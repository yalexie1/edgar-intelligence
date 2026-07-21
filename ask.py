"""
Phase 3: ask a question and get a grounded, cited answer.

Retrieves evidence from the multi-company Chroma index via search() and hands it
to Claude with a strict answer contract: every claim must be quoted, sourced, and
rated for confidence. If evidence is thin the model is instructed to abstain.

P2 upgrades (structured retrieval):
  - Cross-company questions that name specific tickers retrieve evidence per-company
    separately, so every named company is guaranteed at least k chunks in the prompt.
  - Temporal trend questions retrieve from a large pool and group results by filing
    period, presenting evidence chronologically for the model to synthesize.
  - "Which companies" questions (no specific tickers) use an expanded diverse pool.

Run:  python ask.py
      python ask.py --ticker AAPL
      python ask.py --ticker NVDA --form 10-K
      python ask.py --diverse
"""

import argparse
import os
import sys
import time

import cohere
from anthropic import Anthropic
from dotenv import load_dotenv

from embed_and_search import (
    CANDIDATE_K,
    DIVERSE_CANDIDATE_MULTIPLIER,
    TOP_K,
    build_where,
    diversify_results,
    get_pinecone_index,
    search,
)

load_dotenv()  # loads ANTHROPIC_API_KEY (and OPENAI_API_KEY, used by search)

# Haiku is cheap and fast during development; swap to "claude-sonnet-4-6" for
# sharper answers once the pipeline feels solid.
ANSWER_MODEL = "claude-haiku-4-5-20251001"

# Rerank scores below this are treated as "too weak to answer confidently."
LOW_SIMILARITY_THRESHOLD = 0.30

# Maps every name/alias a user might type to the corpus ticker.
# Multi-word aliases must be checked before single-word ones (see detect_tickers).
COMPANY_ALIASES = {
    "advanced micro devices": "AMD",
    "alphabet": "GOOGL",
    "broadcom": "AVGO",
    "facebook": "META",
    "salesforce": "CRM",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "oracle": "ORCL",
    "google": "GOOGL",
    "tesla": "TSLA",
    "intel": "INTC",
    "apple": "AAPL",
    "meta": "META",
    "nvda": "NVDA",
    "msft": "MSFT",
    "amzn": "AMZN",
    "avgo": "AVGO",
    "orcl": "ORCL",
    "tsla": "TSLA",
    "intc": "INTC",
    "aapl": "AAPL",
    "nflx": "NFLX",
    "crm": "CRM",
    "amd": "AMD",
    "aws": "AMZN",
}


def detect_tickers(question):
    """Return corpus tickers explicitly named in the question.

    Checks longer aliases first so "advanced micro devices" doesn't also match
    "micro" from some other rule. Returns a sorted, deduplicated list.
    """
    q = question.lower()
    found = set()
    for alias, ticker in sorted(COMPANY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            found.add(ticker)
    return sorted(found)


# Phrases that signal the user wants a cross-company comparison, even when no
# specific ticker is named. These are unambiguous enough to safely auto-enable
# diverse mode without false-positives on single-company questions.
_CROSS_COMPANY_PHRASES = [
    "which company", "which companies",
    "what company", "what companies",
    "compare companies", "across companies",
    "between companies", "each company",
]

# Phrases that signal the user wants a temporal trend across multiple periods.
# Note: "year over year" is intentionally excluded — it often appears in point-in-time
# questions asking for a YoY growth rate in a single filing (e.g. "Q2 revenue grew
# 154% YoY"), not a trend across multiple filings.
_TEMPORAL_PHRASES = [
    "trended", "trend", "over time", "quarter over quarter",
    "how has", "how have", "grown over", "grown across", "changed over",
    "changed between", "changed across", "multiple quarter", "multiple period",
    "across quarter", "across period", "past year", "past two year",
    "last several", "historically", "over the past", "over recent",
]

# How many results to retrieve per company in structured cross-company mode.
CROSS_COMPANY_K_PER_TICKER = 3

# How many extra candidates to pull for "which companies" style questions.
BROAD_DIVERSE_MULTIPLIER = 16


def is_cross_company_question(question):
    """Return True if the question is asking for a comparison across companies."""
    q = question.lower()
    return any(phrase in q for phrase in _CROSS_COMPANY_PHRASES)


def is_temporal_question(question):
    """Return True if the question is asking about a trend or change over time."""
    q = question.lower()
    return any(phrase in q for phrase in _TEMPORAL_PHRASES)


# Phrases that signal a "what changed" comparison between two specific filing
# periods, as opposed to a general multi-period trend (_TEMPORAL_PHRASES). Kept
# narrow and checked before the temporal check in _route() — a phrase like
# "changed between" alone (already in _TEMPORAL_PHRASES) is deliberately NOT
# included here, since it also matches ordinary trend questions.
_DIFF_PHRASES = [
    "what changed", "what's changed", "what has changed",
    "since last year", "since the prior year", "since its last",
    "compared to last year", "compared to the prior year",
    "compare to last year", "compare to the prior year",
    "vs last year", "versus last year",
    "newly added", "no longer mentions", "new risk factor",
]

# Maps a phrase that names a section to its canonical section label (see
# canonical_section() in embed_and_search.py for the full label set). Longer,
# more specific hints should be listed before shorter ones only where one is a
# substring of another; none currently overlap.
_SECTION_HINTS = {
    "md&a": "mda",
    "management's discussion": "mda",
    "managements discussion": "mda",
    "management discussion": "mda",
    "results of operations": "results_of_operations",
    "financial statement": "financial_statements",
    "market risk": "market_risk",
    "legal proceeding": "legal_proceedings",
    "controls and procedures": "controls",
    "internal controls": "controls",
    "risk factor": "risk_factors",
}


def is_section_diff_question(question):
    """Return True if the question asks what changed in a section between periods."""
    q = question.lower()
    return any(phrase in q for phrase in _DIFF_PHRASES)


def _match_section_hint(question):
    """Return the canonical section explicitly named in the question, or None."""
    q = question.lower()
    for hint, section in _SECTION_HINTS.items():
        if hint in q:
            return section
    return None


def detect_section(question):
    """Map a question to the canonical section it's asking about.

    Defaults to "risk_factors" when no section is named — it's the section
    analysts diff most often ("what changed in the risk factors") and the
    section-diff feature needs some section to filter on even when the user
    doesn't name one explicitly.
    """
    return _match_section_hint(question) or "risk_factors"


def get_collection():
    """Connect to the Pinecone index."""
    try:
        return get_pinecone_index()
    except Exception:
        sys.exit("Could not connect to Pinecone. Check PINECONE_API_KEY in your .env.")


def fmt_source(meta):
    """Format result metadata into a short, human-readable citation label."""
    parts = [
        meta.get("ticker", "?"),
        meta.get("form", "?"),
        meta.get("period") or meta.get("filing_date", "?"),
    ]
    section = meta.get("section") or meta.get("item_title")
    if section:
        parts.append(f"§{section}")
    return " | ".join(parts)


def build_prompt(question, results, history=None):
    """Assemble the question and retrieved passages into a grounded prompt.

    history is an optional list of {"question": str, "answer": str} dicts
    representing prior turns. The model sees them for context but is instructed
    to ground its answer in the freshly retrieved passages, not the prior answers.
    """
    passages = []
    for n, r in enumerate(results, start=1):
        meta = r["metadata"]
        source_line = fmt_source(meta)
        url = meta.get("source_url", "")
        header = f"[{n}] Source: {source_line}"
        if url:
            header += f"\n    URL: {url}"
        passages.append(f"{header}\n{r['text']}")
    context = "\n\n".join(passages)

    history_section = ""
    if history:
        turns = []
        for h in history:
            # Truncate long prior answers so the prompt doesn't balloon
            prior_answer = h["answer"][:500] + ("…" if len(h["answer"]) > 500 else "")
            turns.append(f"User: {h['question']}\nAssistant: {prior_answer}")
        history_section = (
            "Prior conversation (use for context only — "
            "answer the NEW question using the passages below):\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    return f"""You are a financial analyst assistant with access to SEC filings from 13 large public companies (AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, INTC). Answer ONLY from the numbered passages below.

Answer contract — follow every rule:
1. Ground every claim in the passages. Do not add facts from outside them.
2. After each claim, cite the passage number(s) in square brackets, e.g. [1] or [2, 3].
3. For each cited claim, include: the claim, a short supporting quote (use "..."), the source (ticker, form, period, section), and a confidence level (High / Medium / Low).
4. If the passages do not contain enough information to answer confidently, start your entire response with the exact token INSUFFICIENT_EVIDENCE on its own line, then explain why you cannot answer.
5. If the passages feel off-topic or similarities are low, flag that the evidence is thin.
6. Be concise and factual. Prefer exact figures and direct quotes.

{history_section}Passages:
{context}

Question: {question}

Answer (cite each claim with passage number, quote, source, and confidence):"""


def build_cross_company_prompt(question, company_results, history=None):
    """Build a structured comparison prompt with evidence grouped by company.

    company_results is a dict {ticker: [result, ...]} ordered by company.
    Passage numbers are assigned sequentially across all companies so the
    model can cite them unambiguously.
    """
    history_section = ""
    if history:
        turns = []
        for h in history:
            prior = h["answer"][:500] + ("…" if len(h["answer"]) > 500 else "")
            turns.append(f"User: {h['question']}\nAssistant: {prior}")
        history_section = (
            "Prior conversation (for context only):\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    passage_num = 1
    sections = []
    for ticker, results in company_results.items():
        if not results:
            sections.append(f"--- {ticker} ---\n(No relevant filings found for this company.)")
            continue
        company_name = results[0]["metadata"].get("company", ticker)
        header = f"--- {ticker} ({company_name}) ---"
        passages = []
        for r in results:
            meta = r["metadata"]
            source_line = fmt_source(meta)
            url = meta.get("source_url", "")
            h = f"[{passage_num}] Source: {source_line}"
            if url:
                h += f"\n    URL: {url}"
            passages.append(f"{h}\n{r['text']}")
            passage_num += 1
        sections.append(header + "\n" + "\n\n".join(passages))

    context = "\n\n".join(sections)

    return f"""You are a financial analyst comparing multiple companies based on their SEC filings. Answer ONLY from the numbered passages below.

Answer contract — follow every rule:
1. Ground every claim in the passages. Do not add facts from outside them.
2. After each claim, cite the passage number(s) in square brackets, e.g. [1] or [2, 3].
3. Structure your answer by company: for each company, state the key point, include a short supporting quote, cite the passage, and rate confidence (High / Medium / Low).
4. End with a brief synthesis comparing the companies.
5. If passages for a company are missing or very weak, say so explicitly rather than skipping it.
6. If the overall evidence is too thin to compare confidently, start your entire response with INSUFFICIENT_EVIDENCE on its own line, then explain.

{history_section}Passages by company:
{context}

Question: {question}

Answer (company-by-company, then synthesize):"""


def build_temporal_prompt(question, results, ticker, history=None):
    """Build a structured temporal prompt with passages ordered chronologically.

    results is a list of result dicts already sorted by period (oldest first).
    """
    history_section = ""
    if history:
        turns = []
        for h in history:
            prior = h["answer"][:500] + ("…" if len(h["answer"]) > 500 else "")
            turns.append(f"User: {h['question']}\nAssistant: {prior}")
        history_section = (
            "Prior conversation (for context only):\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    passages = []
    current_period = None
    for n, r in enumerate(results, start=1):
        meta = r["metadata"]
        period = meta.get("period", "")
        if period != current_period:
            current_period = period
            passages.append(f"\n--- {period or 'Unknown period'} ---")
        source_line = fmt_source(meta)
        url = meta.get("source_url", "")
        h = f"[{n}] Source: {source_line}"
        if url:
            h += f"\n    URL: {url}"
        passages.append(f"{h}\n{r['text']}")

    context = "\n".join(passages)

    return f"""You are a financial analyst tracking changes over time in SEC filings for {ticker}. Answer ONLY from the numbered passages below.

Answer contract — follow every rule:
1. Ground every claim in the passages. Do not add facts from outside them.
2. After each claim, cite the passage number(s) in square brackets.
3. Work chronologically: for each period, state the key figure or fact, quote briefly, and cite.
4. After the period-by-period breakdown, describe the overall trend (up/down/mixed) and how it changed.
5. Be specific with numbers and dates. If evidence for a period is thin, say so.
6. If the passages are too sparse to describe a trend, start your entire response with INSUFFICIENT_EVIDENCE on its own line.

{history_section}Passages by filing period (chronological):
{context}

Question: {question}

Answer (period-by-period, then trend summary):"""


def build_diff_prompt(question, period_chunks, ticker, section, history=None):
    """Build a prompt comparing the same section across two consecutive filing periods.

    period_chunks is a dict {period: [result, ...]} containing exactly the two
    periods to compare, in chronological order (oldest first) — the dict's own
    iteration order drives the passage layout below, so callers must insert in
    that order.
    """
    history_section = ""
    if history:
        turns = []
        for h in history:
            prior = h["answer"][:500] + ("…" if len(h["answer"]) > 500 else "")
            turns.append(f"User: {h['question']}\nAssistant: {prior}")
        history_section = (
            "Prior conversation (for context only):\n"
            + "\n\n".join(turns)
            + "\n\n"
        )

    passage_num = 1
    sections = []
    for period, results in period_chunks.items():
        header = f"--- {period or 'Unknown period'} ---"
        passages = []
        for r in results:
            meta = r["metadata"]
            source_line = fmt_source(meta)
            url = meta.get("source_url", "")
            h = f"[{passage_num}] Source: {source_line}"
            if url:
                h += f"\n    URL: {url}"
            passages.append(f"{h}\n{r['text']}")
            passage_num += 1
        sections.append(header + "\n" + "\n\n".join(passages))

    context = "\n\n".join(sections)

    return f"""You are a financial analyst comparing {ticker}'s "{section}" section across two consecutive SEC filing periods. Answer ONLY from the numbered passages below.

Answer contract — follow every rule:
1. Ground every claim in the passages. Do not add facts from outside them.
2. After each claim, cite the passage number(s) in square brackets.
3. Structure your answer in three parts:
   a. Added or strengthened — new language, new risks, or emphasis that appears
      in the later period but not the earlier one.
   b. Removed or softened — language present in the earlier period that is
      absent or weakened in the later one.
   c. Unchanged — language or disclosures that remain materially the same
      across both periods.
4. For each point, include a short supporting quote and cite the passage.
5. If the two periods' passages are too thin or too similar to compare
   confidently, start your entire response with INSUFFICIENT_EVIDENCE on its
   own line, then explain.

{history_section}Passages by filing period (chronological, earliest first):
{context}

Question: {question}

Answer (added/strengthened, then removed/softened, then unchanged):"""


def _has_ticker_filter(where):
    """True if `where` already pins a specific ticker (top-level or inside $and)."""
    if not where:
        return False
    if "ticker" in where:
        return True
    return any("ticker" in f for f in where.get("$and", []))


def _extract_section_filter(where):
    """Return the section value already pinned in `where` (top-level or inside
    $and), or None if `where` doesn't pin one.

    Used so an explicit section filter (e.g. the UI's Section dropdown) always
    wins over whatever section a diff question's own wording implies — keeping
    the retrieval filter and the prompt's displayed section label in sync.
    """
    if not where:
        return None
    if "section" in where:
        return where["section"]
    for f in where.get("$and", []):
        if "section" in f:
            return f["section"]
    return None


def _merge_where(where, extra):
    """Combine an existing filter with an additional one, preserving both."""
    if where is None:
        return extra
    if extra is None:
        return where
    if "$and" in where:
        return {"$and": where["$and"] + [extra]}
    return {"$and": [where, extra]}


def _record(timing, key, start):
    """Record elapsed time since `start` into `timing[key]` (v4 #1 tracing).

    No-op when `timing` is None so instrumentation costs nothing for callers
    that don't ask for it (CLI `main()`, `ask_with_contexts`, evals — none of
    which pass a `timing` dict).
    """
    if timing is not None:
        timing[key] = time.perf_counter() - start


def _thin_evidence_abstain(results, message):
    """Shared 'best evidence still too weak' check for all three _prepare_* paths.

    `message` is a format string using `{best}`. Returns the abstain text, or
    None if the best-scored result clears LOW_SIMILARITY_THRESHOLD.
    """
    best = max((r.get("rerank_score", r["similarity"]) for r in results), default=0)
    if best < LOW_SIMILARITY_THRESHOLD:
        return message.format(best=best)
    return None


_cohere_client = None


def _get_cohere_client():
    """Lazily construct the Cohere client. Returns None if no API key is set."""
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            return None
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def _rerank_with_cohere(results, question, k):
    """Cross-encoder re-rank: read question + passage together for precision.

    The local `rerank_score` (cosine + a lexical boost) can't tell a chunk that
    *answers* the question from one that merely shares its vocabulary. Cohere's
    rerank-v3.5 reads both together and re-orders for actual relevance.

    Falls back to the original order (truncated to k) if COHERE_API_KEY is unset
    or the API call fails — reranking is a precision upgrade, not a hard
    dependency, and the free tier is capped at 1000 calls/month.
    """
    if not results:
        return results
    client = _get_cohere_client()
    if client is None:
        return results[:k]
    try:
        response = client.rerank(
            model="rerank-v3.5",
            query=question,
            documents=[r["text"] for r in results],
            top_n=min(k, len(results)),
        )
    except Exception:
        return results[:k]
    return [results[item.index] for item in response.results]


def _prepare_cross_company(collection, question, tickers, k, history, extra_where=None, timing=None):
    """Retrieve evidence per named company and build the comparison prompt (or abstain).

    `extra_where` is an additional metadata filter (e.g. {"section": "risk_factors"})
    merged into each per-ticker filter — carries through a UI filter that was set
    without an explicit ticker, so it survives structured cross-company retrieval.

    Returns (prompt_or_none, abstain_text_or_none, results, effective_where, max_tokens).
    effective_where is always None here — there is no single filter, each company has
    its own. Shared by _ask_cross_company (blocking) and ask_stream (streaming).
    """
    max_tokens = 1400
    k_per = max(CROSS_COMPANY_K_PER_TICKER, k)
    company_results = {}
    # retrieval_s covers the whole per-ticker loop, including each ticker's own
    # Cohere rerank pass below — the two interleave per-ticker, so splitting
    # them into separate retrieval_s/rerank_s timers here would require timing
    # N partial slices instead of one clean stage; not worth the complexity for
    # a per-request tracing feature (see v4 #1 DEVLOG entry).
    t0 = time.perf_counter()
    for ticker in tickers:
        where = _merge_where({"ticker": ticker}, extra_where)
        # Rerank per-ticker (not across the flattened pool) so a weaker company's
        # best chunk can't be crowded out by a stronger company's — the coverage
        # guarantee (every named ticker gets k_per chunks) has to hold post-rerank.
        candidates = search(collection, question, where=where, k=CANDIDATE_K)
        company_results[ticker] = _rerank_with_cohere(candidates, question, k_per)
    _record(timing, "retrieval_s", t0)

    flat = [r for rs in company_results.values() for r in rs]
    if not flat:
        return (
            None,
            "INSUFFICIENT_EVIDENCE\n\nNo evidence found for any of the specified companies.",
            [],
            None,
            max_tokens,
        )

    abstain = _thin_evidence_abstain(
        flat,
        "INSUFFICIENT_EVIDENCE\n\nEvidence is too thin across all companies "
        "(best rerank score: {best:.3f}).",
    )
    if abstain:
        return None, abstain, flat, None, max_tokens

    prompt = build_cross_company_prompt(question, company_results, history=history)
    return prompt, None, flat, None, max_tokens


def _ask_cross_company(collection, question, tickers, k, history, extra_where=None, model=None, timing=None):
    """Structured retrieval: pull evidence per named company, then synthesize.

    Returns (answer_text, flat_results_list, None). effective_where is None
    because there is no single filter — each company has its own.
    """
    prompt, abstain, results, effective_where, max_tokens = _prepare_cross_company(
        collection, question, tickers, k, history, extra_where=extra_where, timing=timing
    )
    if abstain is not None:
        return abstain, results, effective_where

    client = Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _record(timing, "generation_s", t0)
    return msg.content[0].text, results, effective_where


def _prepare_temporal(collection, question, ticker, k, history, extra_where=None, timing=None):
    """Structured temporal retrieval: group evidence by period (or abstain).

    Retrieves a large candidate pool for the ticker, then picks the top result
    from each distinct period so the prompt covers the full time range.
    `extra_where` behaves as in _prepare_cross_company.

    Returns (prompt_or_none, abstain_text_or_none, results, effective_where, max_tokens).
    Shared by _ask_temporal (blocking) and ask_stream (streaming).
    """
    max_tokens = 1200
    # Pull a big candidate pool so we span as many periods as possible. No Cohere
    # rerank here — period order must stay chronological, and reordering by
    # relevance before the per-period grouping below would undermine that.
    raw_k = k * DIVERSE_CANDIDATE_MULTIPLIER * 2
    where = _merge_where({"ticker": ticker}, extra_where)
    t0 = time.perf_counter()
    candidates = search(collection, question, where=where, k=raw_k)
    _record(timing, "retrieval_s", t0)

    if not candidates:
        return (
            None,
            "INSUFFICIENT_EVIDENCE\n\nThe corpus returned no results for this ticker.",
            [],
            where,
            max_tokens,
        )

    # Group by period, keep the best-scored chunk(s) per period.
    period_buckets: dict = {}
    for r in candidates:
        period = r["metadata"].get("period") or r["metadata"].get("filing_date", "")
        if period not in period_buckets:
            period_buckets[period] = []
        period_buckets[period].append(r)

    # Take up to 2 chunks per period (best-ranked already from search()).
    chunks_per_period = max(1, k // max(len(period_buckets), 1))
    selected = []
    for period in sorted(period_buckets.keys()):
        selected.extend(period_buckets[period][:chunks_per_period])

    # Hard cap at k * 2 so the prompt stays manageable.
    selected = selected[:k * 2]

    abstain = _thin_evidence_abstain(
        selected, "INSUFFICIENT_EVIDENCE\n\nEvidence is too thin (best rerank score: {best:.3f})."
    )
    if abstain:
        return None, abstain, selected, where, max_tokens

    prompt = build_temporal_prompt(question, selected, ticker, history=history)
    return prompt, None, selected, where, max_tokens


def _ask_temporal(collection, question, ticker, k, history, extra_where=None, model=None, timing=None):
    """Structured temporal retrieval: group evidence by period, sort chronologically.

    Returns (answer_text, results_list, effective_where).
    """
    prompt, abstain, results, effective_where, max_tokens = _prepare_temporal(
        collection, question, ticker, k, history, extra_where=extra_where, timing=timing
    )
    if abstain is not None:
        return abstain, results, effective_where

    client = Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _record(timing, "generation_s", t0)
    return msg.content[0].text, results, effective_where


def _prepare_section_diff(collection, question, ticker, section, k, history, extra_where=None, timing=None):
    """Structured section-diff retrieval: same section, two most recent periods.

    Filters to the ticker and (unless the caller's own filter already pins a
    section — e.g. the UI's Section dropdown) the detected section, retrieves
    a large candidate pool, groups by period, and keeps the top 2 chunks from
    each of the 2 most recent distinct periods. Falls back to _prepare_temporal
    (N-period grouping) if fewer than 2 periods are found — there's nothing to
    diff, but a trend answer over however many periods exist is still useful.

    Returns (prompt_or_none, abstain_text_or_none, results, effective_where, max_tokens).
    Shared by _ask_section_diff (blocking) and ask_stream (streaming).
    """
    max_tokens = 1200
    where = _merge_where({"ticker": ticker}, extra_where)
    if _extract_section_filter(where) is None:
        where = _merge_where(where, {"section": section})

    t0 = time.perf_counter()
    candidates = search(collection, question, where=where, k=CANDIDATE_K)
    _record(timing, "retrieval_s", t0)

    if not candidates:
        return (
            None,
            "INSUFFICIENT_EVIDENCE\n\nThe corpus returned no results for this "
            "ticker and section.",
            [],
            where,
            max_tokens,
        )

    period_buckets: dict = {}
    for r in candidates:
        period = r["metadata"].get("period") or r["metadata"].get("filing_date", "")
        period_buckets.setdefault(period, []).append(r)

    periods_sorted = sorted(period_buckets.keys())
    if len(periods_sorted) < 2:
        # Nothing to diff — fall back to the temporal path's N-period grouping.
        # Note: this makes a second search() call, so timing["retrieval_s"]
        # ends up reflecting that second (temporal) call, not the one above —
        # an acceptable simplification for a per-request tracing feature.
        return _prepare_temporal(
            collection, question, ticker, k, history, extra_where=extra_where, timing=timing
        )

    two_most_recent = periods_sorted[-2:]
    period_chunks = {p: period_buckets[p][:2] for p in two_most_recent}
    selected = [r for p in two_most_recent for r in period_chunks[p]]

    abstain = _thin_evidence_abstain(
        selected, "INSUFFICIENT_EVIDENCE\n\nEvidence is too thin to compare periods "
        "(best rerank score: {best:.3f})."
    )
    if abstain:
        return None, abstain, selected, where, max_tokens

    prompt = build_diff_prompt(question, period_chunks, ticker, section, history=history)
    return prompt, None, selected, where, max_tokens


def _ask_section_diff(collection, question, ticker, section, k, history, extra_where=None, model=None, timing=None):
    """Structured section-diff retrieval: same section across two consecutive periods.

    Returns (answer_text, results_list, effective_where).
    """
    prompt, abstain, results, effective_where, max_tokens = _prepare_section_diff(
        collection, question, ticker, section, k, history, extra_where=extra_where, timing=timing
    )
    if abstain is not None:
        return abstain, results, effective_where

    client = Anthropic()
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _record(timing, "generation_s", t0)
    return msg.content[0].text, results, effective_where


def _prepare_plain(collection, question, where, k, diverse, history, timing=None):
    """Retrieval for the default (non-structured) path: retrieve, build the prompt,
    or abstain.

    Returns (prompt_or_none, abstain_text_or_none, results, effective_where, max_tokens).
    Shared by _ask_plain (blocking) and ask_stream (streaming).
    """
    max_tokens = 900
    if diverse:
        # Diverse mode fetches more candidates so diversify_results has variety to
        # pick from; ticker spread is the goal there, not raw relevance, so it
        # skips the Cohere pass — reranking pre-diversify would just let the
        # cross-encoder's favorite ticker crowd out the others.
        # "Which companies" questions use BROAD_DIVERSE_MULTIPLIER for more ticker coverage.
        multiplier = BROAD_DIVERSE_MULTIPLIER if is_cross_company_question(question) else DIVERSE_CANDIDATE_MULTIPLIER
        raw_k = k * multiplier
        t0 = time.perf_counter()
        results = search(collection, question, where=where, k=raw_k)
        _record(timing, "retrieval_s", t0)
        results = diversify_results(results, k=k, by="ticker")
    else:
        # Two-stage retrieval: pull the full CANDIDATE_K pool from Pinecone (broad
        # recall via cosine + lexical boost), then let Cohere's cross-encoder
        # re-rank for precision before truncating to k.
        t0 = time.perf_counter()
        candidates = search(collection, question, where=where, k=CANDIDATE_K)
        _record(timing, "retrieval_s", t0)
        t1 = time.perf_counter()
        results = _rerank_with_cohere(candidates, question, k)
        _record(timing, "rerank_s", t1)

    if not results:
        return (
            None,
            "INSUFFICIENT_EVIDENCE\n\nThe corpus returned no results for this question.",
            [],
            where,
            max_tokens,
        )

    abstain = _thin_evidence_abstain(
        results,
        "INSUFFICIENT_EVIDENCE\n\nEvidence is too thin to answer confidently "
        "(best rerank score: {best:.3f}). The corpus may not cover this question.",
    )
    if abstain:
        return None, abstain, results, where, max_tokens

    prompt = build_prompt(question, results, history=history)
    return prompt, None, results, where, max_tokens


def _ask_plain(collection, question, where, k, diverse, history, model=None, timing=None):
    """Default single-search path: retrieve, build the prompt, and answer.

    Returns (answer_text, results_list, effective_where).
    """
    prompt, abstain, results, effective_where, max_tokens = _prepare_plain(
        collection, question, where, k, diverse, history, timing=timing
    )
    if abstain is not None:
        return abstain, results, effective_where

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _record(timing, "generation_s", t0)
    return msg.content[0].text, results, effective_where


def _route(question, where, k, diverse, history):
    """Decide which retrieval strategy a question needs.

    Shared by ask() and ask_stream() so the two never drift out of sync. Ticker
    detection runs unless `where` already pins a specific ticker — a section- or
    form-only filter (e.g. from the UI's Section dropdown, with no ticker chosen)
    still lets the question be scanned for company names, and any ticker found is
    merged into the existing filter via `extra_where` rather than replacing it, so
    the section/form constraint survives structured cross-company/temporal
    retrieval too. When the follow-up doesn't name a company, history is scanned
    so retrieval stays focused on the right ticker.

    Returns a dict:
      {"kind": "cross_company", "tickers": [...], "extra_where": dict|None}
      {"kind": "section_diff", "ticker": str, "section": str, "extra_where": dict|None}
      {"kind": "temporal", "ticker": str, "extra_where": dict|None}
      {"kind": "plain", "where": dict|None, "k": int, "diverse": bool}
    """
    if not _has_ticker_filter(where):
        extra_where = where
        tickers = detect_tickers(question)
        ticker_from_history = False

        # Fall back to the company named in prior turns when the follow-up
        # question doesn't explicitly name one (e.g. "how did they explain that?").
        if not tickers and history:
            for h in reversed(history):
                tickers = detect_tickers(h["question"])
                if tickers:
                    ticker_from_history = True
                    break

        # P2: structured per-company retrieval for explicit multi-company questions.
        if len(tickers) > 1 and not diverse:
            return {"kind": "cross_company", "tickers": tickers, "extra_where": extra_where}

        # v3 §4: "what changed" questions about a single company's section,
        # checked before the (broader) temporal check — a diff question is a
        # special case of a trend question and needs the 2-period grouping in
        # _prepare_section_diff rather than _prepare_temporal's N-period one.
        #
        # Bug fix (2026-07-10): _DIFF_PHRASES includes generic comparative
        # language ("compare to the prior year") that an ordinary conversational
        # follow-up naturally uses regardless of topic. When the ticker is only
        # known via history fallback (not named in this question) and no
        # section is named either, that combination is the signature of a
        # generic follow-up continuing whatever topic was already being
        # discussed — not a deliberate request to diff a section. Caught via
        # live testing: "How does that compare to the prior year?" after a
        # revenue question was silently rerouted into a risk_factors diff for
        # the prior turn's ticker, discarding the actual (revenue) topic.
        # A ticker named explicitly in *this* question, or a section named
        # explicitly in *this* question, still means the user deliberately
        # wants a diff even with unrelated history present.
        explicit_section = _extract_section_filter(extra_where) or _match_section_hint(question)
        if (
            len(tickers) == 1
            and is_section_diff_question(question)
            and not diverse
            and (explicit_section or not ticker_from_history)
        ):
            return {
                "kind": "section_diff",
                "ticker": tickers[0],
                "section": explicit_section or "risk_factors",
                "extra_where": extra_where,
            }

        # P2: structured temporal retrieval for single-company trend questions.
        if len(tickers) == 1 and is_temporal_question(question) and not diverse:
            return {"kind": "temporal", "ticker": tickers[0], "extra_where": extra_where}

        if len(tickers) == 1:
            where = _merge_where(where, {"ticker": tickers[0]})
        elif len(tickers) == 0 and not diverse and is_cross_company_question(question):
            # "Which companies" style: use a larger candidate pool so diversify_results
            # has a better chance of surfacing many distinct tickers.
            diverse = True
            k = max(k, TOP_K)

    return {"kind": "plain", "where": where, "k": k, "diverse": diverse}


def ask(collection, question, where=None, k=TOP_K, diverse=False, history=None, model=None, timing=None):
    """Retrieve evidence and produce a grounded, cited answer.

    Returns (answer_text, results_list, effective_where). Each result dict has:
    id, similarity, lexical_score, rerank_score, text, metadata.

    Routing (P2/v3 §4), decided by _route():
      - Multiple named companies → _ask_cross_company: guaranteed evidence per ticker.
      - Single named company + "what changed" signals → _ask_section_diff: same
        section, two most recent periods, before/after comparison.
      - Single named company + temporal signals → _ask_temporal: per-period grouping.
      - "Which company" questions (no specific tickers) → expanded diverse pool.
      - Everything else → original single-search path (_ask_plain).

    `timing` (v4 #1, optional): a dict the caller supplies to be filled in with
    per-stage latencies (retrieval_s, rerank_s, generation_s — a key is absent
    when that stage didn't run) plus `route`. Mutated in place rather than
    returned, so ask()'s existing 3-tuple return contract (relied on by
    api.py's cache and by evals) stays unchanged. Callers that don't pass a
    dict (CLI `main()`, `ask_with_contexts`) get zero instrumentation cost.

    See ask_stream() for the token-streaming counterpart used by /query/stream.
    """
    route = _route(question, where, k, diverse, history)
    if timing is not None:
        timing["route"] = route["kind"]
    if route["kind"] == "cross_company":
        return _ask_cross_company(
            collection, question, route["tickers"], k, history,
            extra_where=route.get("extra_where"), model=model, timing=timing,
        )
    if route["kind"] == "section_diff":
        return _ask_section_diff(
            collection, question, route["ticker"], route["section"], k, history,
            extra_where=route.get("extra_where"), model=model, timing=timing,
        )
    if route["kind"] == "temporal":
        return _ask_temporal(
            collection, question, route["ticker"], k, history,
            extra_where=route.get("extra_where"), model=model, timing=timing,
        )
    return _ask_plain(
        collection, question, route["where"], route["k"], route["diverse"], history,
        model=model, timing=timing,
    )


def ask_stream(collection, question, where=None, k=TOP_K, diverse=False, history=None, model=None, timing=None):
    """Streaming counterpart to ask(): yields answer text as it arrives from the model.

    Mirrors ask()'s routing (via the same _route() call) so retrieval and abstain
    behavior are identical between the blocking and streaming paths. When the
    question is answerable, tokens are yielded one at a time as they stream in
    from the Anthropic API. When retrieval comes back empty or too weak, the
    abstain message is yielded as a single chunk instead (there is nothing to
    stream — no LLM call is made, matching ask()'s behavior).

    The metadata sentinel is yielded FIRST, before any answer text — retrieval is
    already complete by this point (it happens inside _prepare_*, before the model
    is ever called), so there's no reason to hold sources back until generation
    finishes. This also means a connection drop partway through generation still
    leaves the citations intact for whatever partial answer text did arrive,
    instead of silently losing them:
        {"__meta__": True, "results": [...], "effective_where": ...}

    Does not touch the LRU cache in api.py — streaming responses are not cached.

    `timing` (v4 #1, optional): same contract as ask()'s — a dict the caller
    supplies to be filled in with per-stage latencies plus `route`, mutated in
    place as the generator progresses. retrieval_s/rerank_s are populated by
    the time the __meta__ sentinel is yielded (retrieval already happened by
    then); generation_s is only populated once the generator is exhausted.
    """
    route = _route(question, where, k, diverse, history)
    if timing is not None:
        timing["route"] = route["kind"]

    if route["kind"] == "cross_company":
        prompt, abstain, results, effective_where, max_tokens = _prepare_cross_company(
            collection, question, route["tickers"], k, history,
            extra_where=route.get("extra_where"), timing=timing,
        )
    elif route["kind"] == "section_diff":
        prompt, abstain, results, effective_where, max_tokens = _prepare_section_diff(
            collection, question, route["ticker"], route["section"], k, history,
            extra_where=route.get("extra_where"), timing=timing,
        )
    elif route["kind"] == "temporal":
        prompt, abstain, results, effective_where, max_tokens = _prepare_temporal(
            collection, question, route["ticker"], k, history,
            extra_where=route.get("extra_where"), timing=timing,
        )
    else:
        prompt, abstain, results, effective_where, max_tokens = _prepare_plain(
            collection, question, route["where"], route["k"], route["diverse"], history, timing=timing,
        )

    yield {"__meta__": True, "results": results, "effective_where": effective_where}

    if abstain is not None:
        yield abstain
        return

    client = Anthropic()
    t0 = time.perf_counter()
    with client.messages.stream(
        model=model or ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
    _record(timing, "generation_s", t0)


# Fixed set of themes relevant to SEC filings, each mapped to a natural-language
# query that the embedding model can match against chunk text. Order is display order.
THEMES = {
    "ai_ml":             "artificial intelligence machine learning generative AI large language model compute GPU",
    "cybersecurity":     "cybersecurity data breach cyber attack security incident vulnerability",
    "supply_chain":      "supply chain supplier component shortage inventory disruption",
    "regulation":        "regulatory antitrust regulation compliance government oversight",
    "china_geopolitics": "China geopolitical export control tariff sanction trade restriction",
    "climate_esg":       "climate change sustainability carbon emission environmental regulation ESG",
    "competition":       "competition competitive market share competitor emerging displacement",
    "cloud_platform":    "cloud revenue growth infrastructure platform SaaS subscription services",
}

THEME_LABELS = {
    "ai_ml":             "AI & ML",
    "cybersecurity":     "Cybersecurity",
    "supply_chain":      "Supply Chain",
    "regulation":        "Regulation",
    "china_geopolitics": "China & Geopolitics",
    "climate_esg":       "Climate & ESG",
    "competition":       "Competition",
    "cloud_platform":    "Cloud & Platform",
}


def track_themes(collection, ticker, k=5):
    """Retrieve the strongest evidence for each predefined theme across filing periods.

    Returns retrieval-only results — no LLM call. For each theme the response
    contains the best-matching chunk per distinct filing period, sorted
    chronologically by filing_date.

    Returns:
        {
          "ticker": str,
          "themes": {
            theme_key: {
              "label": str,
              "periods": [
                {"period": str, "filing_date": str, "score": float,
                 "text": str, "section": str, "form": str, "source_url": str},
                ...
              ]
            }, ...
          }
        }
    """
    where = {"ticker": ticker}
    themes_out = {}

    for theme_key, query in THEMES.items():
        # Broad pool so we span many periods; 8-K dates are noisy for trend analysis.
        hits = search(collection, query, where=where, k=k * 4)
        hits = [h for h in hits if h["metadata"].get("form", "").startswith("10-")]

        # Keep the best-scored chunk per distinct period.
        by_period: dict = {}
        for hit in hits:
            period = hit["metadata"].get("period", "")
            score = hit.get("rerank_score", hit["similarity"])
            if period not in by_period or score > by_period[period]["score"]:
                by_period[period] = {
                    "period":     period,
                    "filing_date": hit["metadata"].get("filing_date", ""),
                    "score":      round(score, 3),
                    "text":       hit["text"][:400],
                    "section":    hit["metadata"].get("section", ""),
                    "form":       hit["metadata"].get("form", ""),
                    "source_url": hit["metadata"].get("source_url", ""),
                }

        periods_sorted = sorted(by_period.values(), key=lambda x: x["filing_date"])
        themes_out[theme_key] = {"label": THEME_LABELS[theme_key], "periods": periods_sorted}

    return {"ticker": ticker, "themes": themes_out}


def ask_with_contexts(question, collection, where=None, k=TOP_K, diverse=False, history=None):
    """Thin wrapper around ask() that returns retrieved contexts alongside the answer.

    Used by evals/eval_ragas.py so RAGAS can judge grounding without duplicating
    the retrieval pipeline.

    Returns:
        {
          "answer":   str,
          "contexts": [str, ...],   # raw chunk texts, one per retrieved passage
          "sources":  [dict, ...],  # ticker/form/period/section/url per passage
        }
    """
    answer, results, _ = ask(
        collection, question, where=where, k=k, diverse=diverse, history=history
    )
    return {
        "answer": answer,
        "contexts": [r["text"] for r in results],
        "sources": [
            {
                "ticker":     r["metadata"].get("ticker", ""),
                "form":       r["metadata"].get("form", ""),
                "period":     r["metadata"].get("period") or r["metadata"].get("filing_date", ""),
                "section":    r["metadata"].get("section", ""),
                "source_url": r["metadata"].get("source_url", ""),
            }
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the SEC corpus.")
    parser.add_argument("--ticker", help="filter to a single ticker, e.g. AAPL")
    parser.add_argument("--form", help="filter by form type, e.g. 10-K, 10-Q, 8-K")
    parser.add_argument("--section", help="filter by section, e.g. mda, risk_factors")
    parser.add_argument("--item", help="filter by SEC item number, e.g. 1A, 7")
    parser.add_argument("--period", help="filter by period, e.g. FY2025")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of evidence chunks")
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="pull at least one result per ticker (good for cross-company questions)",
    )
    args = parser.parse_args()

    collection = get_collection()
    where = build_where(args)

    print("EDGAR Intelligence — SEC filing Q&A with citations.")
    print(f"Model: {ANSWER_MODEL} | Filters: {where or 'none'} | diverse: {args.diverse} | k: {args.k}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in {"quit", "exit", ""}:
            break

        answer, results, effective_where = ask(
            collection, question, where=where, k=args.k, diverse=args.diverse
        )
        if effective_where and effective_where != where:
            print(f"[auto-detected filter: {effective_where}]")

        print("\n" + answer.strip())

        if results:
            print("\n--- sources used ---")
            for n, r in enumerate(results, start=1):
                meta = r["metadata"]
                url = meta.get("source_url", "")
                print(
                    f"[{n}] {fmt_source(meta)}"
                    f"  sim={r['similarity']:.3f}"
                    f"  rank={r.get('rerank_score', r['similarity']):.3f}"
                )
                if url:
                    print(f"     {url}")
        print()


if __name__ == "__main__":
    main()