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
import sys

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

from embed_and_search import (
    CHROMA_DIR,
    COLLECTION,
    DIVERSE_CANDIDATE_MULTIPLIER,
    TOP_K,
    build_where,
    diversify_results,
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


def get_collection():
    """Connect to the Chroma vector store built in Phase 2."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        sys.exit("No vector store found. Run `python embed_and_search.py` first.")


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


def _ask_cross_company(collection, question, tickers, k, history):
    """Structured retrieval: pull evidence per named company, then synthesize.

    Returns (answer_text, flat_results_list, None). effective_where is None
    because there is no single filter — each company has its own.
    """
    k_per = max(CROSS_COMPANY_K_PER_TICKER, k)
    company_results = {}
    for ticker in tickers:
        where = {"ticker": ticker}
        results = search(collection, question, where=where, k=k_per)
        company_results[ticker] = results

    flat = [r for rs in company_results.values() for r in rs]
    if not flat:
        return (
            "INSUFFICIENT_EVIDENCE\n\nNo evidence found for any of the specified companies.",
            [],
            None,
        )

    # Abstain if no company has even weak evidence.
    best = max((r.get("rerank_score", r["similarity"]) for r in flat), default=0)
    if best < LOW_SIMILARITY_THRESHOLD:
        return (
            f"INSUFFICIENT_EVIDENCE\n\nEvidence is too thin across all companies "
            f"(best rerank score: {best:.3f}).",
            flat,
            None,
        )

    prompt = build_cross_company_prompt(question, company_results, history=history)
    client = Anthropic()
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, flat, None


def _ask_temporal(collection, question, ticker, k, history):
    """Structured temporal retrieval: group evidence by period, sort chronologically.

    Retrieves a large candidate pool for the ticker, then picks the top result
    from each distinct period so the prompt covers the full time range.

    Returns (answer_text, results_list, effective_where).
    """
    # Pull a big candidate pool so we span as many periods as possible.
    raw_k = k * DIVERSE_CANDIDATE_MULTIPLIER * 2
    where = {"ticker": ticker}
    candidates = search(collection, question, where=where, k=raw_k)

    if not candidates:
        return (
            "INSUFFICIENT_EVIDENCE\n\nThe corpus returned no results for this ticker.",
            [],
            where,
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

    best = max((r.get("rerank_score", r["similarity"]) for r in selected), default=0)
    if best < LOW_SIMILARITY_THRESHOLD:
        return (
            f"INSUFFICIENT_EVIDENCE\n\nEvidence is too thin (best rerank score: {best:.3f}).",
            selected,
            where,
        )

    prompt = build_temporal_prompt(question, selected, ticker, history=history)
    client = Anthropic()
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, selected, where


def ask(collection, question, where=None, k=TOP_K, diverse=False, history=None):
    """Retrieve evidence and produce a grounded, cited answer.

    Returns (answer_text, results_list, effective_where). Each result dict has:
    id, similarity, lexical_score, rerank_score, text, metadata.

    Routing (P2):
      - Multiple named companies → _ask_cross_company: guaranteed evidence per ticker.
      - Single named company + temporal signals → _ask_temporal: per-period grouping.
      - "Which company" questions (no specific tickers) → expanded diverse pool.
      - Everything else → original single-search path.

    If no explicit `where` filter is given, the question is scanned for company
    names. When the follow-up doesn't name a company, history is scanned so
    retrieval stays focused on the right ticker.
    """
    if where is None:
        tickers = detect_tickers(question)

        # Fall back to the company named in prior turns when the follow-up
        # question doesn't explicitly name one (e.g. "how did they explain that?").
        if not tickers and history:
            for h in reversed(history):
                tickers = detect_tickers(h["question"])
                if tickers:
                    break

        # P2: structured per-company retrieval for explicit multi-company questions.
        if len(tickers) > 1 and not diverse:
            return _ask_cross_company(collection, question, tickers, k, history)

        # P2: structured temporal retrieval for single-company trend questions.
        if len(tickers) == 1 and is_temporal_question(question) and not diverse:
            return _ask_temporal(collection, question, tickers[0], k, history)

        if len(tickers) == 1:
            where = {"ticker": tickers[0]}
        elif len(tickers) == 0 and not diverse and is_cross_company_question(question):
            # "Which companies" style: use a larger candidate pool so diversify_results
            # has a better chance of surfacing many distinct tickers.
            diverse = True
            k = max(k, TOP_K)

    # Diverse mode fetches more candidates so diversify_results has variety to pick from.
    # "Which companies" questions use BROAD_DIVERSE_MULTIPLIER for more ticker coverage.
    if diverse:
        multiplier = BROAD_DIVERSE_MULTIPLIER if is_cross_company_question(question) else DIVERSE_CANDIDATE_MULTIPLIER
        raw_k = k * multiplier
    else:
        raw_k = k
    results = search(collection, question, where=where, k=raw_k)

    if diverse:
        results = diversify_results(results, k=k, by="ticker")

    if not results:
        return (
            "INSUFFICIENT_EVIDENCE\n\nThe corpus returned no results for this question.",
            [],
            where,
        )

    # If the best evidence is very weak, abstain rather than risk hallucination.
    best_score = results[0].get("rerank_score", results[0]["similarity"])
    if best_score < LOW_SIMILARITY_THRESHOLD:
        return (
            f"INSUFFICIENT_EVIDENCE\n\nEvidence is too thin to answer confidently "
            f"(best rerank score: {best_score:.3f}). "
            "The corpus may not cover this question.",
            results,
            where,
        )

    prompt = build_prompt(question, results, history=history)
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text, results, where


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