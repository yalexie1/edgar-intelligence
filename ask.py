"""
Phase 3: ask a question and get a grounded, cited answer.

Retrieves evidence from the multi-company Chroma index via search() and hands it
to Claude with a strict answer contract: every claim must be quoted, sourced, and
rated for confidence. If evidence is thin the model is instructed to abstain.

For cross-company questions use --diverse so diversify_results() ensures each ticker
gets at least one evidence slot before the model compares across companies.

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


def is_cross_company_question(question):
    """Return True if the question is asking for a comparison across companies."""
    q = question.lower()
    return any(phrase in q for phrase in _CROSS_COMPANY_PHRASES)


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


def ask(collection, question, where=None, k=TOP_K, diverse=False, history=None):
    """Retrieve evidence and produce a grounded, cited answer.

    Returns (answer_text, results_list, effective_where). Each result dict has:
    id, similarity, lexical_score, rerank_score, text, metadata.

    If no explicit `where` filter is given, the question is scanned for company
    names. A single named company → filter to that ticker. Multiple named
    companies → enable diverse mode so each gets at least one evidence slot.

    history is an optional list of {"question": str, "answer": str} dicts from
    prior turns. When the current question doesn't name a company, the history
    is scanned for one so retrieval stays focused on the right ticker across
    follow-up questions.
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

        if len(tickers) == 1:
            where = {"ticker": tickers[0]}
        elif len(tickers) > 1 and not diverse:
            diverse = True
        elif len(tickers) == 0 and not diverse and is_cross_company_question(question):
            diverse = True

    # Diverse mode fetches more candidates so diversify_results has variety to pick from.
    raw_k = k * DIVERSE_CANDIDATE_MULTIPLIER if diverse else k
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