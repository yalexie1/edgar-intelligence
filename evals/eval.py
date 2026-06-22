"""
Phase 6: Evaluation harness for EDGAR Intelligence.

Runs each case in dataset.json against the live API and scores two things:

  Retrieval hit-rate  — did the expected ticker(s) appear in the top-k sources?
                        A miss means the vector search + reranking didn't surface
                        the right document, regardless of what the model said.

  Answer faithfulness — did the answer contain the expected key fact(s)?
                        Measured by case-insensitive substring matching on the
                        strings listed in answer_contains.

  Abstain precision   — for out-of-corpus questions, did the model correctly
                        decline to answer rather than hallucinating?

  Diversity check     — for cross-company cases with min_unique_tickers set,
                        did the sources cover enough distinct companies?

Run:
  python evals/eval.py                         # assumes API on localhost:8000
  python evals/eval.py --url http://host:8000
  python evals/eval.py --verbose               # print full answer for every case
  python evals/eval.py --group factual         # run one group only
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET = Path(__file__).parent / "dataset.json"

# Strings that signal the model abstained rather than answered.
ABSTAIN_SIGNALS = [
    "cannot answer",
    "cannot be answered",
    "do not contain",
    "not contain",
    "not cover",
    "no information",
    "not available",
    "no passages",
    "no results",
    "not in the",
    "not provided",
    "insufficient",
    "filings don't",
    "filings do not",
    "not found",
]

# ANSI colours (disabled on Windows or when not a TTY)
USE_COLOR = sys.stdout.isatty() and sys.platform != "win32"
GREEN  = "\033[32m" if USE_COLOR else ""
RED    = "\033[31m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
RESET  = "\033[0m"  if USE_COLOR else ""
BOLD   = "\033[1m"  if USE_COLOR else ""
DIM    = "\033[2m"  if USE_COLOR else ""


def call_api(base_url, question):
    payload = json.dumps({
        "question": question,
        "ticker": "",
        "form": "",
        "diverse": False,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/query",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def did_abstain(answer):
    a = answer.lower()
    return any(sig in a for sig in ABSTAIN_SIGNALS)


def score_case(case, data):
    """Return a dict with all scoring fields for one eval case."""
    sources = data.get("sources", [])
    answer  = data.get("answer", "")
    source_tickers  = [s["ticker"] for s in sources]
    unique_tickers  = set(source_tickers)
    source_periods  = [s.get("period", "") for s in sources]
    unique_periods  = set(p for p in source_periods if p)

    should_abstain    = case.get("should_abstain", False)
    expected_tickers  = case.get("expected_tickers", [])
    answer_needles    = case.get("answer_contains", [])
    min_unique_tickers = case.get("min_unique_tickers", 0)
    min_unique_periods = case.get("min_unique_periods", 0)

    answer_lower = answer.lower()

    # --- retrieval hit ---
    # All expected tickers must appear somewhere in the top-k sources.
    tickers_hit = all(t in unique_tickers for t in expected_tickers)
    diversity_hit = len(unique_tickers) >= min_unique_tickers if min_unique_tickers else True
    period_hit    = len(unique_periods) >= min_unique_periods  if min_unique_periods  else True
    retrieval_hit = tickers_hit and diversity_hit and period_hit

    # --- answer faithfulness ---
    if answer_needles:
        answer_hit = all(needle.lower() in answer_lower for needle in answer_needles)
    else:
        # No specific strings required; pass if retrieval hit (we can only check grounding)
        answer_hit = True

    # --- abstain check ---
    if should_abstain:
        abstain_correct = did_abstain(answer)
        # Override the other scores for abstain cases — retrieval is not meaningful
        retrieval_hit = True
        answer_hit    = abstain_correct
        correct       = abstain_correct
    else:
        abstain_correct = None
        correct = retrieval_hit and answer_hit

    return {
        "correct":         correct,
        "retrieval_hit":   retrieval_hit,
        "answer_hit":      answer_hit,
        "abstain_correct": abstain_correct,
        "tickers_hit":     tickers_hit,
        "diversity_hit":   diversity_hit,
        "period_hit":      period_hit,
        "unique_tickers":  sorted(unique_tickers),
        "filter_applied":  data.get("filter_applied"),
        "answer":          answer,
    }


def fmt_bool(b, label=""):
    if b is None:
        return f"{DIM}n/a{RESET}"
    icon = f"{GREEN}✓{RESET}" if b else f"{RED}✗{RESET}"
    return f"{icon} {label}" if label else icon


def run(base_url, verbose=False, group_filter=None):
    cases = json.loads(DATASET.read_text())
    if group_filter:
        cases = [c for c in cases if c.get("group") == group_filter]
    if not cases:
        sys.exit(f"No cases matched group '{group_filter}'.")

    results = []
    col_w = max(len(c["id"]) for c in cases)

    print(f"\n{BOLD}EDGAR Intelligence — Eval Suite{RESET}  ({len(cases)} cases)\n")

    for i, case in enumerate(cases, 1):
        cid   = case["id"]
        group = case.get("group", "")
        print(f"  {DIM}[{i:02d}/{len(cases)}]{RESET}  {cid:{col_w}}  ", end="", flush=True)

        t0 = time.time()
        try:
            data = call_api(base_url, case["question"])
        except urllib.error.URLError as e:
            print(f"{RED}API ERROR: {e}{RESET}")
            results.append({"id": cid, "group": group, "correct": False,
                             "retrieval_hit": False, "answer_hit": False,
                             "abstain_correct": None, "error": str(e)})
            continue
        elapsed = time.time() - t0

        sc = score_case(case, data)
        sc["id"]      = cid
        sc["group"]   = group
        sc["elapsed"] = elapsed
        results.append(sc)

        status = f"{GREEN}PASS{RESET}" if sc["correct"] else f"{RED}FAIL{RESET}"
        detail_parts = [
            f"ret={fmt_bool(sc['retrieval_hit'])}",
            f"ans={fmt_bool(sc['answer_hit'])}",
        ]
        if case.get("should_abstain"):
            detail_parts = [f"abstain={fmt_bool(sc['abstain_correct'])}"]
        detail = "  ".join(detail_parts)
        tickers_str = f"{DIM}{','.join(sc['unique_tickers']) or '-'}{RESET}"
        print(f"{status}  {detail}  {tickers_str}  {DIM}{elapsed:.1f}s{RESET}")

        if verbose or not sc["correct"]:
            if sc["filter_applied"]:
                print(f"         {DIM}filter:  {sc['filter_applied']}{RESET}")
            excerpt = sc["answer"][:300].replace("\n", " ")
            print(f"         {DIM}answer:  {excerpt}…{RESET}")
            if not sc["correct"] and case.get("answer_contains"):
                missing = [n for n in case["answer_contains"]
                           if n.lower() not in sc["answer"].lower()]
                if missing:
                    print(f"         {RED}missing: {missing}{RESET}")
            print()

    # ── Summary ────────────────────────────────────────────────────────────────
    factual_results = [r for r in results if r.get("group") != "abstain" and "error" not in r]
    abstain_results = [r for r in results if r.get("group") == "abstain" and "error" not in r]
    all_valid       = [r for r in results if "error" not in r]

    n_total   = len(all_valid)
    n_pass    = sum(r["correct"] for r in all_valid)
    n_ret_hit = sum(r["retrieval_hit"] for r in factual_results)
    n_ans_hit = sum(r["answer_hit"] for r in factual_results)
    n_f       = len(factual_results)
    n_ab_ok   = sum(r["abstain_correct"] for r in abstain_results if r["abstain_correct"] is not None)
    n_ab      = len(abstain_results)

    def pct(n, d):
        return f"{100 * n // d}%" if d else "n/a"

    print(f"\n{'─'*56}")
    print(f"  {BOLD}Overall{RESET}               {n_pass}/{n_total}  {pct(n_pass, n_total)}")
    print(f"  Retrieval hit-rate    {n_ret_hit}/{n_f}  {pct(n_ret_hit, n_f)}")
    print(f"  Answer faithfulness   {n_ans_hit}/{n_f}  {pct(n_ans_hit, n_f)}")
    print(f"  Abstain precision     {n_ab_ok}/{n_ab}  {pct(n_ab_ok, n_ab)}")

    # Per-group breakdown
    groups = sorted({r["group"] for r in all_valid})
    if len(groups) > 1:
        print(f"\n  {'Group':<22} {'Pass':>6}  {'Total':>6}  {'%':>5}")
        print(f"  {'─'*22} {'─'*6}  {'─'*6}  {'─'*5}")
        for g in groups:
            gr = [r for r in all_valid if r["group"] == g]
            gp = sum(r["correct"] for r in gr)
            print(f"  {g:<22} {gp:>6}  {len(gr):>6}  {pct(gp, len(gr)):>5}")
    print(f"{'─'*56}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR Intelligence evals.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--verbose", action="store_true", help="print answer excerpt for every case")
    parser.add_argument("--group", help="run only this group: factual, temporal, cross_company, abstain")
    args = parser.parse_args()
    run(args.url, verbose=args.verbose, group_filter=args.group)
