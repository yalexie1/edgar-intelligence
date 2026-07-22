"""
Optional RAGAS evaluation for EDGAR Intelligence.

Computes LLM-as-a-judge quality metrics on top of the golden dataset:
  faithfulness       — are all claims in the answer grounded in the retrieved context?
  answer_relevancy   — is the answer relevant to the question asked?
  context_precision  — are the most relevant contexts ranked first?
  context_recall     — do the retrieved contexts cover the expected answer?

IMPORTANT: RAGAS uses an LLM (OpenAI GPT by default) as the judge, so scores vary
by model version and are not deterministic. Treat them as directional signals, not
ground truth. The deterministic suite in eval.py remains the primary regression test.

Requires: pip install "ragas>=0.1.9" datasets pandas
  (these are optional; the rest of the project runs without them)

Saves:
  evals/results/ragas_results.json   — per-case scores
  evals/results/ragas_results.csv    — tabular version (good for Excel/pandas)
  evals/results/ragas_summary.json   — aggregate metrics (loaded by dashboard)

Run:
  python evals/eval_ragas.py --subset 10   # quick smoke test, ~10 LLM calls
  python evals/eval_ragas.py               # full 100-case run
  python evals/eval_ragas.py --group factual --subset 20
"""

import argparse
import asyncio
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

# ── nest_asyncio + C-accelerated asyncio.current_task() incompatibility ────────
# ragas.executor applies nest_asyncio.apply() at import time (unconditionally, on
# every Python version — this is not 3.14-specific). nest_asyncio swaps
# asyncio.Task for its pure-Python re-entrant implementation so nested event
# loops work, but asyncio.current_task() is the C-accelerated `_asyncio` builtin,
# which tracks tasks via its own C-level bookkeeping and has no visibility into
# nest_asyncio's pure-Python Task instances. The result: current_task() always
# returns None while a task is genuinely running, for the lifetime of the process
# (confirmed via a minimal repro: `nest_asyncio.apply(); asyncio.run(...)` prints
# `current_task(): None` even though asyncio.run() always runs its coroutine as
# a real task).
#
# A prior fix here patched only asyncio.timeouts.Timeout.__aenter__ (whose
# "Timeout should be used inside a task" RuntimeError was the first symptom
# found). That was a single-call-site patch, not a root-cause one: it left every
# *other* consumer of current_task() unfixed. The one that actually produces the
# NaN RAGAS scores is sniffio.current_async_library() (sniffio/_impl.py), called
# deep inside anyio/httpcore on every async OpenAI/Anthropic HTTP call ragas
# makes as an LLM judge — it also checks `current_task() is not None` and raises
# AsyncLibraryNotFoundError when that's false, which ragas's executor swallows
# per-job (logged, not raised) and records as `nan`.
#
# Root-cause fix: patch asyncio.current_task itself (the actual function both
# call sites — and any future one — depend on) to fall back to the *real*
# running task when the C-accelerated lookup returns None. nest_asyncio's
# pure-Python Task.__step still calls the legacy _enter_task/_leave_task
# bookkeeping functions (kept in asyncio.tasks for backward compatibility from
# before current_task() was C-accelerated), which populate
# asyncio.tasks._current_tasks — a dict keyed by loop, holding the actual,
# fully-functional running Task object. So the real task is recoverable; it's
# just invisible to the C builtin. This is confirmed by direct inspection
# (nest_asyncio.apply() + asyncio.run(...): current_task() -> None, but
# asyncio.tasks._current_tasks[loop] -> the real, correct Task).
#
# Recovering the real task (rather than fabricating a fake stand-in) matters
# because anyio's cancellation machinery introspects real Task internals
# (e.g. `task._must_cancel`) — a fake object without those attributes breaks
# with AttributeError as soon as anyio's TaskGroup tries to deliver a
# cancellation/timeout, which is a real, different failure than the one this
# patch is fixing (found via a first attempt at this fix, which used a fake
# object and immediately hit that AttributeError under the same live test).
#
# This must run before anything imports anyio, since anyio's asyncio backend
# does `from asyncio import current_task` (a name binding captured at import
# time, not a live module-attribute lookup) — patching asyncio.current_task
# after that import would not affect anyio's already-bound reference. Placing
# this at the top of the file, before `ragas`/`openai`/`anthropic` are
# imported, guarantees the patch is in place first.
import asyncio
import asyncio.tasks as _tasks

_real_current_task = asyncio.current_task


def _patched_current_task(loop=None):
    task = _real_current_task(loop)
    if task is not None:
        return task
    try:
        current_loop = loop if loop is not None else asyncio.get_running_loop()
    except RuntimeError:
        return None
    return _tasks._current_tasks.get(current_loop)


asyncio.current_task = _patched_current_task
asyncio.tasks.current_task = _patched_current_task

# ── optional imports ───────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset
    import pandas as pd
except ImportError as _e:
    sys.exit(
        f"RAGAS dependencies not installed: {_e}\n"
        'Run: pip install "ragas>=0.1.9" datasets pandas'
    )

# ── project imports ────────────────────────────────────────────────────────────
# Adjust path so this script can be run from either the project root or evals/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ask import ask_with_contexts, get_collection

DATASET_PATH  = Path(__file__).parent / "dataset.json"
RESULTS_DIR   = Path(__file__).parent / "results"
RESULTS_JSON  = RESULTS_DIR / "ragas_results.json"
RESULTS_CSV   = RESULTS_DIR / "ragas_results.csv"
SUMMARY_JSON  = RESULTS_DIR / "ragas_summary.json"

# RAGAS's four metrics (see module docstring) all assume a single retrieval
# pass answering a single, self-contained question — that's the shape
# `factual` cases have. It's not the shape the other three groups have:
#   - abstain: the correct answer is "the filings don't cover this," and
#     retrieval is *supposed* to surface weak/no evidence. answer_relevancy
#     (built by generating questions from the answer and comparing them back
#     to the original question) and context_precision (built on "are the
#     retrieved contexts relevant") both score this as if it were a failure,
#     when it's the system working as designed.
#   - cross_company / temporal: `_ask_cross_company()` / `_ask_temporal()`
#     deliberately retrieve chunks spanning multiple companies or multiple
#     filing periods so the answer can compare across them. Most of those
#     chunks are individually irrelevant to any single sub-claim — necessary
#     for the comparison, not noise — but RAGAS's per-chunk relevance judge
#     has no notion of "this chunk is here for period-grouping."
# So the headline RAGAS score is scoped to `factual` cases, where the metrics
# measure what they're designed to measure. The other three groups are still
# scored and saved (see metrics_by_group below) — nothing is hidden, just not
# folded into the headline number, since averaging all four groups together
# mixes "system worked as designed" with "system answered a normal question
# poorly" into one indistinguishable mean.
RAGAS_APPLICABLE_GROUPS = ["factual"]


def _mean_ignoring_nan(values: list) -> tuple:
    """Aggregate a metric column, treating NaN the same as a missing score.

    ragas's Executor catches exceptions per-job (rate limits, timeouts — see
    ragas/executor.py's wrap_callable_with_index) and records np.nan rather
    than raising, so a single case's transient failure doesn't crash the
    whole run. But `sum(values) / len(values)` propagates NaN through
    *any* arithmetic it touches — so if even one of 110 cases came back
    NaN, the entire aggregate silently became NaN too, masking an otherwise
    real score for the rest. This is what actually produced the "all 4
    metrics NaN" result CLAUDE.md attributed to a Python-version blocker:
    once the real nest_asyncio bug was fixed, every job failed identically,
    which is indistinguishable from a handful of genuine per-job rate-limit
    failures without this fix.

    Returns (mean_or_None, n_scored) — n_scored lets callers report how many
    of the total cases actually contributed to the mean.
    """
    scored = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not scored:
        return None, 0
    return round(sum(scored) / len(scored), 4), len(scored)


def _aggregate(case_results: list, metric_cols: list) -> tuple:
    """Compute (metrics, metrics_n_scored) over a list of per-case result dicts."""
    metrics = {}
    metrics_n_scored = {}
    for col in metric_cols:
        mean_val, n_scored = _mean_ignoring_nan([r[col] for r in case_results])
        metrics[col] = mean_val
        metrics_n_scored[col] = n_scored
    return metrics, metrics_n_scored


def build_summary(case_results: list, run_date: str, n_cases_total: int,
                   subset: bool, group_filter: str | None) -> dict:
    """Build the aggregate summary dict from per-case RAGAS scores.

    Split out from run() so --recompute-only can rebuild the summary (with
    the by-group / applicable-subset breakdown) from an already-saved
    ragas_results.json's case data, at zero additional LLM cost.
    """
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    metrics, metrics_n_scored = _aggregate(case_results, metric_cols)

    applicable_cases = [r for r in case_results if r.get("group") in RAGAS_APPLICABLE_GROUPS]
    metrics_applicable, metrics_applicable_n_scored = _aggregate(applicable_cases, metric_cols)

    metrics_by_group = {}
    for grp in sorted({r.get("group", "unknown") for r in case_results}):
        grp_cases = [r for r in case_results if r.get("group") == grp]
        grp_metrics, grp_n_scored = _aggregate(grp_cases, metric_cols)
        metrics_by_group[grp] = {
            "n_cases": len(grp_cases),
            "metrics": grp_metrics,
            "metrics_n_scored": grp_n_scored,
        }

    return {
        "run_date":   run_date,
        "n_cases":    n_cases_total,
        "subset":     subset,
        "group":      group_filter,
        "judge_note": (
            "RAGAS uses an LLM judge (OpenAI GPT by default). "
            "Scores vary by model version and are directional signals, not ground truth."
        ),
        # All 110 cases, all groups — kept for full transparency (nothing
        # hidden), but NOT the headline number — see applicable_note.
        "metrics": metrics,
        "metrics_n_scored": metrics_n_scored,
        # The headline: only cases in RAGAS_APPLICABLE_GROUPS (factual),
        # where RAGAS's methodology assumptions actually hold.
        "metrics_applicable": metrics_applicable,
        "metrics_applicable_n_scored": metrics_applicable_n_scored,
        "applicable_groups": RAGAS_APPLICABLE_GROUPS,
        "applicable_note": (
            "Headline metrics are scored on the 'factual' group only (single "
            "question, single retrieval pass, single answer — what RAGAS's "
            "four metrics assume). abstain/cross_company/temporal cases use "
            "retrieval patterns RAGAS wasn't designed to score (see "
            "metrics_by_group for their scores and CLAUDE.md v4 §4 for why)."
        ),
        "metrics_by_group": metrics_by_group,
    }


def build_ground_truth(case: dict) -> str:
    """Derive a reference answer string for context_precision / context_recall.

    Uses answer_contains strings joined with '; ' when available, falling back
    to the notes field. For abstain cases, the expected answer is an abstention.
    """
    if case.get("should_abstain"):
        return "The corpus does not contain information about this question."
    needles = case.get("answer_contains", [])
    if needles:
        return "; ".join(needles)
    # Use notes as a rough ground truth for retrieval-only cases.
    return case.get("notes", "")


def run(subset: int | None = None, group_filter: str | None = None) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    cases = json.loads(DATASET_PATH.read_text())
    if group_filter:
        cases = [c for c in cases if c.get("group") == group_filter]
    if not cases:
        sys.exit(f"No cases matched group '{group_filter}'.")
    if subset:
        cases = cases[:subset]

    print(f"\nEDGAR Intelligence — RAGAS Eval  ({len(cases)} cases)")
    print("Loading Chroma index…")
    collection = get_collection()

    # ── collect answers and contexts ──────────────────────────────────────────
    rows = []   # one dict per case, accumulates question/answer/contexts/ground_truth
    print("Fetching answers (this calls the LLM for each case)…\n")

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"  [{i:02d}/{len(cases)}]  {cid}", flush=True)
        t0 = time.time()
        try:
            result = ask_with_contexts(case["question"], collection)
        except Exception as e:
            print(f"           ERROR: {e}")
            # Skip this case — don't include broken results in RAGAS
            continue
        elapsed = time.time() - t0

        rows.append({
            "id":           cid,
            "group":        case.get("group", ""),
            "question":     case["question"],
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": build_ground_truth(case),
            "elapsed":      round(elapsed, 2),
        })
        print(f"           {elapsed:.1f}s  {len(result['contexts'])} contexts")

    if not rows:
        sys.exit("No cases completed successfully. Check the Chroma index and API keys.")

    # ── run RAGAS ─────────────────────────────────────────────────────────────
    print(f"\nRunning RAGAS on {len(rows)} cases (calls OpenAI — may take a minute)…")

    ragas_dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    except Exception as e:
        sys.exit(f"RAGAS evaluation failed: {e}")

    # ragas_result.to_pandas() gives a DataFrame with one row per case
    scores_df = ragas_result.to_pandas()

    # ── assemble per-case output ───────────────────────────────────────────────
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    case_results = []
    for i, row in enumerate(rows):
        case_scores = {col: round(float(scores_df.iloc[i][col]), 4) for col in metric_cols}
        case_results.append({
            "id":      row["id"],
            "group":   row["group"],
            "question": row["question"],
            "elapsed": row["elapsed"],
            **case_scores,
        })

    # ── aggregate summary ─────────────────────────────────────────────────────
    summary = build_summary(
        case_results,
        run_date=datetime.datetime.now().isoformat(),
        n_cases_total=len(case_results),
        subset=subset is not None,
        group_filter=group_filter,
    )

    # ── save results ──────────────────────────────────────────────────────────
    RESULTS_JSON.write_text(json.dumps({"summary": summary, "cases": case_results}, indent=2))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    # CSV — one row per case, flat columns
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "question", "elapsed"] + metric_cols)
        writer.writeheader()
        writer.writerows(case_results)

    _print_summary(summary, len(case_results))


def _print_summary(summary: dict, n_cases: int) -> None:
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"\n{'─'*60}")
    print(f"  RAGAS headline  (factual group only, n={summary['metrics_applicable_n_scored'].get('faithfulness', '?')} scored)")
    for col in metric_cols:
        v = summary["metrics_applicable"][col]
        label = v if v is None else f"{v:.4f}"
        print(f"  {col:<24} {label}")
    print(f"\n  All groups combined ({n_cases} cases, includes abstain/cross_company/temporal —")
    print(f"  not RAGAS-applicable, see applicable_note):")
    for col in metric_cols:
        v = summary["metrics"][col]
        label = v if v is None else f"{v:.4f}"
        print(f"  {col:<24} {label}")
    print(f"\n  By group:")
    for grp, g in summary["metrics_by_group"].items():
        vals = "  ".join(f"{col}={g['metrics'][col]}" for col in metric_cols)
        print(f"    {grp:<15} (n={g['n_cases']:>2})  {vals}")
    print(f"{'─'*60}")
    print(f"\n  Saved:")
    print(f"    {RESULTS_JSON}")
    print(f"    {RESULTS_CSV}")
    print(f"    {SUMMARY_JSON}\n")
    print(
        "  Note: RAGAS scores are LLM-judge estimates. They complement the\n"
        "  deterministic suite in eval.py but are not ground truth.\n"
    )


def recompute_summary_only() -> None:
    """Rebuild ragas_summary.json / ragas_results.json / .csv from the cases
    already saved by a prior real run — zero new LLM calls, zero new cost.

    Exists so the applicable-subset / by-group breakdown (added after the
    2026-07-22 full 110-case run) can be applied to that run's real,
    already-collected per-case scores without re-spending ~110 answer calls
    + ~440 judge calls to regenerate them.
    """
    if not RESULTS_JSON.exists():
        sys.exit(f"{RESULTS_JSON} does not exist — run a real eval first (no --recompute-only).")

    existing = json.loads(RESULTS_JSON.read_text())
    case_results = existing["cases"]
    old_summary = existing["summary"]
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    summary = build_summary(
        case_results,
        run_date=old_summary["run_date"],   # preserve: no new run happened
        n_cases_total=old_summary["n_cases"],
        subset=old_summary["subset"],
        group_filter=old_summary["group"],
    )

    RESULTS_JSON.write_text(json.dumps({"summary": summary, "cases": case_results}, indent=2))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "question", "elapsed"] + metric_cols)
        writer.writeheader()
        writer.writerows(case_results)

    print(f"Recomputed summary from {len(case_results)} already-scored cases (run_date preserved: {summary['run_date']}).")
    _print_summary(summary, len(case_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on EDGAR Intelligence.")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="evaluate only the first N cases (default: all 100)",
    )
    parser.add_argument(
        "--group", default=None,
        help="filter to one group: factual, temporal, cross_company, abstain",
    )
    parser.add_argument(
        "--recompute-only", action="store_true",
        help="rebuild summary/results/csv from an already-saved ragas_results.json's "
             "per-case scores — no LLM calls, no new cost. Use after a schema change "
             "to the summary aggregation (e.g. applicable-subset / by-group breakdown).",
    )
    args = parser.parse_args()
    if args.recompute_only:
        recompute_summary_only()
    else:
        run(subset=args.subset, group_filter=args.group)
