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
import csv
import datetime
import json
import sys
import time
from pathlib import Path

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
    def mean(col):
        vals = [r[col] for r in case_results if r[col] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "run_date":   datetime.datetime.now().isoformat(),
        "n_cases":    len(case_results),
        "subset":     subset is not None,
        "group":      group_filter,
        "judge_note": (
            "RAGAS uses an LLM judge (OpenAI GPT by default). "
            "Scores vary by model version and are directional signals, not ground truth."
        ),
        "metrics": {col: mean(col) for col in metric_cols},
    }

    # ── save results ──────────────────────────────────────────────────────────
    RESULTS_JSON.write_text(json.dumps({"summary": summary, "cases": case_results}, indent=2))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    # CSV — one row per case, flat columns
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group", "question", "elapsed"] + metric_cols)
        writer.writeheader()
        writer.writerows(case_results)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  RAGAS results  ({len(case_results)} cases)")
    for col in metric_cols:
        v = summary["metrics"][col]
        label = v if v is None else f"{v:.4f}"
        print(f"  {col:<24} {label}")
    print(f"{'─'*52}")
    print(f"\n  Saved:")
    print(f"    {RESULTS_JSON}")
    print(f"    {RESULTS_CSV}")
    print(f"    {SUMMARY_JSON}\n")
    print(
        "  Note: RAGAS scores are LLM-judge estimates. They complement the\n"
        "  deterministic suite in eval.py but are not ground truth.\n"
    )


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
    args = parser.parse_args()
    run(subset=args.subset, group_filter=args.group)
