"""
Tests for the nest_asyncio/asyncio.current_task() compatibility patch in
evals/eval_ragas.py (v4 #4).

Background: evals/eval_ragas.py's last recorded run (2026-06-23) showed all
four RAGAS metrics as NaN. CLAUDE.md attributed this to "blocked on Python
3.14 + nest_asyncio". Direct investigation (2026-07-22) found the real cause:
ragas.executor applies nest_asyncio.apply() unconditionally at import time
(on every Python version, not just 3.14), which swaps asyncio.Task for a
pure-Python implementation. asyncio.current_task() remains the C-accelerated
builtin, which has no visibility into that pure-Python Task, so it always
returns None even while a task is genuinely running. Every downstream
consumer that gates on "current_task() is not None" breaks identically —
including sniffio.current_async_library() (called deep inside anyio/httpcore
on every async OpenAI/Anthropic HTTP call ragas makes as an LLM judge), which
is what actually produces the NaN scores. A prior session's patch covered
only one such consumer (asyncio.timeouts.Timeout.__aenter__) and missed this
one entirely.

These tests exercise the patched behavior directly (asyncio/nest_asyncio/
sniffio/anyio only) — no network calls, no LLM calls. The full, real
evals/eval_ragas.py --subset 3 run (which does call OpenAI/Anthropic) was
separately confirmed live to produce real, non-NaN scores; see DEVLOG.md.

Importing evals.eval_ragas requires the optional ragas/datasets/pandas
dependencies (see requirements.txt) — skipped entirely if they're not
installed, matching the free tests.yml CI environment, which doesn't install
them.
"""

import asyncio

import pytest

ragas = pytest.importorskip("ragas", reason="ragas is an optional dependency (see requirements.txt)")
pytest.importorskip("datasets", reason="datasets is an optional dependency (see requirements.txt)")
pytest.importorskip("pandas", reason="pandas is an optional dependency (see requirements.txt)")

import sniffio

from evals import eval_ragas
from evals.eval_ragas import _mean_ignoring_nan, build_summary, RAGAS_APPLICABLE_GROUPS


# ── sanity: the module-level patch actually ran ─────────────────────────────

class TestPatchApplied:
    def test_asyncio_current_task_is_patched(self):
        assert asyncio.current_task is eval_ragas._patched_current_task

    def test_asyncio_tasks_current_task_is_patched(self):
        assert asyncio.tasks.current_task is eval_ragas._patched_current_task

    def test_nest_asyncio_was_applied(self):
        # ragas.executor calls nest_asyncio.apply() at import time; importing
        # eval_ragas (which imports ragas) must trigger this as a precondition
        # for the rest of these tests to be testing the real failure mode.
        assert getattr(asyncio, "_nest_patched", False) is True


# ── _patched_current_task itself ────────────────────────────────────────────

class TestPatchedCurrentTask:
    def test_raises_outside_any_event_loop(self):
        # The real, unpatched asyncio.current_task() raises RuntimeError (not
        # None) when called with no running loop at all — confirmed by direct
        # inspection, contrary to what the stdlib docs summary suggests. The
        # patch must preserve this: it only recovers a task when we're inside
        # a genuinely running loop but the C-level lookup came up empty, not
        # paper over the "no loop exists" case.
        with pytest.raises(RuntimeError, match="no running event loop"):
            eval_ragas._patched_current_task()

    def test_returns_real_running_task(self):
        async def inner():
            return eval_ragas._patched_current_task()

        task = asyncio.run(inner())
        assert task is not None
        # Must be a genuine Task (has real internals anyio's cancellation
        # machinery introspects), not a fabricated stand-in — the first fix
        # attempt used a fake object here and broke with AttributeError on
        # `_must_cancel` as soon as anyio tried to deliver a cancellation.
        assert hasattr(task, "_must_cancel")
        assert hasattr(task, "cancelling")
        assert task.cancelling() == 0

    def test_recovers_real_task_when_c_lookup_returns_none(self):
        # This is the exact failure mode: nest_asyncio's pure-Python Task
        # makes the C-accelerated current_task() return None mid-task. The
        # patch's fallback path (asyncio.tasks._current_tasks) must recover
        # the *same* real task the coroutine is actually running in.
        async def inner():
            unpatched = eval_ragas._real_current_task()
            recovered = eval_ragas._patched_current_task()
            return unpatched, recovered

        unpatched, recovered = asyncio.run(inner())
        assert unpatched is None, (
            "if this now returns non-None, the underlying nest_asyncio/"
            "current_task() incompatibility this patch works around may "
            "have been fixed upstream — re-evaluate whether this patch is "
            "still needed"
        )
        assert recovered is not None


# ── the actual consumers that broke before this fix ─────────────────────────

class TestDownstreamConsumers:
    def test_sniffio_detects_asyncio(self):
        async def inner():
            return sniffio.current_async_library()

        assert asyncio.run(inner()) == "asyncio"

    def test_asyncio_timeout_context_manager_works(self):
        async def inner():
            async with asyncio.timeout(5):
                return "ok"

        assert asyncio.run(inner()) == "ok"

    # Not unit-tested here: a real anyio TaskGroup/CancelScope, which is what
    # actually caught the first fix attempt's bug (a fake-task stand-in with
    # no _must_cancel attribute). anyio ships a pytest11 plugin
    # (anyio.pytest_plugin) that pytest auto-loads before any test module is
    # collected, and that plugin eagerly imports anyio._backends._asyncio —
    # binding its `from asyncio import current_task` reference before this
    # patch (which only runs when evals.eval_ragas itself is imported) has a
    # chance to run. That's a pytest-process artifact, not a real ordering
    # problem: in actual `python evals/eval_ragas.py` usage this patch is the
    # first thing that runs, well before ragas/openai/anthropic/anyio ever
    # get imported (confirmed: importing ragas/datasets/pandas/ask directly,
    # with no pytest involved, does not pull in anyio._backends._asyncio
    # eagerly). The real, live `--subset 3` run producing genuine non-NaN
    # scores (see DEVLOG.md) is the actual end-to-end proof for the anyio
    # path; this file covers the isolated mechanism instead.


# ── _mean_ignoring_nan: the second, separate bug found via the full 110-case run ──

class TestMeanIgnoringNan:
    def test_all_valid_values(self):
        mean, n = _mean_ignoring_nan([0.8, 0.9, 1.0])
        assert mean == 0.9
        assert n == 3

    def test_filters_none(self):
        mean, n = _mean_ignoring_nan([0.5, None, 0.7])
        assert mean == 0.6
        assert n == 2

    def test_filters_nan(self):
        # The actual bug: a real 110-case run had 7/110 NaN faithfulness
        # scores (rate-limit/timeout job failures ragas records as np.nan)
        # and the old `sum(vals) / len(vals)` with only a None-filter turned
        # the entire aggregate into NaN as a result.
        mean, n = _mean_ignoring_nan([0.8, float("nan"), 0.6, float("nan")])
        assert mean == 0.7
        assert n == 2

    def test_all_nan_returns_none_and_zero(self):
        mean, n = _mean_ignoring_nan([float("nan"), float("nan")])
        assert mean is None
        assert n == 0

    def test_empty_list_returns_none_and_zero(self):
        mean, n = _mean_ignoring_nan([])
        assert mean is None
        assert n == 0

    def test_mixed_none_and_nan(self):
        mean, n = _mean_ignoring_nan([1.0, None, float("nan"), 0.5])
        assert mean == 0.75
        assert n == 2


# ── build_summary: applicable-subset / by-group breakdown ───────────────────
# Real finding (2026-07-22, same session as the NaN fix): the raw 110-case
# aggregate looked bad (context_precision 0.330, answer_relevancy 0.507)
# because RAGAS's four metrics assume a single question / single retrieval
# pass / single answer — true for `factual` cases, not for `abstain`
# (correct answer is a deliberate non-answer) or `cross_company`/`temporal`
# (retrieval deliberately spans multiple companies/periods). These tests
# use synthetic per-case dicts, not real RAGAS output, to pin down the
# aggregation logic itself.

def _case(group, faithfulness=0.9, answer_relevancy=0.9,
          context_precision=0.9, context_recall=0.9):
    return {
        "id": f"{group}_case", "group": group, "question": "q", "elapsed": 1.0,
        "faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
        "context_precision": context_precision, "context_recall": context_recall,
    }


class TestBuildSummary:
    def test_applicable_groups_is_factual_only(self):
        assert RAGAS_APPLICABLE_GROUPS == ["factual"]

    def test_metrics_applicable_only_uses_factual_cases(self):
        cases = [
            _case("factual", faithfulness=1.0),
            _case("factual", faithfulness=0.8),
            _case("abstain", faithfulness=0.0),   # must NOT affect the headline
        ]
        summary = build_summary(cases, run_date="t", n_cases_total=3,
                                 subset=False, group_filter=None)
        assert summary["metrics_applicable"]["faithfulness"] == 0.9
        assert summary["metrics_applicable_n_scored"]["faithfulness"] == 2

    def test_metrics_all_groups_still_includes_everything(self):
        # Full transparency: the un-scoped `metrics` field must still cover
        # every case, not just the applicable subset — nothing is deleted.
        cases = [_case("factual", faithfulness=1.0), _case("abstain", faithfulness=0.0)]
        summary = build_summary(cases, run_date="t", n_cases_total=2,
                                 subset=False, group_filter=None)
        assert summary["metrics"]["faithfulness"] == 0.5
        assert summary["metrics_n_scored"]["faithfulness"] == 2

    def test_metrics_by_group_breaks_down_each_group_separately(self):
        cases = [
            _case("factual", faithfulness=1.0), _case("factual", faithfulness=1.0),
            _case("abstain", faithfulness=0.0),
        ]
        summary = build_summary(cases, run_date="t", n_cases_total=3,
                                 subset=False, group_filter=None)
        by_group = summary["metrics_by_group"]
        assert by_group["factual"]["n_cases"] == 2
        assert by_group["factual"]["metrics"]["faithfulness"] == 1.0
        assert by_group["abstain"]["n_cases"] == 1
        assert by_group["abstain"]["metrics"]["faithfulness"] == 0.0

    def test_no_cases_in_applicable_group_yields_none(self):
        cases = [_case("abstain"), _case("temporal")]
        summary = build_summary(cases, run_date="t", n_cases_total=2,
                                 subset=False, group_filter=None)
        assert summary["metrics_applicable"]["faithfulness"] is None
        assert summary["metrics_applicable_n_scored"]["faithfulness"] == 0
