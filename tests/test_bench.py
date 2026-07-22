"""
Unit tests for pure functions in bench/load_test.py — no network calls.

Covers:
  - parse_levels    — parses a "1,5,10,20,50" concurrency-level spec into ints
  - percentile       — linear-interpolation percentile over sorted floats
  - summarize_run    — aggregates raw per-request samples into p50/p95/p99/etc.
"""

import pytest

from bench.load_test import parse_levels, percentile, summarize_run


# ── parse_levels ─────────────────────────────────────────────────────────────

class TestParseLevels:
    def test_basic_csv(self):
        assert parse_levels("1,5,10,20,50") == [1, 5, 10, 20, 50]

    def test_single_value(self):
        assert parse_levels("10") == [10]

    def test_strips_whitespace(self):
        assert parse_levels(" 1, 5 , 10 ") == [1, 5, 10]

    def test_ignores_empty_segments(self):
        assert parse_levels("1,,5,") == [1, 5]


# ── percentile ───────────────────────────────────────────────────────────────

class TestPercentile:
    def test_empty_returns_none(self):
        assert percentile([], 0.5) is None

    def test_single_value(self):
        assert percentile([3.0], 0.95) == 3.0

    def test_p50_odd_count(self):
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p0_is_min(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0

    def test_p100_is_max(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0

    def test_interpolates_between_values(self):
        # 4 values, p95 -> k = 3 * 0.95 = 2.85 -> interpolate between index 2 and 3
        # 30 + (40 - 30) * 0.85 = 38.5
        result = percentile([10.0, 20.0, 30.0, 40.0], 0.95)
        assert result == pytest.approx(38.5)


# ── summarize_run ────────────────────────────────────────────────────────────

class TestSummarizeRun:
    def test_all_successful(self):
        samples = [
            {"latency_s": 1.0, "ok": True, "status": 200},
            {"latency_s": 2.0, "ok": True, "status": 200},
            {"latency_s": 3.0, "ok": True, "status": 200},
        ]
        result = summarize_run(samples)
        assert result["count"] == 3
        assert result["errors"] == 0
        assert result["error_rate"] == 0.0
        assert result["p50"] == 2.0
        assert result["mean"] == pytest.approx(2.0)
        assert result["min"] == 1.0
        assert result["max"] == 3.0

    def test_counts_errors_by_ok_flag(self):
        samples = [
            {"latency_s": 1.0, "ok": True, "status": 200},
            {"latency_s": 5.0, "ok": False, "status": 500},
        ]
        result = summarize_run(samples)
        assert result["count"] == 2
        assert result["errors"] == 1
        assert result["error_rate"] == pytest.approx(0.5)

    def test_latency_percentiles_only_over_successful_requests(self):
        # A failed request's latency (e.g. a fast connection-refused) shouldn't
        # pull the success-path latency percentiles down.
        samples = [
            {"latency_s": 10.0, "ok": True, "status": 200},
            {"latency_s": 10.0, "ok": True, "status": 200},
            {"latency_s": 0.001, "ok": False, "status": 0},
        ]
        result = summarize_run(samples)
        assert result["p50"] == 10.0
        assert result["min"] == 10.0

    def test_empty_samples(self):
        result = summarize_run([])
        assert result["count"] == 0
        assert result["errors"] == 0
        assert result["error_rate"] == 0.0
        assert result["p50"] is None

    def test_all_failed_no_success_latencies(self):
        samples = [{"latency_s": 0.1, "ok": False, "status": 500}]
        result = summarize_run(samples)
        assert result["count"] == 1
        assert result["errors"] == 1
        assert result["error_rate"] == 1.0
        assert result["p50"] is None
        assert result["mean"] is None

    def test_ttft_included_when_present(self):
        samples = [
            {"latency_s": 5.0, "ok": True, "status": 200, "ttft_s": 1.0},
            {"latency_s": 6.0, "ok": True, "status": 200, "ttft_s": 2.0},
        ]
        result = summarize_run(samples)
        assert result["ttft_p50"] == pytest.approx(1.5)

    def test_ttft_absent_when_no_samples_have_it(self):
        samples = [{"latency_s": 5.0, "ok": True, "status": 200}]
        result = summarize_run(samples)
        assert "ttft_p50" not in result
