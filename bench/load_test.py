"""
Load testing & performance benchmarking for the EDGAR Intelligence API (V4.md #2).

Ramps concurrency against a LOCALLY RUNNING `uvicorn api:app` instance and reports
p50/p95/p99 latency + error rate per concurrency level, for both /query and
/query/stream. NEVER point this at the deployed Render URL — see the module-level
warning below and CLAUDE.md's Known Limitations.

Three scenarios:
  - warm:   /query, one fixed question, cache primed once up front. Every request
            after the first is a cache hit (near-zero marginal API cost), so this
            is where the full concurrency ladder (default 1,5,10,20,50) gets swept.
  - cold:   /query, 5 distinct real questions fired in a single concurrent wave
            (not swept across levels) — bounded to exactly 5 real LLM calls,
            per V4.md's explicit "keep the cold path small-N" guidance.
  - stream: /query/stream. IMPORTANT: this endpoint is never cached (see its
            docstring in api.py) — every request in the sweep is a real, billed
            call, not just the first. Also records time-to-first-token (TTFT),
            not just total latency.

Usage (server must already be running, e.g. `uvicorn api:app --port 8000`):
    python -m bench.load_test --scenario all
    python -m bench.load_test --scenario warm --levels 1,5,10,20,50
    python -m bench.load_test --scenario cold
    python -m bench.load_test --scenario stream --levels 1,5,10

Results are written as JSON to bench/results/<timestamp>_<scenario>.json.
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── safety guard ─────────────────────────────────────────────────────────────
# Hammering the deployed Render API would burn through Cohere's free tier
# (1,000 calls/month) and real Anthropic/OpenAI dollars in minutes, and would
# immediately trip the app's own per-IP rate limiter. Local only.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_LEVELS = "1,5,10,20,50"

WARM_QUESTION = "What was Apple's total revenue in fiscal year 2025?"

# 5 distinct real questions for the cache-cold snapshot — each a genuine cache
# miss (question, where=None, diverse=False) not reused anywhere else in this
# script, so every one is a real embedding + retrieval + generation call.
COLD_QUESTIONS = [
    "What did Microsoft say about AI investment in its most recent 10-K?",
    "What are Tesla's main supply chain risk factors?",
    "How did NVIDIA's data center revenue trend over the last two quarters?",
    "What acquisitions has Broadcom completed recently?",
    "What does Amazon disclose about its AWS segment margins?",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ── pure helpers (unit tested in tests/test_bench.py) ──────────────────────────

def parse_levels(spec: str) -> list[int]:
    """Parse a comma-separated concurrency-level spec ("1,5,10,20,50") into ints."""
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def percentile(sorted_values: list[float], pct: float):
    """Linear-interpolation percentile over an already-sorted list of floats.

    Mirrors api.py's `_percentile` — duplicated rather than imported so this
    script stays a standalone, dependency-light tool. `pct` is a fraction in
    [0, 1] (0.95 = p95). Returns None for an empty list.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize_run(samples: list[dict]) -> dict:
    """Aggregate raw per-request samples into p50/p95/p99/mean/min/max/error_rate.

    Each sample: {"latency_s": float, "ok": bool, "status": int, "ttft_s": float?}.
    Latency percentiles are computed only over successful requests — a failed
    request's (often very fast, e.g. connection-refused) latency shouldn't drag
    down the success-path percentiles. `ttft_p50/p95/p99` are included only when
    at least one sample carries a `ttft_s` value (streaming scenario).
    """
    count = len(samples)
    errors = sum(1 for s in samples if not s["ok"])
    result = {
        "count": count,
        "errors": errors,
        "error_rate": (errors / count) if count else 0.0,
    }

    success_latencies = sorted(s["latency_s"] for s in samples if s["ok"])
    if success_latencies:
        result["p50"] = percentile(success_latencies, 0.50)
        result["p95"] = percentile(success_latencies, 0.95)
        result["p99"] = percentile(success_latencies, 0.99)
        result["mean"] = statistics.mean(success_latencies)
        result["min"] = success_latencies[0]
        result["max"] = success_latencies[-1]
    else:
        result.update(p50=None, p95=None, p99=None, mean=None, min=None, max=None)

    ttft_values = sorted(
        s["ttft_s"] for s in samples if s.get("ok") and s.get("ttft_s") is not None
    )
    if ttft_values:
        result["ttft_p50"] = percentile(ttft_values, 0.50)
        result["ttft_p95"] = percentile(ttft_values, 0.95)
        result["ttft_p99"] = percentile(ttft_values, 0.99)

    return result


# ── network orchestration (not unit tested — live spot-check only, same
#    convention as ask.py's/api.py's network-touching code) ────────────────────

def _assert_local(base_url: str) -> None:
    host = httpx.URL(base_url).host
    if host not in _ALLOWED_HOSTS:
        raise SystemExit(
            f"Refusing to load-test '{base_url}' — host '{host}' is not localhost.\n"
            "This benchmark must only run against a locally running uvicorn instance. "
            "Hammering the deployed Render API would burn real API budget and trip "
            "its own rate limiter. See CLAUDE.md's Known Limitations."
        )


async def _fire_one(client: httpx.AsyncClient, url: str, payload: dict, stream: bool) -> dict:
    start = time.perf_counter()
    try:
        if stream:
            first_token_t = None
            async with client.stream("POST", url, json=payload, timeout=60.0) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    if line.startswith("data:") and first_token_t is None:
                        first_token_t = time.perf_counter()
            latency = time.perf_counter() - start
            ttft = (first_token_t - start) if first_token_t is not None else None
            return {"latency_s": latency, "ok": status == 200, "status": status, "ttft_s": ttft}
        else:
            resp = await client.post(url, json=payload, timeout=60.0)
            latency = time.perf_counter() - start
            return {"latency_s": latency, "ok": resp.status_code == 200, "status": resp.status_code}
    except Exception:
        return {"latency_s": time.perf_counter() - start, "ok": False, "status": 0}


async def _fire_wave(client, url, payload, n, concurrency, stream=False) -> list[dict]:
    """Fire `n` requests to `url` with at most `concurrency` in flight at once."""
    sem = asyncio.Semaphore(concurrency)

    async def one():
        async with sem:
            return await _fire_one(client, url, payload, stream)

    return list(await asyncio.gather(*(one() for _ in range(n))))


async def _get_metrics(client: httpx.AsyncClient, base_url: str) -> dict:
    try:
        resp = await client.get(f"{base_url}/metrics", timeout=10.0)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def run_warm_scenario(base_url: str, levels: list[int], requests_per_level_min: int = 20) -> dict:
    """Sweep the concurrency ladder against one cache-primed /query question."""
    url = f"{base_url}/query"
    payload = {"question": WARM_QUESTION}
    async with httpx.AsyncClient() as client:
        # Prime the cache — this one call is the only real LLM cost in this scenario.
        prime = await _fire_one(client, url, payload, stream=False)

        per_level = {}
        for level in levels:
            n = max(requests_per_level_min, level * 2)
            samples = await _fire_wave(client, url, payload, n=n, concurrency=level, stream=False)
            per_level[level] = {"n_requests": n, **summarize_run(samples)}

        server_metrics = await _get_metrics(client, base_url)

    return {"scenario": "warm", "prime_request": prime, "levels": per_level, "server_metrics": server_metrics}


async def run_cold_scenario(base_url: str) -> dict:
    """Fire all COLD_QUESTIONS concurrently, once — bounded real API cost."""
    url = f"{base_url}/query"
    async with httpx.AsyncClient() as client:
        payloads = [{"question": q} for q in COLD_QUESTIONS]
        sem = asyncio.Semaphore(len(payloads))

        async def one(payload):
            async with sem:
                return await _fire_one(client, url, payload, stream=False)

        samples = list(await asyncio.gather(*(one(p) for p in payloads)))
        server_metrics = await _get_metrics(client, base_url)

    return {
        "scenario": "cold",
        "n_unique_questions": len(COLD_QUESTIONS),
        "summary": summarize_run(samples),
        "raw_samples": samples,
        "server_metrics": server_metrics,
    }


async def run_stream_scenario(base_url: str, levels: list[int]) -> dict:
    """Sweep the concurrency ladder against /query/stream.

    /query/stream is never cached, so every request across every level is a
    real, billed call — one wave of exactly `level` requests per level (not
    requests_per_level_min padding like the warm scenario) to keep total real
    cost bounded to sum(levels) calls.
    """
    url = f"{base_url}/query/stream"
    payload = {"question": WARM_QUESTION}
    async with httpx.AsyncClient() as client:
        per_level = {}
        for level in levels:
            samples = await _fire_wave(client, url, payload, n=level, concurrency=level, stream=True)
            per_level[level] = {"n_requests": level, **summarize_run(samples)}

        server_metrics = await _get_metrics(client, base_url)

    return {"scenario": "stream", "levels": per_level, "server_metrics": server_metrics}


def _save_results(result: dict, scenario: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{ts}_{scenario}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path


async def _run(args) -> None:
    _assert_local(args.base_url)
    levels = parse_levels(args.levels)

    scenarios = ["warm", "cold", "stream"] if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario} ===", file=sys.stderr)
        if scenario == "warm":
            result = await run_warm_scenario(args.base_url, levels)
        elif scenario == "cold":
            result = await run_cold_scenario(args.base_url)
        elif scenario == "stream":
            result = await run_stream_scenario(args.base_url, levels)
        else:
            raise SystemExit(f"Unknown scenario: {scenario}")

        path = _save_results(result, scenario)
        print(json.dumps(result, indent=2, default=str))
        print(f"\nSaved: {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"default: {DEFAULT_BASE_URL} (must be localhost)")
    parser.add_argument("--scenario", choices=["warm", "cold", "stream", "all"], default="all")
    parser.add_argument("--levels", default=DEFAULT_LEVELS, help=f"comma-separated concurrency levels, default: {DEFAULT_LEVELS}")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
