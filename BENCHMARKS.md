# BENCHMARKS.md — Load testing & performance benchmarking (v4 #2)

Measured with `bench/load_test.py` against a **locally running** `uvicorn api:app`
(single process, no `--workers`, matching `render.yaml`/`start.sh`'s deployed topology
exactly). Never run against the deployed Render URL — see the safety guard in
`bench/load_test.py` and CLAUDE.md's Known Limitations.

**Run conditions (state these explicitly — they affect the numbers):**
- Local dev machine's network path to Pinecone/OpenAI/Anthropic/Cohere, not Render's
  actual container network. Real deployed numbers will differ.
- `ANSWER_MODEL` unset → dev-default `claude-haiku-4-5-20251001`, not the deployed
  `claude-sonnet-4-6` (set via `render.yaml`). Haiku generation is faster and cheaper
  than what's actually live — treat `generation_s` here as a lower bound for Render.
- Corpus/index unchanged: 8,141 vectors, same as production.
- Raw JSON for every run below: `bench/results/20260722T040319Z_cold.json`,
  `20260722T040341Z_warm.json`, `20260722T040447Z_stream.json`.

## 1. Cache-warm sweep (`/query`)

One question (`WARM_QUESTION`) primes the LRU cache with a single real call, then that
same cached `(question, where, diverse)` tuple is hammered at each concurrency level.
Real API cost: **1 call total** (the priming request) — everything else is a cache hit.

| Concurrency | Requests | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1  | 20  | 1.2ms  | 1.5ms   | 2.2ms   | 0 |
| 5  | 20  | 2.6ms  | 5.2ms   | 5.5ms   | 0 |
| 10 | 20  | 6.6ms  | 11.8ms  | 13.1ms  | 0 |
| 20 | 40  | 14.0ms | 27.8ms  | 32.9ms  | 0 |
| 50 | 100 | 64.4ms | 161.4ms | 169.5ms | 0 |

**Finding:** cache-hit traffic scales gracefully all the way to 50 concurrent requests —
p95 stays under 200ms and there isn't a single error. The FastAPI/uvicorn request-handling
layer itself is not the bottleneck; latency here is almost entirely queueing for the
in-process cache lookup, not any external API. This isolates the real constraint to the
*uncached* path, tested next.

## 2. Streaming sweep (`/query/stream`) — never cached

**Important finding, discovered while reading `api.py` before building this:**
`/query/stream` has no cache tier at all (its own docstring says so: streamed responses
"can't be replayed from a cache key"). Unlike the warm `/query` scenario above, **every
one of the 86 requests below is a real, billed call** — embedding + Pinecone retrieval +
Cohere rerank + Claude generation, every time, regardless of repeated question text.

| Concurrency | Requests | p50 total | p95 total | p50 TTFT | p95 TTFT | Errors |
|---|---|---|---|---|---|---|
| 1  | 1  | 4.44s  | 4.44s  | 2.19s  | 2.19s  | 0 |
| 5  | 5  | 4.59s  | 6.97s  | 2.29s  | 4.25s  | 0 |
| 10 | 10 | 8.21s  | 9.63s  | 5.76s  | 5.87s  | 0 |
| 20 | 20 | 10.50s | 11.51s | 7.54s  | 8.14s  | 0 |
| 50 | 50 | 15.76s | 19.56s | 12.28s | 17.13s | 0 |

Zero request failures at every level tested — the server never returned an error or
dropped a connection, even at 50 simultaneous streaming requests. But latency degrades
sharply: p95 total latency goes from 4.4s (serial) to 19.6s at concurrency 50 — a ~4.4x
latency increase for a 50x concurrency increase. More strikingly, **time-to-first-token**
— the exact metric v3 §1's streaming work was added to fix — degrades from p50 2.2s
(serial) to p50 12.3s at concurrency 50. Under real concurrent load, streaming's UX
benefit (tokens appear as generated instead of a blank screen) is substantially eroded:
users would wait 12+ seconds staring at nothing before the first token arrives, not far
off from the pre-streaming blocking experience v3 §1 was measuring against.

Server-side stage breakdown for this run (from `/metrics`, `retrieval_s`/`rerank_s` cover
the full concurrent window, not just level 50):

| Stage | p50 | p95 |
|---|---|---|
| retrieval_s | 3.88s | 6.80s |
| rerank_s    | 4.13s | 6.28s |
| generation_s| 2.72s | 3.64s |

**Finding:** under concurrency, `retrieval_s` and `rerank_s` (the embedding + Pinecone +
Cohere network calls) inflate more than `generation_s` (the Claude call) — the opposite
of the single-request baseline in CLAUDE.md's v3 §1 note, where generation dominated.
This is consistent with (not proven by this benchmark alone) both `/query` and
`/query/stream` being defined as synchronous (`def`, not `async def`) route handlers —
FastAPI/Starlette dispatches sync handlers to a bounded thread pool, so once concurrent
in-flight requests approach that pool's size, later requests queue before their network
calls even start. Worth a follow-up investigation, not concluded here.

## 3. Cache-cold snapshot (`/query`, N=5)

5 distinct real questions (never reused elsewhere in this benchmark), fired concurrently
in one wave — per V4.md's explicit "keep the cold path small-N" guidance, this is 5
samples, not a real percentile distribution. Real cost: 5 calls total.

| | |
|---|---|
| Requests | 5 |
| Errors | 0 |
| p50 | 6.57s |
| p95 | 9.81s |
| min / max | 4.15s / 10.50s |

Consistent with the streaming concurrency=5 row above (p95 6.97s) — both independently
land in the same ~5-10s range for 5 concurrent real pipeline runs.

## Practical ceiling for the free-tier single-instance deployment

Cache-hit traffic (scenario 1) is not the constraint — it comfortably handles 50+
concurrent requests. The real ceiling is **concurrent uncached traffic** (any
`/query/stream` request, or a `/query` cache miss): using a 10-second p95 as a reasonable
UX bar for a chat app that already streams, that threshold is crossed somewhere between
concurrency 5 (p95 6.97s) and concurrency 10 (p95 9.63s, right at the edge). **Practical
ceiling: roughly 5-10 simultaneous in-flight uncached requests** before p95 latency
exceeds what's likely acceptable for an interactive demo.

Note this is a similar order of magnitude to the app's existing per-IP rate limit
(10 req/min) — that limit was set for abuse prevention and predates this benchmark, not
derived from it, but this is an independent confirmation that it also happens to sit
near the server's actual uncached-throughput ceiling, not just an arbitrary number.

## Caveats (measured, not assumed — same standard as the rest of this project)

- Local machine, not Render's actual container — real deployed numbers will differ,
  likely worse given Render's free-tier CPU/network constraints.
- Dev-default Haiku model, not the deployed Sonnet — deployed `generation_s` will be
  higher than what's shown here.
- The cache-cold and cold-concurrency=5 numbers are N=5 — informative, not a real
  percentile distribution.
- The thread-pool-contention explanation for the retrieval/rerank slowdown under load is
  a plausible hypothesis based on how FastAPI dispatches sync route handlers, not a
  root-caused, isolated finding — no experiment here varied thread pool size directly.
- All error rates were 0% at every concurrency level tested, up to 50. This benchmark
  did not find a breaking point where the server returns errors or drops connections —
  only a latency-degradation curve. A higher concurrency ceiling than 50 was not tested,
  to keep real API cost bounded per V4.md's explicit guidance.
