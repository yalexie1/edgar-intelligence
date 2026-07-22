# v4 — Detailed Rundown

KEEP THIS FILE UNTRACKED.

Everything shipped in v4, across three commits plus one not-yet-committed branch:

- `6a01606` — request tracing & latency observability (§1), merged to `main` via PR #3 (2026-07-21)
- `bc5ca9c` — CI/CD pipeline with eval-gated merges (§3), merged to `main` via PR #3 (2026-07-21)
- `7144cad` — load testing & performance benchmarking (§2), merged to `main` via PR #4 (2026-07-22)
- _(not yet committed)_ — fix and verify RAGAS LLM-judge scoring (§4), branch `v4-ragas-fix` (2026-07-22)

v4 originally proposed 5 items (`V4.md`, now removed — see the note on item #5 at the end
of this document). 4 of 5 are done. Item #5 (dependency pinning) was deliberately
descoped, not attempted and abandoned. This document is a from-scratch walkthrough of
what changed across all four shipped items, why, and what to watch out for — written for
someone who wasn't in the room for any of the three sessions.

**Resume lines:**

- Instrumented the full request path with per-stage latency tracing (embedding +
  retrieval, cross-encoder rerank, generation) and structured JSON logging, replacing
  every previously-manual, one-off latency claim with a live `GET /metrics` endpoint
  reporting rolling p50/p95/p99 per pipeline stage.
- Built an async load-testing harness (`httpx` + `asyncio`) and used it to establish,
  for the first time, the free-tier single-instance deployment's real concurrency
  ceiling: cache-hit traffic scales cleanly to 50 concurrent requests (p95 161ms), while
  uncached traffic degrades sharply under load (p95 4.4s → 19.6s from 1 → 50 concurrent
  requests), landing a practical ceiling around 5-10 simultaneous uncached requests.
- Set up GitHub Actions CI: a free, zero-network unit-test workflow on every push/PR, plus
  a paid, eval-gated regression check scoped to merge-to-main that fails the build if the
  110-case golden-set pass rate drops below 97% — confirmed working end-to-end on real
  GitHub infrastructure (108/110, 98.2%, in 11m48s on first real run).
- Root-caused and fixed two distinct, real bugs behind RAGAS's all-`NaN` LLM-judge eval
  results (a nest_asyncio/`asyncio.current_task()` incompatibility, and a separate
  NaN-vs-None aggregation bug), then discovered and fixed a third, more subtle
  measurement problem: RAGAS's four metrics don't fit 60% of this project's own golden
  set by construction, so the dashboard headline is now correctly scoped to the question
  group RAGAS's methodology actually applies to, with full transparency into every
  group's real score.

---

## 1. New libraries / tools

| Package | Where added | Purpose |
|---|---|---|
| `httpx` (explicit pin) | `requirements.txt` | Was already a transitive dependency of other packages; pinned explicitly because `bench/load_test.py` (§2) now imports and uses it directly for the async load-testing harness. |

`ragas`, `datasets`, and `pandas` (the optional RAGAS dependencies) are not new in v4 —
they were already present, commented out, in `requirements.txt` before v4 started. v4 §4
fixed their *usage*, not their presence. No other new packages were added anywhere in v4.

**New environment variables:** none. v4 added no new API integrations — everything
(tracing, load testing, CI, RAGAS) uses infrastructure already in the stack.

---

## 2. Files touched, by item

```
§1 (6a01606)  api.py                       |  ~+121
              ask.py                       |  ~+84
              tests/test_pure.py           |  +130
              3 files changed, 363 insertions(+), 28 deletions(-)

§3 (bc5ca9c)  .github/workflows/eval.yml   |  +74
              .github/workflows/tests.yml  |  +26
              README.md                    |  +2
              3 files changed, 102 insertions(+)

§2 (7144cad)  bench/load_test.py           |  +288
              bench/__init__.py            |  +0
              bench/results/*.json (3)     |  +267
              BENCHMARKS.md                |  +128
              requirements.txt             |  +1
              tests/test_bench.py          |  +123
              .gitignore                   |  +1
              9 files changed, 808 insertions(+)

§4 (uncommitted, v4-ragas-fix)
              evals/eval_ragas.py              |  294 ++++++++--
              evals/results/ragas_results.json | 1194 ++++++++++++++++++++++++++--
              dashboard.html                   |  153 ++++-
              evals/results/ragas_results.csv  |  120 +++-
              evals/results/ragas_summary.json |   98 +++-
              README.md                        |   14 +-
              requirements.txt                 |    4 +-
              tests/test_eval_ragas.py         |  new (19 tests)
              evals/__init__.py                |  new (empty, makes evals importable)
              7 files changed, 1728 insertions(+), 149 deletions(-) + 2 new files
```

`ingest.py` and `embed_and_search.py` — the two FROZEN files — were **not touched** in any
part of v4. No re-ingest or `--rebuild` was needed; the corpus (8,141 chunks, 13
companies) is unchanged. `CLAUDE.md` and `DEVLOG.md` were updated after every item (they're
gitignored, so they don't appear in the diffs above, but both were kept current per this
project's own convention).

---

## 3. §1 — Request Tracing & Latency Observability

**Problem:** zero structured logging anywhere in the request path. Every latency claim in
this project up to v3 (e.g. "~7.1s time-to-first-token, retrieval-bound") was a single
manual, one-off measurement — not something the running system could report on its own,
continuously, or broken down by stage.

**What was added (`ask.py`):** `ask()` and `ask_stream()` gained an optional
`timing: dict | None = None` parameter, threaded through every `_prepare_*`/`_ask_*`
helper. A new `_record(timing, key, start)` helper (pure, tested) mutates the dict in
place with `route` and whichever of `retrieval_s` / `rerank_s` / `generation_s` actually
ran — a stage key is simply absent when that stage didn't run (diverse mode and the
temporal/section-diff paths skip Cohere rerank; an abstain skips generation). `ask()`'s
existing 3-tuple return contract is unchanged — `timing` is an output parameter, so every
existing caller (CLI, evals, the LRU cache) pays zero cost by simply not passing a dict.

**What was added (`api.py`):** adopted the standard `logging` module (previously unused
entirely). Both `/query` and `/query/stream` now emit one structured JSON line per
request: `request_id` (`uuid4`), `endpoint`, `route`, `cache_hit`, `num_results`, the full
question text, `total_s`, and whichever stage timings are present. New pure helpers
`_build_log_payload()` and `_percentile()`/`_metrics_summary()` back a new `GET /metrics`
endpoint: a bounded `deque(maxlen=200)` rolling window of recent per-request timings,
aggregated into p50/p95/p99 per stage plus a route-count breakdown. `request_id` is an
additive field on both `/query`'s response and `/query/stream`'s `sources` event.

**A deliberate privacy call:** logs the full question text. SEC filing Q&A is
low-sensitivity content, and full text is far more useful for debugging routing/retrieval
issues than a truncated one.

**A real bug found via live testing, fixed same session:** `/metrics`'s route breakdown
initially mislabeled cache hits as `"unknown"` — the `/query` handler resolved a local
`route` variable for the *log line*, but pushed the raw (empty, on a cache hit) `timing`
dict onto `_metrics_window` without that same resolved value. Fixed by explicitly setting
`"route": route` when appending to the metrics window.

**Verified live:** ran 4 real queries (plain, temporal, cross-company, cache-hit repeat)
against the real Pinecone index. `retrieval_s` ranged ~0.79s–2.59s, `generation_s`
~1.85s–8.22s. `rerank_s` correctly absent for temporal/cross_company routes (no Cohere
call there); the cache-hit repeat logged `total_s: 0.0` with no stage keys.

**Test coverage:** 18 new pure-logic cases in `tests/test_pure.py`
(`TestRecordTiming`, `TestPercentile`, `TestMetricsSummary`, `TestBuildLogPayload`) — the
network-touching wiring inside `_prepare_*`/`ask()` itself follows this project's existing
convention of live spot-check verification rather than unit mocking. Full suite: 116/116.

**Known limitation carried forward:** retrieval is timed as one combined stage
(embedding + Pinecone ANN) rather than split, since both happen inside frozen
`embed_and_search.py`'s `search()` — splitting further would require modifying a frozen
file, which needs separate confirmation.

---

## 4. §3 — CI/CD Pipeline with Eval-Gated Merges

**Problem:** no CI/CD configuration anywhere in the repo — not even the free, zero-network
unit suite ran automatically. Every test and eval run was a manual local invocation.

**What was added:**
- `.github/workflows/tests.yml` — runs on every push and PR: checkout, Python 3.11 setup
  (matching `render.yaml`'s deployed version), install, `pytest tests/ -v`. No secrets
  required — the suite is zero-network.
- `.github/workflows/eval.yml` — deliberately scoped to `push: branches: [main]` and
  `workflow_dispatch` only, never every PR push, since each run costs real
  Anthropic/OpenAI/Pinecone/Cohere money across 110 cases. Starts `uvicorn api:app` in the
  runner (no local index build needed — the corpus is already committed and Pinecone is
  already cloud-hosted), polls `/health` for readiness, runs `python evals/eval.py`, then
  fails the build if `evals/last_results.json`'s `pass_rate` drops below a 0.97 floor.
  `eval.py` itself was not modified — the threshold check lives entirely in the workflow
  YAML. Eval results are uploaded as a workflow artifact (`if: always()`).
- `README.md` — added the `tests.yml` status badge. Deliberately no `eval.yml` badge yet
  at the time (no real run existed until the 4 required secrets were added and something
  triggered it).

**Not done in this pass:** `eval.yml`'s 4 required secrets (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`) were deliberately not added —
that's a user action against the GitHub repo settings, not something to automate into a
session that hadn't pushed anything yet.

**Confirmed live, same day:** the user added the 4 secrets, pushed the branch, and merged
PR #3. `tests.yml` passed on the push (24s). The merge to `main` triggered `eval.yml` for
the first time ever — it passed: 108/110 (98.2%), comfortably above the 0.97 floor, in
11m48s. Both workflows confirmed working end-to-end on real GitHub infrastructure, not
just validated locally.

---

## 5. §2 — Load Testing & Performance Benchmarking

**Problem:** no load-testing script existed anywhere in the repo. The project had never
established how many concurrent users the single-instance Render free-tier deployment
(no autoscaling, `uvicorn` run directly) can actually serve before latency degrades.

**What was added (`bench/load_test.py`):** an async `httpx` harness with a hard
localhost-only guard (`_assert_local()` raises `SystemExit` on any `--base-url` whose host
isn't `127.0.0.1`/`localhost`/`::1`) so it cannot accidentally hammer the deployed Render
API. Three scenarios, all against a **locally running** `uvicorn api:app`, never Render:
- **warm** (`/query`): primes the LRU cache once, then sweeps a concurrency ladder
  (default `1,5,10,20,50`) against that same cached tuple — near-zero marginal API cost.
- **cold** (`/query`): 5 distinct real questions fired concurrently in one wave — bounded
  to exactly 5 real calls.
- **stream** (`/query/stream`): same concurrency ladder, one wave per level. **Real
  finding, discovered by reading `api.py` before building this, not assumed:**
  `/query/stream` has no cache tier — every one of these requests is real, billed traffic
  regardless of repeated question text. Also records time-to-first-token per request.

Each scenario also pulls the server's own `GET /metrics` (§1) after running, so
client-observed wall-clock latency and server-side per-stage breakdown are both captured
for the same traffic.

**Verified live (real run):** started a local server against the real 8,141-vector
Pinecone index and ran all three scenarios for real.
- **Warm**: scales cleanly to 50 concurrent requests — p95 161ms, zero errors at every
  level.
- **Stream**: degrades sharply under concurrency — p95 total latency 4.4s (concurrency 1)
  → 19.6s (concurrency 50); TTFT p50 2.2s → 12.3s. Server-side `/metrics` showed
  `retrieval_s`/`rerank_s` inflating more under load than `generation_s` — the opposite of
  the single-request baseline, flagged as consistent with (not proven by) synchronous
  route handlers contending for FastAPI/Starlette's bounded sync-handler thread pool.
- **Cold** (N=5): p50 6.57s, p95 9.81s — lands in the same range as the stream scenario's
  concurrency=5 row.
- **Zero request failures** at any tested level up to 50 — only a latency-degradation
  curve, no found breaking point.
- **Practical ceiling:** using a 10s p95 as a reasonable UX bar, that threshold is crossed
  between concurrency 5 (p95 6.97s) and concurrency 10 (p95 9.63s) — roughly **5-10
  simultaneous in-flight uncached requests** is the practical ceiling for this single
  free-tier instance.

**Test coverage:** 17 new pure-function cases (`parse_levels`, `percentile`,
`summarize_run`) in `tests/test_bench.py`, zero network/mocking. Full suite: 133/133.

**Not done / caveats:** measured on the local dev machine's network path, not Render's
actual container. Used the dev-default Haiku model, not the deployed Sonnet — real
`generation_s` on Render will be higher. The cold and concurrency=5 numbers are N=5,
informative but not a real percentile distribution. The thread-pool-contention
explanation is an unproven hypothesis. Total real cost across the whole benchmark: ~92
real LLM calls at dev-default Haiku pricing.

---

## 6. §4 — Fix and Verify RAGAS LLM-Judge Scoring

**Problem:** `evals/eval_ragas.py`'s last recorded run (2026-06-23) showed all four
metrics as `NaN`. CLAUDE.md attributed this to "blocked on Python 3.14 + nest_asyncio" —
a direct check found `ragas==0.2.15` imports cleanly under Python 3.14.2 with no error,
contradicting that framing outright. This was genuinely an investigation first, a fix
second.

**Bug #1 — the actual cause of the `NaN` scores.** `ragas.executor` applies
`nest_asyncio.apply()` unconditionally at import time, on every Python version (not
3.14-specific). This swaps `asyncio.Task` for a pure-Python re-entrant implementation, but
`asyncio.current_task()` remains the C-accelerated builtin, which has no visibility into
that pure-Python task — confirmed via a minimal repro (`nest_asyncio.apply();
asyncio.run(...)` → `current_task()` prints `None`, even mid-task). Every downstream
consumer gated on "`current_task()` is not `None`" breaks identically. A prior session's
patch covered exactly one such consumer (`asyncio.timeouts.Timeout.__aenter__`) and missed
the one that actually produces the `NaN` scores: `sniffio.current_async_library()`, called
deep inside `anyio`/`httpcore` on every async OpenAI/Anthropic call ragas makes as a judge.

**Fix:** patch `asyncio.current_task` itself (in `evals/eval_ragas.py`, before
`ragas`/`openai`/`anthropic`/`anyio` are imported) to fall back to the real running task
from `asyncio.tasks._current_tasks` — a legacy bookkeeping dict nest_asyncio's pure-Python
task still populates, invisible to the C builtin but genuinely correct. A first fix
attempt using a fake `Task` stand-in got further but broke differently
(`AttributeError: '_FakeTask' object has no attribute '_must_cancel'` from `anyio`'s
cancellation delivery) — which is what led to recovering the real task instead of
fabricating one.

**Bug #2 — found only after a full 110-case run.** Even with bug #1 fixed,
`faithfulness` and `context_recall` still showed `NaN`. Root cause: a real 110×4=440-call
run legitimately hits OpenAI `gpt-4o-mini` rate limits and occasional timeouts — 18 of 440
judge calls failed this way, which ragas records as a per-case `NaN` rather than aborting.
`eval_ragas.py`'s own aggregation filtered `None` but not `NaN`, so even one `NaN` in 110
silently poisoned the *entire* metric mean — indistinguishable from bug #1's "everything
fails" case without this second fix. Fixed with a new `_mean_ignoring_nan()` and a
`metrics_n_scored` field so the dashboard can show "N/110 scored."

**First real, non-`NaN` scores this project has ever recorded** (full 110-case run,
all-groups aggregate): faithfulness 0.7993 (103/110 scored), answer_relevancy 0.5073
(110/110), context_precision 0.3296 (110/110), context_recall 0.5303 (99/110 scored).

**Bug #3 (a measurement/methodology problem, not a code bug) — found the same day, when
asked why these numbers looked so bad.** Segmenting the already-collected per-case scores
by question group (free, no new LLM calls) showed the low aggregate wasn't spread evenly:

| Group | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| **factual** | 44 | 0.912 | 0.692 | 0.550 | 0.524 |
| abstain | 26 | 0.777 | 0.000 | 0.038 | 0.880 |
| cross_company | 22 | 0.550 | 0.656 | 0.278 | 0.294 |
| temporal | 18 | 0.816 | 0.606 | 0.274 | 0.233 |

RAGAS's four metrics all assume a single question answered from a single retrieval pass —
true only for `factual`. `abstain` cases (26/110) are *supposed* to retrieve weak/no
evidence and answer "the filings don't cover this" — `answer_relevancy` (built by
generating questions from the answer and comparing back to the original) scores that as
0.000 by construction, and `context_precision` scores the deliberately-weak evidence as
0.038, neither reflecting a real failure. `cross_company`/`temporal` cases (40/110)
deliberately retrieve chunks spanning multiple companies/periods for comparison — mostly
"irrelevant" to any single sub-claim by a per-chunk judge, but structurally necessary for
the comparison the question asked for. Averaging all four groups into one number mixes
"the system worked as designed" with "the system answered a normal question poorly" into
one indistinguishable mean.

**Fix:** added `RAGAS_APPLICABLE_GROUPS = ["factual"]` and `build_summary()` (extracted
from `run()`), computing three things from the same per-case data: `metrics` (all 110,
unchanged, kept for transparency), `metrics_applicable` (factual-only, the new headline),
and `metrics_by_group` (per-group breakdown, all 4 groups, nothing hidden). Added a
`--recompute-only` CLI flag to rebuild `ragas_summary.json`/`ragas_results.json`/`.csv`
from an already-saved run's per-case scores at zero new LLM cost — used to apply this
schema change to the existing full-run data instead of re-spending ~550 real calls.
`dashboard.html`'s RAGAS section now shows the factual-only headline cards, a detailed
callout explaining what each of the four metrics measures (and what a high score does and
doesn't mean) and why the headline is scoped this way, and a collapsible "full breakdown
by question group" table underneath.

**Final, honest headline numbers (factual group, n=44):** faithfulness 0.912
(42/44 scored), answer_relevancy 0.692, context_precision 0.550, context_recall 0.524
(42/44 scored). Even scoped this way, `answer_relevancy` and `context_precision` aren't
high — that's a real, undiagnosed retrieval-quality question, not resolved by the scoping
fix, and remains open.

**Test coverage:** 19 cases in `tests/test_eval_ragas.py`, zero network/LLM calls,
skipped entirely via `pytest.importorskip` when ragas/datasets/pandas aren't installed:
`TestPatchApplied` (3), `TestPatchedCurrentTask` (3, including a real behavior caught by
an initial wrong test assumption — `asyncio.current_task()` *raises* `RuntimeError`
outside any event loop rather than returning `None`), `TestDownstreamConsumers` (2 — a
third, planned `anyio.TaskGroup` test was dropped after it surfaced a pytest-only
artifact: `anyio` ships a `pytest11` plugin that pytest auto-loads before any test module
is collected, eagerly binding `current_task` before this patch has a chance to run;
confirmed this doesn't affect real `python evals/eval_ragas.py` usage), `TestMeanIgnoringNan`
(6), and `TestBuildSummary` (5, covering the group-scoping logic). Full suite: 152/152.

---

## 7. Item #5 (Dependency Pinning & Reproducible Environment) — not implemented

`V4.md` originally proposed pinning `requirements.txt` to exact versions and closing the
discovered local-dev-vs-Render Python version drift (3.14.2 vs 3.11.0), optionally with a
`Dockerfile`. The user decided not to pursue this item. No work was done on it beyond the
original proposal in the now-removed `V4.md` — no lockfile, no `.python-version` file, no
`Dockerfile`. The dev/prod Python version mismatch remains open and undocumented as a
fix-in-progress; it's simply not being worked on.

---

## 8. Testing

| | Before v4 | After §1 | After §3 | After §2 | After §4 |
|---|---|---|---|---|---|
| Unit tests (`tests/`) | 98 | 116 | 116 | 133 | **152** |

All new tests across v4 are pure-function / zero-network-call tests, matching the existing
suite's convention (`test_eval_ragas.py` additionally skips cleanly when the optional
ragas/datasets/pandas dependencies aren't installed). The network-touching orchestration
in `ask.py`/`api.py`/`bench/load_test.py`/`evals/eval_ragas.py` itself isn't
unit-mocked — each item was instead verified via live spot-checks against the real
Pinecone/Anthropic/OpenAI/Cohere stack, the same convention this project has followed
since before v4.

---

## 9. Eval results, run-by-run

| Run | Cases | Overall | Notes |
|---|---|---|---|
| Pre-v4 baseline (v3 final) | 110 | 107/110 (97%) | Unchanged by §1/§2/§3 — none touch retrieval or prompt logic |
| After §1+§3 merge (CI's first real run) | 110 | 108/110 (98.2%) | Confirmed via real GitHub Actions run, 11m48s |
| After §2 | 110 | — (not re-run) | Load testing doesn't touch retrieval/prompt logic; no re-run needed |
| RAGAS (§4), first real non-`NaN` run, all-groups | 110 | faithfulness 0.799, answer_relevancy 0.507, context_precision 0.330, context_recall 0.530 | Deterministic suite unaffected — this is the separate, complementary LLM-judge layer |
| RAGAS (§4), factual-only headline (final) | 44 | faithfulness 0.912, answer_relevancy 0.692, context_precision 0.550, context_recall 0.524 | Same underlying run, correctly re-scoped |

The deterministic golden-set suite (`evals/eval.py`) was not modified by any v4 item —
v4 only added tracing, benchmarking, CI, and a RAGAS fix. Its pass rate moved only because
CI caught a slightly different real-world run (108/110) than the last local run recorded
in CLAUDE.md before v4 (107/110) — normal run-to-run LLM answer variance, not a
regression from any v4 change.

---

## 10. Known limitations added or updated in v4 (see `CLAUDE.md` for the full list)

- **Request tracing (§1):** retrieval is timed as one combined stage (embedding +
  Pinecone ANN) rather than split, since both happen inside frozen `embed_and_search.py`.
- **Load testing (§2):** measured on the local dev machine, not Render's container;
  used the dev-default Haiku model, not the deployed Sonnet; the thread-pool-contention
  explanation for retrieval/rerank slowdown under load is an unproven hypothesis.
- **CI/CD (§3):** `eval.yml` needs 4 GitHub Actions repo secrets to run — added by the
  user post-merge, not automated.
- **RAGAS (§4):** a full 110-case run legitimately hits OpenAI rate limits/timeouts on a
  handful of judge calls (expected, not a bug); the headline is scoped to the `factual`
  question group by design, since RAGAS's methodology doesn't fit abstain/cross_company/
  temporal cases — full per-group scores remain visible, nothing hidden. Even within the
  scoped `factual` group, `answer_relevancy` (0.692) and `context_precision` (0.550)
  aren't high — an open, undiagnosed retrieval-quality question.
- **Item #5** (dependency pinning): not implemented — the dev/prod Python version
  mismatch (3.14.2 vs 3.11.0) remains open.

---

## 11. What was explicitly NOT measured / NOT done (stated honestly, not oversold)

- Request tracing (§1): the ~7.1s TTFT figure from v3 wasn't replaced with a new single
  number — only 4 live spot-check samples were taken, enough to confirm the
  instrumentation works, not enough to produce a new aggregate claim.
- Load testing (§2): no concurrency level above 50 was tested, to keep real API cost
  bounded. The cold-path and concurrency=5 numbers are N=5, not a real distribution.
- CI/CD (§3): no `eval.yml` status badge was added until a real run existed with the
  secrets in place.
- RAGAS (§4): the *reason* `answer_relevancy`/`context_precision` are moderate even
  within the correctly-scoped `factual` group was not diagnosed — that would be a
  retrieval-quality investigation, a plausible next step, not undertaken here.
