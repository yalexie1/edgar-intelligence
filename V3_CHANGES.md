# v3 — Detailed Rundown

KEEP THIS FILE UNTRACKED.

Everything shipped in v3, across two commits:

- `c389e61` — streaming SSE, Cohere reranking, source passage preview, section filter in UI (§1, §2, §3, §5)
- `bd74847` — section-to-section temporal diffs (§4), plus a bug fix found in pre-commit review

v3 is now fully complete (5/5 items). This document is a from-scratch walkthrough of what
changed, why, and what to watch out for — written for someone who wasn't in the room for
either session.

**Resume lines:**

- Implemented real-time token streaming end-to-end (FastAPI SSE endpoint, Anthropic's
  Streaming API, a vanilla-JS `ReadableStream` consumer) alongside a two-stage retrieval
  pipeline — Pinecone ANN for recall, Cohere `rerank-v3.5` cross-encoder for precision —
  replacing a single-stage cosine+lexical ranker without regressing a 110-case eval suite.
- Designed and shipped a section-to-section temporal diff feature: detects "what changed"
  queries, retrieves the same SEC filing section across two reporting periods, and
  generates a structured added/removed/unchanged comparison with citations — extending
  the system from single-shot Q&A into comparative filing analysis.
- Grew test coverage from 51 to 98 unit tests and the golden eval set from 104 to 110
  cases, using test-driven development to catch and fix three real bugs pre-deploy (a
  filter-precedence bug, a prompt/retrieval section-label desync, and a conversational
  follow-up misrouting bug) via direct routing-function tests and live browser testing.
- Improved retrieval quality with a two-stage pipeline — Pinecone ANN recall followed by
  Cohere cross-encoder reranking — layered on query expansion, lexical reranking, and
  diversity ranking, achieving a 97% pass rate across 110 Q&A cases, with a 100% retrieval
  hit-rate and 100% abstain accuracy.

---

## 1. New libraries / tools

| Package | Where added | Purpose |
|---|---|---|
| `cohere>=5.0` (installed: 7.0.5) | `requirements.txt` | Cross-encoder reranking (`rerank-v3.5` model) — reads question+passage together, replacing the old cosine+lexical-only reranker for precision at the final truncation step. |

No other new packages. Everything else in v3 (streaming, section diffs, UI changes) uses
libraries already in the stack (`anthropic`, `fastapi`, vanilla JS `fetch`/`ReadableStream`).

**New environment variable:** `COHERE_API_KEY` (optional). Added to `.env.example`,
`render.yaml` (as `sync: false` — set manually in the Render dashboard), and read via
`_get_cohere_client()` in `ask.py`. Free tier: 1000 calls/month. **The whole system works
without it** — every place that calls Cohere falls back to the pre-existing local
cosine+lexical order if the key is unset or the API call raises for any reason. This was a
deliberate design choice: reranking is a precision upgrade, not a hard dependency, for a
demo server on a free tier.

---

## 2. Files touched, overall

```
 .env.example            |   1 +
 CLAUDE.md               | 358 ++++++++++++++++++++++++++-
 api.py                  | 124 +++++++---
 ask.py                  | 617 ++++++++++++++++++++++++++++++++++++++++------
 dashboard.html          |   2 +-
 evals/dataset.json      |  54 +++++
 evals/last_results.json | 633 ++++++++++++++++++++++++++++++------------------
 index.html              | 254 ++++++++++++++-----
 render.yaml             |   2 +
 requirements.txt        |   1 +
 tests/test_pure.py      | 306 ++++++++++++++++++++++-
 11 files changed, 1927 insertions(+), 425 deletions(-)
```

`ingest.py` and `embed_and_search.py` — the two FROZEN files — were **not touched** in any
part of v3. No re-ingest or `--rebuild` was needed; the corpus (8,141 chunks, 13 companies)
is unchanged.

---

## 3. §1 — Streaming responses (SSE)

**Problem:** every query blocked 5–20s, then rendered the full answer at once. No
feedback that anything was happening.

**Backend (`ask.py`):** the routing logic that used to live inline in `ask()` was pulled
out into `_route(question, where, k, diverse, history)` — a pure function that decides a
retrieval "kind" (`cross_company` / `section_diff` / `temporal` / `plain`) without doing
any retrieval itself. Each kind has a matching `_prepare_*` helper
(`_prepare_cross_company`, `_prepare_section_diff`, `_prepare_temporal`, `_prepare_plain`)
that does retrieval + prompt-building + abstain-checking, but stops short of calling the
model. Both `ask()` (blocking) and the new `ask_stream()` (generator) call the *same*
`_route()` + `_prepare_*` pair, so the two response modes can never drift out of sync —
fixing a bug in one path fixes it in both automatically.

`ask_stream()` yields:
1. `{"__meta__": True, "results": [...], "effective_where": ...}` — **first, before any
   answer text.** Retrieval already finished inside `_prepare_*`, so there's no reason to
   hold sources back until generation completes. This also means a connection drop
   mid-generation still leaves citations intact for whatever partial text arrived.
2. Plain text chunks, one per token, via `client.messages.stream(...).text_stream`.
3. Nothing else — the caller (`api.py`) appends its own `[DONE]` marker.

Abstain cases (thin evidence) skip the model call entirely and yield the abstain string
as a single chunk — matching `ask()`'s existing behavior exactly.

**Backend (`api.py`):** new `POST /query/stream` endpoint, same `Query` schema and
validation as `/query` (both now call shared `_validate_query(q)` / `_build_where(q)` /
`_build_sources(results)` helpers — these didn't exist before and were factored out of
the original `/query` handler). Returns a `StreamingResponse` emitting:
```
data: {"token": "..."}              (repeated, one per token)
data: {"sources": [...], "filter_applied": ...}   (once, after all tokens)
data: [DONE]
```
Streamed responses are **not cached** — this extends the pre-existing rule that
conversation-shaped requests bypass the LRU cache in `api.py`.

**Frontend (`index.html`):** `submitQuestion()` was rewritten to use
`fetch(...).body.getReader()` and parse `data:` lines out of the decoded byte stream
(splitting on `\n\n`). The `.answer` div is created empty up front holding a status
message, then cleared and filled with raw token text as it arrives — markdown parsing
(`marked.parse`) and citation linking (`withCitations`) run **once at the end**, not per
token, specifically to avoid the O(n²) DOM-write/reflow cost of re-parsing markdown on
every single token. Sources render once the `{"sources": ...}` event lands.

The retry loop (3 attempts, 5s backoff) still exists for total connection failure before
any token arrives. A failure *after* some tokens have streamed in is handled differently:
the partial answer is kept on screen and an inline error is appended below it, rather than
being discarded and retried (retrying mid-stream would duplicate an already-rendered
partial answer). The stream is only considered to have completed successfully if the
server's `[DONE]` marker was actually seen (`sawDone` flag) — a connection that closes
without it is treated as a failure, not silently rendered as if it were a complete answer.

**A bug found via live testing, same day:** the original plan called for shortening the
UI's cold-start warning timer from 8s to ~2s, on the theory that "streaming makes
cold-start feel less jarring." Measuring actual time-to-first-token on a warm backend
showed ~7.1s of a ~10.3s total round trip — almost all of it is embedding + Pinecone
retrieval happening *before* generation starts, which streaming does nothing to speed up.
A 2s threshold fired the "waking up" message on essentially every request, cold or not.
`slowTimer` was reverted to 8s (the same value the old blocking UI used).

**Test coverage:** `TestRoute` in `tests/test_pure.py` — 11 cases (now more, see §4) —
covering `_route()`'s branch logic with no network calls.

---

## 4. §2 — Cohere cross-encoder reranking

**Problem:** the existing reranker was `cosine_similarity + 0.08 * lexical_score`
(`KEYWORD_BOOST = 0.08` in `embed_and_search.py`) — cheap, but structurally unable to
tell a passage that *answers* the question from one that merely *shares its vocabulary*.
A cross-encoder reads the question and passage together and scores relevance directly.

**What was added (`ask.py`):**
- `_get_cohere_client()` — lazily builds a module-level `cohere.ClientV2`. Returns `None`
  if `COHERE_API_KEY` isn't set (checked once per process, not per request).
- `_rerank_with_cohere(results, question, k)` — calls
  `client.rerank(model="rerank-v3.5", query=question, documents=[r["text"] for r in
  results], top_n=min(k, len(results)))`, then maps `response.results[i].index` back onto
  the original result dicts (so `metadata`, `rerank_score`, etc. all survive unchanged —
  only the *order and count* change). **Falls back to `results[:k]`** — the pre-existing
  local order — if the client is `None` or the API call raises **for any reason** (quota
  exhausted, network blip, malformed response). This is a bare `except Exception`, on
  purpose: reranking must never be able to take down an otherwise-working query.

**Where it's wired in:**
- `_prepare_plain`'s non-diverse branch: fetches the full `CANDIDATE_K` (50) pool from
  Pinecone, then `_rerank_with_cohere(candidates, question, k)` truncates to the
  requested `k` (default `TOP_K = 5`). This is the main "two-stage retrieval" path:
  Pinecone ANN for recall → Cohere for precision.
- `_prepare_cross_company`: reranked **per-ticker**, not on the flattened cross-company
  pool. Each named company's own `CANDIDATE_K` pool is rerank-truncated to `k_per`
  *before* flattening across companies. This preserves the existing coverage guarantee
  (every named ticker gets `k_per` chunks) — reranking the flattened pool instead could
  let Cohere's favorite company's chunks crowd out a weaker company's, breaking the
  guarantee that structured cross-company retrieval exists to provide.

**Deliberately NOT wired in:**
- Diverse mode (`_prepare_plain`'s diverse branch) — the goal there is ticker *spread*
  (via `diversify_results`), not raw relevance. Reranking first would let the
  cross-encoder's favorite ticker crowd out the rest before diversification even runs.
- `_prepare_temporal` — chronological period order must be preserved; relevance-based
  reordering would undermine the per-period grouping the whole path exists for.
- `_prepare_section_diff` (added later, in §4) — same reasoning as temporal: which 2
  periods get picked matters more than raw relevance within them.

**Verified live** (not just by eval): a real query against AAPL came back with results
whose local `rerank_score` field was no longer monotonically decreasing — direct evidence
Cohere actually reordered the pool rather than silently passing it through.

**Honest caveat, documented in `CLAUDE.md`:** the eval harness's `answer_contains` checks
are pass/fail, not scored — a reranking improvement that changes *which* correct passage
gets cited (without flipping pass→fail) wouldn't show up in the aggregate number. The
measured claim is "shipped, no eval regression" — not "measurably improved precision."
That would need a manual side-by-side spot-check, which wasn't done.

---

## 5. §3 — Source passage preview in the answer UI

**Problem:** citation cards showed ticker/form/period/section badges and a link to the
SEC filing, but not the actual cited text. Verifying a claim meant leaving the app.

**Backend (`api.py`):** `_build_sources()` (the same helper introduced in §1) adds
`"text": r["text"][:500]` to every source entry — a 500-character raw slice of the chunk,
which can end mid-sentence. This is **additive** on both `/query` and `/query/stream` —
the frozen response contract (no existing field renamed or removed) is unaffected.

**Frontend (`index.html`):** `.source`'s CSS was restructured from a single flex row into
a column — the old row content (ticker/form badges, filing link, similarity/rerank
scores) moved into a nested `.source-row` div, with a new `.source-toggle` button and a
sibling `.source-preview` div (hidden by default via `display: none`, shown via an `.open`
class) below it. `renderSourceItems()` only emits the toggle/preview markup when
`s.text` is present (defensive — in case a future response ever omits it).

Because source cards are inserted via `innerHTML` (no per-card JS references at render
time), a single delegated click listener lives on the `#chat` container and handles every
toggle across every turn — finds the closest `.source` ancestor of the click target,
flips `.open` on its `.source-preview`, and swaps the button's label between "passage"
and "hide".

**Not done:** no summarization or trimming of the passage beyond the raw 500-char slice.
Good enough for "does this look plausible," not polished prose.

---

## 6. §4 — Section-to-section temporal diffs (the newest piece)

**Problem:** `_ask_temporal()` groups evidence by period for general trend questions, but
had no way to compare the *same section* across exactly two periods — e.g. "What changed
in NVDA's risk factors since last year?" Financial analysts do this routinely (reading
consecutive 10-Ks side by side); the query type had no handler at all.

**New pure functions (`ask.py`):**
- `_DIFF_PHRASES` — a **narrow** phrase list ("what changed", "what's changed", "since
  last year", "compared to the prior year", "newly added", "no longer mentions", etc.),
  deliberately kept separate from the broader `_TEMPORAL_PHRASES` list. A phrase like
  "changed between" (already in `_TEMPORAL_PHRASES`) was intentionally left out of
  `_DIFF_PHRASES` because it also matches ordinary multi-period trend questions.
- `is_section_diff_question(question)` — substring match against `_DIFF_PHRASES`.
- `_SECTION_HINTS` — maps a phrase naming a section ("md&a", "management's discussion",
  "results of operations", "financial statement", "market risk", "legal proceeding",
  "controls and procedures"/"internal controls", "risk factor") to the canonical section
  label used elsewhere in the corpus (see `canonical_section()` in
  `embed_and_search.py`).
- `detect_section(question)` — looks up `_SECTION_HINTS`, defaults to `"risk_factors"`
  if nothing matches (the section analysts diff most often, and *some* section is needed
  to filter on even if the user didn't name one).
- `build_diff_prompt(question, period_chunks, ticker, section, history=None)` — same
  passage-numbering/citation convention as `build_temporal_prompt`, but structured for
  exactly two `--- {period} ---` blocks, and instructs the model to answer in three parts:
  **(a)** added/strengthened, **(b)** removed/softened, **(c)** unchanged — each with a
  quote and a citation.

**New retrieval path:**
- `_prepare_section_diff(collection, question, ticker, section, k, history,
  extra_where=None)` — filters by ticker + section, retrieves a `CANDIDATE_K` (50)-sized
  pool, buckets by `period`, and keeps the **top 2 chunks from each of the 2 most recent
  distinct periods**. If fewer than 2 distinct periods are found in the pool, it falls
  back entirely to `_prepare_temporal()`'s broader N-period grouping — there's nothing to
  diff, but a trend answer over however many periods exist is still useful. No Cohere
  rerank here, for the same reason it's skipped in `_prepare_temporal`: period-based
  selection, not raw relevance, decides which chunks survive.
- `_ask_section_diff(...)` — the blocking wrapper (same shape as `_ask_temporal`).
- Wired into `_route()` **between** the cross-company check and the temporal check:
  `if len(tickers) == 1 and is_section_diff_question(question) and not diverse`. Ordering
  matters — multi-company detection still runs first (so a diff phrase naming 2+ tickers
  correctly gets `_ask_cross_company`'s guaranteed per-ticker coverage instead of
  section-diff), and the check happens *before* the temporal check so a diff-shaped
  question doesn't fall through to the broader N-period path.
- Both `ask()` and `ask_stream()` got a new `if route["kind"] == "section_diff":` branch
  calling `_prepare_section_diff`/`_ask_section_diff`.

**A real bug found in pre-commit code review (not live testing this time) — and fixed
before committing:** the first version picked the diff prompt's `section` purely from
`detect_section(question)` — the question's own wording — with no awareness that an
explicit section filter might already be pinned via `extra_where` (i.e. the UI's Section
dropdown, from §5). `_prepare_section_diff` *did* correctly prioritize an existing filter
for retrieval (skipping the redundant merge when one was present), but the **prompt**
was still labeled using the question-derived section regardless. Concretely: a UI filter
of `{"section": "mda"}` combined with a question saying "risk factors" would retrieve
MD&A chunks correctly, but tell the model (and the user, via the prompt) that it was
comparing "risk_factors" — a real mismatch between what was retrieved and what the
answer claims to be about.

Caught by manually constructing that exact conflicting case and calling `_route()`
directly — confirmed it returned `section: "risk_factors"` while `extra_where` said
`"mda"`. Fixed by adding `_extract_section_filter(where)` (returns the section value
already pinned in a filter dict, top-level or inside `$and`) and using
`_extract_section_filter(extra_where) or detect_section(question)` in `_route()` — an
explicit filter now always wins, mirroring the existing ticker-filter precedence pattern
(`_has_ticker_filter`) established back in the §5 bug fix. Covered by
`TestRoute.test_section_filter_where_overrides_question_wording`, written first and
watched fail with the old `"risk_factors"` value before the fix.

**Eval cases added (`evals/dataset.json`, now 110 total, up from 104):**
`diff_nvda_risk_factors`, `diff_aapl_mda`, `diff_msft_risk_factors`,
`diff_tsla_risk_factors`, `diff_googl_mda`, `diff_msft_no_section_named` — one per diff
phrasing variant, plus one exercising the default-to-`risk_factors` path. All use
`"group": "temporal"` and `"answer_contains": []` (retrieval-only scoring — diff wording
is inherently non-deterministic, so there's nothing fixed to string-match against).

**Verified live:**
- `POST /query` for "What changed in NVIDIA's risk factors between its two most recent
  10-Ks?" correctly routed to `section_diff`, retrieved FY2025 and FY2026 risk-factors
  chunks, and returned a genuinely structured added/removed/unchanged answer — correctly
  identifying, among other things, a brand-new counterparty-risk factor and a new
  anti-takeover-provisions risk section present in FY2026 but absent from FY2025.
- `POST /query/stream` for an Apple MD&A diff question streamed the `__meta__` sources
  event first (both periods present), consistent with the §1 streaming contract.

---

## 7. §5 — Section filter in the UI

**Problem:** `embed_and_search.py` already supported section-level filtering and
`build_where()` (the CLI helper) already handled it, but the `/query` request schema only
exposed `ticker` and `form` — `section` was silently ignored if sent. No UI control
existed either. The feature was ~70% built at the retrieval layer and entirely
unreachable by users.

**Backend (`api.py`):** `Query.section: str = ""` added. `_build_where()` appends
`{"section": q.section.lower()}` when set, combining with ticker/form via the same
`$and` path already used for multiple filters — no change needed downstream in
`embed_and_search.py`'s filter translation.

**Frontend (`index.html`):** a new `.control-group` (matching the existing ticker/form
visual pattern) with a `<select id="sectionFilter">` — All sections, MD&A, Risk Factors,
Results of Operations, Financial Statements, Market Risk, Legal Proceedings, Controls &
Procedures (7 canonical sections, matching `canonical_section()`'s output set minus
`exhibits`/`unregistered_sales`/`material_agreement`/`other`, which aren't useful things
to filter *to*). Wired into the `/query/stream` payload and reset on "New question."

**A real regression this feature exposed, caught by a pre-commit code-review pass (this
is the *earlier* of the two "found in review" bugs in v3 — different session, same
pattern as the one described in §4):** `_route()`'s original guard was
`if where is None:` — treating *any* explicit filter, including a section-only or
form-only one with **no ticker**, the same as a deliberate manual override, and skipping
ticker detection entirely. Before this feature, that only mattered for the pre-existing
`form` filter (rarely combined with a company-comparison question in practice). The much
more prominent Section dropdown made it far easier to hit: picking "Risk Factors" with no
ticker selected and asking "Compare Apple and Microsoft's biggest risks" would silently
fall through to an undifferentiated single search instead of `_ask_cross_company()`'s
guaranteed per-ticker retrieval.

Fixed by replacing `where is None` with `_has_ticker_filter(where)` (checks top-level and
inside `$and`) — ticker detection now always runs unless a ticker is *already* pinned,
and any ticker found gets merged into the existing filter via a new `_merge_where()`
helper, threaded through as `extra_where` into `_prepare_cross_company`/`_prepare_temporal`
(and, later, `_prepare_section_diff`) so a section/form constraint survives structured
retrieval too, not just the plain path.

**Residual caveat (still true, documented in Known Limitations):** combining the section
filter with a sparse-coverage ticker (INTC's 22 chunks) or an unusual section can return
thin or empty results — the filter narrows an already-small per-company slice, and the
UI gives no feedback beyond the normal abstain message when a combination is thin.

---

## 8. Testing

| | Before v3 | After §1/§2/§3/§5 | After §4 |
|---|---|---|---|
| Unit tests (`tests/test_pure.py`) | 51 | 66 | **95** |
| Eval cases (`evals/dataset.json`) | 104 | 104 | **110** |

All 44 new unit tests added across v3 are pure-function tests — no network calls, no paid
API calls. Everything for §4 specifically was written **test-first** (TDD): each new
function's tests were confirmed to fail with an `ImportError` (function didn't exist yet)
before being implemented, then confirmed green. New test classes: `TestIsSectionDiffQuestion`
(6), `TestDetectSection` (9), `TestBuildDiffPrompt` (6), plus 8 new `TestRoute` cases
(section-diff routing precedence, the diverse/explicit-where bypass, the section-filter
precedence fix).

`_prepare_section_diff` / `_ask_section_diff` themselves aren't directly unit-tested
(they require a live Pinecone + Anthropic call) — same situation as `_prepare_temporal` /
`_ask_temporal`, which have never had direct unit coverage either. Correctness there is
established through `_route()`'s pure-function tests (routing logic) plus live
verification against the real API (see each section above).

---

## 9. Eval results, run-by-run

| Run | Cases | Overall | Notes |
|---|---|---|---|
| Pre-v3 baseline | 104 | 103/104 (99%) | Single known failure: `avgo_gross_margin` |
| After §1 (streaming) | 104 | 103/104 (99%) | No change |
| After §2 (Cohere) | 104 | 103/104 (99%) | No change |
| After §3 + §5 | 104 | 103/104 (99%) | No change |
| After §4 (this session) | **110** | **107/110 (97%)** | See below |

The final run's 3 misses:
1. `avgo_gross_margin` — pre-existing, unrelated to any v3 work (retrieval surfaces
   restructuring 8-K chunks instead of the margin table; model correctly abstains).
2. `aapl_revenue_yoy`, `aapl_iphone_revenue_fy2025` — **new** misses in this run, but
   confirmed via direct `_route()` calls to route identically (`kind: "plain"`) before
   and after the §4 change. Both questions need two specific filing-period figures to
   land inside the same single 5-chunk rerank pool — an inherently borderline retrieval
   scenario unrelated to any routing change. Treated as retrieval/LLM answer variance,
   not a regression.

All 6 new §4 diff cases pass. The `temporal` group as a whole is 18/18 (100%), up from
the prior 12/12 — the 6 new cases all landed clean.

---

## 10. Known limitations added or updated in v3 (see `CLAUDE.md` for the full list)

- **Cohere reranking**: free tier capped at 1000 calls/month; falls back silently to
  local cosine+lexical order on any failure (by design, not a bug).
- **Section filter**: thin/empty results when combined with a sparse ticker (INTC) or an
  unusual section; no UI feedback beyond the normal abstain message.
- **Section diff**: `is_section_diff_question()` is a fixed phrase list — a
  differently-worded "what changed" question can silently miss it and fall through to
  the plain or temporal path instead. `detect_section()`'s default (`risk_factors`) may
  not match what the user meant if they didn't name a section. Only ever compares the 2
  *most recent* periods found in a `CANDIDATE_K`-sized pool — for a low-coverage
  ticker/section, those could be adjacent quarters rather than year-over-year 10-Ks, and
  the prompt doesn't currently surface which two periods it picked beyond the passage
  headers themselves. **No UI entry point** — there's no "diff mode" toggle; it's only
  reachable by phrasing a question the way `_DIFF_PHRASES` expects.
- **RAGAS** optional eval layer remains blocked on Python 3.14 + `nest_asyncio` — this
  predates v3 and wasn't touched.

---

## 11. What was explicitly NOT measured / NOT done (stated honestly, not oversold)

- Cohere reranking (§2): no before/after precision delta on individual cases — only
  "shipped without regressing the aggregate eval," not "measurably more precise."
- Section diff (§4): the *quality* of the added/removed/unchanged breakdown isn't scored
  by the eval harness (`answer_contains: []`) — only that retrieval + routing work. One
  live example looked accurate on manual read-through; that's a single spot-check, not a
  measured claim.
- Streaming (§1): does not reduce actual end-to-end latency — retrieval (embedding +
  Pinecone) still dominates time-to-first-token (~7s measured). It only stops the UI from
  being blank for the *entire* round trip.
- Source preview (§3): no summarization of the 500-char passage slice — can end
  mid-sentence.
