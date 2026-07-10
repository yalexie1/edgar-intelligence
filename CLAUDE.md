# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

EDGAR Intelligence: a retrieval-augmented generation (RAG) system over SEC filings.
It ingests filings for 13 large public companies, chunks and embeds them, stores them
in a Pinecone serverless vector index with rich metadata, and answers plain-English
questions with citations back to the source filing.

Solo learning project by a CS and math student learning production AI engineering.
Prefer clear, well-commented code and explain non-obvious choices. Make minimal,
focused changes; do not refactor unrelated code without asking.

Update this file after any major change to the codebase.

## Current status (read this first)

v2 is complete. P0, P1, P2, and P3 are all done. v3 is complete — all five steps
(streaming SSE, Cohere cross-encoder reranking, source passage preview, section-to-section
temporal diffs, section filter in UI) are done.

- Scale: 13 companies, 8,141 chunks, 10-K / 10-Q / 8-K over ~2 years.
- Frontend: single-page chat UI (`index.html`) with waking-up state and auto-retry on
  cold start. Answers stream token-by-token via SSE. Filter controls now include ticker,
  form, and section (7 canonical sections); each citation card has an expandable inline
  passage preview (see v3 §3, §5 below). Eval dashboard (`dashboard.html`). Standalone
  theme tracker (`themes.html`).
- Backend: FastAPI on Render's free tier. Cold starts are visible in the UI; the Pinecone
  client lazy-inits on first `/query` so `/health` responds instantly. `POST /query/stream`
  streams the same answer as Server-Sent Events (see v3 §1 below). `Query.section` and
  each source's `text` (first 500 chars) are additive fields on the existing contract.
- Vector store: Pinecone serverless (AWS us-east-1, cosine, free tier). No local index —
  Render's 512MB RAM is sufficient because there's nothing to build on boot.
- Retrieval: two-stage — Pinecone ANN (cosine + lexical boost) pulls a 50-candidate pool,
  then Cohere `rerank-v3.5` re-ranks for precision before truncating to k (see v3 §2).
  Falls back to the local rerank order if `COHERE_API_KEY` is unset or the call fails.
  "What changed" questions about a single company's section route to a dedicated
  2-period diff path instead (see v3 §4) — no Cohere rerank there either, since period
  grouping (not raw relevance) drives which chunks get selected.
- Evals: 110-case golden set (104 + 6 new section-diff cases added with v3 §4); last run
  107/110 (97%) on 2026-07-10. Cross-company: 22/22 (100%), temporal: 18/18 (100%,
  includes all 6 new diff cases), factual: 41/44, abstain: 26/26. Retrieval hit-rate:
  100%. The 3 factual-group misses are `avgo_gross_margin` (pre-existing, see Known
  limitations) plus two borderline cases (`aapl_revenue_yoy`, `aapl_iphone_revenue_fy2025`)
  that didn't fail in the prior 103/104 run — confirmed via direct `_route()` calls that
  both still take the unchanged plain path (identical routing before/after v3 §4), so this
  is retrieval/LLM answer variance on questions needing two specific filing periods in one
  5-chunk pool, not a regression from the new routing. RAGAS optional layer is wired up
  but blocked on Python 3.14 + nest_asyncio (see Known limitations). Deterministic suite
  is primary.
- Unit tests: 98/98 passing (`tests/test_pure.py`). No network or paid API calls.

Pipeline file state:
- `ingest.py` and `embed_and_search.py` are FROZEN. Any change requires a full
  re-ingest + `--rebuild` + eval confirmation. Confirm with the user first.
- All other files (`ask.py`, `api.py`, `index.html`, `dashboard.html`, `themes.html`,
  `evals/`, `tests/`) are current and may be edited freely.

## Corpus scope

- Tickers: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, INTC.
- Forms: 10-K, 10-Q, 8-K. Window: last ~2 years. Caps per company: 3 / 8 / 8.
- Ingestion output: `data/corpus.jsonl` (one JSON record per chunk).
- Vector store: Pinecone index `sec-filings`, cosine distance, 1536 dimensions.

## Architecture

- Ingestion (`ingest.py`, offline): resolves tickers to CIKs (SEC ticker map + fallback
  dict), pulls filings via the SEC submissions API, downloads each document, cleans HTML
  (block-aware; strips `display:none` XBRL), splits by SEC Part/Item, chunks
  paragraph-aware (~4000 chars, 600 overlap), writes JSONL with metadata.
- Retrieval (`embed_and_search.py`): loads the corpus, embeds with OpenAI, upserts to
  Pinecone with metadata (chunk text stored in metadata since Pinecone has no separate
  document field). Provides semantic search, metadata filters, query expansion, lexical
  reranking, and diversity mode.
- Answering (`ask.py`): retrieves evidence and asks Claude for a grounded, cited answer.
  Auto-detects company names (falls back to prior turns for follow-ups), uses structured
  per-ticker retrieval for cross-company questions, structured per-period retrieval for
  temporal questions, and a dedicated 2-period diff retrieval for "what changed in
  section X" questions (see v3 §4), enforces the answer contract, abstains on weak
  evidence.
- Backend (`api.py`, FastAPI): `/query` (POST), `/themes` (GET, retrieval-only theme
  tracking), `/evals/results` (GET), `/evals/ragas` (GET), `/health` (GET).
  In-memory LRU cache (256 entries) for `(question, where, diverse)` tuples.
  Per-IP rate limiting (10/min, 200/day) and global daily cap (2000/day).
- Frontend (`index.html`): single-page chat UI with a follow-up bar, autocomplete,
  tooltips, filter controls, and a "New question" reset. Shows waking-up state on cold
  start with auto-retry. Nav links to dashboard and theme tracker.
- Theme tracker (`themes.html`): standalone page. Fetches `GET /themes?ticker=`, renders
  a heat map of retrieval strength across 8 themes × filing periods. Includes a callout
  explaining what rerank_score means and what it does not (not frequency, not sentiment).
- Eval harness (`evals/eval.py`, `evals/dataset.json`): 110-case golden set with per-case
  scoring; writes `evals/last_results.json`, served by the API for the dashboard.
- RAGAS eval (`evals/eval_ragas.py`, optional): LLM-judge layer on top of the golden set.
  Saves `evals/results/ragas_{results,summary}.json` and `ragas_results.csv`.
  Currently blocked on Python 3.14 + nest_asyncio incompatibility (see Known limitations).
- Eval dashboard (`dashboard.html`): fetches `/evals/results` and `/evals/ragas`; renders
  metric cards, per-group bars, question-type breakdown, sortable case table, RAGAS section
  (shows placeholder when not yet run), and a link card to `themes.html`.
- Unit tests (`tests/test_pure.py`): 98 cases, all passing. Pure functions only.

## The retrieval interface (what `ask.py` calls)

Import from `embed_and_search`. Key constants:
`CORPUS_PATH`, `PINECONE_INDEX_NAME` (= "sec-filings"), `TOP_K` (= 5),
`CANDIDATE_K` (= 50), `DIVERSE_CANDIDATE_MULTIPLIER` (= 8).

Connect:
`get_pinecone_index()` → Pinecone Index object. Used by both `api.py` and `ask.py`.

Main function:
`search(index, question, where=None, k=TOP_K)` -> list of result dicts, already
reranked and truncated to `k`. Each result dict has:
- `id`: str
- `similarity`: float (cosine, 1.0 = identical; Pinecone returns this directly)
- `lexical_score`: float
- `rerank_score`: float (sort key; `similarity + KEYWORD_BOOST * lexical_score`)
- `text`: str (the chunk, extracted from Pinecone metadata at query time)
- `metadata`: dict with `ticker, company, cik, form, filing_date, report_date, period,
  accession, source_url, primary_document, part, item, part_item, item_title,
  chunk_index, section_chunk_index, section`

Filter format (passed as `where`): Chroma-style dicts — `{"ticker": "NVDA"}` or
`{"$and": [{"ticker": "AMD"}, {"form": "10-Q"}]}`. `search()` translates to Pinecone
`$eq` format internally via `_to_pinecone_filter()`. `build_where(args)` builds from CLI
args; for programmatic use, build the dict directly.

Other helpers:
- `diversify_results(results, k=TOP_K, by="ticker"|"filing")`: one strong hit per group.
- `expanded_query`, `lexical_score`, `query_terms`, `best_snippet`.

Canonical `section` labels (from `canonical_section`): `mda`, `risk_factors`,
`market_risk`, `exhibits`, `financial_statements`, `unregistered_sales`,
`legal_proceedings`, `controls`, `results_of_operations`, `material_agreement`, `other`.
Section labels are correct after the P2 fix (short item headings like "Item 1A. Risk
Factors" were previously dropped by the MIN_PARAGRAPH_CHARS filter). INTC exception:
only 22 chunks because their XBRL-inline 10-K HTML produces very few leaf text blocks.

## Answer contract (enforced in `ask.py`)

Every claim must carry: (1) the claim, (2) a short supporting quote from the evidence,
(3) a precise source (ticker, form, period, section, source_url), (4) a confidence level.
Answer only from retrieved evidence; abstain on thin evidence (low `rerank_score`).
"The filings don't cover this" is a valid, expected answer.

## Conversation context (`ask.py` + `api.py`)

`ask()` accepts optional `history` ([{"question","answer"}, ...]). Used for (1) retrieval
(reuse a prior turn's ticker when the follow-up names none) and (2) prompting (prepend
recent turns as context). Frontend sends `history: conversationHistory.slice(-3)` and
clears it on "New question." Cache is bypassed when `history` is present.

## Project layout

- `verify_setup.py` — checks OpenAI, Anthropic, and Pinecone API keys.
- `ingest.py` — FROZEN. Builds `data/corpus.jsonl`.
- `embed_and_search.py` — FROZEN. Uploads to Pinecone; `search()` + helpers.
- `ask.py` — current. RAG answering with rigor contract + conversation context.
- `api.py` — current. FastAPI endpoints + LRU cache + rate limiting.
- `index.html` — current. Chat frontend with waking-up state + nav links.
- `dashboard.html` — current. Eval dashboard; links to `themes.html`.
- `themes.html` — current. Standalone theme tracker with score explanation callout.
- `tests/test_pure.py`, `tests/__init__.py` — unit tests (98 cases, no network calls).
- `evals/eval.py`, `evals/dataset.json`, `evals/last_results.json` — current.
- `evals/eval_ragas.py` — current. Optional RAGAS LLM-judge layer (see Known limitations).
- `evals/results/` — RAGAS output files written here when eval_ragas.py is run.
- `data/corpus.jsonl` — committed. Index can rebuild from this without re-scraping.
- `.env` (gitignored) / `.env.example` (committed template).
- `render.yaml`, `start.sh`, `config.js` — deploy config.

## Commands

- Env: `source .venv/bin/activate`
- Install: `python -m pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Backend: `uvicorn api:app --reload --port 8000`
- Frontend: `python -m http.server 5500` then open `http://localhost:5500/index.html`
  (use localhost:5500, not file://, so CORS to the API works)
- Unit tests (no API needed): `python -m pytest tests/`
- Evals (API must be running): `python evals/eval.py` (or `--group factual`)
- RAGAS eval (optional, API must be running): `python evals/eval_ragas.py --subset 10`
  NOTE: blocked on Python 3.14 + nest_asyncio incompatibility; run on Python 3.11/3.12.
- Rebuild index (only if corpus changed): `python embed_and_search.py --rebuild`
  NOTE: costs ~$0.03 in OpenAI fees and ~15 min. Confirm with user before running.
- Interactive search: `python embed_and_search.py [--ticker AAPL --section mda | --diverse]`
- Normal run prints: `Index already built (8141 vectors). Skipping embedding.`

## Tech stack and key decisions

- Python in `.venv`.
- OpenAI `text-embedding-3-small` for embeddings (1536-dim); Anthropic Claude for answers.
  Keys in `.env`.
- Pinecone serverless (AWS us-east-1, cosine) vector store. Replaced Chroma after the P2
  corpus expansion (5,438 → 8,141 chunks) caused Render's free tier (512MB RAM) to OOM
  during index build (Chroma's local HNSW index was 336MB on disk). Pinecone keeps the
  index in the cloud; Render only needs RAM for query embeddings. Chunk text is stored in
  Pinecone metadata since Pinecone has no separate document field.
- Answer model: `ANSWER_MODEL` env var (default: `claude-haiku-4-5-20251001` for dev;
  `claude-sonnet-4-6` set in `render.yaml` for the deployed demo). `ask()` and both
  structured retrieval helpers accept a `model=` kwarg.
- Cohere `rerank-v3.5` (`COHERE_API_KEY`, optional) cross-encoder reranks the top-50
  Pinecone candidates down to k. Free tier: 1000 calls/month. Falls back to the local
  cosine+lexical order if the key is unset or the call fails — never a hard dependency.
- Embedding: BATCH_SIZE=40 texts/call + exponential backoff.
- Pinecone upsert: UPSERT_BATCH_SIZE=100 vectors/call (recommended for large metadata).
- In-memory LRU cache in `api.py` (256 entries, `OrderedDict`). Bypassed when `history`
  is present; resets on restart — fine for a demo server.

## Known limitations (be honest about these; do not oversell)

- The public API has per-IP rate limiting (10 req/min, 200 req/day) and a global daily
  cap (2000 req/day). CORS is tightened to an allowlist. These protect against casual
  abuse; a determined attacker with rotating IPs is not the threat model.
- RAGAS eval is blocked on Python 3.14. `nest_asyncio` (a ragas dependency) patches
  `asyncio.run()` in a way that makes `asyncio.current_task()` always return None.
  Python 3.14 added a strict task-context check to `asyncio.timeout()` (and downstream
  libraries like `anyio`/`sniffio` that the OpenAI async client uses), causing all metric
  calls to fail. Run `eval_ragas.py` on Python 3.11 or 3.12 until ragas fixes this.
- INTC: only 22 chunks because their XBRL-inline 10-K HTML produces very few leaf text
  blocks. Known gap; not worth fixing without a broader corpus refresh.
- Cross-company superlative questions ("which company has the highest gross margin?") use
  broad diverse retrieval (BROAD_DIVERSE_MULTIPLIER=16, top-50 candidates) rather than
  structured per-ticker retrieval. A company's best chunk may not surface in the top 50
  across 8,141 vectors. The model correctly reports only the companies it has evidence
  for, but the caveat can be misleading. Named-company comparisons ("compare AAPL and
  MSFT margins") use `_ask_cross_company()` and guarantee coverage.
- Cold starts. Render free tier spins down after ~15 min idle. Pinecone removed the OOM
  risk, but the Python/FastAPI process cold start still takes ~15-30s.
- Reranking (`KEYWORD_BOOST=0.08`) and query expansion (AI-only) are hardcoded heuristics,
  not tuned against the eval set.
- 8-Ks are noisy (capped at 8 most recent per company).
- `avgo_gross_margin` eval case fails: retrieval surfaces restructuring 8-K chunks instead
  of the margin table; the model correctly abstains. Known retrieval gap.
- Theme tracker scores are rerank_score values (cosine + lexical boost). They measure
  retrievability of a topic — not frequency, not sentiment. Small differences between
  periods are not meaningful. The `themes.html` page explains this explicitly.
- Section filter (v3 §5): combining it with a sparse-coverage ticker (INTC's 22 chunks)
  or an unusual section can return thin or empty results, since the filter narrows an
  already-small per-company slice. `_route()` now merges a section/form-only filter with
  any detected ticker so cross-company/temporal routing still works when the filter is
  set without a ticker (fixed 2026-07-09 — previously any explicit filter silently
  disabled structured retrieval), but the UI gives no feedback when a combination is thin
  beyond the normal abstain message.
- Section diff (v3 §4): `is_section_diff_question()` is a fixed phrase list ("what
  changed", "since last year", "compared to the prior year", etc.) — a differently-worded
  "what changed" question can silently miss the list and fall through to the plain or
  temporal path instead. `detect_section()` defaults to `"risk_factors"` when no section
  is named, which may not be the section the user actually meant. `_prepare_section_diff`
  only ever compares the 2 most recent distinct periods it finds in a `CANDIDATE_K`-sized
  pool — for a low-coverage ticker/section combination (e.g. INTC), those 2 "periods"
  could be adjacent quarters rather than year-over-year 10-Ks, and the prompt doesn't
  currently surface which two periods it picked outside of the passage headers. There is
  no UI entry point for this feature yet (no "diff mode" toggle) — it's reachable only by
  phrasing a question the way `_DIFF_PHRASES` expects.
  **Bug found via live pre-deploy testing, fixed same day (2026-07-10):** several
  `_DIFF_PHRASES` entries ("compare to the prior year", "vs last year", etc.) are generic
  enough that an ordinary conversational follow-up naturally contains one regardless of
  topic. A live Playwright test asked "What was Microsoft's total revenue in fiscal year
  2025?" then followed up with "How does that compare to the prior year?" — the follow-up
  matched `_DIFF_PHRASES`, no section was named so `detect_section()` defaulted to
  `"risk_factors"`, and the ticker fell back to MSFT from history, so `_route()` silently
  returned a full risk-factors diff instead of continuing the revenue conversation.
  Fixed in `_route()`: when the ticker is only known via history fallback (not named in
  the current question) *and* no section is named in the current question either, that
  combination now skips `section_diff` and falls through to temporal/plain instead — the
  signature of a generic follow-up, not a deliberate diff request. A ticker or section
  named explicitly in the current question still routes to `section_diff` even with
  unrelated history present (see `TestRoute.test_diff_phrase_with_history_ticker_but_explicit_section_still_diffs`
  and `test_diff_phrase_with_explicit_ticker_in_question_still_diffs_despite_history`).
  `eval.py` never sends `history`, so this fix cannot change any of the 110 golden-set
  cases — confirmed by grepping `evals/eval.py` for `history` (no matches) before
  skipping a full eval re-run. 3 new unit tests added; full suite: 98/98 passing.

## Conventions and rules

- IMPORTANT: never commit `.env` or any API key. If a key is committed, treat it as
  compromised and rotate it at the provider.
- IMPORTANT: every SEC EDGAR request must send a real User-Agent (name + email) or the SEC
  returns 403. Set it in `ingest.py`.
- When extracting filing text, strip `display:none` elements first (hidden XBRL noise).
- Do not advertise a known-flaky feature (e.g. the section filter) without a caveat.
- Treat CORS as polish, not a security control.
- Keep code readable and commented; explain trade-offs. Minimal, focused changes.

## Freeze rules

- `ingest.py` and `embed_and_search.py` are FROZEN.
- Any change to either requires: re-running `ingest.py`, then
  `python embed_and_search.py --rebuild`, then `python evals/eval.py` to confirm no
  regression. Confirm with the user before rebuilding — re-embedding costs ~$0.03 and
  ~15 min for 8,141 chunks.
- Do not change the `/query` request/response JSON contract (the frontend depends on it).

# v2 — completed work (record, not to-do)

## P0 — Protect the public endpoint ✓ DONE

Per-IP rate limiting (slowapi), global daily cap, 500-char question limit, CORS allowlist,
localhost exemption for the eval harness.

## P1 — Make the evals trustworthy ✓ DONE

`answer_hit=None` for retrieval-only cases (faithfulness denominator is honest — 11/12,
not 11/100); `INSUFFICIENT_EVIDENCE` marker checked first for abstain detection;
numeric-tolerant needle matching. `ask_with_contexts()` added for RAGAS. Dashboard RAGAS
section wired up.

## P2 — Make the shallow features real ✓ DONE

- **Cross-company comparison**: `_ask_cross_company()` retrieves per-ticker separately.
  "Which companies" questions use BROAD_DIVERSE_MULTIPLIER=16. 22/22 (100%).
- **Temporal structured retrieval**: `_ask_temporal()` retrieves 80 candidates, groups by
  period, sorts chronologically. 12/12 (100%).
- **Section labeling**: fixed `_SEC_HEADING_RE` in ingest/embed to preserve short headings
  dropped by MIN_PARAGRAPH_CHARS=40. Corpus grew 5,438 → 8,141 chunks.
- **Theme tracking**: 8 predefined themes, retrieval-only heat map, separated from eval
  dashboard into `themes.html` with a score-meaning explainer.

## P3 — Engineering hygiene ✓ DONE

- **Unit tests**: 51 cases, 51/51 passing. `tests/test_pure.py`.
- **Per-request model config**: `ask()` and helpers accept `model=` kwarg. `ANSWER_MODEL`
  env var on Render set to `claude-sonnet-4-6`.
- **Caching**: in-memory LRU (256 entries) in `api.py`.
- **Pinecone migration**: replaced Chroma with Pinecone serverless to fix Render OOM.
  8,141 vectors uploaded. Evals: 103/104 (99%), no regression.

# v3 — work plan

Five improvements ranked most-to-least urgent. None touch frozen files (`ingest.py`,
`embed_and_search.py`) or break the `/query` response contract.

## 1. Streaming responses (SSE) ✓ DONE (2026-07-08)

**The gap.** Every query blocks for 5–20 s, then renders the full answer at once. No
feedback that anything is happening — the most noticeable gap vs. any production LLM app.

**Resume value.** "Implemented SSE streaming for real-time LLM output across FastAPI,
the Anthropic Streaming API, and a vanilla-JS `ReadableStream` consumer, so users see the
answer forming instead of staring at a blank screen for the full round trip."

**What shipped.**
- `ask.py`: the routing logic previously inlined in `ask()` was extracted into `_route()`
  (decides cross_company / temporal / plain) plus one `_prepare_*` helper per path
  (`_prepare_cross_company`, `_prepare_temporal`, `_prepare_plain`) that does retrieval +
  prompt-building + abstain checks without calling the model. `ask()` and the new
  `ask_stream()` both call `_route()` + the matching `_prepare_*`, so the two paths can't
  drift out of sync. `ask()`'s public signature and return shape are unchanged (evals +
  the LRU cache in `api.py` still call it as before). `ask_stream()` yields text chunks
  via `with client.messages.stream(...) as stream: for text in stream.text_stream: yield
  text`, then a final `{"__meta__": True, "results": ..., "effective_where": ...}` sentinel.
  Abstain cases yield the abstain string as one chunk (no model call), matching `ask()`.
- `api.py`: added `POST /query/stream`, same `Query` schema + validation as `/query`
  (factored into shared `_validate_query()` / `_build_where()` / `_build_sources()`
  helpers). Returns `StreamingResponse(media_type="text/event-stream")` emitting
  `data: {"token": "..."}` lines, then one `data: {"sources": [...], "filter_applied":
  ...}` line, then `data: [DONE]`. Not cached — matches the existing rule that
  conversation-shaped requests bypass the cache; extended it to all streamed responses.
- `index.html`: `submitQuestion()` now does `fetch` + `res.body.getReader()`, parsing
  `data:` lines out of the buffered chunks. The `.answer` div is created empty up front
  (holding a status message), then cleared and appended to as tokens arrive; markdown +
  citation parsing (`marked.parse` + `withCitations`) runs once at the end, not per token,
  to avoid flicker. Sources render once the `sources` event lands. Retries (3x, 5 s
  backoff) still cover total connection failure before any token arrives; a failure
  mid-stream keeps the partial answer and appends an inline error instead of discarding it.
  The slow-start (cold-start warning) timer stayed at 8 s — see the correction below;
  an initial attempt to drop it to 2 s shipped a real bug caught via live testing.
- `tests/test_pure.py`: added `TestRoute` (11 cases) covering `_route()`'s branch logic —
  multi-company, temporal, diverse overrides, history fallback, explicit `where`. Pure,
  no network calls. Full suite: 62/62 passing.

**Measured, not assumed:** time-to-first-token for a real question (`"What was Apple's
total revenue in fiscal year 2025?"`) was ~7.1 s of a ~10.3 s total — most of the latency
is embedding + Pinecone retrieval before generation even starts, not generation itself.
Streaming does not fix retrieval latency; it only stops the UI from being blank for the
*entire* round trip. Do not repeat the "<1 s time-to-first-token" framing from the
original plan — it wasn't measured and isn't accurate for this corpus/retrieval setup.
`python evals/eval.py` re-run afterward: 103/104 (99%), identical to the pre-streaming
baseline (the one failure, `avgo_gross_margin`, is the pre-existing known retrieval gap —
see Known limitations). `/query` itself was not modified.

**Bug found via live testing, fixed same day:** the plan's step 3 said to shorten the
UI's cold-start warning timer from 8 s to ~2 s, reasoning that "streaming makes cold-start
less jarring." That reasoning didn't hold: since normal warm time-to-first-token is
already ~7 s (see above), a 2 s threshold fired the "backend is waking up" message on
essentially every request, cold or not — confirmed live when a second, back-to-back
question still showed the cold-start notice. `index.html`'s `slowTimer` was reverted to
8 s (the same value the old blocking UI used, which was already calibrated against this
same retrieval-bound latency). Lesson: a UI-timing change justified by an unmeasured
latency claim should be checked against the actual measured number before shipping —
the ~7 s figure was already recorded two paragraphs up in this same file when the 2 s
value was chosen.

## 2. Two-stage retrieval with cross-encoder reranking (Cohere) ✓ DONE (2026-07-10)

**The gap.** Current reranker: `cosine + 0.08 * lexical_score`. Can't distinguish a
chunk that *answers* the question from one that merely *mentions* the same terms.
A cross-encoder reads question + passage together — strictly better at relevance.

**Resume value.** "Replaced single-stage cosine retrieval with Pinecone ANN (top-50
recall) → Cohere Rerank cross-encoder (top-5 precision), measured on a 104-case eval."

**What shipped.**
- `requirements.txt` — added `cohere>=5.0` (installed: 7.0.5).
- `.env.example` / `.env` — `COHERE_API_KEY=` added (free tier: 1000 calls/month).
- `ask.py` — added `_rerank_with_cohere(results, question, k)`: lazily builds a
  `cohere.ClientV2` (via `_get_cohere_client()`), calls `.rerank(model="rerank-v3.5",
  query=question, documents=[r["text"] for r in results], top_n=min(k, len(results)))`,
  and maps `response.results[i].index` back onto the original result dicts (so metadata,
  `rerank_score`, etc. all survive). Falls back to `results[:k]` — the existing local
  cosine+lexical order — if `COHERE_API_KEY` is unset *or* the API call raises for any
  reason (quota exhausted, network error, etc.); reranking is a precision upgrade, not a
  hard dependency for a free-tier demo.
- `ask.py` — wired into `_prepare_plain`'s non-diverse branch (fetches the full
  `CANDIDATE_K` pool via `search(..., k=CANDIDATE_K)`, then `_rerank_with_cohere(...,
  k)`) and into `_prepare_cross_company` (per-ticker: each named company's own
  `CANDIDATE_K` pool is reranked down to `k_per` *before* flattening across companies,
  so a weaker company's best chunk can't be crowded out by a stronger company's —
  preserves the existing per-ticker coverage guarantee). Both `ask()` and `ask_stream()`
  go through these same `_prepare_*` functions, so blocking and streaming responses are
  reranked identically. Deliberately **not** wired into the diverse-mode branch of
  `_prepare_plain` (ticker spread via `diversify_results` is the goal there, not raw
  relevance — reranking first would let the cross-encoder's favorite ticker crowd out
  the others) or into `_prepare_temporal` (chronological period order must be preserved;
  reranking would undermine the per-period grouping).
- `render.yaml` — added `COHERE_API_KEY` env var (value set in Render dashboard,
  `sync: false`; optional — falls back to local rerank if unset on the deployed server).

**Verified:** live query against `AAPL` (`_prepare_plain`) came back with results whose
`rerank_score` (the local cosine+lexical field) was no longer monotonically decreasing —
confirming Cohere actually reordered the pool rather than passing it through unchanged.
`python evals/eval.py` re-run after shipping: 103/104 (99%), identical to the
pre-reranking baseline — same single pre-existing failure (`avgo_gross_margin`, the
known retrieval gap, see Known limitations), no new regressions. 66/66 unit tests still
pass (reranking only reorders within `_prepare_*`, which the routing tests don't
exercise against a live Cohere call).

**Not measured:** no before/after precision delta on individual cases beyond the
aggregate 103/104 — the eval set's `answer_contains` checks are pass/fail, not scored,
so a reranking improvement that changes *which* correct passage gets cited (without
changing the pass/fail outcome) wouldn't show up in this harness. Worth a manual
side-by-side spot-check before quoting a "reranking improved precision" claim on a
resume — right now the honest claim is "shipped and did not regress the eval suite,"
not "measurably improved retrieval precision."

## 3. Source passage preview in the answer UI ✓ DONE (2026-07-09)

**The gap.** Citation cards show metadata badges but not the actual text the model cited.
Verifying a claim requires navigating to the SEC filing manually.

**Resume value.** "Added inline evidence preview to citation cards: users expand any
cited passage to read the exact text the model referenced, without leaving the interface."

**What shipped.**
- `api.py`: `_build_sources()` adds `"text": r["text"][:500]` to each source entry.
  Additive field on both `/query` and `/query/stream` — the frozen response contract
  (existing field names) is unchanged.
- `index.html`: `.source` was restructured from a single flex row into a column
  container — the old row markup (`source-n`, badges, link, scores) moved into a nested
  `.source-row` div, with a new `.source-toggle` button appended and a sibling
  `.source-preview` div (escaped via the existing `escHtml()`) below it, hidden by
  default (`display: none`) and shown via an `.open` class. `renderSourceItems()` only
  emits the toggle/preview when `s.text` is present. Since source cards are inserted via
  `innerHTML` (no per-card listeners at render time), one delegated click listener on
  the `#chat` container handles every toggle across every turn — finds the closest
  `.source`, flips `.open` on its `.source-preview`, and swaps the button label between
  "passage" and "hide".
- Verified: `curl` against `/query` confirms each source now carries a `text` field
  (500-char preview); `node --check` on the extracted script confirmed valid JS.

**Not done:** no attempt to trim/summarize the passage — it's a raw 500-char slice of
the chunk, which can end mid-sentence. Acceptable for a "does this passage look
plausible" check; not polished prose.

## 4. Section-to-section temporal diffs ✓ DONE (2026-07-10)

**The gap.** `_ask_temporal()` groups evidence by period for trend questions, but cannot
compare the *same section* across two discrete periods. Financial analysts routinely read
consecutive 10-Ks to spot new disclosures — "What changed in NVDA's risk factors since
last year?" — and this query type has no handler.

**Resume value.** "Implemented section-level diff analysis across consecutive SEC filings:
detects 'what changed' queries, retrieves the same section from two reporting periods, and
generates a structured before/after comparison."

**What shipped.**
- `ask.py` — `_DIFF_PHRASES` (a narrow, "what changed"-style phrase list — deliberately
  distinct from `_TEMPORAL_PHRASES`'s broader trend phrasing) and
  `is_section_diff_question(question)`.
- `ask.py` — `_SECTION_HINTS` dict and `detect_section(question)` (maps "risk factor" →
  `"risk_factors"`, "md&a" and "management's discussion" → `"mda"`, "results of
  operations", "financial statement", "market risk", "legal proceeding", "controls and
  procedures"/"internal controls" → their canonical labels; defaults to `"risk_factors"`
  when no section is named — the section analysts diff most often).
- `ask.py` — `build_diff_prompt(question, period_chunks, ticker, section, history=None)`:
  structures evidence as `--- {period} ---` blocks (like `build_temporal_prompt`, but for
  exactly the two periods being compared) and instructs the model to answer in three parts:
  (a) added/strengthened, (b) removed/softened, (c) unchanged, with quotes and citations.
- `ask.py` — `_prepare_section_diff(collection, question, ticker, section, k, history,
  extra_where=None)` + blocking wrapper `_ask_section_diff(...)`: filters by ticker +
  section (via a new `_extract_section_filter()` helper, so a UI-set section filter is
  used as-is rather than double-filtered against the question's own detected section),
  retrieves `CANDIDATE_K` candidates, groups by `period`, keeps the top 2 chunks from each
  of the 2 most recent distinct periods. Falls back to `_prepare_temporal()`'s N-period
  grouping if fewer than 2 periods are found — there's nothing to diff, but a trend answer
  is still useful. No Cohere rerank here, matching `_prepare_temporal`'s reasoning:
  period-based selection, not raw relevance, drives which chunks get kept.
- `ask.py` — routed in `_route()` (shared by `ask()`/`ask_stream()`) **before** the
  temporal check, so a diff-shaped question doesn't fall through to the broader temporal
  path: `if len(tickers) == 1 and is_section_diff_question(question) and not diverse`.
  Multi-company detection is still checked first, so a diff phrase with 2+ named tickers
  correctly gets `_ask_cross_company()`'s guaranteed per-ticker coverage instead.
- `evals/dataset.json` — added 6 cases (`diff_nvda_risk_factors`, `diff_aapl_mda`,
  `diff_msft_risk_factors`, `diff_tsla_risk_factors`, `diff_googl_mda`,
  `diff_msft_no_section_named`) covering each diff phrase variant and the
  default-to-risk_factors path, with `"group": "temporal"` and `"answer_contains": []`
  (retrieval-only; diff wording is non-deterministic). Corpus now 110 cases.
- `tests/test_pure.py` — 28 new cases: `TestIsSectionDiffQuestion` (6),
  `TestDetectSection` (9), `TestBuildDiffPrompt` (6), plus 7 new `TestRoute` cases covering
  precedence (diff before temporal, multi-company still wins, diverse/explicit-ticker-where
  bypass, section-only-filter merge). Full suite: 94/94 passing, all written test-first
  (RED confirmed via `ImportError` before each function existed, then GREEN).

**Verified live:** `POST /query` for "What changed in NVIDIA's risk factors between its
two most recent 10-Ks?" correctly routed to `section_diff`, retrieved FY2025/FY2026
risk-factors chunks, and returned a structured added/removed/unchanged answer with
citations (e.g. correctly identified a new counterparty-risk factor and a new
anti-takeover-provisions risk added in FY2026, not present in FY2025). `POST
/query/stream` for an MD&A diff question streamed the `__meta__` sources event first (both
periods present) exactly as the streaming contract from v3 §1 specifies.

`python evals/eval.py` re-run after shipping (110 cases): 107/110 (97%). All 6 new
diff cases pass (temporal group: 18/18, 100%). The 3 misses are `avgo_gross_margin`
(pre-existing, see Known limitations) plus two factual-group cases
(`aapl_revenue_yoy`, `aapl_iphone_revenue_fy2025`) that passed in the prior 103/104
run — confirmed via direct `_route()` calls that both still take the identical
unchanged plain path before and after this change, so the miss is retrieval/LLM answer
variance on a question needing two specific filing periods inside one 5-chunk pool, not
a regression introduced by the new routing.

**Not measured:** like v3 §2's reranking claim, the diff quality itself (does the
added/removed/unchanged breakdown actually reflect what changed, beyond citing real
passages) isn't scored by the eval harness — `answer_contains: []` makes these cases
retrieval-only. The one live example above looked accurate on manual inspection, but
that's a single spot-check, not a measured precision claim.

**Bug found via pre-commit code review, fixed same day:** the first version of `_route()`
picked the diff prompt's `section` purely from `detect_section(question)` — the
question's own wording — even when an explicit section filter was already pinned via
`extra_where` (the UI's Section dropdown). `_prepare_section_diff` correctly prioritized
the UI filter for *retrieval* (skipping the redundant merge when one was already present),
but the *prompt* was still labeled with the question-derived section regardless. Caught
by manually constructing the conflicting case — a `{"section": "mda"}` UI filter combined
with a question saying "risk factors" — and confirming via a direct `_route()` call that
it returned `section: "risk_factors"` while `extra_where` said `"mda"`: retrieval would
correctly fetch MD&A chunks, but the prompt would tell the model (and by extension the
user) it was comparing "risk_factors." Fixed by adding `_extract_section_filter(where)`
and using `_extract_section_filter(extra_where) or detect_section(question)` in `_route()`,
so an explicit filter always wins — the same precedence pattern already established for
tickers via `_has_ticker_filter()`. Added `TestRoute.test_section_filter_where_overrides_question_wording`
(written first, watched fail with the old `"risk_factors"` value, then fixed). Full suite:
95/95 passing.

## 5. Section filter in UI (full-stack filter completion) ✓ DONE (2026-07-09)

**The gap.** `embed_and_search.py` supports section filtering and `build_where()` handles
it, but the `Query` schema only exposes `ticker` and `form` — `section` is silently
ignored if sent. The UI has no section control. The feature is ~70% built at the backend
layer but entirely inaccessible to users.

**Resume value.** "Completed the full-stack filter pipeline by exposing section-level
filtering (MD&A, Risk Factors, Financial Statements, 4 others) through the API and a new
frontend control — a capability that existed at the retrieval layer but was unreachable."

**What shipped.**
- `api.py`: added `section: str = ""` to `Query`. `_build_where()` appends
  `{"section": q.section.lower()}` when set, combining with ticker/form via the existing
  `$and` path — no change to filter-format handling downstream in `embed_and_search.py`.
- `index.html`: added a `.control-group` (same visual pattern as ticker/form) with
  `<select id="sectionFilter">` — All sections, MD&A (`mda`), Risk Factors
  (`risk_factors`), Results of Operations (`results_of_operations`), Financial Statements
  (`financial_statements`), Market Risk (`market_risk`), Legal Proceedings
  (`legal_proceedings`), Controls & Procedures (`controls`). Wired `sectionSel.value`
  into the `/query/stream` fetch payload and reset it in the "New question" handler.
- Verified live: `POST /query` with `{"ticker": "AAPL", "section": "risk_factors"}`
  returned `filter_applied: {"$and": [{"ticker": "AAPL"}, {"section": "risk_factors"}]}`
  and both retrieved chunks carried `"section": "risk_factors"`.
- Full 104-case eval re-run after both this and §3 shipped: 103/104 (99%), identical to
  baseline — same pre-existing `avgo_gross_margin` failure, no new regressions from
  either additive change. 62/62 unit tests still pass.

**Bug found via code review, fixed same day:** a code-review pass on this diff caught a
real regression this feature exposed: `_route()`'s `if where is None:` guard treated
*any* explicit filter — including a section-only or form-only one, with no ticker — the
same as an intentional manual override, so it skipped ticker detection entirely. Before
this shipped, that only mattered for the pre-existing `form` filter; the new, far more
prominent Section dropdown made it much easier to hit: picking "Risk Factors" with no
ticker and asking "Compare Apple and Microsoft's biggest risks" silently fell through to
an undifferentiated single search instead of `_ask_cross_company()`'s guaranteed
per-ticker retrieval. Fixed by replacing the `where is None` check with
`_has_ticker_filter(where)` — ticker detection now runs unless a ticker is already
pinned, and any ticker found is merged into the existing filter via a new `_merge_where()`
helper (threaded through as `extra_where` into `_prepare_cross_company`/`_prepare_temporal`
so the section/form constraint survives structured retrieval too, not just the plain
path). See the Known Limitations entry on the section filter for the residual caveat.

## Summary

| # | Improvement | Files | Resume metric | Status |
|---|-------------|-------|---------------|--------|
| 1 | Streaming SSE | `ask.py`, `api.py`, `index.html` | Time-to-first-token: full round trip → tokens render as generated (~7 s TTFT, retrieval-bound — see §1) | ✓ DONE |
| 2 | Cohere cross-encoder reranking | `ask.py`, `requirements.txt`, `.env.example`, `render.yaml` | Pinecone top-50 recall → Cohere top-k precision; no eval regression (103/104, see v3 §2 caveat on precision measurement) | ✓ DONE |
| 3 | Source passage preview | `api.py`, `index.html` | Inline evidence verification, no extra API calls | ✓ DONE |
| 4 | Section-to-section temporal diffs | `ask.py`, `evals/dataset.json` | 6 new passing eval cases for "what changed" queries (temporal group: 18/18) | ✓ DONE |
| 5 | Section filter in UI | `api.py`, `index.html` | 7 canonical sections filterable end-to-end | ✓ DONE |
