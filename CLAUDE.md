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

v2 is complete. P0, P1, P2, and P3 are all done.

- Scale: 13 companies, 8,141 chunks, 10-K / 10-Q / 8-K over ~2 years.
- Frontend: single-page chat UI (`index.html`) with waking-up state and auto-retry on
  cold start. Eval dashboard (`dashboard.html`). Standalone theme tracker (`themes.html`).
- Backend: FastAPI on Render's free tier. Cold starts are visible in the UI; the Pinecone
  client lazy-inits on first `/query` so `/health` responds instantly.
- Vector store: Pinecone serverless (AWS us-east-1, cosine, free tier). No local index —
  Render's 512MB RAM is sufficient because there's nothing to build on boot.
- Evals: 104-case golden set; last run 103/104 (99%). Cross-company: 22/22 (100%),
  temporal: 12/12 (100%), factual: 43/44, abstain: 26/26. Retrieval hit-rate: 100%.
  RAGAS optional layer is wired up but blocked on Python 3.14 + nest_asyncio (see
  Known limitations). Deterministic suite is primary.
- Unit tests: 51/51 passing (`tests/test_pure.py`). No network or paid API calls.

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
  per-ticker retrieval for cross-company questions and structured per-period retrieval for
  temporal questions, enforces the answer contract, abstains on weak evidence.
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
- Eval harness (`evals/eval.py`, `evals/dataset.json`): 104-case golden set with per-case
  scoring; writes `evals/last_results.json`, served by the API for the dashboard.
- RAGAS eval (`evals/eval_ragas.py`, optional): LLM-judge layer on top of the golden set.
  Saves `evals/results/ragas_{results,summary}.json` and `ragas_results.csv`.
  Currently blocked on Python 3.14 + nest_asyncio incompatibility (see Known limitations).
- Eval dashboard (`dashboard.html`): fetches `/evals/results` and `/evals/ragas`; renders
  metric cards, per-group bars, question-type breakdown, sortable case table, RAGAS section
  (shows placeholder when not yet run), and a link card to `themes.html`.
- Unit tests (`tests/test_pure.py`): 51 cases, all passing. Pure functions only.

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
- `tests/test_pure.py`, `tests/__init__.py` — unit tests (51 cases, no network calls).
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

## 1. Streaming responses (SSE)

**The gap.** Every query blocks for 5–20 s, then renders the full answer at once. No
feedback that anything is happening — the most noticeable gap vs. any production LLM app.

**Resume value.** "Implemented SSE streaming for real-time LLM output, cutting
time-to-first-token from ~10 s to <1 s across FastAPI, Anthropic Streaming API, and a
vanilla-JS ReadableStream consumer."

**Steps.**
1. `ask.py` — add `ask_stream()`: mirrors `ask()` routing but uses
   `with client.messages.stream(...) as stream: for text in stream.text_stream: yield text`,
   then yields a sentinel `{"__meta__": True, "results": ..., "effective_where": ...}`.
   Keep `ask()` intact (evals + cache use it).
2. `api.py` — add `POST /query/stream`: same `Query` schema + validation; returns
   `StreamingResponse(media_type="text/event-stream")`. Generator yields
   `data: {"token": "..."}`, then `data: {"sources": [...], "filter_applied": ...}`,
   then `data: [DONE]`. No caching on the stream endpoint.
3. `index.html` — switch `submitQuestion()` to `fetch` + `ReadableStream`. Create the
   answer div immediately (empty) so tokens stream into it. Parse SSE lines: render
   tokens on `evt.token`, render sources on `evt.sources`. Re-parse markdown on `[DONE]`.
   Shorten the slow-timer from 8 s to ~2 s (streaming makes cold-start less jarring).
4. Verify: tokens stream visibly in the UI; `python evals/eval.py` still passes 103/104
   against the unchanged `/query` endpoint.

## 2. Two-stage retrieval with cross-encoder reranking (Cohere)

**The gap.** Current reranker: `cosine + 0.08 * lexical_score`. Can't distinguish a
chunk that *answers* the question from one that merely *mentions* the same terms.
A cross-encoder reads question + passage together — strictly better at relevance.

**Resume value.** "Replaced single-stage cosine retrieval with Pinecone ANN (top-50
recall) → Cohere Rerank cross-encoder (top-5 precision), measured on a 104-case eval."

**Steps.**
1. `requirements.txt` — add `cohere>=5.0`.
2. `.env.example` — add `COHERE_API_KEY=` (free tier: 1000 calls/month).
3. `ask.py` — add `_rerank_with_cohere(results, question, k)`: calls
   `cohere.ClientV2(...).rerank(model="rerank-v3.5", ...)`, falls back to original order
   if `COHERE_API_KEY` is unset.
4. `ask.py` — in `ask()`, change the retrieval call to `k=CANDIDATE_K` to get the full
   50-candidate pool, then pass through `_rerank_with_cohere(..., k=k)`. Do the same in
   `_ask_cross_company()`. Skip reranking in `_ask_temporal()` (chronological order
   must be preserved).
5. `render.yaml` — add `COHERE_API_KEY` env var (value set in Render dashboard).
6. Run `python evals/eval.py` before and after; record delta in CLAUDE.md.

## 3. Source passage preview in the answer UI

**The gap.** Citation cards show metadata badges but not the actual text the model cited.
Verifying a claim requires navigating to the SEC filing manually.

**Resume value.** "Added inline evidence preview to citation cards: users expand any
cited passage to read the exact text the model referenced, without leaving the interface."

**Steps.**
1. `api.py` — add `"text": r["text"][:500]` to each entry in the `sources` list.
   Additive field — doesn't break the frozen response contract.
2. `index.html` — add CSS for `.source-preview` (hidden by default, `display:block` when
   `.open`) and `.source-toggle` button.
3. `index.html` — in `buildTurn()`, append a toggle button + hidden `<div class="source-preview">`
   to each source card. Toggle opens/closes the preview and flips the button label
   (`"passage"` / `"hide"`).
4. Verify: evals still pass (they call `/query`, not the UI); payload grows ~0.5 KB/source
   (note this tradeoff).

## 4. Section-to-section temporal diffs

**The gap.** `_ask_temporal()` groups evidence by period for trend questions, but cannot
compare the *same section* across two discrete periods. Financial analysts routinely read
consecutive 10-Ks to spot new disclosures — "What changed in NVDA's risk factors since
last year?" — and this query type has no handler.

**Resume value.** "Implemented section-level diff analysis across consecutive SEC filings:
detects 'what changed' queries, retrieves the same section from two reporting periods, and
generates a structured before/after comparison."

**Steps.**
1. `ask.py` — add `_DIFF_PHRASES` list and `is_section_diff_question(question)`.
2. `ask.py` — add `_SECTION_HINTS` dict and `detect_section(question)` (maps "risk factor"
   → `"risk_factors"`, "md&a" → `"mda"`, etc.; defaults to `"risk_factors"`).
3. `ask.py` — add `build_diff_prompt(question, period_chunks, ticker, section)`:
   structures evidence as `--- {period} ---` blocks and instructs the model to list
   (a) what was added/strengthened, (b) what was removed/softened, (c) what stayed the
   same, with quotes and citations.
4. `ask.py` — add `_ask_section_diff(collection, question, ticker, k, history, model=None)`:
   filters by `{"$and": [{"ticker": ticker}, {"section": section_key}]}`, retrieves
   `CANDIDATE_K` candidates, groups by `period`, takes top 2 chunks for the 2 most
   recent distinct periods. Falls back to `_ask_temporal()` if fewer than 2 periods found.
5. `ask.py` — route in `ask()` **before** the temporal check:
   `if len(tickers) == 1 and is_section_diff_question(question) and not diverse`.
6. `evals/dataset.json` — add 5+ cases (e.g. "What changed in NVIDIA's risk factors
   between its two most recent 10-Ks?") with `"group": "temporal"` and
   `"answer_contains": []` (retrieval-only; diff wording is non-deterministic).

## 5. Section filter in UI (full-stack filter completion)

**The gap.** `embed_and_search.py` supports section filtering and `build_where()` handles
it, but the `Query` schema only exposes `ticker` and `form` — `section` is silently
ignored if sent. The UI has no section control. The feature is ~70% built at the backend
layer but entirely inaccessible to users.

**Resume value.** "Completed the full-stack filter pipeline by exposing section-level
filtering (MD&A, Risk Factors, Financial Statements, 4 others) through the API and a new
frontend control — a capability that existed at the retrieval layer but was unreachable."

**Steps.**
1. `api.py` — add `section: str = ""` to `Query`. In the filter-building block, add
   `if q.section: filters.append({"section": q.section.lower()})`.
2. `index.html` — add a `.control-group` div (same pattern as ticker/form controls) with
   a `<select id="sectionFilter">` containing: All sections, MD&A (`mda`), Risk Factors
   (`risk_factors`), Results of Operations (`results_of_operations`), Financial Statements
   (`financial_statements`), Market Risk (`market_risk`), Legal Proceedings
   (`legal_proceedings`), Controls & Procedures (`controls`).
3. `index.html` — wire up: declare `const sectionSel`; add `section: sectionSel.value || ""`
   to the fetch payload; reset `sectionSel.value = ""` in the "New question" handler.
4. Verify: select "Risk Factors", ask "What are the biggest risks?" for AAPL; confirm
   `filter_applied` in the response includes `{"section": "risk_factors"}`.

## Summary

| # | Improvement | Files | Resume metric |
|---|-------------|-------|---------------|
| 1 | Streaming SSE | `ask.py`, `api.py`, `index.html` | Time-to-first-token: ~10 s → <1 s |
| 2 | Cohere cross-encoder reranking | `ask.py`, `requirements.txt`, `.env.example` | Retrieval precision delta on 104-case eval |
| 3 | Source passage preview | `api.py`, `index.html` | Inline evidence verification, no extra API calls |
| 4 | Section-to-section temporal diffs | `ask.py`, `evals/dataset.json` | 5+ new passing eval cases for "what changed" queries |
| 5 | Section filter in UI | `api.py`, `index.html` | 7 canonical sections filterable end-to-end |
