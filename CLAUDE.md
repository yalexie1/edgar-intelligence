# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

EDGAR Intelligence: a retrieval-augmented generation (RAG) system over SEC filings.
It ingests filings for 13 large public companies, chunks and embeds them, stores them
in a local Chroma vector database with rich metadata, and answers plain-English
questions with citations back to the source filing.

Solo learning project by a CS and math student learning production AI engineering.
Prefer clear, well-commented code and explain non-obvious choices. Make minimal,
focused changes; do not refactor unrelated code without asking.

Update this file after any major change to the codebase.

## Current status (read this first)

v2 is in progress. P0, P1, and all P2 tasks are done. **Next: P3 or theme tracking.**

- Scale: 13 companies, ~8,141 chunks, 10-K / 10-Q / 8-K over ~2 years.
- Frontend: basic single-page chat UI with waking-up state and auto-retry on cold start.
- Backend: FastAPI on Render's free tier. Cold starts are visible in the UI; the index
  lazy-inits on first request so `/health` responds instantly.
- Evals: 104-case golden set; last run 103/104 (99%). Cross-company: 22/22 (100%),
  temporal: 12/12 (100%), factual: 43/44, abstain: 26/26. Retrieval hit-rate: 100%.
  RAGAS optional layer is wired up but currently blocked on a Python 3.14 +
  nest_asyncio incompatibility (see Known limitations). Deterministic suite is primary.

Pipeline file state:
- `ingest.py` and `embed_and_search.py` are FROZEN for P3 work only. P2 section-label
  fix is complete (re-ingest + `--rebuild` + re-run evals done). See "Freeze rules."
- `ask.py`, `api.py`, `index.html`, `dashboard.html`, `evals/eval.py`,
  `evals/eval_ragas.py`, `evals/dataset.json` are current and may be edited for v2.

**P0 and P1 are done. Start at P2. Work top-down. Do not begin a lower priority until
the one above is done or the user explicitly says to skip it.**

## Corpus scope

- Tickers: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, INTC.
- Forms: 10-K, 10-Q, 8-K. Window: last ~2 years. Caps per company: 3 / 8 / 8.
- Ingestion output: `data/corpus.jsonl` (one JSON record per chunk).
- Vector store: `data/chroma/`, Chroma collection `sec_filings`, cosine distance.

## Architecture

- Ingestion (`ingest.py`, offline): resolves tickers to CIKs (SEC ticker map + fallback
  dict), pulls filings via the SEC submissions API, downloads each document, cleans HTML
  (block-aware; strips `display:none` XBRL), splits by SEC Part/Item, chunks
  paragraph-aware (~4000 chars, 600 overlap), writes JSONL with metadata.
- Retrieval (`embed_and_search.py`): loads the corpus, embeds with OpenAI, stores in
  Chroma with metadata; provides semantic search with metadata filters, query expansion,
  lexical reranking, and diversity mode.
- Answering (`ask.py`): retrieves evidence and asks Claude for a grounded, cited answer.
  Auto-detects company names (falls back to prior turns for follow-ups), diversifies for
  cross-company questions, enforces the answer contract, abstains on weak evidence.
- Backend (`api.py`, FastAPI): `/query` (POST), `/evals/results` (GET),
  `/evals/ragas` (GET), `/health` (GET).
- Frontend (`index.html`): single-page chat UI with a follow-up bar, autocomplete,
  tooltips, filter controls, and a "New question" reset. Shows waking-up state on cold
  start with auto-retry.
- Eval harness (`evals/eval.py`, `evals/dataset.json`): 100-case golden set with per-case
  scoring; writes `evals/last_results.json`, served by the API for the dashboard.
- RAGAS eval (`evals/eval_ragas.py`, optional): LLM-judge layer on top of the golden set.
  Saves `evals/results/ragas_{results,summary}.json` and `ragas_results.csv`.
  Currently blocked on Python 3.14 + nest_asyncio incompatibility (see Known limitations).
- Eval dashboard (`dashboard.html`): fetches `/evals/results` and `/evals/ragas`; renders
  metric cards, per-group bars, question-type breakdown, sortable case table, and a RAGAS
  section (shows placeholder when not yet run).

## The retrieval interface (what `ask.py` calls)

Import from `embed_and_search`. Key constants:
`CHROMA_DIR`, `COLLECTION` (= "sec_filings"), `CORPUS_PATH`, `TOP_K` (= 5),
`CANDIDATE_K` (= 50), `DIVERSE_CANDIDATE_MULTIPLIER` (= 8).

Main function:
`search(collection, question, where=None, k=TOP_K)` -> list of result dicts, already
reranked and truncated to `k`. Each result dict has:
- `id`: str
- `similarity`: float (cosine, 1.0 = identical)
- `lexical_score`: float
- `rerank_score`: float (sort key; `similarity + KEYWORD_BOOST * lexical_score`)
- `text`: str (the chunk)
- `metadata`: dict with `ticker, company, cik, form, filing_date, report_date, period,
  accession, source_url, primary_document, part, item, part_item, item_title,
  chunk_index, section_chunk_index, section`

Other helpers:
- `diversify_results(results, k=TOP_K, by="ticker"|"filing")`: one strong hit per group.
- `build_where(args)`: Chroma `where` filter from an argparse namespace. For programmatic
  use, build the dict directly (e.g. `{"ticker": "NVDA"}`, or
  `{"$and": [{"ticker": "AMD"}, {"form": "10-Q"}]}`).
- `expanded_query`, `lexical_score`, `query_terms`, `best_snippet`.

Canonical `section` labels (from `canonical_section`): `mda`, `risk_factors`,
`market_risk`, `exhibits`, `financial_statements`, `unregistered_sales`,
`legal_proceedings`, `controls`, `results_of_operations`, `material_agreement`, `other`.
Section labels are now generally correct after the P2 fix (short item headings like
"Item 1A. Risk Factors" were previously dropped by the MIN_PARAGRAPH_CHARS filter).

## Answer contract (enforced in `ask.py`)

Every claim must carry: (1) the claim, (2) a short supporting quote from the evidence,
(3) a precise source (ticker, form, period, section, source_url), (4) a confidence level.
Answer only from retrieved evidence; abstain on thin evidence (low `rerank_score`).
"The filings don't cover this" is a valid, expected answer.

## Conversation context (`ask.py` + `api.py`)

`ask()` accepts optional `history` ([{"question","answer"}, ...]). Used for (1) retrieval
(reuse a prior turn's ticker when the follow-up names none) and (2) prompting (prepend
recent turns as context). Frontend sends `history: conversationHistory.slice(-3)` and
clears it on "New question."

## Project layout

- `verify_setup.py` — checks both API keys.
- `ingest.py` — FROZEN (see Freeze rules). Builds `data/corpus.jsonl`.
- `embed_and_search.py` — FROZEN (see Freeze rules). Builds index; `search()` + helpers.
- `ask.py` — current. RAG answering with rigor contract + conversation context.
- `api.py` — current. FastAPI `/query`, `/evals/results`, `/evals/ragas`, `/health`.
- `index.html` — current. Chat frontend with waking-up state.
- `dashboard.html` — current. Eval dashboard with RAGAS section.
- `evals/eval.py`, `evals/dataset.json`, `evals/last_results.json` — current.
- `evals/eval_ragas.py` — current. Optional RAGAS LLM-judge layer (see Known limitations).
- `evals/results/` — RAGAS output files written here when eval_ragas.py is run.
- `data/` — generated. `corpus.jsonl` is committed so the index can rebuild without
  re-scraping; `chroma/` is gitignored.
- `.env` (gitignored) / `.env.example` (committed template).
- `render.yaml`, `start.sh`, `config.js` — deploy config.

## Commands

- Env: `source .venv/bin/activate`
- Install: `python -m pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Backend: `uvicorn api:app --reload --port 8000`
- Frontend: `python -m http.server 5500` then open `http://localhost:5500/dashboard.html`
  (use localhost:5500, not file://, so CORS to the API works)
- Evals (API must be running): `python evals/eval.py` (or `--group factual`)
- RAGAS eval (optional, API must be running): `python evals/eval_ragas.py --subset 10`
  NOTE: blocked on Python 3.14 + nest_asyncio incompatibility; run on Python 3.11/3.12.
- Rebuild index (only if corpus changed): `python embed_and_search.py --rebuild`
- Interactive search: `python embed_and_search.py [--ticker AAPL --section mda | --diverse]`
- Normal run prints: `Index already built (5438 vectors). Skipping embedding.`

## Tech stack and key decisions

- Python in `.venv`.
- OpenAI `text-embedding-3-small` for embeddings; Anthropic Claude for answers. Keys in `.env`.
- Chroma (local, cosine) vector store; may move to hosted pgvector / managed DB at deploy.
- Answer model: `claude-haiku-4-5-20251001` for dev (`ANSWER_MODEL` in `ask.py`); swap to
  `claude-sonnet-4-6` for sharper answers (planned for the deployed demo in P3).
- Embedding uses batching + exponential backoff; index build is incremental.

## Known limitations (be honest about these; do not oversell)

- The public API has per-IP rate limiting (10 req/min, 200 req/day) and a global daily
  cap (2000 req/day). CORS is tightened to an allowlist. These protect against casual
  abuse; a determined attacker with rotating IPs is not the threat model.
- RAGAS eval is blocked on Python 3.14. `nest_asyncio` (a ragas dependency) patches
  `asyncio.run()` in a way that makes `asyncio.current_task()` always return None.
  Python 3.14 added a strict task-context check to `asyncio.timeout()` (and downstream
  libraries like `anyio`/`sniffio` that the OpenAI async client uses), causing all metric
  calls to fail. Run `eval_ragas.py` on Python 3.11 or 3.12 until ragas fixes this.
- Section labels are now correct after the P2 fix: Item 1A (Risk Factors) and other
  Part I sections were absent because html_to_text's MIN_PARAGRAPH_CHARS=40 dropped short
  headings like "Item 1A. Risk Factors" (21 chars). Fixed in ingest.py + embed_and_search.py
  with _SEC_HEADING_RE; re-ingested to 8,141 chunks. INTC is an exception: only 22 chunks
  total because their XBRL-inline 10-K HTML produces very few leaf text blocks. Known gap.
- Cross-company answers use structured per-company retrieval. ✓ Done in P2.
- Temporal answers use structured per-period grouping. ✓ Done in P2.
- Cold starts. Render free tier spins down after ~15 min idle and is slow to wake.
- Reranking (`KEYWORD_BOOST=0.08`) and query expansion (AI-only) are hardcoded heuristics,
  not tuned against the eval set.
- 8-Ks are noisy (capped at 8 most recent per company).
- `avgo_gross_margin` eval case fails: retrieval surfaces restructuring 8-K chunks instead
  of the margin table; the model correctly abstains. Known retrieval gap.

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

- `ingest.py` and `embed_and_search.py` are FROZEN for P3.
- Any change to either requires: re-running `ingest.py`, then
  `python embed_and_search.py --rebuild`, then `python evals/eval.py` to confirm no
  regression. Confirm with the user before rebuilding, since re-embedding costs money
  and time.
- Do not change the `/query` request/response JSON contract (the frontend depends on it).
  P0 may ADD error responses (429/503) but must not alter the success shape.

# v2 — prioritized work plan

Work top-down. Each item lists tasks and acceptance criteria. Do not start a lower item
until the one above is done or skipped by the user.

## P0 — Protect the public endpoint ✓ DONE

Per-IP rate limiting (slowapi), global daily cap, 500-char question limit, CORS allowlist,
localhost exemption for the eval harness. Deployed frontend and localhost both work.

## P1 — Make the evals trustworthy ✓ DONE

Deterministic suite hardened: `answer_hit=None` for retrieval-only cases (faithfulness
denominator is now honest — 11/12, not 11/100); `INSUFFICIENT_EVIDENCE` marker checked
first for abstain detection; numeric-tolerant needle matching. `ask.py` emits the marker
on programmatic abstain paths. `ask_with_contexts()` added for RAGAS without duplicating
retrieval. Dashboard RAGAS section wired up (shows placeholder until scores are run).
RAGAS scoring blocked on Python 3.14 + nest_asyncio — see Known limitations.

## P2 — Make the shallow features real (depth over breadth) ✓ DONE

- **Cross-company comparison** (`ask.py`): when the question names specific companies,
  `_ask_cross_company()` retrieves evidence per-ticker separately (guaranteed coverage),
  assembles a company-by-company evidence table, and uses `build_cross_company_prompt()`.
  "Which companies" questions use `BROAD_DIVERSE_MULTIPLIER=16` for wider ticker coverage.
  Cross-company evals: 22/22 (100%). Added 2 new cross-company cases (3-company structured).

- **Temporal structured retrieval** (`ask.py`): single-company trend questions route to
  `_ask_temporal()`, which retrieves a large pool (80 candidates), groups by period,
  takes the top chunk per period, sorts chronologically, and uses `build_temporal_prompt()`.
  `is_temporal_question()` detects trend intent without false-positives on YoY factual cases.
  Temporal evals: 12/12 (100%). Added 2 new temporal cases (min_unique_periods=2/3).

- **Section labeling** (`ingest.py` + `embed_and_search.py`): fixed root cause where
  `html_to_text`'s MIN_PARAGRAPH_CHARS=40 silently dropped item headings (e.g. "Item 1A.
  Risk Factors" = 21 chars). Added `_SEC_HEADING_RE` to preserve Part/Item boundary markers;
  updated `canonical_section` to use compact (no-space) matching for split-word headings
  (e.g. Microsoft's "RIS K FACTORS" rendering). Corpus grew from 5,438 → 8,141 chunks.
  Re-ingested, rebuilt index, re-ran evals: 103/104 (99%), no regression.

- **Evals**: 104 total cases (4 new P2 cases), 103/104 (99%). Retrieval hit-rate: 100%.

- **Theme tracking** (optional, not done): track a fixed theme list across periods.

### Still possible

- Theme tracking: optional follow-on if desired.
- Theme tracking (optional, after section fix).

Acceptance:
- ✓ Cross-company and temporal answers come from explicit per-entity retrieval, not luck.
- ✓ 4 new eval cases pass (2 cross-company, 2 temporal). Evals: 103/104 (99%).
- Section fix and theme tracking still pending (see above).

## P3 — Engineering hygiene

Tasks:
- Unit tests (pytest) for pure functions: `build_where`, `diversify_results`,
  `canonical_section`, chunking, `detect_tickers`. No network or paid calls.
- Make `ANSWER_MODEL` per-request configurable; use `claude-sonnet-4-6` for the deployed
  demo (cost is low at demo volume), Haiku as the dev default.
- Cache identical `(question, where, diverse)` results to cut repeat cost/latency.
- Keep this `CLAUDE.md` updated as state changes.

## Deploy hardening (fixes the cold-start pain)

Problem: Render free tier spins down after ~15 min idle; cold start is slow, made worse
because boot loads/builds the Chroma index.

Do, in order:
1. Lazy-init the collection on first request, not at import or in `start.sh`, and make
   `/health` respond instantly without touching the index. This stops health checks and
   keep-alive pings from blocking on index load.
2. Frontend "waking up" state: when a request is slow or returns a cold-start error, show a
   clear "backend is waking up, this can take ~30-60s" message and auto-retry, so a cold
   start does not look broken.
3. Persist the index so it never re-embeds on boot: uncomment the persistent-disk block in
   `render.yaml`, mounted at the Chroma path (requires the Starter plan).
4. For no cold starts at all: move to an always-on instance (Render Starter ~$7/mo) or
   Fly.io with `min_machines_running=1`. A keep-alive ping (UptimeRobot every 5 min) only
   reduces, not eliminates, cold starts; document it as a demo tradeoff, not production.

✓ #1 done: collection lazy-inits on first `/query`; `/health` responds instantly.
✓ #2 done: frontend shows waking-up message after 5s and auto-retries.
#3/#4 still need payment (Render Starter plan or Fly.io).

## Definition of done for v2

P0 + P1 complete, the three P2 features working with passing eval cases, P3 tests in place,
and the deploy no longer cold-starts into a broken-looking state. README and this file
reflect reality at every step.