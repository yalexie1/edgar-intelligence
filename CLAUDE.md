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

v1 is a working, deployed MVP. It is not production-grade yet, and v2 (below) is about
closing that gap. Honest framing of where it stands:

- Scale: 13 companies, ~5,438 chunks, 10-K / 10-Q / 8-K over ~2 years.
- Frontend: a basic single-page chat UI. Functional, not polished.
- Backend: FastAPI on Render's free tier. It spins down after ~15 min idle and is slow
  to cold-start. Fixing that is an explicit v2 task (see "Deploy hardening").
- Evals: a 100-case golden set; last run 96/100. Retrieval and abstain numbers are solid;
  the faithfulness number is softer than it looks (see "Known limitations").

Pipeline file state:
- `ingest.py` and `embed_and_search.py` are FROZEN for P0/P1/P3 work. They may only be
  modified for the P2 section-label fix, which requires re-ingest + `--rebuild` + re-run
  evals. See "Freeze rules."
- `ask.py`, `api.py`, `index.html`, `dashboard.html`, `evals/eval.py`,
  `evals/dataset.json` are current and may be edited for v2.

**Start at v2 -> P0. Work top-down. Do not begin a lower priority until the one above is
done or the user explicitly says to skip it.**

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
- Backend (`api.py`, FastAPI): `/query` (POST), `/evals/results` (GET), `/health` (GET).
- Frontend (`index.html`): single-page chat UI with a follow-up bar, autocomplete,
  tooltips, filter controls, and a "New question" reset.
- Eval harness (`evals/eval.py`, `evals/dataset.json`): 100-case golden set with per-case
  scoring; writes `evals/last_results.json`, served by the API for the dashboard.
- Eval dashboard (`dashboard.html`): fetches `/evals/results` and renders metric cards,
  per-group bars, question-type breakdown, and a sortable case table.

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
(See "Known limitations": these labels are sometimes wrong.)

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
- `api.py` — current. FastAPI `/query`, `/evals/results`, `/health`.
- `index.html` — current. Chat frontend.
- `dashboard.html` — current. Eval dashboard.
- `evals/eval.py`, `evals/dataset.json`, `evals/last_results.json` — current.
- `data/` — generated. `corpus.jsonl` is committed so the index can rebuild without
  re-scraping; `chroma/` is gitignored.
- `.env` (gitignored) / `.env.example` (committed template).
- `render.yaml`, `start.sh`, `config.js` — deploy config.

## Commands

- Env: `source .venv/bin/activate`
- Install: `python -m pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Backend: `uvicorn api:app --reload --port 8000`
- Frontend: open `index.html` or `python -m http.server`
- Evals (API must be running): `python evals/eval.py` (or `--group factual`)
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

- The public API is unauthenticated and unthrottled. `/query` spends paid OpenAI +
  Anthropic credits on every call. CORS does NOT protect this: CORS only restricts
  browsers, and any script can call the endpoint directly. Real protection is rate
  limiting + a spend cap (P0).
- Deterministic faithfulness is weak. `eval.py` checks `answer_contains` by substring and
  defaults `answer_hit=True` when a case has no needles, so the faithfulness figure is
  softer than it reads. Abstain detection is keyword-based. (P1 fixes this.)
- Section labels are sometimes wrong. `canonical_section` derives from `item_title`, which
  Part/Item boundary detection mislabels in places (risk factors sometimes tagged
  `market_risk`/`item=3`, seen in AMD, META, CRM, MSFT, AAPL). Do not rely on a hard
  `section=risk_factors` filter; prefer ticker-only. (P2 fixes this.)
- Cross-company and temporal answers are shallow. Cross-company uses one-chunk-per-company
  diversity (3/20 cross-company cases fail); temporal relies on retrieval happening to
  surface two periods, with no real diffing. (P2 builds structured versions.)
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

- `ingest.py` and `embed_and_search.py` are FROZEN for P0, P1, and P3.
- They may ONLY be modified for the P2 section-label fix. Any change to either requires:
  re-running `ingest.py`, then `python embed_and_search.py --rebuild`, then
  `python evals/eval.py` to confirm no regression. Confirm with the user before rebuilding,
  since re-embedding costs money and time.
- Do not change the `/query` request/response JSON contract (the frontend depends on it).
  P0 may ADD error responses (429/503) but must not alter the success shape.

# v2 — prioritized work plan

Work top-down. Each item lists tasks and acceptance criteria. Do not start a lower item
until the one above is done or skipped by the user.

## P0 — Protect the public endpoint (active liability; do first)

The endpoint is live and spends money on every call with no throttle.

Tasks:
- Add per-IP rate limiting to `api.py` (e.g. `slowapi`): ~10 req/min and ~200 req/day per
  IP on `/query`; return 429 with a clear message.
- Add a global daily request counter (in-memory is fine for a single instance); past a cap
  (e.g. 2000/day) return 503.
- Reject questions longer than ~500 chars.
- Then harden CORS as defense-in-depth: replace `allow_origins=["*"]` with an env-driven
  allowlist `ALLOWED_ORIGINS` (comma-separated) plus defaults for the Vercel domain and
  `http://localhost:3000`, `http://localhost:5173`, `http://127.0.0.1:5500`. Set
  `allow_credentials=False`, methods `GET, POST, OPTIONS`.
- Re-add `.env.example` with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, optional
  `ALLOWED_ORIGINS`. Document these in the README.
- Confirm provider spending caps are tight as the backstop.

Acceptance:
- Abusive loops hit 429/503; the deployed frontend and localhost still work.
- `api.py` no longer uses `allow_origins=["*"]`.
- `/query` success JSON is unchanged. README documents env vars.

## P1 — Make the evals trustworthy

Deterministic suite stays the PRIMARY regression test. Strengthen it, then add RAGAS as a
complementary judge.

Tasks:
- Strengthen `evals/eval.py`: numeric-tolerant matching for figure facts; stop defaulting
  `answer_hit=True` when a case lacks `answer_contains` (require needles or mark the case
  retrieval-only); make abstain detection robust by having `ask.py` emit an explicit
  marker (e.g. `INSUFFICIENT_EVIDENCE`) the harness can check instead of sniffing prose.
- Add `evals/eval_ragas.py` (complementary, optional): metrics faithfulness,
  answer_relevancy, context_precision, context_recall. Reuse the pipeline via a new helper
  in `ask.py`: `ask_with_contexts(question, where=None, k=5, diverse=False) ->
  {"answer", "contexts", "sources"}`. Do not duplicate retrieval. Save
  `evals/results/ragas_results.json`, `ragas_results.csv`, `ragas_summary.json`. Run a
  10-case subset first, then the full set. Add `ragas, datasets, pandas` to
  `requirements.txt`, pinned; keep RAGAS optional so the project runs without it.
- README: document both layers; state plainly that RAGAS is an automated LLM judge that
  varies by model/version and is complementary, not ground truth. Add a metrics table with
  real numbers.
- Dashboard: add a RAGAS section (load `ragas_summary.json` or hardcode latest values).

Acceptance:
- Deterministic suite still runs and stays primary; the faithfulness check is no longer a
  pure substring default.
- `python evals/eval_ragas.py` runs and saves all three files.
- README + dashboard show both layers. Nothing breaks `ask.py`, `api.py`, or the frontend.

## P2 — Make the shallow features real (depth over breadth)

Tasks:
- Section labeling (unfreezes `ingest.py`): improve Part/Item boundary detection so risk
  factors stop being mislabeled `market_risk`. Re-ingest, `--rebuild`, re-run evals.
  Confirm with the user before rebuilding (cost/time).
- Cross-company comparison (structured; in `ask.py` or a new `compare.py`): when the
  question names/implies multiple companies, retrieve a scoped evidence set PER company
  (and per period if relevant), assemble an evidence table, then synthesize. Replace the
  one-chunk-per-company diversity approach for these questions. Add eval cases.
- Temporal diffing: "what changed in X from period A to B" — retrieve the same section for
  two periods (filter on `period`/`report_date`), have the model output added/removed/
  changed. Add eval cases.
- Theme tracking (optional, after the above): track a fixed theme list (demand, margins,
  inventory, pricing, China, AI, capex, competition, guidance) across periods for one company.

Acceptance:
- Cross-company and temporal answers come from explicit per-entity retrieval, not luck.
- New eval cases pass. Frozen files touched only for the section fix, with rebuild + re-eval.

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

Recommended now: #1 and #2 (free, immediate UX win). Note #3/#4 as the real fix when the
user is willing to pay.

Acceptance:
- `/health` returns fast even when cold.
- Frontend shows a clear waking state and retries.
- The index does not re-embed on every cold start.

## Definition of done for v2

P0 + P1 complete, the three P2 features working with passing eval cases, P3 tests in place,
and the deploy no longer cold-starts into a broken-looking state. README and this file
reflect reality at every step.