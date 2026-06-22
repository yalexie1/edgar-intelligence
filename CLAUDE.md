# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

EDGAR Intelligence: a retrieval-augmented generation (RAG) system over SEC filings.
It ingests filings for many large public companies, chunks and embeds them, stores
them in a local Chroma vector database with rich metadata, and answers plain-English
questions with citations back to the source filing.

It is a solo learning project by a CS and math student learning production AI
engineering. Prefer clear, well-commented code and explain non-obvious choices.
Make minimal, focused changes; do not refactor unrelated code without asking.

## Current status (read this first)

All core pipeline files are working and tested. Nothing is stale.

- `ingest.py` and `embed_and_search.py` are FROZEN. Do not refactor or re-run them
  unless `data/corpus.jsonl` changes. The Chroma index holds 5,438 vectors across
  13 companies.
- `ask.py` is current. It uses the correct `search()` signature, enforces the rigor
  contract, auto-detects company names from questions, carries conversation history
  across follow-up questions, and abstains when evidence is thin.
- `api.py` is current. FastAPI server exposing `/query` and `/evals/results`.
- `index.html` is current. Chat interface with follow-up bar, autocomplete, (i)
  tooltips, and a "New question" reset button.
- `dashboard.html` is current. Eval dashboard with metric cards, group/type
  breakdowns, sortable case table with expandable rows.
- `evals/eval.py` and `evals/dataset.json` are current. 100-case golden set;
  last run: 96/100 (retrieval 95%, faithfulness 98%, abstain 100%).

**Next up: deploy (Phase 7) — Render for the FastAPI backend, Vercel or Netlify for
the static frontend. After deploy, consider Phase E (temporal diffing) or Phase C
(earnings-call transcripts).**

## Corpus scope

- Tickers: AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, INTC.
- Forms: 10-K, 10-Q, 8-K. Window: last ~2 years. Caps per company: 3 / 8 / 8.
- Output of ingestion: `data/corpus.jsonl` (one JSON record per chunk).
- Vector store: `data/chroma/`, Chroma collection name `sec_filings`, cosine distance.

## Architecture

- Ingestion (`ingest.py`, offline): resolves tickers to CIKs (SEC ticker map, with a
  fallback CIK dict), pulls filings via the SEC submissions API, downloads each
  document, cleans HTML (block-aware; strips `display:none` XBRL), splits by SEC
  Part/Item, chunks paragraph-aware (~4000 chars, 600 overlap), and writes JSONL with
  metadata.
- Retrieval (`embed_and_search.py`): loads the corpus, embeds with OpenAI, stores in
  Chroma with metadata, and provides semantic search with metadata filters, query
  expansion, lexical reranking, and diversity mode.
- Answering (`ask.py`): retrieves evidence and asks Claude for a grounded, cited
  answer. Auto-detects company names (and falls back to prior turns for follow-ups),
  diversifies results for cross-company questions, and enforces the answer contract.
- Backend (`api.py`, FastAPI): exposes `/query` (POST) and `/evals/results` (GET).
- Frontend (`index.html`): single-page chat UI. First question uses the top search
  bar; after each answer a follow-up bar appears inline. Autocomplete, (i) tooltips,
  filter controls, and a "New question" reset.
- Eval harness (`evals/eval.py`, `evals/dataset.json`): 100-case golden set with
  per-case scoring (retrieval hit, answer faithfulness, abstain correctness). Results
  saved to `evals/last_results.json` and served by the API for the dashboard.
- Eval dashboard (`dashboard.html`): fetches `/evals/results` and renders metric
  cards, per-group progress bars, question-type breakdown, and a sortable/filterable
  case table with expandable detail rows.

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
- `rerank_score`: float (the sort key; `similarity + KEYWORD_BOOST * lexical_score`)
- `text`: str (the chunk)
- `metadata`: dict with `ticker, company, cik, form, filing_date, report_date,
  period, accession, source_url, primary_document, part, item, part_item,
  item_title, chunk_index, section_chunk_index, section`

Other helpers available:
- `diversify_results(results, k=TOP_K, by="ticker"|"filing")`: one strong hit per
  group. Use for cross-company questions.
- `build_where(args)`: builds a Chroma `where` filter from an argparse namespace.
  For programmatic use, construct the `where` dict directly with the metadata field
  names above (e.g. `{"ticker": "NVDA"}`, or `{"$and": [{"ticker": "AMD"}, {"form": "10-Q"}]}`).
- `expanded_query`, `lexical_score`, `query_terms`, `best_snippet`.

Canonical `section` labels (from `canonical_section`): `mda`, `risk_factors`,
`market_risk`, `exhibits`, `financial_statements`, `unregistered_sales`,
`legal_proceedings`, `controls`, `results_of_operations`, `material_agreement`, `other`.

## Answer contract (enforced in `ask.py`)

Every answer must be grounded and verifiable. Each claim must carry:
1. the claim itself,
2. a short supporting quote from the retrieved evidence,
3. a precise source (ticker, form, period, section, and source_url),
4. a confidence level.

Rules:
- Answer only from retrieved evidence. If evidence is thin, say so and abstain rather
  than guess. Treat low `rerank_score`/`similarity` as an abstain signal.
- For questions naming more than one company, retrieve per company and use
  `diversify_results(..., by="ticker")` so each company is represented, then have the
  model compare across the assembled evidence.
- Keep "the filings don't cover this" as a valid, expected answer.

## Conversation context (`ask.py` + `api.py`)

`ask()` accepts an optional `history` list of `{"question": str, "answer": str}`
dicts from prior turns in the same session. This is used in two ways:
1. **Retrieval**: if the current question names no company but a prior question did,
   the prior question's ticker is used as the `where` filter, so follow-up questions
   like "how did management explain that?" stay focused on the right company.
2. **Prompting**: the last few prior turns are prepended to the prompt as "Prior
   conversation" context so the model understands references like "the 6% increase."

The frontend sends `history: conversationHistory.slice(-3)` on every request and
appends each successful turn. "New question" clears both the DOM and `conversationHistory`.

## Project layout

- `verify_setup.py` — checks both API keys work.
- `ingest.py` — FROZEN. Builds `data/corpus.jsonl`.
- `embed_and_search.py` — FROZEN. Builds the index; provides `search()` and helpers.
- `ask.py` — current. RAG answering with rigor contract and conversation context.
- `api.py` — current. FastAPI `/query` and `/evals/results`.
- `index.html` — current. Chat-style frontend with follow-up bar, autocomplete, tooltips.
- `dashboard.html` — current. Eval metrics dashboard.
- `evals/eval.py` — current. Runs the 100-case golden set; saves `last_results.json`.
- `evals/dataset.json` — 100 golden cases (44 factual, 10 temporal, 20 cross-company,
  26 abstain).
- `evals/last_results.json` — generated; last run output (gitignored is optional but
  currently committed for dashboard convenience).
- `data/` — generated artifacts (`corpus.jsonl`, `chroma/`). Gitignored.
- `.env` — secret keys. Gitignored. `.env.example` is the committed template.

## Commands

- Activate env: `source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Run backend: `uvicorn api:app --reload --port 8000`
- Open frontend: open `index.html` in a browser (or serve with `python -m http.server`)
- Run evals: `python evals/eval.py` (requires the API to be running on port 8000)
- Run evals for one group only: `python evals/eval.py --group factual`
- Rebuild index (only if corpus changed): `python embed_and_search.py --rebuild`
- Search interactively:
  - `python embed_and_search.py`
  - `python embed_and_search.py --ticker AAPL --section mda`
  - `python embed_and_search.py --ticker NVDA --form 10-Q`
  - `python embed_and_search.py --diverse`
- A normal run prints `Index already built (5438 vectors). Skipping embedding.`

## Tech stack and key decisions

- Python in a `.venv`. Always activate before running.
- Two providers on purpose: OpenAI `text-embedding-3-small` for embeddings, Anthropic
  Claude for answers. Keys in `.env`.
- Chroma (local, cosine) for the vector store; may move to hosted pgvector at deploy.
- Answer model: `claude-haiku-4-5-20251001` for dev speed (`ANSWER_MODEL` in `ask.py`).
  Swap to `claude-sonnet-4-6` for sharper answers.
- Embedding uses batch + exponential backoff on rate limits; index build is incremental.

## Known issues

- Section metadata is sometimes wrong. Some risk-factor text is mislabeled
  `section=market_risk` / `item=3` (seen in AMD, META, CRM, MSFT, AAPL). Do not rely
  on a hard `section=risk_factors` filter for risk questions; prefer filtering by
  ticker only, or no section filter. Fixing the section boundary detection is a future
  ingestion improvement.
- 8-Ks are noisy by nature (capped at 8 most recent per company). Could later filter to
  earnings-related 8-Ks (Item 2.02).
- The `avgo_gross_margin` eval case fails because retrieval surfaces Broadcom 8-K /
  restructuring chunks rather than the gross margin table. The model correctly abstains.
  This is a known retrieval gap; fixing it requires ingestion improvements.
- Duplicate Chroma IDs are handled in `embed_and_search.py` with deterministic suffixes;
  do not reintroduce raw IDs.

## Conventions and rules

- IMPORTANT: never commit `.env` or anything with API keys. If a key is ever committed,
  treat it as compromised and rotate it at the provider.
- IMPORTANT: every SEC EDGAR request must send a real User-Agent (name + email), or the
  SEC returns 403. Set it in `ingest.py`.
- When extracting filing text, strip `display:none` elements first (hidden XBRL noise).
- Keep code readable and commented; explain trade-offs. This is a learning project.
- Make minimal, focused changes. Do not rebuild the index unless `corpus.jsonl` changes.

## Roadmap

- A. Multi-company, multi-form ingestion with metadata. **DONE.**
- B. Filtered retrieval (query expansion, lexical reranking, diversity mode). **DONE.**
- C. Earnings-call transcripts via Financial Modeling Prep (needs an FMP key). PENDING.
- D. Rigor contract: claim + quote + source + confidence; abstain when weak. **DONE.**
- E. Temporal diffing ("what changed in risk factors vs last year"): retrieve the same
  section across two periods and have the model compare. PENDING.
- F. Cross-company comparison: per-company scoped retrieval + diversity mode. **DONE**
  (basic form; `detect_tickers`, `is_cross_company_question`, `diversify_results`).
- G. Theme tracking (demand, margins, inventory, China, AI, capex, competition, guidance)
  across periods for one company. PENDING.
- H. Evaluation harness + dashboard: 100-case golden set, retrieval hit rate, answer
  faithfulness, abstain precision, breakdown by question type. **DONE.**
- I. Conversation context: history-aware retrieval and prompting across follow-ups.
  **DONE.**
- J. Deploy: Render (FastAPI backend) + Vercel/Netlify (static frontend). PENDING.
