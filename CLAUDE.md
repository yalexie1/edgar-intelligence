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

- `ingest.py` and `embed_and_search.py` are UPGRADED and FROZEN. Do not refactor or
  re-run them unless `data/corpus.jsonl` changes. The Chroma index already holds
  ~5,438 vectors across 13 companies.
- `ask.py` and `api.py` are STALE. They were written for the old single-company
  `search()` and no longer match the new one (different signature and result shape,
  see below). They will not work correctly until rewritten.
- `index.html` still says "Apple's 2025 10-K" and needs generalizing to the corpus.

**Immediate next task: rewrite `ask.py` to use the new `search()` and enforce the
rigor contract (see "Answer contract" below). Then update `api.py` and `index.html`
to match.** Do not touch `ingest.py` or `embed_and_search.py` to do this.

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
- Answering (`ask.py`): retrieves evidence and asks Claude for a grounded answer.
  (Currently stale; this is the next thing to rebuild.)
- Backend (`api.py`, FastAPI): exposes `/query` over HTTP. (Stale; rebuild after ask.)
- Frontend (`index.html`): single-page UI that calls `/query`. (Still Apple-only.)

## The retrieval interface (what `ask.py` must call)

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

Note: this is NOT the old `{index, similarity, text}` shape, and the signature takes
`where` before `k`. Old `ask.py` calls `search(collection, question, k)` positionally,
which now passes `k` as the `where` filter. That is the core bug to fix.

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

## Answer contract (the rigor requirement for `ask.py`)

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

## Project layout

- `verify_setup.py` — checks both API keys work.
- `ingest.py` — FROZEN. Builds `data/corpus.jsonl`.
- `embed_and_search.py` — FROZEN. Builds the index; provides `search()` and helpers.
- `ask.py` — STALE. Rewrite next against the new `search()` + answer contract.
- `api.py` — STALE. FastAPI `/query`; update after `ask.py`.
- `index.html` — frontend; generalize beyond Apple.
- `data/` — generated artifacts (`corpus.jsonl`, `chroma/`). Gitignored.
- `.env` — secret keys. Gitignored. `.env.example` is the committed template.

## Commands

- Activate env: `source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Rebuild index (only if corpus changed): `python embed_and_search.py --rebuild`
- Search (interactive), examples:
  - `python embed_and_search.py`
  - `python embed_and_search.py --ticker AAPL --section mda`
  - `python embed_and_search.py --ticker NVDA --form 10-Q`
  - `python embed_and_search.py --diverse` (cross-company)
- A normal run should print `Index already built (5438 vectors). Skipping embedding.`

## Tech stack and key decisions

- Python in a `.venv`. Always activate before running.
- Two providers on purpose: OpenAI `text-embedding-3-small` for embeddings, Anthropic
  Claude for answers. Keys in `.env`.
- Chroma (local, cosine) for the vector store; may move to hosted pgvector at deploy.
- Answer model: Claude Haiku for dev, Sonnet for quality (`ANSWER_MODEL` in `ask.py`).
- Embedding uses batch + exponential backoff on rate limits; index build is incremental.

## Known issues

- IMPORTANT: section metadata is sometimes wrong. Some risk-factor text is mislabeled
  `section=market_risk` / `item=3` (seen in AMD, META, CRM, MSFT, AAPL). Do not rely
  on a hard `section=risk_factors` filter for risk questions; prefer filtering by
  ticker only, or no section filter, and let retrieval + reranking find it. Fixing the
  section boundary detection is a future ingestion improvement, not an urgent one.
- 8-Ks are noisy by nature (capped at 8 most recent per company). Could later filter to
  earnings-related 8-Ks (Item 2.02).
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

## Roadmap (expansion plan)

- A. Multi-company, multi-form ingestion with metadata. DONE.
- B. Filtered retrieval (+ query expansion, lexical reranking, diversity mode). DONE.
- C. Earnings-call transcripts via Financial Modeling Prep (needs an FMP key). PENDING.
- D. Rigor contract: claim + quote + source + confidence; abstain when weak. PENDING
  (build into the `ask.py` rewrite).
- E. Temporal diffing ("what changed in risk factors vs last year"): retrieve the same
  section across two periods and have the model compare. PENDING.
- F. Cross-company comparison ("compare NVDA/AMD/INTC on AI data-center demand"):
  per-company scoped retrieval, assemble an evidence table, reason over it. Diversity
  mode is the seed of this. PENDING.
- G. Theme tracking (demand, margins, inventory, pricing, China, AI, capex, competition,
  guidance) across periods for one company. PENDING.
- H. Evaluation harness + dashboard: retrieval hit rate, citation accuracy, answer
  faithfulness, broken down by single-lookup vs multi-document questions. PENDING.