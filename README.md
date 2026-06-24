# EDGAR Intelligence (v2)

Ask plain-English questions about SEC filings from 13 large public companies and get grounded, cited answers — built with retrieval-augmented generation (RAG).

Try at: https://edgar-intelligence.vercel.app/

NOTE: The backend is hosted on Render and spins down every 15 minutes when there's no activity. Please give up to 30-60 seconds for the backend to reboot if you're accessing the site for the first time.

## What it does

- Searches 8,141 embedded chunks from 10-K, 10-Q, and 8-K filings across AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, and INTC
- Every answer cites the exact passage, filing form, period, section, and a link to the original SEC document
- Follow-up questions carry conversation context — ask "what was Apple's revenue?" then "how did management explain the growth?" and it stays focused
- Auto-detects company names in questions and applies the right metadata filter automatically
- Cross-company questions ("compare AAPL and MSFT cloud margins") retrieve evidence per-ticker separately so every named company is guaranteed representation
- Trend questions ("how has NVDA's gross margin changed?") retrieve across multiple filing periods and present results chronologically
- Abstains honestly when the corpus doesn't cover the question

## Architecture

```
ingest.py          →  data/corpus.jsonl   →  embed_and_search.py  →  Pinecone (cloud)
(offline, frozen)      (chunked filings)      (embedding + upload)

ask.py  ←→  api.py  (FastAPI, Render)  ←→  index.html     (chat UI)
                                        ←→  dashboard.html  (eval metrics)
                                        ←→  themes.html     (theme tracker)
```

- **Embeddings**: OpenAI `text-embedding-3-small` (1536-dim)
- **Answers**: Anthropic Claude (`claude-sonnet-4-6` on Render; `claude-haiku-4-5-20251001` as dev default)
- **Vector store**: Pinecone serverless (AWS us-east-1, cosine, free tier)
- **Retrieval**: semantic search + query expansion + lexical reranking + structured per-entity retrieval for cross-company and temporal questions

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yalexie1/edgar-intelligence.git
cd edgar-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file with your API keys:

```bash
cp .env.example .env
```

Fill in your keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

The vector index lives in Pinecone (cloud). The processed corpus is included at `data/corpus.jsonl`. To upload vectors to your own Pinecone index (creates the `sec-filings` index on first run):

```bash
python embed_and_search.py --rebuild
```

This costs roughly $0.03 in OpenAI embedding fees and only needs to run once. Subsequent runs skip the upload:

```bash
python embed_and_search.py
# Index already built (8141 vectors). Skipping embedding.
```

To run the API locally:

```bash
uvicorn api:app --reload --port 8000
```

Then open the frontend via a local server (not `file://`, so CORS works):

```bash
python -m http.server 5500
# open http://localhost:5500/index.html
```

## Running

```bash
# 1. Start the API server
source .venv/bin/activate
uvicorn api:app --reload --port 8000

# 2. Serve the frontend
python -m http.server 5500
# open http://localhost:5500/index.html
# open http://localhost:5500/dashboard.html
# open http://localhost:5500/themes.html
```

The API exposes:
- `POST /query` — answer a question with cited sources (rate-limited: 10/min, 200/day per IP)
- `GET /themes?ticker=NVDA` — retrieval-only theme heat map (no LLM cost)
- `GET /health` — liveness check
- `GET /evals/results` — last eval run (used by the dashboard)
- `GET /docs` — interactive API docs (FastAPI auto-generated)

## Eval harness

Two complementary evaluation layers. The deterministic suite is the primary regression test; RAGAS is a secondary LLM-judge layer.

### Layer 1 — Deterministic suite (primary)

A 104-case golden dataset (`evals/dataset.json`) covers four question types:

| Group | Cases | What it tests |
|---|---|---|
| factual | 44 | Single-lookup facts (revenue figures, product descriptions) |
| temporal | 12 | Multi-period synthesis (trend questions, min 2–3 unique periods) |
| cross_company | 22 | Per-company structured retrieval (2- and 3-company comparisons) |
| abstain | 26 | Out-of-corpus questions (should refuse to answer) |

Run the full suite (requires the API to be running):

```bash
python evals/eval.py
```

**Answer faithfulness** is scored only on cases with expected strings in `answer_contains` (12 cases). Cases without expected strings are retrieval-only checks and excluded from the faithfulness denominator.

Last result: **103/104 (99%)** — retrieval 100%, abstain precision 100%, cross-company 100%, temporal 100%.  
Results are saved to `evals/last_results.json` and visible at `dashboard.html`.

### Layer 2 — RAGAS (complementary, optional)

RAGAS computes LLM-as-a-judge metrics: faithfulness, answer relevancy, context precision, and context recall. Scores complement the deterministic suite but are not ground truth — they vary by judge model and version.

> **Note:** Currently blocked on Python 3.14 + `nest_asyncio` incompatibility. Run on Python 3.11 or 3.12.

Requires extra dependencies:

```bash
pip install "ragas>=0.1.9" datasets pandas
```

Run a 10-case smoke test first, then the full set:

```bash
python evals/eval_ragas.py --subset 10
python evals/eval_ragas.py
```

Results are saved to `evals/results/` and shown in the eval dashboard under "RAGAS metrics."

### Unit tests

Pure-function tests — no network or paid API calls:

```bash
python -m pytest tests/
# 51 passed
```

Covers `canonical_section`, `build_where`, `diversify_results`, `detect_tickers`, `chunk_section`.

## Theme tracker

`themes.html` shows how strongly 8 predefined themes (AI/ML, Cybersecurity, Supply Chain, Regulation, China/Geopolitics, Climate/ESG, Competition, Cloud/Platform) appear in a company's filings across reporting periods. Scores are cosine + lexical rerank values — not frequency counts or sentiment. Retrieval-only, no LLM cost.

## Notes

- `.env` holds secret keys and is gitignored. Never commit it.
- Every SEC EDGAR request sends a `User-Agent` header (name + email, required by SEC policy).
- `ingest.py` is frozen — do not modify unless you intend to rebuild the full corpus from scratch.
- The public endpoint has per-IP rate limiting (10 req/min, 200 req/day) and a global daily cap (2000 req/day).
