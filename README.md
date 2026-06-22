# EDGAR Intelligence

Ask plain-English questions about SEC filings from 13 large public companies and get grounded, cited answers — built with retrieval-augmented generation (RAG).

## What it does

- Searches 5,438 embedded chunks from 10-K, 10-Q, and 8-K filings across AAPL, MSFT, GOOGL, AMZN, META, NVDA, AVGO, TSLA, ORCL, CRM, AMD, NFLX, and INTC
- Every answer cites the exact passage, filing form, period, section, and a link to the original SEC document
- Follow-up questions carry conversation context — ask "what was Apple's revenue?" then "how did management explain the growth?" and it stays focused
- Auto-detects company names in questions and applies the right metadata filter automatically
- Cross-company questions ("which company has the most hardware risk?") use a diversity mode that ensures each company gets at least one evidence slot
- Abstains honestly when the corpus doesn't cover the question

## Architecture

```
ingest.py          →  data/corpus.jsonl   →  embed_and_search.py  →  data/chroma/
(offline, frozen)      (chunked filings)      (Chroma index)

ask.py  ←→  api.py  (FastAPI, port 8000)  ←→  index.html  (chat UI)
                                           ←→  dashboard.html  (eval metrics)
```

- **Embeddings**: OpenAI `text-embedding-3-small`
- **Answers**: Anthropic Claude Haiku (swap `ANSWER_MODEL` in `ask.py` for Sonnet)
- **Vector store**: Chroma (local, cosine distance)
- **Retrieval**: semantic search + query expansion + lexical reranking + diversity mode

## Setup

1. Create API keys at [platform.openai.com](https://platform.openai.com) and [console.anthropic.com](https://console.anthropic.com).
2. Clone this repo and create a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux
   python -m venv .venv && .venv\Scripts\activate        # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the env template and add your keys:
   ```bash
   cp .env.example .env
   # then edit .env and fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```
5. Verify both providers respond:
   ```bash
   python verify_setup.py
   ```
6. The Chroma index is pre-built (`data/` is gitignored — you'll need to rebuild it):
   ```bash
   python embed_and_search.py --rebuild
   ```
   This embeds `data/corpus.jsonl` and should print `Embedded N chunks` then exit.
   Subsequent runs print `Index already built (5438 vectors). Skipping embedding.`

> **Note**: `data/corpus.jsonl` and `data/chroma/` are gitignored. To regenerate
> the corpus from scratch run `python ingest.py` (makes live SEC EDGAR requests —
> takes several minutes and consumes OpenAI embedding credits).

## Running

```bash
# 1. Start the API server
source .venv/bin/activate
uvicorn api:app --reload --port 8000

# 2. Open the frontend
open index.html          # or serve with: python -m http.server 8080
```

The API also exposes:
- `GET /health` — confirms the index is loaded
- `GET /docs` — interactive API docs (FastAPI auto-generated)
- `GET /evals/results` — last eval run results (used by the dashboard)

## Eval harness

A 100-case golden dataset (`evals/dataset.json`) covers four question types:

| Group | Cases | What it tests |
|---|---|---|
| factual | 44 | Single-lookup facts (revenue figures, product descriptions) |
| temporal | 10 | Multi-period synthesis (trend questions) |
| cross_company | 20 | Cross-company comparison (diversity mode) |
| abstain | 26 | Out-of-corpus questions (should refuse to answer) |

Run the full suite (requires the API to be running):
```bash
python evals/eval.py
```

Last result: **96/100** — retrieval 95%, answer faithfulness 98%, abstain precision 100%.

Results are saved to `evals/last_results.json` and visible at `dashboard.html`.

## Notes

- `.env` holds secret keys and is gitignored. Never commit it.
- Every SEC EDGAR request sends a `User-Agent` header (required by SEC policy).
- `ingest.py` and `embed_and_search.py` are frozen — do not modify them unless
  `data/corpus.jsonl` changes.
