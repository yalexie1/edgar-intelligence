# EDGAR Intelligence

Ask plain-English questions about a public company's SEC filings and get answers with citations, built with retrieval-augmented generation (RAG).

## Status

Phase 0: project setup and API-key check.

## Setup

1. Create accounts and API keys at `platform.openai.com` (embeddings) and `console.anthropic.com` (Claude).
2. Set a spending cap on each account under its billing / usage-limits section. Add a few dollars of credit if prompted.
3. Create and activate a virtual environment:
   - macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy the env template and paste your keys into the new file: `cp .env.example .env`
6. Check that both providers respond: `python verify_setup.py`

When `verify_setup.py` prints "Phase 0 complete," you're ready for Phase 1: downloading and chunking a 10-K.

## Notes

- `.env` holds your secret keys and is gitignored. Never commit it, and never paste keys anywhere public.
- This project targets one company at a time to start. First target: Apple (ticker `AAPL`, CIK `0000320193`).
