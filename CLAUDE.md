# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

EDGAR Intelligence: a retrieval-augmented generation (RAG) app that answers
plain-English questions about a public company's SEC filings, with citations back
to the source passage. It is a solo learning project built by a CS and math
student to learn production AI engineering, so prefer clear, well-commented code
and explain non-obvious choices rather than writing the cleverest possible version.

## Current status

Phase 1 is complete: EDGAR ingestion works end to end. Next up is Phase 2
(embed the chunks, store them in Chroma, build semantic search). See the build
plan at the bottom.

## Architecture

- Frontend (React, later): chat UI, company picker, citations panel.
- Backend (FastAPI, later): the orchestrator. It does no AI itself; it routes.
- Vector DB (Chroma, local): stores chunk embeddings, runs nearest-neighbor search.
- Embedding model (OpenAI `text-embedding-3-small`): turns text into 1,536-dim vectors.
- LLM (Anthropic Claude): writes the grounded, cited answer from retrieved chunks.
- Ingestion (offline Python): downloads a 10-K, cleans it, chunks it. Run once per company.

Query flow: question -> backend -> embed the query -> Chroma returns the top-k
nearest chunks -> backend builds a prompt -> Claude answers with citations ->
backend -> frontend.

## Tech stack and key decisions

- Python, with a `.venv` virtual environment. Always activate it before running anything.
- Two API providers on purpose: OpenAI for embeddings, Anthropic for the Claude answer model. Keys live in `.env`.
- Vector store starts as local Chroma (free, persists to disk); may move to hosted pgvector at deploy time.
- Embeddings: `text-embedding-3-small` is the cheap default. Upgrade only if retrieval quality is poor.
- Answers: Claude Haiku for dev, Sonnet for quality.

## Project layout

- `ingest.py` — Phase 1: fetch, clean, and chunk a 10-K. Saves to `data/`.
- `verify_setup.py` — confirms both API keys work.
- `data/` — generated artifacts (raw filing, `chunks.json`). Gitignored.
- `.env` — secret keys. Gitignored. `.env.example` is the committed template.

## Commands

- Activate env: `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Verify keys: `python verify_setup.py`
- Ingest Apple's latest 10-K: `python ingest.py`

## Conventions and rules

- IMPORTANT: never commit `.env` or anything containing API keys. Keys belong only in `.env`, which is gitignored. If a key is ever committed, treat it as compromised and rotate it at the provider.
- IMPORTANT: every request to SEC EDGAR must send a real User-Agent header (name and email), or the SEC returns 403.
- When extracting text from a filing, strip hidden (`display:none`) elements first. Modern filings hide a block of machine-readable XBRL data there that pollutes the text otherwise.
- Keep code readable and commented; this is a learning project. Explain trade-offs in comments.
- Make minimal, focused changes. Do not refactor unrelated code without asking first.
- Build one vertical slice at a time: get one company and one question working end to end before adding a UI or more companies.

## Build plan (phases)

0. Setup and key check. Done.
1. Ingest and chunk a 10-K, no AI. Done.
2. Embed chunks, store in Chroma, semantic search, no LLM yet. Next.
3. Add Claude to produce cited answers, giving end-to-end RAG in a script.
4. Wrap the logic in a FastAPI backend.
5. Build a React frontend.
6. Evals: question and expected-fact pairs; measure retrieval hit-rate and answer faithfulness.
7. Deploy (Vercel and Render) and write the README.

Later, once the loop is solid: multiple companies, structured risk-factor extraction, a re-ranking step, and caching.