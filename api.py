"""
Phase 4: serve the RAG pipeline over HTTP with FastAPI.

This wraps the same ask() logic from Phase 3 in a web server, so a question comes
in as an HTTP request and the cited answer comes back as JSON. That is what lets
a webpage (Phase 5) or any other program use the system, not just you at a terminal.

Run:  uvicorn api:app --reload --port 8000
Then: open http://127.0.0.1:8000/docs for an interactive page to test it.
"""

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ask import ask
from embed_and_search import CHROMA_DIR, COLLECTION, TOP_K

load_dotenv()

app = FastAPI(title="EDGAR Intelligence API")

# Let a browser frontend on another port (Phase 5) call this API.
# "*" is fine for local development; restrict it to your real domain before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def connect():
    """Connect to the Phase 2 vector store, or return None if it isn't built yet."""
    try:
        return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    except Exception:
        return None


collection = connect()


class Query(BaseModel):
    question: str
    k: int = TOP_K
    ticker: str = ""
    form: str = ""
    diverse: bool = False


@app.get("/health")
def health():
    """Quick check that the server is up and the index is loaded."""
    if collection is None:
        return {"status": "no index", "chunks": 0}
    return {"status": "ok", "chunks": collection.count()}


@app.post("/query")
def query(q: Query):
    """Answer one question from the corpus, with cited sources."""
    if collection is None:
        raise HTTPException(503, "Index not built. Run `python embed_and_search.py` first.")

    # Build an optional metadata filter from the request fields.
    where = None
    filters = []
    if q.ticker:
        filters.append({"ticker": q.ticker.upper()})
    if q.form:
        filters.append({"form": q.form.upper()})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    try:
        answer, results, effective_where = ask(collection, q.question, where=where, k=q.k, diverse=q.diverse)
    except Exception as e:
        raise HTTPException(500, f"Failed to answer: {e}")

    sources = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        sources.append({
            "n": i + 1,
            "ticker": meta.get("ticker", ""),
            "form": meta.get("form", ""),
            "period": meta.get("period") or meta.get("filing_date", ""),
            "section": meta.get("section", ""),
            "source_url": meta.get("source_url", ""),
            "similarity": round(r["similarity"], 3),
            "rerank_score": round(r.get("rerank_score", r["similarity"]), 3),
        })
    return {
        "question": q.question,
        "answer": answer,
        "sources": sources,
        "filter_applied": effective_where,
    }


@app.get("/evals/results")
def evals_results():
    """Return the last saved eval run results for the dashboard."""
    path = Path("evals/last_results.json")
    if not path.exists():
        raise HTTPException(404, "No eval results found. Run `python evals/eval.py` first.")
    return json.loads(path.read_text())