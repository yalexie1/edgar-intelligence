"""
Phase 4: serve the RAG pipeline over HTTP with FastAPI.

This wraps the same ask() logic from Phase 3 in a web server, so a question comes
in as an HTTP request and the cited answer comes back as JSON. That is what lets
a webpage (Phase 5) or any other program use the system, not just you at a terminal.

Run:  uvicorn api:app --reload --port 8000
Then: open http://127.0.0.1:8000/docs for an interactive page to test it.
"""

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


@app.get("/health")
def health():
    """Quick check that the server is up and the index is loaded."""
    if collection is None:
        return {"status": "no index", "chunks": 0}
    return {"status": "ok", "chunks": collection.count()}


@app.post("/query")
def query(q: Query):
    """Answer one question from the filing, with sources."""
    if collection is None:
        raise HTTPException(503, "Index not built. Run `python embed_and_search.py` first.")
    try:
        answer, results = ask(collection, q.question, q.k)
    except Exception as e:
        raise HTTPException(500, f"Failed to answer: {e}")

    sources = [
        {"n": i + 1, "chunk": r["index"], "similarity": round(r["similarity"], 3)}
        for i, r in enumerate(results)
    ]
    return {"question": q.question, "answer": answer, "sources": sources}