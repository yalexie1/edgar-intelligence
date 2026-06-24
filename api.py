"""
FastAPI backend for EDGAR Intelligence.

Endpoints:
  POST /query          — answer a question from the corpus (rate-limited)
  GET  /health         — instant liveness check
  GET  /evals/results  — last eval run results (for the dashboard)
"""

import datetime
import json
import os
from collections import OrderedDict
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ask import ask, track_themes, ANSWER_MODEL as _DEFAULT_MODEL
from embed_and_search import CHROMA_DIR, COLLECTION, TOP_K

load_dotenv()

# ANSWER_MODEL env var lets each deployment choose its model without a code change.
# Set to "claude-sonnet-4-6" on Render; leave unset for Haiku (dev default).
ANSWER_MODEL = os.getenv("ANSWER_MODEL", _DEFAULT_MODEL)

# ── result cache ───────────────────────────────────────────────────────────────
# Caches identical (question, where, diverse) tuples to avoid redundant LLM calls.
# In-memory only — resets on restart, which is fine for a demo server. Max 256
# entries; oldest evicted first (insertion-order OrderedDict).
_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 256


def _cache_key(question: str, where, diverse: bool) -> str:
    return json.dumps({"q": question, "w": where, "d": diverse}, sort_keys=True)


def _cache_get(key: str):
    if key in _CACHE:
        _CACHE.move_to_end(key)  # mark as recently used
        return _CACHE[key]
    return None


def _cache_set(key: str, value) -> None:
    if key in _CACHE:
        _CACHE.move_to_end(key)
    _CACHE[key] = value
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)  # evict oldest


# ── rate limiting (per-IP) ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Hardcoded defaults cover the deployed Vercel frontend and common local dev ports.
# Set ALLOWED_ORIGINS in the environment to add more origins (comma-separated).
_default_origins = [
    "https://edgar-intelligence.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:8080",  # python -m http.server 8080
]
_extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS_ORIGINS = list(set(_default_origins + _extra_origins))

# ── global daily cap ───────────────────────────────────────────────────────────
# Backstop against sustained abuse across many IPs. In-memory is fine for a
# single Render instance; resets naturally on every deploy or restart.
DAILY_CAP = 2000
MAX_QUESTION_LEN = 500
_daily: dict = {"date": None, "count": 0}


def _check_global_cap() -> None:
    today = datetime.date.today()
    if _daily["date"] != today:
        _daily["date"] = today
        _daily["count"] = 0
    _daily["count"] += 1
    if _daily["count"] > DAILY_CAP:
        raise HTTPException(
            503,
            f"Global daily limit of {DAILY_CAP} requests reached. Try again tomorrow.",
        )


# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EDGAR Intelligence API")
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a moment before trying again."},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── vector store ───────────────────────────────────────────────────────────────
def connect():
    """Connect to the Chroma vector store, or return None if it isn't built yet."""
    try:
        return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    except Exception:
        return None


collection = connect()


# ── request schema ─────────────────────────────────────────────────────────────
class Query(BaseModel):
    question: str
    k: int = TOP_K
    ticker: str = ""
    form: str = ""
    diverse: bool = False
    history: list = []  # prior turns: [{"question": str, "answer": str}, ...]


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Instant liveness check — does not touch the index so cold /health is fast."""
    if collection is None:
        return {"status": "no index", "chunks": 0}
    return {"status": "ok", "chunks": collection.count()}


def _is_localhost(request: Request) -> bool:
    return (request.client and request.client.host in ("127.0.0.1", "::1"))


@app.post("/query")
@limiter.limit("10/minute", exempt_when=_is_localhost)
@limiter.limit("200/day",   exempt_when=_is_localhost)
def query(request: Request, q: Query):
    """Answer one question from the corpus, with cited sources."""
    # Fast validation — no LLM cost.
    if not q.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if len(q.question) > MAX_QUESTION_LEN:
        raise HTTPException(
            400,
            f"Question too long ({len(q.question)} chars). Please keep it under {MAX_QUESTION_LEN} characters.",
        )
    if collection is None:
        raise HTTPException(503, "Index not built. Run `python embed_and_search.py` first.")

    # Global daily cap — checked here so only requests that reach the LLM are counted.
    _check_global_cap()

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
        # Only cache when there's no conversation history — history makes each
        # request contextually unique and caching it would return stale context.
        cache_key = _cache_key(q.question, where, q.diverse) if not q.history else None
        cached = _cache_get(cache_key) if cache_key else None
        if cached:
            answer, results, effective_where = cached
        else:
            answer, results, effective_where = ask(
                collection, q.question, where=where, k=q.k, diverse=q.diverse,
                history=q.history, model=ANSWER_MODEL,
            )
            if cache_key:
                _cache_set(cache_key, (answer, results, effective_where))
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

    # Success shape is frozen — do not change field names or remove fields.
    return {
        "question": q.question,
        "answer": answer,
        "sources": sources,
        "filter_applied": effective_where,
    }


_VALID_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AVGO", "TSLA", "ORCL", "CRM", "AMD", "NFLX", "INTC",
}


@app.get("/themes")
@limiter.limit("20/minute", exempt_when=_is_localhost)
def themes_endpoint(request: Request, ticker: str, themes: str = ""):
    """Theme tracking: best evidence per predefined topic across filing periods.

    Returns retrieval-only results (no LLM cost) grouped by theme and period.
    Useful for spotting which topics a company emphasises more or less over time.

    Args:
        ticker: company ticker (AAPL, NVDA, etc.)
        themes: optional comma-separated list of theme keys to return (default: all)
    """
    if collection is None:
        raise HTTPException(503, "Index not built. Run `python embed_and_search.py` first.")
    t = ticker.upper()
    if t not in _VALID_TICKERS:
        raise HTTPException(
            400,
            f"Unknown ticker '{t}'. Supported: {sorted(_VALID_TICKERS)}",
        )
    result = track_themes(collection, t)
    if themes:
        requested = {s.strip() for s in themes.split(",") if s.strip()}
        result["themes"] = {k: v for k, v in result["themes"].items() if k in requested}
    return result


@app.get("/evals/results")
def evals_results():
    """Return the last saved eval run results for the dashboard."""
    path = Path("evals/last_results.json")
    if not path.exists():
        raise HTTPException(404, "No eval results found. Run `python evals/eval.py` first.")
    return json.loads(path.read_text())


@app.get("/evals/ragas")
def ragas_results():
    """Return the last saved RAGAS eval summary for the dashboard."""
    path = Path("evals/results/ragas_summary.json")
    if not path.exists():
        raise HTTPException(
            404, "No RAGAS results found. Run `python evals/eval_ragas.py` first."
        )
    return json.loads(path.read_text())
