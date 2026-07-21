"""
FastAPI backend for EDGAR Intelligence.

Endpoints:
  POST /query          — answer a question from the corpus (rate-limited)
  GET  /health         — instant liveness check
  GET  /evals/results  — last eval run results (for the dashboard)
"""

import datetime
import json
import logging
import math
import os
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ask import ask, ask_stream, track_themes, ANSWER_MODEL as _DEFAULT_MODEL
from embed_and_search import TOP_K, get_pinecone_index

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


# ── observability (v4 #1: request tracing & latency) ───────────────────────────
# One structured JSON line per request to stdout (Render captures stdout as logs).
# Deliberately logs the full question text — SEC filing Q&A is low-sensitivity
# content (same framing as CLAUDE.md's existing privacy posture), and full text
# is far more useful than a truncated/omitted one for debugging routing/retrieval.
logger = logging.getLogger("edgar_rag")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Rolling window of recent per-request timings, for the /metrics endpoint.
# In-memory only — resets on restart, same tradeoff as the LRU cache above.
_METRICS_WINDOW_SIZE = 200
_metrics_window: deque = deque(maxlen=_METRICS_WINDOW_SIZE)


def _build_log_payload(request_id, endpoint, route, timing, cache_hit, num_results, question, total_s):
    """Assemble one structured, JSON-loggable line for a completed request.

    `timing` is the per-stage dict `ask()`/`ask_stream()` populate (keys:
    retrieval_s, rerank_s, generation_s) — a stage key is simply absent when
    that stage didn't run (e.g. diverse mode skips rerank; an abstain skips
    generation entirely), rather than being recorded as zero.
    """
    payload = {
        "request_id": request_id,
        "endpoint": endpoint,
        "route": route,
        "cache_hit": cache_hit,
        "num_results": num_results,
        "question": question,
        "total_s": round(total_s, 4),
    }
    for stage in ("retrieval_s", "rerank_s", "generation_s"):
        if stage in timing:
            payload[stage] = round(timing[stage], 4)
    return payload


def _percentile(sorted_values, pct):
    """Linear-interpolation percentile over an already-sorted list of floats.

    `pct` is a fraction in [0, 1] (0.95 = p95). Returns None for an empty list.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _metrics_summary(window):
    """Aggregate recent per-request timing entries into p50/p95/p99 per stage.

    `window` is an iterable of dicts shaped like `timing` (plus a `total_s`
    key the request handlers add). A stage absent from an entry (e.g.
    rerank_s when diverse/temporal/section_diff skipped it) is excluded from
    that stage's percentiles rather than counted as zero.
    """
    entries = list(window)
    if not entries:
        return {"count": 0}

    summary: dict = {"count": len(entries)}
    for stage in ("total_s", "retrieval_s", "rerank_s", "generation_s"):
        values = sorted(e[stage] for e in entries if stage in e)
        if not values:
            continue
        summary[stage] = {
            "p50": round(_percentile(values, 0.50), 4),
            "p95": round(_percentile(values, 0.95), 4),
            "p99": round(_percentile(values, 0.99), 4),
            "count": len(values),
        }

    route_counts: dict = {}
    for e in entries:
        r = e.get("route", "unknown")
        route_counts[r] = route_counts.get(r, 0) + 1
    summary["routes"] = route_counts
    return summary


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
    """Connect to the Pinecone index, or return None if the key is missing."""
    try:
        return get_pinecone_index()
    except Exception:
        return None


collection = connect()


# ── request schema ─────────────────────────────────────────────────────────────
class Query(BaseModel):
    question: str
    k: int = TOP_K
    ticker: str = ""
    form: str = ""
    section: str = ""
    diverse: bool = False
    history: list = []  # prior turns: [{"question": str, "answer": str}, ...]


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Instant liveness check — does not touch the index so cold /health is fast."""
    if collection is None:
        return {"status": "no index", "chunks": 0}
    try:
        chunks = collection.describe_index_stats().total_vector_count
    except Exception:
        chunks = -1
    return {"status": "ok", "chunks": chunks}


def _is_localhost(request: Request) -> bool:
    return (request.client and request.client.host in ("127.0.0.1", "::1"))


def _build_sources(results: list) -> list:
    """Shape retrieval results into the citation list shared by /query and /query/stream."""
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
            "text": r["text"][:500],
        })
    return sources


def _build_where(q: Query):
    """Build an optional metadata filter dict from ticker/form/section request fields."""
    filters = []
    if q.ticker:
        filters.append({"ticker": q.ticker.upper()})
    if q.form:
        filters.append({"form": q.form.upper()})
    if q.section:
        filters.append({"section": q.section.lower()})
    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}
    return None


def _validate_query(q: Query) -> None:
    """Shared request validation for /query and /query/stream."""
    if not q.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if len(q.question) > MAX_QUESTION_LEN:
        raise HTTPException(
            400,
            f"Question too long ({len(q.question)} chars). Please keep it under {MAX_QUESTION_LEN} characters.",
        )
    if collection is None:
        raise HTTPException(503, "Index not built. Run `python embed_and_search.py` first.")


@app.post("/query")
@limiter.limit("10/minute", exempt_when=_is_localhost)
@limiter.limit("200/day",   exempt_when=_is_localhost)
def query(request: Request, q: Query):
    """Answer one question from the corpus, with cited sources."""
    _validate_query(q)  # fast — no LLM cost

    # Global daily cap — checked here so only requests that reach the LLM are counted.
    _check_global_cap()

    where = _build_where(q)
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    timing: dict = {}

    try:
        # Only cache when there's no conversation history — history makes each
        # request contextually unique and caching it would return stale context.
        cache_key = _cache_key(q.question, where, q.diverse) if not q.history else None
        cached = _cache_get(cache_key) if cache_key else None
        cache_hit = cached is not None
        if cached:
            answer, results, effective_where = cached
        else:
            answer, results, effective_where = ask(
                collection, q.question, where=where, k=q.k, diverse=q.diverse,
                history=q.history, model=ANSWER_MODEL, timing=timing,
            )
            if cache_key:
                _cache_set(cache_key, (answer, results, effective_where))
    except Exception as e:
        raise HTTPException(500, f"Failed to answer: {e}")

    total_s = time.perf_counter() - start
    # A cache hit never calls ask(), so timing stays empty — label it "cache"
    # rather than "unknown" for a route that never ran routing at all.
    route = timing.get("route", "cache")
    logger.info(json.dumps(_build_log_payload(
        request_id, "/query", route, timing, cache_hit, len(results), q.question, total_s,
    )))
    # timing["route"] is only set when ask() actually ran _route() — a cache
    # hit skips ask() entirely, so timing stays empty. Set "route" explicitly
    # here (from the same resolved `route` used for logging above) so a cache
    # hit shows up as "cache" in /metrics instead of "unknown".
    _metrics_window.append({**timing, "route": route, "total_s": total_s})

    # Success shape is frozen — do not change field names or remove fields.
    # request_id is additive (same pattern as v3 §3's "text" field on sources).
    return {
        "question": q.question,
        "answer": answer,
        "sources": _build_sources(results),
        "filter_applied": effective_where,
        "request_id": request_id,
    }


@app.post("/query/stream")
@limiter.limit("10/minute", exempt_when=_is_localhost)
@limiter.limit("200/day",   exempt_when=_is_localhost)
def query_stream(request: Request, q: Query):
    """Answer one question, streaming answer tokens as Server-Sent Events.

    Same validation and request contract as /query. Emits `data: {...}` lines:
      - {"token": "..."}                                — one per answer token, in order
      - {"sources": [...], "filter_applied": ...}        — once, after all tokens
      - [DONE]                                            — terminal marker (always last)

    Not cached — conversation-shaped requests already skip the cache in /query, and a
    streamed response can't be replayed from a cache key without re-simulating token
    arrival, which isn't worth it for a demo server.
    """
    _validate_query(q)
    _check_global_cap()
    where = _build_where(q)
    request_id = str(uuid.uuid4())

    def event_stream():
        start = time.perf_counter()
        timing: dict = {}
        results, effective_where = [], where
        try:
            for chunk in ask_stream(
                collection, q.question, where=where, k=q.k, diverse=q.diverse,
                history=q.history, model=ANSWER_MODEL, timing=timing,
            ):
                if isinstance(chunk, dict) and chunk.get("__meta__"):
                    results = chunk["results"]
                    effective_where = chunk["effective_where"]
                    payload = {
                        "sources": _build_sources(results),
                        "filter_applied": effective_where,
                        "request_id": request_id,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        total_s = time.perf_counter() - start
        logger.info(json.dumps(_build_log_payload(
            request_id, "/query/stream", timing.get("route", "unknown"),
            timing, False, len(results), q.question, total_s,
        )))
        _metrics_window.append({**timing, "total_s": total_s})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@app.get("/metrics")
def metrics():
    """Rolling-window p50/p95/p99 latency breakdown per pipeline stage (v4 #1).

    In-memory only — covers the last _METRICS_WINDOW_SIZE (200) requests since
    the process last restarted (same tradeoff as the LRU cache). No external
    dependency, no new service to run.
    """
    return _metrics_summary(_metrics_window)


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
