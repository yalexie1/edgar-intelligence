"""
Phase 2: embed the multi-company SEC corpus, store it in a local vector database,
and search it by meaning with metadata filters.

What this does:
  1. Loads JSONL chunk records produced by ingest.py (data/corpus.jsonl).
  2. Embeds each record's text with OpenAI's text-embedding-3-small.
  3. Stores vectors, source text, and metadata in a persistent local Chroma DB.
  4. Lets you search across all companies, or filter by ticker/form/section/item.

No LLM yet. This is pure semantic search: the retrieval layer of RAG.

Run:  python embed_and_search.py
      python embed_and_search.py --rebuild
      python embed_and_search.py --ticker AAPL
      python embed_and_search.py --ticker NVDA --form 10-Q
      python embed_and_search.py --ticker MSFT --section mda
"""

import argparse
import hashlib
import json
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()  # loads OPENAI_API_KEY from .env

# --- Settings ----------------------------------------------------------------

CORPUS_PATH = os.path.join("data", "corpus.jsonl")
CHROMA_DIR = os.path.join("data", "chroma")     # the vector DB lives here, on disk
COLLECTION = "sec_filings"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 40                                 # chunks sent per embedding API call
ADD_BATCH_SIZE = 1000                           # records added to Chroma per batch
TOP_K = 5                                       # how many chunks each search returns
CANDIDATE_K = 50                                # retrieve more, then rerank locally

KEYWORD_BOOST = 0.08                            # exact-term boost for reranking
DIVERSE_CANDIDATE_MULTIPLIER = 8                # pull more candidates for diversity mode


# --- Corpus loading and metadata --------------------------------------------

def canonical_section(item_title):
    """Map messy SEC item titles into stable section labels for filtering."""
    title = (item_title or "").lower()
    title = title.replace("’", "'")
    title = " ".join(title.split())

    if "management" in title and "discussion" in title:
        return "mda"
    if "risk factor" in title:
        return "risk_factors"
    if "quantitative" in title and "market risk" in title:
        return "market_risk"
    if "financial statement" in title and "exhibit" in title:
        return "exhibits"
    if "financial statement" in title:
        return "financial_statements"
    if "unregistered sales" in title:
        return "unregistered_sales"
    if "legal proceedings" in title:
        return "legal_proceedings"
    if "controls and procedures" in title:
        return "controls"
    if "results of operations" in title:
        return "results_of_operations"
    if "material definitive agreement" in title:
        return "material_agreement"
    return "other"


def chroma_safe_metadata(rec):
    """Return Chroma-compatible metadata with no None values."""
    fields = [
        "ticker", "company", "cik", "form", "filing_date", "report_date",
        "period", "accession", "source_url", "primary_document", "part",
        "item", "part_item", "item_title", "chunk_index", "section_chunk_index",
    ]
    meta = {}
    for key in fields:
        val = rec.get(key)
        if val is None:
            val = ""
        meta[key] = val
    meta["section"] = rec.get("section") or canonical_section(rec.get("item_title"))
    return meta


def stable_id(rec):
    """Create a stable Chroma ID from filing/chunk identity."""
    existing = rec.get("id")
    if existing:
        return existing
    raw = "|".join([
        rec.get("ticker", ""),
        rec.get("form", ""),
        rec.get("accession", ""),
        str(rec.get("chunk_index", "")),
        rec.get("text", "")[:80],
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def unique_chroma_ids(records):
    """Return stable Chroma IDs that are guaranteed unique within this corpus.

    Some filings can produce repeated source IDs when the same accession/item/chunk
    pattern appears more than once. Chroma refuses duplicate IDs, so duplicates get
    a deterministic suffix before any embedding work starts.
    """
    seen = {}
    ids = []
    duplicate_count = 0

    for rec in records:
        base = stable_id(rec)
        count = seen.get(base, 0)
        seen[base] = count + 1

        if count == 0:
            ids.append(base)
        else:
            duplicate_count += 1
            ids.append(f"{base}-dup{count}")

    if duplicate_count:
        print(f"Resolved {duplicate_count} duplicate Chroma IDs with deterministic suffixes.")

    return ids


def load_corpus(path=CORPUS_PATH):
    """Load JSONL corpus records from ingest.py."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            rec["text"] = text
            records.append(rec)
    return records


# --- Embeddings --------------------------------------------------------------

def embed_texts(texts):
    """Turn a list of strings into a list of embedding vectors, with rate-limit retries."""
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        batch_num = start // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for attempt in range(8):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                # sort by .index so the vectors line up with the input order, just in case
                for item in sorted(resp.data, key=lambda d: d.index):
                    vectors.append(item.embedding)
                print(f"  embedded batch {batch_num}/{total_batches}")
                break
            except (RateLimitError, APITimeoutError, APIError) as e:
                if attempt == 7:
                    raise
                sleep_seconds = min(2 ** attempt, 30)
                print(
                    f"  embedding batch {batch_num}/{total_batches} hit {type(e).__name__}; "
                    f"sleeping {sleep_seconds}s then retrying"
                )
                time.sleep(sleep_seconds)

    return vectors


# --- Vector store ------------------------------------------------------------

def build_index(records, rebuild=False):
    """Create or reuse the Chroma collection holding SEC chunk vectors + metadata."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    # "cosine" makes distance line up with the similarity idea: closer = more alike.
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    # If the store already holds exactly these records, don't pay to embed again.
    if collection.count() == len(records):
        print(f"Index already built ({collection.count()} vectors). Skipping embedding.")
        return collection

    # Otherwise rebuild from scratch so the store always matches corpus.jsonl.
    if collection.count() != 0:
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    texts = [rec["text"] for rec in records]
    ids = unique_chroma_ids(records)
    metadatas = [chroma_safe_metadata(rec) for rec in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Internal error: Chroma IDs are still not unique after de-duplication.")

    print(f"Embedding and storing {len(records)} chunks with {EMBED_MODEL}...")

    # Embed and add incrementally. This avoids doing all API work before finding
    # Chroma insert problems, and it keeps memory usage lower for larger corpora.
    for start in range(0, len(records), ADD_BATCH_SIZE):
        end = min(start + ADD_BATCH_SIZE, len(records))
        batch_texts = texts[start:end]
        batch_embeddings = embed_texts(batch_texts)

        if len(batch_embeddings) != len(batch_texts):
            raise ValueError(
                f"Embedding count mismatch for records {start}:{end}: "
                f"got {len(batch_embeddings)} embeddings for {len(batch_texts)} texts."
            )

        collection.add(
            ids=ids[start:end],
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=metadatas[start:end],
        )
        print(f"  added {end}/{len(records)}")

    print(f"Stored {collection.count()} vectors in ./{CHROMA_DIR}/")
    return collection


def build_where(args):
    """Build a Chroma metadata filter from CLI args."""
    filters = []
    if args.ticker:
        filters.append({"ticker": args.ticker.upper()})
    if args.form:
        filters.append({"form": args.form.upper()})
    if args.section:
        filters.append({"section": args.section.lower()})
    if args.item:
        filters.append({"item": args.item.upper()})
    if args.period:
        filters.append({"period": args.period})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


# --- Query expansion and lexical helpers ------------------------------------


def expanded_query(question):
    """Expand short finance/tech abbreviations before embedding the query."""
    q = question.strip()
    lowered = q.lower()

    ai_markers = {" ai ", "ai-", " ai?", " ai.", " ai,"}
    padded = f" {lowered} "
    if any(marker in padded for marker in ai_markers) or "artificial intelligence" in lowered:
        q += " artificial intelligence AI machine learning generative AI AI-related products services regulation adoption misuse"

    return q



def lexical_score(text, metadata, question):
    """Exact-term score used to lift answer-bearing chunks above boilerplate."""
    terms = query_terms(question)
    if not terms:
        return 0.0

    haystack = " ".join([
        text or "",
        metadata.get("item_title", "") or "",
        metadata.get("section", "") or "",
    ]).lower()

    hits = 0.0
    possible = 0.0
    for term in terms:
        # Multi-word terms are more diagnostic than single generic words.
        weight = 2.0 if " " in term else 1.0
        possible += weight
        if term in haystack:
            hits += weight

    return hits / possible if possible else 0.0


def search(collection, question, where=None, k=TOP_K):
    """Embed the question, retrieve broad candidates, then rerank locally.

    Pure vector search can over-rank repeated SEC boilerplate, especially in MD&A
    sections where many filings start with the same forward-looking disclaimer.
    We retrieve more candidates from Chroma and apply a light exact-term boost so
    chunks containing terms like "seasonality" beat generic boilerplate chunks.
    """
    q_vector = embed_texts([expanded_query(question)])[0]
    n_results = max(k, CANDIDATE_K)
    kwargs = {"query_embeddings": [q_vector], "n_results": n_results}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)

    # Chroma wraps each field one level deep (one row per query); we sent one query.
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    ids = res["ids"][0]

    results = []
    for doc_id, doc, dist, meta in zip(ids, docs, dists, metas):
        similarity = 1 - dist     # cosine distance -> similarity (1.0 = identical)
        lex = lexical_score(doc, meta, question)
        results.append({
            "id": doc_id,
            "similarity": similarity,
            "lexical_score": lex,
            "rerank_score": similarity + KEYWORD_BOOST * lex,
            "metadata": meta,
            "text": doc,
        })

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:k]


def diversify_results(results, k=TOP_K, by="ticker"):
    """Prefer variety across tickers or filings while preserving rank quality.

    This is useful for cross-company questions like "Which companies discuss AI
    risks?" where the top vector results may contain several near-duplicate
    quarters from the same company.
    """
    if not results:
        return []

    if by == "filing":
        key_fn = lambda r: (
            r["metadata"].get("ticker", ""),
            r["metadata"].get("accession", ""),
        )
    else:
        key_fn = lambda r: r["metadata"].get("ticker", "")

    selected = []
    used = set()

    # First pass: take the best result from each group.
    for result in results:
        key = key_fn(result)
        if key in used:
            continue
        selected.append(result)
        used.add(key)
        if len(selected) >= k:
            return selected

    # Second pass: fill any remaining slots with the next best results.
    selected_ids = {r["id"] for r in selected}
    for result in results:
        if result["id"] in selected_ids:
            continue
        selected.append(result)
        if len(selected) >= k:
            break

    return selected


def query_terms(question):
    """Extract useful content terms from a question for result previews/reranking."""
    stopwords = {
        "what", "does", "say", "about", "the", "and", "for", "with", "from",
        "that", "this", "are", "was", "were", "how", "why", "when", "where",
        "company", "companies", "business", "apple", "nvidia", "microsoft",
        "meta", "amazon", "google", "salesforce", "tesla", "amd", "intel",
        "oracle", "netflix", "disclose", "discloses", "disclosed", "related",
        "risk", "risks",
    }
    normalized = question.lower().replace("ai-related", "ai related")
    raw_terms = []
    for term in normalized.replace("?", " ").split():
        term = term.strip(".,;:()[]{}\"'")
        if len(term) >= 4 and term not in stopwords:
            raw_terms.append(term)
        elif term == "ai":
            raw_terms.append("ai")

    terms = set(raw_terms)

    # Domain-specific expansion. SEC filings often spell out "artificial
    # intelligence" instead of using the abbreviation "AI".
    if "ai" in terms or "artificial intelligence" in normalized:
        terms.update({
            "ai",
            "artificial intelligence",
            "machine learning",
            "generative ai",
            "ai-related",
            "misuse",
            "adoption",
            "regulation",
            "responsible use",
        })

    # Prefer multi-word and specific terms first.
    return sorted(terms, key=lambda t: (-len(t), t))


def best_snippet(text, question, window=1200):
    """Return the part of a retrieved chunk most relevant to the question.

    The vector search may retrieve the right chunk, but the answer-bearing phrase
    can appear after boilerplate at the beginning. This preview centers the output
    around a specific query term when possible.
    """
    if len(text) <= window:
        return text.strip()

    lowered = text.lower()
    terms = query_terms(question)

    center = None
    matched_term = None
    for term in terms:
        pos = lowered.find(term)
        if pos >= 0:
            center = pos
            matched_term = term
            break

    if center is None:
        start = 0
    else:
        start = max(0, center - window // 3)

    end = min(len(text), start + window)
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet = snippet + " ..."
    if matched_term:
        snippet = f"[preview centered on: {matched_term}]\n" + snippet
    return snippet


# --- Run ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed and search the SEC corpus.")
    parser.add_argument("--rebuild", action="store_true", help="force re-embedding from scratch")
    parser.add_argument("--ticker", help="filter search by ticker, e.g. AAPL")
    parser.add_argument("--form", help="filter search by form, e.g. 10-K, 10-Q, 8-K")
    parser.add_argument("--section", help="filter by canonical section, e.g. mda, risk_factors")
    parser.add_argument("--item", help="filter by SEC item, e.g. 1A, 7, 2.02")
    parser.add_argument("--period", help="filter by period, e.g. FY2025 or 2026-03")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of search results")
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="prefer one strong result per ticker for cross-company questions",
    )
    parser.add_argument(
        "--diverse-by",
        choices=["ticker", "filing"],
        default="ticker",
        help="diversity grouping: ticker for cross-company, filing for one-company repeated filings",
    )
    args = parser.parse_args()

    if not os.path.exists(CORPUS_PATH):
        sys.exit(f"Could not find {CORPUS_PATH}. Run `python ingest.py` first.")

    records = load_corpus(CORPUS_PATH)
    collection = build_index(records, rebuild=args.rebuild)
    where = build_where(args)

    print("\nSearch filters:", where or "none")
    if args.diverse:
        print(f"Diversity mode: on, grouped by {args.diverse_by}")
    print("Ask a question about the corpus. Type 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        raw_k = args.k * DIVERSE_CANDIDATE_MULTIPLIER if args.diverse else args.k
        results = search(collection, question, where=where, k=raw_k)
        if args.diverse:
            results = diversify_results(results, k=args.k, by=args.diverse_by)

        if not results:
            print("\nNo results matched the current filters.")
            print("Try removing one filter, for example: python embed_and_search.py --ticker AMD")
            continue

        for r in results:
            m = r["metadata"]
            source = (
                f"{m.get('ticker')} {m.get('form')} {m.get('filing_date')} "
                f"section={m.get('section')} item={m.get('part_item') or m.get('item')}"
            )
            print(
                f"\n[sim {r['similarity']:.3f} | lex {r.get('lexical_score', 0):.2f} | "
                f"rank {r.get('rerank_score', r['similarity']):.3f}] {source}"
            )
            print(f"{m.get('item_title')}")
            print(best_snippet(r["text"], question))
            print(m.get("source_url"))


if __name__ == "__main__":
    main()