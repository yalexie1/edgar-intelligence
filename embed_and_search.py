"""
Phase 2: embed the 10-K chunks, store them in a local vector database, and
search them by meaning.

What this does:
  1. Loads the chunks produced by ingest.py (data/chunks.json).
  2. Embeds each chunk with OpenAI's text-embedding-3-small.
  3. Stores the vectors in a local Chroma database that persists to disk.
  4. Lets you type a question and returns the chunks closest to it in meaning.

No LLM yet. This is pure semantic search: the embedding-space idea made real.

Building the index costs a fraction of a cent and happens once. After that,
searches are instant and run for free against the local store.

Run:  python embed_and_search.py
      python embed_and_search.py --rebuild    (force re-embedding from scratch)
"""

import json
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENAI_API_KEY from .env

# --- Settings ----------------------------------------------------------------

CHUNKS_PATH = os.path.join("data", "chunks.json")
CHROMA_DIR = os.path.join("data", "chroma")     # the vector DB lives here, on disk
COLLECTION = "filings"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100                                # chunks sent per embedding API call
TOP_K = 4                                       # how many chunks each search returns


# --- Embeddings --------------------------------------------------------------

def embed_texts(texts):
    """Turn a list of strings into a list of embedding vectors, in batches."""
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # sort by .index so the vectors line up with the input order, just in case
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return vectors


# --- Vector store ------------------------------------------------------------

def build_index(chunks, rebuild=False):
    """Create or reuse the Chroma collection holding the chunk vectors."""
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

    # If the store already holds exactly these chunks, don't pay to embed again.
    if collection.count() == len(chunks):
        print(f"Index already built ({collection.count()} vectors). Skipping embedding.")
        return collection

    # Otherwise rebuild from scratch so the store always matches chunks.json.
    if collection.count() != 0:
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} (a fraction of a cent)...")
    embeddings = embed_texts(chunks)

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"index": i} for i in range(len(chunks))],
    )
    print(f"Stored {collection.count()} vectors in ./{CHROMA_DIR}/")
    return collection


def search(collection, question, k=TOP_K):
    """Embed the question and return the k chunks closest to it in meaning."""
    q_vector = embed_texts([question])[0]
    res = collection.query(query_embeddings=[q_vector], n_results=k)

    # Chroma wraps each field one level deep (one row per query); we sent one query.
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]

    results = []
    for doc, dist, meta in zip(docs, dists, metas):
        results.append({
            "index": meta["index"],
            "similarity": 1 - dist,     # cosine distance -> similarity (1.0 = identical)
            "text": doc,
        })
    return results


# --- Run ---------------------------------------------------------------------

def main():
    if not os.path.exists(CHUNKS_PATH):
        sys.exit(f"Could not find {CHUNKS_PATH}. Run `python ingest.py` first.")

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    collection = build_index(chunks, rebuild="--rebuild" in sys.argv)

    print("\nAsk a question about the filing. Type 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        for r in search(collection, question):
            print(f"\n[chunk {r['index']}   similarity {r['similarity']:.3f}]")
            print(r["text"][:400].strip() + " ...")


if __name__ == "__main__":
    main()