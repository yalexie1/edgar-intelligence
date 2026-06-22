"""
Phase 3: ask a question and get a grounded, cited answer.

This ties the whole loop together. It reuses search() from Phase 2 to pull the
most relevant chunks, then hands those chunks plus the question to Claude with a
strict instruction: answer ONLY from the passages, and cite which one each fact
came from. The result is the first time the system actually answers you.

Requires the Phase 2 index to exist. If it doesn't, run embed_and_search.py first.

Run:  python ask.py
"""

import sys

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

from embed_and_search import search, CHROMA_DIR, COLLECTION, TOP_K

load_dotenv()  # loads ANTHROPIC_API_KEY (and OPENAI_API_KEY, used by search)

# Haiku is cheap and fast for development. Swap to "claude-sonnet-4-6" for sharper
# answers once the pipeline feels right.
ANSWER_MODEL = "claude-haiku-4-5-20251001"


def get_collection():
    """Connect to the vector store built in Phase 2."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        sys.exit("No vector store found. Run `python embed_and_search.py` first.")


def build_prompt(question, results):
    """Assemble the question and the retrieved passages into one grounded prompt."""
    # Number the passages [1], [2], ... so Claude can cite them.
    passages = []
    for n, r in enumerate(results, start=1):
        passages.append(f"[{n}] (chunk {r['index']})\n{r['text']}")
    context = "\n\n".join(passages)

    return f"""You are a financial analyst assistant answering questions about a company's SEC 10-K filing. Use ONLY the numbered passages below.

Rules:
- Answer only from the passages. If they do not contain the answer, say so plainly instead of guessing.
- After each claim, cite the passage number(s) it came from in square brackets, like [1] or [2].
- Be concise and factual. Quote exact figures when the passages give them.

Passages:
{context}

Question: {question}

Answer:"""


def ask(collection, question, k=TOP_K):
    """Retrieve, prompt Claude, and return (answer_text, retrieved_chunks)."""
    results = search(collection, question, k)
    prompt = build_prompt(question, results)

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = msg.content[0].text
    return answer, results


def main():
    collection = get_collection()
    print("Ask a question about the filing. Type 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break

        answer, results = ask(collection, question)
        print("\n" + answer.strip())

        # Show which chunks fed the answer so every citation is traceable.
        print("\n--- sources ---")
        for n, r in enumerate(results, start=1):
            print(f"[{n}] chunk {r['index']} (similarity {r['similarity']:.3f})")


if __name__ == "__main__":
    main()