"""
Phase 0 setup check for EDGAR Intelligence.

Confirms both API keys actually work before you build anything else.
Run it with:  python verify_setup.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads your keys from a local .env file (never committed to git)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")


def check_keys_present():
    missing = []
    if not OPENAI_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print("MISSING KEYS:", ", ".join(missing))
        print("Copy .env.example to .env, paste your keys in, then run this again.")
        sys.exit(1)
    print("[1/3] Found both keys in .env")


def check_openai_embeddings():
    from openai import OpenAI

    client = OpenAI()  # picks up OPENAI_API_KEY from the environment automatically
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="The company's gross margin declined this quarter.",
    )
    dims = len(resp.data[0].embedding)
    print(f"[2/3] OpenAI embeddings OK   ->  got a vector of length {dims}")


def check_anthropic_claude():
    import anthropic

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY automatically
    # If this model name ever errors, check the current names at docs.claude.com
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    reply = msg.content[0].text.strip()
    print(f"[3/3] Claude OK             ->  it replied: {reply!r}")


if __name__ == "__main__":
    check_keys_present()
    try:
        check_openai_embeddings()
        check_anthropic_claude()
    except Exception as e:
        print("\nA call failed:", repr(e))
        print("Common causes: a wrong or expired key, no billing credit added yet,")
        print("or a model name that has changed since this was written.")
        sys.exit(1)
    print("\nPhase 0 complete. Both providers respond. You're ready for Phase 1.")
