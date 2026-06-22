"""
Phase 1: fetch one company's latest 10-K from SEC EDGAR, clean it, and chunk it.

No AI yet. The goal is a list of clean text chunks, saved to disk and previewed
on screen, so you can confirm the text came through readable before we embed it.

Run:  python ingest.py
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

# --- Settings you can change -------------------------------------------------

# SEC REQUIRES a User-Agent that identifies you. Put your real name and email.
# Without this, EDGAR rejects the request with a 403 error.
USER_AGENT = "Yale Xie yale.xie@yale.edu"

CIK = "0000320193"        # Apple, zero-padded to 10 digits
TARGET_FORM = "10-K"

CHUNK_SIZE = 3000         # characters per chunk (a rough first pass; we tune this later)
CHUNK_OVERLAP = 300       # characters repeated between neighboring chunks

DATA_DIR = "data"

# --- EDGAR fetching ----------------------------------------------------------

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def get_latest_filing(cik, form):
    """Ask EDGAR's JSON API for the newest filing of a given form type.

    Returns (accession_number, primary_document, filing_date).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    # "recent" is a set of parallel lists: index i describes one filing across
    # every field. They come newest-first, so the first 10-K we hit is the latest.
    for i, this_form in enumerate(recent["form"]):
        if this_form == form:
            return (
                recent["accessionNumber"][i],
                recent["primaryDocument"][i],
                recent["filingDate"][i],
            )
    raise RuntimeError(f"No {form} found in recent filings for CIK {cik}")


def download_filing_html(cik, accession_no, primary_document):
    """Fetch the actual filing document (an HTML / inline-XBRL file)."""
    cik_int = str(int(cik))                       # the path uses CIK with no leading zeros
    acc_nodash = accession_no.replace("-", "")    # and the accession with no dashes
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodash}/{primary_document}"
    )
    time.sleep(0.3)                               # be polite: SEC asks for under 10 requests/sec
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text, url


# --- Cleaning and chunking ---------------------------------------------------

def html_to_text(html):
    """Strip tags and collapse whitespace into readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Modern filings hide a big block of machine-readable XBRL data with
    # display:none. It's not meant for humans, so drop it before extracting text.
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)              # collapse runs of whitespace into single spaces
    return text.strip()


def chunk_text(text, size, overlap):
    """Split text into overlapping fixed-size windows of characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap                   # step forward, leaving an overlap behind
    return chunks


# --- Run ---------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Looking up the latest {TARGET_FORM} for CIK {CIK}...")
    accession_no, primary_document, filing_date = get_latest_filing(CIK, TARGET_FORM)
    print(f"  found one filed {filing_date}: {primary_document}")

    print("Downloading the filing...")
    html, url = download_filing_html(CIK, accession_no, primary_document)
    print(f"  {len(html):,} characters of HTML")

    print("Cleaning to plain text...")
    text = html_to_text(html)
    print(f"  {len(text):,} characters of clean text")

    print(f"Chunking into ~{CHUNK_SIZE}-character pieces...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks")

    # Save the raw filing and the chunks for the next phase and your own inspection.
    with open(os.path.join(DATA_DIR, "raw_10k.htm"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"\nSaved raw_10k.htm and chunks.json into ./{DATA_DIR}/")

    # Preview the first few chunks so you can eyeball the text quality.
    print("\n----- preview of the first 3 chunks -----")
    for i, c in enumerate(chunks[:3]):
        print(f"\n[chunk {i}] ({len(c)} chars)")
        print(c[:500].strip() + " ...")


if __name__ == "__main__":
    main()