"""
Phase A: build a multi-company, multi-document corpus from SEC EDGAR.

Pulls 10-K, 10-Q, and 8-K filings for a list of companies over the last ~2 years,
cleans each one, chunks it, and tags every chunk with metadata: company, ticker,
form, filing date, period, accession, source URL, and a best-effort section.

Output is one file, data/corpus.jsonl, with one JSON record per chunk. The next
phase embeds this corpus (with its metadata) so search can filter by company,
form, date, or section.

Run:  python ingest.py
"""

import json
import os
import re
import time
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

# --- Settings you can change -------------------------------------------------

# SEC REQUIRES a real User-Agent (name + email). Without it, EDGAR returns 403.
USER_AGENT = "Yale Xie yale.xie@yale.edu"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO",
           "TSLA", "ORCL", "CRM", "AMD", "NFLX", "INTC"]

FORMS = ["10-K", "10-Q", "8-K"]
YEARS_BACK = 2

# Caps per form per company. 8-Ks are frequent and often routine, so we limit
# them to the most recent few to keep the corpus focused and the run fast.
MAX_PER_FORM = {"10-K": 3, "10-Q": 8, "8-K": 8}

# Character-based safety limit after paragraph grouping. This is intentionally
# larger than the old 3,000-char window so each chunk has enough context for RAG.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 600
MIN_PARAGRAPH_CHARS = 40
MIN_CHUNK_CHARS = 250
BLOCK_TAG_NAMES = ["h1", "h2", "h3", "h4", "p", "div", "li", "tr"]
OUT_PATH = os.path.join("data", "corpus.jsonl")

# Fallback CIKs, used only if the live ticker->CIK lookup fails.
FALLBACK_CIK = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    "AMZN": "0001018724", "META": "0001326801", "NVDA": "0001045810",
    "AVGO": "0001730168", "TSLA": "0001318605", "ORCL": "0001341439",
    "CRM": "0001108524", "AMD": "0000002488", "NFLX": "0001065280",
    "INTC": "0000050863",
}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# --- Resolving tickers to CIKs ----------------------------------------------

def load_ticker_map():
    """Fetch the SEC's official ticker->CIK map. Returns {TICKER: cik10} or None."""
    try:
        resp = session.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
        resp.raise_for_status()
        out = {}
        for row in resp.json().values():
            out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
        return out
    except Exception as e:
        print(f"  (ticker map fetch failed: {e}; using fallback CIKs)")
        return None


def resolve_cik(ticker, ticker_map):
    if ticker_map and ticker in ticker_map:
        return ticker_map[ticker]
    return FALLBACK_CIK.get(ticker)


# --- Pulling filings ---------------------------------------------------------

def fetch_submissions(cik):
    """Return (company_name, recent_filings_dict) for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["name"], data["filings"]["recent"]


def pick_filings(recent, since_date):
    """Choose the filings we want: requested forms, recent enough, capped per form."""
    chosen = []
    counts = {form: 0 for form in FORMS}
    n = len(recent["form"])
    for i in range(n):  # recent arrays are newest-first
        form = recent["form"][i]
        if form not in FORMS:
            continue
        filing_date = recent["filingDate"][i]
        if filing_date < since_date:
            continue
        if counts[form] >= MAX_PER_FORM[form]:
            continue
        counts[form] += 1
        chosen.append({
            "form": form,
            "accession": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            "filing_date": filing_date,
            "report_date": recent["reportDate"][i],
        })
    return chosen


def download_filing_html(cik, accession, primary_document):
    cik_int = str(int(cik))
    acc_nodash = accession.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{cik_int}/{acc_nodash}/{primary_document}")
    time.sleep(0.25)  # stay well under SEC's 10 req/sec limit
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text, url


# --- Cleaning, sections, chunking -------------------------------------------

def html_to_text(html):
    """Strip noise while preserving paragraph-like breaks for better chunking."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()

    # Keep block-level structure. RAG chunks are much better when they respect
    # paragraphs/headings instead of flattening the whole filing into one line.
    block_tags = soup.find_all(BLOCK_TAG_NAMES)
    parts = []
    seen = set()
    for tag in block_tags:
        # SEC pages often wrap useful leaf text inside large parent divs. If we
        # keep both parent and child text, the corpus gets duplicated and the
        # natural filing order becomes noisy. Prefer leaf-like blocks.
        if tag.find(BLOCK_TAG_NAMES):
            continue

        txt = tag.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) < MIN_PARAGRAPH_CHARS:
            continue
        # SEC HTML often repeats the same text through nested div/table tags.
        # Dedup exact repeats so the vector index is not polluted.
        if txt in seen:
            continue
        seen.add(txt)
        parts.append(txt)

    if not parts:
        text = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()

    return "\n\n".join(parts)


# Matches standalone SEC Part headings like "Part I" or "Part II Other Information".
PART_RE = re.compile(
    r"(?im)^part\s+(?P<label>[ivx]+)\b\s*(?P<title>[^\n]{0,140})$"
)

# Matches standalone SEC Item headings like "Item 1A. Risk Factors" or
# 8-K headings like "Item 2.02 Results of Operations and Financial Condition".
ITEM_RE = re.compile(
    r"(?im)^item\s+"
    r"(?P<label>\d{1,2}\.\d{2}|\d{1,2}[A-Z]?)"
    r"\s*[.:-]?\s*(?P<title>[^\n]{0,140})$"
)


def normalize_item_label(label):
    """Normalize Item labels so filtering is consistent, e.g. 1a -> 1A."""
    return label.upper().strip()


def normalize_part_label(label):
    """Normalize Part labels so 10-Q Part I and Part II items are distinguishable."""
    if not label:
        return None
    return label.upper().strip()


def clean_item_title(title):
    """Clean a detected item title for metadata display."""
    title = re.sub(r"\s+", " ", title or "").strip(" .:-")
    # Table-of-contents headings often end with a page number, e.g.
    # "Management's Discussion and Analysis 13". Strip that noise.
    title = re.sub(r"\s+\d{1,4}$", "", title).strip(" .:-")
    return title if title else None


def split_into_item_sections(text):
    """Return [(part_label, item_label, item_title, section_text)] using SEC headings.

    10-Q filings reuse item numbers across Part I and Part II. For example,
    Part I Item 2 is MD&A, while Part II Item 2 is Unregistered Sales of Equity
    Securities. Tracking Part separately avoids misleading section metadata.
    """
    part_matches = list(PART_RE.finditer(text))
    item_matches = list(ITEM_RE.finditer(text))
    if not item_matches:
        return [(None, None, None, text)]

    # Attach the most recent Part heading to each Item heading.
    raw_items = []
    part_idx = 0
    current_part = None
    current_part_title = None

    for item_match in item_matches:
        while part_idx < len(part_matches) and part_matches[part_idx].start() < item_match.start():
            current_part = normalize_part_label(part_matches[part_idx].group("label"))
            current_part_title = clean_item_title(part_matches[part_idx].group("title"))
            part_idx += 1

        raw_items.append({
            "match": item_match,
            "part": current_part,
            "part_title": current_part_title,
            "label": normalize_item_label(item_match.group("label")),
            "title": clean_item_title(item_match.group("title")),
            "raw_title": re.sub(r"\s+", " ", item_match.group("title") or "").strip(),
        })

    # Drop likely table-of-contents item lines. In TOCs, each Item heading is
    # usually followed almost immediately by the next Item and often ends with
    # a page number. Keeping those creates tiny chunks and bad item labels.
    filtered_items = []
    for i, item in enumerate(raw_items):
        match = item["match"]
        next_start = raw_items[i + 1]["match"].start() if i + 1 < len(raw_items) else len(text)
        section_len = next_start - match.start()
        looks_like_toc_line = bool(re.search(r"\s\d{1,4}$", item["raw_title"])) and section_len < 900
        if looks_like_toc_line:
            continue
        filtered_items.append(item)

    if not filtered_items:
        return [(None, None, None, text)]

    sections = []
    for i, item in enumerate(filtered_items):
        match = item["match"]
        start = match.start()
        end = filtered_items[i + 1]["match"].start() if i + 1 < len(filtered_items) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) < MIN_CHUNK_CHARS:
            continue
        sections.append((
            item["part"],
            item["label"],
            item["title"],
            section_text,
        ))

    return sections if sections else [(None, None, None, text)]


def split_paragraphs(text):
    """Split preserved filing text into useful paragraph-like units."""
    paragraphs = []
    for para in re.split(r"\n\s*\n+", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(para)
    return paragraphs


def make_overlap(paragraphs):
    """Return trailing paragraphs whose total size is near CHUNK_OVERLAP."""
    overlap = []
    total = 0
    for para in reversed(paragraphs):
        overlap.insert(0, para)
        total += len(para)
        if total >= CHUNK_OVERLAP:
            break
    return overlap


def chunk_section(section_text):
    """Group paragraphs into coherent chunks without crossing Item boundaries."""
    paragraphs = split_paragraphs(section_text)
    if not paragraphs:
        return []

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is huge, split it with a character window as a fallback.
        if para_len > CHUNK_SIZE:
            if current:
                chunks.append("\n\n".join(current))
                current = make_overlap(current)
                current_len = len("\n\n".join(current))

            start = 0
            while start < para_len:
                piece = para[start:start + CHUNK_SIZE]
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        if current and current_len + para_len + 2 > CHUNK_SIZE:
            piece = "\n\n".join(current)
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append(piece)
            current = make_overlap(current)
            current_len = len("\n\n".join(current))

        current.append(para)
        current_len += para_len + 2

    if current:
        piece = "\n\n".join(current)
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)

    return chunks


def chunk_filing(text, base_meta):
    """Split text by SEC Item, then paragraph groups, carrying rich metadata."""
    records = []
    sections = split_into_item_sections(text)
    global_chunk_index = 0

    for part_label, item_label, item_title, section_text in sections:
        section_chunks = chunk_section(section_text)
        for section_chunk_index, piece in enumerate(section_chunks):
            rec = dict(base_meta)
            part_item = f"P{part_label}-I{item_label}" if part_label and item_label else (item_label or "full")
            rec["id"] = (
                f"{base_meta['ticker']}-{base_meta['form']}-"
                f"{base_meta['accession']}-{part_item}-c{section_chunk_index}"
            )
            rec["chunk_index"] = global_chunk_index
            rec["section_chunk_index"] = section_chunk_index
            rec["part"] = part_label
            rec["item"] = item_label
            rec["part_item"] = part_item
            rec["item_title"] = item_title
            rec["text"] = piece
            records.append(rec)
            global_chunk_index += 1

    return records



# --- Period Derivation -------------------------------------------------------

def derive_period(form, report_date, filing_date):
    """A human-readable period label. report_date is the authoritative sort key."""
    year = (report_date or filing_date)[:4]
    if form == "10-K":
        return f"FY{year}"
    if form == "10-Q":
        return report_date[:7] if report_date else filing_date  # year-month
    return filing_date  # 8-Ks are events; the filing date is the natural label

# --- Run ---------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    since_date = (date.today() - timedelta(days=365 * YEARS_BACK)).isoformat()
    ticker_map = load_ticker_map()

    total_chunks = 0
    total_filings = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for ticker in TICKERS:
            cik = resolve_cik(ticker, ticker_map)
            if not cik:
                print(f"{ticker}: no CIK, skipping")
                continue

            try:
                company, recent = fetch_submissions(cik)
            except Exception as e:
                print(f"{ticker}: could not fetch submissions ({e}), skipping")
                continue

            filings = pick_filings(recent, since_date)
            per_company_chunks = 0

            for f in filings:
                try:
                    html, url = download_filing_html(cik, f["accession"], f["primary_document"])
                    text = html_to_text(html)
                except Exception as e:
                    print(f"  {ticker} {f['form']} {f['filing_date']}: download/parse failed ({e})")
                    continue

                base_meta = {
                    "ticker": ticker,
                    "company": company,
                    "cik": cik,
                    "form": f["form"],
                    "filing_date": f["filing_date"],
                    "report_date": f["report_date"],
                    "period": derive_period(f["form"], f["report_date"], f["filing_date"]),
                    "accession": f["accession"],
                    "source_url": url,
                    "primary_document": f["primary_document"],
                }
                for rec in chunk_filing(text, base_meta):
                    out.write(json.dumps(rec) + "\n")
                    per_company_chunks += 1
                total_filings += 1

            total_chunks += per_company_chunks
            print(f"{ticker:5s} {company[:28]:28s}  {len(filings):2d} filings  {per_company_chunks:5d} chunks")

    print(f"\nDone. {total_filings} filings, {total_chunks} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()