"""
Unit tests for pure functions — no network calls, no paid API calls.

Covers:
  - canonical_section  (embed_and_search)
  - build_where        (embed_and_search)
  - diversify_results  (embed_and_search)
  - detect_tickers     (ask)
  - chunk_section      (ingest)
"""

import types

import pytest

from embed_and_search import build_where, canonical_section, diversify_results, TOP_K
from ask import detect_tickers
from ingest import chunk_section, CHUNK_SIZE, MIN_CHUNK_CHARS


# ── canonical_section ─────────────────────────────────────────────────────────

class TestCanonicalSection:
    def test_mda(self):
        assert canonical_section("Management's Discussion and Analysis") == "mda"

    def test_mda_of_financial_condition(self):
        assert canonical_section(
            "Management's Discussion and Analysis of Financial Condition and Results of Operations"
        ) == "mda"

    def test_risk_factors(self):
        assert canonical_section("Item 1A. Risk Factors") == "risk_factors"

    def test_risk_factors_split_word(self):
        # Microsoft renders headings as split spans: "RIS K FACTORS"
        assert canonical_section("ITEM 1A. RIS K FACTORS") == "risk_factors"

    def test_risk_factors_uppercase(self):
        assert canonical_section("RISK FACTORS") == "risk_factors"

    def test_market_risk(self):
        assert canonical_section(
            "Quantitative and Qualitative Disclosures About Market Risk"
        ) == "market_risk"

    def test_financial_statements(self):
        assert canonical_section("Financial Statements") == "financial_statements"

    def test_financial_statements_supplementary(self):
        assert canonical_section(
            "Financial Statements and Supplementary Data"
        ) == "financial_statements"

    def test_legal_proceedings(self):
        assert canonical_section("Legal Proceedings") == "legal_proceedings"

    def test_legal_proceedings_split(self):
        assert canonical_section("LEGAL PROCEEDINGS") == "legal_proceedings"

    def test_controls(self):
        assert canonical_section("Controls and Procedures") == "controls"

    def test_disclosure_controls(self):
        assert canonical_section(
            "Disclosure Controls and Procedures"
        ) == "controls"

    def test_results_of_operations(self):
        assert canonical_section("Results of Operations") == "results_of_operations"

    def test_unregistered_sales(self):
        assert canonical_section(
            "Unregistered Sales of Equity Securities"
        ) == "unregistered_sales"

    def test_material_agreement(self):
        assert canonical_section(
            "Entry into a Material Definitive Agreement"
        ) == "material_agreement"

    def test_other(self):
        assert canonical_section("Item 5. Properties") == "other"

    def test_none(self):
        assert canonical_section(None) == "other"

    def test_empty(self):
        assert canonical_section("") == "other"

    def test_curly_apostrophe_normalised(self):
        # Curly apostrophes in "Management’s" should not break the mda match.
        assert canonical_section("Management’s Discussion and Analysis") == "mda"


# ── build_where ───────────────────────────────────────────────────────────────

def _ns(**kwargs):
    """Build a SimpleNamespace with all expected fields, defaulting to empty string."""
    defaults = dict(ticker="", form="", section="", item="", period="")
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class TestBuildWhere:
    def test_no_filters(self):
        assert build_where(_ns()) is None

    def test_ticker_only(self):
        assert build_where(_ns(ticker="AAPL")) == {"ticker": "AAPL"}

    def test_ticker_uppercased(self):
        assert build_where(_ns(ticker="aapl")) == {"ticker": "AAPL"}

    def test_form_only(self):
        assert build_where(_ns(form="10-k")) == {"form": "10-K"}

    def test_section_only(self):
        assert build_where(_ns(section="MDA")) == {"section": "mda"}

    def test_ticker_and_form(self):
        result = build_where(_ns(ticker="NVDA", form="10-K"))
        assert result == {"$and": [{"ticker": "NVDA"}, {"form": "10-K"}]}

    def test_three_filters(self):
        result = build_where(_ns(ticker="MSFT", form="10-Q", section="risk_factors"))
        assert result == {"$and": [
            {"ticker": "MSFT"},
            {"form": "10-Q"},
            {"section": "risk_factors"},
        ]}

    def test_period_only(self):
        assert build_where(_ns(period="FY2024")) == {"period": "FY2024"}


# ── diversify_results ─────────────────────────────────────────────────────────

def _make_result(id_, ticker, accession="acc1", score=0.8):
    return {
        "id": id_,
        "similarity": score,
        "rerank_score": score,
        "text": "some text",
        "metadata": {"ticker": ticker, "accession": accession},
    }


class TestDiversifyResults:
    def test_empty(self):
        assert diversify_results([]) == []

    def test_single(self):
        r = [_make_result("a", "AAPL")]
        assert diversify_results(r) == r

    def test_all_same_ticker_returns_first_k(self):
        results = [_make_result(f"a{i}", "AAPL", score=0.9 - i * 0.1) for i in range(10)]
        out = diversify_results(results, k=3)
        # All are AAPL — first pass yields only 1; second pass fills to k.
        assert len(out) == 3
        assert all(r["metadata"]["ticker"] == "AAPL" for r in out)

    def test_diverse_by_ticker_one_per_ticker(self):
        results = [
            _make_result("a1", "AAPL", score=0.9),
            _make_result("n1", "NVDA", score=0.85),
            _make_result("a2", "AAPL", score=0.8),
            _make_result("m1", "MSFT", score=0.75),
        ]
        out = diversify_results(results, k=3, by="ticker")
        tickers = [r["metadata"]["ticker"] for r in out]
        assert tickers == ["AAPL", "NVDA", "MSFT"]

    def test_diverse_by_filing(self):
        results = [
            _make_result("a1", "AAPL", accession="acc1", score=0.9),
            _make_result("a2", "AAPL", accession="acc1", score=0.85),
            _make_result("a3", "AAPL", accession="acc2", score=0.8),
        ]
        out = diversify_results(results, k=3, by="filing")
        accessions = [r["metadata"]["accession"] for r in out]
        # First two slots go to distinct accessions; third fills from remaining.
        assert accessions[0] != accessions[1]

    def test_respects_k(self):
        results = [_make_result(f"r{i}", f"T{i}") for i in range(10)]
        out = diversify_results(results, k=4)
        assert len(out) == 4

    def test_k_larger_than_results(self):
        results = [_make_result("a", "AAPL"), _make_result("b", "MSFT")]
        out = diversify_results(results, k=10)
        assert len(out) == 2


# ── detect_tickers ────────────────────────────────────────────────────────────

class TestDetectTickers:
    def test_ticker_symbol(self):
        assert detect_tickers("What did AAPL report?") == ["AAPL"]

    def test_full_name(self):
        assert detect_tickers("What did Apple report?") == ["AAPL"]

    def test_multiple_companies(self):
        result = detect_tickers("Compare Apple and Microsoft cloud revenue")
        assert "AAPL" in result
        assert "MSFT" in result

    def test_multi_word_alias(self):
        # "advanced micro devices" must not also match a shorter alias
        assert detect_tickers("Advanced Micro Devices GPU roadmap") == ["AMD"]

    def test_aws_alias(self):
        assert detect_tickers("What is AWS revenue growth?") == ["AMZN"]

    def test_facebook_alias(self):
        assert detect_tickers("What did Facebook say about AI?") == ["META"]

    def test_no_match(self):
        assert detect_tickers("What is the weather in New York?") == []

    def test_deduplication(self):
        # "Apple" and "AAPL" both name the same company.
        result = detect_tickers("Apple (AAPL) reported strong earnings")
        assert result == ["AAPL"]

    def test_case_insensitive(self):
        assert detect_tickers("nvidia revenue") == ["NVDA"]

    def test_broadcom(self):
        assert detect_tickers("Broadcom acquisition strategy") == ["AVGO"]

    def test_alphabet(self):
        assert detect_tickers("Alphabet cloud growth") == ["GOOGL"]


# ── chunk_section ─────────────────────────────────────────────────────────────

class TestChunkSection:
    def test_empty(self):
        assert chunk_section("") == []

    def test_short_text_single_chunk(self):
        # Paragraphs must be >= MIN_CHUNK_CHARS (250) to survive the filter.
        para1 = "The company reported strong results. " * 8   # ~296 chars
        para2 = "Revenue grew across all segments. " * 8       # ~272 chars
        text = para1 + "\n\n" + para2
        chunks = chunk_section(text)
        assert len(chunks) == 1
        assert "strong results" in chunks[0]

    def test_long_text_splits(self):
        # Build text clearly over CHUNK_SIZE by repeating a paragraph
        para = "A" * 200 + " word filler text.\n\n"
        text = para * 30  # ~6000+ chars, should split into 2+ chunks
        chunks = chunk_section(text)
        assert len(chunks) >= 2

    def test_chunks_not_too_small(self):
        para = "B" * 200 + " some content.\n\n"
        text = para * 30
        chunks = chunk_section(text)
        assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks)

    def test_chunks_respect_size_limit(self):
        # Each chunk (barring overlap and the very last) should be <= CHUNK_SIZE.
        para = "C" * 100 + " content here.\n\n"
        text = para * 50
        chunks = chunk_section(text)
        # Allow a small margin for the overlap stitching at boundaries.
        for c in chunks:
            assert len(c) <= CHUNK_SIZE + 200

    def test_whitespace_only(self):
        assert chunk_section("   \n\n   ") == []
