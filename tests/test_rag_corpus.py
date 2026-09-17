"""
Tests for Gazette RAG Corpus (rules_corpus.json) and corpus_loader.py
"""

import json
import pytest
from pathlib import Path


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _find_corpus_path() -> Path:
    """Locate rules_corpus.json using the same candidate list as corpus_loader.py."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.knowledge.corpus_loader import _CORPUS_CANDIDATES
    for c in _CORPUS_CANDIDATES:
        if c.exists():
            return c
    return _CORPUS_CANDIDATES[0]  # will fail in test_corpus_file_exists, surfacing the right error


# -----------------------------------------------------------------------
# Test 1: corpus file existence and validity
# -----------------------------------------------------------------------

def test_corpus_file_exists():
    corpus_path = _find_corpus_path()
    assert corpus_path.exists(), (
        f"rules_corpus.json not found at {corpus_path}. "
        "Run: python scripts/ingest_rules_pdfs.py --no-llm"
    )
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "rules_corpus.json must be a JSON array"
    assert len(data) > 0, "rules_corpus.json is empty — ingestion may have failed"


# -----------------------------------------------------------------------
# Test 2: English query returns results with source provenance
# -----------------------------------------------------------------------

def test_corpus_english_query():
    """Keyword search on the corpus should find MRP-related chunks."""
    corpus_path = _find_corpus_path()
    if not corpus_path.exists():
        pytest.skip("rules_corpus.json not present — run ingest_rules_pdfs.py first")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    query_words = {"mrp", "maximum", "retail", "price", "inclusive", "taxes"}
    hits = [
        doc for doc in corpus
        if any(w in doc.get("text", "").lower() for w in query_words)
    ]
    assert len(hits) > 0, "No chunks found for English MRP query"

    # Every hit must have a non-empty 'source' field for provenance
    for hit in hits[:5]:
        assert hit.get("source"), f"Chunk missing 'source' field: {hit.get('id')}"


# -----------------------------------------------------------------------
# Test 3: Hindi query path (Devanagari characters in corpus)
# -----------------------------------------------------------------------

def test_corpus_hindi_query():
    """
    The corpus must contain at least some Hindi/Devanagari text for Hindi gazette
    notifications. Query for common Hindi terms for 'maximum retail price'.
    """
    corpus_path = _find_corpus_path()
    if not corpus_path.exists():
        pytest.skip("rules_corpus.json not present — run ingest_rules_pdfs.py first")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Look for chunks marked as Hindi or mixed language
    hindi_chunks = [
        doc for doc in corpus
        if doc.get("language") in ("hi", "mixed")
    ]

    # Also keyword search for any Devanagari characters
    import re
    re_deva = re.compile(r"[\u0900-\u097F]")
    deva_chunks = [
        doc for doc in corpus
        if re_deva.search(doc.get("text", ""))
    ]

    total_hindi = max(len(hindi_chunks), len(deva_chunks))
    assert total_hindi > 0, (
        "No Hindi/Devanagari chunks found in corpus. "
        "Verify that Hindi gazette PDFs were extracted correctly by PyMuPDF."
    )


# -----------------------------------------------------------------------
# Test 4: corpus_loader integration — loads into vector store
# -----------------------------------------------------------------------

def test_corpus_loader_loads_documents():
    """load_corpus_into_vector_store() should return > 0 chunks when corpus is present."""
    corpus_path = _find_corpus_path()
    if not corpus_path.exists():
        pytest.skip("rules_corpus.json not present — run ingest_rules_pdfs.py first")

    from app.knowledge.vector_store import ComplianceVectorStore
    from app.knowledge.corpus_loader import load_corpus_into_vector_store

    store = ComplianceVectorStore()
    n = load_corpus_into_vector_store(store)
    assert n > 0, "load_corpus_into_vector_store() returned 0 — corpus may be empty"
    assert len(store._documents) == n, "Vector store document count mismatch"
