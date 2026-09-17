"""
Knowledge and Gazette PDF ingestion package for Legal Metrology.
"""

from .vector_store import ComplianceVectorStore
from .gazette_loader import (
    get_gazette_pdf_dir,
    discover_gazette_pdfs,
    inspect_gazette_pdf,
    validate_all_gazettes,
    load_gazette_documents,
)
from .corpus_loader import load_corpus_into_vector_store

__all__ = [
    "ComplianceVectorStore",
    "get_gazette_pdf_dir",
    "discover_gazette_pdfs",
    "inspect_gazette_pdf",
    "validate_all_gazettes",
    "load_gazette_documents",
    "load_corpus_into_vector_store",
]
