"""
app/knowledge/corpus_loader.py
-------------------------------
Loads backend/data/rules_corpus.json into a ComplianceVectorStore at server startup.
Produced by: python scripts/ingest_rules_pdfs.py
"""

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import ComplianceVectorStore

log = logging.getLogger(__name__)

# Candidate paths checked in priority order
_CORPUS_CANDIDATES = [
    # Standard project layout (backend/data/ beside this file's repo root)
    Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "rules_corpus.json",
    # One level up (when running from SIH_26034-main parent)
    Path(__file__).resolve().parent.parent.parent.parent / "backend" / "data" / "rules_corpus.json",
    # Absolute fallback paths for this machine
    Path(r"c:\Users\arajk\Downloads\SIH_26034-main\SIH_26034-main\backend\data\rules_corpus.json"),
    Path(r"c:\Users\arajk\Downloads\SIH_26034-main\backend\data\rules_corpus.json"),
]


def _resolve_corpus_path() -> Optional[Path]:
    """Locate rules_corpus.json. Returns None if not found."""
    env_path = __import__("os").getenv("GAZETTE_CORPUS_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    for candidate in _CORPUS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_corpus_into_vector_store(store: "ComplianceVectorStore") -> int:
    """
    Reads rules_corpus.json and bulk-loads all chunks into the given ComplianceVectorStore.

    Each chunk document is fed as:
        {
          "title":      "<gsr_number> -- <source_file>",
          "text":       "<chunk text>",
          "source":     "<gsr_number> -- <source_file> (Page N/M)",
          "gsr_number": "G.S.R. 779(E)" | None,
          "amends":     ["Rule 6(1)(e)", ...]
        }

    Returns:
        Number of corpus chunks loaded (0 if corpus file not present).
    """
    corpus_path = _resolve_corpus_path()
    if corpus_path is None:
        log.info(
            "rules_corpus.json not found — gazette RAG corpus not loaded. "
            "Run: python scripts/ingest_rules_pdfs.py"
        )
        return 0

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus: list = json.load(f)
    except Exception as exc:
        log.error(f"Failed to read rules_corpus.json: {exc}")
        return 0

    if not isinstance(corpus, list) or not corpus:
        log.warning("rules_corpus.json is empty or not a list — skipping.")
        return 0

    # Build document dicts for add_documents()
    docs = []
    for entry in corpus:
        text = entry.get("text", "").strip()
        if not text:
            continue
        gsr = entry.get("gsr_number") or ""
        source_file = entry.get("source_file", "")
        title = f"{gsr} -- {source_file}" if gsr else source_file
        docs.append({
            "title": title,
            "text": text,
            "source": entry.get("source", title),
            "gsr_number": gsr or None,
            "amends": entry.get("amends", []),
            "notification_date": entry.get("notification_date"),
            "effective_date": entry.get("effective_date"),
            "language": entry.get("language", "en"),
            "applies_to": entry.get("applies_to", "all packages"),
            "key_changes": entry.get("key_changes", []),
        })

    store.add_documents(docs)
    log.info(
        f"[corpus_loader] Loaded {len(docs)} gazette corpus chunks from {corpus_path.name} "
        f"({corpus_path})"
    )
    return len(docs)
