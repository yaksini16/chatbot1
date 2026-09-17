"""
Gazette PDF Loader and Ingestion Discovery Pipeline
Automatically discovers, validates, and loads statutory Gazette notification PDFs from backend/data/gazette_pdfs/.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_gazette_pdf_dir() -> Path:
    """
    Resolve the official Gazette PDF directory dynamically across execution environments.
    Checks environment variable GAZETTE_PDF_DIR first, then standard paths relative to current directory and workspace.
    """
    env_dir = os.getenv("GAZETTE_PDF_DIR")
    if env_dir and os.path.isdir(env_dir):
        return Path(env_dir).resolve()

    # Potential candidate paths based on directory structure
    candidates = [
        # Relative to current working directory
        Path("backend/data/gazette_pdfs"),
        Path("data/gazette_pdfs"),
        # Relative to repo root from app/knowledge
        Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "gazette_pdfs",
        # Relative to SIH parent workspace root
        Path(__file__).resolve().parent.parent.parent.parent / "backend" / "data" / "gazette_pdfs",
        # Explicit fallback paths
        Path(r"c:\Users\arajk\Downloads\SIH_26034-main\SIH_26034-main\backend\data\gazette_pdfs"),
        Path(r"c:\Users\arajk\Downloads\SIH_26034-main\backend\data\gazette_pdfs"),
    ]

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    # Default fallback
    return (Path(__file__).resolve().parent.parent.parent / "backend" / "data" / "gazette_pdfs").resolve()


def discover_gazette_pdfs(directory: Optional[Path] = None) -> List[Path]:
    """
    Automatically discovers every PDF file in the gazette directory.
    Preserves original filenames and returns sorted list of Path objects.
    """
    target_dir = Path(directory) if directory else get_gazette_pdf_dir()
    if not target_dir.exists():
        logger.warning(f"Gazette directory does not exist: {target_dir}")
        return []

    pdf_files = [
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]
    # Sort for deterministic processing
    return sorted(pdf_files, key=lambda p: p.name.lower())


def inspect_gazette_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Inspects a single Gazette PDF file:
    - Verifies it can be opened
    - Computes page count
    - Extracts basic metadata
    """
    import pypdf

    info = {
        "filename": file_path.name,
        "path": str(file_path.resolve()),
        "size_bytes": file_path.stat().st_size,
        "can_open": False,
        "page_count": 0,
        "error": None,
    }

    try:
        reader = pypdf.PdfReader(str(file_path))
        info["can_open"] = True
        info["page_count"] = len(reader.pages)
    except Exception as exc:
        info["error"] = str(exc)
        logger.error(f"Failed to open PDF {file_path.name}: {exc}")

    return info


def validate_all_gazettes(directory: Optional[Path] = None) -> Dict[str, Any]:
    """
    Discovers and validates every PDF in the directory.
    Returns summary statistics and individual report per file.
    """
    pdf_files = discover_gazette_pdfs(directory)
    file_reports = [inspect_gazette_pdf(f) for f in pdf_files]

    total_files = len(file_reports)
    all_can_open = all(r["can_open"] for r in file_reports) if total_files > 0 else False
    total_pages = sum(r["page_count"] for r in file_reports)

    return {
        "total_pdfs": total_files,
        "all_can_open": all_can_open,
        "total_pages": total_pages,
        "files": file_reports,
    }


def load_gazette_documents(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Extracts text per page from all discovered Gazette PDFs, preparing document chunks
    ready for RAG indexing.
    """
    import pypdf

    documents = []
    pdf_files = discover_gazette_pdfs(directory)

    for pdf_path in pdf_files:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                documents.append({
                    "filename": pdf_path.name,
                    "source": f"{pdf_path.name} (Page {page_num}/{total_pages})",
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "text": text.strip(),
                })
        except Exception as exc:
            logger.error(f"Error loading document {pdf_path.name}: {exc}")

    return documents
