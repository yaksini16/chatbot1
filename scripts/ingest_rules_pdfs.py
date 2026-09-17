#!/usr/bin/env python3
"""
scripts/ingest_rules_pdfs.py
----------------------------
PRAMAN v4 — Official Gazette PDF -> RAG Corpus Ingestion Pipeline

Turns every PDF in backend/data/gazette_pdfs/ into:
  1. backend/data/rules_corpus.json       -- RAG chunk corpus (per-page chunks with metadata)
  2. backend/data/rules_patch_review.json -- proposed patches to pipeline/rules/rules.json (human review)

Usage:
  python scripts/ingest_rules_pdfs.py             # LLM-enriched (default, uses GOOGLE_API_KEY)
  python scripts/ingest_rules_pdfs.py --no-llm    # Regex-only; no API calls
  python scripts/ingest_rules_pdfs.py --dir PATH  # Custom PDF directory
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_rules_pdfs")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
GAZETTE_PDF_DIR_DEFAULT = PROJECT_ROOT / "backend" / "data" / "gazette_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "data"

# Devanagari Unicode block: U+0900-U+097F
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Gazette header / watermark patterns to strip
RE_STRIP_LINES = re.compile(
    r"^(?:"
    r"REGD\. NO\..*|"
    r"eGazette.*|"
    r"Gazette\s+of\s+India.*|"
    r"\[\s*PART\s+II.*\].*|"
    r"www\.egazette\.nic\.in.*|"
    r"\d+\s*$"
    r")",
    re.IGNORECASE,
)

RE_FILENAME_DATE = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})\s+")
RE_GSR = re.compile(r"G\.?\s*S\.?\s*R\.?\s*(\d+)\s*\(E\)", re.IGNORECASE)
RE_RULE_CLAUSE = re.compile(r"Rule\s+(\d+\s*\([^\)]+\)(?:\([^\)]+\))?)", re.IGNORECASE)
RE_DATE_LONG = re.compile(
    r"\b(\d{1,2})\s+("
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|"
    r"[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember"
    r")\s+(\d{4})\b"
)

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

FIELD_CLAUSE_MAP = {
    "net_quantity": ["6(1)(c)"],
    "mrp": ["6(1)(e)"],
    "manufacturer": ["6(1)(a)"],
    "manufacture_date": ["6(1)(d)"],
    "use_by": ["6(1)(da)"],
    "consumer_care": ["6(2)"],
    "fssai": [],
    "country_of_origin": ["6(1)(aa)"],
}

LLM_SYSTEM_PROMPT = (
    "You are a Legal Metrology regulatory analyst. Below is text extracted from an Indian "
    "government gazette notification PDF (may be Hindi, English, or mixed).\n\n"
    "TASK - return STRICT JSON only (no markdown, no code fences):\n"
    "{\n"
    '  "gsr_number": "G.S.R. ___(E)" or null,\n'
    '  "notification_date": "YYYY-MM-DD" or null,\n'
    '  "effective_date": "YYYY-MM-DD" or null,\n'
    '  "language": "hi" or "en" or "mixed",\n'
    '  "amends": ["Rule 6(1)(e)"],\n'
    '  "summary_en": "2-3 sentence English summary",\n'
    '  "key_changes": ["each substantive change, one line, cite clause"],\n'
    '  "quotable_clauses": [\n'
    '    {"clause": "Rule 6(1)(e)", "text_en": "...", "text_original": "..."}\n'
    '  ],\n'
    '  "penalty_or_enforcement": "..." or null,\n'
    '  "applies_to": "all packages" or "e-commerce" or "garments" or describe precisely\n'
    "}\n\n"
    "RULES:\n"
    "- Quote statutory text VERBATIM in text_original (keep Hindi as-is, keep Rs/rupee symbol, numerals).\n"
    "- Never invent G.S.R. numbers or dates; use null when absent.\n"
    "- Dates may be in Hindi (e.g. tithi) - normalize to YYYY-MM-DD.\n"
    "- If the PDF is an SOP/circular (not an amendment), set amends=[] and summarize in key_changes.\n"
    "- Return ONLY the JSON object, nothing else.\n\n"
    "PDF TEXT:\n<<<TEXT>>>"
)


# ===========================================================================
# TEXT EXTRACTION  (PyMuPDF / fitz)
# ===========================================================================

def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text per page using PyMuPDF. Returns [{page_num, raw_text}]."""
    try:
        import pymupdf as fitz  # pymupdf >= 1.24 preferred import
    except ImportError:
        try:
            import fitz  # legacy alias fallback
        except ImportError:
            log.error("pymupdf not installed. Run: pip install pymupdf")
            sys.exit(1)

    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            pages.append({"page_num": page_num, "raw_text": raw or ""})
        doc.close()
    except Exception as exc:
        log.warning(f"  Could not extract {pdf_path.name}: {exc}")
    return pages


# ===========================================================================
# LANGUAGE DETECTION
# ===========================================================================

def detect_language(text: str) -> str:
    if not text.strip():
        return "en"
    deva_chars = len(RE_DEVANAGARI.findall(text))
    ratio = deva_chars / max(len(text), 1)
    if ratio > 0.40:
        return "hi"
    if ratio > 0.08:
        return "mixed"
    return "en"


# ===========================================================================
# TEXT CLEANING
# ===========================================================================

def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if RE_STRIP_LINES.match(line_s):
            continue
        cleaned.append(line_s)
    return "\n".join(cleaned)


# ===========================================================================
# CHUNKING
# ===========================================================================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sliding-window chunker. Extends boundary to next newline so clauses
    are never split mid-sentence. Always terminates.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Extend to nearest newline (up to 200 chars) to avoid mid-clause splits
        if end < len(text):
            nl = text.find("\n", end)
            if nl != -1 and nl - end < 200:
                end = nl + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # If we've consumed to end of text, stop — avoids infinite loop
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ===========================================================================
# REGEX METADATA EXTRACTION
# ===========================================================================

def parse_metadata_from_filename(filename: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "notification_date": None,
        "effective_date": None,
        "gsr_number": None,
    }
    m = RE_FILENAME_DATE.match(filename)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        meta["notification_date"] = f"{y}-{mo}-{d}"
    return meta


def parse_metadata_from_text(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "gsr_number": None,
        "notification_date": None,
        "effective_date": None,
        "amends": [],
        "language": detect_language(text),
    }

    gsr_m = RE_GSR.search(text)
    if gsr_m:
        meta["gsr_number"] = f"G.S.R. {gsr_m.group(1)}(E)"

    date_matches = RE_DATE_LONG.findall(text)
    parsed_dates = []
    for day, month_str, year in date_matches:
        mo = MONTH_MAP.get(month_str.lower())
        if mo:
            parsed_dates.append(f"{year}-{mo}-{day.zfill(2)}")
    if parsed_dates:
        meta["notification_date"] = meta["notification_date"] or parsed_dates[0]
        if len(parsed_dates) > 1:
            meta["effective_date"] = parsed_dates[1]

    clauses = list({f"Rule {m}" for m in RE_RULE_CLAUSE.findall(text)})
    meta["amends"] = sorted(clauses)

    return meta


# ===========================================================================
# LLM ENRICHMENT  (Gemini)
# ===========================================================================

def classify_with_llm(full_text: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    try:
        from google import genai
    except ImportError:
        log.warning("  google-genai not installed - skipping LLM enrichment")
        return None

    try:
        client = genai.Client(api_key=api_key)
        prompt = LLM_SYSTEM_PROMPT.replace("<<<TEXT>>>", full_text[:6000])
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        raw_json = response.text.strip()
        raw_json = re.sub(r"^```(?:json)?", "", raw_json).strip()
        raw_json = re.sub(r"```$", "", raw_json).strip()
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        log.warning(f"  LLM returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"  LLM call failed: {type(e).__name__}")
        return None



# ===========================================================================
# EMIT CHUNKS
# ===========================================================================

def emit_chunks(
    text_chunks: List[str],
    base_meta: Dict[str, Any],
    llm_meta: Optional[Dict[str, Any]],
    pdf_name: str,
    page_num: int,
    total_pages: int,
) -> List[Dict[str, Any]]:
    gsr = (llm_meta or {}).get("gsr_number") or base_meta.get("gsr_number")
    notif_date = (llm_meta or {}).get("notification_date") or base_meta.get("notification_date")
    eff_date = (llm_meta or {}).get("effective_date") or base_meta.get("effective_date")
    amends = (llm_meta or {}).get("amends") or base_meta.get("amends") or []
    language = (llm_meta or {}).get("language") or base_meta.get("language", "en")
    applies_to = (llm_meta or {}).get("applies_to", "all packages")
    summary_en = (llm_meta or {}).get("summary_en", "")
    key_changes = (llm_meta or {}).get("key_changes", [])
    quotable_clauses = (llm_meta or {}).get("quotable_clauses", [])

    source_label = f"{pdf_name} (Page {page_num}/{total_pages})"
    if gsr:
        source_label = f"{gsr} -- {source_label}"

    docs = []
    for i, chunk in enumerate(text_chunks):
        docs.append({
            "id": str(uuid.uuid4()),
            "source_file": pdf_name,
            "page_number": page_num,
            "chunk_index": i,
            "source": source_label,
            "gsr_number": gsr,
            "notification_date": notif_date,
            "effective_date": eff_date,
            "amends": amends,
            "language": language,
            "applies_to": applies_to,
            "summary_en": summary_en,
            "key_changes": key_changes,
            "quotable_clauses": quotable_clauses,
            "text": chunk,
        })
    return docs


# ===========================================================================
# RULES PATCH REVIEW GENERATION
# ===========================================================================

def build_rules_patch_review(corpus_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patches = []
    for field_key, clauses in FIELD_CLAUSE_MAP.items():
        if not clauses:
            continue
        best_gsr: Optional[str] = None
        best_eff: Optional[str] = None
        best_source: Optional[str] = None

        for doc in corpus_docs:
            for clause_num in clauses:
                full_clause = f"Rule {clause_num}"
                if full_clause in doc.get("amends", []):
                    gsr = doc.get("gsr_number")
                    eff = doc.get("effective_date") or doc.get("notification_date")
                    if gsr and best_gsr is None:
                        best_gsr = gsr
                        best_eff = eff
                        best_source = doc.get("source_file")

        if best_gsr:
            patches.append({
                "field_key": field_key,
                "proposed_gsr_number": best_gsr,
                "proposed_effective_date": best_eff,
                "proposed_gazette_source": best_source,
                "review_status": "PENDING",
                "note": (
                    "Auto-proposed by ingest_rules_pdfs.py. "
                    "Verify against gazette text before merging into rules.json."
                ),
            })
    return patches


# ===========================================================================
# MAIN
# ===========================================================================

def resolve_gazette_dir(cli_dir: Optional[str]) -> Path:
    if cli_dir:
        p = Path(cli_dir)
        if not p.exists():
            log.error(f"Supplied --dir does not exist: {p}")
            sys.exit(1)
        return p

    candidates = [
        GAZETTE_PDF_DIR_DEFAULT,
        PROJECT_ROOT.parent / "backend" / "data" / "gazette_pdfs",
        Path(r"c:\Users\arajk\Downloads\SIH_26034-main\SIH_26034-main\backend\data\gazette_pdfs"),
        Path(r"c:\Users\arajk\Downloads\SIH_26034-main\backend\data\gazette_pdfs"),
    ]
    for c in candidates:
        if c.exists():
            return c

    log.error("Could not locate gazette_pdfs directory. Use --dir to specify.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest official Gazette PDFs into PRAMAN RAG corpus."
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Gemini LLM enrichment; use regex-only metadata.")
    parser.add_argument("--dir", default=None,
                        help="Custom path to gazette PDF directory.")
    args = parser.parse_args()

    gazette_dir = resolve_gazette_dir(args.dir)
    pdf_files = sorted(
        [p for p in gazette_dir.iterdir() if p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )

    if not pdf_files:
        log.error(f"No PDF files found in: {gazette_dir}")
        sys.exit(1)

    use_llm = not args.no_llm
    api_key = None
    if use_llm:
        # Load .env if present; fall back to .env.example (common in dev setups)
        for env_filename in (".env", ".env.example"):
            env_path = PROJECT_ROOT / env_filename
            if env_path.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_path)
                    break
                except ImportError:
                    pass
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            log.warning(
                "GEMINI_API_KEY not set — falling back to regex-only mode. "
                "Set GEMINI_API_KEY in .env to enable LLM enrichment."
            )
            use_llm = False

    print("=" * 70)
    print("  PRAMAN v4 -- Official Gazette PDF -> RAG Corpus Ingestion")
    print("=" * 70)
    print(f"  PDF directory : {gazette_dir}")
    print(f"  PDFs found    : {len(pdf_files)}")
    print(f"  LLM mode      : {'ON (Gemini gemini-3.6-flash)' if use_llm else 'OFF (regex only)'}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print("=" * 70)


    all_corpus_docs: List[Dict[str, Any]] = []
    failed: List[str] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx:02d}/{len(pdf_files)}] {pdf_path.name}")
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            failed.append(pdf_path.name)
            continue

        full_text = "\n".join(p["raw_text"] for p in pages)
        filename_meta = parse_metadata_from_filename(pdf_path.name)
        text_meta = parse_metadata_from_text(full_text)
        base_meta = {**text_meta}
        if filename_meta["notification_date"]:
            base_meta["notification_date"] = filename_meta["notification_date"]

        llm_meta: Optional[Dict[str, Any]] = None
        if use_llm:
            print(f"   -> LLM classification ...", end=" ", flush=True)
            t0 = time.time()
            llm_meta = classify_with_llm(full_text, api_key)
            elapsed = time.time() - t0
            if llm_meta:
                print(f"OK ({elapsed:.1f}s) | GSR: {llm_meta.get('gsr_number')} | amends: {llm_meta.get('amends')}")
            else:
                print("FAILED (using regex fallback)")

        total_pages = len(pages)
        for page_info in pages:
            page_num = page_info["page_num"]
            cleaned = clean_text(page_info["raw_text"])
            chunks = chunk_text(cleaned)
            if not chunks:
                continue
            docs = emit_chunks(
                chunks, base_meta, llm_meta, pdf_path.name, page_num, total_pages
            )
            all_corpus_docs.extend(docs)
        print(f"   -> {total_pages} pages, {sum(1 for d in all_corpus_docs if d['source_file'] == pdf_path.name)} chunks total")

    all_corpus_docs.sort(
        key=lambda d: (d.get("notification_date") or "9999-99-99", d["source_file"])
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = OUTPUT_DIR / "rules_corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_corpus_docs, f, ensure_ascii=False, indent=2)

    patch_review = build_rules_patch_review(all_corpus_docs)
    patch_path = OUTPUT_DIR / "rules_patch_review.json"
    with open(patch_path, "w", encoding="utf-8") as f:
        json.dump(patch_review, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"  Corpus chunks written : {len(all_corpus_docs)}")
    print(f"  Corpus file           : {corpus_path}")
    print(f"  Patch review file     : {patch_path}")
    print(f"  PDFs processed        : {len(pdf_files) - len(failed)}/{len(pdf_files)}")
    if failed:
        print(f"  FAILED PDFs           : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
