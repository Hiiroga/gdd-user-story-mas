"""
ingestion.py — SRS-001, SRS-002
=================================
GDD file acceptance and document / run ID assignment.

Validation rules implemented here:
  VAL-001 — Reject files with extensions other than .txt / .pdf.
  VAL-002 — Reject empty files (0 bytes).
  VAL-003 — Flag (not reject) files with no detectable English content.
  VAL-004 — document_id and run_id must be unique (caller's responsibility
             to check the registry; this module generates UUIDs which are
             probabilistically unique by construction).

Design notes
------------
* All IDs are UUID4 strings for global uniqueness without central state
  (resolves OPEN-019 for this implementation; the format is documented as
  an assumption in the plan).
* Language detection (VAL-003) is deliberately a warning, not a hard block,
  consistent with the "flag not reject" wording in the SRS.
* This module does NOT read file content — that happens in pdf_converter.py
  and text_cleaner.py.  Ingestion only validates metadata.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# langdetect is optional at import time so tests can run without it
try:
    from langdetect import detect as _langdetect_detect
    from langdetect.lang_detect_exception import LangDetectException

    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False
    LangDetectException = Exception  # type: ignore[misc, assignment]


# ── Supported formats (VAL-001) ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf"})


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IngestResult:
    """
    Output of a successful ingestion step.

    All downstream stages receive these identifiers via the run context.
    """

    document_id: str
    """UUID4 string uniquely identifying this GDD document."""

    run_id: str
    """UUID4 string uniquely identifying this pipeline run."""

    file_path: Path
    """Resolved absolute path to the ingested file."""

    file_name: str
    """Original filename (basename only)."""

    format: str  # "txt" | "pdf"
    """Validated file format."""

    ingestion_timestamp: datetime
    """UTC datetime of ingestion."""

    language_warning: Optional[str] = field(default=None)
    """
    Non-null when best-effort language detection suspects the file is not
    English.  A warning, not a rejection (VAL-003).
    """


# ── Public function ────────────────────────────────────────────────────────────

def ingest_file(
    file_path: str | Path,
    *,
    document_id: Optional[str] = None,
    run_id: Optional[str] = None,
    language_sample_chars: int = 2000,
    reject_non_english: bool = False,
) -> IngestResult:
    """
    Validate and register a GDD file for preprocessing.

    Parameters
    ----------
    file_path:
        Path to the GDD file (.txt or .pdf).
    document_id:
        Optionally supply a pre-generated document ID (e.g., from the
        dataset manifest).  If omitted, a UUID4 is generated.
    run_id:
        Optionally supply a pre-generated run ID.  If omitted, a UUID4
        is generated.
    language_sample_chars:
        Number of characters from the start of a .txt file used for
        language detection (VAL-003).  Ignored for PDF (cannot read
        without conversion).
    reject_non_english:
        When True, raise ``ValueError`` if non-English content is
        detected (strict mode).  Default False matches VAL-003 semantics
        (flag, not reject).

    Returns
    -------
    IngestResult
        Metadata bundle for the ingested document.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file fails VAL-001, VAL-002, or VAL-003 (strict mode).
    """
    resolved = Path(file_path).resolve()

    # ── File existence ─────────────────────────────────────────────────────
    if not resolved.exists():
        raise FileNotFoundError(f"GDD file not found: {resolved}")

    # ── VAL-001: Supported extension ───────────────────────────────────────
    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # ── VAL-002: Non-empty file ────────────────────────────────────────────
    file_size = resolved.stat().st_size
    if file_size == 0:
        raise ValueError(f"GDD file is empty (0 bytes): {resolved}")

    # ── VAL-003: Language detection (best-effort, .txt only) ──────────────
    language_warning: Optional[str] = None
    if ext == ".txt" and _LANGDETECT_AVAILABLE:
        language_warning = _detect_language_warning(
            resolved,
            sample_chars=language_sample_chars,
            reject=reject_non_english,
        )

    # ── Assign identifiers (SRS-002) ───────────────────────────────────────
    doc_id = document_id if document_id else str(uuid.uuid4())
    rid = run_id if run_id else str(uuid.uuid4())

    return IngestResult(
        document_id=doc_id,
        run_id=rid,
        file_path=resolved,
        file_name=resolved.name,
        format=ext.lstrip("."),  # "txt" or "pdf"
        ingestion_timestamp=datetime.now(tz=timezone.utc),
        language_warning=language_warning,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _detect_language_warning(
    path: Path,
    sample_chars: int,
    reject: bool,
) -> Optional[str]:
    """
    Read the first ``sample_chars`` characters from a .txt file and try
    to detect the language.

    Returns a warning string if the language appears non-English, or
    ``None`` if detection passes or cannot run.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            sample = fh.read(sample_chars)
    except OSError:
        return None  # can't read → can't warn

    if not sample.strip():
        return None  # empty sample → skip detection

    try:
        detected = _langdetect_detect(sample)
    except LangDetectException:
        return None  # ambiguous / too short → skip

    if detected != "en":
        msg = (
            f"Language detection suggests '{detected}' (expected 'en'). "
            "Proceeding with caution (VAL-003)."
        )
        if reject:
            raise ValueError(msg)
        return msg

    return None
