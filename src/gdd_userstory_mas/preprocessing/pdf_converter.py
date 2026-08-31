"""
pdf_converter.py — SRS-003
============================
PDF-to-text conversion using ``pdfminer.six``.

Validation rules:
  VAL-005 — Extracted text must not be empty for a non-empty PDF.
  VAL-006 — Conversion failures must be logged with the specific
             page/section that failed.

Design notes
------------
* ``pdfminer.six`` is the selected library (OPEN-010 resolved).  It
  handles text-layer PDFs deterministically.  Scanned/image-only PDFs
  will produce an empty or near-empty result — this is detected and
  raised as ``PDFConversionError``, never silently ignored.
* Page-level extraction is attempted first so that partial failures
  (e.g., one corrupt page) can be logged and the successful pages still
  returned (fail-loudly with graceful degradation for individual pages).
* The caller (pipeline.py) is responsible for logging
  ``ErrorFailureLog`` entries; this module raises typed exceptions and
  returns structured results so the caller has full control.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# pdfminer imports — grouped so the ImportError message is clear
try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter

    _PDFMINER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDFMINER_AVAILABLE = False


# ── Custom exception ───────────────────────────────────────────────────────────

class PDFConversionError(RuntimeError):
    """
    Raised when PDF-to-text conversion fails in a way that cannot be
    recovered from (e.g., completely empty result for a non-empty file,
    or pdfminer not installed).
    """


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PDFConversionResult:
    """
    Outcome of a PDF-to-text conversion attempt.

    Attributes
    ----------
    text:
        The concatenated extracted text across all successfully converted
        pages.
    page_count:
        Total number of pages attempted.
    failed_pages:
        List of 1-based page numbers that produced errors or empty text.
    warnings:
        Non-fatal messages accumulated during conversion (logged by the
        caller).
    """

    text: str
    page_count: int
    failed_pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ── Public function ────────────────────────────────────────────────────────────

def convert_pdf_to_text(
    pdf_path: str | Path,
    *,
    codec: str = "utf-8",
    laparams: Optional[object] = None,  # pdfminer LAParams or None
) -> PDFConversionResult:
    """
    Convert a PDF file to plain text, page by page.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    codec:
        Character encoding for the output text (default UTF-8).
    laparams:
        Optional ``pdfminer.layout.LAParams`` instance.  ``None`` uses
        pdfminer's defaults (sufficient for most GDD layouts).

    Returns
    -------
    PDFConversionResult
        Contains the extracted text and metadata about failures.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist.
    PDFConversionError
        If pdfminer is not installed, or if the entire document produces
        empty text (VAL-005).
    """
    if not _PDFMINER_AVAILABLE:
        raise PDFConversionError(  # pragma: no cover
            "pdfminer.six is not installed.  Run: pip install pdfminer.six"
        )

    resolved = Path(pdf_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"PDF file not found: {resolved}")

    if laparams is None:
        laparams = LAParams()

    page_texts: List[str] = []
    failed_pages: List[int] = []
    warnings: List[str] = []
    page_count = 0

    try:
        with open(resolved, "rb") as pdf_fh:
            for page_num, page in enumerate(PDFPage.get_pages(pdf_fh), start=1):
                page_count += 1
                page_text = _extract_page_text(page, laparams, codec)
                if page_text is None:
                    # Partial failure — log page, continue (VAL-006)
                    failed_pages.append(page_num)
                    warnings.append(
                        f"Page {page_num}: extraction failed or returned empty text."
                    )
                else:
                    page_texts.append(page_text)
    except Exception as exc:  # noqa: BLE001
        raise PDFConversionError(
            f"Fatal error reading PDF '{resolved}': {exc}"
        ) from exc

    full_text = "\n".join(page_texts)

    # VAL-005: total extracted text must not be empty
    if not full_text.strip():
        raise PDFConversionError(
            f"PDF conversion produced empty text for '{resolved}'. "
            "The file may be image-only (scanned). OCR is not supported."
        )

    return PDFConversionResult(
        text=full_text,
        page_count=page_count,
        failed_pages=failed_pages,
        warnings=warnings,
    )


# ── Internal helper ────────────────────────────────────────────────────────────

def _extract_page_text(
    page: object,  # pdfminer.pdfpage.PDFPage
    laparams: object,  # pdfminer.layout.LAParams
    codec: str,
) -> Optional[str]:
    """
    Extract text from a single pdfminer page object.

    Returns the extracted string, or ``None`` if the page failed.
    """
    try:
        output = io.StringIO()
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(page)
        device.close()
        text = output.getvalue()
        output.close()
        return text if text.strip() else None
    except Exception:  # noqa: BLE001
        return None
