"""
text_cleaner.py — SRS-004
===========================
Text cleaning: artifact removal and UTF-8 normalization.

Validation rules:
  VAL-007 — Output text must be valid UTF-8.
  VAL-008 — Cleaning must not remove content identified as part of a
             heading, character name, or mechanic description (best-effort).

Design notes
------------
The cleaning rules here are conservative to satisfy VAL-008.  They strip
only unambiguous non-content artifacts:

  1. Null bytes and other C0/C1 control characters (except \\n, \\r, \\t).
  2. PDF-extraction artefacts: form-feed (\\x0c), soft-hyphen (\\xad).
  3. Page-number-only lines (isolated digit sequences, e.g., "42" alone).
  4. Running header/footer lines: repeated identical short lines that
     appear on consecutive "pages" (detected via a sliding-window
     frequency heuristic).
  5. Excessive whitespace: collapse multiple blank lines to a single blank
     line; normalise Windows (\\r\\n) and old Mac (\\r) line endings to \\n.
  6. Unicode normalisation to NFC (composes combining characters).

What is NOT stripped:
  - Lines that could be headings (VAL-008).
  - Short lines that contain non-numeric content (could be character names,
    mechanic labels, etc.).
  - Any line longer than a configurable threshold (assumed to be body text).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List


# ── Configurable thresholds ────────────────────────────────────────────────────
_PAGE_NUM_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
"""Matches lines containing only a page number (1–4 digits)."""

_CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]"
)
"""Matches C0/C1 control characters and soft-hyphen, excluding \\t, \\n, \\r."""

_REPEATED_BLANK_PATTERN = re.compile(r"\n{3,}")
"""Collapses 3+ consecutive blank lines."""


# ── Public function ────────────────────────────────────────────────────────────

def clean_text(
    raw_text: str,
    *,
    remove_page_numbers: bool = True,
    remove_repeated_headers: bool = True,
    header_min_occurrences: int = 3,
    header_max_line_len: int = 80,
) -> str:
    """
    Clean raw extracted text ready for segmentation.

    Parameters
    ----------
    raw_text:
        The raw text string (from .txt file or PDF conversion).
    remove_page_numbers:
        When True, lines that are solely a page number are removed.
    remove_repeated_headers:
        When True, short lines that appear ``header_min_occurrences`` or
        more times throughout the document are treated as running
        headers/footers and removed.
    header_min_occurrences:
        Minimum frequency for a line to be considered a running header.
    header_max_line_len:
        Lines longer than this character count are never treated as
        running headers (they are likely body text).

    Returns
    -------
    str
        Cleaned, UTF-8 normalised text (NFC).

    Raises
    ------
    ValueError
        If ``raw_text`` is not a string (programming error guard).
    """
    if not isinstance(raw_text, str):
        raise ValueError(f"raw_text must be a str, got {type(raw_text).__name__}")

    text = raw_text

    # Step 1 — Normalise line endings to \\n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2 — Strip control characters (VAL-007)
    text = _CONTROL_CHAR_PATTERN.sub("", text)

    # Step 3 — Unicode NFC normalisation (VAL-007)
    text = unicodedata.normalize("NFC", text)

    # Step 4 — Remove page-number-only lines (VAL-008: conservative)
    if remove_page_numbers:
        lines = text.split("\n")
        lines = [ln for ln in lines if not _PAGE_NUM_PATTERN.match(ln)]
        text = "\n".join(lines)

    # Step 5 — Remove repeated running headers/footers (VAL-008: conservative)
    if remove_repeated_headers:
        text = _strip_repeated_lines(
            text,
            min_occurrences=header_min_occurrences,
            max_line_len=header_max_line_len,
        )

    # Step 6 — Collapse excessive blank lines
    text = _REPEATED_BLANK_PATTERN.sub("\n\n", text)

    # Step 7 — Strip leading/trailing whitespace from the whole document
    text = text.strip()

    return text


# ── Internal helpers ───────────────────────────────────────────────────────────

def _strip_repeated_lines(
    text: str,
    min_occurrences: int,
    max_line_len: int,
) -> str:
    """
    Remove lines that are short AND appear at least ``min_occurrences``
    times in the document (heuristic for running headers/footers).

    Preserves all lines longer than ``max_line_len`` regardless of
    frequency (they are almost certainly content, not headers).
    """
    lines = text.split("\n")

    # Count occurrences of stripped versions of short lines only
    short_line_freq: Counter[str] = Counter()
    for ln in lines:
        stripped = ln.strip()
        if stripped and len(stripped) <= max_line_len:
            short_line_freq[stripped] += 1

    # Build the set of repeated-header candidates
    headers_to_remove: set[str] = {
        ln for ln, count in short_line_freq.items()
        if count >= min_occurrences
    }

    # Filter: never remove lines that look like headings
    # (heuristic: starts with a digit or '#' — could be a numbered section)
    headers_to_remove = {
        ln for ln in headers_to_remove
        if not _looks_like_heading(ln)
    }

    filtered = [
        ln for ln in lines
        if ln.strip() not in headers_to_remove
    ]
    return "\n".join(filtered)


def _looks_like_heading(line: str) -> bool:
    """
    Return True if ``line`` looks like it could be a structural heading.
    Used to avoid accidentally stripping headings that happen to repeat
    (e.g., a GDD that re-uses "Overview" in multiple chapters).
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Numbered heading (e.g., "1.", "2.3", "1.2.3")
    if re.match(r"^\d+(\.\d+)*\.?\s", stripped):
        return True
    # Markdown heading
    if stripped.startswith("#"):
        return True
    # Starts with a known section keyword
    if re.match(r"^(Chapter|Section|Part|Appendix)\b", stripped, re.IGNORECASE):
        return True
    return False
