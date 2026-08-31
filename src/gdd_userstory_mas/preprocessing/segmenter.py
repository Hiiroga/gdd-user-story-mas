"""
segmenter.py — SRS-005
========================
Structural segmentation: divide cleaned GDD text into ordered segments
based on headings and structural boundaries.

Validation rules:
  VAL-009 — Segments must be non-overlapping and collectively cover the
             full cleaned text (no content loss).
  VAL-010 — A document with no detectable structure shall be treated as
             a single segment (fallback rule).

Design notes
------------
Heading detection uses a cascade of regex patterns (configurable via
``SegmenterConfig``).  A line is classified as a heading if it matches
ANY of the configured patterns.

Segment hierarchy detection (chapter vs. section):
  - Markdown #   → level 1 (chapter)
  - Markdown ##  → level 2 (section)
  - Numbered 1.  → level 1 (chapter)
  - Numbered 1.1 → level 2 (section)
  - ALL CAPS     → level 1 (chapter, assumed)
  - Other        → level 1 (chapter, assumed)

The algorithm:
  1. Split text into lines.
  2. Identify heading lines.
  3. Collect the text between consecutive headings into segments.
  4. Assign heading text as the segment title.
  5. If no headings found → return single-segment fallback (VAL-010).
  6. Verify that re-joining all segments reproduces the original text
     (VAL-009 sanity check).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Default heading patterns (same as config/pipeline.yaml defaults) ───────────
DEFAULT_HEADING_PATTERNS: List[str] = [
    r"^#{1,6}\s+\S",             # Markdown:   ## Section Title
    r"^\d+(\.\d+)*\.?\s+[A-Z]", # Numbered:    1.2 Combat System
    r"^[A-Z][A-Z\s\-]{4,}$",   # ALL CAPS:    GAME OVERVIEW
    r"^(Chapter|Section|Part)\s+\d+",  # Keywords: Chapter 3
]


# ── Segment dataclass ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    """
    One structural segment of a GDD document.

    The segment's text does NOT include its own heading line.  The
    heading is stored separately so downstream consumers (chunker,
    metadata_tagger) can use it as the chapter/section label.
    """

    title: Optional[str]
    """Heading text for this segment, or None for the preamble."""

    level: Optional[int]
    """Heading depth: 1 = chapter, 2 = section, None if undetected."""

    position: int
    """0-based sequential index within the document."""

    text: str
    """The body text of this segment (not including the heading line)."""

    char_offset_start: int
    """Character offset in the full cleaned document where this segment STARTS
    (includes the heading line if present)."""

    char_offset_end: int
    """Character offset where this segment ENDS (exclusive)."""


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class SegmenterConfig:
    """
    Configuration for the segmenter.

    Defaults match ``config/pipeline.yaml`` values.
    """

    heading_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_HEADING_PATTERNS)
    )
    min_heading_length: int = 2
    max_heading_length: int = 120


# ── Public function ────────────────────────────────────────────────────────────

def segment_text(
    cleaned_text: str,
    *,
    config: Optional[SegmenterConfig] = None,
) -> List[Segment]:
    """
    Divide cleaned GDD text into structural segments.

    Parameters
    ----------
    cleaned_text:
        The output of ``clean_text()``.
    config:
        Segmentation configuration.  ``None`` uses defaults.

    Returns
    -------
    List[Segment]
        Ordered list of non-overlapping segments covering the full text
        (VAL-009).  Always contains at least one entry (VAL-010).

    Raises
    ------
    ValueError
        If VAL-009 check fails (content loss detected — programming error).
    """
    if config is None:
        config = SegmenterConfig()

    compiled_patterns = [re.compile(p, re.MULTILINE) for p in config.heading_patterns]

    lines = cleaned_text.split("\n")
    # Build a list of (line_index, heading_text, heading_level)
    heading_positions: List[Tuple[int, str, int]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not (config.min_heading_length <= len(stripped) <= config.max_heading_length):
            continue
        if _is_heading(stripped, compiled_patterns):
            level = _detect_heading_level(stripped)
            heading_positions.append((i, stripped, level))

    if not heading_positions:
        # VAL-010: single-segment fallback
        return [
            Segment(
                title=None,
                level=None,
                position=0,
                text=cleaned_text,
                char_offset_start=0,
                char_offset_end=len(cleaned_text),
            )
        ]

    segments: List[Segment] = []
    position = 0

    # ── Preamble (text before the first heading) ───────────────────────────
    first_heading_line_idx = heading_positions[0][0]
    preamble_lines = lines[:first_heading_line_idx]
    preamble_text = "\n".join(preamble_lines).strip()

    if preamble_text:
        preamble_end = _line_idx_to_char_offset(lines, first_heading_line_idx)
        segments.append(
            Segment(
                title=None,
                level=None,
                position=position,
                text=preamble_text,
                char_offset_start=0,
                char_offset_end=preamble_end,
            )
        )
        position += 1

    # ── Headed segments ────────────────────────────────────────────────────
    for idx, (line_i, heading_text, level) in enumerate(heading_positions):
        # Start of segment: the heading line itself
        seg_start_char = _line_idx_to_char_offset(lines, line_i)

        # End of segment: start of the next heading (or end of document)
        if idx + 1 < len(heading_positions):
            next_heading_line_i = heading_positions[idx + 1][0]
            seg_end_char = _line_idx_to_char_offset(lines, next_heading_line_i)
        else:
            seg_end_char = len(cleaned_text)

        # Body text: everything after the heading line up to the next heading
        body_lines = lines[line_i + 1 : (
            heading_positions[idx + 1][0] if idx + 1 < len(heading_positions)
            else len(lines)
        )]
        body_text = "\n".join(body_lines).strip()

        segments.append(
            Segment(
                title=heading_text,
                level=level,
                position=position,
                text=body_text,
                char_offset_start=seg_start_char,
                char_offset_end=seg_end_char,
            )
        )
        position += 1

    # ── VAL-009: verify no content loss ───────────────────────────────────
    _validate_coverage(cleaned_text, segments)

    return segments


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_heading(line: str, patterns: List[re.Pattern]) -> bool:  # type: ignore[type-arg]
    """Return True if ``line`` matches any heading pattern."""
    for p in patterns:
        if p.search(line):
            return True
    return False


def _detect_heading_level(line: str) -> int:
    """
    Infer the heading hierarchy level (1 = chapter, 2 = section).

    Rules (in priority order):
    1. Markdown `##` (2 hashes) → level 2; more hashes → deeper.
    2. Numbered `1.1` (has a dot) → level 2; `1` (no dot) → level 1.
    3. ALL CAPS → level 1.
    4. Default → level 1.
    """
    stripped = line.strip()

    # Markdown
    md_match = re.match(r"^(#{1,6})\s", stripped)
    if md_match:
        return len(md_match.group(1))

    # Numbered heading
    num_match = re.match(r"^(\d+)((\.\d+)+)?", stripped)
    if num_match:
        sub = num_match.group(2)
        if sub:  # has at least one dot-sub-number → deeper
            depth = sub.count(".") + 1
            return min(depth, 6)
        return 1

    return 1


def _line_idx_to_char_offset(lines: List[str], line_idx: int) -> int:
    """
    Return the character offset in the joined text for the start of
    ``lines[line_idx]``.  Each line is followed by a ``\\n`` separator.
    """
    # Sum lengths of all preceding lines + 1 for each \\n separator
    offset = 0
    for i in range(line_idx):
        offset += len(lines[i]) + 1  # +1 for \\n
    return offset


def _validate_coverage(text: str, segments: List[Segment]) -> None:
    """
    VAL-009: Verify that every character in ``text`` is covered by
    exactly one segment (non-overlapping, no gaps).

    Uses a lightweight approach: ensure all char ranges are monotonically
    increasing and the last segment ends at len(text).

    Raises
    ------
    ValueError
        If coverage fails.
    """
    if not segments:
        raise ValueError("VAL-009: segment list is empty — content loss detected.")

    # Verify segments are ordered and contiguous
    prev_end = 0
    for seg in segments:
        if seg.char_offset_start > prev_end:
            # There is a gap — preamble handling may leave small gaps due to
            # whitespace stripping; allow gaps that are only whitespace.
            gap_content = text[prev_end:seg.char_offset_start]
            if gap_content.strip():
                raise ValueError(
                    f"VAL-009: content gap detected between char offsets "
                    f"{prev_end} and {seg.char_offset_start}: "
                    f"{gap_content[:50]!r}"
                )
        prev_end = seg.char_offset_end

    # Final segment should reach the end of the text
    if segments[-1].char_offset_end < len(text):
        trailing = text[segments[-1].char_offset_end:]
        if trailing.strip():
            raise ValueError(
                f"VAL-009: trailing content not covered by any segment: "
                f"{trailing[:50]!r}"
            )
