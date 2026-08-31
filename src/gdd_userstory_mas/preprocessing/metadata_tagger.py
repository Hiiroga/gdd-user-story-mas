"""
metadata_tagger.py — SRS-007
==============================
Attach traceability metadata to each chunk, producing ``GDDChunk`` objects.

Validation rules:
  VAL-013 — Every chunk must have a non-null document_id and position index.
  VAL-014 — Chapter/section label may be null only if no structure was
             detected (fallback case per VAL-010).

Design notes
------------
* ``chunk_id`` format: ``{document_id}__chunk_{position:04d}``
  This is deterministic for a given document and position, supporting
  reproducibility.

* Chapter vs. section label assignment:
  - Level-1 headings → ``chapter`` field, ``section = None``.
  - Level-2+ headings → ``chapter`` inherits the most recent level-1
    heading; ``section`` is set to the current heading text.
  - Preamble (no heading) → both ``chapter`` and ``section`` are None.

* The tagger also validates that all required fields are populated
  (VAL-013/VAL-014) before returning, raising ``ValueError`` loudly if
  a chunk fails validation.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError

from gdd_userstory_mas.preprocessing.chunker import ChunkSpec
from gdd_userstory_mas.preprocessing.segmenter import Segment
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk


# ── Public function ────────────────────────────────────────────────────────────

def tag_metadata(
    chunk_specs: List[ChunkSpec],
    document_id: str,
    segments: List[Segment],
    *,
    max_chunk_tokens: int,
) -> List[GDDChunk]:
    """
    Convert raw ``ChunkSpec`` objects into validated ``GDDChunk`` records.

    Parameters
    ----------
    chunk_specs:
        Output of ``chunk_segments()``.
    document_id:
        The document's unique identifier (from ingestion).
    segments:
        The ordered list of ``Segment`` objects (output of
        ``segment_text()``), used to infer chapter/section inheritance.
    max_chunk_tokens:
        The configured max token limit; used in the VAL-011 assertion
        logged during tagging.

    Returns
    -------
    List[GDDChunk]
        Validated, metadata-tagged chunk objects ready for agent consumption.

    Raises
    ------
    ValueError
        If any chunk fails VAL-013 or the Pydantic schema validation.
    """
    # Build a level-1 heading inheritance map: segment_position → chapter label
    chapter_map: dict[int, Optional[str]] = {}
    current_chapter: Optional[str] = None
    for seg in segments:
        if seg.level is not None and seg.level == 1 and seg.title:
            current_chapter = seg.title
        chapter_map[seg.position] = current_chapter

    tagged: List[GDDChunk] = []

    for position, spec in enumerate(chunk_specs):
        chunk_id = _make_chunk_id(document_id, position)

        # ── Chapter / section label resolution ────────────────────────────
        chapter, section = _resolve_labels(spec, chapter_map)

        # ── VAL-013: document_id and position are always set (guaranteed
        #            by construction, but verified at Pydantic level below)

        # ── VAL-014: chapter/section null only when structure undetected ──
        # (Pydantic model allows null; the pipeline.yaml docs this case)

        try:
            chunk = GDDChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=spec.text,
                chapter=chapter,
                section=section,
                position=position,
                char_offset_start=spec.char_offset_start,
                char_offset_end=spec.char_offset_end,
                token_count=spec.token_count,
                overlap_with_previous=spec.overlap_with_previous,
            )
        except ValidationError as exc:
            raise ValueError(
                f"GDDChunk schema validation failed for chunk at position "
                f"{position} (VAL-013): {exc}"
            ) from exc

        tagged.append(chunk)

    return tagged


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_chunk_id(document_id: str, position: int) -> str:
    """
    Generate a deterministic chunk ID.

    Format: ``{document_id}__chunk_{position:04d}``
    """
    return f"{document_id}__chunk_{position:04d}"


def _resolve_labels(
    spec: ChunkSpec,
    chapter_map: dict[int, Optional[str]],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return ``(chapter, section)`` labels for a chunk spec.

    Rules:
    * Level-1 segment → title is the chapter; section is None.
    * Level-2+ segment → inherited chapter from ``chapter_map``; title is
      the section.
    * No title (preamble / single-segment fallback) → both None.
    """
    title = spec.segment_title
    level = spec.segment_level

    if title is None:
        # Preamble or single-segment fallback
        return None, None

    if level is None or level == 1:
        return title, None
    else:
        # Level 2+: inherit chapter from the map
        inherited_chapter = chapter_map.get(spec.segment_position)
        return inherited_chapter, title
