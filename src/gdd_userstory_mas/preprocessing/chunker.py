"""
chunker.py — SRS-006
======================
Context-aware chunking: split segments into LLM-context-compliant chunks.

Validation rules:
  VAL-011 — No chunk shall exceed the configured maximum token size.
  VAL-012 — Chunk splitting shall avoid breaking mid-sentence where
             technically feasible.

Design notes
------------
* Token counting uses ``tiktoken`` with the ``cl100k_base`` encoding
  (GPT-4 family approximation).  This is a proxy — the exact count
  depends on the final LLM choice (OPEN-002).  The encoding can be
  changed via ``ChunkerConfig.tiktoken_encoding``.

* Sentence splitting uses ``nltk.sent_tokenize`` (requires the
  ``punkt_tab`` corpus; downloaded lazily on first use).

* Overlap (OPEN-013):
  When ``overlap_tokens > 0``, the last ``overlap_tokens`` worth of
  sentences from chunk N are prepended to chunk N+1.  The
  ``GDDChunk.overlap_with_previous`` field is set accordingly.

* Splitting strategy:
  1. If a segment fits within ``max_chunk_tokens`` → one chunk.
  2. Otherwise, split segment into sentences.
  3. Greedily accumulate sentences into a chunk until the next sentence
     would exceed the limit → emit the chunk and start a new one.
  4. If a single sentence exceeds the limit → emit it as its own chunk
     (cannot split further without breaking semantic units; logged as a
     warning).

* Determinism: for a fixed input and configuration, the output is fully
  deterministic (no random elements).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# tiktoken is optional at import so tests can mock it
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False

# nltk is optional at import so tests can mock it
try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NLTK_AVAILABLE = False

from gdd_userstory_mas.preprocessing.segmenter import Segment

logger = logging.getLogger(__name__)

# ── ChunkSpec dataclass ────────────────────────────────────────────────────────

@dataclass
class ChunkSpec:
    """
    Intermediate representation of a chunk produced by the chunker,
    before metadata is attached by ``metadata_tagger``.
    """

    text: str
    """The chunk's text content."""

    segment_position: int
    """0-based index of the parent segment."""

    segment_title: Optional[str]
    """Heading of the parent segment (None for preamble)."""

    segment_level: Optional[int]
    """Hierarchy level of the parent segment heading."""

    token_count: int
    """Approximate token count (via tiktoken)."""

    char_offset_start: int
    """Character offset in the cleaned document text."""

    char_offset_end: int
    """Character offset end (exclusive)."""

    overlap_with_previous: Optional[bool] = None
    """True if this chunk shares content with the preceding chunk."""


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class ChunkerConfig:
    """
    Configuration for the chunker.

    All defaults match ``config/pipeline.yaml``.
    """

    max_chunk_tokens: int = 1500
    """Maximum tokens per chunk (VAL-011)."""

    overlap_tokens: int = 0
    """Tokens of trailing context repeated at the start of the next chunk."""

    tiktoken_encoding: str = "cl100k_base"
    """tiktoken encoding name to use for token counting."""

    sentence_splitter: str = "nltk"
    """Sentence splitting backend.  Only 'nltk' is currently supported."""


# ── Public function ────────────────────────────────────────────────────────────

def chunk_segments(
    segments: List[Segment],
    *,
    config: Optional[ChunkerConfig] = None,
    cleaned_text: str = "",
) -> List[ChunkSpec]:
    """
    Split a list of segments into token-limited chunks.

    Parameters
    ----------
    segments:
        Output of ``segment_text()``.
    config:
        Chunker configuration.  ``None`` uses defaults.
    cleaned_text:
        The full cleaned document text.  Used to compute accurate
        ``char_offset_*`` values.  May be empty (offsets will be
        approximated from segment data).

    Returns
    -------
    List[ChunkSpec]
        Ordered list of chunks, each within ``config.max_chunk_tokens``.

    Raises
    ------
    ValueError
        If any chunk exceeds the token limit after splitting (indicates a
        bug in the splitting logic).
    """
    if config is None:
        config = ChunkerConfig()

    encoder = _get_encoder(config.tiktoken_encoding)
    splitter = _get_splitter(config.sentence_splitter)

    chunks: List[ChunkSpec] = []
    overlap_carry: List[str] = []  # sentences carried from the previous chunk

    for seg in segments:
        seg_text = seg.text
        if not seg_text.strip():
            continue  # skip empty segments (e.g., heading-only with no body)

        seg_chunks = _chunk_one_segment(
            seg=seg,
            encoder=encoder,
            splitter=splitter,
            config=config,
            overlap_carry=overlap_carry,
        )
        chunks.extend(seg_chunks)

        # Update overlap carry for the next segment boundary
        if config.overlap_tokens > 0 and seg_chunks:
            last_chunk_text = seg_chunks[-1].text
            overlap_carry = _build_overlap_carry(
                last_chunk_text, encoder, splitter, config.overlap_tokens
            )
        else:
            overlap_carry = []

    # ── VAL-011 post-condition check ───────────────────────────────────────
    for chunk in chunks:
        if chunk.token_count > config.max_chunk_tokens:
            raise ValueError(
                f"VAL-011 violation: chunk exceeds max_chunk_tokens "
                f"({chunk.token_count} > {config.max_chunk_tokens}): "
                f"{chunk.text[:80]!r}"
            )

    return chunks


# ── Internal helpers ───────────────────────────────────────────────────────────

def _chunk_one_segment(
    seg: Segment,
    encoder: object,
    splitter: object,
    config: ChunkerConfig,
    overlap_carry: List[str],
) -> List[ChunkSpec]:
    """
    Produce one or more ``ChunkSpec`` objects from a single segment.
    """
    full_text = seg.text
    token_count = _count_tokens(full_text, encoder)

    # Fast path: segment fits in one chunk (common case)
    if token_count <= config.max_chunk_tokens and not overlap_carry:
        return [
            ChunkSpec(
                text=full_text,
                segment_position=seg.position,
                segment_title=seg.title,
                segment_level=seg.level,
                token_count=token_count,
                char_offset_start=seg.char_offset_start,
                char_offset_end=seg.char_offset_end,
                overlap_with_previous=False,
            )
        ]

    # Split into sentences
    sentences: List[str] = _split_sentences(full_text, splitter)
    if not sentences:
        return []

    chunks: List[ChunkSpec] = []
    current_sentences: List[str] = list(overlap_carry)  # start with overlap carry
    is_first = True
    char_cursor = seg.char_offset_start

    for sentence in sentences:
        probe = current_sentences + [sentence]
        probe_text = " ".join(probe)
        probe_tokens = _count_tokens(probe_text, encoder)

        if probe_tokens <= config.max_chunk_tokens:
            current_sentences.append(sentence)
        else:
            # Emit the current batch if non-empty
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                ct = _count_tokens(chunk_text, encoder)
                char_end = char_cursor + len(chunk_text)
                chunks.append(
                    ChunkSpec(
                        text=chunk_text,
                        segment_position=seg.position,
                        segment_title=seg.title,
                        segment_level=seg.level,
                        token_count=ct,
                        char_offset_start=char_cursor,
                        char_offset_end=char_end,
                        overlap_with_previous=(not is_first and bool(overlap_carry)),
                    )
                )
                char_cursor = char_end
                is_first = False

            # Build overlap carry from what we just emitted
            if config.overlap_tokens > 0:
                current_sentences = _build_overlap_carry(
                    " ".join(current_sentences), encoder, splitter,
                    config.overlap_tokens
                )
            else:
                current_sentences = []

            # Handle oversized single sentence (VAL-012: best-effort)
            single_tokens = _count_tokens(sentence, encoder)
            if single_tokens > config.max_chunk_tokens:
                logger.warning(
                    "VAL-012: single sentence exceeds max_chunk_tokens "
                    "(%d > %d); emitting as its own chunk.",
                    single_tokens, config.max_chunk_tokens,
                )
                char_end = char_cursor + len(sentence)
                chunks.append(
                    ChunkSpec(
                        text=sentence,
                        segment_position=seg.position,
                        segment_title=seg.title,
                        segment_level=seg.level,
                        token_count=single_tokens,
                        char_offset_start=char_cursor,
                        char_offset_end=char_end,
                        overlap_with_previous=False,
                    )
                )
                char_cursor = char_end
                is_first = False
            else:
                current_sentences.append(sentence)

    # Emit any remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        ct = _count_tokens(chunk_text, encoder)
        char_end = char_cursor + len(chunk_text)
        chunks.append(
            ChunkSpec(
                text=chunk_text,
                segment_position=seg.position,
                segment_title=seg.title,
                segment_level=seg.level,
                token_count=ct,
                char_offset_start=char_cursor,
                char_offset_end=char_end,
                overlap_with_previous=(not is_first and bool(overlap_carry)),
            )
        )

    return chunks


def _build_overlap_carry(
    text: str,
    encoder: object,
    splitter: object,
    overlap_tokens: int,
) -> List[str]:
    """
    Build the list of sentences from the END of ``text`` that together
    fill ``overlap_tokens`` or fewer tokens.
    """
    sentences = _split_sentences(text, splitter)
    carry: List[str] = []
    total = 0
    for s in reversed(sentences):
        s_tokens = _count_tokens(s, encoder)
        if total + s_tokens <= overlap_tokens:
            carry.insert(0, s)
            total += s_tokens
        else:
            break
    return carry


def _count_tokens(text: str, encoder: object) -> int:
    """Return the number of tokens in ``text`` using ``encoder``."""
    if encoder is None:
        # Fallback: rough word-based estimate (for testing without tiktoken)
        return len(text.split())
    try:
        return len(encoder.encode(text))  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return len(text.split())  # fallback


def _split_sentences(text: str, splitter: object) -> List[str]:
    """Split ``text`` into sentences using ``splitter``."""
    if splitter is None:
        # Fallback: split on period-space (for testing without nltk)
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
    try:
        sentences = splitter(text)  # type: ignore[operator]
        return [s.strip() for s in sentences if s.strip()]
    except Exception:  # noqa: BLE001
        return [text.strip()] if text.strip() else []


def _get_encoder(encoding_name: str) -> Optional[object]:
    """Load and return a tiktoken encoder, or None if unavailable."""
    if not _TIKTOKEN_AVAILABLE:
        logger.warning(
            "tiktoken not available. Using word-count approximation for token counting."
        )
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001  pragma: no cover
        logger.warning("Failed to load tiktoken encoding '%s': %s", encoding_name, exc)
        return None


def _get_splitter(splitter_name: str) -> Optional[object]:
    """Return the sentence splitter callable, or None if unavailable."""
    if splitter_name == "nltk":
        if not _NLTK_AVAILABLE:
            logger.warning(
                "nltk not available. Using regex sentence splitting fallback."
            )
            return None
        # Ensure punkt is downloaded
        try:
            _nltk_sent_tokenize("test")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to download nltk punkt_tab data.")
        return _nltk_sent_tokenize
    logger.warning("Unknown sentence_splitter '%s'. Using fallback.", splitter_name)
    return None
