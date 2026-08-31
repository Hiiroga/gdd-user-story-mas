"""
tests/unit/preprocessing/test_chunker.py
==========================================
Unit tests for the chunker module (SRS-006).

All tests use a mocked tokeniser and sentence splitter so they run
without tiktoken or NLTK installed.  Integration-level tests that use
real libraries are marked with ``@pytest.mark.integration``.

Covers:
  VAL-011 — No chunk exceeds configured max_chunk_tokens.
  VAL-012 — Splitting avoids mid-sentence breaks where feasible.
  Overlap token carry-forward (OPEN-013).
  Deterministic output.
  Oversized single-sentence handling.
"""
import pytest
from unittest.mock import patch

from gdd_userstory_mas.preprocessing.chunker import (
    ChunkSpec,
    ChunkerConfig,
    _build_overlap_carry,
    _count_tokens,
    _split_sentences,
    chunk_segments,
)
from gdd_userstory_mas.preprocessing.segmenter import Segment


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_segment(
    text: str,
    position: int = 0,
    title: str = "Test Section",
    level: int = 1,
    start: int = 0,
) -> Segment:
    return Segment(
        title=title,
        level=level,
        position=position,
        text=text,
        char_offset_start=start,
        char_offset_end=start + len(text),
    )


def _word_count_encoder(text: str) -> int:
    """Simple word-count approximation used in tests (replaces tiktoken)."""
    return len(text.split())


def _simple_splitter(text: str):
    """Split on period-space for deterministic sentence splitting in tests."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ── VAL-011: Token limit enforcement ──────────────────────────────────────────

class TestTokenLimitEnforcement:
    def test_single_chunk_when_within_limit(self) -> None:
        """A short segment produces exactly one chunk."""
        text = "The player attacks enemies. The enemy defends."
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=100, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch("gdd_userstory_mas.preprocessing.chunker._get_splitter", return_value=None),
        ):
            chunks = chunk_segments([seg], config=config)

        assert len(chunks) == 1
        assert chunks[0].token_count <= config.max_chunk_tokens

    def test_splits_when_segment_exceeds_limit(self) -> None:
        """A long segment is split into multiple chunks."""
        # Build a long segment: 50 words per sentence × 5 sentences = 250 words
        sentences = [
            "Word " * 50 + "sentence end."
        ] * 5
        text = " ".join(sentences)
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=60, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks = chunk_segments([seg], config=config)

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should be at most max_chunk_tokens
            # (oversized single sentence exception aside)
            assert chunk.token_count <= config.max_chunk_tokens or len(chunk.text.split(".")) <= 1

    def test_all_chunks_within_token_limit(self) -> None:
        """VAL-011: post-condition asserted on all output chunks."""
        sentences = ["This is sentence number %d." % i for i in range(20)]
        text = " ".join(sentences)
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=10, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks = chunk_segments([seg], config=config)

        assert len(chunks) > 0
        for chunk in chunks:
            # All chunks must satisfy the limit
            assert chunk.token_count <= config.max_chunk_tokens


# ── VAL-012: Sentence boundary respect ────────────────────────────────────────

class TestSentenceBoundaryRespect:
    def test_chunks_end_at_sentence_boundaries(self) -> None:
        """
        Chunks should not cut a sentence mid-word.
        We verify by checking that each chunk's text ends with sentence-terminal
        punctuation or is the last chunk.
        """
        sentences = [
            "The player can dodge incoming attacks.",
            "Enemies adapt to player behaviour over time.",
            "The inventory system uses a grid layout.",
            "Players can compare equipment in real time.",
        ]
        text = " ".join(sentences)
        seg = _make_segment(text)
        # Limit to ~8 words per chunk (forces splitting)
        config = ChunkerConfig(max_chunk_tokens=8, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks = chunk_segments([seg], config=config)

        for chunk in chunks[:-1]:
            # Non-final chunks should end at a sentence boundary
            assert chunk.text.strip().endswith((".", "?", "!"))


# ── Overlap ────────────────────────────────────────────────────────────────────

class TestOverlapCarry:
    def test_overlap_with_previous_is_false_for_first_chunk(self) -> None:
        text = " ".join(["Sentence %d." % i for i in range(10)])
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=5, overlap_tokens=3)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks = chunk_segments([seg], config=config)

        assert chunks[0].overlap_with_previous is False or chunks[0].overlap_with_previous is None

    def test_no_overlap_flag_when_overlap_tokens_is_zero(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=4, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks = chunk_segments([seg], config=config)

        # No chunk should be flagged as overlapping when overlap=0
        for chunk in chunks:
            assert not chunk.overlap_with_previous


# ── Empty segment handling ─────────────────────────────────────────────────────

class TestEmptySegments:
    def test_empty_segment_text_is_skipped(self) -> None:
        seg = _make_segment("")
        config = ChunkerConfig(max_chunk_tokens=100, overlap_tokens=0)
        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch("gdd_userstory_mas.preprocessing.chunker._get_splitter", return_value=None),
        ):
            chunks = chunk_segments([seg], config=config)
        assert chunks == []


# ── Determinism ────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        text = " ".join(["Sentence number %d." % i for i in range(15)])
        seg = _make_segment(text)
        config = ChunkerConfig(max_chunk_tokens=8, overlap_tokens=0)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch(
                "gdd_userstory_mas.preprocessing.chunker._get_splitter",
                return_value=_simple_splitter,
            ),
        ):
            chunks_1 = chunk_segments([seg], config=config)
            chunks_2 = chunk_segments([seg], config=config)

        assert len(chunks_1) == len(chunks_2)
        for c1, c2 in zip(chunks_1, chunks_2):
            assert c1.text == c2.text
            assert c1.token_count == c2.token_count


# ── ChunkSpec fields ───────────────────────────────────────────────────────────

class TestChunkSpecFields:
    def test_chunk_carries_segment_metadata(self) -> None:
        seg = _make_segment("Content sentence here.", title="Combat System", level=1)
        config = ChunkerConfig(max_chunk_tokens=100)

        with (
            patch("gdd_userstory_mas.preprocessing.chunker._get_encoder", return_value=None),
            patch("gdd_userstory_mas.preprocessing.chunker._get_splitter", return_value=None),
        ):
            chunks = chunk_segments([seg], config=config)

        assert chunks[0].segment_title == "Combat System"
        assert chunks[0].segment_level == 1
        assert chunks[0].segment_position == 0


# ── _count_tokens fallback ─────────────────────────────────────────────────────

class TestCountTokensFallback:
    def test_falls_back_to_word_count_when_encoder_is_none(self) -> None:
        # str.split() on this text gives 8 tokens: the period attaches to "system."
        text = "The player attacks enemies using a combo system."
        count = _count_tokens(text, None)
        # Word-count fallback: len("The player attacks enemies using a combo system.".split()) == 8
        assert count == len(text.split())


# ── _split_sentences fallback ──────────────────────────────────────────────────

class TestSplitSentencesFallback:
    def test_fallback_splits_on_punctuation(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_sentences(text, None)
        assert len(parts) >= 2

    def test_returns_empty_list_for_empty_string(self) -> None:
        parts = _split_sentences("", None)
        assert parts == []


# ── _build_overlap_carry ───────────────────────────────────────────────────────

class TestBuildOverlapCarry:
    def test_carry_respects_token_budget(self) -> None:
        text = "Short. Medium length sentence. Another sentence here."
        carry = _build_overlap_carry(text, None, _simple_splitter, overlap_tokens=5)
        # Carry sentences together should be <= 5 words
        total_words = sum(len(s.split()) for s in carry)
        assert total_words <= 5

    def test_carry_is_empty_when_zero_budget(self) -> None:
        text = "First sentence. Second sentence."
        carry = _build_overlap_carry(text, None, _simple_splitter, overlap_tokens=0)
        assert carry == []
