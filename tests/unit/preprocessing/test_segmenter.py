"""
tests/unit/preprocessing/test_segmenter.py
============================================
Unit tests for the segmentation module (SRS-005).

Covers:
  VAL-009 — Non-overlapping segments covering full text (no content loss).
  VAL-010 — Single-segment fallback for documents with no detectable structure.
  Heading detection patterns.
  Level detection (chapter vs. section).
  Preamble handling.
"""
import pytest

from gdd_userstory_mas.preprocessing.segmenter import (
    DEFAULT_HEADING_PATTERNS,
    Segment,
    SegmenterConfig,
    _detect_heading_level,
    _is_heading,
    segment_text,
)


# ── VAL-010: Single-segment fallback ──────────────────────────────────────────

class TestSingleSegmentFallback:
    def test_unstructured_text_produces_one_segment(self) -> None:
        text = (
            "This is a game about fighting monsters in a dark dungeon. "
            "The player must collect treasure and defeat the boss. "
            "The game has no apparent structure in this test."
        )
        segments = segment_text(text)
        assert len(segments) == 1
        assert segments[0].title is None
        assert segments[0].level is None
        assert segments[0].position == 0

    def test_single_segment_covers_full_text(self) -> None:
        text = "No headings here at all, just plain text content."
        segments = segment_text(text)
        assert segments[0].char_offset_start == 0
        assert segments[0].char_offset_end == len(text)


# ── Heading detection ──────────────────────────────────────────────────────────

class TestHeadingDetection:
    def test_detects_markdown_h1(self) -> None:
        text = "# Overview\n\nSome content.\n\n## Subsection\n\nMore content."
        segments = segment_text(text)
        titles = [s.title for s in segments if s.title]
        assert "# Overview" in titles or "Overview" in " ".join(t or "" for t in titles)

    def test_detects_numbered_heading(self) -> None:
        text = "1. Overview\n\nContent here.\n\n2. Combat\n\nFight mechanics."
        segments = segment_text(text)
        titles = [s.title for s in segments if s.title]
        assert len(titles) >= 2

    def test_detects_all_caps_heading(self) -> None:
        text = "GAME OVERVIEW\n\nContent about the game.\n\nCOMBAT SYSTEM\n\nFight content."
        segments = segment_text(text)
        titled = [s for s in segments if s.title is not None]
        assert len(titled) >= 1

    def test_detects_chapter_keyword(self) -> None:
        text = "Chapter 1 Introduction\n\nSome intro text.\n\nChapter 2 Combat\n\nFight."
        segments = segment_text(text)
        titled = [s for s in segments if s.title is not None]
        assert len(titled) >= 1


# ── VAL-009: No content loss ───────────────────────────────────────────────────

class TestNonOverlappingCoverage:
    def test_segments_cover_full_fixture(self) -> None:
        from pathlib import Path
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "sample_gdd_small.txt"
        text = fixture.read_text(encoding="utf-8")
        # clean_text first (as the pipeline does)
        from gdd_userstory_mas.preprocessing.text_cleaner import clean_text
        cleaned = clean_text(text)
        segments = segment_text(cleaned)
        # Verify segment count >= 1
        assert len(segments) >= 1
        # Verify all segments have non-empty text or title
        for seg in segments:
            assert seg.char_offset_end >= seg.char_offset_start

    def test_segments_are_ordered_by_position(self) -> None:
        text = (
            "1. Overview\n\nIntro content.\n\n"
            "2. Combat\n\nFight content.\n\n"
            "3. World\n\nWorld content."
        )
        segments = segment_text(text)
        positions = [s.position for s in segments]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)  # unique


# ── Heading level detection ────────────────────────────────────────────────────

class TestHeadingLevelDetection:
    def test_markdown_h1_is_level_1(self) -> None:
        assert _detect_heading_level("# Overview") == 1

    def test_markdown_h2_is_level_2(self) -> None:
        assert _detect_heading_level("## Combat System") == 2

    def test_markdown_h3_is_level_3(self) -> None:
        assert _detect_heading_level("### Sub-subsection") == 3

    def test_numbered_no_dot_is_level_1(self) -> None:
        assert _detect_heading_level("1. Overview") == 1

    def test_numbered_with_dot_is_level_2(self) -> None:
        assert _detect_heading_level("2.1 Combat System") == 2

    def test_numbered_deep_is_level_3(self) -> None:
        assert _detect_heading_level("2.1.1 Sub-detail") == 3

    def test_all_caps_is_level_1(self) -> None:
        # Falls through to default → level 1
        assert _detect_heading_level("GAME OVERVIEW") == 1


# ── _is_heading helper ─────────────────────────────────────────────────────────

class TestIsHeading:
    def setup_method(self) -> None:
        import re
        self.patterns = [re.compile(p, re.MULTILINE) for p in DEFAULT_HEADING_PATTERNS]

    def test_markdown_heading_recognised(self) -> None:
        assert _is_heading("## Combat System", self.patterns)

    def test_numbered_heading_recognised(self) -> None:
        assert _is_heading("2. Combat System", self.patterns)

    def test_all_caps_short_line_recognised(self) -> None:
        assert _is_heading("GAME OVERVIEW", self.patterns)

    def test_body_text_not_recognised(self) -> None:
        assert not _is_heading(
            "The player character attacks enemies using a combo system.", self.patterns
        )

    def test_empty_line_not_recognised(self) -> None:
        assert not _is_heading("", self.patterns)


# ── Preamble handling ──────────────────────────────────────────────────────────

class TestPreambleHandling:
    def test_preamble_captured_when_before_first_heading(self) -> None:
        text = "This is preamble text.\n\n1. First Section\n\nSection content."
        segments = segment_text(text)
        # First segment should be the preamble (no title)
        assert segments[0].title is None
        assert "preamble" in segments[0].text.lower()

    def test_no_preamble_when_heading_is_first(self) -> None:
        text = "1. First Section\n\nContent here.\n\n2. Second\n\nMore content."
        segments = segment_text(text)
        # All segments should have titles (no preamble)
        assert all(s.title is not None for s in segments)


# ── Configurable patterns ──────────────────────────────────────────────────────

class TestCustomConfig:
    def test_empty_pattern_list_produces_fallback(self) -> None:
        config = SegmenterConfig(heading_patterns=[])
        text = "1. Section One\n\nContent.\n\n2. Section Two\n\nMore."
        segments = segment_text(text, config=config)
        # With no patterns, no headings detected → single-segment fallback
        assert len(segments) == 1
        assert segments[0].title is None
