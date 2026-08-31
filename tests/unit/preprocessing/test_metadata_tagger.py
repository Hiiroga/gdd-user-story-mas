"""
tests/unit/preprocessing/test_metadata_tagger.py
==================================================
Unit tests for the metadata tagger module (SRS-007).

Covers:
  VAL-013 — Every chunk has a non-null document_id and position index.
  VAL-014 — chapter/section null only when no structure detected.
  Chunk ID determinism.
  Chapter / section label inheritance.
  Pydantic validation pass-through.
"""
import pytest

from gdd_userstory_mas.preprocessing.chunker import ChunkSpec
from gdd_userstory_mas.preprocessing.metadata_tagger import (
    _make_chunk_id,
    _resolve_labels,
    tag_metadata,
)
from gdd_userstory_mas.preprocessing.segmenter import Segment
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk


# ── Helpers ────────────────────────────────────────────────────────────────────

DOC_ID = "test-document-001"


def _make_spec(
    text: str = "Sample text content.",
    segment_position: int = 0,
    segment_title: str = "Test Section",
    segment_level: int = 1,
    token_count: int = 3,
    start: int = 0,
) -> ChunkSpec:
    end = start + len(text)
    return ChunkSpec(
        text=text,
        segment_position=segment_position,
        segment_title=segment_title,
        segment_level=segment_level,
        token_count=token_count,
        char_offset_start=start,
        char_offset_end=end,
        overlap_with_previous=False,
    )


def _make_seg(
    title=None, level=None, position=0, text="content", start=0
) -> Segment:
    return Segment(
        title=title,
        level=level,
        position=position,
        text=text,
        char_offset_start=start,
        char_offset_end=start + len(text),
    )


# ── VAL-013: Required fields ───────────────────────────────────────────────────

class TestRequiredFields:
    def test_document_id_is_propagated(self) -> None:
        spec = _make_spec()
        seg = _make_seg(title="Overview", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)

        assert chunks[0].document_id == DOC_ID

    def test_position_is_set_sequentially(self) -> None:
        specs = [_make_spec(text=f"Content {i}.", segment_position=i) for i in range(3)]
        segs = [_make_seg(title=f"Section {i}", level=1, position=i) for i in range(3)]
        chunks = tag_metadata(specs, DOC_ID, segs, max_chunk_tokens=100)

        assert [c.position for c in chunks] == [0, 1, 2]

    def test_returns_gdd_chunk_objects(self) -> None:
        spec = _make_spec()
        seg = _make_seg(title="Overview", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)

        assert all(isinstance(c, GDDChunk) for c in chunks)


# ── Chunk ID determinism ───────────────────────────────────────────────────────

class TestChunkIdDeterminism:
    def test_chunk_id_format(self) -> None:
        chunk_id = _make_chunk_id(DOC_ID, 0)
        assert chunk_id == f"{DOC_ID}__chunk_0000"

    def test_chunk_id_zero_padded(self) -> None:
        assert _make_chunk_id("doc", 5) == "doc__chunk_0005"
        assert _make_chunk_id("doc", 42) == "doc__chunk_0042"
        assert _make_chunk_id("doc", 9999) == "doc__chunk_9999"

    def test_chunk_ids_are_unique_for_different_positions(self) -> None:
        ids = [_make_chunk_id(DOC_ID, i) for i in range(10)]
        assert len(set(ids)) == 10

    def test_chunk_id_in_output(self) -> None:
        spec = _make_spec(segment_position=0)
        seg = _make_seg(title="Overview", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        expected_id = f"{DOC_ID}__chunk_0000"
        assert chunks[0].chunk_id == expected_id


# ── VAL-014: Chapter / section label inheritance ───────────────────────────────

class TestLabelInheritance:
    def test_level1_segment_sets_chapter_only(self) -> None:
        spec = _make_spec(segment_title="Combat System", segment_level=1)
        seg = _make_seg(title="Combat System", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        assert chunks[0].chapter == "Combat System"
        assert chunks[0].section is None

    def test_level2_segment_sets_section(self) -> None:
        """Level-2 chunk inherits chapter from the most recent level-1 segment."""
        segs = [
            _make_seg(title="Combat System", level=1, position=0, text="Fight."),
            _make_seg(title="Light Attack", level=2, position=1, text="Fast strike."),
        ]
        specs = [
            _make_spec(text="Fight.", segment_title="Combat System", segment_level=1, segment_position=0),
            _make_spec(text="Fast strike.", segment_title="Light Attack", segment_level=2, segment_position=1),
        ]
        chunks = tag_metadata(specs, DOC_ID, segs, max_chunk_tokens=100)
        # Second chunk: should inherit "Combat System" as chapter
        assert chunks[1].chapter == "Combat System"
        assert chunks[1].section == "Light Attack"

    def test_preamble_has_null_labels(self) -> None:
        spec = _make_spec(segment_title=None, segment_level=None)
        seg = _make_seg(title=None, level=None, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        # VAL-014: null only when no structure detected
        assert chunks[0].chapter is None
        assert chunks[0].section is None

    def test_chapter_inherited_across_level2_chunks(self) -> None:
        """Multiple level-2 sections under the same level-1 chapter all inherit it."""
        segs = [
            _make_seg(title="World Design", level=1, position=0, text="Overview."),
            _make_seg(title="Hub World", level=2, position=1, text="The city."),
            _make_seg(title="Regions", level=2, position=2, text="Five regions."),
        ]
        specs = [
            _make_spec(text="Overview.", segment_title="World Design", segment_level=1, segment_position=0),
            _make_spec(text="The city.", segment_title="Hub World", segment_level=2, segment_position=1),
            _make_spec(text="Five regions.", segment_title="Regions", segment_level=2, segment_position=2),
        ]
        chunks = tag_metadata(specs, DOC_ID, segs, max_chunk_tokens=100)
        assert chunks[1].chapter == "World Design"
        assert chunks[2].chapter == "World Design"


# ── _resolve_labels helper ─────────────────────────────────────────────────────

class TestResolveLabels:
    def test_no_title_returns_none_none(self) -> None:
        spec = _make_spec(segment_title=None, segment_level=None)
        chapter_map: dict = {}
        chapter, section = _resolve_labels(spec, chapter_map)
        assert chapter is None
        assert section is None

    def test_level1_returns_title_as_chapter(self) -> None:
        spec = _make_spec(segment_title="Overview", segment_level=1)
        chapter_map: dict = {0: "Overview"}
        chapter, section = _resolve_labels(spec, chapter_map)
        assert chapter == "Overview"
        assert section is None

    def test_level2_returns_inherited_chapter_and_section(self) -> None:
        spec = _make_spec(segment_title="Combat Details", segment_level=2, segment_position=1)
        chapter_map: dict = {1: "Combat System"}
        chapter, section = _resolve_labels(spec, chapter_map)
        assert chapter == "Combat System"
        assert section == "Combat Details"


# ── Offset and token count preservation ───────────────────────────────────────

class TestOffsetPreservation:
    def test_char_offsets_are_preserved(self) -> None:
        spec = _make_spec(text="Hello world.", start=42)
        seg = _make_seg(title="Section", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        assert chunks[0].char_offset_start == 42
        assert chunks[0].char_offset_end == 42 + len("Hello world.")

    def test_token_count_is_preserved(self) -> None:
        spec = _make_spec(token_count=7)
        seg = _make_seg(title="Section", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        assert chunks[0].token_count == 7

    def test_overlap_with_previous_is_preserved(self) -> None:
        spec = ChunkSpec(
            text="Overlap text here.",
            segment_position=0,
            segment_title="Section",
            segment_level=1,
            token_count=3,
            char_offset_start=0,
            char_offset_end=18,
            overlap_with_previous=True,
        )
        seg = _make_seg(title="Section", level=1, position=0)
        chunks = tag_metadata([spec], DOC_ID, [seg], max_chunk_tokens=100)
        assert chunks[0].overlap_with_previous is True
