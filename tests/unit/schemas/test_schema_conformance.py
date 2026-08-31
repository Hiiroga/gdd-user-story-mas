"""
tests/unit/schemas/test_schema_conformance.py
==============================================
Unit tests that validate example payloads against every preprocessing-
relevant schema in ``gdd_userstory_mas/schemas/``.

Tests do NOT require an LLM and run purely from Pydantic validation.

Covers:
  GDDDocument — required fields, format enum, language enum.
  GDDMetadata — required fields, segment_count >= 1, TOCEntry.
  GDDChunk    — required fields, char offset consistency, text not empty.
  ErrorFailureLog — required fields, enum literals.
"""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.gdd_document import GDDDocument
from gdd_userstory_mas.schemas.gdd_metadata import GDDMetadata, TOCEntry


# ── Helpers ────────────────────────────────────────────────────────────────────

NOW = datetime.now(tz=timezone.utc)
DOC_ID = "550e8400-e29b-41d4-a716-446655440000"


def _valid_document(**overrides) -> dict:
    base = {
        "document_id": DOC_ID,
        "file_name": "game_design_doc.txt",
        "format": "txt",
        "language": "en",
        "ingestion_timestamp": NOW,
    }
    base.update(overrides)
    return base


def _valid_metadata(**overrides) -> dict:
    base = {
        "document_id": DOC_ID,
        "structure_detected": True,
        "segment_count": 3,
    }
    base.update(overrides)
    return base


def _valid_chunk(**overrides) -> dict:
    base = {
        "chunk_id": f"{DOC_ID}__chunk_0000",
        "document_id": DOC_ID,
        "text": "The player attacks enemies using a combo system.",
        "position": 0,
        "token_count": 9,
    }
    base.update(overrides)
    return base


def _valid_error_log(**overrides) -> dict:
    base = {
        "error_id": "err-001",
        "stage": "ingestion",
        "error_type": "other",
        "message": "Test error",
        "resolution": "excluded_and_logged",
        "timestamp": NOW,
    }
    base.update(overrides)
    return base


# ── GDDDocument ────────────────────────────────────────────────────────────────

class TestGDDDocument:
    def test_valid_minimal_document(self) -> None:
        doc = GDDDocument(**_valid_document())
        assert doc.document_id == DOC_ID
        assert doc.format == "txt"
        assert doc.language == "en"

    def test_valid_pdf_format(self) -> None:
        doc = GDDDocument(**_valid_document(format="pdf", file_name="gdd.pdf"))
        assert doc.format == "pdf"

    def test_optional_fields_can_be_null(self) -> None:
        doc = GDDDocument(**_valid_document(genre=None, raw_text=None, source_provenance=None))
        assert doc.genre is None
        assert doc.raw_text is None

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValidationError):
            GDDDocument(**_valid_document(format="docx"))

    def test_rejects_invalid_language(self) -> None:
        with pytest.raises(ValidationError):
            GDDDocument(**_valid_document(language="fr"))

    def test_rejects_empty_document_id(self) -> None:
        with pytest.raises(ValidationError):
            GDDDocument(**_valid_document(document_id=""))

    def test_rejects_empty_file_name(self) -> None:
        with pytest.raises(ValidationError):
            GDDDocument(**_valid_document(file_name=""))

    def test_missing_required_fields_raise(self) -> None:
        with pytest.raises(ValidationError):
            GDDDocument(document_id=DOC_ID)  # type: ignore[call-arg]

    def test_serialises_to_json(self) -> None:
        doc = GDDDocument(**_valid_document())
        json_str = doc.model_dump_json()
        assert DOC_ID in json_str


# ── GDDMetadata ────────────────────────────────────────────────────────────────

class TestGDDMetadata:
    def test_valid_minimal_metadata(self) -> None:
        meta = GDDMetadata(**_valid_metadata())
        assert meta.document_id == DOC_ID
        assert meta.structure_detected is True
        assert meta.segment_count == 3

    def test_segment_count_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            GDDMetadata(**_valid_metadata(segment_count=0))

    def test_table_of_contents_is_optional(self) -> None:
        meta = GDDMetadata(**_valid_metadata())
        assert meta.table_of_contents == []

    def test_toc_entry_requires_title_and_position(self) -> None:
        meta = GDDMetadata(**_valid_metadata(
            table_of_contents=[{"title": "Overview", "position": 0, "level": 1}]
        ))
        assert meta.table_of_contents[0].title == "Overview"
        assert meta.table_of_contents[0].level == 1

    def test_toc_entry_rejects_empty_title(self) -> None:
        with pytest.raises(ValidationError):
            GDDMetadata(**_valid_metadata(
                table_of_contents=[{"title": "", "position": 0}]
            ))

    def test_structure_detected_false_is_valid(self) -> None:
        meta = GDDMetadata(**_valid_metadata(structure_detected=False))
        assert meta.structure_detected is False


# ── GDDChunk ───────────────────────────────────────────────────────────────────

class TestGDDChunk:
    def test_valid_minimal_chunk(self) -> None:
        chunk = GDDChunk(**_valid_chunk())
        assert chunk.chunk_id == f"{DOC_ID}__chunk_0000"
        assert chunk.document_id == DOC_ID
        assert chunk.position == 0
        assert chunk.token_count == 9

    def test_optional_fields_can_be_null(self) -> None:
        chunk = GDDChunk(**_valid_chunk(
            chapter=None, section=None,
            char_offset_start=None, char_offset_end=None,
            overlap_with_previous=None,
        ))
        assert chunk.chapter is None
        assert chunk.section is None

    def test_with_full_optional_fields(self) -> None:
        chunk = GDDChunk(**_valid_chunk(
            chapter="Combat System",
            section="Light Attack",
            char_offset_start=100,
            char_offset_end=200,
            overlap_with_previous=True,
        ))
        assert chunk.chapter == "Combat System"
        assert chunk.section == "Light Attack"
        assert chunk.overlap_with_previous is True

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            GDDChunk(**_valid_chunk(text=""))

    def test_rejects_whitespace_only_text(self) -> None:
        with pytest.raises(ValidationError):
            GDDChunk(**_valid_chunk(text="   \n\t  "))

    def test_rejects_negative_position(self) -> None:
        with pytest.raises(ValidationError):
            GDDChunk(**_valid_chunk(position=-1))

    def test_rejects_inconsistent_char_offsets(self) -> None:
        """char_offset_start must be < char_offset_end."""
        with pytest.raises(ValidationError):
            GDDChunk(**_valid_chunk(char_offset_start=100, char_offset_end=50))

    def test_rejects_equal_char_offsets(self) -> None:
        with pytest.raises(ValidationError):
            GDDChunk(**_valid_chunk(char_offset_start=100, char_offset_end=100))

    def test_valid_char_offsets_accepted(self) -> None:
        chunk = GDDChunk(**_valid_chunk(char_offset_start=0, char_offset_end=48))
        assert chunk.char_offset_start == 0
        assert chunk.char_offset_end == 48

    def test_serialises_to_dict(self) -> None:
        chunk = GDDChunk(**_valid_chunk())
        d = chunk.model_dump(mode="json")
        assert d["chunk_id"] == f"{DOC_ID}__chunk_0000"
        assert d["text"] == "The player attacks enemies using a combo system."


# ── ErrorFailureLog ────────────────────────────────────────────────────────────

class TestErrorFailureLog:
    def test_valid_minimal_error_log(self) -> None:
        log = ErrorFailureLog(**_valid_error_log())
        assert log.stage == "ingestion"
        assert log.error_type == "other"
        assert log.resolution == "excluded_and_logged"

    def test_all_stage_values_accepted(self) -> None:
        stages = [
            "ingestion", "pdf_conversion", "text_cleaning", "segmentation",
            "chunking", "metadata_tagging", "gdd_reader", "requirements_analyst",
            "user_story_generator", "reviewer", "redundancy_checker", "evaluator",
            "single_agent_baseline", "orchestration", "result_storage",
        ]
        for stage in stages:
            log = ErrorFailureLog(**_valid_error_log(stage=stage))
            assert log.stage == stage

    def test_all_error_type_values_accepted(self) -> None:
        error_types = [
            "malformed_output", "schema_violation", "llm_api_failure", "llm_timeout",
            "pdf_conversion_failure", "max_iterations_reached_reviewer",
            "max_iterations_reached_redundancy", "missing_source_content",
            "reviewer_contradiction", "other",
        ]
        for et in error_types:
            log = ErrorFailureLog(**_valid_error_log(error_type=et))
            assert log.error_type == et

    def test_all_resolution_values_accepted(self) -> None:
        resolutions = [
            "resolved_via_retry", "excluded_and_logged",
            "run_halted", "flagged_unresolved",
        ]
        for res in resolutions:
            log = ErrorFailureLog(**_valid_error_log(resolution=res))
            assert log.resolution == res

    def test_rejects_invalid_stage(self) -> None:
        with pytest.raises(ValidationError):
            ErrorFailureLog(**_valid_error_log(stage="unknown_stage"))

    def test_rejects_invalid_error_type(self) -> None:
        with pytest.raises(ValidationError):
            ErrorFailureLog(**_valid_error_log(error_type="bad_type"))

    def test_rejects_invalid_resolution(self) -> None:
        with pytest.raises(ValidationError):
            ErrorFailureLog(**_valid_error_log(resolution="silently_ignored"))

    def test_optional_fields_can_be_null(self) -> None:
        log = ErrorFailureLog(**_valid_error_log(
            run_id=None, document_id=None, related_entity_id=None, retry_count=None,
        ))
        assert log.run_id is None

    def test_serialises_to_json_lines(self) -> None:
        log = ErrorFailureLog(**_valid_error_log())
        json_str = log.model_dump_json()
        assert '"ingestion"' in json_str
        assert "Test error" in json_str
