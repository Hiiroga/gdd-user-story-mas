"""
tests/unit/baseline/test_output_validator.py
=============================================
Unit tests for validate_baseline_output — no LLM required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from gdd_userstory_mas.baseline.output_validator import validate_baseline_output
from gdd_userstory_mas.schemas.final_user_story import ConfidenceEvidence, FinalUserStory
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk

NOW = datetime.now(tz=timezone.utc)
DOC_ID = "doc-001"
RUN_ID = "run-001"
CHUNK_ID = "doc-001__chunk_0000"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str = CHUNK_ID) -> GDDChunk:
    return GDDChunk(
        chunk_id=chunk_id,
        document_id=DOC_ID,
        text="The player can jump to reach higher platforms.",
        position=0,
        token_count=10,
    )


def _make_story(
    id: str = "US-0001",
    chunk_id: str = CHUNK_ID,
    **overrides,
) -> FinalUserStory:
    base = dict(
        id=id,
        run_id=RUN_ID,
        pipeline_type="Baseline",
        source_document_id=DOC_ID,
        source_section="Gameplay",
        source_chunk_ids=[chunk_id],
        role="player",
        action="jump to reach high platforms",
        benefit="explore previously inaccessible areas",
        full_text="As a player, I want to jump to reach high platforms, so that I can explore previously inaccessible areas.",
        requirement_type="player",
        game_domain="gameplay",
        validation_status="Unreviewed",
        validation_feedback=[],
        merge_history=None,
        confidence_evidence=ConfidenceEvidence(source_excerpt="player can jump", reviewer_traceability_confirmed=None),
        iteration_count_final=None,
    )
    base.update(overrides)
    return FinalUserStory(**base)


# ── Happy path ─────────────────────────────────────────────────────────────────

class TestValidBaselineOutput:
    def test_valid_single_story_passes(self) -> None:
        report = validate_baseline_output([_make_story()], [_make_chunk()])
        assert report.passed
        assert report.valid_count == 1
        assert report.invalid_count == 0

    def test_multiple_unique_stories_pass(self) -> None:
        chunks = [_make_chunk(f"doc-001__chunk_{i:04d}") for i in range(3)]
        stories = [
            _make_story(id=f"US-{i+1:04d}", chunk_id=f"doc-001__chunk_{i:04d}")
            for i in range(3)
        ]
        report = validate_baseline_output(stories, chunks)
        assert report.passed
        assert report.valid_count == 3

    def test_empty_story_list_passes_with_warning(self) -> None:
        report = validate_baseline_output([], [_make_chunk()])
        assert report.passed  # no errors, only warning
        assert any("Zero" in w or "No user stories" in w for w in report.warnings)

    def test_empty_story_and_empty_chunks_no_warning(self) -> None:
        report = validate_baseline_output([], [])
        assert report.passed
        assert len(report.warnings) == 0


# ── VAL-031: pipeline_type ─────────────────────────────────────────────────────

class TestPipelineTypeValidation:
    def test_mas_pipeline_type_fails(self) -> None:
        story = _make_story(pipeline_type="MAS")
        report = validate_baseline_output([story], [_make_chunk()])
        assert not report.passed
        assert any("VAL-031" in e for e in report.errors)


# ── VAL-032: validation_status ────────────────────────────────────────────────

class TestValidationStatusValidation:
    def test_valid_status_fails(self) -> None:
        story = _make_story(validation_status="Valid")
        report = validate_baseline_output([story], [_make_chunk()])
        assert not report.passed
        assert any("VAL-032" in e for e in report.errors)


# ── VAL-032: iteration_count_final ───────────────────────────────────────────

class TestIterationCountValidation:
    def test_non_null_iteration_count_raises_in_schema(self) -> None:
        """iteration_count_final=1 must be rejected by Pydantic (VAL-032)."""
        with pytest.raises(Exception):  # pydantic ValidationError
            _make_story(iteration_count_final=1)


# ── VAL-020: full_text format ─────────────────────────────────────────────────

class TestFullTextFormat:
    def test_full_text_not_starting_with_as_a_fails(self) -> None:
        # Pydantic validator enforces this; force it through model_construct
        story = FinalUserStory.model_construct(
            id="US-0001",
            run_id=RUN_ID,
            pipeline_type="Baseline",
            source_document_id=DOC_ID,
            source_section=None,
            source_chunk_ids=[CHUNK_ID],
            role="player",
            action="jump",
            benefit="reach high",
            full_text="Player wants to jump so that they reach high.",  # bad format
            requirement_type="player",
            game_domain="gameplay",
            validation_status="Unreviewed",
            validation_feedback=[],
            merge_history=None,
            confidence_evidence=None,
            iteration_count_final=None,
        )
        report = validate_baseline_output([story], [_make_chunk()])
        assert not report.passed
        assert any("VAL-020" in e for e in report.errors)


# ── TRACE-001: source_chunk_ids ───────────────────────────────────────────────

class TestSourceChunkIds:
    def test_unknown_chunk_id_fails(self) -> None:
        story = _make_story(chunk_id="doc-001__chunk_9999")  # not in chunks list
        report = validate_baseline_output([story], [_make_chunk()])
        assert not report.passed
        assert any("TRACE-001" in e for e in report.errors)

    def test_known_chunk_id_passes(self) -> None:
        report = validate_baseline_output([_make_story()], [_make_chunk(CHUNK_ID)])
        assert report.passed


# ── Uniqueness ────────────────────────────────────────────────────────────────

class TestStoryIdUniqueness:
    def test_duplicate_story_ids_fail(self) -> None:
        chunks = [_make_chunk(f"doc-001__chunk_{i:04d}") for i in range(2)]
        stories = [
            _make_story(id="US-0001", chunk_id="doc-001__chunk_0000"),
            _make_story(id="US-0001", chunk_id="doc-001__chunk_0001"),  # duplicate id
        ]
        report = validate_baseline_output(stories, chunks)
        assert not report.passed
        assert any("Duplicate" in e for e in report.errors)
