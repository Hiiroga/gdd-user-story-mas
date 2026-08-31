"""
tests/unit/schemas/test_final_user_story.py
============================================
Unit tests for FinalUserStory and ExperimentRun Pydantic schemas.
All tests are offline — no LLM required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import (
    ConfidenceEvidence,
    FinalUserStory,
    MergeHistoryEntry,
)

NOW = datetime.now(tz=timezone.utc)
DOC_ID = "doc-001"
RUN_ID = "run-001"
CHUNK_ID = "doc-001__chunk_0000"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _valid_baseline(**overrides) -> dict:
    base = dict(
        id="US-0001",
        run_id=RUN_ID,
        pipeline_type="Baseline",
        source_document_id=DOC_ID,
        source_section="Gameplay",
        source_chunk_ids=[CHUNK_ID],
        role="player",
        action="perform a double-jump",
        benefit="reach high platforms",
        full_text="As a player, I want to perform a double-jump, so that I can reach high platforms.",
        requirement_type="player",
        game_domain="gameplay",
        validation_status="Unreviewed",
    )
    base.update(overrides)
    return base


def _valid_mas(**overrides) -> dict:
    base = _valid_baseline(
        pipeline_type="MAS",
        validation_status="Valid",
    )
    base.update(overrides)
    return base


# ── FinalUserStory — Baseline ──────────────────────────────────────────────────

class TestFinalUserStoryBaseline:
    def test_valid_baseline_story(self) -> None:
        s = FinalUserStory(**_valid_baseline())
        assert s.pipeline_type == "Baseline"
        assert s.validation_status == "Unreviewed"
        assert s.iteration_count_final is None

    def test_optional_fields_default_correctly(self) -> None:
        s = FinalUserStory(**_valid_baseline())
        assert s.validation_feedback == []
        assert s.merge_history is None
        assert s.confidence_evidence is None
        assert s.iteration_count_final is None

    def test_source_section_can_be_none(self) -> None:
        s = FinalUserStory(**_valid_baseline(source_section=None))
        assert s.source_section is None

    def test_confidence_evidence_populated(self) -> None:
        s = FinalUserStory(**_valid_baseline(
            confidence_evidence=ConfidenceEvidence(
                source_excerpt="player can double-jump",
                reviewer_traceability_confirmed=None,
            )
        ))
        assert s.confidence_evidence.source_excerpt == "player can double-jump"
        assert s.confidence_evidence.reviewer_traceability_confirmed is None

    def test_rejects_iteration_count_not_none_for_baseline(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(iteration_count_final=0))

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(id=""))

    def test_rejects_whitespace_id(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(id="   "))

    def test_rejects_empty_source_chunk_ids(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(source_chunk_ids=[]))

    def test_rejects_full_text_not_starting_with_as_a(self) -> None:
        with pytest.raises(ValidationError, match="VAL-020"):
            FinalUserStory(**_valid_baseline(
                full_text="Player wants to double-jump to reach platforms."
            ))

    def test_rejects_invalid_pipeline_type(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(pipeline_type="SingleAgent"))

    def test_rejects_invalid_requirement_type(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(requirement_type="npc"))

    def test_rejects_invalid_game_domain(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(game_domain="economics"))

    def test_rejects_invalid_validation_status(self) -> None:
        with pytest.raises(ValidationError):
            FinalUserStory(**_valid_baseline(validation_status="Pending"))

    def test_serialises_to_json(self) -> None:
        s = FinalUserStory(**_valid_baseline())
        d = s.model_dump(mode="json")
        assert d["id"] == "US-0001"
        assert d["pipeline_type"] == "Baseline"
        assert d["validation_status"] == "Unreviewed"


# ── FinalUserStory — MAS ───────────────────────────────────────────────────────

class TestFinalUserStoryMAS:
    def test_valid_mas_story(self) -> None:
        s = FinalUserStory(**_valid_mas())
        assert s.pipeline_type == "MAS"
        assert s.validation_status == "Valid"

    def test_mas_story_with_iteration_count(self) -> None:
        s = FinalUserStory(**_valid_mas(iteration_count_final=2))
        assert s.iteration_count_final == 2

    def test_mas_story_with_merge_history(self) -> None:
        s = FinalUserStory(**_valid_mas(
            validation_status="Merged",
            merge_history=[
                MergeHistoryEntry(
                    original_story_id="US-0002",
                    original_source_chunk_id=CHUNK_ID,
                )
            ]
        ))
        assert len(s.merge_history) == 1
        assert s.merge_history[0].original_story_id == "US-0002"

    def test_mas_story_with_validation_feedback(self) -> None:
        s = FinalUserStory(**_valid_mas(
            validation_feedback=["clarity criterion failed on iteration 1"]
        ))
        assert len(s.validation_feedback) == 1

    def test_all_requirement_types_accepted(self) -> None:
        for rt in ("player", "system", "in-game entity", "dev team"):
            s = FinalUserStory(**_valid_mas(requirement_type=rt))
            assert s.requirement_type == rt

    def test_all_game_domains_accepted(self) -> None:
        for domain in ("gameplay", "UI", "narrative", "systems", "level design", "audio", "other"):
            s = FinalUserStory(**_valid_mas(game_domain=domain))
            assert s.game_domain == domain

    def test_all_validation_statuses_accepted(self) -> None:
        for status in ("Valid", "Rejected-ManualReview", "Unreviewed", "Merged", "Redundancy-Unresolved"):
            s = FinalUserStory(**_valid_mas(validation_status=status))
            assert s.validation_status == status

    def test_multiple_source_chunk_ids(self) -> None:
        s = FinalUserStory(**_valid_mas(
            source_chunk_ids=["doc-001__chunk_0000", "doc-001__chunk_0001"]
        ))
        assert len(s.source_chunk_ids) == 2


# ── ConfidenceEvidence ─────────────────────────────────────────────────────────

class TestConfidenceEvidence:
    def test_all_null(self) -> None:
        ce = ConfidenceEvidence()
        assert ce.source_excerpt is None
        assert ce.reviewer_traceability_confirmed is None

    def test_with_excerpt_only(self) -> None:
        ce = ConfidenceEvidence(source_excerpt="double-jump to reach")
        assert ce.source_excerpt == "double-jump to reach"

    def test_reviewer_confirmed_true(self) -> None:
        ce = ConfidenceEvidence(reviewer_traceability_confirmed=True)
        assert ce.reviewer_traceability_confirmed is True


# ── MergeHistoryEntry ──────────────────────────────────────────────────────────

class TestMergeHistoryEntry:
    def test_valid_entry(self) -> None:
        entry = MergeHistoryEntry(
            original_story_id="US-0003",
            original_source_chunk_id=CHUNK_ID,
        )
        assert entry.original_story_id == "US-0003"

    def test_serialises_to_dict(self) -> None:
        entry = MergeHistoryEntry(
            original_story_id="US-0003",
            original_source_chunk_id=CHUNK_ID,
        )
        d = entry.model_dump()
        assert "original_story_id" in d


# ── ExperimentRun ──────────────────────────────────────────────────────────────

class TestExperimentRun:
    def test_valid_baseline_run(self) -> None:
        run = ExperimentRun(
            run_id="run-001",
            document_id=DOC_ID,
            pipeline_type="Baseline",
            config_snapshot_ref="outputs/run-001/02_baseline/config_snapshot.yaml",
            started_at=NOW,
        )
        assert run.status == "running"
        assert run.completed_at is None
        assert run.final_user_story_ids == []

    def test_valid_mas_run(self) -> None:
        run = ExperimentRun(
            run_id="run-002",
            document_id=DOC_ID,
            pipeline_type="MAS",
            config_snapshot_ref="outputs/run-002/config_snapshot.yaml",
            started_at=NOW,
            status="completed",
            completed_at=NOW,
            final_user_story_ids=["US-0001", "US-0002"],
        )
        assert run.status == "completed"
        assert len(run.final_user_story_ids) == 2

    def test_rejects_invalid_pipeline_type(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentRun(
                run_id="run-003",
                document_id=DOC_ID,
                pipeline_type="Unknown",
                config_snapshot_ref="x",
                started_at=NOW,
            )

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentRun(
                run_id="run-004",
                document_id=DOC_ID,
                pipeline_type="Baseline",
                config_snapshot_ref="x",
                started_at=NOW,
                status="paused",
            )

    def test_serialises_to_json(self) -> None:
        run = ExperimentRun(
            run_id="run-005",
            document_id=DOC_ID,
            pipeline_type="Baseline",
            config_snapshot_ref="config.yaml",
            started_at=NOW,
        )
        j = run.model_dump_json()
        assert "run-005" in j
        assert "Baseline" in j
