"""
tests/unit/mas/test_result_store.py
=====================================
Unit tests for ResultStore — artifact persistence layer.
No LLM calls. All tests use tmp_path (pytest fixture).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gdd_userstory_mas.storage.result_store import ResultStore
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import ReaderOutput, EvidencedItem
from gdd_userstory_mas.schemas.candidate_requirement import (
    AnalystOutput, CandidateRequirement,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory, ConfidenceEvidence
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_store(tmp_path: Path) -> ResultStore:
    return ResultStore(run_id="test-run-001", output_dir=tmp_path / "outputs")


def _make_experiment_run() -> ExperimentRun:
    return ExperimentRun(
        run_id="test-run-001",
        document_id="doombible",
        pipeline_type="MAS",
        config_snapshot_ref="outputs/test-run-001/config_snapshot.yaml",
        started_at=datetime.now(timezone.utc),
        status="running",
    )


def _make_chunk(i: int = 0) -> GDDChunk:
    return GDDChunk(
        chunk_id=f"doombible__chunk_{i:04d}",
        document_id="doombible",
        text="The player can attack enemies.",
        section=f"Section {i}",
        position=i,
        token_count=10,
    )


def _make_reader_output(chunk: GDDChunk) -> ReaderOutput:
    return ReaderOutput(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        gameplay_elements=[EvidencedItem(content="attack enemies", source_excerpt="player can attack")],
        systems=[],
        characters=[],
        ui_elements=[],
        narrative=[],
        themes=[],
    )


def _make_analyst_output(chunk: GDDChunk) -> AnalystOutput:
    return AnalystOutput(
        source_chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        candidates=[
            CandidateRequirement(
                candidate_id=f"{chunk.chunk_id}__cand_0000",
                source_chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                perspective="player",
                domain="gameplay",
                requirement_text="Player attacks enemies",
            )
        ],
    )


def _make_draft(candidate: CandidateRequirement) -> GeneratedUserStory:
    return GeneratedUserStory(
        draft_id=f"{candidate.candidate_id}__draft",
        candidate_id=candidate.candidate_id,
        source_chunk_id=candidate.source_chunk_id,
        document_id=candidate.document_id,
        role="player",
        action="attack enemies with melee weapons",
        benefit="defeat them and progress",
        full_text=(
            "As a player, I want attack enemies with melee weapons, "
            "so that defeat them and progress."
        ),
        iteration_count=0,
    )


def _make_redundancy_analysis(run_id: str) -> RedundancyAnalysis:
    return RedundancyAnalysis(
        analysis_id=f"redundancy__{run_id}__pass1",
        run_id=run_id,
        analyzed_at_pass=1,
        duplicate_groups=[],
        unique_story_ids=[],
        resolution_status="resolved",
        merge_actions=[],
        resolved_story_count=0,
        input_story_count=0,
    )


def _make_final_story(run_id: str, i: int = 0) -> FinalUserStory:
    return FinalUserStory(
        id=f"US-{i+1:04d}",
        run_id=run_id,
        pipeline_type="MAS",
        source_document_id="doombible",
        source_chunk_ids=["doombible__chunk_0000"],
        role="player",
        action="attack enemies",
        benefit="progress through levels",
        full_text=(
            "As a player, I want attack enemies, "
            "so that progress through levels."
        ),
        requirement_type="player",
        game_domain="gameplay",
        validation_status="Valid",
        confidence_evidence=ConfidenceEvidence(
            reviewer_traceability_confirmed=True
        ),
        iteration_count_final=0,
    )


def _make_evaluation_result(run_id: str) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=f"eval__{run_id}",
        run_id=run_id,
        document_id="doombible",
        pipeline_type="MAS",
        total_stories=1,
        aqusa_score=80.0,
        coverage=75.0,
        diversity=0.4,
        redundancy_rate=0.0,
        hallucination_rate=5.0,
    )


def _make_error(run_id: str) -> ErrorFailureLog:
    return ErrorFailureLog(
        error_id="err-001",
        run_id=run_id,
        document_id="doombible",
        stage="gdd_reader",
        error_type="llm_api_failure",
        related_entity_id="doombible__chunk_0000",
        message="LLM API timeout after 3 retries.",
        retry_count=3,
        resolution="excluded_and_logged",
        timestamp=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ResultStore — construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_run_dir_created(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.run_dir.exists()
        assert store.run_dir.is_dir()

    def test_run_dir_path_property(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.run_dir_path == store.run_dir

    def test_run_id_stored(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.run_id == "test-run-001"


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRun
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentRun:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        run = _make_experiment_run()
        store.save_experiment_run(run)
        loaded = store.load_experiment_run()
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.document_id == run.document_id

    def test_load_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_experiment_run() is None

    def test_overwrite_updates(self, tmp_path):
        store = _make_store(tmp_path)
        run = _make_experiment_run()
        store.save_experiment_run(run)
        # Simulate update
        run2 = run.model_copy(update={"status": "completed"})
        store.save_experiment_run(run2)
        loaded = store.load_experiment_run()
        assert loaded.status == "completed"


# ─────────────────────────────────────────────────────────────────────────────
# Config snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigSnapshot:

    def test_save_returns_path(self, tmp_path):
        store = _make_store(tmp_path)
        path = store.save_config_snapshot("model:\n  name: gemini-flash\n")
        assert path.exists()
        assert path.name == "config_snapshot.yaml"

    def test_content_preserved(self, tmp_path):
        store = _make_store(tmp_path)
        yaml_text = "model:\n  name: gemini-flash\n"
        path = store.save_config_snapshot(yaml_text)
        assert path.read_text(encoding="utf-8") == yaml_text


# ─────────────────────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestChunks:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        chunks = [_make_chunk(0), _make_chunk(1)]
        store.save_chunks(chunks)
        loaded = store.load_chunks()
        assert len(loaded) == 2
        assert loaded[0].chunk_id == chunks[0].chunk_id

    def test_empty_chunks(self, tmp_path):
        store = _make_store(tmp_path)
        store.save_chunks([])
        loaded = store.load_chunks()
        assert loaded == []

    def test_load_missing_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_chunks() == []


# ─────────────────────────────────────────────────────────────────────────────
# Reader outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestReaderOutput:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk(0)
        out = _make_reader_output(chunk)
        store.save_reader_output(chunk.chunk_id, out)
        loaded = store.load_reader_output(chunk.chunk_id)
        assert loaded is not None
        assert loaded.chunk_id == out.chunk_id

    def test_load_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_reader_output("nonexistent") is None

    def test_chunk_id_with_slashes_safe(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk(0)
        chunk2 = chunk.model_copy(update={"chunk_id": "doc/subdir/chunk_0000"})
        out = _make_reader_output(chunk2)
        store.save_reader_output(chunk2.chunk_id, out)
        loaded = store.load_reader_output(chunk2.chunk_id)
        assert loaded is not None


# ─────────────────────────────────────────────────────────────────────────────
# Analyst outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalystOutput:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk(0)
        out = _make_analyst_output(chunk)
        store.save_analyst_output(chunk.chunk_id, out)
        loaded = store.load_analyst_output(chunk.chunk_id)
        assert loaded is not None
        assert loaded.candidate_count == 1

    def test_load_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_analyst_output("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
# Draft iterations
# ─────────────────────────────────────────────────────────────────────────────

class TestDraftIteration:

    def test_save_draft_iteration(self, tmp_path):
        store = _make_store(tmp_path)
        chunk = _make_chunk(0)
        analyst_out = _make_analyst_output(chunk)
        draft = _make_draft(analyst_out.candidates[0])
        store.save_draft_iteration(draft, "Valid")
        path = store.run_dir / "generator_reviewer" / f"{draft.draft_id}.json"
        assert path.exists()

    def test_saved_content_has_review_status(self, tmp_path):
        import json
        store = _make_store(tmp_path)
        chunk = _make_chunk(0)
        analyst_out = _make_analyst_output(chunk)
        draft = _make_draft(analyst_out.candidates[0])
        store.save_draft_iteration(draft, "Invalid")
        path = store.run_dir / "generator_reviewer" / f"{draft.draft_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["review_status"] == "Invalid"
        assert data["draft"]["draft_id"] == draft.draft_id


# ─────────────────────────────────────────────────────────────────────────────
# Redundancy analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestRedundancyAnalysis:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        ra = _make_redundancy_analysis("test-run-001")
        store.save_redundancy_analysis(ra)
        loaded = store.load_redundancy_analysis()
        assert loaded is not None
        assert loaded.analysis_id == ra.analysis_id

    def test_load_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_redundancy_analysis() is None


# ─────────────────────────────────────────────────────────────────────────────
# Final stories
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalStories:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        stories = [_make_final_story("test-run-001", i) for i in range(3)]
        store.save_final_stories(stories)
        loaded = store.load_final_stories()
        assert len(loaded) == 3
        assert loaded[0].id == "US-0001"

    def test_empty_final_stories(self, tmp_path):
        store = _make_store(tmp_path)
        store.save_final_stories([])
        loaded = store.load_final_stories()
        assert loaded == []

    def test_load_missing_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_final_stories() == []

    def test_round_trip_fidelity(self, tmp_path):
        store = _make_store(tmp_path)
        story = _make_final_story("test-run-001", 0)
        store.save_final_stories([story])
        loaded = store.load_final_stories()
        assert loaded[0].role == story.role
        assert loaded[0].validation_status == story.validation_status


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationResult:

    def test_save_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        er = _make_evaluation_result("test-run-001")
        store.save_evaluation_result(er)
        loaded = store.load_evaluation_result()
        assert loaded is not None
        assert loaded.evaluation_id == er.evaluation_id
        assert loaded.aqusa_score == er.aqusa_score

    def test_load_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_evaluation_result() is None


# ─────────────────────────────────────────────────────────────────────────────
# Error log
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorLog:

    def test_append_and_load(self, tmp_path):
        store = _make_store(tmp_path)
        err = _make_error("test-run-001")
        store.append_error_log(err)
        loaded = store.load_error_log()
        assert len(loaded) == 1
        assert loaded[0].error_id == err.error_id

    def test_multiple_appends(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(5):
            err = _make_error("test-run-001")
            err2 = err.model_copy(update={"error_id": f"err-{i:03d}"})
            store.append_error_log(err2)
        loaded = store.load_error_log()
        assert len(loaded) == 5

    def test_load_empty_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_error_log() == []

    def test_error_stage_preserved(self, tmp_path):
        store = _make_store(tmp_path)
        err = _make_error("test-run-001")
        store.append_error_log(err)
        loaded = store.load_error_log()
        assert loaded[0].stage == "gdd_reader"
        assert loaded[0].resolution == "excluded_and_logged"
