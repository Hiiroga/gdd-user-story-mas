"""
tests/unit/mas/test_mas_pipeline.py
=====================================
Unit tests for MASPipeline orchestration layer.
All agents are mocked — no real LLM calls made.
Tests cover: happy path, failure isolation, max iterations, empty results.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.mas.pipeline import MASPipeline, MASPipelineError
from gdd_userstory_mas.mas.pipeline_config import MASPipelineConfig
from gdd_userstory_mas.mas.pipeline_result import MASPipelineResult
from gdd_userstory_mas.schemas.candidate_requirement import (
    AnalystOutput,
    CandidateRequirement,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import ReaderOutput
from gdd_userstory_mas.schemas.redundancy_analysis import (
    DuplicateGroup,
    MergeAction,
    RedundancyAnalysis,
)
from gdd_userstory_mas.schemas.reviewer_result import ReviewerResult
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────────────

# Real project root — needed for agents to find their prompt files
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def _make_config(tmp_path: Path, max_iterations: int = 3) -> MASPipelineConfig:
    """Return a MASPipelineConfig with no-op defaults and disk I/O to tmp_path."""
    from gdd_userstory_mas.mas.pipeline_config import MASPipelineConfig
    from gdd_userstory_mas.mas.reader_agent import ReaderAgentConfig
    from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig
    from gdd_userstory_mas.mas.generator_agent import GeneratorAgentConfig
    from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgentConfig
    from gdd_userstory_mas.mas.redundancy_agent import RedundancyAgentConfig
    from gdd_userstory_mas.mas.evaluator_agent import EvaluatorAgentConfig
    from gdd_userstory_mas.preprocessing.pipeline import PreprocessingConfig

    return MASPipelineConfig(
        max_reviewer_iterations=max_iterations,
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
        save_intermediate=False,
        reader_config=ReaderAgentConfig(),
        analyst_config=AnalystAgentConfig(),
        generator_config=GeneratorAgentConfig(),
        reviewer_config=ReviewerAgentConfig(),
        redundancy_config=RedundancyAgentConfig(),
        evaluator_config=EvaluatorAgentConfig(),
        preprocessing_config=PreprocessingConfig.defaults(),
    )


def _make_chunk(i: int = 0, document_id: str = "doc1") -> GDDChunk:
    return GDDChunk(
        chunk_id=f"{document_id}__chunk_{i:04d}",
        document_id=document_id,
        section=f"Section {i}",
        text=f"The player can attack enemies. Feature {i}.",
        position=i,
        token_count=20,
    )


def _make_reader_output(chunk: GDDChunk) -> ReaderOutput:
    from gdd_userstory_mas.schemas.reader_output import EvidencedItem
    return ReaderOutput(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        gameplay_elements=[
            EvidencedItem(content="attack enemies", source_excerpt="player can attack")
        ],
        systems=[],
        characters=[],
        ui_elements=[],
        narrative=[],
        themes=[],
    )


def _make_analyst_output(chunk: GDDChunk, n: int = 1) -> AnalystOutput:
    candidates = []
    for i in range(n):
        candidates.append(CandidateRequirement(
            candidate_id=f"{chunk.chunk_id}__cand_{i:04d}",
            source_chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            perspective="player",
            domain="gameplay",
            requirement_text=f"Player attacks enemies (variant {i})",
        ))
    return AnalystOutput(
        source_chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        candidates=candidates,
    )


def _make_draft(candidate: CandidateRequirement, iteration: int = 0) -> GeneratedUserStory:
    return GeneratedUserStory(
        draft_id=f"{candidate.candidate_id}__draft",
        candidate_id=candidate.candidate_id,
        source_chunk_id=candidate.source_chunk_id,
        document_id=candidate.document_id,
        role="player",
        action="attack enemies with melee weapons",
        benefit="defeat them and progress through levels",
        full_text=(
            "As a player, I want attack enemies with melee weapons, "
            "so that defeat them and progress through levels."
        ),
        iteration_count=iteration,
    )


def _make_valid_reviewer_result(draft: GeneratedUserStory) -> ReviewerResult:
    return ReviewerResult(
        review_id=f"{draft.draft_id}__review_iter0",
        draft_id=draft.draft_id,
        status="Valid",
        evaluated_at_iteration=draft.iteration_count,
        failing_criteria=[],
    )


def _make_invalid_reviewer_result(draft: GeneratedUserStory) -> ReviewerResult:
    return ReviewerResult(
        review_id=f"{draft.draft_id}__review_iter0",
        draft_id=draft.draft_id,
        status="Invalid",
        evaluated_at_iteration=draft.iteration_count,
        failing_criteria=["clarity"],
    )


def _make_redundancy_analysis(run_id: str, story_count: int) -> RedundancyAnalysis:
    return RedundancyAnalysis(
        analysis_id=f"redundancy__{run_id}__pass1",
        run_id=run_id,
        analyzed_at_pass=1,
        duplicate_groups=[],
        unique_story_ids=[],
        resolution_status="resolved",
        merge_actions=[],
        resolved_story_count=story_count,
        input_story_count=story_count,
    )


def _make_evaluation_result(run_id: str, story_count: int) -> EvaluationResult:
    return EvaluationResult(
        evaluation_id=f"eval__{run_id}",
        run_id=run_id,
        document_id="doc1",
        pipeline_type="MAS",
        total_stories=story_count,
        aqusa_score=80.0,
        coverage=75.0,
        diversity=0.4,
        redundancy_rate=0.0,
        hallucination_rate=5.0,
    )


def _make_pipeline(tmp_path: Path, max_iterations: int = 3) -> MASPipeline:
    config = _make_config(tmp_path, max_iterations)
    # Patch LLMClient so agents don't try to validate API key on __init__
    with patch("gdd_userstory_mas.baseline.llm_client.LLMClient.__init__", return_value=None):
        pipeline = MASPipeline(
            config=config,
            api_key="test-key-placeholder",
            project_root=_PROJECT_ROOT,
        )
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helper: patch all 6 agents on an existing pipeline instance
# ─────────────────────────────────────────────────────────────────────────────

def _mock_all_agents(
    pipeline: MASPipeline,
    chunks: List[GDDChunk],
    reader_outputs=None,
    analyst_outputs=None,
    drafts=None,
    reviewer_results=None,
    redundancy_result=None,
    evaluation_result=None,
) -> None:
    """Replace agent internals with mocks. All params are optional lists indexed by call order."""
    # Preprocessing
    pre_result = MagicMock()
    pre_result.chunks = chunks
    pipeline._config.preprocessing_config = MagicMock()

    # Build defaults if not supplied
    if reader_outputs is None:
        reader_outputs = [_make_reader_output(c) for c in chunks]
    if analyst_outputs is None:
        analyst_outputs = [_make_analyst_output(c) for c in chunks]
    if drafts is None:
        all_candidates = [cand for ao in analyst_outputs for cand in ao.candidates]
        drafts = [_make_draft(cand) for cand in all_candidates]
    if reviewer_results is None:
        reviewer_results = [_make_valid_reviewer_result(d) for d in drafts]
    if redundancy_result is None:
        run_id_mock = "test_run"
        redundancy_result = _make_redundancy_analysis(run_id_mock, len(drafts))
    if evaluation_result is None:
        evaluation_result = _make_evaluation_result("test_run", len(drafts))

    # Mock preprocessing pipeline
    with patch.object(
        pipeline._config, "preprocessing_config", MagicMock()
    ):
        pass

    pipeline._config.preprocessing_config = MagicMock()

    # Replace agent methods
    pipeline._reader.process_chunk = MagicMock(side_effect=reader_outputs)
    pipeline._analyst.process_reader_output = MagicMock(side_effect=analyst_outputs)

    draft_iter = iter(drafts)
    pipeline._generator.generate = MagicMock(side_effect=lambda *a, **kw: next(draft_iter))

    rev_iter = iter(reviewer_results)
    pipeline._reviewer.review = MagicMock(
        side_effect=lambda d, t: (next(rev_iter), None)
    )
    pipeline._redundancy.check = MagicMock(return_value=redundancy_result)
    pipeline._evaluator.evaluate = MagicMock(return_value=evaluation_result)

    return pre_result


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — single chunk, single candidate
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPathSingleChunk:

    def test_returns_pipeline_result(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        candidate = analyst_out.candidates[0]
        draft = _make_draft(candidate, iteration=0)
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch(
            "gdd_userstory_mas.mas.pipeline.PreprocessingPipeline"
        ) as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert isinstance(result, MASPipelineResult)
        assert result.succeeded

    def test_experiment_run_completed(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        candidate = analyst_out.candidates[0]
        draft = _make_draft(candidate)
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.experiment_run.status == "completed"
        assert result.experiment_run.completed_at is not None

    def test_produces_final_stories(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=2)
        drafts = [_make_draft(c) for c in analyst_out.candidates]
        rev_results = [_make_valid_reviewer_result(d) for d in drafts]

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 2)
        er = _make_evaluation_result("run1", 2)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)

        draft_iter = iter(drafts)
        pipeline._generator.generate = MagicMock(
            side_effect=lambda *a, **kw: next(draft_iter)
        )
        rev_iter = iter(rev_results)
        pipeline._reviewer.review = MagicMock(
            side_effect=lambda d, t: (next(rev_iter), None)
        )
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.total_stories >= 1

    def test_evaluation_result_attached(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        draft = _make_draft(analyst_out.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.evaluation_result is not None
        assert result.evaluation_result.aqusa_score == 80.0


# ─────────────────────────────────────────────────────────────────────────────
# Failure isolation: Reader fails for one chunk
# ─────────────────────────────────────────────────────────────────────────────

class TestReaderFailureIsolation:

    def test_reader_failure_skips_chunk(self, tmp_path):
        """Reader fails on chunk 0 — should not abort the run."""
        pipeline = _make_pipeline(tmp_path)
        chunk0 = _make_chunk(0)
        chunk1 = _make_chunk(1)

        reader_out1 = _make_reader_output(chunk1)
        analyst_out1 = _make_analyst_output(chunk1, n=1)
        draft = _make_draft(analyst_out1.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk0, chunk1]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        # Reader: fail on first call, succeed on second
        pipeline._reader.process_chunk = MagicMock(
            side_effect=[RuntimeError("LLM timeout"), reader_out1]
        )
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out1)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.succeeded
        assert result.experiment_run.status == "completed"
        assert result.total_errors >= 1
        # At least one error logged for the reader failure
        assert any("chunk" in e.message.lower() for e in result.error_log)

    def test_reader_failure_logs_error(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk(0)
        pre_result = MagicMock()
        pre_result.chunks = [chunk]

        pipeline._reader.process_chunk = MagicMock(
            side_effect=RuntimeError("API error")
        )
        pipeline._analyst.process_reader_output = MagicMock(
            return_value=_make_analyst_output(chunk, 0)
        )
        pipeline._redundancy.check = MagicMock(
            return_value=_make_redundancy_analysis("r", 0)
        )
        pipeline._evaluator.evaluate = MagicMock(
            return_value=_make_evaluation_result("r", 0)
        )

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.total_errors >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Failure: max reviewer iterations
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxReviewerIterations:

    def test_max_iterations_produces_rejected_story(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, max_iterations=2)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        candidate = analyst_out.candidates[0]

        # All reviewer calls return Invalid
        drafts = [_make_draft(candidate, iteration=i) for i in range(2)]
        rev_results = [_make_invalid_reviewer_result(d) for d in drafts]

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 0)
        er = _make_evaluation_result("run1", 0)
        er_null = EvaluationResult(
            evaluation_id="eval__run1",
            run_id="run1",
            document_id="doc1",
            pipeline_type="MAS",
            total_stories=0,
            aqusa_score=None,
            coverage=None,
            diversity=None,
            redundancy_rate=None,
            hallucination_rate=None,
            null_reasons={
                "aqusa_score": "No stories.",
                "coverage": "No stories.",
                "diversity": "No stories.",
                "redundancy_rate": "No stories.",
                "hallucination_rate": "No stories.",
            },
        )

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)

        draft_iter = iter(drafts)
        pipeline._generator.generate = MagicMock(
            side_effect=lambda *a, **kw: next(draft_iter)
        )
        rev_iter = iter(rev_results)
        pipeline._reviewer.review = MagicMock(
            side_effect=lambda d, t: (next(rev_iter), None)
        )
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er_null)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.succeeded
        # Should have a max_iterations error logged
        assert any(
            "max_iterations" in e.error_type for e in result.error_log
        )

    def test_max_iterations_logs_error(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, max_iterations=1)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        candidate = analyst_out.candidates[0]
        draft = _make_draft(candidate, iteration=0)
        rev_result = _make_invalid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(
            return_value=_make_redundancy_analysis("r", 0)
        )
        pipeline._evaluator.evaluate = MagicMock(
            return_value=_make_evaluation_result("r", 0)
        )

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.total_errors >= 1
        error_types = [e.error_type for e in result.error_log]
        assert "max_iterations_reached_reviewer" in error_types


# ─────────────────────────────────────────────────────────────────────────────
# Failure: preprocessing fails fatally
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingFailure:

    def test_preprocessing_exception_halts_run(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.side_effect = RuntimeError("PDF corrupt")
            result = pipeline.run(tmp_path / "test.pdf")

        assert not result.succeeded
        assert result.experiment_run.status == "failed"

    def test_empty_chunks_with_abort_halts(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._config.abort_on_empty_chunks = True

        pre_result = MagicMock()
        pre_result.chunks = []

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert not result.succeeded

    def test_empty_chunks_without_abort_succeeds(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        pipeline._config.abort_on_empty_chunks = False

        pre_result = MagicMock()
        pre_result.chunks = []

        pipeline._redundancy.check = MagicMock(
            return_value=_make_redundancy_analysis("r", 0)
        )
        pipeline._evaluator.evaluate = MagicMock(
            return_value=_make_evaluation_result("r", 0)
        )

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        # Should succeed with 0 stories
        assert result.succeeded


# ─────────────────────────────────────────────────────────────────────────────
# Traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceability:

    def test_final_stories_have_run_id(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        draft = _make_draft(analyst_out.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        for story in result.final_stories:
            assert story.run_id == result.run_id
            assert story.source_chunk_ids

    def test_experiment_run_has_story_ids(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        draft = _make_draft(analyst_out.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert len(result.experiment_run.final_user_story_ids) == result.total_stories

    def test_output_dir_created(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        draft = _make_draft(analyst_out.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        assert result.output_dir.exists()
        assert (result.output_dir / "experiment_run.json").exists()

    def test_result_summary(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        chunk = _make_chunk()
        reader_out = _make_reader_output(chunk)
        analyst_out = _make_analyst_output(chunk, n=1)
        draft = _make_draft(analyst_out.candidates[0])
        rev_result = _make_valid_reviewer_result(draft)

        pre_result = MagicMock()
        pre_result.chunks = [chunk]
        ra = _make_redundancy_analysis("run1", 1)
        er = _make_evaluation_result("run1", 1)

        pipeline._reader.process_chunk = MagicMock(return_value=reader_out)
        pipeline._analyst.process_reader_output = MagicMock(return_value=analyst_out)
        pipeline._generator.generate = MagicMock(return_value=draft)
        pipeline._reviewer.review = MagicMock(return_value=(rev_result, None))
        pipeline._redundancy.check = MagicMock(return_value=ra)
        pipeline._evaluator.evaluate = MagicMock(return_value=er)

        with patch("gdd_userstory_mas.mas.pipeline.PreprocessingPipeline") as MockPre:
            MockPre.return_value.run.return_value = pre_result
            result = pipeline.run(tmp_path / "test.pdf")

        summary = result.summary()
        assert "completed" in summary
        assert "stories=" in summary
