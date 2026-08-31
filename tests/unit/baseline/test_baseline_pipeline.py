"""
tests/unit/baseline/test_baseline_pipeline.py
=============================================
Unit tests for BaselinePipeline — LLM is mocked; no API key required.
Tests cover: successful run, per-chunk agent call count, story ID assignment,
config loading, error handling, and output persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.baseline_agent import BaselineAgentResult, RawStory
from gdd_userstory_mas.baseline.llm_client import LLMAPIError, LLMMalformedOutputError
from gdd_userstory_mas.baseline.pipeline import BaselineConfig, BaselinePipeline, BaselineResult
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk


# ── Helpers ────────────────────────────────────────────────────────────────────

DOC_ID = "doc-gdd-001"
CHUNK_IDS = [f"doc-gdd-001__chunk_{i:04d}" for i in range(3)]


def _make_chunks(n: int = 3) -> List[GDDChunk]:
    return [
        GDDChunk(
            chunk_id=CHUNK_IDS[i],
            document_id=DOC_ID,
            text=f"Chunk {i}: The player performs action {i}.",
            position=i,
            token_count=10,
            chapter="Gameplay",
            section=f"Section {i}",
        )
        for i in range(n)
    ]


def _make_valid_raw_story(role: str = "player", idx: int = 0) -> RawStory:
    return RawStory(
        role=role,
        action=f"perform action {idx}",
        benefit=f"achieve goal {idx}",
        full_text=f"As a {role}, I want to perform action {idx}, so that I can achieve goal {idx}.",
        requirement_type="player",
        game_domain="gameplay",
        source_excerpt=f"player performs action {idx}",
        parse_errors=[],
    )


def _make_agent_result(
    chunk_id: str,
    valid_count: int = 1,
    invalid_count: int = 0,
) -> BaselineAgentResult:
    valid = [_make_valid_raw_story(idx=i) for i in range(valid_count)]
    invalid = [
        RawStory("", "", "", "", "", "", None, parse_errors=["Missing role"])
        for _ in range(invalid_count)
    ]
    return BaselineAgentResult(
        chunk_id=chunk_id,
        raw_stories=valid + invalid,
        valid_stories=valid,
        invalid_stories=invalid,
        prompt_tokens=100,
        completion_tokens=50,
        model_name="gpt-4o",
        llm_response_raw='{"stories": []}',
    )


def _patch_agent(return_values: List[BaselineAgentResult]):
    """Patch BaselineAgent to return a fixed sequence of results."""
    mock_agent = MagicMock()
    mock_agent.process_chunk.side_effect = return_values
    return mock_agent


# ── BaselineConfig ─────────────────────────────────────────────────────────────

class TestBaselineConfig:
    def test_defaults(self) -> None:
        config = BaselineConfig.defaults()
        assert config.provider == "openai"
        assert config.temperature == 0.0
        assert config.strategy == "per_chunk"
        assert config.pipeline_type == "Baseline"

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "baseline.yaml"
        yaml_path.write_text(
            "model:\n  provider: anthropic\n  name: claude-3-5-sonnet-20241022\n"
            "  temperature: 0\n  max_tokens: 1024\n  top_p: null\n  response_format: json_object\n"
            "invocation:\n  strategy: per_chunk\n  max_retries: 2\n"
            "  retry_backoff_seconds: 1\n  max_stories_per_chunk: 5\n"
            "prompt:\n  version: baseline_v2\n  system_prompt_file: config/prompts/p.txt\n"
            "output:\n  pipeline_type: Baseline\n  validation_status_default: Unreviewed\n",
            encoding="utf-8",
        )
        config = BaselineConfig.from_yaml(yaml_path)
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-5-sonnet-20241022"
        assert config.max_retries == 2
        assert config.prompt_version == "baseline_v2"


# ── BaselinePipeline.run ───────────────────────────────────────────────────────

class TestBaselinePipelineRun:
    def _build_pipeline(self, prompt_file: Path) -> BaselinePipeline:
        config = BaselineConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.0,
            system_prompt_file=str(prompt_file),
        )
        return BaselinePipeline(config=config, project_root=prompt_file.parent.parent)

    @pytest.fixture()
    def prompt_file(self, tmp_path: Path) -> Path:
        prompts_dir = tmp_path / "config" / "prompts"
        prompts_dir.mkdir(parents=True)
        p = prompts_dir / "baseline_system_prompt.txt"
        p.write_text(
            "System. Chapter:{chapter} Section:{section} {position}/{total_chunks}",
            encoding="utf-8",
        )
        return p

    def test_run_returns_baseline_result(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)
        results = [_make_agent_result(c.chunk_id, valid_count=1) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert isinstance(result, BaselineResult)

    def test_agent_called_once_per_chunk(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(3)
        results = [_make_agent_result(c.chunk_id) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            mock_agent = _patch_agent(results)
            MockAgent.return_value = mock_agent
            pipeline.run(chunks, document_id=DOC_ID)

        assert mock_agent.process_chunk.call_count == 3

    def test_stories_have_sequential_ids(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)
        results = [_make_agent_result(c.chunk_id, valid_count=2) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert len(result.stories) == 4
        story_ids = [s.id for s in result.stories]
        assert story_ids == ["US-0001", "US-0002", "US-0003", "US-0004"]

    def test_all_stories_have_pipeline_type_baseline(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)
        results = [_make_agent_result(c.chunk_id) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        for story in result.stories:
            assert story.pipeline_type == "Baseline"

    def test_all_stories_have_validation_status_unreviewed(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)
        results = [_make_agent_result(c.chunk_id) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        for story in result.stories:
            assert story.validation_status == "Unreviewed"

    def test_all_stories_have_iteration_count_none(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)
        results = [_make_agent_result(c.chunk_id) for c in chunks]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        for story in result.stories:
            assert story.iteration_count_final is None

    def test_source_chunk_id_matches_chunk(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id, valid_count=1)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert result.stories[0].source_chunk_ids == [chunks[0].chunk_id]

    def test_custom_run_id_is_used(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID, run_id="my-run-007")

        assert result.run_id == "my-run-007"

    def test_token_usage_is_summed(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(3)
        results = [_make_agent_result(c.chunk_id) for c in chunks]
        # Each mock result has 100 prompt + 50 completion tokens

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert result.total_prompt_tokens == 300
        assert result.total_completion_tokens == 150

    def test_llm_api_error_halts_run(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.process_chunk.side_effect = LLMAPIError("API down")
            MockAgent.return_value = mock_agent
            with pytest.raises(RuntimeError, match="LLM API failure"):
                pipeline.run(chunks, document_id=DOC_ID)

    def test_malformed_output_skips_chunk_and_continues(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(2)

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.process_chunk.side_effect = [
                LLMMalformedOutputError("bad json"),          # chunk 0 — skip
                _make_agent_result(chunks[1].chunk_id, 1),   # chunk 1 — ok
            ]
            MockAgent.return_value = mock_agent
            result = pipeline.run(chunks, document_id=DOC_ID)

        # One story from chunk 1, chunk 0 skipped
        assert len(result.stories) == 1
        assert len(result.errors) >= 1

    def test_output_files_created(self, prompt_file: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "outputs" / "run-baseline"
        pipeline = BaselinePipeline(
            config=BaselineConfig(system_prompt_file=str(prompt_file)),
            output_dir=output_dir,
            project_root=prompt_file.parent.parent,
        )
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            pipeline.run(chunks, document_id=DOC_ID)

        baseline_dir = output_dir / "02_baseline"
        assert (baseline_dir / "final_user_stories.json").exists()
        assert (baseline_dir / "experiment_run.json").exists()
        assert (baseline_dir / "validation_report.json").exists()
        assert (baseline_dir / "token_usage.json").exists()

    def test_final_user_stories_json_is_valid(self, prompt_file: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "outputs" / "run-baseline-json"
        pipeline = BaselinePipeline(
            config=BaselineConfig(system_prompt_file=str(prompt_file)),
            output_dir=output_dir,
            project_root=prompt_file.parent.parent,
        )
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id, valid_count=2)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            pipeline.run(chunks, document_id=DOC_ID)

        data = json.loads(
            (output_dir / "02_baseline" / "final_user_stories.json").read_text()
        )
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["pipeline_type"] == "Baseline"
        assert data[0]["validation_status"] == "Unreviewed"

    def test_experiment_run_status_completed(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert result.experiment_run.status == "completed"

    def test_validation_report_attached(self, prompt_file: Path) -> None:
        pipeline = self._build_pipeline(prompt_file)
        chunks = _make_chunks(1)
        results = [_make_agent_result(chunks[0].chunk_id)]

        with patch("gdd_userstory_mas.baseline.pipeline.LLMClient"), \
             patch("gdd_userstory_mas.baseline.pipeline.BaselineAgent") as MockAgent:
            MockAgent.return_value = _patch_agent(results)
            result = pipeline.run(chunks, document_id=DOC_ID)

        assert result.validation_report is not None
        assert result.validation_report.total_stories >= 0
