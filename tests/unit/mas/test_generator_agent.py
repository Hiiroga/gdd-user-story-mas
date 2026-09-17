"""
tests/unit/mas/test_generator_agent.py
========================================
Unit tests for UserStoryGeneratorAgent (C6).
All tests use a mock LLMClient — no real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError, LLMResponse
from gdd_userstory_mas.mas.generator_agent import (
    GeneratorAgentConfig,
    UserStoryGeneratorAgent,
)
from gdd_userstory_mas.schemas.candidate_requirement import CandidateRequirement
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_name="gemini-flash-latest",
        prompt_tokens=80,
        completion_tokens=40,
        finish_reason="stop",
    )


def _make_candidate(**kwargs) -> CandidateRequirement:
    defaults = dict(
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        requirement_text="The game must allow the player to attack enemies using melee weapons.",
        perspective="player",
        domain="gameplay",
        source_reader_output_ref="Melee weapon attacks",
    )
    defaults.update(kwargs)
    return CandidateRequirement(**defaults)


def _make_agent(tmp_path: Path, mock_llm_content: str) -> UserStoryGeneratorAgent:
    """Build agent with mocked LLMClient and real prompt file."""
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "generator_system_prompt.txt").write_text(
        "Generate a user story.\n"
        "Candidate: {candidate_id}\n"
        "Requirement: {requirement_text}\n"
        "Perspective: {perspective}\nDomain: {domain}\nRef: {source_ref}\n",
        encoding="utf-8",
    )
    config = GeneratorAgentConfig(
        min_request_interval_seconds=0.0,
        system_prompt_file="config/prompts/generator_system_prompt.txt",
    )
    with patch(
        "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
        return_value=MagicMock(),
    ):
        agent = UserStoryGeneratorAgent(config=config, project_root=tmp_path)

    agent._client.call = MagicMock(return_value=_make_llm_response(mock_llm_content))
    return agent


def _valid_llm_response() -> str:
    return json.dumps({
        "role": "player",
        "action": "attack enemies using melee weapons",
        "benefit": "I can defeat them and progress through the level",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:

    def test_returns_generated_user_story(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert isinstance(result, GeneratedUserStory)

    def test_draft_id_format(self, tmp_path):
        """draft_id must be {candidate_id}__draft (TRACE-003)."""
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.draft_id == "doc1__cand_0000__draft"

    def test_candidate_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.candidate_id == "doc1__cand_0000"

    def test_source_chunk_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.source_chunk_id == "doc1__chunk_0000"

    def test_document_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.document_id == "doc1"

    def test_role_action_benefit_extracted(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.role == "player"
        assert "melee" in result.action
        assert result.benefit

    def test_full_text_format(self, tmp_path):
        """full_text must follow 'As a ..., I want ..., so that ....' (VAL-020)."""
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.full_text.startswith("As a ")
        assert ", I want " in result.full_text
        assert ", so that " in result.full_text
        assert result.full_text.endswith(".")

    def test_full_text_matches_components(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        expected = GeneratedUserStory.build_full_text(result.role, result.action, result.benefit)
        assert result.full_text == expected

    def test_iteration_count_zero_first_draft(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate(), iteration_count=0)
        assert result.iteration_count == 0
        assert result.is_first_draft()

    def test_iteration_count_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate(), iteration_count=2)
        assert result.iteration_count == 2

    def test_previous_chain_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        chain = ["doc1__cand_0000__draft_v0"]
        result = agent.generate(_make_candidate(), previous_draft_id_chain=chain)
        assert result.previous_draft_id_chain == chain

    def test_empty_chain_for_first_draft(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.generate(_make_candidate())
        assert result.previous_draft_id_chain == []


# ─────────────────────────────────────────────────────────────────────────────
# Prefix stripping
# ─────────────────────────────────────────────────────────────────────────────

class TestPrefixStripping:

    def test_strips_as_a_from_role(self, tmp_path):
        payload = json.dumps({
            "role": "As a player",
            "action": "attack with melee weapons",
            "benefit": "defeat enemies efficiently",
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.generate(_make_candidate())
        assert result.role == "player"
        assert not result.role.lower().startswith("as a")

    def test_strips_i_want_from_action(self, tmp_path):
        payload = json.dumps({
            "role": "player",
            "action": "I want to attack with melee weapons",
            "benefit": "defeat enemies efficiently",
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.generate(_make_candidate())
        assert not result.action.lower().startswith("i want")

    def test_strips_so_that_from_benefit(self, tmp_path):
        payload = json.dumps({
            "role": "player",
            "action": "attack with melee weapons",
            "benefit": "so that I can defeat enemies efficiently",
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.generate(_make_candidate())
        assert not result.benefit.lower().startswith("so that")


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_malformed_json_raises(self, tmp_path):
        agent = _make_agent(tmp_path, "NOT JSON {{{")
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.generate(_make_candidate())

    def test_empty_role_raises(self, tmp_path):
        payload = json.dumps({
            "role": "",
            "action": "attack enemies",
            "benefit": "defeat them",
        })
        agent = _make_agent(tmp_path, payload)
        with pytest.raises(LLMMalformedOutputError, match="role"):
            agent.generate(_make_candidate())

    def test_empty_action_raises(self, tmp_path):
        payload = json.dumps({
            "role": "player",
            "action": "",
            "benefit": "defeat them",
        })
        agent = _make_agent(tmp_path, payload)
        with pytest.raises(LLMMalformedOutputError, match="action"):
            agent.generate(_make_candidate())

    def test_empty_benefit_raises(self, tmp_path):
        payload = json.dumps({
            "role": "player",
            "action": "attack enemies",
            "benefit": "",
        })
        agent = _make_agent(tmp_path, payload)
        with pytest.raises(LLMMalformedOutputError, match="benefit"):
            agent.generate(_make_candidate())


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptRendering:

    def test_candidate_id_injected(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        candidate = _make_candidate()
        rendered = agent._render_system_prompt(candidate)
        assert candidate.candidate_id in rendered

    def test_requirement_text_injected(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        candidate = _make_candidate()
        rendered = agent._render_system_prompt(candidate)
        assert candidate.requirement_text in rendered

    def test_null_source_ref_becomes_na(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        candidate = _make_candidate(source_reader_output_ref=None)
        rendered = agent._render_system_prompt(candidate)
        assert "N/A" in rendered

    def test_no_key_error_from_json_braces(self, tmp_path):
        """Prompt template with JSON braces must not raise KeyError."""
        prompts_dir = tmp_path / "config" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "generator_system_prompt.txt").write_text(
            'Return: {"role": "", "action": "", "benefit": ""}\n'
            "Candidate: {candidate_id}\n"
            "Requirement: {requirement_text}\n"
            "Perspective: {perspective}\nDomain: {domain}\nRef: {source_ref}\n",
            encoding="utf-8",
        )
        config = GeneratorAgentConfig(
            min_request_interval_seconds=0.0,
            system_prompt_file="config/prompts/generator_system_prompt.txt",
        )
        with patch(
            "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
            return_value=MagicMock(),
        ):
            agent = UserStoryGeneratorAgent(config=config, project_root=tmp_path)

        rendered = agent._render_system_prompt(_make_candidate())
        assert '{"role": "", "action": "", "benefit": ""}' in rendered


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorAgentConfig:

    def test_defaults(self):
        cfg = GeneratorAgentConfig.defaults()
        assert cfg.provider == "gemini"
        assert cfg.temperature == 0.0
        assert cfg.prompt_version == "generator_v1"

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
model:
  provider: "gemini"
  name: "gemini-flash-latest"
  temperature: 0
  max_tokens: 4096
  top_p: null
  response_format: "json_object"
invocation:
  max_retries: 5
  retry_backoff_seconds: 2
  min_request_interval_seconds: 13
  wait_on_overload_seconds: 60
generator_agent:
  prompt_version: "generator_v1"
  max_tokens: 1024
  system_prompt_file: "config/prompts/generator_system_prompt.txt"
"""
        p = tmp_path / "config" / "mas.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml_content)
        cfg = GeneratorAgentConfig.from_yaml(p, project_root=tmp_path)
        assert cfg.prompt_version == "generator_v1"
        assert cfg.max_tokens == 1024

    def test_missing_prompt_file_raises(self, tmp_path):
        config = GeneratorAgentConfig(
            min_request_interval_seconds=0.0,
            system_prompt_file="config/prompts/NONEXISTENT.txt",
        )
        with pytest.raises(FileNotFoundError, match="NONEXISTENT"):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                UserStoryGeneratorAgent(config=config, project_root=tmp_path)
