"""
tests/unit/baseline/test_baseline_agent.py
==========================================
Unit tests for BaselineAgent — LLM is fully mocked; no API key required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gdd_userstory_mas.baseline.baseline_agent import (
    BaselineAgent,
    BaselineAgentConfig,
    BaselineAgentResult,
    RawStory,
)
from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "baseline_system_prompt.txt"
    p.write_text(
        "You are a baseline agent.\n"
        "Chapter: {chapter}\nSection: {section}\n"
        "Chunk {position} of {total_chunks}.",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def agent_config(prompt_file: Path) -> BaselineAgentConfig:
    return BaselineAgentConfig(
        system_prompt_path=prompt_file,
        max_stories_per_chunk=10,
        prompt_version="baseline_v1",
    )


def _make_chunk(**overrides) -> GDDChunk:
    base = {
        "chunk_id": "doc-001__chunk_0000",
        "document_id": "doc-001",
        "text": "The player can perform a double-jump to reach high platforms.",
        "position": 0,
        "token_count": 13,
        "chapter": "Gameplay",
        "section": "Movement",
    }
    base.update(overrides)
    return GDDChunk(**base)


def _make_mock_client(content: str) -> MagicMock:
    """Return a mock LLMClient that returns ``content`` for any call."""
    mock = MagicMock()
    resp = MagicMock()
    resp.content = content
    resp.prompt_tokens = 120
    resp.completion_tokens = 60
    resp.model_name = "gpt-4o"
    mock.call.return_value = resp
    return mock


_VALID_STORY_JSON = json.dumps({
    "stories": [
        {
            "role": "player",
            "action": "perform a double-jump",
            "benefit": "reach high platforms that would otherwise be inaccessible",
            "full_text": "As a player, I want to perform a double-jump, so that I can reach high platforms that would otherwise be inaccessible.",
            "requirement_type": "player",
            "game_domain": "gameplay",
            "source_excerpt": "The player can perform a double-jump to reach high platforms.",
        }
    ]
})

_EMPTY_STORIES_JSON = '{"stories": []}'


# ── RawStory.from_dict ─────────────────────────────────────────────────────────

class TestRawStoryFromDict:
    def test_valid_dict_produces_no_errors(self) -> None:
        d = {
            "role": "player", "action": "jump", "benefit": "reach high",
            "full_text": "As a player, I want to jump, so that I can reach high.",
            "requirement_type": "player", "game_domain": "gameplay",
            "source_excerpt": "The player jumps.",
        }
        story = RawStory.from_dict(d)
        assert story.parse_errors == []

    def test_missing_role_produces_error(self) -> None:
        d = {
            "role": "", "action": "jump", "benefit": "reach",
            "full_text": "As a player, I want to jump, so that I reach.",
            "requirement_type": "player", "game_domain": "gameplay",
        }
        story = RawStory.from_dict(d)
        assert any("role" in e for e in story.parse_errors)

    def test_missing_action_produces_error(self) -> None:
        d = {
            "role": "player", "action": "", "benefit": "reach",
            "full_text": "As a player, I want , so that I reach.",
            "requirement_type": "player", "game_domain": "gameplay",
        }
        story = RawStory.from_dict(d)
        assert any("action" in e for e in story.parse_errors)

    def test_optional_source_excerpt_is_none(self) -> None:
        d = {
            "role": "player", "action": "jump", "benefit": "reach",
            "full_text": "As a player, I want to jump, so that I reach.",
            "requirement_type": "player", "game_domain": "gameplay",
        }
        story = RawStory.from_dict(d)
        assert story.source_excerpt is None


# ── BaselineAgent.process_chunk ────────────────────────────────────────────────

class TestBaselineAgentProcessChunk:
    def test_returns_baseline_agent_result(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client(_VALID_STORY_JSON), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=5)
        assert isinstance(result, BaselineAgentResult)

    def test_valid_story_extracted(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client(_VALID_STORY_JSON), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=5)
        assert len(result.valid_stories) == 1
        assert result.valid_stories[0].role == "player"

    def test_token_counts_populated(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client(_VALID_STORY_JSON), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=5)
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 60

    def test_model_name_populated(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client(_VALID_STORY_JSON), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=5)
        assert result.model_name == "gpt-4o"

    def test_chunk_id_preserved(self, agent_config: BaselineAgentConfig) -> None:
        chunk = _make_chunk(chunk_id="doc-999__chunk_0042")
        agent = BaselineAgent(_make_mock_client(_VALID_STORY_JSON), agent_config)
        result = agent.process_chunk(chunk, total_chunks=1)
        assert result.chunk_id == "doc-999__chunk_0042"

    def test_empty_stories_returns_zero_valid(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client(_EMPTY_STORIES_JSON), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=1)
        assert result.valid_stories == []
        assert result.invalid_stories == []

    def test_malformed_json_raises_malformed_error(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client("not json"), agent_config)
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.process_chunk(_make_chunk(), total_chunks=1)

    def test_missing_stories_key_raises(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client('{"result": []}'), agent_config)
        with pytest.raises(LLMMalformedOutputError, match="missing 'stories' key"):
            agent.process_chunk(_make_chunk(), total_chunks=1)

    def test_stories_not_a_list_raises(self, agent_config: BaselineAgentConfig) -> None:
        agent = BaselineAgent(_make_mock_client('{"stories": "bad"}'), agent_config)
        with pytest.raises(LLMMalformedOutputError, match="must be a list"):
            agent.process_chunk(_make_chunk(), total_chunks=1)

    def test_invalid_story_dicts_are_in_invalid_list(self, agent_config: BaselineAgentConfig) -> None:
        bad_json = json.dumps({
            "stories": [
                {"role": "", "action": "", "benefit": "x",
                 "full_text": "As a , I want , so that x.", "requirement_type": "player",
                 "game_domain": "gameplay"}
            ]
        })
        agent = BaselineAgent(_make_mock_client(bad_json), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=1)
        assert len(result.invalid_stories) == 1
        assert len(result.valid_stories) == 0

    def test_multiple_stories_extracted(self, agent_config: BaselineAgentConfig) -> None:
        multi_json = json.dumps({
            "stories": [
                {
                    "role": "player", "action": "double-jump",
                    "benefit": "reach platforms",
                    "full_text": "As a player, I want to double-jump, so that I can reach platforms.",
                    "requirement_type": "player", "game_domain": "gameplay",
                    "source_excerpt": "double-jump",
                },
                {
                    "role": "game system", "action": "detect landing",
                    "benefit": "trigger landing animation",
                    "full_text": "As a game system, I want to detect landing, so that I can trigger landing animation.",
                    "requirement_type": "system", "game_domain": "systems",
                    "source_excerpt": "detect landing",
                },
            ]
        })
        agent = BaselineAgent(_make_mock_client(multi_json), agent_config)
        result = agent.process_chunk(_make_chunk(), total_chunks=1)
        assert len(result.valid_stories) == 2

    def test_prompt_template_is_rendered(self, agent_config: BaselineAgentConfig) -> None:
        """Verify that the system prompt sent to the LLM contains the chunk context."""
        mock_client = _make_mock_client(_VALID_STORY_JSON)
        agent = BaselineAgent(mock_client, agent_config)
        chunk = _make_chunk(chapter="Combat", section="Light Attacks", position=2)
        agent.process_chunk(chunk, total_chunks=7)

        call_args = mock_client.call.call_args
        system_prompt_sent = call_args[0][0]
        assert "Combat" in system_prompt_sent
        assert "Light Attacks" in system_prompt_sent

    def test_missing_prompt_file_raises_file_not_found(self, tmp_path: Path) -> None:
        config = BaselineAgentConfig(
            system_prompt_path=tmp_path / "nonexistent.txt",
        )
        with pytest.raises(FileNotFoundError):
            BaselineAgent(MagicMock(), config)
