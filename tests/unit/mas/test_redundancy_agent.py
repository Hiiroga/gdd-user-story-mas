"""
tests/unit/mas/test_redundancy_agent.py
=========================================
Unit tests for RedundancyCheckerAgent (C8 / Agent 5).
All tests use a mock LLMClient — no real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError, LLMResponse
from gdd_userstory_mas.mas.redundancy_agent import (
    RedundancyAgentConfig,
    RedundancyCheckerAgent,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_name="gemini-flash-latest",
        prompt_tokens=500,
        completion_tokens=200,
        finish_reason="stop",
    )


def _make_story(
    draft_id: str = "doc1__cand_0000__draft",
    role: str = "player",
    action: str = "attack enemies using melee weapons",
    benefit: str = "I can defeat them",
    chunk_id: str = "doc1__chunk_0000",
) -> GeneratedUserStory:
    full_text = f"As a {role}, I want {action}, so that {benefit}."
    return GeneratedUserStory(
        draft_id=draft_id,
        candidate_id=draft_id.replace("__draft", ""),
        source_chunk_id=chunk_id,
        document_id="doc1",
        role=role,
        action=action,
        benefit=benefit,
        full_text=full_text,
        iteration_count=0,
    )


def _make_stories() -> list:
    """Create a collection of 4 stories, 2 of which are redundant."""
    return [
        _make_story("d1__draft", "player", "attack enemies with swords", "defeat them", "c1"),
        _make_story("d2__draft", "player", "attack foes using melee weapons", "defeat enemies", "c2"),
        _make_story("d3__draft", "player", "jump over obstacles", "navigate the level", "c1"),
        _make_story("d4__draft", "game system", "save progress automatically", "player does not lose data", "c3"),
    ]


def _make_agent(tmp_path: Path, mock_llm_content: str) -> RedundancyCheckerAgent:
    """Build agent with mocked LLMClient and real prompt file."""
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "redundancy_system_prompt.txt").write_text(
        "Analyze stories for redundancy.\n"
        "Stories:\n{stories_text}\n",
        encoding="utf-8",
    )
    config = RedundancyAgentConfig(
        min_request_interval_seconds=0.0,
        system_prompt_file="config/prompts/redundancy_system_prompt.txt",
    )
    with patch(
        "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
        return_value=MagicMock(),
    ):
        agent = RedundancyCheckerAgent(config=config, project_root=tmp_path)

    agent._client.call = MagicMock(return_value=_make_llm_response(mock_llm_content))
    return agent


def _no_duplicates_response() -> str:
    return json.dumps({
        "duplicate_groups": [],
        "resolutions": [],
    })


def _one_duplicate_remove_response() -> str:
    return json.dumps({
        "duplicate_groups": [
            {
                "group_id": "dup_group_0",
                "story_ids": ["d1__draft", "d2__draft"],
                "similarity_basis": "llm_judgment",
                "reason": "Both describe attacking enemies with melee.",
            }
        ],
        "resolutions": [
            {
                "group_id": "dup_group_0",
                "action": "remove_inferior",
                "kept_story_id": "d1__draft",
                "removed_story_ids": ["d2__draft"],
                "merged_role": None,
                "merged_action": None,
                "merged_benefit": None,
                "reason": "d1 is more specific (mentions swords).",
            }
        ],
    })


def _one_duplicate_merge_response() -> str:
    return json.dumps({
        "duplicate_groups": [
            {
                "group_id": "dup_group_0",
                "story_ids": ["d1__draft", "d2__draft"],
                "similarity_basis": "llm_judgment",
                "reason": "Both describe melee combat.",
            }
        ],
        "resolutions": [
            {
                "group_id": "dup_group_0",
                "action": "merge",
                "kept_story_id": "d1__draft",
                "removed_story_ids": ["d2__draft"],
                "merged_role": "player",
                "merged_action": "attack enemies using melee weapons including swords",
                "merged_benefit": "defeat them and progress",
                "reason": "Combined melee attack details.",
            }
        ],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — empty/singleton
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_stories(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        result = agent.check([], "run1")
        assert isinstance(result, RedundancyAnalysis)
        assert result.resolution_status == "resolved"
        assert result.resolved_story_count == 0
        assert result.input_story_count == 0
        # LLM should NOT be called
        agent._client.call.assert_not_called()

    def test_single_story(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        stories = [_make_story()]
        result = agent.check(stories, "run1")
        assert result.resolution_status == "resolved"
        assert result.resolved_story_count == 1
        assert result.unique_story_ids == [stories[0].draft_id]
        agent._client.call.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# No duplicates found
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDuplicates:

    def test_all_unique(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        assert result.resolution_status == "resolved"
        assert len(result.duplicate_groups) == 0
        assert len(result.merge_actions) == 0
        assert result.resolved_story_count == 4
        assert set(result.unique_story_ids) == {s.draft_id for s in stories}


# ─────────────────────────────────────────────────────────────────────────────
# Duplicates found — remove inferior
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveInferior:

    def test_one_group_remove(self, tmp_path):
        agent = _make_agent(tmp_path, _one_duplicate_remove_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        assert len(result.duplicate_groups) == 1
        assert len(result.merge_actions) == 1
        assert result.merge_actions[0].action == "remove_inferior"
        assert result.merge_actions[0].kept_story_id == "d1__draft"
        assert result.merge_actions[0].removed_story_ids == ["d2__draft"]

    def test_traceability_source_chunks(self, tmp_path):
        agent = _make_agent(tmp_path, _one_duplicate_remove_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        action = result.merge_actions[0]
        assert "c1" in action.source_chunk_ids
        assert "c2" in action.source_chunk_ids

    def test_unique_stories_excludes_grouped(self, tmp_path):
        agent = _make_agent(tmp_path, _one_duplicate_remove_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        assert "d1__draft" not in result.unique_story_ids
        assert "d2__draft" not in result.unique_story_ids
        assert "d3__draft" in result.unique_story_ids
        assert "d4__draft" in result.unique_story_ids


# ─────────────────────────────────────────────────────────────────────────────
# Duplicates found — merge
# ─────────────────────────────────────────────────────────────────────────────

class TestMerge:

    def test_one_group_merge(self, tmp_path):
        agent = _make_agent(tmp_path, _one_duplicate_merge_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        assert len(result.merge_actions) == 1
        action = result.merge_actions[0]
        assert action.action == "merge"
        assert action.merged_role == "player"
        assert action.merged_action is not None
        assert action.merged_benefit is not None
        assert action.merged_full_text is not None
        assert action.merged_full_text.startswith("As a ")

    def test_merge_full_text_format(self, tmp_path):
        agent = _make_agent(tmp_path, _one_duplicate_merge_response())
        stories = _make_stories()
        result = agent.check(stories, "run1")
        action = result.merge_actions[0]
        assert "I want" in action.merged_full_text
        assert "so that" in action.merged_full_text
        assert action.merged_full_text.endswith(".")


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicIds:

    def test_analysis_id_format(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        result = agent.check(_make_stories(), "run_abc")
        assert result.analysis_id == "redundancy__run_abc__pass1"

    def test_run_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        result = agent.check(_make_stories(), "my_run_123")
        assert result.run_id == "my_run_123"


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_non_json_raises(self, tmp_path):
        agent = _make_agent(tmp_path, "not json")
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.check(_make_stories(), "run1")

    def test_groups_not_list_raises(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({
            "duplicate_groups": "not a list",
            "resolutions": [],
        }))
        with pytest.raises(LLMMalformedOutputError, match="list"):
            agent.check(_make_stories(), "run1")

    def test_unknown_story_ids_filtered(self, tmp_path):
        """Groups with unknown story IDs are filtered to valid ones only."""
        response = json.dumps({
            "duplicate_groups": [
                {
                    "group_id": "g0",
                    "story_ids": ["d1__draft", "nonexistent_draft"],
                    "similarity_basis": "llm_judgment",
                    "reason": "test",
                }
            ],
            "resolutions": [],
        })
        agent = _make_agent(tmp_path, response)
        result = agent.check(_make_stories(), "run1")
        # Group should be skipped (only 1 valid ID < 2)
        assert len(result.duplicate_groups) == 0

    def test_unknown_group_in_resolution_skipped(self, tmp_path):
        response = json.dumps({
            "duplicate_groups": [],
            "resolutions": [
                {
                    "group_id": "nonexistent_group",
                    "action": "remove_inferior",
                    "kept_story_id": "d1__draft",
                    "removed_story_ids": ["d2__draft"],
                    "reason": "test",
                }
            ],
        })
        agent = _make_agent(tmp_path, response)
        result = agent.check(_make_stories(), "run1")
        assert len(result.merge_actions) == 0

    def test_merge_without_fields_falls_back(self, tmp_path):
        """Merge with missing merged fields → fallback to remove_inferior."""
        response = json.dumps({
            "duplicate_groups": [
                {
                    "group_id": "g0",
                    "story_ids": ["d1__draft", "d2__draft"],
                    "similarity_basis": "llm_judgment",
                    "reason": "same thing",
                }
            ],
            "resolutions": [
                {
                    "group_id": "g0",
                    "action": "merge",
                    "kept_story_id": "d1__draft",
                    "removed_story_ids": ["d2__draft"],
                    "merged_role": "",
                    "merged_action": "",
                    "merged_benefit": "",
                    "reason": "incomplete merge",
                }
            ],
        })
        agent = _make_agent(tmp_path, response)
        result = agent.check(_make_stories(), "run1")
        assert len(result.merge_actions) == 1
        assert result.merge_actions[0].action == "remove_inferior"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptRendering:

    def test_system_prompt_contains_stories(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        stories = _make_stories()
        rendered = agent._render_system_prompt(stories)
        for s in stories:
            assert s.draft_id in rendered

    def test_stories_text_format(self, tmp_path):
        agent = _make_agent(tmp_path, _no_duplicates_response())
        stories = _make_stories()
        text = agent._format_stories_text(stories)
        assert "[1]" in text
        assert "[4]" in text

    def test_missing_prompt_file_raises(self, tmp_path):
        config = RedundancyAgentConfig(
            system_prompt_file="nonexistent.txt",
        )
        with pytest.raises(FileNotFoundError):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                RedundancyCheckerAgent(config=config, project_root=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        config = RedundancyAgentConfig.defaults()
        assert config.provider == "gemini"
        assert config.prompt_version == "redundancy_v1"
        assert config.max_resolution_passes == 2
        assert config.max_tokens == 4096

    def test_from_yaml(self, tmp_path):
        yaml_content = (
            "model:\n"
            "  provider: gemini\n"
            "  name: gemini-flash-latest\n"
            "  temperature: 0\n"
            "  max_tokens: 4096\n"
            "invocation:\n"
            "  max_retries: 3\n"
            "redundancy_agent:\n"
            "  prompt_version: redundancy_v2\n"
            "  max_tokens: 8192\n"
            "  max_resolution_passes: 3\n"
        )
        yaml_file = tmp_path / "config" / "mas.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = RedundancyAgentConfig.from_yaml(yaml_file)
        assert config.prompt_version == "redundancy_v2"
        assert config.max_tokens == 8192
        assert config.max_resolution_passes == 3

    def test_from_yaml_missing_section(self, tmp_path):
        yaml_content = "model:\n  provider: gemini\n"
        yaml_file = tmp_path / "config" / "mas.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = RedundancyAgentConfig.from_yaml(yaml_file)
        assert config.prompt_version == "redundancy_v1"
        assert config.max_resolution_passes == 2
