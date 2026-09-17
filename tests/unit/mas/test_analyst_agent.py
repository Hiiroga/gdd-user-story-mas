"""
tests/unit/mas/test_analyst_agent.py
======================================
Unit tests for RequirementsAnalystAgent (C5).
All tests use a mock LLMClient — no real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError, LLMResponse
from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig, RequirementsAnalystAgent
from gdd_userstory_mas.schemas.candidate_requirement import AnalystOutput
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_name="gemini-flash-latest",
        prompt_tokens=150,
        completion_tokens=80,
        finish_reason="stop",
    )


def _make_reader_output(**kwargs) -> ReaderOutput:
    defaults = dict(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        themes=["combat", "survival"],
        gameplay_elements=[
            EvidencedItem(content="Melee weapon attacks", source_excerpt="attack enemies"),
        ],
        systems=[],
        characters=[EvidencedItem(content="Player character", source_excerpt="the player")],
        ui_elements=[EvidencedItem(content="Health bar in HUD", source_excerpt="HUD health bar")],
        narrative=[],
    )
    defaults.update(kwargs)
    return ReaderOutput(**defaults)


def _make_agent(tmp_path: Path, mock_llm_content: str) -> RequirementsAnalystAgent:
    """Build agent with mocked LLMClient and real prompt file."""
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "analyst_system_prompt.txt").write_text(
        "Analyse the GDD elements.\n"
        "Chapter: {chapter}\nSection: {section}\n"
        "Chunk: {position}/{total_chunks}\n",
        encoding="utf-8",
    )
    config = AnalystAgentConfig(
        min_request_interval_seconds=0.0,
        system_prompt_file="config/prompts/analyst_system_prompt.txt",
    )
    with patch(
        "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
        return_value=MagicMock(),
    ):
        agent = RequirementsAnalystAgent(config=config, project_root=tmp_path)

    agent._client.call = MagicMock(return_value=_make_llm_response(mock_llm_content))
    return agent


def _valid_llm_response() -> str:
    return json.dumps({
        "candidates": [
            {
                "requirement_text": "The game must allow the player to attack enemies using melee weapons.",
                "perspective": "player",
                "domain": "gameplay",
                "source_reader_output_ref": "Melee weapon attacks",
            },
            {
                "requirement_text": "The HUD must display the player's current health at all times.",
                "perspective": "player",
                "domain": "ui",
                "source_reader_output_ref": "Health bar in HUD",
            },
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessReaderOutput:

    def test_returns_analyst_output(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert isinstance(result, AnalystOutput)

    def test_traceability_source_chunk_id(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert result.source_chunk_id == "doc1__chunk_0000"

    def test_traceability_document_id(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert result.document_id == "doc1"

    def test_correct_candidate_count(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 2

    def test_candidate_ids_are_deterministic(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output(), id_offset=0)
        assert result.candidates[0].candidate_id == "doc1__cand_0000"
        assert result.candidates[1].candidate_id == "doc1__cand_0001"

    def test_id_offset_applied(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output(), id_offset=10)
        assert result.candidates[0].candidate_id == "doc1__cand_0010"
        assert result.candidates[1].candidate_id == "doc1__cand_0011"

    def test_all_candidates_have_correct_source_chunk_id(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        for c in result.candidates:
            assert c.source_chunk_id == "doc1__chunk_0000"

    def test_perspective_and_domain_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidates[0].perspective == "player"
        assert result.candidates[0].domain == "gameplay"
        assert result.candidates[1].domain == "ui"

    def test_source_ref_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidates[0].source_reader_output_ref == "Melee weapon attacks"

    def test_empty_candidates_valid(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({"candidates": []}))
        result = agent.process_reader_output(_make_reader_output())
        assert result.is_empty

    def test_empty_reader_output_skips_llm(self, tmp_path):
        """When ReaderOutput is empty, no LLM call should be made."""
        agent = _make_agent(tmp_path, _valid_llm_response())
        empty_ro = ReaderOutput(
            chunk_id="doc1__chunk_0000",
            document_id="doc1",
            themes=[], gameplay_elements=[], systems=[],
            characters=[], ui_elements=[], narrative=[],
        )
        result = agent.process_reader_output(empty_ro)
        agent._client.call.assert_not_called()
        assert result.is_empty


# ─────────────────────────────────────────────────────────────────────────────
# Error handling & normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalisation:

    def test_user_story_format_skipped(self, tmp_path):
        """Candidates starting with 'As a' should be silently skipped."""
        payload = json.dumps({
            "candidates": [
                {
                    "requirement_text": "As a player, I want to see my health.",
                    "perspective": "player",
                    "domain": "ui",
                    "source_reader_output_ref": None,
                },
                {
                    "requirement_text": "The HUD must show current health.",
                    "perspective": "player",
                    "domain": "ui",
                    "source_reader_output_ref": None,
                },
            ]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 1
        assert result.candidates[0].requirement_text == "The HUD must show current health."

    def test_invalid_perspective_defaults_to_system(self, tmp_path):
        payload = json.dumps({
            "candidates": [{
                "requirement_text": "Something must happen.",
                "perspective": "protagonist",  # invalid
                "domain": "gameplay",
                "source_reader_output_ref": None,
            }]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 1
        assert result.candidates[0].perspective == "system"

    def test_invalid_domain_defaults_to_other(self, tmp_path):
        payload = json.dumps({
            "candidates": [{
                "requirement_text": "Something must happen.",
                "perspective": "system",
                "domain": "economy",  # invalid
                "source_reader_output_ref": None,
            }]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidates[0].domain == "other"

    def test_duplicate_requirement_texts_deduplicated(self, tmp_path):
        payload = json.dumps({
            "candidates": [
                {"requirement_text": "Health must be shown.", "perspective": "player",
                 "domain": "ui", "source_reader_output_ref": None},
                {"requirement_text": "Health must be shown.", "perspective": "system",
                 "domain": "systems", "source_reader_output_ref": None},
            ]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 1

    def test_non_dict_candidate_skipped(self, tmp_path):
        payload = json.dumps({
            "candidates": [
                "NOT A DICT",
                {"requirement_text": "Valid requirement.", "perspective": "player",
                 "domain": "gameplay", "source_reader_output_ref": None},
            ]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 1

    def test_empty_requirement_text_skipped(self, tmp_path):
        payload = json.dumps({
            "candidates": [
                {"requirement_text": "", "perspective": "player",
                 "domain": "gameplay", "source_reader_output_ref": None},
                {"requirement_text": "Valid requirement.", "perspective": "player",
                 "domain": "gameplay", "source_reader_output_ref": None},
            ]
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.candidate_count == 1

    def test_non_list_candidates_treated_as_empty(self, tmp_path):
        payload = json.dumps({"candidates": "not a list"})
        agent = _make_agent(tmp_path, payload)
        result = agent.process_reader_output(_make_reader_output())
        assert result.is_empty

    def test_malformed_json_raises(self, tmp_path):
        agent = _make_agent(tmp_path, "NOT JSON {{{")
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.process_reader_output(_make_reader_output())


# ─────────────────────────────────────────────────────────────────────────────
# User message construction
# ─────────────────────────────────────────────────────────────────────────────

class TestUserMessage:

    def test_user_message_contains_chunk_id(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({"candidates": []}))
        ro = _make_reader_output()
        msg = agent._build_user_message(ro)
        assert "doc1__chunk_0000" in msg

    def test_user_message_contains_items(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({"candidates": []}))
        ro = _make_reader_output()
        msg = agent._build_user_message(ro)
        assert "Melee weapon attacks" in msg
        assert "Health bar in HUD" in msg

    def test_user_message_contains_themes(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({"candidates": []}))
        ro = _make_reader_output(themes=["combat", "survival"])
        msg = agent._build_user_message(ro)
        assert "combat" in msg
        assert "survival" in msg

    def test_empty_reader_output_message_minimal(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({"candidates": []}))
        ro = ReaderOutput(
            chunk_id="c1", document_id="d1",
            themes=[], gameplay_elements=[], systems=[],
            characters=[], ui_elements=[], narrative=[],
        )
        msg = agent._build_user_message(ro)
        assert "c1" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalystAgentConfig:

    def test_defaults(self):
        cfg = AnalystAgentConfig.defaults()
        assert cfg.provider == "gemini"
        assert cfg.temperature == 0.0
        assert cfg.prompt_version == "analyst_v1"

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
analyst_agent:
  prompt_version: "analyst_v1"
  system_prompt_file: "config/prompts/analyst_system_prompt.txt"
"""
        p = tmp_path / "config" / "mas.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml_content)
        cfg = AnalystAgentConfig.from_yaml(p, project_root=tmp_path)
        assert cfg.model_name == "gemini-flash-latest"
        assert cfg.prompt_version == "analyst_v1"
        assert cfg.min_request_interval_seconds == 13.0

    def test_missing_prompt_file_raises(self, tmp_path):
        config = AnalystAgentConfig(
            min_request_interval_seconds=0.0,
            system_prompt_file="config/prompts/NONEXISTENT.txt",
        )
        with pytest.raises(FileNotFoundError, match="NONEXISTENT"):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                RequirementsAnalystAgent(config=config, project_root=tmp_path)
