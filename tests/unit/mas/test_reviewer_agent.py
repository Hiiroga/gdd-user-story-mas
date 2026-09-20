"""
tests/unit/mas/test_reviewer_agent.py
=======================================
Unit tests for ReviewerAgent (C7).
All tests use a mock LLMClient — no real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError, LLMResponse
from gdd_userstory_mas.mas.reviewer_agent import (
    ReviewerAgent,
    ReviewerAgentConfig,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reviewer_result import ReviewerFeedback, ReviewerResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_name="gemini-flash-latest",
        prompt_tokens=200,
        completion_tokens=100,
        finish_reason="stop",
    )


def _make_draft(**kwargs) -> GeneratedUserStory:
    defaults = dict(
        draft_id="doc1__cand_0000__draft",
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        role="player",
        action="attack enemies using melee weapons",
        benefit="I can defeat them and progress through the level",
        full_text="As a player, I want attack enemies using melee weapons, so that I can defeat them and progress through the level.",
        iteration_count=0,
        previous_draft_id_chain=[],
    )
    defaults.update(kwargs)
    return GeneratedUserStory(**defaults)


SOURCE_CHUNK_TEXT = (
    "The player can attack enemies using various melee weapons such as "
    "swords and axes. Defeating enemies allows progression through levels."
)


def _make_agent(tmp_path: Path, mock_llm_content: str) -> ReviewerAgent:
    """Build agent with mocked LLMClient and real prompt file."""
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "reviewer_system_prompt.txt").write_text(
        "Review this user story draft.\n"
        "Draft ID: {draft_id}\n"
        "Candidate: {candidate_id}\n"
        "Source chunk: {source_chunk_id}\n"
        "Role: {role}\nAction: {action}\nBenefit: {benefit}\n"
        "Full text: {full_text}\n"
        "Source: {source_chunk_text}\n",
        encoding="utf-8",
    )
    config = ReviewerAgentConfig(
        min_request_interval_seconds=0.0,
        system_prompt_file="config/prompts/reviewer_system_prompt.txt",
    )
    with patch(
        "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
        return_value=MagicMock(),
    ):
        agent = ReviewerAgent(config=config, project_root=tmp_path)

    agent._client.call = MagicMock(return_value=_make_llm_response(mock_llm_content))
    return agent


def _valid_llm_response() -> str:
    return json.dumps({
        "status": "Valid",
        "failing_criteria": [],
        "feedback_text": "",
        "unsupported_claim_excerpt": None,
    })


def _invalid_llm_response(
    criteria: list | None = None,
    feedback: str = "The role is too vague. Use 'player' instead of 'user'.",
    excerpt: str | None = None,
) -> str:
    return json.dumps({
        "status": "Invalid",
        "failing_criteria": criteria or ["completeness"],
        "feedback_text": feedback,
        "unsupported_claim_excerpt": excerpt,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core behaviour — Valid verdict
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewValid:

    def test_returns_tuple(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_valid_returns_result_and_none(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert isinstance(result, ReviewerResult)
        assert feedback is None

    def test_valid_status(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.status == "Valid"
        assert result.failing_criteria == []

    def test_valid_no_feedback_ref(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.feedback_ref is None


# ─────────────────────────────────────────────────────────────────────────────
# Core behaviour — Invalid verdict
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewInvalid:

    def test_invalid_returns_result_and_feedback(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert isinstance(result, ReviewerResult)
        assert isinstance(feedback, ReviewerFeedback)

    def test_invalid_status(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.status == "Invalid"
        assert result.failing_criteria == ["completeness"]

    def test_invalid_feedback_text(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        _, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert "role" in feedback.feedback_text.lower()

    def test_invalid_multiple_criteria(self, tmp_path):
        agent = _make_agent(
            tmp_path,
            _invalid_llm_response(
                criteria=["completeness", "clarity", "gdd_traceability"],
                feedback="Multiple issues found.",
            ),
        )
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert len(result.failing_criteria) == 3
        assert len(feedback.failing_criteria) == 3

    def test_invalid_with_unsupported_claim(self, tmp_path):
        agent = _make_agent(
            tmp_path,
            _invalid_llm_response(
                criteria=["gdd_traceability"],
                feedback="Unsupported claim detected.",
                excerpt="multiplayer lobby system",
            ),
        )
        _, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert feedback.unsupported_claim_excerpt == "multiplayer lobby system"

    def test_invalid_feedback_ref_populated(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.feedback_ref is not None
        assert "feedback" in result.feedback_ref


# ─────────────────────────────────────────────────────────────────────────────
# Traceability — deterministic IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceability:

    def test_review_id_format(self, tmp_path):
        """review_id = {draft_id}__review_iter{N}."""
        agent = _make_agent(tmp_path, _valid_llm_response())
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.review_id == "doc1__cand_0000__draft__review_iter0"

    def test_review_id_at_iteration(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft(iteration_count=3)
        result, _ = agent.review(draft, SOURCE_CHUNK_TEXT)
        assert result.review_id == "doc1__cand_0000__draft__review_iter3"

    def test_feedback_id_format(self, tmp_path):
        """feedback_id = {review_id}__feedback."""
        agent = _make_agent(tmp_path, _invalid_llm_response())
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert feedback.feedback_id == f"{result.review_id}__feedback"

    def test_draft_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft()
        result, _ = agent.review(draft, SOURCE_CHUNK_TEXT)
        assert result.draft_id == draft.draft_id

    def test_evaluated_at_iteration(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft(iteration_count=2)
        result, _ = agent.review(draft, SOURCE_CHUNK_TEXT)
        assert result.evaluated_at_iteration == 2

    def test_feedback_draft_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        draft = _make_draft()
        _, feedback = agent.review(draft, SOURCE_CHUNK_TEXT)
        assert feedback.draft_id == draft.draft_id

    def test_feedback_review_id_matches_result(self, tmp_path):
        agent = _make_agent(tmp_path, _invalid_llm_response())
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert feedback.review_id == result.review_id


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_non_json_raises(self, tmp_path):
        agent = _make_agent(tmp_path, "not json at all")
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.review(_make_draft(), SOURCE_CHUNK_TEXT)

    def test_invalid_status_raises(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Maybe",
            "failing_criteria": [],
            "feedback_text": "",
        }))
        with pytest.raises(LLMMalformedOutputError, match="status"):
            agent.review(_make_draft(), SOURCE_CHUNK_TEXT)

    def test_criteria_not_list_raises(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Invalid",
            "failing_criteria": "completeness",
            "feedback_text": "fix it",
        }))
        with pytest.raises(LLMMalformedOutputError, match="list"):
            agent.review(_make_draft(), SOURCE_CHUNK_TEXT)

    def test_invalid_no_criteria_raises_val023(self, tmp_path):
        """VAL-023: Invalid + empty criteria from LLM → error."""
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Invalid",
            "failing_criteria": [],
            "feedback_text": "something",
        }))
        with pytest.raises(LLMMalformedOutputError, match="VAL-023"):
            agent.review(_make_draft(), SOURCE_CHUNK_TEXT)

    def test_unknown_criteria_ignored_with_warning(self, tmp_path):
        """Unknown criteria are filtered out silently."""
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Invalid",
            "failing_criteria": ["completeness", "nonexistent_criterion"],
            "feedback_text": "fix it",
        }))
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.failing_criteria == ["completeness"]

    def test_valid_with_criteria_clears_criteria(self, tmp_path):
        """If LLM says Valid but provides criteria, trust Valid verdict."""
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Valid",
            "failing_criteria": ["clarity"],
            "feedback_text": "",
        }))
        result, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.status == "Valid"
        assert result.failing_criteria == []
        assert feedback is None

    def test_duplicate_criteria_deduplicated(self, tmp_path):
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Invalid",
            "failing_criteria": ["completeness", "completeness", "clarity"],
            "feedback_text": "fix both",
        }))
        result, _ = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert result.failing_criteria == ["completeness", "clarity"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptRendering:

    def test_system_prompt_contains_draft_fields(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft()
        rendered = agent._render_system_prompt(draft, "some GDD text")
        assert draft.draft_id in rendered
        assert draft.role in rendered
        assert draft.action in rendered
        assert "some GDD text" in rendered

    def test_user_message_contains_draft(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft()
        msg = agent._build_user_message(draft, "chunk text")
        assert draft.draft_id in msg
        assert "chunk text" in msg
        assert draft.full_text in msg

    def test_missing_prompt_file_raises(self, tmp_path):
        config = ReviewerAgentConfig(
            system_prompt_file="config/prompts/nonexistent.txt",
        )
        with pytest.raises(FileNotFoundError):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                ReviewerAgent(config=config, project_root=tmp_path)

    def test_empty_prompt_file_raises(self, tmp_path):
        prompts_dir = tmp_path / "config" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "reviewer_system_prompt.txt").write_text("", encoding="utf-8")
        config = ReviewerAgentConfig(
            system_prompt_file="config/prompts/reviewer_system_prompt.txt",
        )
        with pytest.raises(ValueError, match="empty"):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                ReviewerAgent(config=config, project_root=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        config = ReviewerAgentConfig.defaults()
        assert config.provider == "gemini"
        assert config.prompt_version == "reviewer_v1"
        assert config.max_tokens == 2048

    def test_from_yaml(self, tmp_path):
        yaml_content = (
            "model:\n"
            "  provider: gemini\n"
            "  name: gemini-flash-latest\n"
            "  temperature: 0\n"
            "  max_tokens: 4096\n"
            "  response_format: json_object\n"
            "invocation:\n"
            "  max_retries: 3\n"
            "  min_request_interval_seconds: 10\n"
            "reviewer_agent:\n"
            "  prompt_version: reviewer_v2\n"
            "  max_tokens: 2048\n"
            "  system_prompt_file: config/prompts/reviewer_system_prompt.txt\n"
        )
        yaml_file = tmp_path / "config" / "mas.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = ReviewerAgentConfig.from_yaml(yaml_file)
        assert config.provider == "gemini"
        assert config.prompt_version == "reviewer_v2"
        assert config.max_tokens == 2048
        assert config.max_retries == 3

    def test_from_yaml_missing_reviewer_section(self, tmp_path):
        """Should use defaults if reviewer_agent section is absent."""
        yaml_content = (
            "model:\n"
            "  provider: gemini\n"
            "  name: gemini-flash-latest\n"
        )
        yaml_file = tmp_path / "config" / "mas.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = ReviewerAgentConfig.from_yaml(yaml_file)
        assert config.prompt_version == "reviewer_v1"


# ─────────────────────────────────────────────────────────────────────────────
# LLM client interaction
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMInteraction:

    def test_llm_call_receives_system_and_user_message(self, tmp_path):
        agent = _make_agent(tmp_path, _valid_llm_response())
        draft = _make_draft()
        agent.review(draft, SOURCE_CHUNK_TEXT)
        agent._client.call.assert_called_once()
        args = agent._client.call.call_args
        system_msg = args[0][0]
        user_msg = args[0][1]
        assert draft.draft_id in system_msg
        assert draft.draft_id in user_msg

    def test_default_feedback_when_llm_gives_empty_text(self, tmp_path):
        """If LLM gives Invalid but empty feedback_text, agent provides default."""
        agent = _make_agent(tmp_path, json.dumps({
            "status": "Invalid",
            "failing_criteria": ["clarity"],
            "feedback_text": "",
        }))
        _, feedback = agent.review(_make_draft(), SOURCE_CHUNK_TEXT)
        assert "clarity" in feedback.feedback_text
