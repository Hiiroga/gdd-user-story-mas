"""
tests/unit/mas/test_evaluator_agent.py
========================================
Unit tests for EvaluatorAgent (C9 / Agent 6).
Deterministic metrics are tested without mocking.
Hallucination Rate (Baseline) uses a mocked LLMClient.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMResponse
from gdd_userstory_mas.mas.evaluator_agent import (
    GAME_DOMAINS,
    REQUIREMENT_TYPES,
    TOTAL_DIVERSITY_CATEGORIES,
    EvaluatorAgent,
    EvaluatorAgentConfig,
)
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.final_user_story import (
    ConfidenceEvidence,
    FinalUserStory,
)
from gdd_userstory_mas.schemas.redundancy_analysis import (
    DuplicateGroup,
    MergeAction,
    RedundancyAnalysis,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_story(
    story_id: str = "US-0001",
    run_id: str = "run1",
    chunk_id: str = "doc1__chunk_0000",
    requirement_type: str = "player",
    game_domain: str = "gameplay",
    role: str = "player",
    action: str = "attack enemies with melee weapons",
    benefit: str = "defeat them and progress",
    validation_status: str = "Valid",
    reviewer_traceability: Optional[bool] = True,
    pipeline_type: str = "MAS",
    iteration_count: int = 0,
) -> FinalUserStory:
    full_text = f"As a {role}, I want {action}, so that {benefit}."
    confidence = None
    if pipeline_type == "MAS":
        confidence = ConfidenceEvidence(
            reviewer_traceability_confirmed=reviewer_traceability
        )
    return FinalUserStory(
        id=story_id,
        run_id=run_id,
        pipeline_type=pipeline_type,
        source_document_id="doc1",
        source_chunk_ids=[chunk_id],
        role=role,
        action=action,
        benefit=benefit,
        full_text=full_text,
        requirement_type=requirement_type,
        game_domain=game_domain,
        validation_status=validation_status,
        confidence_evidence=confidence,
        iteration_count_final=iteration_count if pipeline_type == "MAS" else None,
    )


def _make_agent(tmp_path: Path) -> EvaluatorAgent:
    config = EvaluatorAgentConfig(
        min_request_interval_seconds=0.0,
        hallucination_prompt_file="config/prompts/evaluator_hallucination_prompt.txt",
    )
    # Create prompt file
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "evaluator_hallucination_prompt.txt").write_text(
        "Review this story.\n"
        "Role: {role}\nAction: {action}\nBenefit: {benefit}\n"
        "Full: {full_text}\nSource: {source_chunk_text}\n",
        encoding="utf-8",
    )
    return EvaluatorAgent(config=config, project_root=tmp_path)


GDD_CHUNKS = {
    "doc1__chunk_0000": "The player can attack enemies using melee weapons.",
    "doc1__chunk_0001": "The game supports saving progress automatically.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Empty collection
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyCollection:

    def test_empty_returns_result(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert isinstance(result, EvaluationResult)

    def test_empty_all_metrics_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.hallucination_rate is None
        assert result.aqusa_score is None
        assert result.coverage is None
        assert result.diversity is None

    def test_empty_null_reasons_populated(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        for m in ["hallucination_rate", "aqusa_score", "coverage", "diversity"]:
            assert m in result.null_reasons


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-003: AQUSA Score
# ─────────────────────────────────────────────────────────────────────────────

class TestAQUSAScore:

    def test_all_passing(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story(f"US-{i:04d}") for i in range(5)]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.aqusa_score == 100.0

    def test_empty_role_pydantic_prevents(self, tmp_path):
        """Pydantic prevents constructing a FinalUserStory with empty role.
        AQUSA score reflects stories that actually have whitespace-only content.
        This test verifies all-valid stories get 100%."""
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", role="player")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.aqusa_score == 100.0

    def test_ambiguity_heuristic(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story(
                "US-0001",
                action="do stuff",
                benefit="it is beneficial",
                # "it is" matches the ambiguity pattern
            ),
        ]
        # Build story with ambiguous full_text manually
        s = stories[0]
        # Patch full_text to include ambiguity
        object.__setattr__(s, 'full_text',
            "As a player, I want do stuff, so that it is beneficial.")
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.aqusa_score is not None
        assert result.aqusa_score < 100.0

    def test_aqusa_detail_populated(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.aqusa_detail is not None
        assert result.aqusa_detail.total_stories == 1


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-004: Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverage:

    def test_full_coverage(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", chunk_id="doc1__chunk_0000"),
            _make_story("US-0002", chunk_id="doc1__chunk_0001"),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 2)
        assert result.coverage == 100.0

    def test_partial_coverage(self, tmp_path):
        agent = _make_agent(tmp_path)
        # Only 1 of 4 chunks covered
        stories = [_make_story("US-0001", chunk_id="doc1__chunk_0000")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 4)
        assert result.coverage == 25.0

    def test_no_chunks_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, {}, "run1", "doc1", "MAS", 5)
        assert result.coverage is None
        assert "coverage" in result.null_reasons

    def test_zero_total_chunks_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 0)
        assert result.coverage is None

    def test_coverage_metric_detail(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", chunk_id="doc1__chunk_0000")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 2)
        md = result.metric_details.get("coverage")
        assert md is not None
        assert md.numerator == 1.0
        assert md.denominator == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-005: Diversity
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversity:

    def test_single_category(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", requirement_type="player", game_domain="gameplay"),
            _make_story("US-0002", requirement_type="player", game_domain="gameplay"),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        # 1 req type + 1 domain = 2 unique / 11 total
        expected = 2 / TOTAL_DIVERSITY_CATEGORIES
        assert abs(result.diversity - expected) < 0.001

    def test_max_diversity(self, tmp_path):
        """All possible categories represented."""
        agent = _make_agent(tmp_path)
        stories = []
        i = 0
        for rt in REQUIREMENT_TYPES:
            for gd in GAME_DOMAINS:
                stories.append(
                    _make_story(f"US-{i:04d}", requirement_type=rt, game_domain=gd)
                )
                i += 1
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.diversity == 1.0

    def test_diversity_metric_detail(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        md = result.metric_details.get("diversity")
        assert md is not None
        assert md.denominator == float(TOTAL_DIVERSITY_CATEGORIES)


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-001: Hallucination Rate (MAS — from Reviewer signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationRateMAS:

    def test_no_hallucinations(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", reviewer_traceability=True),
            _make_story("US-0002", reviewer_traceability=True),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.hallucination_rate == 0.0

    def test_some_hallucinations(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", reviewer_traceability=True),
            _make_story("US-0002", reviewer_traceability=False),
            _make_story("US-0003", reviewer_traceability=False),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        # 2 / 3 = 66.67%
        assert abs(result.hallucination_rate - (2 / 3 * 100)) < 0.01

    def test_unverifiable_excluded(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", reviewer_traceability=True),
            _make_story("US-0002", reviewer_traceability=None),  # unverifiable
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        # 0 / 1 verifiable = 0%
        assert result.hallucination_rate == 0.0
        assert any("unverifiable" in n.lower() for n in result.consistency_notes), (
            f"Expected 'unverifiable' in consistency_notes, got: {result.consistency_notes}"
        )

    def test_all_unverifiable_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001", reviewer_traceability=None),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.hallucination_rate is None
        assert "hallucination_rate" in result.null_reasons

    def test_no_confidence_evidence_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        story = _make_story("US-0001", reviewer_traceability=None)
        object.__setattr__(story, 'confidence_evidence', None)
        result = agent.evaluate([story], GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.hallucination_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# EVAL-002: Redundancy Rate (MAS)
# ─────────────────────────────────────────────────────────────────────────────

def _make_redundancy_analysis(input_count: int, grouped_ids: list) -> RedundancyAnalysis:
    groups = []
    if len(grouped_ids) >= 2:
        groups = [DuplicateGroup(
            group_id="g0",
            story_ids=grouped_ids,
            similarity_basis="llm_judgment",
        )]
    return RedundancyAnalysis(
        analysis_id="redundancy__run1__pass1",
        run_id="run1",
        analyzed_at_pass=1,
        duplicate_groups=groups,
        unique_story_ids=[],
        resolution_status="resolved",
        merge_actions=[],
        resolved_story_count=input_count - len(grouped_ids) + (1 if grouped_ids else 0),
        input_story_count=input_count,
    )


class TestRedundancyRateMAS:

    def test_no_duplicates(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story(f"US-{i:04d}") for i in range(3)]
        ra = _make_redundancy_analysis(3, [])
        result = agent.evaluate(
            stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5, ra
        )
        assert result.redundancy_rate == 0.0

    def test_two_of_four_duplicate(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story(f"US-{i:04d}") for i in range(4)]
        ra = _make_redundancy_analysis(4, ["US-0000", "US-0001"])
        result = agent.evaluate(
            stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5, ra
        )
        # 2 / 4 = 50%
        assert result.redundancy_rate == 50.0

    def test_no_analysis_null(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.redundancy_rate is None
        assert "redundancy_rate" in result.null_reasons

    def test_baseline_null(self, tmp_path):
        """Baseline pipeline: redundancy_rate must be null (no Redundancy Checker)."""
        agent = _make_agent(tmp_path)
        story = _make_story(
            "US-0001",
            pipeline_type="Baseline",
            reviewer_traceability=None,
        )
        # For Baseline hallucination check, patch LLMClient.call to avoid network.
        mock_client = MagicMock()
        mock_client.call.return_value = LLMResponse(
            content='{"is_hallucinated": false, "unsupported_excerpt": null, "confidence": "high"}',
            model_name="gemini-flash-latest",
            prompt_tokens=100,
            completion_tokens=20,
            finish_reason="stop",
        )
        with patch(
            "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
            return_value=MagicMock(),
        ):
            agent._client = mock_client
            result = agent.evaluate(
                [story], GDD_CHUNKS, "run1", "doc1", "Baseline", 5
            )
        assert result.redundancy_rate is None
        assert "redundancy_rate" in result.null_reasons



# ─────────────────────────────────────────────────────────────────────────────
# Consistency Notes
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyNotes:

    def test_rejected_stories_noted(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [
            _make_story("US-0001"),
            _make_story("US-0002", validation_status="Rejected-ManualReview"),
        ]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert any("Rejected-ManualReview" in n for n in result.consistency_notes)

    def test_merged_stories_noted(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", validation_status="Merged")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert any("Merged" in n or "merge" in n.lower() for n in result.consistency_notes)

    def test_unresolved_stories_noted(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", validation_status="Redundancy-Unresolved")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert any("Redundancy-Unresolved" in n for n in result.consistency_notes)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation ID and traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceability:

    def test_evaluation_id_format(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run_abc", "doc1", "MAS", 5)
        assert result.evaluation_id == "eval__run_abc"

    def test_run_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "my_run", "doc1", "MAS", 5)
        assert result.run_id == "my_run"

    def test_document_id_preserved(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run1", "my_doc", "MAS", 5)
        assert result.document_id == "my_doc"

    def test_computed_at_is_set(self, tmp_path):
        agent = _make_agent(tmp_path)
        result = agent.evaluate([], GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        assert result.computed_at
        assert "T" in result.computed_at  # ISO format


# ─────────────────────────────────────────────────────────────────────────────
# Metric detail auditability
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricDetails:

    def test_all_details_present(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", reviewer_traceability=True)]
        ra = _make_redundancy_analysis(1, [])
        result = agent.evaluate(
            stories, GDD_CHUNKS, "run1", "doc1", "MAS", 2, ra
        )
        for m in ["aqusa_score", "coverage", "diversity",
                  "redundancy_rate", "hallucination_rate"]:
            assert m in result.metric_details

    def test_formula_documented(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001")]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        for md in result.metric_details.values():
            assert md.formula

    def test_limitation_documented(self, tmp_path):
        agent = _make_agent(tmp_path)
        stories = [_make_story("US-0001", reviewer_traceability=True)]
        result = agent.evaluate(stories, GDD_CHUNKS, "run1", "doc1", "MAS", 5)
        for md in result.metric_details.values():
            assert md.limitation or md.value is None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        config = EvaluatorAgentConfig.defaults()
        assert config.provider == "gemini"
        assert config.max_tokens == 512
        assert config.count_unresolved_in_redundancy is False

    def test_from_yaml(self, tmp_path):
        yaml_content = (
            "model:\n"
            "  provider: gemini\n"
            "  name: gemini-flash-latest\n"
            "  temperature: 0\n"
            "invocation:\n"
            "  max_retries: 3\n"
            "evaluator_agent:\n"
            "  max_tokens: 1024\n"
            "  count_unresolved_in_redundancy: true\n"
        )
        yaml_file = tmp_path / "config" / "mas.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = EvaluatorAgentConfig.from_yaml(yaml_file)
        assert config.max_tokens == 1024
        assert config.count_unresolved_in_redundancy is True
