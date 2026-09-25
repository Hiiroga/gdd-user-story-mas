"""
tests/unit/mas/test_evaluation_metrics.py
==========================================
Unit tests for the research evaluation metric calculation functions.

Tests verify:
  - Correct formula application (EVAL-001 to EVAL-005)
  - Edge cases: empty collections, zero denominators, missing fields
  - Raw observation counts match input count
  - null_reason is set whenever value is None (VAL-029)
  - Formula version strings are stable (used for reproducibility tracking)

All tests use synthetic data — no I/O, no LLM calls.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List, Optional

from gdd_userstory_mas.evaluation.metrics import (
    FORMULA_VERSION_AQUSA,
    FORMULA_VERSION_COVERAGE,
    FORMULA_VERSION_DIVERSITY,
    FORMULA_VERSION_HALLUCINATION_BASELINE,
    FORMULA_VERSION_HALLUCINATION_MAS,
    FORMULA_VERSION_REDUNDANCY_BASELINE,
    FORMULA_VERSION_REDUNDANCY_MAS,
    TOTAL_DIVERSITY_CATEGORIES,
    compute_aqusa_score,
    compute_coverage,
    compute_diversity,
    compute_hallucination_rate_baseline_heuristic,
    compute_hallucination_rate_mas,
    compute_redundancy_rate_baseline,
    compute_redundancy_rate_mas,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_story(
    id: str = "US-0001",
    role: str = "player",
    action: str = "navigate the main menu",
    benefit: str = "I can access all game options",
    full_text: Optional[str] = None,
    requirement_type: str = "player",
    game_domain: str = "UI",
    source_chunk_ids: Optional[List[str]] = None,
    reviewer_traceability: Optional[bool] = True,
    source_excerpt: Optional[str] = None,
    pipeline_type: str = "MAS",
) -> Dict[str, Any]:
    if full_text is None:
        full_text = f"As a {role}, I want {action}, so that {benefit}."
    if source_chunk_ids is None:
        source_chunk_ids = ["chunk_0001"]
    return {
        "id": id,
        "role": role,
        "action": action,
        "benefit": benefit,
        "full_text": full_text,
        "requirement_type": requirement_type,
        "game_domain": game_domain,
        "source_chunk_ids": source_chunk_ids,
        "pipeline_type": pipeline_type,
        "confidence_evidence": {
            "reviewer_traceability_confirmed": reviewer_traceability,
            "source_excerpt": source_excerpt,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-003: AQUSA Score
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAqusaScore:
    """Tests for compute_aqusa_score (EVAL-003)."""

    def test_empty_collection_returns_null(self):
        result = compute_aqusa_score([])
        assert result.value is None
        assert result.null_reason is not None
        assert result.input_count == 0
        assert result.formula_version == FORMULA_VERSION_AQUSA

    def test_all_passing_stories(self):
        stories = [
            make_story("US-0001"),
            make_story("US-0002", action="load a saved game", benefit="I can continue from where I left off"),
            make_story("US-0003", action="adjust the volume", benefit="the audio matches my preferences"),
        ]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(100.0)
        assert result.numerator == 3.0
        assert result.denominator == 3.0
        assert result.input_count == 3
        assert len(result.raw_observations) == 3
        assert all(obs.value is True for obs in result.raw_observations)

    def test_empty_role_fails(self):
        stories = [make_story("US-0001", role="")]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(0.0)
        obs = result.raw_observations[0]
        assert obs.value is False
        assert "role_empty" in obs.detail

    def test_empty_action_fails(self):
        stories = [make_story("US-0001", action="")]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(0.0)

    def test_empty_benefit_fails(self):
        stories = [make_story("US-0001", benefit="")]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(0.0)

    def test_format_violated_fails(self):
        # Missing "so that" clause
        stories = [make_story("US-0001", full_text="As a player, I want navigate.")]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(0.0)
        obs = result.raw_observations[0]
        assert "format_violated" in obs.detail

    def test_ambiguity_heuristic_fails(self):
        # "it can" pattern triggers heuristic
        stories = [make_story(
            "US-0001",
            full_text="As a player, I want to click, so that it can be done.",
        )]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(0.0)
        obs = result.raw_observations[0]
        assert "ambiguity_heuristic" in obs.detail

    def test_mixed_pass_fail(self):
        stories = [
            make_story("US-0001"),   # passes
            make_story("US-0002", role=""),  # fails
            make_story("US-0003"),   # passes
        ]
        result = compute_aqusa_score(stories)
        assert result.value == pytest.approx(2 / 3 * 100)
        assert result.numerator == 2.0
        assert result.denominator == 3.0

    def test_raw_observations_count_matches_input(self):
        stories = [make_story(f"US-{i:04d}") for i in range(7)]
        result = compute_aqusa_score(stories)
        assert len(result.raw_observations) == 7

    def test_formula_version_stable(self):
        assert compute_aqusa_score([]).formula_version == "aqusa_v1_syntactic_5criteria"


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-004: Coverage
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeCoverage:
    """Tests for compute_coverage (EVAL-004)."""

    def test_zero_total_chunks_returns_null(self):
        result = compute_coverage([make_story()], 0)
        assert result.value is None
        assert result.null_reason is not None

    def test_empty_stories_gives_zero(self):
        result = compute_coverage([], 5)
        assert result.value == pytest.approx(0.0)
        assert result.numerator == 0.0
        assert result.denominator == 5.0

    def test_full_coverage(self):
        stories = [
            make_story("US-0001", source_chunk_ids=["chunk_0001"]),
            make_story("US-0002", source_chunk_ids=["chunk_0002"]),
        ]
        result = compute_coverage(stories, 2)
        assert result.value == pytest.approx(100.0)
        assert result.numerator == 2.0

    def test_partial_coverage(self):
        # 1 of 4 chunks covered
        stories = [make_story("US-0001", source_chunk_ids=["chunk_0001"])]
        result = compute_coverage(stories, 4)
        assert result.value == pytest.approx(25.0)

    def test_duplicate_chunk_ids_counted_once(self):
        # Two stories both reference the same chunk
        stories = [
            make_story("US-0001", source_chunk_ids=["chunk_0001"]),
            make_story("US-0002", source_chunk_ids=["chunk_0001"]),
        ]
        result = compute_coverage(stories, 4)
        # Still only 1 unique chunk covered
        assert result.numerator == 1.0
        assert result.value == pytest.approx(25.0)

    def test_story_references_multiple_chunks(self):
        stories = [
            make_story("US-0001", source_chunk_ids=["chunk_0001", "chunk_0002"]),
        ]
        result = compute_coverage(stories, 4)
        assert result.numerator == 2.0
        assert result.value == pytest.approx(50.0)

    def test_formula_version_stable(self):
        assert compute_coverage([], 1).formula_version == "coverage_v1_chunk_ratio"

    def test_input_count_equals_total_chunks(self):
        result = compute_coverage([make_story()], 10)
        assert result.input_count == 10


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-005: Diversity
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeDiversity:
    """Tests for compute_diversity (EVAL-005)."""

    def test_empty_collection_returns_null(self):
        result = compute_diversity([])
        assert result.value is None
        assert result.null_reason is not None

    def test_single_category_low_diversity(self):
        stories = [make_story(requirement_type="player", game_domain="UI") for _ in range(5)]
        result = compute_diversity(stories)
        # 1 req type + 1 domain = 2 unique categories out of 11
        assert result.numerator == 2.0
        assert result.denominator == float(TOTAL_DIVERSITY_CATEGORIES)
        assert result.value == pytest.approx(2 / TOTAL_DIVERSITY_CATEGORIES)

    def test_all_categories_max_diversity(self):
        req_types = ["player", "system", "in-game entity", "dev team"]
        domains = ["gameplay", "UI", "narrative", "systems", "level design", "audio", "other"]
        stories = []
        for i, (rt, gd) in enumerate(zip(req_types + req_types, domains)):
            stories.append(make_story(
                f"US-{i:04d}",
                requirement_type=rt,
                game_domain=gd,
            ))
        result = compute_diversity(stories)
        # Should cover 4 types + 7 domains = 11 = TOTAL_DIVERSITY_CATEGORIES
        assert result.numerator == float(TOTAL_DIVERSITY_CATEGORIES)
        assert result.value == pytest.approx(1.0)

    def test_value_range_zero_to_one(self):
        stories = [make_story()]
        result = compute_diversity(stories)
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0

    def test_raw_observations_count_matches_input(self):
        stories = [make_story(f"US-{i:04d}") for i in range(6)]
        result = compute_diversity(stories)
        assert len(result.raw_observations) == 6

    def test_raw_observation_contains_category_info(self):
        story = make_story("US-0001", requirement_type="system", game_domain="narrative")
        result = compute_diversity([story])
        obs = result.raw_observations[0]
        assert obs.value["requirement_type"] == "system"
        assert obs.value["game_domain"] == "narrative"

    def test_formula_version_stable(self):
        assert compute_diversity([make_story()]).formula_version == "diversity_v1_category_ratio"


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-002: Redundancy Rate — MAS
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRedundancyRateMAS:
    """Tests for compute_redundancy_rate_mas (EVAL-002, MAS)."""

    def test_no_redundancy_analysis_returns_null(self):
        result = compute_redundancy_rate_mas([], None)
        assert result.value is None
        assert result.null_reason is not None

    def test_zero_input_count_returns_null(self):
        analysis = {"input_story_count": 0, "duplicate_groups": []}
        result = compute_redundancy_rate_mas([], analysis)
        assert result.value is None

    def test_no_duplicate_groups(self):
        analysis = {"input_story_count": 5, "duplicate_groups": []}
        result = compute_redundancy_rate_mas([make_story()], analysis)
        assert result.value == pytest.approx(0.0)
        assert result.numerator == 0.0
        assert result.denominator == 5.0

    def test_one_duplicate_group(self):
        analysis = {
            "input_story_count": 5,
            "duplicate_groups": [
                {"story_ids": ["US-0001", "US-0002"]}
            ],
            "resolution_status": "resolved",
        }
        result = compute_redundancy_rate_mas([], analysis)
        assert result.value == pytest.approx(2 / 5 * 100)
        assert result.numerator == 2.0

    def test_overlapping_group_ids_counted_once(self):
        # Same story in two groups — should count only once
        analysis = {
            "input_story_count": 5,
            "duplicate_groups": [
                {"story_ids": ["US-0001", "US-0002"]},
                {"story_ids": ["US-0002", "US-0003"]},
            ],
        }
        result = compute_redundancy_rate_mas([], analysis)
        # US-0001, US-0002, US-0003 = 3 unique
        assert result.numerator == 3.0
        assert result.value == pytest.approx(3 / 5 * 100)

    def test_formula_version_stable(self):
        analysis = {"input_story_count": 1, "duplicate_groups": []}
        r = compute_redundancy_rate_mas([], analysis)
        assert r.formula_version == "redundancy_v1_mas_checker_groups"


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-002: Redundancy Rate — Baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRedundancyRateBaseline:
    """Tests for compute_redundancy_rate_baseline (EVAL-002, Baseline)."""

    def test_empty_collection_returns_null(self):
        result = compute_redundancy_rate_baseline([])
        assert result.value is None
        assert result.null_reason is not None

    def test_no_duplicates_gives_zero(self):
        stories = [
            make_story("US-0001", action="navigate the menu", benefit="I can see options"),
            make_story("US-0002", action="shoot the enemies", benefit="they are defeated"),
            make_story("US-0003", action="load a save file", benefit="I can resume"),
        ]
        result = compute_redundancy_rate_baseline(stories, threshold=0.5)
        # These have very different vocabularies — should be 0%
        assert result.value == pytest.approx(0.0)
        assert result.numerator == 0.0

    def test_identical_stories_are_redundant(self):
        s = make_story("US-0001", action="shoot all enemies fast", benefit="I win the level quickly")
        # Duplicate with slightly different ID
        s2 = dict(s)
        s2["id"] = "US-0002"
        result = compute_redundancy_rate_baseline([s, s2], threshold=0.5)
        # Jaccard of identical sets = 1.0 > 0.5 → both flagged
        assert result.numerator == 2.0
        assert result.value == pytest.approx(100.0)

    def test_partial_overlap_below_threshold_not_redundant(self):
        s1 = make_story("US-0001", action="run fast through levels", benefit="save time efficiently")
        s2 = make_story("US-0002", action="shoot enemies with pistol", benefit="eliminate threats now")
        result = compute_redundancy_rate_baseline([s1, s2], threshold=0.9)
        # Very different vocabularies — Jaccard < 0.9
        assert result.numerator == 0.0

    def test_raw_observations_cover_all_stories(self):
        stories = [make_story(f"US-{i:04d}") for i in range(4)]
        result = compute_redundancy_rate_baseline(stories, threshold=0.99)
        # All observations (both redundant and non-redundant) should be present
        assert len(result.raw_observations) == 4

    def test_formula_version_stable(self):
        r = compute_redundancy_rate_baseline([make_story()], threshold=0.7)
        assert r.formula_version == "redundancy_v2_jaccard_pairwise_0.7"

    def test_custom_threshold(self):
        """Lower threshold should flag more stories as redundant."""
        s1 = make_story("US-0001", action="walk through the map area", benefit="explore new zones")
        s2 = make_story("US-0002", action="walk through map area zones", benefit="see new places")
        r_strict = compute_redundancy_rate_baseline([s1, s2], threshold=0.95)
        r_loose = compute_redundancy_rate_baseline([s1, s2], threshold=0.10)
        # Strict should flag fewer (or equal) stories
        assert (r_strict.numerator or 0) <= (r_loose.numerator or 0)


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-001: Hallucination Rate — MAS
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHallucinationRateMAS:
    """Tests for compute_hallucination_rate_mas (EVAL-001, MAS)."""

    def test_empty_collection_returns_null(self):
        result = compute_hallucination_rate_mas([])
        assert result.value is None
        assert result.null_reason is not None

    def test_all_traceability_confirmed(self):
        stories = [make_story(reviewer_traceability=True) for _ in range(5)]
        result = compute_hallucination_rate_mas(stories)
        assert result.value == pytest.approx(0.0)
        assert result.numerator == 0.0
        assert result.denominator == 5.0

    def test_all_traceability_failed(self):
        stories = [make_story(reviewer_traceability=False) for _ in range(3)]
        result = compute_hallucination_rate_mas(stories)
        assert result.value == pytest.approx(100.0)
        assert result.numerator == 3.0

    def test_mixed_traceability(self):
        stories = [
            make_story("US-0001", reviewer_traceability=True),
            make_story("US-0002", reviewer_traceability=False),
            make_story("US-0003", reviewer_traceability=True),
            make_story("US-0004", reviewer_traceability=False),
        ]
        result = compute_hallucination_rate_mas(stories)
        assert result.value == pytest.approx(50.0)
        assert result.numerator == 2.0
        assert result.denominator == 4.0

    def test_none_traceability_excluded_from_denominator(self):
        stories = [
            make_story("US-0001", reviewer_traceability=None),  # unverifiable
            make_story("US-0002", reviewer_traceability=False), # hallucinated
            make_story("US-0003", reviewer_traceability=True),  # ok
        ]
        result = compute_hallucination_rate_mas(stories)
        # 1 hallucinated / 2 verifiable = 50%
        assert result.value == pytest.approx(50.0)
        assert result.denominator == 2.0

    def test_all_none_traceability_returns_null(self):
        stories = [make_story(reviewer_traceability=None) for _ in range(3)]
        result = compute_hallucination_rate_mas(stories)
        assert result.value is None
        assert result.null_reason is not None

    def test_missing_confidence_evidence_treated_as_unverifiable(self):
        s = make_story()
        s["confidence_evidence"] = None
        result = compute_hallucination_rate_mas([s])
        # No confidence evidence → unverifiable → null
        assert result.value is None

    def test_raw_observations_classify_correctly(self):
        stories = [
            make_story("US-0001", reviewer_traceability=True),
            make_story("US-0002", reviewer_traceability=False),
            make_story("US-0003", reviewer_traceability=None),
        ]
        result = compute_hallucination_rate_mas(stories)
        obs_by_id = {o.story_id: o for o in result.raw_observations}
        assert obs_by_id["US-0001"].value is False   # not hallucinated
        assert obs_by_id["US-0002"].value is True    # hallucinated
        assert obs_by_id["US-0003"].value is None    # unverifiable

    def test_formula_version_stable(self):
        r = compute_hallucination_rate_mas([make_story()])
        assert r.formula_version == "hallucination_v1_reviewer_traceability"


# ══════════════════════════════════════════════════════════════════════════════
# EVAL-001: Hallucination Rate — Baseline (keyword heuristic)
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHallucinationRateBaselineHeuristic:
    """Tests for compute_hallucination_rate_baseline_heuristic (EVAL-001, Baseline)."""

    def test_empty_collection_returns_null(self):
        result = compute_hallucination_rate_baseline_heuristic([], {})
        assert result.value is None
        assert result.null_reason is not None

    def test_no_chunk_texts_all_unverifiable(self):
        stories = [make_story("US-0001", source_chunk_ids=["chunk_0001"])]
        result = compute_hallucination_rate_baseline_heuristic(stories, {})
        assert result.value is None
        assert result.null_reason is not None and result.null_reason.strip()

    def test_action_keywords_present_in_source_not_hallucinated(self):
        # Action: "navigate the main menu"
        # Source contains: "navigate", "main", "menu" → overlap ≥ 30%
        chunk_texts = {"chunk_0001": "navigate the main menu options and settings"}
        story = make_story(
            "US-0001",
            action="navigate the main menu",
            source_chunk_ids=["chunk_0001"],
        )
        result = compute_hallucination_rate_baseline_heuristic([story], chunk_texts)
        assert result.value == pytest.approx(0.0)  # no hallucination

    def test_action_keywords_absent_from_source_flagged(self):
        # Action about completely different domain than source
        chunk_texts = {"chunk_0001": "flowers bloom in the spring garden every year"}
        story = make_story(
            "US-0001",
            action="shoot enemies with rocket launcher in arena",
            source_chunk_ids=["chunk_0001"],
        )
        result = compute_hallucination_rate_baseline_heuristic([story], chunk_texts)
        assert result.value is not None
        assert result.value > 0.0  # flagged as hallucinated

    def test_missing_chunk_id_unverifiable(self):
        chunk_texts = {"chunk_other": "some text about enemies"}
        story = make_story("US-0001", source_chunk_ids=["chunk_0001"])
        result = compute_hallucination_rate_baseline_heuristic([story], chunk_texts)
        # chunk_0001 not in chunk_texts → unverifiable → null
        assert result.value is None

    def test_mixed_verifiable_and_unverifiable(self):
        chunk_texts = {"chunk_0001": "navigate the main menu interface"}
        stories = [
            make_story("US-0001", action="navigate the main menu", source_chunk_ids=["chunk_0001"]),
            make_story("US-0002", action="some action", source_chunk_ids=["missing_chunk"]),
        ]
        result = compute_hallucination_rate_baseline_heuristic(stories, chunk_texts)
        # One verifiable (not hallucinated), one unverifiable
        assert result.denominator == 1.0
        assert result.value == pytest.approx(0.0)

    def test_formula_version_stable(self):
        r = compute_hallucination_rate_baseline_heuristic(
            [make_story()],
            {"chunk_0001": "navigate the menu"},
        )
        assert r.formula_version == "hallucination_v2_source_excerpt_keyword"

    def test_raw_observations_count_matches_input(self):
        chunk_texts = {f"chunk_{i:04d}": f"text content {i}" for i in range(4)}
        stories = [
            make_story(f"US-{i:04d}", source_chunk_ids=[f"chunk_{i:04d}"])
            for i in range(4)
        ]
        result = compute_hallucination_rate_baseline_heuristic(stories, chunk_texts)
        assert len(result.raw_observations) == 4


# ══════════════════════════════════════════════════════════════════════════════
# Cross-metric invariants (VAL-029 equivalent for MetricObservation)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricObservationInvariants:
    """
    Invariant: whenever value is None, null_reason must be set.
    (Equivalent to VAL-029 from EvaluationResult schema.)
    """

    def _check_null_reason(self, obs):
        if obs.value is None:
            assert obs.null_reason is not None and obs.null_reason.strip(), (
                f"Metric '{obs.metric_name}' has value=None but no null_reason set."
            )

    def test_aqusa_null_has_reason(self):
        self._check_null_reason(compute_aqusa_score([]))

    def test_coverage_null_has_reason(self):
        self._check_null_reason(compute_coverage([], 0))

    def test_diversity_null_has_reason(self):
        self._check_null_reason(compute_diversity([]))

    def test_redundancy_mas_null_has_reason(self):
        self._check_null_reason(compute_redundancy_rate_mas([], None))

    def test_redundancy_baseline_null_has_reason(self):
        self._check_null_reason(compute_redundancy_rate_baseline([]))

    def test_hallucination_mas_null_has_reason(self):
        self._check_null_reason(compute_hallucination_rate_mas([]))

    def test_hallucination_baseline_null_has_reason(self):
        self._check_null_reason(
            compute_hallucination_rate_baseline_heuristic([], {})
        )

    def test_null_mas_no_traceability_has_reason(self):
        stories = [make_story(reviewer_traceability=None)]
        self._check_null_reason(compute_hallucination_rate_mas(stories))
