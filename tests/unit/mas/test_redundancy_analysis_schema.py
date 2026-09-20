"""
tests/unit/mas/test_redundancy_analysis_schema.py
====================================================
Unit tests for RedundancyAnalysis, DuplicateGroup, and MergeAction
Pydantic schemas (Data Contracts §9).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.redundancy_analysis import (
    DuplicateGroup,
    MergeAction,
    RedundancyAnalysis,
)


# ─────────────────────────────────────────────────────────────────────────────
# DuplicateGroup
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateGroup:

    def test_valid_construction(self):
        g = DuplicateGroup(
            group_id="dup_group_0",
            story_ids=["draft_a", "draft_b"],
            similarity_basis="llm_judgment",
        )
        assert g.group_id == "dup_group_0"
        assert len(g.story_ids) == 2

    def test_with_score(self):
        g = DuplicateGroup(
            group_id="dup_group_0",
            story_ids=["a", "b"],
            similarity_basis="llm_judgment",
            similarity_score=0.95,
        )
        assert g.similarity_score == 0.95

    def test_three_stories(self):
        g = DuplicateGroup(
            group_id="g1",
            story_ids=["a", "b", "c"],
            similarity_basis="llm_judgment",
        )
        assert len(g.story_ids) == 3

    def test_min_two_stories_val026(self):
        """VAL-026: must have ≥2 story_ids."""
        with pytest.raises(ValidationError):
            DuplicateGroup(
                group_id="g1",
                story_ids=["only_one"],
                similarity_basis="llm_judgment",
            )

    def test_empty_stories(self):
        with pytest.raises(ValidationError):
            DuplicateGroup(
                group_id="g1",
                story_ids=[],
                similarity_basis="llm_judgment",
            )

    def test_empty_group_id(self):
        with pytest.raises(ValidationError):
            DuplicateGroup(
                group_id="",
                story_ids=["a", "b"],
                similarity_basis="llm_judgment",
            )

    def test_empty_story_id_in_list(self):
        with pytest.raises(ValidationError):
            DuplicateGroup(
                group_id="g1",
                story_ids=["a", ""],
                similarity_basis="llm_judgment",
            )

    def test_frozen(self):
        g = DuplicateGroup(
            group_id="g1",
            story_ids=["a", "b"],
            similarity_basis="llm_judgment",
        )
        with pytest.raises(ValidationError):
            g.group_id = "changed"


# ─────────────────────────────────────────────────────────────────────────────
# MergeAction
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeAction:

    def test_remove_inferior(self):
        a = MergeAction(
            group_id="g1",
            action="remove_inferior",
            kept_story_id="draft_a",
            removed_story_ids=["draft_b"],
            source_chunk_ids=["chunk_1", "chunk_2"],
            reason="draft_b is less informative.",
        )
        assert a.action == "remove_inferior"
        assert a.merged_role is None

    def test_merge(self):
        a = MergeAction(
            group_id="g1",
            action="merge",
            kept_story_id="draft_a",
            removed_story_ids=["draft_b"],
            merged_role="player",
            merged_action="attack and dodge",
            merged_benefit="survive combat encounters",
            merged_full_text="As a player, I want attack and dodge, so that survive combat encounters.",
            source_chunk_ids=["chunk_1"],
            reason="Both describe combat mechanics.",
        )
        assert a.action == "merge"
        assert a.merged_role == "player"

    def test_merge_missing_fields_raises(self):
        """Merge requires all merged_* fields."""
        with pytest.raises(ValidationError, match="merged_role"):
            MergeAction(
                group_id="g1",
                action="merge",
                kept_story_id="draft_a",
                removed_story_ids=["draft_b"],
                merged_role=None,
                merged_action="attack",
                merged_benefit="survive",
                merged_full_text="As a player, I want attack, so that survive.",
                source_chunk_ids=["c1"],
                reason="test",
            )

    def test_empty_removed_ids(self):
        with pytest.raises(ValidationError):
            MergeAction(
                group_id="g1",
                action="remove_inferior",
                kept_story_id="draft_a",
                removed_story_ids=[],
                source_chunk_ids=["c1"],
                reason="test",
            )

    def test_empty_source_chunk_ids(self):
        with pytest.raises(ValidationError):
            MergeAction(
                group_id="g1",
                action="remove_inferior",
                kept_story_id="draft_a",
                removed_story_ids=["draft_b"],
                source_chunk_ids=[],
                reason="test",
            )

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            MergeAction(
                group_id="g1",
                action="delete",
                kept_story_id="a",
                removed_story_ids=["b"],
                source_chunk_ids=["c1"],
                reason="test",
            )

    def test_empty_reason(self):
        with pytest.raises(ValidationError):
            MergeAction(
                group_id="g1",
                action="remove_inferior",
                kept_story_id="a",
                removed_story_ids=["b"],
                source_chunk_ids=["c1"],
                reason="",
            )


# ─────────────────────────────────────────────────────────────────────────────
# RedundancyAnalysis
# ─────────────────────────────────────────────────────────────────────────────

class TestRedundancyAnalysis:

    def test_no_duplicates(self):
        a = RedundancyAnalysis(
            analysis_id="redundancy__run1__pass1",
            run_id="run1",
            analyzed_at_pass=1,
            duplicate_groups=[],
            unique_story_ids=["d1", "d2", "d3"],
            resolution_status="resolved",
            merge_actions=[],
            resolved_story_count=3,
            input_story_count=3,
        )
        assert a.resolution_status == "resolved"
        assert a.resolved_story_count == 3

    def test_with_groups_and_actions(self):
        a = RedundancyAnalysis(
            analysis_id="redundancy__run1__pass1",
            run_id="run1",
            analyzed_at_pass=1,
            duplicate_groups=[
                DuplicateGroup(
                    group_id="g0",
                    story_ids=["d1", "d2"],
                    similarity_basis="llm_judgment",
                ),
            ],
            unique_story_ids=["d3"],
            resolution_status="resolved",
            merge_actions=[
                MergeAction(
                    group_id="g0",
                    action="remove_inferior",
                    kept_story_id="d1",
                    removed_story_ids=["d2"],
                    source_chunk_ids=["c1", "c2"],
                    reason="d2 is less complete.",
                ),
            ],
            resolved_story_count=2,
            input_story_count=3,
        )
        assert len(a.duplicate_groups) == 1
        assert len(a.merge_actions) == 1

    def test_unresolved_status(self):
        a = RedundancyAnalysis(
            analysis_id="redundancy__run1__pass2",
            run_id="run1",
            analyzed_at_pass=2,
            duplicate_groups=[
                DuplicateGroup(
                    group_id="g0",
                    story_ids=["d1", "d2"],
                    similarity_basis="llm_judgment",
                ),
            ],
            unique_story_ids=["d3"],
            resolution_status="unresolved_max_iterations",
            merge_actions=[],
            resolved_story_count=3,
            input_story_count=3,
        )
        assert a.resolution_status == "unresolved_max_iterations"

    def test_empty_analysis_id(self):
        with pytest.raises(ValidationError):
            RedundancyAnalysis(
                analysis_id="",
                run_id="run1",
                analyzed_at_pass=1,
                resolution_status="resolved",
                resolved_story_count=0,
                input_story_count=0,
            )

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            RedundancyAnalysis(
                analysis_id="id1",
                run_id="run1",
                analyzed_at_pass=1,
                resolution_status="unknown",
                resolved_story_count=0,
                input_story_count=0,
            )

    def test_zero_pass(self):
        with pytest.raises(ValidationError):
            RedundancyAnalysis(
                analysis_id="id1",
                run_id="run1",
                analyzed_at_pass=0,
                resolution_status="resolved",
                resolved_story_count=0,
                input_story_count=0,
            )
