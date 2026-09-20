"""
tests/unit/mas/test_redundancy_output_validator.py
=====================================================
Unit tests for validate_redundancy_output() — VAL-D01 through VAL-D07.
"""
from __future__ import annotations

import pytest

from gdd_userstory_mas.mas.redundancy_output_validator import (
    RedundancyValidationReport,
    validate_redundancy_output,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import (
    DuplicateGroup,
    MergeAction,
    RedundancyAnalysis,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_story(draft_id: str, chunk_id: str = "c1") -> GeneratedUserStory:
    role = "player"
    action = f"do action for {draft_id}"
    benefit = "some benefit"
    return GeneratedUserStory(
        draft_id=draft_id,
        candidate_id=draft_id.replace("__draft", ""),
        source_chunk_id=chunk_id,
        document_id="doc1",
        role=role,
        action=action,
        benefit=benefit,
        full_text=f"As a {role}, I want {action}, so that {benefit}.",
        iteration_count=0,
    )


def _make_analysis(
    stories: list,
    groups: list | None = None,
    unique_ids: list | None = None,
    actions: list | None = None,
    status: str = "resolved",
) -> RedundancyAnalysis:
    all_ids = [s.draft_id for s in stories]
    grouped_ids = set()
    for g in (groups or []):
        grouped_ids.update(g.story_ids)
    if unique_ids is None:
        unique_ids = [sid for sid in all_ids if sid not in grouped_ids]

    return RedundancyAnalysis(
        analysis_id="redundancy__run1__pass1",
        run_id="run1",
        analyzed_at_pass=1,
        duplicate_groups=groups or [],
        unique_story_ids=unique_ids,
        resolution_status=status,
        merge_actions=actions or [],
        resolved_story_count=len(unique_ids) + len(actions or []),
        input_story_count=len(stories),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_no_duplicates_passes(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        analysis = _make_analysis(stories)
        report = validate_redundancy_output(analysis, stories)
        assert report.passed
        assert report.errors == []

    def test_with_resolved_group_passes(self):
        stories = [
            _make_story("d1__draft", "c1"),
            _make_story("d2__draft", "c2"),
            _make_story("d3__draft", "c3"),
        ]
        group = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "d2__draft"],
            similarity_basis="llm_judgment",
        )
        action = MergeAction(
            group_id="g0",
            action="remove_inferior",
            kept_story_id="d1__draft",
            removed_story_ids=["d2__draft"],
            source_chunk_ids=["c1", "c2"],
            reason="d2 is less complete.",
        )
        analysis = _make_analysis(stories, groups=[group], actions=[action])
        report = validate_redundancy_output(analysis, stories)
        assert report.passed

    def test_report_type(self):
        stories = [_make_story("d1__draft")]
        analysis = _make_analysis(stories)
        report = validate_redundancy_output(analysis, stories)
        assert isinstance(report, RedundancyValidationReport)

    def test_summary_contains_passed(self):
        stories = [_make_story("d1__draft")]
        analysis = _make_analysis(stories)
        report = validate_redundancy_output(analysis, stories)
        assert "PASSED" in report.summary()


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D01: Groups have ≥2 story_ids
# ─────────────────────────────────────────────────────────────────────────────

class TestValD01:
    """Schema-level validation blocks <2 story_ids already.
    This test confirms defence-in-depth at the validator level."""

    def test_valid_group_passes(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        group = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "d2__draft"],
            similarity_basis="llm_judgment",
        )
        analysis = _make_analysis(stories, groups=[group])
        report = validate_redundancy_output(analysis, stories)
        assert report.passed


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D02: All story_ids reference actual input stories
# ─────────────────────────────────────────────────────────────────────────────

class TestValD02:

    def test_unknown_story_id_fails(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        group = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "nonexistent__draft"],
            similarity_basis="llm_judgment",
        )
        analysis = _make_analysis(
            stories, groups=[group],
            unique_ids=["d2__draft"],
        )
        report = validate_redundancy_output(analysis, stories)
        assert not report.passed
        assert any("VAL-D02" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D03: No story in multiple groups
# ─────────────────────────────────────────────────────────────────────────────

class TestValD03:

    def test_story_in_two_groups_fails(self):
        stories = [
            _make_story("d1__draft"),
            _make_story("d2__draft"),
            _make_story("d3__draft"),
        ]
        g1 = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "d2__draft"],
            similarity_basis="llm_judgment",
        )
        g2 = DuplicateGroup(
            group_id="g1",
            story_ids=["d2__draft", "d3__draft"],
            similarity_basis="llm_judgment",
        )
        analysis = _make_analysis(
            stories, groups=[g1, g2],
            unique_ids=[],
        )
        report = validate_redundancy_output(analysis, stories)
        assert not report.passed
        assert any("VAL-D03" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D04: unique + grouped = all input
# ─────────────────────────────────────────────────────────────────────────────

class TestValD04:

    def test_missing_story_fails(self):
        stories = [
            _make_story("d1__draft"),
            _make_story("d2__draft"),
            _make_story("d3__draft"),
        ]
        # Only d1 and d2 accounted for — d3 missing
        analysis = _make_analysis(
            stories,
            unique_ids=["d1__draft", "d2__draft"],
        )
        report = validate_redundancy_output(analysis, stories)
        assert not report.passed
        assert any("VAL-D04" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D05: Merge action references valid group
# ─────────────────────────────────────────────────────────────────────────────

class TestValD05:

    def test_unknown_group_fails(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        action = MergeAction(
            group_id="nonexistent_group",
            action="remove_inferior",
            kept_story_id="d1__draft",
            removed_story_ids=["d2__draft"],
            source_chunk_ids=["c1"],
            reason="test",
        )
        analysis = _make_analysis(stories, actions=[action])
        report = validate_redundancy_output(analysis, stories)
        assert not report.passed
        assert any("VAL-D05" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D06: Merged full_text format
# ─────────────────────────────────────────────────────────────────────────────

class TestValD06:

    def test_bad_merge_format_warns(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        group = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "d2__draft"],
            similarity_basis="llm_judgment",
        )
        action = MergeAction(
            group_id="g0",
            action="merge",
            kept_story_id="d1__draft",
            removed_story_ids=["d2__draft"],
            merged_role="player",
            merged_action="do stuff",
            merged_benefit="good things",
            merged_full_text="This is not proper format",
            source_chunk_ids=["c1"],
            reason="test",
        )
        analysis = _make_analysis(stories, groups=[group], actions=[action])
        report = validate_redundancy_output(analysis, stories)
        assert report.passed  # warning, not error
        assert any("VAL-D06" in w for w in report.warnings)

    def test_good_merge_format_no_warning(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        group = DuplicateGroup(
            group_id="g0",
            story_ids=["d1__draft", "d2__draft"],
            similarity_basis="llm_judgment",
        )
        action = MergeAction(
            group_id="g0",
            action="merge",
            kept_story_id="d1__draft",
            removed_story_ids=["d2__draft"],
            merged_role="player",
            merged_action="do stuff",
            merged_benefit="good things happen.",
            merged_full_text="As a player, I want do stuff, so that good things happen.",
            source_chunk_ids=["c1"],
            reason="test",
        )
        analysis = _make_analysis(stories, groups=[group], actions=[action])
        report = validate_redundancy_output(analysis, stories)
        assert report.passed
        assert report.warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# VAL-D07: Input story count match
# ─────────────────────────────────────────────────────────────────────────────

class TestValD07:

    def test_wrong_input_count_fails(self):
        stories = [_make_story("d1__draft"), _make_story("d2__draft")]
        analysis = RedundancyAnalysis(
            analysis_id="redundancy__run1__pass1",
            run_id="run1",
            analyzed_at_pass=1,
            duplicate_groups=[],
            unique_story_ids=["d1__draft", "d2__draft"],
            resolution_status="resolved",
            merge_actions=[],
            resolved_story_count=2,
            input_story_count=999,  # Wrong!
        )
        report = validate_redundancy_output(analysis, stories)
        assert not report.passed
        assert any("VAL-D07" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple errors
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleErrors:

    def test_summary_contains_failed(self):
        stories = [_make_story("d1__draft")]
        analysis = RedundancyAnalysis(
            analysis_id="redundancy__run1__pass1",
            run_id="run1",
            analyzed_at_pass=1,
            duplicate_groups=[],
            unique_ids=[],
            unique_story_ids=[],
            resolution_status="resolved",
            merge_actions=[],
            resolved_story_count=0,
            input_story_count=999,
        )
        report = validate_redundancy_output(analysis, stories)
        assert "FAILED" in report.summary()
