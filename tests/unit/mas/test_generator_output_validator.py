"""
tests/unit/mas/test_generator_output_validator.py
==================================================
Unit tests for validate_generator_output() — VAL-G01 through VAL-G09.
All tests are OFFLINE — no LLM calls.
"""
from __future__ import annotations

import pytest

from gdd_userstory_mas.mas.generator_output_validator import (
    GeneratorValidationReport,
    validate_generator_output,
)
from gdd_userstory_mas.schemas.candidate_requirement import CandidateRequirement
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate(**kwargs) -> CandidateRequirement:
    defaults = dict(
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        requirement_text="The game must allow the player to attack enemies.",
        perspective="player",
        domain="gameplay",
    )
    defaults.update(kwargs)
    return CandidateRequirement(**defaults)


def _make_story(**kwargs) -> GeneratedUserStory:
    defaults = dict(
        draft_id="doc1__cand_0000__draft",
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        role="player",
        action="attack enemies using melee weapons",
        benefit="I can defeat them and progress through the level",
        full_text=(
            "As a player, I want attack enemies using melee weapons, "
            "so that I can defeat them and progress through the level."
        ),
        iteration_count=0,
    )
    defaults.update(kwargs)
    return GeneratedUserStory(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-G01 — draft_id format (TRACE-003)
# ─────────────────────────────────────────────────────────────────────────────

class TestDraftIdTraceability:

    def test_correct_draft_id_passes(self):
        c = _make_candidate()
        s = _make_story()
        assert validate_generator_output(s, c).passed

    def test_wrong_draft_id_fails(self):
        c = _make_candidate()
        s = _make_story(draft_id="WRONG__draft")
        report = validate_generator_output(s, c)
        assert not report.passed
        assert any("VAL-G01" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-G02 — candidate_id match
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateIdMatch:

    def test_matching_passes(self):
        c = _make_candidate(candidate_id="doc1__cand_0000")
        s = _make_story(candidate_id="doc1__cand_0000")
        assert validate_generator_output(s, c).passed

    def test_mismatch_fails(self):
        c = _make_candidate(candidate_id="doc1__cand_0000")
        s = _make_story(candidate_id="WRONG_CAND", draft_id="WRONG_CAND__draft")
        report = validate_generator_output(s, c)
        assert not report.passed
        assert any("VAL-G02" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-G03 — source_chunk_id (TRACE-001)
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceChunkIdTraceability:

    def test_matching_passes(self):
        c = _make_candidate(source_chunk_id="doc1__chunk_0000")
        s = _make_story(source_chunk_id="doc1__chunk_0000")
        assert validate_generator_output(s, c).passed

    def test_mismatch_fails(self):
        c = _make_candidate(source_chunk_id="doc1__chunk_0000")
        s = _make_story(source_chunk_id="WRONG_CHUNK")
        report = validate_generator_output(s, c)
        assert not report.passed
        assert any("VAL-G03" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-G04 — document_id
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdTraceability:

    def test_matching_passes(self):
        c = _make_candidate(document_id="doc1")
        s = _make_story(document_id="doc1")
        assert validate_generator_output(s, c).passed

    def test_mismatch_fails(self):
        c = _make_candidate(document_id="doc1")
        s = _make_story(document_id="WRONG_DOC")
        report = validate_generator_output(s, c)
        assert not report.passed
        assert any("VAL-G04" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-G07 — iteration_count >= 0 (VAL-022)
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationCount:

    def test_zero_passes(self):
        c = _make_candidate()
        s = _make_story(iteration_count=0)
        assert validate_generator_output(s, c).passed

    def test_positive_passes(self):
        c = _make_candidate()
        s = _make_story(iteration_count=3)
        assert validate_generator_output(s, c).passed


# ─────────────────────────────────────────────────────────────────────────────
# Multiple errors at once
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleErrors:

    def test_two_id_mismatches_two_errors(self):
        c = _make_candidate(candidate_id="doc1__cand_0000", document_id="doc1")
        s = _make_story(
            candidate_id="WRONG",
            draft_id="WRONG__draft",
            document_id="WRONG_DOC",
        )
        report = validate_generator_output(s, c)
        assert not report.passed
        assert len(report.errors) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_fully_valid_passes_cleanly(self):
        c = _make_candidate()
        s = _make_story()
        report = validate_generator_output(s, c)
        assert report.passed
        assert report.errors == []
        assert report.warnings == []

    def test_report_summary_contains_draft_id(self):
        c = _make_candidate()
        s = _make_story()
        report = validate_generator_output(s, c)
        assert s.draft_id in report.summary()
        assert "PASSED" in report.summary()

    def test_failed_report_summary_shows_failed(self):
        c = _make_candidate()
        s = _make_story(draft_id="WRONG__draft")
        report = validate_generator_output(s, c)
        assert "FAILED" in report.summary()


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReport dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReport:

    def test_passed_true_no_errors(self):
        report = GeneratorValidationReport(draft_id="d1__draft", passed=True)
        assert report.passed
        assert report.errors == []
        assert report.warnings == []

    def test_summary_shows_all_errors(self):
        report = GeneratorValidationReport(
            draft_id="d1__draft",
            passed=False,
            errors=["VAL-G01: something", "VAL-G02: something else"],
        )
        summary = report.summary()
        assert "VAL-G01" in summary
        assert "VAL-G02" in summary
        assert "FAILED" in summary
