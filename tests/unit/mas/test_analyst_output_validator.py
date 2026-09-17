"""
tests/unit/mas/test_analyst_output_validator.py
================================================
Unit tests for validate_analyst_output() — VAL-A01 through VAL-A07.
All tests are OFFLINE — no LLM calls.
"""
from __future__ import annotations

import pytest

from gdd_userstory_mas.mas.analyst_output_validator import (
    AnalystValidationReport,
    validate_analyst_output,
)
from gdd_userstory_mas.schemas.candidate_requirement import (
    AnalystOutput,
    CandidateRequirement,
)
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_reader_output(**kwargs) -> ReaderOutput:
    defaults = dict(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        themes=["combat"],
        gameplay_elements=[
            EvidencedItem(content="Melee attacks", source_excerpt="melee"),
        ],
        systems=[], characters=[], ui_elements=[], narrative=[],
    )
    defaults.update(kwargs)
    return ReaderOutput(**defaults)


def _make_candidate(**kwargs) -> CandidateRequirement:
    defaults = dict(
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        requirement_text="The player must be able to perform melee attacks.",
        perspective="player",
        domain="gameplay",
    )
    defaults.update(kwargs)
    return CandidateRequirement(**defaults)


def _make_output(**kwargs) -> AnalystOutput:
    defaults = dict(
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        candidates=[],
    )
    defaults.update(kwargs)
    return AnalystOutput(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A01 — source_chunk_id traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkIdTraceability:

    def test_matching_chunk_id_passes(self):
        ro = _make_reader_output()
        output = _make_output()
        report = validate_analyst_output(output, ro)
        assert report.passed

    def test_mismatched_chunk_id_fails(self):
        ro = _make_reader_output(chunk_id="doc1__chunk_0000")
        output = _make_output(source_chunk_id="WRONG_ID")
        report = validate_analyst_output(output, ro)
        assert not report.passed
        assert any("VAL-A01" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A02 — document_id traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdTraceability:

    def test_matching_document_id_passes(self):
        ro = _make_reader_output(document_id="doc1")
        output = _make_output(document_id="doc1")
        assert validate_analyst_output(output, ro).passed

    def test_mismatched_document_id_fails(self):
        ro = _make_reader_output(document_id="doc1")
        output = _make_output(document_id="WRONG_DOC")
        report = validate_analyst_output(output, ro)
        assert not report.passed
        assert any("VAL-A02" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A03 — candidate_id uniqueness
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateIdUniqueness:

    def test_unique_ids_pass(self):
        c1 = _make_candidate(candidate_id="doc1__cand_0000")
        c2 = _make_candidate(
            candidate_id="doc1__cand_0001",
            requirement_text="The system must save progress automatically.",
            perspective="system",
            domain="systems",
        )
        output = _make_output(candidates=[c1, c2])
        ro = _make_reader_output()
        assert validate_analyst_output(output, ro).passed

    def test_duplicate_ids_fail(self):
        c1 = _make_candidate(candidate_id="doc1__cand_0000")
        c2 = _make_candidate(
            candidate_id="doc1__cand_0000",  # DUPLICATE
            requirement_text="Different text.",
            perspective="system",
            domain="systems",
        )
        output = _make_output(candidates=[c1, c2])
        ro = _make_reader_output()
        report = validate_analyst_output(output, ro)
        assert not report.passed
        assert any("VAL-A03" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A04 — all candidates' source_chunk_id match
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateChunkIdConsistency:

    def test_all_matching_passes(self):
        c = _make_candidate(source_chunk_id="doc1__chunk_0000")
        output = _make_output(candidates=[c])
        ro = _make_reader_output(chunk_id="doc1__chunk_0000")
        assert validate_analyst_output(output, ro).passed

    def test_candidate_wrong_source_chunk_fails(self):
        c = _make_candidate(source_chunk_id="WRONG_CHUNK")
        output = _make_output(candidates=[c])
        ro = _make_reader_output(chunk_id="doc1__chunk_0000")
        report = validate_analyst_output(output, ro)
        assert not report.passed
        assert any("VAL-A04" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A06 — empty output warning when ReaderOutput non-empty
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyOutputWarning:

    def test_empty_output_with_nonempty_reader_warns(self):
        ro = _make_reader_output()  # has gameplay_elements
        assert not ro.is_empty
        output = _make_output()  # empty candidates
        report = validate_analyst_output(output, ro)
        assert report.passed  # not a blocking error
        assert any("VAL-A06" in w for w in report.warnings)

    def test_empty_output_with_empty_reader_no_warning(self):
        ro = ReaderOutput(
            chunk_id="doc1__chunk_0000", document_id="doc1",
            themes=[], gameplay_elements=[], systems=[],
            characters=[], ui_elements=[], narrative=[],
        )
        output = _make_output()
        report = validate_analyst_output(output, ro)
        assert report.passed
        assert not any("VAL-A06" in w for w in report.warnings)

    def test_non_empty_output_no_warning(self):
        ro = _make_reader_output()
        output = _make_output(candidates=[_make_candidate()])
        report = validate_analyst_output(output, ro)
        assert report.passed
        assert not any("VAL-A06" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-A07 — source_reader_output_ref cross-reference
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceRefCrossReference:

    def test_ref_matches_evidenced_item_no_warning(self):
        ro = _make_reader_output()  # has "Melee attacks"
        c = _make_candidate(source_reader_output_ref="Melee attacks")
        output = _make_output(candidates=[c])
        report = validate_analyst_output(output, ro)
        assert report.passed
        assert not any("VAL-A07" in w for w in report.warnings)

    def test_ref_does_not_match_warns(self):
        ro = _make_reader_output()
        c = _make_candidate(source_reader_output_ref="This item does not exist")
        output = _make_output(candidates=[c])
        report = validate_analyst_output(output, ro)
        assert report.passed  # warning, not error
        assert any("VAL-A07" in w for w in report.warnings)

    def test_null_ref_no_warning(self):
        ro = _make_reader_output()
        c = _make_candidate(source_reader_output_ref=None)
        output = _make_output(candidates=[c])
        report = validate_analyst_output(output, ro)
        assert report.passed
        assert not any("VAL-A07" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_fully_valid_passes_cleanly(self):
        ro = _make_reader_output()
        c = _make_candidate(source_reader_output_ref="Melee attacks")
        output = _make_output(candidates=[c])
        report = validate_analyst_output(output, ro)
        assert report.passed
        assert report.errors == []
        assert report.warnings == []

    def test_report_summary_format(self):
        ro = _make_reader_output()
        output = _make_output()
        report = validate_analyst_output(output, ro)
        summary = report.summary()
        assert ro.chunk_id in summary
