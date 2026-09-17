"""
tests/unit/mas/test_reader_output_validator.py
================================================
Unit tests for validate_reader_output() — the post-processing validator
that checks traceability and suspicious outputs (VAL-R01 through VAL-R05).

All tests are OFFLINE — no LLM calls.
"""
from __future__ import annotations

import pytest

from gdd_userstory_mas.mas.reader_output_validator import (
    ReaderValidationReport,
    validate_reader_output,
)
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(**kwargs) -> GDDChunk:
    defaults = dict(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        text="Some GDD text here.",
        position=0,
        token_count=10,
    )
    defaults.update(kwargs)
    return GDDChunk(**defaults)


def _make_output(**kwargs) -> ReaderOutput:
    defaults = dict(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        themes=[],
        gameplay_elements=[],
        systems=[],
        characters=[],
        ui_elements=[],
        narrative=[],
    )
    defaults.update(kwargs)
    return ReaderOutput(**defaults)


def _item(content: str, excerpt: str | None = "some quote") -> EvidencedItem:
    return EvidencedItem(content=content, source_excerpt=excerpt)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-R01 — chunk_id traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkIdTraceability:

    def test_matching_chunk_id_passes(self):
        chunk = _make_chunk(chunk_id="doc1__chunk_0000")
        output = _make_output(chunk_id="doc1__chunk_0000")
        report = validate_reader_output(output, chunk)
        assert report.passed
        assert not any("VAL-R01" in e for e in report.errors)

    def test_mismatched_chunk_id_fails(self):
        chunk = _make_chunk(chunk_id="doc1__chunk_0000")
        output = _make_output(chunk_id="WRONG_CHUNK_ID")
        report = validate_reader_output(output, chunk)
        assert not report.passed
        assert any("VAL-R01" in e for e in report.errors)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-R02 — document_id traceability
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdTraceability:

    def test_matching_document_id_passes(self):
        chunk = _make_chunk(document_id="doc1")
        output = _make_output(document_id="doc1")
        report = validate_reader_output(output, chunk)
        assert report.passed

    def test_mismatched_document_id_fails(self):
        chunk = _make_chunk(document_id="doc1")
        output = _make_output(document_id="WRONG_DOC")
        report = validate_reader_output(output, chunk)
        assert not report.passed
        assert any("VAL-R02" in e for e in report.errors)

    def test_both_ids_wrong_two_errors(self):
        chunk = _make_chunk(chunk_id="c1", document_id="doc1")
        output = _make_output(chunk_id="WRONG_C", document_id="WRONG_D")
        report = validate_reader_output(output, chunk)
        assert not report.passed
        assert len(report.errors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# VAL-R04 — empty output warning
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyOutputWarning:

    def test_all_empty_issues_warning_but_passes(self):
        chunk = _make_chunk()
        output = _make_output()
        assert output.is_empty
        report = validate_reader_output(output, chunk)
        assert report.passed           # not a blocking error
        assert any("VAL-R04" in w for w in report.warnings)

    def test_non_empty_output_no_warning(self):
        chunk = _make_chunk()
        output = _make_output(themes=["action"])
        report = validate_reader_output(output, chunk)
        assert report.passed
        assert not any("VAL-R04" in w for w in report.warnings)

    def test_only_items_no_themes_no_warning(self):
        chunk = _make_chunk()
        output = _make_output(gameplay_elements=[_item("A mechanic")])
        report = validate_reader_output(output, chunk)
        assert report.passed
        assert not any("VAL-R04" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# VAL-R05 — source_excerpt non-empty when present (warning only)
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceExcerptWarning:

    def test_null_excerpt_no_warning(self):
        chunk = _make_chunk()
        output = _make_output(
            gameplay_elements=[EvidencedItem(content="Something", source_excerpt=None)]
        )
        report = validate_reader_output(output, chunk)
        assert report.passed
        # VAL-R05 should NOT fire for null excerpts (null is intentional)
        assert not any("VAL-R05" in w for w in report.warnings)

    def test_non_null_valid_excerpt_no_warning(self):
        chunk = _make_chunk()
        output = _make_output(
            systems=[EvidencedItem(content="Combat", source_excerpt="combat system")]
        )
        report = validate_reader_output(output, chunk)
        assert report.passed
        assert not any("VAL-R05" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — full valid output
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_full_valid_output_passes_with_no_warnings(self):
        chunk = _make_chunk()
        output = _make_output(
            themes=["action", "survival"],
            gameplay_elements=[_item("Melee combat", "melee combat")],
            systems=[_item("Health system", "health points")],
            characters=[_item("Player character", "the player")],
            ui_elements=[_item("Health bar", "health bar in the HUD")],
            narrative=[_item("Apocalyptic setting", "wasteland")],
        )
        report = validate_reader_output(output, chunk)
        assert report.passed
        assert report.errors == []
        assert report.warnings == []

    def test_report_summary_format(self):
        chunk = _make_chunk()
        output = _make_output()
        report = validate_reader_output(output, chunk)
        summary = report.summary()
        assert "PASSED" in summary
        assert chunk.chunk_id in summary


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReport dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReport:

    def test_passed_true_when_no_errors(self):
        report = ReaderValidationReport(chunk_id="c1", passed=True)
        assert report.passed
        assert report.errors == []
        assert report.warnings == []

    def test_summary_shows_failed_with_errors(self):
        report = ReaderValidationReport(
            chunk_id="c1",
            passed=False,
            errors=["VAL-R01: something wrong"],
        )
        summary = report.summary()
        assert "FAILED" in summary
        assert "VAL-R01" in summary
