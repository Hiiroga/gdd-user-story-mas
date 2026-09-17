"""
tests/unit/mas/test_candidate_requirement_schema.py
=====================================================
Unit tests for CandidateRequirement and AnalystOutput Pydantic schemas.
All tests are OFFLINE — no LLM calls, no file I/O.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.candidate_requirement import (
    AnalystOutput,
    CandidateRequirement,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate(**kwargs) -> CandidateRequirement:
    defaults = dict(
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        requirement_text="The game must display the player's health in the HUD.",
        perspective="player",
        domain="ui",
        source_reader_output_ref=None,
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
# CandidateRequirement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateRequirementConstruction:

    def test_valid_minimal(self):
        c = _make_candidate()
        assert c.candidate_id == "doc1__cand_0000"
        assert c.perspective == "player"
        assert c.domain == "ui"

    def test_all_perspectives_accepted(self):
        for p in ("player", "system", "in-game entity", "dev team"):
            c = _make_candidate(perspective=p)
            assert c.perspective == p

    def test_all_domains_accepted(self):
        for d in (
            "gameplay", "ui", "narrative", "systems", "level design",
            "audio", "visuals", "accessibility", "meta", "other",
        ):
            c = _make_candidate(domain=d)
            assert c.domain == d

    def test_source_ref_optional(self):
        c = _make_candidate(source_reader_output_ref=None)
        assert c.source_reader_output_ref is None

    def test_source_ref_populated(self):
        c = _make_candidate(source_reader_output_ref="Melee weapon attacks")
        assert c.source_reader_output_ref == "Melee weapon attacks"

    def test_source_ref_empty_string_becomes_none(self):
        c = _make_candidate(source_reader_output_ref="   ")
        assert c.source_reader_output_ref is None

    def test_frozen(self):
        c = _make_candidate()
        with pytest.raises(Exception):
            c.candidate_id = "changed"  # type: ignore[misc]


class TestCandidateRequirementValidation:

    def test_empty_candidate_id_raises(self):
        with pytest.raises(ValidationError, match="candidate_id"):
            _make_candidate(candidate_id="")

    def test_empty_source_chunk_id_raises(self):
        with pytest.raises(ValidationError, match="source_chunk_id"):
            _make_candidate(source_chunk_id="")

    def test_empty_document_id_raises(self):
        with pytest.raises(ValidationError, match="document_id"):
            _make_candidate(document_id="")

    def test_empty_requirement_text_raises(self):
        with pytest.raises(ValidationError, match="requirement_text"):
            _make_candidate(requirement_text="")

    def test_user_story_format_raises(self):
        """requirement_text must NOT start with 'As a'."""
        with pytest.raises(ValidationError, match="user-story format"):
            _make_candidate(requirement_text="As a player, I want health displayed.")

    def test_user_story_format_case_insensitive(self):
        """Check is case-insensitive."""
        with pytest.raises(ValidationError, match="user-story format"):
            _make_candidate(requirement_text="AS A player, I want something.")

    def test_invalid_perspective_raises(self):
        with pytest.raises(ValidationError):
            _make_candidate(perspective="developer")  # type: ignore[arg-type]

    def test_invalid_domain_raises(self):
        with pytest.raises(ValidationError):
            _make_candidate(domain="economy")  # type: ignore[arg-type]

    def test_requirement_text_stripped(self):
        c = _make_candidate(requirement_text="  Health must be shown.  ")
        assert c.requirement_text == "Health must be shown."


# ─────────────────────────────────────────────────────────────────────────────
# AnalystOutput tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalystOutput:

    def test_empty_candidates_valid(self):
        o = _make_output()
        assert o.is_empty
        assert o.candidate_count == 0

    def test_with_candidates(self):
        c1 = _make_candidate(candidate_id="doc1__cand_0000")
        c2 = _make_candidate(
            candidate_id="doc1__cand_0001",
            requirement_text="Enemy units must patrol their area when idle.",
            perspective="in-game entity",
            domain="gameplay",
        )
        o = _make_output(candidates=[c1, c2])
        assert o.candidate_count == 2
        assert not o.is_empty

    def test_empty_source_chunk_id_raises(self):
        with pytest.raises(ValidationError, match="source_chunk_id"):
            _make_output(source_chunk_id="")

    def test_empty_document_id_raises(self):
        with pytest.raises(ValidationError, match="document_id"):
            _make_output(document_id="")

    def test_perspective_counts(self):
        candidates = [
            _make_candidate(candidate_id=f"doc1__cand_{i:04d}", perspective=p)
            for i, p in enumerate(["player", "player", "system", "in-game entity"])
        ]
        o = _make_output(candidates=candidates)
        counts = o.perspective_counts()
        assert counts["player"] == 2
        assert counts["system"] == 1
        assert counts["in-game entity"] == 1
        assert counts["dev team"] == 0

    def test_domain_counts(self):
        candidates = [
            _make_candidate(
                candidate_id=f"doc1__cand_{i:04d}",
                domain=d,
                requirement_text=f"Requirement {i}.",
            )
            for i, d in enumerate(["gameplay", "gameplay", "ui"])
        ]
        o = _make_output(candidates=candidates)
        counts = o.domain_counts()
        assert counts["gameplay"] == 2
        assert counts["ui"] == 1

    def test_frozen(self):
        o = _make_output()
        with pytest.raises(Exception):
            o.source_chunk_id = "changed"  # type: ignore[misc]
