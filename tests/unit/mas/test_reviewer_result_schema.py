"""
tests/unit/mas/test_reviewer_result_schema.py
===============================================
Unit tests for ReviewerResult and ReviewerFeedback Pydantic schemas
(Data Contracts §7–§8).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.reviewer_result import (
    VALID_CRITERIA,
    ReviewerFeedback,
    ReviewerResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _valid_result(**kwargs) -> dict:
    """Default fields for a Valid ReviewerResult."""
    defaults = dict(
        review_id="doc1__cand_0000__draft__review_iter0",
        draft_id="doc1__cand_0000__draft",
        status="Valid",
        evaluated_at_iteration=0,
        failing_criteria=[],
        feedback_ref=None,
    )
    defaults.update(kwargs)
    return defaults


def _invalid_result(**kwargs) -> dict:
    """Default fields for an Invalid ReviewerResult."""
    defaults = dict(
        review_id="doc1__cand_0000__draft__review_iter0",
        draft_id="doc1__cand_0000__draft",
        status="Invalid",
        evaluated_at_iteration=0,
        failing_criteria=["completeness"],
        feedback_ref="doc1__cand_0000__draft__review_iter0__feedback",
    )
    defaults.update(kwargs)
    return defaults


def _valid_feedback(**kwargs) -> dict:
    """Default fields for a ReviewerFeedback."""
    defaults = dict(
        feedback_id="doc1__cand_0000__draft__review_iter0__feedback",
        review_id="doc1__cand_0000__draft__review_iter0",
        draft_id="doc1__cand_0000__draft",
        failing_criteria=["completeness"],
        feedback_text="The role field is too vague. Use a specific stakeholder.",
        unsupported_claim_excerpt=None,
    )
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# ReviewerResult — Valid construction
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerResultValid:

    def test_valid_construction(self):
        r = ReviewerResult(**_valid_result())
        assert r.status == "Valid"
        assert r.failing_criteria == []
        assert r.feedback_ref is None

    def test_valid_with_all_fields(self):
        r = ReviewerResult(**_valid_result(
            evaluated_at_iteration=3,
        ))
        assert r.evaluated_at_iteration == 3

    def test_frozen_model(self):
        r = ReviewerResult(**_valid_result())
        with pytest.raises(ValidationError):
            r.status = "Invalid"


# ─────────────────────────────────────────────────────────────────────────────
# ReviewerResult — Invalid construction
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerResultInvalid:

    def test_invalid_construction(self):
        r = ReviewerResult(**_invalid_result())
        assert r.status == "Invalid"
        assert r.failing_criteria == ["completeness"]

    def test_invalid_multiple_criteria(self):
        r = ReviewerResult(**_invalid_result(
            failing_criteria=["completeness", "clarity", "gdd_traceability"],
        ))
        assert len(r.failing_criteria) == 3

    def test_invalid_all_criteria(self):
        r = ReviewerResult(**_invalid_result(
            failing_criteria=list(VALID_CRITERIA),
        ))
        assert len(r.failing_criteria) == 4

    def test_invalid_at_higher_iteration(self):
        r = ReviewerResult(**_invalid_result(evaluated_at_iteration=2))
        assert r.evaluated_at_iteration == 2


# ─────────────────────────────────────────────────────────────────────────────
# ReviewerResult — Validation errors
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerResultValidation:

    def test_invalid_empty_criteria_raises_val023(self):
        """VAL-023: Invalid + empty criteria → error."""
        with pytest.raises(ValidationError, match="VAL-023"):
            ReviewerResult(**_invalid_result(failing_criteria=[]))

    def test_valid_nonempty_criteria_contradiction(self):
        """Contradiction: Valid + non-empty criteria → error."""
        with pytest.raises(ValidationError, match="[Cc]ontradiction"):
            ReviewerResult(**_valid_result(failing_criteria=["clarity"]))

    def test_invalid_criterion_value(self):
        """Unknown criterion value → error."""
        with pytest.raises(ValidationError):
            ReviewerResult(**_invalid_result(failing_criteria=["nonexistent"]))

    def test_invalid_status_value(self):
        """Status not in enum → error."""
        with pytest.raises(ValidationError):
            ReviewerResult(**_valid_result(status="Maybe"))

    def test_empty_review_id(self):
        with pytest.raises(ValidationError):
            ReviewerResult(**_valid_result(review_id=""))

    def test_empty_draft_id(self):
        with pytest.raises(ValidationError):
            ReviewerResult(**_valid_result(draft_id=""))

    def test_whitespace_only_review_id(self):
        with pytest.raises(ValidationError):
            ReviewerResult(**_valid_result(review_id="   "))

    def test_negative_iteration(self):
        with pytest.raises(ValidationError):
            ReviewerResult(**_valid_result(evaluated_at_iteration=-1))


# ─────────────────────────────────────────────────────────────────────────────
# ReviewerFeedback — Valid construction
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerFeedbackValid:

    def test_basic_construction(self):
        f = ReviewerFeedback(**_valid_feedback())
        assert f.feedback_text
        assert f.failing_criteria == ["completeness"]

    def test_with_unsupported_claim(self):
        f = ReviewerFeedback(**_valid_feedback(
            failing_criteria=["gdd_traceability"],
            unsupported_claim_excerpt="multiplayer lobby system",
        ))
        assert f.unsupported_claim_excerpt == "multiplayer lobby system"

    def test_multiple_criteria(self):
        f = ReviewerFeedback(**_valid_feedback(
            failing_criteria=["completeness", "unambiguous"],
        ))
        assert len(f.failing_criteria) == 2

    def test_frozen_model(self):
        f = ReviewerFeedback(**_valid_feedback())
        with pytest.raises(ValidationError):
            f.feedback_text = "changed"


# ─────────────────────────────────────────────────────────────────────────────
# ReviewerFeedback — Validation errors
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerFeedbackValidation:

    def test_empty_criteria_raises(self):
        """Feedback must have at least one failing criterion."""
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(failing_criteria=[]))

    def test_empty_feedback_text(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(feedback_text=""))

    def test_whitespace_feedback_text(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(feedback_text="   "))

    def test_empty_feedback_id(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(feedback_id=""))

    def test_empty_review_id(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(review_id=""))

    def test_empty_draft_id(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(draft_id=""))

    def test_invalid_criterion_value(self):
        with pytest.raises(ValidationError):
            ReviewerFeedback(**_valid_feedback(
                failing_criteria=["nonexistent_criterion"],
            ))


# ─────────────────────────────────────────────────────────────────────────────
# VALID_CRITERIA constant
# ─────────────────────────────────────────────────────────────────────────────

class TestValidCriteria:

    def test_contains_expected_values(self):
        expected = {"completeness", "clarity", "unambiguous", "gdd_traceability"}
        assert VALID_CRITERIA == expected

    def test_is_frozen(self):
        with pytest.raises((TypeError, AttributeError)):
            VALID_CRITERIA.add("new_criterion")
