"""
reviewer_result.py — Data Contracts §7–§8
============================================
Reviewer Agent output schemas: ReviewerResult and ReviewerFeedback.

Schema source: Data Contracts Specification §7 (ReviewerResult)
              and §8 (ReviewerFeedback).

Design notes:
  * ``ReviewerResult`` captures the verdict (Valid/Invalid) and which
    AQUSA criteria failed.
  * ``ReviewerFeedback`` provides structured, actionable feedback to the
    Generator Agent when a draft is Invalid — this is the feedback loop
    input to SRS-010.
  * VAL-023: When status = Invalid, ``failing_criteria`` MUST be non-empty.
  * Contradiction check: When status = Valid, ``failing_criteria`` MUST be
    empty (per MAS Arch §4 failure conditions).
  * ``failing_criteria`` enum values are fixed to the 4 AQUSA categories
    defined in Data Contracts §7: ``completeness``, ``clarity``,
    ``unambiguous``, ``gdd_traceability``.
  * One ``ReviewerResult`` per Reviewer invocation; one ``ReviewerFeedback``
    per ``Invalid`` result (one-to-one when present).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Criterion type ─────────────────────────────────────────────────────────────

VALID_CRITERIA = frozenset({"completeness", "clarity", "unambiguous", "gdd_traceability"})

CriterionType = Literal["completeness", "clarity", "unambiguous", "gdd_traceability"]


# ── ReviewerResult ─────────────────────────────────────────────────────────────

class ReviewerResult(BaseModel):
    """
    Verdict produced by the Reviewer Agent for a given draft submission.

    One ``ReviewerResult`` per Reviewer invocation (i.e., per draft
    submission, including each resubmission after revision).  Many
    ``ReviewerResult`` records may exist for the same ``draft_id`` over
    its revision history, ordered by ``evaluated_at_iteration``.

    Attributes
    ----------
    review_id:
        Unique identifier for this review.
        Format: ``{draft_id}__review_iter{N}``.
    draft_id:
        Foreign key to ``GeneratedUserStory.draft_id``.
    status:
        ``"Valid"`` or ``"Invalid"``.
    evaluated_at_iteration:
        The ``iteration_count`` of the draft at the time of this review.
    failing_criteria:
        AQUSA criteria that failed.  Required non-empty when
        status = Invalid (VAL-023).  Must be empty when status = Valid.
    feedback_ref:
        Foreign key to ``ReviewerFeedback.feedback_id``.
        Populated when status = Invalid.
    """

    model_config = ConfigDict(frozen=True)

    review_id: str = Field(
        ...,
        description="Unique review identifier. Format: {draft_id}__review_iter{N}.",
    )
    draft_id: str = Field(
        ...,
        description="FK to GeneratedUserStory.draft_id.",
    )
    status: Literal["Valid", "Invalid"] = Field(
        ...,
        description='Verdict: "Valid" or "Invalid".',
    )
    evaluated_at_iteration: int = Field(
        ...,
        ge=0,
        description="The iteration_count of the draft at review time.",
    )
    failing_criteria: List[CriterionType] = Field(
        default_factory=list,
        description=(
            "AQUSA criteria that failed. Must be non-empty when status=Invalid "
            "(VAL-023). Must be empty when status=Valid."
        ),
    )
    feedback_ref: Optional[str] = Field(
        default=None,
        description=(
            "FK to ReviewerFeedback.feedback_id. "
            "Populated when status=Invalid."
        ),
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("review_id", "draft_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v

    @field_validator("failing_criteria")
    @classmethod
    def criteria_values_valid(cls, v: List[str]) -> List[str]:
        for c in v:
            if c not in VALID_CRITERIA:
                raise ValueError(
                    f"Invalid criterion {c!r}. "
                    f"Allowed: {sorted(VALID_CRITERIA)}"
                )
        return v

    @model_validator(mode="after")
    def status_criteria_consistency(self) -> "ReviewerResult":
        """
        Enforce VAL-023 and the contradiction check:
        - Invalid → failing_criteria must be non-empty.
        - Valid → failing_criteria must be empty.
        """
        if self.status == "Invalid" and len(self.failing_criteria) == 0:
            raise ValueError(
                "VAL-023: status='Invalid' requires at least one entry "
                "in failing_criteria."
            )
        if self.status == "Valid" and len(self.failing_criteria) > 0:
            raise ValueError(
                "Contradiction: status='Valid' but failing_criteria is "
                f"non-empty: {self.failing_criteria}. A Valid verdict "
                "must have empty failing_criteria."
            )
        return self


# ── ReviewerFeedback ───────────────────────────────────────────────────────────

class ReviewerFeedback(BaseModel):
    """
    Structured, actionable feedback returned to the Generator Agent when
    a draft is Invalid (Data Contracts §8).

    One ``ReviewerFeedback`` per ``Invalid`` ``ReviewerResult`` (one-to-one
    when present).  Consumed directly by the next ``GeneratedUserStory``
    revision pass.

    Attributes
    ----------
    feedback_id:
        Unique identifier.  Format: ``{review_id}__feedback``.
    review_id:
        FK to ``ReviewerResult.review_id``.
    draft_id:
        FK to ``GeneratedUserStory.draft_id``.
    failing_criteria:
        Same criteria list as in the corresponding ``ReviewerResult``.
    feedback_text:
        Specific, actionable explanation per failing criterion (VAL-023).
    unsupported_claim_excerpt:
        If ``gdd_traceability`` failed, the specific unsupported phrase
        from the draft, to guide revision without inventing new content.
    """

    model_config = ConfigDict(frozen=True)

    feedback_id: str = Field(
        ...,
        description="Unique feedback ID. Format: {review_id}__feedback.",
    )
    review_id: str = Field(
        ...,
        description="FK to ReviewerResult.review_id.",
    )
    draft_id: str = Field(
        ...,
        description="FK to GeneratedUserStory.draft_id.",
    )
    failing_criteria: List[CriterionType] = Field(
        ...,
        min_length=1,
        description="AQUSA criteria that failed (same as ReviewerResult).",
    )
    feedback_text: str = Field(
        ...,
        description="Specific, actionable feedback per failing criterion (VAL-023).",
    )
    unsupported_claim_excerpt: Optional[str] = Field(
        default=None,
        description=(
            "If gdd_traceability failed, the specific unsupported phrase "
            "from the draft."
        ),
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("feedback_id", "review_id", "draft_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v

    @field_validator("feedback_text")
    @classmethod
    def feedback_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("feedback_text must not be empty.")
        return v.strip()

    @field_validator("failing_criteria")
    @classmethod
    def criteria_values_valid(cls, v: List[str]) -> List[str]:
        for c in v:
            if c not in VALID_CRITERIA:
                raise ValueError(
                    f"Invalid criterion {c!r}. "
                    f"Allowed: {sorted(VALID_CRITERIA)}"
                )
        return v
