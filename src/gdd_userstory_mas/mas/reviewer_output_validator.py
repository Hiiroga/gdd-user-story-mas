"""
reviewer_output_validator.py — ReviewerResult post-processing validator
========================================================================
Validates a ``ReviewerResult`` (and optional ``ReviewerFeedback``) against
the source ``GeneratedUserStory`` to catch traceability violations and
logical contradictions.

Validation rules:
  VAL-R01  draft_id in result matches the story's draft_id.
  VAL-R02  evaluated_at_iteration matches story's iteration_count.
  VAL-R03  Valid status with non-empty failing_criteria → contradiction.
  VAL-R04  Invalid status with empty failing_criteria → error (VAL-023).
  VAL-R05  If Invalid, feedback must be provided (not None).
  VAL-R06  If feedback provided, feedback.draft_id matches story's draft_id.
  VAL-R07  If feedback provided, feedback.review_id matches result's review_id.
  VAL-R08  If gdd_traceability in failing_criteria, warn if
           unsupported_claim_excerpt is empty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reviewer_result import ReviewerFeedback, ReviewerResult

logger = logging.getLogger(__name__)


# ── Validation Report ──────────────────────────────────────────────────────────

@dataclass
class ReviewerValidationReport:
    """
    Result of validating a ``ReviewerResult`` + optional ``ReviewerFeedback``.

    Attributes
    ----------
    review_id:
        The review_id this report pertains to.
    passed:
        True when no blocking errors found.
    errors:
        Blocking violations — result should be rejected.
    warnings:
        Non-blocking observations.
    """
    review_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"ReviewerValidation [{self.review_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


# ── Validator function ─────────────────────────────────────────────────────────

def validate_reviewer_output(
    result: ReviewerResult,
    feedback: Optional[ReviewerFeedback],
    story: GeneratedUserStory,
) -> ReviewerValidationReport:
    """
    Validate a ``ReviewerResult`` and optional ``ReviewerFeedback``
    against the source ``GeneratedUserStory``.

    Parameters
    ----------
    result:
        The ``ReviewerResult`` produced by the reviewer agent.
    feedback:
        The ``ReviewerFeedback`` (if status=Invalid), or None.
    story:
        The ``GeneratedUserStory`` that was reviewed.

    Returns
    -------
    ReviewerValidationReport
        Report with ``passed=True`` if no blocking errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # VAL-R01 — draft_id match
    if result.draft_id != story.draft_id:
        errors.append(
            f"VAL-R01: result.draft_id={result.draft_id!r} does not match "
            f"story.draft_id={story.draft_id!r}."
        )

    # VAL-R02 — evaluated_at_iteration match
    if result.evaluated_at_iteration != story.iteration_count:
        errors.append(
            f"VAL-R02: result.evaluated_at_iteration="
            f"{result.evaluated_at_iteration} does not match "
            f"story.iteration_count={story.iteration_count}."
        )

    # VAL-R03 — Valid + non-empty failing_criteria → contradiction
    if result.status == "Valid" and len(result.failing_criteria) > 0:
        errors.append(
            f"VAL-R03: status='Valid' but failing_criteria is non-empty: "
            f"{result.failing_criteria}. Contradiction."
        )

    # VAL-R04 — Invalid + empty failing_criteria → VAL-023 violation
    if result.status == "Invalid" and len(result.failing_criteria) == 0:
        errors.append(
            f"VAL-R04 VAL-023: status='Invalid' but failing_criteria is "
            f"empty. At least one criterion must be specified."
        )

    # VAL-R05 — Invalid requires feedback
    if result.status == "Invalid" and feedback is None:
        errors.append(
            f"VAL-R05: status='Invalid' but no ReviewerFeedback was "
            f"provided. Feedback is required for Invalid verdicts."
        )

    # VAL-R06 — feedback.draft_id match
    if feedback is not None and feedback.draft_id != story.draft_id:
        errors.append(
            f"VAL-R06: feedback.draft_id={feedback.draft_id!r} does not "
            f"match story.draft_id={story.draft_id!r}."
        )

    # VAL-R07 — feedback.review_id match
    if feedback is not None and feedback.review_id != result.review_id:
        errors.append(
            f"VAL-R07: feedback.review_id={feedback.review_id!r} does not "
            f"match result.review_id={result.review_id!r}."
        )

    # VAL-R08 — gdd_traceability without unsupported_claim_excerpt
    if (
        feedback is not None
        and "gdd_traceability" in feedback.failing_criteria
        and not feedback.unsupported_claim_excerpt
    ):
        warnings.append(
            f"VAL-R08: 'gdd_traceability' is in failing_criteria but "
            f"unsupported_claim_excerpt is empty/null. Consider "
            f"providing the specific unsupported phrase."
        )

    passed = len(errors) == 0
    report = ReviewerValidationReport(
        review_id=result.review_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "ReviewerOutput validation passed for review %s "
                "(draft=%s, status=%s)",
                result.review_id,
                result.draft_id,
                result.status,
            )
    else:
        logger.error(report.summary())

    return report
