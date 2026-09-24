"""
evaluator_output_validator.py — EvaluationResult post-processing validator
===========================================================================
Validates an ``EvaluationResult`` against the input story collection.

Validation rules:
  VAL-E01  All 5 metrics present (numeric or null) — no field missing.
  VAL-E02  run_id and document_id not empty.
  VAL-E03  Null metrics have corresponding entry in null_reasons (VAL-029).
  VAL-E04  Numeric metrics within valid ranges (rates 0–100, diversity 0–1).
  VAL-E05  total_stories matches len(input_stories).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorValidationReport:
    """
    Validation report for an EvaluationResult.

    Attributes
    ----------
    evaluation_id:
        The evaluation_id this report pertains to.
    passed:
        True when no blocking errors found.
    errors:
        Blocking violations.
    warnings:
        Non-blocking observations.
    """
    evaluation_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"EvaluatorValidation [{self.evaluation_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


def validate_evaluator_output(
    result: EvaluationResult,
    input_stories: List[FinalUserStory],
) -> EvaluatorValidationReport:
    """
    Validate an ``EvaluationResult`` against the input story collection.

    Parameters
    ----------
    result:
        The ``EvaluationResult`` produced by the evaluator agent.
    input_stories:
        The stories that were evaluated.

    Returns
    -------
    EvaluatorValidationReport
        Report with ``passed=True`` if no blocking errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # VAL-E01 — All 5 metric fields are present (always true via Pydantic schema,
    # but we check that no field is unexpectedly absent from the model)
    metric_fields = [
        "hallucination_rate",
        "redundancy_rate",
        "aqusa_score",
        "coverage",
        "diversity",
    ]
    for mf in metric_fields:
        if not hasattr(result, mf):
            errors.append(
                f"VAL-E01: Metric field '{mf}' is missing from EvaluationResult."
            )

    # VAL-E02 — run_id and document_id not empty
    if not result.run_id.strip():
        errors.append("VAL-E02: run_id is empty.")
    if not result.document_id.strip():
        errors.append("VAL-E02: document_id is empty.")

    # VAL-E03 — Null metrics have null_reasons entry (VAL-029)
    for mf in metric_fields:
        value = getattr(result, mf, None)
        if value is None and mf not in result.null_reasons:
            errors.append(
                f"VAL-E03 VAL-029: Metric '{mf}' is null but has no "
                f"entry in null_reasons."
            )

    # VAL-E04 — Numeric metrics within valid ranges
    rate_fields = ["hallucination_rate", "redundancy_rate", "aqusa_score", "coverage"]
    for mf in rate_fields:
        value = getattr(result, mf, None)
        if value is not None:
            if not (0.0 <= value <= 100.0):
                errors.append(
                    f"VAL-E04: {mf}={value:.4f} is outside valid range [0, 100]."
                )

    if result.diversity is not None:
        if not (0.0 <= result.diversity <= 1.0):
            errors.append(
                f"VAL-E04: diversity={result.diversity:.4f} is outside "
                f"valid range [0.0, 1.0]."
            )

    # VAL-E05 — total_stories matches input
    if result.total_stories != len(input_stories):
        errors.append(
            f"VAL-E05: result.total_stories={result.total_stories} does not "
            f"match len(input_stories)={len(input_stories)}."
        )

    # Warnings: sanity checks
    if result.total_stories == 0 and input_stories:
        warnings.append(
            "total_stories is 0 but input_stories is non-empty."
        )
    if result.aqusa_score == 100.0:
        warnings.append(
            "aqusa_score=100.0: all stories passed every AQUSA criterion. "
            "Verify this is expected."
        )
    if result.hallucination_rate == 0.0:
        warnings.append(
            "hallucination_rate=0.0: no stories flagged as hallucinated. "
            "Verify this is expected."
        )

    passed = len(errors) == 0
    report = EvaluatorValidationReport(
        evaluation_id=result.evaluation_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "EvaluatorOutput validation passed for %s "
                "(pipeline=%s, stories=%d)",
                result.evaluation_id,
                result.pipeline_type,
                result.total_stories,
            )
    else:
        logger.error(report.summary())

    return report
