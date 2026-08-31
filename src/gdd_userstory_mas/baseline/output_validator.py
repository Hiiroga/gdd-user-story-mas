"""
output_validator.py — VAL-031 / VAL-032 output validation
===========================================================
Post-generation validation of the baseline's ``FinalUserStory`` list.

Validation rules applied here:

  VAL-031: Baseline output schema must match MAS output schema field-
           for-field (NFR-006).  Enforced by Pydantic at construction;
           this module adds cross-story checks.
  VAL-032: No intermediate review/redundancy logic is invoked (structural,
           enforced by the pipeline; this module can only detect
           post-hoc schema violations).
  VAL-020: full_text must begin with "As a " (enforced in FinalUserStory
           field_validator; re-checked here for reporting).

Additional cross-story checks:
  - No two stories in the same run have the same id.
  - Every source_chunk_id references a chunk_id from the input chunks.
  - story count is within reasonable bounds (warn if 0 for a non-empty doc).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Set

from gdd_userstory_mas.schemas.final_user_story import FinalUserStory
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk

logger = logging.getLogger(__name__)


# ── Validation result dataclass ────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """
    Structured report of the baseline output validation pass.

    Attributes
    ----------
    total_stories:
        Number of ``FinalUserStory`` objects validated.
    valid_count:
        Stories that passed all checks.
    invalid_count:
        Stories that failed one or more checks.
    errors:
        Per-story error messages keyed by story id (or index if id is empty).
    warnings:
        Non-fatal observations.
    passed:
        True if no errors were found (warnings are non-blocking).
    """

    total_stories: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.invalid_count == 0 and not self.errors


# ── Public function ────────────────────────────────────────────────────────────

def validate_baseline_output(
    stories: List[FinalUserStory],
    source_chunks: List[GDDChunk],
) -> ValidationReport:
    """
    Validate a list of ``FinalUserStory`` objects produced by the baseline.

    Parameters
    ----------
    stories:
        The list of ``FinalUserStory`` objects to validate.
    source_chunks:
        The ``GDDChunk`` list that was fed to the baseline agent (used
        to verify ``source_chunk_ids`` references).

    Returns
    -------
    ValidationReport
        Structured result; check ``report.passed`` for a pass/fail signal.
    """
    report = ValidationReport(total_stories=len(stories))
    valid_chunk_ids: Set[str] = {c.chunk_id for c in source_chunks}
    seen_ids: Set[str] = set()

    for idx, story in enumerate(stories):
        story_label = story.id or f"story[{idx}]"
        story_errors: List[str] = []

        # VAL-031: pipeline_type must be "Baseline"
        if story.pipeline_type != "Baseline":
            story_errors.append(
                f"VAL-031: pipeline_type must be 'Baseline', got '{story.pipeline_type}'."
            )

        # VAL-031: validation_status must be "Unreviewed" (no Reviewer stage)
        if story.validation_status != "Unreviewed":
            story_errors.append(
                f"VAL-032: validation_status must be 'Unreviewed' for Baseline, "
                f"got '{story.validation_status}'."
            )

        # VAL-031: iteration_count_final must be None
        if story.iteration_count_final is not None:
            story_errors.append(
                "VAL-032: iteration_count_final must be None for Baseline."
            )

        # VAL-020: full_text format
        if not story.full_text.strip().startswith("As a"):
            story_errors.append(
                f"VAL-020: full_text does not start with 'As a': {story.full_text[:80]!r}"
            )

        # Traceability: every source_chunk_id must be a known chunk
        unknown_chunks = [
            cid for cid in story.source_chunk_ids
            if cid not in valid_chunk_ids
        ]
        if unknown_chunks:
            story_errors.append(
                f"TRACE-001: unknown source_chunk_ids: {unknown_chunks}"
            )

        # Uniqueness: story IDs must not repeat within a run
        if story.id in seen_ids:
            story_errors.append(f"Duplicate story id: '{story.id}'.")
        else:
            seen_ids.add(story.id)

        if story_errors:
            report.invalid_count += 1
            for err in story_errors:
                report.errors.append(f"[{story_label}] {err}")
            logger.warning("[output_validator] Story %s failed: %s", story_label, story_errors)
        else:
            report.valid_count += 1

    # Cross-collection checks
    if len(stories) == 0 and len(source_chunks) > 0:
        report.warnings.append(
            "No user stories were extracted from a non-empty document. "
            "Check that the LLM prompt is reaching the model and the GDD "
            "contains extractable requirements."
        )
        logger.warning("[output_validator] Zero stories extracted from %d chunks.", len(source_chunks))

    if report.passed:
        logger.info(
            "[output_validator] Validation passed: %d stories, all valid.",
            report.valid_count,
        )
    else:
        logger.error(
            "[output_validator] Validation FAILED: %d/%d stories invalid. Errors: %s",
            report.invalid_count, report.total_stories, report.errors,
        )

    return report
