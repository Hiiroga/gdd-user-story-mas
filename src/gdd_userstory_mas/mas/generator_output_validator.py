"""
generator_output_validator.py — GeneratedUserStory post-processing validator
=============================================================================
Validates a ``GeneratedUserStory`` against its source ``CandidateRequirement``
to catch traceability violations before the draft is passed to the Reviewer.

Validation rules:
  VAL-G01  draft_id follows {candidate_id}__draft (TRACE-003).
  VAL-G02  candidate_id in story matches source candidate.candidate_id.
  VAL-G03  source_chunk_id inherited correctly from candidate (TRACE-001).
  VAL-G04  document_id matches candidate.document_id.
  VAL-G05  full_text format: must start with "As a " and end with "." (VAL-020).
  VAL-G06  full_text is consistent with role/action/benefit (VAL-020 cross-check).
  VAL-G07  iteration_count >= 0 (VAL-022).
  VAL-G08  role, action, benefit are all non-empty.
  VAL-G09  Warning if role, action, or benefit contain "As a", "I want",
           or "so that" (indicates LLM field leakage).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from gdd_userstory_mas.schemas.candidate_requirement import CandidateRequirement
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory

logger = logging.getLogger(__name__)


# ── Validation Report ──────────────────────────────────────────────────────────

@dataclass
class GeneratorValidationReport:
    """
    Result of validating a ``GeneratedUserStory``.

    Attributes
    ----------
    draft_id:
        The draft_id this report pertains to.
    passed:
        True when no blocking errors found.
    errors:
        Blocking violations — draft should be rejected.
    warnings:
        Non-blocking observations.
    """
    draft_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"GeneratorValidation [{self.draft_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


# ── Validator function ─────────────────────────────────────────────────────────

_FORMAT_PATTERN = re.compile(
    r"^As a .+,\s+I want .+,\s+so that .+\.$",
    re.IGNORECASE,
)


def validate_generator_output(
    story: GeneratedUserStory,
    candidate: CandidateRequirement,
) -> GeneratorValidationReport:
    """
    Validate a ``GeneratedUserStory`` against its source ``CandidateRequirement``.

    Parameters
    ----------
    story:
        The ``GeneratedUserStory`` produced by the generator agent.
    candidate:
        The ``CandidateRequirement`` that was passed to the agent.

    Returns
    -------
    GeneratorValidationReport
        Report with ``passed=True`` if no blocking errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # VAL-G01 — draft_id format (TRACE-003)
    expected_draft_id = f"{candidate.candidate_id}__draft"
    if story.draft_id != expected_draft_id:
        errors.append(
            f"VAL-G01 TRACE-003: story.draft_id={story.draft_id!r} does not match "
            f"expected={expected_draft_id!r}."
        )

    # VAL-G02 — candidate_id match
    if story.candidate_id != candidate.candidate_id:
        errors.append(
            f"VAL-G02: story.candidate_id={story.candidate_id!r} != "
            f"candidate.candidate_id={candidate.candidate_id!r}."
        )

    # VAL-G03 — source_chunk_id (TRACE-001)
    if story.source_chunk_id != candidate.source_chunk_id:
        errors.append(
            f"VAL-G03 TRACE-001: story.source_chunk_id={story.source_chunk_id!r} != "
            f"candidate.source_chunk_id={candidate.source_chunk_id!r}."
        )

    # VAL-G04 — document_id
    if story.document_id != candidate.document_id:
        errors.append(
            f"VAL-G04: story.document_id={story.document_id!r} != "
            f"candidate.document_id={candidate.document_id!r}."
        )

    # VAL-G05 — full_text format (VAL-020)
    if not _FORMAT_PATTERN.match(story.full_text.strip()):
        errors.append(
            f"VAL-G05 VAL-020: full_text does not match "
            f'"As a ..., I want ..., so that ....": {story.full_text[:120]!r}.'
        )

    # VAL-G06 — full_text consistency with components (VAL-020)
    def _normalise(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    expected_text = GeneratedUserStory.build_full_text(
        story.role, story.action, story.benefit
    )
    if _normalise(story.full_text) != _normalise(expected_text):
        errors.append(
            f"VAL-G06 VAL-020: full_text does not match role/action/benefit.\n"
            f"  Expected: {expected_text!r}\n"
            f"  Got     : {story.full_text!r}"
        )

    # VAL-G07 — iteration_count >= 0 (VAL-022)
    if story.iteration_count < 0:
        errors.append(
            f"VAL-G07 VAL-022: iteration_count={story.iteration_count} is negative."
        )

    # VAL-G08 — non-empty story fields (defence-in-depth; schema already enforces)
    for field_name in ("role", "action", "benefit"):
        val = getattr(story, field_name)
        if not val.strip():
            errors.append(
                f"VAL-G08: story.{field_name} is empty — should have been "
                f"caught by schema validation."
            )

    # VAL-G09 — detect field leakage (LLM included "As a/I want/so that" inside fields)
    leakage_checks = [
        ("role", ["as a ", "as an "]),
        ("action", ["i want ", "so that "]),
        ("benefit", ["as a ", "i want "]),
    ]
    for field_name, bad_prefixes in leakage_checks:
        val = getattr(story, field_name).lower()
        for prefix in bad_prefixes:
            if prefix in val:
                warnings.append(
                    f"VAL-G09: story.{field_name} contains suspicious phrase "
                    f"{prefix!r} — possible LLM field leakage."
                )
                break

    passed = len(errors) == 0
    report = GeneratorValidationReport(
        draft_id=story.draft_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "GeneratorOutput validation passed for draft %s (iter=%d): %r",
                story.draft_id,
                story.iteration_count,
                story.full_text[:60],
            )
    else:
        logger.error(report.summary())

    return report
