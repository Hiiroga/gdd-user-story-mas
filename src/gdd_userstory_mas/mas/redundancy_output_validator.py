"""
redundancy_output_validator.py — RedundancyAnalysis post-processing validator
===============================================================================
Validates a ``RedundancyAnalysis`` against the input story collection
to catch traceability violations and logical inconsistencies.

Validation rules:
  VAL-D01  Every duplicate group has ≥2 story_ids (VAL-026).
  VAL-D02  All story_ids in groups reference actual input stories.
  VAL-D03  No story appears in multiple duplicate groups.
  VAL-D04  unique_story_ids + grouped story_ids = all input story IDs.
  VAL-D05  Every merge action references a valid group.
  VAL-D06  Merged stories conform to user story format (if action=merge).
  VAL-D07  No information loss — all input stories accounted for.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis

logger = logging.getLogger(__name__)


# ── Validation Report ──────────────────────────────────────────────────────────

@dataclass
class RedundancyValidationReport:
    """
    Result of validating a ``RedundancyAnalysis``.

    Attributes
    ----------
    analysis_id:
        The analysis_id this report pertains to.
    passed:
        True when no blocking errors found.
    errors:
        Blocking violations.
    warnings:
        Non-blocking observations.
    """
    analysis_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"RedundancyValidation [{self.analysis_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


# ── Format pattern ─────────────────────────────────────────────────────────────

_FORMAT_PATTERN = re.compile(
    r"^As a .+,\s+I want .+,\s+so that .+\.$",
    re.IGNORECASE,
)


# ── Validator function ─────────────────────────────────────────────────────────

def validate_redundancy_output(
    analysis: RedundancyAnalysis,
    input_stories: List[GeneratedUserStory],
) -> RedundancyValidationReport:
    """
    Validate a ``RedundancyAnalysis`` against the input story collection.

    Parameters
    ----------
    analysis:
        The ``RedundancyAnalysis`` produced by the redundancy agent.
    input_stories:
        The original input stories passed to the agent.

    Returns
    -------
    RedundancyValidationReport
        Report with ``passed=True`` if no blocking errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    input_ids: Set[str] = {s.draft_id for s in input_stories}

    # VAL-D01 — Every group has ≥2 story_ids
    for group in analysis.duplicate_groups:
        if len(group.story_ids) < 2:
            errors.append(
                f"VAL-D01 VAL-026: group {group.group_id} has "
                f"{len(group.story_ids)} story_ids (minimum is 2)."
            )

    # VAL-D02 — All story_ids reference actual input stories
    for group in analysis.duplicate_groups:
        for sid in group.story_ids:
            if sid not in input_ids:
                errors.append(
                    f"VAL-D02: group {group.group_id} references unknown "
                    f"story_id {sid!r}."
                )

    # VAL-D03 — No story in multiple groups
    seen_in_groups: Set[str] = set()
    for group in analysis.duplicate_groups:
        for sid in group.story_ids:
            if sid in seen_in_groups:
                errors.append(
                    f"VAL-D03: story {sid!r} appears in multiple "
                    f"duplicate groups."
                )
            seen_in_groups.add(sid)

    # VAL-D04 — unique + grouped = all input
    all_accounted = set(analysis.unique_story_ids) | seen_in_groups
    missing = input_ids - all_accounted
    extra = all_accounted - input_ids
    if missing:
        errors.append(
            f"VAL-D04: {len(missing)} input stories not accounted for "
            f"in unique_story_ids or duplicate_groups: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}."
        )
    if extra:
        warnings.append(
            f"VAL-D04: {len(extra)} story IDs in output not found in "
            f"input: {sorted(extra)[:5]}."
        )

    # VAL-D05 — Every merge action references a valid group
    group_ids = {g.group_id for g in analysis.duplicate_groups}
    for action in analysis.merge_actions:
        if action.group_id not in group_ids:
            errors.append(
                f"VAL-D05: merge action references unknown group "
                f"{action.group_id!r}."
            )

    # VAL-D06 — Merged stories conform to format (if action=merge)
    for action in analysis.merge_actions:
        if action.action == "merge" and action.merged_full_text:
            if not _FORMAT_PATTERN.match(action.merged_full_text.strip()):
                warnings.append(
                    f"VAL-D06: merged full_text for group "
                    f"{action.group_id} does not match the expected "
                    f"format: {action.merged_full_text[:80]!r}."
                )

    # VAL-D07 — No information loss
    if analysis.input_story_count != len(input_stories):
        errors.append(
            f"VAL-D07: analysis.input_story_count="
            f"{analysis.input_story_count} does not match "
            f"len(input_stories)={len(input_stories)}."
        )

    passed = len(errors) == 0
    report = RedundancyValidationReport(
        analysis_id=analysis.analysis_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "RedundancyOutput validation passed for %s "
                "(%d groups, %d actions, status=%s)",
                analysis.analysis_id,
                len(analysis.duplicate_groups),
                len(analysis.merge_actions),
                analysis.resolution_status,
            )
    else:
        logger.error(report.summary())

    return report
