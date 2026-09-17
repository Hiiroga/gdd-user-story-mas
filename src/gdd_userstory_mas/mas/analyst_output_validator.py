"""
analyst_output_validator.py — AnalystOutput post-processing validator
======================================================================
Validates an ``AnalystOutput`` against its source ``ReaderOutput`` to
catch traceability violations and suspicious outputs.

Validation rules:
  VAL-A01  source_chunk_id in output matches reader_output.chunk_id (TRACE-001).
  VAL-A02  document_id in output matches reader_output.document_id.
  VAL-A03  All candidate_id values within the output are unique.
  VAL-A04  All source_chunk_id values in candidates match the output's
           source_chunk_id (TRACE-001 defence-in-depth).
  VAL-A05  requirement_text does not start with "As a" (user-story format guard).
  VAL-A06  If output is empty and ReaderOutput was non-empty, emit WARNING.
  VAL-A07  source_reader_output_ref, when set, must reference an EvidencedItem
           content string that exists in the ReaderOutput (traceability check).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Set

from gdd_userstory_mas.schemas.candidate_requirement import AnalystOutput
from gdd_userstory_mas.schemas.reader_output import ReaderOutput

logger = logging.getLogger(__name__)


# ── Validation Report ──────────────────────────────────────────────────────────

@dataclass
class AnalystValidationReport:
    """
    Result of validating a single ``AnalystOutput``.

    Attributes
    ----------
    source_chunk_id:
        The chunk ID this report pertains to.
    passed:
        True when no blocking errors found.
    errors:
        Blocking violations — output should be rejected if any present.
    warnings:
        Non-blocking observations — logged but do not reject.
    """
    source_chunk_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"AnalystValidation [{self.source_chunk_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


# ── Validator function ─────────────────────────────────────────────────────────

def validate_analyst_output(
    output: AnalystOutput,
    reader_output: ReaderOutput,
) -> AnalystValidationReport:
    """
    Validate an ``AnalystOutput`` against its source ``ReaderOutput``.

    Parameters
    ----------
    output:
        The ``AnalystOutput`` produced by the Requirements Analyst Agent.
    reader_output:
        The ``ReaderOutput`` that was passed to the agent.

    Returns
    -------
    AnalystValidationReport
        Report with ``passed=True`` if no blocking errors found.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # VAL-A01 — source_chunk_id traceability (TRACE-001)
    if output.source_chunk_id != reader_output.chunk_id:
        errors.append(
            f"VAL-A01 TRACE-001: output.source_chunk_id={output.source_chunk_id!r} "
            f"!= reader_output.chunk_id={reader_output.chunk_id!r}."
        )

    # VAL-A02 — document_id traceability
    if output.document_id != reader_output.document_id:
        errors.append(
            f"VAL-A02 TRACE-001: output.document_id={output.document_id!r} "
            f"!= reader_output.document_id={reader_output.document_id!r}."
        )

    # VAL-A03 — candidate_id uniqueness within this output
    seen_ids: Set[str] = set()
    for c in output.candidates:
        if c.candidate_id in seen_ids:
            errors.append(
                f"VAL-A03: duplicate candidate_id={c.candidate_id!r} found."
            )
        seen_ids.add(c.candidate_id)

    # VAL-A04 — all candidates have matching source_chunk_id
    for c in output.candidates:
        if c.source_chunk_id != reader_output.chunk_id:
            errors.append(
                f"VAL-A04 TRACE-001: candidate {c.candidate_id}.source_chunk_id="
                f"{c.source_chunk_id!r} != chunk_id={reader_output.chunk_id!r}."
            )

    # VAL-A05 — requirement_text must not be in user-story format
    for c in output.candidates:
        if c.requirement_text.strip().lower().startswith("as a"):
            errors.append(
                f"VAL-A05: candidate {c.candidate_id}.requirement_text starts with "
                f"'As a' (user-story format leakage): {c.requirement_text[:60]!r}."
            )

    # VAL-A06 — empty output when ReaderOutput was non-empty
    if output.is_empty and not reader_output.is_empty:
        warnings.append(
            "VAL-A06: AnalystOutput has zero candidates but ReaderOutput was "
            "non-empty. This may indicate the LLM found no actionable requirements "
            "(valid) or failed to extract any (check the output)."
        )

    # VAL-A07 — source_reader_output_ref must match an EvidencedItem content
    all_evidenced_contents = set()
    for cat in ("gameplay_elements", "systems", "characters", "ui_elements", "narrative"):
        for item in getattr(reader_output, cat):
            all_evidenced_contents.add(item.content)

    for c in output.candidates:
        if c.source_reader_output_ref is not None:
            if c.source_reader_output_ref not in all_evidenced_contents:
                warnings.append(
                    f"VAL-A07: candidate {c.candidate_id}.source_reader_output_ref="
                    f"{c.source_reader_output_ref!r} does not match any "
                    f"EvidencedItem.content in the ReaderOutput (traceability gap)."
                )

    passed = len(errors) == 0
    report = AnalystValidationReport(
        source_chunk_id=reader_output.chunk_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "AnalystOutput validation passed for chunk %s (%d candidates)",
                reader_output.chunk_id,
                output.candidate_count,
            )
    else:
        logger.error(report.summary())

    return report
