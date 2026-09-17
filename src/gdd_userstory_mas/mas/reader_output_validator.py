"""
reader_output_validator.py — ReaderOutput post-processing validator
====================================================================
Validates a ``ReaderOutput`` instance against the source ``GDDChunk``
to catch traceability violations and suspicious outputs before the
artifact is passed to the Requirements Analyst Agent.

Validation rules implemented:
  VAL-R01  chunk_id in output matches chunk_id of source chunk (TRACE-001).
  VAL-R02  document_id in output matches document_id of source chunk.
  VAL-R03  All EvidencedItem.content strings are non-empty (redundant with
           schema, but re-checked here for defence-in-depth).
  VAL-R04  If all categories are completely empty, emit a WARNING (suspicious
           but not fatal — a genuinely empty chunk is valid).
  VAL-R05  source_excerpt, if present, must be non-empty string (schema
           already enforces this; re-checked for audit logging).

Note: This validator does NOT reject outputs for having too few items —
zero items per category is valid (the agent must not hallucinate).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput

logger = logging.getLogger(__name__)


# ── Validation Report ──────────────────────────────────────────────────────────

@dataclass
class ReaderValidationReport:
    """
    Result of validating a single ``ReaderOutput``.

    Attributes
    ----------
    chunk_id:
        The chunk ID this report pertains to.
    passed:
        True when no blocking violations were found.
    errors:
        Blocking validation failures (cause the output to be rejected).
    warnings:
        Non-blocking observations (logged but do not reject the output).
    """

    chunk_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"ReaderValidation [{self.chunk_id}]: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN : {w}")
        return "\n".join(parts)


# ── Validator function ─────────────────────────────────────────────────────────

def validate_reader_output(
    output: ReaderOutput,
    chunk: GDDChunk,
) -> ReaderValidationReport:
    """
    Validate a ``ReaderOutput`` against its source ``GDDChunk``.

    Parameters
    ----------
    output:
        The ``ReaderOutput`` produced by the GDD Reader Agent.
    chunk:
        The source ``GDDChunk`` that was passed to the agent.

    Returns
    -------
    ReaderValidationReport
        A report with ``passed=True`` if no blocking errors found.
        Warnings are always logged even when passed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # VAL-R01 — chunk_id traceability (TRACE-001)
    if output.chunk_id != chunk.chunk_id:
        errors.append(
            f"VAL-R01 TRACE-001 violation: output.chunk_id={output.chunk_id!r} "
            f"does not match source chunk.chunk_id={chunk.chunk_id!r}."
        )

    # VAL-R02 — document_id traceability
    if output.document_id != chunk.document_id:
        errors.append(
            f"VAL-R02 TRACE-001 violation: output.document_id={output.document_id!r} "
            f"does not match source chunk.document_id={chunk.document_id!r}."
        )

    # VAL-R03 — EvidencedItem.content non-empty (defence-in-depth)
    for category_name in (
        "gameplay_elements", "systems", "characters", "ui_elements", "narrative"
    ):
        items: List[EvidencedItem] = getattr(output, category_name)
        for idx, item in enumerate(items):
            if not item.content.strip():
                errors.append(
                    f"VAL-R03: {category_name}[{idx}].content is empty — "
                    f"should have been stripped by the agent."
                )

    # VAL-R04 — suspicious all-empty output
    if output.is_empty:
        warnings.append(
            "VAL-R04: All categories (including themes) are empty. "
            "This may indicate the chunk contained no design content, "
            "or that the LLM failed to extract any elements. "
            "Verify the chunk text is meaningful."
        )

    # VAL-R05 — source_excerpt non-empty strings (when present)
    for category_name in (
        "gameplay_elements", "systems", "characters", "ui_elements", "narrative"
    ):
        items = getattr(output, category_name)
        for idx, item in enumerate(items):
            if item.source_excerpt is not None and not item.source_excerpt.strip():
                warnings.append(
                    f"VAL-R05: {category_name}[{idx}].source_excerpt is set "
                    f"but empty — should be null instead."
                )

    passed = len(errors) == 0
    report = ReaderValidationReport(
        chunk_id=chunk.chunk_id,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

    if passed:
        if warnings:
            logger.warning(report.summary())
        else:
            logger.debug(
                "ReaderOutput validation passed for chunk %s (%d items total)",
                chunk.chunk_id,
                output.total_items,
            )
    else:
        logger.error(report.summary())

    return report
