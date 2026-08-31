"""
ErrorFailureLog — Data Contract §13
=======================================
Captures every failure condition defined in the MAS Detailed Architecture §4
and SRS error-handling requirements.

Schema source: Data Contracts Specification §13, ErrorFailureLog.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Allowed stage identifiers ──────────────────────────────────────────────────
StageType = Literal[
    "ingestion",
    "pdf_conversion",
    "text_cleaning",
    "segmentation",
    "chunking",
    "metadata_tagging",
    "gdd_reader",
    "requirements_analyst",
    "user_story_generator",
    "reviewer",
    "redundancy_checker",
    "evaluator",
    "single_agent_baseline",
    "orchestration",
    "result_storage",
]

# ── Allowed error types ────────────────────────────────────────────────────────
ErrorType = Literal[
    "malformed_output",
    "schema_violation",
    "llm_api_failure",
    "llm_timeout",
    "pdf_conversion_failure",
    "max_iterations_reached_reviewer",
    "max_iterations_reached_redundancy",
    "missing_source_content",
    "reviewer_contradiction",
    "other",
]

# ── Allowed resolution strategies ─────────────────────────────────────────────
ResolutionType = Literal[
    "resolved_via_retry",
    "excluded_and_logged",
    "run_halted",
    "flagged_unresolved",
]


class ErrorFailureLog(BaseModel):
    """
    Structured log entry for any failure event in the pipeline.

    Design principle: "fail loudly, never silently degrade"
    (Technical Architecture §10).
    """

    error_id: str = Field(..., description="Unique error identifier (UUID4).")
    run_id: Optional[str] = Field(
        default=None,
        description=(
            "Foreign key to ExperimentRun.run_id, if the error occurred "
            "within a run context."
        ),
    )
    document_id: Optional[str] = Field(default=None)
    stage: StageType = Field(
        ...,
        description=(
            "Pipeline stage where the error occurred. "
            "Maps to SRS sections 1–15 / MAS agent stages."
        ),
    )
    error_type: ErrorType = Field(..., description="Enumerated error classification.")
    related_entity_id: Optional[str] = Field(
        default=None,
        description=(
            "e.g., chunk_id, candidate_id, draft_id — "
            "whichever artifact the error pertains to."
        ),
    )
    message: str = Field(..., description="Human-readable error description.")
    retry_count: Optional[int] = Field(
        default=None, ge=0, description="Number of retries attempted before logging."
    )
    resolution: ResolutionType = Field(
        ...,
        description=(
            "How the error was handled. Consistent with the "
            "'fail loudly, never silently degrade' principle."
        ),
    )
    timestamp: datetime = Field(
        ..., description="ISO 8601 datetime when the error was recorded."
    )

    model_config = ConfigDict()
