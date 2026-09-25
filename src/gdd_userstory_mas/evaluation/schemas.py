"""
evaluation/schemas.py — Pydantic Schemas for Comparative Evaluation
====================================================================
Defines the machine-readable output schemas for the research evaluation
pipeline, designed for later statistical analysis.

Key schemas:
  MetricResult       — One metric's computed value + raw observations.
  PipelineEvaluation — All 5 metrics for one pipeline run.
  ComparisonResult   — Side-by-side comparison of MAS vs. Baseline.
  AnnotationTaskSpec — Declares human annotation tasks required for
                       ground-truth metrics (explicitly separated).

All schemas serialize to JSON-serializable types for storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── MetricResult ───────────────────────────────────────────────────────────────

class RawObservation(BaseModel):
    """One per-story (or per-pair) raw measurement for a metric."""

    model_config = ConfigDict(frozen=True)

    story_id: str = Field(..., description="Story ID or pair identifier.")
    value: Any = Field(
        ...,
        description=(
            "The raw observed value for this story. "
            "Type depends on the metric: bool (pass/fail), float (score), "
            "list (chunk ids), dict (category labels), None (unverifiable)."
        ),
    )
    detail: str = Field(
        default="",
        description="Human-readable explanation of this observation.",
    )


class MetricResult(BaseModel):
    """
    Complete result for one metric from one pipeline run.

    Designed to be fully self-describing for statistical analysis:
    the formula_version field identifies exactly which computation
    was used, enabling reproducibility verification.
    """

    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(
        ...,
        description="Metric identifier (e.g., 'aqusa_score', 'coverage').",
    )
    formula_version: str = Field(
        ...,
        description=(
            "Identifies the exact computation formula/version used. "
            "E.g., 'aqusa_v1_syntactic_5criteria'. "
            "Used to detect formula changes across runs."
        ),
    )
    input_count: int = Field(
        ...,
        ge=0,
        description="Number of inputs (stories or chunks) that went into computation.",
    )
    value: Optional[float] = Field(
        default=None,
        description="Computed scalar value. None if not computable (see null_reason).",
    )
    numerator: Optional[float] = Field(
        default=None,
        description="Numerator of the ratio formula, for transparency.",
    )
    denominator: Optional[float] = Field(
        default=None,
        description="Denominator of the ratio formula, for transparency.",
    )
    null_reason: Optional[str] = Field(
        default=None,
        description=(
            "Required explanation when value is None (VAL-029). "
            "Documents WHY the metric could not be computed."
        ),
    )
    interpretation: str = Field(
        default="",
        description="Human-readable interpretation of the computed value.",
    )
    limitation: str = Field(
        default="",
        description="Known limitations of this metric for this run.",
    )
    annotation_required: bool = Field(
        default=False,
        description=(
            "True if this metric value is an automatic APPROXIMATION that "
            "requires human annotation for ground-truth validation."
        ),
    )
    annotation_task: Optional[str] = Field(
        default=None,
        description=(
            "Key into ComparisonResult.annotation_tasks if annotation_required=True."
        ),
    )
    raw_observations: List[RawObservation] = Field(
        default_factory=list,
        description=(
            "Per-story (or per-pair) raw observations. "
            "Enables post-hoc statistical analysis and audit."
        ),
    )


# ── AnnotationTaskSpec ─────────────────────────────────────────────────────────

class AnnotationTaskSpec(BaseModel):
    """
    Declares a human annotation task required to obtain ground-truth values
    for metrics that cannot be computed automatically.

    This EXPLICITLY SEPARATES the automated evaluation from human judgment,
    as required by the research design.
    """

    model_config = ConfigDict(frozen=True)

    task_key: str = Field(
        ...,
        description="Unique identifier for this annotation task.",
    )
    metric_name: str = Field(
        ...,
        description="Which metric this annotation provides ground truth for.",
    )
    reason: str = Field(
        ...,
        description="Why automatic computation is insufficient.",
    )
    instructions: str = Field(
        ...,
        description="Step-by-step instructions for human annotators.",
    )
    annotation_schema: Dict[str, str] = Field(
        ...,
        description=(
            "JSON-serializable schema describing the fields that annotators "
            "must record for each item."
        ),
    )
    status: Literal["pending", "in_progress", "complete"] = Field(
        default="pending",
        description="Annotation completion status.",
    )


# ── PipelineEvaluation ─────────────────────────────────────────────────────────

class PipelineEvaluation(BaseModel):
    """
    All 5 thesis metrics computed for one pipeline run (MAS or Baseline).

    Designed as a self-contained record for a single experimental arm.
    """

    model_config = ConfigDict(frozen=True)

    pipeline_type: Literal["MAS", "Baseline"] = Field(
        ...,
        description="'MAS' or 'Baseline'.",
    )
    run_id: str = Field(
        ...,
        description="Experiment run identifier from the pipeline output.",
    )
    document_id: str = Field(
        ...,
        description="Source GDD document identifier.",
    )
    total_stories: int = Field(
        ...,
        ge=0,
        description="Total number of final stories in the evaluated collection.",
    )
    total_chunks: int = Field(
        ...,
        ge=0,
        description="Total number of GDD chunks produced by preprocessing.",
    )

    # ── Metric results ─────────────────────────────────────────────────────────

    hallucination_rate: MetricResult = Field(
        ...,
        description="EVAL-001: Hallucination rate result.",
    )
    redundancy_rate: MetricResult = Field(
        ...,
        description="EVAL-002: Redundancy rate result.",
    )
    aqusa_score: MetricResult = Field(
        ...,
        description="EVAL-003: AQUSA syntactic quality score result.",
    )
    coverage: MetricResult = Field(
        ...,
        description="EVAL-004: GDD chunk coverage result.",
    )
    diversity: MetricResult = Field(
        ...,
        description="EVAL-005: Category diversity result.",
    )

    # ── Provenance ─────────────────────────────────────────────────────────────

    model_config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model configuration used for this pipeline run.",
    )
    prompt_config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prompt configuration used for this pipeline run.",
    )
    pipeline_output_ref: str = Field(
        default="",
        description="Path to the pipeline output directory.",
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this evaluation was computed.",
    )


# ── ComparisonResult ───────────────────────────────────────────────────────────

class ComparisonResult(BaseModel):
    """
    Machine-readable side-by-side comparison of MAS vs. Baseline.

    This is the top-level output of the research evaluation pipeline.
    Suitable for:
      - Statistical analysis (scipy, pandas)
      - Thesis reporting
      - Replication by other researchers

    All metric values include raw_observations for full auditability.
    Human annotation tasks are explicitly declared and separated.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(
        ...,
        description=(
            "Unique identifier for this comparison. "
            "Format: 'compare__{mas_run_id}__{baseline_run_id}'."
        ),
    )
    document_id: str = Field(
        ...,
        description="Source GDD document. Both pipelines must use the same document.",
    )
    gdd_filename: str = Field(
        default="",
        description="Filename of the source GDD (for reporting).",
    )

    # ── Pipeline results ───────────────────────────────────────────────────────

    mas: PipelineEvaluation = Field(
        ...,
        description="MAS pipeline evaluation.",
    )
    baseline: PipelineEvaluation = Field(
        ...,
        description="Baseline pipeline evaluation.",
    )

    # ── Experimental conditions ────────────────────────────────────────────────

    controlled_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Documents that both pipelines used identical experimental conditions. "
            "Keys: 'same_document', 'same_model', 'same_chunk_count', etc."
        ),
    )

    # ── Human annotation tasks (explicitly separated) ──────────────────────────

    annotation_tasks: List[AnnotationTaskSpec] = Field(
        default_factory=list,
        description=(
            "Human annotation tasks required for ground-truth validation. "
            "EXPLICITLY SEPARATED from automatic evaluation. "
            "These tasks are declared but NOT executed automatically."
        ),
    )

    # ── Metadata ───────────────────────────────────────────────────────────────

    pipeline_version: str = Field(
        default="gdd-userstory-mas-0.1.0",
        description="Pipeline software version.",
    )
    evaluation_pipeline_version: str = Field(
        default="eval-pipeline-1.0.0",
        description="Version of the evaluation pipeline that produced this result.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this comparison was created.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Free-text notes about this comparison run.",
    )
