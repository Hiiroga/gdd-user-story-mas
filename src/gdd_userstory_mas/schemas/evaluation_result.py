"""
evaluation_result.py — Data Contracts §11
==========================================
EvaluationResult schema: structured output of the Evaluator Agent
for one pipeline run (SRS-014).

Schema source: Data Contracts Specification §11 (EvaluationResult).

Metric definitions (per thesis / MAS Architecture EVAL-001–EVAL-005):

  EVAL-001  hallucination_rate — % of stories with unsupported claims
  EVAL-002  redundancy_rate    — % of stories flagged as redundant pre-dedup
  EVAL-003  aqusa_score        — % of stories meeting AQUSA syntactic criteria
  EVAL-004  coverage           — % of GDD chunks covered by ≥1 story
  EVAL-005  diversity          — ratio of unique role/domain categories seen

Design notes:
  * All five metric fields are nullable with a documented reason (VAL-029).
  * ``MetricDetail`` carries per-metric numerator/denominator for transparency.
  * ``consistency_notes`` discloses excluded/flagged stories per the
    transparency principle (MAS Arch §4).
  * ``computed_at`` is set by the Evaluator at computation time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── MetricDetail ───────────────────────────────────────────────────────────────

class MetricDetail(BaseModel):
    """
    Per-metric computation breakdown for transparency and reproducibility.

    Attributes
    ----------
    metric_name:
        The metric identifier (e.g., ``"aqusa_score"``).
    formula:
        Human-readable formula used.
    numerator:
        Numerator value used in the formula (null if metric is null).
    denominator:
        Denominator value (null if metric is null or not applicable).
    value:
        The computed metric value (mirrors the parent field).
    interpretation:
        Brief interpretation of the value.
    limitation:
        Documented limitations of this metric for this run.
    """

    model_config = ConfigDict(frozen=True)

    metric_name: str
    formula: str
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    value: Optional[float] = None
    interpretation: str = ""
    limitation: str = ""


# ── AQUSADetail ───────────────────────────────────────────────────────────────

class AQUSADetail(BaseModel):
    """
    Per-criterion breakdown of AQUSA score computation (EVAL-003).

    Each criterion is a count of stories that failed it.
    """

    model_config = ConfigDict(frozen=True)

    total_stories: int
    role_empty_count: int = 0
    action_empty_count: int = 0
    benefit_empty_count: int = 0
    format_violated_count: int = 0
    ambiguity_heuristic_count: int = 0
    stories_passing_all: int = 0


# ── EvaluationResult ──────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """
    Final evaluation report for one pipeline run (Data Contracts §11).

    One ``EvaluationResult`` per ``run_id``. Consumed by the
    Evaluation & Comparison Module to produce MAS-vs-Baseline comparisons.

    Attributes
    ----------
    evaluation_id:
        Unique identifier. Format: ``eval__{run_id}``.
    run_id:
        FK to ``ExperimentRun.run_id``.
    document_id:
        FK to ``GDDDocument.document_id``.
    pipeline_type:
        ``"MAS"`` or ``"Baseline"``.
    total_stories:
        Total number of final stories in the evaluated collection.
    hallucination_rate:
        EVAL-001. % of stories with unsupported claims. Lower is better.
    redundancy_rate:
        EVAL-002. % of stories flagged redundant pre-dedup. Lower is better.
    aqusa_score:
        EVAL-003. % of stories meeting all AQUSA syntactic criteria.
        Higher is better.
    coverage:
        EVAL-004. % of GDD chunks covered by ≥1 story. Higher is better.
    diversity:
        EVAL-005. Ratio of unique role/domain categories seen (0.0–1.0).
        Higher is better.
    null_reasons:
        Required explanation for any metric left null (VAL-029).
    consistency_notes:
        Transparency disclosures for excluded/flagged stories.
    computed_at:
        ISO-8601 UTC timestamp of computation.
    metric_details:
        Per-metric breakdown for auditability.
    aqusa_detail:
        Per-criterion AQUSA breakdown.
    """

    model_config = ConfigDict(frozen=True)

    evaluation_id: str = Field(
        ...,
        description="Unique ID. Format: eval__{run_id}.",
    )
    run_id: str = Field(
        ...,
        description="FK to ExperimentRun.run_id.",
    )
    document_id: str = Field(
        ...,
        description="FK to GDDDocument.document_id.",
    )
    pipeline_type: Literal["MAS", "Baseline"] = Field(
        ...,
        description="'MAS' or 'Baseline'.",
    )
    total_stories: int = Field(
        ...,
        ge=0,
        description="Total stories in the evaluated collection.",
    )

    # ── Thesis metrics (EVAL-001 to EVAL-005) ─────────────────────────────────

    hallucination_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="EVAL-001. % stories with unsupported claims (0–100). Null with reason if unavailable.",
    )
    redundancy_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="EVAL-002. % stories flagged redundant pre-dedup (0–100).",
    )
    aqusa_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="EVAL-003. % stories passing all AQUSA syntactic criteria (0–100).",
    )
    coverage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="EVAL-004. % GDD chunks covered by ≥1 story (0–100).",
    )
    diversity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="EVAL-005. Ratio of unique role/domain categories seen (0.0–1.0).",
    )

    # ── Transparency fields ────────────────────────────────────────────────────

    null_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Required explanation for any metric left null (VAL-029).",
    )
    consistency_notes: List[str] = Field(
        default_factory=list,
        description="Transparency disclosures for excluded/flagged stories.",
    )
    computed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of computation.",
    )

    # ── Audit fields ───────────────────────────────────────────────────────────

    metric_details: Dict[str, MetricDetail] = Field(
        default_factory=dict,
        description="Per-metric computation breakdown keyed by metric name.",
    )
    aqusa_detail: Optional[AQUSADetail] = Field(
        default=None,
        description="Per-criterion AQUSA breakdown for transparency.",
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("evaluation_id", "run_id", "document_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v

    @model_validator(mode="after")
    def null_metrics_have_reasons(self) -> "EvaluationResult":
        """VAL-029: Every null metric must have an entry in null_reasons."""
        metric_fields = [
            "hallucination_rate",
            "redundancy_rate",
            "aqusa_score",
            "coverage",
            "diversity",
        ]
        missing_reasons = []
        for field_name in metric_fields:
            if getattr(self, field_name) is None:
                if field_name not in self.null_reasons:
                    missing_reasons.append(field_name)
        if missing_reasons:
            raise ValueError(
                f"VAL-029: Null metric(s) {missing_reasons} must have an "
                f"entry in null_reasons explaining why they could not be computed."
            )
        return self
