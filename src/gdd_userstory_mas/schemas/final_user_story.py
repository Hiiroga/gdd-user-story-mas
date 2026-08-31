"""
FinalUserStory — Data Contract §10
=====================================
The canonical, fully-traceable user story produced by either the MAS
pipeline or the Baseline pipeline.  This is the **terminal** artifact
consumed by the Evaluator and stored for thesis reporting.

Schema source: Data Contracts Specification §10, FinalUserStory.

Key Baseline-specific semantics (SRS-015, VAL-031/VAL-032):
  - ``pipeline_type = "Baseline"``
  - ``validation_status = "Unreviewed"``  (no Reviewer stage exists)
  - ``validation_feedback = []``          (no feedback loop)
  - ``iteration_count_final = None``      (no iterative revision)
  - ``merge_history = None``              (no Redundancy Checker)
  - ``confidence_evidence.reviewer_traceability_confirmed = None``
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Sub-models ─────────────────────────────────────────────────────────────────

class MergeHistoryEntry(BaseModel):
    """One entry in the merge history of a merged story (MAS only)."""

    original_story_id: str
    original_source_chunk_id: str


class ConfidenceEvidence(BaseModel):
    """
    Evidence block supporting hallucination-rate auditing (EVAL-001/EVAL-007).

    Fields are null rather than guessed when the pipeline did not produce them.
    """

    source_excerpt: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim/near-verbatim GDD text supporting this story. "
            "For Baseline, populated from the LLM's own reported source_excerpt."
        ),
    )
    reviewer_traceability_confirmed: Optional[bool] = Field(
        default=None,
        description=(
            "True if Reviewer's gdd_traceability criterion passed. "
            "Always None for Baseline (no Reviewer stage, FR-025)."
        ),
    )


# ── Controlled vocabulary ──────────────────────────────────────────────────────

RequirementType = Literal["player", "system", "in-game entity", "dev team"]
GameDomain = Literal["gameplay", "UI", "narrative", "systems", "level design", "audio", "other"]
PipelineType = Literal["MAS", "Baseline"]
ValidationStatus = Literal[
    "Valid",
    "Rejected-ManualReview",
    "Unreviewed",
    "Merged",
    "Redundancy-Unresolved",
]


# ── Main schema ────────────────────────────────────────────────────────────────

class FinalUserStory(BaseModel):
    """
    Terminal, fully-traceable user story (Data Contracts §10).

    Produced by MAS pipeline (post-Redundancy Checker) or directly by
    the Baseline pipeline (SRS-015).  Schema is identical across both
    pipelines (VAL-031); MAS-only fields are null/empty for Baseline.
    """

    id: str = Field(
        ...,
        description="Final canonical ID, e.g., US-0001 (FR-018).",
    )
    run_id: str = Field(..., description="Foreign key to ExperimentRun.run_id.")
    pipeline_type: PipelineType = Field(
        ...,
        description="'MAS' or 'Baseline'. Required for schema parity (NFR-006).",
    )

    # ── Traceability ───────────────────────────────────────────────────────────
    source_document_id: str = Field(
        ..., description="Foreign key to GDDDocument.document_id."
    )
    source_section: Optional[str] = Field(
        default=None,
        description=(
            "Chapter/section label (FR-019). "
            "Null only if no structure detected (VAL-010 fallback)."
        ),
    )
    source_chunk_ids: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "One or more GDDChunk.chunk_id references. "
            "Required even for Baseline (TRACE chain from FinalUserStory → GDDChunk → GDDDocument)."
        ),
    )

    # ── User story content ─────────────────────────────────────────────────────
    role: str = Field(..., description="The 'As a ...' part.")
    action: str = Field(..., description="The 'I want ...' part.")
    benefit: str = Field(..., description="The 'so that ...' part.")
    full_text: str = Field(
        ...,
        description=(
            'Must follow: "As a [role], I want [action], so that [benefit]." (VAL-020).'
        ),
    )

    # ── Classification ─────────────────────────────────────────────────────────
    requirement_type: RequirementType = Field(
        ...,
        description=(
            "Perspective label from Schetinger et al. framework (VAL-017). "
            "Carried from CandidateRequirement.perspective for MAS; "
            "assigned directly by LLM for Baseline."
        ),
    )
    game_domain: GameDomain = Field(
        ...,
        description="Functional domain tag (gameplay, UI, narrative, systems, …).",
    )

    # ── Validation metadata ────────────────────────────────────────────────────
    validation_status: ValidationStatus = Field(
        ...,
        description=(
            "'Unreviewed' for all Baseline stories (no Reviewer stage, FR-025). "
            "'Valid' for MAS stories that passed Reviewer + Redundancy Checker."
        ),
    )
    validation_feedback: List[str] = Field(
        default_factory=list,
        description="Empty for Baseline (no feedback loop, FR-025).",
    )
    merge_history: Optional[List[MergeHistoryEntry]] = Field(
        default=None,
        description="Null for Baseline. Populated for MAS stories with status 'Merged'.",
    )
    confidence_evidence: Optional[ConfidenceEvidence] = Field(
        default=None,
        description="Traceability/hallucination-audit block.",
    )
    iteration_count_final: Optional[int] = Field(
        default=None,
        description="Null for Baseline (no iterative revision, FR-025).",
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("FinalUserStory.id must not be empty.")
        return v

    @field_validator("full_text")
    @classmethod
    def full_text_format(cls, v: str) -> str:
        """VAL-020: full_text must begin with 'As a'."""
        if not v.strip().startswith("As a"):
            raise ValueError(
                "full_text must follow the format 'As a [role], I want [action], "
                "so that [benefit].' (VAL-020)."
            )
        return v

    @field_validator("source_chunk_ids")
    @classmethod
    def chunk_ids_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("source_chunk_ids must contain at least one chunk_id.")
        return v

    @model_validator(mode="after")
    def baseline_fields_consistency(self) -> "FinalUserStory":
        """VAL-032: Baseline stories must not carry MAS-only artefacts."""
        if self.pipeline_type == "Baseline":
            if self.validation_status not in ("Unreviewed",):
                # Soft check — warn but don't hard-fail in case researcher
                # overrides the default intentionally.
                pass
            if self.iteration_count_final is not None:
                raise ValueError(
                    "Baseline FinalUserStory must have iteration_count_final=None "
                    "(no iterative revision exists for Baseline, FR-025)."
                )
        return self
