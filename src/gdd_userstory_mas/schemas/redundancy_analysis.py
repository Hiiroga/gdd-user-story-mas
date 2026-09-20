"""
redundancy_analysis.py — Data Contracts §9
=============================================
Redundancy Checker Agent output schemas: RedundancyAnalysis,
DuplicateGroup, and MergeAction.

Schema source: Data Contracts Specification §9 (RedundancyAnalysis).

Design notes:
  * ``DuplicateGroup`` captures a set of semantically equivalent or
    substantially overlapping stories identified during detection.
    Each group must contain ≥2 story IDs (VAL-026).
  * ``MergeAction`` records the resolution decision for a duplicate group:
    merge into a combined story or remove an inferior one.  Every action
    preserves the original story IDs for traceability (TRACE-004).
  * ``RedundancyAnalysis`` is the top-level output containing detection
    results, resolution actions, and the final resolved story list.
  * ``resolution_status`` reflects the loop outcome:
    - ``pending`` — detection done, resolution not yet applied
    - ``resolved`` — all duplicate groups resolved
    - ``unresolved_max_iterations`` — iteration cap reached, some groups
      passed through with a flag (MAS Arch §4 failure handling).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── DuplicateGroup ─────────────────────────────────────────────────────────────

class DuplicateGroup(BaseModel):
    """
    A group of semantically equivalent or substantially overlapping stories.

    Attributes
    ----------
    group_id:
        Unique identifier for this duplicate group.
    story_ids:
        Draft IDs of the stories in this group.  Must contain ≥2 (VAL-026).
    similarity_basis:
        Method used: e.g., ``llm_judgment``, ``embedding_score``, ``hybrid``.
    similarity_score:
        Quantitative similarity score if available; null otherwise.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str = Field(
        ...,
        description="Unique duplicate group identifier.",
    )
    story_ids: List[str] = Field(
        ...,
        min_length=2,
        description="Draft IDs in this group. Must be ≥2 (VAL-026).",
    )
    similarity_basis: str = Field(
        ...,
        description="Detection method: llm_judgment, embedding_score, hybrid.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Quantitative score if available.",
    )

    @field_validator("group_id")
    @classmethod
    def group_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("group_id must not be empty.")
        return v

    @field_validator("story_ids")
    @classmethod
    def story_ids_not_empty(cls, v: List[str]) -> List[str]:
        for sid in v:
            if not sid.strip():
                raise ValueError("story_ids must not contain empty strings.")
        return v


# ── MergeAction ────────────────────────────────────────────────────────────────

class MergeAction(BaseModel):
    """
    Resolution decision for a duplicate group.

    Records whether stories were merged into one or an inferior story was
    removed.  Every action preserves the original story IDs and source
    chunk IDs for traceability (TRACE-004).

    Attributes
    ----------
    group_id:
        FK to ``DuplicateGroup.group_id``.
    action:
        ``merge`` — combine into a single comprehensive story.
        ``remove_inferior`` — keep the best, discard the rest.
    kept_story_id:
        The draft_id of the story that survives (or the merged result ID).
    removed_story_ids:
        Draft IDs of stories removed/subsumed.
    merged_role, merged_action, merged_benefit, merged_full_text:
        The resulting story fields after merge.  Populated only when
        ``action = "merge"``.  For ``remove_inferior``, these are null
        (the kept story's fields are unchanged).
    source_chunk_ids:
        All source chunk IDs from the merged/kept stories (TRACE-004).
    reason:
        Explanation of why these stories are duplicates and the resolution
        rationale.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str = Field(
        ...,
        description="FK to DuplicateGroup.group_id.",
    )
    action: Literal["merge", "remove_inferior"] = Field(
        ...,
        description="Resolution type.",
    )
    kept_story_id: str = Field(
        ...,
        description="Draft ID of the surviving/merged story.",
    )
    removed_story_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Draft IDs of removed/subsumed stories.",
    )
    merged_role: Optional[str] = Field(
        default=None,
        description="Merged role field (only for action=merge).",
    )
    merged_action: Optional[str] = Field(
        default=None,
        description="Merged action field (only for action=merge).",
    )
    merged_benefit: Optional[str] = Field(
        default=None,
        description="Merged benefit field (only for action=merge).",
    )
    merged_full_text: Optional[str] = Field(
        default=None,
        description="Merged full_text (only for action=merge).",
    )
    source_chunk_ids: List[str] = Field(
        ...,
        min_length=1,
        description="All source chunk IDs from merged/kept stories (TRACE-004).",
    )
    reason: str = Field(
        ...,
        description="Explanation of the redundancy and resolution rationale.",
    )

    @field_validator("group_id", "kept_story_id", "reason")
    @classmethod
    def string_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v

    @model_validator(mode="after")
    def merge_fields_consistency(self) -> "MergeAction":
        """If action=merge, merged story fields must be populated."""
        if self.action == "merge":
            for field_name in ("merged_role", "merged_action",
                               "merged_benefit", "merged_full_text"):
                val = getattr(self, field_name)
                if not val or not val.strip():
                    raise ValueError(
                        f"action='merge' requires {field_name} to be "
                        f"non-empty, but got {val!r}."
                    )
        return self


# ── RedundancyAnalysis ─────────────────────────────────────────────────────────

class RedundancyAnalysis(BaseModel):
    """
    Top-level output of the Redundancy Checker Agent (Data Contracts §9).

    Contains detection results (duplicate groups), resolution actions
    (merge/remove decisions), and the final resolved story list.

    Attributes
    ----------
    analysis_id:
        Unique identifier.  Format: ``redundancy__{run_id}__pass{N}``.
    run_id:
        FK to ``ExperimentResult.run_id``.
    analyzed_at_pass:
        Which detection pass this is (1, 2, etc.).
    duplicate_groups:
        Groups of semantically duplicate stories detected.
    unique_story_ids:
        Draft IDs of stories NOT in any duplicate group.
    resolution_status:
        ``pending``, ``resolved``, or ``unresolved_max_iterations``.
    merge_actions:
        Resolution decisions applied to duplicate groups.
    resolved_story_count:
        Number of stories in the final cleaned collection.
    input_story_count:
        Number of stories received as input.
    """

    model_config = ConfigDict(frozen=True)

    analysis_id: str = Field(
        ...,
        description="Unique ID. Format: redundancy__{run_id}__pass{N}.",
    )
    run_id: str = Field(
        ...,
        description="FK to ExperimentResult.run_id.",
    )
    analyzed_at_pass: int = Field(
        ...,
        ge=1,
        description="Detection pass number (1-indexed).",
    )
    duplicate_groups: List[DuplicateGroup] = Field(
        default_factory=list,
        description="Groups of semantically duplicate stories.",
    )
    unique_story_ids: List[str] = Field(
        default_factory=list,
        description="Draft IDs not in any duplicate group.",
    )
    resolution_status: Literal["pending", "resolved", "unresolved_max_iterations"] = Field(
        ...,
        description="Outcome of the resolution loop.",
    )
    merge_actions: List[MergeAction] = Field(
        default_factory=list,
        description="Resolution decisions for duplicate groups.",
    )
    resolved_story_count: int = Field(
        ...,
        ge=0,
        description="Final number of stories after resolution.",
    )
    input_story_count: int = Field(
        ...,
        ge=0,
        description="Number of stories received as input.",
    )

    @field_validator("analysis_id", "run_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v
