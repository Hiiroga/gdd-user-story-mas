"""
generated_user_story.py — Data Contract §6
============================================
Draft user story produced by the User Story Generator Agent (SRS-010).

Schema source: Data Contracts Specification §6, GeneratedUserStory.

Design notes:
  * ``draft_id`` is STABLE across all revisions of the same underlying
    candidate (TRACE-003). Format: ``{candidate_id}__draft``.
    Using candidate_id as the base makes the link explicit and deterministic.
  * ``full_text`` MUST exactly match
    "As a [role], I want [action], so that [benefit]." (VAL-020).
    The validator reconstructs the string from role/action/benefit and
    compares it to full_text to catch any mismatch.
  * ``iteration_count`` starts at 0 for a fresh first draft and increments
    by exactly 1 per revision pass (VAL-022).
  * ``previous_draft_id_chain`` is empty for a fresh draft; the Reviewer
    feedback loop appends entries as revisions occur (OPEN DECISION on
    storage strategy — we use an ordered list of prior draft_ids).
  * This is a PRE-VALIDATION state; see FinalUserStory (§10) for the
    post-pipeline canonical representation.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── GeneratedUserStory ─────────────────────────────────────────────────────────

class GeneratedUserStory(BaseModel):
    """
    A draft user story produced or revised by the User Story Generator Agent.

    One ``CandidateRequirement`` produces exactly one ``GeneratedUserStory``
    identified by a stable ``draft_id``.  The draft may be revised multiple
    times via the Reviewer feedback loop; each revision increments
    ``iteration_count`` by 1.

    Attributes
    ----------
    draft_id:
        Stable identifier across all revisions of the same candidate.
        Format: ``{candidate_id}__draft`` (TRACE-003).
    candidate_id:
        Foreign key to ``CandidateRequirement.candidate_id``.
    source_chunk_id:
        Inherited from the candidate; foreign key to ``GDDChunk.chunk_id``
        (TRACE-001).
    document_id:
        Foreign key to ``GDDDocument.document_id``.
    role:
        The stakeholder role (e.g., "player", "game designer").  Must not be
        empty.
    action:
        The desired capability or action.  Must not be empty.
    benefit:
        The rationale or outcome achieved.  Must not be empty.
    full_text:
        Complete user story string.  Must follow exactly:
        ``"As a [role], I want [action], so that [benefit]."`` (VAL-020).
    iteration_count:
        0 for a fresh first draft; increments by 1 per revision (VAL-022).
    previous_draft_id_chain:
        Ordered list of prior ``draft_id`` states for audit trail.  Empty
        for a fresh first draft.
    """

    model_config = ConfigDict(frozen=True)

    # ── Traceability fields ────────────────────────────────────────────────────
    draft_id: str = Field(
        ...,
        description="Stable ID across revisions. Format: {candidate_id}__draft (TRACE-003).",
    )
    candidate_id: str = Field(
        ...,
        description="FK to CandidateRequirement.candidate_id.",
    )
    source_chunk_id: str = Field(
        ...,
        description="Inherited from candidate; FK to GDDChunk.chunk_id (TRACE-001).",
    )
    document_id: str = Field(
        ...,
        description="FK to GDDDocument.document_id.",
    )

    # ── Story fields ───────────────────────────────────────────────────────────
    role: str = Field(
        ...,
        description="Stakeholder role (e.g., 'player', 'game designer'). Must not be empty.",
    )
    action: str = Field(
        ...,
        description="The desired capability or behaviour. Must not be empty.",
    )
    benefit: str = Field(
        ...,
        description="The rationale or benefit achieved. Must not be empty.",
    )
    full_text: str = Field(
        ...,
        description=(
            'Complete user story. Must be exactly: '
            '"As a [role], I want [action], so that [benefit]." (VAL-020).'
        ),
    )

    # ── Iteration tracking ─────────────────────────────────────────────────────
    iteration_count: int = Field(
        default=0,
        ge=0,
        description="0 for fresh draft; increments by 1 per revision (VAL-022).",
    )
    previous_draft_id_chain: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of prior draft_id states. Empty for first draft. "
            "Populated by Reviewer feedback loop."
        ),
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("draft_id", "candidate_id", "source_chunk_id", "document_id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(f"Field must not be empty.")
        return v

    @field_validator("role", "action", "benefit")
    @classmethod
    def story_field_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("role, action, and benefit must not be empty.")
        return stripped

    @field_validator("full_text")
    @classmethod
    def full_text_format(cls, v: str) -> str:
        """Enforce 'As a ..., I want ..., so that ....' format (VAL-020)."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("full_text must not be empty.")
        pattern = re.compile(
            r"^As a .+,\s+I want .+,\s+so that .+\.$",
            re.IGNORECASE,
        )
        if not pattern.match(stripped):
            raise ValueError(
                f"full_text must follow the format "
                f'"As a [role], I want [action], so that [benefit]." (VAL-020). '
                f"Got: {stripped[:120]!r}"
            )
        return stripped

    @model_validator(mode="after")
    def full_text_matches_components(self) -> "GeneratedUserStory":
        """
        Verify that full_text is consistent with role, action, benefit (VAL-020).

        Constructs the canonical form and checks it against full_text
        (case-insensitive, whitespace-normalised).
        """
        expected = f"As a {self.role}, I want {self.action}, so that {self.benefit}."
        # Normalise whitespace for comparison
        def normalise(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        if normalise(self.full_text) != normalise(expected):
            raise ValueError(
                f"full_text does not match role/action/benefit components (VAL-020).\n"
                f"  Expected (from components): {expected!r}\n"
                f"  Got full_text            : {self.full_text!r}"
            )
        return self

    # ── Helpers ────────────────────────────────────────────────────────────────

    @classmethod
    def build_full_text(cls, role: str, action: str, benefit: str) -> str:
        """Construct the canonical full_text string from components."""
        return f"As a {role.strip()}, I want {action.strip()}, so that {benefit.strip()}."

    def is_first_draft(self) -> bool:
        """True when this is the initial generation (not a revision)."""
        return self.iteration_count == 0
