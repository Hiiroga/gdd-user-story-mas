"""
candidate_requirement.py — Data Contract §5
=============================================
Output of the Requirements Analyst Agent (SRS-009).

Schema source: Data Contracts Specification §5, CandidateRequirement.

Design notes:
  * ``perspective`` is exactly one of four fixed values per Schetinger et al.
    framework (VAL-017): player | system | in-game entity | dev team.
  * ``domain`` uses a controlled vocabulary resolved from OPEN DECISION in the
    data contracts. Vocabulary is fixed here based on the examples given in §5
    and the game domain taxonomy used throughout the thesis.
  * ``source_reader_output_ref`` is Optional — it strengthens traceability by
    pointing to the specific EvidencedItem this candidate derived from, but is
    not required because zero-evidence items can still yield requirements.
  * ``candidate_id`` is deterministic: ``{document_id}__cand_{sequence:04d}``
    so that re-runs over identical input produce the same IDs.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Controlled vocabularies ────────────────────────────────────────────────────

# Perspective enum — from Schetinger et al. framework (VAL-017, Data Contracts §5)
PerspectiveType = Literal["player", "system", "in-game entity", "dev team"]

# Domain vocabulary — resolved from OPEN DECISION in §5.
# Based on the illustrative examples in §5 and §10 (game_domain field).
VALID_DOMAINS = frozenset({
    "gameplay",
    "ui",
    "narrative",
    "systems",
    "level design",
    "audio",
    "visuals",
    "accessibility",
    "meta",
    "other",
})

DomainType = Literal[
    "gameplay",
    "ui",
    "narrative",
    "systems",
    "level design",
    "audio",
    "visuals",
    "accessibility",
    "meta",
    "other",
]


# ── CandidateRequirement ───────────────────────────────────────────────────────

class CandidateRequirement(BaseModel):
    """
    A single candidate functional requirement extracted by the Requirements
    Analyst Agent from one ``ReaderOutput``.

    Many ``CandidateRequirement`` records may derive from one ``ReaderOutput``
    (one-to-many). Zero candidates is a valid outcome when the Reader Output
    contains no actionable requirements.

    Each candidate feeds exactly one initial ``GeneratedUserStory`` draft
    (Data Contracts §6).

    Attributes
    ----------
    candidate_id:
        Unique identifier within a document run.
        Format: ``{document_id}__cand_{sequence:04d}`` (deterministic).
    source_chunk_id:
        Foreign key to ``GDDChunk.chunk_id`` (TRACE-001).
    document_id:
        Foreign key to ``GDDDocument.document_id``.
    requirement_text:
        The reformulated requirement statement — a clear, atomic functional
        requirement. NOT yet in user-story format ("As a...").
    perspective:
        Exactly one perspective label (VAL-017) per Schetinger et al.:
        ``player`` | ``system`` | ``in-game entity`` | ``dev team``.
    domain:
        Functional domain tag from the controlled vocabulary.
    source_reader_output_ref:
        Optional reference to the ``EvidencedItem.content`` string this
        candidate was derived from, for additional traceability.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(
        ...,
        description=(
            "Unique identifier within the document run. "
            "Format: {document_id}__cand_{sequence:04d}."
        ),
    )
    source_chunk_id: str = Field(
        ...,
        description="Foreign key to GDDChunk.chunk_id (TRACE-001).",
    )
    document_id: str = Field(
        ...,
        description="Foreign key to GDDDocument.document_id.",
    )
    requirement_text: str = Field(
        ...,
        description=(
            "Reformulated requirement statement. Not in user-story format. "
            "Must be atomic, clear, and directly supported by the GDD."
        ),
    )
    perspective: PerspectiveType = Field(
        ...,
        description=(
            "Exactly one perspective per Schetinger et al. framework (VAL-017): "
            "player | system | in-game entity | dev team."
        ),
    )
    domain: DomainType = Field(
        ...,
        description=(
            "Functional domain tag from the controlled vocabulary: "
            "gameplay | ui | narrative | systems | level design | "
            "audio | visuals | accessibility | meta | other."
        ),
    )
    source_reader_output_ref: Optional[str] = Field(
        default=None,
        description=(
            "Optional: the EvidencedItem.content string this candidate was "
            "derived from. Strengthens traceability (Data Contracts §5)."
        ),
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CandidateRequirement.candidate_id must not be empty.")
        return v

    @field_validator("source_chunk_id")
    @classmethod
    def source_chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "CandidateRequirement.source_chunk_id must not be empty (TRACE-001)."
            )
        return v

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CandidateRequirement.document_id must not be empty.")
        return v

    @field_validator("requirement_text")
    @classmethod
    def requirement_text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("CandidateRequirement.requirement_text must not be empty.")
        if stripped.lower().startswith("as a"):
            raise ValueError(
                "CandidateRequirement.requirement_text must NOT be in user-story format "
                "(must not start with 'As a'). That is the Generator Agent's job."
            )
        return stripped

    @field_validator("source_reader_output_ref")
    @classmethod
    def ref_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return v


# ── AnalystOutput ──────────────────────────────────────────────────────────────

class AnalystOutput(BaseModel):
    """
    Full output of the Requirements Analyst Agent for one ``ReaderOutput``.

    Wraps the list of ``CandidateRequirement`` records with traceability
    metadata so the Orchestrator can log and route the output (IF-5).

    Attributes
    ----------
    source_chunk_id:
        The chunk_id of the ``ReaderOutput`` that was analysed (TRACE-001).
    document_id:
        The document_id of the source chunk.
    candidates:
        Zero or more ``CandidateRequirement`` records. An empty list is a
        valid outcome (no actionable requirements found in this chunk).
    """

    model_config = ConfigDict(frozen=True)

    source_chunk_id: str = Field(
        ...,
        description="chunk_id of the ReaderOutput that was analysed (TRACE-001).",
    )
    document_id: str = Field(
        ...,
        description="document_id of the source.",
    )
    candidates: List[CandidateRequirement] = Field(
        default_factory=list,
        description=(
            "Candidate requirements extracted from this chunk. "
            "Empty list is valid (no actionable requirements found)."
        ),
    )

    @field_validator("source_chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("AnalystOutput.source_chunk_id must not be empty.")
        return v

    @field_validator("document_id")
    @classmethod
    def doc_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("AnalystOutput.document_id must not be empty.")
        return v

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def is_empty(self) -> bool:
        return len(self.candidates) == 0

    def perspective_counts(self) -> dict:
        counts: dict = {
            "player": 0, "system": 0,
            "in-game entity": 0, "dev team": 0,
        }
        for c in self.candidates:
            counts[c.perspective] += 1
        return counts

    def domain_counts(self) -> dict:
        counts: dict = {}
        for c in self.candidates:
            counts[c.domain] = counts.get(c.domain, 0) + 1
        return counts
