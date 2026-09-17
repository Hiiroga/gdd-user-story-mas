"""
reader_output.py — Data Contract §4
=====================================
Structured thematic summary produced by the GDD Reader Agent (SRS-008).

Schema source: Data Contracts Specification §4, ReaderOutput.

Design notes:
  * EvidencedItem.source_excerpt is Optional[str] — the agent MUST try to
    provide it, but null is permitted when no verbatim phrase supports the
    item (VAL-016 note: prefer quoting over omitting).
  * All list fields (gameplay_elements, systems, etc.) may be empty []. Zero
    items in a category is valid — the agent must not fabricate content.
  * This schema is the ONLY output of the GDD Reader Agent. It feeds the
    Requirements Analyst Agent (Data Contracts §5). It does NOT contain
    requirements or user stories.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── EvidencedItem ──────────────────────────────────────────────────────────────

class EvidencedItem(BaseModel):
    """
    A single identified element with its supporting GDD evidence.

    Used for every non-theme category in ``ReaderOutput``:
    gameplay_elements, systems, characters, ui_elements, narrative.

    Attributes
    ----------
    content:
        The identified element, in the Reader Agent's own summarized wording.
        Must not be empty.
    source_excerpt:
        Verbatim or near-verbatim phrase from the GDD chunk that supports
        this item. Used for traceability and hallucination auditing (VAL-016).
        Null only when the agent cannot identify a specific supporting phrase.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(
        ...,
        description=(
            "The extracted item in the agent's own summarized wording. "
            "Must not be empty."
        ),
    )
    source_excerpt: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim or near-verbatim phrase from the chunk supporting "
            "this item (VAL-016 traceability). Null only when no specific "
            "phrase can be identified."
        ),
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "EvidencedItem.content must not be empty. "
                "If no content was found, omit the item from the list entirely."
            )
        return v.strip()

    @field_validator("source_excerpt")
    @classmethod
    def source_excerpt_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return v


# ── ReaderOutput ───────────────────────────────────────────────────────────────

class ReaderOutput(BaseModel):
    """
    Structured thematic summary produced by the GDD Reader Agent for one chunk.

    One ``ReaderOutput`` is produced per ``GDDChunk`` (one-to-one).
    This output feeds exactly the corresponding chunk's ``CandidateRequirement``
    generation by the Requirements Analyst Agent (Data Contracts §5).

    Attributes
    ----------
    chunk_id:
        Foreign key to ``GDDChunk.chunk_id`` (TRACE-001).
    document_id:
        Foreign key to ``GDDDocument.document_id``.
    themes:
        High-level thematic keywords extracted from the chunk
        (e.g., "stealth", "resource management"). Plain strings, not evidenced.
    gameplay_elements:
        Mechanics, rules, player interactions, win/lose conditions identified
        in the chunk. Each carries a supporting ``source_excerpt``.
    systems:
        Game systems (e.g., combat system, economy, inventory, progression)
        mentioned or described in the chunk.
    characters:
        Player characters, NPCs, enemies, or other entities described.
    ui_elements:
        Menus, HUD components, screens, prompts, or other interface elements.
    narrative:
        Story beats, world-building, lore, dialogue structures, or
        narrative mechanics described in the chunk.
    """

    model_config = ConfigDict(frozen=True)

    # ── Traceability fields (TRACE-001) ────────────────────────────────────────
    chunk_id: str = Field(
        ...,
        description="Foreign key to GDDChunk.chunk_id (TRACE-001).",
    )
    document_id: str = Field(
        ...,
        description="Foreign key to GDDDocument.document_id.",
    )

    # ── Thematic summary categories ────────────────────────────────────────────
    themes: List[str] = Field(
        default_factory=list,
        description=(
            "High-level thematic keywords from the chunk. "
            "Plain strings — no source_excerpt required for themes."
        ),
    )
    gameplay_elements: List[EvidencedItem] = Field(
        default_factory=list,
        description="Mechanics, rules, and player interactions.",
    )
    systems: List[EvidencedItem] = Field(
        default_factory=list,
        description="Game systems described or mentioned.",
    )
    characters: List[EvidencedItem] = Field(
        default_factory=list,
        description="Player characters, NPCs, enemies, or entities.",
    )
    ui_elements: List[EvidencedItem] = Field(
        default_factory=list,
        description="Menus, HUD elements, screens, or interface components.",
    )
    narrative: List[EvidencedItem] = Field(
        default_factory=list,
        description="Story beats, lore, dialogue structures, or narrative mechanics.",
    )

    # ── Validators ─────────────────────────────────────────────────────────────

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ReaderOutput.chunk_id must not be empty (TRACE-001).")
        return v

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ReaderOutput.document_id must not be empty (TRACE-001).")
        return v

    @field_validator("themes")
    @classmethod
    def themes_no_empty_strings(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        return cleaned

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def total_items(self) -> int:
        """Total number of identified items across all non-theme categories."""
        return (
            len(self.gameplay_elements)
            + len(self.systems)
            + len(self.characters)
            + len(self.ui_elements)
            + len(self.narrative)
        )

    @property
    def is_empty(self) -> bool:
        """True when all categories (including themes) are empty."""
        return not self.themes and self.total_items == 0

    def category_counts(self) -> dict:
        """Return a dict of category → item count for logging."""
        return {
            "themes": len(self.themes),
            "gameplay_elements": len(self.gameplay_elements),
            "systems": len(self.systems),
            "characters": len(self.characters),
            "ui_elements": len(self.ui_elements),
            "narrative": len(self.narrative),
        }
