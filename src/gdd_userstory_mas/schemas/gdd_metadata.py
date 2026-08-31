"""
GDDMetadata — Data Contract §2
================================
Document-level structural metadata derived during segmentation, distinct
from per-chunk metadata.

Schema source: Data Contracts Specification §2, GDDMetadata.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TOCEntry(BaseModel):
    """A single entry in the detected table of contents."""

    title: str = Field(..., description="Section/chapter heading text.")
    level: Optional[int] = Field(
        default=None,
        ge=1,
        description="Heading depth if detectable (1 = top-level chapter).",
    )
    position: int = Field(
        ...,
        ge=0,
        description="Sequential order index within the TOC.",
    )

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("TOCEntry.title must not be empty.")
        return v


class GDDMetadata(BaseModel):
    """
    Document-level structural metadata.

    One-to-one with ``GDDDocument``.  Provides the structural map used
    to assign ``chapter``/``section`` labels in each ``GDDChunk``.
    """

    document_id: str = Field(
        ..., description="Foreign key to GDDDocument.document_id."
    )
    structure_detected: bool = Field(
        ...,
        description=(
            "False triggers the single-segment fallback rule "
            "(SRS-005, VAL-010)."
        ),
    )
    segment_count: int = Field(
        ...,
        ge=1,
        description="Total number of structural segments detected.",
    )
    table_of_contents: List[TOCEntry] = Field(
        default_factory=list,
        description="Ordered list of detected headings.",
    )
    preprocessing_config_ref: Optional[str] = Field(
        default=None,
        description=(
            "Reference to the versioned config snapshot used "
            "(Technical Architecture §8/§11)."
        ),
    )

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty.")
        return v
