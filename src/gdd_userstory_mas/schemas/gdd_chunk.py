"""
GDDChunk — Data Contract §3
=============================
A single context-limit-compliant unit of text passed to the GDD Reader Agent.

Schema source: Data Contracts Specification §3, GDDChunk.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GDDChunk(BaseModel):
    """
    A preprocessed text chunk with full traceability metadata.

    Many ``GDDChunk`` records are produced per ``GDDDocument``
    (one-to-many).  Each chunk is the direct input to exactly one
    ``ReaderOutput`` (produced by the GDD Reader Agent).
    """

    chunk_id: str = Field(
        ...,
        description="Unique identifier within document_id.",
    )
    document_id: str = Field(
        ...,
        description="Foreign key to GDDDocument.document_id.",
    )
    text: str = Field(
        ...,
        description="The text content of this chunk.",
    )
    chapter: Optional[str] = Field(
        default=None,
        description="Chapter label derived from segmentation (null if undetected).",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section label derived from segmentation (null if undetected).",
    )
    position: int = Field(
        ...,
        ge=0,
        description="Sequential order index within the document (SRS-007).",
    )
    char_offset_start: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Character index in the cleaned document text where this chunk starts."
        ),
    )
    char_offset_end: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Character index in the cleaned document text where this chunk ends "
            "(exclusive)."
        ),
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Approximate token count. Must not exceed configured max (VAL-011).",
    )
    overlap_with_previous: Optional[bool] = Field(
        default=None,
        description=(
            "Whether this chunk overlaps prior chunk content. "
            "Overlap strategy: OPEN DECISION (OPEN-013)."
        ),
    )

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_id must not be empty.")
        return v

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty.")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("GDDChunk.text must not be empty (VAL-009).")
        return v

    @model_validator(mode="after")
    def char_offsets_consistent(self) -> "GDDChunk":
        """Ensure start < end when both offsets are provided."""
        if (
            self.char_offset_start is not None
            and self.char_offset_end is not None
        ):
            if self.char_offset_start >= self.char_offset_end:
                raise ValueError(
                    "char_offset_start must be strictly less than char_offset_end."
                )
        return self
