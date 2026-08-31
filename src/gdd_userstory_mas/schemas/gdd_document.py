"""
GDDDocument — Data Contract §1
================================
Represents the raw ingested source document before preprocessing.

Schema source: Data Contracts Specification §1, GDDDocument.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import ConfigDict

from pydantic import BaseModel, Field, field_validator


class GDDDocument(BaseModel):
    """
    Root artifact of the pipeline.

    One ``GDDDocument`` produces exactly one ``GDDMetadata`` and many
    ``GDDChunk`` records via Preprocessing (SRS-003–SRS-007).
    """

    document_id: str = Field(
        ...,
        description=(
            "Unique identifier assigned at ingestion (SRS-001/SRS-002). "
            "Format: UUID4 string."
        ),
    )
    file_name: str = Field(..., description="Original filename of the GDD.")
    format: Literal["txt", "pdf"] = Field(
        ..., description="Input file format. Per DATA-002."
    )
    language: Literal["en"] = Field(
        default="en",
        description="Document language. English-only per DATA-001.",
    )
    genre: Optional[str] = Field(
        default=None,
        description=(
            "Game genre, e.g., RPG, action-adventure, puzzle (DATA-004). "
            "Optional metadata for dataset diversity reporting."
        ),
    )
    raw_text: Optional[str] = Field(
        default=None,
        description=(
            "Populated after PDF-to-text conversion (SRS-003) or directly "
            "for .txt input."
        ),
    )
    ingestion_timestamp: datetime = Field(
        ..., description="ISO 8601 datetime when the document was ingested."
    )
    source_provenance: Optional[str] = Field(
        default=None,
        description=(
            "e.g., public repository, academic publication, researcher-created "
            "(per DATA-006/ASM-002)."
        ),
    )

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty.")
        return v

    @field_validator("file_name")
    @classmethod
    def file_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("file_name must not be empty.")
        return v

    model_config = ConfigDict()
