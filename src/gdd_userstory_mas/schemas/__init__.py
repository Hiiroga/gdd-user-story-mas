"""
gdd_userstory_mas.schemas
=========================
Pydantic data-contract models derived 1-to-1 from the approved
Data Contracts Specification document.

Schemas implemented here cover:
  - Preprocessing (§1–§3)
  - MAS Reader Agent output (§4 ReaderOutput, EvidencedItem)
  - Baseline and shared terminal output (§10, §12, §13)

MAS-only intermediate schemas (CandidateRequirement §5,
GeneratedUserStory §6, ReviewerResult §7–8, RedundancyAnalysis §9) will be
added when the corresponding MAS agents are implemented.
"""
from gdd_userstory_mas.schemas.gdd_document import GDDDocument
from gdd_userstory_mas.schemas.gdd_metadata import GDDMetadata, TOCEntry
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import ReaderOutput, EvidencedItem
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.final_user_story import (
    FinalUserStory,
    ConfidenceEvidence,
    MergeHistoryEntry,
)
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun

__all__ = [
    "GDDDocument",
    "GDDMetadata",
    "TOCEntry",
    "GDDChunk",
    "ReaderOutput",
    "EvidencedItem",
    "ErrorFailureLog",
    "FinalUserStory",
    "ConfidenceEvidence",
    "MergeHistoryEntry",
    "ExperimentRun",
]
