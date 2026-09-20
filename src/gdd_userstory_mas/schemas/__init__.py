"""
gdd_userstory_mas.schemas
=========================
Pydantic data-contract models derived 1-to-1 from the approved
Data Contracts Specification document.

Schemas implemented here cover:
  - Preprocessing (§1–§3)
  - MAS Reader Agent output (§4 ReaderOutput, EvidencedItem)
  - MAS Analyst Agent output (§5 CandidateRequirement, AnalystOutput)
  - MAS Generator Agent output (§6 GeneratedUserStory)
  - MAS Reviewer Agent output (§7 ReviewerResult, §8 ReviewerFeedback)
  - MAS Redundancy Checker output (§9 RedundancyAnalysis)
  - Baseline and shared terminal output (§10, §12, §13)
"""
from gdd_userstory_mas.schemas.gdd_document import GDDDocument
from gdd_userstory_mas.schemas.gdd_metadata import GDDMetadata, TOCEntry
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import ReaderOutput, EvidencedItem
from gdd_userstory_mas.schemas.candidate_requirement import (
    CandidateRequirement,
    AnalystOutput,
    DomainType,
    PerspectiveType,
    VALID_DOMAINS,
)
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reviewer_result import (
    ReviewerResult,
    ReviewerFeedback,
    CriterionType,
    VALID_CRITERIA,
)
from gdd_userstory_mas.schemas.redundancy_analysis import (
    RedundancyAnalysis,
    DuplicateGroup,
    MergeAction,
)
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
    "CandidateRequirement",
    "AnalystOutput",
    "DomainType",
    "PerspectiveType",
    "VALID_DOMAINS",
    "ErrorFailureLog",
    "GeneratedUserStory",
    "ReviewerResult",
    "ReviewerFeedback",
    "CriterionType",
    "VALID_CRITERIA",
    "RedundancyAnalysis",
    "DuplicateGroup",
    "MergeAction",
    "FinalUserStory",
    "ConfidenceEvidence",
    "MergeHistoryEntry",
    "ExperimentRun",
]


