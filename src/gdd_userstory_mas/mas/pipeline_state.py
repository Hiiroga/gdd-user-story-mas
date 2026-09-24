"""
pipeline_state.py — MASPipelineState
======================================
Tracks in-flight state for one MAS pipeline run.

Holds all intermediate artifacts accumulated across stages so the
orchestrator can:
  1. Persist intermediate results at each stage boundary.
  2. Provide a consistent view of run progress.
  3. Be resumed (partially) if interrupted.

Design:
  - Plain dataclass (not Pydantic) — mutable, accumulates across stages.
  - The orchestrator is the sole writer; agents never touch this directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from gdd_userstory_mas.schemas.candidate_requirement import AnalystOutput
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reader_output import ReaderOutput
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis


# Canonical stage names (aligned with StageType in ErrorFailureLog)
STAGE_PREPROCESSING = "chunking"
STAGE_READER = "gdd_reader"
STAGE_ANALYST = "requirements_analyst"
STAGE_GENERATOR = "user_story_generator"
STAGE_REVIEWER = "reviewer"
STAGE_REDUNDANCY = "redundancy_checker"
STAGE_EVALUATOR = "evaluator"
STAGE_RESULT_STORAGE = "result_storage"


@dataclass
class StoryIteration:
    """
    Records one Generator↔Reviewer iteration for a single candidate.
    Stored for traceability (TRACE-003).
    """

    iteration: int
    draft: GeneratedUserStory
    review_status: str          # "Valid" | "Invalid"
    failing_criteria: List[str] = field(default_factory=list)
    feedback_text: str = ""


@dataclass
class CandidateTrace:
    """
    Full generation trace for one CandidateRequirement.
    One entry per candidate in the pipeline state.
    """

    candidate_id: str
    chunk_id: str
    iterations: List[StoryIteration] = field(default_factory=list)
    final_status: str = "pending"  # "valid" | "rejected" | "error"
    final_story: Optional[GeneratedUserStory] = None


@dataclass
class MASPipelineState:
    """
    Mutable in-flight state for one MAS pipeline run.

    Attributes
    ----------
    run_id:
        Unique run identifier.
    experiment_run:
        ExperimentRun record (mutable — status/timestamps updated in-place).
    chunks:
        All preprocessing chunks (set after Stage 1).
    reader_outputs:
        One ReaderOutput per successfully processed chunk.
    analyst_outputs:
        One AnalystOutput per successfully processed chunk.
    candidate_traces:
        Full iteration trace per candidate (for traceability/audit).
    valid_generator_stories:
        GeneratedUserStory objects that passed the Reviewer (pre-redundancy).
    redundancy_analysis:
        Output of the Redundancy Checker (set after Stage 5).
    final_stories:
        Terminal FinalUserStory collection (post-redundancy + build step).
    error_log:
        All ErrorFailureLog entries recorded during this run.
    evaluation_result:
        Output of the Evaluator (set after Stage 6).
    stage_completed:
        Boolean flag per stage — used to guard idempotent re-runs.
    started_at:
        UTC timestamp when the run was initiated.
    """

    run_id: str
    experiment_run: ExperimentRun

    # Stage 1 outputs
    chunks: List[GDDChunk] = field(default_factory=list)

    # Stage 2 outputs
    reader_outputs: List[ReaderOutput] = field(default_factory=list)

    # Stage 3 outputs
    analyst_outputs: List[AnalystOutput] = field(default_factory=list)

    # Stage 4 outputs
    candidate_traces: List[CandidateTrace] = field(default_factory=list)
    valid_generator_stories: List[GeneratedUserStory] = field(default_factory=list)

    # Stage 5 outputs
    redundancy_analysis: Optional[RedundancyAnalysis] = None

    # Post-redundancy build
    final_stories: List[FinalUserStory] = field(default_factory=list)

    # Stage 6 outputs
    evaluation_result: Optional[EvaluationResult] = None

    # Cross-stage
    error_log: List[ErrorFailureLog] = field(default_factory=list)
    stage_completed: Dict[str, bool] = field(default_factory=dict)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Running counters for candidate IDs
    _candidate_id_offset: int = 0

    # ── Convenience helpers ────────────────────────────────────────────────────

    def mark_stage_complete(self, stage: str) -> None:
        self.stage_completed[stage] = True

    def is_stage_complete(self, stage: str) -> bool:
        return self.stage_completed.get(stage, False)

    def add_error(self, entry: ErrorFailureLog) -> None:
        self.error_log.append(entry)
        # Track in ExperimentRun
        if entry.error_id not in self.experiment_run.error_log_refs:
            self.experiment_run.error_log_refs.append(entry.error_id)

    def next_candidate_offset(self, count: int) -> int:
        """Return current offset, then advance by count."""
        offset = self._candidate_id_offset
        self._candidate_id_offset += count
        return offset

    @property
    def total_valid(self) -> int:
        return len(self.valid_generator_stories)

    @property
    def total_rejected(self) -> int:
        return sum(
            1 for t in self.candidate_traces if t.final_status == "rejected"
        )

    @property
    def total_errors(self) -> int:
        return len(self.error_log)
