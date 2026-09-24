"""
pipeline_result.py — MASPipelineResult
=========================================
Terminal output of a complete MAS pipeline run.

This is what ``MASPipeline.run()`` returns to the caller.
All individual artifacts are also persisted to disk in
``outputs/{run_id}/`` by the ``ResultStore`` during the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis


@dataclass
class MASPipelineResult:
    """
    Terminal output of one complete MAS pipeline run.

    Attributes
    ----------
    experiment_run:
        The ``ExperimentRun`` record with final status, timestamps, and
        references to all produced artefacts.
    final_stories:
        The canonical final ``FinalUserStory`` collection.
        Includes Valid, Merged, and Rejected-ManualReview stories.
    redundancy_analysis:
        The ``RedundancyAnalysis`` from Stage 5.
        ``None`` if no valid stories existed to check or Stage 5 was skipped.
    evaluation_result:
        The ``EvaluationResult`` from Stage 6 (the Evaluator Agent).
        ``None`` if the Evaluator failed or was skipped.
    error_log:
        All ``ErrorFailureLog`` entries recorded during the run.
        Empty list if no errors occurred.
    output_dir:
        Path to the run output directory (``outputs/{run_id}/``).
        All artifacts are persisted here.
    succeeded:
        ``True`` if the run completed (even partially) without a
        run-halting error.  ``False`` if the run was aborted (e.g.,
        preprocessing failure with ``abort_on_empty_chunks=True``).
    """

    experiment_run: ExperimentRun
    final_stories: List[FinalUserStory] = field(default_factory=list)
    redundancy_analysis: Optional[RedundancyAnalysis] = None
    evaluation_result: Optional[EvaluationResult] = None
    error_log: List[ErrorFailureLog] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    succeeded: bool = True

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self.experiment_run.run_id

    @property
    def total_stories(self) -> int:
        return len(self.final_stories)

    @property
    def total_errors(self) -> int:
        return len(self.error_log)

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = self.experiment_run.status
        return (
            f"MASPipelineResult(run={self.run_id}, status={status}, "
            f"stories={self.total_stories}, errors={self.total_errors}, "
            f"aqusa={self.evaluation_result.aqusa_score if self.evaluation_result else 'n/a'}%)"
        )
