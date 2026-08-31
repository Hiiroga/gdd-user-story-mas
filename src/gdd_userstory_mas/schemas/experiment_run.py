"""
ExperimentRun — Data Contract §12
====================================
Top-level record aggregating one complete pipeline run (MAS or Baseline)
for one GDD document.

Schema source: Data Contracts Specification §12, ExperimentRun.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


RunStatus = Literal["running", "completed", "failed", "partially_completed"]
PipelineType = Literal["MAS", "Baseline"]


class ExperimentRun(BaseModel):
    """
    One complete pipeline execution for one GDD document.

    One ``ExperimentRun`` is created at the start of any run (MAS or
    Baseline) and updated as the run progresses.  Stored to
    ``outputs/{run_id}/experiment_run.json``.
    """

    run_id: str = Field(..., description="Unique run identifier (UUID4).")
    document_id: str = Field(
        ..., description="Foreign key to GDDDocument.document_id."
    )
    pipeline_type: PipelineType = Field(..., description="'MAS' or 'Baseline'.")
    config_snapshot_ref: str = Field(
        ...,
        description=(
            "Reference to the versioned config snapshot used for this run "
            "(Technical Architecture §8/§11). Typically the path to the "
            "frozen config file written at run start."
        ),
    )
    started_at: datetime = Field(..., description="ISO 8601 UTC start timestamp.")
    completed_at: Optional[datetime] = Field(default=None)
    status: RunStatus = Field(default="running")
    final_user_story_ids: List[str] = Field(
        default_factory=list,
        description="FinalUserStory.id records produced by this run.",
    )
    evaluation_result_ref: Optional[str] = Field(
        default=None,
        description="EvaluationResult.evaluation_id, populated on completion.",
    )
    error_log_refs: List[str] = Field(
        default_factory=list,
        description="ErrorFailureLog.error_id entries associated with this run.",
    )

    model_config = ConfigDict()
