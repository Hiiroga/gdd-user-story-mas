"""
GDD Multi-Agent System package.

Public API
----------
MASPipeline          — Orchestrator (Prompt 15)
MASPipelineConfig    — Configuration dataclass
MASPipelineResult    — Terminal result
MASPipelineState     — In-flight state (internal use)

Individual agents (C4–C9):
  GDDReaderAgent, RequirementsAnalystAgent, UserStoryGeneratorAgent,
  ReviewerAgent, RedundancyCheckerAgent, EvaluatorAgent

Persistence:
  ResultStore (gdd_userstory_mas.storage)
"""
from gdd_userstory_mas.mas.pipeline import MASPipeline
from gdd_userstory_mas.mas.pipeline_config import MASPipelineConfig
from gdd_userstory_mas.mas.pipeline_result import MASPipelineResult
from gdd_userstory_mas.mas.pipeline_state import MASPipelineState

__all__ = [
    "MASPipeline",
    "MASPipelineConfig",
    "MASPipelineResult",
    "MASPipelineState",
]
