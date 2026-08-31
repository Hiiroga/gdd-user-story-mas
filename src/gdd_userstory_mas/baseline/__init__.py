"""
gdd_userstory_mas.baseline
===========================
Single-Agent Baseline pipeline (C10 — SRS-015).

Public entry point: ``BaselinePipeline`` in ``pipeline.py``.

Components:
  llm_client.py        — Thin, provider-agnostic LLM interface.
  baseline_agent.py    — One-shot extraction logic (prompt + parse + validate).
  output_validator.py  — Post-generation schema validation (VAL-031).
  pipeline.py          — Orchestrator wiring preprocessing → agent → storage.
"""
from gdd_userstory_mas.baseline.llm_client import LLMClient, LLMResponse
from gdd_userstory_mas.baseline.baseline_agent import BaselineAgent, BaselineAgentConfig
from gdd_userstory_mas.baseline.output_validator import validate_baseline_output
from gdd_userstory_mas.baseline.pipeline import BaselinePipeline, BaselineConfig

__all__ = [
    "LLMClient",
    "LLMResponse",
    "BaselineAgent",
    "BaselineAgentConfig",
    "validate_baseline_output",
    "BaselinePipeline",
    "BaselineConfig",
]
