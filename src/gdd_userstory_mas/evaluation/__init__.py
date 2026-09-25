"""
evaluation — Research Evaluation Package
=========================================
Implements the comparative evaluation pipeline for the thesis experiment:
    Single-Agent Baseline vs. Multi-Agent System (MAS)

Modules:
    metrics    — Pure, deterministic metric calculation functions (testable).
    schemas    — Pydantic schemas for ComparisonResult and ObservationRecord.
    loader     — Functions to load experiment outputs from disk.
    comparator — Orchestrates evaluation of both pipelines and produces
                 the final ComparisonResult.
"""
