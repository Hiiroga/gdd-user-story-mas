"""
gdd_userstory_mas.logging
==========================
Structured logging helpers for the GDD User Story MAS pipeline.

Two output streams are maintained (per Technical Architecture §9 and the
Project Structure logging strategy §8):

* ``run.log``      — human-readable chronological narrative.
* ``errors.jsonl`` — machine-readable ``ErrorFailureLog`` entries (JSON Lines).
* ``audit_trail.jsonl`` — feedback-loop / redundancy-loop iteration history.
"""
from gdd_userstory_mas.logging.run_logger import RunLogger

__all__ = ["RunLogger"]
