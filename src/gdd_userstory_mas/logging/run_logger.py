"""
run_logger.py — SRS-018 / Technical Architecture §9
======================================================
Per-run, per-stage structured logger.

Writes to two streams simultaneously:
  * <logs_dir>/<run_id>/run.log        — human-readable, plain-text
  * <logs_dir>/<run_id>/errors.jsonl   — machine-readable ErrorFailureLog entries

Usage
-----
::

    logger = RunLogger(run_id="abc-123", logs_dir=Path("logs"))
    logger.info("ingestion", "Accepted file 'sample.pdf'")
    logger.log_error(error_log_entry)
    logger.close()

Or use as a context manager::

    with RunLogger(run_id="abc-123", logs_dir=Path("logs")) as logger:
        logger.info("ingestion", "Accepted file 'sample.pdf'")
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog


class RunLogger:
    """
    Structured logger for one pipeline run.

    Parameters
    ----------
    run_id:
        The unique run identifier (used as the sub-directory name).
    logs_dir:
        Root logs directory.  Sub-directory ``<logs_dir>/<run_id>/`` is
        created automatically.
    level:
        Python logging level string (``"DEBUG"``, ``"INFO"``, etc.).
    echo_stdout:
        When ``True`` (default in dev mode), also write human-readable
        messages to ``stdout`` so the researcher can watch runs live.
    """

    def __init__(
        self,
        run_id: str,
        logs_dir: Path = Path("logs"),
        level: str = "INFO",
        echo_stdout: bool = True,
    ) -> None:
        self.run_id = run_id
        self.run_log_dir = logs_dir / run_id
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        # ── Human-readable log ─────────────────────────────────────────────
        log_path = self.run_log_dir / "run.log"
        self._logger = logging.getLogger(f"gdd_mas.run.{run_id}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False  # don't bubble to root

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        if echo_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

        # ── Machine-readable error log (JSON Lines) ────────────────────────
        errors_path = self.run_log_dir / "errors.jsonl"
        self._errors_fh = open(errors_path, "a", encoding="utf-8")  # noqa: SIM115

        self._logger.info("RunLogger initialised for run_id=%s", run_id)

    # ── Public logging API ─────────────────────────────────────────────────────

    def debug(self, stage: str, message: str) -> None:
        """Write a DEBUG-level message to run.log."""
        self._logger.debug("[%s] %s", stage, message)

    def info(self, stage: str, message: str) -> None:
        """Write an INFO-level message to run.log."""
        self._logger.info("[%s] %s", stage, message)

    def warning(self, stage: str, message: str) -> None:
        """Write a WARNING-level message to run.log."""
        self._logger.warning("[%s] %s", stage, message)

    def error(self, stage: str, message: str) -> None:
        """Write an ERROR-level message to run.log (without a structured entry)."""
        self._logger.error("[%s] %s", stage, message)

    def log_error(self, entry: ErrorFailureLog) -> None:
        """
        Write a structured ``ErrorFailureLog`` entry to both run.log (as
        ERROR) and errors.jsonl.
        """
        self._logger.error(
            "[%s] %s (type=%s, resolution=%s)",
            entry.stage,
            entry.message,
            entry.error_type,
            entry.resolution,
        )
        json_line = entry.model_dump_json()
        self._errors_fh.write(json_line + "\n")
        self._errors_fh.flush()

    # ── Context-manager support ────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close all file handles."""
        self._errors_fh.flush()
        self._errors_fh.close()
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
        return False  # do not suppress exceptions
