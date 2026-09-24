"""
result_store.py — ResultStore
================================
Handles all JSON/JSONL persistence for one MAS pipeline run.

Writes artifacts to::

    {output_dir}/{run_id}/
        experiment_run.json
        config_snapshot.yaml
        preprocessing/
            chunks.jsonl
        reader/
            {chunk_id}.json
        analyst/
            {chunk_id}.json
        generator_reviewer/
            {draft_id}_iter{N}.json
        redundancy/
            analysis.json
        final_stories.jsonl
        evaluation_result.json
        errors/
            errors.jsonl

Design:
  - Stateless per call — no in-memory cache.
  - All write operations are atomic (write-then-rename not implemented here
    for simplicity; files are small enough that partial writes are harmless
    at research scale).
  - All read operations return ``None`` on missing file (graceful degradation
    for re-run scenarios).
  - Uses Pydantic ``.model_dump_json()`` for deterministic serialization.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from gdd_userstory_mas.schemas.candidate_requirement import AnalystOutput
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reader_output import ReaderOutput
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis

logger = logging.getLogger(__name__)


class ResultStore:
    """
    Artifact persistence layer for one MAS pipeline run.

    All paths are rooted at ``{output_dir}/{run_id}/``.
    Directories are created on first use.

    Parameters
    ----------
    run_id:
        Unique run identifier.
    output_dir:
        Root output directory (``outputs/`` by default).
    """

    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ResultStore initialised: %s", self.run_dir)

    # ── Directory helpers ──────────────────────────────────────────────────────

    def _dir(self, *parts: str) -> Path:
        d = self.run_dir.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        logger.debug("Saved: %s", path)

    def _read(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ── ExperimentRun ──────────────────────────────────────────────────────────

    def save_experiment_run(self, run: ExperimentRun) -> None:
        """Write (or overwrite) the ExperimentRun record."""
        path = self.run_dir / "experiment_run.json"
        self._write(path, run.model_dump_json(indent=2))

    def load_experiment_run(self) -> Optional[ExperimentRun]:
        """Load a previously persisted ExperimentRun, or None."""
        raw = self._read(self.run_dir / "experiment_run.json")
        if raw is None:
            return None
        return ExperimentRun.model_validate_json(raw)

    # ── Config snapshot ────────────────────────────────────────────────────────

    def save_config_snapshot(self, yaml_text: str) -> Path:
        """Write the frozen YAML config snapshot for reproducibility."""
        path = self.run_dir / "config_snapshot.yaml"
        self._write(path, yaml_text)
        return path

    # ── Preprocessing / Chunks ─────────────────────────────────────────────────

    def save_chunks(self, chunks: List[GDDChunk]) -> None:
        """Write all chunks as JSONL."""
        d = self._dir("preprocessing")
        path = d / "chunks.jsonl"
        lines = "\n".join(c.model_dump_json() for c in chunks)
        self._write(path, lines + "\n" if lines else "")

    def load_chunks(self) -> List[GDDChunk]:
        """Load persisted chunks, or empty list."""
        raw = self._read(self.run_dir / "preprocessing" / "chunks.jsonl")
        if not raw:
            return []
        return [GDDChunk.model_validate_json(line) for line in raw.splitlines() if line.strip()]

    # ── Reader outputs ─────────────────────────────────────────────────────────

    def save_reader_output(self, chunk_id: str, output: ReaderOutput) -> None:
        d = self._dir("reader")
        safe = chunk_id.replace("/", "_").replace("\\", "_")
        self._write(d / f"{safe}.json", output.model_dump_json(indent=2))

    def load_reader_output(self, chunk_id: str) -> Optional[ReaderOutput]:
        safe = chunk_id.replace("/", "_").replace("\\", "_")
        raw = self._read(self.run_dir / "reader" / f"{safe}.json")
        if raw is None:
            return None
        return ReaderOutput.model_validate_json(raw)

    # ── Analyst outputs ────────────────────────────────────────────────────────

    def save_analyst_output(self, chunk_id: str, output: AnalystOutput) -> None:
        d = self._dir("analyst")
        safe = chunk_id.replace("/", "_").replace("\\", "_")
        self._write(d / f"{safe}.json", output.model_dump_json(indent=2))

    def load_analyst_output(self, chunk_id: str) -> Optional[AnalystOutput]:
        safe = chunk_id.replace("/", "_").replace("\\", "_")
        raw = self._read(self.run_dir / "analyst" / f"{safe}.json")
        if raw is None:
            return None
        return AnalystOutput.model_validate_json(raw)

    # ── Generator / Reviewer iterations ───────────────────────────────────────

    def save_draft_iteration(
        self,
        draft: GeneratedUserStory,
        review_status: str,
    ) -> None:
        """Persist one draft + its review verdict (one file per iteration)."""
        d = self._dir("generator_reviewer")
        filename = f"{draft.draft_id}.json"
        payload = {
            "draft": json.loads(draft.model_dump_json()),
            "review_status": review_status,
        }
        self._write(d / filename, json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Redundancy ─────────────────────────────────────────────────────────────

    def save_redundancy_analysis(self, ra: RedundancyAnalysis) -> None:
        d = self._dir("redundancy")
        self._write(d / "analysis.json", ra.model_dump_json(indent=2))

    def load_redundancy_analysis(self) -> Optional[RedundancyAnalysis]:
        raw = self._read(self.run_dir / "redundancy" / "analysis.json")
        if raw is None:
            return None
        return RedundancyAnalysis.model_validate_json(raw)

    # ── Final stories ──────────────────────────────────────────────────────────

    def save_final_stories(self, stories: List[FinalUserStory]) -> None:
        path = self.run_dir / "final_stories.jsonl"
        lines = "\n".join(s.model_dump_json() for s in stories)
        self._write(path, lines + "\n" if lines else "")

    def load_final_stories(self) -> List[FinalUserStory]:
        raw = self._read(self.run_dir / "final_stories.jsonl")
        if not raw:
            return []
        return [
            FinalUserStory.model_validate_json(line)
            for line in raw.splitlines()
            if line.strip()
        ]

    # ── Evaluation result ──────────────────────────────────────────────────────

    def save_evaluation_result(self, er: EvaluationResult) -> None:
        self._write(self.run_dir / "evaluation_result.json", er.model_dump_json(indent=2))

    def load_evaluation_result(self) -> Optional[EvaluationResult]:
        raw = self._read(self.run_dir / "evaluation_result.json")
        if raw is None:
            return None
        return EvaluationResult.model_validate_json(raw)

    # ── Error log ──────────────────────────────────────────────────────────────

    def append_error_log(self, entry: ErrorFailureLog) -> None:
        """Append one error entry to errors.jsonl (JSONL, one entry per line)."""
        d = self._dir("errors")
        path = d / "errors.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def load_error_log(self) -> List[ErrorFailureLog]:
        raw = self._read(self.run_dir / "errors" / "errors.jsonl")
        if not raw:
            return []
        return [
            ErrorFailureLog.model_validate_json(line)
            for line in raw.splitlines()
            if line.strip()
        ]

    # ── Summary ────────────────────────────────────────────────────────────────

    @property
    def run_dir_path(self) -> Path:
        return self.run_dir
