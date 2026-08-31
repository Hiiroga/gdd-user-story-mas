"""
pipeline.py — Baseline Pipeline Orchestrator (C10 — SRS-015)
==============================================================
Wires preprocessing output → BaselineAgent → validation → storage.

Execution flow:
  1.  Load config from ``config/baseline.yaml``.
  2.  Initialise ``LLMClient`` + ``BaselineAgent``.
  3.  For each ``GDDChunk`` (in position order):
        a.  Call ``BaselineAgent.process_chunk()``.
        b.  Convert valid ``RawStory`` objects → ``FinalUserStory`` objects.
        c.  Log any per-chunk failures.
  4.  Run ``validate_baseline_output()`` on the full collection.
  5.  Persist outputs to ``outputs/{run_id}/02_baseline/``.
  6.  Update and return ``BaselineResult``.

Fairness guarantees (NFR-006, DATA-005):
  * The pipeline receives exactly the same ``GDDChunk`` list produced by
    ``PreprocessingPipeline``.  It does NOT re-preprocess.
  * Model config (provider, name, temperature) is documented in
    ``BaselineConfig`` and snapshotted alongside outputs.
  * Per-chunk invocation strategy (OPEN-016 resolved as "per_chunk") is
    explicitly documented.

Error handling (Technical Architecture §10):
  * ``LLMAPIError``          → logged + ``ErrorFailureLog`` entry; run halted.
  * ``LLMMalformedOutputError`` → retry once with the same chunk (re-prompt);
    if still malformed → log + skip chunk + continue (not halt).
  * Schema validation failure  → story excluded and logged.
  * All errors are appended to ``BaselineResult.errors`` and written to
    ``errors.jsonl`` via ``RunLogger``.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import ValidationError

from gdd_userstory_mas.baseline.baseline_agent import BaselineAgent, BaselineAgentConfig
from gdd_userstory_mas.baseline.llm_client import LLMAPIError, LLMClient, LLMMalformedOutputError
from gdd_userstory_mas.baseline.output_validator import ValidationReport, validate_baseline_output
from gdd_userstory_mas.logging.run_logger import RunLogger
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import (
    ConfidenceEvidence,
    FinalUserStory,
)
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    """
    Full configuration for the baseline pipeline run.

    Loaded from ``config/baseline.yaml`` via ``BaselineConfig.from_yaml()``.
    All fields are snapshotted alongside run outputs (Technical Architecture §8).
    """

    # Model
    provider: str = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: Optional[float] = None
    response_format: str = "json_object"

    # Invocation
    strategy: str = "per_chunk"
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    max_stories_per_chunk: int = 10

    # Prompt
    prompt_version: str = "baseline_v1"
    system_prompt_file: str = "config/prompts/baseline_system_prompt.txt"

    # Output
    pipeline_type: str = "Baseline"
    validation_status_default: str = "Unreviewed"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, project_root: Path = Path(".")) -> "BaselineConfig":
        """Load a ``BaselineConfig`` from ``config/baseline.yaml``."""
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        model = data.get("model", {})
        inv = data.get("invocation", {})
        prompt = data.get("prompt", {})
        output = data.get("output", {})

        return cls(
            provider=model.get("provider", "openai"),
            model_name=model.get("name", "gpt-4o"),
            temperature=model.get("temperature", 0.0),
            max_tokens=model.get("max_tokens", 2048),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            strategy=inv.get("strategy", "per_chunk"),
            max_retries=inv.get("max_retries", 3),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            max_stories_per_chunk=inv.get("max_stories_per_chunk", 10),
            prompt_version=prompt.get("version", "baseline_v1"),
            system_prompt_file=prompt.get(
                "system_prompt_file", "config/prompts/baseline_system_prompt.txt"
            ),
            pipeline_type=output.get("pipeline_type", "Baseline"),
            validation_status_default=output.get("validation_status_default", "Unreviewed"),
        )

    @classmethod
    def defaults(cls) -> "BaselineConfig":
        """Return a config with all defaults (no YAML file required)."""
        return cls()


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """
    Full output of one baseline pipeline run.

    Attributes
    ----------
    run_id:
        Unique run identifier.
    document_id:
        The processed GDD document's ID.
    stories:
        All validated ``FinalUserStory`` objects produced.
    validation_report:
        Cross-story validation summary.
    experiment_run:
        The ``ExperimentRun`` metadata record.
    errors:
        ``ErrorFailureLog`` entries generated during the run.
    total_prompt_tokens:
        Sum of prompt tokens across all LLM calls.
    total_completion_tokens:
        Sum of completion tokens across all LLM calls.
    """

    run_id: str
    document_id: str
    stories: List[FinalUserStory]
    validation_report: ValidationReport
    experiment_run: ExperimentRun
    errors: List[ErrorFailureLog] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


# ── Pipeline class ─────────────────────────────────────────────────────────────

class BaselinePipeline:
    """
    Orchestrator for the single-agent baseline (C10 — SRS-015).

    Parameters
    ----------
    config:
        ``BaselineConfig`` instance.  ``None`` uses defaults.
    api_key:
        LLM provider API key.  If ``None``, the SDK reads from its
        standard environment variable.
    logger:
        ``RunLogger`` instance.  If ``None``, a no-op stub is used.
    output_dir:
        If set, all outputs are written to ``<output_dir>/02_baseline/``.
    project_root:
        Project root for resolving relative paths (prompt file, etc.).
    """

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        api_key: Optional[str] = None,
        logger: Optional[RunLogger] = None,
        output_dir: Optional[Path] = None,
        project_root: Path = Path("."),
    ) -> None:
        self._config = config or BaselineConfig.defaults()
        self._api_key = api_key
        self._logger = logger
        self._output_dir = output_dir
        self._project_root = project_root

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(
        self,
        chunks: List[GDDChunk],
        *,
        document_id: str,
        run_id: Optional[str] = None,
    ) -> BaselineResult:
        """
        Run the baseline pipeline on a pre-processed list of ``GDDChunk`` objects.

        Parameters
        ----------
        chunks:
            Output of ``PreprocessingPipeline.run().chunks`` — MUST be the
            identical list used (or to be used) by the MAS pipeline
            (DATA-005, NFR-006).
        document_id:
            The GDD document's unique identifier.
        run_id:
            Optional pre-assigned run ID; UUID4 generated if omitted.

        Returns
        -------
        BaselineResult
        """
        rid = run_id or str(uuid.uuid4())
        started_at = datetime.now(tz=timezone.utc)
        errors: List[ErrorFailureLog] = []
        stories: List[FinalUserStory] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        self._log("info", "single_agent_baseline",
                  f"Baseline run starting: run_id={rid}, document_id={document_id}, "
                  f"chunks={len(chunks)}, model={self._config.model_name}")

        # ── Snapshot config alongside outputs ──────────────────────────────────
        config_snapshot_ref = self._snapshot_config(rid)

        # ── Initialise ExperimentRun ───────────────────────────────────────────
        experiment_run = ExperimentRun(
            run_id=rid,
            document_id=document_id,
            pipeline_type="Baseline",
            config_snapshot_ref=config_snapshot_ref,
            started_at=started_at,
            status="running",
        )

        # ── Build agent ────────────────────────────────────────────────────────
        prompt_path = (
            self._project_root / self._config.system_prompt_file
        )
        agent_config = BaselineAgentConfig(
            system_prompt_path=prompt_path,
            max_stories_per_chunk=self._config.max_stories_per_chunk,
            prompt_version=self._config.prompt_version,
        )

        try:
            llm_client = LLMClient(
                provider=self._config.provider,
                model_name=self._config.model_name,
                api_key=self._api_key,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                top_p=self._config.top_p,
                response_format=self._config.response_format,
                max_retries=self._config.max_retries,
                retry_backoff_seconds=self._config.retry_backoff_seconds,
            )
        except ImportError as exc:
            err = self._make_error(rid, document_id, "llm_api_failure", str(exc), "run_halted")
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            experiment_run.status = "failed"
            experiment_run.completed_at = datetime.now(tz=timezone.utc)
            raise RuntimeError(f"LLM client initialisation failed: {exc}") from exc

        agent = BaselineAgent(llm_client, agent_config)

        # ── Process chunks ─────────────────────────────────────────────────────
        sorted_chunks = sorted(chunks, key=lambda c: c.position)
        story_counter = 0

        for chunk in sorted_chunks:
            try:
                result = agent.process_chunk(chunk, total_chunks=len(sorted_chunks))
            except LLMAPIError as exc:
                # API failure — all retries exhausted; halt this run
                err = self._make_error(
                    rid, document_id, "llm_api_failure", str(exc), "run_halted",
                    related_entity_id=chunk.chunk_id,
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)
                experiment_run.status = "failed"
                experiment_run.completed_at = datetime.now(tz=timezone.utc)
                raise RuntimeError(
                    f"LLM API failure on chunk {chunk.chunk_id}: {exc}"
                ) from exc
            except LLMMalformedOutputError as exc:
                # Malformed output — skip this chunk, continue
                err = self._make_error(
                    rid, document_id, "malformed_output",
                    str(exc), "excluded_and_logged",
                    related_entity_id=chunk.chunk_id,
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)
                self._log("error", "single_agent_baseline",
                          f"Skipping chunk {chunk.chunk_id} due to malformed LLM output.")
                continue

            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens

            # Log the raw LLM response for auditability
            self._write_chunk_log(rid, chunk.chunk_id, result.llm_response_raw)

            # Convert valid RawStory → FinalUserStory
            for raw in result.valid_stories:
                try:
                    story = self._make_final_story(
                        raw=raw,
                        chunk=chunk,
                        story_counter=story_counter,
                        run_id=rid,
                        document_id=document_id,
                        model_name=result.model_name,
                    )
                    stories.append(story)
                    story_counter += 1
                except ValidationError as exc:
                    err = self._make_error(
                        rid, document_id, "schema_violation",
                        f"FinalUserStory validation failed: {exc}",
                        "excluded_and_logged",
                        related_entity_id=chunk.chunk_id,
                    )
                    errors.append(err)
                    if self._logger:
                        self._logger.log_error(err)

            # Log skipped stories
            for bad in result.invalid_stories:
                err = self._make_error(
                    rid, document_id, "malformed_output",
                    f"Invalid story in chunk {chunk.chunk_id}: {bad.parse_errors}",
                    "excluded_and_logged",
                    related_entity_id=chunk.chunk_id,
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)

        # ── Validate full collection ───────────────────────────────────────────
        self._log("info", "single_agent_baseline", "Running output validation (VAL-031/032).")
        validation_report = validate_baseline_output(stories, chunks)

        if not validation_report.passed:
            for ve in validation_report.errors:
                err = self._make_error(
                    rid, document_id, "schema_violation", ve,
                    "excluded_and_logged",
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)

        # ── Update ExperimentRun ───────────────────────────────────────────────
        experiment_run.status = "partially_completed" if errors else "completed"
        experiment_run.completed_at = datetime.now(tz=timezone.utc)
        experiment_run.final_user_story_ids = [s.id for s in stories]
        experiment_run.error_log_refs = [e.error_id for e in errors]

        self._log(
            "info", "single_agent_baseline",
            f"Baseline run complete: {len(stories)} stories, "
            f"{len(errors)} errors, {total_prompt_tokens} prompt tokens, "
            f"{total_completion_tokens} completion tokens.",
        )

        result_obj = BaselineResult(
            run_id=rid,
            document_id=document_id,
            stories=stories,
            validation_report=validation_report,
            experiment_run=experiment_run,
            errors=errors,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
        )

        # ── Persist ────────────────────────────────────────────────────────────
        if self._output_dir:
            self._persist(result_obj)

        return result_obj

    # ── FinalUserStory factory ─────────────────────────────────────────────────

    def _make_final_story(
        self,
        raw,
        chunk: GDDChunk,
        story_counter: int,
        run_id: str,
        document_id: str,
        model_name: str,
    ) -> FinalUserStory:
        """Convert a validated ``RawStory`` into a ``FinalUserStory``."""
        story_id = f"US-{story_counter + 1:04d}"
        section_label = chunk.chapter
        if chunk.section:
            section_label = (
                f"{chunk.chapter} > {chunk.section}"
                if chunk.chapter
                else chunk.section
            )

        # Normalise requirement_type: fall back to "player" if unrecognised
        req_type = raw.requirement_type.lower() if raw.requirement_type else ""
        valid_req_types = {"player", "system", "in-game entity", "dev team"}
        if req_type not in valid_req_types:
            req_type = "player"

        # Normalise game_domain
        valid_domains = {"gameplay", "ui", "narrative", "systems", "level design", "audio", "other"}
        game_domain = raw.game_domain.lower() if raw.game_domain else "other"
        # normalise "UI" case
        if game_domain == "ui":
            game_domain = "UI"
        elif game_domain not in valid_domains:
            game_domain = "other"
        # Restore capitalisation for UI
        if raw.game_domain and raw.game_domain.upper() == "UI":
            game_domain = "UI"

        return FinalUserStory(
            id=story_id,
            run_id=run_id,
            pipeline_type="Baseline",
            source_document_id=document_id,
            source_section=section_label,
            source_chunk_ids=[chunk.chunk_id],
            role=raw.role,
            action=raw.action,
            benefit=raw.benefit,
            full_text=raw.full_text,
            requirement_type=req_type,  # type: ignore[arg-type]
            game_domain=game_domain,  # type: ignore[arg-type]
            validation_status="Unreviewed",
            validation_feedback=[],
            merge_history=None,
            confidence_evidence=ConfidenceEvidence(
                source_excerpt=raw.source_excerpt,
                reviewer_traceability_confirmed=None,
            ),
            iteration_count_final=None,
        )

    # ── Persistence helpers ────────────────────────────────────────────────────

    def _persist(self, result: BaselineResult) -> None:
        """Write all baseline outputs to ``<output_dir>/02_baseline/``."""
        out_dir = self._output_dir / "02_baseline"
        out_dir.mkdir(parents=True, exist_ok=True)

        # User stories JSON
        stories_data = [s.model_dump(mode="json") for s in result.stories]
        (out_dir / "final_user_stories.json").write_text(
            json.dumps(stories_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Experiment run
        (out_dir / "experiment_run.json").write_text(
            result.experiment_run.model_dump_json(indent=2), encoding="utf-8"
        )

        # Validation report
        report_data = {
            "total_stories": result.validation_report.total_stories,
            "valid_count": result.validation_report.valid_count,
            "invalid_count": result.validation_report.invalid_count,
            "passed": result.validation_report.passed,
            "errors": result.validation_report.errors,
            "warnings": result.validation_report.warnings,
        }
        (out_dir / "validation_report.json").write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Token usage summary
        token_summary = {
            "total_prompt_tokens": result.total_prompt_tokens,
            "total_completion_tokens": result.total_completion_tokens,
            "total_tokens": result.total_prompt_tokens + result.total_completion_tokens,
        }
        (out_dir / "token_usage.json").write_text(
            json.dumps(token_summary, indent=2), encoding="utf-8"
        )

        self._log(
            "info", "single_agent_baseline",
            f"Baseline output written to {out_dir}",
        )

    def _snapshot_config(self, run_id: str) -> str:
        """
        Copy ``config/baseline.yaml`` into the run output directory as a
        frozen snapshot.  Returns the path string as the config_snapshot_ref.
        """
        if not self._output_dir:
            return str(self._project_root / self._config.system_prompt_file)

        snap_dir = self._output_dir / "02_baseline"
        snap_dir.mkdir(parents=True, exist_ok=True)
        src = self._project_root / "config" / "baseline.yaml"
        dst = snap_dir / "config_snapshot.yaml"

        if src.exists():
            shutil.copy2(src, dst)
            return str(dst)
        return str(src)

    def _write_chunk_log(self, run_id: str, chunk_id: str, raw_json: str) -> None:
        """Append the raw LLM JSON response for a chunk to the audit trail."""
        if not self._output_dir:
            return
        log_dir = self._output_dir / "02_baseline" / "chunk_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{chunk_id}.json"
        log_file.write_text(raw_json, encoding="utf-8")

    # ── Logging / error helpers ────────────────────────────────────────────────

    def _log(self, level: str, stage: str, message: str) -> None:
        if self._logger:
            getattr(self._logger, level)(stage, message)

    @staticmethod
    def _make_error(
        run_id: str,
        document_id: str,
        error_type: str,
        message: str,
        resolution: str,
        *,
        related_entity_id: Optional[str] = None,
    ) -> ErrorFailureLog:
        return ErrorFailureLog(
            error_id=str(uuid.uuid4()),
            run_id=run_id,
            document_id=document_id,
            stage="single_agent_baseline",  # type: ignore[arg-type]
            error_type=error_type,  # type: ignore[arg-type]
            message=message,
            resolution=resolution,  # type: ignore[arg-type]
            related_entity_id=related_entity_id,
            timestamp=datetime.now(tz=timezone.utc),
        )
