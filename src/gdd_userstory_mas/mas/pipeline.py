"""
pipeline.py — MASPipeline (Orchestration Layer)
=================================================
Integrates all 6 MAS agents into a single deterministic execution flow:

    Preprocessing → Reader → Analyst → Generator ↔ Reviewer → Redundancy Checker → Evaluator

This module is the ONLY place that knows the execution order and the
handoff contracts between agents.  Individual agents are NOT modified.

Key design rules:
  - Per-story failure isolation: one LLM failure for one candidate does
    NOT abort sibling candidates or other chunks.
  - Preprocessing failure IS fatal (nothing to process).
  - All failures are logged via RunLogger AND persisted as ErrorFailureLog
    entries in ResultStore.
  - Schema validation gates every agent-to-agent handoff.
  - Generator↔Reviewer feedback loop is bounded by ``max_reviewer_iterations``.
  - All intermediate results are persisted at each stage boundary.
  - Full traceability: every FinalUserStory carries source_chunk_ids,
    run_id, and confidence_evidence.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError
from gdd_userstory_mas.logging.run_logger import RunLogger
from gdd_userstory_mas.mas.analyst_agent import RequirementsAnalystAgent
from gdd_userstory_mas.mas.analyst_output_validator import validate_analyst_output
from gdd_userstory_mas.mas.evaluator_agent import EvaluatorAgent
from gdd_userstory_mas.mas.evaluator_output_validator import validate_evaluator_output
from gdd_userstory_mas.mas.generator_agent import UserStoryGeneratorAgent
from gdd_userstory_mas.mas.generator_output_validator import validate_generator_output
from gdd_userstory_mas.mas.pipeline_config import MASPipelineConfig
from gdd_userstory_mas.mas.pipeline_result import MASPipelineResult
from gdd_userstory_mas.mas.pipeline_state import (
    STAGE_ANALYST,
    STAGE_EVALUATOR,
    STAGE_GENERATOR,
    STAGE_PREPROCESSING,
    STAGE_READER,
    STAGE_REDUNDANCY,
    STAGE_RESULT_STORAGE,
    STAGE_REVIEWER,
    CandidateTrace,
    MASPipelineState,
    StoryIteration,
)
from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent
from gdd_userstory_mas.mas.reader_output_validator import validate_reader_output
from gdd_userstory_mas.mas.redundancy_agent import RedundancyCheckerAgent
from gdd_userstory_mas.mas.redundancy_output_validator import validate_redundancy_output
from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgent
from gdd_userstory_mas.mas.reviewer_output_validator import validate_reviewer_output
from gdd_userstory_mas.preprocessing.pipeline import PreprocessingPipeline
from gdd_userstory_mas.schemas.candidate_requirement import CandidateRequirement
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.evaluation_result import EvaluationResult
from gdd_userstory_mas.schemas.experiment_run import ExperimentRun
from gdd_userstory_mas.schemas.final_user_story import (
    ConfidenceEvidence,
    FinalUserStory,
)
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis
from gdd_userstory_mas.schemas.reviewer_result import ReviewerFeedback, ReviewerResult
from gdd_userstory_mas.storage.result_store import ResultStore

logger = logging.getLogger(__name__)


class MASPipelineError(RuntimeError):
    """Raised when a fatal (run-halting) pipeline error occurs."""


class MASPipeline:
    """
    MAS Orchestration Pipeline.

    Runs the full GDD → User Story extraction pipeline for one document:
        Preprocessing → Reader → Analyst → Generator ↔ Reviewer
        → Redundancy Checker → Evaluator

    Parameters
    ----------
    config:
        ``MASPipelineConfig`` with all agent configs and orchestration settings.
    api_key:
        LLM API key.  If None, reads from the ``LLM_API_KEY`` environment variable.
    project_root:
        Project root directory (for resolving prompt file paths).

    Usage
    -----
    ::

        config = MASPipelineConfig.from_yaml(
            "config/mas.yaml", "config/pipeline.yaml"
        )
        result = MASPipeline(config, api_key=os.getenv("LLM_API_KEY")).run(
            Path("datasets/raw/doombible.pdf")
        )
        print(result.summary())
    """

    def __init__(
        self,
        config: MASPipelineConfig,
        api_key: Optional[str] = None,
        project_root: Path = Path("."),
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._project_root = Path(project_root)

        # Instantiate all agents (stateless — can be reused across chunks)
        self._reader = GDDReaderAgent(
            config=config.reader_config,
            api_key=api_key,
            project_root=self._project_root,
        )
        self._analyst = RequirementsAnalystAgent(
            config=config.analyst_config,
            api_key=api_key,
            project_root=self._project_root,
        )
        self._generator = UserStoryGeneratorAgent(
            config=config.generator_config,
            api_key=api_key,
            project_root=self._project_root,
        )
        self._reviewer = ReviewerAgent(
            config=config.reviewer_config,
            api_key=api_key,
            project_root=self._project_root,
        )
        self._redundancy = RedundancyCheckerAgent(
            config=config.redundancy_config,
            api_key=api_key,
            project_root=self._project_root,
        )
        self._evaluator = EvaluatorAgent(
            config=config.evaluator_config,
            api_key=api_key,
            project_root=self._project_root,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, gdd_path: Path) -> MASPipelineResult:
        """
        Execute the full MAS pipeline for one GDD document.

        Parameters
        ----------
        gdd_path:
            Absolute path to the GDD file (PDF or text).

        Returns
        -------
        MASPipelineResult
            Terminal result containing all produced artifacts.
        """
        run_id = self._new_run_id()
        document_id = gdd_path.stem

        store = ResultStore(run_id, self._config.output_dir)

        with RunLogger(
            run_id=run_id,
            logs_dir=self._config.logs_dir,
            echo_stdout=True,
        ) as run_log:
            return self._execute(run_id, document_id, gdd_path, store, run_log)

    # ══════════════════════════════════════════════════════════════════════════
    # Internal execution
    # ══════════════════════════════════════════════════════════════════════════

    def _execute(
        self,
        run_id: str,
        document_id: str,
        gdd_path: Path,
        store: ResultStore,
        run_log: RunLogger,
    ) -> MASPipelineResult:
        """Main execution flow — called within RunLogger context."""

        # ── Bootstrap ExperimentRun ────────────────────────────────────────────
        config_ref = self._snapshot_config(store)
        experiment_run = ExperimentRun(
            run_id=run_id,
            document_id=document_id,
            pipeline_type="MAS",
            config_snapshot_ref=str(config_ref),
            started_at=datetime.now(timezone.utc),
            status="running",
        )
        state = MASPipelineState(
            run_id=run_id,
            experiment_run=experiment_run,
        )
        store.save_experiment_run(experiment_run)
        run_log.info("orchestration", f"Run {run_id} started — doc={document_id}")

        try:
            # ── Stage 1: Preprocessing ─────────────────────────────────────────
            chunks = self._stage_preprocess(gdd_path, state, store, run_log)
            if chunks is None:
                # Fatal preprocessing failure — run halted
                return self._finalize_halted(state, store, run_log)

            # Build chunk text map for Evaluator (needs all chunk texts)
            gdd_chunk_texts: Dict[str, str] = {
                c.chunk_id: c.text for c in chunks
            }

            # ── Stages 2–4: Per-chunk Reader → Analyst → Generator/Reviewer ───
            id_offset = 0
            for ci, chunk in enumerate(chunks):
                run_log.info(
                    "orchestration",
                    f"Chunk [{ci+1}/{len(chunks)}] id={chunk.chunk_id} "
                    f"tokens={chunk.token_count}",
                )
                id_offset = self._process_chunk(
                    chunk, ci, id_offset, len(chunks), state, store, run_log
                )

            run_log.info(
                "orchestration",
                f"Chunk processing complete: "
                f"{state.total_valid} valid stories, "
                f"{state.total_rejected} rejected, "
                f"{state.total_errors} errors so far",
            )

            # ── Stage 5: Redundancy Checker ────────────────────────────────────
            redundancy_analysis = self._stage_redundancy(
                state, store, run_log
            )

            # ── Build FinalUserStory collection ────────────────────────────────
            final_stories = self._build_final_stories(
                state, redundancy_analysis, run_id, document_id, store, run_log
            )

            # ── Stage 6: Evaluator ─────────────────────────────────────────────
            evaluation_result = self._stage_evaluator(
                final_stories, gdd_chunk_texts, redundancy_analysis,
                run_id, document_id, len(chunks), state, store, run_log,
            )

            # ── Finalize ExperimentRun ─────────────────────────────────────────
            experiment_run.status = "completed"
            experiment_run.completed_at = datetime.now(timezone.utc)
            experiment_run.final_user_story_ids = [s.id for s in final_stories]
            if evaluation_result:
                experiment_run.evaluation_result_ref = evaluation_result.evaluation_id
            store.save_experiment_run(experiment_run)

            run_log.info(
                "orchestration",
                f"Run {run_id} COMPLETED — "
                f"stories={len(final_stories)}, "
                f"errors={len(state.error_log)}, "
                f"AQUSA={evaluation_result.aqusa_score if evaluation_result else 'n/a'}%",
            )

            state.mark_stage_complete(STAGE_RESULT_STORAGE)

        except MASPipelineError:
            return self._finalize_halted(state, store, run_log)

        except Exception as exc:  # noqa: BLE001
            run_log.error(
                "orchestration",
                f"Unexpected error in run {run_id}: {exc}",
            )
            experiment_run.status = "failed"
            experiment_run.completed_at = datetime.now(timezone.utc)
            store.save_experiment_run(experiment_run)
            raise

        return MASPipelineResult(
            experiment_run=experiment_run,
            final_stories=final_stories,
            redundancy_analysis=redundancy_analysis,
            evaluation_result=evaluation_result,
            error_log=state.error_log,
            output_dir=store.run_dir_path,
            succeeded=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: Preprocessing
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_preprocess(
        self,
        gdd_path: Path,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> Optional[List[GDDChunk]]:
        """Returns chunk list, or None on fatal failure."""
        run_log.info(STAGE_PREPROCESSING, f"Preprocessing: {gdd_path.name}")
        try:
            if self._config.preprocessing_config is None:
                raise MASPipelineError("preprocessing_config is required.")
            pre_pipeline = PreprocessingPipeline(
                config=self._config.preprocessing_config
            )
            pre_result = pre_pipeline.run(gdd_path)
            chunks = pre_result.chunks

            if not chunks:
                msg = f"Preprocessing yielded 0 chunks for {gdd_path.name}."
                run_log.warning(STAGE_PREPROCESSING, msg)
                if self._config.abort_on_empty_chunks:
                    err = self._make_error(
                        state.run_id, state.experiment_run.document_id,
                        STAGE_PREPROCESSING, "other", None,
                        msg, None, "run_halted",
                    )
                    state.add_error(err)
                    store.append_error_log(err)
                    run_log.log_error(err)
                    return None

            state.chunks = chunks
            state.mark_stage_complete(STAGE_PREPROCESSING)

            if self._config.save_intermediate:
                store.save_chunks(chunks)

            run_log.info(
                STAGE_PREPROCESSING,
                f"Produced {len(chunks)} chunks from {gdd_path.name}.",
            )
            return chunks

        except MASPipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            msg = f"Preprocessing failed for {gdd_path}: {exc}"
            run_log.error(STAGE_PREPROCESSING, msg)
            err = self._make_error(
                state.run_id, state.experiment_run.document_id,
                STAGE_PREPROCESSING, "other", None, msg, None, "run_halted",
            )
            state.add_error(err)
            store.append_error_log(err)
            run_log.log_error(err)
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # Per-chunk processing (Stages 2–4)
    # ══════════════════════════════════════════════════════════════════════════

    def _process_chunk(
        self,
        chunk: GDDChunk,
        chunk_index: int,
        id_offset: int,
        total_chunks: int,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> int:
        """
        Run Reader → Analyst → Generator/Reviewer for one chunk.
        Returns the updated candidate id_offset.
        """
        # ── Stage 2: Reader ────────────────────────────────────────────────────
        reader_output = self._stage_reader(
            chunk, total_chunks, state, store, run_log
        )
        if reader_output is None:
            return id_offset  # chunk skipped

        if self._config.abort_on_empty_reader and reader_output.is_empty:
            run_log.warning(
                STAGE_READER,
                f"Chunk {chunk.chunk_id} produced 0 evidence items — skipping.",
            )
            return id_offset

        # ── Stage 3: Analyst ───────────────────────────────────────────────────
        analyst_output = self._stage_analyst(
            reader_output, chunk, id_offset, total_chunks, state, store, run_log
        )
        if analyst_output is None:
            return id_offset  # chunk skipped

        new_offset = id_offset + analyst_output.candidate_count

        if analyst_output.is_empty:
            run_log.info(
                STAGE_ANALYST,
                f"Chunk {chunk.chunk_id} produced 0 candidates — nothing to generate.",
            )
            return new_offset

        # ── Stage 4: Generator ↔ Reviewer (per candidate) ─────────────────────
        for candidate in analyst_output.candidates:
            self._stage_generator_reviewer_loop(
                candidate, chunk, state, store, run_log
            )

        return new_offset

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Reader Agent
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_reader(
        self,
        chunk: GDDChunk,
        total_chunks: int,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ):
        """Returns ReaderOutput or None on failure."""
        run_log.debug(STAGE_READER, f"Reading chunk {chunk.chunk_id}")
        try:
            reader_output = self._reader.process_chunk(
                chunk, total_chunks=total_chunks
            )
        except Exception as exc:  # noqa: BLE001
            self._log_chunk_error(
                state, store, run_log, chunk.chunk_id,
                STAGE_READER, exc,
                "excluded_and_logged",
            )
            return None

        # Schema validation gate
        val_report = validate_reader_output(reader_output, chunk)
        if not val_report.passed:
            msg = (
                f"Reader output validation FAILED for chunk {chunk.chunk_id}: "
                f"{val_report.errors}"
            )
            run_log.warning(STAGE_READER, msg)
            err = self._make_error(
                state.run_id, state.experiment_run.document_id,
                STAGE_READER, "schema_violation", chunk.chunk_id,
                msg, None, "excluded_and_logged",
            )
            state.add_error(err)
            store.append_error_log(err)
            run_log.log_error(err)
            return None

        state.reader_outputs.append(reader_output)
        if self._config.save_intermediate:
            store.save_reader_output(chunk.chunk_id, reader_output)

        run_log.debug(
            STAGE_READER,
            f"Chunk {chunk.chunk_id}: {reader_output.total_items} evidence items.",
        )
        return reader_output

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Analyst Agent
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_analyst(
        self,
        reader_output,
        chunk: GDDChunk,
        id_offset: int,
        total_chunks: int,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ):
        """Returns AnalystOutput or None on failure."""
        run_log.debug(STAGE_ANALYST, f"Analysing chunk {chunk.chunk_id}")
        try:
            analyst_output = self._analyst.process_reader_output(
                reader_output,
                id_offset=id_offset,
                total_chunks=total_chunks,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_chunk_error(
                state, store, run_log, chunk.chunk_id,
                STAGE_ANALYST, exc,
                "excluded_and_logged",
            )
            return None

        val_report = validate_analyst_output(analyst_output)
        if not val_report.passed:
            msg = (
                f"Analyst output validation FAILED for chunk {chunk.chunk_id}: "
                f"{val_report.errors}"
            )
            run_log.warning(STAGE_ANALYST, msg)
            err = self._make_error(
                state.run_id, state.experiment_run.document_id,
                STAGE_ANALYST, "schema_violation", chunk.chunk_id,
                msg, None, "excluded_and_logged",
            )
            state.add_error(err)
            store.append_error_log(err)
            run_log.log_error(err)
            return None

        state.analyst_outputs.append(analyst_output)
        if self._config.save_intermediate:
            store.save_analyst_output(chunk.chunk_id, analyst_output)

        run_log.debug(
            STAGE_ANALYST,
            f"Chunk {chunk.chunk_id}: {analyst_output.candidate_count} candidates.",
        )
        return analyst_output

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Generator ↔ Reviewer feedback loop
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_generator_reviewer_loop(
        self,
        candidate: CandidateRequirement,
        chunk: GDDChunk,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> None:
        """
        Run the Generator↔Reviewer loop for one candidate.
        Mutates ``state.valid_generator_stories`` and ``state.candidate_traces``.
        """
        trace = CandidateTrace(
            candidate_id=candidate.candidate_id,
            chunk_id=chunk.chunk_id,
        )
        state.candidate_traces.append(trace)

        feedback: Optional[ReviewerFeedback] = None
        iteration = 0
        max_iter = self._config.max_reviewer_iterations

        while iteration < max_iter:
            # ── Generate ───────────────────────────────────────────────────────
            try:
                draft = self._generator.generate(
                    candidate,
                    iteration_count=iteration,
                    reviewer_feedback=feedback,
                )
            except Exception as exc:  # noqa: BLE001
                self._log_candidate_error(
                    state, store, run_log,
                    candidate.candidate_id, chunk.chunk_id,
                    STAGE_GENERATOR, exc, "excluded_and_logged",
                )
                trace.final_status = "error"
                return

            # Generator schema validation
            gen_report = validate_generator_output(draft, candidate)
            if not gen_report.passed:
                msg = (
                    f"Generator output invalid for candidate "
                    f"{candidate.candidate_id} iter={iteration}: {gen_report.errors}"
                )
                run_log.warning(STAGE_GENERATOR, msg)
                err = self._make_error(
                    state.run_id, state.experiment_run.document_id,
                    STAGE_GENERATOR, "schema_violation",
                    candidate.candidate_id, msg, iteration, "excluded_and_logged",
                )
                state.add_error(err)
                store.append_error_log(err)
                run_log.log_error(err)
                trace.final_status = "error"
                return

            # ── Review ─────────────────────────────────────────────────────────
            try:
                result, feedback = self._reviewer.review(draft, chunk.text)
            except Exception as exc:  # noqa: BLE001
                self._log_candidate_error(
                    state, store, run_log,
                    candidate.candidate_id, chunk.chunk_id,
                    STAGE_REVIEWER, exc, "excluded_and_logged",
                )
                trace.final_status = "error"
                return

            # Reviewer schema validation
            rev_report = validate_reviewer_output(result, draft)
            if not rev_report.passed:
                run_log.warning(
                    STAGE_REVIEWER,
                    f"Reviewer output invalid for {draft.draft_id}: {rev_report.errors}",
                )

            # Record iteration
            trace.iterations.append(
                StoryIteration(
                    iteration=iteration,
                    draft=draft,
                    review_status=result.status,
                    failing_criteria=result.failing_criteria or [],
                    feedback_text=feedback.feedback_text if feedback else "",
                )
            )

            if self._config.save_intermediate:
                store.save_draft_iteration(draft, result.status)

            run_log.debug(
                STAGE_REVIEWER,
                f"{draft.draft_id} iter={iteration}: {result.status} "
                f"failing={result.failing_criteria}",
            )

            if result.status == "Valid":
                # ── Story passed review ────────────────────────────────────────
                state.valid_generator_stories.append(draft)
                trace.final_status = "valid"
                trace.final_story = draft
                run_log.info(
                    STAGE_REVIEWER,
                    f"✓ Candidate {candidate.candidate_id} → "
                    f"Valid on iteration {iteration}.",
                )
                return

            # ── Invalid — try next iteration ───────────────────────────────────
            iteration += 1

        # ── Max iterations exhausted ───────────────────────────────────────────
        msg = (
            f"Candidate {candidate.candidate_id} reached max_reviewer_iterations "
            f"({max_iter}) without passing review. "
            f"Last failing: {result.failing_criteria}."
        )
        run_log.warning(STAGE_REVIEWER, msg)
        err = self._make_error(
            state.run_id, state.experiment_run.document_id,
            STAGE_REVIEWER, "max_iterations_reached_reviewer",
            candidate.candidate_id, msg, max_iter - 1, "excluded_and_logged",
        )
        state.add_error(err)
        store.append_error_log(err)
        run_log.log_error(err)
        trace.final_status = "rejected"

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Redundancy Checker
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_redundancy(
        self,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> Optional[RedundancyAnalysis]:
        """Returns RedundancyAnalysis or None if skipped/failed."""
        valid_stories = state.valid_generator_stories
        if not valid_stories:
            run_log.warning(
                STAGE_REDUNDANCY,
                "No valid stories to check for redundancy — skipping.",
            )
            return None

        run_log.info(
            STAGE_REDUNDANCY,
            f"Redundancy check on {len(valid_stories)} stories…",
        )
        try:
            ra = self._redundancy.check(valid_stories, state.run_id)
        except Exception as exc:  # noqa: BLE001
            msg = f"Redundancy Checker failed: {exc}"
            run_log.error(STAGE_REDUNDANCY, msg)
            err = self._make_error(
                state.run_id, state.experiment_run.document_id,
                STAGE_REDUNDANCY, "other", None, msg, None, "flagged_unresolved",
            )
            state.add_error(err)
            store.append_error_log(err)
            run_log.log_error(err)
            return None

        val_report = validate_redundancy_output(ra, valid_stories)
        if not val_report.passed:
            run_log.warning(
                STAGE_REDUNDANCY,
                f"Redundancy output validation FAILED: {val_report.errors}",
            )

        state.redundancy_analysis = ra
        state.mark_stage_complete(STAGE_REDUNDANCY)
        store.save_redundancy_analysis(ra)

        run_log.info(
            STAGE_REDUNDANCY,
            f"Redundancy complete: status={ra.resolution_status}, "
            f"groups={len(ra.duplicate_groups)}, "
            f"resolved={ra.resolved_story_count}",
        )
        return ra

    # ══════════════════════════════════════════════════════════════════════════
    # Build FinalUserStory collection
    # ══════════════════════════════════════════════════════════════════════════

    def _build_final_stories(
        self,
        state: MASPipelineState,
        redundancy_analysis: Optional[RedundancyAnalysis],
        run_id: str,
        document_id: str,
        store: ResultStore,
        run_log: RunLogger,
    ) -> List[FinalUserStory]:
        """
        Convert GeneratedUserStory objects to FinalUserStory, applying
        redundancy resolution decisions (merge/remove) from RedundancyAnalysis.
        """
        run_log.info("orchestration", "Building FinalUserStory collection…")

        # Map draft_id → GeneratedUserStory for O(1) lookup
        draft_map: Dict[str, GeneratedUserStory] = {
            s.draft_id: s for s in state.valid_generator_stories
        }

        # Map draft_id → reviewer_traceability from traces
        traceability_map: Dict[str, Optional[bool]] = {}
        for trace in state.candidate_traces:
            for it in trace.iterations:
                if it.review_status == "Valid":
                    # Extract gdd_traceability verdict from trace
                    # We use the final iteration's review as the signal
                    traceability_map[it.draft.draft_id] = True
                elif it.review_status == "Invalid":
                    # Check if gdd_traceability specifically failed
                    if "source_relevance" in (it.failing_criteria or []) or \
                       "no_unsupported_information" in (it.failing_criteria or []):
                        traceability_map[it.draft.draft_id] = False

        # Determine which story IDs to include and with what status
        # Start: all valid stories are "Valid" by default
        status_map: Dict[str, str] = {
            s.draft_id: "Valid" for s in state.valid_generator_stories
        }
        # Merged/removed IDs from redundancy analysis
        merged_pairs: Dict[str, str] = {}   # removed_id → merged_into_id
        merged_stories_new: List[FinalUserStory] = []

        if redundancy_analysis:
            for action in redundancy_analysis.merge_actions:
                if action.action == "merge":
                    # Mark all removed stories as Merged
                    for removed_id in action.removed_story_ids:
                        status_map[removed_id] = "Merged"
                    # The kept/merged story is also marked Merged
                    status_map[action.kept_story_id] = "Merged"
                elif action.action == "remove_inferior":
                    for removed_id in action.removed_story_ids:
                        status_map[removed_id] = "Merged"  # consumed
                elif action.action == "keep":
                    pass  # stays "Valid"

        # Build FinalUserStory for each valid story
        final_stories: List[FinalUserStory] = []
        counter = 1

        for draft in state.valid_generator_stories:
            v_status = status_map.get(draft.draft_id, "Valid")
            traceability = traceability_map.get(draft.draft_id)

            confidence = ConfidenceEvidence(
                reviewer_traceability_confirmed=traceability,
                source_excerpt=None,
            )

            try:
                fs = FinalUserStory(
                    id=f"US-{counter:04d}",
                    run_id=run_id,
                    pipeline_type="MAS",
                    source_document_id=document_id,
                    source_chunk_ids=[draft.source_chunk_id],
                    role=draft.role,
                    action=draft.action,
                    benefit=draft.benefit,
                    full_text=draft.full_text,
                    requirement_type=self._infer_requirement_type(draft),
                    game_domain=self._infer_game_domain(draft),
                    validation_status=v_status,
                    confidence_evidence=confidence,
                    iteration_count_final=draft.iteration_count,
                )
                final_stories.append(fs)
                counter += 1
            except Exception as exc:  # noqa: BLE001
                run_log.warning(
                    "orchestration",
                    f"Could not build FinalUserStory for {draft.draft_id}: {exc}",
                )

        # Add Rejected-ManualReview stories (candidates that hit max iterations)
        # Use the last known draft from the trace
        for trace in state.candidate_traces:
            if trace.final_status == "rejected" and trace.iterations:
                last_draft = trace.iterations[-1].draft
                try:
                    fs = FinalUserStory(
                        id=f"US-{counter:04d}",
                        run_id=run_id,
                        pipeline_type="MAS",
                        source_document_id=document_id,
                        source_chunk_ids=[last_draft.source_chunk_id],
                        role=last_draft.role,
                        action=last_draft.action,
                        benefit=last_draft.benefit,
                        full_text=last_draft.full_text,
                        requirement_type=self._infer_requirement_type(last_draft),
                        game_domain=self._infer_game_domain(last_draft),
                        validation_status="Rejected-ManualReview",
                        confidence_evidence=ConfidenceEvidence(
                            reviewer_traceability_confirmed=False
                        ),
                        iteration_count_final=last_draft.iteration_count,
                    )
                    final_stories.append(fs)
                    counter += 1
                except Exception as exc:  # noqa: BLE001
                    run_log.warning(
                        "orchestration",
                        f"Could not build Rejected FinalUserStory "
                        f"for {trace.candidate_id}: {exc}",
                    )

        state.final_stories = final_stories
        store.save_final_stories(final_stories)

        run_log.info(
            "orchestration",
            f"FinalUserStory collection: {len(final_stories)} stories "
            f"(valid={sum(1 for s in final_stories if s.validation_status == 'Valid')}, "
            f"merged={sum(1 for s in final_stories if s.validation_status == 'Merged')}, "
            f"rejected={sum(1 for s in final_stories if s.validation_status == 'Rejected-ManualReview')})",
        )
        return final_stories

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 6: Evaluator
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_evaluator(
        self,
        final_stories: List[FinalUserStory],
        gdd_chunk_texts: Dict[str, str],
        redundancy_analysis: Optional[RedundancyAnalysis],
        run_id: str,
        document_id: str,
        total_chunk_count: int,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> Optional[EvaluationResult]:
        """Returns EvaluationResult or None on failure."""
        run_log.info(
            STAGE_EVALUATOR,
            f"Evaluating {len(final_stories)} final stories…",
        )
        try:
            er = self._evaluator.evaluate(
                stories=final_stories,
                gdd_chunks=gdd_chunk_texts,
                run_id=run_id,
                document_id=document_id,
                pipeline_type="MAS",
                total_chunk_count=total_chunk_count,
                redundancy_analysis=redundancy_analysis,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Evaluator failed: {exc}"
            run_log.error(STAGE_EVALUATOR, msg)
            err = self._make_error(
                state.run_id, document_id,
                STAGE_EVALUATOR, "other", None, msg, None, "excluded_and_logged",
            )
            state.add_error(err)
            store.append_error_log(err)
            run_log.log_error(err)
            return None

        val_report = validate_evaluator_output(er, final_stories)
        if not val_report.passed:
            run_log.warning(
                STAGE_EVALUATOR,
                f"Evaluator output validation warnings: {val_report.errors}",
            )

        state.evaluation_result = er
        state.mark_stage_complete(STAGE_EVALUATOR)
        store.save_evaluation_result(er)

        run_log.info(
            STAGE_EVALUATOR,
            f"Evaluation complete: "
            f"AQUSA={er.aqusa_score:.1f}% "
            f"Coverage={er.coverage:.1f}% "
            f"Diversity={er.diversity:.3f} "
            f"Redundancy={er.redundancy_rate:.1f}% "
            f"Hallucination={er.hallucination_rate:.1f}%"
            if all(
                v is not None for v in [
                    er.aqusa_score, er.coverage, er.diversity,
                    er.redundancy_rate, er.hallucination_rate,
                ]
            )
            else "Evaluation complete (some metrics null — see null_reasons).",
        )
        return er

    # ══════════════════════════════════════════════════════════════════════════
    # Error helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _log_chunk_error(
        self,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
        chunk_id: str,
        stage: str,
        exc: Exception,
        resolution: str,
    ) -> None:
        msg = f"Stage {stage} failed for chunk {chunk_id}: {exc}"
        run_log.error(stage, msg)
        error_type = (
            "llm_api_failure"
            if "api" in str(exc).lower() or "rate" in str(exc).lower()
            else "malformed_output"
            if isinstance(exc, LLMMalformedOutputError)
            else "other"
        )
        err = self._make_error(
            state.run_id, state.experiment_run.document_id,
            stage, error_type, chunk_id, msg, None, resolution,
        )
        state.add_error(err)
        store.append_error_log(err)
        run_log.log_error(err)

    def _log_candidate_error(
        self,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
        candidate_id: str,
        chunk_id: str,
        stage: str,
        exc: Exception,
        resolution: str,
    ) -> None:
        msg = (
            f"Stage {stage} failed for candidate {candidate_id} "
            f"(chunk {chunk_id}): {exc}"
        )
        run_log.error(stage, msg)
        error_type = (
            "llm_api_failure"
            if "api" in str(exc).lower() or "rate" in str(exc).lower()
            else "malformed_output"
            if isinstance(exc, LLMMalformedOutputError)
            else "other"
        )
        err = self._make_error(
            state.run_id, state.experiment_run.document_id,
            stage, error_type, candidate_id, msg, None, resolution,
        )
        state.add_error(err)
        store.append_error_log(err)
        run_log.log_error(err)

    @staticmethod
    def _make_error(
        run_id: str,
        document_id: str,
        stage: str,
        error_type: str,
        related_id: Optional[str],
        message: str,
        retry_count: Optional[int],
        resolution: str,
    ) -> ErrorFailureLog:
        return ErrorFailureLog(
            error_id=str(uuid.uuid4()),
            run_id=run_id,
            document_id=document_id,
            stage=stage,
            error_type=error_type,
            related_entity_id=related_id,
            message=message,
            retry_count=retry_count,
            resolution=resolution,
            timestamp=datetime.now(timezone.utc),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Finalize helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _finalize_halted(
        self,
        state: MASPipelineState,
        store: ResultStore,
        run_log: RunLogger,
    ) -> MASPipelineResult:
        """Mark the run as failed and return an empty result."""
        run_log.error(
            "orchestration",
            f"Run {state.run_id} HALTED due to fatal error.",
        )
        state.experiment_run.status = "failed"
        state.experiment_run.completed_at = datetime.now(timezone.utc)
        store.save_experiment_run(state.experiment_run)
        return MASPipelineResult(
            experiment_run=state.experiment_run,
            final_stories=[],
            error_log=state.error_log,
            output_dir=store.run_dir_path,
            succeeded=False,
        )

    def _snapshot_config(self, store: ResultStore) -> Path:
        """Write a frozen copy of mas.yaml to the run output directory."""
        mas_yaml_path = self._project_root / "config" / "mas.yaml"
        try:
            text = mas_yaml_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = "# config/mas.yaml not found at snapshot time\n"
        return store.save_config_snapshot(text)

    # ══════════════════════════════════════════════════════════════════════════
    # Inference helpers (classification fields for FinalUserStory)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _infer_requirement_type(draft: GeneratedUserStory) -> str:
        """
        Best-effort requirement_type from the draft's role text.
        Falls back to 'player' if unrecognised.
        """
        role_lower = draft.role.lower()
        if "player" in role_lower or "gamer" in role_lower or "user" in role_lower:
            return "player"
        if "system" in role_lower or "engine" in role_lower or "server" in role_lower:
            return "system"
        if "developer" in role_lower or "dev" in role_lower or "designer" in role_lower:
            return "dev team"
        if "entity" in role_lower or "npc" in role_lower or "enemy" in role_lower \
                or "character" in role_lower:
            return "in-game entity"
        return "player"  # safe default

    @staticmethod
    def _infer_game_domain(draft: GeneratedUserStory) -> str:
        """
        Best-effort game_domain from the draft's action + benefit text.
        Falls back to 'gameplay' if unrecognised.
        """
        text_lower = (draft.action + " " + draft.benefit).lower()
        if any(w in text_lower for w in ["ui", "interface", "menu", "hud", "screen", "button"]):
            return "UI"
        if any(w in text_lower for w in ["story", "narrative", "dialogue", "lore", "cutscene"]):
            return "narrative"
        if any(w in text_lower for w in ["audio", "sound", "music", "sfx"]):
            return "audio"
        if any(w in text_lower for w in ["level", "map", "environment", "dungeon", "zone"]):
            return "level design"
        if any(w in text_lower for w in ["system", "engine", "database", "save", "load", "config"]):
            return "systems"
        return "gameplay"  # safe default

    @staticmethod
    def _new_run_id() -> str:
        return str(uuid.uuid4())
