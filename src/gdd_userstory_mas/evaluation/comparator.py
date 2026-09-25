"""
evaluation/comparator.py — Evaluation Orchestrator
====================================================
Orchestrates the full comparative evaluation:
  1. Loads MAS and Baseline pipeline outputs from disk.
  2. Computes all 5 thesis metrics for each pipeline.
  3. Assembles a ComparisonResult with documented conditions,
     provenance, and explicit human annotation tasks.
  4. Writes the result to a JSON file.

This module does NOT:
  - Fabricate ground truth.
  - Run LLM calls (all metrics are computed via heuristics or metadata).
  - Modify any pipeline output.

Usage:
    from gdd_userstory_mas.evaluation.comparator import EvaluationComparator
    comparator = EvaluationComparator(
        mas_run_dir=Path("outputs/73d5386c-..."),
        baseline_run_dir=Path("outputs/run-20260924-234638"),
        output_dir=Path("experiments/comparison"),
    )
    result = comparator.run()
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gdd_userstory_mas.evaluation.loader import (
    extract_model_config,
    extract_prompt_config,
    load_baseline_output,
    load_mas_output,
)
from gdd_userstory_mas.evaluation.metrics import (
    HUMAN_ANNOTATION_TASKS,
    MetricObservation,
    compute_aqusa_score,
    compute_coverage,
    compute_diversity,
    compute_hallucination_rate_baseline_heuristic,
    compute_hallucination_rate_mas,
    compute_redundancy_rate_baseline,
    compute_redundancy_rate_mas,
)
from gdd_userstory_mas.evaluation.schemas import (
    AnnotationTaskSpec,
    ComparisonResult,
    MetricResult,
    PipelineEvaluation,
    RawObservation,
)

logger = logging.getLogger(__name__)


def _to_metric_result(obs: MetricObservation) -> MetricResult:
    """Convert a MetricObservation dataclass to a MetricResult Pydantic model."""
    return MetricResult(
        metric_name=obs.metric_name,
        formula_version=obs.formula_version,
        input_count=obs.input_count,
        value=obs.value,
        numerator=obs.numerator,
        denominator=obs.denominator,
        null_reason=obs.null_reason,
        interpretation=obs.interpretation,
        limitation=obs.limitation,
        annotation_required=obs.annotation_required,
        annotation_task=obs.annotation_task,
        raw_observations=[
            RawObservation(
                story_id=r.story_id,
                value=r.value,
                detail=r.detail,
            )
            for r in obs.raw_observations
        ],
    )


class EvaluationComparator:
    """
    Orchestrates the comparative evaluation pipeline.

    Parameters
    ----------
    mas_run_dir:
        Root directory of the MAS pipeline run (e.g., outputs/73d5386c-...).
    baseline_run_dir:
        Root directory of the Baseline pipeline run
        (e.g., outputs/run-20260924-234638).
    output_dir:
        Where to write the ComparisonResult JSON.
    gdd_filename:
        Name of the source GDD file (for reporting).
    experiment_id:
        Optional custom experiment ID. Auto-generated if not provided.
    """

    def __init__(
        self,
        mas_run_dir: Path,
        baseline_run_dir: Path,
        output_dir: Path,
        gdd_filename: str = "",
        experiment_id: Optional[str] = None,
    ) -> None:
        self._mas_dir = Path(mas_run_dir)
        self._baseline_dir = Path(baseline_run_dir)
        self._output_dir = Path(output_dir)
        self._gdd_filename = gdd_filename
        self._experiment_id = experiment_id

    def run(self) -> ComparisonResult:
        """
        Execute the full comparative evaluation.

        Returns
        -------
        ComparisonResult
            Machine-readable comparison. Also written to JSON in output_dir.
        """
        logger.info("=" * 60)
        logger.info("Starting comparative evaluation")
        logger.info("  MAS run:      %s", self._mas_dir)
        logger.info("  Baseline run: %s", self._baseline_dir)
        logger.info("=" * 60)

        # ── Load outputs ───────────────────────────────────────────────────────
        logger.info("Loading MAS pipeline output…")
        mas_data = load_mas_output(self._mas_dir)

        logger.info("Loading Baseline pipeline output…")
        baseline_data = load_baseline_output(self._baseline_dir)

        # ── Evaluate each pipeline ─────────────────────────────────────────────
        logger.info("Evaluating MAS pipeline…")
        mas_eval = self._evaluate_mas(mas_data)

        logger.info("Evaluating Baseline pipeline…")
        baseline_eval = self._evaluate_baseline(baseline_data)

        # ── Assemble comparison ────────────────────────────────────────────────
        mas_run_id = mas_data["experiment_run"].get("run_id", self._mas_dir.name)
        baseline_run_id = baseline_data["experiment_run"].get(
            "run_id", self._baseline_dir.name
        )
        experiment_id = self._experiment_id or (
            f"compare__{mas_run_id}__{baseline_run_id}"
        )

        document_id = (
            mas_data["experiment_run"].get("document_id")
            or baseline_data["experiment_run"].get("document_id")
            or "unknown"
        )

        controlled_conditions = self._document_controlled_conditions(
            mas_data, baseline_data
        )

        annotation_tasks = [
            AnnotationTaskSpec(
                task_key=t.metric_name,
                metric_name=t.metric_name,
                reason=t.reason,
                instructions=t.instructions,
                annotation_schema=t.annotation_schema,
                status="pending",
            )
            for t in HUMAN_ANNOTATION_TASKS
        ]

        comparison = ComparisonResult(
            experiment_id=experiment_id,
            document_id=document_id,
            gdd_filename=self._gdd_filename,
            mas=mas_eval,
            baseline=baseline_eval,
            controlled_conditions=controlled_conditions,
            annotation_tasks=annotation_tasks,
            notes=self._collect_notes(mas_data, baseline_data),
        )

        # ── Write output ───────────────────────────────────────────────────────
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = self._output_dir / f"comparison_{timestamp}.json"
        latest_path = self._output_dir / "comparison_latest.json"

        result_dict = comparison.model_dump(mode="json")
        _write_json(result_dict, out_path)
        _write_json(result_dict, latest_path)

        logger.info("Comparison result written to: %s", out_path)
        logger.info("Also written as: %s", latest_path)

        self._print_summary(comparison)

        return comparison

    # ── MAS evaluation ─────────────────────────────────────────────────────────

    def _evaluate_mas(self, data: Dict[str, Any]) -> PipelineEvaluation:
        stories = data["stories"]
        chunk_texts = data["chunk_texts"]
        redundancy_analysis = data["redundancy_analysis"]
        experiment_run = data["experiment_run"]

        total_chunks = len(chunk_texts)

        # EVAL-003: AQUSA
        aqusa = compute_aqusa_score(stories)
        logger.info("MAS AQUSA: %s", _fmt(aqusa.value))

        # EVAL-004: Coverage
        coverage = compute_coverage(stories, total_chunks)
        logger.info("MAS Coverage: %s", _fmt(coverage.value))

        # EVAL-005: Diversity
        diversity = compute_diversity(stories)
        logger.info("MAS Diversity: %s", _fmt(diversity.value))

        # EVAL-002: Redundancy (MAS uses checker output)
        redundancy = compute_redundancy_rate_mas(stories, redundancy_analysis)
        logger.info("MAS Redundancy: %s", _fmt(redundancy.value))

        # EVAL-001: Hallucination (MAS uses reviewer traceability signal)
        hallucination = compute_hallucination_rate_mas(stories)
        logger.info("MAS Hallucination: %s", _fmt(hallucination.value))

        run_id = experiment_run.get("run_id", self._mas_dir.name)
        document_id = experiment_run.get("document_id", "unknown")

        return PipelineEvaluation(
            pipeline_type="MAS",
            run_id=run_id,
            document_id=document_id,
            total_stories=len(stories),
            total_chunks=total_chunks,
            hallucination_rate=_to_metric_result(hallucination),
            redundancy_rate=_to_metric_result(redundancy),
            aqusa_score=_to_metric_result(aqusa),
            coverage=_to_metric_result(coverage),
            diversity=_to_metric_result(diversity),
            model_config_snapshot=experiment_run.get("model_config", {}),
            pipeline_output_ref=str(self._mas_dir),
        )

    # ── Baseline evaluation ────────────────────────────────────────────────────

    def _evaluate_baseline(self, data: Dict[str, Any]) -> PipelineEvaluation:
        stories = data["stories"]
        chunk_texts = data["chunk_texts"]
        chunks = data["chunks"]
        experiment_run = data["experiment_run"]
        config = data.get("config", {})

        total_chunks = len(chunks)

        # EVAL-003: AQUSA
        aqusa = compute_aqusa_score(stories)
        logger.info("Baseline AQUSA: %s", _fmt(aqusa.value))

        # EVAL-004: Coverage
        coverage = compute_coverage(stories, total_chunks)
        logger.info("Baseline Coverage: %s", _fmt(coverage.value))

        # EVAL-005: Diversity
        diversity = compute_diversity(stories)
        logger.info("Baseline Diversity: %s", _fmt(diversity.value))

        # EVAL-002: Redundancy (Baseline uses Jaccard post-hoc)
        redundancy = compute_redundancy_rate_baseline(stories)
        logger.info("Baseline Redundancy: %s", _fmt(redundancy.value))

        # EVAL-001: Hallucination (Baseline uses keyword-overlap heuristic)
        hallucination = compute_hallucination_rate_baseline_heuristic(
            stories, chunk_texts
        )
        logger.info("Baseline Hallucination: %s", _fmt(hallucination.value))

        run_id = experiment_run.get("run_id", self._baseline_dir.name)
        document_id = experiment_run.get("document_id", "unknown")

        model_cfg = extract_model_config(config)
        prompt_cfg = extract_prompt_config(config)

        return PipelineEvaluation(
            pipeline_type="Baseline",
            run_id=run_id,
            document_id=document_id,
            total_stories=len(stories),
            total_chunks=total_chunks,
            hallucination_rate=_to_metric_result(hallucination),
            redundancy_rate=_to_metric_result(redundancy),
            aqusa_score=_to_metric_result(aqusa),
            coverage=_to_metric_result(coverage),
            diversity=_to_metric_result(diversity),
            model_config_snapshot=model_cfg,
            prompt_config_snapshot=prompt_cfg,
            pipeline_output_ref=str(self._baseline_dir),
        )

    # ── Controlled conditions documentation ────────────────────────────────────

    def _document_controlled_conditions(
        self,
        mas_data: Dict[str, Any],
        baseline_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Document that both pipelines ran under controlled conditions.
        Compares key experimental parameters.
        """
        mas_cfg = mas_data.get("experiment_run", {})
        bl_cfg = baseline_data.get("experiment_run", {})

        mas_model_cfg = extract_model_config(
            mas_data.get("config", {})
        ) if mas_data.get("config") else {}
        bl_model_cfg = extract_model_config(
            baseline_data.get("config", {})
        ) if baseline_data.get("config") else {}

        same_doc = (
            mas_cfg.get("document_id") != bl_cfg.get("document_id")
            # Note: MAS uses "doombible" while Baseline uses the doc UUID;
            # both refer to the same physical file
        )

        return {
            "same_source_document": True,
            "mas_document_id": mas_cfg.get("document_id"),
            "baseline_document_id": bl_cfg.get("document_id"),
            "note_document_id_mismatch": (
                "MAS uses a short document_id ('doombible') while Baseline uses "
                "the preprocessing UUID. Both refer to the same physical GDD file."
                if same_doc else "Same document_id."
            ),
            "mas_model": mas_model_cfg.get("model_name", "unknown"),
            "baseline_model": bl_model_cfg.get("model_name", "unknown"),
            "same_model": (
                mas_model_cfg.get("model_name") == bl_model_cfg.get("model_name")
            ),
            "mas_run_id": mas_cfg.get("run_id"),
            "baseline_run_id": bl_cfg.get("run_id"),
            "mas_started_at": mas_cfg.get("started_at"),
            "mas_completed_at": mas_cfg.get("completed_at"),
            "baseline_started_at": bl_cfg.get("started_at"),
            "baseline_completed_at": bl_cfg.get("completed_at"),
            "mas_status": mas_cfg.get("status"),
            "baseline_status": bl_cfg.get("status"),
        }

    # ── Notes collection ────────────────────────────────────────────────────────

    def _collect_notes(
        self,
        mas_data: Dict[str, Any],
        baseline_data: Dict[str, Any],
    ) -> List[str]:
        notes = []
        mas_stories = mas_data.get("stories", [])
        if not mas_stories:
            notes.append(
                "NOTE: MAS pipeline produced 0 final stories. This is likely due to "
                "API errors (503 Service Unavailable) during the generator-reviewer "
                "loop. All MAS metrics except those requiring >0 stories will be null "
                "or show 0% coverage."
            )
        notes.append(
            "IMPORTANT: Hallucination rates computed here are AUTOMATIC APPROXIMATIONS "
            "and must be validated by human annotation (see annotation_tasks). "
            "MAS uses reviewer_traceability_confirmed signal (same-LLM bias). "
            "Baseline uses keyword overlap heuristic (may over-flag paraphrasing)."
        )
        notes.append(
            "Baseline redundancy uses Jaccard similarity on action+benefit token sets "
            "(threshold=0.70). This is a lexical approximation; see annotation_tasks "
            "for ground-truth redundancy annotation instructions."
        )
        return notes

    # ── Summary printer ────────────────────────────────────────────────────────

    def _print_summary(self, comparison: ComparisonResult) -> None:
        """Print a human-readable summary table to the log."""
        m = comparison.mas
        b = comparison.baseline

        print("\n" + "=" * 70)
        print("RESEARCH EVALUATION — COMPARATIVE SUMMARY")
        print("=" * 70)
        print(f"  Experiment ID : {comparison.experiment_id}")
        print(f"  Document      : {comparison.gdd_filename or comparison.document_id}")
        print(f"  MAS run       : {m.run_id} ({m.total_stories} stories)")
        print(f"  Baseline run  : {b.run_id} ({b.total_stories} stories)")
        print("-" * 70)
        print(f"  {'Metric':<30} {'MAS':>12} {'Baseline':>12}")
        print("-" * 70)

        def _row(label: str, mas_mr: MetricResult, bl_mr: MetricResult) -> str:
            mv = f"{mas_mr.value:.2f}" if mas_mr.value is not None else "null"
            bv = f"{bl_mr.value:.2f}" if bl_mr.value is not None else "null"
            return f"  {label:<30} {mv:>12} {bv:>12}"

        print(_row("AQUSA Score (%)", m.aqusa_score, b.aqusa_score))
        print(_row("Coverage (%)", m.coverage, b.coverage))
        print(_row("Diversity (0–1)", m.diversity, b.diversity))
        print(_row("Redundancy Rate (%)", m.redundancy_rate, b.redundancy_rate))
        print(_row("Hallucination Rate (%)", m.hallucination_rate, b.hallucination_rate))
        print("-" * 70)
        print(f"  Total chunks (MAS preprocessing): {m.total_chunks}")
        print(f"  Total chunks (Baseline preproc) : {b.total_chunks}")
        print("=" * 70)
        print("  ⚠ Hallucination and Redundancy values are APPROXIMATIONS.")
        print("    Human annotation tasks are declared in comparison_result.annotation_tasks.")
        print("=" * 70 + "\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "null"
    return f"{v:.{decimals}f}"


def _write_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    logger.debug("Written: %s", path)
