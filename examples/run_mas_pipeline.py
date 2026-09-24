"""
examples/run_mas_pipeline.py
==============================
End-to-end MAS pipeline demonstration for one GDD document.

Usage
-----
::

    # Minimal (reads LLM_API_KEY from environment)
    python examples/run_mas_pipeline.py datasets/raw/doombible.pdf

    # With options
    python examples/run_mas_pipeline.py datasets/raw/doombible.pdf \\
        --max-chunks 3 \\
        --output-dir outputs \\
        --log-dir logs

Requirements
------------
* LLM_API_KEY environment variable (Gemini API key)
* config/mas.yaml and config/pipeline.yaml present
* A GDD PDF file at the specified path

Output
------
All artifacts written to ``outputs/{run_id}/``:
* experiment_run.json
* config_snapshot.yaml
* preprocessing/chunks.jsonl
* reader/{chunk_id}.json (if --save-intermediate)
* analyst/{chunk_id}.json (if --save-intermediate)
* generator_reviewer/{draft_id}_iter{N}.json (if --save-intermediate)
* redundancy/analysis.json
* final_stories.jsonl
* evaluation_result.json
* errors/errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── ensure project root on sys.path ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full MAS pipeline on a GDD PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "gdd_path",
        type=Path,
        help="Path to the GDD PDF (or text) file.",
    )
    parser.add_argument(
        "--mas-yaml",
        type=Path,
        default=_PROJECT_ROOT / "config" / "mas.yaml",
        help="Path to config/mas.yaml.",
    )
    parser.add_argument(
        "--pipeline-yaml",
        type=Path,
        default=_PROJECT_ROOT / "config" / "pipeline.yaml",
        help="Path to config/pipeline.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs",
        help="Root output directory for run artifacts.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_PROJECT_ROOT / "logs",
        help="Root log directory.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Process only the first N chunks (for quick testing).",
    )
    parser.add_argument(
        "--max-reviewer-iterations",
        type=int,
        default=None,
        help="Override max_reviewer_iterations from config.",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Skip writing per-chunk intermediate artifacts.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key (defaults to LLM_API_KEY env var).",
    )
    return parser.parse_args()


def _truncate_pipeline_to_n_chunks(pipeline, n: int) -> None:
    """
    Monkey-patch the preprocessing stage so only the first N chunks
    are processed (for quick demo runs).
    """
    original_execute = pipeline._execute

    def patched_execute(run_id, document_id, gdd_path, store, run_log):
        # Call original _stage_preprocess through normal execute path
        # but cap chunks afterward
        result = original_execute(run_id, document_id, gdd_path, store, run_log)
        return result

    # We patch at the preprocessing level via a wrapper
    from gdd_userstory_mas.preprocessing.pipeline import PreprocessingPipeline as _PP

    original_run = _PP.run

    def patched_run(self, path):
        pre_result = original_run(self, path)
        pre_result.chunks = pre_result.chunks[:n]
        return pre_result

    _PP.run = patched_run


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERROR: No API key found. Set LLM_API_KEY environment variable "
            "or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    gdd_path = args.gdd_path.resolve()
    if not gdd_path.exists():
        print(f"ERROR: GDD file not found: {gdd_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load config ────────────────────────────────────────────────────────────
    from gdd_userstory_mas.mas.pipeline_config import MASPipelineConfig

    print(f"Loading config from {args.mas_yaml} + {args.pipeline_yaml}…")
    config = MASPipelineConfig.from_yaml(
        mas_yaml=args.mas_yaml,
        pipeline_yaml=args.pipeline_yaml,
        project_root=_PROJECT_ROOT,
    )

    # Apply CLI overrides
    config.output_dir = args.output_dir
    config.logs_dir = args.log_dir
    if args.max_reviewer_iterations is not None:
        config.max_reviewer_iterations = args.max_reviewer_iterations
    if args.no_save_intermediate:
        config.save_intermediate = False

    # ── Create pipeline ────────────────────────────────────────────────────────
    from gdd_userstory_mas.mas.pipeline import MASPipeline

    pipeline = MASPipeline(
        config=config,
        api_key=api_key,
        project_root=_PROJECT_ROOT,
    )

    # ── Optionally limit chunk count ───────────────────────────────────────────
    if args.max_chunks is not None:
        _truncate_pipeline_to_n_chunks(pipeline, args.max_chunks)

    # ── Run ────────────────────────────────────────────────────────────────────
    print(f"\nStarting MAS pipeline run on: {gdd_path.name}")
    print(f"  max_reviewer_iterations : {config.max_reviewer_iterations}")
    print(f"  max_chunks              : {args.max_chunks or 'all'}")
    print(f"  output_dir              : {config.output_dir}")
    print("-" * 60)

    result = pipeline.run(gdd_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    er = result.evaluation_result
    if er:
        print("\n── Evaluation Metrics ─────────────────────────────────────")
        print(f"  AQUSA Score      : {er.aqusa_score:.1f}%" if er.aqusa_score is not None else "  AQUSA Score      : n/a")
        print(f"  Coverage         : {er.coverage:.1f}%" if er.coverage is not None else "  Coverage         : n/a")
        print(f"  Diversity        : {er.diversity:.3f}" if er.diversity is not None else "  Diversity        : n/a")
        print(f"  Redundancy Rate  : {er.redundancy_rate:.1f}%" if er.redundancy_rate is not None else "  Redundancy Rate  : n/a")
        print(f"  Hallucination    : {er.hallucination_rate:.1f}%" if er.hallucination_rate is not None else "  Hallucination    : n/a")

    print(f"\n── Final Stories ({result.total_stories}) ─────────────────────────────")
    for story in result.final_stories[:10]:  # show first 10
        print(
            f"  [{story.validation_status:20s}] {story.id}: "
            f"As a {story.role[:20]}, I want {story.action[:35]}…"
        )
    if result.total_stories > 10:
        print(f"  … and {result.total_stories - 10} more.")

    if result.error_log:
        print(f"\n── Errors ({result.total_errors}) ─────────────────────────────────────")
        for err in result.error_log[:5]:
            print(f"  [{err.stage}] {err.error_type}: {err.message[:80]}")
        if result.total_errors > 5:
            print(f"  … and {result.total_errors - 5} more.")

    print(f"\nArtifacts saved to: {result.output_dir}")
    print(f"Run ID: {result.run_id}")

    # Exit with non-zero if the run failed
    sys.exit(0 if result.succeeded else 1)


if __name__ == "__main__":
    main()
