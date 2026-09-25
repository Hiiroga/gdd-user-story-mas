"""
run_evaluation.py — Research Evaluation Pipeline CLI
=====================================================
Compares MAS pipeline vs. Single-Agent Baseline pipeline outputs.

Usage:
    python run_evaluation.py \\
        --mas-run-dir outputs/73d5386c-ce3c-459a-a03c-8472fe65025f \\
        --baseline-run-dir outputs/run-20260924-234638 \\
        --output-dir experiments/comparison \\
        --gdd-filename doombible.pdf

The output comparison JSON will be written to:
    experiments/comparison/comparison_<timestamp>.json
    experiments/comparison/comparison_latest.json

No LLM calls are made — all metrics are computed deterministically
from existing pipeline output files.

For ground-truth validation of hallucination and redundancy metrics,
see the annotation_tasks section in the comparison JSON output.
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# ── UTF-8 output (Windows) ─────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Evaluation Pipeline: MAS vs. Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mas-run-dir",
        type=Path,
        required=True,
        help=(
            "Path to MAS pipeline run directory "
            "(e.g., outputs/73d5386c-ce3c-459a-a03c-8472fe65025f)."
        ),
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        required=True,
        help=(
            "Path to Baseline pipeline run directory "
            "(e.g., outputs/run-20260924-234638)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/comparison"),
        help=(
            "Directory to write the comparison JSON result "
            "(default: experiments/comparison)."
        ),
    )
    parser.add_argument(
        "--gdd-filename",
        type=str,
        default="",
        help="Name of the source GDD file (for reporting only).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Optional custom experiment ID. "
            "Auto-generated from run IDs if not specified."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not args.mas_run_dir.exists():
        print(f"ERROR: MAS run directory not found: {args.mas_run_dir}")
        print(
            "       Run the MAS pipeline first: "
            "python examples/run_mas_pipeline.py datasets/raw/doombible.pdf"
        )
        sys.exit(1)

    if not args.baseline_run_dir.exists():
        print(f"ERROR: Baseline run directory not found: {args.baseline_run_dir}")
        print(
            "       Run the Baseline pipeline first: "
            "python run_baseline.py datasets/raw/doombible.pdf"
        )
        sys.exit(1)

    # ── Run evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GDD User Story Extraction — Research Evaluation Pipeline")
    print("=" * 60)
    print(f"  MAS run dir      : {args.mas_run_dir}")
    print(f"  Baseline run dir : {args.baseline_run_dir}")
    print(f"  Output dir       : {args.output_dir}")
    if args.gdd_filename:
        print(f"  GDD file         : {args.gdd_filename}")
    print("=" * 60 + "\n")

    try:
        from gdd_userstory_mas.evaluation.comparator import EvaluationComparator

        comparator = EvaluationComparator(
            mas_run_dir=args.mas_run_dir,
            baseline_run_dir=args.baseline_run_dir,
            output_dir=args.output_dir,
            gdd_filename=args.gdd_filename,
            experiment_id=args.experiment_id,
        )

        result = comparator.run()

        # ── Report annotation tasks ────────────────────────────────────────────
        if result.annotation_tasks:
            print("\nHUMAN ANNOTATION TASKS REQUIRED:")
            print("-" * 50)
            for task in result.annotation_tasks:
                print(f"  [{task.task_key}]")
                print(f"  Reason: {task.reason[:120]}...")
                print(f"  Status: {task.status}")
                print()
            print(
                "  See annotation_tasks in the comparison JSON for full instructions."
            )

        print(
            f"\n✓ Evaluation complete. Results in: {args.output_dir}/comparison_latest.json"
        )

    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Evaluation pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
