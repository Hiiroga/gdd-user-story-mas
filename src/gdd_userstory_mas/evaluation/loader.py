"""
evaluation/loader.py — Experiment Output Loader
=================================================
Loads pipeline output data from disk into Python dicts/objects
suitable for the evaluation pipeline.

Supports:
  - MAS pipeline outputs (UUID-based run dirs)
  - Baseline pipeline outputs (run-YYYYMMDD-HHMMSS dirs)
  - Preprocessing outputs (chunks.json, gdd_metadata.json)

All functions raise descriptive errors when required files are missing.
Returns plain dicts (not Pydantic models) for flexibility.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ── Baseline loader ────────────────────────────────────────────────────────────

def load_baseline_output(run_dir: Path) -> Dict[str, Any]:
    """
    Load all relevant outputs from a Baseline pipeline run directory.

    Expected structure:
        <run_dir>/
          02_baseline/
            final_user_stories.json
            experiment_run.json
            config_snapshot.yaml
            token_usage.json
            validation_report.json
          01_preprocessing/
            chunks.json
            gdd_metadata.json

    Parameters
    ----------
    run_dir:
        Root run directory (e.g., outputs/run-20260924-234638).

    Returns
    -------
    dict with keys:
        stories       — list of story dicts
        experiment_run — experiment metadata dict
        config        — config snapshot dict
        chunks        — list of chunk dicts (from preprocessing)
        gdd_metadata  — GDD metadata dict
        chunk_texts   — {chunk_id: text} mapping
        run_dir_str   — str path to run dir
    """
    run_dir = Path(run_dir)
    baseline_dir = run_dir / "02_baseline"
    preproc_dir = run_dir / "01_preprocessing"

    _require_dir(baseline_dir, "Baseline output dir")
    _require_dir(preproc_dir, "Preprocessing output dir")

    stories = _load_json(baseline_dir / "final_user_stories.json", default=[])
    experiment_run = _load_json(baseline_dir / "experiment_run.json", default={})
    token_usage = _load_json(baseline_dir / "token_usage.json", default={})
    validation_report = _load_json(baseline_dir / "validation_report.json", default={})
    config = _load_yaml(baseline_dir / "config_snapshot.yaml", default={})

    chunks = _load_json(preproc_dir / "chunks.json", default=[])
    gdd_metadata = _load_json(preproc_dir / "gdd_metadata.json", default={})

    chunk_texts = _build_chunk_texts(chunks)

    logger.info(
        "Loaded Baseline run: %s — %d stories, %d chunks",
        run_dir.name, len(stories), len(chunks),
    )

    return {
        "stories": stories,
        "experiment_run": experiment_run,
        "config": config,
        "token_usage": token_usage,
        "validation_report": validation_report,
        "chunks": chunks,
        "gdd_metadata": gdd_metadata,
        "chunk_texts": chunk_texts,
        "run_dir_str": str(run_dir),
    }


# ── MAS loader ─────────────────────────────────────────────────────────────────

def load_mas_output(
    run_dir: Path,
    preprocessing_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load all relevant outputs from a MAS pipeline run directory.

    Expected structure (UUID-based):
        <run_dir>/
          experiment_run.json
          evaluation_result.json
          config_snapshot.yaml   (optional)
          generator_reviewer/
            <chunk_id>/
              iteration_*.json
          reader/
          redundancy/
            analysis.json

    Parameters
    ----------
    run_dir:
        Root MAS run directory (e.g., outputs/73d5386c-...).
    preprocessing_dir:
        Optional path to a preprocessing dir (01_preprocessing/) to get
        chunk texts. If None, will try to find in sibling run dirs.

    Returns
    -------
    dict with keys:
        stories           — list of final story dicts (may be empty)
        experiment_run    — experiment metadata dict
        evaluation_result — existing evaluation result dict (may be empty)
        redundancy_analysis — RedundancyAnalysis dict or None
        chunk_texts       — {chunk_id: text} mapping
        run_dir_str       — str path to run dir
    """
    run_dir = Path(run_dir)
    _require_dir(run_dir, "MAS run dir")

    experiment_run = _load_json(run_dir / "experiment_run.json", default={})
    evaluation_result = _load_json(run_dir / "evaluation_result.json", default={})

    # Load redundancy analysis if present
    redundancy_dir = run_dir / "redundancy"
    redundancy_analysis = None
    if (redundancy_dir / "analysis.json").exists():
        redundancy_analysis = _load_json(redundancy_dir / "analysis.json", default=None)

    # Load final stories from generator_reviewer outputs
    stories = _load_mas_final_stories(run_dir, experiment_run)

    # Load chunk texts
    chunk_texts = {}
    if preprocessing_dir and Path(preprocessing_dir).exists():
        chunks = _load_json(Path(preprocessing_dir) / "chunks.json", default=[])
        chunk_texts = _build_chunk_texts(chunks)
    else:
        # Try to find preprocessing in outputs parent
        chunk_texts = _find_preprocessing_chunks(run_dir)

    logger.info(
        "Loaded MAS run: %s — %d stories, %d chunk texts",
        run_dir.name, len(stories), len(chunk_texts),
    )

    return {
        "stories": stories,
        "experiment_run": experiment_run,
        "evaluation_result": evaluation_result,
        "redundancy_analysis": redundancy_analysis,
        "chunk_texts": chunk_texts,
        "run_dir_str": str(run_dir),
    }


def _load_mas_final_stories(
    run_dir: Path, experiment_run: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Attempt to reconstruct final story list from MAS run.

    Strategy:
    1. Use final_user_story_ids from experiment_run.json to find stories.
    2. Search generator_reviewer/ subdirs for story JSONs.
    3. Return whatever stories are found (list may be empty if pipeline errored).
    """
    final_ids = set(experiment_run.get("final_user_story_ids", []))
    stories: Dict[str, Dict] = {}

    gr_dir = run_dir / "generator_reviewer"
    if gr_dir.exists():
        for chunk_dir in gr_dir.iterdir():
            if not chunk_dir.is_dir():
                continue
            # Look for the last iteration file (highest iteration number)
            iteration_files = sorted(chunk_dir.glob("iteration_*.json"))
            if iteration_files:
                data = _load_json(iteration_files[-1], default={})
                # The last iteration may contain final stories
                for s in data.get("stories", []):
                    sid = s.get("id", "")
                    if sid and (not final_ids or sid in final_ids):
                        stories[sid] = s

    result = list(stories.values())
    if not result and final_ids:
        logger.warning(
            "MAS run %s has %d final_user_story_ids but no story files found. "
            "Pipeline may have failed before producing output.",
            run_dir.name, len(final_ids),
        )
    return result


def _find_preprocessing_chunks(run_dir: Path) -> Dict[str, str]:
    """
    Try to find a preprocessing chunks.json by searching the outputs directory.
    Returns chunk_texts dict or empty dict.
    """
    # Walk up to outputs/, then look for any 01_preprocessing dir
    outputs_dir = run_dir.parent
    for candidate in sorted(outputs_dir.iterdir()):
        preproc = candidate / "01_preprocessing"
        chunks_file = preproc / "chunks.json"
        if chunks_file.exists():
            chunks = _load_json(chunks_file, default=[])
            if chunks:
                logger.info(
                    "Found preprocessing chunks from %s (%d chunks)",
                    candidate.name, len(chunks),
                )
                return _build_chunk_texts(chunks)
    return {}


# ── Config snapshot loader ─────────────────────────────────────────────────────

def load_config_snapshot(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config snapshot file."""
    return _load_yaml(config_path, default={})


def extract_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration for provenance tracking."""
    model = config.get("model", {})
    invocation = config.get("invocation", {})
    return {
        "provider": model.get("provider", "unknown"),
        "model_name": model.get("name", "unknown"),
        "temperature": model.get("temperature"),
        "top_p": model.get("top_p"),
        "response_format": model.get("response_format"),
        "max_retries": invocation.get("max_retries"),
        "max_reviewer_iterations": config.get("pipeline", {}).get(
            "max_reviewer_iterations"
        ),
        "wait_on_overload_seconds": invocation.get("wait_on_overload_seconds"),
    }


def extract_prompt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract prompt configuration for provenance tracking."""
    prompts = {}
    for section in ["generator_agent", "reviewer_agent", "redundancy_agent",
                    "analyst_agent", "reader_agent", "evaluator_agent"]:
        agent_cfg = config.get(section, {})
        for key in ["prompt_file", "system_prompt_file", "hallucination_prompt_file"]:
            if key in agent_cfg:
                prompts[f"{section}.{key}"] = agent_cfg[key]
    return prompts


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_chunk_texts(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build {chunk_id: text} mapping from chunks list."""
    result = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id", "")
        text = c.get("text") or c.get("content") or c.get("raw_text", "")
        if cid:
            result[cid] = text
    return result


def _require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            f"Ensure the pipeline run completed successfully."
        )


def _load_json(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        logger.debug("JSON file not found (returning default): %s", path)
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
        return default


def _load_yaml(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        logger.debug("YAML file not found (returning default): %s", path)
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load YAML %s: %s", path, exc)
        return default
