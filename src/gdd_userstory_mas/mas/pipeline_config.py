"""
pipeline_config.py — MASPipelineConfig
========================================
Single configuration dataclass for the MAS orchestration pipeline.
Aggregates per-agent configs and orchestration-level settings from
``config/mas.yaml`` and ``config/pipeline.yaml``.

Design: one ``from_yaml()`` call populates every sub-config by delegating
to each agent's existing ``from_yaml()`` method.  The orchestrator only
holds this dataclass and never reads YAML directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig
from gdd_userstory_mas.mas.evaluator_agent import EvaluatorAgentConfig
from gdd_userstory_mas.mas.generator_agent import GeneratorAgentConfig
from gdd_userstory_mas.mas.reader_agent import ReaderAgentConfig
from gdd_userstory_mas.mas.redundancy_agent import RedundancyAgentConfig
from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgentConfig
from gdd_userstory_mas.preprocessing.pipeline import PreprocessingConfig


@dataclass
class MASPipelineConfig:
    """
    Full pipeline configuration.

    Attributes
    ----------
    max_reviewer_iterations:
        Maximum Generator↔Reviewer iterations per candidate story
        (inclusive of the first attempt).  If a story fails review
        on all iterations it is flagged ``Rejected-ManualReview``.
    output_dir:
        Root directory for run output artifacts.
        Sub-directory ``{output_dir}/{run_id}/`` is created per run.
    logs_dir:
        Root directory for per-run logs.
        Sub-directory ``{logs_dir}/{run_id}/`` is created per run.
    save_intermediate:
        When True (default), write Reader/Analyst/Generator/Reviewer
        outputs to disk at each stage boundary.
    abort_on_empty_chunks:
        When True, halt the run if preprocessing yields zero chunks.
    abort_on_empty_reader:
        When True, skip a chunk if the Reader produced zero evidence items.
    reader_config:
        Config for GDDReaderAgent (C4).
    analyst_config:
        Config for RequirementsAnalystAgent (C5).
    generator_config:
        Config for UserStoryGeneratorAgent (C6).
    reviewer_config:
        Config for ReviewerAgent (C7).
    redundancy_config:
        Config for RedundancyCheckerAgent (C8).
    evaluator_config:
        Config for EvaluatorAgent (C9).
    preprocessing_config:
        Config for the Preprocessing pipeline.
    """

    # ── Orchestration-level settings ───────────────────────────────────────────
    max_reviewer_iterations: int = 3
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    save_intermediate: bool = True
    abort_on_empty_chunks: bool = False
    abort_on_empty_reader: bool = False

    # ── Per-agent configs ──────────────────────────────────────────────────────
    reader_config: ReaderAgentConfig = field(
        default_factory=ReaderAgentConfig.defaults
        if hasattr(ReaderAgentConfig, "defaults")
        else ReaderAgentConfig
    )
    analyst_config: AnalystAgentConfig = field(
        default_factory=AnalystAgentConfig.defaults
        if hasattr(AnalystAgentConfig, "defaults")
        else AnalystAgentConfig
    )
    generator_config: GeneratorAgentConfig = field(
        default_factory=GeneratorAgentConfig.defaults
        if hasattr(GeneratorAgentConfig, "defaults")
        else GeneratorAgentConfig
    )
    reviewer_config: ReviewerAgentConfig = field(
        default_factory=ReviewerAgentConfig.defaults
        if hasattr(ReviewerAgentConfig, "defaults")
        else ReviewerAgentConfig
    )
    redundancy_config: RedundancyAgentConfig = field(
        default_factory=RedundancyAgentConfig.defaults
        if hasattr(RedundancyAgentConfig, "defaults")
        else RedundancyAgentConfig
    )
    evaluator_config: EvaluatorAgentConfig = field(
        default_factory=EvaluatorAgentConfig.defaults
    )
    preprocessing_config: Optional[PreprocessingConfig] = None

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        mas_yaml: str | Path,
        pipeline_yaml: str | Path,
        project_root: Path = Path("."),
    ) -> "MASPipelineConfig":
        """
        Build a ``MASPipelineConfig`` from ``config/mas.yaml`` and
        ``config/pipeline.yaml``.

        Parameters
        ----------
        mas_yaml:
            Path to ``config/mas.yaml``.
        pipeline_yaml:
            Path to ``config/pipeline.yaml`` (preprocessing config).
        project_root:
            Project root for resolving relative paths.
        """
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PyYAML is required: pip install pyyaml") from exc

        mas_path = Path(mas_yaml)
        if not mas_path.is_absolute():
            mas_path = project_root / mas_path

        with open(mas_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        orch = raw.get("orchestration", {})

        output_dir = Path(orch.get("output_dir", "outputs"))
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

        logs_dir = Path(orch.get("logs_dir", "logs"))
        if not logs_dir.is_absolute():
            logs_dir = project_root / logs_dir

        # Delegate sub-configs to their own from_yaml()
        reader_cfg = ReaderAgentConfig.from_yaml(mas_path, project_root=project_root)
        analyst_cfg = AnalystAgentConfig.from_yaml(mas_path, project_root=project_root)
        generator_cfg = GeneratorAgentConfig.from_yaml(mas_path, project_root=project_root)
        reviewer_cfg = ReviewerAgentConfig.from_yaml(mas_path, project_root=project_root)
        redundancy_cfg = RedundancyAgentConfig.from_yaml(mas_path, project_root=project_root)
        evaluator_cfg = EvaluatorAgentConfig.from_yaml(mas_path, project_root=project_root)
        pre_yaml = Path(pipeline_yaml)
        if not pre_yaml.is_absolute():
            pre_yaml = project_root / pre_yaml
        preprocessing_cfg = PreprocessingConfig.from_yaml(pre_yaml)


        return cls(
            max_reviewer_iterations=int(orch.get("max_reviewer_iterations", 3)),
            output_dir=output_dir,
            logs_dir=logs_dir,
            save_intermediate=bool(orch.get("save_intermediate", True)),
            abort_on_empty_chunks=bool(orch.get("abort_on_empty_chunks", False)),
            abort_on_empty_reader=bool(orch.get("abort_on_empty_reader", False)),
            reader_config=reader_cfg,
            analyst_config=analyst_cfg,
            generator_config=generator_cfg,
            reviewer_config=reviewer_cfg,
            redundancy_config=redundancy_cfg,
            evaluator_config=evaluator_cfg,
            preprocessing_config=preprocessing_cfg,
        )

    @classmethod
    def defaults(cls) -> "MASPipelineConfig":
        """Return a config with all defaults (for testing without YAML files)."""
        return cls(
            reader_config=ReaderAgentConfig(),
            analyst_config=AnalystAgentConfig(),
            generator_config=GeneratorAgentConfig(),
            reviewer_config=ReviewerAgentConfig(),
            redundancy_config=RedundancyAgentConfig(),
            evaluator_config=EvaluatorAgentConfig(),
        )
