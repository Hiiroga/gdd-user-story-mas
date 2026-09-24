"""
evaluator_agent.py — Evaluator Agent (C9 / Agent 6)
=====================================================
Implements the Evaluator Agent as specified in MAS Detailed Architecture
§1 (Agent 6), Data Contracts §11 (EvaluationResult), and SRS-014.

Responsibilities:
  - Compute 5 thesis metrics for the final user story collection:
      EVAL-001  hallucination_rate
      EVAL-002  redundancy_rate
      EVAL-003  aqusa_score
      EVAL-004  coverage
      EVAL-005  diversity
  - Report consistency notes for excluded/flagged stories.
  - Produce a structured EvaluationResult (Data Contracts §11).
  - Does NOT modify any story content (AGENT-010).
  - Does NOT re-run Reviewer or Redundancy Checker logic (AGENT-010).
  - Does NOT silently omit any metric (VAL-029).

Metric computation strategy:
  - AQUSA (EVAL-003), Coverage (EVAL-004), Diversity (EVAL-005) are
    computed deterministically from story metadata — no LLM needed.
  - Redundancy Rate (EVAL-002): for MAS, derived from RedundancyAnalysis;
    for Baseline, requires a post-hoc LLM pairwise comparison (not yet
    implemented — returns null with reason per VAL-029).
  - Hallucination Rate (EVAL-001): for MAS, uses Reviewer Agent's
    ``gdd_traceability`` verdict from ``confidence_evidence``; for
    Baseline, uses LLM-based verification via hallucination prompt.

Design decisions:
  - Reuses ``baseline.llm_client.LLMClient`` for hallucination checking.
  - Config loaded from ``config/mas.yaml`` under ``evaluator_agent:``.
  - Stateless per call.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from gdd_userstory_mas.baseline.llm_client import LLMClient
from gdd_userstory_mas.schemas.evaluation_result import (
    AQUSADetail,
    EvaluationResult,
    MetricDetail,
)
from gdd_userstory_mas.schemas.final_user_story import FinalUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import RedundancyAnalysis

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Controlled vocabulary sizes for diversity computation (EVAL-005)
REQUIREMENT_TYPES = frozenset({"player", "system", "in-game entity", "dev team"})
GAME_DOMAINS = frozenset(
    {"gameplay", "UI", "narrative", "systems", "level design", "audio", "other"}
)
TOTAL_DIVERSITY_CATEGORIES = len(REQUIREMENT_TYPES) + len(GAME_DOMAINS)

# Heuristic: vague subject pronouns that may indicate ambiguity (EVAL-003 criterion 5)
_AMBIGUITY_PATTERN = re.compile(
    r"\b(it|they|this|that|them|these|those)\b\s+(can|will|should|is|are|enables|allows)",
    re.IGNORECASE,
)

# Full-text format pattern for AQUSA format check
_FORMAT_PATTERN = re.compile(
    r"^As a .+,\s+I want .+,\s+so that .+\.$",
    re.IGNORECASE,
)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class EvaluatorAgentConfig:
    """
    Configuration for the Evaluator Agent.
    Loaded from ``config/mas.yaml`` under ``evaluator_agent:``.
    """

    # LLM settings (used only for hallucination checking in Baseline mode)
    provider: str = "gemini"
    model_name: str = "gemini-flash-latest"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: Optional[float] = None
    response_format: str = "json_object"

    # Retry / rate-limit
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 13.0
    wait_on_overload_seconds: float = 60.0

    # Prompt
    hallucination_prompt_file: str = "config/prompts/evaluator_hallucination_prompt.txt"

    # Evaluation settings
    # Whether Redundancy-Unresolved stories count toward redundancy_rate
    # (OPEN-021: left as config option, default = False per conservative choice)
    count_unresolved_in_redundancy: bool = False

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "EvaluatorAgentConfig":
        """Load ``EvaluatorAgentConfig`` from ``config/mas.yaml``."""
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PyYAML is required. Run: pip install pyyaml") from exc

        path = Path(yaml_path)
        if not path.is_absolute():
            path = project_root / path

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        model = raw.get("model", {})
        inv = raw.get("invocation", {})
        ev = raw.get("evaluator_agent", {})

        return cls(
            provider=model.get("provider", "gemini"),
            model_name=model.get("name", "gemini-flash-latest"),
            temperature=float(model.get("temperature", 0.0)),
            max_tokens=int(ev.get("max_tokens", 512)),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            max_retries=int(inv.get("max_retries", 5)),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(
                inv.get("min_request_interval_seconds", 13.0)
            ),
            wait_on_overload_seconds=float(inv.get("wait_on_overload_seconds", 60.0)),
            hallucination_prompt_file=ev.get(
                "hallucination_prompt_file",
                "config/prompts/evaluator_hallucination_prompt.txt",
            ),
            count_unresolved_in_redundancy=bool(
                ev.get("count_unresolved_in_redundancy", False)
            ),
        )

    @classmethod
    def defaults(cls) -> "EvaluatorAgentConfig":
        return cls()


# ── Evaluator Agent ────────────────────────────────────────────────────────────

class EvaluatorAgent:
    """
    Evaluator Agent (Component C9 / Agent 6).

    Computes the 5 thesis metrics for a final user story collection and
    produces a structured ``EvaluationResult``.

    This agent:
    - Is STATELESS — each ``evaluate()`` call is independent.
    - Does NOT modify any story content (AGENT-010).
    - Does NOT re-run Reviewer or Redundancy Checker logic (AGENT-010).
    - Discloses all exclusions in ``consistency_notes``.
    - Returns null with documented reason for any metric that cannot
      be computed (VAL-029).

    Parameters
    ----------
    config:
        ``EvaluatorAgentConfig`` loaded from ``config/mas.yaml``.
    api_key:
        LLM API key. If None, reads from environment.
        Only used for Baseline hallucination checking.
    project_root:
        Project root for resolving prompt file paths.
    """

    def __init__(
        self,
        config: EvaluatorAgentConfig,
        api_key: Optional[str] = None,
        project_root: Path = Path("."),
    ) -> None:
        self._config = config
        self._project_root = Path(project_root)
        self._hallucination_prompt_template: Optional[str] = None

        # LLM client (lazy — only initialised when needed for Baseline)
        self._api_key = api_key
        self._client: Optional[LLMClient] = None

        logger.info(
            "EvaluatorAgent initialized (provider=%s, model=%s)",
            config.provider,
            config.model_name,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        stories: List[FinalUserStory],
        gdd_chunks: Dict[str, str],
        run_id: str,
        document_id: str,
        pipeline_type: str,
        total_chunk_count: int,
        redundancy_analysis: Optional[RedundancyAnalysis] = None,
    ) -> EvaluationResult:
        """
        Compute all thesis metrics and produce an EvaluationResult.

        Parameters
        ----------
        stories:
            Final user story collection (post-Redundancy Checker for MAS,
            direct LLM output for Baseline).
        gdd_chunks:
            Mapping of ``chunk_id → chunk_text`` for the source document.
            Required for Coverage and Hallucination Rate.
        run_id:
            Experiment run identifier.
        document_id:
            Source document identifier.
        pipeline_type:
            ``"MAS"`` or ``"Baseline"``.
        total_chunk_count:
            Total number of chunks produced by preprocessing (EVAL-004 denominator).
        redundancy_analysis:
            RedundancyAnalysis from the Redundancy Checker (MAS only).
            If None, redundancy_rate will be null with a documented reason.

        Returns
        -------
        EvaluationResult
            Structured evaluation report with all metrics.

        Raises
        ------
        ValueError
            If ``gdd_chunks`` is empty but hallucination/coverage computation
            is needed (AGENT-010 failure condition).
        """
        logger.info(
            "Evaluator starting: run=%s, doc=%s, pipeline=%s, "
            "stories=%d, chunks=%d",
            run_id, document_id, pipeline_type,
            len(stories), total_chunk_count,
        )

        if not stories:
            logger.warning("No stories to evaluate — returning empty report.")
            return self._empty_result(run_id, document_id, pipeline_type)

        consistency_notes: List[str] = []
        null_reasons: Dict[str, str] = {}
        metric_details: Dict[str, MetricDetail] = {}

        # Validate GDD chunks availability
        if not gdd_chunks:
            null_reasons["hallucination_rate"] = (
                "Source GDD chunk texts not provided. "
                "Cannot compute hallucination rate without source content "
                "(MAS Arch §4 failure condition)."
            )
            null_reasons["coverage"] = (
                "Source GDD chunk texts not provided. "
                "Cannot verify chunk coverage without chunk content."
            )
            logger.warning(
                "gdd_chunks is empty — hallucination_rate and coverage "
                "will be null."
            )

        # Collect consistency notes for flagged stories
        consistency_notes.extend(
            self._collect_consistency_notes(stories)
        )

        # ── EVAL-003: AQUSA Score (deterministic) ──────────────────────────────
        aqusa_score, aqusa_detail, aqusa_metric = self._compute_aqusa_score(stories)
        metric_details["aqusa_score"] = aqusa_metric
        if aqusa_score is None:
            null_reasons["aqusa_score"] = "No stories available."

        # ── EVAL-004: Coverage (deterministic) ────────────────────────────────
        if gdd_chunks:
            coverage, coverage_metric = self._compute_coverage(
                stories, total_chunk_count
            )
            if coverage is None and "coverage" not in null_reasons:
                null_reasons["coverage"] = (
                    "total_chunk_count is 0 — cannot compute coverage "
                    "(denominator is zero)."
                )
        else:
            coverage = None
            coverage_metric = MetricDetail(
                metric_name="coverage",
                formula="covered_chunks / total_chunks × 100",
                value=None,
                interpretation="Not computed — no GDD chunks provided.",
                limitation="Requires source chunk texts.",
            )
        metric_details["coverage"] = coverage_metric

        # ── EVAL-005: Diversity (deterministic) ───────────────────────────────
        diversity, diversity_metric = self._compute_diversity(stories)
        metric_details["diversity"] = diversity_metric
        if diversity is None:
            null_reasons["diversity"] = "No stories available."

        # ── EVAL-002: Redundancy Rate ──────────────────────────────────────────
        redundancy_rate, redundancy_metric = self._compute_redundancy_rate(
            stories, redundancy_analysis, pipeline_type,
            null_reasons, consistency_notes,
        )
        metric_details["redundancy_rate"] = redundancy_metric

        # ── EVAL-001: Hallucination Rate ───────────────────────────────────────
        if gdd_chunks:
            hallucination_rate, hall_metric = self._compute_hallucination_rate(
                stories, gdd_chunks, pipeline_type,
                null_reasons, consistency_notes,
            )
        else:
            hallucination_rate = None
            hall_metric = MetricDetail(
                metric_name="hallucination_rate",
                formula="hallucinated_stories / total_stories × 100",
                value=None,
                interpretation="Not computed — no GDD chunks provided.",
                limitation="Requires source chunk texts for verification.",
            )
        metric_details["hallucination_rate"] = hall_metric

        result = EvaluationResult(
            evaluation_id=f"eval__{run_id}",
            run_id=run_id,
            document_id=document_id,
            pipeline_type=pipeline_type,
            total_stories=len(stories),
            hallucination_rate=hallucination_rate,
            redundancy_rate=redundancy_rate,
            aqusa_score=aqusa_score,
            coverage=coverage,
            diversity=diversity,
            null_reasons=null_reasons,
            consistency_notes=consistency_notes,
            computed_at=datetime.now(timezone.utc).isoformat(),
            metric_details=metric_details,
            aqusa_detail=aqusa_detail,
        )

        logger.info(
            "Evaluation complete: AQUSA=%.1f%% Coverage=%.1f%% "
            "Diversity=%.2f Redundancy=%.1f%% Hallucination=%s",
            aqusa_score or 0,
            coverage or 0,
            diversity or 0,
            redundancy_rate or 0,
            f"{hallucination_rate:.1f}%" if hallucination_rate is not None else "null",
        )

        return result

    # ── EVAL-003: AQUSA Score ─────────────────────────────────────────────────

    def _compute_aqusa_score(
        self, stories: List[FinalUserStory]
    ) -> tuple[Optional[float], Optional[AQUSADetail], MetricDetail]:
        """
        EVAL-003: % of stories passing all AQUSA syntactic criteria.

        Criteria (deterministic):
          1. role field is non-empty
          2. action field is non-empty
          3. benefit field is non-empty
          4. full_text matches expected format
          5. No vague-pronoun ambiguity heuristic
        """
        if not stories:
            detail = MetricDetail(
                metric_name="aqusa_score",
                formula="stories_passing_all / total_stories × 100",
                value=None,
                interpretation="No stories to evaluate.",
                limitation="Empty collection.",
            )
            return None, None, detail

        role_empty = 0
        action_empty = 0
        benefit_empty = 0
        format_violated = 0
        ambiguity_count = 0
        passing = 0

        for s in stories:
            failed = False
            if not s.role.strip():
                role_empty += 1
                failed = True
            if not s.action.strip():
                action_empty += 1
                failed = True
            if not s.benefit.strip():
                benefit_empty += 1
                failed = True
            if not _FORMAT_PATTERN.match(s.full_text.strip()):
                format_violated += 1
                failed = True
            if _AMBIGUITY_PATTERN.search(s.full_text):
                ambiguity_count += 1
                failed = True
            if not failed:
                passing += 1

        total = len(stories)
        score = (passing / total) * 100.0

        aqusa_detail = AQUSADetail(
            total_stories=total,
            role_empty_count=role_empty,
            action_empty_count=action_empty,
            benefit_empty_count=benefit_empty,
            format_violated_count=format_violated,
            ambiguity_heuristic_count=ambiguity_count,
            stories_passing_all=passing,
        )

        metric = MetricDetail(
            metric_name="aqusa_score",
            formula="stories_passing_all_criteria / total_stories × 100",
            numerator=float(passing),
            denominator=float(total),
            value=score,
            interpretation=(
                f"{passing}/{total} stories ({score:.1f}%) meet all 5 AQUSA "
                f"syntactic criteria: role/action/benefit non-empty, "
                f"correct format, no ambiguity heuristic."
            ),
            limitation=(
                "Criterion 5 (ambiguity) is heuristic-based and may "
                "produce false positives. Syntactic check only — does not "
                "assess semantic quality."
            ),
        )

        return score, aqusa_detail, metric

    # ── EVAL-004: Coverage ────────────────────────────────────────────────────

    def _compute_coverage(
        self,
        stories: List[FinalUserStory],
        total_chunk_count: int,
    ) -> tuple[Optional[float], MetricDetail]:
        """
        EVAL-004: % of GDD chunks covered by ≥1 final story.

        Formula: covered_chunks / total_chunks × 100
        """
        if total_chunk_count == 0:
            detail = MetricDetail(
                metric_name="coverage",
                formula="covered_chunks / total_chunks × 100",
                value=None,
                interpretation="total_chunk_count is 0 — cannot compute coverage.",
                limitation="Denominator is zero.",
            )
            # Caller must add null_reasons["coverage"]; we return None signal
            return None, detail

        # Collect all source chunk IDs referenced by stories
        covered_ids: Set[str] = set()
        for s in stories:
            for cid in s.source_chunk_ids:
                covered_ids.add(cid)

        covered = len(covered_ids)
        cov = (covered / total_chunk_count) * 100.0

        detail = MetricDetail(
            metric_name="coverage",
            formula="covered_chunks / total_chunks × 100",
            numerator=float(covered),
            denominator=float(total_chunk_count),
            value=cov,
            interpretation=(
                f"{covered}/{total_chunk_count} GDD chunks ({cov:.1f}%) "
                f"have ≥1 final user story."
            ),
            limitation=(
                "OPEN-009: Some chunks may legitimately contain no extractable "
                "requirements (e.g., title page, bibliography). Counting all "
                "chunks may understate true coverage."
            ),
        )

        return cov, detail

    # ── EVAL-005: Diversity ───────────────────────────────────────────────────

    def _compute_diversity(
        self, stories: List[FinalUserStory]
    ) -> tuple[Optional[float], MetricDetail]:
        """
        EVAL-005: Ratio of unique role/domain categories seen.

        Formula: unique_categories_seen / total_possible_categories
        Total possible = len(REQUIREMENT_TYPES) + len(GAME_DOMAINS) = 11
        """
        if not stories:
            detail = MetricDetail(
                metric_name="diversity",
                formula="unique_categories_seen / total_possible_categories",
                value=None,
                interpretation="No stories.",
                limitation="Empty collection.",
            )
            return None, detail

        seen_types: Set[str] = {s.requirement_type for s in stories}
        seen_domains: Set[str] = {s.game_domain for s in stories}
        unique_seen = len(seen_types) + len(seen_domains)
        div = unique_seen / TOTAL_DIVERSITY_CATEGORIES

        detail = MetricDetail(
            metric_name="diversity",
            formula=(
                f"(unique_req_types + unique_domains) / "
                f"(total_req_types + total_domains) = "
                f"({len(seen_types)} + {len(seen_domains)}) / "
                f"({len(REQUIREMENT_TYPES)} + {len(GAME_DOMAINS)})"
            ),
            numerator=float(unique_seen),
            denominator=float(TOTAL_DIVERSITY_CATEGORIES),
            value=div,
            interpretation=(
                f"{unique_seen}/{TOTAL_DIVERSITY_CATEGORIES} unique categories "
                f"represented (diversity={div:.3f}). "
                f"Req types seen: {sorted(seen_types)}. "
                f"Domains seen: {sorted(seen_domains)}."
            ),
            limitation=(
                "A GDD focused on one domain (e.g., gameplay only) will "
                "naturally score low regardless of pipeline quality. "
                "Compare diversity between MAS and Baseline for the same GDD."
            ),
        )

        return div, detail

    # ── EVAL-002: Redundancy Rate ─────────────────────────────────────────────

    def _compute_redundancy_rate(
        self,
        stories: List[FinalUserStory],
        redundancy_analysis: Optional[RedundancyAnalysis],
        pipeline_type: str,
        null_reasons: Dict[str, str],
        consistency_notes: List[str],
    ) -> tuple[Optional[float], MetricDetail]:
        """
        EVAL-002: % of stories flagged as redundant pre-deduplication.

        For MAS: derived from RedundancyAnalysis.
        For Baseline: null with reason (no Redundancy Checker stage exists).
        """
        if pipeline_type == "Baseline":
            null_reasons["redundancy_rate"] = (
                "Baseline pipeline has no Redundancy Checker stage. "
                "Redundancy rate is not applicable for Baseline "
                "(no RedundancyAnalysis available)."
            )
            detail = MetricDetail(
                metric_name="redundancy_rate",
                formula="stories_in_duplicate_groups / total_input_stories × 100",
                value=None,
                interpretation=(
                    "Not applicable for Baseline pipeline "
                    "(no Redundancy Checker stage)."
                ),
                limitation="Baseline has no redundancy detection pass.",
            )
            return None, detail

        if redundancy_analysis is None:
            null_reasons["redundancy_rate"] = (
                "RedundancyAnalysis not provided. "
                "Cannot compute redundancy_rate without it."
            )
            detail = MetricDetail(
                metric_name="redundancy_rate",
                formula="stories_in_duplicate_groups / total_input_stories × 100",
                value=None,
                interpretation="RedundancyAnalysis not provided.",
                limitation="Requires RedundancyAnalysis output.",
            )
            return None, detail

        total_input = redundancy_analysis.input_story_count
        if total_input == 0:
            null_reasons["redundancy_rate"] = (
                "RedundancyAnalysis.input_story_count is 0."
            )
            detail = MetricDetail(
                metric_name="redundancy_rate",
                formula="stories_in_duplicate_groups / total_input_stories × 100",
                value=None,
                interpretation="No input stories.",
                limitation="Denominator is zero.",
            )
            return None, detail

        # Count stories in duplicate groups (numerator)
        grouped_ids: Set[str] = set()
        for group in redundancy_analysis.duplicate_groups:
            grouped_ids.update(group.story_ids)

        # OPEN-021: whether Redundancy-Unresolved stories count
        if self._config.count_unresolved_in_redundancy:
            unresolved = [
                s for s in stories
                if s.validation_status == "Redundancy-Unresolved"
            ]
            if unresolved:
                consistency_notes.append(
                    f"OPEN-021 note: {len(unresolved)} Redundancy-Unresolved "
                    f"stories are counted toward redundancy_rate numerator "
                    f"(count_unresolved_in_redundancy=True)."
                )
                for s in unresolved:
                    grouped_ids.add(s.id)
        else:
            unresolved = [
                s for s in stories
                if s.validation_status == "Redundancy-Unresolved"
            ]
            if unresolved:
                consistency_notes.append(
                    f"OPEN-021 note: {len(unresolved)} Redundancy-Unresolved "
                    f"stories are NOT counted toward redundancy_rate "
                    f"(count_unresolved_in_redundancy=False)."
                )

        numerator = len(grouped_ids)
        rate = (numerator / total_input) * 100.0

        detail = MetricDetail(
            metric_name="redundancy_rate",
            formula="stories_in_duplicate_groups / total_input_stories × 100",
            numerator=float(numerator),
            denominator=float(total_input),
            value=rate,
            interpretation=(
                f"{numerator}/{total_input} stories ({rate:.1f}%) were in "
                f"duplicate groups before deduplication. "
                f"Resolution status: {redundancy_analysis.resolution_status}."
            ),
            limitation=(
                "OPEN-021: Whether Redundancy-Unresolved stories count toward "
                "the numerator is configurable (count_unresolved_in_redundancy). "
                "Current setting: "
                f"{self._config.count_unresolved_in_redundancy}."
            ),
        )

        return rate, detail

    # ── EVAL-001: Hallucination Rate ──────────────────────────────────────────

    def _compute_hallucination_rate(
        self,
        stories: List[FinalUserStory],
        gdd_chunks: Dict[str, str],
        pipeline_type: str,
        null_reasons: Dict[str, str],
        consistency_notes: List[str],
    ) -> tuple[Optional[float], MetricDetail]:
        """
        EVAL-001: % of stories with unsupported (hallucinated) claims.

        For MAS: uses Reviewer's ``gdd_traceability`` verdict from
        ``confidence_evidence.reviewer_traceability_confirmed``.
        For Baseline: calls LLM hallucination-check prompt per story.
        """
        if not stories:
            null_reasons["hallucination_rate"] = "No stories to evaluate."
            detail = MetricDetail(
                metric_name="hallucination_rate",
                formula="hallucinated_stories / total_stories × 100",
                value=None,
                interpretation="No stories.",
                limitation="Empty collection.",
            )
            return None, detail

        hallucinated = 0
        total = len(stories)
        unverifiable = 0

        if pipeline_type == "MAS":
            # Use Reviewer's gdd_traceability signal
            for s in stories:
                if s.confidence_evidence is None:
                    # No confidence evidence — cannot determine; treat as unknown
                    unverifiable += 1
                    continue
                confirmed = s.confidence_evidence.reviewer_traceability_confirmed
                if confirmed is False:
                    # gdd_traceability FAILED → story contains unsupported claim
                    hallucinated += 1
                elif confirmed is None:
                    unverifiable += 1

            if unverifiable > 0:
                consistency_notes.append(
                    f"EVAL-001 note: {unverifiable}/{total} stories have no "
                    f"reviewer_traceability_confirmed signal — they are "
                    f"unverifiable and excluded from the hallucination_rate "
                    f"denominator."
                )

            verifiable = total - unverifiable
            if verifiable == 0:
                null_reasons["hallucination_rate"] = (
                    "No stories have reviewer_traceability_confirmed data. "
                    "Cannot compute hallucination_rate."
                )
                detail = MetricDetail(
                    metric_name="hallucination_rate",
                    formula="hallucinated_stories / verifiable_stories × 100",
                    value=None,
                    interpretation="No verifiable stories.",
                    limitation="All stories lack reviewer_traceability_confirmed.",
                )
                return None, detail

            rate = (hallucinated / verifiable) * 100.0
            detail = MetricDetail(
                metric_name="hallucination_rate",
                formula=(
                    "stories_with_gdd_traceability_failed / "
                    "verifiable_stories × 100"
                ),
                numerator=float(hallucinated),
                denominator=float(verifiable),
                value=rate,
                interpretation=(
                    f"{hallucinated}/{verifiable} verifiable stories "
                    f"({rate:.1f}%) failed the Reviewer's gdd_traceability "
                    f"criterion, indicating potential unsupported claims. "
                    f"({unverifiable} stories had no traceability signal "
                    f"and were excluded from denominator.)"
                ),
                limitation=(
                    "Uses Reviewer Agent's gdd_traceability verdict as proxy "
                    "for hallucination — same LLM may have circular bias. "
                    "Only covers stories where reviewer_traceability_confirmed "
                    "is populated."
                ),
            )
            return rate, detail

        else:  # Baseline
            # LLM-based verification per story
            hallucinated, unverifiable, llm_note = self._llm_hallucination_check(
                stories, gdd_chunks
            )
            if llm_note:
                consistency_notes.append(llm_note)

            verifiable = total - unverifiable
            if verifiable == 0:
                null_reasons["hallucination_rate"] = (
                    "All stories lack corresponding source chunk text. "
                    "Cannot compute hallucination_rate."
                )
                detail = MetricDetail(
                    metric_name="hallucination_rate",
                    formula="hallucinated_stories / verifiable_stories × 100",
                    value=None,
                    interpretation="No verifiable stories.",
                    limitation="Missing source chunk texts.",
                )
                return None, detail

            rate = (hallucinated / verifiable) * 100.0
            detail = MetricDetail(
                metric_name="hallucination_rate",
                formula="llm_hallucinated_stories / verifiable_stories × 100",
                numerator=float(hallucinated),
                denominator=float(verifiable),
                value=rate,
                interpretation=(
                    f"{hallucinated}/{verifiable} Baseline stories "
                    f"({rate:.1f}%) were flagged as containing unsupported "
                    f"claims by the LLM hallucination-check prompt."
                ),
                limitation=(
                    "LLM-based check using same model — potential circular bias. "
                    "Confidence 'high'/'medium'/'low' from LLM is not weighted. "
                    f"{unverifiable} stories lacked source chunk text."
                ),
            )
            return rate, detail

    def _llm_hallucination_check(
        self,
        stories: List[FinalUserStory],
        gdd_chunks: Dict[str, str],
    ) -> tuple[int, int, str]:
        """
        LLM-based hallucination check for Baseline stories.

        Returns (hallucinated_count, unverifiable_count, consistency_note).
        """
        self._ensure_client()
        prompt_template = self._load_hallucination_prompt()

        hallucinated = 0
        unverifiable = 0
        notes = []

        for s in stories:
            # Find source chunk text
            chunk_text = ""
            for cid in s.source_chunk_ids:
                if cid in gdd_chunks:
                    chunk_text = gdd_chunks[cid]
                    break

            if not chunk_text:
                unverifiable += 1
                continue

            system_prompt = (
                prompt_template
                .replace("{role}", s.role)
                .replace("{action}", s.action)
                .replace("{benefit}", s.benefit)
                .replace("{full_text}", s.full_text)
                .replace("{source_chunk_text}", chunk_text)
            )

            try:
                llm_resp = self._client.call(
                    system_prompt,
                    "Evaluate the user story above for hallucination. Return JSON.",
                )
                data: Dict[str, Any] = json.loads(llm_resp.content)
                is_hallucinated = bool(data.get("is_hallucinated", False))
                if is_hallucinated:
                    hallucinated += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hallucination check failed for story %s: %s — skipping.",
                    s.id, exc,
                )
                unverifiable += 1

        note = ""
        if unverifiable > 0:
            note = (
                f"EVAL-001 Baseline note: {unverifiable}/{len(stories)} "
                f"stories could not be verified (missing chunk text or LLM error)."
            )

        return hallucinated, unverifiable, note

    # ── Consistency notes ─────────────────────────────────────────────────────

    def _collect_consistency_notes(
        self, stories: List[FinalUserStory]
    ) -> List[str]:
        """Collect transparency disclosures for flagged/excluded stories."""
        notes = []

        rejected = [s for s in stories if s.validation_status == "Rejected-ManualReview"]
        if rejected:
            notes.append(
                f"TRANSPARENCY: {len(rejected)} stories have "
                f"validation_status='Rejected-ManualReview' and are "
                f"included in the collection but represent terminal-Invalid "
                f"outcomes. IDs: {[s.id for s in rejected[:5]]}"
                f"{'...' if len(rejected) > 5 else ''}."
            )

        unresolved = [s for s in stories if s.validation_status == "Redundancy-Unresolved"]
        if unresolved:
            notes.append(
                f"TRANSPARENCY: {len(unresolved)} stories have "
                f"validation_status='Redundancy-Unresolved' (redundancy "
                f"checker loop did not converge). IDs: "
                f"{[s.id for s in unresolved[:5]]}"
                f"{'...' if len(unresolved) > 5 else ''}."
            )

        merged = [s for s in stories if s.validation_status == "Merged"]
        if merged:
            notes.append(
                f"INFO: {len(merged)} stories resulted from Redundancy "
                f"Checker merge operations and combine content from "
                f"multiple original stories (TRACE-004)."
            )

        return notes

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        """Lazily initialise the LLM client (only for Baseline hallucination)."""
        if self._client is None:
            try:
                from gdd_userstory_mas.baseline.llm_client import LLMClient  # noqa
            except ImportError as exc:
                raise ImportError("LLMClient not available.") from exc
            self._client = LLMClient(
                provider=self._config.provider,
                model_name=self._config.model_name,
                api_key=self._api_key,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                top_p=self._config.top_p,
                response_format=self._config.response_format,
                max_retries=self._config.max_retries,
                retry_backoff_seconds=self._config.retry_backoff_seconds,
                min_request_interval_seconds=self._config.min_request_interval_seconds,
                wait_on_overload_seconds=self._config.wait_on_overload_seconds,
            )

    def _load_hallucination_prompt(self) -> str:
        if self._hallucination_prompt_template is None:
            path = self._project_root / self._config.hallucination_prompt_file
            if not path.exists():
                raise FileNotFoundError(
                    f"Hallucination prompt not found: {path}"
                )
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Hallucination prompt is empty: {path}")
            self._hallucination_prompt_template = text
        return self._hallucination_prompt_template

    def _empty_result(
        self, run_id: str, document_id: str, pipeline_type: str
    ) -> EvaluationResult:
        """Return a null-metric result for an empty story collection."""
        reason = "No stories in the evaluated collection."
        null_reasons = {
            "hallucination_rate": reason,
            "redundancy_rate": (
                reason
                if pipeline_type == "MAS"
                else "Baseline pipeline has no Redundancy Checker stage."
            ),
            "aqusa_score": reason,
            "coverage": reason,
            "diversity": reason,
        }
        if pipeline_type == "Baseline":
            null_reasons["redundancy_rate"] = (
                "Baseline pipeline has no Redundancy Checker stage."
            )
        return EvaluationResult(
            evaluation_id=f"eval__{run_id}",
            run_id=run_id,
            document_id=document_id,
            pipeline_type=pipeline_type,
            total_stories=0,
            hallucination_rate=None,
            redundancy_rate=None,
            aqusa_score=None,
            coverage=None,
            diversity=None,
            null_reasons=null_reasons,
            consistency_notes=["No stories in the evaluated collection."],
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
