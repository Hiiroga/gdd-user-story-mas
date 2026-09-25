"""
evaluation/metrics.py — Pure Metric Calculation Functions
==========================================================
All metric functions in this module are:
  - Pure (no side effects, no I/O)
  - Fully deterministic given identical input
  - Unit-testable without mocking

Metrics implemented (thesis EVAL-001 to EVAL-005):
  EVAL-001  hallucination_rate
  EVAL-002  redundancy_rate
  EVAL-003  aqusa_score
  EVAL-004  coverage
  EVAL-005  diversity

Each metric function returns a MetricObservation dataclass containing:
  - raw_observations : list of per-story observations (the "raw data")
  - value            : computed scalar value (or None)
  - formula_version  : identifies the exact computation formula used
  - input_count      : number of inputs that went into the computation
  - null_reason      : documented reason when value is None (VAL-029)

Design principles:
  - No fabrication of ground truth.
  - Human-annotation requirements are explicitly documented and separated
    (see ANNOTATION_REQUIRED flags and HumanAnnotationTask).
  - Baseline redundancy is computable post-hoc via a pairwise similarity
    method that does NOT require an LLM (Jaccard on action/benefit tokens).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

FORMULA_VERSION_AQUSA = "aqusa_v1_syntactic_5criteria"
FORMULA_VERSION_COVERAGE = "coverage_v1_chunk_ratio"
FORMULA_VERSION_DIVERSITY = "diversity_v1_category_ratio"
FORMULA_VERSION_REDUNDANCY_MAS = "redundancy_v1_mas_checker_groups"
FORMULA_VERSION_REDUNDANCY_BASELINE = "redundancy_v2_jaccard_pairwise_0.7"
FORMULA_VERSION_HALLUCINATION_MAS = "hallucination_v1_reviewer_traceability"
FORMULA_VERSION_HALLUCINATION_BASELINE = "hallucination_v2_source_excerpt_keyword"

# Controlled vocabulary for diversity (EVAL-005)
REQUIREMENT_TYPES: frozenset = frozenset({"player", "system", "in-game entity", "dev team"})
GAME_DOMAINS: frozenset = frozenset(
    {"gameplay", "UI", "narrative", "systems", "level design", "audio", "other"}
)
TOTAL_DIVERSITY_CATEGORIES: int = len(REQUIREMENT_TYPES) + len(GAME_DOMAINS)

# AQUSA syntactic pattern — "As a ..., I want ..., so that ...."
_FORMAT_PATTERN = re.compile(
    r"^As a .+,\s+I want .+,\s+so that .+\.$",
    re.IGNORECASE,
)
# Vague pronoun heuristic (EVAL-003 criterion 5)
_AMBIGUITY_PATTERN = re.compile(
    r"\b(it|they|this|that|them|these|those)\b\s+(can|will|should|is|are|enables|allows)",
    re.IGNORECASE,
)

# Jaccard similarity threshold for Baseline redundancy detection
JACCARD_REDUNDANCY_THRESHOLD: float = 0.7


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class StoryObservation:
    """Raw per-story observation for a given metric."""
    story_id: str
    value: Any  # metric-specific raw value (bool, float, str, etc.)
    detail: str  # human-readable explanation of this observation


@dataclass
class MetricObservation:
    """
    Complete metric computation result, including raw observations.

    Stored verbatim in the ComparisonResult for later statistical analysis.
    """
    metric_name: str
    formula_version: str
    input_count: int
    raw_observations: List[StoryObservation] = field(default_factory=list)
    value: Optional[float] = None
    null_reason: Optional[str] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    interpretation: str = ""
    limitation: str = ""
    annotation_required: bool = False
    annotation_task: Optional[str] = None


@dataclass
class HumanAnnotationTask:
    """
    Declares a metric that CANNOT be computed automatically and requires
    human expert annotation.

    This separates the automated evaluation process from human judgment.
    Included in the ComparisonResult to make the annotation requirement
    machine-readable and explicit.
    """
    metric_name: str
    reason: str
    instructions: str
    annotation_schema: Dict[str, str]


# ── EVAL-003: AQUSA Score ──────────────────────────────────────────────────────

def compute_aqusa_score(stories: List[Dict[str, Any]]) -> MetricObservation:
    """
    EVAL-003: % of stories passing all AQUSA syntactic criteria.

    Formula: stories_passing_all_criteria / total_stories × 100
    Version: aqusa_v1_syntactic_5criteria

    Criteria (deterministic, no LLM required):
      1. role is non-empty
      2. action is non-empty
      3. benefit is non-empty
      4. full_text matches "As a ..., I want ..., so that ...." format
      5. No vague-pronoun ambiguity heuristic in full_text

    Parameters
    ----------
    stories:
        List of story dicts with keys: id, role, action, benefit, full_text.

    Returns
    -------
    MetricObservation with per-story raw observations.
    """
    if not stories:
        return MetricObservation(
            metric_name="aqusa_score",
            formula_version=FORMULA_VERSION_AQUSA,
            input_count=0,
            value=None,
            null_reason="Empty story collection — no stories to evaluate.",
            interpretation="No stories available.",
            limitation="Empty collection.",
        )

    observations: List[StoryObservation] = []
    passing = 0
    total = len(stories)

    for s in stories:
        story_id = s.get("id", "unknown")
        role = (s.get("role") or "").strip()
        action = (s.get("action") or "").strip()
        benefit = (s.get("benefit") or "").strip()
        full_text = (s.get("full_text") or "").strip()

        failures = []
        if not role:
            failures.append("role_empty")
        if not action:
            failures.append("action_empty")
        if not benefit:
            failures.append("benefit_empty")
        if not _FORMAT_PATTERN.match(full_text):
            failures.append("format_violated")
        if _AMBIGUITY_PATTERN.search(full_text):
            failures.append("ambiguity_heuristic")

        passed = len(failures) == 0
        if passed:
            passing += 1

        observations.append(StoryObservation(
            story_id=story_id,
            value=passed,
            detail=(
                "PASS" if passed
                else f"FAIL: {', '.join(failures)}"
            ),
        ))

    score = (passing / total) * 100.0

    return MetricObservation(
        metric_name="aqusa_score",
        formula_version=FORMULA_VERSION_AQUSA,
        input_count=total,
        raw_observations=observations,
        value=score,
        numerator=float(passing),
        denominator=float(total),
        interpretation=(
            f"{passing}/{total} stories ({score:.1f}%) meet all 5 AQUSA "
            f"syntactic criteria."
        ),
        limitation=(
            "Criterion 5 (ambiguity) is a heuristic regex — may produce false "
            "positives. Syntactic only; does not assess semantic quality."
        ),
        annotation_required=False,
    )


# ── EVAL-004: Coverage ─────────────────────────────────────────────────────────

def compute_coverage(
    stories: List[Dict[str, Any]],
    total_chunk_count: int,
) -> MetricObservation:
    """
    EVAL-004: % of GDD chunks covered by ≥1 final story.

    Formula: covered_chunks / total_chunks × 100
    Version: coverage_v1_chunk_ratio

    Parameters
    ----------
    stories:
        List of story dicts with key: source_chunk_ids (List[str]).
    total_chunk_count:
        Total number of GDD chunks produced by preprocessing.

    Returns
    -------
    MetricObservation with per-chunk coverage observations.
    """
    if total_chunk_count == 0:
        return MetricObservation(
            metric_name="coverage",
            formula_version=FORMULA_VERSION_COVERAGE,
            input_count=0,
            value=None,
            null_reason="total_chunk_count is 0 — denominator is zero.",
            interpretation="Cannot compute: no chunks.",
            limitation="Denominator is zero.",
        )

    if not stories:
        return MetricObservation(
            metric_name="coverage",
            formula_version=FORMULA_VERSION_COVERAGE,
            input_count=total_chunk_count,
            raw_observations=[],
            value=0.0,
            numerator=0.0,
            denominator=float(total_chunk_count),
            interpretation="0 chunks covered — no stories produced.",
            limitation=(
                "OPEN-009: Some chunks may legitimately contain no extractable "
                "requirements (e.g., title page)."
            ),
        )

    covered_ids: Set[str] = set()
    for s in stories:
        for cid in (s.get("source_chunk_ids") or []):
            covered_ids.add(cid)

    covered = len(covered_ids)
    cov = (covered / total_chunk_count) * 100.0

    observations = [
        StoryObservation(
            story_id=s.get("id", "unknown"),
            value=s.get("source_chunk_ids", []),
            detail=f"Covers chunks: {s.get('source_chunk_ids', [])}",
        )
        for s in stories
    ]

    return MetricObservation(
        metric_name="coverage",
        formula_version=FORMULA_VERSION_COVERAGE,
        input_count=total_chunk_count,
        raw_observations=observations,
        value=cov,
        numerator=float(covered),
        denominator=float(total_chunk_count),
        interpretation=(
            f"{covered}/{total_chunk_count} GDD chunks ({cov:.1f}%) "
            f"have ≥1 final story referencing them."
        ),
        limitation=(
            "OPEN-009: Some chunks may legitimately have no extractable "
            "requirements. All chunks are counted in the denominator."
        ),
        annotation_required=False,
    )


# ── EVAL-005: Diversity ────────────────────────────────────────────────────────

def compute_diversity(stories: List[Dict[str, Any]]) -> MetricObservation:
    """
    EVAL-005: Ratio of unique role/domain categories seen.

    Formula: (unique_req_types + unique_game_domains) /
             (total_req_types + total_game_domains)
    Version: diversity_v1_category_ratio
    Max value = 1.0 when all 11 categories are represented.

    Parameters
    ----------
    stories:
        List of story dicts with keys: id, requirement_type, game_domain.

    Returns
    -------
    MetricObservation with per-story category observations.
    """
    if not stories:
        return MetricObservation(
            metric_name="diversity",
            formula_version=FORMULA_VERSION_DIVERSITY,
            input_count=0,
            value=None,
            null_reason="Empty story collection.",
            interpretation="No stories available.",
            limitation="Empty collection.",
        )

    seen_types: Set[str] = set()
    seen_domains: Set[str] = set()
    observations: List[StoryObservation] = []

    for s in stories:
        rt = (s.get("requirement_type") or "").strip()
        gd = (s.get("game_domain") or "").strip()
        seen_types.add(rt)
        seen_domains.add(gd)
        observations.append(StoryObservation(
            story_id=s.get("id", "unknown"),
            value={"requirement_type": rt, "game_domain": gd},
            detail=f"type={rt!r}, domain={gd!r}",
        ))

    unique_seen = len(seen_types) + len(seen_domains)
    div = unique_seen / TOTAL_DIVERSITY_CATEGORIES

    return MetricObservation(
        metric_name="diversity",
        formula_version=FORMULA_VERSION_DIVERSITY,
        input_count=len(stories),
        raw_observations=observations,
        value=div,
        numerator=float(unique_seen),
        denominator=float(TOTAL_DIVERSITY_CATEGORIES),
        interpretation=(
            f"{unique_seen}/{TOTAL_DIVERSITY_CATEGORIES} unique categories "
            f"represented (diversity={div:.4f}). "
            f"Req types seen: {sorted(seen_types)}. "
            f"Domains seen: {sorted(seen_domains)}."
        ),
        limitation=(
            f"Total possible = {TOTAL_DIVERSITY_CATEGORIES} "
            f"({len(REQUIREMENT_TYPES)} req types + {len(GAME_DOMAINS)} domains). "
            "A GDD focused on one domain will naturally score low regardless of "
            "pipeline quality. Always compare MAS vs. Baseline on the same GDD."
        ),
        annotation_required=False,
    )


# ── EVAL-002: Redundancy Rate ──────────────────────────────────────────────────

def compute_redundancy_rate_mas(
    stories: List[Dict[str, Any]],
    redundancy_analysis: Optional[Dict[str, Any]],
) -> MetricObservation:
    """
    EVAL-002 (MAS): % of stories that were in duplicate groups before
    the Redundancy Checker deduplicated them.

    Formula: stories_in_duplicate_groups / input_story_count × 100
    Version: redundancy_v1_mas_checker_groups

    Source: RedundancyAnalysis.duplicate_groups from the MAS pipeline.
    If not available (e.g., pipeline errored before this stage), returns null.

    Parameters
    ----------
    stories:
        Final story list (post-dedup). Used for cross-referencing.
    redundancy_analysis:
        Dict representation of RedundancyAnalysis schema, or None.

    Returns
    -------
    MetricObservation.
    """
    if redundancy_analysis is None:
        return MetricObservation(
            metric_name="redundancy_rate",
            formula_version=FORMULA_VERSION_REDUNDANCY_MAS,
            input_count=len(stories),
            value=None,
            null_reason=(
                "RedundancyAnalysis not provided. "
                "MAS pipeline may have produced 0 stories, preventing "
                "the Redundancy Checker from running."
            ),
            interpretation="Cannot compute: no RedundancyAnalysis available.",
            limitation="Requires RedundancyAnalysis from MAS pipeline.",
        )

    total_input = redundancy_analysis.get("input_story_count", 0)
    if total_input == 0:
        return MetricObservation(
            metric_name="redundancy_rate",
            formula_version=FORMULA_VERSION_REDUNDANCY_MAS,
            input_count=0,
            value=None,
            null_reason="RedundancyAnalysis.input_story_count is 0.",
            interpretation="No input stories for redundancy checking.",
            limitation="Denominator is zero.",
        )

    duplicate_groups = redundancy_analysis.get("duplicate_groups", [])
    grouped_ids: Set[str] = set()
    for group in duplicate_groups:
        for sid in group.get("story_ids", []):
            grouped_ids.add(sid)

    numerator = len(grouped_ids)
    rate = (numerator / total_input) * 100.0

    observations = [
        StoryObservation(
            story_id=gid,
            value=True,
            detail="Was in a duplicate group before deduplication.",
        )
        for gid in sorted(grouped_ids)
    ]

    return MetricObservation(
        metric_name="redundancy_rate",
        formula_version=FORMULA_VERSION_REDUNDANCY_MAS,
        input_count=total_input,
        raw_observations=observations,
        value=rate,
        numerator=float(numerator),
        denominator=float(total_input),
        interpretation=(
            f"{numerator}/{total_input} stories ({rate:.1f}%) were in "
            f"duplicate groups before deduplication. "
            f"Groups found: {len(duplicate_groups)}. "
            f"Resolution: {redundancy_analysis.get('resolution_status', 'unknown')}."
        ),
        limitation=(
            "OPEN-021: Whether Redundancy-Unresolved stories count toward "
            "the numerator is configurable. Here, only stories in "
            "duplicate_groups are counted."
        ),
        annotation_required=False,
    )


def compute_redundancy_rate_baseline(
    stories: List[Dict[str, Any]],
    threshold: float = JACCARD_REDUNDANCY_THRESHOLD,
) -> MetricObservation:
    """
    EVAL-002 (Baseline): Post-hoc redundancy detection via Jaccard similarity
    on tokenized action + benefit fields.

    Formula: stories_in_near_duplicate_pairs / total_stories × 100
    Version: redundancy_v2_jaccard_pairwise_0.7

    This is an AUTOMATIC approximation — it does NOT require a human annotator
    or an LLM call. It serves as an objective proxy for semantic overlap.

    Jaccard similarity = |A ∩ B| / |A ∪ B| where A, B are token sets
    derived from lowercased "action + benefit" text of two stories.

    A pair is considered near-duplicate if Jaccard ≥ threshold (default 0.7).

    IMPORTANT: This is a lexical heuristic. It may miss paraphrastic
    duplicates and may flag coincidentally similar vocabulary.
    For ground truth redundancy, use the HUMAN ANNOTATION TASK below.

    Parameters
    ----------
    stories:
        List of story dicts with keys: id, action, benefit.
    threshold:
        Jaccard similarity threshold (default 0.7).

    Returns
    -------
    MetricObservation.
    """
    if not stories:
        return MetricObservation(
            metric_name="redundancy_rate",
            formula_version=FORMULA_VERSION_REDUNDANCY_BASELINE,
            input_count=0,
            value=None,
            null_reason="Empty story collection.",
            interpretation="No stories to check.",
            limitation="Empty collection.",
        )

    n = len(stories)

    def tokenize(s: Dict[str, Any]) -> Set[str]:
        text = f"{s.get('action', '')} {s.get('benefit', '')}".lower()
        return set(re.findall(r"\b\w{3,}\b", text))

    token_sets = [tokenize(s) for s in stories]

    redundant_ids: Set[str] = set()
    pair_observations: List[Tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            if not union:
                continue
            jaccard = len(a & b) / len(union)
            if jaccard >= threshold:
                redundant_ids.add(stories[i].get("id", str(i)))
                redundant_ids.add(stories[j].get("id", str(j)))
                pair_observations.append(
                    (stories[i].get("id", str(i)), stories[j].get("id", str(j)), jaccard)
                )

    observations = [
        StoryObservation(
            story_id=sid,
            value=True,
            detail=(
                f"Near-duplicate with ≥1 other story (Jaccard ≥ {threshold}). "
                f"Pairs: "
                + ", ".join(
                    f"({a}↔{b} J={j:.3f})"
                    for a, b, j in pair_observations
                    if a == sid or b == sid
                )
            ),
        )
        for sid in sorted(redundant_ids)
    ]
    # Also add non-redundant stories as negative observations
    all_ids = {s.get("id", str(i)) for i, s in enumerate(stories)}
    for sid in sorted(all_ids - redundant_ids):
        observations.append(StoryObservation(
            story_id=sid,
            value=False,
            detail=f"No near-duplicate found (Jaccard < {threshold}).",
        ))

    numerator = len(redundant_ids)
    rate = (numerator / n) * 100.0

    return MetricObservation(
        metric_name="redundancy_rate",
        formula_version=FORMULA_VERSION_REDUNDANCY_BASELINE,
        input_count=n,
        raw_observations=observations,
        value=rate,
        numerator=float(numerator),
        denominator=float(n),
        interpretation=(
            f"{numerator}/{n} stories ({rate:.1f}%) are in near-duplicate pairs "
            f"(Jaccard ≥ {threshold} on action+benefit token sets). "
            f"Near-duplicate pairs found: {len(pair_observations)}."
        ),
        limitation=(
            f"Lexical heuristic (Jaccard on tokens, threshold={threshold}). "
            "May miss paraphrastic duplicates or flag coincidental vocabulary overlap. "
            "For ground-truth redundancy, see HUMAN ANNOTATION TASK: redundancy_ground_truth."
        ),
        annotation_required=False,
    )


# ── EVAL-001: Hallucination Rate ───────────────────────────────────────────────

def compute_hallucination_rate_mas(
    stories: List[Dict[str, Any]],
) -> MetricObservation:
    """
    EVAL-001 (MAS): % of MAS stories that failed the Reviewer's
    gdd_traceability criterion.

    Formula: stories_with_reviewer_traceability_false / verifiable_stories × 100
    Version: hallucination_v1_reviewer_traceability

    Uses confidence_evidence.reviewer_traceability_confirmed field:
      True  → traceability passed (not hallucinated)
      False → traceability FAILED (potential hallucination)
      None  → unverifiable (excluded from denominator)

    LIMITATION: Uses same LLM as the Reviewer — potential circular bias.
    For independent ground truth, see HUMAN ANNOTATION TASK.

    Parameters
    ----------
    stories:
        List of story dicts with nested confidence_evidence dict.

    Returns
    -------
    MetricObservation.
    """
    if not stories:
        return MetricObservation(
            metric_name="hallucination_rate",
            formula_version=FORMULA_VERSION_HALLUCINATION_MAS,
            input_count=0,
            value=None,
            null_reason="Empty story collection.",
            interpretation="No stories.",
            limitation="Empty collection.",
        )

    observations: List[StoryObservation] = []
    hallucinated = 0
    unverifiable = 0

    for s in stories:
        sid = s.get("id", "unknown")
        ce = s.get("confidence_evidence") or {}
        confirmed = ce.get("reviewer_traceability_confirmed")

        if confirmed is None:
            unverifiable += 1
            observations.append(StoryObservation(
                story_id=sid,
                value=None,
                detail="reviewer_traceability_confirmed=None → unverifiable.",
            ))
        elif confirmed is False:
            hallucinated += 1
            observations.append(StoryObservation(
                story_id=sid,
                value=True,  # is_hallucinated = True
                detail=(
                    "reviewer_traceability_confirmed=False → "
                    "Reviewer flagged unsupported claim."
                ),
            ))
        else:
            observations.append(StoryObservation(
                story_id=sid,
                value=False,  # is_hallucinated = False
                detail="reviewer_traceability_confirmed=True → traceability passed.",
            ))

    verifiable = len(stories) - unverifiable
    if verifiable == 0:
        return MetricObservation(
            metric_name="hallucination_rate",
            formula_version=FORMULA_VERSION_HALLUCINATION_MAS,
            input_count=len(stories),
            raw_observations=observations,
            value=None,
            null_reason=(
                "No verifiable stories — all stories have "
                "reviewer_traceability_confirmed=None."
            ),
            interpretation="Cannot compute: no traceability signals.",
            limitation=(
                "All stories lack reviewer_traceability_confirmed. "
                "MAS pipeline may have produced 0 reviewed stories."
            ),
        )

    rate = (hallucinated / verifiable) * 100.0

    return MetricObservation(
        metric_name="hallucination_rate",
        formula_version=FORMULA_VERSION_HALLUCINATION_MAS,
        input_count=len(stories),
        raw_observations=observations,
        value=rate,
        numerator=float(hallucinated),
        denominator=float(verifiable),
        interpretation=(
            f"{hallucinated}/{verifiable} verifiable stories ({rate:.1f}%) "
            f"failed the Reviewer's gdd_traceability criterion. "
            f"({unverifiable} stories unverifiable — excluded from denominator.)"
        ),
        limitation=(
            "EVAL-001 MAS: Uses Reviewer Agent's gdd_traceability verdict as "
            "proxy for hallucination. Same underlying LLM — potential circular "
            "bias. Only covers stories where traceability signal is populated."
        ),
        annotation_required=False,
    )


def compute_hallucination_rate_baseline_heuristic(
    stories: List[Dict[str, Any]],
    gdd_chunk_texts: Dict[str, str],
) -> MetricObservation:
    """
    EVAL-001 (Baseline): Keyword-presence heuristic for hallucination detection.

    Formula: stories_where_action_keywords_absent_from_source / verifiable × 100
    Version: hallucination_v2_source_excerpt_keyword

    Method (fully automatic, no LLM):
    1. For each story, retrieve its source chunk text via source_chunk_ids.
    2. Tokenize the story's `action` field and the chunk text.
    3. Compute keyword overlap: if <30% of action tokens (≥3 chars) appear
       in the source chunk, flag the story as potentially hallucinated.

    This is a conservative AUTOMATIC proxy. Stories that draw on broader
    context or use paraphrasing will be over-flagged.

    IMPORTANT — HUMAN ANNOTATION REQUIRED for ground truth:
    See HUMAN_ANNOTATION_TASKS list. This heuristic is documented as
    formula_version="hallucination_v2_source_excerpt_keyword" and is
    separate from any human judgment.

    Parameters
    ----------
    stories:
        Baseline story dicts with source_chunk_ids and action fields.
    gdd_chunk_texts:
        {chunk_id: chunk_text} mapping for all source chunks.

    Returns
    -------
    MetricObservation.
    """
    if not stories:
        return MetricObservation(
            metric_name="hallucination_rate",
            formula_version=FORMULA_VERSION_HALLUCINATION_BASELINE,
            input_count=0,
            value=None,
            null_reason="Empty story collection.",
            interpretation="No stories.",
            limitation="Empty collection.",
        )

    KEYWORD_OVERLAP_THRESHOLD = 0.30

    def tokens(text: str) -> Set[str]:
        return set(re.findall(r"\b\w{3,}\b", text.lower()))

    observations: List[StoryObservation] = []
    hallucinated = 0
    unverifiable = 0

    for s in stories:
        sid = s.get("id", "unknown")
        action_text = s.get("action", "") or ""
        action_tokens = tokens(action_text)
        source_ids = s.get("source_chunk_ids") or []

        # Find first available chunk text
        chunk_text = ""
        for cid in source_ids:
            if cid in gdd_chunk_texts:
                chunk_text = gdd_chunk_texts[cid]
                break

        if not chunk_text:
            unverifiable += 1
            observations.append(StoryObservation(
                story_id=sid,
                value=None,
                detail="Source chunk text not available — unverifiable.",
            ))
            continue

        chunk_tokens = tokens(chunk_text)

        # Also check source_excerpt if present
        excerpt = ""
        ce = s.get("confidence_evidence") or {}
        if isinstance(ce, dict):
            excerpt = ce.get("source_excerpt", "") or ""
        excerpt_tokens = tokens(excerpt) if excerpt else set()

        combined_source = chunk_tokens | excerpt_tokens

        if not action_tokens:
            # Empty action — AQUSA will catch this, not hallucination
            observations.append(StoryObservation(
                story_id=sid,
                value=False,
                detail="Empty action field — not classified as hallucination.",
            ))
            continue

        overlap = len(action_tokens & combined_source) / len(action_tokens)

        is_hallucinated = overlap < KEYWORD_OVERLAP_THRESHOLD
        if is_hallucinated:
            hallucinated += 1

        observations.append(StoryObservation(
            story_id=sid,
            value=is_hallucinated,
            detail=(
                f"Action keyword overlap with source: {overlap:.2%} "
                f"(threshold={KEYWORD_OVERLAP_THRESHOLD:.0%}) → "
                f"{'HALLUCINATED' if is_hallucinated else 'SUPPORTED'}. "
                f"Action tokens: {sorted(action_tokens)[:5]}..."
            ),
        ))

    verifiable = len(stories) - unverifiable
    if verifiable == 0:
        return MetricObservation(
            metric_name="hallucination_rate",
            formula_version=FORMULA_VERSION_HALLUCINATION_BASELINE,
            input_count=len(stories),
            raw_observations=observations,
            value=None,
            null_reason="All stories lack source chunk text — cannot verify.",
            interpretation="No verifiable stories.",
            limitation="Missing chunk texts for all stories.",
        )

    rate = (hallucinated / verifiable) * 100.0

    return MetricObservation(
        metric_name="hallucination_rate",
        formula_version=FORMULA_VERSION_HALLUCINATION_BASELINE,
        input_count=len(stories),
        raw_observations=observations,
        value=rate,
        numerator=float(hallucinated),
        denominator=float(verifiable),
        interpretation=(
            f"{hallucinated}/{verifiable} verifiable stories ({rate:.1f}%) "
            f"have <{KEYWORD_OVERLAP_THRESHOLD:.0%} action keyword overlap "
            f"with their source chunk, indicating potential hallucination. "
            f"({unverifiable} unverifiable.)"
        ),
        limitation=(
            f"Lexical heuristic: keyword overlap threshold={KEYWORD_OVERLAP_THRESHOLD:.0%}. "
            "Stories using paraphrasing or broad domain knowledge will be over-flagged. "
            "For ground-truth hallucination judgments, see HUMAN_ANNOTATION_TASKS."
        ),
        annotation_required=False,
        annotation_task="hallucination_ground_truth",
    )


# ── Human Annotation Tasks ─────────────────────────────────────────────────────

HUMAN_ANNOTATION_TASKS: List[HumanAnnotationTask] = [
    HumanAnnotationTask(
        metric_name="hallucination_rate_ground_truth",
        reason=(
            "Automatic hallucination detection (keyword overlap, LLM self-check) "
            "has known limitations: (a) keyword heuristics miss paraphrasing; "
            "(b) LLM-based checks have circular bias (same model generates and judges). "
            "Ground-truth hallucination rate requires a domain expert to inspect "
            "each story against the source GDD text."
        ),
        instructions=(
            "For each user story, a human annotator must:\n"
            "1. Read the story's full_text.\n"
            "2. Read the source_chunk text referenced by source_chunk_ids.\n"
            "3. Judge: Is the story's action and benefit SUPPORTED by the "
            "source chunk (or is it fabricated/extrapolated beyond the text)?\n"
            "4. Record: story_id, is_hallucinated (bool), confidence (high/medium/low), "
            "and a one-sentence justification.\n"
            "5. Two independent annotators required; resolve disagreements "
            "via adjudication."
        ),
        annotation_schema={
            "story_id": "str — FinalUserStory.id",
            "is_hallucinated": "bool — True if action/benefit not supported by source",
            "confidence": "Literal['high','medium','low']",
            "justification": "str — one-sentence rationale",
            "annotator_id": "str — anonymized annotator identifier",
        },
    ),
    HumanAnnotationTask(
        metric_name="redundancy_ground_truth",
        reason=(
            "Automatic Jaccard-based redundancy uses lexical overlap and may miss "
            "semantic duplicates (same requirement expressed differently) or "
            "falsely flag stories that share domain vocabulary but express "
            "distinct requirements."
        ),
        instructions=(
            "For each pair of user stories from the SAME pipeline run, "
            "a human annotator must:\n"
            "1. Read both stories' full_text.\n"
            "2. Judge: Are these stories expressing the SAME functional requirement "
            "(redundant) or distinct requirements?\n"
            "3. Record: story_id_a, story_id_b, is_redundant (bool), "
            "confidence (high/medium/low), justification.\n"
            "NOTE: Only flag as redundant if both stories would be satisfied by "
            "implementing the same feature. Near-synonyms but different scopes "
            "are NOT redundant."
        ),
        annotation_schema={
            "story_id_a": "str",
            "story_id_b": "str",
            "is_redundant": "bool",
            "confidence": "Literal['high','medium','low']",
            "justification": "str",
            "annotator_id": "str",
        },
    ),
]
