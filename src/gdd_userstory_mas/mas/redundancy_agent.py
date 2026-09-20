"""
redundancy_agent.py — Redundancy Checker Agent (C8 / Agent 5)
===============================================================
Implements the Redundancy Checker Agent as specified in MAS Detailed
Architecture §1 (Agent 5) and Data Contracts §9 (SRS-012/SRS-013).

Responsibilities:
  - Receive the COMPLETE collection of Reviewer-validated user stories.
  - Detect semantically duplicate or substantially overlapping stories
    using LLM-based pairwise comparison.
  - Resolve each duplicate group: merge or remove inferior.
  - Produce a cleaned story collection with full traceability.
  - Preserve merge_history for every merge/remove action (TRACE-004).
  - Never introduce new requirement content during a merge.
  - Never alter unique (non-duplicate) stories.

Design decisions:
  - Uses LLM-only judgment (no embedding dependency) — OPEN-007 resolved
    as llm_judgment for simplicity and fairness with single Gemini model.
  - Detection + resolution in a single LLM call to minimize API usage
    on free-tier rate limits.
  - Reuses ``baseline.llm_client.LLMClient``.
  - Stateless per call; operates on the whole collection at once.
  - Config loaded from ``config/mas.yaml`` under ``redundancy_agent:``.
  - Deterministic IDs: ``analysis_id = redundancy__{run_id}__pass{N}``,
    ``group_id = dup_group_{i}``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from gdd_userstory_mas.baseline.llm_client import (
    LLMClient,
    LLMMalformedOutputError,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.redundancy_analysis import (
    DuplicateGroup,
    MergeAction,
    RedundancyAnalysis,
)

logger = logging.getLogger(__name__)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class RedundancyAgentConfig:
    """
    Configuration for the Redundancy Checker Agent.

    Loaded from ``config/mas.yaml``.  Mirrors the pattern of other
    MAS agent configs.
    """

    # LLM model settings
    provider: str = "gemini"
    model_name: str = "gemini-flash-latest"
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: Optional[float] = None
    response_format: str = "json_object"

    # Retry / rate-limit settings
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 13.0
    wait_on_overload_seconds: float = 60.0

    # Prompt settings
    prompt_version: str = "redundancy_v1"
    system_prompt_file: str = "config/prompts/redundancy_system_prompt.txt"

    # Redundancy-specific settings
    max_resolution_passes: int = 2

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "RedundancyAgentConfig":
        """Load ``RedundancyAgentConfig`` from ``config/mas.yaml``."""
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
        red = raw.get("redundancy_agent", {})

        return cls(
            provider=model.get("provider", "gemini"),
            model_name=model.get("name", "gemini-flash-latest"),
            temperature=float(model.get("temperature", 0.0)),
            max_tokens=int(red.get("max_tokens", model.get("max_tokens", 4096))),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            max_retries=int(inv.get("max_retries", 5)),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(
                inv.get("min_request_interval_seconds", 13.0)
            ),
            wait_on_overload_seconds=float(inv.get("wait_on_overload_seconds", 60.0)),
            prompt_version=red.get("prompt_version", "redundancy_v1"),
            system_prompt_file=red.get(
                "system_prompt_file",
                "config/prompts/redundancy_system_prompt.txt",
            ),
            max_resolution_passes=int(red.get("max_resolution_passes", 2)),
        )

    @classmethod
    def defaults(cls) -> "RedundancyAgentConfig":
        return cls()


# ── Redundancy Checker Agent ──────────────────────────────────────────────────

class RedundancyCheckerAgent:
    """
    Redundancy Checker Agent (Component C8 / Agent 5).

    Analyzes the complete collection of Reviewer-validated user stories
    to detect and resolve semantic duplicates.

    This agent:
    - Operates on the WHOLE collection — never on individual stories.
    - Does NOT alter unique (non-duplicate) stories.
    - Does NOT introduce new requirement content during a merge.
    - Preserves full traceability: merge_history, source_chunk_ids
      (TRACE-004).

    Parameters
    ----------
    config:
        ``RedundancyAgentConfig`` loaded from ``config/mas.yaml``.
    api_key:
        LLM API key. If None, reads from environment.
    project_root:
        Project root for resolving prompt file paths.
    """

    def __init__(
        self,
        config: RedundancyAgentConfig,
        api_key: Optional[str] = None,
        project_root: Path = Path("."),
    ) -> None:
        self._config = config
        self._project_root = Path(project_root)
        self._system_prompt_template = self._load_prompt()

        self._client = LLMClient(
            provider=config.provider,
            model_name=config.model_name,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            response_format=config.response_format,
            max_retries=config.max_retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
            min_request_interval_seconds=config.min_request_interval_seconds,
            wait_on_overload_seconds=config.wait_on_overload_seconds,
        )

        logger.info(
            "RedundancyCheckerAgent initialized (provider=%s, model=%s, "
            "prompt=%s, max_passes=%d)",
            config.provider,
            config.model_name,
            config.prompt_version,
            config.max_resolution_passes,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        stories: List[GeneratedUserStory],
        run_id: str,
    ) -> RedundancyAnalysis:
        """
        Analyze the complete story collection for redundancy.

        Parameters
        ----------
        stories:
            The full list of Reviewer-validated ``GeneratedUserStory``
            objects.
        run_id:
            The experiment run identifier.

        Returns
        -------
        RedundancyAnalysis
            Detection results, resolution actions, and metadata.

        Raises
        ------
        LLMAPIError
            If all LLM call attempts fail.
        LLMMalformedOutputError
            If the LLM returns invalid JSON or a schema-violating response.
        """
        logger.info(
            "Redundancy check starting: %d stories, run=%s",
            len(stories),
            run_id,
        )

        # Edge case: 0 or 1 stories — no redundancy possible
        if len(stories) <= 1:
            logger.info("≤1 stories — no redundancy possible.")
            return RedundancyAnalysis(
                analysis_id=f"redundancy__{run_id}__pass1",
                run_id=run_id,
                analyzed_at_pass=1,
                duplicate_groups=[],
                unique_story_ids=[s.draft_id for s in stories],
                resolution_status="resolved",
                merge_actions=[],
                resolved_story_count=len(stories),
                input_story_count=len(stories),
            )

        # Build story lookup
        story_map: Dict[str, GeneratedUserStory] = {
            s.draft_id: s for s in stories
        }

        # Run detection + resolution
        pass_num = 1
        duplicate_groups, merge_actions = self._detect_and_resolve(
            stories, pass_num
        )

        # Determine unique stories (those not in any duplicate group)
        grouped_ids: Set[str] = set()
        for g in duplicate_groups:
            grouped_ids.update(g.story_ids)

        unique_ids = [s.draft_id for s in stories if s.draft_id not in grouped_ids]

        # Determine resolution status
        if not duplicate_groups:
            resolution_status = "resolved"
        elif len(merge_actions) == len(duplicate_groups):
            resolution_status = "resolved"
        elif pass_num >= self._config.max_resolution_passes:
            resolution_status = "unresolved_max_iterations"
        else:
            resolution_status = "resolved"

        # Calculate resolved story count
        resolved_count = len(unique_ids) + len(merge_actions)

        analysis = RedundancyAnalysis(
            analysis_id=f"redundancy__{run_id}__pass{pass_num}",
            run_id=run_id,
            analyzed_at_pass=pass_num,
            duplicate_groups=duplicate_groups,
            unique_story_ids=unique_ids,
            resolution_status=resolution_status,
            merge_actions=merge_actions,
            resolved_story_count=resolved_count,
            input_story_count=len(stories),
        )

        logger.info(
            "Redundancy check complete: %d groups found, %d actions, "
            "status=%s, %d→%d stories",
            len(duplicate_groups),
            len(merge_actions),
            resolution_status,
            len(stories),
            resolved_count,
        )

        return analysis

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        path = self._project_root / self._config.system_prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"Redundancy Checker system prompt not found: {path}\n"
                f"Expected at: config/prompts/redundancy_system_prompt.txt"
            )
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(
                f"Redundancy Checker system prompt is empty: {path}"
            )
        logger.debug("Loaded prompt from %s (%d chars)", path, len(text))
        return text

    def _format_stories_text(self, stories: List[GeneratedUserStory]) -> str:
        """Format the story collection for the LLM prompt."""
        lines = []
        for i, s in enumerate(stories):
            lines.append(
                f"[{i+1}] draft_id: {s.draft_id}\n"
                f"    source_chunk_id: {s.source_chunk_id}\n"
                f"    role: {s.role}\n"
                f"    action: {s.action}\n"
                f"    benefit: {s.benefit}\n"
                f"    full_text: {s.full_text}"
            )
        return "\n\n".join(lines)

    def _render_system_prompt(self, stories: List[GeneratedUserStory]) -> str:
        """Inject the stories list into the prompt template."""
        stories_text = self._format_stories_text(stories)
        return self._system_prompt_template.replace(
            "{stories_text}", stories_text
        )

    def _build_user_message(self, stories: List[GeneratedUserStory]) -> str:
        """Build the user-turn message."""
        return (
            f"Analyze the {len(stories)} user stories above for "
            f"semantic redundancy. Identify duplicate groups and provide "
            f"resolution actions (merge or remove_inferior). "
            f"Return your JSON response."
        )

    def _detect_and_resolve(
        self,
        stories: List[GeneratedUserStory],
        pass_num: int,
    ) -> Tuple[List[DuplicateGroup], List[MergeAction]]:
        """
        Send stories to LLM for combined detection + resolution.

        Returns (duplicate_groups, merge_actions).
        """
        system_prompt = self._render_system_prompt(stories)
        user_message = self._build_user_message(stories)

        llm_response = self._client.call(system_prompt, user_message)

        logger.debug(
            "Redundancy pass %d: LLM returned %d prompt tokens, "
            "%d completion tokens",
            pass_num,
            llm_response.prompt_tokens,
            llm_response.completion_tokens,
        )

        return self._parse_and_validate(llm_response.content, stories)

    def _parse_and_validate(
        self,
        raw_content: str,
        stories: List[GeneratedUserStory],
    ) -> Tuple[List[DuplicateGroup], List[MergeAction]]:
        """
        Parse LLM JSON and build validated DuplicateGroups + MergeActions.
        """
        # Step 1: Parse JSON
        try:
            data: Dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Redundancy Checker: LLM returned non-JSON: "
                f"{raw_content[:300]!r}"
            ) from exc

        # Build valid story ID set for validation
        valid_ids: Set[str] = {s.draft_id for s in stories}
        story_map: Dict[str, GeneratedUserStory] = {
            s.draft_id: s for s in stories
        }

        # Step 2: Parse duplicate groups
        raw_groups = data.get("duplicate_groups", [])
        if not isinstance(raw_groups, list):
            raise LLMMalformedOutputError(
                f"Redundancy Checker: 'duplicate_groups' must be a list, "
                f"got {type(raw_groups).__name__}."
            )

        duplicate_groups: List[DuplicateGroup] = []
        seen_story_ids: Set[str] = set()

        for i, rg in enumerate(raw_groups):
            if not isinstance(rg, dict):
                logger.warning("Skipping non-dict duplicate group at index %d", i)
                continue

            group_id = str(rg.get("group_id", f"dup_group_{i}")).strip()
            raw_story_ids = rg.get("story_ids", [])
            if not isinstance(raw_story_ids, list):
                logger.warning("Skipping group %s: story_ids not a list", group_id)
                continue

            # Filter to valid story IDs and skip already-grouped stories
            story_ids = []
            for sid in raw_story_ids:
                sid_str = str(sid).strip()
                if sid_str in valid_ids and sid_str not in seen_story_ids:
                    story_ids.append(sid_str)

            if len(story_ids) < 2:
                logger.warning(
                    "Skipping group %s: fewer than 2 valid story IDs "
                    "after filtering (had %d raw, %d valid)",
                    group_id, len(raw_story_ids), len(story_ids),
                )
                continue

            seen_story_ids.update(story_ids)

            similarity_basis = str(
                rg.get("similarity_basis", "llm_judgment")
            ).strip()

            try:
                group = DuplicateGroup(
                    group_id=group_id,
                    story_ids=story_ids,
                    similarity_basis=similarity_basis,
                    similarity_score=None,
                )
                duplicate_groups.append(group)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed group %s: %s", group_id, exc
                )

        # Step 3: Parse resolutions
        raw_resolutions = data.get("resolutions", [])
        if not isinstance(raw_resolutions, list):
            raw_resolutions = []

        merge_actions: List[MergeAction] = []
        group_id_set = {g.group_id for g in duplicate_groups}

        for j, rr in enumerate(raw_resolutions):
            if not isinstance(rr, dict):
                continue

            res_group_id = str(rr.get("group_id", "")).strip()
            if res_group_id not in group_id_set:
                logger.warning(
                    "Skipping resolution for unknown group %s", res_group_id
                )
                continue

            action = str(rr.get("action", "")).strip().lower()
            if action not in ("merge", "remove_inferior"):
                logger.warning(
                    "Skipping resolution for group %s: invalid action %r",
                    res_group_id, action,
                )
                continue

            kept_id = str(rr.get("kept_story_id", "")).strip()
            removed_ids = [
                str(rid).strip()
                for rid in rr.get("removed_story_ids", [])
                if str(rid).strip() in valid_ids
            ]

            if not kept_id or not removed_ids:
                logger.warning(
                    "Skipping resolution for group %s: missing kept/removed IDs",
                    res_group_id,
                )
                continue

            # Collect all source_chunk_ids from involved stories
            involved_ids = [kept_id] + removed_ids
            source_chunk_ids = list(dict.fromkeys(
                story_map[sid].source_chunk_id
                for sid in involved_ids
                if sid in story_map
            ))
            if not source_chunk_ids:
                source_chunk_ids = ["unknown"]

            reason = str(rr.get("reason", "Duplicate detected by LLM.")).strip()

            # Build merge fields
            merged_role = None
            merged_action_text = None
            merged_benefit = None
            merged_full_text = None

            if action == "merge":
                merged_role = str(rr.get("merged_role", "")).strip() or None
                merged_action_text = str(
                    rr.get("merged_action", "")
                ).strip() or None
                merged_benefit = str(
                    rr.get("merged_benefit", "")
                ).strip() or None

                if merged_role and merged_action_text and merged_benefit:
                    # Strip prefixes if LLM included them
                    merged_role = self._strip_prefix(
                        merged_role, ["as a ", "as an "]
                    )
                    merged_action_text = self._strip_prefix(
                        merged_action_text, ["i want ", "to "]
                    )
                    merged_benefit = self._strip_prefix(
                        merged_benefit, ["so that ", "in order to "]
                    )
                    merged_full_text = (
                        f"As a {merged_role}, I want {merged_action_text}, "
                        f"so that {merged_benefit}."
                    )
                else:
                    # Fallback: if LLM didn't provide merge fields,
                    # keep the best story unchanged (downgrade to remove)
                    logger.warning(
                        "Group %s: merge requested but LLM didn't provide "
                        "all merged fields — falling back to remove_inferior",
                        res_group_id,
                    )
                    action = "remove_inferior"
                    merged_role = None
                    merged_action_text = None
                    merged_benefit = None
                    merged_full_text = None

            try:
                merge_action = MergeAction(
                    group_id=res_group_id,
                    action=action,
                    kept_story_id=kept_id,
                    removed_story_ids=removed_ids,
                    merged_role=merged_role,
                    merged_action=merged_action_text,
                    merged_benefit=merged_benefit,
                    merged_full_text=merged_full_text,
                    source_chunk_ids=source_chunk_ids,
                    reason=reason,
                )
                merge_actions.append(merge_action)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed resolution for group %s: %s",
                    res_group_id, exc,
                )

        return duplicate_groups, merge_actions

    @staticmethod
    def _strip_prefix(text: str, prefixes: List[str]) -> str:
        """Strip any matching prefix (case-insensitive)."""
        lower = text.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return text[len(prefix):].strip()
        return text
