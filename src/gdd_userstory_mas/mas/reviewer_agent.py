"""
reviewer_agent.py — Reviewer Agent (C7)
=========================================
Implements the Reviewer Agent as specified in Technical Architecture §2
and Data Contracts §7–§8 (SRS-011).

Responsibilities:
  - Receive ONE ``GeneratedUserStory`` draft and its source GDD chunk text.
  - Call the LLM to evaluate the draft against AQUSA-based criteria.
  - Produce a ``ReviewerResult`` verdict (Valid/Invalid).
  - If Invalid, also produce a ``ReviewerFeedback`` with actionable
    feedback for the Generator Agent's revision pass.
  - Preserve full traceability: review_id, draft_id,
    evaluated_at_iteration (TRACE-003).
  - Never rewrite or edit the user story content itself (AGENT-008).
  - Never perform redundancy checking (that is Agent 5's role).

Design decisions:
  - Reuses ``baseline.llm_client.LLMClient`` (same as other MAS agents).
  - Prompt rendering uses ``str.replace()`` (not ``.format()``).
  - Stateless: no cross-draft state.
  - ``review_id`` = ``{draft_id}__review_iter{N}`` — deterministic,
    directly traceable (TRACE-003).
  - ``feedback_id`` = ``{review_id}__feedback`` — deterministic.
  - Config loaded from ``config/mas.yaml`` via
    ``ReviewerAgentConfig.from_yaml()``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gdd_userstory_mas.baseline.llm_client import (
    LLMClient,
    LLMMalformedOutputError,
)
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory
from gdd_userstory_mas.schemas.reviewer_result import (
    VALID_CRITERIA,
    ReviewerFeedback,
    ReviewerResult,
)

logger = logging.getLogger(__name__)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class ReviewerAgentConfig:
    """
    Configuration for the Reviewer Agent.

    Loaded from ``config/mas.yaml``. Mirrors the pattern of
    ``GeneratorAgentConfig`` and other MAS agent configs.
    """

    # LLM model settings
    provider: str = "gemini"
    model_name: str = "gemini-flash-latest"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: Optional[float] = None
    response_format: str = "json_object"

    # Retry / rate-limit settings
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 13.0
    wait_on_overload_seconds: float = 60.0

    # Prompt settings
    prompt_version: str = "reviewer_v1"
    system_prompt_file: str = "config/prompts/reviewer_system_prompt.txt"

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "ReviewerAgentConfig":
        """Load ``ReviewerAgentConfig`` from ``config/mas.yaml``."""
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
        rev = raw.get("reviewer_agent", {})

        return cls(
            provider=model.get("provider", "gemini"),
            model_name=model.get("name", "gemini-flash-latest"),
            temperature=float(model.get("temperature", 0.0)),
            max_tokens=int(rev.get("max_tokens", model.get("max_tokens", 2048))),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            max_retries=int(inv.get("max_retries", 5)),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(
                inv.get("min_request_interval_seconds", 13.0)
            ),
            wait_on_overload_seconds=float(inv.get("wait_on_overload_seconds", 60.0)),
            prompt_version=rev.get("prompt_version", "reviewer_v1"),
            system_prompt_file=rev.get(
                "system_prompt_file",
                "config/prompts/reviewer_system_prompt.txt",
            ),
        )

    @classmethod
    def defaults(cls) -> "ReviewerAgentConfig":
        return cls()


# ── Reviewer Agent ─────────────────────────────────────────────────────────────

class ReviewerAgent:
    """
    Reviewer Agent (Component C7).

    Evaluates one ``GeneratedUserStory`` against the source GDD chunk text
    and returns a ``ReviewerResult`` (Valid/Invalid) plus optional
    ``ReviewerFeedback`` when Invalid.

    This agent:
    - Is STATELESS — each ``review()`` call is independent.
    - Does NOT rewrite the user story (AGENT-008).
    - Does NOT perform redundancy checking (that is Agent 5's role).
    - Produces structured, actionable feedback for the Generator's
      revision pass when the verdict is Invalid.

    Parameters
    ----------
    config:
        ``ReviewerAgentConfig`` loaded from ``config/mas.yaml``.
    api_key:
        LLM API key. If None, reads from environment.
    project_root:
        Project root for resolving prompt file paths.

    Example
    -------
    >>> agent = ReviewerAgent(config=ReviewerAgentConfig.defaults())
    >>> result, feedback = agent.review(story, source_chunk_text="...")
    >>> print(result.status)
    'Valid'
    """

    def __init__(
        self,
        config: ReviewerAgentConfig,
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
            "ReviewerAgent initialized (provider=%s, model=%s, prompt=%s)",
            config.provider,
            config.model_name,
            config.prompt_version,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def review(
        self,
        draft: GeneratedUserStory,
        source_chunk_text: str,
    ) -> Tuple[ReviewerResult, Optional[ReviewerFeedback]]:
        """
        Review a user story draft against the source GDD chunk text.

        Parameters
        ----------
        draft:
            The ``GeneratedUserStory`` to evaluate.
        source_chunk_text:
            The original GDD chunk text linked via ``source_chunk_id``.
            Required for relevance/traceability checking (hallucination
            detection).

        Returns
        -------
        tuple[ReviewerResult, ReviewerFeedback | None]
            A ``(result, feedback)`` tuple.  ``feedback`` is ``None``
            when the verdict is ``Valid``.

        Raises
        ------
        LLMAPIError
            If all LLM call attempts fail.
        LLMMalformedOutputError
            If the LLM returns invalid JSON or a schema-violating response.
        """
        logger.info(
            "Reviewing draft %s (iter=%d, chunk=%s)",
            draft.draft_id,
            draft.iteration_count,
            draft.source_chunk_id,
        )

        system_prompt = self._render_system_prompt(draft, source_chunk_text)
        user_message = self._build_user_message(draft, source_chunk_text)

        llm_response = self._client.call(system_prompt, user_message)

        logger.debug(
            "Draft %s: LLM returned %d prompt tokens, %d completion tokens",
            draft.draft_id,
            llm_response.prompt_tokens,
            llm_response.completion_tokens,
        )

        result, feedback = self._parse_and_validate(
            llm_response.content,
            draft,
        )

        logger.info(
            "Draft %s → review %s: status=%s, failing=%s",
            draft.draft_id,
            result.review_id,
            result.status,
            result.failing_criteria,
        )

        return result, feedback

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        path = self._project_root / self._config.system_prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"Reviewer Agent system prompt not found: {path}\n"
                f"Expected at: config/prompts/reviewer_system_prompt.txt"
            )
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Reviewer Agent system prompt is empty: {path}")
        logger.debug("Loaded prompt from %s (%d chars)", path, len(text))
        return text

    def _render_system_prompt(
        self,
        draft: GeneratedUserStory,
        source_chunk_text: str,
    ) -> str:
        """
        Inject draft and source context into the prompt template.
        Uses str.replace() to avoid KeyError from JSON braces in the template.
        """
        return (
            self._system_prompt_template
            .replace("{draft_id}", draft.draft_id)
            .replace("{candidate_id}", draft.candidate_id)
            .replace("{source_chunk_id}", draft.source_chunk_id)
            .replace("{role}", draft.role)
            .replace("{action}", draft.action)
            .replace("{benefit}", draft.benefit)
            .replace("{full_text}", draft.full_text)
            .replace("{source_chunk_text}", source_chunk_text)
        )

    def _build_user_message(
        self,
        draft: GeneratedUserStory,
        source_chunk_text: str,
    ) -> str:
        """Build the user-turn message containing the draft for review."""
        return (
            f"Review this user story draft:\n\n"
            f"Draft ID      : {draft.draft_id}\n"
            f"Candidate ID  : {draft.candidate_id}\n"
            f"Source Chunk   : {draft.source_chunk_id}\n"
            f"Iteration     : {draft.iteration_count}\n"
            f"Role          : {draft.role}\n"
            f"Action        : {draft.action}\n"
            f"Benefit       : {draft.benefit}\n"
            f"Full Text     : {draft.full_text}\n\n"
            f"Source GDD Chunk:\n{source_chunk_text}\n\n"
            f"Evaluate and return your JSON verdict."
        )

    def _parse_and_validate(
        self,
        raw_content: str,
        draft: GeneratedUserStory,
    ) -> Tuple[ReviewerResult, Optional[ReviewerFeedback]]:
        """
        Parse the LLM JSON and build validated ReviewerResult + optional
        ReviewerFeedback.

        Injects all traceability fields from the draft so they cannot
        be corrupted by LLM output.
        """
        # Step 1: Parse JSON
        try:
            data: Dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Reviewer Agent: LLM returned non-JSON for draft "
                f"{draft.draft_id}: {raw_content[:300]!r}"
            ) from exc

        # Step 2: Extract status
        status = str(data.get("status", "")).strip()
        if status not in ("Valid", "Invalid"):
            raise LLMMalformedOutputError(
                f"Reviewer Agent: 'status' must be 'Valid' or 'Invalid', "
                f"got {status!r} for draft {draft.draft_id}."
            )

        # Step 3: Extract and validate failing_criteria
        raw_criteria = data.get("failing_criteria", [])
        if not isinstance(raw_criteria, list):
            raise LLMMalformedOutputError(
                f"Reviewer Agent: 'failing_criteria' must be a list, "
                f"got {type(raw_criteria).__name__} for draft {draft.draft_id}."
            )

        # Filter to valid criteria and deduplicate while preserving order
        failing_criteria: List[str] = []
        seen = set()
        for c in raw_criteria:
            c_str = str(c).strip().lower()
            if c_str in VALID_CRITERIA and c_str not in seen:
                failing_criteria.append(c_str)
                seen.add(c_str)
            elif c_str not in VALID_CRITERIA:
                logger.warning(
                    "Reviewer Agent: ignoring unknown criterion %r from LLM "
                    "for draft %s",
                    c_str,
                    draft.draft_id,
                )

        # Step 4: Consistency enforcement
        # If status=Valid but criteria non-empty → force Valid (trust status)
        if status == "Valid" and failing_criteria:
            logger.warning(
                "Reviewer Agent: LLM said Valid but provided failing_criteria "
                "%s for draft %s — clearing criteria to trust Valid verdict.",
                failing_criteria,
                draft.draft_id,
            )
            failing_criteria = []

        # If status=Invalid but no criteria → raise error (cannot proceed)
        if status == "Invalid" and not failing_criteria:
            raise LLMMalformedOutputError(
                f"Reviewer Agent: LLM said Invalid but provided no "
                f"failing_criteria for draft {draft.draft_id}. "
                f"VAL-023 requires at least one criterion."
            )

        # Step 5: Extract feedback fields
        feedback_text = str(data.get("feedback_text", "")).strip()
        unsupported_claim = data.get("unsupported_claim_excerpt")
        if unsupported_claim is not None:
            unsupported_claim = str(unsupported_claim).strip() or None

        # Step 6: Build deterministic IDs
        review_id = f"{draft.draft_id}__review_iter{draft.iteration_count}"
        feedback_id = f"{review_id}__feedback"

        # Step 7: Build ReviewerResult
        feedback_ref = feedback_id if status == "Invalid" else None

        try:
            result = ReviewerResult(
                review_id=review_id,
                draft_id=draft.draft_id,
                status=status,
                evaluated_at_iteration=draft.iteration_count,
                failing_criteria=failing_criteria,
                feedback_ref=feedback_ref,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMMalformedOutputError(
                f"Reviewer Agent: ReviewerResult schema validation failed "
                f"for draft {draft.draft_id}: {exc}"
            ) from exc

        # Step 8: Build ReviewerFeedback if Invalid
        feedback: Optional[ReviewerFeedback] = None
        if status == "Invalid":
            if not feedback_text:
                feedback_text = (
                    f"Draft failed criteria: {', '.join(failing_criteria)}. "
                    f"Please revise accordingly."
                )
            try:
                feedback = ReviewerFeedback(
                    feedback_id=feedback_id,
                    review_id=review_id,
                    draft_id=draft.draft_id,
                    failing_criteria=failing_criteria,
                    feedback_text=feedback_text,
                    unsupported_claim_excerpt=unsupported_claim,
                )
            except Exception as exc:  # noqa: BLE001
                raise LLMMalformedOutputError(
                    f"Reviewer Agent: ReviewerFeedback schema validation "
                    f"failed for draft {draft.draft_id}: {exc}"
                ) from exc

        return result, feedback
