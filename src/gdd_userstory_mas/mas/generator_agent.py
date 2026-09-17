"""
generator_agent.py — User Story Generator Agent (C6)
======================================================
Implements the User Story Generator Agent as specified in Technical
Architecture §2 and Data Contracts §6 (SRS-010).

Responsibilities:
  - Receive ONE ``CandidateRequirement`` at a time (stateless).
  - Call the LLM to generate a single user story with an explicit role,
    action, and benefit.
  - Produce a ``GeneratedUserStory`` that strictly follows the format:
    "As a [role], I want [action], so that [benefit]." (VAL-020).
  - Preserve full source traceability: draft_id, candidate_id,
    source_chunk_id, document_id (TRACE-001, TRACE-003).
  - Never invent information not present in the candidate requirement.

Design decisions:
  - Reuses ``baseline.llm_client.LLMClient`` (same as other MAS agents).
  - Prompt rendering uses ``str.replace()`` (not ``.format()``).
  - Stateless: no cross-candidate state.
  - ``draft_id`` = ``{candidate_id}__draft`` — stable, deterministic,
    directly traceable to its candidate (TRACE-003).
  - ``iteration_count`` = 0 for fresh generation.  The Reviewer feedback
    loop (C7) is responsible for revision; this agent only produces the
    initial draft (and revised drafts when given feedback — but that
    orchestration logic is in C3, not here).
  - Config loaded from ``config/mas.yaml`` via
    ``GeneratorAgentConfig.from_yaml()``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gdd_userstory_mas.baseline.llm_client import (
    LLMClient,
    LLMMalformedOutputError,
)
from gdd_userstory_mas.schemas.candidate_requirement import CandidateRequirement
from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory

logger = logging.getLogger(__name__)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class GeneratorAgentConfig:
    """
    Configuration for the User Story Generator Agent.

    Loaded from ``config/mas.yaml``. Mirrors the pattern of
    ``ReaderAgentConfig`` and ``AnalystAgentConfig``.
    """

    # LLM model settings
    provider: str = "gemini"
    model_name: str = "gemini-flash-latest"
    temperature: float = 0.0
    max_tokens: int = 1024   # shorter than other agents — one story at a time
    top_p: Optional[float] = None
    response_format: str = "json_object"

    # Retry / rate-limit settings
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 13.0
    wait_on_overload_seconds: float = 60.0

    # Prompt settings
    prompt_version: str = "generator_v1"
    system_prompt_file: str = "config/prompts/generator_system_prompt.txt"

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "GeneratorAgentConfig":
        """Load ``GeneratorAgentConfig`` from ``config/mas.yaml``."""
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
        gen = raw.get("generator_agent", {})

        return cls(
            provider=model.get("provider", "gemini"),
            model_name=model.get("name", "gemini-flash-latest"),
            temperature=float(model.get("temperature", 0.0)),
            max_tokens=int(gen.get("max_tokens", model.get("max_tokens", 1024))),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            max_retries=int(inv.get("max_retries", 5)),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(
                inv.get("min_request_interval_seconds", 13.0)
            ),
            wait_on_overload_seconds=float(inv.get("wait_on_overload_seconds", 60.0)),
            prompt_version=gen.get("prompt_version", "generator_v1"),
            system_prompt_file=gen.get(
                "system_prompt_file",
                "config/prompts/generator_system_prompt.txt",
            ),
        )

    @classmethod
    def defaults(cls) -> "GeneratorAgentConfig":
        return cls()


# ── User Story Generator Agent ─────────────────────────────────────────────────

class UserStoryGeneratorAgent:
    """
    User Story Generator Agent (Component C6).

    Generates one ``GeneratedUserStory`` from one ``CandidateRequirement``.

    This agent:
    - Is STATELESS — each ``generate()`` call is independent.
    - Does NOT implement the Reviewer feedback loop (that is C3/C7).
    - Produces a first draft (``iteration_count=0``) or a revised draft
      (``iteration_count=N``) — the caller controls the iteration state.
    - Validates the LLM output against the ``GeneratedUserStory`` schema
      before returning.

    Parameters
    ----------
    config:
        ``GeneratorAgentConfig`` loaded from ``config/mas.yaml``.
    api_key:
        LLM API key. If None, reads from environment.
    project_root:
        Project root for resolving prompt file paths.

    Example
    -------
    >>> agent = UserStoryGeneratorAgent(config=GeneratorAgentConfig.defaults())
    >>> story = agent.generate(candidate)
    >>> print(story.full_text)
    'As a player, I want to attack enemies with melee weapons, so that I can defeat them.'
    """

    def __init__(
        self,
        config: GeneratorAgentConfig,
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
            "UserStoryGeneratorAgent initialized (provider=%s, model=%s, prompt=%s)",
            config.provider,
            config.model_name,
            config.prompt_version,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        candidate: CandidateRequirement,
        iteration_count: int = 0,
        previous_draft_id_chain: Optional[List[str]] = None,
    ) -> GeneratedUserStory:
        """
        Generate a user story draft from one candidate requirement.

        Parameters
        ----------
        candidate:
            The ``CandidateRequirement`` to convert into a user story.
        iteration_count:
            0 for a fresh first draft.  Pass the current iteration number
            when re-generating after Reviewer feedback (VAL-022).
        previous_draft_id_chain:
            Prior draft_id chain for audit trail; empty for first draft.

        Returns
        -------
        GeneratedUserStory
            Validated draft with full traceability.

        Raises
        ------
        LLMAPIError
            If all LLM call attempts fail.
        LLMMalformedOutputError
            If the LLM returns invalid JSON or a schema-violating response.
        """
        logger.info(
            "Generating story for candidate %s (iter=%d, domain=%s, perspective=%s)",
            candidate.candidate_id,
            iteration_count,
            candidate.domain,
            candidate.perspective,
        )

        system_prompt = self._render_system_prompt(candidate)
        user_message = self._build_user_message(candidate)

        llm_response = self._client.call(system_prompt, user_message)

        logger.debug(
            "Candidate %s: LLM returned %d prompt tokens, %d completion tokens",
            candidate.candidate_id,
            llm_response.prompt_tokens,
            llm_response.completion_tokens,
        )

        story = self._parse_and_validate(
            llm_response.content,
            candidate,
            iteration_count,
            previous_draft_id_chain or [],
        )

        logger.info(
            "Candidate %s → draft %s (iter=%d): %r",
            candidate.candidate_id,
            story.draft_id,
            story.iteration_count,
            story.full_text[:80],
        )

        return story

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        path = self._project_root / self._config.system_prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"Generator Agent system prompt not found: {path}\n"
                f"Expected at: config/prompts/generator_system_prompt.txt"
            )
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Generator Agent system prompt is empty: {path}")
        logger.debug("Loaded prompt from %s (%d chars)", path, len(text))
        return text

    def _render_system_prompt(self, candidate: CandidateRequirement) -> str:
        """
        Inject candidate context into the prompt template.
        Uses str.replace() to avoid KeyError from JSON braces in the template.
        """
        ref = candidate.source_reader_output_ref or "N/A"
        return (
            self._system_prompt_template
            .replace("{candidate_id}", candidate.candidate_id)
            .replace("{requirement_text}", candidate.requirement_text)
            .replace("{perspective}", candidate.perspective)
            .replace("{domain}", candidate.domain)
            .replace("{source_ref}", ref)
        )

    def _build_user_message(self, candidate: CandidateRequirement) -> str:
        """Build the user-turn message containing the requirement."""
        return (
            f"Candidate requirement:\n"
            f"  ID         : {candidate.candidate_id}\n"
            f"  Text       : {candidate.requirement_text}\n"
            f"  Perspective: {candidate.perspective}\n"
            f"  Domain     : {candidate.domain}\n"
            f"  Source ref : {candidate.source_reader_output_ref or 'N/A'}\n\n"
            f"Generate one user story JSON for this requirement."
        )

    def _parse_and_validate(
        self,
        raw_content: str,
        candidate: CandidateRequirement,
        iteration_count: int,
        previous_draft_id_chain: List[str],
    ) -> GeneratedUserStory:
        """
        Parse the LLM JSON and build a validated ``GeneratedUserStory``.

        Injects all traceability fields from the candidate so they cannot
        be corrupted by LLM output.
        """
        # Step 1: Parse JSON
        try:
            data: Dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Generator Agent: LLM returned non-JSON for candidate "
                f"{candidate.candidate_id}: {raw_content[:300]!r}"
            ) from exc

        # Step 2: Extract and clean story fields
        role = str(data.get("role", "")).strip()
        action = str(data.get("action", "")).strip()
        benefit = str(data.get("benefit", "")).strip()

        if not role:
            raise LLMMalformedOutputError(
                f"Generator Agent: 'role' is empty for candidate "
                f"{candidate.candidate_id}. LLM response: {raw_content[:200]!r}"
            )
        if not action:
            raise LLMMalformedOutputError(
                f"Generator Agent: 'action' is empty for candidate "
                f"{candidate.candidate_id}. LLM response: {raw_content[:200]!r}"
            )
        if not benefit:
            raise LLMMalformedOutputError(
                f"Generator Agent: 'benefit' is empty for candidate "
                f"{candidate.candidate_id}. LLM response: {raw_content[:200]!r}"
            )

        # Step 3: Strip "As a/I want/so that" if LLM included them in fields
        role = self._strip_prefix(role, ["as a ", "as an "])
        action = self._strip_prefix(action, ["i want ", "to "])
        benefit = self._strip_prefix(benefit, ["so that ", "in order to "])

        # Step 4: Build canonical full_text from components
        full_text = GeneratedUserStory.build_full_text(role, action, benefit)

        # Step 5: Deterministic draft_id (TRACE-003)
        draft_id = f"{candidate.candidate_id}__draft"

        # Step 6: Build and validate schema
        try:
            return GeneratedUserStory(
                draft_id=draft_id,
                candidate_id=candidate.candidate_id,
                source_chunk_id=candidate.source_chunk_id,
                document_id=candidate.document_id,
                role=role,
                action=action,
                benefit=benefit,
                full_text=full_text,
                iteration_count=iteration_count,
                previous_draft_id_chain=previous_draft_id_chain,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMMalformedOutputError(
                f"Generator Agent: GeneratedUserStory schema validation failed "
                f"for candidate {candidate.candidate_id}: {exc}\n"
                f"role={role!r}, action={action!r}, benefit={benefit!r}"
            ) from exc

    @staticmethod
    def _strip_prefix(text: str, prefixes: List[str]) -> str:
        """
        Strip any matching prefix (case-insensitive) from a string.

        Prevents the LLM from accidentally including "As a", "I want",
        or "so that" inside the role/action/benefit fields themselves.
        """
        lower = text.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return text[len(prefix):].strip()
        return text
