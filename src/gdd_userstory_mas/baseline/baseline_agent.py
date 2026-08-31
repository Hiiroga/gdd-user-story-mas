"""
baseline_agent.py — SRS-015 core extraction logic
===================================================
The single LLM call per chunk that replaces the entire MAS pipeline
(Reader → Analyst → Generator → Reviewer → Redundancy Checker).

Design decisions:
  * One call per ``GDDChunk`` (OPEN-016 resolved as "per_chunk").
    This matches the MAS Reader Agent's granularity exactly, ensuring
    the baseline sees the same context window size and boundary conditions
    as each MAS agent call.  Experimental fairness is preserved at the
    input level (DATA-005, NFR-006).
  * The prompt is loaded from the versioned file specified in config
    (``config/prompts/baseline_system_prompt.txt``) so prompt changes
    are tracked independently from code changes.
  * Schema validation of the LLM output happens here (before the caller
    receives results) so malformed output is caught early and retried at
    the pipeline level.
  * The agent is STATELESS — no conversation history across chunks.
    Each chunk is processed independently, matching the MAS Reader
    Agent's independent-per-chunk design.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from gdd_userstory_mas.baseline.llm_client import LLMClient, LLMMalformedOutputError
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk

logger = logging.getLogger(__name__)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class BaselineAgentConfig:
    """
    Configuration passed to BaselineAgent at construction time.

    All values come from ``config/baseline.yaml`` and are snapshotted
    alongside the run output for reproducibility (Technical Architecture §8).
    """

    system_prompt_path: Path
    """Absolute path to the system prompt template file."""

    max_stories_per_chunk: int = 10
    """Soft upper bound on stories per chunk (hint to LLM, not enforced)."""

    prompt_version: str = "baseline_v1"
    """Prompt version string; written to run metadata."""


# ── Raw LLM story dataclass (pre-schema-validation) ───────────────────────────

@dataclass
class RawStory:
    """
    A single story extracted from an LLM response before Pydantic validation.
    Holds the raw dict from the LLM JSON to allow graceful partial-failure
    handling (log invalid stories without halting the whole chunk).
    """

    role: str
    action: str
    benefit: str
    full_text: str
    requirement_type: str
    game_domain: str
    source_excerpt: Optional[str]
    parse_errors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "RawStory":
        """Parse one story dict from the LLM response."""
        errors: List[str] = []
        for required_field in ("role", "action", "benefit", "full_text",
                               "requirement_type", "game_domain"):
            if not d.get(required_field):
                errors.append(f"Missing or empty field: '{required_field}'")

        return cls(
            role=d.get("role", ""),
            action=d.get("action", ""),
            benefit=d.get("benefit", ""),
            full_text=d.get("full_text", ""),
            requirement_type=d.get("requirement_type", ""),
            game_domain=d.get("game_domain", "other"),
            source_excerpt=d.get("source_excerpt"),
            parse_errors=errors,
        )


# ── Agent result dataclass ─────────────────────────────────────────────────────

@dataclass
class BaselineAgentResult:
    """
    Result of processing one GDDChunk through the baseline agent.

    Attributes
    ----------
    chunk_id:
        The chunk that was processed.
    raw_stories:
        All parsed story dicts from the LLM (some may have parse_errors).
    valid_stories:
        Subset of raw_stories with no parse_errors.
    invalid_stories:
        Subset with parse_errors (logged but not included in final output).
    prompt_tokens:
        LLM prompt token count (for cost tracking / logging).
    completion_tokens:
        LLM completion token count.
    model_name:
        Exact model version string reported by the provider.
    llm_response_raw:
        Raw JSON string returned by the LLM.
    """

    chunk_id: str
    raw_stories: List[RawStory]
    valid_stories: List[RawStory]
    invalid_stories: List[RawStory]
    prompt_tokens: int
    completion_tokens: int
    model_name: str
    llm_response_raw: str


# ── Agent class ────────────────────────────────────────────────────────────────

class BaselineAgent:
    """
    Single-agent baseline extractor (C10 — SRS-015).

    Processes one ``GDDChunk`` per call and returns extracted user story
    candidates as ``BaselineAgentResult``.  Does NOT produce
    ``FinalUserStory`` objects — that responsibility belongs to the
    pipeline orchestrator (``pipeline.py``) which applies IDs, run_id,
    and validation metadata.

    Parameters
    ----------
    llm_client:
        Initialised ``LLMClient`` instance.
    config:
        ``BaselineAgentConfig`` controlling the prompt and extraction limits.
    """

    def __init__(self, llm_client: LLMClient, config: BaselineAgentConfig) -> None:
        self._client = llm_client
        self._config = config
        self._system_prompt_template = self._load_prompt_template()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_chunk(
        self,
        chunk: GDDChunk,
        total_chunks: int,
    ) -> BaselineAgentResult:
        """
        Process a single ``GDDChunk`` and return extracted user stories.

        Parameters
        ----------
        chunk:
            The preprocessed GDD chunk to process.
        total_chunks:
            Total number of chunks in the document (for context labelling
            in the prompt, so the LLM knows its positional context).

        Returns
        -------
        BaselineAgentResult
            Contains valid and invalid story candidates.

        Raises
        ------
        LLMMalformedOutputError
            If the LLM response cannot be parsed as JSON.
        """
        system_prompt = self._render_prompt(chunk, total_chunks)
        user_message = self._build_user_message(chunk)

        logger.info(
            "[baseline_agent] Processing chunk %s (position %d/%d, %d tokens)",
            chunk.chunk_id, chunk.position + 1, total_chunks, chunk.token_count,
        )

        llm_response = self._client.call(system_prompt, user_message)
        raw_json = llm_response.content

        stories = self._parse_response(raw_json, chunk.chunk_id)
        valid = [s for s in stories if not s.parse_errors]
        invalid = [s for s in stories if s.parse_errors]

        if invalid:
            logger.warning(
                "[baseline_agent] Chunk %s: %d/%d stories had parse errors and will be skipped.",
                chunk.chunk_id, len(invalid), len(stories),
            )
            for bad in invalid:
                logger.warning(
                    "  → Errors: %s | raw: role=%r action=%r",
                    bad.parse_errors, bad.role[:50], bad.action[:50],
                )

        logger.info(
            "[baseline_agent] Chunk %s: extracted %d valid stories.",
            chunk.chunk_id, len(valid),
        )

        return BaselineAgentResult(
            chunk_id=chunk.chunk_id,
            raw_stories=stories,
            valid_stories=valid,
            invalid_stories=invalid,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            model_name=llm_response.model_name,
            llm_response_raw=raw_json,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt_template(self) -> str:
        path = self._config.system_prompt_path
        if not path.exists():
            raise FileNotFoundError(
                f"Baseline system prompt not found: {path}"
            )
        return path.read_text(encoding="utf-8")

    def _render_prompt(self, chunk: GDDChunk, total_chunks: int) -> str:
        """
        Render the system prompt template with chunk-specific context
        (chapter, section, position).
        """
        return self._system_prompt_template.format(
            chapter=chunk.chapter or "Unknown",
            section=chunk.section or "Unknown",
            position=chunk.position + 1,
            total_chunks=total_chunks,
        )

    def _build_user_message(self, chunk: GDDChunk) -> str:
        """Build the user-turn message containing the GDD text excerpt."""
        header_parts = []
        if chunk.chapter:
            header_parts.append(f"Chapter: {chunk.chapter}")
        if chunk.section:
            header_parts.append(f"Section: {chunk.section}")

        header = "\n".join(header_parts)
        if header:
            return f"{header}\n\n---\n\n{chunk.text}"
        return chunk.text

    def _parse_response(
        self, raw_json: str, chunk_id: str
    ) -> List[RawStory]:
        """
        Parse the LLM's JSON response into a list of ``RawStory`` objects.

        Raises ``LLMMalformedOutputError`` if the top-level structure is wrong.
        Individual story dicts with missing fields are returned as
        ``RawStory`` with ``parse_errors`` set (not raised — allows partial
        recovery within a chunk).
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Chunk {chunk_id}: LLM returned non-JSON: {raw_json[:200]!r}"
            ) from exc

        if not isinstance(data, dict) or "stories" not in data:
            raise LLMMalformedOutputError(
                f"Chunk {chunk_id}: LLM response missing 'stories' key. "
                f"Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
            )

        stories_raw = data["stories"]
        if not isinstance(stories_raw, list):
            raise LLMMalformedOutputError(
                f"Chunk {chunk_id}: 'stories' must be a list, got {type(stories_raw).__name__}"
            )

        return [RawStory.from_dict(s) for s in stories_raw if isinstance(s, dict)]
