"""
analyst_agent.py — Requirements Analyst Agent (C5)
====================================================
Implements the Requirements Analyst Agent as specified in Technical
Architecture §2 and Data Contracts §5 (SRS-009).

Responsibilities:
  - Receive one ``ReaderOutput`` at a time (stateless, independent per chunk).
  - Call the LLM to identify candidate functional requirements from the
    structured thematic summary produced by the GDD Reader Agent.
  - Classify each requirement by perspective (player | system | in-game entity
    | dev team) and domain (gameplay | ui | narrative | systems | ...).
  - Assign deterministic ``candidate_id`` values.
  - Preserve source traceability via ``source_reader_output_ref``.
  - Return a validated ``AnalystOutput`` containing zero or more
    ``CandidateRequirement`` records.
  - Never generate user stories.
  - Never infer requirements not supported by the input.

Design decisions:
  - Reuses ``baseline.llm_client.LLMClient`` (same as Reader Agent).
  - Prompt rendering uses ``str.replace()`` (not ``.format()``).
  - Stateless: no cross-chunk state.
  - Config loaded from ``config/mas.yaml`` via ``AnalystAgentConfig.from_yaml()``.
  - ``candidate_id`` format: ``{document_id}__cand_{sequence:04d}`` starting
    from a caller-supplied offset so IDs are unique across the full run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gdd_userstory_mas.baseline.llm_client import (
    LLMClient,
    LLMMalformedOutputError,
)
from gdd_userstory_mas.schemas.candidate_requirement import (
    VALID_DOMAINS,
    AnalystOutput,
    CandidateRequirement,
)
from gdd_userstory_mas.schemas.reader_output import ReaderOutput

logger = logging.getLogger(__name__)

# ── Valid perspective values (mirrors schema Literal) ──────────────────────────
_VALID_PERSPECTIVES = frozenset({"player", "system", "in-game entity", "dev team"})
_VALID_DOMAINS = VALID_DOMAINS


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class AnalystAgentConfig:
    """
    Configuration for the Requirements Analyst Agent.

    Loaded from ``config/mas.yaml``. Mirrors the pattern of
    ``ReaderAgentConfig`` for consistency.
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
    prompt_version: str = "analyst_v1"
    system_prompt_file: str = "config/prompts/analyst_system_prompt.txt"

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "AnalystAgentConfig":
        """Load ``AnalystAgentConfig`` from ``config/mas.yaml``."""
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
        analyst = raw.get("analyst_agent", {})

        return cls(
            provider=model.get("provider", "gemini"),
            model_name=model.get("name", "gemini-flash-latest"),
            temperature=float(model.get("temperature", 0.0)),
            max_tokens=int(model.get("max_tokens", 4096)),
            top_p=model.get("top_p"),
            response_format=model.get("response_format", "json_object"),
            max_retries=int(inv.get("max_retries", 5)),
            retry_backoff_seconds=float(inv.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(
                inv.get("min_request_interval_seconds", 13.0)
            ),
            wait_on_overload_seconds=float(inv.get("wait_on_overload_seconds", 60.0)),
            prompt_version=analyst.get("prompt_version", "analyst_v1"),
            system_prompt_file=analyst.get(
                "system_prompt_file",
                "config/prompts/analyst_system_prompt.txt",
            ),
        )

    @classmethod
    def defaults(cls) -> "AnalystAgentConfig":
        return cls()


# ── Requirements Analyst Agent ─────────────────────────────────────────────────

class RequirementsAnalystAgent:
    """
    Requirements Analyst Agent (Component C5).

    Receives one ``ReaderOutput`` and returns an ``AnalystOutput`` containing
    zero or more ``CandidateRequirement`` records.

    This agent:
    - Is STATELESS — no conversation history across calls.
    - Does NOT generate user stories.
    - Does NOT infer requirements beyond what the ReaderOutput explicitly states.
    - Assigns deterministic ``candidate_id`` values starting from
      ``id_offset`` (to ensure global uniqueness across a full document run).

    Parameters
    ----------
    config:
        ``AnalystAgentConfig`` (loaded from ``config/mas.yaml``).
    api_key:
        LLM provider API key. If None, reads from environment.
    project_root:
        Project root directory used to resolve prompt file paths.

    Example
    -------
    >>> agent = RequirementsAnalystAgent(config=AnalystAgentConfig.defaults())
    >>> output = agent.process_reader_output(reader_output, id_offset=0)
    >>> print(output.candidate_count)
    """

    def __init__(
        self,
        config: AnalystAgentConfig,
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
            "RequirementsAnalystAgent initialized (provider=%s, model=%s, prompt=%s)",
            config.provider,
            config.model_name,
            config.prompt_version,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_reader_output(
        self,
        reader_output: ReaderOutput,
        id_offset: int = 0,
        total_chunks: int = 1,
    ) -> AnalystOutput:
        """
        Analyse one ``ReaderOutput`` and return candidate requirements.

        Parameters
        ----------
        reader_output:
            The structured thematic summary from the GDD Reader Agent.
        id_offset:
            The starting sequence number for ``candidate_id`` generation.
            Pass the running total of candidates produced so far in the run
            to ensure globally unique IDs across all chunks.
        total_chunks:
            Total number of chunks in the document (used in the prompt for
            positional context).

        Returns
        -------
        AnalystOutput
            Validated output with zero or more ``CandidateRequirement`` records.

        Raises
        ------
        LLMAPIError
            If all LLM call attempts fail (triggers run halt per Tech Arch §10).
        LLMMalformedOutputError
            If the LLM returns unparseable JSON.
        """
        logger.info(
            "Analysing ReaderOutput for chunk %s (id_offset=%d)",
            reader_output.chunk_id,
            id_offset,
        )

        if reader_output.is_empty:
            logger.info(
                "Chunk %s: ReaderOutput is empty — returning zero candidates.",
                reader_output.chunk_id,
            )
            return AnalystOutput(
                source_chunk_id=reader_output.chunk_id,
                document_id=reader_output.document_id,
                candidates=[],
            )

        system_prompt = self._render_system_prompt(reader_output, total_chunks)
        user_message = self._build_user_message(reader_output)

        llm_response = self._client.call(system_prompt, user_message)

        logger.debug(
            "Chunk %s: LLM returned %d prompt tokens, %d completion tokens",
            reader_output.chunk_id,
            llm_response.prompt_tokens,
            llm_response.completion_tokens,
        )

        analyst_output = self._parse_and_validate(
            llm_response.content,
            reader_output,
            id_offset,
        )

        logger.info(
            "Chunk %s: extracted %d candidates | perspectives=%s | domains=%s",
            reader_output.chunk_id,
            analyst_output.candidate_count,
            analyst_output.perspective_counts(),
            analyst_output.domain_counts(),
        )

        return analyst_output

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        prompt_path = self._project_root / self._config.system_prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Analyst Agent system prompt not found: {prompt_path}\n"
                f"Expected at: config/prompts/analyst_system_prompt.txt"
            )
        text = prompt_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Analyst Agent system prompt is empty: {prompt_path}")
        logger.debug("Loaded prompt from %s (%d chars)", prompt_path, len(text))
        return text

    def _render_system_prompt(
        self,
        reader_output: ReaderOutput,
        total_chunks: int,
    ) -> str:
        """Render prompt with chunk context using str.replace() (safe with JSON braces)."""
        # Infer chapter/section from the ReaderOutput's metadata if available.
        # ReaderOutput doesn't carry chapter/section directly, so we use
        # generic positional context from what is known.
        return (
            self._system_prompt_template
            .replace("{chapter}", "—")
            .replace("{section}", "—")
            .replace("{position}", "—")
            .replace("{total_chunks}", str(total_chunks))
        )

    def _build_user_message(self, reader_output: ReaderOutput) -> str:
        """
        Serialize the ReaderOutput into a structured user message.

        The LLM receives the full structured summary so it can identify
        requirements from ANY category (gameplay, systems, UI, etc.).
        """
        lines = [
            f"[ReaderOutput — chunk_id: {reader_output.chunk_id}]",
            f"[Document: {reader_output.document_id}]",
            "---",
            "",
        ]

        if reader_output.themes:
            lines.append("THEMES: " + ", ".join(reader_output.themes))
            lines.append("")

        for category_name in (
            "gameplay_elements", "systems", "characters", "ui_elements", "narrative"
        ):
            items = getattr(reader_output, category_name)
            if not items:
                continue
            label = category_name.replace("_", " ").upper()
            lines.append(f"{label}:")
            for item in items:
                lines.append(f"  - content: {item.content}")
                if item.source_excerpt:
                    lines.append(f"    source_excerpt: \"{item.source_excerpt}\"")
            lines.append("")

        return "\n".join(lines)

    def _parse_and_validate(
        self,
        raw_content: str,
        reader_output: ReaderOutput,
        id_offset: int,
    ) -> AnalystOutput:
        """
        Parse the LLM JSON response and build a validated ``AnalystOutput``.

        Steps:
        1. Parse JSON.
        2. Extract "candidates" list.
        3. Normalise each item — skip invalid entries with a warning.
        4. Assign deterministic ``candidate_id`` values.
        5. Build and validate the Pydantic model.
        """
        # Step 1: Parse JSON
        try:
            data: Dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Analyst Agent: LLM returned non-JSON for chunk "
                f"{reader_output.chunk_id}: {raw_content[:300]!r}"
            ) from exc

        # Step 2: Extract candidates list
        raw_candidates = data.get("candidates", [])
        if not isinstance(raw_candidates, list):
            logger.warning(
                "Chunk %s: 'candidates' field is not a list (got %s) — treating as []",
                reader_output.chunk_id,
                type(raw_candidates).__name__,
            )
            raw_candidates = []

        # Step 3: Normalise + validate each candidate
        valid_candidates: List[CandidateRequirement] = []
        seen_texts: set = set()

        for idx, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                logger.warning(
                    "Chunk %s: candidates[%d] is not a dict — skipping",
                    reader_output.chunk_id, idx,
                )
                continue

            req_text = str(raw.get("requirement_text", "")).strip()
            if not req_text:
                logger.warning(
                    "Chunk %s: candidates[%d] has empty requirement_text — skipping",
                    reader_output.chunk_id, idx,
                )
                continue

            # Guard against user-story format leaking through
            if req_text.lower().startswith("as a"):
                logger.warning(
                    "Chunk %s: candidates[%d] requirement_text starts with 'As a' "
                    "(user-story format) — skipping: %r",
                    reader_output.chunk_id, idx, req_text[:80],
                )
                continue

            # Deduplicate
            text_key = req_text.lower()
            if text_key in seen_texts:
                logger.warning(
                    "Chunk %s: candidates[%d] is duplicate — skipping",
                    reader_output.chunk_id, idx,
                )
                continue
            seen_texts.add(text_key)

            # Validate perspective
            perspective = str(raw.get("perspective", "")).strip().lower()
            if perspective not in _VALID_PERSPECTIVES:
                logger.warning(
                    "Chunk %s: candidates[%d] has invalid perspective %r "
                    "— defaulting to 'system'",
                    reader_output.chunk_id, idx, perspective,
                )
                perspective = "system"

            # Validate domain
            domain = str(raw.get("domain", "")).strip().lower()
            if domain not in _VALID_DOMAINS:
                logger.warning(
                    "Chunk %s: candidates[%d] has invalid domain %r "
                    "— defaulting to 'other'",
                    reader_output.chunk_id, idx, domain,
                )
                domain = "other"

            # source_reader_output_ref
            ref = raw.get("source_reader_output_ref")
            if ref is not None:
                ref = str(ref).strip() or None

            # Build candidate with deterministic ID
            seq = id_offset + len(valid_candidates)
            candidate_id = f"{reader_output.document_id}__cand_{seq:04d}"

            try:
                candidate = CandidateRequirement(
                    candidate_id=candidate_id,
                    source_chunk_id=reader_output.chunk_id,
                    document_id=reader_output.document_id,
                    requirement_text=req_text,
                    perspective=perspective,  # type: ignore[arg-type]
                    domain=domain,  # type: ignore[arg-type]
                    source_reader_output_ref=ref,
                )
                valid_candidates.append(candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Chunk %s: candidates[%d] failed schema validation — skipping: %s",
                    reader_output.chunk_id, idx, exc,
                )
                continue

        # Step 5: Build AnalystOutput
        try:
            return AnalystOutput(
                source_chunk_id=reader_output.chunk_id,
                document_id=reader_output.document_id,
                candidates=valid_candidates,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMMalformedOutputError(
                f"Analyst Agent: AnalystOutput validation failed for chunk "
                f"{reader_output.chunk_id}: {exc}"
            ) from exc
