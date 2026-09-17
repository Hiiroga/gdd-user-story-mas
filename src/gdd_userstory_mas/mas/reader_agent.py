"""
reader_agent.py — GDD Reader Agent (C4)
=========================================
Implements the GDD Reader Agent as specified in Technical Architecture §2
and Data Contracts §4 (SRS-008).

Responsibilities:
  - Receive one GDDChunk at a time (stateless, independent per chunk).
  - Call the LLM to identify themes, gameplay elements, systems, characters,
    UI elements, and narrative information.
  - Return a validated ``ReaderOutput`` with source traceability.
  - Never generate user stories or requirements.
  - Never infer content not present in the chunk text.

Design decisions:
  - Reuses ``baseline.llm_client.LLMClient`` (provider-agnostic, retry logic,
    429/503 handling already implemented).
  - Prompt rendering uses ``str.replace()`` (not ``.format()``) to avoid
    KeyError when the prompt template contains JSON example braces.
  - Stateless: no cross-chunk conversation history.
  - Config loaded from ``config/mas.yaml`` via ``ReaderAgentConfig.from_yaml()``.
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
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput

logger = logging.getLogger(__name__)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class ReaderAgentConfig:
    """
    Configuration for the GDD Reader Agent.

    All values come from ``config/mas.yaml`` and are loaded once at
    construction time. A snapshot is persisted alongside each run's output
    for reproducibility (Technical Architecture §8).
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
    prompt_version: str = "reader_v1"
    system_prompt_file: str = "config/prompts/reader_system_prompt.txt"

    # Resolved at load time (not serialized to YAML)
    _system_prompt_text: str = field(default="", init=False, repr=False)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        project_root: Path = Path("."),
    ) -> "ReaderAgentConfig":
        """
        Load ``ReaderAgentConfig`` from ``config/mas.yaml``.

        Parameters
        ----------
        yaml_path:
            Path to the YAML file (usually ``config/mas.yaml``).
        project_root:
            Root of the project — used to resolve ``system_prompt_file``
            relative paths.
        """
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required. Run: pip install pyyaml"
            ) from exc

        path = Path(yaml_path)
        if not path.is_absolute():
            path = project_root / path

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        model = raw.get("model", {})
        inv = raw.get("invocation", {})
        reader = raw.get("reader_agent", {})

        cfg = cls(
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
            prompt_version=reader.get("prompt_version", "reader_v1"),
            system_prompt_file=reader.get(
                "system_prompt_file",
                "config/prompts/reader_system_prompt.txt",
            ),
        )
        return cfg

    @classmethod
    def defaults(cls) -> "ReaderAgentConfig":
        """Return a config with all default values (useful for testing)."""
        return cls()


# ── GDD Reader Agent ───────────────────────────────────────────────────────────

class GDDReaderAgent:
    """
    GDD Reader Agent (Component C4).

    Analyzes one ``GDDChunk`` at a time and returns a ``ReaderOutput``
    containing structured thematic elements with source traceability.

    This agent:
    - Is STATELESS — no conversation history across chunks.
    - Does NOT generate user stories or requirements.
    - Does NOT infer content not present in the chunk text.
    - Validates LLM output against the ``ReaderOutput`` schema before returning.

    Parameters
    ----------
    config:
        ``ReaderAgentConfig`` (loaded from ``config/mas.yaml``).
    api_key:
        LLM provider API key. If None, reads from environment.
    project_root:
        Project root directory used to resolve prompt file paths.

    Example
    -------
    >>> agent = GDDReaderAgent(config=ReaderAgentConfig.defaults(), api_key="sk-...")
    >>> output = agent.process_chunk(chunk)
    >>> print(output.category_counts())
    """

    def __init__(
        self,
        config: ReaderAgentConfig,
        api_key: Optional[str] = None,
        project_root: Path = Path("."),
    ) -> None:
        self._config = config
        self._project_root = Path(project_root)

        # Load and cache the system prompt template
        self._system_prompt_template = self._load_prompt()

        # Initialise shared LLM client (reuses baseline infrastructure)
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
            "GDDReaderAgent initialized (provider=%s, model=%s, prompt=%s)",
            config.provider,
            config.model_name,
            config.prompt_version,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_chunk(
        self,
        chunk: GDDChunk,
        total_chunks: int = 1,
    ) -> ReaderOutput:
        """
        Analyze one GDD chunk and return a structured ``ReaderOutput``.

        This method is STATELESS — it can be called in any order and
        does not depend on prior calls.

        Parameters
        ----------
        chunk:
            The preprocessed GDD chunk to analyze.
        total_chunks:
            Total number of chunks in the document (used in the prompt for
            positional context only).

        Returns
        -------
        ReaderOutput
            Validated structured output with traceability preserved.

        Raises
        ------
        LLMAPIError
            If all LLM call attempts fail (triggers run halt per Tech Arch §10).
        LLMMalformedOutputError
            If the LLM returns unparseable JSON (logged, chunk skipped).
        """
        logger.info(
            "Reading chunk %s (pos=%d/%d, tokens=%d)",
            chunk.chunk_id,
            chunk.position + 1,
            total_chunks,
            chunk.token_count,
        )

        system_prompt = self._render_system_prompt(chunk, total_chunks)
        user_message = self._build_user_message(chunk)

        llm_response = self._client.call(system_prompt, user_message)

        logger.debug(
            "Chunk %s: LLM returned %d prompt tokens, %d completion tokens",
            chunk.chunk_id,
            llm_response.prompt_tokens,
            llm_response.completion_tokens,
        )

        reader_output = self._parse_and_validate(llm_response.content, chunk)

        logger.info(
            "Chunk %s: extracted %s",
            chunk.chunk_id,
            reader_output.category_counts(),
        )

        return reader_output

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        """Load the system prompt template from the configured file path."""
        prompt_path = self._project_root / self._config.system_prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Reader Agent system prompt not found: {prompt_path}\n"
                f"Expected at: config/prompts/reader_system_prompt.txt"
            )
        text = prompt_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(
                f"Reader Agent system prompt is empty: {prompt_path}"
            )
        logger.debug("Loaded prompt from %s (%d chars)", prompt_path, len(text))
        return text

    def _render_system_prompt(self, chunk: GDDChunk, total_chunks: int) -> str:
        """
        Render the system prompt with chunk-specific context variables.

        Uses ``str.replace()`` (not ``.format()``) to avoid KeyError when
        the prompt template contains JSON example braces.
        """
        return (
            self._system_prompt_template
            .replace("{chapter}", chunk.chapter or "Unknown")
            .replace("{section}", chunk.section or "Unknown")
            .replace("{position}", str(chunk.position + 1))
            .replace("{total_chunks}", str(total_chunks))
        )

    def _build_user_message(self, chunk: GDDChunk) -> str:
        """Build the user-turn message containing the chunk text."""
        header = (
            f"[GDD Chunk — chunk_id: {chunk.chunk_id}]\n"
            f"[Document: {chunk.document_id}]\n"
            "---\n"
        )
        return header + chunk.text

    def _parse_and_validate(
        self,
        raw_content: str,
        chunk: GDDChunk,
    ) -> ReaderOutput:
        """
        Parse the LLM JSON response and validate against ``ReaderOutput``.

        Parameters
        ----------
        raw_content:
            Raw JSON string from the LLM.
        chunk:
            The source chunk (for traceability injection and validation).

        Returns
        -------
        ReaderOutput
            Fully validated schema instance.

        Raises
        ------
        LLMMalformedOutputError
            If JSON is unparseable or the schema cannot be constructed.
        """
        # Step 1: Parse JSON
        try:
            data: Dict[str, Any] = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LLMMalformedOutputError(
                f"Reader Agent: LLM returned non-JSON for chunk "
                f"{chunk.chunk_id}: {raw_content[:300]!r}"
            ) from exc

        # Step 2: Inject traceability fields (always override from source)
        data["chunk_id"] = chunk.chunk_id
        data["document_id"] = chunk.document_id

        # Step 3: Normalise list fields — ensure they are lists
        for field_name in (
            "themes", "gameplay_elements", "systems",
            "characters", "ui_elements", "narrative"
        ):
            if field_name not in data:
                data[field_name] = []
            elif not isinstance(data[field_name], list):
                logger.warning(
                    "Chunk %s: field '%s' is not a list (got %s) — coercing to []",
                    chunk.chunk_id, field_name, type(data[field_name]).__name__,
                )
                data[field_name] = []

        # Step 4: Normalise themes — strip non-strings
        data["themes"] = [
            t for t in data["themes"] if isinstance(t, str) and t.strip()
        ]

        # Step 5: Normalise EvidencedItem lists
        for field_name in (
            "gameplay_elements", "systems", "characters", "ui_elements", "narrative"
        ):
            normalized: List[Dict[str, Any]] = []
            for idx, item in enumerate(data[field_name]):
                if not isinstance(item, dict):
                    logger.warning(
                        "Chunk %s: %s[%d] is not a dict — skipping",
                        chunk.chunk_id, field_name, idx,
                    )
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    logger.warning(
                        "Chunk %s: %s[%d] has empty content — skipping",
                        chunk.chunk_id, field_name, idx,
                    )
                    continue
                excerpt = item.get("source_excerpt")
                if excerpt is not None:
                    excerpt = str(excerpt).strip() or None
                normalized.append({
                    "content": content,
                    "source_excerpt": excerpt,
                })
            data[field_name] = normalized

        # Step 6: Build and validate the Pydantic model
        try:
            return ReaderOutput(**data)
        except Exception as exc:  # noqa: BLE001
            raise LLMMalformedOutputError(
                f"Reader Agent: ReaderOutput schema validation failed for "
                f"chunk {chunk.chunk_id}: {exc}\nRaw data: {data}"
            ) from exc
