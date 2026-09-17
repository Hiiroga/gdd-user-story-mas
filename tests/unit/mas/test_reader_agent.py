"""
tests/unit/mas/test_reader_agent.py
=====================================
Unit tests for GDDReaderAgent (C4).

All tests use a mock LLMClient — no real API calls are made.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import LLMMalformedOutputError, LLMResponse
from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_chunk() -> GDDChunk:
    return GDDChunk(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        text="Players can attack enemies using melee weapons. "
             "The HUD displays health and stamina bars. "
             "The world is set in a post-apocalyptic wasteland.",
        chapter="Chapter 1",
        section="Combat",
        position=0,
        token_count=42,
    )


@pytest.fixture()
def valid_llm_json(sample_chunk: GDDChunk) -> str:
    """A well-formed LLM response for sample_chunk."""
    return json.dumps({
        "themes": ["combat", "post-apocalyptic"],
        "gameplay_elements": [
            {
                "content": "Melee weapon attacks",
                "source_excerpt": "attack enemies using melee weapons",
            }
        ],
        "systems": [],
        "characters": [],
        "ui_elements": [
            {
                "content": "HUD with health and stamina bars",
                "source_excerpt": "HUD displays health and stamina bars",
            }
        ],
        "narrative": [
            {
                "content": "Post-apocalyptic wasteland setting",
                "source_excerpt": "set in a post-apocalyptic wasteland",
            }
        ],
    })


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_name="gemini-flash-latest",
        prompt_tokens=100,
        completion_tokens=80,
        finish_reason="stop",
    )


def _make_agent(
    tmp_path: Path,
    mock_llm_content: str,
    config_overrides: dict | None = None,
) -> GDDReaderAgent:
    """Build an agent with a mocked LLMClient and a real prompt file."""
    # Write a minimal prompt file
    prompts_dir = tmp_path / "config" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / "reader_system_prompt.txt"
    prompt_file.write_text(
        "Analyze the GDD chunk.\n"
        "Chapter: {chapter}\nSection: {section}\n"
        "Chunk: {position}/{total_chunks}\n",
        encoding="utf-8",
    )

    config = ReaderAgentConfig(
        provider="gemini",
        model_name="gemini-flash-latest",
        min_request_interval_seconds=0.0,
        system_prompt_file="config/prompts/reader_system_prompt.txt",
    )
    if config_overrides:
        for k, v in config_overrides.items():
            object.__setattr__(config, k, v)

    # Patch _build_client so no real API connection is made at construction time.
    with patch(
        "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
        return_value=MagicMock(),
    ):
        agent = GDDReaderAgent(config=config, project_root=tmp_path)

    # Replace the internal call method with our mock response.
    agent._client.call = MagicMock(  # type: ignore[method-assign]
        return_value=_make_llm_response(mock_llm_content)
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Core behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessChunk:

    def test_returns_reader_output(self, tmp_path, sample_chunk, valid_llm_json):
        agent = _make_agent(tmp_path, valid_llm_json)
        result = agent.process_chunk(sample_chunk, total_chunks=5)
        assert isinstance(result, ReaderOutput)

    def test_traceability_chunk_id_preserved(self, tmp_path, sample_chunk, valid_llm_json):
        agent = _make_agent(tmp_path, valid_llm_json)
        result = agent.process_chunk(sample_chunk)
        assert result.chunk_id == sample_chunk.chunk_id

    def test_traceability_document_id_preserved(self, tmp_path, sample_chunk, valid_llm_json):
        agent = _make_agent(tmp_path, valid_llm_json)
        result = agent.process_chunk(sample_chunk)
        assert result.document_id == sample_chunk.document_id

    def test_traceability_overrides_llm_chunk_id(self, tmp_path, sample_chunk):
        """Even if LLM returns wrong chunk_id, we inject the correct one."""
        payload = json.dumps({
            "chunk_id": "WRONG_ID",       # should be overridden
            "document_id": "WRONG_DOC",   # should be overridden
            "themes": ["action"],
            "gameplay_elements": [],
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert result.chunk_id == sample_chunk.chunk_id
        assert result.document_id == sample_chunk.document_id

    def test_correct_items_extracted(self, tmp_path, sample_chunk, valid_llm_json):
        agent = _make_agent(tmp_path, valid_llm_json)
        result = agent.process_chunk(sample_chunk)
        assert len(result.themes) == 2
        assert len(result.gameplay_elements) == 1
        assert len(result.ui_elements) == 1
        assert len(result.narrative) == 1
        assert len(result.systems) == 0
        assert len(result.characters) == 0

    def test_source_excerpts_preserved(self, tmp_path, sample_chunk, valid_llm_json):
        agent = _make_agent(tmp_path, valid_llm_json)
        result = agent.process_chunk(sample_chunk)
        assert result.gameplay_elements[0].source_excerpt == "attack enemies using melee weapons"

    def test_all_empty_categories_valid(self, tmp_path, sample_chunk):
        """LLM returning all-empty categories is valid — don't force hallucination."""
        payload = json.dumps({
            "themes": [],
            "gameplay_elements": [],
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert result.is_empty is True

    def test_null_source_excerpt_valid(self, tmp_path, sample_chunk):
        payload = json.dumps({
            "themes": [],
            "gameplay_elements": [
                {"content": "Some mechanic", "source_excerpt": None}
            ],
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert result.gameplay_elements[0].source_excerpt is None

    def test_missing_category_defaults_to_empty(self, tmp_path, sample_chunk):
        """If LLM omits a category key entirely, it should default to []."""
        payload = json.dumps({
            "themes": ["action"],
            # systems, characters, etc. omitted
            "gameplay_elements": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert result.systems == []
        assert result.characters == []


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_malformed_json_raises_malformed_error(self, tmp_path, sample_chunk):
        agent = _make_agent(tmp_path, "NOT JSON AT ALL {{{")
        with pytest.raises(LLMMalformedOutputError, match="non-JSON"):
            agent.process_chunk(sample_chunk)

    def test_empty_content_item_skipped(self, tmp_path, sample_chunk):
        """Items with empty content should be skipped (not raise)."""
        payload = json.dumps({
            "themes": [],
            "gameplay_elements": [
                {"content": "", "source_excerpt": "something"},   # empty → skip
                {"content": "Valid mechanic", "source_excerpt": "valid"},
            ],
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert len(result.gameplay_elements) == 1
        assert result.gameplay_elements[0].content == "Valid mechanic"

    def test_non_dict_item_skipped(self, tmp_path, sample_chunk):
        """Non-dict items in a list should be skipped gracefully."""
        payload = json.dumps({
            "themes": [],
            "gameplay_elements": ["NOT A DICT", {"content": "Good one"}],
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert len(result.gameplay_elements) == 1
        assert result.gameplay_elements[0].content == "Good one"

    def test_non_list_category_coerced_to_empty(self, tmp_path, sample_chunk):
        """If LLM returns a non-list for a category, coerce to []."""
        payload = json.dumps({
            "themes": "action",  # string instead of list
            "gameplay_elements": "not a list",  # should become []
            "systems": [],
            "characters": [],
            "ui_elements": [],
            "narrative": [],
        })
        agent = _make_agent(tmp_path, payload)
        result = agent.process_chunk(sample_chunk)
        assert result.gameplay_elements == []


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptRendering:

    def test_chapter_section_injected(self, tmp_path, sample_chunk, valid_llm_json):
        """Verify rendered prompt contains chunk context (no KeyError from JSON braces)."""
        agent = _make_agent(tmp_path, valid_llm_json)
        rendered = agent._render_system_prompt(sample_chunk, total_chunks=10)
        assert "Chapter 1" in rendered
        assert "Combat" in rendered
        assert "1/10" in rendered

    def test_no_key_error_from_json_braces(self, tmp_path):
        """Prompt template with JSON example braces must not raise KeyError."""
        # Write a prompt that contains JSON braces like the real one
        prompts_dir = tmp_path / "config" / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "reader_system_prompt.txt").write_text(
            'Return: {"themes": [], "gameplay_elements": []}\n'
            "Chapter: {chapter}\nSection: {section}\n"
            "Chunk: {position}/{total_chunks}\n",
            encoding="utf-8",
        )
        config = ReaderAgentConfig(
            min_request_interval_seconds=0.0,
            system_prompt_file="config/prompts/reader_system_prompt.txt",
        )
        with patch(
            "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
            return_value=MagicMock(),
        ):
            agent = GDDReaderAgent(config=config, project_root=tmp_path)
        chunk = GDDChunk(
            chunk_id="c1", document_id="d1", text="test",
            position=0, token_count=1,
        )
        # Must not raise KeyError
        rendered = agent._render_system_prompt(chunk, total_chunks=1)
        assert '{"themes": [], "gameplay_elements": []}' in rendered
        assert "Unknown" in rendered  # chapter/section default

    def test_unknown_chapter_default(self, tmp_path, valid_llm_json):
        """Chunk with no chapter/section → 'Unknown' inserted."""
        chunk = GDDChunk(
            chunk_id="c1", document_id="d1", text="Some text",
            position=0, token_count=5,
        )
        agent = _make_agent(tmp_path, valid_llm_json)
        rendered = agent._render_system_prompt(chunk, total_chunks=1)
        assert "Unknown" in rendered


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

class TestReaderAgentConfig:

    def test_defaults(self):
        cfg = ReaderAgentConfig.defaults()
        assert cfg.provider == "gemini"
        assert cfg.temperature == 0.0
        assert cfg.prompt_version == "reader_v1"

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
model:
  provider: "gemini"
  name: "gemini-flash-latest"
  temperature: 0
  max_tokens: 4096
  top_p: null
  response_format: "json_object"
invocation:
  max_retries: 5
  retry_backoff_seconds: 2
  min_request_interval_seconds: 13
  wait_on_overload_seconds: 60
reader_agent:
  prompt_version: "reader_v1"
  system_prompt_file: "config/prompts/reader_system_prompt.txt"
"""
        yaml_path = tmp_path / "config" / "mas.yaml"
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(yaml_content, encoding="utf-8")

        cfg = ReaderAgentConfig.from_yaml(yaml_path, project_root=tmp_path)
        assert cfg.provider == "gemini"
        assert cfg.model_name == "gemini-flash-latest"
        assert cfg.temperature == 0.0
        assert cfg.max_retries == 5
        assert cfg.min_request_interval_seconds == 13.0
        assert cfg.prompt_version == "reader_v1"

    def test_missing_prompt_file_raises(self, tmp_path):
        config = ReaderAgentConfig(
            min_request_interval_seconds=0.0,
            system_prompt_file="config/prompts/NONEXISTENT.txt",
        )
        with pytest.raises(FileNotFoundError, match="NONEXISTENT"):
            with patch(
                "gdd_userstory_mas.baseline.llm_client.LLMClient._build_client",
                return_value=MagicMock(),
            ):
                GDDReaderAgent(config=config, project_root=tmp_path)
