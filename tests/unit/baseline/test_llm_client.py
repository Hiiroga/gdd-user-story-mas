"""
tests/unit/baseline/test_llm_client.py
========================================
Unit tests for LLMClient — all LLM calls are mocked so no real API key
or internet access is required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.baseline.llm_client import (
    LLMAPIError,
    LLMClient,
    LLMMalformedOutputError,
    LLMResponse,
    _validate_json,
)


# ── _validate_json ─────────────────────────────────────────────────────────────

class TestValidateJson:
    def test_valid_json_object_passes(self) -> None:
        _validate_json('{"key": "value"}')  # must not raise

    def test_valid_json_array_passes(self) -> None:
        _validate_json('[1, 2, 3]')

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(LLMMalformedOutputError):
            _validate_json("{not valid json}")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(LLMMalformedOutputError):
            _validate_json("")


# ── LLMClient construction ─────────────────────────────────────────────────────

class TestLLMClientConstruction:
    def test_unknown_provider_raises_on_build(self) -> None:
        with pytest.raises(LLMAPIError, match="Unknown provider"):
            LLMClient(provider="unsupported_llm", model_name="x")

    def test_openai_import_error_raises_import_error(self) -> None:
        import sys
        # Temporarily hide openai from imports
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises((ImportError, LLMAPIError)):
                LLMClient(provider="openai", model_name="gpt-4o")


# ── LLMClient.call — OpenAI mock ───────────────────────────────────────────────

def _make_mock_openai_client(json_content: str = '{"stories": []}') -> MagicMock:
    """Return a mock openai.OpenAI() client that returns a fixed JSON response."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json_content
    mock_choice.finish_reason = "stop"
    mock_raw = MagicMock()
    mock_raw.choices = [mock_choice]
    mock_raw.model = "gpt-4o"
    mock_raw.usage.prompt_tokens = 100
    mock_raw.usage.completion_tokens = 50
    mock_client.chat.completions.create.return_value = mock_raw
    return mock_client


class TestLLMClientOpenAICall:
    def _get_client(self, json_content: str = '{"stories": []}') -> LLMClient:
        mock_sdk = MagicMock()
        mock_sdk.OpenAI.return_value = _make_mock_openai_client(json_content)
        with patch.dict("sys.modules", {"openai": mock_sdk}):
            return LLMClient(
                provider="openai",
                model_name="gpt-4o",
                api_key="test-key",
                temperature=0.0,
                max_tokens=512,
                response_format="json_object",
            )

    def test_call_returns_llm_response(self) -> None:
        client = self._get_client()
        # Manually inject a pre-built mock client to avoid re-patching
        client._client = _make_mock_openai_client()
        resp = client.call("system", "user")
        assert isinstance(resp, LLMResponse)
        assert resp.model_name == "gpt-4o"

    def test_call_reports_token_counts(self) -> None:
        client = self._get_client()
        client._client = _make_mock_openai_client()
        resp = client.call("system", "user")
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50

    def test_call_returns_content(self) -> None:
        payload = '{"stories": [{"role": "player", "action": "jump"}]}'
        client = self._get_client(payload)
        client._client = _make_mock_openai_client(payload)
        resp = client.call("system", "user")
        assert "stories" in json.loads(resp.content)

    def test_malformed_json_raises_malformed_error(self) -> None:
        client = self._get_client()
        client._client = _make_mock_openai_client("not json at all")
        with pytest.raises(LLMMalformedOutputError):
            client.call("system", "user")

    def test_retry_on_transient_error(self) -> None:
        """Client retries on generic Exception (simulates network error)."""
        client = self._get_client()
        good_mock = _make_mock_openai_client()
        # First call raises, second succeeds
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = [
            Exception("network timeout"),
            good_mock.chat.completions.create.return_value,
        ]
        client.max_retries = 2
        client.retry_backoff_seconds = 0  # no sleep in tests

        with patch("gdd_userstory_mas.baseline.llm_client.time.sleep"):
            resp = client.call("system", "user")
        assert resp.model_name == "gpt-4o"

    def test_all_retries_exhausted_raises_api_error(self) -> None:
        client = self._get_client()
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = Exception("always fails")
        client.max_retries = 2
        client.retry_backoff_seconds = 0

        with patch("gdd_userstory_mas.baseline.llm_client.time.sleep"):
            with pytest.raises(LLMAPIError, match="All 2 LLM call attempts failed"):
                client.call("system", "user")
