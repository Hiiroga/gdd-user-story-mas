"""
llm_client.py — Provider-agnostic LLM interface
=================================================
Thin abstraction layer so the baseline (and future MAS agents) can swap
providers without changing calling code.

Design decisions:
  * Only ONE external dependency is required at runtime: the provider's
    own Python SDK (e.g., ``openai``).  No LangChain/CrewAI dependency.
  * The client is stateless (no conversation history); each call is
    independent — correct for both the baseline (one call per chunk) and
    MAS agents (each agent call is a fresh context).
  * Provider-specific code is isolated to ``_call_openai`` /
    ``_call_anthropic`` / ``_call_gemini`` helpers so adding a new
    provider touches only this file.
  * Retry with exponential back-off is built in (Technical Architecture §10).
  * Structured JSON output is requested via provider JSON mode where
    available; raw text with manual JSON parsing is the fallback.

Currently supported providers:
  - ``openai``    — uses ``openai`` SDK (openai>=1.0)
  - ``anthropic`` — uses ``anthropic`` SDK
  - ``gemini``    — uses ``google-generativeai`` SDK (stub only for now)

For unit testing WITHOUT hitting a real API, pass a ``mock_response``
fixture — see ``_MockLLMClient`` pattern in tests.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Normalised response from any LLM provider.

    Attributes
    ----------
    content:
        The text/JSON content returned by the model.
    model_name:
        Exact model identifier reported by the provider.
    prompt_tokens:
        Number of input tokens consumed (for cost/logging).
    completion_tokens:
        Number of output tokens generated (for cost/logging).
    finish_reason:
        Provider-specific finish reason string (e.g., 'stop', 'length').
    raw_response:
        The full raw response object from the SDK (for debugging).
    """

    content: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw_response: Any = None


# ── LLMClient ──────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin, provider-agnostic LLM client with retry logic.

    Parameters
    ----------
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"gemini"``.
    model_name:
        The model identifier string (e.g., ``"gpt-4o"``).
    api_key:
        API key.  If ``None``, the client attempts to read from the
        standard environment variable for the provider.
    temperature:
        Sampling temperature.  Use 0 for maximum reproducibility.
    max_tokens:
        Maximum completion tokens.
    response_format:
        ``"json_object"`` to request JSON mode, ``"text"`` otherwise.
    max_retries:
        Number of retry attempts on transient failure.
    retry_backoff_seconds:
        Base backoff interval (doubles on each retry).
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: Optional[float] = None,
        response_format: str = "json_object",
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.response_format = response_format
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        self._client: Any = self._build_client()

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_message: str,
    ) -> LLMResponse:
        """
        Send a system + user message and return a normalised response.

        Retries up to ``max_retries`` times with exponential back-off on
        transient errors (API failures, timeouts).

        Raises
        ------
        LLMAPIError
            If all retries are exhausted.
        LLMMalformedOutputError
            If the provider returns a non-parseable response when JSON mode
            was requested.
        """
        last_exc: Optional[Exception] = None
        backoff = self.retry_backoff_seconds

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "LLM call attempt %d/%d (provider=%s, model=%s)",
                    attempt, self.max_retries, self.provider, self.model_name,
                )
                response = self._dispatch(system_prompt, user_message)
                logger.debug(
                    "LLM call succeeded: prompt_tokens=%d, completion_tokens=%d, finish=%s",
                    response.prompt_tokens, response.completion_tokens, response.finish_reason,
                )
                return response
            except (LLMAPIError, LLMMalformedOutputError):
                raise  # don't retry on schema errors — they need a different prompt
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d failed: %s. Retrying in %.1fs…",
                    attempt, self.max_retries, exc, backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2  # exponential back-off

        raise LLMAPIError(
            f"All {self.max_retries} LLM call attempts failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    # ── Internal dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, system_prompt: str, user_message: str) -> LLMResponse:
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_message)
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_message)
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_message)
        raise LLMAPIError(f"Unknown provider: '{self.provider}'")

    def _build_client(self) -> Any:
        """Initialise and return the provider SDK client (lazy import)."""
        if self.provider == "openai":
            try:
                import openai  # noqa: PLC0415
                return openai.OpenAI(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai SDK not installed. Run: pip install openai>=1.0"
                ) from exc

        if self.provider == "anthropic":
            try:
                import anthropic  # noqa: PLC0415
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "anthropic SDK not installed. Run: pip install anthropic"
                ) from exc

        if self.provider == "gemini":
            try:
                import google.generativeai as genai  # noqa: PLC0415
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai SDK not installed. "
                    "Run: pip install google-generativeai"
                ) from exc

        raise LLMAPIError(f"Unknown provider: '{self.provider}'")

    def _call_openai(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Call OpenAI-compatible API (openai>=1.0)."""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        raw = self._client.chat.completions.create(**kwargs)
        choice = raw.choices[0]
        content = choice.message.content or ""

        if self.response_format == "json_object":
            _validate_json(content)

        return LLMResponse(
            content=content,
            model_name=raw.model,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            finish_reason=choice.finish_reason or "",
            raw_response=raw,
        )

    def _call_anthropic(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Call Anthropic Messages API."""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        raw = self._client.messages.create(**kwargs)
        content = raw.content[0].text if raw.content else ""

        if self.response_format == "json_object":
            _validate_json(content)

        return LLMResponse(
            content=content,
            model_name=raw.model,
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
            finish_reason=raw.stop_reason or "",
            raw_response=raw,
        )

    def _call_gemini(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Call Google Gemini API (stub — expand when provider is confirmed)."""
        # Combine system and user for Gemini (no system-message support in basic API)
        combined = f"{system_prompt}\n\n---\n\n{user_message}"
        raw = self._client.generate_content(combined)
        content = raw.text or ""

        if self.response_format == "json_object":
            _validate_json(content)

        # Gemini usage metadata varies; use 0 as placeholder
        return LLMResponse(
            content=content,
            model_name=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
            raw_response=raw,
        )


# ── Custom exceptions ──────────────────────────────────────────────────────────

class LLMAPIError(RuntimeError):
    """Raised when an LLM API call fails after all retries."""


class LLMMalformedOutputError(ValueError):
    """Raised when the LLM returns content that cannot be parsed as expected."""


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_json(content: str) -> None:
    """
    Assert that ``content`` is valid JSON.

    Raises ``LLMMalformedOutputError`` if parsing fails.
    """
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMMalformedOutputError(
            f"LLM returned non-JSON content when JSON mode was requested: "
            f"{content[:200]!r}"
        ) from exc
