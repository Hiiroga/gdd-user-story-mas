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
        min_request_interval_seconds: float = 0.0,
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
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_call_time: float = 0.0  # unix timestamp of last successful call

        self._client: Any = self._build_client()

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_message: str,
    ) -> LLMResponse:
        """
        Send a system + user message and return a normalised response.

        Handles three retry strategies:

        - **429 RESOURCE_EXHAUSTED**: waits the retryDelay from the API
          response, then retries. Does NOT consume ``max_retries`` budget.
        - **503 UNAVAILABLE** (model overloaded): waits longer, also does NOT
          consume ``max_retries`` budget.
        - **All other errors**: exponential back-off, capped at ``max_retries``.

        Also throttles to ``min_request_interval_seconds`` between calls.

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
        regular_attempts = 0  # only counts non-429/503 failures

        # ── Per-call throttle ───────────────────────────────────────────────
        if self.min_request_interval_seconds > 0:
            elapsed = time.time() - self._last_call_time
            wait = self.min_request_interval_seconds - elapsed
            if wait > 0:
                logger.debug("Rate-limit throttle: sleeping %.1fs", wait)
                time.sleep(wait)

        while True:
            regular_attempts += 1
            try:
                logger.debug(
                    "LLM call attempt %d (provider=%s, model=%s)",
                    regular_attempts, self.provider, self.model_name,
                )
                response = self._dispatch(system_prompt, user_message)
                self._last_call_time = time.time()
                logger.debug(
                    "LLM call succeeded: prompt_tokens=%d, completion_tokens=%d, finish=%s",
                    response.prompt_tokens, response.completion_tokens, response.finish_reason,
                )
                return response
            except (LLMAPIError, LLMMalformedOutputError):
                raise  # schema errors need a different prompt
            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)

                # ── 429 RESOURCE_EXHAUSTED — wait and retry, free attempt ──
                retry_after = _parse_retry_after(exc)
                if retry_after is not None:
                    logger.warning(
                        "429 rate-limit — API requests %.0fs wait. Sleeping (free retry)…",
                        retry_after,
                    )
                    time.sleep(retry_after + 2)
                    regular_attempts -= 1  # restore — 429 is not our error
                    continue

                # ── 503 UNAVAILABLE — model overloaded, wait and free retry ─
                if "503" in exc_str or "UNAVAILABLE" in exc_str:
                    wait_503 = min(backoff * 4, 120)
                    logger.warning(
                        "503 model overloaded — sleeping %.0fs (free retry)…", wait_503,
                    )
                    time.sleep(wait_503)
                    regular_attempts -= 1  # restore
                    continue

                # ── Regular failure — counts against max_retries ───────────
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d failed: %s. Retrying in %.1fs…",
                    regular_attempts, self.max_retries, exc, backoff,
                )
                if regular_attempts >= self.max_retries:
                    break
                time.sleep(backoff)
                backoff *= 2

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
                from google import genai  # noqa: PLC0415
                return genai.Client(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai SDK not installed. Run: pip install google-genai>=1.0"
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
        """Call Google Gemini API using the new google-genai SDK (>=1.0)."""
        from google.genai import types  # noqa: PLC0415

        # Build generation config
        gen_config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "system_instruction": system_prompt,
        }
        if self.top_p is not None:
            gen_config_kwargs["top_p"] = self.top_p

        # Request JSON output when json_object mode is set
        if self.response_format == "json_object":
            gen_config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**gen_config_kwargs)

        raw = self._client.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=config,
        )

        content = raw.text or ""

        if self.response_format == "json_object":
            _validate_json(content)

        # Extract token usage (available in usage_metadata)
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(raw, "usage_metadata") and raw.usage_metadata:
            prompt_tokens = getattr(raw.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(raw.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=content,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=str(raw.candidates[0].finish_reason) if raw.candidates else "stop",
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


def _parse_retry_after(exc: Exception) -> Optional[float]:
    """
    Extract the suggested retry delay (in seconds) from a 429 error.

    Tries in order:
    1. ``exc.details`` list → looks for a ``RetryInfo`` entry with
       ``retryDelay`` (Google genai SDK format).
    2. String search in ``str(exc)`` for ``"Please retry in Xs"`` pattern.

    Returns ``None`` if the exception is not a 429 or no delay is found.
    """
    import re  # noqa: PLC0415

    exc_str = str(exc)

    # Only apply to 429 errors
    if "429" not in exc_str and "RESOURCE_EXHAUSTED" not in exc_str:
        return None

    # Strategy 1: parse from structured details (google-genai SDK)
    try:
        details = getattr(exc, "details", None) or []
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, dict):
                    delay_str = detail.get("retryDelay", "")
                    if delay_str:
                        # e.g. "58s" or "58.13s"
                        m = re.search(r"([\d.]+)s?$", str(delay_str))
                        if m:
                            return float(m.group(1))
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: parse from error message string
    # e.g. "Please retry in 58.130224109s."
    m = re.search(r"[Rr]etry in ([\d.]+)s", exc_str)
    if m:
        return float(m.group(1))

    # Fallback: generic 429 — wait 60s
    return 60.0
