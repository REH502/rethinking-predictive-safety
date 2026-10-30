"""
Unified LLM Client Module.

Supports three providers only:
- openai: OpenAI SDK (openai library)
- gemini: Google Gemini SDK (google-generativeai library)
- custom: HTTP via requests (OpenAI-compatible or other APIs)
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Dict, Any, List, Type
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import math
import os
import requests
import json
import re
import time
from urllib.parse import urlsplit
from .logger import get_logger

logger = get_logger("SafePred.LLMClient")

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_CUSTOM = "custom"
_LEDGER_STRING_LIMIT = 4000
_MAX_RETRY_AFTER_SECONDS = 300.0
_LEDGER_EXCEPTION_TYPE_LIMIT = 200
_LEDGER_EXCEPTION_MESSAGE_LIMIT = 2000
_NON_SECRET_API_KEY_SENTINELS = frozenset({"local"})
_CREDENTIAL_FIELD_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "api_key",
    "apikey",
    "access-token",
    "access_token",
}
_QWEN_THINKING_EXTENSION_MODELS = {
    "qwen-agentworld-35b-a3b",
}


def _supports_qwen_thinking_extensions(model_name: str) -> bool:
    """Return whether the served model accepts vLLM Qwen thinking controls."""

    normalized = str(model_name or "").strip().lower()
    return "qwen3" in normalized or normalized in _QWEN_THINKING_EXTENSION_MODELS


def _configured_world_model_request_extra_body() -> Optional[Dict[str, Any]]:
    """Read the public, profile-resolved extension for the world-model role."""

    name = "WM_SAFETY_WORLD_MODEL_REQUEST_EXTRA_BODY_JSON"
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _post_without_loopback_proxy(url: str, **kwargs: Any) -> Any:
    """POST directly to loopback even when the process has hostile proxies."""

    hostname = str(urlsplit(str(url)).hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        with requests.Session() as session:
            session.trust_env = False
            return session.post(url, **kwargs)
    return requests.post(
        url,
        proxies={"http": None, "https": None},
        **kwargs,
    )


def normalize_chat_completions_url(api_url: str) -> str:
    """Return a chat/completions URL without assuming every provider uses /v1."""

    base = str(api_url or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _uses_openai_compatible_chat_endpoint(api_url: str) -> bool:
    """Recognize custom endpoints whose transport is OpenAI chat-completions.

    DeepSeek's official OpenAI-compatible base URL deliberately has no ``/v1``
    suffix.  Its direct HTTP endpoint is ``/chat/completions``.
    """

    base = str(api_url or "").strip()
    lowered = base.lower()
    parsed = urlsplit(base)
    path = (parsed.path or "").rstrip("/").lower()
    official_deepseek_base = (
        str(parsed.hostname or "").lower() == "api.deepseek.com"
        and path in {"", "/v1"}
    )
    official_bigmodel_base = (
        str(parsed.hostname or "").lower() == "open.bigmodel.cn"
        and path == "/api/paas/v4"
    )
    return (
        "/v1" in lowered
        or "openai" in lowered
        or "chat/completions" in lowered
        or official_deepseek_base
        or official_bigmodel_base
    )


def _request_ledger_secrets(active_api_key: Optional[str] = None) -> list[str]:
    """Return active credential values without ever persisting their names."""

    secrets = {
        str(value)
        for name, value in os.environ.items()
        if value
        and str(value).strip().lower() not in _NON_SECRET_API_KEY_SENTINELS
        and (
            "API_KEY" in name.upper()
            or "ACCESS_TOKEN" in name.upper()
            or name.upper().endswith("_TOKEN")
        )
    }
    if (
        active_api_key
        and str(active_api_key).strip().lower() not in _NON_SECRET_API_KEY_SENTINELS
    ):
        secrets.add(str(active_api_key))
    return sorted((value for value in secrets if value), key=len, reverse=True)


def _redact_request_ledger_string(
    value: Any,
    *,
    active_api_key: Optional[str] = None,
    limit: Optional[int] = _LEDGER_STRING_LIMIT,
) -> str:
    """Bound and redact a string before it can reach the request ledger."""

    redacted = str(value or "")
    redacted = re.sub(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)[^&#\s]*",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(?<![?&])\b(?:authorization|proxy-authorization|x-api-key|"
        r"api[_-]?key|access[_-]?token)\s*[:=]\s*"
        r"(?:Bearer\s+)?[^\s,;\"'}\]]+",
        "[REDACTED_CREDENTIAL]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\bBearer\s+[^\s,;\"'}\]]+",
        "[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}",
        "[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\bAIza[0-9A-Za-z_-]{20,}",
        "[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\bgh[pousr]_[A-Za-z0-9]{20,}",
        "[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
        "[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        "[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\bya29\.[0-9A-Za-z_-]{10,}",
        "[REDACTED]",
        redacted,
    )
    for secret in _request_ledger_secrets(active_api_key):
        redacted = redacted.replace(secret, "[REDACTED]")
    if limit is None:
        return redacted
    return redacted[: max(0, int(limit))]


def sanitize_request_ledger_value(
    value: Any,
    *,
    active_api_key: Optional[str] = None,
) -> Any:
    """Recursively redact credential-bearing fields and persisted strings."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        successful_response = value.get("status") == "success"
        for raw_key, raw_value in value.items():
            key = _redact_request_ledger_string(
                raw_key,
                active_api_key=active_api_key,
                limit=200,
            )
            if str(raw_key).strip().lower() in _CREDENTIAL_FIELD_NAMES:
                continue
            if (
                (
                    str(raw_key) == "reasoning_content"
                    or (
                        successful_response
                        and str(raw_key) == "response_text"
                    )
                )
                and isinstance(raw_value, str)
            ):
                sanitized[key] = _redact_request_ledger_string(
                    raw_value,
                    active_api_key=active_api_key,
                    limit=None,
                )
            else:
                sanitized[key] = sanitize_request_ledger_value(
                    raw_value,
                    active_api_key=active_api_key,
                )
        return sanitized
    if isinstance(value, (list, tuple)):
        return [
            sanitize_request_ledger_value(item, active_api_key=active_api_key)
            for item in value
        ]
    if isinstance(value, str):
        return _redact_request_ledger_string(
            value,
            active_api_key=active_api_key,
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_request_ledger_string(
        value,
        active_api_key=active_api_key,
    )


_request_context_override = ContextVar('safepred_request_context_override', default=None)


@contextmanager
def request_context_scope(context: Mapping[str, Any]):
    """Task-local ledger identity; never mutate the shared process environment."""
    token = _request_context_override.set(dict(context))
    try:
        yield
    finally:
        _request_context_override.reset(token)


def request_context_from_env(
    *,
    active_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Load a non-secret request identity without affecting model execution."""

    override = _request_context_override.get()
    if override is not None:
        return sanitize_request_ledger_value(override, active_api_key=active_api_key)
    raw_context = os.environ.get("SAFEPRED_REQUEST_CONTEXT_JSON", "")
    if not raw_context.strip():
        return {}
    try:
        parsed = json.loads(raw_context)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    sanitized = sanitize_request_ledger_value(
        parsed,
        active_api_key=active_api_key,
    )
    return sanitized if isinstance(sanitized, dict) else {}


def append_request_ledger_row(
    row: Dict[str, Any],
    *,
    active_api_key: Optional[str] = None,
    ledger_path: Optional[str | os.PathLike[str]] = None,
    required: bool = False,
) -> None:
    """Append one sanitized JSON object under an advisory file lock.

    ``ledger_path`` permits post-run evaluators to keep their evidence separate
    from the world-model request ledger without mutating process environment.
    """

    raw_path = (
        os.fspath(ledger_path).strip()
        if ledger_path is not None
        else os.environ.get("SAFEPRED_REQUEST_LEDGER_PATH", "").strip()
    )
    if not raw_path:
        if required:
            raise ValueError("request ledger path is required but not configured")
        return
    path = os.path.abspath(os.path.expanduser(raw_path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sanitized = sanitize_request_ledger_value(
        row,
        active_api_key=active_api_key,
    )
    serialized = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _canonical_prompt_sha256(
    *,
    prompt: str,
    messages: Optional[List[Dict[str, str]]],
) -> str:
    effective_messages: Any = (
        messages
        if messages is not None
        else [{"role": "user", "content": prompt}]
    )
    encoded = json.dumps(
        {"messages": effective_messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_sdk_response_envelope(response: Any) -> str:
    """Best-effort JSON serialization for an SDK response, without metadata."""

    for method_name in ("model_dump_json", "model_dump", "to_dict"):
        method = getattr(response, method_name, None)
        if not callable(method):
            continue
        try:
            serialized = method()
            if isinstance(serialized, str):
                return serialized
            return json.dumps(
                serialized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            continue
    return ""


class LLMConnectionError(Exception):
    """Exception raised when LLM API connection fails after maximum retries."""
    
    def __init__(self, message: str, max_retries: int, last_error: Exception):
        """
        Initialize connection error.
        
        Args:
            message: Error message
            max_retries: Maximum number of retry attempts
            last_error: The last exception that occurred
        """
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(message)


class LLMInvalidContentError(ValueError):
    """A completed LLM response did not contain usable assistant content."""


class LLMClient:
    """
    Unified LLM client for calling different providers.
    
    - openai: Uses openai library (OpenAI API)
    - gemini: Uses google-generativeai library (Gemini API)
    - custom: Uses requests library (OpenAI-compatible or other HTTP APIs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: str = "gpt-4",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: LLM API key
            api_url: LLM API URL
            model_name: Model name/identifier
            provider: One of 'openai', 'gemini', 'custom'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Explicit maximum number of attempts. When omitted,
                SAFEPRED_LLM_MAX_ATTEMPTS is used if set, otherwise 3.
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            retry_backoff: Multiplier for exponential backoff (default: 2.0)
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.provider = (provider or "openai").strip().lower()
        if self.provider not in (PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_CUSTOM):
            self.provider = PROVIDER_CUSTOM
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        configured = os.environ.get("SAFEPRED_LLM_MAX_ATTEMPTS")
        self.max_retries = max(
            1,
            int(max_retries) if max_retries is not None
            else int(configured)
            if configured is not None and configured.strip()
            else 3,
        )
        configured_retry_delay = os.environ.get(
            "SAFEPRED_LLM_RETRY_DELAY_SECONDS"
        )
        configured_retry_backoff = os.environ.get(
            "SAFEPRED_LLM_RETRY_BACKOFF"
        )
        self.retry_delay = (
            float(configured_retry_delay)
            if configured_retry_delay is not None
            and configured_retry_delay.strip()
            else float(retry_delay)
        )
        self.retry_backoff = (
            float(configured_retry_backoff)
            if configured_retry_backoff is not None
            and configured_retry_backoff.strip()
            else float(retry_backoff)
        )
        if self.retry_delay < 0:
            raise ValueError("LLM retry delay must be non-negative")
        if self.retry_backoff <= 0:
            raise ValueError("LLM retry backoff must be positive")
        self._last_token_usage = None  # Store token usage from last API call
        self._last_response_snapshot = None
        self._last_transport_attempt_count = 0
        self._token_usage_total = {
            "call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_token_usage(self, usage: Dict[str, Any]) -> None:
        """Record per-call and cumulative token usage when an API returns usage."""

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        self._last_token_usage = {
            "model": usage.get("model") or self.model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self._token_usage_total["call_count"] += 1
        self._token_usage_total["prompt_tokens"] += prompt_tokens
        self._token_usage_total["completion_tokens"] += completion_tokens
        self._token_usage_total["total_tokens"] += total_tokens

    def token_usage_snapshot(self) -> Dict[str, Any]:
        """Return cumulative token usage for this client."""

        return {
            **self._token_usage_total,
            "last": dict(self._last_token_usage) if self._last_token_usage else None,
        }

    def _capture_response_snapshot(
        self,
        response: Any,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Retain response diagnostics before envelope/content validation."""

        snapshot = {
            "http_status": getattr(response, "status_code", None),
            "response_text": str(getattr(response, "text", "") or ""),
            "reasoning_content": self._response_reasoning_content(
                result if result is not None else response
            ),
            "token_usage": None,
        }
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            self._record_token_usage(
                {
                    "model": self.model_name,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            )
            snapshot["token_usage"] = dict(self._last_token_usage)
        elif isinstance(self._last_response_snapshot, dict):
            snapshot["token_usage"] = self._last_response_snapshot.get(
                "token_usage"
            )
        self._last_response_snapshot = snapshot

    @staticmethod
    def _response_reasoning_content(response_or_payload: Any) -> str:
        """Extract provider-exposed CoT without changing the parsed answer."""

        payload = response_or_payload
        choices = (
            payload.get("choices")
            if isinstance(payload, Mapping)
            else getattr(payload, "choices", None)
        )
        if not isinstance(choices, (list, tuple)) or not choices:
            return ""
        first_choice = choices[0]
        message = (
            first_choice.get("message")
            if isinstance(first_choice, Mapping)
            else getattr(first_choice, "message", None)
        )
        reasoning = (
            message.get("reasoning_content")
            if isinstance(message, Mapping)
            else getattr(message, "reasoning_content", None)
        )
        return str(reasoning) if reasoning is not None else ""
    
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using LLM with automatic retry on transient errors.
        
        Args:
            prompt: Text prompt (used if messages is None). If both prompt and messages are None, raises ValueError.
            messages: List of message dicts with 'role' and 'content' keys (used if provided, overrides prompt)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        
        Raises:
            ValueError: If both prompt and messages are None
            LLMConnectionError: If all retry attempts fail and connection cannot be established.
                This exception will terminate the program execution.
            Exception: For non-retryable errors (raised immediately without retry)
        """
        # Validate that at least one input is provided
        if prompt is None and messages is None:
            raise ValueError("LLMClient.generate() requires either 'prompt' or 'messages' argument")
        
        # If prompt is None but messages is provided, set prompt to empty string
        # This maintains backward compatibility with internal implementations
        if prompt is None:
            prompt = ""
        
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        request_ledger_attempt = kwargs.pop("request_ledger_attempt", 1)
        if (
            isinstance(request_ledger_attempt, bool)
            or not isinstance(request_ledger_attempt, int)
            or request_ledger_attempt < 1
        ):
            raise ValueError("request_ledger_attempt must be a positive integer")
        if "seed" not in kwargs:
            seed = os.environ.get("SAFEPRED_WM_SEED")
            if seed is not None and str(seed).strip():
                kwargs["seed"] = int(seed)

        prompt_sha256 = _canonical_prompt_sha256(
            prompt=prompt,
            messages=messages,
        )
        endpoint = self._request_endpoint()
        self._last_token_usage = None
        self._last_response_snapshot = None
        self._last_transport_attempt_count = 0
        
        # Dispatch by provider: openai (SDK), gemini (SDK), custom (requests)
        if self.provider == PROVIDER_OPENAI:
            generate_func = lambda: self._generate_openai(prompt, messages, temp, max_toks, **kwargs)
        elif self.provider == PROVIDER_GEMINI:
            generate_func = lambda: self._generate_gemini(prompt, messages, temp, max_toks, **kwargs)
        else:
            generate_func = lambda: self._generate_custom(prompt, messages, temp, max_toks, **kwargs)
        
        # Execute with retry mechanism. Ledger persistence is best-effort:
        # it must never replace the model result or the original model error.
        try:
            response_text = self._execute_with_retry(generate_func)
        except Exception as exc:
            ledger_exc = (
                exc.last_error
                if isinstance(exc, LLMConnectionError)
                else exc
            )
            response = getattr(ledger_exc, "response", None)
            response_snapshot = (
                self._last_response_snapshot
                if isinstance(self._last_response_snapshot, dict)
                else {}
            )
            response_text = str(
                response_snapshot.get("response_text")
                if "response_text" in response_snapshot
                else getattr(response, "text", "")
                if response is not None
                else ""
            )
            row = {
                **request_context_from_env(active_api_key=self.api_key),
                "timestamp_utc": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "component": "world_model",
                "model": self.model_name,
                "endpoint": endpoint,
                "prompt_sha256": prompt_sha256,
                "attempt": request_ledger_attempt,
                "semantic_attempt": request_ledger_attempt,
                "transport_attempt_count": self._last_transport_attempt_count,
                "status": "error",
                "response_text": response_text,
                "token_usage": (
                    response_snapshot.get("token_usage")
                    if response_snapshot.get("token_usage") is not None
                    else dict(self._last_token_usage)
                    if self._last_token_usage is not None
                    else None
                ),
                "exception_type": type(ledger_exc).__name__[
                    :_LEDGER_EXCEPTION_TYPE_LIMIT
                ],
                "exception_message": _redact_request_ledger_string(
                    ledger_exc,
                    active_api_key=self.api_key,
                    limit=_LEDGER_EXCEPTION_MESSAGE_LIMIT,
                ),
            }
            error_reasoning_content = str(
                response_snapshot.get("reasoning_content") or ""
            )
            if error_reasoning_content:
                row["reasoning_content_present"] = True
                row["reasoning_content"] = error_reasoning_content
            status_code = response_snapshot.get("http_status")
            if status_code is None:
                status_code = getattr(response, "status_code", None)
            if status_code is not None:
                row["http_status"] = int(status_code)
            self._try_append_request_ledger(row)
            raise

        response_snapshot = (
            self._last_response_snapshot
            if isinstance(self._last_response_snapshot, dict)
            else {}
        )
        reasoning_content = str(
            response_snapshot.get("reasoning_content") or ""
        )
        success_row = {
            **request_context_from_env(active_api_key=self.api_key),
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "component": "world_model",
            "model": self.model_name,
            "endpoint": endpoint,
            "prompt_sha256": prompt_sha256,
            "attempt": request_ledger_attempt,
            "semantic_attempt": request_ledger_attempt,
            "transport_attempt_count": self._last_transport_attempt_count,
            "status": "success",
            "response_text": response_text,
            "token_usage": (
                dict(self._last_token_usage)
                if self._last_token_usage is not None
                else None
            ),
        }
        if reasoning_content:
            success_row["reasoning_content_present"] = True
            success_row["reasoning_content"] = reasoning_content
        status_code = response_snapshot.get("http_status")
        if status_code is not None:
            success_row["http_status"] = int(status_code)
        self._try_append_request_ledger(success_row)
        return response_text

    def _request_endpoint(self) -> str:
        base = str(self.api_url or "").strip()
        if self.provider == PROVIDER_OPENAI:
            return normalize_chat_completions_url(
                base or "https://api.openai.com/v1"
            )
        if self.provider == PROVIDER_CUSTOM and _uses_openai_compatible_chat_endpoint(
            base
        ):
            return normalize_chat_completions_url(base)
        return base

    def _try_append_request_ledger(self, row: Dict[str, Any]) -> None:
        try:
            append_request_ledger_row(
                row,
                active_api_key=self.api_key,
            )
        except Exception as ledger_error:
            logger.warning(
                "[LLM Client] Request ledger append failed: %s",
                _redact_request_ledger_string(
                    type(ledger_error).__name__,
                    active_api_key=self.api_key,
                    limit=200,
                ),
            )
    
    def _generate_openai(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI SDK (openai library)."""
        from openai import OpenAI

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        client_kwargs = {"api_key": self.api_key}
        if self.api_url:
            client_kwargs["base_url"] = self.api_url.rstrip("/")
        client_kwargs["max_retries"] = 0
        client = OpenAI(**client_kwargs)

        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        # These are vLLM/Qwen chat-template extensions used by the custom
        # provider.  The official OpenAI SDK does not accept them.
        create_kwargs.pop("enable_thinking", None)
        create_kwargs.pop("chat_template_kwargs", None)
        model_lower = self.model_name.lower()
        configured_extra_body = _configured_world_model_request_extra_body()
        if configured_extra_body is not None:
            existing_extra_body = create_kwargs.get("extra_body")
            if existing_extra_body is not None and not isinstance(
                existing_extra_body,
                Mapping,
            ):
                raise ValueError(
                    "World-model extra_body must be a mapping when provided"
                )
            merged_extra_body = dict(existing_extra_body or {})
            merged_extra_body.update(configured_extra_body)
            if merged_extra_body:
                create_kwargs["extra_body"] = merged_extra_body
            else:
                create_kwargs.pop("extra_body", None)
        elif model_lower.startswith("deepseek-v4"):
            existing_extra_body = create_kwargs.get("extra_body")
            if existing_extra_body is not None and not isinstance(
                existing_extra_body,
                Mapping,
            ):
                raise ValueError(
                    "DeepSeek V4 extra_body must be a mapping when provided"
                )
            extra_body = dict(existing_extra_body or {})
            extra_body["thinking"] = {"type": "disabled"}
            create_kwargs["extra_body"] = extra_body
        if "gpt-5" in model_lower or "o3" in model_lower or "o4" in model_lower:
            create_kwargs.pop("temperature", None)
            create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens", max_tokens)

        response = client.chat.completions.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        usage_result = None
        if usage is not None:
            usage_result = {
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        usage,
                        "completion_tokens",
                        0,
                    ),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            }
        self._capture_response_snapshot(response, usage_result)
        response_text = _serialize_sdk_response_envelope(response)
        if (
            response_text
            and isinstance(self._last_response_snapshot, dict)
        ):
            self._last_response_snapshot["response_text"] = response_text
            self._last_response_snapshot["reasoning_content"] = (
                self._response_reasoning_content(response)
            )

        content = response.choices[0].message.content
        if content is None:
            raise LLMInvalidContentError("LLM returned None content")
        return str(content)

    def _generate_gemini(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Gemini SDK (google-generativeai library)."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        contents = prompt
        if messages:
            contents = self._messages_to_prompt(messages)

        try:
            config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if kwargs.get("seed") is not None:
                config_kwargs["seed"] = int(kwargs["seed"])
            response_format = kwargs.get("response_format")
            if response_format is not None:
                if response_format != {"type": "json_object"}:
                    raise ValueError(
                        "Gemini supports only response_format={'type': 'json_object'}"
                    )
                config_kwargs["response_mime_type"] = "application/json"
            gen_config = genai.GenerationConfig(**config_kwargs)
        except Exception as exc:
            if (
                kwargs.get("response_format") is not None
                and os.environ.get("SAFEPRED_REQUIRE_RESPONSE_FORMAT") == "1"
            ):
                raise RuntimeError(
                    "Formal prediction requires Gemini JSON response MIME support; "
                    "the installed SDK/configuration rejected it"
                ) from exc
            gen_config = None  # use SDK defaults
        response = model.generate_content(contents, generation_config=gen_config)
        text = getattr(response, "text", None) if response else None
        if not text:
            raise ValueError("Gemini returned empty or None content")
        return text

    def _generate_requests(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using requests for OpenAI-compatible API (used by provider=custom)."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Check if model requires special parameters
        model_lower = self.model_name.lower()
        configured_extra_body = _configured_world_model_request_extra_body()
        qwen_thinking_extensions = _supports_qwen_thinking_extensions(
            self.model_name
        )
        if configured_extra_body is not None:
            qwen_thinking_extensions = False
        if not qwen_thinking_extensions:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("chat_template_kwargs", None)
        if "gpt-5" in model_lower or "o3" in model_lower or "o4" in model_lower:
            # Use max_completion_tokens for newer models
            # Don't pass temperature - these models only support default temperature=1
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                **kwargs
            }
            # Remove temperature from kwargs if present (these models don't support it)
            payload.pop("temperature", None)
        elif qwen_thinking_extensions:
            # Qwen3 models require enable_thinking parameter (default to False)
            # If enable_thinking is explicitly set in kwargs, it will override the default
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": False,  # Default for qwen3 models
                "chat_template_kwargs": {"enable_thinking": False},
                **kwargs  # kwargs can override enable_thinking if explicitly set
            }
        else:
            # Use max_tokens for standard models
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

        if configured_extra_body is not None:
            payload.update(configured_extra_body)
        elif "deepseek-v4" in model_lower:
            # DeepSeek V4 defaults to thinking mode, which can spend the whole
            # completion budget in reasoning_content and leave content empty.
            payload["thinking"] = {"type": "disabled"}
        
        # For requests, we need the full endpoint URL: https://api.openai-proxy.org/v1/chat/completions
        # Input URL is base URL: https://api.openai-proxy.org/v1
        # IMPORTANT: Always use self.api_url from config.yaml, never use a cached or modified URL
        base_api_url = self.api_url or "https://api.openai.com/v1"
        if base_api_url != self.api_url:
            logger.warning(f"[LLM Client] Using default URL instead of config: {base_api_url}")
        
        api_url = normalize_chat_completions_url(base_api_url)
        
        # Log the URL being used (for debugging retry issues)
        logger.debug(f"[LLM Client] Using API URL from config: {self.api_url} -> {api_url}")
        
        # Explicitly disable proxy to avoid using system proxy (e.g., 127.0.0.1:7897)
        # which could cause URL to change during retries
        response = _post_without_loopback_proxy(
            api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        self._capture_response_snapshot(response)
        
        # Check if error is due to max_tokens not being supported
        if response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', '') or str(error_data)
                # Log detailed error information
                logger.error("=" * 80)
                logger.error(f"[LLM Client] 400 Bad Request - Model: {self.model_name}")
                logger.error(f"API URL: {api_url}")
                logger.error(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                logger.error(f"Error response: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                logger.error(f"Error message: {error_msg}")
                logger.error("=" * 80)
                
                if 'max_tokens' in error_msg.lower() and ('max_completion_tokens' in error_msg.lower() or 'unsupported parameter' in error_msg.lower()):
                    if self.max_retries == 1:
                        raise requests.exceptions.HTTPError(
                            f"400 Bad Request: {error_msg}. "
                            f"API URL: {api_url}, Model: {self.model_name}. "
                            "Formal one-attempt mode disables parameter resubmission.",
                            response=response,
                        )
                    # Retry with max_completion_tokens instead
                    logger.warning(f"[LLM Client] Model {self.model_name} doesn't support max_tokens, retrying with max_completion_tokens")
                    payload.pop('max_tokens', None)
                    payload['max_completion_tokens'] = max_tokens
                    # Remove temperature if present (some models don't support it)
                    payload.pop('temperature', None)
                    response = _post_without_loopback_proxy(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    self._capture_response_snapshot(response)
                    # If retry also fails, log the error again
                    if response.status_code == 400:
                        retry_error_msg = error_msg  # Default to original error message
                        try:
                            retry_error_data = response.json()
                            retry_error_msg = retry_error_data.get('error', {}).get('message', '') or str(retry_error_data)
                            logger.error("=" * 80)
                            logger.error(f"[LLM Client] Retry also failed with 400 Bad Request")
                            logger.error(f"Retry payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                            logger.error(f"Retry error response: {json.dumps(retry_error_data, indent=2, ensure_ascii=False)}")
                            logger.error(f"Retry error message: {retry_error_msg}")
                            logger.error("=" * 80)
                        except Exception:
                            logger.error(f"[LLM Client] Retry error response text: {response.text[:500]}")
                        # Raise exception with detailed error information
                        raise requests.exceptions.HTTPError(
                            f"400 Bad Request: {retry_error_msg}. "
                            f"API URL: {api_url}, Model: {self.model_name}. "
                            f"Retry with max_completion_tokens also failed.",
                            response=response
                        )
                else:
                    # Not a max_tokens error, raise immediately with detailed error information
                    raise requests.exceptions.HTTPError(
                        f"400 Bad Request: {error_msg}. "
                        f"API URL: {api_url}, Model: {self.model_name}. "
                        f"This is not a max_tokens issue, cannot auto-retry.",
                        response=response
                    )
            except requests.exceptions.HTTPError:
                # Re-raise HTTPError (already formatted with detailed info)
                raise
            except (ValueError, KeyError) as e:
                # If we can't parse the error, log the raw response and raise with details
                logger.error("=" * 80)
                logger.error(f"[LLM Client] Failed to parse 400 error response")
                logger.error(f"Raw response text: {response.text[:1000]}")
                logger.error(f"Parse error: {e}")
                logger.error("=" * 80)
                raise requests.exceptions.HTTPError(
                    f"400 Bad Request: Failed to parse error response. "
                    f"API URL: {api_url}, Model: {self.model_name}. "
                    f"Raw response: {response.text[:500]}",
                    response=response
                ) from e
        
        response.raise_for_status()
        result = response.json()
        self._capture_response_snapshot(response, result)
        
        # Debug: Log response structure
        logger.debug(f"[LLM Client] API Response (requests) - Model: {self.model_name}")
        logger.debug(f"[LLM Client] Response status: {response.status_code}")
        logger.debug(f"[LLM Client] Response keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error("=" * 80)
            logger.error("[LLM Client] No choices in response (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Response JSON: {result}")
            logger.error("=" * 80)
            raise ValueError("LLM response has no choices")
        
        first_choice = result["choices"][0]
        message = (
            first_choice.get("message")
            if isinstance(first_choice, dict)
            else None
        )
        if not isinstance(message, dict) or "content" not in message:
            logger.error("=" * 80)
            logger.error(
                "[LLM Client] LLM response message is missing content "
                "(requests)"
            )
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Full response JSON: {result}")
            logger.error("=" * 80)
            raise ValueError("LLM response message is missing content")
        content = message["content"]
        if content is None:
            # Log full response for debugging
            logger.error("=" * 80)
            logger.error("[LLM Client] LLM returned None content (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Full response JSON: {result}")
            logger.error("=" * 80)
            raise LLMInvalidContentError(
                "LLM returned None content in response"
            )
        content_str = str(content)
        # Token usage was captured before envelope/content validation.
        if "usage" in result:
            usage = result["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            logger.info(
                f"[LLM Client] Token usage - Model: {self.model_name}, "
                f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
            )
        # Log if content is empty
        if not content_str or len(content_str.strip()) == 0:
            logger.error("=" * 80)
            logger.error("[LLM Client] LLM returned empty content (requests)")
            logger.error("=" * 80)
            logger.error(f"Model: {self.model_name}")
            logger.error(f"Payload: {payload}")
            logger.error(f"Response JSON: {result}")
            logger.error(f"Content (repr): {repr(content)}")
            logger.error("=" * 80)
        return content_str

    def _generate_custom(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using requests for custom provider."""
        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
            "Content-Type": "application/json",
        }
        # Remove None values
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Prepare messages if not provided
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Check if API URL looks like OpenAI-compatible format (contains /v1)
        # If so, use OpenAI-format requests instead
        if _uses_openai_compatible_chat_endpoint(self.api_url or ""):
            # Use OpenAI-compatible format
            return self._generate_requests(prompt, messages, temperature, max_tokens, **kwargs)
        
        # Custom providers may use different formats
        # Try prompt-based format first
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # If messages provided, try messages format
        if messages:
            payload_messages = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
        else:
            payload_messages = payload
        
        # IMPORTANT: Always use self.api_url from config.yaml, never use a cached or modified URL
        # Log the URL being used (for debugging retry issues)
        logger.debug(f"[LLM Client] Using API URL from config: {self.api_url}")
        
        # Try messages format first, then prompt format
        last_error = None
        payloads_to_try = (
            [payload_messages]
            if self.max_retries == 1
            else [payload_messages, payload]
        )
        for idx, payload_to_try in enumerate(payloads_to_try):
            format_name = "messages" if idx == 0 else "prompt"
            try:
                # Explicitly disable proxy to avoid using system proxy (e.g., 127.0.0.1:7897)
                # which could cause URL to change during retries
                response = _post_without_loopback_proxy(
                    self.api_url,  # Always use self.api_url from config.yaml
                    headers=headers,
                    json=payload_to_try,
                    timeout=self.timeout,
                )
                self._capture_response_snapshot(response)
                response.raise_for_status()
                result = response.json()
                self._capture_response_snapshot(response, result)
                
                # Token usage was captured before response-text extraction.
                if "usage" in result:
                    usage = result["usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    logger.info(
                        f"[LLM Client] Token usage (Custom) - Model: {self.model_name}, "
                        f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
                    )
                
                # Extract text from various response formats
                return self._extract_text_from_response(result)
                
            except requests.exceptions.RequestException:
                raise
            except Exception as e:
                last_error = e
                logger.debug(f"[LLM Client] Custom API call failed with {format_name} format: {e}")
                # Log response details if available
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_result = e.response.json()
                        logger.debug(f"[LLM Client] Error response JSON: {error_result}")
                    except Exception:
                        logger.debug(f"[LLM Client] Error response text: {e.response.text[:500]}")
                continue
        
        # If all attempts failed, raise with diagnostic information
        error_msg = f"Failed to extract text from custom API response"
        if last_error:
            error_msg += f": {last_error}"
        logger.error("=" * 80)
        logger.error(f"[LLM Client] {error_msg}")
        logger.error("=" * 80)
        logger.error(f"API URL: {self.api_url}")
        logger.error(f"Model: {self.model_name}")
        logger.error(f"Provider: {self.provider}")
        logger.error(f"Last error: {last_error}")
        logger.error("=" * 80)
        raise Exception(error_msg)
    
    def _extract_text_from_response(self, result: Dict[str, Any]) -> str:
        """
        Extract text from various API response formats.
        
        Args:
            result: API response dictionary
        
        Returns:
            Extracted text content
        
        Raises:
            ValueError: If text cannot be extracted
        """
        # Try different response formats
        if "text" in result:
            return result["text"]
        elif "content" in result:
            return result["content"]
        elif "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "")
            elif "text" in choice:
                return choice["text"]
        elif "output" in result:
            if isinstance(result["output"], str):
                return result["output"]
            elif "text" in result["output"]:
                return result["output"]["text"]
        
        # If no standard format found, return first string value
        for value in result.values():
            if isinstance(value, str) and value:
                return value
        
        raise ValueError("Failed to extract text from response")
    
    def _execute_with_retry(self, func, max_retries: Optional[int] = None) -> Any:
        """
        Execute a function with retry mechanism for transient errors.
        
        Args:
            func: Function to execute (callable that takes no arguments)
            max_retries: Override default max_retries (optional)
        
        Returns:
            Function result
        
        Raises:
            LLMConnectionError: If all retry attempts fail and connection cannot be established.
                This exception will terminate the program execution.
            Exception: For non-retryable errors (raised immediately without retry).
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        retries = 0
        current_delay = self.retry_delay
        
        # Define retryable exceptions
        retryable_exceptions = self._get_retryable_exceptions()
        
        last_exception = None
        
        while retries < max_retries:
            try:
                self._last_transport_attempt_count += 1
                return func()
            except requests.exceptions.HTTPError as e:
                # Special handling for HTTP errors - check if retryable
                if self._is_retryable_http_error(e):
                    retries += 1
                    last_exception = e
                    
                    # Check if we've exhausted retries
                    if retries >= max_retries:
                        error_msg = (
                            f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                            f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                            f"Last error: {type(e).__name__}: {e}"
                        )
                        logger.error("=" * 80)
                        logger.error(error_msg)
                        logger.error("=" * 80)
                        raise LLMConnectionError(error_msg, max_retries, e) from e
                    
                    # Log retry attempt with URL from config
                    response = getattr(e, "response", None)
                    status_code = getattr(response, "status_code", "unknown")
                    retry_delay = self._http_retry_delay(e, current_delay)
                    logger.warning(
                        f"[LLM Client] Retryable HTTP error (status {status_code}): {e} "
                        f"(attempt {retries}/{max_retries}, waiting {retry_delay:.2f}s) "
                        f"Using API URL from config: {self.api_url}"
                    )
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(retry_delay)
                    current_delay *= self.retry_backoff
                else:
                    # Non-retryable HTTP error (e.g., 4xx except 429), raise immediately
                    logger.error(f"[LLM Client] Non-retryable HTTP error: {e}")
                    raise
            except tuple(retryable_exceptions) as e:
                retries += 1
                last_exception = e
                
                # Check if we've exhausted retries
                if retries >= max_retries:
                    error_type = type(e).__name__
                    error_msg = (
                        f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                        f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                        f"Last error ({error_type}): {e}"
                    )
                    logger.error("=" * 80)
                    logger.error(error_msg)
                    logger.error("=" * 80)
                    raise LLMConnectionError(error_msg, max_retries, e) from e
                
                # Log retry attempt with URL from config
                error_type = type(e).__name__
                retry_delay = self._http_retry_delay(e, current_delay)
                logger.warning(
                    f"[LLM Client] Retryable error ({error_type}): {e} "
                    f"(attempt {retries}/{max_retries}, waiting {retry_delay:.2f}s) "
                    f"Using API URL from config: {self.api_url}"
                )
                
                # Wait before retrying (exponential backoff)
                time.sleep(retry_delay)
                current_delay *= self.retry_backoff
                
            except Exception as e:
                # Non-retryable exception, raise immediately
                logger.error(f"[LLM Client] Non-retryable error: {e}")
                raise
        
        # Should not reach here, but just in case (fallback)
        if last_exception:
            error_msg = (
                f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}. "
                f"Last error: {type(last_exception).__name__}: {last_exception}"
            )
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise LLMConnectionError(error_msg, max_retries, last_exception) from last_exception
        else:
            error_msg = (
                f"[LLM Client] Failed to connect to LLM API after {max_retries} retry attempts. "
                f"Provider: {self.provider}, Model: {self.model_name}, API URL: {self.api_url}."
            )
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise LLMConnectionError(error_msg, max_retries, Exception("Unknown error"))
    
    def _get_retryable_exceptions(self) -> List[Type[Exception]]:
        """
        Get list of retryable exception types based on provider.
        
        Returns:
            List of exception types that should trigger retry
        """
        retryable = []
        
        # Common network/HTTP errors (from requests library)
        retryable.extend([
            requests.exceptions.ConnectionError,  # Connection failed
            requests.exceptions.Timeout,  # Request timeout
            requests.exceptions.HTTPError,  # HTTP error (will check status code)
            requests.exceptions.SSLError,  # SSL/TLS connection error
            requests.exceptions.RequestException,  # Base exception for all requests errors
        ])
        
        # OpenAI SDK exceptions (provider=openai)
        if self.provider == PROVIDER_OPENAI:
            try:
                import openai
                retryable.extend([
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ])
            except ImportError:
                pass
        # Gemini/Google API exceptions (provider=gemini)
        if self.provider == PROVIDER_GEMINI:
            try:
                from google.api_core.exceptions import (
                    ResourceExhausted,
                    ServiceUnavailable,
                    DeadlineExceeded,
                )
                retryable.extend([ResourceExhausted, ServiceUnavailable, DeadlineExceeded])
            except ImportError:
                pass
        return retryable
    
    @staticmethod
    def _retry_after_seconds(error: BaseException) -> Optional[float]:
        """Parse a bounded Retry-After delta or HTTP date from an error."""

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        raw_value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if raw_value is None:
            return None
        try:
            provider_delay = float(raw_value)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(raw_value))
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            provider_delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        if not math.isfinite(provider_delay) or provider_delay < 0:
            return None
        return min(provider_delay, _MAX_RETRY_AFTER_SECONDS)

    @classmethod
    def _http_retry_delay(
        cls, error: BaseException, current_delay: float
    ) -> float:
        """Use provider backpressure without weakening local backoff."""

        retry_after = cls._retry_after_seconds(error)
        return max(float(current_delay), retry_after or 0.0)

    def _is_retryable_http_error(self, error: requests.exceptions.HTTPError) -> bool:
        """
        Check if an HTTP error should trigger retry.
        
        Args:
            error: HTTPError exception
        
        Returns:
            True if error is retryable (5xx, 429, or 404 for custom providers), False otherwise
        """
        if not hasattr(error, 'response') or error.response is None:
            return True  # If we can't check, assume retryable
        
        status_code = error.response.status_code
        
        # Retry on server errors (5xx) and rate limits (429)
        # Also retry on 404 for custom providers (vLLM may return 404 temporarily during model loading)
        if status_code >= 500 or status_code == 429:
            return True
        
        # For custom providers (like vLLM), 404 might be temporary (model loading, server restart)
        # Check if this is a custom provider by checking if api_url contains non-standard domains
        if status_code == 404 and self.provider == PROVIDER_CUSTOM:
            # Check if error message suggests model not found (permanent) vs server issue (temporary)
            error_msg = ""
            try:
                if hasattr(error, 'response') and error.response:
                    error_data = error.response.json()
                    error_msg = str(error_data.get('error', {}).get('message', '')).lower()
            except:
                pass
            
            # If error message indicates model doesn't exist, don't retry
            if 'model' in error_msg and ('does not exist' in error_msg or 'not found' in error_msg):
                return False
            
            # Otherwise, treat 404 as retryable for custom providers (might be temporary)
            return True
        
        return False
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts)
    
    @classmethod
    def from_config(cls, llm_config: Dict[str, Any]) -> "LLMClient":
        """
        Create LLMClient from configuration dictionary.
        
        Args:
            llm_config: Configuration dict from SafetyConfig.get_llm_config()
        
        Returns:
            LLMClient instance
        """
        return cls(
            api_key=llm_config.get("api_key"),
            api_url=llm_config.get("api_url"),
            model_name=llm_config.get("model_name"),
            provider=llm_config.get("provider", PROVIDER_OPENAI),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 512),
            timeout=llm_config.get("timeout", 30),
            max_retries=llm_config.get("max_retries", 3),
            retry_delay=llm_config.get("retry_delay", 1.0),
            retry_backoff=llm_config.get("retry_backoff", 2.0),
        )
