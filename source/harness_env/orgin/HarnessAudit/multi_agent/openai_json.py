"""Shared OpenAI SDK helpers for JSON-only model calls."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import import_module
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .model_request_options import role_request_extra_body


class OpenAIJSONError(RuntimeError):
    """Raised when the OpenAI SDK is unavailable or returns invalid JSON."""


class OpenAIJSONTransportError(OpenAIJSONError):
    """Raised after a transient OpenAI request exhausts its retry budget."""

class OpenAIJSONSemanticError(OpenAIJSONError):
    """Raised when a transported response is not valid requested JSON."""


class OpenAIJSONValidationError(OpenAIJSONSemanticError):
    """Raised when JSON violates the frozen evaluator response schema."""



log = logging.getLogger(__name__)
_ASYNC_OPENAI_CLIENT: Any | None = None
_ASYNC_OPENAI_CLIENT_CONFIG: tuple[str | None, ...] | None = None

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionTimeout",
    "ReadTimeout",
    "Timeout",
}


def _retry_attempts() -> int:
    raw = os.getenv("WM_SAFETY_JUDGE_TRANSPORT_MAX_ATTEMPTS")
    if raw is None:
        raw = os.getenv("MASP_OPENAI_RETRY_ATTEMPTS", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _retry_base_delay() -> float:
    raw = os.getenv("MASP_OPENAI_RETRY_BASE_DELAY", "1.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _is_transient_openai_error(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status in _TRANSIENT_STATUS_CODES:
        return True
    return exc.__class__.__name__ in _TRANSIENT_ERROR_NAMES


def _request_error_message(exc: BaseException) -> str:
    status = _status_code(exc)
    status_part = f" status={status}" if status is not None else ""
    return f"OpenAI request failed{status_part}: {exc}"


def _retry_delay(base_delay: float, attempt: int) -> float:
    return base_delay * (2 ** max(0, attempt - 1))


def _extract_output_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "".join(parts)

    raise OpenAIJSONError("OpenAI response did not contain output text")


def _parse_json_response(response: Any) -> Any:
    content = _extract_output_text(response).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.IGNORECASE | re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", content):
            try:
                parsed, _ = decoder.raw_decode(content[match.start() :])
                return parsed
            except json.JSONDecodeError:
                continue

        raise OpenAIJSONError(f"Model returned non-JSON content: {content[:500]}") from exc


def _response_usage_metadata(response: Any, *, attempt: int) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage_payload = usage.model_dump()
    elif isinstance(usage, dict):
        usage_payload = usage
    else:
        usage_payload = {}
    prompt_tokens = int(
        usage_payload.get("prompt_tokens")
        or usage_payload.get("input_tokens")
        or 0
    )
    completion_tokens = int(
        usage_payload.get("completion_tokens")
        or usage_payload.get("output_tokens")
        or 0
    )
    total_tokens = int(
        usage_payload.get("total_tokens")
        or prompt_tokens + completion_tokens
    )
    return {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "attempt_count": attempt,
        "api_mode": "chat_completions" if _use_chat_completions() else "responses",
    }


def _use_response_format() -> bool:
    if os.getenv("MASP_OPENAI_DISABLE_RESPONSE_FORMAT", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return False
    raw = os.getenv("MASP_OPENAI_RESPONSE_FORMAT")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no"}
    return True


def _use_chat_completions() -> bool:
    model_stack_mode = os.getenv("WM_SAFETY_JUDGE_API_MODE")
    if model_stack_mode:
        if model_stack_mode not in {"chat_completions", "responses"}:
            raise ValueError(
                "WM_SAFETY_JUDGE_API_MODE must be chat_completions or responses"
            )
        return model_stack_mode == "chat_completions"
    base_url = (
        os.getenv("WM_SAFETY_JUDGE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).lower()
    if os.getenv("MASP_OPENAI_USE_CHAT_COMPLETIONS"):
        return os.getenv("MASP_OPENAI_USE_CHAT_COMPLETIONS", "").lower() not in {
            "0",
            "false",
            "no",
        }
    return bool(base_url and "api.openai.com" not in base_url)


def _judge_runtime_config(*, requested_max_tokens: int) -> dict[str, Any]:
    """Resolve the judge role without mutating the actor's global transport."""

    configured_max = os.getenv("WM_SAFETY_JUDGE_MAX_TOKENS")
    max_tokens = max(1, int(requested_max_tokens))
    if configured_max is not None and configured_max.strip():
        profile_max = max(1, int(configured_max))
        # Explicit opt-in for reasoning evaluators; preserve historical caps otherwise.
        if os.getenv('WM_SAFETY_JUDGE_USE_PROFILE_OUTPUT_BUDGET') == '1':
            max_tokens = profile_max
        else:
            max_tokens = min(max_tokens, profile_max)
    timeout_raw = os.getenv("WM_SAFETY_JUDGE_TIMEOUT_SECONDS")
    timeout = float(timeout_raw) if timeout_raw and timeout_raw.strip() else None
    transport_raw = os.getenv("WM_SAFETY_JUDGE_TRANSPORT_MAX_ATTEMPTS")
    transport_max_attempts = (
        max(1, int(transport_raw))
        if transport_raw is not None and transport_raw.strip()
        else _retry_attempts()
    )
    return {
        "api_key": (
            os.getenv("WM_SAFETY_JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        ),
        "base_url": (
            os.getenv("WM_SAFETY_JUDGE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None
        ),
        "temperature": float(os.getenv("WM_SAFETY_JUDGE_TEMPERATURE", "0")),
        "max_tokens": max_tokens,
        "timeout": timeout,
        "transport_max_attempts": transport_max_attempts,
    }


_JUDGE_LEDGER_SCHEMA_VERSION = "harnessaudit-judge-request-ledger-v1"
_JUDGE_LEDGER_COMPONENTS = frozenset(
    {"completion_judge", "operational_judge", "perturbation_judge"}
)
_JUDGE_LEDGER_CONTEXT_FIELDS = frozenset(
    {
        "task_id",
        "policy_mode",
        "intervention_mode",
        "experiment_seed",
        "stage",
        "wm_return_perturbation",
        "wm_return_signal_source_policy",
        "signal_source",
        "local_control_policy",
        "wm_request_expected",
        "oracle_label_horizon",
    }
)


def _judge_ledger_required() -> bool:
    return os.getenv("HARNESS_JUDGE_LEDGER_REQUIRED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _judge_ledger_path() -> str:
    return os.getenv("HARNESS_JUDGE_LEDGER_PATH", "").strip()


def _require_judge_ledger_configuration() -> None:
    if _judge_ledger_required() and not _judge_ledger_path():
        raise OpenAIJSONError(
            "judge request ledger is required but HARNESS_JUDGE_LEDGER_PATH is unset"
        )


def _positive_semantic_attempt(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("semantic_attempt must be a positive integer")
    return value


def judge_semantic_max_attempts() -> int:
    """Return the bounded evaluator semantic-regeneration budget."""

    raw = os.getenv("HARNESS_JUDGE_SEMANTIC_MAX_ATTEMPTS", "1").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise OpenAIJSONError(
            "HARNESS_JUDGE_SEMANTIC_MAX_ATTEMPTS must be a positive integer"
        ) from exc
    if not 1 <= value <= 8:
        raise OpenAIJSONError(
            "HARNESS_JUDGE_SEMANTIC_MAX_ATTEMPTS must be a positive integer no greater than 8"
        )
    return value


def judge_semantic_repair_prompt(
    user_prompt: str,
    *,
    semantic_attempt: int,
) -> str:
    """Make a new semantic request after a transported JSON/schema failure.

    The feedback deliberately does not embed the prior raw response: malformed
    provider output can contain arbitrary text, while the frozen schema is
    already present in the original prompt.
    """

    semantic_attempt = _positive_semantic_attempt(semantic_attempt)
    return (
        f"{user_prompt}\n\n"
        f"Semantic repair attempt {semantic_attempt}: your previous response did "
        "not satisfy the required JSON schema. "
        "Return only a corrected JSON object matching the requested schema exactly."
    )


def judge_semantic_group_id(
    *,
    component: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Build a stable non-secret semantic request-group identity."""

    return _default_semantic_group_id(
        component=component,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def validate_score_reasoning_payload(payload: Any) -> dict[str, Any]:
    """Validate the exact score/reasoning schema used by L2 judges.

    JSON booleans and numeric strings are rejected even though Python would
    normally coerce them through ``float``. A malformed evaluator response is
    an explicit semantic failure, never a zero or missing score.
    """

    if not isinstance(payload, dict) or set(payload) != {"score", "reasoning"}:
        raise OpenAIJSONValidationError(
            "response validation failed: expected exactly the keys score and reasoning"
        )
    score = payload["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise OpenAIJSONValidationError(
            "response validation failed: score must be a finite JSON number"
        )
    normalized_score = float(score)
    if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
        raise OpenAIJSONValidationError(
            "response validation failed: score must be finite and in [0, 1]"
        )
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str):
        raise OpenAIJSONValidationError(
            "response validation failed: reasoning must be a string"
        )
    return {"score": normalized_score, "reasoning": reasoning}


def _safe_judge_endpoint(runtime: dict[str, Any]) -> str:
    raw = str(runtime.get("base_url") or "https://api.openai.com/v1").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise OpenAIJSONError("judge endpoint identity is invalid") from exc
    if not parsed.scheme or not parsed.netloc:
        raise OpenAIJSONError("judge endpoint identity must be an absolute URL")
    endpoint = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc.rsplit("@", 1)[-1],
            parsed.path.rstrip("/"),
            "",
            "",
        )
    ).rstrip("/")
    suffix = "/chat/completions" if _use_chat_completions() else "/responses"
    return endpoint + suffix


def _judge_prompt_sha256(*, system_prompt: str, user_prompt: str) -> str:
    encoded = json.dumps(
        {"system": system_prompt, "user": user_prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_semantic_group_id(
    *,
    component: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    encoded = json.dumps(
        {
            "component": component,
            "model": model,
            "system": system_prompt,
            "user": user_prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "judge:" + hashlib.sha256(encoded).hexdigest()


def _judge_request_context(*, runtime: dict[str, Any]) -> dict[str, Any]:
    """Return only fields permitted by the frozen judge-ledger schema."""

    from SafePred.utils.llm_client import request_context_from_env

    context = request_context_from_env(active_api_key=runtime.get("api_key"))
    return {
        key: context[key]
        for key in _JUDGE_LEDGER_CONTEXT_FIELDS
        if key in context
    }


_JUDGE_REQUIRED_CONTEXT_FIELDS = (
    "task_id",
    "policy_mode",
    "intervention_mode",
    "experiment_seed",
    "stage",
)


def _require_judge_ledger_context(*, runtime: dict[str, Any]) -> dict[str, Any]:
    """Fail before a required judge call if its condition cannot be audited."""

    context = _judge_request_context(runtime=runtime)
    for field in _JUDGE_REQUIRED_CONTEXT_FIELDS:
        value = context.get(field)
        if field == "experiment_seed":
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise OpenAIJSONError(
                "judge request ledger is required but condition context is incomplete"
            )
    return context


def _extract_reasoning_content(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    message = getattr(choices[0], "message", None)
    if isinstance(message, dict):
        value = message.get("reasoning_content")
    else:
        value = getattr(message, "reasoning_content", None)
    return value if isinstance(value, str) and value else None


def _safe_response_text(response: Any) -> str | None:
    try:
        return _extract_output_text(response)
    except OpenAIJSONError:
        return None


def _judge_ledger_row(
    *,
    runtime: dict[str, Any],
    model: str,
    component: str,
    system_prompt: str,
    user_prompt: str,
    semantic_attempt: int,
    semantic_group_id: str,
    transport_attempt_count: int,
    status: str,
    response: Any | None = None,
    response_text: str | None = None,
    error: BaseException | None = None,
    error_kind: str | None = None,
) -> dict[str, Any]:
    if component not in _JUDGE_LEDGER_COMPONENTS:
        raise ValueError(f"unknown judge ledger component: {component!r}")
    semantic_attempt = _positive_semantic_attempt(semantic_attempt)
    if (
        isinstance(transport_attempt_count, bool)
        or not isinstance(transport_attempt_count, int)
        or transport_attempt_count < 0
    ):
        raise ValueError("transport_attempt_count must be a non-negative integer")
    if status not in {"success", "error"}:
        raise ValueError("judge ledger status must be success or error")
    if not isinstance(semantic_group_id, str) or not semantic_group_id:
        raise ValueError("semantic_group_id must be a nonempty string")

    token_usage = None
    if response is not None:
        try:
            token_usage = _response_usage_metadata(
                response,
                attempt=max(1, transport_attempt_count),
            )["usage"]
        except (TypeError, ValueError):
            token_usage = None
    context = (
        _require_judge_ledger_context(runtime=runtime)
        if _judge_ledger_required()
        else _judge_request_context(runtime=runtime)
    )
    row: dict[str, Any] = {
        **context,
        "schema_version": _JUDGE_LEDGER_SCHEMA_VERSION,
        "component": component,
        "model": str(model),
        "endpoint": _safe_judge_endpoint(runtime),
        "api_mode": "chat_completions" if _use_chat_completions() else "responses",
        "prompt_sha256": _judge_prompt_sha256(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ),
        "semantic_group_id": semantic_group_id,
        "semantic_attempt": semantic_attempt,
        "transport_attempt_count": transport_attempt_count,
        "status": status,
        "token_usage": token_usage,
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
    reasoning_content = _extract_reasoning_content(response)
    if reasoning_content is not None:
        row["reasoning_content"] = reasoning_content
    http_status = _status_code(response if response is not None else error) if (
        response is not None or error is not None
    ) else None
    if status == "success":
        if response_text is not None:
            row["response_text"] = response_text
        if http_status is not None:
            row["http_status"] = http_status
        return row

    if error is None or not error_kind:
        raise ValueError("error judge-ledger row requires an error and error_kind")
    row.update(
        {
            "error_kind": error_kind,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "http_status": http_status,
        }
    )
    if response_text is not None:
        row["response_text"] = response_text
    return row


def _append_judge_ledger_row(*, runtime: dict[str, Any], row: dict[str, Any]) -> None:
    path = _judge_ledger_path()
    if not path:
        if _judge_ledger_required():
            raise OpenAIJSONError(
                "judge request ledger is required but HARNESS_JUDGE_LEDGER_PATH is unset"
            )
        return
    try:
        from SafePred.utils.llm_client import append_request_ledger_row

        append_request_ledger_row(
            row,
            active_api_key=runtime.get("api_key"),
            ledger_path=path,
            required=True,
        )
    except Exception as exc:  # noqa: BLE001
        if _judge_ledger_required():
            raise OpenAIJSONError("judge request ledger append failed") from exc
        log.warning("Judge request ledger append failed: %s", type(exc).__name__)


def _serialized_payload_for_ledger(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(payload)


def _semantic_error_kind(error: BaseException) -> str:
    if isinstance(error, OpenAIJSONValidationError):
        return "semantic_validation_error"
    if isinstance(error, OpenAIJSONSemanticError):
        return "semantic_json_error"
    if isinstance(error, OpenAIJSONTransportError):
        return "transport_error"
    return "request_error"


def _validate_response_payload(
    payload: Any,
    *,
    response_validator: Callable[[Any], Any] | None,
) -> Any:
    if response_validator is None:
        return payload
    try:
        return response_validator(payload)
    except OpenAIJSONValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OpenAIJSONValidationError(
            f"response validation failed: {type(exc).__name__}: {str(exc)[:400]}"
        ) from exc
def _close_sync_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to close OpenAI client cleanly: %s", exc)


async def _close_async_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        close = getattr(client, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to close async OpenAI client cleanly: %s", exc)


async def _get_async_openai_client() -> Any:
    """Return a process-local async OpenAI client for JSON judge calls."""

    global _ASYNC_OPENAI_CLIENT, _ASYNC_OPENAI_CLIENT_CONFIG

    openai = import_module("openai")
    runtime = _judge_runtime_config(requested_max_tokens=1)
    config = (
        runtime["api_key"],
        runtime["base_url"],
        str(runtime["timeout"]),
        os.getenv("OPENAI_ORG_ID"),
        os.getenv("OPENAI_PROJECT"),
        str(id(asyncio.get_running_loop())),
    )
    if _ASYNC_OPENAI_CLIENT is None or _ASYNC_OPENAI_CLIENT_CONFIG != config:
        if _ASYNC_OPENAI_CLIENT is not None:
            await _close_async_client(_ASYNC_OPENAI_CLIENT)
        client_kwargs = {
            "api_key": runtime["api_key"],
            "base_url": runtime["base_url"],
            "max_retries": 0,
        }
        if runtime["timeout"] is not None:
            client_kwargs["timeout"] = runtime["timeout"]
        _ASYNC_OPENAI_CLIENT = openai.AsyncOpenAI(**client_kwargs)
        _ASYNC_OPENAI_CLIENT_CONFIG = config
    return _ASYNC_OPENAI_CLIENT


def create_json_response(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1024,
    include_metadata: bool = False,
) -> Any:
    runtime = _judge_runtime_config(requested_max_tokens=max_output_tokens)
    try:
        openai = import_module("openai")
        client_kwargs = {
            "api_key": runtime["api_key"],
            "base_url": runtime["base_url"],
            "max_retries": 0,
        }
        if runtime["timeout"] is not None:
            client_kwargs["timeout"] = runtime["timeout"]
        client = openai.OpenAI(**client_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise OpenAIJSONError(
            "OpenAI SDK is required for non-dry-run generation. "
            "Install the optional oai dependencies and configure OpenAI credentials."
        ) from exc

    attempts = _retry_attempts()
    base_delay = _retry_base_delay()
    try:
        for attempt in range(1, attempts + 1):
            try:
                if _use_chat_completions():
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": runtime["temperature"],
                        "max_tokens": runtime["max_tokens"],
                        "seed": int(os.getenv("MASP_OPENAI_SEED", "0")),
                    }
                    if _use_response_format():
                        payload["response_format"] = {"type": "json_object"}
                    extra_body = role_request_extra_body("judge", model=model)
                    if extra_body:
                        payload["extra_body"] = extra_body
                    response = client.chat.completions.create(**payload)
                else:
                    payload = {
                        "model": model,
                        "instructions": system_prompt,
                        "input": user_prompt,
                        "temperature": runtime["temperature"],
                        "max_output_tokens": runtime["max_tokens"],
                    }
                    extra_body = role_request_extra_body("judge", model=model)
                    if extra_body:
                        payload["extra_body"] = extra_body
                    response = client.responses.create(**payload)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < attempts and _is_transient_openai_error(exc):
                    delay = _retry_delay(base_delay, attempt)
                    log.warning(
                        "Transient OpenAI JSON request failure on attempt %d/%d; "
                        "retrying in %.1fs: %s",
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    if delay:
                        time.sleep(delay)
                    continue
                if _is_transient_openai_error(exc):
                    raise OpenAIJSONTransportError(
                        "OpenAI transport retries exhausted "
                        f"attempts={attempts}: {_request_error_message(exc)}"
                    ) from exc
                raise OpenAIJSONError(_request_error_message(exc)) from exc

        parsed = _parse_json_response(response)
        if not include_metadata:
            return parsed
        return {
            "payload": parsed,
            **_response_usage_metadata(response, attempt=attempt),
        }
    finally:
        _close_sync_client(client)


async def create_json_response_async(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1024,
    include_metadata: bool = False,
    ledger_component: str | None = None,
    semantic_attempt: int = 1,
    semantic_group_id: str | None = None,
    response_validator: Callable[[Any], Any] | None = None,
) -> Any:
    """Call the judge with distinct transport, semantic, and ledger identities."""

    _require_judge_ledger_configuration()
    try:
        semantic_attempt = _positive_semantic_attempt(semantic_attempt)
    except ValueError as exc:
        raise OpenAIJSONError(str(exc)) from exc
    if ledger_component is not None:
        if (
            not isinstance(ledger_component, str)
            or ledger_component not in _JUDGE_LEDGER_COMPONENTS
        ):
            raise OpenAIJSONError("judge ledger component is not approved")
        if semantic_group_id is None:
            semantic_group_id = _default_semantic_group_id(
                component=ledger_component,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        if not isinstance(semantic_group_id, str) or not semantic_group_id:
            raise OpenAIJSONError("judge semantic_group_id must be a nonempty string")
    elif semantic_group_id is not None:
        raise OpenAIJSONError(
            "semantic_group_id requires an approved judge ledger component"
        )
    elif _judge_ledger_required():
        raise OpenAIJSONError(
            "judge request ledger is required but no approved ledger component was supplied"
        )

    runtime = _judge_runtime_config(requested_max_tokens=max_output_tokens)
    if _judge_ledger_required():
        _require_judge_ledger_context(runtime=runtime)

    def append_event(
        *,
        transport_attempt_count: int,
        status: str,
        response: Any | None = None,
        response_text: str | None = None,
        error: BaseException | None = None,
        error_kind: str | None = None,
    ) -> None:
        if ledger_component is None:
            return
        row = _judge_ledger_row(
            runtime=runtime,
            model=model,
            component=ledger_component,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            semantic_attempt=semantic_attempt,
            semantic_group_id=semantic_group_id or "",
            transport_attempt_count=transport_attempt_count,
            status=status,
            response=response,
            response_text=response_text,
            error=error,
            error_kind=error_kind,
        )
        _append_judge_ledger_row(runtime=runtime, row=row)

    try:
        client = await _get_async_openai_client()
    except Exception as exc:  # noqa: BLE001
        failure = OpenAIJSONError(
            "OpenAI SDK is required for LLM-as-judge scoring. "
            "Install the optional oai dependencies and configure OpenAI credentials."
        )
        append_event(
            transport_attempt_count=0,
            status="error",
            error=failure,
            error_kind="client_initialization_error",
        )
        raise failure from exc

    attempts = int(runtime["transport_max_attempts"])
    base_delay = _retry_base_delay()
    for attempt in range(1, attempts + 1):
        try:
            if _use_chat_completions():
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": runtime["temperature"],
                    "max_tokens": runtime["max_tokens"],
                    "seed": int(os.getenv("MASP_OPENAI_SEED", "0")),
                }
                if _use_response_format():
                    payload["response_format"] = {"type": "json_object"}
                extra_body = role_request_extra_body("judge", model=model)
                if extra_body:
                    payload["extra_body"] = extra_body
                response = await client.chat.completions.create(**payload)
            else:
                payload = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": user_prompt,
                    "temperature": runtime["temperature"],
                    "max_output_tokens": runtime["max_tokens"],
                }
                extra_body = role_request_extra_body("judge", model=model)
                if extra_body:
                    payload["extra_body"] = extra_body
                response = await client.responses.create(**payload)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt < attempts and _is_transient_openai_error(exc):
                delay = _retry_delay(base_delay, attempt)
                log.warning(
                    "Transient OpenAI JSON request failure on attempt %d/%d; "
                    "retrying in %.1fs: %s",
                    attempt,
                    attempts,
                    delay,
                    exc,
                )
                if delay:
                    await asyncio.sleep(delay)
                continue
            if _is_transient_openai_error(exc):
                failure = OpenAIJSONTransportError(
                    "OpenAI transport retries exhausted "
                    f"attempts={attempts}: {_request_error_message(exc)}"
                )
                append_event(
                    transport_attempt_count=attempt,
                    status="error",
                    error=exc,
                    error_kind="transport_error",
                )
                raise failure from exc
            failure = OpenAIJSONError(_request_error_message(exc))
            append_event(
                transport_attempt_count=attempt,
                status="error",
                error=exc,
                error_kind="request_error",
            )
            raise failure from exc

    response_text = _safe_response_text(response)
    try:
        parsed = _parse_json_response(response)
    except OpenAIJSONError as exc:
        failure = OpenAIJSONSemanticError("response JSON parsing failed")
        append_event(
            transport_attempt_count=attempt,
            status="error",
            response=response,
            response_text=response_text,
            error=failure,
            error_kind=_semantic_error_kind(failure),
        )
        raise failure from exc
    try:
        validated = _validate_response_payload(
            parsed,
            response_validator=response_validator,
        )
    except OpenAIJSONSemanticError as exc:
        append_event(
            transport_attempt_count=attempt,
            status="error",
            response=response,
            response_text=response_text,
            error=exc,
            error_kind=_semantic_error_kind(exc),
        )
        raise

    append_event(
        transport_attempt_count=attempt,
        status="success",
        response=response,
        response_text=response_text,
    )
    if not include_metadata:
        return validated
    return {
        "payload": validated,
        **_response_usage_metadata(response, attempt=attempt),
    }
