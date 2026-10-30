"""Model-specific OpenAI-compatible request options.

Endpoint location is deliberately irrelevant: an official API, relay, and local
server use the same experiment path. Only the declared model identity controls
whether a vendor-specific request extension is attached.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configured_extra_body(role: str | None) -> dict[str, Any] | None:
    if role is None:
        return None
    normalized_role = str(role).strip().upper()
    if not normalized_role or not normalized_role.replace("_", "").isalnum():
        raise ValueError(f"Invalid model-stack role: {role!r}")
    name = f"WM_SAFETY_{normalized_role}_REQUEST_EXTRA_BODY_JSON"
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


def model_specific_extra_body(
    model: str,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    """Return the externally resolved role extension, with legacy fallback.

    Canonical experiments always provide the role-specific environment value,
    including an explicit empty object. Model-name inference remains only for
    older direct IS-Bench entry points that do not use a model-stack profile.
    """

    configured = _configured_extra_body(role)
    if configured is not None:
        return configured

    normalized = str(model or "").strip().lower()
    if "qwen3" in normalized and _enabled("QWEN_DISABLE_THINKING", default=True):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "deepseek-v4" in normalized and _enabled(
        "DEEPSEEK_DISABLE_THINKING",
        default=True,
    ):
        return {"thinking": {"type": "disabled"}}
    return {}


def request_extra_body_disables_thinking(payload: dict[str, Any]) -> bool:
    """Return whether a resolved extension explicitly disables reasoning."""

    chat_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_kwargs, dict) and chat_kwargs.get("enable_thinking") is False:
        return True
    if payload.get("enable_thinking") is False:
        return True
    thinking = payload.get("thinking")
    return isinstance(thinking, dict) and thinking.get("type") == "disabled"
