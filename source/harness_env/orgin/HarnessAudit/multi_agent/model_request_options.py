"""Resolved provider request extensions for HarnessAudit model roles."""

from __future__ import annotations

import json
import os
from typing import Any


def role_request_extra_body(
    role: str,
    *,
    model: str,
) -> dict[str, Any]:
    """Return an externally configured role mapping, with legacy fallback.

    Canonical model-stack launches always set the environment value, including
    an explicit empty object. Model-family inference is retained only for old
    direct HarnessAudit entry points that predate model-stack profiles.
    """

    normalized_role = str(role).strip().upper()
    if not normalized_role or not normalized_role.replace("_", "").isalnum():
        raise ValueError(f"Invalid model-stack role: {role!r}")
    name = f"WM_SAFETY_{normalized_role}_REQUEST_EXTRA_BODY_JSON"
    raw = os.environ.get(name)
    if raw is not None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return payload

    normalized_model = str(model or "").strip().lower()
    if "qwen3" in normalized_model:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "deepseek-v4" in normalized_model:
        return {"thinking": {"type": "disabled"}}
    return {}

