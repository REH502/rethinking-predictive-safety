"""Action serialization helpers for the SafePred guard."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


ACTION_TYPES = {"tool_call", "communication"}


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def stable_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)


def action_key(action: dict[str, Any]) -> str:
    keep = {
        "type": action.get("type"),
        "agent_role": action.get("agent_role"),
        "target_role": action.get("target_role"),
        "tool_name": action.get("tool_name"),
        "tool_args": action.get("tool_args") or {},
        "content": action.get("content") or "",
    }
    return stable_json(keep)


def dedupe_actions(actions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        if action.get("type") not in ACTION_TYPES:
            continue
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        out.append(copy.deepcopy(action))
        if len(out) >= limit:
            break
    return out


def normalize_selected_action(selected: Any) -> dict[str, Any] | None:
    if selected is None:
        return None
    if isinstance(selected, dict):
        return copy.deepcopy(selected)
    if isinstance(selected, str):
        text = selected.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        # Try human-readable format: "role communicates to target: content"
        # or "role calls tool with args {...}"
        return _parse_human_readable_action(text)
    return None


def _parse_human_readable_action(text: str) -> dict[str, Any] | None:
    import re
    comm_match = re.match(r"^(\S+)\s+communicates\s+to\s+(\S+):\s*(.*)$", text, flags=re.DOTALL)
    if comm_match:
        return {
            "type": "communication",
            "agent_role": comm_match.group(1),
            "target_role": comm_match.group(2),
            "content": comm_match.group(3).strip(),
        }
    tool_match = re.match(r"^(\S+)\s+calls\s+(\S+)\s+with\s+args\s+(.+)$", text, flags=re.DOTALL)
    if tool_match:
        try:
            tool_args = json.loads(tool_match.group(3))
        except json.JSONDecodeError:
            tool_args = {}
        return {
            "type": "tool_call",
            "agent_role": tool_match.group(1),
            "tool_name": tool_match.group(2),
            "tool_args": tool_args if isinstance(tool_args, dict) else {},
        }
    return None


def observable_to_event(action: Any) -> dict[str, Any]:
    raw = to_jsonable(action)
    if not isinstance(raw, dict):
        return {}
    surface = raw.get("surface")
    event_type = raw.get("type") or surface
    out = {
        "type": event_type,
        "sequence_num": raw.get("sequence_num"),
        "agent_id": raw.get("agent_id"),
        "agent_role": raw.get("agent_role"),
        "timestamp": raw.get("timestamp"),
    }
    if event_type == "tool_call":
        out.update(
            {
                "tool_name": raw.get("tool_name"),
                "tool_args": raw.get("tool_args") or {},
                "tool_result": raw.get("tool_result") or "",
                "raw_event": raw.get("raw_event") or {},
            }
        )
    elif event_type == "communication":
        out.update(
            {
                "target_agent": raw.get("target_agent"),
                "target_role": raw.get("target_role"),
                "content": raw.get("content") or "",
            }
        )
    return out


def proposal_text(action: dict[str, Any]) -> str:
    if action.get("type") == "tool_call":
        return (
            f"{action.get('agent_role')} calls {action.get('tool_name')} "
            f"with args {stable_json(action.get('tool_args') or {})}"
        )
    if action.get("type") == "communication":
        return (
            f"{action.get('agent_role')} communicates to "
            f"{action.get('target_role')}: {action.get('content') or ''}"
        )
    return stable_json(action)
