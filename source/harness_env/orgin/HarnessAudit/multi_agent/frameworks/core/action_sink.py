"""Action capture sink that emits both normalized actions and JSONL trace events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from multi_agent.schemas.actions import ActionSurface, ObservableAction
from multi_agent.trace.events import (
    CommunicationEvent,
    SafetyAbstentionEvent,
    ToolCallEvent,
)
from multi_agent.trace.writer import TraceWriter


@dataclass
class _PendingAction:
    """Buffered action waiting for global chronological resequencing."""

    ordinal: int
    timestamp: datetime
    surface: ActionSurface
    agent_id: str
    agent_role: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    target_agent: str | None = None
    target_role: str | None = None
    content: str | None = None
    raw_event: dict[str, Any] | None = None


def _coerce_timestamp(value: str | datetime | None) -> datetime:
    """Normalize source timestamps from harness logs into aware UTC datetimes."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ActionSink:
    """Collect normalized actions, then resequence them by source timestamp."""

    def __init__(
        self,
        *,
        run_id: str,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.run_id: str = run_id
        self.trace_writer: TraceWriter | None = trace_writer
        self._actions: list[ObservableAction] = []
        self._pending_actions: list[_PendingAction] = []
        self._sequence = 0
        self._ordinal = 0

    @property
    def actions(self) -> list[ObservableAction]:
        self.finalize()
        return self._actions

    def snapshot_actions(self) -> list[ObservableAction]:
        """Return current actions plus pending actions without mutating the sink."""

        snapshot = list(self._actions)
        sequence = self._sequence
        for pending in sorted(self._pending_actions, key=self._sort_key):
            sequence += 1
            snapshot.append(
                ObservableAction(
                    timestamp=pending.timestamp,
                    sequence_num=sequence,
                    run_id=self.run_id,
                    surface=pending.surface,
                    agent_id=pending.agent_id,
                    agent_role=pending.agent_role,
                    tool_name=pending.tool_name,
                    tool_args=pending.tool_args,
                    tool_result=pending.tool_result,
                    target_agent=pending.target_agent,
                    target_role=pending.target_role,
                    content=pending.content,
                    raw_event=pending.raw_event or {},
                )
            )
        return snapshot

    def restore_actions(self, values: list[ObservableAction | dict[str, Any]]) -> None:
        """Restore one committed action prefix before new actions are accepted."""

        if self._actions or self._pending_actions or self._sequence or self._ordinal:
            raise RuntimeError("ActionSink restore requires a fresh sink")
        restored = [
            value
            if isinstance(value, ObservableAction)
            else ObservableAction.model_validate(value)
            for value in values
        ]
        for expected_sequence, action in enumerate(restored, start=1):
            if action.run_id != self.run_id:
                raise ValueError("Restored action run_id does not match the active run")
            if action.sequence_num != expected_sequence:
                raise ValueError("Restored actions must have contiguous sequence numbers")
        self._actions = [action.model_copy(deep=True) for action in restored]
        self._sequence = len(restored)
        self._ordinal = len(restored)

    def replay_restored_actions_to_trace(self) -> None:
        """Write the committed action prefix after a rebuilt trace_start event."""

        if self._pending_actions:
            raise RuntimeError("Cannot replay restored actions with pending actions")
        if self.trace_writer is None:
            return
        for action in self._actions:
            self._write_action_event(action)

    def serialized_actions(self) -> list[dict[str, Any]]:
        """Return a JSON-safe deep snapshot for a durable turn checkpoint."""

        self.finalize()
        return [action.model_dump(mode="json") for action in self._actions]

    def _write_action_event(self, action: ObservableAction) -> None:
        if self.trace_writer is None:
            return
        common = {
            "trace_id": self.run_id,
            "sequence_num": action.sequence_num,
            "agent_id": action.agent_id,
            "agent_role": action.agent_role,
            "raw_event": action.raw_event or {},
            "timestamp": action.timestamp.isoformat(),
        }
        if action.surface == ActionSurface.TOOL_CALL:
            self.trace_writer.write_event(
                ToolCallEvent(
                    **common,
                    tool_name=action.tool_name or "",
                    tool_args=action.tool_args or {},
                    tool_result=action.tool_result or "",
                )
            )
        else:
            self.trace_writer.write_event(
                CommunicationEvent(
                    **common,
                    target_agent=action.target_agent or "",
                    target_role=action.target_role or "",
                    content=action.content or "",
                )
            )

    def emit_safety_abstention(self, terminal: dict[str, Any]) -> None:
        """Persist a framework-owned non-action terminal after prior actions flush."""

        self.finalize()
        if self.trace_writer is not None:
            self.trace_writer.write_event(
                SafetyAbstentionEvent(trace_id=self.run_id, terminal=terminal)
            )

    def reserve_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _reserve_ordinal(self) -> int:
        self._ordinal += 1
        return self._ordinal

    @staticmethod
    def _sort_key(action: _PendingAction) -> tuple[datetime, str, int, int]:
        raw_event = action.raw_event or {}
        return (
            action.timestamp,
            str(raw_event.get("source_file", "")),
            int(raw_event.get("source_line", 0) or 0),
            action.ordinal,
        )

    def finalize(self) -> None:
        """Materialize buffered actions in globally chronological order."""

        if not self._pending_actions:
            return

        for pending in sorted(self._pending_actions, key=self._sort_key):
            seq = self.reserve_sequence()
            action = ObservableAction(
                timestamp=pending.timestamp,
                sequence_num=seq,
                run_id=self.run_id,
                surface=pending.surface,
                agent_id=pending.agent_id,
                agent_role=pending.agent_role,
                tool_name=pending.tool_name,
                tool_args=pending.tool_args,
                tool_result=pending.tool_result,
                target_agent=pending.target_agent,
                target_role=pending.target_role,
                content=pending.content,
                raw_event=pending.raw_event or {},
            )
            self._actions.append(action)
            self._write_action_event(action)

        self._pending_actions.clear()

    def emit_tool_call(
        self,
        *,
        agent_id: str,
        agent_role: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        raw_event: dict[str, Any] | None = None,
        timestamp: str | datetime | None = None,
    ) -> int:
        self._pending_actions.append(
            _PendingAction(
                ordinal=self._reserve_ordinal(),
                timestamp=_coerce_timestamp(timestamp),
                surface=ActionSurface.TOOL_CALL,
                agent_id=agent_id,
                agent_role=agent_role,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                raw_event=raw_event or {},
            )
        )
        return self._ordinal

    def emit_communication(
        self,
        *,
        agent_id: str,
        agent_role: str,
        target_agent: str,
        target_role: str,
        content: str,
        raw_event: dict[str, Any] | None = None,
        timestamp: str | datetime | None = None,
    ) -> int:
        self._pending_actions.append(
            _PendingAction(
                ordinal=self._reserve_ordinal(),
                timestamp=_coerce_timestamp(timestamp),
                surface=ActionSurface.COMMUNICATION,
                agent_id=agent_id,
                agent_role=agent_role,
                target_agent=target_agent,
                target_role=target_role,
                content=content,
                raw_event=raw_event or {},
            )
        )
        return self._ordinal
