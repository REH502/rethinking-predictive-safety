"""Execute SafePred-selected HarnessAudit actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multi_agent.schemas.task import ToolCatalog, ToolDef


@dataclass(frozen=True)
class RoutedExecution:
    executed: bool
    action: dict[str, Any]
    result: str
    reason: str | None = None


class HarnessActionRouter:
    """Route a selected SafePred action into HarnessAudit's mock tool layer."""

    def __init__(self, *, catalog: ToolCatalog) -> None:
        self._tool_defs: dict[str, ToolDef] = {tool.name: tool for tool in catalog.tools}

    def execute_selected_action(
        self,
        *,
        selected_action: dict[str, Any],
        bank: Any,
        workspace: Any,
        action_sink: Any,
        raw_event: dict[str, Any] | None = None,
    ) -> RoutedExecution:
        action_type = selected_action.get("type")
        if action_type == "tool_call":
            return self._execute_tool_call(
                selected_action=selected_action,
                bank=bank,
                workspace=workspace,
                action_sink=action_sink,
                raw_event=raw_event,
            )
        if action_type == "communication":
            return self._execute_communication(
                selected_action=selected_action,
                action_sink=action_sink,
                raw_event=raw_event,
            )
        return RoutedExecution(
            executed=False,
            action=selected_action,
            result="",
            reason=f"Unsupported selected action type: {action_type}",
        )

    def _execute_tool_call(
        self,
        *,
        selected_action: dict[str, Any],
        bank: Any,
        workspace: Any,
        action_sink: Any,
        raw_event: dict[str, Any] | None,
    ) -> RoutedExecution:
        from multi_agent.frameworks.core.tool_dispatch import dispatch_tool

        tool_name = str(selected_action.get("tool_name") or "")
        tool_def = self._tool_defs.get(tool_name)
        if tool_def is None:
            return RoutedExecution(
                executed=False,
                action=selected_action,
                result="",
                reason=f"Selected tool is not in the HarnessAudit catalog: {tool_name}",
            )

        agent_role = str(selected_action.get("agent_role") or "")
        tool_args = selected_action.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            return RoutedExecution(
                executed=False,
                action=selected_action,
                result="",
                reason="Selected tool_args is not a JSON object.",
            )

        try:
            result = dispatch_tool(
                tool_name,
                tool_args,
                bank=bank,
                tool_def=tool_def,
                workspace=workspace,
            )
        except Exception as exc:
            result = f"[ERROR] Tool {tool_name} failed: {exc}"
        result = str(result)

        action_sink.emit_tool_call(
            agent_id=agent_role,
            agent_role=agent_role,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=result,
            raw_event=raw_event or {},
        )
        return RoutedExecution(
            executed=True,
            action=selected_action,
            result=result,
        )

    @staticmethod
    def _execute_communication(
        *,
        selected_action: dict[str, Any],
        action_sink: Any,
        raw_event: dict[str, Any] | None,
    ) -> RoutedExecution:
        agent_role = str(selected_action.get("agent_role") or "")
        target_role = str(selected_action.get("target_role") or "")
        content = str(selected_action.get("content") or "")
        if not agent_role or not target_role or not content:
            return RoutedExecution(
                executed=False,
                action=selected_action,
                result="",
                reason="Selected communication requires agent_role, target_role, and content.",
            )

        action_sink.emit_communication(
            agent_id=agent_role,
            agent_role=agent_role,
            target_agent=target_role,
            target_role=target_role,
            content=content,
            raw_event=raw_event or {},
        )
        return RoutedExecution(
            executed=True,
            action=selected_action,
            result=f"[SAFEPRED_SELECTED_COMMUNICATION_EXECUTED] {content}",
        )
