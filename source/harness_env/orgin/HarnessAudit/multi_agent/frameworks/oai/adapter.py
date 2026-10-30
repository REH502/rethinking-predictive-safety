"""OpenAI Agents SDK adapter for HarnessAudit.

Mode A (in-process): all agents run in a single Python process.  The hub
agent delegates to spoke agents via ``delegate_to_agent`` FunctionTool calls.
When the LLM proposes *parallel* tool calls, the SDK executes them
concurrently (asyncio), giving true wall-clock concurrency for LLM I/O.

Bank is accessed directly via ``bank.dispatch()`` — no MCP server, no
serialisation overhead.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import threading
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from multi_agent.model_request_options import role_request_extra_body
from multi_agent.frameworks.core.sde_policy import SDE_DOMAIN
from multi_agent.safepred_guard.intervention import (
    SafePredTerminalBlock,
    advisory_message,
    enforced_control_contract,
    is_advisory_only,
    is_enforced_terminal_block_outcome,
    is_prediction_only,
    should_return_advisory,
)
from multi_agent.trace.provenance import runtime_versions
from multi_agent.trace.runtime_checkpoint import (
    DurableCheckpointSession,
    RuntimeCheckpointCoordinator,
    RuntimeCheckpointDriftError,
    RuntimeCommitState,
    RuntimeRecoveryState,
    validate_oai_actor_completion_state,
)

from multi_agent.safepred_guard.terminal import SAFETY_ABSTENTION_FINAL_OUTPUT

from ..core.base import MASFrameworkAdapter, RunContext, RunOutcome
from ..core.task_input import extract_input_text

try:
    # Runner validates hooks by nominal type, so structural compatibility is
    # insufficient even when every callback is implemented.
    from agents.lifecycle import RunHooksBase as _SDKRunHooksBase
except ImportError:  # Preserve the adapter's optional-SDK import contract.
    class _SDKRunHooksBase:  # type: ignore[no-redef]
        pass


def _actor_tool_error_message(args: Any) -> str:
    """Give the actor bounded in-conversation guidance for a malformed tool call."""

    tool_name = str(getattr(args, "tool_name", "") or "")
    if tool_name == "final_output":
        return (
            "`final_output` is not a tool. It is only a candidate type nested "
            "inside `consult_safepred`. Do not repeat this tool call. If you are "
            "finished, return the revised user-facing answer now as ordinary "
            "assistant text without calling another tool."
        )
    default_message = str(
        getattr(args, "default_message", "The requested tool is unavailable.")
    )
    return (
        f"{default_message} Do not repeat the unavailable tool call. Use only an "
        "advertised tool, or answer directly as ordinary assistant text."
    )


def _required_actor_output(value: Any, *, error_message: str) -> str:
    """Normalize an actor response and reject empty/whitespace-only output."""

    output = str(value) if value is not None else ""
    if not output.strip():
        raise RuntimeError(error_message)
    return output


def _actor_reasoning_text(response: Any) -> str:
    """Return provider-exposed reasoning items from one SDK model response."""

    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if str(getattr(item, "type", "") or "") != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = str(getattr(summary, "text", "") or "")
            if text:
                chunks.append(text)
        for content in getattr(item, "content", None) or []:
            text = str(getattr(content, "text", "") or "")
            if text:
                chunks.append(text)
    return "\n".join(chunks)


class _ActorReasoningLedgerHook(_SDKRunHooksBase):
    """Persist actor CoT as an unscored, sanitized sidecar ledger."""

    schema_version = "harnessaudit-actor-reasoning-ledger-v1"
    checkpoint_schema_version = "harnessaudit-actor-reasoning-checkpoint-v2"

    def __init__(self, *, model: str, hub_role: str = "") -> None:
        self._model = str(model)
        self._hub_role = str(hub_role)
        self._counter = 0
        self._pending_hub_final_output: str | None = None
        self._legacy_sdk_start_turn = 0
        self._counter_lock = threading.Lock()

    @staticmethod
    def _checkpoint_json_copy(value: Any) -> Any:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            return json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise TypeError("actor reasoning checkpoint state must be JSON-safe") from exc

    def export_checkpoint_state(self) -> dict[str, Any]:
        """Return the next-call identity state without sharing live references."""

        with self._counter_lock:
            state = {
                    "schema_version": self.checkpoint_schema_version,
                    "identity": {
                        "model": self._model,
                        "hub_role": self._hub_role,
                    },
                    "model_call_counter": self._counter,
                    "pending_hub_final_output": self._pending_hub_final_output,
                }
            if enforced_control_contract() == "legacy_online_v16":
                state["legacy_sdk_start_turn"] = self._legacy_sdk_start_turn
            return self._checkpoint_json_copy(state)

    def restore_checkpoint_state(self, state: Any) -> None:
        """Restore the call index before this hook records its first model call."""

        with self._counter_lock:
            if (
                self._counter != 0
                or self._pending_hub_final_output is not None
            ):
                raise RuntimeError(
                    "actor reasoning checkpoint restore is allowed only before "
                    "the first model call"
                )
            restored = self._checkpoint_json_copy(state)
            if not isinstance(restored, dict):
                raise TypeError("actor reasoning checkpoint state must be an object")
            expected = {
                "schema_version",
                "identity",
                "model_call_counter",
                "pending_hub_final_output",
            }
            if enforced_control_contract() == "legacy_online_v16":
                expected.add("legacy_sdk_start_turn")
            if set(restored) != expected:
                raise ValueError("actor reasoning checkpoint state has an invalid schema")
            if restored["schema_version"] != self.checkpoint_schema_version:
                raise ValueError("actor reasoning checkpoint schema version mismatch")
            if restored["identity"] != {
                "model": self._model,
                "hub_role": self._hub_role,
            }:
                raise ValueError("actor reasoning checkpoint identity mismatch")
            counter = restored["model_call_counter"]
            if type(counter) is not int:
                raise TypeError("model_call_counter must be an integer")
            if counter < 0:
                raise ValueError("model_call_counter must be nonnegative")
            pending = restored["pending_hub_final_output"]
            if pending is not None and not isinstance(pending, str):
                raise TypeError("pending_hub_final_output must be a string or null")
            start = restored.get("legacy_sdk_start_turn", 0)
            if type(start) is not int or start < 0:
                raise ValueError("Invalid legacy SDK invocation start")
            self._counter = counter
            self._pending_hub_final_output = pending
            self._legacy_sdk_start_turn = start

    def remaining_model_turns(self, maximum: int, committed: int, *, new_invocation: bool = False) -> int:
        """Preserve the historical per-SDK-run limit without resetting on resume."""
        with self._counter_lock:
            if enforced_control_contract() != "legacy_online_v16":
                return max(0, maximum - committed)
            if new_invocation:
                self._legacy_sdk_start_turn = committed
            if committed < self._legacy_sdk_start_turn:
                raise RuntimeCheckpointDriftError("Legacy invocation starts after committed model turns")
            return max(0, maximum - (committed - self._legacy_sdk_start_turn))

    def pending_hub_final_output(self) -> str | None:
        """Return a committed hub final waiting for adapter post-processing."""

        with self._counter_lock:
            return self._pending_hub_final_output

    def clear_pending_hub_final_output(self) -> None:
        """Clear an earlier final before starting a continuation or regeneration."""

        with self._counter_lock:
            self._pending_hub_final_output = None

    def __getattr__(self, name: str):
        if not name.startswith("on_"):
            raise AttributeError(name)

        async def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    async def on_agent_end(
        self,
        context: Any,
        agent: Any,
        output: Any,
    ) -> None:
        """Stage only the hub final before the SDK persists that completed turn."""

        del context
        if str(getattr(agent, "name", "") or "") != self._hub_role:
            return
        if output is not None and not isinstance(output, str):
            raise RuntimeCheckpointDriftError(
                "HarnessAudit hub final output must be a string"
            )
        final_output = "" if output is None else output
        with self._counter_lock:
            self._pending_hub_final_output = final_output

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        ledger_value = str(
            os.environ.get("HARNESS_ACTOR_REASONING_LEDGER_PATH") or ""
        ).strip()
        actor_context = getattr(context, "context", None)
        guard = getattr(actor_context, "safepred_guard", None)
        mark_turn = getattr(guard, "mark_actor_turn", None)
        if not ledger_value and not callable(mark_turn):
            return
        with self._counter_lock:
            self._counter += 1
            model_call_index = self._counter
        if callable(mark_turn):
            mark_turn(str(getattr(agent, "name", "") or actor_context.agent_role), model_call_index)
        if not ledger_value:
            return
        reasoning_content = _actor_reasoning_text(response)
        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        actor_context = getattr(context, "context", None)
        action_sink = getattr(actor_context, "action_sink", None)

        from SafePred.utils.llm_client import (
            request_context_from_env,
            sanitize_request_ledger_value,
        )

        active_api_key = (
            os.environ.get("WM_SAFETY_ACTOR_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )
        row = {
            **request_context_from_env(active_api_key=active_api_key),
            "schema_version": self.schema_version,
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "component": "actor",
            "model": self._model,
            "model_call_index": model_call_index,
            "trace_id": str(getattr(action_sink, "run_id", "") or ""),
            "agent_role": str(getattr(agent, "name", "") or ""),
            "reasoning_content_present": bool(reasoning_content),
            "reasoning_content": reasoning_content,
            "usage": {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "reasoning_tokens": int(
                    getattr(output_details, "reasoning_tokens", 0) or 0
                ),
            },
        }
        sanitized = sanitize_request_ledger_value(
            row,
            active_api_key=active_api_key,
        )
        path = Path(ledger_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(
                    json.dumps(
                        sanitized,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _actor_sampling_controls_from_environment(
) -> tuple[float | None, int | None] | None:
    """Resolve the optional first-class actor sampling controls fail-closed."""

    raw_top_p = os.environ.get("WM_SAFETY_ACTOR_TOP_P")
    raw_top_k = os.environ.get("WM_SAFETY_ACTOR_TOP_K")
    if raw_top_p is None and raw_top_k is None:
        return None
    top_p: float | None = None
    top_k: int | None = None
    if raw_top_p is not None:
        try:
            top_p = float(raw_top_p)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("WM_SAFETY_ACTOR_TOP_P must be in (0, 1]") from exc
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise RuntimeError("WM_SAFETY_ACTOR_TOP_P must be in (0, 1]")
    if raw_top_k is not None:
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "WM_SAFETY_ACTOR_TOP_K must be a nonnegative integer"
            ) from exc
        if top_k < 0 or str(raw_top_k).strip() not in {str(top_k), f"{top_k}.0"}:
            raise RuntimeError("WM_SAFETY_ACTOR_TOP_K must be a nonnegative integer")
    return top_p, top_k


# ---- Shared context threaded through Runner → tools → nested runs ----

@dataclass
class MultiAgentContext:
    """Propagated to every tool via ``ToolContext[MultiAgentContext].context``."""

    action_sink: Any
    """ActionSink instance for trace emission."""

    bank: Any
    """Domain Bank (or None if fixture-only domain)."""

    workspace: Any
    """SDE workspace path (or None)."""

    agent_role: str = ""
    """Current agent's role name — set per-run so tool handlers know the caller."""

    spoke_agents: dict[str, Any] = field(default_factory=dict)
    """role → Agent mapping, used by delegate_to_agent tool."""

    run_config: Any = None
    """OpenAI Agents SDK RunConfig shared with nested specialist runs."""

    run_hooks: Any = None
    """Shared actor-reasoning persistence hook for hub and spoke calls."""

    depth: int = 0
    """Recursion guard: bumped on each nested Runner.run inside delegate_to_agent."""

    safepred_guard: Any = None
    """Optional SafePred online guard shared by hub and spoke tool handlers."""

    hub_role: str = ""
    """Coordinator role; used to validate hub-only advisory candidates."""

    actor_roles: tuple[str, ...] = ()
    """Registered actor roles available in this episode."""

    safepred_consultation_counts: dict[str, int] = field(default_factory=dict)
    """Shared per-caller consultation counters for the current episode."""

    safepred_consultation_limit_per_caller: int = 3
    safepred_consultation_limit_per_episode: int = 12



def _runtime_root_path(value: Any) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise RuntimeCheckpointDriftError("runtime root must not be a symbolic link")
    root = Path(os.path.abspath(os.fspath(raw)))
    if not root.is_dir() or root.resolve() != root:
        raise RuntimeCheckpointDriftError("runtime root is unavailable or unsafe")
    return root


def _runtime_resolve_path(value: Any, root: Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else root / raw
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeCheckpointDriftError(f"{label} is outside runtime root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeCheckpointDriftError(f"{label} traverses a symbolic link")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise RuntimeCheckpointDriftError(f"{label} escapes runtime root") from exc
    return path


def _secure_regular_file(path: Path, root: Path, *, label: str) -> Path:
    """Create or validate a private, single-link regular runtime file."""

    path = _runtime_resolve_path(path, root, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _runtime_resolve_path(path, root, label=label)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeCheckpointDriftError(
                f"{label} must be a single-link regular file"
            )
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _secure_session_database(path: Path, root: Path) -> Path:
    return _secure_regular_file(path, root, label="session database")


def _runtime_workspace_path(workspace: Any, root: Path) -> Path | None:
    if workspace is None:
        return None
    raw = getattr(workspace, "path", workspace)
    if not isinstance(raw, (str, os.PathLike)):
        raise RuntimeCheckpointDriftError("workspace path is unavailable")
    path = _runtime_resolve_path(raw, root, label="workspace")
    if path.is_symlink() or not path.is_dir():
        raise RuntimeCheckpointDriftError("workspace is unavailable or unsafe")
    return path


def _runtime_read_bytes(path: Path, root: Path, *, label: str) -> bytes:
    path = _runtime_resolve_path(path, root, label=label)
    if path.is_symlink() or not path.is_file():
        raise RuntimeCheckpointDriftError(f"{label} is unavailable or unsafe")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != before.st_size
    ):
        raise RuntimeCheckpointDriftError(f"{label} changed while being read")
    return data


def _runtime_ledger_paths(root: Path, safepred_guard: Any) -> dict[str, Path]:
    """Freeze all known append-only ledger paths before the first model call."""

    candidates: list[tuple[str, Path]] = []
    for name, env_name in (
        ("request", "SAFEPRED_REQUEST_LEDGER_PATH"),
        ("actor_reasoning", "HARNESS_ACTOR_REASONING_LEDGER_PATH"),
        ("judge", "HARNESS_JUDGE_LEDGER_PATH"),
    ):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            candidates.append((name, Path(raw)))
    if safepred_guard is not None:
        for name, attribute in (
            ("guard_decisions", "decision_path"),
            ("consultation_attempts", "consultation_attempt_path"),
            ("safety_abstention_terminal", "safety_abstention_path"),
        ):
            raw = getattr(safepred_guard, attribute, None)
            if not isinstance(raw, (str, os.PathLike)):
                raise RuntimeCheckpointDriftError(
                    f"guard ledger {name} path is unavailable"
                )
            candidates.append((name, Path(raw)))

    ledgers: dict[str, Path] = {}
    seen: set[Path] = set()
    for name, raw_path in candidates:
        path = _secure_regular_file(raw_path, root, label=f"ledger {name}")
        resolved = path.resolve(strict=True)
        if resolved in seen:
            raise RuntimeCheckpointDriftError(
                "two runtime ledgers resolve to the same file"
            )
        seen.add(resolved)
        ledgers[name] = path
    return ledgers



def _validate_runtime_ledger_bindings(
    paths: dict[str, Path],
    recovery: RuntimeRecoveryState,
    root: Path,
) -> None:
    if set(paths) != set(recovery.ledger_offsets):
        raise RuntimeCheckpointDriftError("runtime ledger set changed across recovery")
    for name, path in paths.items():
        relative = path.relative_to(root).as_posix()
        if recovery.ledger_offsets[name].relative_path != relative:
            raise RuntimeCheckpointDriftError("runtime ledger path changed across recovery")


def _validate_runtime_recovery(
    recovery: RuntimeRecoveryState,
    coordinator: RuntimeCheckpointCoordinator,
) -> None:
    observed = coordinator.inspect()
    if replace(recovery, recovery_orphans=()) != observed:
        raise RuntimeCheckpointDriftError(
            "runtime recovery state differs from the durable checkpoint"
        )


def _restore_adapter_runtime_state(
    recovery: RuntimeRecoveryState,
    *,
    actor_reasoning_hook: Any,
    safepred_guard: Any,
    actor_roles: tuple[str, ...],
) -> dict[str, int]:
    committed = recovery.model_turn_count > 0 or recovery.actor_complete
    hook_state = recovery.hook_state
    if hook_state:
        if set(hook_state) != {"actor_reasoning"}:
            raise RuntimeCheckpointDriftError("actor hook checkpoint schema drift")
        actor_reasoning_hook.restore_checkpoint_state(hook_state["actor_reasoning"])
    elif committed:
        raise RuntimeCheckpointDriftError("committed actor hook state is missing")
    if safepred_guard is not None:
        if committed and not recovery.guard_state:
            raise RuntimeCheckpointDriftError("committed guard state is missing")
        if recovery.guard_state:
            safepred_guard.restore_checkpoint_state(recovery.guard_state)
    elif recovery.guard_state:
        raise RuntimeCheckpointDriftError("guard checkpoint exists without a guard")
    if committed and set(recovery.counts_state) != {
        "safepred_consultation_counts"
    }:
        raise RuntimeCheckpointDriftError("consultation count checkpoint schema drift")
    if not committed and recovery.counts_state:
        raise RuntimeCheckpointDriftError("empty checkpoint contains consultation counts")
    raw_counts = recovery.counts_state.get("safepred_consultation_counts", {})
    if not isinstance(raw_counts, dict):
        raise RuntimeCheckpointDriftError("consultation count checkpoint is malformed")
    allowed_roles = set(actor_roles)
    counts: dict[str, int] = {}
    for role, value in raw_counts.items():
        if (
            not isinstance(role, str)
            or role not in allowed_roles
            or type(value) is not int
            or value < 0
        ):
            raise RuntimeCheckpointDriftError(
                "consultation count checkpoint is malformed"
            )
        counts[role] = value
    return counts


def _runtime_error_text(exc: BaseException) -> str:
    from SafePred.utils.llm_client import sanitize_request_ledger_value

    active_api_key = (
        os.environ.get("WM_SAFETY_ACTOR_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or None
    )
    sanitized = sanitize_request_ledger_value(
        {"type": type(exc).__name__, "message": str(exc)},
        active_api_key=active_api_key,
    )
    error_type = str(sanitized.get("type") or "ActorError")[:200]
    message = str(sanitized.get("message") or "")[:2000]
    return f"{error_type}: {message}" if message else error_type


def _outcome_from_actor_checkpoint(recovery: RuntimeRecoveryState) -> RunOutcome | None:
    state = recovery.actor_complete_state
    if state is None:
        return None
    state = validate_oai_actor_completion_state(state)
    if state["schema_version"] == 1:
        return RunOutcome(
            framework="oai",
            final_output=state["final_output"],
            error=state["error"],
        )
    return RunOutcome(
        framework="oai",
        final_output=state["final_output"],
        error=None,
        terminal_disposition=state["terminal_disposition"],
        task_success_status=state["task_success_status"],
        safety_abstention=state["safety_abstention"],
        safety_abstention_artifact=state["safety_abstention_artifact"],
    )


def _unbound_safety_abstention_artifact(path: Path) -> dict[str, Any]:
    """Build a diagnostic-only artifact binding outside trace-resume roots."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Safety-abstention sidecar is unavailable or unsafe")
    with path.open("rb") as handle:
        data = handle.read()
    return {
        "relative_path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


# ---- Adapter ----

class OAIAdapter(MASFrameworkAdapter):
    """Run multi-agent tasks using the OpenAI Agents SDK (in-process)."""

    framework_id: ClassVar[str] = "oai"

    def __init__(self, agent_timeout: int = 300, **_kwargs: Any) -> None:
        self._agent_timeout = agent_timeout

    # ---- MASFrameworkAdapter.run ----

    async def run(
        self,
        ctx: RunContext,
        initial_input: str | list[dict[str, Any]],
    ) -> RunOutcome:
        try:
            from agents import Agent, ModelSettings, Runner, RunConfig
            from agents.tracing import set_tracing_disabled
            from openai import AsyncOpenAI
            try:
                from agents import OpenAIProvider
            except ImportError:
                from agents.models.openai_provider import OpenAIProvider
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Agents SDK not installed.  pip install openai-agents"
            ) from exc

        from .prompts import (
            HARNESSAUDIT_ACTOR_CONSULTATION_PROMPT_VERSION,
            build_oai_hub_prompt,
            build_oai_spoke_prompt,
        )
        from .tools import (
            build_consult_safepred_tool,
            build_delegate_tool,
            build_domain_tools,
        )

        hub_role = ctx.task.agents[0].role
        spoke_cfgs = ctx.task.agents[1:]

        # Disable SDK tracing — ActionSink hooks are the primary trace source.
        set_tracing_disabled(True)

        runtime_coordinator: RuntimeCheckpointCoordinator | None = None
        runtime_session: DurableCheckpointSession | None = None
        runtime_state_factory = None
        runtime_underlying_session = None
        runtime_recovery = ctx.runtime_recovery
        runtime_root: Path | None = None
        runtime_workspace: Path | None = None
        active_ledger_paths: dict[str, Path] = {}
        base_ledger_paths: dict[str, Path] = {}

        try:
            runtime_fields = (
                ctx.runtime_checkpoint,
                runtime_recovery,
                ctx.runtime_root,
                ctx.session_path,
                ctx.trace_path,
            )
            runtime_requested = any(value is not None for value in runtime_fields)
            if runtime_requested and any(value is None for value in runtime_fields):
                raise RuntimeCheckpointDriftError(
                    "trace resume requires checkpoint, recovery, root, session, and trace"
                )
            if runtime_requested:
                if not isinstance(ctx.runtime_checkpoint, RuntimeCheckpointCoordinator):
                    raise RuntimeCheckpointDriftError(
                        "runtime checkpoint has the wrong type"
                    )
                if not isinstance(runtime_recovery, RuntimeRecoveryState):
                    raise RuntimeCheckpointDriftError(
                        "runtime recovery state has the wrong type"
                    )
                runtime_root = _runtime_root_path(ctx.runtime_root)
                runtime_coordinator = ctx.runtime_checkpoint
                if runtime_coordinator.root != runtime_root:
                    raise RuntimeCheckpointDriftError("runtime root identity mismatch")
                if runtime_coordinator.checkpoint.run_id != ctx.run_id:
                    raise RuntimeCheckpointDriftError("runtime run_id mismatch")
                _validate_runtime_recovery(runtime_recovery, runtime_coordinator)
                recovered_outcome = _outcome_from_actor_checkpoint(runtime_recovery)
                if recovered_outcome is not None:
                    return recovered_outcome
                base_ledger_paths = _runtime_ledger_paths(runtime_root, None)
                if runtime_recovery.model_turn_count == 0 and not runtime_recovery.actor_complete:
                    # Clear only stale pre-initialization bytes here. Guard
                    # construction may legitimately append policy-extraction
                    # evidence to the same request ledger.
                    runtime_coordinator.prepare_initial_ledgers(base_ledger_paths)

            # 1. Build domain tools (shared across all agents)
            all_domain_tools = build_domain_tools(ctx.catalog, ctx.workspace)

            # 2. Resolve model string
            model = self._resolve_model(ctx.model)
            actor_provenance = ctx.actor_provenance or {}
            self._validate_runtime_provenance(actor_provenance)
            run_config = self._build_run_config(
                model,
                RunConfig=RunConfig,
                OpenAIProvider=OpenAIProvider,
                AsyncOpenAI=AsyncOpenAI,
                base_url=actor_provenance.get("endpoint"),
            )
            actor_reasoning_hook = _ActorReasoningLedgerHook(
                model=model,
                hub_role=hub_role,
            )
            safepred_guard = None
            if ctx.safepred_guard_config:
                from multi_agent.safepred_guard import SafePredHarnessGuard

                safepred_guard = SafePredHarnessGuard.from_run_context(ctx)
            restored_consultation_counts: dict[str, int] = {}
            if runtime_recovery is not None:
                if not isinstance(runtime_recovery, RuntimeRecoveryState):
                    raise RuntimeCheckpointDriftError(
                        "runtime recovery state has the wrong type"
                    )
                restored_consultation_counts = _restore_adapter_runtime_state(
                    runtime_recovery,
                    actor_reasoning_hook=actor_reasoning_hook,
                    safepred_guard=safepred_guard,
                    actor_roles=tuple(agent.role for agent in ctx.task.agents),
                )
            advisory_active = bool(
                safepred_guard is not None and is_advisory_only(safepred_guard)
            )
            consultation_limit_per_caller = self._consultation_limit(
                ctx.safepred_guard_config,
                config_key="max_consultations_per_caller",
                env_key="HARNESS_SAFEPRED_MAX_CONSULTATIONS_PER_CALLER",
                default=3,
            )
            consultation_limit_per_episode = self._consultation_limit(
                ctx.safepred_guard_config,
                config_key="max_consultations_per_episode",
                env_key="HARNESS_SAFEPRED_MAX_CONSULTATIONS_PER_EPISODE",
                default=12,
            )
            if any(
                count > consultation_limit_per_caller
                for count in restored_consultation_counts.values()
            ) or sum(restored_consultation_counts.values()) > (
                consultation_limit_per_episode
            ):
                raise RuntimeCheckpointDriftError(
                    "restored consultation counts exceed the configured budget"
                )
            consultation_tool = None
            if advisory_active:
                consultation_tool = build_consult_safepred_tool(
                    catalog=ctx.catalog,
                    actor_roles=tuple(agent.role for agent in ctx.task.agents),
                    hub_role=hub_role,
                )
            spoke_tool_sets, hub_domain_tools = self._actor_tool_sets(
                all_domain_tools,
                spoke_cfgs,
                consultation_tool=consultation_tool,
            )
            computed_parallel_proposals = bool(
                safepred_guard is None or is_prediction_only(safepred_guard)
            ) and os.environ.get("HARNESS_DISABLE_PARALLEL_TOOL_CALLS", "0") != "1"
            allow_parallel_proposals = computed_parallel_proposals and bool(
                actor_provenance.get(
                    "actor_parallel_tool_calls", computed_parallel_proposals
                )
            )
            settings = self._build_model_settings(
                model,
                allow_parallel_proposals=allow_parallel_proposals,
                ModelSettings=ModelSettings,
                actor_seed=actor_provenance.get("actor_trace_seed"),
                actor_temperature=actor_provenance.get("actor_temperature"),
                require_seed=actor_provenance.get("actor_seed_required"),
            )

            # 3. Create spoke agents (all domain tools, NO delegate tool)
            spoke_agents: dict[str, Agent[MultiAgentContext]] = {}
            for agent_cfg in spoke_cfgs:
                spoke = Agent[MultiAgentContext](
                    name=agent_cfg.role,
                    instructions=build_oai_spoke_prompt(
                        agent_cfg,
                        hub_role,
                        safepred_advisory=advisory_active,
                    ),
                    tools=spoke_tool_sets[agent_cfg.role],
                    model=model,
                    model_settings=settings,
                )
                spoke_agents[agent_cfg.role] = spoke

            # 4. Create hub agent (all domain tools plus delegate_to_agent)
            delegate_tool = build_delegate_tool(
                catalog=ctx.catalog,
                max_spoke_turns=ctx.max_turns,
            )
            hub = Agent[MultiAgentContext](
                name=hub_role,
                instructions=build_oai_hub_prompt(
                    ctx.task.agents[0],
                    spoke_cfgs,
                    ctx.task.goal,
                    safepred_advisory=advisory_active,
                ),
                tools=[*hub_domain_tools, delegate_tool],
                model=model,
                model_settings=settings,
            )
            if (safepred_guard is not None and
                getattr(safepred_guard.settings, "wm_return_perturbation", "") == "environment_h1"):
                from multi_agent.safepred_guard.environment_h1 import PROMPT
                for actor in [hub, *spoke_agents.values()]:
                    actor.instructions = str(actor.instructions) + "\n\n" + PROMPT
            if advisory_active:
                installed_actors = [
                    (hub_role, hub),
                    *[
                        (role, spoke_agents[role])
                        for role in spoke_agents
                    ],
                ]
                for role, actor in installed_actors:
                    if not any(
                        getattr(tool, "name", None) == "consult_safepred"
                        for tool in actor.tools
                    ):
                        raise RuntimeError(
                            f"advisory consultation tool was not installed for actor {role!r}"
                        )
                    if "consult_safepred" not in str(actor.instructions):
                        raise RuntimeError(
                            f"advisory consultation prompt was not installed for actor {role!r}"
                        )
                safepred_guard.mark_consultation_available(
                    tool_or_command_name="consult_safepred",
                    candidate_schema_version="harnessaudit-consultation-candidate-v1",
                    prompt_version=(
                        HARNESSAUDIT_ACTOR_CONSULTATION_PROMPT_VERSION
                    ),
                    parser_version="harnessaudit-consultation-parser-v1",
                    decision_ledger_schema_version=(
                        "safepred-consultation-decision-ledger-v1"
                    ),
                    per_caller=consultation_limit_per_caller,
                    per_episode=consultation_limit_per_episode,
                    actor_roles=[role for role, _actor in installed_actors],
                )

            # 5. Build context
            multi_agent_ctx = MultiAgentContext(
                action_sink=ctx.action_sink,
                bank=ctx.bank,
                workspace=ctx.workspace,
                agent_role=hub_role,
                spoke_agents=spoke_agents,
                run_config=run_config,
                run_hooks=actor_reasoning_hook,
                safepred_guard=safepred_guard,
                hub_role=hub_role,
                actor_roles=tuple(agent.role for agent in ctx.task.agents),
                safepred_consultation_counts=restored_consultation_counts,
                safepred_consultation_limit_per_caller=(
                    consultation_limit_per_caller
                ),
                safepred_consultation_limit_per_episode=(
                    consultation_limit_per_episode
                ),
            )

            # 6. Run (with timeout guard)
            import asyncio

            if runtime_coordinator is not None:
                assert runtime_root is not None
                assert isinstance(runtime_recovery, RuntimeRecoveryState)
                runtime_workspace = _runtime_workspace_path(ctx.workspace, runtime_root)
                active_ledger_paths = _runtime_ledger_paths(
                    runtime_root,
                    safepred_guard,
                )
                if runtime_recovery.model_turn_count == 0 and not runtime_recovery.actor_complete:
                    guard_ledger_paths = {
                        name: path
                        for name, path in active_ledger_paths.items()
                        if name not in base_ledger_paths
                    }
                    runtime_coordinator.prepare_initial_ledgers(guard_ledger_paths)
                else:
                    _validate_runtime_ledger_bindings(
                        active_ledger_paths,
                        runtime_recovery,
                        runtime_root,
                    )
                session_path = _secure_session_database(
                    _runtime_resolve_path(
                        ctx.session_path,
                        runtime_root,
                        label="session database",
                    ),
                    runtime_root,
                )
                from agents.memory.sqlite_session import SQLiteSession

                runtime_underlying_session = SQLiteSession(
                    ctx.run_id,
                    db_path=session_path,
                )
                _secure_session_database(session_path, runtime_root)

                def build_runtime_state() -> RuntimeCommitState:
                    actions = ctx.action_sink.serialized_actions()
                    trace_path = _runtime_resolve_path(
                        ctx.trace_path,
                        runtime_root,
                        label="trace path",
                    )
                    hook_state = {
                        "actor_reasoning": actor_reasoning_hook.export_checkpoint_state()
                    }
                    guard_state = (
                        safepred_guard.export_checkpoint_state()
                        if safepred_guard is not None
                        else {}
                    )
                    return RuntimeCommitState(
                        normalized_actions=actions,
                        guard_state=guard_state,
                        hook_state=hook_state,
                        counts_state={
                            "safepred_consultation_counts": dict(
                                multi_agent_ctx.safepred_consultation_counts
                            )
                        },
                        usage={
                            "actor_model_calls": hook_state[
                                "actor_reasoning"
                            ]["model_call_counter"]
                        },
                        ledger_paths=active_ledger_paths,
                        trace_bytes=_runtime_read_bytes(
                            trace_path,
                            runtime_root,
                            label="trace path",
                        ),
                        workspace=runtime_workspace,
                    )

                runtime_state_factory = build_runtime_state
                bankless_sde = (
                    ctx.bank is None
                    and ctx.task.domain == SDE_DOMAIN
                    and ctx.workspace is not None
                )
                if ctx.bank is None and not bankless_sde:
                    raise RuntimeCheckpointDriftError(
                        "bankless runtime is restricted to SDE with a workspace"
                    )
                runtime_session = DurableCheckpointSession(
                    runtime_underlying_session,
                    runtime_coordinator,
                    bank=ctx.bank,
                    state_factory=build_runtime_state,
                    allow_bankless=bankless_sde,
                )
                await runtime_session.prepare(
                    bank=ctx.bank,
                    trace_path=ctx.trace_path,
                    workspace=runtime_workspace,
                    recovery_state=runtime_recovery,
                )

            recovered_pending_output = (
                actor_reasoning_hook.pending_hub_final_output()
                if runtime_session is not None
                else None
            )
            if recovered_pending_output is None:
                committed_turns = (
                    runtime_session.committed_model_turns
                    if runtime_session is not None
                    else 0
                )
                remaining_turns = actor_reasoning_hook.remaining_model_turns(ctx.max_turns, committed_turns)
                if remaining_turns == 0:
                    raise RuntimeError(
                        "MaxTurnsExceeded: no remaining model turns after recovery"
                    )
                runner_input = (
                    []
                    if runtime_session is not None and committed_turns > 0
                    else extract_input_text(initial_input) or ctx.task.goal
                )
                actor_reasoning_hook.clear_pending_hub_final_output()
                result = await asyncio.wait_for(
                    Runner.run(
                        hub,
                        runner_input,
                        context=multi_agent_ctx,
                        max_turns=remaining_turns,
                        run_config=run_config,
                        hooks=actor_reasoning_hook,
                        **(
                            {"session": runtime_session}
                            if runtime_session is not None
                            else {}
                        ),
                    ),
                    timeout=self._agent_timeout,
                )
                final_output = _required_actor_output(
                    result.final_output,
                    error_message="HarnessAudit hub produced no final user response",
                )
            else:
                # SDK on_agent_end ran before the completed turn was committed.
                # Resume adapter/guard post-processing without another actor call,
                # including when the cumulative model-turn budget is exhausted.
                final_output = _required_actor_output(
                    recovered_pending_output,
                    error_message="Recovered HarnessAudit hub response is empty",
                )
            if advisory_active:
                safepred_guard.close_pending_consultation(
                    agent_role=hub_role,
                    next_action={
                        "type": "communication",
                        "agent_role": hub_role,
                        "target_role": "user",
                        "content": final_output,
                    },
                )
            final_guard_decision = None
            while (
                safepred_guard
                and not is_advisory_only(safepred_guard)
                and final_output
            ):
                from .tools import _evaluate_guard_call
                final_guard_decision = await _evaluate_guard_call(
                    safepred_guard.evaluate_communication,
                    agent_role=hub_role,
                    target_role="user",
                    content=final_output,
                )
                if is_enforced_terminal_block_outcome(final_guard_decision):
                    raise SafePredTerminalBlock(
                        safepred_guard.build_safety_abstention_terminal(
                            final_guard_decision,
                            agent_role=hub_role,
                            surface="final_output",
                        )
                    )
                if getattr(final_guard_decision, "requires_regeneration", False):
                    feedback = {
                        "role": "user",
                        "content": _actor_feedback_message(
                            final_guard_decision,
                            reason="Revise the final user response before it is released.",
                        ),
                    }
                    continuation = (
                        [feedback]
                        if runtime_session is not None
                        else [*result.to_input_list(mode="normalized"), feedback]
                    )
                    committed_turns = (
                        runtime_session.committed_model_turns
                        if runtime_session is not None
                        else 0
                    )
                    remaining_turns = actor_reasoning_hook.remaining_model_turns(
                        ctx.max_turns, committed_turns, new_invocation=True)
                    if remaining_turns == 0:
                        raise RuntimeError(
                            "MaxTurnsExceeded: no remaining turns for final regeneration"
                        )
                    actor_reasoning_hook.clear_pending_hub_final_output()
                    result = await asyncio.wait_for(
                        Runner.run(
                            hub,
                            continuation,
                            context=multi_agent_ctx,
                            max_turns=remaining_turns,
                            run_config=run_config,
                            hooks=actor_reasoning_hook,
                            **(
                                {"session": runtime_session}
                                if runtime_session is not None
                                else {}
                            ),
                        ),
                        timeout=self._agent_timeout,
                    )
                    final_output = _required_actor_output(
                        result.final_output,
                        error_message=(
                            "HarnessAudit hub produced no revised response "
                            "after SafePred feedback"
                        ),
                    )
                    continue
                final_output = _apply_final_output_guard(
                    original_output=final_output,
                    decision=final_guard_decision,
                    safepred_guard=safepred_guard,
                )
                break
            final_output = _required_actor_output(
                final_output,
                error_message="HarnessAudit hub produced no releasable final response",
            )

        except SafePredTerminalBlock as exc:
            guard_for_cleanup = locals().get("safepred_guard")
            if guard_for_cleanup is None:
                raise RuntimeError("SafePred terminal escaped without a guard") from exc
            terminal = exc.terminal
            close_all = getattr(
                guard_for_cleanup,
                "close_all_pending_consultations",
                None,
            )
            if callable(close_all):
                # Lifecycle closeouts append to the guard decision ledger, so
                # complete them before the immutable terminal binds its hash.
                close_all(status="no_subsequent_action")
            sidecar_path = guard_for_cleanup.persist_safety_abstention_terminal(
                terminal
            )
            committed_actions = ctx.action_sink.serialized_actions()
            if len(committed_actions) != terminal["trace_action_cutoff"]:
                raise RuntimeCheckpointDriftError(
                    "safety abstention action cutoff does not match the trace"
                )
            ctx.action_sink.emit_safety_abstention(terminal)
            if runtime_coordinator is not None and runtime_root is not None:
                artifact_binding = runtime_coordinator.bind_existing_artifact(
                    sidecar_path,
                    label="safety abstention artifact",
                ).as_json_value()
            else:
                artifact_binding = _unbound_safety_abstention_artifact(sidecar_path)
            if runtime_coordinator is not None and runtime_state_factory is not None:
                runtime_coordinator.checkpoint_actor_complete(
                    bank=ctx.bank,
                    state=runtime_state_factory(),
                    actor_state={
                        "schema_version": 2,
                        "framework": "oai",
                        "final_output": SAFETY_ABSTENTION_FINAL_OUTPUT,
                        "error": None,
                        "terminal_disposition": terminal["terminal_disposition"],
                        "task_success_status": terminal["task_success_status"],
                        "safety_abstention": terminal,
                        "safety_abstention_artifact": artifact_binding,
                    },
                )
            if runtime_underlying_session is not None:
                try:
                    runtime_underlying_session.close()
                except Exception:
                    pass
            return RunOutcome(
                framework=self.framework_id,
                final_output=SAFETY_ABSTENTION_FINAL_OUTPUT,
                error=None,
                terminal_disposition=terminal["terminal_disposition"],
                task_success_status=terminal["task_success_status"],
                safety_abstention=terminal,
                safety_abstention_artifact=artifact_binding,
            )
        except Exception as exc:
            import traceback

            guard_for_cleanup = locals().get("safepred_guard")
            if guard_for_cleanup is not None:
                close_all = getattr(
                    guard_for_cleanup,
                    "close_all_pending_consultations",
                    None,
                )
                if callable(close_all):
                    close_all(status="no_subsequent_action")
            error_text = _runtime_error_text(exc)
            traceback.print_tb(exc.__traceback__)
            print(error_text, file=sys.stderr)
            if runtime_underlying_session is not None:
                try:
                    runtime_underlying_session.close()
                except Exception:
                    pass
            if runtime_coordinator is not None:
                # The failing SDK turn is intentionally not committed. A later
                # invocation restores the last complete turn; the matrix layer
                # records this task failure and advances to the next condition.
                raise RuntimeError(error_text) from None
            return RunOutcome(
                framework=self.framework_id,
                final_output="",
                error=error_text,
            )

        # 7. Emit final user_output
        ctx.action_sink.emit_communication(
            agent_id=hub_role,
            agent_role=hub_role,
            target_agent="user",
            target_role="user",
            content=final_output,
            raw_event={
                "framework": "oai",
                "communication_kind": "user_output",
                **self._guard_raw_event(final_guard_decision if 'final_guard_decision' in locals() else None),
            },
        )
        if safepred_guard is not None:
            safepred_guard.close_all_pending_consultations(
                status="no_subsequent_action"
            )
        if runtime_coordinator is not None and runtime_state_factory is not None:
            runtime_coordinator.checkpoint_actor_complete(
                bank=ctx.bank,
                state=runtime_state_factory(),
                actor_state={
                    "schema_version": 1,
                    "framework": "oai",
                    "final_output": final_output,
                    "error": None,
                },
            )
        if runtime_underlying_session is not None:
            runtime_underlying_session.close()
        return RunOutcome(framework=self.framework_id, final_output=final_output)

    # ---- Helpers ----

    @staticmethod
    def _actor_tool_sets(
        all_domain_tools,
        spoke_cfgs,
        *,
        consultation_tool=None,
    ):
        actor_tools = [*all_domain_tools]
        if consultation_tool is not None:
            actor_tools.append(consultation_tool)
        spoke_tools = {
            agent_cfg.role: list(actor_tools)
            for agent_cfg in spoke_cfgs
        }
        return spoke_tools, list(actor_tools)

    @staticmethod
    def _consultation_limit(
        raw_config: dict[str, Any] | None,
        *,
        config_key: str,
        env_key: str,
        default: int,
    ) -> int:
        configured = (raw_config or {}).get(config_key)
        value = configured if configured is not None else os.environ.get(env_key, default)
        return max(0, int(value))

    @staticmethod
    def _validate_runtime_provenance(
        actor_provenance: dict[str, Any],
        *,
        observed_versions: dict[str, Any] | None = None,
    ) -> None:
        """Ensure the adapter executes in the runtime recorded pre-call."""

        recorded = actor_provenance.get("runtime_versions")
        if recorded is None:
            return
        observed = observed_versions or runtime_versions()
        if recorded != observed:
            raise RuntimeError(
                "Actor runtime versions changed between trace_start and the "
                "OpenAI adapter call"
            )

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Map multi_agent model identifier to an OpenAI SDK model string.

        ``MultiProvider`` splits on the first ``/``:
        - No prefix → OpenAI directly (``"gpt-4.1"``)
        - ``litellm/`` → ``LitellmProvider`` (``"litellm/anthropic/claude-sonnet-4"``)
        """
        if any(k in model for k in ("claude", "sonnet", "opus", "haiku")):
            return f"litellm/anthropic/{model}"
        if "gemini" in model:
            # A model-stack profile with an explicit endpoint is already an
            # OpenAI-compatible route (official API, relay, or local server).
            # Keep the plain model id so _build_run_config can bind that
            # endpoint.  Only legacy, endpoint-less Gemini invocations use the
            # optional LiteLLM provider.
            if os.environ.get("WM_SAFETY_ACTOR_BASE_URL") or os.environ.get(
                "OPENAI_BASE_URL"
            ):
                return model
            return f"litellm/gemini/{model}"
        return model

    @staticmethod
    def _build_run_config(
        model: str,
        *,
        RunConfig: Any,
        OpenAIProvider: Any,
        AsyncOpenAI: Any,
        base_url: str | None = None,
    ) -> Any:
        """Create a RunConfig that respects OpenAI-compatible relay settings."""
        runtime_controls = {
            # Unknown tool calls are model-output protocol errors, not transport
            # failures. Return a deterministic error in the same conversation so
            # the actor can repair without replaying already-executed domain tools.
            # Runner.max_turns remains the hard bound on repeated malformed calls.
            "tool_not_found_behavior": "return_error_to_model",
            "tool_error_formatter": _actor_tool_error_message,
        }
        if model.startswith("litellm/"):
            return RunConfig(model=model, **runtime_controls)

        api_key = (
            os.environ.get("WM_SAFETY_ACTOR_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )
        base_url = (
            os.environ.get("WM_SAFETY_ACTOR_BASE_URL")
            or base_url
            or os.environ.get("OPENAI_BASE_URL")
            or None
        )
        if api_key or base_url:
            timeout_raw = os.environ.get("WM_SAFETY_ACTOR_TIMEOUT_SECONDS")
            transport_attempts_raw = os.environ.get(
                "WM_SAFETY_ACTOR_TRANSPORT_MAX_ATTEMPTS"
            )
            transport_attempts = (
                max(1, int(transport_attempts_raw))
                if transport_attempts_raw is not None
                and transport_attempts_raw.strip()
                else 3
            )
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                # OpenAI names this retry count, not total attempts.
                max_retries=transport_attempts - 1,
                **(
                    {"timeout": float(timeout_raw)}
                    if timeout_raw is not None and timeout_raw.strip()
                    else {}
                ),
            )
            provider = OpenAIProvider(
                openai_client=client,
                use_responses=(
                    os.environ.get("WM_SAFETY_ACTOR_API_MODE", "chat_completions")
                    == "responses"
                ),
            )
            return RunConfig(
                model=model,
                model_provider=provider,
                **runtime_controls,
            )
        return RunConfig(model=model, **runtime_controls)

    @staticmethod
    def _build_model_settings(
        model: str,
        *,
        allow_parallel_proposals: bool,
        ModelSettings: Any,
        actor_seed: int | None = None,
        actor_temperature: float | None = None,
        require_seed: bool | None = None,
    ) -> Any:
        settings: dict[str, Any] = {
            "parallel_tool_calls": allow_parallel_proposals,
        }
        max_tokens = os.environ.get("WM_SAFETY_ACTOR_MAX_TOKENS")
        if max_tokens is not None and max_tokens.strip():
            settings["max_tokens"] = int(max_tokens)
        model_lower = str(model).lower()
        if not any(name in model_lower for name in ("gpt-5", "o3", "o4")):
            settings["temperature"] = float(
                actor_temperature
                if actor_temperature is not None
                else os.environ.get("HARNESS_ACTOR_TEMPERATURE", "0")
            )
        sampling_controls = _actor_sampling_controls_from_environment()
        if sampling_controls is not None and sampling_controls[0] is not None:
            settings["top_p"] = sampling_controls[0]
        seed: int | str | None = actor_seed
        if seed is None:
            seed = os.environ.get("HARNESS_ACTOR_SEED")
        extra_body = role_request_extra_body("actor", model=model)
        if {"top_p", "top_k"}.intersection(extra_body):
            raise RuntimeError(
                "Actor sampling controls must use the model-stack top_p/top_k "
                "fields, not request_extra_body"
            )
        configured_extra_body_required = bool(extra_body) and (
            "WM_SAFETY_ACTOR_REQUEST_EXTRA_BODY_JSON" in os.environ
        )
        configured_sampling_required = sampling_controls is not None
        if seed is not None and str(seed).strip():
            extra_body["seed"] = int(seed)
        if sampling_controls is not None and sampling_controls[1] is not None:
            extra_body["top_k"] = sampling_controls[1]
        if extra_body:
            settings["extra_body"] = extra_body
        try:
            return ModelSettings(**settings)
        except TypeError as exc:
            seed_is_required = (
                bool(require_seed)
                if require_seed is not None
                else os.environ.get("HARNESS_ACTOR_REQUIRE_SEED") == "1"
            )
            if seed is not None and seed_is_required:
                raise RuntimeError(
                    "Formal run requested a deterministic actor seed, but this "
                    "agents SDK ModelSettings does not support extra_body.seed"
                ) from exc
            if configured_sampling_required:
                raise RuntimeError(
                    "OpenAI Agents SDK cannot represent the actor top_p/top_k "
                    "sampling controls required by the model-stack profile"
                ) from exc
            if configured_extra_body_required:
                raise RuntimeError(
                    "OpenAI Agents SDK cannot represent the actor "
                    "request_extra_body required by the model-stack profile"
                ) from exc
            if extra_body:
                settings.pop("extra_body", None)
                try:
                    return ModelSettings(**settings)
                except TypeError as retry_exc:
                    raise RuntimeError(
                        "OpenAI Agents SDK cannot represent required actor "
                        "model settings; refusing unconstrained fallback"
                    ) from retry_exc
            raise RuntimeError(
                "OpenAI Agents SDK cannot represent required actor model "
                "settings; refusing unconstrained fallback"
            ) from exc

    @staticmethod
    def _guard_raw_event(decision: Any) -> dict[str, Any]:
        if decision is None:
            return {}
        return {
            "safepred_guard": {
                "decision_id": getattr(decision, "decision_id", None),
                "allowed": getattr(decision, "allowed", None),
                "requires_regeneration": getattr(decision, "requires_regeneration", None),
                "risk_score": getattr(decision, "risk_score", None),
                "selected_action": getattr(decision, "selected_action", None),
                "intervention_mode": getattr(decision, "intervention_mode", None),
                "proposed_action_filtered": getattr(decision, "proposed_action_filtered", None),
                "feedback_attempt": getattr(decision, "feedback_attempt", None),
                "feedback_budget": getattr(decision, "feedback_budget", None),
                "feedback_budget_exhausted": getattr(
                    decision,
                    "feedback_budget_exhausted",
                    None,
                ),
                "feedback_chain_id": getattr(decision, "feedback_chain_id", None),
                "feedback_parent_decision_id": getattr(
                    decision,
                    "feedback_parent_decision_id",
                    None,
                ),
                "feedback_chain_status": getattr(
                    decision,
                    "feedback_chain_status",
                    None,
                ),
                "intervention_outcome": getattr(decision, "intervention_outcome", None),
            }
        }


def _apply_final_output_guard(
    *,
    original_output: str,
    decision: Any,
    safepred_guard: Any,
) -> str:
    if decision is None:
        return original_output
    if is_prediction_only(safepred_guard):
        return original_output
    if is_advisory_only(safepred_guard):
        if should_return_advisory(decision):
            return advisory_message(
                decision,
                "Final user output was predicted risky.",
            )
        return original_output
    if getattr(decision, "should_execute", False) and getattr(decision, "selected_action", None):
        selected = decision.selected_action
        if (
            isinstance(selected, dict)
            and selected.get("type") == "communication"
            and selected.get("target_role") == "user"
        ):
            return str(selected.get("content") or original_output)
    return (
        "[SAFEPRED_REPLAN_REQUIRED] "
        + (
            getattr(decision, "risk_guidance", "")
            or "Final output was blocked by SafePred."
        )
    )


def _actor_feedback_message(decision: Any, *, reason: str) -> str:
    if (getattr(decision, "safepred_result", {}) or {}).get("environment_h1"):
        return str(decision.risk_guidance)
    if getattr(decision, "intervention_mode", "") == "advisory_only":
        return advisory_message(decision, reason)
    guidance = str(getattr(decision, "risk_guidance", "") or "").strip()
    risk = getattr(decision, "risk_score", None)
    parts = ["[SAFEPRED_REPLAN_REQUIRED]", reason]
    if risk is not None:
        parts.append(f"risk_score={risk}")
    if guidance:
        parts.append(guidance)
    else:
        parts.append("Regenerate a safer response within the current role and task scope.")
    return " ".join(parts)
