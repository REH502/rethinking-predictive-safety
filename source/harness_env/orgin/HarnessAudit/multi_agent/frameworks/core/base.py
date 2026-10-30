"""Framework adapter base contracts for execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from multi_agent.perturbation import PerturbationConfig
from multi_agent.schemas.task import TaskConfig, ToolCatalog


@dataclass(slots=True)
class RunContext:
    """Shared runtime context passed into framework adapters."""

    run_id: str
    task: TaskConfig
    catalog: ToolCatalog
    model: str
    max_turns: int
    bank: Any
    workspace: Any
    action_sink: Any
    bank_source: str = "multi_agent"
    perturbation: PerturbationConfig | None = None
    safepred_guard_config: dict[str, Any] | None = None
    actor_provenance: dict[str, Any] | None = None
    runtime_checkpoint: Any = None
    runtime_recovery: Any = None
    session_path: Any = None
    trace_path: Any = None
    runtime_root: Any = None


@dataclass(slots=True)
class RunOutcome:
    """Adapter execution result consumed by benchmark runner."""

    framework: str
    final_output: str
    error: str | None = None
    terminal_disposition: str | None = None
    task_success_status: str = "scored"
    safety_abstention: dict[str, Any] | None = None
    safety_abstention_artifact: dict[str, Any] | None = None


class MASFrameworkAdapter(Protocol):
    """Framework-specific adapter contract."""

    framework_id: ClassVar[str]

    async def run(self, ctx: RunContext, initial_input: str | list[dict[str, Any]]) -> RunOutcome:
        ...
