"""Task runner — framework-agnostic orchestration with adapters."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_agent.banks import get_bank
from multi_agent.completion_judge import (
    CompletionEvaluation,
    evaluate_completion_checkpoints,
)
from multi_agent.frameworks import (
    RunContext,
    RunOutcome,
    get_framework_adapter,
)
from multi_agent.safepred_guard.terminal import (
    SAFETY_ABSTENTION_DISPOSITION,
    SAFETY_ABSTENTION_TASK_STATUS,
    validate_safety_abstention_terminal,
)
from multi_agent.frameworks.core.action_sink import ActionSink
from multi_agent.frameworks.core.sde_policy import SDE_DOMAIN
from multi_agent.frameworks.core.task_input import build_task_input
from multi_agent.perturbation import PerturbationConfig, apply_task_perturbation
from multi_agent.schemas.actions import ObservableAction
from multi_agent.schemas.task import TaskConfig, ToolCatalog
from multi_agent.schemas.trace import ExecutionTrace
from multi_agent.trace.events import AgentInfo, TraceStart
from multi_agent.trace.provenance import (
    build_actor_prompt_payload,
    build_actor_provenance,
)
from multi_agent.trace.checkpoint import DurableTraceCheckpoint
from multi_agent.trace.runtime_checkpoint import (
    ArtifactBinding,
    RuntimeCommitState,
    RuntimeCheckpointCoordinator,
    RuntimeCheckpointDriftError,
    RuntimeRecoveryState,
    validate_oai_actor_completion_state,
)
from multi_agent.trace.writer import TraceWriter
from multi_agent.workspace import Workspace, WorkspaceManager


def framework_run_id_slug(framework: str) -> str:
    """Map framework id to stable run-id prefix."""
    mapping = {"clawteam": "ct", "oai": "oa", "adk": "adk", "openclaw_local": "ol"}
    return mapping.get(framework, framework[:3])


def next_run_id(
    model: str = "deepseek-v4-flash",
    framework_slug: str = "ct",
    harness_slug: str = "claude",
) -> str:
    """Return a unique run id: ``{framework_slug}-{harness}-{model}-{uuid}``."""
    short = uuid.uuid4().hex[:8]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return f"{framework_slug}-{harness_slug}-{safe_model}-{short}"


def _build_execution_trace(
    *,
    run_id: str,
    task_id: str,
    domain: str,
    framework: str,
    llm: str,
    actions: list[ObservableAction],
    started_at: datetime,
    completion: CompletionEvaluation | None,
    terminal_disposition: str | None = None,
    task_success_status: str = "scored",
    safety_abstention: dict[str, Any] | None = None,
    safety_abstention_artifact: dict[str, Any] | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
    perturbation: dict[str, Any] | None = None,
    workspace_path: str | None = None,
) -> ExecutionTrace:
    completion_fields = (
        {
            "task_completion_score": completion.task_completion_score,
            "deterministic_completion_score": completion.deterministic_completion_score,
            "deterministic_completion_weight": completion.deterministic_completion_weight,
            "deterministic_completion_contribution": (
                completion.deterministic_completion_contribution
            ),
            "llm_completion_score": completion.llm_completion_score,
            "llm_completion_weight": completion.llm_completion_weight,
            "llm_completion_contribution": completion.llm_completion_contribution,
            "completion_weight_total": completion.completion_weight_total,
        }
        if completion is not None
        else {
            "task_completion_score": None,
            "deterministic_completion_score": None,
            "deterministic_completion_weight": None,
            "deterministic_completion_contribution": None,
            "llm_completion_score": None,
            "llm_completion_weight": None,
            "llm_completion_contribution": None,
            "completion_weight_total": None,
        }
    )
    trace = ExecutionTrace(
        run_id=run_id,
        task_id=task_id,
        domain=domain,
        framework=framework,
        llm=llm,
        actions=actions,
        decisions=[],
        terminal_disposition=terminal_disposition,
        task_success_status=task_success_status,
        safety_abstention=safety_abstention,
        safety_abstention_artifact=safety_abstention_artifact,
        **completion_fields,
        workspace_path=workspace_path,
        started_at=started_at,
        error=error,
        warnings=warnings or [],
        perturbation=perturbation,
        perturbation_delivered=None,
        stability_score=None,
        stable=None,
    )
    trace.finalize()
    return trace


class BenchmarkRunner:
    """Framework-agnostic benchmark runner with pluggable adapters."""

    def __init__(
        self,
        task: TaskConfig,
        catalog: ToolCatalog,
        model: str = "deepseek-v4-flash",
        framework: str = "clawteam",
        harness: str = "claude",
        trace_writer: TraceWriter | None = None,
        max_turns: int = 30,
        workspace_manager: WorkspaceManager | None = None,
        bank_factory: Any = None,
        agent_timeout: int = 300,
        judge_model: str = "deepseek-v4-flash",
        skip_judge: bool = False,
        judge_workers: int = 4,
        perturbation: PerturbationConfig | None = None,
        safepred_guard_config: dict[str, Any] | None = None,
        actor_runtime: dict[str, Any] | None = None,
        task_path: Path | None = None,
        repo_root: Path | None = None,
        package_root: Path | None = None,
        trace_resume: bool = False,
        trace_runtime_root: str | Path | None = None,
        trace_checkpoint_path: str | Path | None = None,
        trace_checkpoint_identity: dict[str, Any] | None = None,
        trace_checkpoint_attempt_id: str | None = None,
    ) -> None:
        self.task = task
        self.catalog = catalog
        self.model = model
        self.framework = framework
        self.harness = harness
        self._trace_writer = trace_writer
        self.max_turns = max_turns
        self._workspace_manager = workspace_manager
        self._bank_factory = bank_factory or get_bank
        self._agent_timeout = agent_timeout
        self._judge_model = judge_model
        self._skip_judge = skip_judge
        self._judge_workers = judge_workers
        self._perturbation = perturbation
        self._safepred_guard_config = safepred_guard_config
        self._actor_runtime = dict(actor_runtime or {})
        self._task_path = Path(task_path).resolve() if task_path is not None else None
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._package_root = (
            Path(package_root).resolve() if package_root is not None else None
        )
        self.actor_provenance: dict[str, Any] | None = None
        self._bank: Any = None
        self._workspace: Workspace | None = None
        if type(trace_resume) is not bool:
            raise TypeError("trace_resume must be a boolean")
        self._trace_resume = trace_resume
        self._trace_runtime_root_config = trace_runtime_root
        self._trace_checkpoint_path_config = trace_checkpoint_path
        self._trace_checkpoint_identity_config = trace_checkpoint_identity
        self._trace_checkpoint_attempt_id_config = trace_checkpoint_attempt_id
        self._runtime_checkpoint: DurableTraceCheckpoint | None = None
        self._runtime_checkpoint_coordinator: RuntimeCheckpointCoordinator | None = None
        self._runtime_recovery_state: RuntimeRecoveryState | None = None
        self._runtime_session_path: Path | None = None
        self._runtime_trace_path: Path | None = None
        self._runtime_root: Path | None = None
        self._runtime_checkpoint_path: Path | None = None
        self._validate_trace_resume_configuration()

    @property
    def runtime_checkpoint_coordinator(self) -> RuntimeCheckpointCoordinator | None:
        return self._runtime_checkpoint_coordinator

    @property
    def runtime_recovery_state(self) -> RuntimeRecoveryState | None:
        return self._runtime_recovery_state

    @property
    def runtime_session_path(self) -> Path | None:
        return self._runtime_session_path

    @property
    def runtime_trace_path(self) -> Path | None:
        return self._runtime_trace_path

    @property
    def runtime_root(self) -> Path | None:
        return self._runtime_root

    @property
    def runtime_checkpoint_path(self) -> Path | None:
        return self._runtime_checkpoint_path

    @property
    def runtime_checkpoint_identity(self) -> dict[str, Any] | None:
        value = self._trace_checkpoint_identity_config
        if value is None:
            return None
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))

    @property
    def runtime_checkpoint_attempt_id(self) -> str | None:
        value = self._trace_checkpoint_attempt_id_config
        return str(value) if value is not None else None

    def checkpoint_evaluation_report(
        self,
        trace: ExecutionTrace,
        report_path: str | Path,
        *,
        status: str = "passed",
    ) -> RuntimeRecoveryState:
        """Bind an evaluation report and seal one resumable trace fail-closed."""

        coordinator = self._runtime_checkpoint_coordinator
        root = self._runtime_root
        trace_path = self._runtime_trace_path
        if coordinator is None or root is None or trace_path is None:
            raise RuntimeError("trace resume runtime is not initialized")
        current = coordinator.inspect()
        self._runtime_recovery_state = current
        if not isinstance(status, str) or not status.strip():
            raise ValueError("terminal status must be a nonempty string")
        status = status.strip()
        if trace.run_id != coordinator.checkpoint.run_id:
            raise RuntimeCheckpointDriftError("evaluation trace run_id mismatch")
        if trace.task_id != self.task.task_id:
            raise RuntimeCheckpointDriftError("evaluation trace task_id mismatch")
        if not current.actor_complete:
            raise RuntimeCheckpointDriftError(
                "evaluation report cannot precede actor completion"
            )
        report_binding = coordinator.bind_existing_artifact(
            report_path,
            label="evaluation report",
        )

        evaluation_state = current.evaluation_complete_state
        if evaluation_state is not None:
            if (
                evaluation_state.get("schema_version") != 1
                or evaluation_state.get("run_id") != trace.run_id
                or evaluation_state.get("task_id") != trace.task_id
                or evaluation_state.get("report_role")
                != "harnessaudit_evaluation_report_v1"
            ):
                raise RuntimeCheckpointDriftError(
                    "evaluation completion identity mismatch"
                )
            stored_report = ArtifactBinding.from_json_value(
                evaluation_state.get("report"),
                label="evaluation report",
            )
            if stored_report != report_binding:
                raise RuntimeCheckpointDriftError(
                    "evaluation report differs from committed binding"
                )

        if current.terminal is not None:
            reused = coordinator.validate_terminal_report_for_reuse(
                expected_status=status,
            )
            if reused != report_binding:
                raise RuntimeCheckpointDriftError(
                    "terminal report differs from requested binding"
                )
            self._runtime_recovery_state = coordinator.inspect()
            return self._runtime_recovery_state

        if evaluation_state is None:
            actions = [action.model_dump(mode="json") for action in trace.actions]
            if actions != list(current.normalized_actions):
                raise RuntimeCheckpointDriftError(
                    "evaluation trace actions differ from actor checkpoint"
                )
            if trace_path.is_symlink() or not trace_path.is_file():
                raise RuntimeCheckpointDriftError("runtime trace is missing or unsafe")
            if self._trace_writer is not None:
                self._trace_writer.close()
            bank_connection = getattr(self._bank, "conn", None)
            if self._bank is None:
                if self.task.domain != SDE_DOMAIN:
                    raise RuntimeError(
                        "bankless evaluation checkpoint is restricted to SDE"
                    )
            elif not isinstance(bank_connection, sqlite3.Connection):
                raise RuntimeError("evaluation checkpoint requires an open SQLite bank")
            ledger_paths = {
                name: root / binding.relative_path
                for name, binding in current.ledger_offsets.items()
            }
            commit_state = RuntimeCommitState(
                normalized_actions=actions,
                guard_state=current.guard_state,
                hook_state=current.hook_state,
                counts_state=current.counts_state,
                usage=current.usage,
                ledger_paths=ledger_paths,
                trace_bytes=trace_path.read_bytes(),
            )
            metric_summary = {
                "completion_weight_total": trace.completion_weight_total,
                "deterministic_completion_score": (
                    trace.deterministic_completion_score
                ),
                "llm_completion_score": trace.llm_completion_score,
                "task_completion_score": trace.task_completion_score,
            }
            coordinator.checkpoint_evaluation_complete(
                bank=self._bank,
                state=commit_state,
                evaluation_state={
                    "schema_version": 1,
                    "run_id": trace.run_id,
                    "task_id": trace.task_id,
                    "report_role": "harnessaudit_evaluation_report_v1",
                    "metric_summary": metric_summary,
                },
                report_path=report_path,
            )
            current = coordinator.inspect()
            self._runtime_recovery_state = current

        coordinator.seal_terminal_report(report_path=report_path, status=status)
        self._runtime_recovery_state = coordinator.inspect()
        return self._runtime_recovery_state

    def close_runtime_checkpoint(self) -> None:
        """Close the runner-owned core after the CLI has sealed terminal state."""

        checkpoint = self._runtime_checkpoint
        self._runtime_checkpoint = None
        self._runtime_checkpoint_coordinator = None
        if checkpoint is not None:
            checkpoint.close()

    def _validate_trace_resume_configuration(self) -> None:
        if not self._trace_resume:
            return
        if self.framework != "oai":
            raise ValueError("trace resume currently supports only framework='oai'")
        if self._trace_runtime_root_config is None:
            raise ValueError("trace_resume requires trace_runtime_root")
        if self._trace_checkpoint_path_config is None:
            raise ValueError("trace_resume requires trace_checkpoint_path")
        identity = self._trace_checkpoint_identity_config
        if not isinstance(identity, dict) or not identity:
            raise ValueError("trace_resume requires a nonempty checkpoint identity")
        if any(not isinstance(key, str) for key in identity):
            raise ValueError("trace checkpoint identity keys must be strings")
        try:
            normalized = json.loads(
                json.dumps(
                    identity,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("trace checkpoint identity must be finite JSON") from exc
        if not isinstance(normalized, dict):
            raise ValueError("trace checkpoint identity must be an object")
        attempt_id = self._trace_checkpoint_attempt_id_config
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("trace_resume requires trace_checkpoint_attempt_id")
        self._trace_checkpoint_identity_config = normalized
        self._trace_checkpoint_attempt_id_config = attempt_id.strip()

    @staticmethod
    def _runtime_child_path(root: Path, value: str | Path, *, label: str) -> Path:
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else root / raw
        if candidate.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link")
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside trace_runtime_root") from exc
        if not relative.parts:
            raise ValueError(f"{label} must not be trace_runtime_root")
        return resolved

    def _open_runtime_checkpoint(
        self,
        *,
        run_id: str,
        action_sink: ActionSink,
    ) -> RuntimeRecoveryState:
        writer = self._trace_writer
        if writer is None or not isinstance(getattr(writer, "path", None), Path):
            raise ValueError("trace_resume requires a path-backed TraceWriter")
        raw_root = Path(self._trace_runtime_root_config).expanduser()
        if raw_root.exists() and raw_root.is_symlink():
            raise ValueError("trace_runtime_root must not be a symbolic link")
        root = raw_root.resolve()
        if root.exists() and not root.is_dir():
            raise ValueError("trace_runtime_root must be a directory")
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._runtime_child_path(
            root,
            self._trace_checkpoint_path_config,
            label="trace_checkpoint_path",
        )
        raw_trace_path = Path(writer.path).expanduser()
        if raw_trace_path.is_symlink():
            raise ValueError("trace path must not be a symbolic link")
        trace_path = raw_trace_path.resolve()
        try:
            trace_relative = trace_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "trace path must be inside trace_runtime_root"
            ) from exc
        if not trace_relative.parts:
            raise ValueError("trace path must not be trace_runtime_root")
        session_path = self._runtime_child_path(
            root,
            checkpoint_path.parent / "actor_session.sqlite",
            label="runtime session path",
        )
        if len({checkpoint_path, trace_path, session_path}) != 3:
            raise ValueError("checkpoint, trace, and session paths must be distinct")
        if session_path.exists() and session_path.is_symlink():
            raise ValueError("runtime session path must not be a symbolic link")

        writer.close()
        writer.path = trace_path
        bank_connection = getattr(self._bank, "conn", None)
        bankless_sde = (
            self._bank is None
            and self.task.domain == SDE_DOMAIN
            and self._workspace is not None
        )
        if not bankless_sde and (
            not isinstance(bank_connection, sqlite3.Connection)
            or not callable(getattr(self._bank, "dump_to_file", None))
        ):
            raise RuntimeError(
                "trace resume requires a SQLite-backed bank or an SDE workspace"
            )
        checkpoint = DurableTraceCheckpoint(
            checkpoint_path,
            identity=self._trace_checkpoint_identity_config,
            run_id=run_id,
            attempt_id=self._trace_checkpoint_attempt_id_config,
        )
        coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
        self._runtime_checkpoint = checkpoint
        self._runtime_checkpoint_coordinator = coordinator
        self._runtime_session_path = session_path
        self._runtime_trace_path = trace_path
        self._runtime_root = root
        self._runtime_checkpoint_path = checkpoint_path

        recover_kwargs: dict[str, Any] = {
            "bank": self._bank,
            "trace_path": trace_path,
            "require_resumable": False,
        }
        if self._workspace is not None:
            if "workspace" not in inspect.signature(coordinator.recover).parameters:
                raise RuntimeError(
                    "trace resume cannot restore a workspace with this runtime "
                    "checkpoint coordinator"
                )
            recover_kwargs["workspace"] = self._workspace.path
        recovery = coordinator.recover(**recover_kwargs)
        action_sink.restore_actions(list(recovery.normalized_actions))
        self._runtime_recovery_state = recovery
        return recovery

    @staticmethod
    def _actor_outcome_from_checkpoint(state: Any) -> RunOutcome:
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

    async def _run_or_recover_actor(
        self,
        *,
        run_id: str,
        active_task: TaskConfig,
        action_sink: ActionSink,
        recovery: RuntimeRecoveryState | None,
    ) -> RunOutcome:
        if recovery is not None and recovery.terminal is not None:
            if not recovery.actor_complete:
                raise RuntimeCheckpointDriftError(
                    "terminal runtime checkpoint lacks actor completion"
                )
            return self._actor_outcome_from_checkpoint(
                recovery.actor_complete_state
            )
        if recovery is not None and recovery.actor_complete:
            return self._actor_outcome_from_checkpoint(
                recovery.actor_complete_state
            )

        adapter = get_framework_adapter(
            self.framework,
            agent_timeout=self._agent_timeout,
            harness=self.harness,
        )
        context_kwargs: dict[str, Any] = {
            "run_id": run_id,
            "task": active_task,
            "catalog": self.catalog,
            "model": self.model,
            "max_turns": self.max_turns,
            "bank": self._bank,
            "workspace": self._workspace,
            "action_sink": action_sink,
            "bank_source": self._bank_source(),
            "perturbation": self._perturbation,
            "safepred_guard_config": self._safepred_guard_config,
            "actor_provenance": self.actor_provenance,
        }
        if self._trace_resume:
            runtime_values = {
                "runtime_checkpoint": self._runtime_checkpoint_coordinator,
                "runtime_recovery": recovery,
                "session_path": self._runtime_session_path,
                "trace_path": self._runtime_trace_path,
                "runtime_root": self._runtime_root,
            }
            supported = inspect.signature(RunContext).parameters
            missing = set(runtime_values) - set(supported)
            if missing:
                raise RuntimeError(
                    "RunContext lacks trace-resume fields: "
                    + ", ".join(sorted(missing))
                )
            context_kwargs.update(runtime_values)
        try:
            outcome = await adapter.run(
                RunContext(**context_kwargs),
                build_task_input(active_task),
            )
        except BaseException:
            coordinator = self._runtime_checkpoint_coordinator
            if self._trace_resume and coordinator is not None:
                self._runtime_recovery_state = coordinator.inspect()
            raise
        if not self._trace_resume:
            return outcome

        coordinator = self._runtime_checkpoint_coordinator
        if coordinator is None:
            raise RuntimeError("trace resume coordinator was not initialized")
        current = coordinator.inspect()
        self._runtime_recovery_state = current
        if not current.actor_complete:
            raise RuntimeCheckpointDriftError(
                "adapter returned before checkpointing actor completion"
            )
        recovered_outcome = self._actor_outcome_from_checkpoint(
            current.actor_complete_state
        )
        if (
            outcome.framework != recovered_outcome.framework
            or outcome.final_output != recovered_outcome.final_output
            or outcome.error != recovered_outcome.error
            or outcome.terminal_disposition != recovered_outcome.terminal_disposition
            or outcome.task_success_status != recovered_outcome.task_success_status
            or outcome.safety_abstention != recovered_outcome.safety_abstention
            or outcome.safety_abstention_artifact
            != recovered_outcome.safety_abstention_artifact
        ):
            raise RuntimeCheckpointDriftError(
                "live actor outcome differs from checkpointed actor completion"
            )
        return outcome

    def _bank_source(self) -> str:
        bank_factory = self._bank_factory
        module_name = getattr(bank_factory, "__module__", "")
        if module_name.startswith("single_agent."):
            return "single_agent"
        return "multi_agent"

    def run(
        self,
        run_id: str | None = None,
    ) -> ExecutionTrace:
        """Synchronous entry point."""
        return asyncio.run(self._run_async(run_id=run_id))

    async def _run_async(
        self,
        run_id: str | None = None,
    ) -> ExecutionTrace:
        if self._trace_resume:
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("trace_resume requires an explicit stable run_id")
            run_id = run_id.strip()
        run_id = run_id or next_run_id(
            model=self.model,
            framework_slug=framework_run_id_slug(self.framework),
            harness_slug=self.harness,
        )
        trace_writer = self._trace_writer
        run_started_at = datetime.now(timezone.utc)

        action_sink = ActionSink(
            run_id=run_id,
            trace_writer=trace_writer,
        )
        final_text = ""
        completion: CompletionEvaluation | None = CompletionEvaluation(
            task_completion_score=0.0,
        )
        terminal_disposition: str | None = None
        task_success_status = "scored"
        safety_abstention: dict[str, Any] | None = None
        safety_abstention_artifact: dict[str, Any] | None = None
        run_error: str | None = None
        run_warnings: list[str] = []
        sde_workspace_checked = False
        workspace_path_for_trace: str | None = None

        if self.task.fixture and self._workspace_manager:
            self._workspace = self._workspace_manager.create(self.task.fixture, run_id)

        try:
            self.close_bank()
            self._bank = self._bank_factory(self.task.domain)
            if self._bank is not None and hasattr(self._bank, "apply_task_metadata"):
                self._bank.apply_task_metadata(self.task.metadata)
            active_task = apply_task_perturbation(self.task, self._perturbation)

            if self._task_path and self._repo_root and self._package_root:
                prompt_payload = self._actor_prompt_payload(active_task)
                self.actor_provenance = build_actor_provenance(
                    task=active_task,
                    catalog=self.catalog,
                    task_path=self._task_path,
                    repo_root=self._repo_root,
                    package_root=self._package_root,
                    framework=self.framework,
                    harness=self.harness,
                    model=self.model,
                    max_turns=self.max_turns,
                    runtime=self._actor_runtime,
                    prompt_payload=prompt_payload,
                )

            runtime_recovery: RuntimeRecoveryState | None = None
            if self._trace_resume:
                runtime_recovery = self._open_runtime_checkpoint(
                    run_id=run_id,
                    action_sink=action_sink,
                )
                if runtime_recovery.initial_state_state is not None:
                    raw_started_at = runtime_recovery.initial_state_state[
                        "run_started_at_utc"
                    ]
                    run_started_at = datetime.fromisoformat(
                        raw_started_at[:-1] + "+00:00"
                    ).astimezone(timezone.utc)

            if trace_writer and (
                not self._trace_resume
                or runtime_recovery is None
                or runtime_recovery.trace_snapshot is None
            ):
                actor_provenance = self.actor_provenance or {}
                trace_writer.write_event(
                    TraceStart(
                        trace_id=run_id,
                        task_id=active_task.task_id,
                        domain=active_task.domain,
                        framework=self.framework,
                        model=self.model,
                        agents=[
                            AgentInfo(
                                agent_id=agent_cfg.role,
                                role=agent_cfg.role,
                                useful_tools=agent_cfg.useful_tools,
                                unnecessary_tools=[
                                    tool.name
                                    for tool in self.catalog.tools
                                    if tool.name
                                    not in set(agent_cfg.useful_tools)
                                    and tool.name
                                    not in set(
                                        agent_cfg.tool_necessity.forbidden
                                        if agent_cfg.tool_necessity else []
                                    )
                                ],
                                forbidden_tools=(
                                    list(agent_cfg.tool_necessity.forbidden)
                                    if agent_cfg.tool_necessity else []
                                ),
                                visible_tools=[
                                    tool.name for tool in self.catalog.tools
                                ]
                                + (
                                    ["delegate_to_agent"]
                                    if self.framework == "oai"
                                    and agent_index == 0
                                    else []
                                ),
                            )
                            for agent_index, agent_cfg in enumerate(active_task.agents)
                        ],
                        n_rules=len(active_task.access_rules),
                        perturbation=(
                            self._perturbation.trace_metadata()
                            if self._perturbation else None
                        ),
                        actor_trace_seed=actor_provenance.get("actor_trace_seed"),
                        actor_temperature=actor_provenance.get("actor_temperature"),
                        actor_parallel_tool_calls=actor_provenance.get(
                            "actor_parallel_tool_calls"
                        ),
                        actor_protocol_sha256=actor_provenance.get(
                            "actor_protocol_sha256"
                        ),
                        actor_stable_protocol_sha256=actor_provenance.get(
                            "actor_stable_protocol_sha256"
                        ),
                        actor_runtime_revision=actor_provenance.get(
                            "runtime_revision"
                        ),
                        actor_runtime_versions=actor_provenance.get(
                            "runtime_versions"
                        ),
                        resource_semantics_mode=actor_provenance.get(
                            "resource_semantics_mode"
                        ),
                        actor_provenance=actor_provenance or None,
                    )
                )

            if self._trace_resume:
                coordinator = self._runtime_checkpoint_coordinator
                trace_path = self._runtime_trace_path
                if (
                    coordinator is None
                    or trace_path is None
                    or runtime_recovery is None
                ):
                    raise RuntimeError("trace resume baseline runtime is unavailable")
                if runtime_recovery.initial_state_state is None:
                    if runtime_recovery.last_step_id != 0:
                        raise RuntimeCheckpointDriftError(
                            "runtime checkpoint has steps but no initial baseline"
                        )
                    if action_sink.actions:
                        raise RuntimeCheckpointDriftError(
                            "initial runtime baseline contains actions"
                        )
                    if trace_writer is None:
                        raise RuntimeError("initial runtime baseline requires a trace writer")
                    trace_writer.close()
                    workspace = (
                        self._workspace.path if self._workspace is not None else None
                    )
                    coordinator.checkpoint_initial_state(
                        bank=self._bank,
                        state=RuntimeCommitState(
                            normalized_actions=(),
                            guard_state={},
                            hook_state={},
                            counts_state={},
                            usage={},
                            ledger_paths={},
                            trace_bytes=trace_path.read_bytes(),
                            workspace=workspace,
                        ),
                        run_started_at_utc=(
                            run_started_at.isoformat().replace("+00:00", "Z")
                        ),
                    )
                    runtime_recovery = coordinator.inspect()
                    self._runtime_recovery_state = runtime_recovery

            outcome = await self._run_or_recover_actor(
                run_id=run_id,
                active_task=active_task,
                action_sink=action_sink,
                recovery=runtime_recovery,
            )
            final_text = outcome.final_output
            run_error = outcome.error
            # Keep third-party/legacy adapter mocks compatible: only the typed
            # safety terminal needs the newly added outcome fields.
            terminal_disposition = getattr(outcome, "terminal_disposition", None)
            task_success_status = getattr(outcome, "task_success_status", "scored")
            safety_abstention = getattr(outcome, "safety_abstention", None)
            safety_abstention_artifact = getattr(
                outcome,
                "safety_abstention_artifact",
                None,
            )
            if terminal_disposition == SAFETY_ABSTENTION_DISPOSITION:
                if (
                    run_error is not None
                    or task_success_status != SAFETY_ABSTENTION_TASK_STATUS
                    or safety_abstention is None
                    or safety_abstention_artifact is None
                ):
                    raise RuntimeCheckpointDriftError(
                        "safety abstention outcome is incomplete"
                    )
                validate_safety_abstention_terminal(safety_abstention)
                completion = None
            elif (
                task_success_status != "scored"
                or safety_abstention is not None
                or safety_abstention_artifact is not None
            ):
                raise RuntimeCheckpointDriftError(
                    "ordinary actor outcome carries a safety-terminal payload"
                )
            if self._workspace is not None:
                workspace_path_for_trace = str(self._workspace.path)
                run_warnings.extend(
                    self._sde_workspace_warnings(action_sink.actions, self._workspace)
                )
                sde_workspace_checked = True
            if self._bank is not None and hasattr(self._bank, "reconcile_trace_actions"):
                self._bank.reconcile_trace_actions(action_sink.actions)
            if (
                not run_error
                and terminal_disposition != SAFETY_ABSTENTION_DISPOSITION
                and not (
                    self._trace_resume
                    and self._runtime_recovery_state is not None
                    and (
                        self._runtime_recovery_state.evaluation_complete
                        or self._runtime_recovery_state.terminal is not None
                    )
                )
            ):
                completion = await self._evaluate_completion(
                    action_sink.actions,
                    final_text,
                )
        finally:
            if self._workspace and self._workspace_manager:
                if workspace_path_for_trace is None:
                    workspace_path_for_trace = str(self._workspace.path)
                if not sde_workspace_checked:
                    run_warnings.extend(
                        self._sde_workspace_warnings(action_sink.actions, self._workspace)
                    )
                keep = (
                    self.task.domain == SDE_DOMAIN
                    or completion is None
                    or completion.task_completion_score < 1.0
                    or os.getenv("MASP_KEEP_WORKSPACES") == "1"
                )
                self._workspace_manager.destroy(self._workspace, keep=keep)
                self._workspace = None

        return _build_execution_trace(
            run_id=run_id,
            task_id=self.task.task_id,
            domain=self.task.domain,
            framework=self.framework,
            llm=self.model,
            actions=action_sink.actions,
            started_at=run_started_at,
            completion=completion,
            terminal_disposition=terminal_disposition,
            task_success_status=task_success_status,
            safety_abstention=safety_abstention,
            safety_abstention_artifact=safety_abstention_artifact,
            error=run_error,
            warnings=run_warnings,
            perturbation=(
                self._perturbation.trace_metadata()
                if self._perturbation else None
            ),
            workspace_path=workspace_path_for_trace,
        )

    def _actor_prompt_payload(self, task: TaskConfig) -> dict[str, Any]:
        """Return the exact prompt/input material whose digest is frozen."""

        return build_actor_prompt_payload(task, self.framework)

    def dump_bank_to_file(self, path: Any) -> bool:
        """Dump bank SQLite DB if available."""
        bank = self._bank
        if bank is None:
            return False
        try:
            bank.dump_to_file(Path(path))
            return True
        except Exception:
            return False

    def close_bank(self) -> None:
        """Close and clear the current per-run bank if present."""
        bank = self._bank
        self._bank = None
        if bank is None:
            return
        close_fn = getattr(bank, "close", None)
        if not callable(close_fn):
            return
        try:
            close_fn()
        except Exception:
            pass

    def _sde_workspace_warnings(
        self,
        actions: list[ObservableAction],
        workspace: Workspace,
    ) -> list[str]:
        if self.task.domain != SDE_DOMAIN:
            return []

        has_write_source_code = any(
            action.surface == "tool_call"
            and action.tool_name == "write_source_code"
            for action in actions
        )
        if has_write_source_code:
            return []

        try:
            status = subprocess.run(
                ["git", "-C", str(workspace.path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            return [f"SDE workspace dirty-check failed: {exc}"]

        if status.returncode != 0:
            stderr = status.stderr.strip()
            detail = f": {stderr}" if stderr else ""
            return [f"SDE workspace dirty-check failed{detail}"]

        if status.stdout.strip():
            return [
                "SDE backend workspace changed but trace contains no "
                "write_source_code tool call."
            ]
        return []

    async def _evaluate_completion(
        self,
        actions: list[ObservableAction],
        final_output: str,
    ) -> CompletionEvaluation:
        """Evaluate L2 TCR completion checkpoints from task config."""
        return await evaluate_completion_checkpoints(
            self.task,
            actions,
            final_output,
            bank=self._bank,
            workspace=self._workspace,
            judge_model=self._judge_model,
            skip_judge=self._skip_judge,
            judge_workers=self._judge_workers,
        )


__all__ = [
    "BenchmarkRunner",
    "build_task_input",
    "framework_run_id_slug",
    "next_run_id",
]
