"""JSONL trace event types for multi-agent access control evaluation.

Each event is a Pydantic model with a ``type`` literal discriminator field.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Helper models ----


class AgentInfo(BaseModel):
    """Lightweight agent descriptor for TraceStart."""

    agent_id: str
    role: str
    useful_tools: list[str] = Field(
        default_factory=list,
        description="Task-useful tool set. Other catalog tools are visible but scored.",
    )
    unnecessary_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Effective mildly out-of-scope domain-tool set, including explicitly "
            "unnecessary and unmentioned catalog tools."
        ),
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Task-declared forbidden tool set.",
    )
    visible_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Complete action/tool names visible to this actor, including the OAI "
            "hub delegation action when applicable."
        ),
    )


# ---- Event types ----


class TraceStart(BaseModel):
    """Emitted once at the start of a run."""

    type: Literal["trace_start"] = "trace_start"
    trace_id: str
    task_id: str
    domain: str
    framework: str
    model: str
    agents: list[AgentInfo] = Field(default_factory=list)
    n_rules: int = 0
    perturbation: dict[str, Any] | None = None
    # Native actor-generation provenance.  These values are resolved and
    # written before the framework adapter (and therefore the model) is called.
    actor_trace_seed: int | None = None
    actor_temperature: float | None = None
    actor_parallel_tool_calls: bool | None = None
    actor_protocol_sha256: str | None = None
    actor_stable_protocol_sha256: str | None = None
    actor_runtime_revision: str | None = None
    actor_runtime_versions: dict[str, Any] | None = None
    resource_semantics_mode: str | None = None
    actor_provenance: dict[str, Any] | None = None
    timestamp: str = Field(default_factory=_now)


class ToolCallEvent(BaseModel):
    """Emitted for every tool invocation.

    ``timestamp`` records when the underlying harness observed the completed
    action, not when HarnessAudit later wrote the trace line.
    """

    type: Literal["tool_call"] = "tool_call"
    trace_id: str
    sequence_num: int
    agent_id: str
    agent_role: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    raw_event: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)



class CommunicationEvent(BaseModel):
    """Emitted for inter-agent messages and final output to user.

    ``timestamp`` records the source-event time from the native artifact when
    available.
    """

    type: Literal["communication"] = "communication"
    trace_id: str
    sequence_num: int
    agent_id: str
    agent_role: str
    target_agent: str
    target_role: str
    content: str = ""
    raw_event: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)


class SafetyAbstentionEvent(BaseModel):
    """Framework-owned task terminal for an enforced SafePred abstention."""

    type: Literal["safety_abstention"] = "safety_abstention"
    trace_id: str
    terminal: dict[str, Any]
    timestamp: str = Field(default_factory=_now)


class AccessDecisionEvent(BaseModel):
    """Emitted for each action's access check result (post-run)."""

    type: Literal["access_decision"] = "access_decision"
    trace_id: str
    action_sequence_num: int
    violated: bool
    violation_code: str | None = None
    severity: str | None = None
    rule_description: str | None = None
    timestamp: str = Field(default_factory=_now)


class TraceEnd(BaseModel):
    """Emitted once at the end (always, even on error)."""

    type: Literal["trace_end"] = "trace_end"
    trace_id: str
    n_actions: int = 0
    n_tool_calls: int = 0
    n_communications: int = 0
    n_violations: int = 0

    # Paper-facing metrics.
    # L1 Boundary Compliance: SAR_t, SAR_r, SAR_f, and average SAR.
    sar_tool: float = 0.0
    sar_resource: float = 0.0
    sar_flow: float = 0.0
    sar_avg: float = 0.0
    # L2 Execution Fidelity: action validity and task completion.
    avs: float | None = None
    tcr: float | None = 0.0

    # Diagnostics / legacy aliases retained for existing analysis scripts.
    protocol_safety_score: float = 0.0
    tool_penalty: float = 0.0
    resource_penalty: float = 0.0
    flow_penalty: float = 0.0
    comm_penalty: float = 0.0
    operational_safety_score: float | None = None
    task_completion_score: float | None = 0.0
    # Raw task-success components are additive trace provenance. A typed
    # safety abstention writes all of them explicitly as null rather than
    # allowing model defaults to masquerade as a scored task outcome.
    trajectory_tcr: float | None = None
    deterministic_completion_score: float | None = None
    deterministic_completion_weight: float | None = None
    deterministic_completion_contribution: float | None = None
    llm_completion_score: float | None = None
    llm_completion_weight: float | None = None
    llm_completion_contribution: float | None = None
    completion_weight_total: float | None = None
    perturbation: dict[str, Any] | None = None
    terminal_disposition: str | None = None
    task_success_status: str = "scored"
    safety_abstention: dict[str, Any] | None = None
    safety_abstention_artifact: dict[str, Any] | None = None

    violations_by_code: dict[str, int] = Field(default_factory=dict)
    wall_time_s: float = 0.0
    error: str | None = None
    timestamp: str = Field(default_factory=_now)
