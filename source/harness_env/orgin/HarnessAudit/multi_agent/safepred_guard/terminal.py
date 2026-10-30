"""Typed, content-safe terminal contract for SafePred-enforced abstention."""

from __future__ import annotations

import json
from typing import Any


SAFETY_ABSTENTION_TERMINAL_SCHEMA_VERSION = (
    "harnessaudit-safety-abstention-terminal-v1"
)
SAFETY_ABSTENTION_ARTIFACT_SCHEMA_VERSION = (
    "harnessaudit-safety-abstention-artifact-v1"
)
SAFETY_ABSTENTION_KIND = "safety_abstention"
SAFETY_ABSTENTION_DISPOSITION = "safety_abstained"
SAFETY_ABSTENTION_TASK_STATUS = "not_scored"
SAFETY_ABSTENTION_FINAL_OUTPUT = (
    "Safety guard ended this task before a safe action was released."
)

# A typed safety abstention is a terminal intervention outcome, not a completed
# task. These task-success fields must therefore be persisted explicitly as
# null in both the terminal report and trace_end event. Keeping this tuple at
# the terminal-contract boundary prevents producer and matrix validators from
# silently drifting apart.
SAFETY_ABSTENTION_UNSCORED_SCORE_FIELDS = (
    "avs",
    "operational_safety_score",
    "tcr",
    "task_completion_score",
    "trajectory_tcr",
    "deterministic_completion_score",
    "deterministic_completion_weight",
    "deterministic_completion_contribution",
    "llm_completion_score",
    "llm_completion_weight",
    "llm_completion_contribution",
    "completion_weight_total",
)

_SAFETY_ABSTENTION_TERMINAL_FIELDS = {
    "schema_version",
    "kind",
    "terminal_disposition",
    "task_success_status",
    "termination_reason",
    "guard_intervention_outcome",
    "run_id",
    "task_id",
    "intervention_mode",
    "decision_id",
    "feedback_chain_id",
    "feedback_parent_decision_id",
    "feedback_attempt",
    "feedback_budget",
    "feedback_chain_status",
    "agent_role",
    "surface",
    "trace_action_cutoff",
    "decision_ledger_file",
}
_SAFETY_ABSTENTION_ARTIFACT_FIELDS = {
    "artifact_schema_version",
    "terminal",
    "decision_ledger",
}
_SAFETY_ABSTENTION_LEDGER_FIELDS = {
    "relative_path",
    "sha256",
    "size_bytes",
}


def _json_copy(value: Any, *, label: str) -> Any:
    """Return a detached JSON value while rejecting lossy/non-finite data."""

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
        raise TypeError(f"{label} must be JSON-safe") from exc


def validate_safety_abstention_terminal(value: Any) -> dict[str, Any]:
    """Validate the secret-free task-level safety-abstention payload."""

    terminal = _json_copy(value, label="safety abstention terminal")
    if (
        not isinstance(terminal, dict)
        or set(terminal) != _SAFETY_ABSTENTION_TERMINAL_FIELDS
    ):
        raise ValueError("safety abstention terminal has an invalid schema")
    if terminal["schema_version"] != SAFETY_ABSTENTION_TERMINAL_SCHEMA_VERSION:
        raise ValueError("safety abstention terminal schema version mismatch")
    if terminal["kind"] != SAFETY_ABSTENTION_KIND:
        raise ValueError("safety abstention terminal kind mismatch")
    if terminal["terminal_disposition"] != SAFETY_ABSTENTION_DISPOSITION:
        raise ValueError("safety abstention terminal disposition mismatch")
    if terminal["task_success_status"] != SAFETY_ABSTENTION_TASK_STATUS:
        raise ValueError("safety abstention terminal task status mismatch")
    for name in (
        "termination_reason",
        "guard_intervention_outcome",
        "run_id",
        "task_id",
        "intervention_mode",
        "feedback_chain_id",
        "feedback_chain_status",
        "agent_role",
        "decision_ledger_file",
    ):
        if not isinstance(terminal[name], str) or not terminal[name]:
            raise ValueError(f"safety abstention terminal {name} must be nonempty")
    if terminal["surface"] not in {"tool_call", "communication", "final_output"}:
        raise ValueError("safety abstention terminal has an invalid surface")
    for name in (
        "decision_id",
        "feedback_attempt",
        "feedback_budget",
        "trace_action_cutoff",
    ):
        if type(terminal[name]) is not int or terminal[name] < 0:
            raise ValueError(f"safety abstention terminal {name} must be nonnegative")
    if terminal["decision_id"] <= 0:
        raise ValueError("safety abstention terminal decision_id must be positive")
    parent = terminal["feedback_parent_decision_id"]
    if parent is not None and (type(parent) is not int or parent <= 0):
        raise ValueError("safety abstention terminal parent decision is invalid")
    return terminal


def validate_artifact_binding(value: Any) -> dict[str, Any]:
    """Validate a portable content-addressed artifact binding."""

    binding = _json_copy(value, label="artifact binding")
    if not isinstance(binding, dict) or set(binding) != _SAFETY_ABSTENTION_LEDGER_FIELDS:
        raise ValueError("artifact binding has an invalid schema")
    if not isinstance(binding["relative_path"], str) or not binding["relative_path"]:
        raise ValueError("artifact binding path is invalid")
    if not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64:
        raise ValueError("artifact binding hash is invalid")
    try:
        int(binding["sha256"], 16)
    except ValueError as exc:
        raise ValueError("artifact binding hash is invalid") from exc
    if type(binding["size_bytes"]) is not int or binding["size_bytes"] < 0:
        raise ValueError("artifact binding size is invalid")
    return binding


def validate_safety_abstention_artifact(value: Any) -> dict[str, Any]:
    """Validate the sidecar payload that binds an abstention to its ledger."""

    artifact = _json_copy(value, label="safety abstention artifact")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != _SAFETY_ABSTENTION_ARTIFACT_FIELDS
    ):
        raise ValueError("safety abstention artifact has an invalid schema")
    if artifact["artifact_schema_version"] != SAFETY_ABSTENTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("safety abstention artifact schema version mismatch")
    artifact["terminal"] = validate_safety_abstention_terminal(artifact["terminal"])
    artifact["decision_ledger"] = validate_artifact_binding(
        artifact["decision_ledger"]
    )
    return artifact


__all__ = [
    "SAFETY_ABSTENTION_ARTIFACT_SCHEMA_VERSION",
    "SAFETY_ABSTENTION_DISPOSITION",
    "SAFETY_ABSTENTION_FINAL_OUTPUT",
    "SAFETY_ABSTENTION_KIND",
    "SAFETY_ABSTENTION_TASK_STATUS",
    "SAFETY_ABSTENTION_TERMINAL_SCHEMA_VERSION",
    "SAFETY_ABSTENTION_UNSCORED_SCORE_FIELDS",
    "validate_artifact_binding",
    "validate_safety_abstention_artifact",
    "validate_safety_abstention_terminal",
]
