"""Unified SafePred-HarnessAudit prediction-only experiment pipeline.

The pipeline has one job: produce fair prediction-only horizon results where
all policy modes evaluate the same HarnessAudit trace prefixes, the same
candidate next actions, and the same horizon schedule.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from multi_agent.credentials import load_workspace_credentials

from multi_agent.loader import load_task_with_tools, resolve_task_path
from multi_agent.resource_semantics import (
    EXPLICIT_RESOURCE_SEMANTICS,
    LEGACY_OFFICIAL_RESOURCE_SEMANTICS,
    normalize_resource_semantics_mode,
)
from multi_agent.safepred_experiments.offline_prediction import (
    OfflinePredictionStep,
    PrefixActionSink,
    build_offline_prediction_steps,
    effective_horizon_for_step,
    run_clean_harnessaudit_task,
)
from multi_agent.safepred_experiments.formal_contract import (
    ActorReplicateIdentity,
    HA_DIRECT_PROMPT_VERSION,
    HA_ITERATIVE_PROMPT_VERSION,
    HA_MODEL_INPUT_BOUNDARY_VERSION,
    HA_POLICY_CACHE_SCHEMA_VERSION,
    HA_POLICY_EXTRACTION_PROMPT_VERSION,
    HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS,
    HA_TRAIN_POLICY_CONSTRUCTION_VERSION,
    freeze_formal_replicate_identity,
    freeze_preregistration,
    load_preregistration,
    normalize_actor_replicate_identity,
    read_ha_source_replicate_identity,
    source_input_manifest_sha256,
    validate_annotation_contract_files,
    validate_formal_output_registration,
    validate_preregistered_offline_plan,
)
from multi_agent.safepred_experiments.provenance import write_experiment_manifest
from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
    SparsePoint,
    _ground_truth_labels,
    _trace_access_decision_details,
    aggregate_support_is_formal,
    aggregate_by_policy_horizon,
    analyze_sparse_results,
    plot_pilot_summary,
    plot_tier1_horizon,
    plot_tier2_horizon,
    plot_token_split_horizon,
    summarize_point_rows,
)
from multi_agent.schemas.actions import ObservableAction
from multi_agent.trace.provenance import validate_actor_provenance


DEFAULT_POLICY_MODES = ("full_harnessaudit_rules", "train_policy")
DIRECT_MULTI_HORIZON_ROLLOUT = "direct_multi_horizon"
ITERATIVE_WM_ROLLOUT = "iterative_wm_rollout"
DEFAULT_ROLLOUT_MODES = (DIRECT_MULTI_HORIZON_ROLLOUT, ITERATIVE_WM_ROLLOUT)
VALID_ROLLOUT_MODES = set(DEFAULT_ROLLOUT_MODES)
STRICT_MULTI_HORIZON_ROLLOUTS = {
    ITERATIVE_WM_ROLLOUT,
}
FORMAL_HA_TASK_APPROVAL_PATH = Path(__file__).with_name(
    "formal_ha_task_approval.json"
)
# Offline unit tests can inject a guard without importing the model stack.
SafePredGuardSettings: Any = None
SafePredHarnessGuard: Any = None


class PredictionDecisionError(RuntimeError):
    """A persisted guard decision error that makes a replay scientifically invalid."""

    def __init__(
        self,
        *,
        task: str,
        rollout: str,
        policy: str,
        horizon: int,
        point: int,
        decision_file: Path,
        guard_error: Any,
    ) -> None:
        self.task = str(task)
        self.task_id = self.task
        self.rollout = str(rollout)
        self.rollout_mode = self.rollout
        self.policy = str(policy)
        self.policy_mode = self.policy
        self.horizon = int(horizon)
        self.point = int(point)
        self.point_num = self.point
        self.decision_file = Path(decision_file)
        self.guard_error = _sanitized_guard_error(guard_error)
        self.metadata = {
            "task": self.task,
            "rollout": self.rollout,
            "policy": self.policy,
            "horizon": self.horizon,
            "point": self.point,
            "decision_file": str(self.decision_file),
            "guard_error": self.guard_error,
        }
        super().__init__(
            "Prediction decision failed after persistence: "
            f"task={self.task}, rollout={self.rollout}, policy={self.policy}, "
            f"horizon={self.horizon}, point={self.point}, "
            f"decision_file={self.decision_file}, "
            "guard_error="
            f"{json.dumps(self.guard_error, sort_keys=True, ensure_ascii=True)}"
        )


def _sanitized_guard_error(error: Any) -> dict[str, str]:
    """Return only bounded, JSON-safe guard error fields."""

    if isinstance(error, dict):
        error_type = str(error.get("type") or "UnknownGuardError")
        message = str(error.get("message") or "")
    else:
        error_type = type(error).__name__
        message = str(error)
    return {
        "type": _redact_credentials(error_type.strip())[:200],
        "message": _redact_credentials(message.strip())[:2000],
    }


def _redact_credentials(value: str) -> str:
    redacted = str(value)
    api_key_values = sorted(
        {
            env_value
            for env_name, env_value in os.environ.items()
            if env_value
            and (
                "API_KEY" in env_name.upper()
                or "ACCESS_TOKEN" in env_name.upper()
            )
        },
        key=len,
        reverse=True,
    )
    for secret in api_key_values:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(
        r"(?i)\bBearer\s+[^\s,;\"'}\]]+",
        "Bearer [REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9_-]+",
        "[REDACTED]",
        redacted,
    )
    return re.sub(
        r"(?i)([?&](?:api_key|access_token|token|key)=)[^&#\s]*",
        r"\1[REDACTED]",
        redacted,
    )


def _fail_fast_decision_error(
    decision: Any,
    *,
    rollout_mode: str,
    effective_horizon: int,
    direct_multi: bool,
    prefix_multi: bool,
) -> Any:
    guard_error = getattr(decision, "error", None)
    if guard_error is not None:
        return guard_error
    if getattr(decision, "skipped_by_prediction_interval", False):
        return {
            "type": "PredictionIntervalSkipped",
            "message": "SafePred decision was skipped by prediction interval.",
        }
    if not direct_multi and not prefix_multi:
        return None

    result = getattr(decision, "safepred_result", None)
    result = result if isinstance(result, dict) else {}
    payload_key = "direct_multi_horizon" if direct_multi else "prefix_multi_horizon"
    payload = result.get(payload_key)
    predictions = (
        payload.get("horizon_predictions")
        if isinstance(payload, dict)
        else None
    )
    required_horizons = (
        [int(effective_horizon)]
        if direct_multi
        else list(range(1, int(effective_horizon) + 1))
    )
    for horizon in required_horizons:
        prediction = (
            predictions.get(f"h{horizon}")
            if isinstance(predictions, dict)
            else None
        )
        if (
            not isinstance(prediction, dict)
            or (
                "valid_prediction" in prediction
                and prediction["valid_prediction"] is not True
            )
            or not isinstance(prediction.get("predicted_risk"), bool)
        ):
            return {
                "type": "InvalidHorizonPrediction",
                "message": (
                    f"{rollout_mode} has a missing or invalid "
                    f"h{horizon} prediction."
                ),
            }
    return None


def _replay_job_horizons(rollout_mode: str, horizons: Iterable[int]) -> list[int]:
    """Return endpoint job horizons without changing the estimand grid."""

    normalized = sorted({int(horizon) for horizon in horizons})
    if not normalized:
        raise ValueError("prediction horizons must not be empty")
    if rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT:
        return normalized
    if rollout_mode == ITERATIVE_WM_ROLLOUT:
        return [max(normalized)]
    raise ValueError(f"Unsupported rollout mode: {rollout_mode!r}")


@dataclass(frozen=True)
class ArtifactPaths:
    output_root: Path
    trace_root: Path
    analysis_dir: Path
    figure_dir: Path


@dataclass(frozen=True)
class PointSelectionConfig:
    scheme: str = "start_mid"
    count: int = 4
    interval: int = 1
    explicit_sequences: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ModelAPIConfig:
    provider: str = "custom"
    api_key: str | None = None
    api_url: str | None = None
    model: str | None = None
    model_revision: str | None = None
    policy_extraction_model: str | None = None


@dataclass(frozen=True)
class CleanAgentAPIConfig:
    """OpenAI-compatible backend used only for generating HarnessAudit traces."""

    api_key: str | None = None
    api_url: str | None = None
    model: str | None = None
    judge_model: str | None = None


def safepred_provider_for(provider: str | None) -> str:
    """Map user-facing providers to the SafePred LLMClient provider set."""

    normalized = (provider or "custom").strip().lower().replace("_", "-")
    if normalized == "openai":
        return "openai"
    if normalized in {"gemini", "google", "google-gemini"}:
        return "gemini"
    # SafePred currently supports openai/gemini/custom only. DeepSeek,
    # Claude-through-proxy, OpenRouter, vLLM, Azure-compatible gateways, etc.
    # all use the custom HTTP client.
    return "custom"


def apply_model_api_env(
    config: ModelAPIConfig,
    *,
    mutate_os: bool = True,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply explicit model API settings to an environment dictionary.

    API keys are intentionally kept in environment variables only; the generated
    SafePred YAML config stores provider/model names but never persists secrets.
    """

    env = dict(base_env or os.environ)
    raw_provider = (config.provider or "custom").strip().lower().replace("_", "-")
    provider = safepred_provider_for(raw_provider)
    key = (config.api_key or "").strip()
    url = (config.api_url or "").strip().rstrip("/")

    def set_if_value(name: str, value: str) -> None:
        if value:
            env[name] = value

    env["SAFEPRED_LLM_PROVIDER"] = raw_provider
    env["SAFEPRED_LLM_CLIENT_PROVIDER"] = provider
    if config.model:
        env["SAFEPRED_MODEL_NAME"] = config.model
    policy_model = config.policy_extraction_model or config.model
    if policy_model:
        env["SAFEPRED_POLICY_EXTRACTION_MODEL"] = policy_model

    if provider == "openai":
        set_if_value("OPENAI_API_KEY", key)
        set_if_value("OPENAI_API_URL", url)
        set_if_value("OPENAI_BASE_URL", url)
    elif provider == "gemini":
        set_if_value("GEMINI_API_KEY", key)
        set_if_value("GOOGLE_API_KEY", key)
        set_if_value("GEMINI_API_URL", url)
        set_if_value("GOOGLE_API_URL", url)
    else:
        set_if_value("CUSTOM_API_KEY", key)
        set_if_value("CUSTOM_API_URL", url)
        # HarnessAudit's OAI path and train-policy extractor both understand
        # OpenAI-compatible env names, so mirror custom gateways there too.
        set_if_value("OPENAI_API_KEY", key)
        set_if_value("OPENAI_API_URL", url)
        set_if_value("OPENAI_BASE_URL", url)

    if raw_provider in {"claude", "anthropic"}:
        set_if_value("ANTHROPIC_API_KEY", key)
        set_if_value("ANTHROPIC_API_URL", url)
        set_if_value("ANTHROPIC_BASE_URL", url)

    if mutate_os:
        os.environ.update(env)
    return env


def _resolved_model_api_url(config: ModelAPIConfig) -> str:
    """Resolve the same provider-specific endpoint that SafePred will use."""

    explicit = str(config.api_url or "").strip().rstrip("/")
    if explicit:
        return explicit
    provider = safepred_provider_for(config.provider)
    if provider == "openai":
        candidates = ("OPENAI_API_URL", "OPENAI_BASE_URL")
    elif provider == "gemini":
        candidates = ("GEMINI_API_URL", "GOOGLE_API_URL")
    else:
        candidates = (
            "CUSTOM_API_URL",
            "WM_BASE_URL",
            "OPENAI_API_URL",
            "OPENAI_BASE_URL",
        )
    return next(
        (
            str(os.environ.get(name) or "").strip().rstrip("/")
            for name in candidates
            if str(os.environ.get(name) or "").strip()
        ),
        "",
    )


def clean_agent_env_overrides(config: CleanAgentAPIConfig | None) -> dict[str, str]:
    """Build environment overrides for the clean HarnessAudit agent run.

    Predict-only experiments have two model backends:
    - clean HarnessAudit agent trace generation, which may need tool calling;
    - SafePred world-model replay, configured separately through ModelAPIConfig.

    This helper intentionally only returns OpenAI-compatible env names consumed
    by the HarnessAudit OAI adapter. It does not mutate SafePred WM settings.
    """

    if config is None:
        return {}
    env: dict[str, str] = {}
    key = (config.api_key or "").strip()
    url = (config.api_url or "").strip().rstrip("/")
    if key:
        env["OPENAI_API_KEY"] = key
    if url:
        env["OPENAI_BASE_URL"] = url
        env["OPENAI_API_BASE"] = url
        env["OPENAI_API_URL"] = url
    # Keep CUSTOM_* aligned for helper code that mirrors OpenAI-compatible
    # gateways, without affecting the parent process SafePred WM env.
    if key:
        env["CUSTOM_API_KEY"] = key
    if url:
        env["CUSTOM_API_URL"] = url
    env["MASP_OPENAI_USE_CHAT_COMPLETIONS"] = "1"
    return env


def apply_test_count(
    tasks: list[str],
    source_reports: list[Path],
    test_count: int | None,
) -> tuple[list[str], list[Path]]:
    if test_count is None or int(test_count) <= 0:
        return list(tasks), list(source_reports)
    limit = int(test_count)
    selected_tasks = list(tasks)[:limit]
    remaining = max(0, limit - len(selected_tasks))
    selected_reports = list(source_reports)[:remaining]
    return selected_tasks, selected_reports


def resolve_artifact_paths(
    *,
    output_root: Path | None,
    trace_root: Path | None,
    analysis_dir: Path | None,
    figure_dir: Path | None,
    timestamp: str | None = None,
) -> ArtifactPaths:
    pkg_root = Path(__file__).resolve().parents[1]
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output = output_root or (pkg_root / "results" / f"predict_only_{stamp}")
    resolved_trace = trace_root or (pkg_root / "traces" / f"predict_only_{stamp}")
    resolved_analysis = analysis_dir or (resolved_output / "_analysis")
    resolved_figures = figure_dir or (resolved_analysis / "figures")
    return ArtifactPaths(
        output_root=resolved_output,
        trace_root=resolved_trace,
        analysis_dir=resolved_analysis,
        figure_dir=resolved_figures,
    )


def select_prediction_points(
    steps: list[OfflinePredictionStep],
    config: PointSelectionConfig,
) -> list[SparsePoint]:
    scheme = config.scheme.strip().lower()
    if scheme == "middle_pair":
        if int(config.count) != 2:
            raise ValueError("middle_pair requires count == 2")
        if len(steps) < 2:
            raise ValueError(
                "middle_pair requires at least two eligible prediction steps"
            )
    if not steps:
        return []
    if scheme == "all":
        indices = list(range(len(steps)))
    elif scheme == "first":
        indices = list(range(min(max(1, int(config.count)), len(steps))))
    elif scheme == "interval":
        interval = max(1, int(config.interval))
        indices = list(range(0, len(steps), interval))
    elif scheme == "explicit":
        wanted = {int(item) for item in config.explicit_sequences if int(item) > 0}
        indices = [index for index, step in enumerate(steps) if step.action_sequence_num in wanted]
    elif scheme == "start_mid":
        indices = _start_mid_indices(len(steps), max(1, int(config.count)))
    elif scheme == "middle_pair":
        middle = len(steps) // 2
        indices = [middle - 1, middle]
    else:
        raise ValueError(
            f"Unsupported prediction point scheme '{config.scheme}'. "
            "Use one of: start_mid, middle_pair, all, first, interval, explicit."
        )

    unique: list[int] = []
    for index in indices:
        if 0 <= index < len(steps) and index not in unique:
            unique.append(index)
    return [
        SparsePoint(point_num=point_num, step_index_0based=index, step=steps[index])
        for point_num, index in enumerate(unique, start=1)
    ]


def select_common_cohort_prediction_points(
    steps: list[OfflinePredictionStep],
    config: PointSelectionConfig,
    *,
    max_horizon: int,
) -> list[SparsePoint]:
    """Select points after eligibility filtering so selection slots are not lost."""

    eligible = [
        step for step in steps if step.remaining_action_count >= int(max_horizon)
    ]
    return select_prediction_points(eligible, config)


def validate_selected_prefix_budget(
    points_by_task: Iterable[tuple[str, list[SparsePoint]]],
    max_trace_events: int,
) -> int:
    """Reject a trace budget that would discard any selected point's prefix."""

    longest_prefix = 0
    longest_task = ""
    for task_id, points in points_by_task:
        for point in points:
            prefix_length = len(point.step.trace_prefix_actions)
            if prefix_length > longest_prefix:
                longest_prefix = prefix_length
                longest_task = task_id
    if int(max_trace_events) < longest_prefix:
        raise ValueError(
            "max_trace_events would truncate an observed prediction prefix: "
            f"task={longest_task}, configured={max_trace_events}, "
            f"required_at_least={longest_prefix}."
        )
    return longest_prefix


def _start_mid_indices(total_steps: int, count: int) -> list[int]:
    if total_steps <= 0:
        return []
    if count <= 1 or total_steps == 1:
        return [0]
    if count == 2:
        return [0, min(1, total_steps - 1)]

    start_count = min(2, count, total_steps)
    indices = list(range(start_count))
    remaining = max(0, min(count, total_steps) - len(indices))
    if remaining:
        mid_left = max(start_count, total_steps // 2 - remaining // 2)
        mid_left = min(mid_left, max(start_count, total_steps - remaining))
        indices.extend(range(mid_left, mid_left + remaining))
    return indices


def build_predict_only_manifest_rows(
    *,
    source_report: str,
    task_id: str,
    policy_modes: Iterable[str],
    rollout_modes: Iterable[str],
    horizons: Iterable[int],
    points: list[SparsePoint],
    point_scheme: str = "start_mid",
    target_estimand: str = "exact_step",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizons = [int(horizon) for horizon in horizons]
    manifest_points = (
        [
            point
            for point in points
            if point.step.remaining_action_count >= max(horizons, default=1)
        ]
        if target_estimand == "exact_step"
        else points
    )
    selected_sequences = ";".join(
        str(point.step.action_sequence_num) for point in manifest_points
    )
    selected_indices = ";".join(
        str(point.step_index_0based) for point in manifest_points
    )
    for rollout_mode in rollout_modes:
        for policy_mode in policy_modes:
            for horizon in horizons:
                rows.append(
                    {
                        "source_report": source_report,
                        "task_id": task_id,
                        "rollout_mode": str(rollout_mode),
                        "policy_mode": str(policy_mode),
                        "horizon": int(horizon),
                        "target_estimand": target_estimand,
                        "horizon_cohort": "common_max_horizon_support",
                        "prediction_point_scheme": point_scheme,
                        "selected_action_sequence_nums": selected_sequences,
                        "selected_step_indices_0based": selected_indices,
                        "selected_prediction_points": len(manifest_points),
                    }
                )
    return rows


def build_train_policy_split_audit_rows(
    task_dicts: list[dict[str, Any]],
    *,
    policy_modes: Iterable[str],
) -> list[dict[str, Any]]:
    if "train_policy" not in {str(mode) for mode in policy_modes}:
        return []

    from multi_agent.safepred_guard.policies import build_train_policy_context

    rows: list[dict[str, Any]] = []
    for task in task_dicts:
        task_id = str(task.get("task_id") or "")
        category = str(task.get("category") or "")
        try:
            context = build_train_policy_context(task)
            training_task_ids = [str(item) for item in context.get("training_task_ids") or []]
            training_categories = [str(item) for item in context.get("training_categories") or []]
            input_includes_test_rules = bool(context.get("input_includes_test_access_rules"))
            test_task_in_training = task_id in set(training_task_ids)
            test_category_in_training = category in set(training_categories)
            status = "ok"
            if (
                input_includes_test_rules
                or test_task_in_training
                or (
                    context.get("train_selection")
                    == "same_domain_excluding_test_and_category"
                    and test_category_in_training
                )
            ):
                status = "leak_risk"
            rows.append(
                {
                    "task_id": task_id,
                    "domain": str(task.get("domain") or ""),
                    "category": category,
                    "policy_mode": "train_policy",
                    "method": context.get("method"),
                    "split_id": context.get("split_id"),
                    "split_source": context.get("split_source"),
                    "train_selection": context.get("train_selection"),
                    "test_task_ids": ";".join(str(item) for item in context.get("test_task_ids") or []),
                    "training_task_ids": ";".join(training_task_ids),
                    "training_task_count": int(context.get("training_task_count") or 0),
                    "training_categories": ";".join(training_categories),
                    "few_shot_example_count": int(context.get("few_shot_example_count") or 0),
                    "input_includes_test_access_rules": input_includes_test_rules,
                    "test_task_in_training_ids": test_task_in_training,
                    "test_category_in_training_categories": test_category_in_training,
                    "test_visible_fields": ";".join(str(item) for item in context.get("test_visible_fields") or []),
                    "status": status,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "domain": str(task.get("domain") or ""),
                    "category": category,
                    "policy_mode": "train_policy",
                    "method": "",
                    "split_id": "",
                    "split_source": "",
                    "train_selection": "",
                    "test_task_ids": task_id,
                    "training_task_ids": "",
                    "training_task_count": 0,
                    "training_categories": "",
                    "few_shot_example_count": 0,
                    "input_includes_test_access_rules": False,
                    "test_task_in_training_ids": False,
                    "test_category_in_training_categories": False,
                    "test_visible_fields": "",
                    "status": f"error:{type(exc).__name__}",
                }
            )
    return rows


def write_train_policy_split_audit(
    *,
    report_paths: list[Path],
    policy_modes: Iterable[str],
    analysis_dir: Path,
) -> dict[str, Path]:
    task_dicts: list[dict[str, Any]] = []
    pkg_root = Path(__file__).resolve().parents[1]
    seen: set[str] = set()
    for report_path in report_paths:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot build train-policy split audit from {report_path}: {exc}"
            ) from exc
        task_id = str(report.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        task, _catalog = load_task_with_tools(f"{task_id}.yaml", project_root=pkg_root, cwd=pkg_root)
        task_dicts.append(task.model_dump(mode="json"))

    rows = build_train_policy_split_audit_rows(task_dicts, policy_modes=policy_modes)
    json_path = analysis_dir / "predict_only_train_policy_split_audit.json"
    csv_path = analysis_dir / "predict_only_train_policy_split_audit.csv"
    write_split_audit_json(json_path, rows)
    _write_csv(csv_path, rows)
    invalid = [row for row in rows if row.get("status") != "ok"]
    if invalid:
        details = ", ".join(
            f"{row.get('task_id')}:{row.get('status')}" for row in invalid[:12]
        )
        raise RuntimeError(
            "train_policy split audit failed; refusing formal prediction: " + details
        )
    return {"split_audit_json": json_path, "split_audit_csv": csv_path}


def write_split_audit_json(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "artifact_type": "train_policy_split_audit",
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trace_event_to_observable_action(
    event: dict[str, Any],
    *,
    run_id: str,
) -> ObservableAction:
    """Reconstruct the checker input exactly from a persisted trace action."""
    agent_role = str(
        event.get("agent_role")
        or event.get("role")
        or event.get("agent_id")
        or "unknown"
    )
    agent_id = str(event.get("agent_id") or agent_role)
    target_role_raw = event.get("target_role")
    target_agent_raw = event.get("target_agent")
    target_role = (
        str(target_role_raw)
        if target_role_raw is not None
        else str(target_agent_raw) if target_agent_raw is not None else None
    )
    target_agent = (
        str(target_agent_raw)
        if target_agent_raw is not None
        else target_role
    )
    return ObservableAction(
        sequence_num=_int(event.get("sequence_num")),
        run_id=run_id,
        surface=str(event.get("type") or ""),
        agent_id=agent_id,
        agent_role=agent_role,
        tool_name=event.get("tool_name"),
        tool_args=deepcopy(event.get("tool_args"))
        if isinstance(event.get("tool_args"), dict)
        else {},
        tool_result=str(event.get("tool_result") or ""),
        target_agent=target_agent,
        target_role=target_role,
        content=str(event.get("content") or ""),
        raw_event=(
            deepcopy(event.get("raw_event"))
            if isinstance(event.get("raw_event"), dict)
            else {}
        ),
    )


def _decision_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _recomputed_decision_record(decision: Any) -> dict[str, Any]:
    rule = getattr(decision, "rule", None)
    return {
        "violated": bool(getattr(decision, "violated", False)),
        "violation_code": _decision_value(getattr(decision, "violation_code", None)),
        "severity": _decision_value(getattr(decision, "severity", None)),
        "rule_description": getattr(rule, "description", None) if rule is not None else None,
    }


def _trace_decision_matches_recomputed(
    trace_event: dict[str, Any],
    recomputed: dict[str, Any],
) -> bool:
    """Match required label fields and any optional fields present in the trace."""
    if trace_event.get("violated") is not recomputed["violated"]:
        return False
    if _decision_value(trace_event.get("violation_code")) != recomputed["violation_code"]:
        return False
    for optional_field in ("severity", "rule_description"):
        if optional_field in trace_event and _decision_value(
            trace_event.get(optional_field)
        ) != _decision_value(recomputed.get(optional_field)):
            return False
    return True


def _trace_decisions_match_recomputed_in_order(
    trace_events: list[dict[str, Any]],
    recomputed: list[dict[str, Any]],
) -> bool:
    return len(trace_events) == len(recomputed) and all(
        _trace_decision_matches_recomputed(trace_event, current_decision)
        for trace_event, current_decision in zip(
            trace_events,
            recomputed,
            strict=True,
        )
    )


def _format_access_decisions(decisions: list[dict[str, Any]]) -> str:
    return json.dumps(decisions, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _audit_source_resource_semantics(
    report: dict[str, Any],
    trace_starts: list[dict[str, Any]],
) -> tuple[str, bool, list[str]]:
    """Resolve one source mode and validate its formal three-way provenance."""

    errors: list[str] = []
    trace_start = trace_starts[0] if len(trace_starts) == 1 else {}
    report_provenance = report.get("actor_provenance")
    trace_provenance = trace_start.get("actor_provenance")
    report_mode = report.get("resource_semantics_mode")
    trace_mode = trace_start.get("resource_semantics_mode")
    provenance_mode = (
        report_provenance.get("resource_semantics_mode")
        if isinstance(report_provenance, dict)
        else None
    )
    supplied_modes = [
        value
        for value in (report_mode, trace_mode, provenance_mode)
        if value is not None and str(value).strip()
    ]
    normalized_supplied_modes: list[str] = []
    for supplied_mode in supplied_modes:
        try:
            normalized_supplied_modes.append(
                normalize_resource_semantics_mode(supplied_mode)
            )
        except ValueError as exc:
            message = str(exc)
            if message not in errors:
                errors.append(message)
    mode = (
        normalized_supplied_modes[0]
        if normalized_supplied_modes
        else EXPLICIT_RESOURCE_SEMANTICS
    )

    requires_three_way = (
        mode == LEGACY_OFFICIAL_RESOURCE_SEMANTICS
        or any(
            str(value or "").strip() == LEGACY_OFFICIAL_RESOURCE_SEMANTICS
            for value in (report_mode, trace_mode, provenance_mode)
        )
    )
    provenance_valid = not requires_three_way
    if requires_three_way:
        if not (
            report_mode
            and report_mode == trace_mode == provenance_mode == mode
        ):
            errors.append(
                "resource semantics mode must agree across report, trace_start, "
                "and actor provenance"
            )
        if report_provenance != trace_provenance:
            errors.append(
                "report and trace_start actor provenance differ for resource semantics"
            )
        provenance_errors = validate_actor_provenance(
            report_provenance,
            formal=True,
        )
        if provenance_errors:
            errors.extend(
                "resource semantics actor provenance: " + str(error)
                for error in provenance_errors
            )
        provenance_valid = not errors
    elif len(set(normalized_supplied_modes)) > 1:
        errors.append("inconsistent resource semantics mode markers")
        provenance_valid = False
    if errors:
        provenance_valid = False

    return mode, provenance_valid, errors


def audit_clean_source_reports(report_paths: list[Path]) -> dict[str, Any]:
    import multi_agent.checker as access_checker

    checker_path = Path(str(access_checker.__file__)).resolve()
    checker_sha256 = _sha256_file(checker_path)
    task_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for supplied_path in report_paths:
        report_path = Path(supplied_path)
        errors: list[str] = []
        report: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        trace_file: Path | None = None
        task_path: Path | None = None
        task_sha256 = ""
        effective_rules_sha256 = ""
        effective_rule_count = 0
        legacy_resource_rule_count = 0
        current_rule_set_formal_valid = False
        recomputed_access_decision_count = 0
        recomputed_violation_count = 0
        access_decisions_match_current_checker = False
        resource_semantics_mode = EXPLICIT_RESOURCE_SEMANTICS
        resource_semantics_provenance_valid = False
        current_task: Any | None = None
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("report root must be a JSON object")
            report = loaded
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid report: {type(exc).__name__}: {exc}")
        if report:
            if not str(report.get("task_id") or ""):
                errors.append("missing task_id")
            if not str(report.get("run_id") or ""):
                errors.append("missing run_id")
            if report.get("error"):
                errors.append(f"runner error: {report.get('error')}")
            if report.get("safepred_offline_replay"):
                errors.append("source report is already an offline SafePred replay")
            try:
                trace_file = _resolve_trace_file(report.get("trace_file"), report_path)
                events = _load_jsonl_strict(trace_file)
            except Exception as exc:
                errors.append(f"invalid trace_file: {type(exc).__name__}: {exc}")

        if report.get("formal_trace") is True and events:
            run_id = str(report.get("run_id") or "")
            mismatched_event_indices = [
                index
                for index, event in enumerate(events, start=1)
                if str(event.get("trace_id") or "") != run_id
            ]
            if mismatched_event_indices:
                errors.append(
                    "formal trace event trace_id mismatch at event(s): "
                    + ",".join(
                        str(index)
                        for index in mismatched_event_indices[:12]
                    )
                )
            if events[0].get("type") != "trace_start":
                errors.append("formal trace_start must be the first event")
            if events[-1].get("type") != "trace_end":
                errors.append("formal trace_end must be the last event")

        actions = [
            event
            for event in events
            if event.get("type") in {"tool_call", "communication"}
        ]
        sequences = [_int(event.get("sequence_num")) for event in actions]
        if not actions:
            errors.append("trace contains no replayable actions")
        if sequences and sequences != sorted(sequences):
            errors.append("action sequence numbers are not monotonic")
        if len(sequences) != len(set(sequences)):
            errors.append("action sequence numbers are not unique")
        if sequences and sequences != list(range(1, len(sequences) + 1)):
            errors.append("action sequence numbers must be contiguous and 1-based")

        trace_starts = [event for event in events if event.get("type") == "trace_start"]
        trace_ends = [event for event in events if event.get("type") == "trace_end"]
        if len(trace_starts) != 1:
            errors.append(f"trace must contain exactly one trace_start (found {len(trace_starts)})")
        if len(trace_ends) != 1:
            errors.append(f"trace must contain exactly one trace_end (found {len(trace_ends)})")
        elif trace_ends[0].get("error"):
            errors.append(f"trace_end reports runner error: {trace_ends[0].get('error')}")
        if trace_ends:
            declared_actions = _int(trace_ends[-1].get("n_actions"))
            if declared_actions != len(actions):
                errors.append(
                    f"trace_end n_actions={declared_actions} does not match parsed actions={len(actions)}"
                )
        if report and _int(report.get("n_actions")) != len(actions):
            errors.append(
                f"report n_actions={_int(report.get('n_actions'))} does not match parsed actions={len(actions)}"
            )
        if report:
            (
                resource_semantics_mode,
                resource_semantics_provenance_valid,
                resource_semantics_errors,
            ) = _audit_source_resource_semantics(report, trace_starts)
            errors.extend(resource_semantics_errors)

        access_events = [event for event in events if event.get("type") == "access_decision"]
        access_sequences = {_int(event.get("action_sequence_num")) for event in access_events}
        missing_labels = sorted(set(sequences) - access_sequences)
        unknown_labels = sorted(access_sequences - set(sequences))
        if missing_labels:
            errors.append(
                "missing access_decision labels for action sequence(s): "
                + ",".join(str(item) for item in missing_labels[:12])
            )
        if unknown_labels:
            errors.append(
                "access_decision labels reference unknown action sequence(s): "
                + ",".join(str(item) for item in unknown_labels[:12])
            )
        malformed_labels = [
            event
            for event in access_events
            if not isinstance(event.get("violated"), bool)
        ]
        if malformed_labels:
            errors.append(
                f"{len(malformed_labels)} access_decision label(s) lack a JSON-boolean violated field"
            )
        missing_violation_codes = [
            event
            for event in access_events
            if event.get("violated") is True
            and str(event.get("violation_code") or "") not in {"V-ID", "V-OT", "V-OR", "V-IC"}
        ]
        if missing_violation_codes:
            errors.append(
                f"{len(missing_violation_codes)} violated access_decision label(s) lack a valid violation_code"
            )
        trace_violation_count = sum(event.get("violated") is True for event in access_events)
        if report and _int(report.get("n_violations")) != trace_violation_count:
            errors.append(
                f"report n_violations={_int(report.get('n_violations'))} does not match "
                f"violated access_decisions={trace_violation_count}"
            )

        task_id = str(report.get("task_id") or "")
        if task_id:
            if task_id not in task_cache:
                try:
                    resolved = resolve_task_path(f"{task_id}.yaml")
                    loaded_task, _ = load_task_with_tools(resolved)
                    effective_rules = [
                        rule.model_dump(mode="json")
                        for rule in loaded_task.access_rules
                    ]
                    legacy_resource_rules = [
                        rule
                        for rule in loaded_task.access_rules
                        if rule.access_type == "operation.resource"
                        and rule.resource_effect is None
                    ]
                    task_cache[task_id] = {
                        "task": loaded_task,
                        "task_path": resolved.resolve(),
                        "task_sha256": _sha256_file(resolved),
                        "effective_rules_sha256": _stable_json_sha256(effective_rules),
                        "effective_rule_count": len(effective_rules),
                        "legacy_resource_rules": [
                            {
                                "role": rule.role,
                                "tool": rule.match.tool,
                                "description": rule.description,
                            }
                            for rule in legacy_resource_rules
                        ],
                    }
                except Exception as exc:
                    task_cache[task_id] = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            cached_task = task_cache[task_id]
            if cached_task.get("error"):
                errors.append(
                    "unable to load current task/rules for access GT recomputation: "
                    + str(cached_task["error"])
                )
            else:
                current_task = cached_task["task"]
                task_path = cached_task["task_path"]
                task_sha256 = str(cached_task["task_sha256"])
                effective_rules_sha256 = str(cached_task["effective_rules_sha256"])
                effective_rule_count = int(cached_task["effective_rule_count"])
                legacy_resource_rules = list(cached_task["legacy_resource_rules"])
                legacy_resource_rule_count = len(legacy_resource_rules)
                current_rule_set_formal_valid = (
                    resource_semantics_mode
                    == LEGACY_OFFICIAL_RESOURCE_SEMANTICS
                    or legacy_resource_rule_count == 0
                )
                if (
                    legacy_resource_rules
                    and resource_semantics_mode == EXPLICIT_RESOURCE_SEMANTICS
                ):
                    first_legacy = legacy_resource_rules[0]
                    errors.append(
                        "current effective task rules contain "
                        f"{legacy_resource_rule_count} legacy operation.resource "
                        "rule(s) lacking explicit resource_effect; first="
                        f"role={first_legacy.get('role')!r}, "
                        f"tool={first_legacy.get('tool')!r}"
                    )
                if str(current_task.task_id) != task_id:
                    errors.append(
                        f"resolved task_id={current_task.task_id!r} does not match "
                        f"report task_id={task_id!r}"
                    )

        if trace_starts and task_id:
            trace_task_id = str(trace_starts[0].get("task_id") or "")
            if trace_task_id and trace_task_id != task_id:
                errors.append(
                    f"trace_start task_id={trace_task_id!r} does not match "
                    f"report task_id={task_id!r}"
                )

        if current_task is not None and actions:
            trace_decisions_by_sequence: dict[int, list[dict[str, Any]]] = {}
            for event in access_events:
                trace_decisions_by_sequence.setdefault(
                    _int(event.get("action_sequence_num")), []
                ).append(event)
            checker_recomputation_failed = False
            decision_mismatch_found = False
            for event in actions:
                sequence_num = _int(event.get("sequence_num"))
                try:
                    action = _trace_event_to_observable_action(
                        event,
                        run_id=str(report.get("run_id") or ""),
                    )
                    current_decisions = [
                        _recomputed_decision_record(decision)
                        for decision in access_checker.check_action(
                            action,
                            current_task.access_rules,
                            str(current_task.domain or ""),
                            resource_semantics_mode=resource_semantics_mode,
                        )
                    ]
                except Exception as exc:
                    checker_recomputation_failed = True
                    errors.append(
                        "current checker access GT recomputation failed for "
                        f"action sequence {sequence_num}: {type(exc).__name__}: {exc}"
                    )
                    continue
                recomputed_access_decision_count += len(current_decisions)
                recomputed_violation_count += sum(
                    decision["violated"] for decision in current_decisions
                )
                trace_decisions = trace_decisions_by_sequence.get(sequence_num, [])
                if not _trace_decisions_match_recomputed_in_order(
                    trace_decisions,
                    current_decisions,
                ):
                    decision_mismatch_found = True
                    errors.append(
                        "persisted access_decision labels do not match current "
                        f"task+checker GT for action sequence {sequence_num}: "
                        f"trace={_format_access_decisions(trace_decisions)}, "
                        f"recomputed={_format_access_decisions(current_decisions)}"
                    )
            if not checker_recomputation_failed:
                if recomputed_violation_count != trace_violation_count:
                    decision_mismatch_found = True
                    errors.append(
                        f"current task+checker recomputed violations="
                        f"{recomputed_violation_count} does not match persisted "
                        f"violated access_decisions={trace_violation_count}"
                    )
                access_decisions_match_current_checker = (
                    not decision_mismatch_found and current_rule_set_formal_valid
                )

        guarded_prefix_events = []
        for event in actions:
            raw_event = event.get("raw_event") or {}
            if not isinstance(raw_event, dict):
                continue
            if raw_event.get("safepred_guard") or str(
                raw_event.get("capture_mode") or ""
            ).startswith("safepred_"):
                guarded_prefix_events.append(_int(event.get("sequence_num")))
        if guarded_prefix_events:
            errors.append(
                "source trace contains prior SafePred intervention metadata at sequence(s): "
                + ",".join(str(item) for item in guarded_prefix_events[:12])
            )

        actor_provenance = report.get("actor_provenance")
        if not isinstance(actor_provenance, dict):
            actor_provenance = {}
        actor_hashes = actor_provenance.get("hashes")
        if not isinstance(actor_hashes, dict):
            actor_hashes = {}
        rows.append(
            {
                "report_path": str(report_path.resolve()),
                "trace_file": str(trace_file.resolve()) if trace_file and trace_file.is_file() else "",
                "task_id": str(report.get("task_id") or ""),
                "run_id": str(report.get("run_id") or ""),
                "action_count": len(actions),
                "access_labeled_action_count": len(set(sequences) & access_sequences),
                "recomputed_access_decision_count": recomputed_access_decision_count,
                "recomputed_violation_count": recomputed_violation_count,
                "access_decisions_match_current_checker": access_decisions_match_current_checker,
                "resource_semantics_mode": resource_semantics_mode,
                "resource_semantics_provenance_valid": (
                    resource_semantics_provenance_valid
                ),
                "actor_protocol_profile": str(
                    actor_provenance.get("actor_protocol_profile") or ""
                ),
                "actor_model": str(
                    actor_provenance.get("model") or ""
                ),
                "actor_model_revision": str(
                    actor_provenance.get("model_revision") or ""
                ),
                "actor_endpoint": str(
                    actor_provenance.get("endpoint") or ""
                ),
                "actor_runtime_versions_sha256": _stable_json_sha256(
                    actor_provenance.get("runtime_versions") or {}
                ),
                "actor_implementation_manifest_sha256": str(
                    actor_hashes.get("implementation_manifest_sha256") or ""
                ),
                "source_report_sha256": (
                    _sha256_file(report_path) if report_path.is_file() else ""
                ),
                "source_trace_sha256": (
                    _sha256_file(trace_file)
                    if trace_file is not None and trace_file.is_file()
                    else ""
                ),
                "task_file": str(task_path) if task_path is not None else "",
                "task_sha256": task_sha256,
                "effective_rule_count": effective_rule_count,
                "legacy_resource_rule_count": legacy_resource_rule_count,
                "current_rule_set_formal_valid": current_rule_set_formal_valid,
                "effective_rules_sha256": effective_rules_sha256,
                "checker_file": str(checker_path),
                "checker_sha256": checker_sha256,
                "is_clean_replayable": not errors,
                "errors": errors,
            }
        )
    resource_semantics_modes = sorted(
        {
            str(row.get("resource_semantics_mode") or EXPLICIT_RESOURCE_SEMANTICS)
            for row in rows
        }
    )
    resource_semantics_uniform = len(resource_semantics_modes) <= 1
    if not resource_semantics_uniform:
        cohort_error = (
            "mixed resource semantics cohort is forbidden: "
            + ",".join(resource_semantics_modes)
        )
        for row in rows:
            row["errors"].append(cohort_error)
            row["is_clean_replayable"] = False
    cohort_uniformity: dict[str, bool] = {}
    cohort_values: dict[str, list[str]] = {}
    for field in (
        "actor_protocol_profile",
        "actor_model",
        "actor_model_revision",
        "actor_endpoint",
        "actor_runtime_versions_sha256",
        "actor_implementation_manifest_sha256",
    ):
        values = sorted(
            {str(row.get(field) or "") for row in rows}
        )
        cohort_values[field] = values
        cohort_uniformity[field] = len(values) <= 1
        if len(values) > 1:
            cohort_error = (
                f"mixed {field} cohort is forbidden: "
                + ",".join(values)
            )
            for row in rows:
                row["errors"].append(cohort_error)
                row["is_clean_replayable"] = False
    return {
        "artifact_type": "harnessaudit_clean_source_audit",
        "report_count": len(rows),
        "valid_report_count": sum(1 for row in rows if row["is_clean_replayable"]),
        "invalid_report_count": sum(1 for row in rows if not row["is_clean_replayable"]),
        "access_gt_recomputation_required": True,
        "access_gt_recomputation_schema_version": 1,
        "resource_semantics_modes": resource_semantics_modes,
        "resource_semantics_uniform": resource_semantics_uniform,
        "actor_cohort_values": cohort_values,
        "actor_cohort_uniformity": cohort_uniformity,
        "actor_cohort_uniform": all(cohort_uniformity.values()),
        "checker_file": str(checker_path),
        "checker_sha256": checker_sha256,
        "reports": rows,
    }


def write_clean_source_audit(
    *,
    report_paths: list[Path],
    analysis_dir: Path,
) -> dict[str, Path]:
    audit = audit_clean_source_reports(report_paths)
    json_path = analysis_dir / "predict_only_source_report_audit.json"
    csv_path = analysis_dir / "predict_only_source_report_audit.csv"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(
        csv_path,
        [
            {
                **row,
                "errors": " | ".join(row.get("errors") or []),
            }
            for row in audit["reports"]
        ],
    )
    if audit["invalid_report_count"]:
        first = next(row for row in audit["reports"] if not row["is_clean_replayable"])
        raise ValueError(
            "HarnessAudit source report audit failed: "
            f"{first['report_path']}: {'; '.join(first['errors'])}. "
            f"See {json_path}"
        )
    return {"source_audit_json": json_path, "source_audit_csv": csv_path}


def build_prediction_quality_report(point_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(point_rows)
    decision_error_rows = [
        row for row in point_rows if _truthy(row.get("decision_error"))
    ]
    skipped_rows = [
        row for row in point_rows if _truthy(row.get("skipped_by_prediction_interval"))
    ]
    invalid_rows = [
        row
        for row in point_rows
        if not _truthy(row.get("valid_prediction"))
        and not _truthy(row.get("decision_error"))
        and not _truthy(row.get("skipped_by_prediction_interval"))
    ]
    valid_rows = [
        row
        for row in point_rows
        if _truthy(row.get("valid_prediction"))
        and not _truthy(row.get("decision_error"))
        and not _truthy(row.get("skipped_by_prediction_interval"))
    ]
    error_tasks = sorted({str(row.get("task_id") or "") for row in decision_error_rows})
    valid_tasks = sorted({str(row.get("task_id") or "") for row in valid_rows})
    by_policy_horizon = Counter(
        (
            str(row.get("rollout_mode") or "wm_free_rollout"),
            str(row.get("policy_mode") or ""),
            str(row.get("horizon") or ""),
            "error" if _truthy(row.get("decision_error")) else "valid" if _truthy(row.get("valid_prediction")) else "invalid",
        )
        for row in point_rows
    )
    return {
        "artifact_type": "predict_only_quality_report",
        "target_estimand": _single_value(point_rows, "gt_estimand"),
        "horizon_cohort": _single_value(point_rows, "horizon_cohort"),
        "total_rows": total_rows,
        "valid_prediction_rows": len(valid_rows),
        "decision_error_rows": len(decision_error_rows),
        "invalid_prediction_rows": len(invalid_rows),
        "skipped_by_prediction_interval_rows": len(skipped_rows),
        "valid_task_count": len(valid_tasks),
        "error_task_count": len(error_tasks),
        "valid_tasks": valid_tasks,
        "error_tasks": error_tasks,
        "by_policy_horizon": [
            {
                "rollout_mode": rollout,
                "policy_mode": policy,
                "horizon": horizon,
                "status": status,
                "rows": count,
            }
            for (rollout, policy, horizon, status), count in sorted(by_policy_horizon.items())
        ],
        "complete": (
            bool(total_rows)
            and len(valid_rows) == total_rows
            and not decision_error_rows
            and not skipped_rows
            and not invalid_rows
        ),
        "warning": (
            ""
            if len(valid_rows) == total_rows and not decision_error_rows and not skipped_rows
            else "Some SafePred rows failed, were skipped, lacked ground truth, or were otherwise invalid. "
            "Figures only reflect valid rows and must not be treated as a complete run."
        ),
    }


def _single_value(rows: Iterable[dict[str, Any]], key: str) -> str:
    values = {
        str(row.get(key) or "").strip().lower()
        for row in rows
        if str(row.get(key) or "").strip()
    }
    if len(values) == 1:
        return next(iter(values))
    return "mixed" if values else "unknown"


def _harnessaudit_offline_task_ids(
    tasks: Iterable[str],
    source_reports: Iterable[Path],
) -> list[str]:
    """Resolve the task set without running an agent or contacting a model."""
    task_ids = {
        Path(str(task)).stem
        for task in tasks
        if str(task).strip()
    }
    for raw_report_path in source_reports:
        report_path = Path(raw_report_path).expanduser().resolve()
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read HarnessAudit source report for formal task gate: {report_path}: {exc}"
            ) from exc
        task_id = str(report.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(
                f"HarnessAudit source report lacks task_id for formal task gate: {report_path}"
            )
        task_ids.add(Path(task_id).stem)
    return sorted(task_ids)


def validate_formal_ha_task_approval(
    task_ids: Iterable[str],
    *,
    manifest_path: Path = FORMAL_HA_TASK_APPROVAL_PATH,
) -> dict[str, Any]:
    """Bind confirmatory HA runs to the manually reviewed task set and bytes."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read HA formal task approval manifest {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("HA formal task approval manifest schema_version must equal 1")
    if payload.get("resource_semantics") != "explicit_effect_v1":
        raise ValueError(
            "HA formal task approval manifest must require explicit_effect_v1"
        )
    approved = payload.get("tasks")
    if not isinstance(approved, dict) or not approved:
        raise ValueError("HA formal task approval manifest contains no approved tasks")

    observed = sorted({Path(str(item)).stem for item in task_ids if str(item).strip()})
    unapproved = sorted(set(observed) - set(approved))
    if unapproved:
        raise ValueError(
            "Formal HarnessAudit predict-only task(s) lack manual approval: "
            + ", ".join(unapproved)
        )

    task_root = Path(__file__).resolve().parents[1] / "tasks"
    checked: list[dict[str, str]] = []
    for task_id in observed:
        row = approved.get(task_id)
        if not isinstance(row, dict):
            raise ValueError(f"Malformed HA approval entry for {task_id}")
        relative_path = str(row.get("relative_path") or "")
        expected_sha = str(row.get("sha256") or "").lower()
        task_path = (task_root / relative_path).resolve()
        try:
            task_path.relative_to(task_root.resolve())
        except ValueError as exc:
            raise ValueError(f"HA approval path escapes task root: {relative_path}") from exc
        if not task_path.is_file():
            raise ValueError(f"Approved HA task file is missing: {task_path}")
        if task_path.stem != task_id:
            raise ValueError(
                f"HA approval task/path mismatch: task={task_id}, path={relative_path}"
            )
        actual_sha = _sha256_file(task_path)
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            raise ValueError(
                f"Approved HA task bytes changed for {task_id}: "
                f"expected={expected_sha}, observed={actual_sha}"
            )
        checked.append(
            {
                "task_id": task_id,
                "task_path": str(task_path),
                "task_sha256": actual_sha,
            }
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "task_count": len(checked),
        "tasks": checked,
    }


def _ha_source_input_manifest_digest(report_paths: Iterable[Path]) -> str:
    """Bind the exact frozen HA reports and traces with portable logical names."""

    files: dict[str, Path] = {}
    identities: set[tuple[str, str]] = set()
    records: list[tuple[str, str, Path, Path]] = []
    for raw_report_path in report_paths:
        report_path = Path(raw_report_path).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        task_id = Path(str(report.get("task_id") or "")).stem
        run_id = str(report.get("run_id") or report_path.stem).strip()
        if not task_id or not run_id:
            raise ValueError(f"HA source report lacks a stable task/run identity: {report_path}")
        identity = (task_id, run_id)
        if identity in identities:
            raise ValueError(
                "HA source manifest contains duplicate task/run identity: "
                f"task={task_id}, run={run_id}"
            )
        identities.add(identity)
        trace_path = _resolve_trace_file(report.get("trace_file"), report_path).resolve()
        records.append((task_id, run_id, report_path, trace_path))
    for task_id, run_id, report_path, trace_path in sorted(
        records, key=lambda item: (item[0], item[1])
    ):
        logical_root = f"{task_id}/{run_id}"
        files[f"{logical_root}/report.json"] = report_path
        files[f"{logical_root}/trace.jsonl"] = trace_path
    return source_input_manifest_sha256(files)


def _validate_ha_actor_seed_provenance(
    report_paths: Iterable[Path],
    *,
    expected_actor_trace_seed: int | None = None,
) -> ActorReplicateIdentity:
    """Compatibility wrapper around the authoritative source identity reader."""

    return read_ha_source_replicate_identity(
        report_paths,
        expected_actor_trace_seed=expected_actor_trace_seed,
    )


def _source_report_actor_identity(
    report: dict[str, Any],
    *,
    legacy_fallback_seed: int,
    expected_actor_protocol_profile: str | None = None,
    expected_actor_replicate_id: str | None = None,
    expected_actor_trace_seed: int | None = None,
) -> ActorReplicateIdentity:
    """Resolve source identity without inventing a seed for expanded reports."""

    has_profile = "actor_protocol_profile" in report
    has_replicate = "actor_replicate_id" in report
    has_seed = "actor_trace_seed" in report
    if has_profile or has_replicate:
        if not (has_profile and has_replicate and has_seed):
            raise ValueError(
                "Expanded HA source identity requires explicit "
                "actor_protocol_profile, actor_replicate_id, and "
                "actor_trace_seed fields"
            )
        identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_protocol_profile=report["actor_protocol_profile"],
            actor_replicate_id=report["actor_replicate_id"],
            actor_trace_seed=report["actor_trace_seed"],
        )
    elif has_seed:
        identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_trace_seed=report["actor_trace_seed"],
        )
    elif (
        expected_actor_protocol_profile is not None
        or expected_actor_replicate_id is not None
    ):
        identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_protocol_profile=expected_actor_protocol_profile,
            actor_replicate_id=expected_actor_replicate_id,
            actor_trace_seed=expected_actor_trace_seed,
        )
    else:
        identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_trace_seed=(
                legacy_fallback_seed
                if expected_actor_trace_seed is None
                else expected_actor_trace_seed
            ),
        )

    if has_profile or has_replicate or has_seed:
        if (
            expected_actor_protocol_profile is not None
            or expected_actor_replicate_id is not None
        ):
            asserted = normalize_actor_replicate_identity(
                benchmark="harnessaudit",
                actor_protocol_profile=expected_actor_protocol_profile,
                actor_replicate_id=expected_actor_replicate_id,
                actor_trace_seed=expected_actor_trace_seed,
            )
            if asserted != identity:
                raise ValueError(
                    "Caller actor identity assertion differs from authoritative "
                    f"source report: expected={asserted.full_key}, "
                    f"observed={identity.full_key}"
                )
        elif (
            expected_actor_trace_seed is not None
            and identity.actor_trace_seed != expected_actor_trace_seed
        ):
            raise ValueError(
                "Caller actor_trace_seed assertion differs from authoritative "
                f"source report: expected={expected_actor_trace_seed}, "
                f"observed={identity.actor_trace_seed}"
            )
    return identity


def _ha_source_action_trajectory_digests(
    report_paths: Iterable[Path],
) -> dict[str, str]:
    """Hash only the ordered actor actions, excluding seed/label metadata."""

    digests: dict[str, str] = {}
    for raw_report_path in report_paths:
        report_path = Path(raw_report_path).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        task_id = str(report.get("task_id") or "")
        if task_id in digests:
            raise ValueError(
                f"Formal HA source set contains multiple reports for task {task_id!r}"
            )
        trace_path = _resolve_trace_file(report.get("trace_file"), report_path)
        steps = build_offline_prediction_steps(_load_jsonl(trace_path))
        canonical_actions = [
            {
                "action_sequence_num": int(step.action_sequence_num),
                "proposed_action": step.proposed_action,
            }
            for step in steps
        ]
        digests[task_id] = hashlib.sha256(
            json.dumps(
                canonical_actions,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return dict(sorted(digests.items()))


def _ha_observed_formal_protocol(
    *,
    point_config: PointSelectionConfig,
    rollout_modes: Iterable[str],
    model: str,
    model_api: ModelAPIConfig,
    base_config: Path,
    max_trace_events: int,
    task_approval_sha256: str,
    actor_trace_seed: int | None,
    source_manifest_sha256: str,
    source_inputs_contract: dict[str, Any],
    actor_protocol_profile: str | None = None,
    actor_replicate_id: str | None = None,
) -> dict[str, Any]:
    model_revision = str(model_api.model_revision or "").strip()
    if not model_revision:
        raise ValueError(
            "Formal HA protocol requires --model-revision naming an immutable "
            "weights or deployment revision"
        )
    if not str(model_api.api_url or "").strip():
        raise ValueError(
            "Formal HA protocol requires an explicit --api-url so the frozen "
            "endpoint and replay cache cannot diverge from provider environment variables"
        )
    try:
        base_payload = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
        risk_threshold = float((base_payload.get("tree_search") or {})["root_risk_threshold"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Cannot resolve root_risk_threshold from HA base config {base_config}: {exc}"
        ) from exc
    normalized_rollouts = {
        str(value).strip() for value in rollout_modes if str(value).strip()
    }
    identity = normalize_actor_replicate_identity(
        benchmark="harnessaudit",
        actor_protocol_profile=actor_protocol_profile,
        actor_replicate_id=actor_replicate_id,
        actor_trace_seed=actor_trace_seed,
    )
    if (
        identity.actor_protocol_profile
        == "deterministic_close_source_v1"
    ):
        source_map = source_inputs_contract.get(
            "manifest_sha256_by_actor_replicate_id"
        )
        source_key = identity.actor_replicate_id
    elif identity.actor_protocol_profile == "seeded_sampling_v1":
        source_map = source_inputs_contract.get(
            "manifest_sha256_by_actor_trace_seed"
        )
        source_key = str(identity.actor_trace_seed)
    else:  # normalize_actor_replicate_identity is fail-closed; keep this explicit.
        raise ValueError(
            "Unsupported formal HA actor protocol profile: "
            f"{identity.actor_protocol_profile!r}"
        )
    expected_source_digest = (
        source_map.get(source_key) if isinstance(source_map, dict) else None
    )
    if str(expected_source_digest or "").lower() != str(source_manifest_sha256).lower():
        raise ValueError(
            "Formal HA source manifest does not match the frozen actor identity "
            f"entry: actor_identity={identity.full_key}"
        )
    execution_semantics: dict[str, str] = {}
    prompt_versions: dict[str, str] = {}
    if DIRECT_MULTI_HORIZON_ROLLOUT in normalized_rollouts:
        execution_semantics[DIRECT_MULTI_HORIZON_ROLLOUT] = "single_horizon_per_call"
        prompt_versions[DIRECT_MULTI_HORIZON_ROLLOUT] = (
            HA_DIRECT_PROMPT_VERSION
        )
    if ITERATIVE_WM_ROLLOUT in normalized_rollouts:
        execution_semantics[ITERATIVE_WM_ROLLOUT] = "single_hmax_nested_rollout"
        prompt_versions[ITERATIVE_WM_ROLLOUT] = (
            HA_ITERATIVE_PROMPT_VERSION
        )
    if set(execution_semantics) != normalized_rollouts:
        unsupported = sorted(normalized_rollouts - set(execution_semantics))
        raise ValueError(
            "Formal HA protocol has no frozen execution semantic for rollout mode(s): "
            f"{unsupported}"
        )
    return {
        "point_selection": {
            "scheme": point_config.scheme,
            "count": int(point_config.count),
            "interval": int(point_config.interval),
            "explicit_sequences": [int(value) for value in point_config.explicit_sequences],
        },
        "rollout_execution_semantics": execution_semantics,
        "model": {
            "provider": str(model_api.provider),
            "model": str(model_api.model or model),
            "revision": model_revision,
            "api_base_url": _resolved_model_api_url(model_api),
            "input_mode": "trace_text",
            "response_format_json_object": True,
            "prompt_versions": prompt_versions,
            "risk_threshold": risk_threshold,
        },
        "experience_replay": {"enable": False},
        "train_policy": {
            "construction_version": HA_TRAIN_POLICY_CONSTRUCTION_VERSION,
            "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
            "policy_extraction_prompt_version": HA_POLICY_EXTRACTION_PROMPT_VERSION,
            "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
            "test_visible_fields": list(HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS),
            "input_includes_test_access_rules": False,
            "input_includes_test_tool_necessity": False,
            "input_includes_tool_necessity_derivatives": False,
            "include_test_tool_necessity_policies": False,
            "policy_extraction_model": str(
                model_api.policy_extraction_model or model_api.model or model
            ),
            "policy_extraction_seed": int(
                os.environ.get("SAFEPRED_POLICY_EXTRACTION_SEED", "0")
            ),
            "policy_inputs_manifest_sha256": _ha_policy_input_manifest_digest(),
        },
        "base_config_sha256": _sha256_file(Path(base_config)),
        "implementation_manifest_sha256": _ha_implementation_manifest_digest(),
        "max_trace_events": int(max_trace_events),
        "approved_task_manifest_sha256": str(task_approval_sha256),
        "source_inputs": deepcopy(source_inputs_contract),
    }


def _ha_label_support_census_rows(
    *,
    report_points: Iterable[tuple[str, Path, list[SparsePoint]]],
    horizons: Iterable[int],
) -> list[dict[str, Any]]:
    """Count exact-step labels before any WM call, clustered by source task."""

    horizon_values = sorted({int(value) for value in horizons})
    support: dict[int, dict[str, Any]] = {
        horizon: {
            "horizon": horizon,
            "selected_prediction_points": 0,
            "positive_prediction_points": 0,
            "negative_prediction_points": 0,
            "positive_tasks": set(),
            "negative_tasks": set(),
        }
        for horizon in horizon_values
    }
    for task_id, trace_file, points in report_points:
        gt_details = _trace_access_decision_details(trace_file)
        for point in points:
            for horizon in horizon_values:
                labels = _ground_truth_labels(
                    gt_details,
                    action_sequence_num=point.step.action_sequence_num,
                    horizon=horizon,
                )
                if not labels["exact_step_available"]:
                    raise ValueError(
                        "HA exact-step GT is unavailable in the common cohort: "
                        f"task={task_id}, sequence={point.step.action_sequence_num}, h={horizon}"
                    )
                row = support[horizon]
                row["selected_prediction_points"] += 1
                if labels["exact_step_violated"]:
                    row["positive_prediction_points"] += 1
                    row["positive_tasks"].add(task_id)
                else:
                    row["negative_prediction_points"] += 1
                    row["negative_tasks"].add(task_id)
    return [
        {
            "target_estimand": "exact_step",
            "horizon_cohort": "common_max_horizon_support",
            "horizon": horizon,
            "selected_prediction_points": row["selected_prediction_points"],
            "positive_prediction_points": row["positive_prediction_points"],
            "negative_prediction_points": row["negative_prediction_points"],
            "positive_task_cluster_count": len(row["positive_tasks"]),
            "negative_task_cluster_count": len(row["negative_tasks"]),
            "positive_tasks": ";".join(sorted(row["positive_tasks"])),
            "negative_tasks": ";".join(sorted(row["negative_tasks"])),
        }
        for horizon, row in sorted(support.items())
    ]


def _validate_ha_formal_label_support(rows: Iterable[dict[str, Any]]) -> None:
    failures = [
        row
        for row in rows
        if int(row.get("positive_task_cluster_count") or 0) < 2
        or int(row.get("negative_task_cluster_count") or 0) < 2
    ]
    if failures:
        details = ", ".join(
            "h{h}: positive_tasks={p}, negative_tasks={n}".format(
                h=row["horizon"],
                p=row["positive_task_cluster_count"],
                n=row["negative_task_cluster_count"],
            )
            for row in failures
        )
        raise ValueError(
            "Formal HA label-only preflight lacks at least two positive and two "
            f"negative task clusters at every horizon ({details}). Change the "
            "frozen task/point cohort before calling the model."
        )


def run_pipeline(
    *,
    tasks: list[str],
    source_reports: list[Path],
    policy_modes: list[str],
    rollout_modes: list[str],
    horizons: list[int],
    point_config: PointSelectionConfig,
    model: str,
    judge_model: str,
    max_turns: int,
    skip_clean_run: bool,
    skip_judge: bool,
    base_config: Path,
    paths: ArtifactPaths,
    candidate_count: int,
    max_trace_events: int,
    skip_existing: bool,
    run_tier3_judge: bool,
    tier3_model: str,
    tier3_api_key: str | None = None,
    tier3_api_url: str | None = None,
    tier3_disable_response_format: bool = False,
    tier3_force_rescore: bool = False,
    model_api: ModelAPIConfig | None = None,
    clean_agent_api: CleanAgentAPIConfig | None = None,
    fail_on_prediction_errors: bool = True,
    fail_fast_decision_errors: bool = False,
    workers: int = 1,
    gt_estimand: str = "exact_step",
    experiment_seed: int = 0,
    actor_trace_seed: int | None = None,
    wm_seed: int | None = None,
    formal: bool = False,
    preregistration_path: Path | None = None,
    compositionality_manifest_path: Path | None = None,
    faithfulness_adjudication_protocol_path: Path | None = None,
    adjudicator_blinding_protocol_path: Path | None = None,
) -> dict[str, Path]:
    _load_project_env()
    if formal and wm_seed is None:
        raise ValueError(
            "Formal HA offline prediction requires explicit wm_seed; "
            "actor identity is read from source reports and --actor-trace-seed "
            "is only a backwards-compatible assertion."
        )
    resolved_wm_seed = int(experiment_seed if wm_seed is None else wm_seed)
    if formal:
        resolved_actor_protocol_profile = ""
        resolved_actor_replicate_id = ""
        resolved_actor_trace_seed = None
    else:
        initial_identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_trace_seed=(
                experiment_seed if actor_trace_seed is None else actor_trace_seed
            ),
        )
        resolved_actor_protocol_profile = initial_identity.actor_protocol_profile
        resolved_actor_replicate_id = initial_identity.actor_replicate_id
        resolved_actor_trace_seed = initial_identity.actor_trace_seed
    os.environ["SAFEPRED_WM_SEED"] = str(resolved_wm_seed)
    os.environ.setdefault("SAFEPRED_POLICY_EXTRACTION_SEED", "0")
    model_api = model_api or ModelAPIConfig(provider="custom", model=model)
    if not model_api.model:
        model_api = replace(model_api, model=model)
    normalized_policy_modes = [
        str(mode).strip() for mode in policy_modes if str(mode).strip()
    ]
    if not normalized_policy_modes:
        normalized_policy_modes = list(DEFAULT_POLICY_MODES)
    if len(normalized_policy_modes) != len(set(normalized_policy_modes)):
        raise ValueError(
            f"Duplicate policy modes are not allowed: {normalized_policy_modes}"
        )
    unsupported_policies = sorted(
        set(normalized_policy_modes) - set(DEFAULT_POLICY_MODES)
    )
    if unsupported_policies:
        raise ValueError(
            f"Unsupported policy mode(s): {unsupported_policies}. "
            f"Use: {sorted(DEFAULT_POLICY_MODES)}"
        )
    policy_modes = normalized_policy_modes
    normalized_rollout_modes = [str(mode).strip() for mode in rollout_modes if str(mode).strip()]
    if len(normalized_rollout_modes) != len(set(normalized_rollout_modes)):
        raise ValueError(
            f"Duplicate rollout modes are not allowed: {normalized_rollout_modes}"
        )
    unsupported = sorted(set(normalized_rollout_modes) - VALID_ROLLOUT_MODES)
    if unsupported:
        raise ValueError(f"Unsupported rollout mode(s): {unsupported}. Use: {sorted(VALID_ROLLOUT_MODES)}")
    if not normalized_rollout_modes:
        normalized_rollout_modes = list(DEFAULT_ROLLOUT_MODES)
    requested_horizons = [int(value) for value in horizons]
    if len(requested_horizons) != len(set(requested_horizons)):
        raise ValueError(
            f"Duplicate prediction horizons are not allowed: {requested_horizons}"
        )
    normalized_horizons = sorted(requested_horizons)
    if gt_estimand != "exact_step":
        raise ValueError(
            "New HarnessAudit predict-only runs support only exact_step. "
            "within_window is a legacy analysis estimand and is not aligned with "
            "single-horizon direct predictions."
        )
    if not normalized_horizons or normalized_horizons[0] < 1:
        raise ValueError("prediction horizons must be positive integers")
    # Iterative replay generates one nested rollout through max(horizons), then
    # analysis retains only the preregistered horizons.  The requested horizon
    # set therefore need not be contiguous (for example [1, 2, 3, 5]).
    horizons = normalized_horizons
    offline_contract_gate: dict[str, Any] = {"status": "not_requested"}
    preregistration: dict[str, Any] | None = None
    annotation_contract_files: dict[str, Path] = {}
    frozen_preregistration_path: Path | None = None
    formal_replicate_identity_path: Path | None = None
    formal_task_approval: dict[str, Any] | None = None
    if preregistration_path is not None and not formal:
        raise ValueError("--preregistration requires --formal for offline prediction")
    if formal:
        if preregistration_path is None:
            raise ValueError("Formal offline prediction requires --preregistration")
        if tasks:
            raise ValueError(
                "Formal HA offline prediction requires pre-generated --source-reports and "
                "does not generate actor traces in the same run. Generate and audit the "
                "source traces first, freeze their manifest, then run prediction-only."
            )
        if not source_reports:
            raise ValueError(
                "Formal HA offline prediction requires at least one frozen --source-reports input"
            )
        formal_task_approval = validate_formal_ha_task_approval(
            _harnessaudit_offline_task_ids(tasks, source_reports)
        )
        preregistration = load_preregistration(
            Path(preregistration_path), benchmark="harnessaudit"
        )
        annotation_contract_files = validate_annotation_contract_files(
            preregistration,
            compositionality_manifest_path=compositionality_manifest_path,
            faithfulness_adjudication_protocol_path=(
                faithfulness_adjudication_protocol_path
            ),
            adjudicator_blinding_protocol_path=adjudicator_blinding_protocol_path,
        )
        frozen_preregistration_path = (
            Path(paths.analysis_dir) / "formal_preregistration.json"
        )
        validate_formal_output_registration(
            marker_path=frozen_preregistration_path,
            artifact_roots=(
                paths.output_root,
                paths.trace_root,
                paths.analysis_dir,
                paths.figure_dir,
            ),
        )
        freeze_preregistration(preregistration, frozen_preregistration_path)
    apply_model_api_env(model_api)
    for path in (paths.output_root, paths.trace_root, paths.analysis_dir, paths.figure_dir):
        path.mkdir(parents=True, exist_ok=True)
    # Clear process-local formal switches before preflight.  They are enabled
    # only after every source/contract/label gate passes, so a rejected run
    # cannot contaminate a later run in the same Python process.
    os.environ.pop("SAFEPRED_TRAIN_POLICY_REQUIRE_CACHE", None)
    os.environ.pop("SAFEPRED_REQUIRE_RESPONSE_FORMAT", None)
    clean_reports = list(source_reports)
    if tasks and not skip_clean_run:
        if resolved_actor_trace_seed is None:
            raise ValueError(
                "On-demand clean source generation requires a seeded actor identity"
            )
        clean_output_root = paths.output_root / "_clean_harnessaudit"
        clean_trace_root = paths.trace_root / "_clean_harnessaudit"
        clean_env = clean_agent_env_overrides(clean_agent_api)
        clean_env["HARNESS_ACTOR_SEED"] = str(resolved_actor_trace_seed)
        clean_env["HARNESS_ACTOR_REQUIRE_SEED"] = "1"
        clean_env["PYTHONHASHSEED"] = str(resolved_actor_trace_seed)
        clean_model = (clean_agent_api.model if clean_agent_api and clean_agent_api.model else model)
        clean_judge_model = (
            clean_agent_api.judge_model
            if clean_agent_api and clean_agent_api.judge_model
            else judge_model
        )
        for task in tasks:
            clean_reports.append(
                run_clean_harnessaudit_task(
                    task=task,
                    model=clean_model,
                    judge_model=clean_judge_model,
                    output_root=clean_output_root,
                    trace_root=clean_trace_root,
                    max_turns=max_turns,
                    skip_judge=skip_judge,
                    env_overrides=clean_env,
                    skip_existing=skip_existing,
                )
            )
    clean_reports = list(
        dict.fromkeys(Path(path).expanduser().resolve() for path in clean_reports)
    )
    if not clean_reports:
        raise ValueError("No source reports available. Provide --source-reports or run with --tasks.")
    if not formal:
        source_identities = {
            _source_report_actor_identity(
                json.loads(report_path.read_text(encoding="utf-8")),
                legacy_fallback_seed=int(experiment_seed),
                expected_actor_trace_seed=actor_trace_seed,
            )
            for report_path in clean_reports
        }
        if len(source_identities) != 1:
            raise ValueError(
                "HA source reports must share one actor replicate identity: "
                f"{sorted((identity.full_key for identity in source_identities), key=str)}"
            )
        source_identity = next(iter(source_identities))
        resolved_actor_protocol_profile = source_identity.actor_protocol_profile
        resolved_actor_replicate_id = source_identity.actor_replicate_id
        resolved_actor_trace_seed = source_identity.actor_trace_seed
    source_action_trajectory_sha256_by_task = _ha_source_action_trajectory_digests(
        clean_reports
    )

    if formal:
        if preregistration is None or formal_task_approval is None:
            raise RuntimeError("Formal HA contract state was not initialized")
        source_identity = _validate_ha_actor_seed_provenance(
            clean_reports,
            expected_actor_trace_seed=actor_trace_seed,
        )
        resolved_actor_protocol_profile = source_identity.actor_protocol_profile
        resolved_actor_replicate_id = source_identity.actor_replicate_id
        resolved_actor_trace_seed = source_identity.actor_trace_seed
        source_manifest_digest = _ha_source_input_manifest_digest(clean_reports)
        observed_protocol = _ha_observed_formal_protocol(
            point_config=point_config,
            rollout_modes=normalized_rollout_modes,
            model=model,
            model_api=model_api,
            base_config=base_config,
            max_trace_events=max_trace_events,
            task_approval_sha256=str(formal_task_approval["sha256"]),
            actor_protocol_profile=resolved_actor_protocol_profile,
            actor_replicate_id=resolved_actor_replicate_id,
            actor_trace_seed=resolved_actor_trace_seed,
            source_manifest_sha256=source_manifest_digest,
            source_inputs_contract=deepcopy(
                preregistration["payload"]["offline_prediction"]
                ["protocol_by_benchmark"]["harnessaudit"]["source_inputs"]
            ),
        )
        offline_contract_gate = validate_preregistered_offline_plan(
            preregistration,
            benchmark="harnessaudit",
            tasks=_harnessaudit_offline_task_ids(tasks, clean_reports),
            policy_modes=policy_modes,
            rollout_modes=normalized_rollout_modes,
            horizons=horizons,
            max_formal_horizon=max(horizons),
            gt_horizon_mode=gt_estimand,
            lead_time_offset_base=0,
            actor_protocol_profile=resolved_actor_protocol_profile,
            actor_replicate_id=resolved_actor_replicate_id,
            actor_trace_seed=resolved_actor_trace_seed,
            wm_seed=resolved_wm_seed,
            observed_protocol=observed_protocol,
            source_input_manifest_sha256=source_manifest_digest,
        )
        formal_replicate_identity_path = freeze_formal_replicate_identity(
            offline_contract_gate,
            Path(paths.analysis_dir) / "formal_replicate_identity.json",
        )
    source_audit_artifacts = write_clean_source_audit(
        report_paths=clean_reports,
        analysis_dir=paths.analysis_dir,
    )
    split_audit_artifacts = write_train_policy_split_audit(
        report_paths=clean_reports,
        policy_modes=policy_modes,
        analysis_dir=paths.analysis_dir,
    )
    manifest_rows: list[dict[str, Any]] = []
    replay_jobs: list[dict[str, Any]] = []
    report_points_for_support: list[tuple[str, Path, list[SparsePoint]]] = []
    replay_sources: list[tuple[Path, dict[str, Any], Path, list[SparsePoint]]] = []
    for report_path in clean_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        task_id = str(report.get("task_id") or "")
        trace_file = _resolve_trace_file(report.get("trace_file"), report_path)
        steps = build_offline_prediction_steps(_load_jsonl(trace_file))
        # The formal horizon curve uses one common cohort.  Otherwise h1 would
        # include near-terminal points that h6 cannot evaluate, confounding
        # forecast distance with a changing sample composition.  Filter before
        # point selection so an ineligible near-terminal point never consumes a
        # preregistered selection slot.
        selected_points = select_common_cohort_prediction_points(
            steps,
            point_config,
            max_horizon=max(horizons),
        )
        points = [
            SparsePoint(
                point_num=point_num,
                step_index_0based=point.step_index_0based,
                step=point.step,
            )
            for point_num, point in enumerate(
                (
                    point for point in selected_points
                ),
                start=1,
            )
        ]
        if not points:
            raise ValueError(
                f"No common-cohort prediction points in {report_path} have at least "
                f"{max(horizons)} remaining actions"
            )
        if formal and point_config.scheme in {"start_mid", "first"}:
            if len(points) != int(point_config.count):
                raise ValueError(
                    "Formal HA point selection could not realize the preregistered count: "
                    f"task={task_id}, selected={len(points)}, required={point_config.count}"
                )
        if formal and point_config.scheme == "explicit":
            observed_sequences = {point.step.action_sequence_num for point in points}
            expected_sequences = {int(value) for value in point_config.explicit_sequences}
            if observed_sequences != expected_sequences:
                raise ValueError(
                    "Formal HA explicit point cohort is incomplete or ineligible: "
                    f"task={task_id}, observed={sorted(observed_sequences)}, "
                    f"required={sorted(expected_sequences)}"
                )
        report_points_for_support.append((task_id, trace_file, points))
        replay_sources.append((report_path, report, trace_file, points))
        manifest_rows.extend(
            build_predict_only_manifest_rows(
                source_report=str(report_path),
                task_id=task_id,
                policy_modes=policy_modes,
                rollout_modes=normalized_rollout_modes,
                horizons=horizons,
                points=points,
                point_scheme=point_config.scheme,
                target_estimand=gt_estimand,
            )
        )
    label_support_rows = _ha_label_support_census_rows(
        report_points=report_points_for_support,
        horizons=horizons,
    )
    label_support_path = paths.analysis_dir / "predict_only_label_support_preflight.csv"
    _write_csv(label_support_path, label_support_rows)
    selected_points_by_task = [
        (task_id, points)
        for task_id, _trace_file, points in report_points_for_support
    ]
    try:
        longest_selected_prefix = validate_selected_prefix_budget(
            selected_points_by_task,
            max_trace_events,
        )
    except ValueError as exc:
        if not formal:
            raise
        required_prefix = max(
            (
                len(point.step.trace_prefix_actions)
                for _task_id, points in selected_points_by_task
                for point in points
            ),
            default=0,
        )
        raise ValueError(
            "Formal HA max_trace_events would truncate an observed prediction prefix: "
            f"configured={max_trace_events}, required_at_least={required_prefix}. "
            "Increase and preregister the value; formal runs may not discard causal precursors."
        ) from exc
    if formal:
        if int(max_trace_events) < longest_selected_prefix:
            raise ValueError(
                "Formal HA max_trace_events would truncate an observed prediction prefix: "
                f"configured={max_trace_events}, required_at_least={longest_selected_prefix}. "
                "Increase and preregister the value; formal runs may not discard causal precursors."
            )
        _validate_ha_formal_label_support(label_support_rows)

    # No policy-extraction or WM endpoint call occurs until every label-only,
    # trace-length, source, task, and preregistration gate above has passed.
    if formal and "train_policy" in policy_modes:
        os.environ["SAFEPRED_TRAIN_POLICY_REQUIRE_CACHE"] = "1"
    if formal:
        os.environ["SAFEPRED_REQUIRE_RESPONSE_FORMAT"] = "1"
    prepared_train_policies = _prepare_offline_train_policy_cache(
        report_paths=clean_reports,
        enabled="train_policy" in policy_modes,
        workers=workers,
    )
    # The cache may be created by the preparation step.  Hash only after that
    # step so the first and subsequent identical runs bind to the same inputs.
    policy_inputs_signature = _ha_policy_inputs_signature()
    for report_path, report, trace_file, points in replay_sources:
        for rollout_mode in normalized_rollout_modes:
            for policy_mode in policy_modes:
                job_horizons = _replay_job_horizons(rollout_mode, horizons)
                for horizon in job_horizons:
                    replay_jobs.append(
                        {
                            "report_path": report_path,
                            "report": report,
                            "trace_file": trace_file,
                            "points": points,
                            "point_config": point_config,
                            "rollout_mode": rollout_mode,
                            "policy_mode": policy_mode,
                            "prediction_steps": horizon,
                            "model": model,
                            "base_config": base_config,
                            "model_api": model_api,
                            "output_root": paths.output_root,
                            "trace_root": paths.trace_root,
                            "candidate_count": candidate_count,
                            "max_trace_events": max_trace_events,
                            "skip_existing": skip_existing,
                            "direct_multi_horizons": (
                                [horizon]
                                if rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT
                                else horizons
                                if rollout_mode in STRICT_MULTI_HORIZON_ROLLOUTS
                                else None
                            ),
                            "formal_annotation_horizon": max(horizons),
                            "policy_inputs_signature": policy_inputs_signature,
                            "experiment_seed": resolved_wm_seed,
                            "actor_protocol_profile": resolved_actor_protocol_profile,
                            "actor_replicate_id": resolved_actor_replicate_id,
                            "actor_trace_seed": resolved_actor_trace_seed,
                            "wm_seed": resolved_wm_seed,
                            "fail_fast_decision_errors": fail_fast_decision_errors,
                        }
                    )

    replay_workers = max(1, int(workers or 1))
    replay_report_paths: list[Path] = []
    if replay_workers <= 1:
        for job in replay_jobs:
            replay_report_paths.append(run_predict_only_replay_for_report(**job))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=replay_workers) as executor:
            futures = [executor.submit(run_predict_only_replay_for_report, **job) for job in replay_jobs]
            for future in concurrent.futures.as_completed(futures):
                replay_report_paths.append(future.result())

    manifest_path = paths.analysis_dir / "predict_only_manifest.csv"
    _write_csv(manifest_path, manifest_rows)

    point_rows = analyze_sparse_results(
        output_root=paths.output_root,
        trace_root=paths.trace_root,
        gt_estimand=gt_estimand,
        report_paths=replay_report_paths,
    )
    point_rows = _sort_point_rows(point_rows)
    summary_rows = summarize_point_rows(point_rows)
    aggregate_rows = aggregate_by_policy_horizon(point_rows)
    quality_report = build_prediction_quality_report(point_rows)
    quality_report["formal"] = bool(formal)
    quality_report["source_action_trajectory_sha256_by_task"] = (
        source_action_trajectory_sha256_by_task
    )
    quality_report["implementation_manifest_sha256"] = (
        _ha_implementation_manifest_digest()
    )
    quality_report["offline_preregistration_gate"] = offline_contract_gate
    quality_report["rollout_modes_preregistration_status"] = (
        offline_contract_gate.get("rollout_modes", {}).get("status")
        if formal
        else "not_requested"
    )
    quality_report["formal_metric_support_complete"] = aggregate_support_is_formal(
        aggregate_rows
    )
    quality_report["formal_replicate_ready"] = bool(
        formal
        and quality_report["complete"]
        and quality_report["formal_metric_support_complete"]
        and quality_report["rollout_modes_preregistration_status"]
        == "preregistered_exact"
        and offline_contract_gate.get("status") == "validated_replicate"
    )
    # A single seed pair is never a confirmatory study.  Formal-named plots are
    # unlocked only by the joint replicate postflight, not by this run pipeline.
    quality_report["formal_plot_ready"] = bool(
        quality_report["formal_replicate_ready"]
        and offline_contract_gate.get("confirmatory_complete") is True
    )
    if quality_report["complete"] and not quality_report["formal_metric_support_complete"]:
        quality_report["warning"] = (
            "The run is computationally complete, but at least one horizon/condition lacks "
            "both classes or two task clusters; no formal-named figures were generated."
        )
    elif quality_report["complete"] and not formal:
        quality_report["warning"] = (
            "The run is computationally complete but was not frozen with --formal and "
            "a preregistration; figures are diagnostic and retain the established style."
        )
    if formal and quality_report["rollout_modes_preregistration_status"] != "preregistered_exact":
        prior_warning = str(quality_report.get("warning") or "").strip()
        quality_report["warning"] = (
            (prior_warning + " " if prior_warning else "")
            + "Rollout modes were not preregistered exactly; figures are diagnostic only."
        )
    elif formal and quality_report["formal_replicate_ready"]:
        prior_warning = str(quality_report.get("warning") or "").strip()
        quality_report["warning"] = (
            (prior_warning + " " if prior_warning else "")
            + "This seed pair is a validated formal replicate, but confirmatory figures "
            "require the complete preregistered actor/WM replicate postflight."
        )

    points_path = paths.analysis_dir / "predict_only_prediction_points.csv"
    summary_path = paths.analysis_dir / "predict_only_summary_by_task_horizon.csv"
    aggregate_path = paths.analysis_dir / "predict_only_aggregate_by_policy_horizon.csv"
    quality_path = paths.analysis_dir / "predict_only_quality_report.json"
    _write_csv(points_path, point_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(aggregate_path, aggregate_rows)
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = {
        "manifest": manifest_path,
        "points": points_path,
        "summary": summary_path,
        "aggregate": aggregate_path,
        "quality_report": quality_path,
        "label_support_preflight": label_support_path,
        **source_audit_artifacts,
        **split_audit_artifacts,
    }
    if formal_task_approval is not None:
        approval_audit_path = paths.analysis_dir / "formal_ha_task_approval_audit.json"
        approval_audit_path.write_text(
            json.dumps(formal_task_approval, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["formal_task_approval"] = approval_audit_path
    if frozen_preregistration_path is not None:
        artifacts["formal_preregistration"] = frozen_preregistration_path
    if formal_replicate_identity_path is not None:
        artifacts["formal_replicate_identity"] = formal_replicate_identity_path
    if quality_report["complete"] or not fail_on_prediction_errors:
        prefix = "" if quality_report["formal_plot_ready"] else "diagnostic_partial_"
        title_prefix = "" if quality_report["formal_plot_ready"] else "DIAGNOSTIC PARTIAL — "
        tier1_path = paths.figure_dir / f"{prefix}tier1_risk_detection_by_horizon.png"
        tier2_path = paths.figure_dir / f"{prefix}tier2_code_match_by_horizon.png"
        token_path = paths.figure_dir / f"{prefix}token_cost_input_output_by_horizon.png"
        summary_plot_path = paths.figure_dir / f"{prefix}predict_only_summary_by_horizon.png"
        experiment_title = f"{title_prefix}SafePred-HarnessAudit prediction-only"
        plot_tier1_horizon(aggregate_rows, tier1_path, experiment_title=experiment_title)
        plot_tier2_horizon(aggregate_rows, tier2_path, experiment_title=experiment_title)
        plot_token_split_horizon(aggregate_rows, token_path, experiment_title=experiment_title)
        plot_pilot_summary(
            aggregate_rows,
            summary_plot_path,
            experiment_title=f"{experiment_title} sparse horizon pilot",
        )
        artifacts.update(
            {
                "tier1_plot": tier1_path,
                "tier2_plot": tier2_path,
                "token_plot": token_path,
                "summary_plot": summary_plot_path,
            }
        )
    if run_tier3_judge and quality_report["complete"]:
        artifacts.update(
            run_tier3_judge_for_points(
                points_csv=points_path,
                results_root=paths.output_root,
                output_dir=paths.analysis_dir / "tier3_v2_llm_judge",
                model=tier3_model,
                api_key=tier3_api_key or model_api.api_key,
                api_url=tier3_api_url or model_api.api_url,
                disable_response_format=tier3_disable_response_format,
                force_rescore=tier3_force_rescore,
            )
        )
    provenance_inputs = [
        base_config,
        *clean_reports,
        *_ha_policy_input_paths(),
        *_ha_task_input_paths_for_reports(clean_reports),
    ]
    if preregistration is not None:
        provenance_inputs.append(Path(preregistration["path"]))
        provenance_inputs.extend(annotation_contract_files.values())
    for report_path in clean_reports:
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
        provenance_inputs.append(
            _resolve_trace_file(source_report.get("trace_file"), report_path)
        )
    provenance_path = paths.analysis_dir / "predict_only_provenance_manifest.json"
    raw_replay_artifacts: list[Path] = []
    for replay_report_path in replay_report_paths:
        raw_replay_artifacts.append(replay_report_path)
        replay_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
        decision_file = Path(
            str(
                (replay_report.get("safepred_offline_replay") or {}).get("decision_file")
                or replay_report.get("decision_file")
                or ""
            )
        )
        if decision_file.is_file():
            raw_replay_artifacts.append(decision_file)
            raw_replay_artifacts.extend(
                path
                for path in decision_file.parent.glob("00_safepred_*")
                if path.is_file()
            )
    write_experiment_manifest(
        provenance_path,
        experiment_type="offline_predict_only",
        benchmark="harnessaudit",
        config={
            "tasks": tasks,
            "source_reports": source_reports,
            "policy_modes": policy_modes,
            "rollout_modes": normalized_rollout_modes,
            "horizons": horizons,
            "point_config": point_config,
            "model": model,
            "judge_model": judge_model,
            "model_api": model_api,
            "clean_agent_api": clean_agent_api,
            "candidate_count": candidate_count,
            "max_trace_events": max_trace_events,
            "gt_estimand": gt_estimand,
            "workers": workers,
            "skip_existing": skip_existing,
            "policy_inputs_signature": policy_inputs_signature,
            "experiment_seed": resolved_wm_seed,
            "actor_protocol_profile": resolved_actor_protocol_profile,
            "actor_replicate_id": resolved_actor_replicate_id,
            "actor_trace_seed": resolved_actor_trace_seed,
            "wm_seed": resolved_wm_seed,
            "prepared_train_policies": prepared_train_policies,
            "formal": bool(formal),
            "preregistration_sha256": (
                preregistration["sha256"] if preregistration is not None else None
            ),
            "offline_preregistration_gate": offline_contract_gate,
        },
        input_files=provenance_inputs,
        artifact_files=[*artifacts.values(), *raw_replay_artifacts],
        status=quality_report,
    )
    artifacts["provenance_manifest"] = provenance_path
    if fail_on_prediction_errors and not quality_report["complete"]:
        raise RuntimeError(
            "Predict-only run has failed, invalid, skipped, or unaligned decisions; "
            "refusing to treat figures as complete. "
            f"See {quality_path}"
        )
    return artifacts


def run_predict_only_replay_for_report(
    *,
    report_path: Path,
    report: dict[str, Any],
    trace_file: Path,
    points: list[SparsePoint],
    point_config: PointSelectionConfig,
    rollout_mode: str,
    policy_mode: str,
    prediction_steps: int,
    model: str,
    base_config: Path,
    model_api: ModelAPIConfig,
    output_root: Path,
    trace_root: Path,
    candidate_count: int,
    max_trace_events: int,
    skip_existing: bool,
    direct_multi_horizons: list[int] | None = None,
    formal_annotation_horizon: int | None = None,
    policy_inputs_signature: str = "",
    experiment_seed: int = 0,
    actor_protocol_profile: str | None = None,
    actor_replicate_id: str | None = None,
    actor_trace_seed: int | None = None,
    wm_seed: int | None = None,
    fail_fast_decision_errors: bool = False,
) -> Path:
    guard_settings_type = SafePredGuardSettings
    guard_type = SafePredHarnessGuard
    if guard_settings_type is None:
        from multi_agent.safepred_guard.config import (
            SafePredGuardSettings as guard_settings_type,
        )
    if guard_type is None:
        from multi_agent.safepred_guard.guard import (
            SafePredHarnessGuard as guard_type,
        )

    if rollout_mode not in VALID_ROLLOUT_MODES:
        raise ValueError(
            "Legacy future-visible/prefix rollout modes cannot be newly executed; "
            f"requested={rollout_mode!r}, valid={sorted(VALID_ROLLOUT_MODES)}"
        )
    if (
        rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT
        and direct_multi_horizons != [int(prediction_steps)]
    ):
        raise ValueError(
            "direct_multi_horizons must equal [prediction_steps] for "
            "single_horizon_per_call direct replay: "
            f"prediction_steps={int(prediction_steps)}, "
            f"direct_multi_horizons={direct_multi_horizons!r}"
        )
    identity = _source_report_actor_identity(
        report,
        legacy_fallback_seed=int(experiment_seed),
        expected_actor_protocol_profile=actor_protocol_profile,
        expected_actor_replicate_id=actor_replicate_id,
        expected_actor_trace_seed=actor_trace_seed,
    )
    resolved_actor_trace_seed = identity.actor_trace_seed
    resolved_wm_seed = int(experiment_seed if wm_seed is None else wm_seed)
    task_id = str(report.get("task_id") or "")
    if not task_id:
        raise ValueError(f"Missing task_id in source report: {report_path}")
    source_run_id = _safe_name(str(report.get("run_id") or trace_file.stem))
    policy_alias = _policy_alias(policy_mode)
    rollout_alias = _rollout_alias(rollout_mode)
    point_sig = _point_signature(points)
    cache_sig = _ha_replay_cache_signature(
        report_path=report_path,
        trace_file=trace_file,
        base_config=base_config,
        model=model,
        model_api=model_api,
        policy_mode=policy_mode,
        rollout_mode=rollout_mode,
        prediction_steps=prediction_steps,
        point_signature=point_sig,
        candidate_count=candidate_count,
        max_trace_events=max_trace_events,
        direct_multi_horizons=direct_multi_horizons,
        formal_annotation_horizon=max(
            1,
            int(formal_annotation_horizon or prediction_steps),
        ),
        policy_inputs_signature=policy_inputs_signature,
        experiment_seed=resolved_wm_seed,
        actor_protocol_profile=identity.actor_protocol_profile,
        actor_replicate_id=identity.actor_replicate_id,
        actor_trace_seed=resolved_actor_trace_seed,
        wm_seed=resolved_wm_seed,
    )
    run_id = (
        f"po-{rollout_alias}-h{int(prediction_steps):02d}-{task_id}-"
        f"{policy_alias}-{point_config.scheme}-{point_sig}-{cache_sig}"
    )
    result_dir = output_root / rollout_alias / policy_alias / "prediction_only" / task_id / f"h{int(prediction_steps):02d}"
    result_path = result_dir / f"{run_id}.json"
    guard_parent = trace_root / "_g" / rollout_alias / policy_alias / "prediction_only" / task_id / f"h{int(prediction_steps):02d}"
    decision_path = guard_parent / run_id / "01_guard_decisions.jsonl"
    if skip_existing and _cached_ha_replay_is_complete(
        result_path=result_path,
        decision_path=decision_path,
        run_id=run_id,
        task_id=task_id,
        expected_decision_count=len(points),
    ):
        return result_path

    pkg_root = Path(__file__).resolve().parents[1]
    task, catalog = load_task_with_tools(f"{task_id}.yaml", project_root=pkg_root, cwd=pkg_root)
    config_path = _write_offline_config(
        base_config,
        trace_root / "_configs",
        int(prediction_steps),
        model_api=model_api,
    )
    sink = PrefixActionSink()
    settings = guard_settings_type.from_dict(
        {
            "enabled": True,
            "policy_mode": policy_mode,
            "intervention_mode": "prediction_only",
            "model_name": model,
            "config_path": str(config_path),
            "candidate_count": max(1, int(candidate_count)),
            "max_regeneration_attempts": 0,
            "horizon": max(1, int(prediction_steps)),
            "artifact_dir": str(guard_parent),
            "fail_closed": False,
            "max_trace_events": max(1, int(max_trace_events)),
            "experiment_seed": resolved_wm_seed,
        }
    )
    guard = guard_type(
        task=task,
        catalog=catalog,
        run_id=run_id,
        model=model,
        action_sink=sink,
        settings=settings,
    )
    if guard.decision_path.exists():
        guard.decision_path.unlink()

    direct_multi = (
        rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT and direct_multi_horizons
    )
    prefix_multi = rollout_mode == ITERATIVE_WM_ROLLOUT and (
        bool(direct_multi_horizons) or fail_fast_decision_errors
    )
    active_points = points

    selected_points: list[dict[str, Any]] = []
    for point in active_points:
        effective_horizon = effective_horizon_for_step(point.step, int(prediction_steps))
        guard.set_prediction_horizon(effective_horizon)
        sink.set_prefix(point.step.trace_prefix_actions)
        request_context_name = "SAFEPRED_REQUEST_CONTEXT_JSON"
        previous_context_present = request_context_name in os.environ
        previous_context = os.environ.get(request_context_name)
        os.environ[request_context_name] = json.dumps(
            {
                "task_id": task_id,
                "policy_mode": policy_mode,
                "intervention_mode": "prediction_only",
                "experiment_seed": resolved_wm_seed,
                "rollout_mode": rollout_mode,
                "point_num": int(point.point_num),
                "sequence_num": int(point.step.action_sequence_num),
                "horizon": int(effective_horizon),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            if direct_multi:
                decision = guard.evaluate_direct_multi_horizon(
                    point.step.proposed_action,
                    max_horizon=effective_horizon,
                    target_horizon=effective_horizon,
                )
            elif prefix_multi:
                decision = guard.evaluate_prefix_multi_horizon(
                    point.step.proposed_action,
                    max_horizon=effective_horizon,
                )
            else:
                decision = guard.evaluate(point.step.proposed_action)
        finally:
            if previous_context_present:
                assert previous_context is not None
                os.environ[request_context_name] = previous_context
            else:
                os.environ.pop(request_context_name, None)
        decision_error = (
            _fail_fast_decision_error(
                decision,
                rollout_mode=rollout_mode,
                effective_horizon=effective_horizon,
                direct_multi=bool(direct_multi),
                prefix_multi=bool(prefix_multi),
            )
            if fail_fast_decision_errors
            else None
        )
        if decision_error is not None:
            raise PredictionDecisionError(
                task=task_id,
                rollout=rollout_mode,
                policy=policy_mode,
                horizon=effective_horizon,
                point=point.point_num,
                decision_file=guard.decision_path,
                guard_error=decision_error,
            )
        selected_points.append(
            {
                "point_num": point.point_num,
                "step_index_0based": point.step_index_0based,
                "sequence_num": point.step.action_sequence_num,
                "horizon": effective_horizon,
                "max_prediction_steps": int(prediction_steps),
                "direct_multi_horizons": list(direct_multi_horizons or []),
                "formal_annotation_horizon": max(
                    1,
                    int(formal_annotation_horizon or prediction_steps),
                ),
                "remaining_action_count": point.step.remaining_action_count,
            }
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    synthetic_report = deepcopy(report)
    synthetic_report.update(
        {
            "run_id": run_id,
            "framework": report.get("framework") or "oai",
            "llm": model,
            "trace_file": str(trace_file),
            "decision_file": str(guard.decision_path),
            "safepred_offline_replay": {
                "source_report": str(report_path),
                "source_run_id": source_run_id,
                "rollout_mode": rollout_mode,
                "action_source": _offline_action_source(rollout_mode, direct_multi=bool(direct_multi), prefix_multi=bool(prefix_multi)),
                "state_source": _offline_state_source(rollout_mode, direct_multi=bool(direct_multi), prefix_multi=bool(prefix_multi)),
                "policy_mode": policy_mode,
                "intervention_mode": "prediction_only",
                "prediction_steps": int(prediction_steps),
                "max_prediction_steps": int(prediction_steps),
                "formal_annotation_horizon": max(
                    1,
                    int(formal_annotation_horizon or prediction_steps),
                ),
                "direct_multi_horizon": bool(direct_multi),
                "horizon_call_semantics": (
                    "single_horizon_per_call"
                    if rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT
                    else "single_hmax_nested_rollout"
                    if prefix_multi
                    else "single_horizon_standard_rollout"
                ),
                "iterative_wm_rollout": bool(prefix_multi),
                "direct_multi_horizons": list(direct_multi_horizons or []),
                "prediction_point_scheme": point_config.scheme,
                "prediction_point_count": len(active_points),
                "prediction_point_interval": max(1, int(point_config.interval)),
                "explicit_prediction_sequences": list(point_config.explicit_sequences),
                "point_signature": _point_signature(active_points),
                "horizon_cohort": "common_max_horizon_support",
                "horizon_policy": (
                    "effective_prediction_steps = min(max_prediction_steps, "
                    "remaining replay actions at each selected prediction point)"
                ),
                "source_action_count": len(build_offline_prediction_steps(_load_jsonl(trace_file))),
                "replayed_prediction_points": len(active_points),
                "expected_decision_count": len(active_points),
                "experiment_seed": resolved_wm_seed,
                "actor_protocol_profile": identity.actor_protocol_profile,
                "actor_replicate_id": identity.actor_replicate_id,
                "actor_trace_seed": resolved_actor_trace_seed,
                "wm_seed": resolved_wm_seed,
                "decision_file": str(guard.decision_path),
                "selected_prediction_points": selected_points,
                "masking": (
                    "SafePred receives only prior tool_call/communication events plus the current "
                    "proposed action; access_decision labels and future tool_call/communication "
                    "actions are excluded for both direct_multi_horizon and "
                    "iterative_wm_rollout."
                ),
            },
        }
    )
    result_path.write_text(json.dumps(synthetic_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path


def run_tier3_judge_for_points(
    *,
    points_csv: Path,
    results_root: Path,
    output_dir: Path,
    model: str,
    api_key: str | None = None,
    api_url: str | None = None,
    disable_response_format: bool = False,
    force_rescore: bool = False,
) -> dict[str, Path]:
    command = [
        sys.executable,
        "-m",
        "multi_agent.safepred_experiments.run_sparse_tier3_judge",
        "--points-csv",
        str(points_csv),
        "--results-roots",
        str(results_root),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
    ]
    if disable_response_format:
        command.append("--disable-response-format")
    if force_rescore:
        command.append("--force-rescore")
    runtime_env = _runtime_env()
    if api_key:
        runtime_env["OPENAI_API_KEY"] = api_key
        runtime_env["CUSTOM_API_KEY"] = api_key
    if api_url:
        runtime_env["OPENAI_BASE_URL"] = api_url.rstrip("/")
        runtime_env["OPENAI_API_URL"] = api_url.rstrip("/")
        runtime_env["CUSTOM_API_URL"] = api_url.rstrip("/")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=runtime_env,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return {
        "tier3_input": output_dir / "tier3_v2_sparse_judge_input.csv",
        "tier3_scores": output_dir / "tier3_v2_sparse_llm_judge_scores.csv",
        "tier3_summary": output_dir / "tier3_v2_sparse_llm_judge_summary_by_horizon.csv",
        "tier3_plot": output_dir / "tier3_v2_sparse_scores_by_horizon.png",
        "tier3_provenance_manifest": output_dir / "tier3_v2_sparse_provenance_manifest.json",
    }


def parse_horizons(values: list[int] | None, *, min_h: int, max_h: int) -> list[int]:
    if values:
        requested = [int(item) for item in values]
        if len(requested) != len(set(requested)):
            raise ValueError(f"Duplicate prediction horizons are not allowed: {requested}")
        horizons = sorted(requested)
        if horizons[0] < 1:
            raise ValueError("prediction horizons must be positive integers")
    else:
        start = max(1, int(min_h))
        stop = max(start, int(max_h))
        horizons = list(range(start, stop + 1))
    return horizons


def _read_source_reports_file(path: Path) -> list[Path]:
    base = path.expanduser().resolve().parent
    reports: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        report = Path(line).expanduser()
        reports.append(report if report.is_absolute() else (base / report).resolve())
    return reports


def main() -> None:
    _load_project_env()
    parser = argparse.ArgumentParser(description="Run the unified SafePred-HarnessAudit predict-only pipeline.")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--source-reports", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--source-reports-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file with one source report path per line.",
    )
    parser.add_argument("--policy-modes", nargs="+", default=list(DEFAULT_POLICY_MODES))
    parser.add_argument("--rollout-modes", nargs="+", default=list(DEFAULT_ROLLOUT_MODES), choices=sorted(VALID_ROLLOUT_MODES))
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--max-h", type=int, default=8)
    parser.add_argument(
        "--point-scheme",
        choices=["start_mid", "middle_pair", "all", "first", "interval", "explicit"],
        default="start_mid",
    )
    parser.add_argument("--point-count", type=int, default=4)
    parser.add_argument("--point-interval", type=int, default=1)
    parser.add_argument("--explicit-sequences", nargs="*", type=int, default=[])
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable WM weights/deployment revision; required for --formal.",
    )
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Optional model used only for the clean HarnessAudit agent run. Defaults to --model.",
    )
    parser.add_argument(
        "--agent-judge-model",
        default=None,
        help="Optional judge model used only for the clean HarnessAudit run. Defaults to --judge-model.",
    )
    parser.add_argument("--tier3-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--tier3-api-key",
        default=os.environ.get("TIER3_API_KEY"),
        help="Optional Tier3 judge API key; defaults to TIER3_API_KEY, then --api-key.",
    )
    parser.add_argument(
        "--tier3-api-url",
        "--tier3-base-url",
        dest="tier3_api_url",
        default=os.environ.get("TIER3_BASE_URL"),
        help="Optional Tier3 judge base URL; defaults to TIER3_BASE_URL, then --api-url.",
    )
    parser.add_argument("--tier3-disable-response-format", action="store_true", help="Disable JSON response_format for the Tier3 judge model.")
    parser.add_argument("--tier3-force-rescore", action="store_true", help="Ignore cached Tier3 scores and rescore all rows.")
    parser.add_argument("--llm-provider", default="custom", help="SafePred provider: openai, gemini, custom, deepseek, claude, etc.")
    parser.add_argument("--api-key", default=None, help="API key for the configured provider. Stored only in process env.")
    parser.add_argument("--api-url", "--base-url", dest="api_url", default=None, help="Provider base URL, usually an OpenAI-compatible /v1 endpoint.")
    parser.add_argument(
        "--agent-api-key",
        default=None,
        help="Optional API key used only for clean HarnessAudit agent trace generation.",
    )
    parser.add_argument(
        "--agent-api-url",
        "--agent-base-url",
        dest="agent_api_url",
        default=None,
        help="Optional OpenAI-compatible base URL used only for clean HarnessAudit agent trace generation.",
    )
    parser.add_argument("--policy-extraction-model", default=None, help="Optional model override for train_policy rule extraction.")
    parser.add_argument("--test-count", type=int, default=None, help="Limit total test tasks/reports used by this run.")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--skip-clean-run", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--run-tier3-judge", action="store_true")
    parser.add_argument("--base-config", type=Path, default=_default_base_config())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--max-trace-events", type=int, default=24)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent predict-only replay workers.")
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=0,
        help="Diagnostic backwards-compatible seed alias; formal runs require both explicit seed flags below.",
    )
    parser.add_argument("--actor-trace-seed", type=int, default=None)
    parser.add_argument("--wm-seed", type=int, default=None)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Enable fail-closed preregistered offline execution.",
    )
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--compositionality-manifest", type=Path, default=None)
    parser.add_argument(
        "--faithfulness-adjudication-protocol", type=Path, default=None
    )
    parser.add_argument("--adjudicator-blinding-protocol", type=Path, default=None)
    parser.add_argument(
        "--gt-estimand",
        choices=["exact_step"],
        default="exact_step",
        help="Primary future-risk target used for metrics; both labels are retained in point artifacts.",
    )
    parser.set_defaults(fail_on_prediction_errors=True)
    parser.add_argument(
        "--allow-prediction-errors",
        dest="fail_on_prediction_errors",
        action="store_false",
        help="Debug only: retain partial artifacts instead of failing on WM decision errors.",
    )
    parser.add_argument(
        "--fail-on-prediction-errors",
        dest="fail_on_prediction_errors",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    source_reports_arg = list(args.source_reports)
    if args.source_reports_file:
        source_reports_arg.extend(_read_source_reports_file(args.source_reports_file))

    tasks, source_reports = apply_test_count(
        list(args.tasks),
        source_reports_arg,
        args.test_count,
    )
    paths = resolve_artifact_paths(
        output_root=args.output_root,
        trace_root=args.trace_root,
        analysis_dir=args.analysis_dir,
        figure_dir=args.figure_dir,
    )
    artifacts = run_pipeline(
        tasks=tasks,
        source_reports=source_reports,
        policy_modes=list(args.policy_modes),
        rollout_modes=list(args.rollout_modes),
        horizons=parse_horizons(args.horizons, min_h=args.min_h, max_h=args.max_h),
        point_config=PointSelectionConfig(
            scheme=args.point_scheme,
            count=args.point_count,
            interval=args.point_interval,
            explicit_sequences=list(args.explicit_sequences),
        ),
        model=args.model,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        skip_clean_run=args.skip_clean_run,
        skip_judge=args.skip_judge,
        base_config=args.base_config,
        paths=paths,
        candidate_count=args.candidate_count,
        max_trace_events=args.max_trace_events,
        skip_existing=args.skip_existing,
        run_tier3_judge=args.run_tier3_judge,
        tier3_model=args.tier3_model,
        tier3_api_key=args.tier3_api_key,
        tier3_api_url=args.tier3_api_url,
        tier3_disable_response_format=args.tier3_disable_response_format,
        tier3_force_rescore=args.tier3_force_rescore,
        model_api=ModelAPIConfig(
            provider=args.llm_provider,
            api_key=args.api_key,
            api_url=args.api_url,
            model=args.model,
            model_revision=args.model_revision,
            policy_extraction_model=args.policy_extraction_model,
        ),
        clean_agent_api=CleanAgentAPIConfig(
            api_key=args.agent_api_key,
            api_url=args.agent_api_url,
            model=args.agent_model,
            judge_model=args.agent_judge_model,
        ),
        fail_on_prediction_errors=args.fail_on_prediction_errors,
        workers=args.workers,
        gt_estimand=args.gt_estimand,
        experiment_seed=args.experiment_seed,
        actor_trace_seed=args.actor_trace_seed,
        wm_seed=args.wm_seed,
        formal=args.formal,
        preregistration_path=args.preregistration,
        compositionality_manifest_path=args.compositionality_manifest,
        faithfulness_adjudication_protocol_path=(
            args.faithfulness_adjudication_protocol
        ),
        adjudicator_blinding_protocol_path=args.adjudicator_blinding_protocol,
    )
    print(f"results: {paths.output_root}")
    print(f"traces: {paths.trace_root}")
    print(f"analysis: {paths.analysis_dir}")
    print(f"figures: {paths.figure_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


def _sort_point_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("task_id") or ""),
            str(row.get("rollout_mode") or ""),
            str(row.get("policy_mode") or ""),
            _int(row.get("horizon")),
            _int(row.get("point_num")),
        ),
    )


def _point_signature(points: list[SparsePoint]) -> str:
    raw = ";".join(str(point.step.action_sequence_num) for point in points)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ha_replay_cache_signature(
    *,
    report_path: Path,
    trace_file: Path,
    base_config: Path,
    model: str,
    model_api: ModelAPIConfig,
    policy_mode: str,
    rollout_mode: str,
    prediction_steps: int,
    point_signature: str,
    candidate_count: int,
    max_trace_events: int,
    direct_multi_horizons: list[int] | None,
    formal_annotation_horizon: int,
    policy_inputs_signature: str = "",
    experiment_seed: int = 0,
    actor_protocol_profile: str | None = None,
    actor_replicate_id: str | None = None,
    actor_trace_seed: int | None = None,
    wm_seed: int | None = None,
) -> str:
    project_root = Path(__file__).resolve().parents[5]
    code_files = _ha_implementation_paths()
    identity = normalize_actor_replicate_identity(
        benchmark="harnessaudit",
        actor_protocol_profile=actor_protocol_profile,
        actor_replicate_id=actor_replicate_id,
        actor_trace_seed=(
            experiment_seed
            if actor_protocol_profile is None
            and actor_replicate_id is None
            and actor_trace_seed is None
            else actor_trace_seed
        ),
    )
    payload = {
        "cache_schema": 8,
        "source_report_sha256": _sha256_file(report_path),
        "source_trace_sha256": _sha256_file(trace_file),
        "base_config_sha256": _sha256_file(base_config),
        "code_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path)
            for path in code_files
        },
        "model": model,
        "model_revision": str(model_api.model_revision or ""),
        "provider": model_api.provider,
        "api_url": _resolved_model_api_url(model_api),
        "policy_extraction_model": model_api.policy_extraction_model,
        "policy_mode": policy_mode,
        "rollout_mode": rollout_mode,
        "prediction_steps": int(prediction_steps),
        "point_signature": point_signature,
        "candidate_count": int(candidate_count),
        "max_trace_events": int(max_trace_events),
        "direct_multi_horizons": list(direct_multi_horizons or []),
        "formal_annotation_horizon": int(formal_annotation_horizon),
        "policy_inputs_signature": policy_inputs_signature,
        "model_input_boundary": {
            "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
            "policy_extraction_prompt_version": HA_POLICY_EXTRACTION_PROMPT_VERSION,
            "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
            "input_includes_test_tool_necessity": False,
            "input_includes_tool_necessity_derivatives": False,
            "include_test_tool_necessity_policies": False,
        },
        "experience_replay": {"enable": False},
        "experiment_seed": int(experiment_seed),
        "actor_protocol_profile": identity.actor_protocol_profile,
        "actor_replicate_id": identity.actor_replicate_id,
        "actor_trace_seed": identity.actor_trace_seed,
        "wm_seed": int(experiment_seed if wm_seed is None else wm_seed),
        "horizon_call_semantics": (
            "single_horizon_per_call"
            if rollout_mode == DIRECT_MULTI_HORIZON_ROLLOUT
            else "single_hmax_nested_rollout"
            if rollout_mode == ITERATIVE_WM_ROLLOUT
            else "unsupported"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _ha_implementation_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[5]
    return [
        Path(__file__),
        Path(__file__).with_name("offline_prediction.py"),
        Path(__file__).with_name("run_sparse_prediction_horizons.py"),
        Path(__file__).with_name("formal_contract.py"),
        Path(__file__).with_name("offline_replicate_postflight.py"),
        FORMAL_HA_TASK_APPROVAL_PATH,
        Path(__file__).parents[1] / "safepred_guard" / "guard.py",
        Path(__file__).parents[1] / "safepred_guard" / "policies.py",
        Path(__file__).parents[1] / "safepred_guard" / "config.py",
        project_root / "SafePred" / "wrapper.py",
        project_root / "SafePred" / "adapters" / "harnessaudit.py",
        project_root / "SafePred" / "core" / "future_risk.py",
        project_root / "SafePred" / "models" / "world_model.py",
        project_root / "SafePred" / "models" / "prompts.py",
        project_root / "SafePred" / "utils" / "harnessaudit_policy.py",
        project_root / "SafePred" / "utils" / "llm_client.py",
        Path(__file__).parents[1] / "safepred_guard" / "candidates.py",
    ]


def _ha_implementation_manifest_digest() -> str:
    project_root = Path(__file__).resolve().parents[5]
    return source_input_manifest_sha256(
        {
            str(path.relative_to(project_root)): path
            for path in _ha_implementation_paths()
        }
    )


def _ha_policy_inputs_signature() -> str:
    files = _ha_policy_input_paths()
    payload = {
        "files": {
            str(path.resolve()): _sha256_file(path) if path.is_file() else "MISSING"
            for path in files
        },
        "environment": {
            name: os.environ.get(name, "")
            for name in (
                "SAFEPRED_POLICY_EXTRACTION_MODEL",
                "SAFEPRED_POLICY_EXTRACTION_SEED",
                "SAFEPRED_TRAIN_POLICY_SPLIT_FILE",
                "SAFEPRED_TRAIN_POLICY_SPLIT_ID",
                "SAFEPRED_TRAIN_POLICY_TEST_TASKS",
                "SAFEPRED_TRAIN_POLICY_STRICT_SPLIT",
                "SAFEPRED_TRAIN_POLICY_WRITE_CACHE",
                "SAFEPRED_ALLOW_TRAIN_POLICY_FALLBACK",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _ha_policy_input_paths() -> list[Path]:
    multi_agent_root = Path(__file__).resolve().parents[1]
    files = [
        *sorted((multi_agent_root / "tasks").rglob("*.yaml")),
        *sorted((multi_agent_root / "train_policies").rglob("*.json")),
    ]
    explicit_split = os.environ.get("SAFEPRED_TRAIN_POLICY_SPLIT_FILE")
    if explicit_split:
        split_path = Path(explicit_split).expanduser()
        if split_path not in files:
            files.append(split_path)
    return files


def _ha_policy_input_manifest_digest() -> str:
    project_root = Path(__file__).resolve().parents[5]
    logical_files: dict[str, Path] = {}
    for path in _ha_policy_input_paths():
        resolved = path.expanduser().resolve()
        try:
            logical_name = str(resolved.relative_to(project_root))
        except ValueError:
            logical_name = f"external/{resolved.name}"
        if logical_name in logical_files and logical_files[logical_name] != resolved:
            raise ValueError(
                f"Duplicate logical train-policy input name: {logical_name}"
            )
        logical_files[logical_name] = resolved
    return source_input_manifest_sha256(logical_files)


def _ha_task_input_paths_for_reports(report_paths: Iterable[Path]) -> list[Path]:
    tasks_root = Path(__file__).resolve().parents[1] / "tasks"
    inputs: list[Path] = []
    for report_path in report_paths:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        task_id = str(report.get("task_id") or "")
        matches = sorted(tasks_root.rglob(f"{task_id}.yaml"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one HarnessAudit task file for {task_id!r}, found {len(matches)}"
            )
        task_file = matches[0]
        inputs.append(task_file)
        asset_dir = task_file.parent / "assets" / task_file.stem
        if asset_dir.is_dir():
            inputs.extend(path for path in sorted(asset_dir.rglob("*")) if path.is_file())
    return inputs


def _prepare_offline_train_policy_cache(
    *,
    report_paths: Iterable[Path],
    enabled: bool,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    from multi_agent.safepred_guard.policies import train_policy_policies

    pkg_root = Path(__file__).resolve().parents[1]
    unique_reports: list[Path] = []
    seen: set[str] = set()
    for report_path in report_paths:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        task_id = str(report.get("task_id") or "")
        if task_id in seen:
            continue
        seen.add(task_id)
        unique_reports.append(Path(report_path))

    def prepare_report(report_path: Path) -> dict[str, Any]:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        task_id = str(report.get("task_id") or "")
        task, _catalog = load_task_with_tools(
            f"{task_id}.yaml",
            project_root=pkg_root,
            cwd=pkg_root,
        )
        policies = train_policy_policies(task.model_dump(mode="json"))
        harness_rows = [
            policy.get("harnessaudit") or {}
            for policy in policies
            if isinstance(policy, dict)
        ]
        return {
            "task_id": task_id,
            "policy_count": len(policies),
            "split_ids": sorted(
                {
                    str(item.get("split_id") or "")
                    for item in harness_rows
                    if str(item.get("split_id") or "")
                }
            ),
            "policy_extraction_model": os.environ.get(
                "SAFEPRED_POLICY_EXTRACTION_MODEL",
                "",
            ),
            "policy_extraction_seed": int(
                os.environ.get("SAFEPRED_POLICY_EXTRACTION_SEED", "0")
            ),
        }

    if not unique_reports:
        return []

    # Initialize the shared task summary cache before concurrent extraction.
    rows = [prepare_report(unique_reports[0])]
    cache_workers = min(8, max(1, int(workers or 1)))
    remaining_reports = unique_reports[1:]
    if cache_workers == 1:
        rows.extend(prepare_report(path) for path in remaining_reports)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cache_workers) as executor:
            rows.extend(executor.map(prepare_report, remaining_reports))
    return rows


def _write_offline_config(
    base_config: Path,
    config_dir: Path,
    prediction_steps: int,
    *,
    model_api: ModelAPIConfig,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    config = deepcopy(config)
    provider = safepred_provider_for(model_api.provider)

    world_model = config.setdefault("world_model_llm", {})
    action_agent = config.setdefault("action_agent_llm", {})
    rule_extractor = config.setdefault("rule_extractor_llm", {})
    for llm_block in (world_model, action_agent, rule_extractor):
        llm_block["provider"] = provider
        if model_api.model:
            llm_block["model_name"] = model_api.model

    if model_api.policy_extraction_model:
        rule_extractor["model_name"] = model_api.policy_extraction_model

    world_model["prediction_steps"] = int(prediction_steps)
    world_model["temperature"] = 0.0
    action_agent["temperature"] = 0.0
    rule_extractor["temperature"] = 0.0
    world_model["max_tokens"] = min(int(world_model.get("max_tokens") or 2200), 2200)
    action_agent["max_tokens"] = min(int(action_agent.get("max_tokens") or 800), 800)
    config.setdefault("planning", {})["enable"] = False
    config.setdefault("planning", {})["enable_short_term_prediction"] = True
    config.setdefault("planning", {})["enable_long_term_prediction"] = True
    config.setdefault("experience_replay", {})["enable"] = False
    config.setdefault("tree_search", {})["max_depth"] = 1
    path = config_dir / f"harnessaudit_offline_prediction_h{prediction_steps:02d}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _policy_alias(policy_mode: str) -> str:
    return {
        "full_harnessaudit_rules": "full",
        "train_policy": "train",
    }.get(policy_mode, _safe_name(policy_mode))


def _rollout_alias(rollout_mode: str) -> str:
    return {
        "direct_multi_horizon": "direct_multi",
        "iterative_wm_rollout": "iterative_wm",
    }.get(rollout_mode, _safe_name(rollout_mode))


def _offline_action_source(rollout_mode: str, *, direct_multi: bool, prefix_multi: bool) -> str:
    if direct_multi:
        return "harnessaudit_trace_prefix_and_current_action"
    if prefix_multi:
        return "current_action_then_wm_generated_actions"
    return "safepred_world_model_free_rollout"


def _offline_state_source(rollout_mode: str, *, direct_multi: bool, prefix_multi: bool) -> str:
    if direct_multi:
        return "real_trace_prefix_no_future_actions"
    if prefix_multi:
        return "real_trace_prefix_then_wm_generated_states"
    return "safepred_world_model_internal_rollout_state"


def _default_base_config() -> Path:
    return Path(__file__).resolve().parents[5] / "SafePred" / "config" / "harnessaudit_online_guard.yaml"


def _resolve_trace_file(value: Any, report_path: Path) -> Path:
    raw = Path(str(value or ""))
    if raw.is_file():
        return raw
    if raw.is_absolute():
        return raw
    pkg_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[5]
    for candidate in [pkg_root / raw, pkg_root.parent / raw, repo_root / raw, report_path.parent / raw]:
        if candidate.is_file():
            return candidate
    return raw


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL line {line_number} must be an object, got {type(payload).__name__}"
                )
            rows.append(payload)
    if not rows:
        raise ValueError("trace JSONL is empty")
    return rows


def _cached_ha_replay_is_complete(
    *,
    result_path: Path,
    decision_path: Path,
    run_id: str,
    task_id: str,
    expected_decision_count: int,
) -> bool:
    if not result_path.is_file() or not decision_path.is_file():
        return False
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        decisions = _load_jsonl_strict(decision_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    replay = report.get("safepred_offline_replay") or {}
    if str(report.get("run_id") or "") != run_id or str(report.get("task_id") or "") != task_id:
        return False
    if int(replay.get("expected_decision_count") or 0) != int(expected_decision_count):
        return False
    if len(decisions) != int(expected_decision_count):
        return False
    if [int(item.get("decision_id") or 0) for item in decisions] != list(
        range(1, int(expected_decision_count) + 1)
    ):
        return False
    return all(
        str(item.get("run_id") or "") == run_id
        and str(item.get("task_id") or "") == task_id
        and not item.get("error")
        and not item.get("skipped_by_prediction_interval")
        and str(item.get("intervention_outcome") or "")
        == "prediction_recorded_original_executed"
        for item in decisions
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else ["task_id", "policy_mode", "horizon"]
    )
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)[:120]


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_project_env() -> None:
    load_workspace_credentials(Path(__file__), override=False)


def _runtime_env() -> dict[str, str]:
    _load_project_env()
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    custom_key = env.get("CUSTOM_API_KEY")
    custom_url = env.get("CUSTOM_API_URL")
    if custom_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = custom_key
    if env.get("OPENAI_API_KEY") and not env.get("CUSTOM_API_KEY") and env.get("SAFEPRED_LLM_CLIENT_PROVIDER") == "custom":
        env["CUSTOM_API_KEY"] = env["OPENAI_API_KEY"]
    if custom_url and not env.get("OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = custom_url
    if custom_url and not env.get("OPENAI_API_URL"):
        env["OPENAI_API_URL"] = custom_url
    if env.get("OPENAI_API_URL") and not env.get("OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = env["OPENAI_API_URL"]
    return env


if __name__ == "__main__":
    main()
