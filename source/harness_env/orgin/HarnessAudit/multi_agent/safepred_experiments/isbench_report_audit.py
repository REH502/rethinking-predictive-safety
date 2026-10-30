"""Audit completed ISBench report files before SafePred predict-only runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from multi_agent.safepred_experiments.isbench_vlm_adapter import (
    group_observation_dirs,
)


EXPECTED_ACTION_ERROR_TYPES = {
    "ActionPrimitiveError",
    "ActionPrimitiveErrorGroup",
}

ACTION_OUTCOME_SCHEMA_VERSION = 1
ACTION_OUTCOME_STATUSES = {
    "completed",
    "failed",
    "terminal_noop",
    "navigation_noop",
    "infrastructure_error",
}
ACTION_OUTCOME_ERROR_STATUSES = {"failed", "infrastructure_error"}

TRACE_PROVENANCE_SCHEMA_VERSION = 2
CLEAN_STRICT_TRACE_PROFILE = "clean_strict_trace_v1"
ONLINE_STRICT_PROFILE = "online_strict_v1"
PROVIDER_REASONING_CAPTURE = "openai_message_reasoning_content_v1"
PROVIDER_REASONING_SOURCE_FIELD = "choices[0].message.reasoning_content"
GENERATION_VALIDATION_PROFILES = {
    CLEAN_STRICT_TRACE_PROFILE,
    ONLINE_STRICT_PROFILE,
}
GENERATION_PROTOCOL_FIELDS = (
    "benchmark",
    "model_requested",
    "model_resolved",
    "provider_mode",
    "model_revision",
    "api_base",
    "runtime_revision",
    "python_version",
    "dependency_versions",
    "prompt_setting",
    "actor_temperature",
    "actor_top_p",
    "actor_max_completion_tokens",
    "actor_max_images",
    "actor_retry_seed_policy",
    "actor_transport_compat",
    "actor_output_contract",
    "disable_thinking",
    "use_initial_setup",
    "use_self_caption",
    "draw_bbox_2d",
    "skip_viewer_obs",
    "online_object_sampling",
    "eval_process_safety",
    "eval_termination_safety",
    "eval_awareness",
    "eval_execution",
    "task_config_sha256",
    "camera_config_sha256",
    "implementation_sha256",
)
TRACE_DEPENDENCY_KEYS = (
    "omnigibson",
    "isaacsim",
    "torch",
    "openai",
    "google-genai",
    "numpy",
    "Pillow",
)
TRACE_IMPLEMENTATION_FILES = (
    "benchmark/__init__.py",
    "benchmark/base_benchmark.py",
    "benchmark/custom_behavior_task.py",
    "benchmark/data_utils.py",
    "benchmark/online_benchmark.py",
    "benchmark/evaluator/__init__.py",
    "benchmark/evaluator/bddl_goal_condition.py",
    "benchmark/evaluator/evaluator.py",
    "benchmark/tracker/__init__.py",
    "benchmark/tracker/base_tracker.py",
    "benchmark/tracker/online_tracker.py",
    "cli/online_benchmark_all.py",
    "cli/online_benchmark_once.py",
    "models/plan_agent.py",
    "models/safepred_safeguard.py",
    "models/server_inference.py",
    "primitives/__init__.py",
    "primitives/attachment.py",
    "primitives/ego_primitives.py",
    "primitives/executor.py",
    "primitives/object_states_utils.py",
    "primitives/primitive_utils.py",
    "utils/action_validation.py",
    "utils/action_recovery.py",
    "utils/episode_checkpoint.py",
    "utils/simulation_replay.py",
    "utils/monkey_patch.py",
    "utils/prompts.py",
    "utils/trace_provenance.py",
)
def audit_reports(
    roots: Iterable[Path],
    *,
    require_unique_tasks: int = 0,
    recommended_count: int = 3,
    tasks_dir: Path | None = None,
    require_complete_safety_labels: bool = False,
    require_explicit_process_timing: bool = False,
    require_image_quality: bool = False,
    require_successful: bool = False,
    require_generation_provenance: bool = False,
    expected_experiment_seed: int | None = None,
    expected_generation_protocol: dict[str, Any] | None = None,
    expected_generation_protocol_by_identity: Mapping[
        tuple[str, str], dict[str, Any]
    ] | None = None,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    expected_intervention_mode: str | None = None,
    expected_policy_mode: str | None = None,
) -> dict[str, Any]:
    generation_validation_profile = _normalize_generation_validation_profile(
        generation_validation_profile
    )
    if (
        expected_generation_protocol is not None
        and expected_generation_protocol_by_identity is not None
    ):
        raise ValueError(
            "expected_generation_protocol and expected_generation_protocol_by_identity "
            "are mutually exclusive"
        )
    if generation_validation_profile == ONLINE_STRICT_PROFILE:
        if expected_intervention_mode not in {
            "no_guard",
            "prediction_only",
            "advisory_only",
            "enforced_guard",
            "distributed_enforced",
        }:
            raise ValueError(
                "online-strict audit requires an explicit expected_intervention_mode"
            )
        if expected_intervention_mode == "no_guard":
            if expected_policy_mode not in (None, "", "none"):
                raise ValueError("online-strict no_guard cannot bind a policy mode")
            expected_policy_mode = None
        elif expected_policy_mode not in {
            "generic_embodied_policy",
            "train_policy",
            "full_isbench_policy_text",
        }:
            raise ValueError(
                "online-strict guarded audit requires an explicit expected_policy_mode"
            )
    report_paths = _find_report_paths(roots)
    current_implementation_hash = ""
    current_implementation_error = ""
    if require_generation_provenance and tasks_dir is not None:
        package_root = Path(tasks_dir).resolve().parent.parent / "og_ego_prim"
        try:
            current_implementation_hash = _implementation_sha256(package_root)
        except (OSError, ValueError) as exc:
            current_implementation_error = (
                f"{type(exc).__name__}: {exc}"
            )
    reports = [
        _summarize_report(
            path,
            tasks_dir=tasks_dir,
            require_complete_safety_labels=require_complete_safety_labels,
            require_explicit_process_timing=require_explicit_process_timing,
            require_image_quality=require_image_quality,
            require_generation_provenance=require_generation_provenance,
            expected_experiment_seed=expected_experiment_seed,
            expected_generation_protocol=expected_generation_protocol,
            expected_generation_protocol_by_identity=(
                expected_generation_protocol_by_identity
            ),
            current_implementation_hash=current_implementation_hash,
            current_implementation_error=current_implementation_error,
            generation_validation_profile=generation_validation_profile,
            expected_intervention_mode=expected_intervention_mode,
            expected_policy_mode=expected_policy_mode,
        )
        for path in report_paths
    ]
    replayable = [item for item in reports if item["is_replayable"]]
    successful = [item for item in reports if item["is_successful"]]
    eligible = successful if require_successful else replayable
    eligible_paths = {str(item["path"]) for item in eligible}
    for item in reports:
        item["is_eligible"] = str(item["path"]) in eligible_paths
    by_task: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_task.setdefault(str(item["task"]), []).append(item)
    recommended: list[dict[str, Any]] = []
    for task in sorted(by_task):
        candidates = sorted(
            by_task[task],
            key=lambda item: (
                -int(item["safety_check_count"]),
                -int(item["observation_image_count"]),
                str(item["path"]),
            ),
        )
        recommended.append(candidates[0])
    recommended = recommended[: max(0, int(recommended_count))]
    required = int(require_unique_tasks or 0)
    unique_eligible_task_count = len(by_task)
    unique_successful_task_count = len({str(item["task"]) for item in successful})
    expected_protocol_rows = []
    if expected_generation_protocol_by_identity is not None:
        expected_protocol_rows = [
            {
                "task": str(identity[0]),
                "scene": str(identity[1]),
                "generation_protocol_sha256": hashlib.sha256(
                    json.dumps(
                        protocol,
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for identity, protocol in sorted(
                expected_generation_protocol_by_identity.items(),
                key=lambda item: (str(item[0][0]), str(item[0][1])),
            )
        ]
    expected_protocol_set_sha256 = (
        hashlib.sha256(
            json.dumps(
                expected_protocol_rows,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if expected_generation_protocol_by_identity is not None
        else ""
    )
    observed_identities = {
        (str(item.get("task") or ""), str(item.get("scene") or ""))
        for item in reports
    }
    expected_identities = (
        {
            (str(identity[0]), str(identity[1]))
            for identity in expected_generation_protocol_by_identity
        }
        if expected_generation_protocol_by_identity is not None
        else set()
    )
    generation_protocol_binding_passed = bool(
        expected_generation_protocol_by_identity is not None
        and observed_identities == expected_identities
        and len(reports) == len(expected_identities)
        and all(
            item.get("generation_protocol_matches_expected") is True
            for item in reports
        )
    )
    return {
        "total_reports": len(reports),
        "replayable_reports": len(replayable),
        "successful_reports": len(successful),
        "unique_replayable_task_count": len(
            {str(item["task"]) for item in replayable}
        ),
        "unique_successful_task_count": unique_successful_task_count,
        "unique_eligible_task_count": len(by_task),
        "require_unique_tasks": required,
        "meets_unique_task_requirement": unique_eligible_task_count >= required if required else True,
        "require_complete_safety_labels": bool(require_complete_safety_labels),
        "require_explicit_process_timing": bool(require_explicit_process_timing),
        "require_image_quality": bool(require_image_quality),
        "require_successful": bool(require_successful),
        "require_generation_provenance": bool(require_generation_provenance),
        "generation_validation_profile": (
            generation_validation_profile if require_generation_provenance else None
        ),
        "expected_intervention_mode": (
            expected_intervention_mode if require_generation_provenance else None
        ),
        "expected_policy_mode": (
            expected_policy_mode if require_generation_provenance else None
        ),
        "expected_generation_protocol_set_sha256": expected_protocol_set_sha256,
        "generation_protocol_binding_passed": generation_protocol_binding_passed,
        "recommended_reports": recommended,
        "reports": reports,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit ISBench report.json files for SafePred runs.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--require-unique-tasks", type=int, default=0)
    parser.add_argument("--recommended-count", type=int, default=3)
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--require-complete-safety-labels", action="store_true")
    parser.add_argument("--require-explicit-process-timing", action="store_true")
    parser.add_argument("--require-image-quality", action="store_true")
    parser.add_argument("--require-successful", action="store_true")
    parser.add_argument("--require-generation-provenance", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--replayable-paths-only",
        action="store_true",
        help=(
            "Print only eligible report paths, one per line (successful paths "
            "when --require-successful is set, otherwise replayable paths)."
        ),
    )
    args = parser.parse_args(argv)

    payload = audit_reports(
        args.roots,
        require_unique_tasks=args.require_unique_tasks,
        recommended_count=args.recommended_count,
        tasks_dir=args.tasks_dir,
        require_complete_safety_labels=args.require_complete_safety_labels,
        require_explicit_process_timing=args.require_explicit_process_timing,
        require_image_quality=args.require_image_quality,
        require_successful=args.require_successful,
        require_generation_provenance=args.require_generation_provenance,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        _write_csv(args.output_csv, payload["reports"])
    if args.replayable_paths_only:
        for item in payload["reports"]:
            if item.get("is_eligible"):
                print(item["path"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["meets_unique_task_requirement"]:
        raise SystemExit(2)


def _find_report_paths(roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file() and path.name == "report.json":
            paths.append(path)
        elif path.exists():
            paths.extend(path.rglob("report.json"))
    return sorted({path.resolve() for path in paths})


def _summarize_report(
    path: Path,
    *,
    tasks_dir: Path | None,
    require_complete_safety_labels: bool,
    require_explicit_process_timing: bool,
    require_image_quality: bool,
    require_generation_provenance: bool,
    expected_experiment_seed: int | None,
    expected_generation_protocol: dict[str, Any] | None,
    expected_generation_protocol_by_identity: Mapping[
        tuple[str, str], dict[str, Any]
    ] | None,
    current_implementation_hash: str,
    current_implementation_error: str,
    generation_validation_profile: str,
    expected_intervention_mode: str | None,
    expected_policy_mode: str | None,
) -> dict[str, Any]:
    error: str | None = None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = {}
        error = f"{type(exc).__name__}: {exc}"
    error_stack = report.get("error_stack") if isinstance(report, dict) else []
    if not isinstance(error_stack, list):
        error_stack = [error_stack] if error_stack else []
    action_error_types = _ordered_unique(
        str(item.get("err_type") or "") if isinstance(item, dict) else ""
        for item in error_stack
    )
    infrastructure_error_types = [
        error_type
        for error_type in action_error_types
        if error_type not in EXPECTED_ACTION_ERROR_TYPES
    ]
    plans = report.get("plans") if isinstance(report, dict) else []
    if not isinstance(plans, list):
        plans = []
    process_checks = report.get("process_safety_goal_condition") if isinstance(report, dict) else []
    termination_checks = report.get("termination_safety_goal_condition") if isinstance(report, dict) else []
    process_checks = process_checks if isinstance(process_checks, list) else []
    termination_checks = termination_checks if isinstance(termination_checks, list) else []
    termination = report.get("termination") if isinstance(report, dict) else {}
    if not isinstance(termination, dict):
        termination = {}
    image_audit = _audit_observation_images(path.parent, len(plans))
    observation_step_count = int(image_audit["observation_step_count"])
    missing_observation_steps = list(image_audit["missing_observation_steps"])
    invalid_plan_indices = [
        index
        for index, plan in enumerate(plans, start=1)
        if not isinstance(plan, dict)
        or not isinstance(plan.get("action"), str)
        or not str(plan.get("action") or "").strip()
    ]
    step_numbers = [
        plan.get("step")
        for plan in plans
        if isinstance(plan, dict) and isinstance(plan.get("step"), int)
    ]
    expected_zero_based = list(range(len(plans)))
    expected_one_based = list(range(1, len(plans) + 1))
    non_monotonic_steps = bool(
        plans and step_numbers not in (expected_zero_based, expected_one_based)
    )
    step_index_base = (
        0
        if step_numbers == expected_zero_based
        else 1
        if step_numbers == expected_one_based
        else None
    )
    safety_audit = _safety_label_audit(
        report,
        tasks_dir=tasks_dir,
        require_explicit_process_timing=require_explicit_process_timing,
    ) if require_complete_safety_labels else {
        "errors": [],
        "task_config_path": "",
        "task_config_sha256": "",
        "condition_bindings": [],
        "condition_bindings_sha256": "",
    }
    safety_label_errors = list(safety_audit["errors"])
    image_quality_errors = list(image_audit["image_quality_errors"])
    task_id = str(report.get("task") or "").strip() if isinstance(report, dict) else ""
    expected_protocol_lookup_error = ""
    resolved_expected_generation_protocol = expected_generation_protocol
    scene_id = str(report.get("scene") or "").strip() if isinstance(report, dict) else ""
    if expected_generation_protocol_by_identity is not None:
        candidate = expected_generation_protocol_by_identity.get((task_id, scene_id))
        if isinstance(candidate, dict):
            resolved_expected_generation_protocol = candidate
        else:
            resolved_expected_generation_protocol = None
            expected_protocol_lookup_error = (
                "no expected generation protocol is registered for "
                f"task={task_id!r}, scene={scene_id!r}"
            )
    provenance_audit = _generation_provenance_audit(
        report,
        tasks_dir=tasks_dir,
        expected_experiment_seed=expected_experiment_seed,
        expected_generation_protocol=resolved_expected_generation_protocol,
        expected_protocol_lookup_error=expected_protocol_lookup_error,
        current_implementation_hash=current_implementation_hash,
        current_implementation_error=current_implementation_error,
        generation_validation_profile=generation_validation_profile,
        expected_intervention_mode=expected_intervention_mode,
        expected_policy_mode=expected_policy_mode,
    ) if require_generation_provenance else {
        "errors": [],
        "schema_version": None,
        "experiment_seed": None,
        "actor_trace_seed": None,
        "task_seed": None,
        "generation_protocol_sha256": "",
        "expected_generation_protocol_sha256": "",
        "generation_protocol_matches_expected": False,
    }
    generation_provenance_errors = list(provenance_audit["errors"])
    is_structurally_replayable = (
        not error
        and bool(plans)
        and not invalid_plan_indices
        and not non_monotonic_steps
        and bool(str(termination.get("reason") or ""))
        and not missing_observation_steps
        and not safety_label_errors
        and (not require_image_quality or not image_quality_errors)
        and not generation_provenance_errors
    )
    # A completed trace can be structurally inspectable while still containing
    # failed actions.  Formal replay excludes those traces because the realized
    # state transition is not a clean benchmark trajectory.
    is_replayable = is_structurally_replayable and not error_stack
    is_successful = (
        is_replayable
        and str(termination.get("reason") or "") == "done"
    )
    return {
        "path": str(path),
        "task": str(report.get("task") or "") if isinstance(report, dict) else "",
        "scene": str(report.get("scene") or "") if isinstance(report, dict) else "",
        "plan_count": len(plans),
        "observation_image_count": image_audit["observation_image_count"],
        "observation_step_count": observation_step_count,
        "missing_observation_steps": ";".join(str(step) for step in missing_observation_steps),
        "observation_view_names": image_audit["observation_view_names"],
        "observation_views_by_step": image_audit["observation_views_by_step"],
        "observation_view_contract": image_audit["observation_view_contract"],
        "observation_view_contract_sha256": image_audit[
            "observation_view_contract_sha256"
        ],
        "image_quality_passed": not image_quality_errors,
        "image_quality_errors": " | ".join(image_quality_errors),
        "image_audit_sha256": image_audit["image_audit_sha256"],
        "generation_provenance_passed": not generation_provenance_errors,
        "generation_validation_profile": (
            generation_validation_profile if require_generation_provenance else None
        ),
        "generation_provenance_errors": " | ".join(generation_provenance_errors),
        "generation_provenance_schema_version": provenance_audit["schema_version"],
        "experiment_seed": provenance_audit["experiment_seed"],
        "actor_trace_seed": provenance_audit["actor_trace_seed"],
        "task_seed": provenance_audit["task_seed"],
        "generation_protocol_sha256": provenance_audit[
            "generation_protocol_sha256"
        ],
        "expected_generation_protocol_sha256": provenance_audit[
            "expected_generation_protocol_sha256"
        ],
        "generation_protocol_matches_expected": provenance_audit[
            "generation_protocol_matches_expected"
        ],
        "invalid_plan_indices": ";".join(str(index) for index in invalid_plan_indices),
        "non_monotonic_steps": non_monotonic_steps,
        "step_index_base": step_index_base,
        "safety_check_count": len(process_checks) + len(termination_checks),
        "complete_safety_labels": not safety_label_errors,
        "safety_label_errors": " | ".join(safety_label_errors),
        "task_config_path": safety_audit["task_config_path"],
        "task_config_sha256": safety_audit["task_config_sha256"],
        "safety_condition_binding_count": len(safety_audit["condition_bindings"]),
        "safety_condition_bindings": safety_audit["condition_bindings"],
        "safety_condition_bindings_sha256": safety_audit[
            "condition_bindings_sha256"
        ],
        "risk_types": ";".join(
            _ordered_unique(
                str(item.get("risk_type") or "")
                for item in [*process_checks, *termination_checks]
                if isinstance(item, dict) and item.get("risk_type")
            )
        ),
        "termination_reason": str(termination.get("reason") or ""),
        "error_count": len(error_stack) + (1 if error else 0),
        "has_action_errors": bool(error_stack),
        "action_error_types": ";".join(action_error_types),
        "infrastructure_error_types": ";".join(infrastructure_error_types),
        "has_infrastructure_errors": bool(infrastructure_error_types),
        "is_structurally_replayable": is_structurally_replayable,
        "is_replayable": is_replayable,
        "is_successful": is_successful,
        "parse_error": error or "",
    }


def _safety_label_errors(
    report: dict[str, Any],
    *,
    tasks_dir: Path | None,
    require_explicit_process_timing: bool = False,
) -> list[str]:
    return list(
        _safety_label_audit(
            report,
            tasks_dir=tasks_dir,
            require_explicit_process_timing=require_explicit_process_timing,
        )["errors"]
    )


def _generation_provenance_audit(
    report: dict[str, Any],
    *,
    tasks_dir: Path | None,
    expected_experiment_seed: int | None,
    expected_generation_protocol: dict[str, Any] | None,
    expected_protocol_lookup_error: str,
    current_implementation_hash: str,
    current_implementation_error: str,
    generation_validation_profile: str,
    expected_intervention_mode: str | None,
    expected_policy_mode: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_protocol_lookup_error:
        errors.append(expected_protocol_lookup_error)
    metadata = report.get("experiment_metadata")
    if not isinstance(metadata, dict):
        return {
            "errors": ["experiment_metadata is missing or not an object"],
            "schema_version": None,
            "experiment_seed": None,
            "actor_trace_seed": None,
            "task_seed": None,
            "generation_protocol_sha256": "",
            "expected_generation_protocol_sha256": "",
            "generation_protocol_matches_expected": False,
        }

    schema_version = metadata.get("schema_version")
    if schema_version != TRACE_PROVENANCE_SCHEMA_VERSION:
        errors.append(
            "experiment_metadata.schema_version="
            f"{schema_version!r} expected={TRACE_PROVENANCE_SCHEMA_VERSION}"
        )
    if metadata.get("recorded_before_actor_call") is not True:
        errors.append(
            "experiment_metadata was not recorded natively before the actor call"
        )

    task = str(report.get("task") or "").strip()
    scene = str(report.get("scene") or "").strip()
    experiment_seed = _integer_seed(metadata.get("experiment_seed"))
    actor_trace_seed = _integer_seed(metadata.get("actor_trace_seed"))
    task_seed = _integer_seed(metadata.get("task_seed"))
    actor_seed = _integer_seed(metadata.get("actor_seed"))
    safepred_seed = _integer_seed(metadata.get("safepred_seed"))
    for name, value in (
        ("experiment_seed", experiment_seed),
        ("actor_trace_seed", actor_trace_seed),
        ("task_seed", task_seed),
        ("actor_seed", actor_seed),
        ("safepred_seed", safepred_seed),
    ):
        if value is None:
            errors.append(f"experiment_metadata.{name} must be an integer")
        elif value < 0:
            errors.append(f"experiment_metadata.{name} must be non-negative")

    if experiment_seed is not None:
        if actor_trace_seed != experiment_seed:
            errors.append(
                "actor_trace_seed must equal the experiment-level actor seed"
            )
        if expected_experiment_seed is not None and experiment_seed != int(
            expected_experiment_seed
        ):
            errors.append(
                f"experiment_seed={experiment_seed} expected={int(expected_experiment_seed)}"
            )
        if task and scene:
            derived_seed = _derive_task_seed(experiment_seed, task, scene)
            for name, value in (
                ("task_seed", task_seed),
                ("actor_seed", actor_seed),
                ("safepred_seed", safepred_seed),
            ):
                if value != derived_seed:
                    errors.append(
                        f"{name}={value!r} expected derived seed={derived_seed}"
                    )
        else:
            errors.append("report task and scene are required for seed derivation")

    protocol = metadata.get("generation_protocol")
    if not isinstance(protocol, dict):
        errors.append("experiment_metadata.generation_protocol is missing or not an object")
        protocol = {}
    for field in GENERATION_PROTOCOL_FIELDS:
        if field not in protocol:
            errors.append(f"generation_protocol lacks {field}")
    if protocol.get("benchmark") != "isbench":
        errors.append("generation_protocol.benchmark must equal 'isbench'")
    for field in (
        "model_requested",
        "model_resolved",
        "provider_mode",
        "model_revision",
        "prompt_setting",
        "api_base",
        "runtime_revision",
        "python_version",
    ):
        if not str(protocol.get(field) or "").strip():
            errors.append(f"generation_protocol.{field} must be non-empty")
    _validate_generation_parameters(
        protocol,
        errors,
        generation_validation_profile=generation_validation_profile,
        expected_output_contract=(
            expected_generation_protocol.get("actor_output_contract")
            if expected_generation_protocol is not None else None
        ),
    )
    reasoning_capture = protocol.get("provider_reasoning_capture")
    if reasoning_capture is not None and reasoning_capture != PROVIDER_REASONING_CAPTURE:
        errors.append(
            "generation_protocol.provider_reasoning_capture is unsupported"
        )

    encoded = json.dumps(
        protocol,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    protocol_digest = hashlib.sha256(encoded).hexdigest()
    if metadata.get("generation_protocol_sha256") != protocol_digest:
        errors.append(
            "experiment_metadata.generation_protocol_sha256 does not match "
            "generation_protocol"
        )

    errors.extend(
        _actor_generation_attempt_errors(
            report,
            task_seed=task_seed,
            actor_retry_seed_policy=str(
                protocol.get("actor_retry_seed_policy") or ""
            ),
            generation_validation_profile=generation_validation_profile,
            expected_intervention_mode=expected_intervention_mode,
            expected_policy_mode=expected_policy_mode,
            provider_reasoning_capture=(
                str(reasoning_capture) if reasoning_capture is not None else None
            ),
        )
    )
    if generation_validation_profile == ONLINE_STRICT_PROFILE:
        errors.extend(_action_outcome_errors(report, required=True))
    execution_condition = report.get("execution_goal_condition")
    if not isinstance(execution_condition, dict) or not isinstance(
        execution_condition.get("eval"), bool
    ):
        errors.append(
            "execution_goal_condition.eval must be boolean; this validates "
            "that execution evaluation ran without filtering on its outcome"
        )

    if tasks_dir is None:
        errors.append("tasks_dir is required for generation-provenance validation")
    elif task:
        tasks_path = Path(tasks_dir).resolve()
        task_path = tasks_path / f"{task}.json"
        camera_path = tasks_path.parent / "cameras" / "camera.json"
        try:
            task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"cannot validate default scene from {task_path}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            scene_info = (
                task_payload.get("scene_info")
                if isinstance(task_payload, dict)
                else None
            )
            default_scene = str(
                scene_info.get("default_scene_model")
                if isinstance(scene_info, dict)
                else ""
            ).strip()
            if not default_scene:
                errors.append(
                    f"task config lacks scene_info.default_scene_model: {task_path}"
                )
            elif scene != default_scene:
                errors.append(
                    f"report scene={scene!r} does not match task default scene="
                    f"{default_scene!r}"
                )
        _compare_current_file_hash(
            protocol,
            field="task_config_sha256",
            path=task_path,
            errors=errors,
        )
        _compare_current_file_hash(
            protocol,
            field="camera_config_sha256",
            path=camera_path,
            errors=errors,
        )
        if current_implementation_error:
            errors.append(
                "cannot hash current ISBench implementation: "
                f"{current_implementation_error}"
            )
        elif protocol.get("implementation_sha256") != current_implementation_hash:
            errors.append(
                "generation_protocol.implementation_sha256 does not match "
                "the current trace implementation"
            )

    expected_protocol_digest = ""
    protocol_matches_expected = False
    if expected_generation_protocol is not None:
        expected_protocol_digest = hashlib.sha256(
            json.dumps(
                expected_generation_protocol,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        compared_fields = list(GENERATION_PROTOCOL_FIELDS)
        if "provider_reasoning_capture" in expected_generation_protocol:
            compared_fields.append("provider_reasoning_capture")
        protocol_matches_expected = all(
            protocol.get(field) == expected_generation_protocol.get(field)
            for field in compared_fields
        )
        _compare_expected_protocol(
            protocol,
            expected_generation_protocol,
            errors=errors,
        )

    return {
        "errors": _ordered_unique(errors),
        "schema_version": schema_version,
        "experiment_seed": experiment_seed,
        "actor_trace_seed": actor_trace_seed,
        "task_seed": task_seed,
        "generation_protocol_sha256": protocol_digest,
        "expected_generation_protocol_sha256": expected_protocol_digest,
        "generation_protocol_matches_expected": protocol_matches_expected,
    }


def _action_outcome_errors(
    report: dict[str, Any],
    *,
    required: bool,
) -> list[str]:
    """Validate the one-to-one yielded-plan/executor-outcome ledger."""

    errors: list[str] = []
    plans = report.get("plans")
    outcomes = report.get("action_outcomes")
    schema_version = report.get("action_outcome_schema_version")
    if not required and outcomes is None and schema_version is None:
        return []
    if schema_version != ACTION_OUTCOME_SCHEMA_VERSION:
        errors.append(
            "action_outcome_schema_version="
            f"{schema_version!r} expected={ACTION_OUTCOME_SCHEMA_VERSION}"
        )
    if not isinstance(plans, list):
        return [*errors, "plans is not a list for action-outcome validation"]
    if not isinstance(outcomes, list):
        return [*errors, "action_outcomes is missing or not a list"]
    if len(outcomes) != len(plans):
        errors.append(
            f"action_outcomes count={len(outcomes)} expected plans={len(plans)}"
        )

    expected_keys = {
        "sequence_num",
        "plan_step",
        "action",
        "status",
        "attempted",
        "state_may_have_changed",
        "error_type",
        "error_message",
    }
    error_bindings: list[tuple[Any, ...]] = []
    for index, outcome in enumerate(outcomes, start=1):
        prefix = f"action_outcomes[{index - 1}]"
        if not isinstance(outcome, dict):
            errors.append(f"{prefix} is not an object")
            continue
        if set(outcome) != expected_keys:
            errors.append(
                f"{prefix} keys={sorted(outcome)} expected={sorted(expected_keys)}"
            )
        if outcome.get("sequence_num") != index:
            errors.append(f"{prefix}.sequence_num must equal {index}")
        plan = plans[index - 1] if index <= len(plans) else None
        if not isinstance(plan, dict):
            errors.append(f"{prefix} has no matching plan object")
            continue
        if outcome.get("plan_step") != plan.get("step"):
            errors.append(f"{prefix}.plan_step does not match plans[{index - 1}].step")
        if str(outcome.get("action") or "").strip().lower() != str(
            plan.get("action") or ""
        ).strip().lower():
            errors.append(f"{prefix}.action does not match plans[{index - 1}].action")
        status = str(outcome.get("status") or "")
        if status not in ACTION_OUTCOME_STATUSES:
            errors.append(f"{prefix}.status is invalid: {status!r}")
        attempted = outcome.get("attempted")
        state_may_have_changed = outcome.get("state_may_have_changed")
        if not isinstance(attempted, bool):
            errors.append(f"{prefix}.attempted must be boolean")
        if not isinstance(state_may_have_changed, bool):
            errors.append(f"{prefix}.state_may_have_changed must be boolean")
        error_type = outcome.get("error_type")
        error_message = outcome.get("error_message")
        if status in ACTION_OUTCOME_ERROR_STATUSES:
            if attempted is not True or state_may_have_changed is not True:
                errors.append(
                    f"{prefix} error status must be attempted with uncertain state"
                )
            if not isinstance(error_type, str) or not error_type.strip():
                errors.append(f"{prefix}.error_type must be non-empty for {status}")
            if not isinstance(error_message, str) or not error_message.strip():
                errors.append(f"{prefix}.error_message must be non-empty for {status}")
            error_bindings.append(
                (
                    outcome.get("plan_step"),
                    str(outcome.get("action") or "").strip().lower(),
                    str(error_type or ""),
                    str(error_message or ""),
                    status,
                )
            )
        else:
            if error_type is not None or error_message is not None:
                errors.append(f"{prefix} non-error status must have null error fields")
            if status in {"terminal_noop", "navigation_noop"} and (
                attempted is not False or state_may_have_changed is not False
            ):
                errors.append(f"{prefix} no-op status has inconsistent state fields")
            if status == "completed" and attempted is not True:
                errors.append(f"{prefix} completed status must be attempted")

    stack = report.get("error_stack")
    if not isinstance(stack, list):
        errors.append("error_stack is not a list for action-outcome validation")
        stack = []
    stack_bindings = []
    for row in stack:
        if not isinstance(row, dict):
            errors.append("error_stack contains a non-object row")
            continue
        stack_bindings.append(
            (
                row.get("step"),
                str(row.get("action") or "").strip().lower(),
                str(row.get("err_type") or ""),
                str(row.get("msg") or ""),
                str(row.get("execution_status") or ""),
            )
        )
    if stack_bindings != error_bindings:
        errors.append("error_stack does not bind one-to-one with error action_outcomes")
    return errors


def _integer_seed(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _derive_task_seed(experiment_seed: int, task: str, scene: str) -> int:
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:{task}:{scene}".encode("utf-8")
    ).hexdigest()
    return (int(experiment_seed) + int(digest[:8], 16)) % (2**31 - 1)


def _normalize_generation_validation_profile(value: Any) -> str:
    profile = str(value or "").strip()
    if profile not in GENERATION_VALIDATION_PROFILES:
        raise ValueError(
            "Unsupported ISBench generation validation profile: "
            f"{profile!r}"
        )
    return profile


def _validate_generation_parameters(
    protocol: dict[str, Any],
    errors: list[str],
    *,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    expected_output_contract: str | None = None,
) -> None:
    generation_validation_profile = _normalize_generation_validation_profile(
        generation_validation_profile
    )
    actor_max_images = protocol.get("actor_max_images")
    if (
        not isinstance(actor_max_images, int)
        or isinstance(actor_max_images, bool)
        or actor_max_images < 0
    ):
        errors.append(
            "generation_protocol.actor_max_images must be a non-negative integer"
        )
    model_requested = str(protocol.get("model_requested") or "").strip()
    if model_requested.lower() in {"none", "example"}:
        errors.append("generation_protocol.model_requested must identify the actor model")
    model_resolved = str(protocol.get("model_resolved") or "").strip()
    if model_resolved.lower() in {"none", "example", "local"}:
        errors.append(
            "generation_protocol.model_resolved must identify the immutable served actor model"
        )
    if "gemini_direct" in model_resolved.lower():
        errors.append(
            "generation_protocol.model_resolved uses unsupported gemini_direct routing"
        )
    if protocol.get("provider_mode") not in {"local", "close_source"}:
        errors.append(
            "generation_protocol.provider_mode must be 'local' or 'close_source'"
        )
    api_base = str(protocol.get("api_base") or "").strip()
    parsed_api_base = urlsplit(api_base)
    if (
        parsed_api_base.scheme not in {"http", "https"}
        or not parsed_api_base.hostname
        or parsed_api_base.username is not None
        or parsed_api_base.password is not None
        or bool(parsed_api_base.query)
        or bool(parsed_api_base.fragment)
    ):
        errors.append(
            "generation_protocol.api_base must be an absolute HTTP(S) URL "
            "without credentials, query, or fragment"
        )
    is_aifast_endpoint = (
        parsed_api_base.scheme in {"http", "https"}
        and str(parsed_api_base.hostname or "").lower() == "aifast.site"
    )
    expected_transport_compat = (
        "aifast_noncalling_dummy_tool_v1"
        if is_aifast_endpoint
        else "none"
    )
    if protocol.get("actor_transport_compat") != expected_transport_compat:
        errors.append(
            "generation_protocol.actor_transport_compat must equal "
            f"{expected_transport_compat!r} for api_base={api_base!r}"
        )
    # An explicit caller-bound contract takes precedence over legacy endpoint
    # inference. Never infer this override from the report under audit or the
    # audit process's environment; the existing full expected-protocol binding
    # still checks the report against the pre-generation declaration.
    if expected_output_contract is None:
        expected_output_contract = (
            "upstream_fenced_json_or_strict_whole_json_v1"
            if is_aifast_endpoint
            else "upstream_lowercase_fenced_json_v1"
        )
    elif expected_output_contract not in {
        "upstream_fenced_json_or_strict_whole_json_v1",
        "upstream_lowercase_fenced_json_v1",
    }:
        errors.append("expected generation_protocol.actor_output_contract is unsupported")
    if protocol.get("actor_output_contract") != expected_output_contract:
        errors.append(
            "generation_protocol.actor_output_contract must equal "
            f"{expected_output_contract!r} for api_base={api_base!r}"
        )
    dependency_versions = protocol.get("dependency_versions")
    if not isinstance(dependency_versions, dict):
        errors.append("generation_protocol.dependency_versions must be an object")
    else:
        for key in TRACE_DEPENDENCY_KEYS:
            if not str(dependency_versions.get(key) or "").strip():
                errors.append(
                    f"generation_protocol.dependency_versions.{key} must be non-empty"
                )
    provider_mode = protocol.get("provider_mode")
    temperature = protocol.get("actor_temperature")
    valid_temperature = (
        not isinstance(temperature, bool)
        and isinstance(temperature, (int, float))
        and (
            float(temperature) == 0.0
            if provider_mode == "close_source"
            else float(temperature) > 0.0
        )
    )
    if not valid_temperature:
        expected_temperature = (
            "equal 0.0 for close_source clean generation"
            if provider_mode == "close_source"
            else "be > 0 for independently seeded local generation"
        )
        errors.append(
            f"generation_protocol.actor_temperature must {expected_temperature}"
        )
    top_p = protocol.get("actor_top_p")
    valid_top_p = (
        not isinstance(top_p, bool)
        and isinstance(top_p, (int, float))
        and (
            float(top_p) == 1.0
            if provider_mode == "close_source"
            else 0 < float(top_p) <= 1
        )
    )
    if not valid_top_p:
        if provider_mode == "close_source":
            errors.append(
                "generation_protocol.actor_top_p must equal 1.0 for "
                "close_source clean generation"
            )
        else:
            errors.append("generation_protocol.actor_top_p must be in (0, 1]")
    max_tokens = protocol.get("actor_max_completion_tokens")
    valid_max_tokens = (
        not isinstance(max_tokens, bool)
        and isinstance(max_tokens, int)
        and (
            max_tokens == 512
            if provider_mode == "close_source"
            else max_tokens > 0
        )
    )
    if not valid_max_tokens:
        if provider_mode == "close_source":
            errors.append(
                "generation_protocol.actor_max_completion_tokens must equal "
                "512 for close_source clean generation"
            )
        else:
            errors.append(
                "generation_protocol.actor_max_completion_tokens must be a "
                "positive integer"
            )
    expected_retry_seed_policy = (
        "none"
        if provider_mode == "close_source"
        else "base_plus_validation_attempt_v1"
    )
    if protocol.get("actor_retry_seed_policy") != expected_retry_seed_policy:
        errors.append(
            "generation_protocol.actor_retry_seed_policy must equal "
            f"{expected_retry_seed_policy!r} for provider_mode={provider_mode!r}"
        )
    retry_fields = {
        "actor_transport_retry_policy",
        "actor_transport_max_retries",
        "actor_transport_retry_base_delay_seconds",
    }
    if retry_fields.intersection(protocol):
        missing_retry_fields = retry_fields.difference(protocol)
        for field in sorted(missing_retry_fields):
            errors.append(f"generation_protocol.{field} must be recorded")
        max_retries = protocol.get("actor_transport_max_retries")
        valid_max_retries = (
            not isinstance(max_retries, bool)
            and isinstance(max_retries, int)
            and max_retries >= 0
        )
        if not valid_max_retries:
            errors.append(
                "generation_protocol.actor_transport_max_retries must be a "
                "nonnegative integer"
            )
        base_delay = protocol.get("actor_transport_retry_base_delay_seconds")
        valid_base_delay = (
            not isinstance(base_delay, bool)
            and isinstance(base_delay, (int, float))
            and float(base_delay) >= 0
        )
        if not valid_base_delay:
            errors.append(
                "generation_protocol.actor_transport_retry_base_delay_seconds "
                "must be nonnegative"
            )
        expected_transport_retry_policy = (
            "same_request_transient_http_v1"
            if valid_max_retries and max_retries > 0
            else "none"
        )
        if (
            protocol.get("actor_transport_retry_policy")
            != expected_transport_retry_policy
        ):
            errors.append(
                "generation_protocol.actor_transport_retry_policy must equal "
                f"{expected_transport_retry_policy!r}"
            )
    if protocol.get("prompt_setting") != "v1":
        errors.append("generation_protocol.prompt_setting must equal 'v1'")
    for field in (
        "use_initial_setup",
        "use_self_caption",
        "skip_viewer_obs",
        "online_object_sampling",
    ):
        if protocol.get(field) is not False:
            errors.append(
                f"generation_protocol.{field} must be false for clean source traces"
            )
    for field in (
        "disable_thinking",
        "use_initial_setup",
        "use_self_caption",
        "draw_bbox_2d",
        "skip_viewer_obs",
        "online_object_sampling",
        "eval_process_safety",
        "eval_termination_safety",
        "eval_awareness",
        "eval_execution",
    ):
        if not isinstance(protocol.get(field), bool):
            errors.append(f"generation_protocol.{field} must be boolean")
    expected_evaluators = {
        "eval_process_safety": True,
        "eval_termination_safety": True,
        "eval_awareness": False,
        "eval_execution": True,
    }
    for field, expected in expected_evaluators.items():
        if protocol.get(field) is not expected:
            errors.append(
                f"generation_protocol.{field} must be {str(expected).lower()} "
                "for clean source traces"
            )


def _actor_generation_attempt_errors(
    report: dict[str, Any],
    *,
    task_seed: int | None,
    actor_retry_seed_policy: str,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    expected_intervention_mode: str | None = None,
    expected_policy_mode: str | None = None,
    provider_reasoning_capture: str | None = None,
) -> list[str]:
    generation_validation_profile = _normalize_generation_validation_profile(
        generation_validation_profile
    )
    online_strict = generation_validation_profile == ONLINE_STRICT_PROFILE
    errors: list[str] = []
    if online_strict and expected_intervention_mode not in {
        "no_guard",
        "prediction_only",
        "advisory_only",
        "enforced_guard",
        "distributed_enforced",
    }:
        errors.append(
            "online-strict actor-attempt validation requires an explicit "
            "expected intervention assignment"
        )
    attempts = report.get("actor_generation_attempts")
    plans = report.get("plans")
    if not isinstance(attempts, list):
        return ["actor_generation_attempts is missing or not a list"]
    if not isinstance(plans, list):
        return ["plans is not a list for actor-generation-attempt validation"]
    expected_steps = list(range(1, len(plans) + 1))
    termination = report.get("termination")
    trailing_plan_error_step = (
        len(plans) + 1
        if online_strict
        and isinstance(termination, dict)
        and termination.get("reason") == "plan_error"
        else None
    )
    attempts_by_step: dict[int, list[dict[str, Any]]] = {}
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f"actor_generation_attempts[{index}] is not an object")
            continue
        step = attempt.get("step")
        validation_attempt = attempt.get("validation_attempt")
        regeneration_attempt = attempt.get("safepred_regeneration_attempt")
        request_seed = attempt.get("request_seed")
        accepted = attempt.get("accepted")
        raw_output = attempt.get("raw_output")
        if provider_reasoning_capture is not None:
            errors.extend(
                _provider_reasoning_record_errors(
                    attempt.get("provider_reasoning"),
                    prefix=f"actor_generation_attempts[{index}].provider_reasoning",
                    expected_contract=provider_reasoning_capture,
                )
            )
        if isinstance(step, bool) or not isinstance(step, int):
            errors.append(f"actor_generation_attempts[{index}].step must be an integer")
            continue
        attempts_by_step.setdefault(step, []).append(attempt)
        if (
            isinstance(validation_attempt, bool)
            or not isinstance(validation_attempt, int)
            or validation_attempt < 0
        ):
            errors.append(
                f"actor_generation_attempts[{index}].validation_attempt "
                "must be a non-negative integer"
            )
        if online_strict:
            if (
                isinstance(regeneration_attempt, bool)
                or not isinstance(regeneration_attempt, int)
                or regeneration_attempt < 0
            ):
                errors.append(
                    f"actor_generation_attempts[{index}].safepred_regeneration_attempt "
                    "must be a non-negative integer for online-strict generation"
                )
            for field in (
                "consultation_requested",
                "consultation_budget_exhausted",
            ):
                if not isinstance(attempt.get(field), bool):
                    errors.append(
                        f"actor_generation_attempts[{index}].{field} must be boolean "
                        "for online-strict generation"
                    )
        elif regeneration_attempt != 0:
            errors.append(
                f"actor_generation_attempts[{index}] has SafePred regeneration "
                f"attempt={regeneration_attempt!r}; clean trace generation must be unguarded"
            )
        if not isinstance(accepted, bool):
            errors.append(
                f"actor_generation_attempts[{index}].accepted must be boolean"
            )
        if not isinstance(raw_output, str) or not raw_output.strip():
            errors.append(
                f"actor_generation_attempts[{index}].raw_output must be non-empty"
            )
        expected_request_seed = None
        if (
            actor_retry_seed_policy != "none"
            and task_seed is not None
            and isinstance(validation_attempt, int)
            and not isinstance(validation_attempt, bool)
        ):
            expected_request_seed = task_seed + validation_attempt
        if request_seed != expected_request_seed:
            errors.append(
                f"actor_generation_attempts[{index}].request_seed={request_seed!r} "
                f"expected={expected_request_seed!r}"
            )

    if provider_reasoning_capture is not None:
        raw_outputs = report.get("raw_outputs")
        if not isinstance(raw_outputs, list):
            errors.append("raw_outputs is missing or not a list")
        else:
            if len(raw_outputs) != len(plans):
                errors.append(
                    "raw_outputs must contain one accepted output per yielded plan"
                )
            for raw_index, raw_record in enumerate(raw_outputs):
                prefix = f"raw_outputs[{raw_index}]"
                if not isinstance(raw_record, dict):
                    errors.append(f"{prefix} is not an object")
                    continue
                errors.extend(
                    _provider_reasoning_record_errors(
                        raw_record.get("provider_reasoning"),
                        prefix=f"{prefix}.provider_reasoning",
                        expected_contract=provider_reasoning_capture,
                    )
                )
                step = raw_record.get("step")
                matching_attempts = [
                    row
                    for row in attempts_by_step.get(step, [])
                    if row.get("accepted") is True
                    and row.get("raw_output") == raw_record.get("content")
                    and row.get("provider_reasoning")
                    == raw_record.get("provider_reasoning")
                ]
                if not matching_attempts:
                    errors.append(
                        f"{prefix} is not bound to an accepted actor attempt"
                    )

    if online_strict:
        observed_steps = [
            attempt.get("step")
            for attempt in attempts
            if isinstance(attempt, dict)
            and isinstance(attempt.get("step"), int)
            and not isinstance(attempt.get("step"), bool)
        ]
        if observed_steps != sorted(observed_steps):
            errors.append(
                "actor_generation_attempts steps must be non-decreasing for "
                "online-strict generation"
            )
        invalid_steps = sorted(
            {
                step
                for step in observed_steps
                if step < 1 or step > len(plans) + 1
            }
        )
        if invalid_steps:
            errors.append(
                "actor_generation_attempts online-strict step range is invalid: "
                f"{invalid_steps}"
            )
        for step in expected_steps:
            rows = attempts_by_step.get(step, [])
            if not rows or not any(row.get("accepted") is True for row in rows):
                errors.append(
                    f"actor_generation_attempts step={step} lacks an accepted "
                    "proposal for an executed online action"
                )
        for step, rows in sorted(attempts_by_step.items()):
            expected_validation_attempt = 0
            for row in rows:
                observed_attempt = row.get("validation_attempt")
                if observed_attempt != expected_validation_attempt:
                    errors.append(
                        f"actor_generation_attempts step={step} validation attempt="
                        f"{observed_attempt!r} expected={expected_validation_attempt}"
                    )
                if row.get("accepted") is True:
                    expected_validation_attempt = 0
                else:
                    expected_validation_attempt += 1
        safeguards = report.get("safepred_safeguard")
        configuration = report.get("safepred_configuration")
        if not isinstance(safeguards, list):
            errors.append(
                "online-strict generation requires safepred_safeguard to be a list"
            )
            safeguards = []
        if expected_intervention_mode == "no_guard":
            if safeguards:
                errors.append("online-strict no_guard must not contain SafePred decisions")
            if configuration is not None:
                errors.append(
                    "online-strict no_guard must not contain a safeguard configuration"
                )
            expected_attempt_steps = list(expected_steps)
            if trailing_plan_error_step is not None:
                trailing_rows = attempts_by_step.get(trailing_plan_error_step, [])
                if not trailing_rows or any(
                    row.get("accepted") is not False for row in trailing_rows
                ):
                    errors.append(
                        "online-strict no_guard plan_error must end with only "
                        "rejected actor-generation attempts"
                    )
                expected_attempt_steps.append(trailing_plan_error_step)
            if sorted(attempts_by_step) != expected_attempt_steps:
                errors.append(
                    "online-strict no_guard actor-attempt step coverage="
                    f"{sorted(attempts_by_step)} expected={expected_attempt_steps}"
                )
            for step in expected_attempt_steps:
                rows = attempts_by_step.get(step, [])
                if any(row.get("safepred_regeneration_attempt") != 0 for row in rows):
                    errors.append(
                        f"online-strict no_guard step={step} contains safeguard regeneration"
                    )
                if any(
                    row.get("consultation_requested") is not False
                    or row.get("consultation_budget_exhausted") is not False
                    for row in rows
                ):
                    errors.append(
                        f"online-strict no_guard step={step} contains consultation state"
                    )
                if step in expected_steps:
                    accepted_indices = [
                        index for index, row in enumerate(rows)
                        if row.get("accepted") is True
                    ]
                    if accepted_indices != [len(rows) - 1]:
                        errors.append(
                            f"online-strict no_guard step={step} must have exactly one "
                            "accepted final validation attempt"
                        )
        else:
            if not isinstance(configuration, dict):
                errors.append(
                    "online-strict guarded generation requires safepred_configuration"
                )
                configuration = {}
            if configuration.get("enabled") is not True:
                errors.append("online-strict safeguard configuration must be enabled")
            if configuration.get("assigned_condition") != expected_intervention_mode:
                errors.append(
                    "online-strict safeguard assigned_condition differs from assignment"
                )
            if configuration.get("intervention_mode") != expected_intervention_mode:
                errors.append(
                    "online-strict safeguard intervention_mode differs from assignment"
                )
            if configuration.get("policy_mode") != expected_policy_mode:
                errors.append(
                    "online-strict safeguard policy_mode differs from assignment"
                )
            for retry_field in (
                "semantic_max_attempts",
                "transport_max_attempts",
            ):
                retry_budget = configuration.get(retry_field)
                if (
                    isinstance(retry_budget, bool)
                    or not isinstance(retry_budget, int)
                    or retry_budget < 1
                ):
                    errors.append(
                        f"online-strict safeguard {retry_field} must be a "
                        "positive integer"
                    )
                elif any(
                    decision.get(retry_field) != retry_budget
                    for decision in safeguards
                    if isinstance(decision, dict)
                ):
                    errors.append(
                        f"online-strict safeguard decisions disagree on {retry_field}"
                    )
            if expected_intervention_mode in {"prediction_only", "advisory_only"}:
                for step, rows in sorted(attempts_by_step.items()):
                    if any(
                        row.get("safepred_regeneration_attempt") != 0
                        for row in rows
                    ):
                        errors.append(
                            f"online-strict {expected_intervention_mode} step={step} "
                            "contains automatic safeguard regeneration"
                        )
            if expected_intervention_mode == "advisory_only":
                for step in expected_steps:
                    rows = attempts_by_step.get(step, [])
                    real_action_indices = [
                        index for index, row in enumerate(rows)
                        if row.get("accepted") is True
                        and row.get("consultation_requested") is False
                    ]
                    if real_action_indices != [len(rows) - 1]:
                        errors.append(
                            f"online-strict advisory_only step={step} must end with "
                            "exactly one accepted non-consultation action"
                        )
            elif expected_intervention_mode == "prediction_only":
                for step in expected_steps:
                    rows = attempts_by_step.get(step, [])
                    if any(row.get("consultation_requested") is not False for row in rows):
                        errors.append(
                            f"online-strict prediction_only step={step} contains consultation state"
                        )
                    accepted_indices = [
                        index for index, row in enumerate(rows)
                        if row.get("accepted") is True
                    ]
                    if accepted_indices != [len(rows) - 1]:
                        errors.append(
                            f"online-strict prediction_only step={step} must have exactly "
                            "one accepted final validation attempt"
                        )
            elif expected_intervention_mode in {"enforced_guard", "distributed_enforced"}:
                for step, rows in sorted(attempts_by_step.items()):
                    if any(row.get("consultation_requested") is not False for row in rows):
                        errors.append(
                            f"online-strict enforced_guard step={step} contains consultation state"
                        )
                    expected_regeneration_attempt = 0
                    for row in rows:
                        observed_regeneration_attempt = row.get(
                            "safepred_regeneration_attempt"
                        )
                        if observed_regeneration_attempt != expected_regeneration_attempt:
                            errors.append(
                                f"online-strict enforced_guard step={step} regeneration "
                                f"attempt={observed_regeneration_attempt!r} "
                                f"expected={expected_regeneration_attempt}"
                            )
                        if row.get("accepted") is True:
                            expected_regeneration_attempt += 1
    else:
        if sorted(attempts_by_step) != expected_steps:
            errors.append(
                "actor_generation_attempts step coverage="
                f"{sorted(attempts_by_step)} expected={expected_steps}"
            )
        for step in expected_steps:
            rows = attempts_by_step.get(step, [])
            validation_attempts = [row.get("validation_attempt") for row in rows]
            if validation_attempts != list(range(len(rows))):
                errors.append(
                    f"actor_generation_attempts step={step} validation attempts="
                    f"{validation_attempts} expected={list(range(len(rows)))}"
                )
            accepted_indices = [
                index for index, row in enumerate(rows) if row.get("accepted") is True
            ]
            if accepted_indices != [len(rows) - 1]:
                errors.append(
                    f"actor_generation_attempts step={step} must have exactly one "
                    "accepted final validation attempt"
                )
        safeguards = report.get("safepred_safeguard")
        if safeguards not in (None, []):
            errors.append("clean trace generation must not contain SafePred safeguards")
    return errors


def _provider_reasoning_record_errors(
    value: Any,
    *,
    prefix: str,
    expected_contract: str,
) -> list[str]:
    expected_keys = {
        "capture_contract",
        "source_field",
        "present",
        "content",
        "char_count",
        "sha256",
    }
    if not isinstance(value, dict):
        return [f"{prefix} is missing or not an object"]
    errors: list[str] = []
    if set(value) != expected_keys:
        errors.append(f"{prefix} must contain exactly {sorted(expected_keys)}")
    if value.get("capture_contract") != expected_contract:
        errors.append(f"{prefix}.capture_contract does not match provenance")
    if value.get("source_field") != PROVIDER_REASONING_SOURCE_FIELD:
        errors.append(f"{prefix}.source_field is invalid")
    present = value.get("present")
    content = value.get("content")
    char_count = value.get("char_count")
    digest = value.get("sha256")
    if not isinstance(present, bool):
        errors.append(f"{prefix}.present must be boolean")
    if not isinstance(content, str):
        errors.append(f"{prefix}.content must be a string")
        content = ""
    if isinstance(char_count, bool) or not isinstance(char_count, int):
        errors.append(f"{prefix}.char_count must be an integer")
    elif char_count != len(content):
        errors.append(f"{prefix}.char_count does not match content")
    if present is True:
        expected_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not content:
            errors.append(f"{prefix}.present cannot describe empty content")
        if digest != expected_digest:
            errors.append(f"{prefix}.sha256 does not match content")
    elif present is False and (content != "" or char_count != 0 or digest is not None):
        errors.append(f"{prefix} absent record must use the canonical empty shape")
    return errors


def _compare_current_file_hash(
    protocol: dict[str, Any],
    *,
    field: str,
    path: Path,
    errors: list[str],
) -> None:
    try:
        current_hash = _sha256_file(path)
    except OSError as exc:
        errors.append(f"cannot hash current {path}: {type(exc).__name__}: {exc}")
        return
    if protocol.get(field) != current_hash:
        errors.append(f"generation_protocol.{field} does not match current {path.name}")


def _implementation_sha256(package_root: Path) -> str:
    root = Path(package_root).resolve()
    source_paths: list[tuple[str, Path]] = []
    for relative in sorted(TRACE_IMPLEMENTATION_FILES):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"implementation source escapes package root: {relative}")
        source_paths.append((relative, path))
    digest = hashlib.sha256()
    for relative, path in source_paths:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _compare_expected_protocol(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    errors: list[str],
) -> None:
    for field in GENERATION_PROTOCOL_FIELDS:
        if observed.get(field) != expected.get(field):
            errors.append(
                f"generation_protocol.{field}={observed.get(field)!r} "
                f"expected={expected.get(field)!r}"
            )
    if "provider_reasoning_capture" in expected:
        field = "provider_reasoning_capture"
        if observed.get(field) != expected.get(field):
            errors.append(
                f"generation_protocol.{field}={observed.get(field)!r} "
                f"expected={expected.get(field)!r}"
            )


def _safety_label_audit(
    report: dict[str, Any],
    *,
    tasks_dir: Path | None,
    require_explicit_process_timing: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    bindings: list[dict[str, Any]] = []
    task_id = str(report.get("task") or "").strip()
    if not task_id:
        errors.append("missing task id for safety-label validation")
        return _safety_audit_payload(errors=errors, bindings=bindings)
    if tasks_dir is None:
        errors.append("tasks_dir is required for complete safety-label validation")
        return _safety_audit_payload(errors=errors, bindings=bindings)
    task_path = (Path(tasks_dir) / f"{task_id}.json").resolve()
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            f"cannot load task safety config {task_path}: {type(exc).__name__}: {exc}"
        )
        return _safety_audit_payload(
            errors=errors,
            bindings=bindings,
            task_path=task_path,
        )
    task_hash = _sha256_file(task_path)
    declared_task_id = str(
        (task.get("task_info") or {}).get("task_name")
        if isinstance(task.get("task_info"), dict)
        else ""
    ).strip()
    if declared_task_id and declared_task_id != task_id:
        errors.append(
            "task config identity mismatch: "
            f"report task={task_id!r}, task_info.task_name={declared_task_id!r}"
        )
    expected = task.get("evaluation_goal_conditions") or {}
    plans = report.get("plans") or []
    for key in ("process_safety_goal_condition", "termination_safety_goal_condition"):
        expected_rows = expected.get(key) or []
        report_rows = report.get(key)
        # ISBench's native tracker omits an empty condition family instead of
        # serializing it as an empty list. Match that representation against
        # the task definition without weakening checks for non-empty labels.
        if report_rows is None and not expected_rows:
            report_rows = []
        if not isinstance(report_rows, list):
            errors.append(f"{key} is missing or not a list")
            continue
        if len(report_rows) != len(expected_rows):
            errors.append(
                f"{key} count={len(report_rows)} expected={len(expected_rows)}"
            )
        observed_indices_by_identity: dict[str, list[int]] = {}
        for observed_index, report_row in enumerate(report_rows):
            observed_indices_by_identity.setdefault(
                _condition_identity(report_row), []
            ).append(observed_index)
        matched_observed_indices: set[int] = set()
        for expected_index, expected_row in enumerate(expected_rows):
            expected_identity = _condition_identity(expected_row)
            candidate_indices = observed_indices_by_identity.get(expected_identity, [])
            observed_index = next(
                (
                    item
                    for item in candidate_indices
                    if item not in matched_observed_indices
                ),
                None,
            )
            report_row = (
                report_rows[observed_index]
                if observed_index is not None
                else None
            )
            observed_identity = _condition_identity(report_row)
            matched = observed_index is not None and bool(expected_identity)
            if observed_index is not None:
                matched_observed_indices.add(observed_index)
            binding = {
                "family": key,
                "index": expected_index,
                "expected_index": expected_index,
                "observed_index": observed_index,
                "condition_identity": expected_identity,
                "report_condition_identity": observed_identity,
                "matched": matched,
                "expected": _condition_identity_fields(expected_row),
                "observed": _condition_identity_fields(report_row),
                "eval": report_row.get("eval") if isinstance(report_row, dict) else None,
                "sequence_num": (
                    report_row.get("sequence_num")
                    if isinstance(report_row, dict)
                    else None
                ),
            }
            bindings.append(binding)
            if not matched:
                errors.append(
                    f"{key} expected[{expected_index}] does not match task condition identity: "
                    f"expected={expected_identity or 'missing'}"
                )
        for observed_index, report_row in enumerate(report_rows):
            if observed_index in matched_observed_indices:
                continue
            observed_identity = _condition_identity(report_row)
            bindings.append(
                {
                    "family": key,
                    "index": observed_index,
                    "expected_index": None,
                    "observed_index": observed_index,
                    "condition_identity": "",
                    "report_condition_identity": observed_identity,
                    "matched": False,
                    "expected": {},
                    "observed": _condition_identity_fields(report_row),
                    "eval": (
                        report_row.get("eval")
                        if isinstance(report_row, dict)
                        else None
                    ),
                    "sequence_num": (
                        report_row.get("sequence_num")
                        if isinstance(report_row, dict)
                        else None
                    ),
                }
            )
            errors.append(
                f"{key} observed[{observed_index}] does not match task condition identity: "
                f"observed={observed_identity or 'missing'}"
            )
        for index, row in enumerate(report_rows):
            if not isinstance(row, dict):
                errors.append(f"{key}[{index}] is not an object")
                continue
            if "eval" not in row or not isinstance(row.get("eval"), (bool, type(None))):
                errors.append(f"{key}[{index}].eval must be true, false, or null")
            if not str(row.get("risk_type") or "").strip():
                errors.append(f"{key}[{index}] lacks risk_type")
            if key == "process_safety_goal_condition" and require_explicit_process_timing:
                raw_sequence = row.get("sequence_num")
                if isinstance(row.get("eval"), bool):
                    if not isinstance(raw_sequence, int) or isinstance(raw_sequence, bool):
                        errors.append(
                            f"{key}[{index}] evaluated label lacks explicit integer sequence_num"
                        )
                    elif raw_sequence < 1 or raw_sequence > len(plans):
                        errors.append(
                            f"{key}[{index}].sequence_num={raw_sequence} is outside "
                            f"the 1..{len(plans)} plan range"
                        )
                    elif isinstance(plans[raw_sequence - 1], dict):
                        expected_action = _canonical_action(row.get("action") or "")
                        observed_action = _canonical_action(
                            plans[raw_sequence - 1].get("action") or ""
                        )
                        if expected_action != observed_action:
                            errors.append(
                                f"{key}[{index}] action/sequence mismatch: "
                                f"{expected_action} != {observed_action}"
                            )
                elif raw_sequence is not None:
                    errors.append(
                        f"{key}[{index}] unevaluated label must have sequence_num=null"
                    )
    return _safety_audit_payload(
        errors=errors,
        bindings=bindings,
        task_path=task_path,
        task_hash=task_hash,
    )


def _safety_audit_payload(
    *,
    errors: list[str],
    bindings: list[dict[str, Any]],
    task_path: Path | None = None,
    task_hash: str = "",
) -> dict[str, Any]:
    encoded = json.dumps(
        bindings,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "errors": errors,
        "task_config_path": str(task_path) if task_path is not None else "",
        "task_config_sha256": task_hash,
        "condition_bindings": bindings,
        "condition_bindings_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _condition_identity(row: Any) -> str:
    fields = _condition_identity_fields(row)
    if not fields:
        return ""
    encoded = json.dumps(
        fields,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition_identity_fields(row: Any) -> dict[str, str]:
    if not isinstance(row, dict):
        return {}
    return {
        "action": _canonical_action(row.get("action")),
        "type": _canonical_text(row.get("type")),
        "risk_type": _canonical_text(row.get("risk_type")),
        "condition": _canonical_condition(
            row.get("condition")
            if row.get("condition") is not None
            else row.get("safety_bddl")
        ),
    }


def _canonical_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _canonical_condition(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def _canonical_action(action: Any) -> str:
    return "".join(str(action or "").strip().lower().split())


def _audit_observation_images(run_dir: Path, plan_count: int) -> dict[str, Any]:
    expected_steps = set(range(max(0, int(plan_count))))
    images_by_step: dict[int, list[Path]] = {step: [] for step in expected_steps}
    errors: list[str] = []
    observation_dirs = group_observation_dirs(
        run_dir,
        expected_steps=expected_steps,
    )
    duplicate_step_directories = {
        step: [str(path.resolve()) for path in paths]
        for step, paths in observation_dirs.items()
        if len(paths) > 1
    }
    for step, paths in sorted(observation_dirs.items()):
        if len(paths) > 1:
            errors.append(
                "step "
                f"{step} has multiple observation directories with the same "
                "numeric prefix: "
                + ", ".join(path.name for path in paths)
            )
        for directory in paths:
            images_by_step[step].extend(sorted(directory.glob("*.png")))

    try:
        from PIL import Image, ImageStat
    except ImportError as exc:
        Image = None  # type: ignore[assignment]
        ImageStat = None  # type: ignore[assignment]
        errors.append(
            "Pillow is required to decode and quality-audit formal observation images: "
            f"{exc}"
        )
    audit_rows: list[dict[str, Any]] = []
    view_names: set[str] = set()
    views_by_step: dict[str, list[str]] = {}
    for step in sorted(images_by_step):
        paths = images_by_step[step]
        views_by_step[str(step)] = [path.name for path in paths]
        view_names.update(path.name for path in paths)
        pixel_hashes: dict[str, Path] = {}
        for path in paths:
            row: dict[str, Any] = {"step": step, "path": str(path.resolve())}
            if Image is None or ImageStat is None:
                row["decode_error"] = "Pillow unavailable"
                audit_rows.append(row)
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    rgb = image.convert("RGB")
                    luminance = ImageStat.Stat(rgb.convert("L")).mean[0]
                    extrema = rgb.convert("L").getextrema()
                    pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    row.update(
                        {
                            "width": width,
                            "height": height,
                            "mean_luminance": round(float(luminance), 4),
                            "luminance_max": int(extrema[1]),
                            "pixel_sha256": pixel_hash,
                        }
                    )
                    if width < 32 or height < 32:
                        errors.append(
                            f"step {step} image is too small ({width}x{height}): {path}"
                        )
                    if float(luminance) < 3.0 and int(extrema[1]) < 16:
                        errors.append(
                            f"step {step} image is effectively black/over-dark: {path}"
                        )
                    if pixel_hash in pixel_hashes:
                        errors.append(
                            f"step {step} contains duplicate views: "
                            f"{pixel_hashes[pixel_hash].name} and {path.name}"
                        )
                    else:
                        pixel_hashes[pixel_hash] = path
            except Exception as exc:
                row["decode_error"] = f"{type(exc).__name__}: {exc}"
                errors.append(
                    f"step {step} image is not decodable: {path}: "
                    f"{type(exc).__name__}: {exc}"
                )
            audit_rows.append(row)

    missing_steps = sorted(step for step, paths in images_by_step.items() if not paths)
    nonempty_view_sets = {
        step: {path.name for path in paths}
        for step, paths in images_by_step.items()
        if paths
    }
    reference_step = min(nonempty_view_sets) if nonempty_view_sets else None
    reference_views = (
        nonempty_view_sets[reference_step]
        if reference_step is not None
        else set()
    )
    inconsistent_view_steps: dict[str, dict[str, list[str]]] = {}
    for step, observed_views in sorted(nonempty_view_sets.items()):
        if observed_views == reference_views:
            continue
        missing_views = sorted(reference_views - observed_views)
        unexpected_views = sorted(observed_views - reference_views)
        inconsistent_view_steps[str(step)] = {
            "missing_views": missing_views,
            "unexpected_views": unexpected_views,
        }
        errors.append(
            f"step {step} camera view-name set differs from reference step "
            f"{reference_step}: missing={missing_views}, unexpected={unexpected_views}"
        )
    view_contract = {
        "schema_version": 1,
        "input_step_range": "0..plan_count-1",
        "post_final_step_excluded": True,
        "reference_step": reference_step,
        "required_view_names": sorted(reference_views),
        "consistent": not inconsistent_view_steps,
        "inconsistent_steps": inconsistent_view_steps,
        "duplicate_step_directories": duplicate_step_directories,
    }
    view_contract_encoded = json.dumps(
        view_contract,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = json.dumps(
        audit_rows,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "observation_image_count": sum(len(paths) for paths in images_by_step.values()),
        "observation_step_count": len(expected_steps - set(missing_steps)),
        "missing_observation_steps": missing_steps,
        "observation_view_names": ";".join(sorted(view_names)),
        "observation_views_by_step": views_by_step,
        "observation_view_contract": view_contract,
        "observation_view_contract_sha256": hashlib.sha256(
            view_contract_encoded
        ).hexdigest(),
        "image_quality_errors": errors,
        "image_audit_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_step_coverage(run_dir: Path, plan_count: int) -> tuple[int, list[int]]:
    expected_steps = set(range(max(0, int(plan_count))))
    observed_steps = set(
        group_observation_dirs(
            run_dir,
            expected_steps=expected_steps,
        )
    )
    return len(observed_steps & expected_steps), sorted(expected_steps - observed_steps)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["path"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


if __name__ == "__main__":
    main()
