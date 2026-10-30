"""Freeze derived identities into one joint HA + IS-Bench preregistration."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from experiments.formal_readiness import _placeholder_paths
from experiments.model_stack import ResolvedModelStack, load_model_stack_profile
from multi_agent.safepred_experiments.formal_contract import (
    HA_DIRECT_PROMPT_VERSION,
    HA_ITERATIVE_PROMPT_VERSION,
    HA_MODEL_INPUT_BOUNDARY_VERSION,
    HA_POLICY_CACHE_SCHEMA_VERSION,
    HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION,
    HA_POLICY_EXTRACTION_PROMPT_VERSION,
    HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS,
    HA_SINGLE_FIXED_SEED_SAMPLING_CONTRACT,
    HA_TRAIN_POLICY_CONSTRUCTION_VERSION,
    ISBENCH_FORMAL_EXECUTION_CONTRACT,
    ISBENCH_FORMAL_SCHEDULE_SEED,
    ISBENCH_FORMAL_SCHEDULE_VERSION,
    canonical_source_input_manifest_sha256,
    canonical_preregistration_sha256,
    current_online_implementation_manifest_sha256,
    isbench_formal_execution_contract_errors,
    isbench_formal_condition_order,
    isbench_formal_condition_order_sha256,
    validate_isbench_hardware_contract,
    validate_ha_online_task_cohort_manifest,
    validate_preregistration,
)


BENCHMARKS = ("harnessaudit", "isbench")


def freeze_formal_preregistration(
    *,
    draft_path: Path,
    output_path: Path,
    model_profile_path: Path,
    compositionality_manifest_path: Path,
    faithfulness_adjudication_protocol_path: Path,
    adjudicator_blinding_protocol_path: Path,
    multiplicity_analysis_code_path: Path,
    frozen_at_utc: str | None = None,
    environ: dict[str, str] | None = None,
    benchmarks: Iterable[str] | None = None,
    isbench_task_list_path: Path | None = None,
    isbench_root: Path | None = None,
) -> dict[str, Any]:
    """Create, validate, and atomically publish an immutable formal input."""

    draft_path = Path(draft_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing preregistration: {output_path}")
    if draft_path == output_path:
        raise ValueError("draft and output paths must differ")
    try:
        raw = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read preregistration draft: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("preregistration draft must be a JSON object")
    payload = deepcopy(raw)
    if payload.get("schema_version") != 3:
        raise ValueError("preregistration draft schema_version must equal 3")
    requested_benchmarks = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in (BENCHMARKS if benchmarks is None else benchmarks)
        )
    )
    if not requested_benchmarks or any(
        benchmark not in BENCHMARKS for benchmark in requested_benchmarks
    ):
        raise ValueError(
            "benchmarks must be a non-empty subset of harnessaudit and isbench"
        )
    scope = {
        str(item).strip().lower() for item in payload.get("benchmark_scope") or []
    }
    if scope != set(requested_benchmarks):
        raise ValueError(
            "benchmark_scope must exactly equal the requested benchmark set"
        )
    timestamp = str(frozen_at_utc or "").strip()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError("frozen_at_utc must be an ISO-8601 timestamp with timezone")
    payload["frozen_at_utc"] = timestamp

    stack = load_model_stack_profile(
        model_profile_path,
        environ=dict(os.environ if environ is None else environ),
        require_credentials=False,
    )
    _bind_online_model_stack(
        payload,
        stack,
        benchmarks=requested_benchmarks,
    )
    if "isbench" in requested_benchmarks:
        if isbench_task_list_path is None:
            raise ValueError(
                "IS-Bench freezes require --isbench-task-list so the exact "
                "online cohort and task-derived seeds are bound before model access"
            )
        _bind_isbench_formal_inputs(
            payload,
            stack,
            task_list_path=isbench_task_list_path,
            isbench_root=(
                Path(isbench_root)
                if isbench_root is not None
                else Path(__file__).resolve().parents[1] / "IS-Bench"
            ),
        )
    _validate_ha_registered_model_input_boundary(
        payload,
        benchmarks=requested_benchmarks,
    )
    _validate_local_serving_sources(payload, benchmarks=requested_benchmarks)
    _validate_ha_task_cohort_source(payload, benchmarks=requested_benchmarks)
    annotation_paths = {
        "compositionality_manifest_sha256": compositionality_manifest_path,
        "faithfulness_adjudication_protocol_sha256": (
            faithfulness_adjudication_protocol_path
        ),
        "adjudicator_blinding_protocol_sha256": adjudicator_blinding_protocol_path,
        "multiplicity_analysis_code_sha256": multiplicity_analysis_code_path,
    }
    annotation = payload.get("annotation_contract")
    if not isinstance(annotation, dict):
        raise ValueError("annotation_contract must be an object")
    for field, raw_path in annotation_paths.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{field} input does not exist: {path}")
        annotation[field] = _sha256_file(path)

    placeholders = _placeholder_paths(payload)
    if placeholders:
        raise ValueError(
            "draft contains unresolved placeholders: " + ", ".join(placeholders[:30])
        )
    for benchmark in requested_benchmarks:
        validate_preregistration(payload, benchmark=benchmark)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return {
        "schema_version": (
            "joint-formal-preregistration-freeze-v1"
            if set(requested_benchmarks) == set(BENCHMARKS)
            else "scoped-formal-preregistration-freeze-v1"
        ),
        "status": "frozen",
        "study_id": payload["study_id"],
        "benchmarks": list(requested_benchmarks),
        "output_path": str(output_path),
        "preregistration_sha256": canonical_preregistration_sha256(payload),
        "model_profile_sha256": stack.source_sha256,
        "implementation_manifest_sha256_by_benchmark": {
            benchmark: payload["run_plans"][benchmark]["online_advisory_contract"][
                "provenance"
            ]["implementation_manifest_sha256"]
            for benchmark in requested_benchmarks
        },
        "annotation_contract": dict(annotation),
    }


def _bind_online_model_stack(
    payload: dict[str, Any],
    stack: ResolvedModelStack,
    *,
    benchmarks: Iterable[str] = BENCHMARKS,
) -> None:
    actor = stack.roles["actor"]
    world_model = stack.roles["world_model"]
    actor_identity = {
        "provider": "openai_compatible",
        "model": actor.model,
        "endpoint": actor.base_url,
        "revision": actor.revision,
    }
    world_model_identity = {
        "provider": "openai_compatible",
        "model": world_model.model,
        "endpoint": world_model.base_url,
        "revision": world_model.revision,
    }
    normalized_benchmarks = tuple(
        dict.fromkeys(str(item).strip().lower() for item in benchmarks)
    )
    for benchmark in normalized_benchmarks:
        try:
            contract = payload["run_plans"][benchmark]["online_advisory_contract"]
            provenance = contract["provenance"]
        except (KeyError, TypeError):
            raise ValueError(
                f"run_plans.{benchmark}.online_advisory_contract is incomplete"
            ) from None
        contract["actor"] = dict(actor_identity)
        contract["world_model"] = dict(world_model_identity)
        contract["model_stack"] = stack.public_identity
        if benchmark == "harnessaudit":
            study_id = str(payload.get("study_id") or "").strip()
            is_v11_or_later = (
                study_id.rsplit("-v", 1)[-1].isdigit()
                and int(study_id.rsplit("-v", 1)[-1]) >= 11
            )
            if is_v11_or_later and (actor.top_p is None or actor.top_k is None):
                raise ValueError(
                    "V11-or-later HarnessAudit freezes require explicit actor "
                    "top_p and top_k sampling controls"
                )
            if not isinstance(
                payload["run_plans"][benchmark].get("task_cohort_contract"),
                dict,
            ):
                raise ValueError(
                    "new HarnessAudit freezes require task_cohort_contract"
                )
            sampling_contract = payload["run_plans"][benchmark].get(
                "sampling_contract"
            )
            if sampling_contract != HA_SINGLE_FIXED_SEED_SAMPLING_CONTRACT:
                raise ValueError(
                    "new HarnessAudit freezes require the exact fixed-seed "
                    "task-census sampling_contract"
                )
            runtime = contract.get("runtime")
            if not isinstance(runtime, dict) or runtime.get(
                "credential_scan_scope"
            ) != "managed_roots_plus_declared_paths_v1":
                raise ValueError(
                    "new HarnessAudit freezes require the complete credential scan scope"
                )
            if not isinstance(contract.get("safepred_config"), dict):
                raise ValueError(
                    "new HarnessAudit freezes require safepred_config identity"
                )
            policy_modes = set(
                payload["run_plans"][benchmark].get("policy_modes") or []
            )
            if "train_policy" in policy_modes:
                policy_extraction_contract = contract.get(
                    "policy_extraction_contract"
                )
                if not isinstance(policy_extraction_contract, dict):
                    raise ValueError(
                        "new HarnessAudit freezes require policy_extraction_contract"
                    )
                _validate_ha_policy_extraction_input_boundary(
                    policy_extraction_contract
                )
                _validate_ha_policy_extraction_model_binding(
                    policy_extraction_contract,
                    stack,
                )
            local_role_names = [
                role_name
                for role_name, role in stack.roles.items()
                if _is_local_serving_endpoint(str(role.base_url or ""))
            ]
            if local_role_names:
                if not isinstance(runtime, dict) or not isinstance(
                    runtime.get("serving_contract"), dict
                ):
                    raise ValueError(
                        "local HarnessAudit freezes require runtime.serving_contract"
                    )
        provenance["implementation_manifest_sha256"] = (
            current_online_implementation_manifest_sha256(benchmark)
        )
    if "isbench" in normalized_benchmarks:
        is_runtime = payload["run_plans"]["isbench"]["online_advisory_contract"][
            "runtime"
        ]
        is_runtime.update(
            {
                "max_tokens": world_model.max_tokens,
                "temperature": world_model.temperature,
                "request_timeout_seconds": world_model.timeout_seconds,
                "transport_max_attempts": world_model.transport_max_attempts,
            }
        )
        is_runtime["transport_retry_base_delay_seconds"] = float(
            is_runtime.get("transport_retry_base_delay_seconds", 0.0)
        )


def _bind_isbench_formal_inputs(
    payload: dict[str, Any],
    stack: ResolvedModelStack,
    *,
    task_list_path: Path,
    isbench_root: Path,
) -> None:
    """Bind the online cohort, task seeds, and current offline code identity.

    Schema v3 still carries an offline-protocol section for an online-only
    Decision Faithfulness component.  This function fills those identities
    from real input bytes and the resolved model stack instead of accepting
    hand-written digest-like values in a draft.
    """

    task_list_path = Path(task_list_path).expanduser().resolve()
    isbench_root = Path(isbench_root).expanduser().resolve()
    if not task_list_path.is_file():
        raise ValueError(f"IS-Bench task list does not exist: {task_list_path}")
    if not isbench_root.is_dir():
        raise ValueError(f"IS-Bench root does not exist: {isbench_root}")
    task_ids = [
        line.strip()
        for line in task_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("IS-Bench formal task list must be non-empty and unique")
    if int((payload.get("support") or {}).get("min_tasks") or 0) >= 161:
        expected_snapshot = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "task_lists"
            / "isbench_qwen35_27b_formal_full161_v1.txt"
        ).resolve()
        if task_list_path != expected_snapshot or len(task_ids) != 161:
            raise ValueError(
                "full IS-Bench formal freezes require the immutable 161-task snapshot"
            )
    try:
        plan = payload["run_plans"]["isbench"]
        contract = plan["online_advisory_contract"]
        runtime = contract["runtime"]
        protocol = payload["offline_prediction"]["protocol_by_benchmark"][
            "isbench"
        ]
    except (KeyError, TypeError):
        raise ValueError("IS-Bench formal draft is incomplete") from None
    if list(plan.get("task_ids") or []) != task_ids:
        raise ValueError(
            "run_plans.isbench.task_ids must exactly match --isbench-task-list"
        )
    seeds = list(plan.get("experiment_seeds") or [])
    if any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("IS-Bench experiment_seeds must be unique exact integers")
    execution_errors = isbench_formal_execution_contract_errors(
        runtime.get("execution_contract")
    )
    if execution_errors:
        raise ValueError(
            "IS-Bench formal draft must bind a registered execution contract: "
            + "; ".join(execution_errors)
        )
    schedule = runtime.get("execution_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("IS-Bench formal draft requires execution_schedule")
    if schedule.get("version") != ISBENCH_FORMAL_SCHEDULE_VERSION or schedule.get(
        "schedule_seed"
    ) != ISBENCH_FORMAL_SCHEDULE_SEED:
        raise ValueError("IS-Bench execution schedule version/seed drift")
    condition_order = isbench_formal_condition_order(
        experiment_seeds=seeds,
        policy_modes=plan.get("policy_modes") or [],
        intervention_modes=plan.get("intervention_modes") or [],
        schedule_seed=ISBENCH_FORMAL_SCHEDULE_SEED,
        version=ISBENCH_FORMAL_SCHEDULE_VERSION,
    )
    existing_order = schedule.get("condition_order")
    if existing_order not in (None, []) and existing_order != condition_order:
        raise ValueError("IS-Bench draft execution condition order drift")
    existing_digest = schedule.get("condition_order_sha256")
    condition_order_digest = isbench_formal_condition_order_sha256(condition_order)
    if existing_digest not in (None, "") and existing_digest != condition_order_digest:
        raise ValueError("IS-Bench draft execution schedule digest drift")
    runtime["execution_schedule"] = {
        "version": ISBENCH_FORMAL_SCHEDULE_VERSION,
        "schedule_seed": ISBENCH_FORMAL_SCHEDULE_SEED,
        "condition_order": condition_order,
        "condition_order_sha256": condition_order_digest,
    }

    bindings: list[dict[str, Any]] = []
    tasks_root = isbench_root / "data" / "tasks"
    for task_id in sorted(task_ids):
        task_path = tasks_root / f"{Path(task_id).stem}.json"
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot bind IS-Bench task identity for {task_path}: {exc}"
            ) from exc
        observed_task_id = str(
            (task.get("task_info") or {}).get("task_name") or ""
        ).strip()
        scene = str(
            (task.get("scene_info") or {}).get("default_scene_model") or ""
        ).strip()
        if observed_task_id != task_id or not scene:
            raise ValueError(
                f"IS-Bench task identity/default scene differs for {task_path}"
            )
        for seed in sorted(seeds):
            digest = hashlib.sha256(
                f"{seed}:{task_id}:{scene}".encode("utf-8")
            ).hexdigest()
            bindings.append(
                {
                    "task_id": task_id,
                    "scene": scene,
                    "experiment_seed": seed,
                    "inference_seed": (seed + int(digest[:8], 16)) % (2**31 - 1),
                }
            )
    runtime["inference_seed_policy"] = "task_scene_sha256_v1"
    runtime["inference_seed_bindings"] = bindings

    world_model = stack.roles["world_model"]
    registered_rollouts = list(
        payload["offline_prediction"]["rollout_modes_by_benchmark"]["isbench"]
    )
    supported_prompts = {
        "direct_multi_horizon": (
            "isbench-direct-vision-single-horizon-v2-neutral-schema"
        ),
        "wm_free_rollout": (
            "isbench-direct-vision-wm-free-current-action-v2-neutral-schema"
        ),
    }
    supported_semantics = {
        "direct_multi_horizon": "single_horizon_per_call",
        "wm_free_rollout": "single_hmax_nested_rollout",
    }
    if any(mode not in supported_prompts for mode in registered_rollouts):
        raise ValueError("IS-Bench draft contains an unsupported offline rollout mode")
    protocol["implementation_manifest_sha256"] = (
        _current_isbench_offline_implementation_manifest_sha256()
    )
    protocol["rollout_execution_semantics"] = {
        mode: supported_semantics[mode] for mode in registered_rollouts
    }
    protocol["model"] = {
        "provider": "openai_compatible",
        "model": world_model.model,
        "revision": world_model.revision,
        "api_base_url": world_model.base_url,
        "input_mode": "direct_vision",
        "response_format_json_object": True,
        "prompt_versions": {
            mode: supported_prompts[mode] for mode in registered_rollouts
        },
        "risk_threshold": float(contract["prediction"]["risk_threshold"]),
    }
    protocol["train_policy"] = {
        "construction_version": "isbench-held-out-train-policy-v1",
        "max_tasks": int(runtime["train_policy_max_tasks"]),
        "same_domain_only": bool(runtime["train_policy_same_domain_only"]),
    }
    task_list_sha256 = _sha256_file(task_list_path)
    protocol["source_inputs"] = {
        "manifest_kind": (
            "not_applicable_to_online_only_decision_faithfulness_component_v1"
        ),
        "manifest_sha256_by_actor_trace_seed": {
            str(seed): task_list_sha256 for seed in sorted(seeds)
        },
    }


def _current_isbench_offline_implementation_manifest_sha256() -> str:
    """Compute the offline IS-Bench implementation identity without imports.

    Importing the simulator-facing pipeline can initialize heavyweight vision
    dependencies during a pre-model freeze.  The logical file set mirrors the
    pipeline's implementation manifest and remains cheap and deterministic.
    """

    project_root = Path(__file__).resolve().parents[1]
    experiment_root = (
        project_root
        / "harness_env"
        / "orgin"
        / "HarnessAudit"
        / "multi_agent"
        / "safepred_experiments"
    )
    paths = {
        "harnessaudit/isbench_predict_only_pipeline.py": (
            experiment_root / "isbench_predict_only_pipeline.py"
        ),
        "harnessaudit/isbench_vlm_adapter.py": (
            experiment_root / "isbench_vlm_adapter.py"
        ),
        "harnessaudit/isbench_report_audit.py": (
            experiment_root / "isbench_report_audit.py"
        ),
        "harnessaudit/formal_contract.py": experiment_root / "formal_contract.py",
        "harnessaudit/run_sparse_prediction_horizons.py": (
            experiment_root / "run_sparse_prediction_horizons.py"
        ),
        "harnessaudit/offline_replicate_postflight.py": (
            experiment_root / "offline_replicate_postflight.py"
        ),
        "safepred/wrapper.py": project_root / "SafePred" / "wrapper.py",
        "safepred/adapters/isbench.py": (
            project_root / "SafePred" / "adapters" / "isbench.py"
        ),
        "safepred/core/future_risk.py": (
            project_root / "SafePred" / "core" / "future_risk.py"
        ),
        "safepred/models/world_model.py": (
            project_root / "SafePred" / "models" / "world_model.py"
        ),
        "safepred/models/prompts.py": (
            project_root / "SafePred" / "models" / "prompts.py"
        ),
        "safepred/utils/llm_client.py": (
            project_root / "SafePred" / "utils" / "llm_client.py"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "IS-Bench offline implementation manifest references missing files: "
            + ", ".join(missing)
        )
    return canonical_source_input_manifest_sha256(
        {name: _sha256_file(path) for name, path in paths.items()}
    )


def _validate_ha_policy_extraction_input_boundary(
    contract: dict[str, Any],
) -> None:
    """Reject legacy caches or held-out tool-necessity model inputs at freeze."""

    from multi_agent.safepred_guard import policies as policy_module

    expected = {
        "schema_version": HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION,
        "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
        "policy_extraction_prompt_version": HA_POLICY_EXTRACTION_PROMPT_VERSION,
        "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
        "test_visible_fields": list(HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS),
        "input_includes_test_access_rules": False,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if contract.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "new HarnessAudit freeze policy_extraction_contract has a legacy "
            "or leaky model-input boundary: "
            + ", ".join(mismatches)
        )
    implementation_expected = {
        "_TRAIN_POLICY_CACHE_SCHEMA_VERSION": HA_POLICY_CACHE_SCHEMA_VERSION,
        "_MODEL_INPUT_BOUNDARY_VERSION": HA_MODEL_INPUT_BOUNDARY_VERSION,
        "_POLICY_EXTRACTION_PROMPT_VERSION": HA_POLICY_EXTRACTION_PROMPT_VERSION,
        "_TEST_VISIBLE_FIELDS": list(HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS),
    }
    implementation_mismatches = [
        field
        for field, expected_value in implementation_expected.items()
        if getattr(policy_module, field, None) != expected_value
    ]
    if implementation_mismatches:
        raise ValueError(
            "new HarnessAudit freeze active policy implementation differs "
            "from the registered model-input boundary: "
            + ", ".join(implementation_mismatches)
        )


def _validate_ha_registered_model_input_boundary(
    payload: dict[str, Any],
    *,
    benchmarks: Iterable[str],
) -> None:
    """Require a fresh offline protocol identity for the corrected HA boundary."""

    if "harnessaudit" not in set(benchmarks):
        return
    try:
        protocol = payload["offline_prediction"]["protocol_by_benchmark"][
            "harnessaudit"
        ]
        train_policy = protocol["train_policy"]
        prompt_versions = protocol["model"]["prompt_versions"]
    except (KeyError, TypeError):
        raise ValueError(
            "new HarnessAudit freezes require the corrected offline model-input "
            "boundary identity"
        ) from None
    if not isinstance(train_policy, dict) or not isinstance(prompt_versions, dict):
        raise ValueError(
            "new HarnessAudit freezes require the corrected offline model-input "
            "boundary identity"
        )
    expected_train_policy = {
        "construction_version": HA_TRAIN_POLICY_CONSTRUCTION_VERSION,
        "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
        "policy_extraction_prompt_version": HA_POLICY_EXTRACTION_PROMPT_VERSION,
        "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
        "test_visible_fields": list(HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS),
        "input_includes_test_access_rules": False,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
    }
    mismatches = [
        f"train_policy.{field}"
        for field, expected_value in expected_train_policy.items()
        if train_policy.get(field) != expected_value
    ]
    if protocol.get("experience_replay") != {"enable": False}:
        mismatches.append("experience_replay")
    registered_rollouts = set(
        payload.get("offline_prediction", {})
        .get("rollout_modes_by_benchmark", {})
        .get("harnessaudit", [])
    )
    supported_prompt_versions = {
        "direct_multi_horizon": HA_DIRECT_PROMPT_VERSION,
        "iterative_wm_rollout": HA_ITERATIVE_PROMPT_VERSION,
    }
    expected_prompt_versions = {
        rollout: supported_prompt_versions[rollout]
        for rollout in registered_rollouts
        if rollout in supported_prompt_versions
    }
    mismatches.extend(
        f"model.prompt_versions.{field}"
        for field, expected_value in expected_prompt_versions.items()
        if prompt_versions.get(field) != expected_value
    )
    if mismatches:
        raise ValueError(
            "new HarnessAudit freeze requires a new no-test-tool-necessity "
            "offline protocol identity: "
            + ", ".join(mismatches)
        )


def _validate_ha_policy_extraction_model_binding(
    contract: dict[str, Any],
    stack: ResolvedModelStack,
) -> None:
    """Reject a freeze whose policy-extractor request identity can drift at run time."""

    policy_extractor = stack.roles["policy_extractor"]
    canonical_extra_body = json.dumps(
        policy_extractor.request_extra_body,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = {
        "max_tokens": policy_extractor.max_tokens,
        "timeout_seconds": float(policy_extractor.timeout_seconds),
        "transport_max_attempts": policy_extractor.transport_max_attempts,
        "request_extra_body_json": canonical_extra_body,
        # Model-stack execution always publishes the resolved request-extra
        # body environment variable, including an explicitly resolved empty
        # object. The runtime therefore disables the legacy implicit bounded-
        # thinking path and the frozen contract must record that behavior.
        "bounded_thinking_enabled": False,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if contract.get(field) != expected_value
        or (
            field == "bounded_thinking_enabled"
            and contract.get(field) is not expected_value
        )
    ]
    if mismatches:
        raise ValueError(
            "new HarnessAudit freeze policy_extraction_contract differs from "
            "the policy_extractor model profile: "
            + ", ".join(mismatches)
        )


def _validate_local_serving_sources(
    payload: dict[str, Any],
    *,
    benchmarks: Iterable[str],
    project_root: Path | None = None,
) -> None:
    """Bind every self-hosted endpoint to source and launcher bytes.

    A hostname such as ``gpu18`` is local serving even though it is not
    loopback.  IS-Bench v3 additionally uses a public HTTPS ingress, but its
    explicit serving schema still marks it as self-hosted.  The endpoint string
    alone is therefore never accepted as checkpoint/runtime evidence.
    """

    root = Path(
        Path(__file__).resolve().parents[1]
        if project_root is None
        else project_root
    ).resolve()
    for benchmark in tuple(dict.fromkeys(str(item) for item in benchmarks)):
        contract = payload["run_plans"][benchmark]["online_advisory_contract"]
        runtime = contract.get("runtime")
        if benchmark == "isbench" and isinstance(runtime, dict) and (
            "hardware_contract" in runtime
        ):
            _bind_isbench_hardware_source(
                contract,
                project_root=root,
            )
        role_identities = _online_contract_role_identities(contract)
        local_roles = {
            name: identity
            for name, identity in role_identities.items()
            if _is_local_serving_endpoint(
                str(identity.get("base_url") or identity.get("endpoint") or "")
            )
        }
        local_service_identities = {
            (
                str(identity.get("base_url") or identity.get("endpoint") or "").rstrip(
                    "/"
                ),
                str(identity.get("model") or ""),
            )
            for identity in local_roles.values()
        }
        if len(local_service_identities) > 1:
            raise ValueError(
                f"{benchmark} local model roles do not share one serving identity"
            )
        serving = runtime.get("serving_contract") if isinstance(runtime, dict) else None
        public_isbench_v3 = benchmark == "isbench" and _is_isbench_v3_serving_draft(
            serving,
            project_root=root,
        )
        if not local_roles and not public_isbench_v3:
            continue
        if not isinstance(serving, dict):
            raise ValueError(f"self-hosted {benchmark} serving contract is missing")
        source_relative = str(serving.get("source_path") or "")
        source_path = (root / source_relative).resolve()
        if (
            not source_relative
            or Path(source_relative).is_absolute()
            or root not in source_path.parents
            or not source_path.is_file()
        ):
            raise ValueError(
                f"self-hosted {benchmark} serving source is missing or outside project"
            )
        try:
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError(f"self-hosted {benchmark} serving source is invalid") from None
        if not isinstance(source_payload, dict):
            raise ValueError(f"self-hosted {benchmark} serving source must be an object")
        observed_identity = {
            "source_path": source_relative,
            "source_sha256": _sha256_file(source_path),
            **source_payload,
        }
        # A draft may carry only the immutable project-relative source pointer;
        # all other identity fields are derived from current bytes.  A fully
        # specified draft must already match exactly.
        if set(serving) != {"source_path"} and observed_identity != serving:
            raise ValueError(
                f"self-hosted {benchmark} serving source differs from draft identity"
            )
        source_bindings: list[tuple[dict[str, Any], str, str, str]] = [
            (
                observed_identity,
                "launcher_source_path",
                "launcher_source_sha256",
                f"{benchmark} launcher source",
            )
        ]
        if (
            benchmark == "isbench"
            and observed_identity.get("schema_version")
            == "isbench-vllm-serving-contract-v4"
        ):
            source_bindings.append(
                (
                    observed_identity,
                    "submitter_source_path",
                    "submitter_source_sha256",
                    "IS-Bench submitter source",
                )
            )
        if (
            benchmark == "isbench"
            and observed_identity.get("schema_version")
            == "isbench-vllm-serving-contract-v3"
        ):
            ingress = observed_identity.get("ingress")
            if not isinstance(ingress, dict):
                raise ValueError("IS-Bench v3 ingress source identity is missing")
            source_bindings.extend(
                (
                    (
                        ingress,
                        "gateway_source_path",
                        "gateway_source_sha256",
                        "IS-Bench gateway source",
                    ),
                    (
                        ingress,
                        "gateway_launcher_source_path",
                        "gateway_launcher_source_sha256",
                        "IS-Bench gateway launcher source",
                    ),
                )
            )
        for source_record, path_field, digest_field, label in source_bindings:
            relative = str(source_record.get(path_field) or "")
            candidate = (root / relative).resolve()
            if (
                not relative
                or Path(relative).is_absolute()
                or root not in candidate.parents
                or not candidate.is_file()
                or _sha256_file(candidate)
                != str(source_record.get(digest_field) or "").lower()
            ):
                raise ValueError(f"{label} SHA256 mismatch")
        if benchmark == "isbench":
            _validate_isbench_serving_source_identity(
                observed_identity,
                contract=contract,
                local_roles=local_roles,
            )
        elif local_roles:
            _validate_ha_serving_source_identity(
                observed_identity,
                local_roles=local_roles,
            )
        runtime["serving_contract"] = observed_identity


def _online_contract_role_identities(
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the five-role identity when present, with legacy fallback.

    The unified model-stack contract permits local and external roles to be
    mixed.  Older registrations only carried actor/world-model identities, so
    those remain readable without weakening new five-role validation.
    """

    model_stack = contract.get("model_stack") or {}
    roles = model_stack.get("roles") if isinstance(model_stack, dict) else None
    if isinstance(roles, dict) and roles:
        return {
            str(name): dict(identity)
            for name, identity in roles.items()
            if isinstance(identity, dict)
        }
    return {
        name: dict(identity)
        for name in ("actor", "world_model")
        if isinstance((identity := contract.get(name)), dict)
    }


def _validate_ha_serving_source_identity(
    serving: dict[str, Any],
    *,
    local_roles: dict[str, dict[str, Any]],
) -> None:
    served_model = str(serving.get("served_model_name") or "")
    if not served_model or any(
        str(identity.get("model") or "") != served_model
        for identity in local_roles.values()
    ):
        raise ValueError(
            "local HarnessAudit served model differs from local model-stack roles"
        )
    base_url = str(serving.get("base_url") or "").rstrip("/")
    if not base_url:
        host = str(serving.get("host") or "").strip()
        port = serving.get("port")
        if host and isinstance(port, int):
            base_url = f"http://{host}:{port}/v1"
    if not base_url or any(
        str(identity.get("base_url") or identity.get("endpoint") or "").rstrip("/")
        != base_url
        for identity in local_roles.values()
    ):
        raise ValueError(
            "local HarnessAudit serving endpoint differs from local model-stack roles"
        )


def _is_isbench_v3_serving_draft(
    serving: Any,
    *,
    project_root: Path,
) -> bool:
    """Recognize the explicit public ingress schema without classifying APIs.

    An ordinary public provider has no local serving source and must remain
    unbound.  A v3 source pointer, however, deliberately asserts a self-hosted
    gateway and must be frozen even when its public hostname is not private.
    """

    if not isinstance(serving, dict):
        return False
    if serving.get("schema_version") == "isbench-vllm-serving-contract-v3":
        return True
    source_relative = str(serving.get("source_path") or "")
    source_path = (project_root / source_relative).resolve()
    if (
        not source_relative
        or Path(source_relative).is_absolute()
        or project_root not in source_path.parents
        or not source_path.is_file()
    ):
        return False
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(source, dict) and (
        source.get("schema_version") == "isbench-vllm-serving-contract-v3"
    )


def _bind_isbench_hardware_source(
    contract: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Replace an IS-Bench hardware pointer with its exact frozen identity.

    The hardware profile is prospective infrastructure metadata, not benchmark
    evidence.  Freezing nevertheless binds the profile and both programs that
    generate its later empirical qualification evidence, so neither hardware
    eligibility nor its interpretation can change after preregistration.
    """

    root = Path(
        Path(__file__).resolve().parents[1]
        if project_root is None
        else project_root
    ).resolve()
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("IS-Bench runtime is missing")
    hardware = runtime.get("hardware_contract")
    if not isinstance(hardware, dict):
        raise ValueError("IS-Bench hardware contract is missing")
    source_relative = str(hardware.get("source_path") or "")
    source_path = (root / source_relative).resolve()
    if (
        not source_relative
        or Path(source_relative).is_absolute()
        or root not in source_path.parents
        or not source_path.is_file()
    ):
        raise ValueError(
            "IS-Bench hardware profile source is missing or outside project"
        )
    try:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("IS-Bench hardware profile source is invalid") from None
    if not isinstance(source_payload, dict):
        raise ValueError("IS-Bench hardware profile source must be an object")
    observed_identity = {
        "source_path": source_relative,
        "source_sha256": _sha256_file(source_path),
        **source_payload,
    }
    if set(hardware) != {"source_path"} and observed_identity != hardware:
        raise ValueError(
            "IS-Bench hardware profile source differs from draft identity"
        )
    errors = validate_isbench_hardware_contract(observed_identity)
    if errors:
        raise ValueError("IS-Bench hardware profile is invalid: " + "; ".join(errors))

    probe_contract = observed_identity["probe_contract"]
    for path_field, digest_field in (
        ("launcher_source_path", "launcher_source_sha256"),
        ("probe_python_source_path", "probe_python_source_sha256"),
    ):
        nested_relative = str(probe_contract.get(path_field) or "")
        nested_path = (root / nested_relative).resolve()
        expected_digest = str(probe_contract.get(digest_field) or "").lower()
        if (
            not nested_relative
            or Path(nested_relative).is_absolute()
            or root not in nested_path.parents
            or not nested_path.is_file()
            or _sha256_file(nested_path) != expected_digest
        ):
            raise ValueError(
                f"IS-Bench hardware profile {path_field} SHA256 mismatch"
            )

    runtime["hardware_contract"] = observed_identity
    return observed_identity


def _is_local_serving_endpoint(endpoint: str) -> bool:
    try:
        host = str(urlsplit(str(endpoint)).hostname or "").lower()
    except ValueError:
        return False
    if host in {"localhost", "::1"} or host.startswith("gpu"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return bool(re.fullmatch(r"(?:node|compute|slurm)[-_.]?\d+", host))
    return address.is_loopback or address.is_private or address.is_link_local


def _validate_isbench_serving_source_identity(
    serving: dict[str, Any],
    *,
    contract: dict[str, Any],
    local_roles: dict[str, dict[str, Any]] | None = None,
) -> None:
    source_payload_keys = {
        "schema_version",
        "backend",
        "backend_version",
        "model_path",
        "host",
        "advertised_host",
        "port",
        "base_url",
        "served_model_name",
        "slurm_node",
        "slurm_gpu_type",
        "slurm_gpu_count",
        "checkpoint_revision",
        "checkpoint_commit",
        "lfs_manifest_sha256",
        "runtime_freeze_sha256",
        "max_model_len",
        "dtype",
        "gpu_memory_utilization",
        "launcher_source_path",
        "launcher_source_sha256",
        "environment",
        "launch_arguments",
    }
    schema_version = serving.get("schema_version")
    if schema_version in {
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        source_payload_keys |= {
            "hardware_profile_source_path",
            "hardware_profile_source_sha256",
        }
    if schema_version == "isbench-vllm-serving-contract-v3":
        source_payload_keys.add("ingress")
    if schema_version == "isbench-vllm-serving-contract-v4":
        source_payload_keys |= {
            "submitter_source_path",
            "submitter_source_sha256",
        }
    if set(serving) != {"source_path", "source_sha256", *source_payload_keys}:
        raise ValueError("local IS-Bench serving source has unexpected fields")
    if schema_version not in {
        "isbench-vllm-serving-contract-v1",
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        raise ValueError("local IS-Bench serving schema is invalid")
    hardware = (contract.get("runtime") or {}).get("hardware_contract")
    if schema_version in {
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        errors = validate_isbench_hardware_contract(hardware)
        if errors:
            raise ValueError(
                "local IS-Bench hardware contract is invalid: " + "; ".join(errors)
            )
        if (
            serving.get("hardware_profile_source_path")
            != hardware.get("source_path")
            or serving.get("hardware_profile_source_sha256")
            != hardware.get("source_sha256")
        ):
            raise ValueError(
                "local IS-Bench serving source differs from hardware profile"
            )
    elif hardware is not None:
        raise ValueError(
            "local IS-Bench v1 serving source cannot bind a hardware profile"
        )
    # A mixed stack can bind a self-hosted actor/evaluator/policy service while
    # sending only the world-model role to an external provider.  Validate the
    # service against the roles that actually resolve to that local endpoint.
    # The fallback preserves legacy public-ingress contracts whose HTTPS URL
    # is self-hosted but is not classified as a private/local address.
    bound_roles = dict(local_roles or {})
    if not bound_roles:
        bound_roles = {
            name: dict(identity)
            for name in ("actor", "world_model")
            if isinstance((identity := contract.get(name)), dict)
        }
    endpoint = str(serving.get("base_url") or "").rstrip("/")
    if any(
        str(
            identity.get("base_url") or identity.get("endpoint") or ""
        ).rstrip("/")
        != endpoint
        for identity in bound_roles.values()
    ):
        raise ValueError("local IS-Bench serving endpoint differs from model profile")
    if any(
        identity.get("model") != serving.get("served_model_name")
        for identity in bound_roles.values()
    ):
        raise ValueError("local IS-Bench served model differs from model profile")
    if any(
        identity.get("revision") != serving.get("checkpoint_revision")
        for identity in bound_roles.values()
    ):
        raise ValueError("local IS-Bench checkpoint revision differs from model profile")
    model_stack = contract.get("model_stack") or {}
    roles = model_stack.get("roles") or {}
    runtime_roles = (
        [roles[name] for name in bound_roles if isinstance(roles.get(name), dict)]
        if local_roles
        else [role for role in roles.values() if isinstance(role, dict)]
    )
    runtime_digest = str(serving.get("runtime_freeze_sha256") or "")
    expected_runtime_suffix = f"pip-freeze-sha256:{runtime_digest}"
    if not runtime_roles or any(
        not str(role.get("runtime_revision") or "").endswith(expected_runtime_suffix)
        for role in runtime_roles
    ):
        raise ValueError("local IS-Bench vLLM runtime differs from model profile")


def _validate_ha_task_cohort_source(
    payload: dict[str, Any], *, benchmarks: Iterable[str]
) -> None:
    """Bind a new HA freeze to the independently audited cohort source."""

    if "harnessaudit" not in set(benchmarks):
        return
    try:
        plan = payload["run_plans"]["harnessaudit"]
        contract = plan["task_cohort_contract"]
    except (KeyError, TypeError):
        raise ValueError(
            "new HarnessAudit freezes require task_cohort_contract"
        ) from None
    if not isinstance(contract, dict):
        raise ValueError("new HarnessAudit freezes require task_cohort_contract")
    project_root = Path(__file__).resolve().parents[1]
    source_relative = str(contract.get("source_path") or "")
    source_path = (project_root / source_relative).resolve()
    if (
        not source_relative
        or Path(source_relative).is_absolute()
        or project_root not in source_path.parents
        or not source_path.is_file()
    ):
        raise ValueError(
            "HarnessAudit task cohort source is missing or outside project"
        )
    validated = validate_ha_online_task_cohort_manifest(
        source_path,
        project_root=project_root,
        expected_contract=contract,
        require_project_relative=True,
    )
    plan_ids = sorted(Path(str(item)).stem for item in plan.get("tasks") or [])
    if plan_ids != validated["selected_task_ids"]:
        raise ValueError(
            "HarnessAudit run-plan tasks differ from validated task cohort"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze one joint formal preregistration from a user-edited draft"
    )
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-profile", required=True, type=Path)
    parser.add_argument("--compositionality-manifest", required=True, type=Path)
    parser.add_argument(
        "--faithfulness-adjudication-protocol", required=True, type=Path
    )
    parser.add_argument("--adjudicator-blinding-protocol", required=True, type=Path)
    parser.add_argument("--multiplicity-analysis-code", required=True, type=Path)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument(
        "--isbench-task-list",
        type=Path,
        help=(
            "Exact online IS-Bench cohort; required whenever isbench is in "
            "--benchmarks."
        ),
    )
    parser.add_argument(
        "--isbench-root",
        type=Path,
        default=project_root / "IS-Bench",
        help="IS-Bench source root used to bind task/default-scene identities.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
        help=(
            "Exact benchmark scope to freeze. The default remains the joint HA + "
            "IS-Bench registration; use harnessaudit for an HA-only component."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = freeze_formal_preregistration(
        draft_path=args.draft,
        output_path=args.output,
        model_profile_path=args.model_profile,
        compositionality_manifest_path=args.compositionality_manifest,
        faithfulness_adjudication_protocol_path=(
            args.faithfulness_adjudication_protocol
        ),
        adjudicator_blinding_protocol_path=args.adjudicator_blinding_protocol,
        multiplicity_analysis_code_path=args.multiplicity_analysis_code,
        frozen_at_utc=args.frozen_at_utc,
        benchmarks=args.benchmarks,
        isbench_task_list_path=args.isbench_task_list,
        isbench_root=args.isbench_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
