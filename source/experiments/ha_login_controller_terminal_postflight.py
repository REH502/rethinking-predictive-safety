#!/usr/bin/env python3
"""Terminal postflight for the explicit HA login-controller topology.

This is intentionally separate from ``ha_formal_terminal_postflight``'s
SLURM-controller schema.  A Qwen vLLM service can be a SLURM job while the
formal controller remains on the login host for external Gemini egress; this
validator proves that topology rather than pretending the service allocation
was the controller.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import uuid

from experiments import ha_formal_terminal_postflight as base
from experiments.ha_login_controller_registry import (
    EXTERNAL_SERVICE_OWNERSHIP,
    LOGIN_CONTROLLER_STATUS_SCHEMA,
    LoginControllerRegistryError,
    validate_login_controller_registry,
)


LOGIN_AUDIT_SCHEMA = "ha-formal-login-controller-terminal-postflight-v1"
LOGIN_EVIDENCE_SCHEMA = "ha-formal-login-controller-terminal-evidence-manifest-v1"


def _fail(code: str) -> None:
    base._fail(code)


def _validate_expected_controller_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        _fail("expected_login_controller_id_invalid")
    if str(parsed) != value.lower():
        _fail("expected_login_controller_id_invalid")
    return str(parsed)


def _audit_login_status(
    row: Mapping[str, Any],
    *,
    expected_condition_count: int,
    safety_abstained_count: int = 0,
    terminal_scoring_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the login-controller mirror of the frozen terminal gate."""

    registered_contract = base._terminal_scoring_contract(terminal_scoring_contract)
    if safety_abstained_count and registered_contract is None:
        _fail(base.SAFETY_ABSTAINED_POSTFLIGHT_CODE)
    expected_counts = base._expected_terminal_status_counts(
        condition_count=expected_condition_count,
        safety_abstained_count=safety_abstained_count,
    )
    status = base._read_json(
        Path(str(row["paths"]["invocation_status"])),
        code="login_controller_status_invalid",
    )
    if not isinstance(status, dict):
        _fail("login_controller_status_invalid")
    controller = row["controller"]
    service = row["actor_service"]
    observed_counts = status.get("status_counts")
    try:
        started = base._parse_utc_timestamp(
            status.get("controller_started_at_utc"),
            code="login_controller_status_invalid",
        )
        finished = base._parse_utc_timestamp(
            status.get("controller_finished_at_utc"),
            code="login_controller_status_invalid",
        )
    except base.PostflightFailure:
        raise
    observed_service = status.get("actor_service")
    if not isinstance(observed_service, dict):
        _fail("login_controller_status_invalid")
    expected_service = {
        **service,
        "observed_state_at_finish": "RUNNING",
    }
    if (
        status.get("schema_version") != LOGIN_CONTROLLER_STATUS_SCHEMA
        or status.get("controller_id") != controller["id"]
        or status.get("controller_pid") != controller["pid"]
        or status.get("controller_hostname") != controller["hostname"]
        or status.get("controller_started_at_utc") != controller["started_at_utc"]
        or finished < started
        or status.get("controller_process_finished") is not True
        or status.get("controller_exit_status") != 0
        or status.get("launcher_phase") != "formal_complete"
        or status.get("resumable_incomplete") is not False
        or observed_service != expected_service
        or status.get("service_process_ownership") != EXTERNAL_SERVICE_OWNERSHIP
        or status.get("service_stopped_by_controller") is not False
        or status.get("total_condition_count") != expected_condition_count
        or status.get("terminal_condition_count") != expected_condition_count
        or status.get("pending_condition_count") != 0
        or observed_counts != expected_counts
        or status.get("formal_candidate_eligible")
        is not (safety_abstained_count == 0)
        or (
            safety_abstained_count
            and status.get("itt_fail_closed_candidate_eligible") is not True
        )
        or status.get("registry_record_sha256") != row["record_sha256"]
    ):
        _fail("login_controller_status_not_formal_complete")
    outer = base._read_json(
        Path(str(row["paths"]["outer_credential_scan"])),
        code="login_controller_outer_credential_scan_invalid",
    )
    if (
        not isinstance(outer, dict)
        or outer.get("status") != "passed"
        or outer.get("leak_path_count") != 0
        or outer.get("missing_root_count", 0) != 0
        or not isinstance(outer.get("scanned_file_count"), int)
        or outer["scanned_file_count"] < 1
    ):
        _fail("login_controller_outer_credential_scan_invalid")
    result: dict[str, Any] = {
        "controller_id": str(controller["id"]),
        "controller_finished_at_utc": str(status["controller_finished_at_utc"]),
        "runner_exit_status": 0,
        "outer_scan_status": "passed",
        "outer_scan_scanned_file_count": int(outer["scanned_file_count"]),
    }
    if safety_abstained_count:
        assert registered_contract is not None
        result["itt_fail_closed_candidate_eligible"] = True
        result["itt_endpoint"] = registered_contract["itt_endpoint"]
    return result

def _audit_controller_lifecycle_logs(
    row: Mapping[str, Any], *, formal_root: Path
) -> dict[str, Any]:
    """Require bounded public lifecycle records, never raw controller output."""

    controller = row["controller"]
    service = row["actor_service"]
    expected_common = {
        "schema_version": "ha-login-controller-structured-lifecycle-v1",
        "capture_policy": "structured_public_lifecycle_only_v1",
        "controller_id": controller["id"],
        "controller_started_at_utc": controller["started_at_utc"],
        "formal_root": str(formal_root),
        "service_slurm_job_id": service["slurm_job_id"],
    }
    observed: dict[str, dict[str, Any]] = {}
    for name, expected_event in (
        ("controller_stdout", "registered"),
        ("controller_stderr", "finished"),
    ):
        rows = base._read_jsonl(
            Path(str(row["paths"][name])), code="login_controller_lifecycle_log_invalid"
        )
        if len(rows) != 1:
            _fail("login_controller_lifecycle_log_invalid")
        payload = rows[0]
        allowed = set(expected_common) | {"event", "written_at_utc"}
        if expected_event == "finished":
            allowed |= {"runner_exit_status", "service_state_at_finish"}
        if set(payload) != allowed or any(
            payload.get(key) != value for key, value in expected_common.items()
        ):
            _fail("login_controller_lifecycle_log_invalid")
        try:
            written = base._parse_utc_timestamp(
                payload.get("written_at_utc"), code="login_controller_lifecycle_log_invalid"
            )
            started = base._parse_utc_timestamp(
                controller["started_at_utc"], code="login_controller_lifecycle_log_invalid"
            )
        except base.PostflightFailure:
            raise
        if written < started or payload.get("event") != expected_event:
            _fail("login_controller_lifecycle_log_invalid")
        if expected_event == "finished" and (
            payload.get("runner_exit_status") != 0
            or payload.get("service_state_at_finish") != "RUNNING"
        ):
            _fail("login_controller_lifecycle_log_invalid")
        observed[name] = {
            "event": expected_event,
            "written_at_utc": str(payload["written_at_utc"]),
        }
    return observed


def _audit_preregistration_from_registry(
    formal_root: Path,
    *,
    registry_row: Mapping[str, Any],
    identity: Mapping[str, Any],
    expected_study_id: str,
    expected_task_count: int,
) -> dict[str, Any]:
    from multi_agent.safepred_experiments.formal_contract import (
        canonical_preregistration_sha256,
        validate_preregistration,
    )

    prereg_path = Path(
        str(registry_row["immutable_input_bindings"]["preregistration"]["path"])
    )
    registered = base._read_json(prereg_path, code="login_controller_preregistration_missing")
    frozen = base._read_json(
        formal_root / "harnessaudit/analysis/formal_preregistration.json",
        code="formal_preregistration_missing",
    )
    if not isinstance(registered, dict) or registered != frozen:
        _fail("login_controller_preregistration_copy_drift")
    try:
        validate_preregistration(registered, benchmark="harnessaudit")
    except ValueError:
        _fail("preregistration_invalid")
    if registered.get("study_id") != expected_study_id:
        _fail("preregistration_study_identity_mismatch")
    runtime = (
        registered.get("run_plans", {})
        .get("harnessaudit", {})
        .get("online_advisory_contract", {})
        .get("runtime", {})
    )
    if not isinstance(runtime, dict):
        _fail("preregistration_runtime_invalid")
    terminal_scoring_contract = base._terminal_scoring_contract(
        runtime.get("terminal_scoring_contract")
    )
    if identity.get("schema_version") == "ha-online-formal-resume-identity-v2":
        for field in (
            "execution_workers",
            "execution_schedule_version",
            "execution_lane_count",
            "execution_lane_assignment_version",
            "execution_lane_plan_sha256",
        ):
            if identity.get(field) != runtime.get(field):
                _fail("preregistration_resume_lane_binding_mismatch")
    elif runtime.get("execution_workers") != 1:
        _fail("preregistration_resume_worker_binding_mismatch")
    digest = canonical_preregistration_sha256(registered)
    if digest != identity.get("preregistration_sha256"):
        _fail("preregistration_resume_binding_mismatch")
    source_manifest = base._read_json(
        formal_root / "harnessaudit/analysis/formal_online_source_manifest.json",
        code="formal_source_manifest_missing",
    )
    if not isinstance(source_manifest, dict):
        _fail("formal_source_manifest_invalid")
    source_digest = base._sha256_bytes(
        json.dumps(
            source_manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    if source_digest != identity.get("source_manifest_sha256"):
        _fail("formal_source_manifest_binding_mismatch")
    task_files = source_manifest.get("task_files")
    if not isinstance(task_files, list) or len(task_files) != expected_task_count:
        _fail("formal_source_manifest_task_count_invalid")
    task_ids = [str(row.get("task_id") or "") for row in task_files if isinstance(row, dict)]
    if len(task_ids) != expected_task_count or len(set(task_ids)) != expected_task_count:
        _fail("formal_source_manifest_task_identity_invalid")
    result: dict[str, Any] = {
        "preregistration_sha256": digest,
        "source_manifest_sha256": source_digest,
        "task_count": len(task_ids),
        "task_ids": sorted(task_ids),
    }
    if terminal_scoring_contract is not None:
        result["terminal_scoring_contract"] = terminal_scoring_contract
    return result


def _audit_external_service(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = row["immutable_input_bindings"]
    contract_path = Path(str(bindings["serving_contract"]["path"]))
    service_info_path = Path(str(bindings["service_info"]["path"]))
    contract = base._read_json(contract_path, code="login_controller_serving_contract_invalid")
    service_info = base._read_json(service_info_path, code="login_controller_service_info_invalid")
    if not isinstance(contract, dict) or not isinstance(service_info, dict):
        _fail("login_controller_serving_contract_invalid")
    service = row["actor_service"]
    arguments = contract.get("launch_arguments")
    checkpoint_path = Path(str(row["paths"]["service_checkpoint_verification"]))
    gpu_path = Path(str(row["paths"]["service_gpu_info"]))
    vllm_log = Path(str(row["paths"]["service_vllm_log"]))
    try:
        base_url = f"http://{contract['host']}:{contract['port']}/v1"
    except (KeyError, TypeError, ValueError):
        _fail("login_controller_serving_contract_invalid")
    if (
        contract.get("schema_version") != "vllm-serving-contract-v1"
        or contract.get("backend") != "vllm"
        or contract.get("host") != service["slurm_node"]
        or not isinstance(contract.get("port"), int)
        or base_url != service["base_url"]
        or not isinstance(arguments, list)
        or base._launch_argument_value(arguments, "--max-model-len")
        != str(base.EXPECTED_MODEL_CONTEXT_TOKENS)
        or service_info.get("slurm_job_id") != service["slurm_job_id"]
        or service_info.get("slurm_node") != service["slurm_node"]
        or service_info.get("base_url") != service["base_url"]
        or str(service_info.get("checkpoint_report_path") or "") != str(checkpoint_path)
        or str(service_info.get("gpu_info_path") or "") != str(gpu_path)
        or not vllm_log.is_file()
        or vllm_log.parent != service_info_path.parent
        or service_info.get("checkpoint_report_sha256") != base._sha256_file(checkpoint_path)
        or service_info.get("gpu_info_sha256") != base._sha256_file(gpu_path)
        or service_info.get("launcher_sha256") != contract.get("launcher_source_sha256")
    ):
        _fail("login_controller_external_service_identity_invalid")
    return {
        "topology": "login_controller_external_slurm_vllm_service_v1",
        "service_slurm_job_id": str(service["slurm_job_id"]),
        "service_node": str(service["slurm_node"]),
        "base_url": str(service["base_url"]),
        "served_model_name": str(contract.get("served_model_name") or ""),
        "max_model_len": base.EXPECTED_MODEL_CONTEXT_TOKENS,
        "service_end_epoch_seconds": int(service["service_end_epoch_seconds"]),
        "invocation_safety_margin_seconds": int(
            service["invocation_safety_margin_seconds"]
        ),
    }


def run_login_controller_terminal_postflight(
    *,
    formal_root: Path,
    controller_root: Path,
    promotion_root: Path,
    expected_login_controller_id: str,
    expected_study_id: str,
    postflight_launcher: Path,
    stability_window_seconds: float = 2.0,
    sleeper: Callable[[float], None] = __import__("time").sleep,
    supplied_credentials: Sequence[bytes] | None = None,
) -> dict[str, Any]:
    """Audit one explicit login-controller run without any scheduler coercion."""

    controller_id = _validate_expected_controller_id(expected_login_controller_id)
    if not expected_study_id.strip():
        _fail("expected_study_id_missing")
    if stability_window_seconds < 0:
        _fail("stability_window_invalid")
    runs_root = base._existing_directory(base.RUNS_ROOT, code="runs_root_missing")
    formal = base._existing_directory(formal_root, code="formal_root_missing")
    controller = base._existing_directory(controller_root, code="controller_root_missing")
    launcher = base._regular_file(
        postflight_launcher, code="postflight_launcher_missing", allow_empty=False
    )
    if formal.parent != runs_root or controller.parent != runs_root:
        _fail("input_roots_must_be_direct_children_of_runs_root")
    promotion = base._reject_symlink_components(promotion_root, allow_missing_leaf=True)
    if promotion.exists():
        _fail("promotion_root_already_exists")
    registry_path = controller / "invocation_registry.jsonl"
    try:
        rows = validate_login_controller_registry(
            registry_path,
            runs_root=runs_root,
            project_root=base.PROJECT_ROOT,
            require_paths_exist=True,
        )
    except LoginControllerRegistryError as exc:
        _fail(exc.code)
    if (
        len(rows) != 1
        or rows[-1]["controller"]["id"] != controller_id
        or rows[-1].get("study_id") != expected_study_id
    ):
        _fail("expected_login_controller_invocation_not_unique_tail")
    row = rows[-1]
    paths = row["paths"]
    lock_paths = {
        formal.parent / f".{formal.name}.ha_online_active.lock",
        Path(str(paths["launcher_lock"])),
    }
    with ExitStack() as locks:
        for lock_path in sorted(lock_paths):
            locks.enter_context(base._existing_exclusive_lock(lock_path))
        stable_paths = [
            Path(str(paths["controller_stdout"])),
            Path(str(paths["controller_stderr"])),
            Path(str(paths["service_vllm_log"])),
            Path(str(paths["service_checkpoint_verification"])),
            Path(str(paths["service_gpu_info"])),
            Path(str(paths["invocation_status"])),
            Path(str(paths["outer_credential_scan"])),
            registry_path,
        ]
        stable = base._stable_bindings(
            stable_paths,
            stability_window_seconds=stability_window_seconds,
            sleeper=sleeper,
        )
        plan, terminal, identity, denominators = base._audit_resume_state(
            formal, expected_study_id=expected_study_id
        )
        prereg = _audit_preregistration_from_registry(
            formal,
            registry_row=row,
            identity=identity,
            expected_study_id=expected_study_id,
            expected_task_count=int(denominators["task_count"]),
        )
        terminal_scoring_contract = prereg.pop("terminal_scoring_contract", None)
        terminal_matrix = base._audit_terminal_matrix_state(
            plan,
            terminal=terminal,
            expected_condition_count=int(denominators["condition_count"]),
            terminal_scoring_contract=terminal_scoring_contract,
        )
        status = _audit_login_status(
            row,
            expected_condition_count=int(denominators["condition_count"]),
            safety_abstained_count=int(
                terminal_matrix["safety_abstained_condition_count"]
            ),
            terminal_scoring_contract=terminal_scoring_contract,
        )
        controller_lifecycle_logs = _audit_controller_lifecycle_logs(
            row, formal_root=formal
        )
        policy = base._audit_policy_preparation(
            formal, expected_task_ids=prereg.pop("task_ids")
        )
        analysis = formal / "harnessaudit" / "analysis"
        manifest = base._audit_run_manifest(
            analysis / "run_manifest.jsonl",
            terminal=terminal,
            denominators=denominators,
            terminal_scoring_contract=terminal_scoring_contract,
        )
        lifecycle = base._audit_lifecycle(
            analysis / "formal_run_lifecycle.jsonl",
            expected_condition_count=int(denominators["condition_count"]),
            safety_abstained_count=int(
                terminal_matrix["safety_abstained_condition_count"]
            ),
            terminal_scoring_contract=terminal_scoring_contract,
        )
        itt_fail_closed: dict[str, Any] | None = None
        itt_artifact_path: Path | None = None
        if terminal_matrix["safety_abstained_condition_count"]:
            itt_artifact_path = analysis / base.ITT_FAIL_CLOSED_ARTIFACT_FILENAME
            itt_fail_closed = base._audit_tcr_itt_fail_closed_artifact(
                itt_artifact_path,
                terminal=terminal,
                expected_condition_count=int(denominators["condition_count"]),
                terminal_scoring_contract=terminal_scoring_contract,
            )
        provenance = base._audit_provenance(
            analysis / "online_matrix_provenance_manifest.json",
            formal_root=formal,
            expected_condition_count=int(denominators["condition_count"]),
            required_artifact_paths=(
                [itt_artifact_path] if itt_artifact_path is not None else []
            ),
        )
        serving = _audit_external_service(row)
        checkpoint = base._audit_checkpoint(
            Path(str(paths["service_checkpoint_verification"]))
        )
        internal_scan = base._read_json(
            analysis / "active_credential_scan.json",
            code="internal_credential_scan_invalid",
        )
        if (
            not isinstance(internal_scan, dict)
            or internal_scan.get("status") != "passed"
            or internal_scan.get("leak_path_count") != 0
            or internal_scan.get("missing_root_count") != 0
        ):
            _fail("internal_credential_scan_invalid")
        if supplied_credentials is None:
            secrets, credential_source = base._credential_values()
        else:
            secrets = sorted({bytes(value) for value in supplied_credentials if len(value) >= 12})
            if not secrets:
                _fail("active_credential_set_empty")
            credential_source = {
                "source_exists": True,
                "source_owner_only": True,
                "active_credential_count": len(secrets),
            }
        scan_files = base._inventory_scan_paths(
            [formal, controller, *[Path(str(value)) for value in paths.values()]],
            runs_root=runs_root,
        )
        evidence_bindings, credential_scan = base._scan_and_bind(scan_files, secrets=secrets)
        if credential_scan["status"] != "passed":
            _fail("terminal_active_credential_scan_failed")

    implementation = [
        base._binding(Path(__file__), code="postflight_implementation_missing", allow_empty=False),
        base._binding(launcher, code="postflight_launcher_missing", allow_empty=False),
    ]
    evidence: dict[str, Any] = {
        "schema_version": LOGIN_EVIDENCE_SCHEMA,
        "created_at_utc": base._utc_now(),
        "study_id": expected_study_id,
        "controller_topology": "login_controller_external_slurm_vllm_service_v1",
        "login_controller": {
            "id": controller_id,
            "pid": int(row["controller"]["pid"]),
            "hostname": str(row["controller"]["hostname"]),
        },
        "formal_root": str(formal),
        "controller_root": str(controller),
        "stable_terminal_files": [binding.as_dict() for binding in stable],
        "implementation": [binding.as_dict() for binding in implementation],
        "scanned_files": [binding.as_dict() for binding in evidence_bindings],
    }
    audit: dict[str, Any] = {
        "schema_version": LOGIN_AUDIT_SCHEMA,
        "artifact_type": "ha_formal_login_controller_terminal_promotion_audit",
        "created_at_utc": base._utc_now(),
        "study_id": expected_study_id,
        "login_controller": {
            "id": controller_id,
            "registry_sha256": str(row["record_sha256"]),
            "structured_lifecycle_logs": controller_lifecycle_logs,
            **status,
        },
        "terminal_postflight_status": "passed",
        "formal_matrix_candidate": True,
        "scientific_claim_classification": "unevaluable",
        "scientific_claim_reason": "pending_blinded_decision_faithfulness_postflight",
        "condition_plan": {
            "total_condition_count": int(plan["total_condition_count"]),
            "terminal_condition_count": int(plan["terminal_condition_count"]),
            "pending_condition_count": int(plan["pending_condition_count"]),
            "status_counts": dict(plan["status_counts"]),
            "history_line_count": int(plan["history_line_count"]),
            "history_tail_sha256": str(plan["history_tail_sha256"]),
        },
        "run_manifest": manifest,
        "terminal_matrix": terminal_matrix,
        "policy_preparation": policy,
        "lifecycle": lifecycle,
        "provenance": provenance,
        "preregistration_and_source": prereg,
        "serving": serving,
        "model_checkpoint": checkpoint,
        "credential_source": credential_source,
        "terminal_active_credential_scan": credential_scan,
        "promotion_gate": {
            "expected_conditions": int(denominators["condition_count"]),
            "expected_tasks": int(denominators["task_count"]),
            "expected_arms": list(denominators["arms"]),
            "expected_arm_counts": dict(denominators["arm_counts"]),
            "wm_return_perturbations": list(denominators["wm_return_perturbations"]),
            "expected_seeds": [0],
            "matrix_outputs_validated": True,
            "provenance_complete": True,
            "active_credential_scan_passed": True,
            "leak_path_count": 0,
            "missing_path_count": 0,
        },
    }
    if itt_fail_closed is not None:
        audit["safety_abstention_itt"] = itt_fail_closed
        audit["promotion_gate"]["safety_abstention_itt_endpoint"] = (
            itt_fail_closed["itt_endpoint"]
        )
        audit["promotion_gate"]["raw_tcr_status"] = itt_fail_closed[
            "raw_tcr_status"
        ]
    artifacts = base._write_promotion_root(promotion, evidence=evidence, audit=audit)
    return {
        "status": "passed",
        "formal_matrix_candidate": True,
        "promotion_root": str(promotion),
        "artifacts": {name: binding.as_dict() for name, binding in artifacts.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--promotion-root", type=Path, required=True)
    parser.add_argument("--expected-login-controller-id", required=True)
    parser.add_argument("--expected-study-id", required=True)
    parser.add_argument("--postflight-launcher", type=Path, required=True)
    parser.add_argument("--stability-window-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        result = run_login_controller_terminal_postflight(
            formal_root=args.formal_root,
            controller_root=args.controller_root,
            promotion_root=args.promotion_root,
            expected_login_controller_id=args.expected_login_controller_id,
            expected_study_id=args.expected_study_id,
            postflight_launcher=args.postflight_launcher,
            stability_window_seconds=args.stability_window_seconds,
        )
    except base.PostflightFailure as exc:
        raise SystemExit(f"HA login-controller terminal postflight failed: {exc.code}") from None
    print(json.dumps({"status": result["status"], "promotion_root": result["promotion_root"]}, sort_keys=True))


if __name__ == "__main__":
    main()
