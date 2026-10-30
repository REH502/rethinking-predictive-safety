"""Pre-execution fail-closed readiness gate for the joint HA + IS-Bench study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from experiments.model_stack import load_model_stack_profile
from experiments.wm_return_perturbations import (
    DISTRIBUTED_ENFORCED,
    WM_RETURN_PERTURBATIONS,
)
from multi_agent.safepred_experiments.formal_contract import (
    current_online_implementation_manifest_sha256,
    isbench_formal_condition_order,
    isbench_formal_condition_order_sha256,
    isbench_omnigibson_hardware_identity,
    load_preregistration,
    validate_annotation_contract_files,
    validate_isbench_hardware_contract,
)
from scripts.probe_omnigibson_gpu import _empirical_result_failure_reason


BENCHMARKS = ("harnessaudit", "isbench")

_ISBENCH_HARDWARE_QUALIFICATION_SCHEMAS = {
    "omnigibson_a10040_empirical_v1": {
        "gpu_identity": "isbench-a10040-heterogeneous-gpu-identity-v1",
        "preregistration_verification": (
            "isbench-a10040-preregistration-verification-v1"
        ),
        "serving_contract_verification": (
            "isbench-a10040-serving-contract-verification-v1"
        ),
        "hardware_probe_verification": (
            "isbench-a10040-hardware-probe-verification-v1"
        ),
        "slurm_window": "isbench-a10040-qualification-slurm-window-v1",
    },
    "omnigibson_rtx_a6000_empirical_v1": {
        "gpu_identity": (
            "isbench-a6000-a100sxm-heterogeneous-gpu-identity-v1"
        ),
        "preregistration_verification": (
            "isbench-a6000-preregistration-verification-v1"
        ),
        "serving_contract_verification": (
            "isbench-a6000-serving-contract-verification-v1"
        ),
        "hardware_probe_verification": (
            "isbench-a6000-hardware-probe-verification-v1"
        ),
        "slurm_window": "isbench-a6000-qualification-slurm-window-v1",
    },
    "omnigibson_rtx_a6000_empirical_v2": {
        "gpu_identity": (
            "isbench-a6000-a100sxm-heterogeneous-gpu-identity-v1"
        ),
        "preregistration_verification": (
            "isbench-a6000-preregistration-verification-v1"
        ),
        "serving_contract_verification": (
            "isbench-a6000-serving-contract-verification-v1"
        ),
        "hardware_probe_verification": (
            "isbench-a6000-hardware-probe-verification-v1"
        ),
        "slurm_window": "isbench-a6000-qualification-slurm-window-v1",
    },
}

_ISBENCH_QWEN_SERVING_GPU_IDENTITIES = {
    "nvidia_a100_80gb_pcie": {
        "name": "NVIDIA A100 80GB PCIe",
        "minimum_memory_mib": 80000,
    },
    "nvidia_a100_80gb_sxm4": {
        "name": "NVIDIA A100-SXM4-80GB",
        "minimum_memory_mib": 80000,
    },
}

_ISBENCH_V4_SERVING_SCHEMA = "isbench-vllm-serving-contract-v4"
_ISBENCH_V4_QUALIFICATION_SCHEMA = "isbench-online-qualification-v3"
_ISBENCH_SERVICE_HEALTH_KEYS = {
    "http_4xx",
    "http_5xx",
    "oom",
    "traceback",
    "cuda_xid",
    "segfault",
    "engine_dead",
    "timeout",
}
_ISBENCH_V4_SERVICE_BINDING_NAMES = {
    "service_ready",
    "service_active_credential_scan_at_ready",
    "service_startup_health_summary",
    "service_vllm_log_snapshot",
    "service_runtime_health",
    "service_active_credential_scan_terminal",
    "advisory_active_path_code_verification",
}
_ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES = {
    "submission_identity_manifest",
    "submission_identity_verification_preflight",
    "submission_identity_verification_terminal",
}
_ISBENCH_SUBMISSION_IDENTITY_INPUT_NAMES = (
    "submitted_launcher_copy",
    "launcher_source",
    "submitter_source",
    "formal_preregistration",
    "serving_contract",
    "hardware_profile",
    "model_profile",
    "formal_task_list",
    "qualification_task_list",
    "compositionality_manifest",
    "faithfulness_protocol",
    "blinding_protocol",
    "probe_launcher_source",
    "submitted_probe_launcher_copy",
    "probe_python_source",
    "workflow_entrypoint",
    "registered_probe_summary",
    "shared_service_launcher",
    "shared_service_info",
    "shared_service_checkpoint",
    "shared_service_gpu_info",
)
_ISBENCH_ADVISORY_ACTIVE_PATH_TEST_IDS = [
    "harness_env/orgin/HarnessAudit/tests/test_safepred_guard.py::"
    "test_advisory_adapter_exposes_consultation_to_every_actor_and_"
    "releases_unconsulted_final",
    "harness_env/orgin/HarnessAudit/tests/test_safepred_guard.py::"
    "test_consult_safepred_evaluates_exact_candidate_without_action_side_effect",
    "harness_env/orgin/HarnessAudit/tests/test_safepred_guard.py::"
    "test_explicit_consultation_bypasses_prediction_interval_and_evaluates_once",
    "harness_env/orgin/HarnessAudit/tests/test_safepred_guard.py::"
    "test_advisory_consultation_links_only_same_caller_next_action",
]
_ISBENCH_ADVISORY_ACTIVE_PATH_SOURCE_FILES = [
    "harness_env/orgin/HarnessAudit/multi_agent/frameworks/oai/tools.py",
    "harness_env/orgin/HarnessAudit/multi_agent/frameworks/oai/adapter.py",
    "harness_env/orgin/HarnessAudit/multi_agent/frameworks/oai/prompts.py",
    "harness_env/orgin/HarnessAudit/tests/test_safepred_guard.py",
]
_ISBENCH_SERVICE_LOG_PATTERNS = {
    "http_4xx": re.compile(
        r'(?:"\s+4\d{2}\s+|status(?:_code)?\s*[:=]\s*4\d{2}\b)',
        re.IGNORECASE,
    ),
    "http_5xx": re.compile(
        r'(?:"\s+5\d{2}\s+|status(?:_code)?\s*[:=]\s*5\d{2}\b)',
        re.IGNORECASE,
    ),
    "oom": re.compile(
        r"(?:cuda\s+out\s+of\s+memory|outofmemoryerror|oom_kill|\boom\b)",
        re.IGNORECASE,
    ),
    "traceback": re.compile(
        r"^traceback \(most recent call last\):",
        re.IGNORECASE | re.MULTILINE,
    ),
    "cuda_xid": re.compile(
        r"(?:nvrm:\s*)?xid(?:\s*\(pci:[^)]+\))?\s*[: ]\s*\d+",
        re.IGNORECASE,
    ),
    "segfault": re.compile(
        r"\b(?:segmentation fault|segfault|sigsegv)\b",
        re.IGNORECASE,
    ),
    "engine_dead": re.compile(
        r"\b(?:enginedeaderror|engine(?: process)? (?:is )?"
        r"(?:dead|died|terminated unexpectedly))\b",
        re.IGNORECASE,
    ),
    "timeout": re.compile(
        r"\b(?:timeouterror|request timed out|engine startup timed out|"
        r"health check timed out)\b",
        re.IGNORECASE,
    ),
}


def _isbench_v5_submission_identity_required(serving: Any) -> bool:
    """Identify the v5 launcher/source identity without inventing a v5 schema."""

    if not isinstance(serving, dict):
        return False
    for field in (
        "source_path",
        "launcher_source_path",
        "submitter_source_path",
    ):
        for match in re.finditer(
            r"(?:^|[_./-])v(\d+)(?=[^0-9]|$)",
            str(serving.get(field) or ""),
        ):
            if int(match.group(1)) >= 5:
                return True
    return False


def _validate_isbench_submission_identity_evidence(
    *,
    artifacts: dict[str, Path],
    bindings: dict[str, Any],
    expected_preflight_phase: str,
    expected_terminal_phase: str,
    expected_input_paths: dict[str, Path] | None = None,
) -> list[str]:
    """Rehash the immutable submission identity and both temporal checks.

    Qualification binds the original manifest.  Formal invocation postflight
    binds a byte-identical copy, while each verification continues to bind the
    original read-only manifest in its submission root.
    """

    errors: list[str] = []
    expected_artifacts = _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
    if set(artifacts) != expected_artifacts:
        errors.append("IS-Bench submission identity evidence set is incomplete")
        return errors

    expected_filenames = {
        "submission_identity_manifest": "submission_identity_manifest.json",
        "submission_identity_verification_preflight": (
            "submission_identity_verification_preflight.json"
        ),
        "submission_identity_verification_terminal": (
            "submission_identity_verification_terminal.json"
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_artifacts):
        raw_path = Path(artifacts[name]).expanduser()
        path = raw_path.resolve()
        binding = bindings.get(name)
        if path.name != expected_filenames[name]:
            errors.append(f"IS-Bench submission identity {name} filename drift")
        raw_binding_path = (
            Path(str(binding.get("path") or "")).expanduser()
            if isinstance(binding, dict)
            else Path("")
        )
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or raw_binding_path.is_symlink()
            or raw_binding_path.resolve() != path
        ):
            errors.append(f"IS-Bench submission identity {name} binding is invalid")
        if raw_path.is_symlink():
            errors.append(f"IS-Bench submission identity {name} is a symlink")
            continue
        if not path.is_file():
            errors.append(f"IS-Bench submission identity {name} is missing")
            continue
        observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if not isinstance(binding, dict) or binding.get("sha256") != observed_sha:
            errors.append(f"IS-Bench submission identity {name} binding changed")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"IS-Bench submission identity {name} is invalid JSON")
            continue
        if not isinstance(payload, dict):
            errors.append(f"IS-Bench submission identity {name} must be an object")
            continue
        payloads[name] = payload

    raw_manifest_path = Path(artifacts["submission_identity_manifest"]).expanduser()
    manifest_path = raw_manifest_path.resolve()
    manifest = payloads.get("submission_identity_manifest")
    if manifest is None:
        return errors
    if set(manifest) != {"schema_version", "status", "submission_root", "inputs"}:
        errors.append("IS-Bench submission identity manifest schema drift")
    if manifest.get("schema_version") != "isbench-submission-identity-manifest-v1":
        errors.append("IS-Bench submission identity manifest schema version drift")
    if manifest.get("status") != "frozen":
        errors.append("IS-Bench submission identity manifest status drift")

    root_text = str(manifest.get("submission_root") or "")
    submission_root = Path(root_text).expanduser()
    root_valid = bool(root_text and submission_root.is_absolute())
    if root_valid:
        submission_root = submission_root.resolve()
        root_valid = submission_root.is_dir() and str(submission_root) == root_text
    if not root_valid:
        errors.append("IS-Bench submission identity submission root is invalid")
        original_manifest = None
    else:
        original_manifest = submission_root / "submission_identity_manifest.json"
        if (
            original_manifest.is_symlink()
            or not original_manifest.is_file()
            or original_manifest.stat().st_mode & 0o222
        ):
            errors.append(
                "IS-Bench submission identity original manifest must be read-only"
            )
        else:
            original_sha = hashlib.sha256(original_manifest.read_bytes()).hexdigest()
            copy_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if original_sha != copy_sha:
                errors.append(
                    "IS-Bench submission identity manifest copy differs from original"
                )
    if manifest_path.stat().st_mode & 0o222:
        errors.append("IS-Bench submission identity manifest must be read-only")

    inputs = manifest.get("inputs")
    expected_input_names = set(_ISBENCH_SUBMISSION_IDENTITY_INPUT_NAMES)
    if not isinstance(inputs, dict) or set(inputs) != expected_input_names:
        errors.append("IS-Bench submission identity input entry census drift")
        inputs = inputs if isinstance(inputs, dict) else {}
    expected_paths = {
        name: Path(path).expanduser().resolve()
        for name, path in (expected_input_paths or {}).items()
    }
    if root_valid:
        expected_paths.setdefault(
            "submitted_launcher_copy",
            submission_root / "component_launcher.sbatch",
        )
        expected_paths.setdefault(
            "submitted_probe_launcher_copy",
            submission_root / "probe_launcher.sbatch",
        )
    if not set(expected_paths).issubset(expected_input_names):
        errors.append("IS-Bench submission identity expected input census drift")
    for name in sorted(expected_input_names):
        item = inputs.get(name)
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            errors.append(f"IS-Bench submission identity input {name} is invalid")
            continue
        raw_path = str(item.get("path") or "")
        candidate = Path(raw_path).expanduser()
        if not raw_path or not candidate.is_absolute():
            errors.append(f"IS-Bench submission identity input {name} path is invalid")
            continue
        if candidate.is_symlink():
            errors.append(f"IS-Bench submission identity input {name} path is invalid")
            continue
        resolved = candidate.resolve()
        if (
            str(resolved) != raw_path
            or not resolved.is_file()
        ):
            errors.append(f"IS-Bench submission identity input {name} path is invalid")
            continue
        if expected_paths.get(name, resolved) != resolved:
            errors.append(f"IS-Bench submission identity input {name} path drift")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != item.get("sha256"):
            errors.append(f"IS-Bench submission identity input {name} changed")

    phases = {
        "submission_identity_verification_preflight": expected_preflight_phase,
        "submission_identity_verification_terminal": expected_terminal_phase,
    }
    original_binding = None
    if original_manifest is not None and original_manifest.is_file():
        original_binding = {
            "path": str(original_manifest),
            "sha256": hashlib.sha256(original_manifest.read_bytes()).hexdigest(),
        }
    for name, expected_phase in phases.items():
        verification = payloads.get(name)
        if verification is None:
            continue
        if set(verification) != {
            "schema_version",
            "status",
            "phase",
            "manifest",
            "inputs",
        }:
            errors.append(f"IS-Bench submission identity {name} schema drift")
        if verification.get("schema_version") != (
            "isbench-submission-identity-verification-v1"
        ):
            errors.append(f"IS-Bench submission identity {name} schema version drift")
        if verification.get("status") != "verified":
            errors.append(f"IS-Bench submission identity {name} status drift")
        if verification.get("phase") != expected_phase:
            errors.append(f"IS-Bench submission identity {name} phase drift")
        if original_binding is None or verification.get("manifest") != original_binding:
            errors.append(f"IS-Bench submission identity {name} manifest binding drift")
        if verification.get("inputs") != inputs:
            errors.append(f"IS-Bench submission identity {name} input entry census drift")
    return errors


def _isbench_service_log_signature_counts(text: str) -> dict[str, int]:
    """Recompute the frozen v4 fatal service signatures from one log snapshot."""

    return {
        name: len(pattern.findall(str(text)))
        for name, pattern in _ISBENCH_SERVICE_LOG_PATTERNS.items()
    }


def _validate_zero_leak_credential_scan(payload: Any, *, label: str) -> list[str]:
    if not isinstance(payload, dict) or not (
        payload.get("status") == "passed"
        and payload.get("leak_path_count") == 0
        and payload.get("leak_paths") == []
    ):
        return [f"{label} credential scan failed"]
    return []


def _validate_isbench_v4_service_evidence(
    *,
    artifacts: dict[str, Path],
    bindings: dict[str, Any],
    serving: dict[str, Any],
    expected_service_job_id: str,
    expected_runtime_phase: str,
    expected_implementation_sha256: str,
) -> list[str]:
    """Rehash and semantically validate the complete v4 Qwen service bundle."""

    errors: list[str] = []
    missing = sorted(_ISBENCH_V4_SERVICE_BINDING_NAMES - set(artifacts))
    if missing:
        return ["IS-Bench v4 service evidence is missing: " + ", ".join(missing)]

    expected_names = {
        "service_ready": "service_ready.json",
        "service_active_credential_scan_at_ready": (
            "service_active_credential_scan_at_ready.json"
        ),
        "service_startup_health_summary": "service_startup_health_summary.json",
        "service_vllm_log_snapshot": "service_vllm_log_snapshot.log",
        "service_runtime_health": (
            "service_runtime_health_after_qualification.json"
            if expected_runtime_phase == "qualification_terminal"
            else "service_runtime_health_after_invocation.json"
        ),
        "service_active_credential_scan_terminal": (
            "service_active_credential_scan_after_qualification.json"
            if expected_runtime_phase == "qualification_terminal"
            else "service_active_credential_scan_after_invocation.json"
        ),
        "advisory_active_path_code_verification": (
            "advisory_active_path_code_verification.json"
        ),
    }
    for name, expected_name in expected_names.items():
        path = artifacts[name]
        binding = bindings.get(name)
        if path.name != expected_name:
            errors.append(f"IS-Bench v4 service evidence {name} filename drift")
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            errors.append(f"IS-Bench v4 service evidence {name} binding is invalid")
        elif (
            Path(str(binding.get("path") or "")).expanduser().resolve() != path.resolve()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != binding.get("sha256")
        ):
            errors.append(f"IS-Bench v4 service evidence {name} binding changed")

    def load_json(name: str) -> dict[str, Any] | None:
        try:
            value = json.loads(artifacts[name].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"IS-Bench v4 service evidence {name} is invalid")
            return None
        if not isinstance(value, dict):
            errors.append(f"IS-Bench v4 service evidence {name} is invalid")
            return None
        return value

    def validate_nested_binding(
        binding: Any,
        *,
        label: str,
    ) -> Path | None:
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            errors.append(f"IS-Bench v4 service {label} binding is invalid")
            return None
        nested_path = Path(str(binding.get("path") or "")).expanduser().resolve()
        if not nested_path.is_file() or hashlib.sha256(
            nested_path.read_bytes()
        ).hexdigest() != binding.get("sha256"):
            errors.append(f"IS-Bench v4 service {label} binding changed")
            return None
        return nested_path

    ready = load_json("service_ready")
    expected_ready_keys = {
        "schema_version",
        "status",
        "service_job_id",
        "node",
        "base_url",
        "model",
        "max_model_len",
        "dtype",
        "ready_at_utc",
        "launcher",
        "checkpoint_verification",
        "qwen_gpu_identity",
        "qwen_gpu_identity_text",
        "endpoint_preflight",
        "serving_contract",
        "active_credential_scan_at_ready",
        "startup_health_summary",
    }
    expected_identity = {
        "service_job_id": str(expected_service_job_id),
        "node": str(serving.get("slurm_node") or ""),
        "base_url": str(serving.get("base_url") or ""),
        "model": str(serving.get("served_model_name") or ""),
        "max_model_len": serving.get("max_model_len"),
        "dtype": serving.get("dtype"),
    }
    if not isinstance(ready, dict) or set(ready) != expected_ready_keys:
        errors.append("IS-Bench v4 service_ready schema is invalid")
    else:
        if ready.get("schema_version") != "isbench-qwen-v4-service-ready-v1" or (
            ready.get("status") != "passed"
        ):
            errors.append("IS-Bench v4 service_ready status/schema drift")
        for field, expected in expected_identity.items():
            if ready.get(field) != expected:
                errors.append(f"IS-Bench v4 service_ready {field} drift")
        try:
            ready_at = datetime.fromisoformat(
                str(ready.get("ready_at_utc") or "").replace("Z", "+00:00")
            )
            if ready_at.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append("IS-Bench v4 service_ready timestamp is invalid")
        nested_paths = {
            name: validate_nested_binding(ready.get(name), label=name)
            for name in (
                "launcher",
                "checkpoint_verification",
                "qwen_gpu_identity",
                "qwen_gpu_identity_text",
                "endpoint_preflight",
                "serving_contract",
                "active_credential_scan_at_ready",
                "startup_health_summary",
            )
        }
        if (ready.get("launcher") or {}).get("sha256") != serving.get(
            "launcher_source_sha256"
        ):
            errors.append("IS-Bench v4 service launcher differs from serving contract")
        if (ready.get("serving_contract") or {}).get("sha256") != serving.get(
            "source_sha256"
        ):
            errors.append(
                "IS-Bench v4 service contract copy differs from serving contract"
            )
        for ready_name, binding_name in (
            (
                "active_credential_scan_at_ready",
                "service_active_credential_scan_at_ready",
            ),
            ("startup_health_summary", "service_startup_health_summary"),
        ):
            if (ready.get(ready_name) or {}).get("sha256") != (
                bindings.get(binding_name) or {}
            ).get("sha256"):
                errors.append(
                    f"IS-Bench v4 service_ready {ready_name} digest drift"
                )

        checkpoint_path = nested_paths.get("checkpoint_verification")
        checkpoint: Any = None
        if checkpoint_path is not None:
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                checkpoint = None
        if not isinstance(checkpoint, dict) or not (
            checkpoint.get("schema_version") == "model-checkpoint-verification-v1"
            and checkpoint.get("status") == "verified"
            and checkpoint.get("lfs_content_verified") is True
            and checkpoint.get("commit") == serving.get("checkpoint_commit")
            and checkpoint.get("lfs_manifest_sha256")
            == serving.get("lfs_manifest_sha256")
            and checkpoint.get("runtime_pip_freeze_sha256")
            == serving.get("runtime_freeze_sha256")
        ):
            errors.append("IS-Bench v4 service checkpoint identity drift")

        endpoint_path = nested_paths.get("endpoint_preflight")
        endpoint: Any = None
        if endpoint_path is not None:
            try:
                endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                endpoint = None
        if not isinstance(endpoint, dict) or not (
            endpoint.get("status") == "passed"
            and str(endpoint.get("base_url") or "").rstrip("/")
            == str(serving.get("base_url") or "").rstrip("/")
            and endpoint.get("requested_model") == serving.get("served_model_name")
            and endpoint.get("inference_probe") == "passed"
        ):
            errors.append("IS-Bench v4 service endpoint identity drift")

    ready_scan = load_json("service_active_credential_scan_at_ready")
    errors.extend(
        _validate_zero_leak_credential_scan(
            ready_scan,
            label="IS-Bench v4 service ready",
        )
    )
    terminal_scan = load_json("service_active_credential_scan_terminal")
    errors.extend(
        _validate_zero_leak_credential_scan(
            terminal_scan,
            label="IS-Bench v4 service terminal",
        )
    )

    validated_health: dict[str, dict[str, Any]] = {}

    def validate_health(name: str, expected_phase: str) -> None:
        health = load_json(name)
        v1_keys = {
            "schema_version",
            "status",
            "phase",
            "service_job_id",
            "node",
            "model",
            "base_url",
            "checked_at_utc",
            "log_snapshot",
            "probes",
            "counts",
        }
        v2_keys = v1_keys | {
            "baseline_policy",
            "baseline_counts",
            "delta_counts",
            "baseline_health_summary",
        }
        schema_version = health.get("schema_version") if isinstance(health, dict) else None
        expected_keys = (
            v2_keys
            if schema_version == "isbench-qwen-service-health-v2"
            else v1_keys
        )
        if not isinstance(health, dict) or set(health) != expected_keys:
            errors.append(f"IS-Bench v4 service {name} schema is invalid")
            return
        if not (
            schema_version in {
                "isbench-qwen-service-health-v1",
                "isbench-qwen-service-health-v2",
            }
            and health.get("status") == "passed"
            and health.get("phase") == expected_phase
            and health.get("probes")
            == {
                "model_listing_probe": "passed",
                "service_reachable": "passed",
            }
        ):
            errors.append(f"IS-Bench v4 service {name} status/schema drift")
        for field, expected in expected_identity.items():
            if field in {"max_model_len", "dtype"}:
                continue
            if health.get(field) != expected:
                errors.append(f"IS-Bench v4 service {name} {field} drift")
        try:
            checked_at = datetime.fromisoformat(
                str(health.get("checked_at_utc") or "").replace("Z", "+00:00")
            )
            if checked_at.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append(f"IS-Bench v4 service {name} timestamp is invalid")
        counts = health.get("counts")
        counts_valid = bool(
            isinstance(counts, dict)
            and set(counts) == _ISBENCH_SERVICE_HEALTH_KEYS
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in counts.values()
            )
        )
        if not counts_valid:
            errors.append(f"IS-Bench v4 service {name} health counts are invalid")
        elif schema_version == "isbench-qwen-service-health-v1" and any(
            counts.values()
        ):
            errors.append(f"IS-Bench v4 service {name} health counts are not all zero")
        elif schema_version == "isbench-qwen-service-health-v2":
            baseline_counts = health.get("baseline_counts")
            delta_counts = health.get("delta_counts")
            zero_counts = {key: 0 for key in _ISBENCH_SERVICE_HEALTH_KEYS}
            if not (
                health.get("baseline_policy")
                == "allow_preexisting_http_4xx_only_and_require_zero_delta_v1"
                and isinstance(baseline_counts, dict)
                and set(baseline_counts) == _ISBENCH_SERVICE_HEALTH_KEYS
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in baseline_counts.values()
                )
                and delta_counts == zero_counts
            ):
                errors.append(
                    f"IS-Bench v4 service {name} baseline/delta health is invalid"
                )
            elif expected_phase == "startup":
                if not (
                    health.get("baseline_health_summary") is None
                    and baseline_counts == counts
                    and all(
                        value == 0
                        for key, value in counts.items()
                        if key != "http_4xx"
                    )
                ):
                    errors.append(
                        "IS-Bench v4 startup health baseline is invalid"
                    )
            else:
                startup = validated_health.get("service_startup_health_summary") or {}
                baseline_path = validate_nested_binding(
                    health.get("baseline_health_summary"),
                    label=f"{name} baseline_health_summary",
                )
                expected_baseline_binding = bindings.get(
                    "service_startup_health_summary"
                )
                if not (
                    health.get("baseline_health_summary")
                    == expected_baseline_binding
                    and baseline_path is not None
                    and baseline_counts == startup.get("baseline_counts")
                    and counts == baseline_counts
                ):
                    errors.append(
                        f"IS-Bench v4 service {name} baseline binding/count drift"
                    )
        log_path = validate_nested_binding(
            health.get("log_snapshot"),
            label=f"{name} log_snapshot",
        )
        if log_path is not None:
            if (
                name == "service_startup_health_summary"
                and log_path.name != "service_vllm_log_at_ready_snapshot.log"
            ):
                errors.append("IS-Bench v4 startup log snapshot filename drift")
            recomputed = _isbench_service_log_signature_counts(
                log_path.read_text(encoding="utf-8", errors="replace")
            )
            if recomputed != counts or (
                schema_version == "isbench-qwen-service-health-v1"
                and any(recomputed.values())
            ):
                errors.append(f"IS-Bench v4 service {name} log health failed")
        if name == "service_runtime_health" and health.get(
            "log_snapshot"
        ) != bindings.get("service_vllm_log_snapshot"):
            errors.append("IS-Bench v4 runtime health/log binding drift")
        validated_health[name] = health

    validate_health("service_startup_health_summary", "startup")
    validate_health("service_runtime_health", expected_runtime_phase)

    active_path = load_json("advisory_active_path_code_verification")
    expected_checks = {
        "actor_initiated_pull": "passed",
        "nonexecuting": "passed",
        "zero_uptake_valid": "passed",
        "strict_child_parent": "passed",
    }
    if not isinstance(active_path, dict) or set(active_path) != {
        "schema_version",
        "status",
        "classification",
        "implementation_manifest_sha256",
        "checks",
        "test_ids",
        "source_files",
        "test_log",
        "exit_code",
        "model_calls_made",
    } or not (
        active_path.get("schema_version")
        == "isbench-advisory-active-path-code-verification-v1"
        and active_path.get("status") == "verified"
        and active_path.get("classification")
        == "diagnostic_code_path_verification_not_actor_uptake"
        and active_path.get("implementation_manifest_sha256")
        == expected_implementation_sha256
        and active_path.get("checks") == expected_checks
        and active_path.get("test_ids") == _ISBENCH_ADVISORY_ACTIVE_PATH_TEST_IDS
        and isinstance(active_path.get("exit_code"), int)
        and not isinstance(active_path.get("exit_code"), bool)
        and active_path.get("exit_code") == 0
        and active_path.get("model_calls_made") is False
    ):
        errors.append("IS-Bench v4 advisory active-path verification failed")
    elif isinstance(active_path, dict):
        source_rows = active_path.get("source_files")
        source_gate = bool(
            isinstance(source_rows, list)
            and len(source_rows) == len(_ISBENCH_ADVISORY_ACTIVE_PATH_SOURCE_FILES)
            and all(
                isinstance(row, dict)
                and set(row) == {"path", "source_sha256"}
                for row in source_rows
            )
            and [row.get("path") for row in source_rows]
            == _ISBENCH_ADVISORY_ACTIVE_PATH_SOURCE_FILES
        )
        project_root = Path(__file__).resolve().parents[1]
        if source_gate:
            for row in source_rows:
                source = (project_root / str(row["path"])).resolve()
                if (
                    project_root not in source.parents
                    or not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest()
                    != row.get("source_sha256")
                ):
                    source_gate = False
                    break
        if not source_gate:
            errors.append("IS-Bench v4 advisory active-path source binding failed")
        test_log = active_path.get("test_log")
        test_log_path = validate_nested_binding(
            test_log,
            label="advisory active-path test_log",
        )
        active_path_parent = artifacts[
            "advisory_active_path_code_verification"
        ].parent.resolve()
        if test_log_path is None or not (
            test_log_path.parent == active_path_parent
            and test_log_path.name == "advisory_active_path_code_verification.log"
        ):
            errors.append("IS-Bench v4 advisory active-path test log binding failed")
    return errors


def _isbench_hardware_qualification_schemas(
    hardware_contract: dict[str, Any],
) -> dict[str, str]:
    """Return the exact auxiliary schemas for one registered hardware arm."""

    errors = validate_isbench_hardware_contract(hardware_contract)
    if errors:
        raise ValueError("invalid IS-Bench hardware contract: " + "; ".join(errors))
    profile_id = str(hardware_contract.get("profile_id") or "")
    schemas = _ISBENCH_HARDWARE_QUALIFICATION_SCHEMAS.get(profile_id)
    if schemas is None:
        raise ValueError("unregistered IS-Bench hardware qualification profile")
    return dict(schemas)


def evaluate_formal_readiness(
    *,
    preregistration_path: Path,
    model_profile_path: Path,
    compositionality_manifest_path: Path,
    faithfulness_adjudication_protocol_path: Path,
    adjudicator_blinding_protocol_path: Path,
    checkpoint_verification_path: Path | None = None,
    environ: dict[str, str] | None = None,
    benchmarks: Iterable[str] | None = None,
    isbench_task_list_path: Path | None = None,
    omnigibson_infrastructure_summary_path: Path | None = None,
    isbench_online_qualification_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Return a credential-free report; every mismatch blocks formal execution."""

    errors: list[str] = []
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
    inputs = {
        "preregistration": Path(preregistration_path).expanduser().resolve(),
        "model_profile": Path(model_profile_path).expanduser().resolve(),
        "compositionality_manifest": Path(compositionality_manifest_path)
        .expanduser()
        .resolve(),
        "faithfulness_adjudication_protocol": Path(
            faithfulness_adjudication_protocol_path
        )
        .expanduser()
        .resolve(),
        "adjudicator_blinding_protocol": Path(adjudicator_blinding_protocol_path)
        .expanduser()
        .resolve(),
    }
    if isbench_task_list_path is not None:
        inputs["isbench_task_list"] = (
            Path(isbench_task_list_path).expanduser().resolve()
        )
    if omnigibson_infrastructure_summary_path is not None:
        inputs["omnigibson_infrastructure_summary"] = Path(
            omnigibson_infrastructure_summary_path
        ).expanduser().resolve()
    if isbench_online_qualification_summary_path is not None:
        inputs["isbench_online_qualification_summary"] = Path(
            isbench_online_qualification_summary_path
        ).expanduser().resolve()
    raw_payload: dict[str, Any] = {}
    try:
        raw_value = json.loads(inputs["preregistration"].read_text(encoding="utf-8"))
        if not isinstance(raw_value, dict):
            errors.append("preregistration input is not a JSON object")
        else:
            raw_payload = raw_value
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            f"cannot read preregistration input: {type(exc).__name__}: {exc}"
        )
    preregistrations: dict[str, dict[str, Any]] = {}
    for benchmark in requested_benchmarks:
        try:
            preregistrations[benchmark] = load_preregistration(
                inputs["preregistration"], benchmark=benchmark
            )
        except Exception as exc:
            errors.append(
                f"{benchmark} preregistration invalid: {type(exc).__name__}: {exc}"
            )
    payload = next(
        (item["payload"] for item in preregistrations.values()), raw_payload
    )
    if set(
        str(item).strip().lower()
        for item in payload.get("benchmark_scope") or []
    ) != set(requested_benchmarks):
        errors.append("benchmark_scope must exactly equal requested benchmarks")
    placeholder_paths = _placeholder_paths(payload)
    if placeholder_paths:
        errors.append(
            "preregistration contains unresolved placeholders: "
            + ", ".join(placeholder_paths[:20])
        )
    if "isbench" in requested_benchmarks:
        task_list_path = inputs.get("isbench_task_list")
        if task_list_path is None:
            errors.append("isbench readiness requires --isbench-task-list")
        else:
            errors.extend(
                _validate_isbench_task_list_binding(payload, task_list_path)
            )

    stack = None
    try:
        stack = load_model_stack_profile(
            inputs["model_profile"],
            environ=dict(os.environ if environ is None else environ),
            require_credentials=False,
        )
    except Exception as exc:
        errors.append(f"model profile invalid: {type(exc).__name__}: {exc}")

    implementation_digests: dict[str, str] = {}
    serving_source_bindings: dict[str, Any] = {}
    hardware_source_bindings: dict[str, Any] = {}
    hardware_contracts: dict[str, dict[str, Any]] = {}
    for benchmark, registration in preregistrations.items():
        contract = (
            registration["payload"]["run_plans"][benchmark][
                "online_advisory_contract"
            ]
        )
        expected_implementation = str(
            contract["provenance"]["implementation_manifest_sha256"]
        ).strip().lower()
        try:
            observed_implementation = current_online_implementation_manifest_sha256(
                benchmark
            )
            implementation_digests[benchmark] = observed_implementation
            if observed_implementation != expected_implementation:
                errors.append(
                    f"{benchmark} implementation digest differs from preregistration"
                )
        except Exception as exc:
            errors.append(
                f"{benchmark} implementation digest failed: {type(exc).__name__}: {exc}"
            )
        if stack is not None:
            if contract.get("model_stack") != stack.public_identity:
                errors.append(
                    f"{benchmark} online contract model_stack differs from model profile"
                )
            actor = stack.roles["actor"]
            world_model = stack.roles["world_model"]
            expected_actor = {
                "provider": "openai_compatible",
                "model": actor.model,
                "endpoint": actor.base_url,
                "revision": actor.revision,
            }
            expected_world_model = {
                "provider": "openai_compatible",
                "model": world_model.model,
                "endpoint": world_model.base_url,
                "revision": world_model.revision,
            }
            if contract.get("actor") != expected_actor:
                errors.append(f"{benchmark} actor identity differs from model profile")
            if contract.get("world_model") != expected_world_model:
                errors.append(
                    f"{benchmark} world-model identity differs from model profile"
                )
        serving_result = _validate_local_serving_binding(
            benchmark=benchmark,
            contract=contract,
        )
        errors.extend(serving_result.pop("errors"))
        if serving_result:
            serving_source_bindings[benchmark] = serving_result
        if benchmark == "isbench" and isinstance(
            (contract.get("runtime") or {}).get("hardware_contract"), dict
        ):
            hardware_contracts[benchmark] = contract["runtime"][
                "hardware_contract"
            ]
            hardware_result = _validate_local_hardware_binding(contract=contract)
            errors.extend(hardware_result.pop("errors"))
            if hardware_result:
                hardware_source_bindings[benchmark] = hardware_result
        try:
            validate_annotation_contract_files(
                registration,
                compositionality_manifest_path=inputs["compositionality_manifest"],
                faithfulness_adjudication_protocol_path=inputs[
                    "faithfulness_adjudication_protocol"
                ],
                adjudicator_blinding_protocol_path=inputs[
                    "adjudicator_blinding_protocol"
                ],
            )
        except Exception as exc:
            errors.append(
                f"{benchmark} annotation contract invalid: {type(exc).__name__}: {exc}"
            )

    checkpoint_summary: dict[str, Any] | None = None
    if checkpoint_verification_path is not None:
        checkpoint_path = Path(checkpoint_verification_path).expanduser().resolve()
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_summary = _validate_checkpoint(stack, checkpoint)
            errors.extend(checkpoint_summary.pop("errors"))
        except Exception as exc:
            errors.append(
                f"checkpoint verification invalid: {type(exc).__name__}: {exc}"
            )
    elif stack is not None and all(
        role.auth == "none" for role in stack.roles.values()
    ):
        errors.append("local model profile requires --checkpoint-verification")

    prereg_hashes = {
        benchmark: registration["sha256"]
        for benchmark, registration in preregistrations.items()
    }
    if len(set(prereg_hashes.values())) > 1:
        errors.append("benchmarks resolved different canonical preregistration hashes")
    if "isbench" in requested_benchmarks and int(
        (payload.get("support") or {}).get("min_tasks") or 0
    ) >= 161:
        infrastructure_path = inputs.get("omnigibson_infrastructure_summary")
        qualification_path = inputs.get("isbench_online_qualification_summary")
        isbench_hardware = hardware_contracts.get("isbench")
        if infrastructure_path is None:
            errors.append(
                "full IS-Bench readiness requires the schema-v2 OmniGibson "
                "infrastructure probe summary"
            )
        else:
            errors.extend(
                _validate_omnigibson_infrastructure_summary(
                    infrastructure_path,
                    hardware_contract=isbench_hardware,
                )
            )
        if qualification_path is None:
            errors.append(
                "full IS-Bench readiness requires registered online qualification"
            )
        elif infrastructure_path is not None:
            errors.extend(
                _validate_isbench_online_qualification(
                    qualification_path,
                    preregistration_path=inputs["preregistration"],
                    preregistration_sha256=prereg_hashes.get("isbench", ""),
                    model_profile_path=inputs["model_profile"],
                    checkpoint_verification_path=(
                        Path(checkpoint_verification_path).expanduser().resolve()
                        if checkpoint_verification_path is not None
                        else None
                    ),
                    infrastructure_summary_path=infrastructure_path,
                    payload=payload,
                    hardware_contract=isbench_hardware,
                )
            )
    return {
        "schema_version": (
            "joint-formal-readiness-v1"
            if set(requested_benchmarks) == set(BENCHMARKS)
            else "scoped-formal-readiness-v1"
        ),
        "formal_ready": not errors,
        "claim_allowed": False,
        "claim_allowed_reason": (
            "pre-execution readiness never substitutes for completed postflight"
        ),
        "benchmarks": list(requested_benchmarks),
        "study_id": payload.get("study_id"),
        "preregistration_sha256_by_benchmark": prereg_hashes,
        "model_profile_public_identity": (
            stack.public_identity if stack is not None else None
        ),
        "implementation_manifest_sha256_by_benchmark": implementation_digests,
        "serving_source_bindings": serving_source_bindings,
        "hardware_source_bindings": hardware_source_bindings,
        "isbench_hardware_claims": (
            hardware_contracts.get("isbench", {}).get("hardware_claims")
            if "isbench" in hardware_contracts
            else None
        ),
        "isbench_primary_estimand": (
            (
                "within_hardware_"
                f"{len((payload.get('run_plans', {}).get('isbench', {}) or {}).get('intervention_modes') or [])}"
                "_arm_itt_single_seed_task_census"
            )
            if "isbench" in hardware_contracts
            else None
        ),
        "checkpoint_verification": checkpoint_summary,
        "inputs": {
            name: _fingerprint(path) for name, path in inputs.items()
        },
        "errors": _ordered_unique(errors),
    }


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


def _validate_local_serving_binding(
    *, benchmark: str, contract: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    serving = (contract.get("runtime") or {}).get("serving_contract")
    model_stack = contract.get("model_stack") or {}
    role_identities = (
        model_stack.get("roles") if isinstance(model_stack, dict) else None
    )
    if not isinstance(role_identities, dict) or not role_identities:
        role_identities = {
            role: contract.get(role)
            for role in ("actor", "world_model")
            if isinstance(contract.get(role), dict)
        }
    local_roles = {
        str(role): identity
        for role, identity in role_identities.items()
        if isinstance(identity, dict)
        and _is_local_serving_endpoint(
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
        return {
            "errors": [
                f"{benchmark} local model roles do not share one serving identity"
            ]
        }
    public_isbench_v3 = (
        benchmark == "isbench"
        and isinstance(serving, dict)
        and serving.get("schema_version") == "isbench-vllm-serving-contract-v3"
    )
    if not local_roles and not public_isbench_v3:
        return {"errors": []}
    if not isinstance(serving, dict):
        return {"errors": [f"local {benchmark} serving contract is missing"]}
    project_root = Path(__file__).resolve().parents[1]
    source_relative = str(serving.get("source_path") or "")
    source_path = (project_root / source_relative).resolve()
    if (
        not source_relative
        or Path(source_relative).is_absolute()
        or project_root not in source_path.parents
        or not source_path.is_file()
    ):
        return {"errors": [f"local {benchmark} serving source is invalid"]}
    try:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"errors": [f"local {benchmark} serving source is unreadable"]}
    observed = {
        "source_path": source_relative,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        **(source_payload if isinstance(source_payload, dict) else {}),
    }
    if observed != serving:
        errors.append(f"local {benchmark} serving source bytes differ")
    launcher_relative = str(serving.get("launcher_source_path") or "")
    launcher_path = (project_root / launcher_relative).resolve()
    if (
        not launcher_relative
        or Path(launcher_relative).is_absolute()
        or project_root not in launcher_path.parents
        or not launcher_path.is_file()
    ):
        errors.append(f"local {benchmark} launcher source is invalid")
    elif hashlib.sha256(launcher_path.read_bytes()).hexdigest() != str(
        serving.get("launcher_source_sha256") or ""
    ):
        errors.append(f"local {benchmark} launcher source SHA256 differs")
    submitter_present = bool(
        "submitter_source_path" in serving
        or "submitter_source_sha256" in serving
    )
    submitter_path: Path | None = None
    submitter_path_valid = False
    if submitter_present:
        submitter_relative = str(serving.get("submitter_source_path") or "")
        submitter_path = (project_root / submitter_relative).resolve()
        if (
            not submitter_relative
            or Path(submitter_relative).is_absolute()
            or project_root not in submitter_path.parents
            or not submitter_path.is_file()
        ):
            errors.append(f"local {benchmark} submitter source is invalid")
        else:
            submitter_path_valid = True
            if hashlib.sha256(submitter_path.read_bytes()).hexdigest() != str(
                serving.get("submitter_source_sha256") or ""
            ):
                errors.append(f"local {benchmark} submitter source SHA256 differs")
    if benchmark == "harnessaudit" and local_roles:
        served_model = str(serving.get("served_model_name") or "")
        if not served_model or any(
            str(identity.get("model") or "") != served_model
            for identity in local_roles.values()
        ):
            errors.append(
                "local HarnessAudit served model differs from local model-stack roles"
            )
        served_base_url = str(serving.get("base_url") or "").rstrip("/")
        if not served_base_url:
            host = str(serving.get("host") or "").strip()
            port = serving.get("port")
            if host and isinstance(port, int):
                served_base_url = f"http://{host}:{port}/v1"
        if not served_base_url or any(
            str(identity.get("base_url") or identity.get("endpoint") or "").rstrip(
                "/"
            )
            != served_base_url
            for identity in local_roles.values()
        ):
            errors.append(
                "local HarnessAudit serving endpoint differs from local model-stack roles"
            )
    result = {
        "source": _fingerprint(source_path),
        "launcher": _fingerprint(launcher_path),
        "schema_version": serving.get("schema_version"),
        "backend": serving.get("backend"),
        "backend_version": serving.get("backend_version"),
        "base_url": serving.get("base_url"),
        "served_model_name": serving.get("served_model_name"),
        "slurm_node": serving.get("slurm_node"),
        "slurm_gpu_type": serving.get("slurm_gpu_type"),
        "slurm_gpu_count": serving.get("slurm_gpu_count"),
        "max_model_len": serving.get("max_model_len"),
        "dtype": serving.get("dtype"),
        "gpu_memory_utilization": serving.get("gpu_memory_utilization"),
        "errors": errors,
    }
    if submitter_present and submitter_path is not None and submitter_path_valid:
        result["submitter"] = _fingerprint(submitter_path)
    return result


def _validate_local_hardware_binding(
    *, contract: dict[str, Any]
) -> dict[str, Any]:
    """Re-read the prospective hardware profile and its probe programs."""

    hardware = (contract.get("runtime") or {}).get("hardware_contract")
    errors = validate_isbench_hardware_contract(hardware)
    if errors:
        return {"errors": errors}
    project_root = Path(__file__).resolve().parents[1]
    source_relative = str(hardware.get("source_path") or "")
    source_path = (project_root / source_relative).resolve()
    if (
        not source_relative
        or Path(source_relative).is_absolute()
        or project_root not in source_path.parents
        or not source_path.is_file()
    ):
        return {"errors": ["IS-Bench hardware profile source is invalid"]}
    try:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"errors": ["IS-Bench hardware profile source is unreadable"]}
    observed = {
        "source_path": source_relative,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        **(source_payload if isinstance(source_payload, dict) else {}),
    }
    if observed != hardware:
        errors.append("IS-Bench hardware profile source bytes differ")
    probe_bindings: dict[str, Any] = {}
    probe_contract = hardware.get("probe_contract") or {}
    for name, path_field, digest_field in (
        (
            "launcher",
            "launcher_source_path",
            "launcher_source_sha256",
        ),
        (
            "probe_python",
            "probe_python_source_path",
            "probe_python_source_sha256",
        ),
    ):
        relative = str(probe_contract.get(path_field) or "")
        candidate = (project_root / relative).resolve()
        if (
            not relative
            or Path(relative).is_absolute()
            or project_root not in candidate.parents
            or not candidate.is_file()
        ):
            errors.append(f"IS-Bench hardware {name} source is invalid")
            continue
        fingerprint = _fingerprint(candidate)
        if fingerprint["sha256"] != probe_contract.get(digest_field):
            errors.append(f"IS-Bench hardware {name} source SHA256 differs")
        probe_bindings[name] = fingerprint
    claims = hardware["hardware_claims"]
    return {
        "source": _fingerprint(source_path),
        "profile_id": hardware["profile_id"],
        "vendor_supported": claims["vendor_supported"],
        "rt_cores": claims["rt_cores"],
        "external_benchmark_comparability": claims[
            "external_benchmark_comparability"
        ],
        "primary_claim_scope": claims["claim_scope"],
        "probe_sources": probe_bindings,
        "errors": errors,
    }


def _validate_omnigibson_infrastructure_summary(
    path: Path,
    *,
    hardware_contract: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"OmniGibson infrastructure summary invalid: {type(exc).__name__}: {exc}"]
    if not isinstance(summary, dict):
        return ["OmniGibson infrastructure summary must be an object"]
    if hardware_contract is not None:
        return _validate_empirical_hardware_infrastructure_summary(
            path,
            summary=summary,
            hardware_contract=hardware_contract,
        )
    expected = {
        "schema_version": "omnigibson-gpu-infrastructure-probe-v3",
        "artifact_class": "infrastructure_diagnostic_not_experiment_evidence",
        "qualification_scope": "omnigibson_infrastructure_only",
        "status": "qualified",
        "gpu_label": "nvidia_rtx_a6000",
        "required_repetitions": 2,
        "observed_repetitions": 2,
        "functional_success_repetitions": 2,
        "hardware_gate_pass_repetitions": 2,
        "passed_repetitions": 2,
        "requested_repetitions": 2,
        "require_rtx_gpu": True,
        "result_path_set_exact": True,
        "scene_template_binding_consistent": True,
        "input_sha256_binding_consistent": True,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            errors.append(f"OmniGibson infrastructure summary {field} drift")
    if summary.get("failure_reasons") != []:
        errors.append("OmniGibson infrastructure probe contains failures")
    if summary.get("formal_workload_gate") != {
        "status": "requires_full_task_compatible_smoke",
        "eligible_for_formal_experiment": False,
        "required_next_evidence": "registered_full_task_compatible_online_smoke",
    }:
        errors.append(
            "OmniGibson infrastructure probe cannot substitute for online qualification"
        )
    return errors


def _validate_empirical_hardware_infrastructure_summary(
    path: Path,
    *,
    summary: dict[str, Any],
    hardware_contract: dict[str, Any],
) -> list[str]:
    """Validate prospective cold starts for one registered hardware profile."""

    errors = validate_isbench_hardware_contract(hardware_contract)
    if errors:
        return errors
    expected_keys = {
        "schema_version",
        "artifact_class",
        "qualification_scope",
        "status",
        "hardware_profile_id",
        "hardware_profile_sha256",
        "hardware_claims",
        "observed_hardware_constraints",
        "registered_workload",
        "prospective_evidence",
        "prior_artifact_reuse",
        "required_repetitions",
        "observed_repetitions",
        "functional_success_repetitions",
        "hardware_gate_pass_repetitions",
        "image_evidence_pass_repetitions",
        "passed_repetitions",
        "failure_reasons",
        "formal_workload_gate",
        "listed_gpu_cost_units_per_hour",
        "estimated_gpu_cost_units_for_probe",
        "total_elapsed_seconds",
        "mean_elapsed_seconds",
        "maximum_peak_memory_used_mib",
        "maximum_peak_memory_fraction",
        "result_files",
        "slurm_job_id",
        "finished_at_utc",
    }
    optional_scene_binding_key = "scene_template_binding_consistent"
    if set(summary) not in {
        frozenset(expected_keys),
        frozenset({*expected_keys, optional_scene_binding_key}),
    }:
        errors.append(
            "OmniGibson empirical infrastructure summary has unexpected fields"
        )
        return errors
    if (
        optional_scene_binding_key in summary
        and summary.get(optional_scene_binding_key) is not True
    ):
        errors.append(
            "OmniGibson empirical infrastructure summary "
            "scene_template_binding_consistent drift"
        )
    required_repetitions = int(
        hardware_contract["probe_contract"]["required_repetitions"]
    )
    listed_cost = float(
        hardware_contract["cost"]["listed_gpu_cost_units_per_hour"]
    )
    result_files = [
        f"repeat_{index}/result.json"
        for index in range(1, required_repetitions + 1)
    ]
    expected = {
        "schema_version": "omnigibson-empirical-hardware-qualification-v1",
        "artifact_class": "infrastructure_qualification_not_experiment_evidence",
        "qualification_scope": "registered_omnigibson_workload_hardware_arm_only",
        "status": "empirical_hardware_qualified_for_registered_workload_smoke",
        "hardware_profile_id": hardware_contract["profile_id"],
        "hardware_profile_sha256": hardware_contract["source_sha256"],
        "hardware_claims": hardware_contract["hardware_claims"],
        "observed_hardware_constraints": hardware_contract[
            "observed_hardware_constraints"
        ],
        "registered_workload": hardware_contract["registered_workload"],
        "prospective_evidence": True,
        "prior_artifact_reuse": False,
        "required_repetitions": required_repetitions,
        "observed_repetitions": required_repetitions,
        "functional_success_repetitions": required_repetitions,
        "hardware_gate_pass_repetitions": required_repetitions,
        "image_evidence_pass_repetitions": required_repetitions,
        "passed_repetitions": required_repetitions,
        "failure_reasons": [],
        "formal_workload_gate": {
            "eligible_for_formal_experiment": False,
            "required_next_evidence": "registered_full_task_compatible_online_smoke",
            "status": "requires_registered_full_task_compatible_online_smoke",
        },
        "listed_gpu_cost_units_per_hour": listed_cost,
        "result_files": result_files,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            errors.append(f"OmniGibson empirical infrastructure summary {field} drift")
    job_id = str(summary.get("slurm_job_id") or "")
    if not job_id.isdigit():
        errors.append("OmniGibson empirical infrastructure SLURM job id is invalid")
    try:
        finished_at = datetime.fromisoformat(
            str(summary.get("finished_at_utc") or "").replace("Z", "+00:00")
        )
    except ValueError:
        finished_at = None
    if finished_at is None or finished_at.tzinfo is None:
        errors.append("OmniGibson empirical infrastructure finish time is invalid")

    numeric_fields = (
        "estimated_gpu_cost_units_for_probe",
        "total_elapsed_seconds",
        "mean_elapsed_seconds",
        "maximum_peak_memory_used_mib",
        "maximum_peak_memory_fraction",
    )
    numeric: dict[str, float] = {}
    for field in numeric_fields:
        try:
            numeric[field] = float(summary[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"OmniGibson empirical infrastructure {field} is invalid")
    if numeric.get("estimated_gpu_cost_units_for_probe", -1) < 0:
        errors.append("OmniGibson empirical infrastructure cost is invalid")
    if numeric.get("total_elapsed_seconds", 0) <= 0 or numeric.get(
        "mean_elapsed_seconds", 0
    ) <= 0:
        errors.append("OmniGibson empirical infrastructure elapsed time is invalid")
    maximum_fraction = numeric.get("maximum_peak_memory_fraction", -1)
    if not 0 <= maximum_fraction <= hardware_contract["registered_workload"][
        "maximum_peak_memory_fraction"
    ]:
        errors.append("OmniGibson empirical infrastructure peak memory is invalid")

    source_profile = {
        key: hardware_contract[key]
        for key in hardware_contract
        if key not in {"source_path", "source_sha256"}
    }
    result_paths = [path.parent / item for item in expected["result_files"]]
    elapsed: list[float] = []
    peaks: list[float] = []
    fractions: list[float] = []
    for index, result_path in enumerate(result_paths, 1):
        if not result_path.is_file():
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} is missing"
            )
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} is invalid"
            )
            continue
        if not isinstance(result, dict):
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} is invalid"
            )
            continue
        reason = _empirical_result_failure_reason(
            result,
            source_profile,
            hardware_contract["source_sha256"],
            result_path=result_path,
        )
        if reason is not None:
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} failed: {reason}"
            )
        if str((result.get("slurm") or {}).get("job_id") or "") != job_id:
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} job drift"
            )
        try:
            elapsed.append(float(result["elapsed_seconds"]))
            peaks.append(float(result["gpu_measurements"]["peak_memory_used_mib"]))
            fractions.append(float(result["gpu_measurements"]["peak_memory_fraction"]))
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"OmniGibson empirical infrastructure repetition {index} metrics invalid"
            )
    if len(elapsed) == required_repetitions:
        derived = {
            "total_elapsed_seconds": sum(elapsed),
            "mean_elapsed_seconds": sum(elapsed) / required_repetitions,
            "maximum_peak_memory_used_mib": max(peaks),
            "maximum_peak_memory_fraction": max(fractions),
            "estimated_gpu_cost_units_for_probe": (
                listed_cost * sum(elapsed) / 3600.0
            ),
        }
        for field, expected_value in derived.items():
            try:
                observed_value = float(summary[field])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(observed_value - expected_value) > 1e-9:
                errors.append(
                    f"OmniGibson empirical infrastructure {field} aggregate drift"
                )
    return errors


def _validate_a10040_empirical_infrastructure_summary(
    path: Path,
    *,
    summary: dict[str, Any],
    hardware_contract: dict[str, Any],
) -> list[str]:
    """Backward-compatible alias for archived callers and tests."""

    return _validate_empirical_hardware_infrastructure_summary(
        path,
        summary=summary,
        hardware_contract=hardware_contract,
    )


def _validate_isbench_gpu_identity_payload(
    payload: Any,
    *,
    hardware_contract: dict[str, Any] | None = None,
    serving_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Validate the exact simulator and model-serving GPU identities."""

    errors: list[str] = []
    hardware_identity: dict[str, Any] | None = None
    if hardware_contract is not None:
        hardware_errors = validate_isbench_hardware_contract(hardware_contract)
        errors.extend(hardware_errors)
        if hardware_errors:
            return errors
        hardware_identity = isbench_omnigibson_hardware_identity(hardware_contract)
    if hardware_identity is not None:
        try:
            schemas = _isbench_hardware_qualification_schemas(hardware_contract)
        except ValueError as exc:
            return [*errors, str(exc)]
        expected_schema = schemas["gpu_identity"]
        if (
            hardware_identity.get("profile_id")
            == "omnigibson_rtx_a6000_empirical_v1"
            and isinstance(serving_contract, dict)
            and serving_contract.get("slurm_gpu_type")
            == "nvidia_a100_80gb_pcie"
        ):
            expected_schema = (
                "isbench-a6000-shared-a100pcie-heterogeneous-gpu-identity-v1"
            )
    else:
        expected_schema = "isbench-heterogeneous-gpu-identity-v1"
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "status",
        "omnigibson",
        "qwen",
    }:
        return [*errors, "IS-Bench qualification GPU identity schema is invalid"]
    if payload.get("schema_version") != expected_schema:
        errors.append("IS-Bench qualification GPU identity schema drift")
    if payload.get("status") != "verified":
        errors.append("IS-Bench qualification GPU identity status is invalid")
    expected_omni = {
        "node": hardware_identity["node"] if hardware_identity else "gpu07",
        "name": (
            hardware_identity["exact_nvidia_smi_name"]
            if hardware_identity
            else "NVIDIA RTX A6000"
        ),
        "minimum_memory_mib": (
            hardware_identity["minimum_memory_total_mib"]
            if hardware_identity
            else 48000
        ),
    }
    if hardware_identity is not None and serving_contract is not None:
        serving_gpu_type = str(serving_contract.get("slurm_gpu_type") or "")
        serving_gpu = _ISBENCH_QWEN_SERVING_GPU_IDENTITIES.get(serving_gpu_type)
        serving_node = str(serving_contract.get("slurm_node") or "")
        if serving_gpu is None or not serving_node:
            errors.append("IS-Bench qualification Qwen serving GPU identity is invalid")
            expected_qwen = {
                "node": serving_node,
                "name": "",
                "minimum_memory_mib": 0,
            }
        else:
            expected_qwen = {"node": serving_node, **serving_gpu}
    else:
        # Archived v1 and A100-40 v2 qualifications predate explicit serving
        # injection here. Preserve their byte interpretation without using the
        # fallback for a new A6000 qualification.
        if (
            hardware_identity is not None
            and hardware_identity.get("profile_id")
            == "omnigibson_rtx_a6000_empirical_v1"
        ):
            errors.append(
                "IS-Bench qualification A6000 identity requires serving contract"
            )
        expected_qwen = {
            "node": "gpu18",
            "name": "NVIDIA A100 80GB PCIe",
            "minimum_memory_mib": 80000,
        }
    expected_by_role = {
        "omnigibson": expected_omni,
        "qwen": expected_qwen,
    }
    require_job_ids = bool(
        hardware_identity is not None
        and hardware_identity.get("profile_id")
        == "omnigibson_rtx_a6000_empirical_v1"
    )
    for role, expected in expected_by_role.items():
        observed = payload.get(role)
        expected_role_keys = {
            "node",
            "name",
            "memory_mib",
            "source_sha256",
        }
        if require_job_ids:
            expected_role_keys.add("job_id")
        if not isinstance(observed, dict) or set(observed) != expected_role_keys:
            errors.append(f"IS-Bench qualification {role} GPU identity is invalid")
            continue
        try:
            memory_mib = int(observed.get("memory_mib"))
        except (TypeError, ValueError):
            memory_mib = -1
        if (
            observed.get("node") != expected["node"]
            or observed.get("name") != expected["name"]
            or memory_mib < expected["minimum_memory_mib"]
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(observed.get("source_sha256") or "")
            )
        ):
            errors.append(f"IS-Bench qualification {role} GPU identity drift")
        if require_job_ids and not str(observed.get("job_id") or "").isdigit():
            errors.append(f"IS-Bench qualification {role} GPU job identity is invalid")
    return errors


def _validate_isbench_v2_launcher_bindings(
    *,
    bindings: dict[str, Any],
    resolved: dict[str, Path],
    serving: dict[str, Any],
    require_serving_launcher_identity: bool = True,
) -> list[str]:
    """Require source/copy identity and, for archived v2, the service launcher."""

    errors: list[str] = []
    expected_digest = str(serving.get("launcher_source_sha256") or "")
    observed: dict[str, str] = {}
    for name in ("launcher", "submitted_launcher_copy"):
        path = resolved.get(name)
        binding = bindings.get(name)
        if path is None or not path.is_file() or not isinstance(binding, dict):
            errors.append(f"IS-Bench qualification {name} binding is missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[name] = digest
        if digest != binding.get("sha256"):
            errors.append(f"IS-Bench qualification {name} binding changed")
        if require_serving_launcher_identity and digest != expected_digest:
            label = "submitted launcher" if name == "submitted_launcher_copy" else "launcher"
            errors.append(f"IS-Bench qualification {label} differs from serving contract")
    if len(observed) == 2 and observed["launcher"] != observed[
        "submitted_launcher_copy"
    ]:
        errors.append(
            "IS-Bench qualification submitted launcher differs from persistent launcher"
        )
    return errors


def _validate_isbench_qualification_preregistration_sha(
    summary: dict[str, Any],
    *,
    expected_sha256: str,
    hardware_contract: dict[str, Any] | None,
) -> list[str]:
    """Apply the explicit canonical-SHA requirement only to A100-40 v2."""

    observed = summary.get("formal_preregistration_sha256")
    if hardware_contract is not None:
        if observed != expected_sha256:
            return ["IS-Bench qualification canonical preregistration SHA differs"]
    elif observed is not None and observed != expected_sha256:
        return ["IS-Bench qualification canonical preregistration SHA differs"]
    return []


def _validate_isbench_qualification_slurm_window(
    payload: Any,
    *,
    hardware_contract: dict[str, Any],
    gpu_identity_payload: Any,
) -> list[str]:
    """Validate the scheduler window for one registered qualification profile."""

    errors = validate_isbench_hardware_contract(hardware_contract)
    if errors:
        return errors
    profile_id = str(hardware_contract.get("profile_id") or "")
    schemas = _isbench_hardware_qualification_schemas(hardware_contract)
    if profile_id == "omnigibson_a10040_empirical_v1":
        expected_keys = {
            "schema_version",
            "leader_job_id",
            "group0_job_id",
            "group1_job_id",
            "start_epoch",
            "end_epoch",
            "allocated_walltime_seconds",
            "invocation_deadline_epoch",
            "postflight_reserve_seconds",
            "deadline_policy_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            return ["IS-Bench qualification SLURM window schema drift"]
        try:
            leader = str(payload["leader_job_id"])
            start = int(payload["start_epoch"])
            end = int(payload["end_epoch"])
            deadline = int(payload["invocation_deadline_epoch"])
        except (KeyError, TypeError, ValueError):
            return ["IS-Bench qualification SLURM window is invalid"]
        if not (
            payload["schema_version"] == schemas["slurm_window"]
            and leader.isdigit()
            and payload["group0_job_id"] == f"{leader}+0"
            and payload["group1_job_id"] == f"{leader}+1"
            and end - start == 43200
            and payload["allocated_walltime_seconds"] == 43200
            and payload["postflight_reserve_seconds"] == 900
            and deadline == end - 900
            and payload["deadline_policy_version"]
            == "condition_start_budget_v1"
        ):
            errors.append("IS-Bench qualification SLURM window drift")
        return errors

    if profile_id != "omnigibson_rtx_a6000_empirical_v1":
        return ["IS-Bench qualification SLURM hardware profile is unregistered"]
    expected_keys = {
        "schema_version",
        "service_job_id",
        "qualification_runner_job_id",
        "start_epoch",
        "end_epoch",
        "allocated_walltime_seconds",
        "invocation_deadline_epoch",
        "postflight_reserve_seconds",
        "deadline_policy_version",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        return ["IS-Bench qualification SLURM window schema drift"]
    try:
        service_job_id = str(payload["service_job_id"])
        runner_job_id = str(payload["qualification_runner_job_id"])
        start = int(payload["start_epoch"])
        end = int(payload["end_epoch"])
        allocated = int(payload["allocated_walltime_seconds"])
        deadline = int(payload["invocation_deadline_epoch"])
        reserve = int(payload["postflight_reserve_seconds"])
    except (KeyError, TypeError, ValueError):
        return ["IS-Bench qualification SLURM window is invalid"]
    if not (
        payload["schema_version"] == schemas["slurm_window"]
        and service_job_id.isdigit()
        and runner_job_id.isdigit()
        and service_job_id != runner_job_id
        and start < end
        and end - start == 43200
        and allocated == 43200
        and reserve == 900
        and deadline == end - reserve
        and start < deadline < end
        and payload["deadline_policy_version"] == "condition_start_budget_v1"
    ):
        errors.append("IS-Bench qualification SLURM window drift")
    if not isinstance(gpu_identity_payload, dict):
        errors.append("IS-Bench qualification SLURM/GPU job binding is invalid")
    else:
        qwen_job_id = str(
            (gpu_identity_payload.get("qwen") or {}).get("job_id") or ""
        )
        runner_gpu_job_id = str(
            (gpu_identity_payload.get("omnigibson") or {}).get("job_id") or ""
        )
        if (
            qwen_job_id != service_job_id
            or runner_gpu_job_id != runner_job_id
        ):
            errors.append("IS-Bench qualification SLURM/GPU job binding drift")
    return errors


def _isbench_qualification_matrix_expectations(
    *,
    policies: Iterable[str],
    intervention_modes: Iterable[str],
) -> tuple[int, dict[str, int], dict[str, bool]]:
    policy_list = list(policies)
    modes = list(intervention_modes)
    guarded_modes = [mode for mode in modes if mode != "no_guard"]
    guarded_multiplicity = sum(
        len(WM_RETURN_PERTURBATIONS)
        if mode == DISTRIBUTED_ENFORCED
        else 1
        for mode in guarded_modes
    )
    condition_count = 1 + len(policy_list) * guarded_multiplicity
    arm_counts = {
        mode: (
            1
            if mode == "no_guard"
            else len(policy_list)
            * (
                len(WM_RETURN_PERTURBATIONS)
                if mode == DISTRIBUTED_ENFORCED
                else 1
            )
        )
        for mode in modes
    }
    ledger_gate = {
        "enforced_guard_complete": True,
        "advisory_complete_or_zero_uptake_valid": True,
    }
    if "prediction_only" in modes:
        ledger_gate = {"prediction_only_complete": True, **ledger_gate}
    if DISTRIBUTED_ENFORCED in modes:
        ledger_gate = {
            **ledger_gate,
            "distributed_enforced_complete": True,
        }
    return condition_count, arm_counts, ledger_gate


def _validate_isbench_online_qualification(
    path: Path,
    *,
    preregistration_path: Path,
    preregistration_sha256: str,
    model_profile_path: Path,
    checkpoint_verification_path: Path | None,
    infrastructure_summary_path: Path,
    payload: dict[str, Any],
    hardware_contract: dict[str, Any] | None = None,
) -> list[str]:
    """Validate the registered two-task diagnostic smoke matrix.

    This gate remains diagnostic.  It can authorize starting the frozen full
    cohort only when its bytes still match the preregistration frozen before
    any qualification model access.
    """

    errors: list[str] = []
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"IS-Bench online qualification invalid: {type(exc).__name__}: {exc}"]
    if not isinstance(summary, dict):
        return ["IS-Bench online qualification must be an object"]
    plan = payload.get("run_plans", {}).get("isbench", {})
    contract = plan.get("online_advisory_contract", {})
    policies = list(plan.get("policy_modes") or [])
    intervention_modes = list(plan.get("intervention_modes") or [])
    guarded_modes = [
        mode for mode in intervention_modes if mode != "no_guard"
    ]
    (
        expected_condition_count,
        expected_arm_counts,
        expected_wm_ledger_gate,
    ) = _isbench_qualification_matrix_expectations(
        policies=policies,
        intervention_modes=intervention_modes,
    )
    expected_task_count = 2
    expected_task_episode_count = expected_task_count * expected_condition_count
    prediction_only_registered = "prediction_only" in intervention_modes
    serving = (contract.get("runtime") or {}).get("serving_contract") or {}
    serving_schema = str(serving.get("schema_version") or "")
    v4_service_evidence_required = serving_schema == _ISBENCH_V4_SERVING_SCHEMA
    v5_submission_identity_required = _isbench_v5_submission_identity_required(
        serving
    )
    embedded_hardware = (contract.get("runtime") or {}).get("hardware_contract")
    if hardware_contract is None:
        hardware_contract = embedded_hardware
    elif hardware_contract != embedded_hardware:
        errors.append(
            "IS-Bench qualification hardware contract differs from preregistration"
        )
    hardware_identity: dict[str, Any] | None = None
    qualification_schemas: dict[str, str] | None = None
    if hardware_contract is not None:
        hardware_errors = validate_isbench_hardware_contract(hardware_contract)
        errors.extend(hardware_errors)
        if not hardware_errors:
            hardware_identity = isbench_omnigibson_hardware_identity(
                hardware_contract
            )
            try:
                qualification_schemas = _isbench_hardware_qualification_schemas(
                    hardware_contract
                )
            except ValueError as exc:
                errors.append(str(exc))
                # Keep an unregistered profile fail-closed without allowing a
                # secondary NoneType crash to suppress the readiness report.
                hardware_identity = None
    expected_scalars = {
        "schema_version": (
            _ISBENCH_V4_QUALIFICATION_SCHEMA
            if v4_service_evidence_required
            else (
                "isbench-online-qualification-v2"
                if hardware_identity is not None
                else "isbench-online-qualification-v1"
            )
        ),
        "artifact_class": "diagnostic",
        "qualification_scope": "registered_full_task_compatible_online_smoke",
        "status": "passed",
        "eligible_for_formal_claim": False,
        "preregistration_frozen_before_model_access": True,
        "post_qualification_mutation_allowed": False,
        "required_condition_count": expected_condition_count,
        "passed_condition_count": expected_condition_count,
        "failed_condition_count": 0,
        "failed_validation_condition_count": 0,
        "skipped_condition_count": 0,
        "required_task_count": expected_task_count,
        "task_episode_denominator": expected_task_episode_count,
        "passed_task_episode_count": expected_task_episode_count,
        "all_condition_validations_passed": True,
        "checkpoint_status": "verified",
        "credential_scan_status": "passed",
        "credential_leak_path_count": 0,
    }
    if hardware_identity is not None:
        expected_scalars.update(
            {
                "outer_credential_scan_status": "passed",
                "outer_credential_leak_path_count": 0,
                "request_ledger_error_count": 0,
                "infrastructure_error_report_count": 0,
                "structurally_invalid_report_count": 0,
            }
        )
    for field, expected in expected_scalars.items():
        if summary.get(field) != expected:
            errors.append(f"IS-Bench online qualification {field} drift")
    if (
        summary.get("schema_version") == _ISBENCH_V4_QUALIFICATION_SCHEMA
        and not v4_service_evidence_required
    ):
        errors.append(
            "IS-Bench v4 qualification schema requires a v4 serving contract"
        )
    if v4_service_evidence_required:
        for field in (
            "prediction_only_pairing_gate_passed",
            "strict_child_parent_gate_passed",
            "provenance_gate_passed",
            "credential_scan_gate_passed",
            "service_evidence_gate_passed",
            "service_runtime_health_gate_passed",
            "service_credential_scan_gate_passed",
            "advisory_active_path_code_gate_passed",
        ):
            if summary.get(field) is not True:
                errors.append(f"IS-Bench v4 qualification {field} failed")
    if v5_submission_identity_required and summary.get(
        "submission_identity_gate_passed"
    ) is not True:
        errors.append("IS-Bench v5 qualification submission identity gate failed")
    if hardware_identity is not None:
        expected_hardware_fields = {
            "hardware_profile_id": hardware_identity["profile_id"],
            "hardware_profile_sha256": hardware_identity["source_sha256"],
            "hardware_claims": hardware_identity["hardware_claims"],
        }
        for field, expected in expected_hardware_fields.items():
            if summary.get(field) != expected:
                errors.append(f"IS-Bench online qualification {field} drift")
    if summary.get("failure_reasons") != []:
        errors.append("IS-Bench online qualification contains failures")
    if hardware_identity is not None:
        if not isinstance(summary.get("action_error_report_count"), int) or summary.get(
            "action_error_report_count", -1
        ) < 0:
            errors.append("IS-Bench qualification action error count is invalid")
        signature_counts = summary.get("runtime_log_signature_counts")
        if not isinstance(signature_counts, dict) or set(signature_counts) != {
            "sigsegv",
            "oom",
            "infrastructure_timeout",
            "traceback_report_only",
        } or any(
            not isinstance(value, int) or value < 0
            for value in (signature_counts or {}).values()
        ):
            errors.append("IS-Bench qualification runtime signature counts are invalid")
        elif any(
            signature_counts[name] != 0
            for name in ("sigsegv", "oom", "infrastructure_timeout")
        ):
            errors.append("IS-Bench qualification contains fatal runtime signatures")
    if summary.get("condition_count_by_arm") != expected_arm_counts:
        errors.append(
            "IS-Bench online qualification does not cover the registered paths"
        )
    if summary.get("wm_ledger_gate") != expected_wm_ledger_gate:
        errors.append("IS-Bench qualification WM request ledger gate failed")
    actor = contract.get("actor", {})
    if summary.get("endpoint") != actor.get("endpoint"):
        errors.append("IS-Bench qualification endpoint differs from preregistration")
    if summary.get("model") != actor.get("model"):
        errors.append("IS-Bench qualification model differs from preregistration")
    implementation = (
        contract.get("provenance", {}).get("implementation_manifest_sha256")
    )
    if summary.get("implementation_manifest_sha256") != implementation:
        errors.append("IS-Bench qualification implementation identity differs")
    try:
        infrastructure_summary = json.loads(
            infrastructure_summary_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        infrastructure_summary = {}
    if summary.get("infrastructure_probe_job_id") != infrastructure_summary.get(
        "slurm_job_id"
    ):
        errors.append("IS-Bench qualification probe job identity differs")
    gpu_payload: Any = None
    gpu_verification = summary.get("gpu_identity_verification")
    if not isinstance(gpu_verification, dict) or set(gpu_verification) != {
        "status",
        "binding",
    } or gpu_verification.get("status") != "verified":
        errors.append("IS-Bench qualification GPU identity verification is invalid")
    else:
        gpu_binding = gpu_verification.get("binding")
        if not isinstance(gpu_binding, dict) or set(gpu_binding) != {
            "path",
            "sha256",
        }:
            errors.append("IS-Bench qualification GPU identity binding is invalid")
        else:
            gpu_path = Path(str(gpu_binding.get("path") or "")).resolve()
            if not gpu_path.is_file() or hashlib.sha256(
                gpu_path.read_bytes()
            ).hexdigest() != gpu_binding.get("sha256"):
                errors.append("IS-Bench qualification GPU identity binding changed")
            else:
                try:
                    gpu_payload = json.loads(gpu_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    gpu_payload = None
                errors.extend(
                    _validate_isbench_gpu_identity_payload(
                        gpu_payload,
                        hardware_contract=hardware_contract,
                        serving_contract=(
                            (contract.get("runtime") or {}).get(
                                "serving_contract"
                            )
                        ),
                    )
                )

    bindings = summary.get("bindings")
    if hardware_identity is not None:
        required_binding_names = {
            "formal_preregistration",
            "launcher",
            "submitted_launcher_copy",
            "model_profile",
            "serving_contract",
            "qualification_task_list",
            "checkpoint_verification",
            "endpoint_preflight",
            "hardware_profile",
            "infrastructure_probe_summary",
            "infrastructure_probe_result_1",
            "infrastructure_probe_result_2",
            "gpu_identity",
            "hardware_probe_verification",
            "preregistration_verification",
            "serving_contract_verification",
            "slurm_time_window",
            "matrix_manifest",
            "provenance_manifest",
            "active_credential_scan",
            "outer_active_credential_scan",
            "task_scores",
        }
        if v4_service_evidence_required:
            required_binding_names |= {
                "prediction_only_pairing",
                *_ISBENCH_V4_SERVICE_BINDING_NAMES,
            }
        if v5_submission_identity_required:
            required_binding_names |= _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
    else:
        required_binding_names = {
            "formal_preregistration",
            "launcher",
            "serving_contract",
            "model_profile",
            "checkpoint_verification",
            "qualification_task_list",
            "matrix_manifest",
            "task_scores",
            "provenance_manifest",
            "active_credential_scan",
            "infrastructure_probe_summary",
        }
    if not isinstance(bindings, dict) or set(bindings) != required_binding_names:
        errors.append(
            "IS-Bench qualification bindings must cover all registered inputs/outputs"
        )
        return errors
    resolved: dict[str, Path] = {}
    for name in sorted(required_binding_names):
        binding = bindings.get(name)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            errors.append(f"IS-Bench qualification binding {name} is invalid")
            continue
        candidate = Path(str(binding.get("path") or "")).expanduser().resolve()
        if not candidate.is_file():
            errors.append(f"IS-Bench qualification binding {name} is missing")
            continue
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != str(binding.get("sha256") or ""):
            errors.append(f"IS-Bench qualification binding {name} changed")
        resolved[name] = candidate
    expected_paths = {
        "formal_preregistration": preregistration_path.resolve(),
        "model_profile": model_profile_path.resolve(),
        "infrastructure_probe_summary": infrastructure_summary_path.resolve(),
    }
    project_root = Path(__file__).resolve().parents[1]
    expected_paths["serving_contract"] = (
        project_root / str(serving.get("source_path") or "")
    ).resolve()
    if not v5_submission_identity_required:
        expected_paths["launcher"] = (
            project_root / str(serving.get("launcher_source_path") or "")
        ).resolve()
    expected_paths["qualification_task_list"] = (
        project_root
        / "config"
        / "task_lists"
        / "isbench_qwen35_27b_qualification_v1.txt"
    ).resolve()
    if hardware_identity is not None:
        expected_paths["hardware_profile"] = (
            project_root / hardware_identity["source_path"]
        ).resolve()
        expected_paths.update(
            {
                "submitted_launcher_copy": (path.parent / "launch.sbatch").resolve(),
                "checkpoint_verification": (
                    path.parent / "model_checkpoint_verification.json"
                ).resolve(),
                "endpoint_preflight": (
                    path.parent / "endpoint_preflight.json"
                ).resolve(),
                "gpu_identity": (
                    path.parent / "gpu_identity_verification.json"
                ).resolve(),
                "hardware_probe_verification": (
                    path.parent / "hardware_probe_verification.json"
                ).resolve(),
                "preregistration_verification": (
                    path.parent / "preregistration_verification.json"
                ).resolve(),
                "serving_contract_verification": (
                    path.parent / "serving_contract_verification.json"
                ).resolve(),
                "slurm_time_window": (
                    path.parent / "slurm_time_window.json"
                ).resolve(),
                "matrix_manifest": (
                    path.parent / "matrix" / "online_matrix_manifest.jsonl"
                ).resolve(),
                "provenance_manifest": (
                    path.parent
                    / "matrix"
                    / "online_matrix_provenance_manifest.json"
                ).resolve(),
                "active_credential_scan": (
                    path.parent / "matrix" / "active_credential_scan.json"
                ).resolve(),
                "outer_active_credential_scan": (
                    path.parent / "outer_active_credential_scan.json"
                ).resolve(),
                "task_scores": (
                    path.parent / "matrix" / "online_task_scores.csv"
                ).resolve(),
            }
        )
        if v4_service_evidence_required:
            expected_paths.update(
                {
                    "prediction_only_pairing": (
                        path.parent / "matrix" / "prediction_only_pairing.json"
                    ).resolve(),
                    "advisory_active_path_code_verification": (
                        path.parent
                        / "advisory_active_path_code_verification.json"
                    ).resolve(),
                }
            )
        if v5_submission_identity_required:
            expected_paths.update(
                {
                    "submission_identity_verification_preflight": (
                        path.parent
                        / "submission_identity_verification_preflight.json"
                    ).resolve(),
                    "submission_identity_verification_terminal": (
                        path.parent
                        / "submission_identity_verification_terminal.json"
                    ).resolve(),
                }
            )
        try:
            infrastructure_payload = json.loads(
                infrastructure_summary_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            infrastructure_payload = {}
        result_files = infrastructure_payload.get("result_files") or []
        if result_files != ["repeat_1/result.json", "repeat_2/result.json"]:
            errors.append(
                "IS-Bench qualification infrastructure result cohort differs"
            )
        else:
            for index, relative in enumerate(result_files, 1):
                expected_paths[f"infrastructure_probe_result_{index}"] = (
                    infrastructure_summary_path.parent / relative
                ).resolve()
    for name, expected in expected_paths.items():
        if resolved.get(name) != expected:
            errors.append(f"IS-Bench qualification binding {name} path drift")
    if hardware_identity is not None:
        errors.extend(
            _validate_isbench_v2_launcher_bindings(
                bindings=bindings,
                resolved=resolved,
                serving=serving,
                require_serving_launcher_identity=(
                    not v5_submission_identity_required
                ),
            )
        )
        for name, expected_digest in (
            ("serving_contract", serving.get("source_sha256")),
            ("hardware_profile", hardware_identity["source_sha256"]),
        ):
            binding = bindings.get(name) or {}
            if binding.get("sha256") != expected_digest:
                errors.append(f"IS-Bench qualification binding {name} identity drift")
        if gpu_verification.get("binding") != bindings.get("gpu_identity"):
            errors.append(
                "IS-Bench qualification GPU identity bindings are inconsistent"
            )
        expected_probe_bindings = [
            bindings.get("infrastructure_probe_result_1"),
            bindings.get("infrastructure_probe_result_2"),
        ]
        if summary.get("infrastructure_probe_result_bindings") != (
            expected_probe_bindings
        ):
            errors.append(
                "IS-Bench qualification probe result bindings are inconsistent"
            )

        def load_bound_json(name: str) -> dict[str, Any] | None:
            candidate = resolved.get(name)
            if candidate is None:
                return None
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append(f"IS-Bench qualification {name} audit is invalid")
                return None
            if not isinstance(value, dict):
                errors.append(f"IS-Bench qualification {name} audit is invalid")
                return None
            return value

        endpoint_payload = load_bound_json("endpoint_preflight")
        if not isinstance(endpoint_payload, dict) or not (
            endpoint_payload.get("status") == "passed"
            and str(endpoint_payload.get("base_url") or "").rstrip("/")
            == str(actor.get("endpoint") or "").rstrip("/")
            and endpoint_payload.get("requested_model") == actor.get("model")
            and endpoint_payload.get("inference_probe") == "passed"
        ):
            errors.append("IS-Bench qualification endpoint preflight audit failed")

        prereg_audit = load_bound_json("preregistration_verification")
        expected_prereg_audit = {
            "schema_version": qualification_schemas[
                "preregistration_verification"
            ],
            "status": "verified",
            "preregistration_path": str(preregistration_path.resolve()),
            "preregistration_sha256": preregistration_sha256,
            "hardware_profile_id": hardware_identity["profile_id"],
            "hardware_profile_sha256": hardware_identity["source_sha256"],
        }
        if prereg_audit != expected_prereg_audit:
            errors.append(
                "IS-Bench qualification preregistration verification drift"
            )

        serving_audit = load_bound_json("serving_contract_verification")
        if serving_audit != {
            "schema_version": qualification_schemas[
                "serving_contract_verification"
            ],
            "status": "verified",
            "serving_contract_sha256": serving.get("source_sha256"),
            "launcher_source_sha256": serving.get("launcher_source_sha256"),
            "hardware_profile_source_sha256": hardware_identity["source_sha256"],
        }:
            errors.append("IS-Bench qualification serving verification drift")

        hardware_audit = load_bound_json("hardware_probe_verification")
        expected_probe_result_bindings = [
            {
                "path": str(expected_paths[f"infrastructure_probe_result_{index}"]),
                "sha256": bindings[f"infrastructure_probe_result_{index}"][
                    "sha256"
                ],
            }
            for index in (1, 2)
            if f"infrastructure_probe_result_{index}" in expected_paths
        ]
        expected_hardware_audit = {
            "schema_version": qualification_schemas[
                "hardware_probe_verification"
            ],
            "status": "verified",
            "hardware_profile_id": hardware_identity["profile_id"],
            "hardware_profile_sha256": hardware_identity["source_sha256"],
            "hardware_claims": hardware_identity["hardware_claims"],
            "probe_job_id": str(infrastructure_summary.get("slurm_job_id") or ""),
            "hardware_profile": {
                "path": str(expected_paths["hardware_profile"]),
                "sha256": hardware_identity["source_sha256"],
            },
            "probe_summary": {
                "path": str(infrastructure_summary_path.resolve()),
                "sha256": bindings["infrastructure_probe_summary"]["sha256"],
            },
            "probe_results": expected_probe_result_bindings,
        }
        if hardware_audit != expected_hardware_audit:
            errors.append("IS-Bench qualification hardware probe audit drift")

        slurm_window = load_bound_json("slurm_time_window")
        errors.extend(
            _validate_isbench_qualification_slurm_window(
                slurm_window,
                hardware_contract=hardware_contract,
                gpu_identity_payload=gpu_payload,
            )
        )
        if v4_service_evidence_required:
            service_job_id = (
                str(slurm_window.get("service_job_id") or "")
                if isinstance(slurm_window, dict)
                else ""
            )
            errors.extend(
                _validate_isbench_v4_service_evidence(
                    artifacts={
                        name: resolved[name]
                        for name in _ISBENCH_V4_SERVICE_BINDING_NAMES
                        if name in resolved
                    },
                    bindings=bindings,
                    serving=serving,
                    expected_service_job_id=service_job_id,
                    expected_runtime_phase="qualification_terminal",
                    expected_implementation_sha256=str(implementation or ""),
                )
            )
        if v5_submission_identity_required:
            probe_contract = hardware_contract.get("probe_contract") or {}
            expected_identity_inputs = {
                "launcher_source": resolved.get(
                    "launcher", project_root / "__missing_launcher__"
                ),
                "submitter_source": (
                    project_root / str(serving.get("submitter_source_path") or "")
                ).resolve(),
                "formal_preregistration": preregistration_path.resolve(),
                "serving_contract": expected_paths["serving_contract"],
                "hardware_profile": expected_paths["hardware_profile"],
                "model_profile": model_profile_path.resolve(),
                "formal_task_list": (
                    project_root
                    / "config/task_lists/isbench_qwen35_27b_formal_full161_v1.txt"
                ).resolve(),
                "qualification_task_list": expected_paths[
                    "qualification_task_list"
                ],
                "compositionality_manifest": (
                    project_root
                    / "config/decision_faithfulness/compositionality_manifest_v1.json"
                ).resolve(),
                "faithfulness_protocol": (
                    project_root
                    / "config/decision_faithfulness/faithfulness_adjudication_protocol_v1.md"
                ).resolve(),
                "blinding_protocol": (
                    project_root
                    / "config/decision_faithfulness/adjudicator_blinding_protocol_v1.md"
                ).resolve(),
                "probe_launcher_source": (
                    project_root
                    / str(probe_contract.get("launcher_source_path") or "")
                ).resolve(),
                "probe_python_source": (
                    project_root
                    / str(probe_contract.get("probe_python_source_path") or "")
                ).resolve(),
                "workflow_entrypoint": (project_root / "wm-safety").resolve(),
                "registered_probe_summary": infrastructure_summary_path.resolve(),
                "shared_service_launcher": (
                    project_root
                    / str(serving.get("launcher_source_path") or "")
                ).resolve(),
            }
            errors.extend(
                _validate_isbench_submission_identity_evidence(
                    artifacts={
                        name: resolved[name]
                        for name in _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
                        if name in resolved
                    },
                    bindings=bindings,
                    expected_preflight_phase="qualification_preflight",
                    expected_terminal_phase="qualification_terminal",
                    expected_input_paths=expected_identity_inputs,
                )
            )

        outer_scan = load_bound_json("outer_active_credential_scan")
        if not isinstance(outer_scan, dict) or not (
            outer_scan.get("status") == "passed"
            and int(outer_scan.get("leak_path_count") or 0) == 0
            and not outer_scan.get("leak_paths")
        ):
            errors.append(
                "IS-Bench qualification outer active credential scan failed"
            )
    qualification_checkpoint_path = resolved.get("checkpoint_verification")
    if qualification_checkpoint_path is not None and checkpoint_verification_path is not None:
        try:
            qualification_checkpoint = json.loads(
                qualification_checkpoint_path.read_text(encoding="utf-8")
            )
            formal_checkpoint = json.loads(
                checkpoint_verification_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            errors.append("IS-Bench qualification checkpoint audit is invalid")
        else:
            identity_fields = {
                "schema_version",
                "status",
                "repository",
                "commit",
                "lfs_manifest_sha256",
                "lfs_content_verified",
                "runtime_pip_freeze_sha256",
            }
            if (
                not isinstance(qualification_checkpoint, dict)
                or not isinstance(formal_checkpoint, dict)
                or {field: qualification_checkpoint.get(field) for field in identity_fields}
                != {field: formal_checkpoint.get(field) for field in identity_fields}
                or qualification_checkpoint.get("status") != "verified"
                or formal_checkpoint.get("status") != "verified"
            ):
                errors.append(
                    "IS-Bench qualification/formal checkpoint semantic identity differs"
                )
    errors.extend(
        _validate_isbench_qualification_preregistration_sha(
            summary,
            expected_sha256=preregistration_sha256,
            hardware_contract=hardware_contract,
        )
    )

    task_list_path = resolved.get("qualification_task_list")
    qualification_task_ids: list[str] = []
    if task_list_path is not None:
        qualification_task_ids = [
            line.strip()
            for line in task_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if len(qualification_task_ids) != 2 or summary.get(
        "qualification_task_ids"
    ) != qualification_task_ids:
        errors.append("IS-Bench qualification task cohort differs from registered 2-task smoke")

    matrix_path = resolved.get("matrix_manifest")
    expected_condition_ids: set[str] = {"baseline__no_guard__seed-0"}
    aliases = {
        "generic_embodied_policy": "generic",
        "train_policy": "train",
        "full_isbench_policy_text": "full_text",
    }
    expected_condition_ids = set(
        isbench_formal_condition_order(
            experiment_seeds=[0],
            policy_modes=policies,
            intervention_modes=intervention_modes,
        )
    )
    if matrix_path is not None:
        try:
            rows = [
                json.loads(line)
                for line in matrix_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            errors.append("IS-Bench qualification matrix manifest is invalid")
            rows = []
        observed_ids = [str(row.get("condition_id") or "") for row in rows]
        if (
            len(rows) != expected_condition_count
            or set(observed_ids) != expected_condition_ids
            or len(
            observed_ids
            ) != len(set(observed_ids))
        ):
            errors.append("IS-Bench qualification matrix denominator is not exact")
        if any(
            row.get("status") != "passed"
            or (row.get("validation") or {}).get("valid") is not True
            for row in rows
        ):
            errors.append("IS-Bench qualification matrix contains invalid conditions")
        if v4_service_evidence_required:
            strict_gate = len(rows) == expected_condition_count and all(
                (row.get("validation") or {}).get("online_strict_required") is True
                and (row.get("validation") or {}).get(
                    "online_strict_child_passed"
                )
                is True
                and (row.get("validation") or {}).get(
                    "online_strict_parent_audit_passed"
                )
                is True
                for row in rows
            )
            if not strict_gate:
                errors.append("IS-Bench v4 qualification strict gate failed")

            def ledger_complete(
                mode: str,
                *,
                require_decisions: bool,
                expected_row_count: int | None = None,
            ) -> bool:
                selected = [
                    row for row in rows if row.get("intervention_mode") == mode
                ]
                def valid_row(row: dict[str, Any]) -> bool:
                    validation = row.get("validation")
                    if not isinstance(validation, dict):
                        return False
                    ledger_errors = validation.get("request_ledger_errors")
                    if (
                        validation.get("valid") is not True
                        or isinstance(ledger_errors, bool)
                        or not isinstance(ledger_errors, int)
                        or ledger_errors != 0
                    ):
                        return False
                    if not require_decisions:
                        return True
                    decisions = validation.get("safepred_decisions")
                    reports_without = validation.get(
                        "reports_without_safepred_decisions"
                    )
                    return bool(
                        isinstance(decisions, int)
                        and not isinstance(decisions, bool)
                        and decisions > 0
                        and isinstance(reports_without, int)
                        and not isinstance(reports_without, bool)
                        and reports_without == 0
                    )

                return len(selected) == (
                    len(policies)
                    if expected_row_count is None
                    else expected_row_count
                ) and all(
                    valid_row(row) for row in selected
                )

            observed_ledger_gate = {
                "enforced_guard_complete": ledger_complete(
                    "enforced_guard", require_decisions=True
                ),
                "advisory_complete_or_zero_uptake_valid": ledger_complete(
                    "advisory_only", require_decisions=False
                ),
            }
            if prediction_only_registered:
                observed_ledger_gate = {
                    "prediction_only_complete": ledger_complete(
                        "prediction_only", require_decisions=True
                    ),
                    **observed_ledger_gate,
                }
            if DISTRIBUTED_ENFORCED in intervention_modes:
                distributed_rows = [
                    row
                    for row in rows
                    if row.get("intervention_mode") == DISTRIBUTED_ENFORCED
                ]
                perturbations_complete = all(
                    {
                        row.get("wm_return_perturbation")
                        for row in distributed_rows
                        if row.get("policy_mode") == policy
                    }
                    == set(WM_RETURN_PERTURBATIONS)
                    for policy in policies
                )
                observed_ledger_gate = {
                    **observed_ledger_gate,
                    "distributed_enforced_complete": bool(
                        perturbations_complete
                        and ledger_complete(
                            DISTRIBUTED_ENFORCED,
                            require_decisions=True,
                            expected_row_count=(
                                len(policies) * len(WM_RETURN_PERTURBATIONS)
                            ),
                        )
                    ),
                }
            if summary.get("wm_ledger_gate") != observed_ledger_gate or not all(
                observed_ledger_gate.values()
            ):
                errors.append("IS-Bench v4 qualification ledger gate failed")

    if v4_service_evidence_required:
        pairing_path = resolved.get("prediction_only_pairing")
        try:
            pairing_rows = json.loads(pairing_path.read_text(encoding="utf-8"))
        except (AttributeError, OSError, json.JSONDecodeError):
            pairing_rows = None
        expected_prediction_ids = {
            f"{aliases[policy]}__prediction__seed-0"
            for policy in policies
        }
        pairing_gate = bool(
            pairing_rows == []
            if not prediction_only_registered
            else (
                isinstance(pairing_rows, list)
                and len(pairing_rows) == len(policies)
                and all(isinstance(row, dict) for row in pairing_rows)
                and {row.get("condition_id") for row in pairing_rows}
                == expected_prediction_ids
                and all(
                    row.get("experiment_seed") == 0
                    and row.get("baseline_condition_id")
                    == "baseline__no_guard__seed-0"
                    and row.get("status") == "matched"
                    and row.get("matched_task_count") == expected_task_count
                    and row.get("mismatched_tasks") == []
                    for row in pairing_rows
                )
            )
        )
        if not pairing_gate:
            errors.append("IS-Bench v4 qualification pairing gate failed")

        provenance_path = resolved.get("provenance_manifest")
        try:
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
        except (AttributeError, OSError, json.JSONDecodeError):
            provenance = None
        provenance_gate = False
        if isinstance(provenance, dict):
            provenance_copy = dict(provenance)
            observed_hash = provenance_copy.pop("manifest_payload_sha256", None)
            expected_hash = hashlib.sha256(
                json.dumps(
                    provenance_copy,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            config = provenance.get("config") or {}
            status = provenance.get("status") or {}
            observed_implementation = (
                ((config.get("online_advisory_contract") or {}).get(
                    "provenance"
                ) or {}).get("implementation_manifest_sha256")
            )
            provenance_gate = bool(
                observed_hash == expected_hash
                and config.get("formal") is False
                and config.get("online_strict_qualification") is True
                and config.get("artifact_class") == "diagnostic"
                and config.get("expected_task_ids") == qualification_task_ids
                and config.get("agent_base_url") == actor.get("endpoint")
                and config.get("safepred_base_url") == actor.get("endpoint")
                and observed_implementation == implementation
                and status.get("matrix_complete") is True
                and status.get("conditions_recorded")
                == expected_condition_count
                and status.get("failed_run_count") == 0
                and status.get("skipped_run_count") == 0
            )
        if not provenance_gate:
            errors.append("IS-Bench v4 qualification provenance gate failed")

    task_scores_path = resolved.get("task_scores")
    if task_scores_path is not None:
        try:
            with task_scores_path.open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                score_rows = list(csv.DictReader(handle))
        except OSError:
            score_rows = []
        identities = {
            (str(row.get("condition_id") or ""), str(row.get("task_id") or ""))
            for row in score_rows
        }
        expected_identities = {
            (condition_id, task_id)
            for condition_id in expected_condition_ids
            for task_id in qualification_task_ids
        }
        if (
            len(score_rows) != expected_task_episode_count
            or identities != expected_identities
        ):
            errors.append(
                "IS-Bench qualification task-episode denominator is not exact"
            )

    scan_path = resolved.get("active_credential_scan")
    if scan_path is not None:
        try:
            scan = json.loads(scan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            scan = None
        if not isinstance(scan, dict) or scan.get("status") != "passed" or int(
            scan.get("leak_path_count") or 0
        ) != 0 or scan.get("leak_paths"):
            errors.append("IS-Bench qualification active credential scan failed")
    return errors


def _validate_checkpoint(stack: Any, checkpoint: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(checkpoint, dict):
        return {"errors": ["checkpoint verification is not a JSON object"]}
    if checkpoint.get("schema_version") != "model-checkpoint-verification-v1":
        errors.append("checkpoint verification schema_version is invalid")
    if checkpoint.get("status") != "verified":
        errors.append("checkpoint status is not verified")
    if checkpoint.get("lfs_content_verified") is not True:
        errors.append("checkpoint LFS content was not fully verified")
    if stack is not None:
        # A checkpoint verification describes the local serving stack only.
        # Endpoint locality is the canonical distinction: a local endpoint
        # can require authentication and a remote endpoint can be unauthenticated.
        # A missing or malformed endpoint never silently claims local-checkpoint
        # provenance.
        local_roles = [
            role
            for role in stack.roles.values()
            if _is_local_serving_endpoint(getattr(role, "base_url", ""))
        ]
        if not local_roles:
            errors.append(
                "checkpoint verification was supplied but model profile has no local roles"
            )
        expected_revisions = {role.revision for role in local_roles}
        expected_runtimes = {role.runtime_revision for role in local_roles}
        if local_roles and len(expected_revisions) != 1:
            errors.append("local profile roles do not share one checkpoint revision")
        if local_roles and len(expected_runtimes) != 1:
            errors.append("local profile roles do not share one runtime revision")
        commit = str(checkpoint.get("commit") or "")
        lfs_digest = str(checkpoint.get("lfs_manifest_sha256") or "")
        runtime_digest = str(checkpoint.get("runtime_pip_freeze_sha256") or "")
        if len(expected_revisions) == 1:
            (revision,) = expected_revisions
            if f"@{commit}" not in revision or not revision.endswith(
                f"lfs-manifest:{lfs_digest}"
            ):
                errors.append("checkpoint verification differs from role revision")
        if len(expected_runtimes) == 1:
            (runtime,) = expected_runtimes
            if not runtime.endswith(f"pip-freeze-sha256:{runtime_digest}"):
                errors.append(
                    "checkpoint runtime differs from role runtime_revision"
                )
    return {
        "schema_version": checkpoint.get("schema_version"),
        "status": checkpoint.get("status"),
        "repository": checkpoint.get("repository"),
        "commit": checkpoint.get("commit"),
        "lfs_manifest_sha256": checkpoint.get("lfs_manifest_sha256"),
        "lfs_content_verified": checkpoint.get("lfs_content_verified"),
        "runtime_pip_freeze_sha256": checkpoint.get(
            "runtime_pip_freeze_sha256"
        ),
        "errors": errors,
    }


def _validate_isbench_task_list_binding(
    payload: dict[str, Any], task_list_path: Path
) -> list[str]:
    """Verify the frozen online cohort and its source-byte digest."""

    errors: list[str] = []
    try:
        task_list_bytes = task_list_path.read_bytes()
        task_ids = [
            line.strip()
            for line in task_list_bytes.decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except (OSError, UnicodeDecodeError) as exc:
        return [f"cannot read IS-Bench task list: {type(exc).__name__}: {exc}"]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        errors.append("IS-Bench task list must be non-empty and unique")
    if int((payload.get("support") or {}).get("min_tasks") or 0) >= 161:
        expected_snapshot = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "task_lists"
            / "isbench_qwen35_27b_formal_full161_v1.txt"
        ).resolve()
        if task_list_path.resolve() != expected_snapshot or len(task_ids) != 161:
            errors.append(
                "full IS-Bench readiness requires the immutable 161-task snapshot"
            )
    try:
        plan = payload["run_plans"]["isbench"]
        protocol = payload["offline_prediction"]["protocol_by_benchmark"][
            "isbench"
        ]
        source_inputs = protocol["source_inputs"]
    except (KeyError, TypeError):
        errors.append("IS-Bench preregistration lacks the frozen task-list binding")
        return errors
    if list(plan.get("task_ids") or []) != task_ids:
        errors.append("IS-Bench task list differs from preregistered task_ids")
    expected_digest = hashlib.sha256(task_list_bytes).hexdigest()
    source_map = source_inputs.get("manifest_sha256_by_actor_trace_seed")
    expected_seed_keys = {str(seed) for seed in plan.get("experiment_seeds") or []}
    if not isinstance(source_map, dict) or set(source_map) != expected_seed_keys:
        errors.append("IS-Bench task-list source map differs from experiment seeds")
    elif any(str(value).lower() != expected_digest for value in source_map.values()):
        errors.append("IS-Bench task-list source digest differs from current bytes")
    if source_inputs.get("manifest_kind") != (
        "not_applicable_to_online_only_decision_faithfulness_component_v1"
    ):
        errors.append("IS-Bench online-only source manifest kind is invalid")
    return errors


def _placeholder_paths(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in _placeholder_paths(item, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in _placeholder_paths(item, f"{prefix}[{index}]")
        ]
    if isinstance(value, str) and value.strip().upper().startswith("REPLACE"):
        return [prefix]
    return []


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate scoped or joint formal pre-execution readiness"
    )
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--model-profile", required=True, type=Path)
    parser.add_argument("--compositionality-manifest", required=True, type=Path)
    parser.add_argument(
        "--faithfulness-adjudication-protocol", required=True, type=Path
    )
    parser.add_argument("--adjudicator-blinding-protocol", required=True, type=Path)
    parser.add_argument("--checkpoint-verification", type=Path)
    parser.add_argument(
        "--isbench-task-list",
        type=Path,
        help="Exact task list bound by an IS-Bench scoped or joint preregistration.",
    )
    parser.add_argument("--omnigibson-infrastructure-summary", type=Path)
    parser.add_argument("--isbench-online-qualification-summary", type=Path)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
        help="Exact benchmark scope to validate.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_formal_readiness(
        preregistration_path=args.preregistration,
        model_profile_path=args.model_profile,
        compositionality_manifest_path=args.compositionality_manifest,
        faithfulness_adjudication_protocol_path=(
            args.faithfulness_adjudication_protocol
        ),
        adjudicator_blinding_protocol_path=args.adjudicator_blinding_protocol,
        checkpoint_verification_path=args.checkpoint_verification,
        benchmarks=args.benchmarks,
        isbench_task_list_path=args.isbench_task_list,
        omnigibson_infrastructure_summary_path=(
            args.omnigibson_infrastructure_summary
        ),
        isbench_online_qualification_summary_path=(
            args.isbench_online_qualification_summary
        ),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
