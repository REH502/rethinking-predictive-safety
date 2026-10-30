"""Independent, non-circular terminal gate for a formal IS-Bench matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.formal_readiness import (
    _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES,
    _ISBENCH_V4_SERVICE_BINDING_NAMES,
    _ISBENCH_V4_SERVING_SCHEMA,
    _isbench_v5_submission_identity_required,
    _validate_isbench_submission_identity_evidence,
    _validate_isbench_v4_service_evidence,
)
from experiments.workflows import load_experiment_credentials
from multi_agent.safepred_experiments.formal_contract import (
    ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS,
    ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS,
    isbench_omnigibson_hardware_identity,
    isbench_uses_registered_single_seed_one_pass_contract,
    load_preregistration,
)
from multi_agent.safepred_experiments.isbench_online_matrix import (
    _verified_completed_formal_artifacts,
)


REQUIRED_FILES = {
    "active_credential_scan.json",
    "formal_condition_lifecycle.jsonl",
    "formal_online_identity.json",
    "formal_preregistration.json",
    "formal_run_lifecycle.jsonl",
    "online_matrix_manifest.jsonl",
    "online_matrix_provenance_manifest.json",
    "online_task_scores.csv",
}


_QWEN_GPU_IDENTITIES_BY_SLURM_TYPE = {
    "nvidia_a100_80gb_pcie": {
        "exact_nvidia_smi_name": "NVIDIA A100 80GB PCIe",
        "minimum_memory_total_mib": 80000,
    },
    "nvidia_a100_80gb_sxm4": {
        "exact_nvidia_smi_name": "NVIDIA A100-SXM4-80GB",
        "minimum_memory_total_mib": 80000,
    },
}


def _qwen_serving_hardware_identity(
    serving: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive exact Qwen node/GPU expectations from the frozen contract."""

    if not isinstance(serving, dict):
        raise ValueError("formal invocation requires a serving contract")
    node = str(serving.get("slurm_node") or "").strip()
    gpu_type = str(serving.get("slurm_gpu_type") or "").strip()
    gpu_count = serving.get("slurm_gpu_count")
    identity = _QWEN_GPU_IDENTITIES_BY_SLURM_TYPE.get(gpu_type)
    if identity is None:
        raise ValueError("formal invocation serving GPU type is unregistered")
    if (
        not node
        or isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or gpu_count != 1
    ):
        raise ValueError("formal invocation serving GPU allocation is invalid")
    return {
        "node": node,
        "slurm_gpu_type": gpu_type,
        **identity,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name}:{number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{number} is not an object")
        rows.append(value)
    return rows


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("formal root contains a symlink")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("formal root contains a non-regular artifact")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _active_credentials() -> list[bytes]:
    loaded = load_experiment_credentials(dict(os.environ))
    values = {
        str(value).encode("utf-8")
        for name, value in loaded.items()
        if (
            "API_KEY" in str(name).upper()
            or "ACCESS_TOKEN" in str(name).upper()
            or str(name).upper().endswith("_TOKEN")
        )
        and not str(name).upper().endswith("_API_KEY_ENV")
        and len(str(value or "")) >= 12
    }
    # The registered local Qwen profile uses auth:none.  Its launcher enters
    # every child with a 0600 empty environment file and removes API-key/token
    # variables, so an empty active set is the expected fail-closed identity.
    return sorted(values)


def _contains_any(path: Path, values: Sequence[bytes]) -> bool:
    overlap = max((len(value) for value in values), default=1) - 1
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            payload = tail + chunk
            if any(value in payload for value in values):
                return True
            tail = payload[-overlap:] if overlap else b""
    return False


def _canonical_payload_sha256(value: dict[str, Any], field: str) -> bool:
    copy = dict(value)
    observed = copy.pop(field, None)
    encoded = json.dumps(copy, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return observed == hashlib.sha256(encoded).hexdigest()


def _compact_payload_sha256(value: dict[str, Any], field: str) -> bool:
    copy = dict(value)
    observed = copy.pop(field, None)
    encoded = json.dumps(
        copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return observed == hashlib.sha256(encoded).hexdigest()


def _scene_binding_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_scene_template_provenance(
    *,
    identity: dict[str, Any],
    provenance: dict[str, Any],
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Cross-check frozen scene bindings, final provenance, and current bytes."""

    if identity.get("schema_version") != 2:
        raise ValueError("formal online identity lacks scene-template schema v2")
    preflight = identity.get("scene_template_preflight")
    if not isinstance(preflight, dict):
        raise ValueError("formal online identity lacks scene-template preflight")
    templates = preflight.get("templates")
    archive = preflight.get("source_archive")
    expected = list(expected_task_ids)
    if (
        preflight.get("schema_version")
        != "isbench-scene-template-preflight-v1"
        or preflight.get("status") != "passed"
        or not isinstance(templates, list)
        or any(not isinstance(row, dict) for row in templates)
        or not isinstance(archive, dict)
        or preflight.get("task_count") != len(expected)
        or preflight.get("validated_template_count") != len(expected)
        or len(templates) != len(expected)
        or [row.get("task_id") for row in templates] != expected
    ):
        raise ValueError("formal scene-template census is incomplete")
    binding_payload = {"source_archive": archive, "templates": templates}
    if preflight.get("bindings_sha256") != _scene_binding_sha256(binding_payload):
        raise ValueError("formal scene-template binding hash drift")
    if any(
        row.get("source_archive_member_sha256") != row.get("sha256")
        for row in templates
    ):
        raise ValueError("formal scene-template archive-member binding drift")

    provenance_config = provenance.get("config")
    if not isinstance(provenance_config, dict) or provenance_config.get(
        "scene_template_preflight"
    ) != preflight:
        raise ValueError("provenance scene-template preflight differs from identity")
    expected_integrity = {
        "status": "passed",
        "task_count": len(expected),
        "validated_template_count": len(templates),
        "bindings_sha256": preflight["bindings_sha256"],
    }
    if provenance_config.get("scene_template_integrity") != expected_integrity:
        raise ValueError("provenance scene-template terminal integrity is missing")

    identity_rows = identity.get("input_files")
    provenance_rows = provenance.get("inputs")
    if not isinstance(identity_rows, list) or not isinstance(provenance_rows, list):
        raise ValueError("scene-template input fingerprints are missing")
    if any(not isinstance(row, dict) for row in [*identity_rows, *provenance_rows]):
        raise ValueError("scene-template input fingerprints are invalid")
    identity_by_path = {str(row.get("path") or ""): row for row in identity_rows}
    provenance_by_path = {str(row.get("path") or ""): row for row in provenance_rows}
    if (
        "" in identity_by_path
        or "" in provenance_by_path
        or len(identity_by_path) != len(identity_rows)
        or len(provenance_by_path) != len(provenance_rows)
    ):
        raise ValueError("scene-template input fingerprints have blank/duplicate paths")

    archive_relative = Path(str(archive.get("relative_path") or ""))
    if (
        archive_relative != Path("data/scenes.tar.gz")
        or archive_relative.is_absolute()
        or ".." in archive_relative.parts
    ):
        raise ValueError("scene-template archive path is invalid")
    archive_candidates = [
        Path(path_text)
        for path_text in identity_by_path
        if Path(path_text).parts[-len(archive_relative.parts) :]
        == archive_relative.parts
    ]
    if len(archive_candidates) != 1:
        raise ValueError("scene-template archive identity is missing or ambiguous")
    is_root = archive_candidates[0].parent.parent

    records = [archive, *templates]
    for record in records:
        relative = Path(str(record.get("relative_path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("scene-template provenance path is invalid")
        path = is_root / relative
        path_text = str(path)
        identity_row = identity_by_path.get(path_text)
        provenance_row = provenance_by_path.get(path_text)
        if identity_row is None or provenance_row is None:
            raise ValueError("scene-template fingerprint is missing from provenance")
        if (
            identity_row.get("size_bytes") != record.get("size_bytes")
            or identity_row.get("sha256") != record.get("sha256")
            or provenance_row.get("exists") is not True
            or provenance_row.get("size_bytes") != record.get("size_bytes")
            or provenance_row.get("sha256") != record.get("sha256")
        ):
            raise ValueError("scene-template identity/provenance fingerprint drift")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("scene-template current bytes changed")
    return expected_integrity


def _validate_invocation_chain(
    outer: Path,
    *,
    expected_serving_contract: dict[str, Any] | None = None,
    expected_hardware_contract: dict[str, Any] | None = None,
    expected_implementation_sha256: str | None = None,
) -> dict[str, Any]:
    v4_service_evidence_required = (
        isinstance(expected_serving_contract, dict)
        and expected_serving_contract.get("schema_version")
        == _ISBENCH_V4_SERVING_SCHEMA
    )
    v5_submission_identity_required = (
        _isbench_v5_submission_identity_required(expected_serving_contract)
    )
    qwen_identity = _qwen_serving_hardware_identity(expected_serving_contract)
    hardware_identity = (
        isbench_omnigibson_hardware_identity(expected_hardware_contract)
        if expected_hardware_contract is not None
        else None
    )
    expected_omnigibson_node = (
        hardware_identity["node"] if hardware_identity is not None else "gpu07"
    )
    expected_omnigibson_name = (
        hardware_identity["exact_nvidia_smi_name"]
        if hardware_identity is not None
        else "NVIDIA RTX A6000"
    )
    expected_omnigibson_memory = (
        hardware_identity["minimum_memory_total_mib"]
        if hardware_identity is not None
        else 48000
    )
    index_path = outer / "formal_invocation_index.jsonl"
    if not index_path.is_file():
        raise ValueError("formal outer root lacks invocation index")
    rows = _jsonl(index_path)
    if not rows:
        raise ValueError("formal invocation index is empty")
    expected_index_keys = {
        "schema_version",
        "sequence",
        "job_id",
        "manifest_path",
        "size_bytes",
        "sha256",
        "status",
        "matrix_lifecycle_status",
    }
    manifests: list[dict[str, Any]] = []
    observed_job_ids: set[str] = set()
    prior_finished_at: datetime | None = None
    for expected_sequence, row in enumerate(rows, 1):
        if set(row) != expected_index_keys or row.get("schema_version") != (
            "isbench-formal-invocation-index-row-v1"
        ):
            raise ValueError("formal invocation index row schema drift")
        if row.get("sequence") != expected_sequence:
            raise ValueError("formal invocation sequence has a gap or duplicate")
        job_id = str(row.get("job_id") or "")
        if not job_id.isdigit() or job_id in observed_job_ids:
            raise ValueError("formal invocation job identity is invalid or duplicate")
        observed_job_ids.add(job_id)
        expected_relative = f"invocations/job-{job_id}/invocation_manifest.json"
        if row.get("manifest_path") != expected_relative:
            raise ValueError("formal invocation manifest path is not canonical")
        manifest_path = (outer / expected_relative).resolve()
        if outer not in manifest_path.parents or not manifest_path.is_file():
            raise ValueError("formal invocation manifest is missing or outside root")
        if manifest_path.stat().st_size != row.get("size_bytes") or _sha256(
            manifest_path
        ) != row.get("sha256"):
            raise ValueError("formal invocation manifest binding changed")
        manifest = _json(manifest_path)
        expected_manifest_keys = {
            "schema_version",
            "sequence",
            "job_id",
            "resume",
            "status",
            "started_at_utc",
            "finished_at_utc",
            "slurm",
            "matrix",
            "evidence",
            "manifest_payload_sha256",
        }
        expected_manifest_schema = (
            "isbench-formal-invocation-manifest-v2"
            if v4_service_evidence_required
            else "isbench-formal-invocation-manifest-v1"
        )
        if set(manifest) != expected_manifest_keys or manifest.get(
            "schema_version"
        ) != expected_manifest_schema:
            raise ValueError("formal invocation manifest schema drift")
        if not _compact_payload_sha256(manifest, "manifest_payload_sha256"):
            raise ValueError("formal invocation manifest self-hash failed")
        if manifest.get("sequence") != expected_sequence or str(
            manifest.get("job_id") or ""
        ) != job_id:
            raise ValueError("formal invocation manifest/index identity differs")
        try:
            started_at = datetime.fromisoformat(
                str(manifest.get("started_at_utc") or "").replace("Z", "+00:00")
            )
            finished_at = datetime.fromisoformat(
                str(manifest.get("finished_at_utc") or "").replace("Z", "+00:00")
            )
        except ValueError:
            raise ValueError("formal invocation timestamps are invalid") from None
        if (
            started_at.tzinfo is None
            or finished_at.tzinfo is None
            or finished_at < started_at
            or (prior_finished_at is not None and started_at < prior_finished_at)
        ):
            raise ValueError("formal invocation timestamps overlap or regress")
        prior_finished_at = finished_at
        if manifest.get("status") != row.get("status") or (
            manifest.get("matrix") or {}
        ).get("lifecycle_status") != row.get("matrix_lifecycle_status"):
            raise ValueError("formal invocation manifest/index status differs")
        slurm = manifest.get("slurm")
        expected_slurm_keys = {
            "leader_job_id",
            "group0_job_id",
            "group1_job_id",
            "start_epoch",
            "end_epoch",
            "allocated_walltime_seconds",
            "invocation_deadline_epoch",
            "postflight_reserve_seconds",
            "omnigibson_node",
            "qwen_node",
        }
        if not isinstance(slurm, dict) or set(slurm) != expected_slurm_keys:
            raise ValueError("formal invocation SLURM identity schema drift")
        if str(slurm.get("leader_job_id") or "") != job_id:
            raise ValueError("formal invocation SLURM leader differs from job id")
        group0_job_id = str(slurm.get("group0_job_id") or "")
        group1_job_id = str(slurm.get("group1_job_id") or "")
        if (
            group0_job_id != job_id
            or not group1_job_id.isdigit()
            or group1_job_id == job_id
        ):
            raise ValueError(
                "formal invocation runner/service job identity is invalid"
            )
        if slurm.get("omnigibson_node") != expected_omnigibson_node or slurm.get(
            "qwen_node"
        ) != qwen_identity["node"]:
            raise ValueError("formal invocation node identity drift")
        try:
            start_epoch = int(slurm["start_epoch"])
            end_epoch = int(slurm["end_epoch"])
            walltime = int(slurm["allocated_walltime_seconds"])
            deadline_epoch = int(slurm["invocation_deadline_epoch"])
            reserve_seconds = int(slurm["postflight_reserve_seconds"])
        except (TypeError, ValueError):
            raise ValueError("formal invocation SLURM time window is invalid") from None
        if (
            start_epoch <= 0
            or end_epoch <= start_epoch
            or walltime != end_epoch - start_epoch
            or reserve_seconds <= 0
            or deadline_epoch != end_epoch - reserve_seconds
            or not start_epoch < deadline_epoch < end_epoch
        ):
            raise ValueError("formal invocation SLURM time window is inconsistent")
        evidence = manifest.get("evidence")
        required_evidence = {
            "launcher",
            "checkpoint_verification",
            "omnigibson_gpu_identity",
            "qwen_gpu_identity",
            "endpoint_preflight",
            "slurm_time_window",
            "readiness_report",
            "invocation_postflight",
        }
        if hardware_identity is not None:
            required_evidence.add("hardware_profile")
        if v4_service_evidence_required:
            required_evidence |= _ISBENCH_V4_SERVICE_BINDING_NAMES
        if v5_submission_identity_required:
            required_evidence |= _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
        if not isinstance(evidence, dict) or set(evidence) != required_evidence:
            if v5_submission_identity_required and (
                not isinstance(evidence, dict)
                or not _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES.issubset(evidence)
            ):
                raise ValueError(
                    "formal invocation submission identity evidence set is incomplete"
                )
            raise ValueError("formal invocation evidence set is incomplete")
        resolved_evidence: dict[str, Path] = {}
        for name, binding in evidence.items():
            if not isinstance(binding, dict) or set(binding) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError(f"formal invocation evidence {name} binding is invalid")
            relative = Path(str(binding.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("formal invocation evidence path is unsafe")
            raw_evidence_path = outer / relative
            if raw_evidence_path.is_symlink():
                raise ValueError("formal invocation evidence path is a symlink")
            evidence_path = raw_evidence_path.resolve()
            expected_parent = outer / "invocations" / f"job-{job_id}"
            if expected_parent not in evidence_path.parents or not evidence_path.is_file():
                raise ValueError("formal invocation evidence is outside its job directory")
            if evidence_path.stat().st_size != binding.get("size_bytes") or _sha256(
                evidence_path
            ) != binding.get("sha256"):
                raise ValueError("formal invocation evidence binding changed")
            resolved_evidence[name] = evidence_path
        if expected_serving_contract is not None and evidence["launcher"].get(
            "sha256"
        ) != expected_serving_contract.get("launcher_source_sha256"):
            raise ValueError("formal invocation launcher differs from serving contract")
        if hardware_identity is not None and evidence["hardware_profile"].get(
            "sha256"
        ) != hardware_identity["source_sha256"]:
            raise ValueError(
                "formal invocation hardware profile differs from preregistration"
            )
        time_window = _json(outer / evidence["slurm_time_window"]["path"])
        if time_window != slurm:
            raise ValueError("formal invocation SLURM evidence differs from manifest")
        checkpoint = _json(
            outer / evidence["checkpoint_verification"]["path"]
        )
        if checkpoint.get("schema_version") != "model-checkpoint-verification-v1" or (
            checkpoint.get("status") != "verified"
            or checkpoint.get("lfs_content_verified") is not True
        ):
            raise ValueError("formal invocation checkpoint verification did not pass")
        if expected_serving_contract is not None and any(
            checkpoint.get(field) != expected_serving_contract.get(expected_field)
            for field, expected_field in (
                ("commit", "checkpoint_commit"),
                ("lfs_manifest_sha256", "lfs_manifest_sha256"),
                ("runtime_pip_freeze_sha256", "runtime_freeze_sha256"),
            )
        ):
            raise ValueError("formal invocation checkpoint differs from serving contract")
        endpoint = _json(outer / evidence["endpoint_preflight"]["path"])
        if endpoint.get("status") != "passed" or endpoint.get(
            "requested_model"
        ) != "qwen3.5-27b" or endpoint.get("inference_probe") != "passed":
            raise ValueError("formal invocation real-inference endpoint preflight failed")
        if expected_serving_contract is not None and str(
            endpoint.get("base_url") or ""
        ).rstrip("/") != str(expected_serving_contract.get("base_url") or "").rstrip(
            "/"
        ):
            raise ValueError("formal invocation endpoint differs from serving contract")
        if v4_service_evidence_required:
            if not expected_implementation_sha256:
                raise ValueError(
                    "formal invocation v4 implementation identity is missing"
                )
            service_errors = _validate_isbench_v4_service_evidence(
                artifacts={
                    name: resolved_evidence[name]
                    for name in _ISBENCH_V4_SERVICE_BINDING_NAMES
                },
                bindings={
                    name: {
                        "path": str(resolved_evidence[name]),
                        "sha256": evidence[name]["sha256"],
                    }
                    for name in _ISBENCH_V4_SERVICE_BINDING_NAMES
                },
                serving=expected_serving_contract,
                expected_service_job_id=group1_job_id,
                expected_runtime_phase="formal_invocation_terminal",
                expected_implementation_sha256=expected_implementation_sha256,
            )
            if service_errors:
                raise ValueError(
                    "formal invocation v4 service evidence failed: "
                    + "; ".join(service_errors)
                )
        if v5_submission_identity_required:
            identity_errors = _validate_isbench_submission_identity_evidence(
                artifacts={
                    name: resolved_evidence[name]
                    for name in _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
                },
                bindings={
                    name: {
                        "path": str(resolved_evidence[name]),
                        "sha256": evidence[name]["sha256"],
                    }
                    for name in _ISBENCH_SUBMISSION_IDENTITY_BINDING_NAMES
                },
                expected_preflight_phase="formal_invocation_preflight",
                expected_terminal_phase="formal_invocation_terminal",
            )
            if identity_errors:
                raise ValueError(
                    "formal invocation submission identity evidence failed: "
                    + "; ".join(identity_errors)
                )
        for evidence_name, expected_name, minimum_mib in (
            (
                "omnigibson_gpu_identity",
                expected_omnigibson_name,
                expected_omnigibson_memory,
            ),
            (
                "qwen_gpu_identity",
                qwen_identity["exact_nvidia_smi_name"],
                qwen_identity["minimum_memory_total_mib"],
            ),
        ):
            gpu_path = outer / evidence[evidence_name]["path"]
            rows = [
                line.strip()
                for line in gpu_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            try:
                name, memory_text = [item.strip() for item in rows[0].rsplit(",", 1)]
                memory_mib = int(memory_text.removesuffix("MiB").strip())
            except (IndexError, TypeError, ValueError):
                raise ValueError("formal invocation GPU identity is invalid") from None
            if len(rows) != 1 or name != expected_name or memory_mib < minimum_mib:
                raise ValueError("formal invocation GPU identity drift")
        readiness = _json(outer / evidence["readiness_report"]["path"])
        if readiness.get("formal_ready") is not True or readiness.get("errors") != [] or (
            readiness.get("benchmarks") != ["isbench"]
        ):
            raise ValueError("formal invocation readiness report did not pass")
        invocation_postflight = _json(
            outer / evidence["invocation_postflight"]["path"]
        )
        expected_classification = (
            "formal_candidate_pending_blinded_postflight"
            if manifest.get("status") == "completed"
            else "unevaluable_resume_required"
        )
        if invocation_postflight.get("classification") != expected_classification or (
            invocation_postflight.get("lifecycle_status")
            != (manifest.get("matrix") or {}).get("lifecycle_status")
        ):
            raise ValueError("formal invocation postflight classification drift")
        manifests.append(manifest)
    invocations_root = outer / "invocations"
    if (
        invocations_root.is_symlink()
        or not invocations_root.is_dir()
        or any(child.is_symlink() or not child.is_dir() for child in invocations_root.iterdir())
        or {child.name for child in invocations_root.iterdir()}
        != {f"job-{job_id}" for job_id in observed_job_ids}
    ):
        raise ValueError("formal invocation directory census is not exactly sealed")
    if any(manifest.get("status") != "paused_clean" for manifest in manifests[:-1]):
        raise ValueError("a prior formal invocation was not a clean resumable pause")
    last = manifests[-1]
    last_matrix = last.get("matrix") or {}
    if last.get("status") != "completed" or last_matrix != {
        "runner_exit_code": 0,
        "invocation_postflight_exit_code": 0,
        "lifecycle_status": "matrix_outputs_validated",
    }:
        raise ValueError("last formal invocation is not a completed validated matrix")
    result = {
        "invocation_count": len(manifests),
        "job_ids": [str(item["job_id"]) for item in manifests],
        "last_job_id": str(last["job_id"]),
        "last_status": "completed",
    }
    if v4_service_evidence_required:
        result["v4_service_evidence_validated"] = True
    if v5_submission_identity_required:
        result["submission_identity_evidence_validated"] = True
    if hardware_identity is not None:
        claims = hardware_identity["hardware_claims"]
        result.update(
            {
                "hardware_profile_id": hardware_identity["profile_id"],
                "external_benchmark_comparability": claims[
                    "external_benchmark_comparability"
                ],
                "primary_claim_scope": claims["claim_scope"],
            }
        )
    return result


def _isbench_terminal_denominators(
    plan: dict[str, Any],
    *,
    hardware_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive and fail-closed check the five-seed or registered one-pass denominator."""

    registered_single_seed = (
        isbench_uses_registered_single_seed_one_pass_contract(plan)
    )
    internal_hardware_arm = hardware_contract is not None
    if internal_hardware_arm:
        isbench_omnigibson_hardware_identity(hardware_contract)
        registered_hardware = (
            ((plan.get("online_advisory_contract") or {}).get("runtime") or {}).get(
                "hardware_contract"
            )
        )
        if registered_hardware != hardware_contract or not registered_single_seed:
            raise ValueError(
                "formal single-seed cohort contract is not the canonical "
                "registered full-161 one-pass design"
            )
    task_count = len(plan.get("task_ids") or [])
    seeds = list(plan.get("experiment_seeds") or [])
    policies = list(plan.get("policy_modes") or [])
    expected_seeds = [0] if internal_hardware_arm else [0, 1, 2, 3, 4]
    if (
        task_count != 161
        or seeds != expected_seeds
        or set(policies) not in ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS
        or len(policies) != len(set(policies))
    ):
        raise ValueError(
            "formal task/seed identity differs from the frozen cohort contract"
        )
    intervention_modes = list(plan.get("intervention_modes") or [])
    registered_modes = set(intervention_modes)
    if (
        registered_modes not in ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS
        or len(intervention_modes) != len(registered_modes)
    ):
        raise ValueError("formal intervention matrix is not a registered arm set")
    guarded_modes = [mode for mode in intervention_modes if mode != "no_guard"]
    seed_count = len(seeds)
    guarded_multiplicity = sum(
        4 if mode == "distributed_enforced" else 1
        for mode in guarded_modes
    )
    condition_count = seed_count * (1 + len(policies) * guarded_multiplicity)
    return {
        "task_count": task_count,
        "seed_count": seed_count,
        "condition_count": condition_count,
        "task_episode_count": task_count * condition_count,
        "condition_count_by_arm": {
            mode: (
                seed_count
                if mode == "no_guard"
                else seed_count
                * len(policies)
                * (4 if mode == "distributed_enforced" else 1)
            )
            for mode in intervention_modes
        },
    }


def run_terminal_postflight(
    *,
    formal_outer_root: Path,
    output_root: Path,
    supplied_credentials: Sequence[bytes] | None = None,
) -> dict[str, Any]:
    outer = Path(formal_outer_root).resolve()
    formal = outer / "matrix"
    output = Path(output_root).resolve()
    if not outer.is_dir() or not formal.is_dir() or output.exists():
        raise ValueError("formal outer/matrix roots must exist and output must be new")
    if output == outer or outer in output.parents:
        raise ValueError("postflight output must be outside the formal outer root")
    missing = sorted(name for name in REQUIRED_FILES if not (formal / name).is_file())
    if missing:
        raise ValueError("formal root is missing: " + ", ".join(missing))
    before = _inventory(outer)
    registration = load_preregistration(
        formal / "formal_preregistration.json", benchmark="isbench"
    )
    payload = registration["payload"]
    plan = payload["run_plans"]["isbench"]
    runtime_contract = plan["online_advisory_contract"]["runtime"]
    hardware_contract = runtime_contract.get("hardware_contract")
    implementation_sha256 = str(
        (
            plan["online_advisory_contract"].get("provenance") or {}
        ).get("implementation_manifest_sha256")
        or ""
    )
    invocation_chain = _validate_invocation_chain(
        outer,
        expected_serving_contract=runtime_contract.get("serving_contract"),
        expected_hardware_contract=hardware_contract,
        expected_implementation_sha256=implementation_sha256,
    )
    task_ids = list(plan["task_ids"])
    seeds = list(plan["experiment_seeds"])
    policies = list(plan["policy_modes"])
    denominators = _isbench_terminal_denominators(
        plan,
        hardware_contract=hardware_contract,
    )
    schedule = plan["online_advisory_contract"]["runtime"]["execution_schedule"]
    condition_ids = list(schedule["condition_order"])
    registered_modes = list(plan.get("intervention_modes") or [])
    guarded_modes = [mode for mode in registered_modes if mode != "no_guard"]
    guarded_multiplicity = sum(
        4 if mode == "distributed_enforced" else 1
        for mode in guarded_modes
    )
    expected_conditions = len(seeds) * (
        1 + len(policies) * guarded_multiplicity
    )
    expected_episodes = len(task_ids) * expected_conditions
    expected_seed_count = denominators["seed_count"]
    expected_condition_count = denominators["condition_count"]
    expected_episode_count = denominators["task_episode_count"]
    if expected_conditions != expected_condition_count or expected_episodes != (
        expected_episode_count
    ):
        raise ValueError(
            "formal denominator differs from the frozen task-census seed identity"
        )
    if len(condition_ids) != expected_condition_count or len(
        set(condition_ids)
    ) != expected_condition_count:
        raise ValueError("registered condition schedule denominator is invalid")

    if _verified_completed_formal_artifacts(
        formal, planned_condition_count=expected_conditions
    ) is None:
        raise ValueError("formal root is not terminal")
    manifest = _jsonl(formal / "online_matrix_manifest.jsonl")
    observed_ids = [str(row.get("condition_id") or "") for row in manifest]
    if observed_ids != condition_ids:
        raise ValueError("matrix manifest order differs from frozen schedule")
    if any(
        row.get("status") != "passed"
        or (row.get("validation") or {}).get("valid") is not True
        for row in manifest
    ):
        raise ValueError("matrix contains failed/failed_validation/skipped conditions")
    arm_counts = {
        arm: sum(row.get("intervention_mode") == arm for row in manifest)
        for arm in registered_modes
    }
    expected_arm_counts = denominators["condition_count_by_arm"]
    if arm_counts != expected_arm_counts:
        raise ValueError("matrix arm counts differ from the frozen seed identity")
    if "distributed_enforced" in registered_modes:
        expected_mode_count = len(seeds) * len(policies)
        perturbation_counts = {
            mode: sum(
                row.get("intervention_mode") == "distributed_enforced"
                and row.get("wm_return_perturbation") == mode
                for row in manifest
            )
            for mode in ("empty", "corrupt", "irrelevant", "filler")
        }
        if any(count != expected_mode_count for count in perturbation_counts.values()):
            raise ValueError(
                "matrix WM-return perturbation counts differ from the frozen identity"
            )
    else:
        perturbation_counts = {}

    events = _jsonl(formal / "formal_condition_lifecycle.jsonl")
    starts = [row for row in events if row.get("event") == "CONDITION_START"]
    terminals = [row for row in events if row.get("event") == "CONDITION_TERMINAL"]
    if len(starts) != expected_condition_count or len(
        terminals
    ) != expected_condition_count:
        raise ValueError("condition lifecycle denominator is invalid")
    if [row.get("condition_id") for row in starts] != condition_ids or [
        row.get("condition_id") for row in terminals
    ] != condition_ids or any(row.get("status") != "passed" for row in terminals):
        raise ValueError("condition lifecycle order/status drift")

    with (formal / "online_task_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        score_rows = list(csv.DictReader(handle))
    identities = {
        (str(row.get("condition_id") or ""), str(row.get("task_id") or ""))
        for row in score_rows
    }
    expected_identities = {
        (condition_id, task_id)
        for condition_id in condition_ids
        for task_id in task_ids
    }
    if len(score_rows) != expected_episode_count or identities != expected_identities:
        raise ValueError("task score denominator differs from the frozen identity")

    identity = _json(formal / "formal_online_identity.json")
    if not _canonical_payload_sha256(identity, "manifest_sha256"):
        raise ValueError("formal online identity self-hash failed")
    if identity.get("preregistration_sha256") != registration["sha256"]:
        raise ValueError("formal online identity preregistration hash drift")
    if [row.get("condition_id") for row in identity.get("runs") or []] != condition_ids:
        raise ValueError("formal online identity run order drift")
    if identity.get("expected_task_ids") != task_ids:
        raise ValueError("formal online identity task cohort drift")
    provenance = _json(formal / "online_matrix_provenance_manifest.json")
    if not _canonical_payload_sha256(provenance, "manifest_payload_sha256"):
        raise ValueError("provenance self-hash failed")
    scene_template_integrity = _validate_scene_template_provenance(
        identity=identity,
        provenance=provenance,
        expected_task_ids=task_ids,
    )

    internal_scan = _json(formal / "active_credential_scan.json")
    if internal_scan.get("status") != "passed" or int(
        internal_scan.get("leak_path_count") or 0
    ) != 0 or internal_scan.get("leak_paths"):
        raise ValueError("runner credential scan failed")
    credentials = list(
        _active_credentials()
        if supplied_credentials is None
        else supplied_credentials
    )
    credentials = sorted({bytes(value) for value in credentials if len(value) >= 12})
    leak_count = sum(_contains_any(outer / row["path"], credentials) for row in before)
    if leak_count:
        raise ValueError("terminal active credential scan failed")
    after = _inventory(outer)
    if after != before:
        raise ValueError("formal root changed during terminal audit")

    tree_digest = hashlib.sha256(
        json.dumps(before, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    audit = {
        "schema_version": "isbench-formal-terminal-postflight-v1",
        "status": "passed",
        "artifact_class": "formal_candidate_pending_blinded_postflight",
        "task_count": 161,
        "condition_count": expected_condition_count,
        "task_episode_denominator": expected_episode_count,
        "condition_count_by_arm": arm_counts,
        "condition_count_by_wm_return_perturbation": perturbation_counts,
        "failed_condition_count": 0,
        "failed_validation_condition_count": 0,
        "skipped_condition_count": 0,
        "matrix_outputs_validated": True,
        "provenance_validated": True,
        "identity_validated": True,
        "scene_template_integrity": scene_template_integrity,
        "lifecycle_validated": True,
        "invocation_chain": invocation_chain,
        "terminal_active_credential_scan": {
            "status": "passed",
            "scanned_file_count": len(before),
            "active_credential_count": len(credentials),
            "leak_path_count": 0,
        },
        "formal_root_tree_sha256": tree_digest,
    }
    if hardware_contract is not None:
        hardware_identity = isbench_omnigibson_hardware_identity(hardware_contract)
        audit.update(
            {
                "hardware_profile_id": hardware_identity["profile_id"],
                "vendor_supported": hardware_identity["hardware_claims"][
                    "vendor_supported"
                ],
                "rt_cores": hardware_identity["hardware_claims"]["rt_cores"],
                "external_benchmark_comparability": hardware_identity[
                    "hardware_claims"
                ]["external_benchmark_comparability"],
                "primary_claim_scope": hardware_identity["hardware_claims"][
                    "claim_scope"
                ],
                "primary_estimand": (
                    "within_hardware_four_arm_itt_single_seed_task_census"
                ),
            }
        )
    evidence = {
        "schema_version": "isbench-formal-terminal-evidence-v1",
        "formal_outer_root": str(outer),
        "formal_matrix_root": str(formal),
        "files": before,
        "formal_root_tree_sha256": tree_digest,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("audit.json", audit),
        ("evidence_manifest.json", evidence),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    seal = {
        "schema_version": "isbench-formal-promotion-seal-v1",
        "status": "passed",
        "audit_sha256": _sha256(output / "audit.json"),
        "evidence_manifest_sha256": _sha256(output / "evidence_manifest.json"),
    }
    (output / "promotion_seal.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-outer-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_terminal_postflight(
        formal_outer_root=args.formal_outer_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
