#!/usr/bin/env python3
"""Cold-start OmniGibson against a real IS-Bench scene without model calls.

This is an infrastructure diagnostic.  It deliberately does not execute an
agent plan or run any safety evaluator, so its artifacts cannot be mistaken
for benchmark observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "omnigibson-gpu-infrastructure-probe-v4"
EMPIRICAL_PROFILE_SCHEMA_VERSION = "omnigibson-empirical-hardware-profile-v1"
EMPIRICAL_RUN_SCHEMA_VERSION = "omnigibson-empirical-hardware-probe-run-v1"
EMPIRICAL_SUMMARY_SCHEMA_VERSION = (
    "omnigibson-empirical-hardware-qualification-v1"
)
EMPIRICAL_SUCCESS_STATUS = (
    "empirical_hardware_qualified_for_registered_workload_smoke"
)

_GENERIC_PROBE_TASK = "removing_lint_from_dryer"
_GENERIC_PROBE_SCENE = "Beechwood_0_garden"
_GENERIC_PROBE_SEED = 0
_GENERIC_PROBE_PHYSICS_STEPS = 60
_GENERIC_PROBE_INITIAL_VIEW_COUNT = 5
_GENERIC_PROBE_REQUIRED_REPETITIONS = 2
_GENERIC_PROBE_MAXIMUM_PEAK_MEMORY_FRACTION = 0.9
_GENERIC_INPUT_SHA256_KEYS = {
    "task_config",
    "camera_config",
    "scene_template",
    "scene_source_archive",
    "probe_script",
}
_GENERIC_STAGE_NAMES = (
    "imports",
    "environment_constructed",
    "initial_render",
    "physics_steps",
    "final_render",
)

_EMPIRICAL_PROFILE_REGISTRY = {
    "omnigibson_a10040_empirical_v1": {
        "slurm": {
            "node": "gpu16",
            "gres": "gpu:a100-pcie-40gb:1",
            "gpu_count": 1,
        },
        "observed_hardware_constraints": {
            "exact_nvidia_smi_name": "NVIDIA A100-PCIE-40GB",
            "minimum_memory_total_mib": 40000,
        },
        "hardware_claims": {
            "vendor_supported": False,
            "rt_cores": False,
            "external_benchmark_comparability": False,
            "claim_scope": "within_hardware_arm_contrasts_only",
        },
        "listed_gpu_cost_units_per_hour": 10,
        "probe_contract": {
            "launcher_source_path": (
                "scripts/slurm/omnigibson_a10040_empirical_probe.sbatch"
            ),
            "probe_python_source_path": "scripts/probe_omnigibson_gpu.py",
            "run_schema_version": EMPIRICAL_RUN_SCHEMA_VERSION,
            "summary_schema_version": EMPIRICAL_SUMMARY_SCHEMA_VERSION,
            "run_root_pattern": (
                "runs/omnigibson_gpu_probe/a100_pcie_40gb_empirical_v2/"
                "job-${SLURM_JOB_ID}"
            ),
            "gpu_label": "a100-pcie-40gb",
            "required_repetitions": 2,
        },
    },
    "omnigibson_rtx_a6000_empirical_v1": {
        "slurm": {
            "node": "gpu07",
            "gres": "gpu:nvidia_rtx_a6000:1",
            "gpu_count": 1,
        },
        "observed_hardware_constraints": {
            "exact_nvidia_smi_name": "NVIDIA RTX A6000",
            "minimum_memory_total_mib": 48000,
        },
        "hardware_claims": {
            "vendor_supported": True,
            "rt_cores": True,
            "external_benchmark_comparability": False,
            "claim_scope": "within_hardware_arm_contrasts_only",
        },
        "listed_gpu_cost_units_per_hour": 12,
        "probe_contract": {
            "launcher_source_path": (
                "scripts/slurm/omnigibson_a6000_empirical_probe.sbatch"
            ),
            "probe_python_source_path": "scripts/probe_omnigibson_gpu.py",
            "run_schema_version": EMPIRICAL_RUN_SCHEMA_VERSION,
            "summary_schema_version": EMPIRICAL_SUMMARY_SCHEMA_VERSION,
            "run_root_pattern": (
                "runs/omnigibson_gpu_probe/rtx_a6000_empirical_v1/"
                "job-${SLURM_JOB_ID}"
            ),
            "gpu_label": "nvidia-rtx-a6000",
            "required_repetitions": 2,
        },
    },
}
_EMPIRICAL_REGISTERED_WORKLOAD = {
    "task": "removing_lint_from_dryer",
    "scene": "Beechwood_0_garden",
    "initial_view_count": 5,
    "physics_steps": 60,
    "final_render_required": True,
    "isolated_cold_starts": 2,
    "maximum_peak_memory_fraction": 0.9,
}
_EMPIRICAL_QUALIFICATION = {
    "success_status": EMPIRICAL_SUCCESS_STATUS,
    "eligible_for_formal_experiment": False,
    "required_next_evidence": "registered_full_task_compatible_online_smoke",
    "prospective_only": True,
    "prior_artifact_reuse": False,
}
_PIXEL_STATISTIC_FIELDS = {
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    "nonzero_fraction",
    "mean_absolute_adjacent_difference",
    "p90_absolute_adjacent_difference",
    "adjacent_difference_over_40_fraction",
}
_GENERIC_RENDERER_IDENTITY_SCHEMA_VERSION = (
    "omnigibson-probe-vulkan-renderer-identity-v1"
)
_ACTIVE_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|SESSION[_-]?TOKEN|"
    r"HF[_-]?TOKEN|PRIVATE[_-]?KEY|SECRET|PASSWORD|CREDENTIAL)",
    re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)((?:authorization\s*:\s*bearer|(?:api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|secret[_-]?key|password)\s*[:=])\s*[\"']?)([^\s\"']+)"
)
_TOKEN_LITERAL_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _active_credential_values() -> tuple[str, ...]:
    """Return active secret values without exposing them to artifacts or output."""

    return tuple(
        sorted(
            {
                value
                for name, value in os.environ.items()
                if value and _ACTIVE_CREDENTIAL_ENV_NAME.search(name)
            },
            key=len,
            reverse=True,
        )
    )


def _redact_sensitive_text(value: Any) -> str:
    """Redact active credential values and common credential renderings."""

    text = str(value)
    for secret in _active_credential_values():
        text = text.replace(secret, "[REDACTED]")
    text = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", text)
    return _TOKEN_LITERAL_PATTERN.sub("[REDACTED]", text)


def _redacted_exception_payload(
    error: BaseException, traceback_text: str | None = None
) -> dict[str, str]:
    """Produce a bounded error record that cannot persist active secret values."""

    trace = traceback.format_exc(limit=20) if traceback_text is None else traceback_text
    return {
        "type": type(error).__name__,
        "message": _redact_sensitive_text(error)[:2000],
        "traceback": _redact_sensitive_text(trace)[:12000],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys are invalid: {observed}")
    return value


def _load_empirical_hardware_profile(path: Path) -> dict[str, Any]:
    """Load the prospective empirical hardware contract fail-closed.

    The profile is intentionally narrower than a vendor compatibility claim.
    It registers one node, one exact inventory name, and one diagnostic
    workload.  Historical probe artifacts are never accepted by this loader.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "profile_id",
            "artifact_class",
            "slurm",
            "observed_hardware_constraints",
            "hardware_claims",
            "registered_workload",
            "qualification",
            "cost",
            "probe_contract",
        },
        label="empirical hardware profile",
    )
    if payload["schema_version"] != EMPIRICAL_PROFILE_SCHEMA_VERSION:
        raise ValueError("empirical hardware profile schema is invalid")
    profile_spec = _EMPIRICAL_PROFILE_REGISTRY.get(payload["profile_id"])
    if profile_spec is None:
        raise ValueError("empirical hardware profile_id is invalid")
    if payload["artifact_class"] != (
        "prospective_infrastructure_qualification_contract_not_experiment_evidence"
    ):
        raise ValueError("empirical hardware profile artifact_class is invalid")

    slurm = _require_exact_keys(
        payload["slurm"], {"node", "gres", "gpu_count"}, label="slurm contract"
    )
    if slurm != profile_spec["slurm"]:
        raise ValueError("empirical hardware SLURM contract is invalid")

    constraints = _require_exact_keys(
        payload["observed_hardware_constraints"],
        {"exact_nvidia_smi_name", "minimum_memory_total_mib"},
        label="observed hardware constraints",
    )
    if constraints != profile_spec["observed_hardware_constraints"]:
        raise ValueError("empirical observed hardware constraints are invalid")

    claims = _require_exact_keys(
        payload["hardware_claims"], set(profile_spec["hardware_claims"]),
        label="hardware claims",
    )
    if claims != profile_spec["hardware_claims"]:
        raise ValueError("empirical hardware claims exceed the registered scope")

    workload = _require_exact_keys(
        payload["registered_workload"],
        {
            "task",
            "scene",
            "initial_view_count",
            "physics_steps",
            "final_render_required",
            "isolated_cold_starts",
            "maximum_peak_memory_fraction",
        },
        label="registered workload",
    )
    if workload != _EMPIRICAL_REGISTERED_WORKLOAD:
        raise ValueError("empirical registered workload is invalid")

    qualification = _require_exact_keys(
        payload["qualification"],
        {
            "success_status",
            "eligible_for_formal_experiment",
            "required_next_evidence",
            "prospective_only",
            "prior_artifact_reuse",
        },
        label="qualification semantics",
    )
    if qualification != _EMPIRICAL_QUALIFICATION:
        raise ValueError("empirical qualification semantics are invalid")

    cost = _require_exact_keys(
        payload["cost"],
        {
            "listed_gpu_cost_units_per_hour",
            "inventory_source",
            "inventory_observed_at_utc",
        },
        label="cost provenance",
    )
    from datetime import datetime

    try:
        observed_at = datetime.fromisoformat(
            str(cost["inventory_observed_at_utc"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        observed_at = None
    if (
        cost["listed_gpu_cost_units_per_hour"]
        != profile_spec["listed_gpu_cost_units_per_hour"]
        or cost["inventory_source"] != "scir-watch -s"
        or observed_at is None
        or observed_at.tzinfo is None
    ):
        raise ValueError("empirical cost provenance is invalid")

    contract = _require_exact_keys(
        payload["probe_contract"],
        {
            "launcher_source_path",
            "launcher_source_sha256",
            "probe_python_source_path",
            "probe_python_source_sha256",
            "run_schema_version",
            "summary_schema_version",
            "run_root_pattern",
            "gpu_label",
            "required_repetitions",
        },
        label="probe contract",
    )
    expected_contract = profile_spec["probe_contract"]
    for field, expected in expected_contract.items():
        if contract[field] != expected:
            raise ValueError(f"empirical probe contract field {field} is invalid")
    for field in ("launcher_source_sha256", "probe_python_source_sha256"):
        if not _is_sha256(contract[field]):
            raise ValueError(f"empirical probe contract field {field} is invalid")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _image_stats(value: Any) -> dict[str, float | int]:
    import numpy as np

    image = np.asarray(value)
    image_float = image.astype(np.float32, copy=False)
    horizontal_difference = np.abs(np.diff(image_float, axis=1))
    vertical_difference = np.abs(np.diff(image_float, axis=0))
    adjacent_difference = np.concatenate(
        (horizontal_difference.reshape(-1), vertical_difference.reshape(-1))
    )
    return {
        "minimum": int(image.min()),
        "maximum": int(image.max()),
        "mean": float(image.mean()),
        "standard_deviation": float(image.std()),
        "nonzero_fraction": float(np.count_nonzero(image) / image.size),
        "mean_absolute_adjacent_difference": float(adjacent_difference.mean()),
        "p90_absolute_adjacent_difference": float(
            np.percentile(adjacent_difference, 90)
        ),
        "adjacent_difference_over_40_fraction": float(
            np.count_nonzero(adjacent_difference > 40) / adjacent_difference.size
        ),
    }


def _array_sha256(value: Any) -> str:
    import numpy as np

    image = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("utf-8"))
    digest.update(json.dumps(list(image.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _image_evidence(
    value: Any, image_path: Path, *, index: int, artifact_root: Path
) -> dict[str, Any]:
    resolved_path = image_path.resolve()
    resolved_root = artifact_root.resolve()
    if not resolved_path.is_relative_to(resolved_root) or not resolved_path.is_file():
        raise RuntimeError(f"expected rendered image is missing: {image_path}")
    return {
        "index": index,
        "relative_path": str(resolved_path.relative_to(resolved_root)),
        "file_sha256": _sha256(resolved_path),
        "array_sha256": _array_sha256(value),
        "pixel_statistics": _image_stats(value),
    }


def _pixel_statistics_are_informative(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _PIXEL_STATISTIC_FIELDS:
        return False
    try:
        return (
            float(value["standard_deviation"]) >= 0.5
            and int(value["maximum"]) - int(value["minimum"]) >= 5
            and float(value["nonzero_fraction"]) >= 0.01
        )
    except (TypeError, ValueError):
        return False


def _recorded_image_evidence_is_valid(
    value: Any, *, artifact_root: Path, expected_index: int
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "index",
        "relative_path",
        "file_sha256",
        "array_sha256",
        "pixel_statistics",
    }:
        return False
    if value["index"] != expected_index:
        return False
    if not _is_sha256(value["file_sha256"]) or not _is_sha256(
        value["array_sha256"]
    ):
        return False
    relative_path = Path(str(value["relative_path"]))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    image_path = (artifact_root / relative_path).resolve()
    if not image_path.is_relative_to(artifact_root.resolve()) or not image_path.is_file():
        return False
    if _sha256(image_path) != value["file_sha256"]:
        return False
    return _pixel_statistics_are_informative(value["pixel_statistics"])


def _assert_informative_render(stats: list[dict[str, float | int]]) -> None:
    if not stats or not any(
        float(item["standard_deviation"]) >= 0.5
        and int(item["maximum"]) - int(item["minimum"]) >= 5
        and float(item["nonzero_fraction"]) >= 0.01
        for item in stats
    ):
        raise RuntimeError(f"renderer produced no informative RGB frame: {stats}")


def _parse_number(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _is_rtx_gpu_name(name: str | None) -> bool:
    """Fail-closed inventory check for the required NVIDIA RTX capability.

    This deliberately does not infer ray-tracing support from compute
    generation or memory size.  A successful lightweight render on an A100,
    A800, H100, or V100 is not sufficient evidence for the full IS-Bench
    actor-subprocess workload.
    """

    return bool(name and "RTX" in name.upper())


def _probe_process_vulkan_renderer_identity(
    expected_gpu_name: str | None,
) -> dict[str, Any]:
    """Record a conservative Vulkan identity from this probe process's environment.

    This is an environment diagnostic only.  It intentionally does not infer
    any vendor support or formal-workload readiness from Vulkan availability.
    """

    expected = expected_gpu_name.strip() if isinstance(expected_gpu_name, str) else ""
    try:
        completed = subprocess.run(
            ["vulkaninfo", "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        device_names = sorted(
            {
                match.strip()
                for match in re.findall(r"deviceName\s*=\s*([^\r\n]+)", output)
                if match.strip()
            }
        )
        software_renderer_detected = any(
            marker in output.lower() for marker in ("llvmpipe", "lavapipe")
        )
        passed = (
            completed.returncode == 0
            and bool(expected)
            and device_names == [expected]
            and not software_renderer_detected
        )
    except (OSError, subprocess.TimeoutExpired):
        device_names = []
        software_renderer_detected = False
        passed = False
    return {
        "schema_version": _GENERIC_RENDERER_IDENTITY_SCHEMA_VERSION,
        "checked_in_probe_process": True,
        "renderer_api": "vulkan",
        "expected_gpu_name": expected or None,
        "device_names": device_names,
        "software_renderer_detected": software_renderer_detected,
        "status": "passed" if passed else "failed",
    }


def _resolve_probe_scene_template(
    *,
    project_root: Path,
    task: str,
    scene: str,
) -> dict[str, Any]:
    """Resolve and hash the exact fixed template the probe workload will use."""

    root = Path(project_root).expanduser().resolve()
    if not isinstance(task, str) or not isinstance(scene, str):
        raise ValueError("probe task and scene ids must be strings")
    task_id = task
    scene_id = scene
    if (
        not task_id
        or task_id != task_id.strip()
        or task_id in {".", ".."}
        or Path(task_id).name != task_id
        or "\\" in task_id
    ):
        raise ValueError(f"invalid probe task id: {task!r}")
    if (
        not scene_id
        or scene_id != scene_id.strip()
        or scene_id in {".", ".."}
        or Path(scene_id).name != scene_id
        or "\\" in scene_id
    ):
        raise ValueError(f"invalid probe scene id: {scene!r}")
    isbench_root = root / "IS-Bench"
    data_root = isbench_root / "data"
    task_path = data_root / "tasks" / f"{task_id}.json"
    try:
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load probe task config {task_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(task_payload, dict):
        raise ValueError(f"probe task config is not a JSON object: {task_path}")
    task_info = task_payload.get("task_info")
    scene_info = task_payload.get("scene_info")
    if not isinstance(task_info, dict) or not isinstance(scene_info, dict):
        raise ValueError(f"probe task config is incomplete: {task_path}")
    if task_info.get("task_name") != task_id:
        raise ValueError(f"probe task identity is ambiguous: {task_path}")
    default_scene = scene_info.get("default_scene_model")
    scene_models = scene_info.get("scene_models")
    if (
        not isinstance(default_scene, str)
        or scene_id != default_scene
        or not isinstance(scene_models, list)
        or scene_models.count(default_scene) != 1
    ):
        raise ValueError(
            f"probe scene must resolve uniquely to the task default scene: "
            f"selected={scene_id!r} default={default_scene!r}"
        )
    if scene_info.get("online_object_sampling") is not False:
        raise ValueError(
            f"probe fixed scene template requires online_object_sampling=false: {task_path}"
        )
    activity_definition_id = task_info.get("activity_definition_id")
    activity_instance_id = task_info.get("activity_instance_id")
    for field, value in (
        ("activity_definition_id", activity_definition_id),
        ("activity_instance_id", activity_instance_id),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"probe {field} must be a non-negative integer")
    expected_name = (
        f"{scene_id}_task_{task_id}_{activity_definition_id}_"
        f"{activity_instance_id}_template.json"
    )
    template_dir = data_root / "scenes" / scene_id / "json"
    matches = (
        sorted(
            path
            for path in template_dir.iterdir()
            if path.name.casefold() == expected_name.casefold()
        )
        if template_dir.is_dir()
        else []
    )
    if not matches:
        raise ValueError(
            f"probe scene template is missing for task={task_id!r} "
            f"scene={scene_id!r}: {template_dir / expected_name}"
        )
    if len(matches) != 1 or matches[0].name != expected_name:
        raise ValueError(
            f"probe scene template is ambiguous for task={task_id!r} "
            f"scene={scene_id!r}: {[path.name for path in matches]}"
        )
    template = matches[0]
    if (
        template.is_symlink()
        or not template.is_file()
        or template.resolve() != template
    ):
        raise ValueError(f"probe scene template is not a regular file: {template}")
    try:
        parsed_template = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"probe scene template is invalid JSON: {template}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(parsed_template, dict):
        raise ValueError(f"probe scene template is not a JSON object: {template}")
    archive = data_root / "scenes.tar.gz"
    if archive.is_symlink() or not archive.is_file() or archive.resolve() != archive:
        raise ValueError(
            f"probe scene source archive is missing or not a regular file: {archive}"
        )
    archive_member = str(template.relative_to(data_root))
    member_hashes: list[str] = []
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                if member.name.removeprefix("./") != archive_member:
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"probe scene source archive member is not a regular file: "
                        f"{archive_member}"
                    )
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"cannot read probe scene source archive member: {archive_member}"
                    )
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                member_hashes.append(digest.hexdigest())
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(
            f"cannot validate probe scene source archive {archive}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not member_hashes:
        raise ValueError(
            f"probe scene source archive member is missing: {archive_member}"
        )
    if len(member_hashes) != 1:
        raise ValueError(
            f"probe scene source archive member is ambiguous: {archive_member}"
        )
    template_sha256 = _sha256(template)
    if member_hashes[0] != template_sha256:
        raise ValueError(
            f"probe scene template differs from source archive member: {archive_member}"
        )
    return {
        "task": task_id,
        "scene": scene_id,
        "activity_definition_id": activity_definition_id,
        "activity_instance_id": activity_instance_id,
        "relative_path": str(template.relative_to(root)),
        "source_archive_member": archive_member,
        "size_bytes": template.stat().st_size,
        "sha256": template_sha256,
        "source_archive_member_sha256": member_hashes[0],
        "source_archive_relative_path": str(archive.relative_to(root)),
        "source_archive_size_bytes": archive.stat().st_size,
        "source_archive_sha256": _sha256(archive),
    }


def _probe_scene_template_integrity(
    binding: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative = Path(str(binding.get("relative_path") or ""))
    archive_relative = Path(
        str(binding.get("source_archive_relative_path") or "")
    )
    valid_relatives = all(
        not path.is_absolute() and ".." not in path.parts
        for path in (relative, archive_relative)
    )
    lexical_template = root / relative
    lexical_archive = root / archive_relative
    template = lexical_template.resolve()
    archive = lexical_archive.resolve()
    try:
        template.relative_to(root)
        archive.relative_to(root)
    except ValueError:
        valid_relatives = False
    valid_relatives = (
        valid_relatives
        and lexical_template == template
        and lexical_archive == archive
        and not lexical_template.is_symlink()
        and not lexical_archive.is_symlink()
    )
    template_sha256 = _sha256(template) if valid_relatives and template.is_file() else None
    archive_sha256 = _sha256(archive) if valid_relatives and archive.is_file() else None
    return {
        "template_sha256": template_sha256,
        "source_archive_sha256": archive_sha256,
        "passed": (
            valid_relatives
            and template_sha256 == binding.get("sha256")
            and archive_sha256 == binding.get("source_archive_sha256")
        ),
    }


def _stage_by_name(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    stages = result.get("stages")
    if not isinstance(stages, list):
        return None
    matches = [stage for stage in stages if isinstance(stage, dict) and stage.get("name") == name]
    return matches[0] if len(matches) == 1 else None


def _empirical_image_evidence_passes(
    result: dict[str, Any], profile: dict[str, Any], *, result_path: Path
) -> bool:
    workload = profile["registered_workload"]
    environment = _stage_by_name(result, "environment_constructed")
    initial = _stage_by_name(result, "initial_render")
    physics = _stage_by_name(result, "physics_steps")
    final = _stage_by_name(result, "final_render")
    if not all((environment, initial, physics, final)):
        return False
    if (
        environment.get("resolved_task") != workload["task"]
        or environment.get("resolved_scene") != workload["scene"]
        or physics.get("step_count") != workload["physics_steps"]
        or initial.get("view_count") != workload["initial_view_count"]
    ):
        return False
    shapes = initial.get("view_shapes")
    evidence = initial.get("view_image_evidence")
    if (
        not isinstance(shapes, list)
        or len(shapes) != workload["initial_view_count"]
        or not all(
            isinstance(shape, list) and len(shape) == 3 and shape[-1] == 3
            for shape in shapes
        )
        or not isinstance(evidence, list)
        or len(evidence) != workload["initial_view_count"]
    ):
        return False
    artifact_root = result_path.parent
    if not all(
        _recorded_image_evidence_is_valid(
            item, artifact_root=artifact_root, expected_index=index
        )
        for index, item in enumerate(evidence)
    ):
        return False
    final_shape = final.get("view_shape")
    if (
        not isinstance(final_shape, list)
        or len(final_shape) != 3
        or final_shape[-1] != 3
    ):
        return False
    return _recorded_image_evidence_is_valid(
        final.get("image_evidence"), artifact_root=artifact_root, expected_index=0
    )


def _empirical_hardware_gate_passes(
    result: dict[str, Any], profile: dict[str, Any]
) -> bool:
    slurm_contract = profile["slurm"]
    constraints = profile["observed_hardware_constraints"]
    measurements = result.get("gpu_measurements")
    compatibility = result.get("hardware_compatibility")
    slurm = result.get("slurm")
    if not all(isinstance(item, dict) for item in (measurements, compatibility, slurm)):
        return False
    observed_memory = measurements.get("memory_total_mib")
    try:
        memory_pass = float(observed_memory) >= float(
            constraints["minimum_memory_total_mib"]
        )
    except (TypeError, ValueError):
        memory_pass = False
    return (
        slurm.get("node") == slurm_contract["node"]
        and measurements.get("gpu_name") == constraints["exact_nvidia_smi_name"]
        and memory_pass
        and measurements.get("sampler_errors") == []
        and compatibility
        == {
            "required_node": slurm_contract["node"],
            "observed_node": slurm.get("node"),
            "node_gate_pass": slurm.get("node") == slurm_contract["node"],
            "required_exact_gpu_name": constraints["exact_nvidia_smi_name"],
            "observed_gpu_name": measurements.get("gpu_name"),
            "exact_name_gate_pass": (
                measurements.get("gpu_name")
                == constraints["exact_nvidia_smi_name"]
            ),
            "required_minimum_memory_total_mib": constraints[
                "minimum_memory_total_mib"
            ],
            "observed_memory_total_mib": observed_memory,
            "minimum_memory_gate_pass": memory_pass,
        }
    )


def _recorded_scene_template_binding_is_valid(result: dict[str, Any]) -> bool:
    binding = result.get("scene_template_binding")
    integrity = result.get("scene_template_integrity")
    input_sha256 = result.get("input_sha256")
    if not all(isinstance(value, dict) for value in (binding, integrity, input_sha256)):
        return False
    template_sha256 = binding.get("sha256")
    archive_sha256 = binding.get("source_archive_sha256")
    task = result.get("task")
    scene = result.get("scene")
    definition_id = binding.get("activity_definition_id")
    instance_id = binding.get("activity_instance_id")
    if (
        not isinstance(task, str)
        or not isinstance(scene, str)
        or not task
        or not scene
        or task != task.strip()
        or scene != scene.strip()
        or task in {".", ".."}
        or scene in {".", ".."}
        or Path(task).name != task
        or Path(scene).name != scene
        or "\\" in task
        or "\\" in scene
        or isinstance(definition_id, bool)
        or not isinstance(definition_id, int)
        or definition_id < 0
        or isinstance(instance_id, bool)
        or not isinstance(instance_id, int)
        or instance_id < 0
    ):
        return False
    expected_name = (
        f"{scene}_task_{task}_{definition_id}_{instance_id}_template.json"
    )
    relative = Path(str(binding.get("relative_path") or ""))
    member_relative = Path(str(binding.get("source_archive_member") or ""))
    archive_relative = Path(
        str(binding.get("source_archive_relative_path") or "")
    )
    return (
        binding.get("task") == task
        and binding.get("scene") == scene
        and _is_sha256(template_sha256)
        and _is_sha256(archive_sha256)
        and binding.get("source_archive_member_sha256") == template_sha256
        and relative
        == Path("IS-Bench/data/scenes") / scene / "json" / expected_name
        and member_relative == Path("scenes") / scene / "json" / expected_name
        and archive_relative == Path("IS-Bench/data/scenes.tar.gz")
        and input_sha256.get("scene_template") == template_sha256
        and input_sha256.get("scene_source_archive") == archive_sha256
        and integrity.get("template_sha256") == template_sha256
        and integrity.get("source_archive_sha256") == archive_sha256
        and integrity.get("passed") is True
    )


def _recorded_scene_template_binding_identity(
    result: dict[str, Any],
) -> tuple[Any, ...] | None:
    if not _recorded_scene_template_binding_is_valid(result):
        return None
    binding = result["scene_template_binding"]
    return (
        binding["task"],
        binding["scene"],
        binding["activity_definition_id"],
        binding["activity_instance_id"],
        binding["relative_path"],
        binding["sha256"],
        binding["source_archive_member"],
        binding["source_archive_member_sha256"],
        binding["source_archive_relative_path"],
        binding["source_archive_sha256"],
    )


def _empirical_result_failure_reason(
    result: dict[str, Any],
    profile: dict[str, Any],
    profile_sha256: str,
    *,
    result_path: Path,
) -> str | None:
    if result.get("schema_version") != EMPIRICAL_RUN_SCHEMA_VERSION:
        return "run_schema_mismatch"
    if result.get("artifact_class") != (
        "infrastructure_diagnostic_not_experiment_evidence"
    ):
        return "run_artifact_class_mismatch"
    input_sha256 = result.get("input_sha256") or {}
    if (
        result.get("hardware_profile_id") != profile["profile_id"]
        or result.get("hardware_profile_sha256") != profile_sha256
        or input_sha256.get("hardware_profile") != profile_sha256
    ):
        return "hardware_profile_binding_mismatch"
    if input_sha256.get("probe_script") != profile["probe_contract"][
        "probe_python_source_sha256"
    ]:
        return "probe_source_binding_mismatch"
    if (
        result.get("prospective_evidence") is not True
        or result.get("prior_artifact_reuse") is not False
    ):
        return "nonprospective_evidence"
    if result.get("hardware_claims") != profile["hardware_claims"]:
        return "hardware_claims_mismatch"
    if result.get("observed_hardware_constraints") != profile[
        "observed_hardware_constraints"
    ]:
        return "hardware_constraints_mismatch"
    if result.get("registered_workload") != profile["registered_workload"]:
        return "registered_workload_mismatch"
    if result.get("status") != "success":
        error_type = str(result.get("error", {}).get("type", ""))
        if error_type == "ProcessTimeout" or result.get("process_exit_code") in {
            124,
            137,
            143,
        }:
            return "process_timeout"
        return "probe_process_failed"
    if not _recorded_scene_template_binding_is_valid(result):
        return "scene_template_provenance_gate_failed"

    measurements = result.get("gpu_measurements") or {}
    slurm = result.get("slurm") or {}
    constraints = profile["observed_hardware_constraints"]
    if slurm.get("node") != profile["slurm"]["node"]:
        return "node_gate_failed"
    if measurements.get("gpu_name") != constraints["exact_nvidia_smi_name"]:
        return "exact_gpu_name_gate_failed"
    try:
        if float(measurements.get("memory_total_mib")) < float(
            constraints["minimum_memory_total_mib"]
        ):
            return "minimum_gpu_memory_gate_failed"
    except (TypeError, ValueError):
        return "minimum_gpu_memory_gate_failed"
    if measurements.get("sampler_errors") != []:
        return "gpu_sampler_failed"
    if not _empirical_hardware_gate_passes(result, profile):
        return "hardware_contract_gate_failed"
    try:
        if float(measurements.get("peak_memory_fraction")) > float(
            profile["registered_workload"]["maximum_peak_memory_fraction"]
        ):
            return "peak_gpu_memory_gate_failed"
    except (TypeError, ValueError):
        return "peak_gpu_memory_gate_failed"
    if not _empirical_image_evidence_passes(
        result, profile, result_path=result_path
    ):
        return "registered_image_or_physics_evidence_failed"
    if result.get("single_run_gate_pass") is not True:
        return "single_run_gate_not_passed"
    return None


class NvidiaSmiSampler:
    """Sample the single GPU visible inside the SLURM device cgroup."""

    def __init__(self, gpu_index: str, interval_seconds: float = 1.0):
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "--id",
            self.gpu_index,
            "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            if message and message not in self.errors:
                self.errors.append(_redact_sensitive_text(message)[:500])
            return
        line = result.stdout.strip().splitlines()[0]
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            self.errors.append(
                _redact_sensitive_text(f"unexpected nvidia-smi row: {line}")[:500]
            )
            return
        self.samples.append(
            {
                "monotonic_seconds": time.monotonic(),
                "uuid": fields[0],
                "name": fields[1],
                "memory_total_mib": _parse_number(fields[2]),
                "memory_used_mib": _parse_number(fields[3]),
                "utilization_percent": _parse_number(fields[4]),
            }
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as exc:  # diagnostic must retain sampler failures
                message = _redact_sensitive_text(f"{type(exc).__name__}: {exc}")
                if message not in self.errors:
                    self.errors.append(message[:500])
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, 2 * self.interval_seconds))
        self._sample()
        totals = [
            item["memory_total_mib"]
            for item in self.samples
            if item["memory_total_mib"] is not None
        ]
        used = [
            item["memory_used_mib"]
            for item in self.samples
            if item["memory_used_mib"] is not None
        ]
        utilization = [
            item["utilization_percent"]
            for item in self.samples
            if item["utilization_percent"] is not None
        ]
        total = max(totals) if totals else None
        peak = max(used) if used else None
        baseline = used[0] if used else None
        peak_memory_delta = (
            peak - baseline if peak is not None and baseline is not None else None
        )
        peak_utilization = max(utilization) if utilization else None
        return {
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.samples[-1]["uuid"] if self.samples else None,
            "gpu_name": self.samples[-1]["name"] if self.samples else None,
            "memory_total_mib": total,
            "baseline_memory_used_mib": baseline,
            "peak_memory_used_mib": peak,
            "peak_memory_delta_mib": peak_memory_delta,
            "peak_memory_fraction": (
                peak / total if peak is not None and total not in (None, 0) else None
            ),
            "peak_utilization_percent": peak_utilization,
            "positive_peak_memory_delta_gate_pass": bool(
                peak_memory_delta is not None and peak_memory_delta > 0
            ),
            "positive_peak_utilization_gate_pass": bool(
                peak_utilization is not None and peak_utilization > 0
            ),
            "sample_count": len(self.samples),
            "sampler_errors": self.errors,
        }


def _nvidia_smi_gpu_index() -> str:
    explicit = os.environ.get("PROBE_NVIDIA_SMI_GPU_ID", "").strip()
    if explicit:
        return explicit
    # SCIR's device cgroup exposes only the allocated device and nvidia-smi
    # enumerates it as ordinal 0 even when SLURM_JOB_GPUS records a different
    # physical host ordinal.
    return "0"


def run_probe(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    project_root = Path(args.project_root).resolve()
    task_config = project_root / "IS-Bench" / "data" / "tasks" / f"{args.task}.json"
    camera_config = project_root / "IS-Bench" / "data" / "cameras" / "camera.json"
    scene_template_binding = _resolve_probe_scene_template(
        project_root=project_root,
        task=args.task,
        scene=args.scene,
    )
    empirical_profile_path = (
        Path(args.hardware_profile).resolve()
        if getattr(args, "hardware_profile", None)
        else None
    )
    empirical_profile = (
        _load_empirical_hardware_profile(empirical_profile_path)
        if empirical_profile_path is not None
        else None
    )
    empirical_profile_sha256 = (
        _sha256(empirical_profile_path) if empirical_profile_path is not None else None
    )
    if empirical_profile is not None:
        workload = empirical_profile["registered_workload"]
        observed_arguments = {
            "task": args.task,
            "scene": args.scene,
            "physics_steps": args.physics_steps,
            "maximum_peak_memory_fraction": args.max_peak_memory_fraction,
        }
        expected_arguments = {
            "task": workload["task"],
            "scene": workload["scene"],
            "physics_steps": workload["physics_steps"],
            "maximum_peak_memory_fraction": workload[
                "maximum_peak_memory_fraction"
            ],
        }
        if observed_arguments != expected_arguments or args.require_rtx:
            raise ValueError("probe arguments differ from empirical hardware profile")
    renderer_identity = _probe_process_vulkan_renderer_identity(
        os.environ.get("PROBE_EXPECTED_GPU_NAME") if args.require_rtx else None
    )
    started = time.monotonic()
    payload: dict[str, Any] = {
        "schema_version": (
            EMPIRICAL_RUN_SCHEMA_VERSION if empirical_profile is not None else SCHEMA_VERSION
        ),
        "artifact_class": "infrastructure_diagnostic_not_experiment_evidence",
        "status": "running",
        "qualification_thresholds": {
            "required_cold_start_successes": 2,
            "required_physics_steps_per_start": args.physics_steps,
            "maximum_peak_memory_fraction": args.max_peak_memory_fraction,
            "require_rtx_gpu": args.require_rtx,
            "required_hardware_capability": (
                "nvidia_rtx_hardware_ray_tracing" if args.require_rtx else None
            ),
        },
        "started_at_utc": _utc_now(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omnigibson_gpu_id": os.environ.get("OMNIGIBSON_GPU_ID"),
        },
        "requested_gpu_label": os.environ.get("PROBE_GPU_LABEL"),
        "listed_gpu_cost_units_per_hour": _parse_number(
            os.environ.get("PROBE_GPU_COST", "")
        ),
        "task": args.task,
        "scene": args.scene,
        "physics_steps": args.physics_steps,
        "input_sha256": {
            "task_config": _sha256(task_config),
            "camera_config": _sha256(camera_config),
            "scene_template": scene_template_binding["sha256"],
            "scene_source_archive": scene_template_binding[
                "source_archive_sha256"
            ],
            "probe_script": _sha256(Path(__file__).resolve()),
        },
        "scene_template_binding": scene_template_binding,
        "runtime_versions": {
            name: _package_version(name)
            for name in ("omnigibson", "isaacsim", "torch", "numpy", "psutil")
        },
        "renderer_identity": renderer_identity,
        "stages": [],
    }
    if empirical_profile is not None:
        payload.update(
            {
                "hardware_profile_id": empirical_profile["profile_id"],
                "hardware_profile_sha256": empirical_profile_sha256,
                "prospective_evidence": True,
                "prior_artifact_reuse": False,
                "registered_workload": empirical_profile["registered_workload"],
                "hardware_claims": empirical_profile["hardware_claims"],
                "observed_hardware_constraints": empirical_profile[
                    "observed_hardware_constraints"
                ],
            }
        )
        payload["listed_gpu_cost_units_per_hour"] = float(
            empirical_profile["cost"]["listed_gpu_cost_units_per_hour"]
        )
        payload["input_sha256"]["hardware_profile"] = empirical_profile_sha256
    else:
        payload["seed"] = args.seed
    sampler = NvidiaSmiSampler(_nvidia_smi_gpu_index())
    sampler.start()
    exit_code = 1
    current_stage = "imports"
    try:
        stage_started = time.monotonic()
        from og_ego_prim.utils.monkey_patch import add_monkey_patch

        add_monkey_patch()
        import numpy as np
        import omnigibson as og
        import torch
        from omnigibson.macros import gm
        from og_ego_prim.benchmark import build_benchmark

        gm.USE_GPU_DYNAMICS = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        payload["stages"].append(
            {"name": "imports", "elapsed_seconds": time.monotonic() - stage_started}
        )

        current_stage = "environment_constructed"
        stage_started = time.monotonic()
        benchmark = build_benchmark(
            task=args.task,
            scene=args.scene,
            ego_view=True,
            draw_bbox_2d=False,
            use_initial_setup=False,
            use_self_caption=False,
            online_object_sampling=False,
            offline_mode=False,
            debug=False,
            eval_process_safety=False,
            eval_termination_safety=False,
            eval_awareness=False,
            eval_execution=False,
        )
        payload["stages"].append(
            {
                "name": "environment_constructed",
                "elapsed_seconds": time.monotonic() - stage_started,
                "resolved_task": benchmark.task_name,
                "resolved_scene": benchmark.scene_name,
            }
        )

        current_stage = "initial_render"
        stage_started = time.monotonic()
        initial_views_dir = output.parent / "initial_views"
        initial_views = benchmark.get_surrounding_viewer_obs(
            save_img=str(initial_views_dir)
        )
        initial_shapes = [list(np.asarray(view).shape) for view in (initial_views or [])]
        if not initial_shapes or any(len(shape) != 3 or shape[-1] != 3 for shape in initial_shapes):
            raise RuntimeError(f"invalid initial viewer observations: {initial_shapes}")
        initial_stats = [_image_stats(view) for view in (initial_views or [])]
        _assert_informative_render(initial_stats)
        initial_stage: dict[str, Any] = {
            "name": "initial_render",
            "elapsed_seconds": time.monotonic() - stage_started,
            "view_count": len(initial_shapes),
            "view_shapes": initial_shapes,
        }
        if empirical_profile is not None:
            expected_view_count = empirical_profile["registered_workload"][
                "initial_view_count"
            ]
            initial_image_paths = sorted(initial_views_dir.glob("*.png"))
            if (
                len(initial_shapes) != expected_view_count
                or len(initial_image_paths) != expected_view_count
            ):
                raise RuntimeError(
                    "registered initial view count mismatch: "
                    f"arrays={len(initial_shapes)} files={len(initial_image_paths)} "
                    f"expected={expected_view_count}"
                )
            initial_stage["view_image_evidence"] = [
                _image_evidence(
                    view,
                    image_path,
                    index=index,
                    artifact_root=output.parent,
                )
                for index, (view, image_path) in enumerate(
                    zip(initial_views, initial_image_paths, strict=True)
                )
            ]
        else:
            initial_stage["view_pixel_statistics"] = initial_stats
        payload["stages"].append(initial_stage)

        current_stage = "physics_steps"
        stage_started = time.monotonic()
        benchmark.executor._simulator_loop(args.physics_steps)
        payload["stages"].append(
            {
                "name": "physics_steps",
                "elapsed_seconds": time.monotonic() - stage_started,
                "step_count": args.physics_steps,
            }
        )

        current_stage = "final_render"
        stage_started = time.monotonic()
        final_view = benchmark.get_viewer_obs(
            save_img=str(output.parent / "final_view.png")
        )
        final_shape = list(np.asarray(final_view).shape)
        if len(final_shape) != 3 or final_shape[-1] != 3:
            raise RuntimeError(f"invalid final viewer observation: {final_shape}")
        final_stats = _image_stats(final_view)
        _assert_informative_render([final_stats])
        final_stage: dict[str, Any] = {
            "name": "final_render",
            "elapsed_seconds": time.monotonic() - stage_started,
            "view_shape": final_shape,
        }
        if empirical_profile is not None:
            final_stage["image_evidence"] = _image_evidence(
                final_view,
                output.parent / "final_view.png",
                index=0,
                artifact_root=output.parent,
            )
        else:
            final_stage["view_pixel_statistics"] = final_stats
        payload["stages"].append(final_stage)
        payload["status"] = "success"
        exit_code = 0
    except BaseException as exc:
        payload["status"] = "failed"
        payload["failure_stage"] = current_stage
        payload["error"] = _redacted_exception_payload(exc)
    finally:
        payload["gpu_measurements"] = sampler.stop()
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["finished_at_utc"] = _utc_now()
        payload["scene_template_integrity"] = _probe_scene_template_integrity(
            scene_template_binding,
            project_root=project_root,
        )
        peak_fraction = payload["gpu_measurements"].get("peak_memory_fraction")
        gpu_name = payload["gpu_measurements"].get("gpu_name")
        if empirical_profile is not None:
            slurm_contract = empirical_profile["slurm"]
            constraints = empirical_profile["observed_hardware_constraints"]
            observed_node = payload["slurm"].get("node")
            observed_memory = payload["gpu_measurements"].get("memory_total_mib")
            try:
                minimum_memory_gate_pass = float(observed_memory) >= float(
                    constraints["minimum_memory_total_mib"]
                )
            except (TypeError, ValueError):
                minimum_memory_gate_pass = False
            payload["hardware_compatibility"] = {
                "required_node": slurm_contract["node"],
                "observed_node": observed_node,
                "node_gate_pass": observed_node == slurm_contract["node"],
                "required_exact_gpu_name": constraints["exact_nvidia_smi_name"],
                "observed_gpu_name": gpu_name,
                "exact_name_gate_pass": (
                    gpu_name == constraints["exact_nvidia_smi_name"]
                ),
                "required_minimum_memory_total_mib": constraints[
                    "minimum_memory_total_mib"
                ],
                "observed_memory_total_mib": observed_memory,
                "minimum_memory_gate_pass": minimum_memory_gate_pass,
            }
            payload["single_run_gate_pass"] = (
                payload["status"] == "success"
                and peak_fraction is not None
                and peak_fraction
                <= empirical_profile["registered_workload"][
                    "maximum_peak_memory_fraction"
                ]
                and _empirical_hardware_gate_passes(payload, empirical_profile)
                and _empirical_image_evidence_passes(
                    payload, empirical_profile, result_path=output
                )
                and payload["scene_template_integrity"]["passed"] is True
            )
        else:
            payload["hardware_compatibility"] = {
                "require_rtx_gpu": args.require_rtx,
                "required_hardware_capability": (
                    "nvidia_rtx_hardware_ray_tracing" if args.require_rtx else None
                ),
                "observed_gpu_name": gpu_name,
                "rt_capable_name_gate_pass": _is_rtx_gpu_name(gpu_name),
                # Kept for backward-compatible readers of v1 diagnostic artifacts.
                "rtx_name_gate_pass": _is_rtx_gpu_name(gpu_name),
            }
            payload["single_run_gate_pass"] = (
                payload["status"] == "success"
                and peak_fraction is not None
                and peak_fraction <= args.max_peak_memory_fraction
                and (not args.require_rtx or _is_rtx_gpu_name(gpu_name))
                and _generic_gpu_measurement_failure_reason(payload) is None
                and _generic_renderer_identity_failure_reason(payload) is None
                and payload["scene_template_integrity"]["passed"] is True
            )
        _atomic_json(output, payload)
        sys.stdout.flush()
        sys.stderr.flush()

    # Isaac Sim native-plugin teardown can segfault after valid artifacts are
    # complete.  The production strict launcher uses the same process-lifetime
    # boundary, so terminate cleanly without treating teardown as GPU evidence.
    os._exit(exit_code)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _exact_int(value: Any, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _generic_pixel_statistics_are_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _PIXEL_STATISTIC_FIELDS
        and all(_finite_number(item) for item in value.values())
    )


def _generic_rgb_shape_is_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in value
        )
        and value[-1] == 3
    )


def _generic_stage_evidence_passes(result: dict[str, Any]) -> bool:
    stages = result.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) != len(_GENERIC_STAGE_NAMES)
        or tuple(
            stage.get("name") if isinstance(stage, dict) else None
            for stage in stages
        )
        != _GENERIC_STAGE_NAMES
    ):
        return False
    imports, environment, initial, physics, final = stages
    if any(
        not _finite_number(stage.get("elapsed_seconds"))
        or float(stage["elapsed_seconds"]) < 0
        for stage in stages
    ):
        return False
    if set(imports) != {"name", "elapsed_seconds"}:
        return False
    if (
        set(environment)
        != {"name", "elapsed_seconds", "resolved_task", "resolved_scene"}
        or environment.get("resolved_task") != _GENERIC_PROBE_TASK
        or environment.get("resolved_scene") != _GENERIC_PROBE_SCENE
    ):
        return False
    initial_shapes = initial.get("view_shapes")
    initial_statistics = initial.get("view_pixel_statistics")
    if (
        set(initial)
        != {
            "name",
            "elapsed_seconds",
            "view_count",
            "view_shapes",
            "view_pixel_statistics",
        }
        or not _exact_int(
            initial.get("view_count"), _GENERIC_PROBE_INITIAL_VIEW_COUNT
        )
        or not isinstance(initial_shapes, list)
        or len(initial_shapes) != _GENERIC_PROBE_INITIAL_VIEW_COUNT
        or not all(_generic_rgb_shape_is_valid(shape) for shape in initial_shapes)
        or not isinstance(initial_statistics, list)
        or len(initial_statistics) != _GENERIC_PROBE_INITIAL_VIEW_COUNT
        or not all(
            _generic_pixel_statistics_are_valid(statistics)
            for statistics in initial_statistics
        )
        or not any(
            _pixel_statistics_are_informative(statistics)
            for statistics in initial_statistics
        )
    ):
        return False
    if (
        set(physics) != {"name", "elapsed_seconds", "step_count"}
        or not _exact_int(
            physics.get("step_count"), _GENERIC_PROBE_PHYSICS_STEPS
        )
    ):
        return False
    return (
        set(final)
        == {"name", "elapsed_seconds", "view_shape", "view_pixel_statistics"}
        and _generic_rgb_shape_is_valid(final.get("view_shape"))
        and _generic_pixel_statistics_are_valid(final.get("view_pixel_statistics"))
        and _pixel_statistics_are_informative(final["view_pixel_statistics"])
    )


def _generic_input_sha256_is_valid(result: dict[str, Any]) -> bool:
    input_sha256 = result.get("input_sha256")
    return (
        isinstance(input_sha256, dict)
        and set(input_sha256) == _GENERIC_INPUT_SHA256_KEYS
        and all(_is_sha256(input_sha256[key]) for key in _GENERIC_INPUT_SHA256_KEYS)
    )


def _generic_rtx_hardware_gate_passes(result: dict[str, Any]) -> bool:
    measurements = result.get("gpu_measurements")
    compatibility = result.get("hardware_compatibility")
    if not isinstance(measurements, dict) or not isinstance(compatibility, dict):
        return False
    gpu_name = measurements.get("gpu_name")
    return (
        _is_rtx_gpu_name(gpu_name)
        and compatibility
        == {
            "require_rtx_gpu": True,
            "required_hardware_capability": "nvidia_rtx_hardware_ray_tracing",
            "observed_gpu_name": gpu_name,
            "rt_capable_name_gate_pass": True,
            "rtx_name_gate_pass": True,
        }
    )


def _generic_renderer_identity_failure_reason(result: dict[str, Any]) -> str | None:
    identity = result.get("renderer_identity")
    measurements = result.get("gpu_measurements")
    if not isinstance(identity, dict) or not isinstance(measurements, dict):
        return "renderer_identity_gate_failed"
    gpu_name = measurements.get("gpu_name")
    expected = {
        "schema_version": _GENERIC_RENDERER_IDENTITY_SCHEMA_VERSION,
        "checked_in_probe_process": True,
        "renderer_api": "vulkan",
        "expected_gpu_name": gpu_name,
        "device_names": [gpu_name],
        "software_renderer_detected": False,
        "status": "passed",
    }
    return None if identity == expected and _is_rtx_gpu_name(gpu_name) else "renderer_identity_gate_failed"


def _generic_gpu_activity_evidence_passes(result: dict[str, Any]) -> bool:
    measurements = result.get("gpu_measurements")
    if not isinstance(measurements, dict):
        return False
    delta = measurements.get("peak_memory_delta_mib")
    utilization = measurements.get("peak_utilization_percent")
    return (
        _finite_number(delta)
        and float(delta) > 0
        and _finite_number(utilization)
        and float(utilization) > 0
        and measurements.get("positive_peak_memory_delta_gate_pass") is True
        and measurements.get("positive_peak_utilization_gate_pass") is True
    )


def _generic_gpu_measurement_failure_reason(result: dict[str, Any]) -> str | None:
    measurements = result.get("gpu_measurements")
    if not isinstance(measurements, dict):
        return "gpu_measurements_not_finite"
    if measurements.get("sampler_errors") != []:
        return "gpu_sampler_failed"
    expected_keys = {
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "memory_total_mib",
        "baseline_memory_used_mib",
        "peak_memory_used_mib",
        "peak_memory_delta_mib",
        "peak_memory_fraction",
        "peak_utilization_percent",
        "positive_peak_memory_delta_gate_pass",
        "positive_peak_utilization_gate_pass",
        "sample_count",
        "sampler_errors",
    }
    numeric_keys = {
        "memory_total_mib",
        "baseline_memory_used_mib",
        "peak_memory_used_mib",
        "peak_memory_delta_mib",
        "peak_memory_fraction",
        "peak_utilization_percent",
    }
    if (
        set(measurements) != expected_keys
        or not all(_finite_number(measurements.get(key)) for key in numeric_keys)
        or isinstance(measurements.get("sample_count"), bool)
        or not isinstance(measurements.get("sample_count"), int)
        or measurements["sample_count"] <= 0
    ):
        return "gpu_measurements_not_finite"
    total = float(measurements["memory_total_mib"])
    baseline = float(measurements["baseline_memory_used_mib"])
    peak = float(measurements["peak_memory_used_mib"])
    delta = float(measurements["peak_memory_delta_mib"])
    fraction = float(measurements["peak_memory_fraction"])
    utilization = float(measurements["peak_utilization_percent"])
    if (
        total <= 0
        or not 0 <= baseline <= total
        or not 0 <= peak <= total
        or not math.isclose(delta, peak - baseline, rel_tol=1e-12, abs_tol=1e-12)
        or not 0 <= fraction <= 1
        or not math.isclose(fraction, peak / total, rel_tol=1e-12, abs_tol=1e-12)
        or not 0 <= utilization <= 100
    ):
        return "gpu_measurements_not_finite"
    if fraction > _GENERIC_PROBE_MAXIMUM_PEAK_MEMORY_FRACTION:
        return "peak_gpu_memory_gate_failed"
    if not _generic_gpu_activity_evidence_passes(result):
        return "gpu_activity_evidence_gate_failed"
    return None


def _probe_failure_reason(result: dict[str, Any], *, require_rtx: bool) -> str | None:
    """Return one sanitized, stable reason for an unsuccessful repetition."""

    if result.get("schema_version") != SCHEMA_VERSION:
        return "run_schema_mismatch"
    if result.get("artifact_class") != (
        "infrastructure_diagnostic_not_experiment_evidence"
    ):
        return "run_artifact_class_mismatch"
    error_type = str(result.get("error", {}).get("type", "")).strip()
    exit_code = result.get("process_exit_code")
    if error_type == "ProcessTimeout" or exit_code in {124, 137, 143}:
        return "process_timeout"
    if result.get("status") != "success":
        return "probe_process_failed"
    if (
        result.get("task") != _GENERIC_PROBE_TASK
        or result.get("scene") != _GENERIC_PROBE_SCENE
        or not _exact_int(result.get("seed"), _GENERIC_PROBE_SEED)
        or not _exact_int(
            result.get("physics_steps"), _GENERIC_PROBE_PHYSICS_STEPS
        )
    ):
        return "registered_workload_mismatch"
    expected_thresholds = {
        "required_cold_start_successes": _GENERIC_PROBE_REQUIRED_REPETITIONS,
        "required_physics_steps_per_start": _GENERIC_PROBE_PHYSICS_STEPS,
        "maximum_peak_memory_fraction": (
            _GENERIC_PROBE_MAXIMUM_PEAK_MEMORY_FRACTION
        ),
        "require_rtx_gpu": True,
        "required_hardware_capability": "nvidia_rtx_hardware_ray_tracing",
    }
    thresholds = result.get("qualification_thresholds")
    if (
        thresholds != expected_thresholds
        or not isinstance(thresholds, dict)
        or not _exact_int(
            thresholds.get("required_cold_start_successes"),
            _GENERIC_PROBE_REQUIRED_REPETITIONS,
        )
        or not _exact_int(
            thresholds.get("required_physics_steps_per_start"),
            _GENERIC_PROBE_PHYSICS_STEPS,
        )
        or type(thresholds.get("maximum_peak_memory_fraction")) is not float
    ):
        return "qualification_threshold_mismatch"
    if not _recorded_scene_template_binding_is_valid(result):
        return "scene_template_provenance_gate_failed"
    if not _generic_input_sha256_is_valid(result):
        return "input_sha256_gate_failed"
    if not _generic_stage_evidence_passes(result):
        return "registered_stage_evidence_failed"
    measurement_reason = _generic_gpu_measurement_failure_reason(result)
    if measurement_reason is not None:
        return measurement_reason
    if require_rtx and not _generic_rtx_hardware_gate_passes(result):
        return "hardware_rt_capability_gate_failed"
    if require_rtx and _generic_renderer_identity_failure_reason(result) is not None:
        return "renderer_identity_gate_failed"
    if result.get("single_run_gate_pass") is not True:
        return "single_run_gate_not_passed"
    if not _finite_number(result.get("elapsed_seconds")) or float(
        result["elapsed_seconds"]
    ) < 0:
        return "elapsed_seconds_invalid"
    return None


def _aggregate_empirical_probe(
    args: argparse.Namespace, profile_path: Path
) -> int:
    root = Path(args.aggregate_root).resolve()
    profile_path = profile_path.resolve()
    profile = _load_empirical_hardware_profile(profile_path)
    profile_sha256 = _sha256(profile_path)
    required = profile["probe_contract"]["required_repetitions"]
    if args.required_repetitions != required:
        raise ValueError("required repetitions differ from empirical hardware profile")

    result_paths = [root / f"repeat_{index}" / "result.json" for index in range(1, required + 1)]
    present_paths = [path for path in result_paths if path.is_file()]
    results = [json.loads(path.read_text(encoding="utf-8")) for path in present_paths]
    failure_counts: Counter[str] = Counter()
    if len(present_paths) != required:
        failure_counts["missing_repetition_artifact"] += required - len(present_paths)

    functional_successes = 0
    hardware_gate_passes = 0
    image_evidence_passes = 0
    passed = 0
    elapsed: list[float] = []
    peaks: list[float] = []
    fractions: list[float] = []
    observed_job_ids: set[str] = set()
    for path, result in zip(present_paths, results, strict=True):
        if result.get("status") == "success":
            functional_successes += 1
        if _empirical_hardware_gate_passes(result, profile):
            hardware_gate_passes += 1
        if _empirical_image_evidence_passes(result, profile, result_path=path):
            image_evidence_passes += 1
        reason = _empirical_result_failure_reason(
            result, profile, profile_sha256, result_path=path
        )
        if reason is None:
            passed += 1
        else:
            failure_counts[reason] += 1
        try:
            elapsed.append(float(result["elapsed_seconds"]))
        except (KeyError, TypeError, ValueError):
            pass
        measurements = result.get("gpu_measurements") or {}
        try:
            peaks.append(float(measurements["peak_memory_used_mib"]))
        except (KeyError, TypeError, ValueError):
            pass
        try:
            fractions.append(float(measurements["peak_memory_fraction"]))
        except (KeyError, TypeError, ValueError):
            pass
        job_id = str((result.get("slurm") or {}).get("job_id") or "").strip()
        if job_id:
            observed_job_ids.add(job_id)

    current_job_id = str(os.environ.get("SLURM_JOB_ID") or "").strip()
    if (
        len(observed_job_ids) != 1
        or (current_job_id and observed_job_ids != {current_job_id})
    ):
        failure_counts["slurm_job_binding_mismatch"] += 1

    scene_binding_identities = [
        identity
        for result in results
        if (
            identity := _recorded_scene_template_binding_identity(result)
        )
        is not None
    ]
    scene_binding_consistent = (
        len(scene_binding_identities) == len(results)
        and len(set(scene_binding_identities)) == 1
    )
    if (
        len(scene_binding_identities) == len(results)
        and len(set(scene_binding_identities)) > 1
    ):
        failure_counts["scene_template_binding_mismatch"] += 1

    qualified = (
        len(present_paths) == required
        and passed == required
        and scene_binding_consistent
        and not failure_counts
    )
    cost = float(profile["cost"]["listed_gpu_cost_units_per_hour"])
    summary = {
        "schema_version": EMPIRICAL_SUMMARY_SCHEMA_VERSION,
        "artifact_class": "infrastructure_qualification_not_experiment_evidence",
        "qualification_scope": "registered_omnigibson_workload_hardware_arm_only",
        "status": (
            EMPIRICAL_SUCCESS_STATUS if qualified else "not_empirically_qualified"
        ),
        "hardware_profile_id": profile["profile_id"],
        "hardware_profile_sha256": profile_sha256,
        "hardware_claims": profile["hardware_claims"],
        "observed_hardware_constraints": profile[
            "observed_hardware_constraints"
        ],
        "registered_workload": profile["registered_workload"],
        "prospective_evidence": True,
        "prior_artifact_reuse": False,
        "required_repetitions": required,
        "observed_repetitions": len(present_paths),
        "functional_success_repetitions": functional_successes,
        "hardware_gate_pass_repetitions": hardware_gate_passes,
        "image_evidence_pass_repetitions": image_evidence_passes,
        "passed_repetitions": passed,
        "scene_template_binding_consistent": scene_binding_consistent,
        "failure_reasons": [
            {"reason": reason, "count": failure_counts[reason]}
            for reason in sorted(failure_counts)
        ],
        "formal_workload_gate": {
            "eligible_for_formal_experiment": False,
            "required_next_evidence": "registered_full_task_compatible_online_smoke",
            "status": (
                "requires_registered_full_task_compatible_online_smoke"
                if qualified
                else "blocked_empirical_hardware_probe"
            ),
        },
        "listed_gpu_cost_units_per_hour": cost,
        "estimated_gpu_cost_units_for_probe": cost * sum(elapsed) / 3600.0,
        "total_elapsed_seconds": sum(elapsed),
        "mean_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "maximum_peak_memory_used_mib": max(peaks) if peaks else None,
        "maximum_peak_memory_fraction": max(fractions) if fractions else None,
        "result_files": [str(path.relative_to(root)) for path in result_paths],
        "slurm_job_id": (
            current_job_id
            or (next(iter(observed_job_ids)) if len(observed_job_ids) == 1 else None)
        ),
        "finished_at_utc": _utc_now(),
    }
    _atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if qualified else 3


def aggregate_probe(args: argparse.Namespace) -> int:
    hardware_profile = getattr(args, "hardware_profile", None)
    if hardware_profile:
        if getattr(args, "candidate_only", False) or getattr(
            args, "summary_output", None
        ):
            raise ValueError("candidate-only aggregation is not an empirical-profile mode")
        return _aggregate_empirical_probe(args, Path(hardware_profile))
    root = Path(args.aggregate_root).resolve()
    required = _GENERIC_PROBE_REQUIRED_REPETITIONS
    expected_result_paths = [
        root / f"repeat_{index}" / "result.json"
        for index in range(1, required + 1)
    ]
    result_paths = sorted(root.glob("repeat_*/result.json"))
    result_path_set_exact = (
        len(result_paths) == len(expected_result_paths)
        and set(result_paths) == set(expected_result_paths)
        and all(
            path.is_file()
            and not path.is_symlink()
            and not path.parent.is_symlink()
            for path in result_paths
        )
    )
    results: list[dict[str, Any]] = []
    invalid_json_count = 0
    for path in result_paths:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.parent.is_symlink()
        ):
            invalid_json_count += 1
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json_count += 1
            continue
        if not isinstance(result, dict):
            invalid_json_count += 1
            continue
        results.append(result)
    elapsed = [
        float(result["elapsed_seconds"])
        for result in results
        if _finite_number(result.get("elapsed_seconds"))
        and float(result["elapsed_seconds"]) >= 0
    ]
    peaks = [
        float(result["gpu_measurements"]["peak_memory_used_mib"])
        for result in results
        if _finite_number(
            result.get("gpu_measurements", {}).get("peak_memory_used_mib")
        )
    ]
    fractions = [
        float(result["gpu_measurements"]["peak_memory_fraction"])
        for result in results
        if _finite_number(
            result.get("gpu_measurements", {}).get("peak_memory_fraction")
        )
    ]
    memory_deltas = [
        float(result["gpu_measurements"]["peak_memory_delta_mib"])
        for result in results
        if _finite_number(
            result.get("gpu_measurements", {}).get("peak_memory_delta_mib")
        )
    ]
    peak_utilizations = [
        float(result["gpu_measurements"]["peak_utilization_percent"])
        for result in results
        if _finite_number(
            result.get("gpu_measurements", {}).get("peak_utilization_percent")
        )
    ]
    cost = _parse_number(os.environ.get("PROBE_GPU_COST", ""))
    if cost is not None and not math.isfinite(cost):
        cost = None
    functional_successes = sum(result.get("status") == "success" for result in results)
    hardware_gate_passes = sum(
        _generic_rtx_hardware_gate_passes(result)
        for result in results
    )
    gpu_activity_gate_passes = sum(
        _generic_gpu_activity_evidence_passes(result)
        for result in results
    )
    renderer_identity_gate_passes = sum(
        _generic_renderer_identity_failure_reason(result) is None
        for result in results
    )
    failure_counts = Counter(
        reason
        for result in results
        if (reason := _probe_failure_reason(result, require_rtx=args.require_rtx))
        is not None
    )
    if invalid_json_count:
        failure_counts["invalid_result_json"] += invalid_json_count
    if not result_path_set_exact:
        failure_counts["result_path_set_mismatch"] += 1
    requested_repetitions = getattr(args, "required_repetitions", None)
    requested_maximum = getattr(
        args,
        "max_peak_memory_fraction",
        None,
    )
    if (
        requested_repetitions != _GENERIC_PROBE_REQUIRED_REPETITIONS
        or getattr(args, "require_rtx", False) is not True
        or not _finite_number(requested_maximum)
        or float(requested_maximum)
        != _GENERIC_PROBE_MAXIMUM_PEAK_MEMORY_FRACTION
    ):
        failure_counts["aggregate_contract_mismatch"] += 1
    scene_binding_identities = [
        identity
        for result in results
        if (
            identity := _recorded_scene_template_binding_identity(result)
        )
        is not None
    ]
    scene_binding_consistent = (
        len(scene_binding_identities) == len(results)
        and len(set(scene_binding_identities)) == 1
    )
    if (
        len(scene_binding_identities) == len(results)
        and len(set(scene_binding_identities)) > 1
    ):
        failure_counts["scene_template_binding_mismatch"] += 1
    input_sha256_identities = [
        tuple(result["input_sha256"][key] for key in sorted(_GENERIC_INPUT_SHA256_KEYS))
        for result in results
        if _generic_input_sha256_is_valid(result)
    ]
    input_sha256_binding_consistent = (
        len(input_sha256_identities) == len(results)
        and len(set(input_sha256_identities)) == 1
    )
    if (
        len(input_sha256_identities) == len(results)
        and len(set(input_sha256_identities)) > 1
    ):
        failure_counts["input_sha256_binding_mismatch"] += 1
    passed = sum(
        _probe_failure_reason(result, require_rtx=args.require_rtx) is None
        for result in results
    )
    qualified = (
        result_path_set_exact
        and len(results) == required
        and passed == required
        and scene_binding_consistent
        and input_sha256_binding_consistent
        and gpu_activity_gate_passes == required
        and renderer_identity_gate_passes == required
        and not failure_counts
    )
    candidate_only = bool(getattr(args, "candidate_only", False))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_class": "infrastructure_diagnostic_not_experiment_evidence",
        "qualification_scope": "omnigibson_infrastructure_only",
        "status": (
            "candidate_qualified"
            if qualified and candidate_only
            else "qualified" if qualified else "not_qualified"
        ),
        "gpu_label": os.environ.get("PROBE_GPU_LABEL"),
        "listed_gpu_cost_units_per_hour": cost,
        "require_rtx_gpu": args.require_rtx,
        "required_repetitions": required,
        "requested_repetitions": requested_repetitions,
        "observed_repetitions": len(result_paths),
        "functional_success_repetitions": functional_successes,
        "hardware_gate_pass_repetitions": hardware_gate_passes,
        "gpu_activity_gate_pass_repetitions": gpu_activity_gate_passes,
        "renderer_identity_gate_pass_repetitions": renderer_identity_gate_passes,
        "passed_repetitions": passed,
        "result_path_set_exact": result_path_set_exact,
        "scene_template_binding_consistent": scene_binding_consistent,
        "input_sha256_binding_consistent": input_sha256_binding_consistent,
        "failure_reasons": [
            {"reason": reason, "count": failure_counts[reason]}
            for reason in sorted(failure_counts)
        ],
        "formal_workload_gate": {
            "status": (
                "requires_full_task_compatible_smoke"
                if qualified
                else "blocked_infrastructure_probe"
            ),
            "eligible_for_formal_experiment": False,
            "required_next_evidence": "registered_full_task_compatible_online_smoke",
        },
        "total_elapsed_seconds": sum(elapsed),
        "mean_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "maximum_peak_memory_used_mib": max(peaks) if peaks else None,
        "maximum_peak_memory_fraction": max(fractions) if fractions else None,
        "minimum_peak_memory_delta_mib": min(memory_deltas) if memory_deltas else None,
        "maximum_peak_memory_delta_mib": max(memory_deltas) if memory_deltas else None,
        "minimum_peak_utilization_percent": (
            min(peak_utilizations) if peak_utilizations else None
        ),
        "maximum_peak_utilization_percent": (
            max(peak_utilizations) if peak_utilizations else None
        ),
        "estimated_gpu_cost_units_for_probe": (
            cost * sum(elapsed) / 3600.0 if cost is not None else None
        ),
        "result_files": [str(path.relative_to(root)) for path in result_paths],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "finished_at_utc": _utc_now(),
    }
    if candidate_only:
        summary["terminal_qualification_pending"] = True
        summary_output = Path(str(args.summary_output)).resolve()
        if (
            summary_output.parent != root
            or summary_output.name != "aggregate_candidate.json"
            or summary_output.exists()
            or summary_output.is_symlink()
        ):
            raise ValueError(
                "candidate-only summary output must be a new aggregate_candidate.json "
                "directly below --aggregate-root"
            )
    else:
        summary_output = root / "summary.json"
    _atomic_json(summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if qualified else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--task", default="removing_lint_from_dryer")
    parser.add_argument("--scene", default="Beechwood_0_garden")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physics-steps", type=int, default=60)
    parser.add_argument("--max-peak-memory-fraction", type=float, default=0.90)
    parser.add_argument("--require-rtx", action="store_true")
    parser.add_argument("--hardware-profile")
    parser.add_argument("--output")
    parser.add_argument("--aggregate-root")
    parser.add_argument("--required-repetitions", type=int, default=2)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--summary-output")
    args = parser.parse_args()
    if bool(args.output) == bool(args.aggregate_root):
        parser.error("provide exactly one of --output or --aggregate-root")
    if args.candidate_only != bool(args.summary_output):
        parser.error("--candidate-only and --summary-output must be provided together")
    if args.candidate_only and not args.aggregate_root:
        parser.error("--candidate-only is valid only with --aggregate-root")
    return args


def main() -> int:
    args = parse_args()
    if args.aggregate_root:
        return aggregate_probe(args)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
