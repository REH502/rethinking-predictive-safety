"""Frozen-manifest and offline-preflight entry point for resilient HA pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

_HORIZONS = [1, 2, 3, 4, 5, 6]
_ROLLOUT_MODES = [
    "direct_multi_horizon",
    "iterative_wm_rollout",
]
_POLICY_MODES = [
    "train_policy",
    "full_harnessaudit_rules",
]
_MODEL_ID = "gpt-5.6-luna"
_DEEPSEEK_MODEL_ID = "deepseek-v4-flash"
_AGENTWORLD_MODEL_ID = "Qwen-AgentWorld-35B-A3B"
_QWEN35_PLUS_MODEL_ID = "qwen3.5-plus-2026-02-15"
_SUPPORTED_MODEL_IDS = {
    _MODEL_ID,
    _DEEPSEEK_MODEL_ID,
    _AGENTWORLD_MODEL_ID,
    _QWEN35_PLUS_MODEL_ID,
}
_RESOURCE_SEMANTICS = "legacy_official_v1"
_STUDY_ID = "ha_luna_resilient_predict_only"
_DEEPSEEK_STUDY_ID = "ha_deepseek_resilient_predict_only"
_AGENTWORLD_STUDY_ID = "ha_agentworld_resilient_predict_only"
_QWEN35_PLUS_STUDY_ID = "ha_qwen35_plus_resilient_predict_only"
_DIAGNOSTIC_LABEL = "diagnostic_only"
_DEFAULT_MODEL_ENDPOINT = "https://aifast.site/v1"
_AIFAST_ROOT = "https://aifast.site"
_AIFAST_ENUMERATION_SCRIPT = Path(
    "/home/zhliu/.codex/skills/aifast-model-calls/scripts/aifast_request.py"
)
_RUN_ID_PATTERN = re.compile(
    r"ha-(?:luna|deepseek|agentworld|qwen35-plus)-predict-only-[0-9a-f]{16}\Z"
)
_ENDPOINT_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(authorization|bearer|api[_-]?key|access[_-]?token|password|"
    r"passwd|secret|sk-[a-z0-9_-]+)"
)
_STATUS_FILENAME = "batch.status.jsonl"
_PROGRESS_FILENAME = "progress.json"
_REQUIRED_CHILD_ARTIFACTS = {
    "points",
    "quality_report",
    "source_audit_json",
    "source_audit_csv",
    "provenance_manifest",
}
_RUNNER_IMPLEMENTATION_PATH = (
    "harness_env/orgin/HarnessAudit/multi_agent/"
    "safepred_experiments/ha_resilient_predict_only.py"
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _frozen_child_workers() -> int:
    """Return the bounded worker count bound into child provenance.

    The original frozen runner used one worker.  Run-local parallel wrappers
    may raise this value, but the value must be recorded and validated rather
    than rejected as a provenance mismatch.
    """

    raw = str(os.environ.get("SAFEPRED_FROZEN_CHILD_WORKERS") or "1").strip()
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError("SAFEPRED_FROZEN_CHILD_WORKERS must be an integer") from exc
    if workers < 1 or workers > 64:
        raise ValueError("SAFEPRED_FROZEN_CHILD_WORKERS must be in [1, 64]")
    return workers


def _is_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _deterministic_run_id(
    *,
    partial_audit_sha256: str,
    ordered_task_ids: list[str],
    model_id: str,
    requested_revision_label: str,
    reasoning_mode: str | None = None,
) -> str:
    identity = {
        "partial_audit_sha256": str(partial_audit_sha256),
        "task_ids": [str(task_id) for task_id in ordered_task_ids],
        "model_id": str(model_id),
        "model_revision_label": str(requested_revision_label),
    }
    if reasoning_mode is not None:
        identity["reasoning_mode"] = str(reasoning_mode)
    if model_id == _MODEL_ID:
        prefix = "ha-luna-predict-only"
    elif model_id == _DEEPSEEK_MODEL_ID:
        prefix = "ha-deepseek-predict-only"
    elif model_id == _AGENTWORLD_MODEL_ID:
        prefix = "ha-agentworld-predict-only"
    elif model_id == _QWEN35_PLUS_MODEL_ID:
        prefix = "ha-qwen35-plus-predict-only"
    else:
        _study_id_for_model(model_id)
        raise AssertionError("unreachable supported model identity")
    return f"{prefix}-{_canonical_sha256(identity)[:16]}"


def _study_id_for_model(model_id: str) -> str:
    if model_id == _MODEL_ID:
        return _STUDY_ID
    if model_id == _DEEPSEEK_MODEL_ID:
        return _DEEPSEEK_STUDY_ID
    if model_id == _AGENTWORLD_MODEL_ID:
        return _AGENTWORLD_STUDY_ID
    if model_id == _QWEN35_PLUS_MODEL_ID:
        return _QWEN35_PLUS_STUDY_ID
    raise ValueError(
        f"unsupported model id {model_id!r}; expected one of "
        f"{sorted(_SUPPORTED_MODEL_IDS)}"
    )


def _validated_reasoning_mode(
    value: Any,
    *,
    model_id: str,
) -> str | None:
    if value is None:
        return None
    reasoning_mode = str(value).strip()
    if reasoning_mode not in {"provider_default", "disabled"}:
        raise ValueError(
            "reasoning_mode must equal provider_default or disabled"
        )
    if reasoning_mode == "disabled" and not str(model_id).lower().startswith(
        "deepseek-v4"
    ):
        raise ValueError(
            "disabled reasoning mode requires a deepseek-v4 model"
        )
    return reasoning_mode


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_model_endpoint(value: Any) -> str:
    endpoint = str(value or "")
    if not endpoint or endpoint != endpoint.strip():
        raise ValueError("model endpoint must be a non-empty safe URL")
    if any(character.isspace() or ord(character) < 32 for character in endpoint):
        raise ValueError("model endpoint must be a non-empty safe URL")
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("model endpoint must be a non-empty safe URL") from exc
    if parsed.scheme not in {"https", "http"} or not hostname:
        raise ValueError("model endpoint must use HTTP(S) with a valid host")
    if parsed.scheme == "http" and hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("model endpoint must use HTTPS for non-local hosts")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("model endpoint must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise ValueError("model endpoint must not contain a query or fragment")
    decoded_endpoint = unquote(endpoint)
    if _ENDPOINT_CREDENTIAL_PATTERN.search(decoded_endpoint):
        raise ValueError("model endpoint must not contain credential material")
    return endpoint.rstrip("/")


def _validated_agentworld_endpoint(value: Any) -> str:
    endpoint = _validated_model_endpoint(value)
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path != "/v1"
    ):
        raise ValueError(
            "AgentWorld local_vllm endpoint must use loopback HTTP "
            "http://127.0.0.1:<port>/v1"
        )
    return endpoint


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path) -> Path:
    absolute = _lexical_absolute(path)
    for candidate in [*reversed(absolute.parents), absolute]:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"run path contains a symlink component: {candidate}")
    return absolute


def _safe_run_path(path: Path, *, run_root: Path) -> Path:
    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    absolute = _lexical_absolute(path)
    if not _is_relative_to(absolute, root):
        raise ValueError(f"run path is outside run_root: {absolute}")
    return _assert_no_symlink_components(absolute)


def _ensure_safe_directory(path: Path, *, run_root: Path) -> Path:
    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    if not root.exists():
        _assert_no_symlink_components(root.parent)
        root.mkdir(parents=True)
        _assert_no_symlink_components(root)
    if not root.is_dir():
        raise ValueError(f"run_root is not a directory: {root}")
    directory = _lexical_absolute(path)
    if not _is_relative_to(directory, root):
        raise ValueError(f"run directory is outside run_root: {directory}")
    relative_parts = directory.relative_to(root).parts
    current = root
    for part in relative_parts:
        current = current / part
        _assert_no_symlink_components(current)
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _assert_no_symlink_components(current)
        if not current.is_dir():
            raise ValueError(f"run path is not a directory: {current}")
    return directory


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    run_root: Path,
) -> None:
    destination = _safe_run_path(path, run_root=run_root)
    _ensure_safe_directory(destination.parent, run_root=run_root)
    _safe_run_path(destination, run_root=run_root)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_task_name(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id)).strip("._")
    if not safe:
        raise ValueError("task_id cannot form a safe artifact path")
    return safe


def _task_paths(run_root: Path, task_id: str) -> dict[str, Path]:
    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    safe_task = _safe_task_name(task_id)
    return {
        "result": root / "results" / safe_task,
        "trace": root / "traces" / safe_task,
        "diagnostic": root / "diagnostics" / safe_task,
    }


def _parse_csv_bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be exactly true or false")


def _split_codes(value: Any) -> list[str]:
    return sorted(
        {
            code.strip()
            for code in str(value or "").replace(",", ";").split(";")
            if code.strip()
        }
    )


def _validate_child_point_grid(
    manifest: dict[str, Any],
    index: int,
    points_path: Path,
) -> dict[str, Any]:
    source_reports = manifest["source_reports"]
    if index < 0 or index >= len(source_reports):
        raise ValueError(f"source index is out of range: {index}")
    source = source_reports[index]
    task_id = str(source["task_id"])
    point_by_sequence = {
        int(point["sequence_num"]): point for point in source["points"]
    }
    expected_keys = {
        (
            int(point["sequence_num"]),
            int(horizon),
            str(rollout_mode),
            str(policy_mode),
        )
        for point in source["points"]
        for horizon in manifest["matrix"]["horizons"]
        for rollout_mode in manifest["matrix"]["rollout_modes"]
        for policy_mode in manifest["matrix"]["policy_modes"]
    }
    if len(expected_keys) != 48:
        raise ValueError("frozen source does not define the exact 48-row grid")

    with Path(points_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 48:
        raise ValueError(f"child point CSV must contain exactly 48 rows, got {len(rows)}")

    observed_keys: list[tuple[int, int, str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if str(row.get("task_id") or "") != task_id:
            raise ValueError(f"child grid row {row_number} has the wrong task")
        try:
            sequence_num = int(str(row.get("action_sequence_num") or ""))
            point_num = int(str(row.get("point_num") or ""))
            horizon = int(str(row.get("horizon") or ""))
        except ValueError as exc:
            raise ValueError(
                f"child grid row {row_number} has invalid point/horizon identity"
            ) from exc
        point = point_by_sequence.get(sequence_num)
        if point is None or point_num != int(point["point_num"]):
            raise ValueError(f"child grid row {row_number} has the wrong point")
        key = (
            sequence_num,
            horizon,
            str(row.get("rollout_mode") or ""),
            str(row.get("policy_mode") or ""),
        )
        observed_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"child grid row {row_number} has the wrong condition")
        if not _parse_csv_bool(row.get("valid_prediction"), field="valid_prediction"):
            raise ValueError(f"child grid row {row_number} is not a valid prediction")
        if _parse_csv_bool(row.get("decision_error"), field="decision_error"):
            raise ValueError(f"child grid row {row_number} contains a decision error")
        if _parse_csv_bool(
            row.get("skipped_by_prediction_interval"),
            field="skipped_by_prediction_interval",
        ):
            raise ValueError(f"child grid row {row_number} was interval-skipped")
        if str(row.get("gt_estimand") or "") != "exact_step":
            raise ValueError(f"child grid row {row_number} has the wrong GT estimand")
        if not _parse_csv_bool(
            row.get("gt_target_available"), field="gt_target_available"
        ) or not _parse_csv_bool(
            row.get("gt_exact_available"), field="gt_exact_available"
        ):
            raise ValueError(f"child grid row {row_number} lacks exact-step GT")
        frozen_gt = next(
            (
                gt
                for gt in point["exact_step_ground_truth"]
                if int(gt["horizon"]) == horizon
            ),
            None,
        )
        if not isinstance(frozen_gt, dict) or frozen_gt.get("available") is not True:
            raise ValueError(f"frozen exact-step GT is unavailable at row {row_number}")
        expected_violated = frozen_gt.get("violated") is True
        if (
            _parse_csv_bool(
                row.get("gt_target_violated"), field="gt_target_violated"
            )
            != expected_violated
            or _parse_csv_bool(
                row.get("gt_exact_violated"), field="gt_exact_violated"
            )
            != expected_violated
        ):
            raise ValueError(f"child grid row {row_number} has changed GT labels")
        expected_codes = sorted(str(code) for code in frozen_gt["violation_codes"])
        if (
            _split_codes(row.get("gt_target_codes")) != expected_codes
            or _split_codes(row.get("gt_exact_codes")) != expected_codes
        ):
            raise ValueError(f"child grid row {row_number} has changed GT codes")

    if len(set(observed_keys)) != 48:
        raise ValueError("child point CSV contains duplicate condition keys")
    if set(observed_keys) != expected_keys:
        raise ValueError("child point CSV does not contain the complete condition grid")
    return {
        "row_count": 48,
        "unique_condition_key_count": 48,
        "point_keys": list(source["point_keys"]),
    }


def _artifact_record(path: Path, *, run_root: Path) -> dict[str, Any]:
    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    absolute = _lexical_absolute(path)
    if not _is_relative_to(absolute, root):
        raise ValueError(f"child artifact is outside run_root: {absolute}")
    _assert_no_symlink_components(absolute)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"child artifact must be a regular file: {absolute}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"child artifact must be a regular file: {absolute}")
    return {
        "path": str(absolute),
        "size": metadata.st_size,
        "sha256": _file_sha256(absolute),
    }


def _active_secret_values() -> list[str]:
    return sorted(
        {
            str(value)
            for name, value in os.environ.items()
            if value
            and (
                "API_KEY" in name.upper()
                or "ACCESS_TOKEN" in name.upper()
                or "PASSWORD" in name.upper()
                or "SECRET" in name.upper()
            )
        },
        key=len,
        reverse=True,
    )


def _sanitize_persisted_text(value: str) -> str:
    sanitized = str(value)
    for secret in _active_secret_values():
        sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*[^\r\n]*",
        "[REDACTED_HEADER]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\bbearer\s+[^\s,;\"'}\]]+",
        "[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(?:WM|CUSTOM|OPENAI|AIFAST)_API_KEY\s*=\s*[^\s]*",
        "[REDACTED_ENV]",
        sanitized,
    )
    return sanitized


def _sanitize_log_file(path: Path, *, run_root: Path) -> None:
    destination = _safe_run_path(path, run_root=run_root)
    sanitized = _sanitize_persisted_text(
        destination.read_text(encoding="utf-8", errors="replace")
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(sanitized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _safe_run_path(destination, run_root=run_root)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_safe_bound_artifact(path: Path) -> None:
    payload = Path(path).read_bytes()
    for secret in _active_secret_values():
        if secret.encode("utf-8") in payload:
            raise ValueError(f"bound artifact contains active credential: {path}")
    text = payload.decode("utf-8", errors="ignore")
    if re.search(
        r"(?ix)"
        r"\b(?:authorization|proxy-authorization)"
        r"[\"']?\s*[:=]\s*[\"']?\s*bearer\s+"
        r"[A-Za-z0-9._~+/\-]{8,}=*",
        text,
    ):
        raise ValueError(f"bound artifact contains a raw credential header: {path}")
    if re.search(
        r"(?i)\b(?:WM|CUSTOM|OPENAI|AIFAST)_API_KEY\s*=",
        text,
    ):
        raise ValueError(f"bound artifact contains an environment credential: {path}")


def _validate_source_audit_artifacts(
    manifest: dict[str, Any],
    index: int,
    source_audit_path: Path,
    source_csv_path: Path,
) -> dict[str, Any]:
    source_audit = _read_json_object(source_audit_path, label="source audit")
    expected_top_fields = {
        "artifact_type",
        "report_count",
        "valid_report_count",
        "invalid_report_count",
        "access_gt_recomputation_required",
        "access_gt_recomputation_schema_version",
        "resource_semantics_modes",
        "resource_semantics_uniform",
        "actor_cohort_values",
        "actor_cohort_uniformity",
        "actor_cohort_uniform",
        "checker_file",
        "checker_sha256",
        "reports",
    }
    if set(source_audit) != expected_top_fields:
        raise ValueError("child source audit has missing or unknown fields")
    reports = source_audit.get("reports")
    if (
        source_audit.get("artifact_type") != "harnessaudit_clean_source_audit"
        or type(source_audit.get("report_count")) is not int
        or source_audit.get("report_count") != 1
        or type(source_audit.get("valid_report_count")) is not int
        or source_audit.get("valid_report_count") != 1
        or type(source_audit.get("invalid_report_count")) is not int
        or source_audit.get("invalid_report_count") != 0
        or not isinstance(reports, list)
        or len(reports) != 1
        or not isinstance(reports[0], dict)
    ):
        raise ValueError("child source audit counters are not a clean single report")
    source = manifest["source_reports"][index]
    report = reports[0]
    expected_report_fields = {
        "report_path",
        "trace_file",
        "task_id",
        "run_id",
        "action_count",
        "access_labeled_action_count",
        "recomputed_access_decision_count",
        "recomputed_violation_count",
        "access_decisions_match_current_checker",
        "resource_semantics_mode",
        "resource_semantics_provenance_valid",
        "actor_protocol_profile",
        "actor_model",
        "actor_model_revision",
        "actor_endpoint",
        "actor_runtime_versions_sha256",
        "actor_implementation_manifest_sha256",
        "source_report_sha256",
        "source_trace_sha256",
        "task_file",
        "task_sha256",
        "effective_rule_count",
        "legacy_resource_rule_count",
        "current_rule_set_formal_valid",
        "effective_rules_sha256",
        "checker_file",
        "checker_sha256",
        "is_clean_replayable",
        "errors",
    }
    if set(report) != expected_report_fields:
        raise ValueError("child source audit report has missing or unknown fields")
    frozen_report = _read_json_object(
        Path(str(source["report_path"])),
        label="frozen source report",
    )
    actor_provenance = frozen_report.get("actor_provenance")
    if not isinstance(actor_provenance, dict):
        raise ValueError("frozen source report lacks actor provenance")
    actor_hashes = actor_provenance.get("hashes")
    if not isinstance(actor_hashes, dict):
        raise ValueError("frozen source report lacks actor implementation hashes")
    actor_fields = {
        "actor_protocol_profile": str(
            actor_provenance.get("actor_protocol_profile") or ""
        ),
        "actor_model": str(actor_provenance.get("model") or ""),
        "actor_model_revision": str(
            actor_provenance.get("model_revision") or ""
        ),
        "actor_endpoint": str(actor_provenance.get("endpoint") or ""),
        "actor_runtime_versions_sha256": _canonical_sha256(
            actor_provenance.get("runtime_versions") or {}
        ),
        "actor_implementation_manifest_sha256": str(
            actor_hashes.get("implementation_manifest_sha256") or ""
        ),
    }
    if any(
        not isinstance(actor_fields[field], str) or not actor_fields[field]
        for field in (
            "actor_protocol_profile",
            "actor_model",
            "actor_model_revision",
            "actor_endpoint",
        )
    ):
        raise ValueError("child source audit actor provenance is incomplete")
    _validated_model_endpoint(actor_fields["actor_endpoint"])
    expected_identity = {
        "report_path": str(Path(str(source["report_path"])).resolve()),
        "trace_file": str(Path(str(source["trace_path"])).resolve()),
        "task_id": str(source["task_id"]),
        "run_id": str(source["run_id"]),
        "action_count": int(source["source_action_count"]),
        "resource_semantics_mode": _RESOURCE_SEMANTICS,
        "resource_semantics_provenance_valid": True,
        **actor_fields,
        "source_report_sha256": str(source["report_sha256"]),
        "source_trace_sha256": str(source["trace_sha256"]),
        "is_clean_replayable": True,
        "errors": [],
    }
    if any(
        not _json_exact(report.get(key), value)
        for key, value in expected_identity.items()
    ):
        raise ValueError("child source audit does not bind the frozen source identity")
    for field in (
        "access_labeled_action_count",
        "recomputed_access_decision_count",
        "recomputed_violation_count",
        "effective_rule_count",
        "legacy_resource_rule_count",
    ):
        if (
            type(report.get(field)) is not int
            or int(report[field]) < 0
        ):
            raise ValueError(f"child source audit {field} must be nonnegative")
    action_count = int(report["action_count"])
    if (
        report["access_labeled_action_count"] != action_count
        or report["recomputed_access_decision_count"]
        < report["access_labeled_action_count"]
        or report["recomputed_violation_count"]
        > report["recomputed_access_decision_count"]
        or report["legacy_resource_rule_count"] > report["effective_rule_count"]
    ):
        raise ValueError("child source audit count relationships are inconsistent")
    for field in (
        "access_decisions_match_current_checker",
        "current_rule_set_formal_valid",
        "is_clean_replayable",
    ):
        if report.get(field) is not True:
            raise ValueError(f"child source audit {field} must be true")
    for field in (
        "actor_runtime_versions_sha256",
        "actor_implementation_manifest_sha256",
        "source_report_sha256",
        "source_trace_sha256",
        "task_sha256",
        "effective_rules_sha256",
        "checker_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(report.get(field) or "")):
            raise ValueError(f"child source audit {field} must be a SHA-256")
    for path_field, hash_field in (
        ("task_file", "task_sha256"),
        ("checker_file", "checker_sha256"),
    ):
        bound_path = Path(str(report.get(path_field) or ""))
        if (
            not bound_path.is_absolute()
            or not bound_path.is_file()
            or _file_sha256(bound_path) != report[hash_field]
        ):
            raise ValueError(f"child source audit {path_field} binding changed")
    if (
        source_audit.get("access_gt_recomputation_required") is not True
        or type(source_audit.get("access_gt_recomputation_schema_version"))
        is not int
        or source_audit["access_gt_recomputation_schema_version"] != 1
        or source_audit.get("resource_semantics_modes")
        != [_RESOURCE_SEMANTICS]
        or source_audit.get("resource_semantics_uniform") is not True
        or source_audit.get("actor_cohort_uniform") is not True
        or source_audit.get("checker_file") != report["checker_file"]
        or source_audit.get("checker_sha256") != report["checker_sha256"]
    ):
        raise ValueError("child source audit cohort/checker semantics changed")
    actor_cohort_values = source_audit.get("actor_cohort_values")
    actor_cohort_uniformity = source_audit.get("actor_cohort_uniformity")
    if (
        not isinstance(actor_cohort_values, dict)
        or set(actor_cohort_values) != set(actor_fields)
        or not isinstance(actor_cohort_uniformity, dict)
        or set(actor_cohort_uniformity) != set(actor_fields)
        or any(
            actor_cohort_values.get(field) != [value]
            for field, value in actor_fields.items()
        )
        or any(
            actor_cohort_uniformity.get(field) is not True
            for field in actor_fields
        )
    ):
        raise ValueError("child source audit actor cohort is not uniformly bound")

    with source_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        csv_fields = list(reader.fieldnames or [])
    expected_fields = list(report)
    if csv_fields != expected_fields or len(csv_rows) != 1:
        raise ValueError("child source audit CSV does not match its JSON report")
    expected_csv_row = {
        key: (
            " | ".join(str(item) for item in value)
            if key == "errors" and isinstance(value, list)
            else str(value)
        )
        for key, value in report.items()
    }
    if csv_rows[0] != expected_csv_row:
        raise ValueError("child source audit CSV does not match its JSON report")
    return source_audit


def _provenance_fingerprint_from_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": record["path"],
        "exists": True,
        "size_bytes": record["size"],
        "sha256": record["sha256"],
    }


def _validate_fingerprint_record(record: Any, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} fingerprint must be an object")
    exists = record.get("exists")
    expected_fields = (
        {"path", "exists", "size_bytes", "sha256"}
        if exists is True
        else {"path", "exists"}
    )
    path = Path(str(record.get("path") or ""))
    if (
        set(record) != expected_fields
        or type(exists) is not bool
        or not path.is_absolute()
    ):
        raise ValueError(f"{label} fingerprint has an invalid schema")
    if exists is True and (
        type(record.get("size_bytes")) is not int
        or record["size_bytes"] < 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""))
    ):
        raise ValueError(f"{label} fingerprint has invalid file metadata")


def _validate_provenance_manifest(
    manifest: dict[str, Any],
    index: int,
    records: dict[str, dict[str, Any]],
    quality: dict[str, Any],
) -> None:
    provenance = _read_json_object(
        Path(records["provenance_manifest"]["path"]),
        label="provenance manifest",
    )
    expected_top_fields = {
        "schema_version",
        "experiment_type",
        "benchmark",
        "created_at_utc",
        "runtime",
        "config",
        "status",
        "inputs",
        "code",
        "artifacts",
        "manifest_payload_sha256",
    }
    if set(provenance) != expected_top_fields:
        raise ValueError("child provenance has missing or unknown fields")
    if (
        type(provenance.get("schema_version")) is not int
        or provenance.get("schema_version") != 1
        or provenance.get("experiment_type") != "offline_predict_only"
        or provenance.get("benchmark") != "harnessaudit"
    ):
        raise ValueError("child provenance has the wrong experiment identity")
    runtime = provenance.get("runtime")
    if (
        not _is_timestamp(provenance.get("created_at_utc"))
        or not isinstance(runtime, dict)
        or set(runtime) != {"python", "executable", "platform"}
        or any(
            not isinstance(runtime.get(field), str) or not runtime[field]
            for field in ("python", "executable", "platform")
        )
    ):
        raise ValueError("child provenance runtime/timestamp is invalid")
    claimed_hash = provenance.get("manifest_payload_sha256")
    unhashed = dict(provenance)
    unhashed.pop("manifest_payload_sha256", None)
    expected_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if claimed_hash != expected_hash:
        raise ValueError("child provenance payload hash changed")

    source = manifest["source_reports"][index]
    model_id = str(manifest["model"]["id"])
    frozen_report = _read_json_object(
        Path(str(source["report_path"])),
        label="frozen source report",
    )
    if (
        isinstance(frozen_report.get("actor_protocol_profile"), str)
        and isinstance(frozen_report.get("actor_replicate_id"), str)
    ):
        actor_protocol_profile = frozen_report["actor_protocol_profile"]
        actor_replicate_id = frozen_report["actor_replicate_id"]
        actor_trace_seed = frozen_report.get("actor_trace_seed")
    else:
        fallback_seed = (
            frozen_report.get("actor_trace_seed")
            if type(frozen_report.get("actor_trace_seed")) is int
            else 0
        )
        actor_protocol_profile = "seeded_sampling_v1"
        actor_replicate_id = f"seed-{fallback_seed}"
        actor_trace_seed = fallback_seed
    expected_bindings = {
        "tasks": [],
        "source_reports": [source["report_path"]],
        "policy_modes": list(manifest["matrix"]["policy_modes"]),
        "rollout_modes": list(manifest["matrix"]["rollout_modes"]),
        "horizons": list(manifest["matrix"]["horizons"]),
        "point_config": {
            "scheme": "explicit",
            "count": 2,
            "interval": 1,
            "explicit_sequences": list(source["selected_sequence_nums"]),
        },
        "model": model_id,
        "judge_model": model_id,
        "clean_agent_api": None,
        "candidate_count": 1,
        "max_trace_events": 64,
        "gt_estimand": "exact_step",
        "workers": _frozen_child_workers(),
        "skip_existing": False,
        "experiment_seed": 0,
        "actor_protocol_profile": actor_protocol_profile,
        "actor_replicate_id": actor_replicate_id,
        "actor_trace_seed": actor_trace_seed,
        "wm_seed": 0,
        "formal": False,
        "preregistration_sha256": None,
        "offline_preregistration_gate": {"status": "not_requested"},
    }
    config = provenance.get("config")
    expected_config_fields = {
        "tasks",
        "source_reports",
        "policy_modes",
        "rollout_modes",
        "horizons",
        "point_config",
        "model",
        "judge_model",
        "model_api",
        "clean_agent_api",
        "candidate_count",
        "max_trace_events",
        "gt_estimand",
        "workers",
        "skip_existing",
        "policy_inputs_signature",
        "experiment_seed",
        "actor_protocol_profile",
        "actor_replicate_id",
        "actor_trace_seed",
        "wm_seed",
        "prepared_train_policies",
        "formal",
        "preregistration_sha256",
        "offline_preregistration_gate",
    }
    if (
        not isinstance(config, dict)
        or set(config) != expected_config_fields
        or any(
        not _json_exact(config.get(key), value)
        for key, value in expected_bindings.items()
        )
    ):
        raise ValueError("child provenance config does not bind the frozen child")
    if (
        actor_protocol_profile
        not in {"deterministic_close_source_v1", "seeded_sampling_v1"}
        or not actor_replicate_id
        or (
            actor_protocol_profile == "deterministic_close_source_v1"
            and actor_trace_seed is not None
        )
        or (
            actor_protocol_profile == "seeded_sampling_v1"
            and type(actor_trace_seed) is not int
        )
    ):
        raise ValueError("child provenance actor identity is invalid")
    model_api = config.get("model_api")
    expected_model_api_fields = {
        "provider",
        "api_key",
        "api_url",
        "model",
        "model_revision",
        "policy_extraction_model",
    }
    if (
        not isinstance(model_api, dict)
        or set(model_api) != expected_model_api_fields
        or model_api.get("provider") != "custom"
        or (
            model_api.get("api_key") is not None
            and model_api.get("api_key") != "[REDACTED]"
        )
        or model_api.get("api_url") != manifest["model"]["endpoint"]
        or model_api.get("model") != model_id
        or model_api.get("model_revision")
        != manifest["model"]["requested_revision_label"]
        or model_api.get("policy_extraction_model") != model_id
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(config.get("policy_inputs_signature") or ""),
        )
    ):
        raise ValueError("child provenance model/policy config changed")
    prepared = config.get("prepared_train_policies")
    if (
        not isinstance(prepared, list)
        or len(prepared) != 1
        or not isinstance(prepared[0], dict)
        or set(prepared[0])
        != {
            "task_id",
            "policy_count",
            "split_ids",
            "policy_extraction_model",
            "policy_extraction_seed",
        }
        or prepared[0].get("task_id") != source["task_id"]
        or type(prepared[0].get("policy_count")) is not int
        or prepared[0]["policy_count"] < 0
        or not isinstance(prepared[0].get("split_ids"), list)
        or not all(
            isinstance(value, str) and value
            for value in prepared[0]["split_ids"]
        )
        or prepared[0]["split_ids"]
        != sorted(set(prepared[0]["split_ids"]))
        or prepared[0].get("policy_extraction_model") != model_id
        or type(prepared[0].get("policy_extraction_seed")) is not int
        or prepared[0]["policy_extraction_seed"] != 0
    ):
        raise ValueError("child provenance prepared policy binding changed")
    if not _json_exact(provenance.get("status"), quality):
        raise ValueError("child provenance status does not equal the quality report")

    inputs = provenance.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("child provenance inputs must be fingerprint records")
    for record in inputs:
        _validate_fingerprint_record(record, label="input")
    expected_inputs = []
    for path_key, hash_key in (
        ("report_path", "report_sha256"),
        ("trace_path", "trace_sha256"),
    ):
        path = Path(str(source[path_key])).expanduser().absolute()
        expected_inputs.append(
            {
                "path": str(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": source[hash_key],
            }
        )
    for expected_input in expected_inputs:
        if not any(
            _json_exact(record, expected_input) for record in inputs
        ):
            raise ValueError("child provenance does not bind the source report and trace")

    code = provenance.get("code")
    if not isinstance(code, list) or not code:
        raise ValueError("child provenance code fingerprints must not be empty")
    for record in code:
        _validate_fingerprint_record(record, label="code")
    for implementation in manifest["implementation_files"]:
        path = Path(str(implementation["path"])).expanduser().absolute()
        if not any(
            record.get("path") == str(path)
            and record.get("exists") is True
            and record.get("sha256") == implementation["sha256"]
            for record in code
        ):
            raise ValueError(
                "child provenance omits a frozen implementation file"
            )

    provenance_artifacts = provenance.get("artifacts")
    if not isinstance(provenance_artifacts, list):
        raise ValueError("child provenance artifacts must be fingerprint records")
    for record in provenance_artifacts:
        _validate_fingerprint_record(record, label="artifact")
    expected_artifacts = [
        _provenance_fingerprint_from_record(record)
        for name, record in sorted(records.items())
        if name != "provenance_manifest"
    ]
    for expected_artifact in expected_artifacts:
        if not any(
            _json_exact(record, expected_artifact)
            for record in provenance_artifacts
        ):
            raise ValueError("child provenance omits an authoritative output artifact")
    provenance_path = records["provenance_manifest"]["path"]
    if any(record.get("path") == provenance_path for record in provenance_artifacts):
        raise ValueError("child provenance must exclude its own last-written file")


def _validate_required_child_artifacts(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    artifacts: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    missing = sorted(_REQUIRED_CHILD_ARTIFACTS - set(artifacts))
    if missing:
        raise ValueError(f"child artifacts are missing required outputs: {missing}")
    records = {
        str(name): _artifact_record(Path(path), run_root=run_root)
        for name, path in sorted(artifacts.items())
    }
    for record in records.values():
        _assert_safe_bound_artifact(Path(record["path"]))
    grid = _validate_child_point_grid(
        manifest,
        index,
        Path(records["points"]["path"]),
    )
    quality = _read_json_object(
        Path(records["quality_report"]["path"]),
        label="quality report",
    )
    expected_quality = {
        "complete": True,
        "total_rows": 48,
        "valid_prediction_rows": 48,
        "decision_error_rows": 0,
        "invalid_prediction_rows": 0,
        "skipped_by_prediction_interval_rows": 0,
    }
    if any(quality.get(key) != value for key, value in expected_quality.items()):
        raise ValueError("child quality report does not describe a complete 48-row run")
    _validate_source_audit_artifacts(
        manifest,
        index,
        Path(records["source_audit_json"]["path"]),
        Path(records["source_audit_csv"]["path"]),
    )
    _validate_provenance_manifest(manifest, index, records, quality)
    return records, grid, quality


def _load_nonempty_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} must not be empty")
    return rows


def finalize_child_result(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    """Validate and atomically bind one authoritative pipeline result."""

    _offline_preflight(manifest)
    source = manifest["source_reports"][index]
    task_id = str(source["task_id"])
    paths = _task_paths(run_root, task_id)
    records, grid, quality = _validate_required_child_artifacts(
        manifest,
        run_root,
        index,
        artifacts,
    )
    ledger_path = paths["diagnostic"] / "request_attempts.jsonl"
    _assert_safe_bound_artifact(ledger_path)
    _load_nonempty_jsonl(ledger_path, label="request ledger")
    ledger_record = _artifact_record(ledger_path, run_root=run_root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_child_result",
        "manifest_sha256": manifest["manifest_sha256"],
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "run_id": manifest["run_id"],
        "index": int(index),
        "task_id": task_id,
        "source": {
            "report_path": source["report_path"],
            "report_sha256": source["report_sha256"],
            "trace_path": source["trace_path"],
            "trace_sha256": source["trace_sha256"],
            "point_keys": list(source["point_keys"]),
        },
        "grid": grid,
        "quality": {
            key: quality[key]
            for key in (
                "complete",
                "total_rows",
                "valid_prediction_rows",
                "decision_error_rows",
                "invalid_prediction_rows",
                "skipped_by_prediction_interval_rows",
            )
        },
        "artifacts": records,
        "request_ledger": ledger_record,
    }
    payload["output_binding_sha256"] = _canonical_sha256(
        {
            "artifacts": records,
            "request_ledger": ledger_record,
            "grid": grid,
        }
    )
    result_path = paths["diagnostic"] / "child_result.json"
    _atomic_write_json(result_path, payload, run_root=run_root)
    return payload


def validate_child_result(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
) -> dict[str, Any]:
    """Independently rehash and revalidate a completed child result."""

    source = manifest["source_reports"][index]
    result_path = _task_paths(run_root, str(source["task_id"]))[
        "diagnostic"
    ] / "child_result.json"
    result = _read_json_object(result_path, label="child result")
    expected_identity = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_child_result",
        "manifest_sha256": manifest["manifest_sha256"],
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "run_id": manifest["run_id"],
        "index": int(index),
        "task_id": str(source["task_id"]),
    }
    if any(result.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("child result identity does not match the frozen source")
    source_binding = result.get("source")
    if not isinstance(source_binding, dict) or source_binding != {
        "report_path": source["report_path"],
        "report_sha256": source["report_sha256"],
        "trace_path": source["trace_path"],
        "trace_sha256": source["trace_sha256"],
        "point_keys": list(source["point_keys"]),
    }:
        raise ValueError("child result source binding does not match the manifest")
    raw_artifacts = result.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("child result artifacts must be an object")
    artifact_paths: dict[str, Path] = {}
    for name, raw_record in raw_artifacts.items():
        if not isinstance(raw_record, dict):
            raise ValueError("child artifact records must be objects")
        current = _artifact_record(
            Path(str(raw_record.get("path") or "")),
            run_root=run_root,
        )
        if current != raw_record:
            raise ValueError(f"child artifact hash/size changed: {name}")
        artifact_paths[str(name)] = Path(current["path"])
    records, grid, quality = _validate_required_child_artifacts(
        manifest,
        run_root,
        index,
        artifact_paths,
    )
    if records != raw_artifacts or result.get("grid") != grid:
        raise ValueError("child result output/grid binding changed")
    expected_quality = {
        key: quality[key]
        for key in (
            "complete",
            "total_rows",
            "valid_prediction_rows",
            "decision_error_rows",
            "invalid_prediction_rows",
            "skipped_by_prediction_interval_rows",
        )
    }
    if result.get("quality") != expected_quality:
        raise ValueError("child result quality binding changed")
    ledger = result.get("request_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("child result request ledger binding is missing")
    current_ledger = _artifact_record(
        Path(str(ledger.get("path") or "")),
        run_root=run_root,
    )
    if current_ledger != ledger:
        raise ValueError("child request ledger hash/size changed")
    _load_nonempty_jsonl(Path(current_ledger["path"]), label="request ledger")
    expected_output_hash = _canonical_sha256(
        {
            "artifacts": records,
            "request_ledger": current_ledger,
            "grid": grid,
        }
    )
    if result.get("output_binding_sha256") != expected_output_hash:
        raise ValueError("child result output binding SHA-256 changed")
    return result


def _implementation_files() -> list[dict[str, str]]:
    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[5]
    paths = [
        module_path,
        module_path.with_name("predict_only_pipeline.py"),
        module_path.with_name("offline_prediction.py"),
        module_path.with_name("run_sparse_prediction_horizons.py"),
        module_path.with_name("provenance.py"),
        module_path.parents[1] / "safepred_guard" / "policies.py",
        repository_root / "SafePred" / "models" / "prompts.py",
        repository_root / "SafePred" / "models" / "world_model.py",
        repository_root / "SafePred" / "utils" / "harnessaudit_policy.py",
        repository_root / "SafePred" / "utils" / "llm_client.py",
        repository_root / "experiments" / "plot_results.py",
    ]
    return [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.resolve().relative_to(repository_root)),
            "sha256": _file_sha256(path.resolve()),
        }
        for path in paths
    ]


def build_frozen_manifest(
    partial_audit_path: Path,
    task_ids: list[str],
    model_id: str,
    model_revision_label: str,
    *,
    reasoning_mode: str | None = None,
    execution_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest entirely from frozen local artifacts."""

    from multi_agent.safepred_experiments.offline_prediction import (
        build_offline_prediction_steps,
    )
    from multi_agent.safepred_experiments.predict_only_pipeline import (
        PointSelectionConfig,
        _load_jsonl_strict,
        _resolve_trace_file,
        select_common_cohort_prediction_points,
    )
    from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
        _ground_truth_labels,
        _trace_access_decision_details,
    )

    if not task_ids:
        raise ValueError("source_reports must not be empty")
    normalized_model_id = str(model_id)
    _study_id_for_model(normalized_model_id)
    normalized_reasoning_mode = _validated_reasoning_mode(
        reasoning_mode,
        model_id=normalized_model_id,
    )
    execution_binding: dict[str, str] | None = None
    if execution_provenance_path is not None:
        if normalized_model_id != _AGENTWORLD_MODEL_ID:
            raise ValueError(
                "execution provenance binding is AgentWorld-only"
            )
        execution_record = _read_json_object(
            Path(execution_provenance_path),
            label="execution provenance",
        )
        record_sha256 = str(execution_record.get("record_sha256") or "")
        expected_record_sha256 = _canonical_sha256(
            {
                key: value
                for key, value in execution_record.items()
                if key != "record_sha256"
            }
        )
        if (
            execution_record.get("artifact_type")
            != "ha_agentworld_execution_provenance"
            or record_sha256 != expected_record_sha256
        ):
            raise ValueError("AgentWorld execution provenance is invalid")
        execution_binding = {
            "artifact_type": "ha_agentworld_execution_provenance",
            "record_sha256": record_sha256,
        }
    if (
        normalized_model_id.lower().startswith("deepseek-v4")
        and normalized_reasoning_mode != "disabled"
    ):
        raise ValueError(
            "new deepseek-v4 manifests require reasoning_mode=disabled"
        )

    audit_path = Path(partial_audit_path).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("partial audit must be a JSON object")
    if audit.get("passed") is not True:
        raise ValueError("partial audit must have passed=true")
    eligible_report_paths = audit.get("eligible_report_paths")
    if not isinstance(eligible_report_paths, list):
        raise ValueError("partial audit eligible_report_paths must be a list")
    eligible_paths = {
        str(Path(path).expanduser().resolve()): Path(path).expanduser().resolve()
        for path in eligible_report_paths
    }
    source_audit = audit.get("source_audit")
    source_audit_rows = (
        source_audit.get("reports")
        if isinstance(source_audit, dict)
        else None
    )
    if not isinstance(source_audit_rows, list):
        raise ValueError("partial audit must contain source audit reports")
    audit_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in source_audit_rows:
        if isinstance(row, dict):
            audit_rows_by_task.setdefault(str(row.get("task_id") or ""), []).append(row)

    normalized_task_ids = [Path(str(task_id)).stem for task_id in task_ids]
    if any(not task_id for task_id in normalized_task_ids):
        raise ValueError("task_ids must not contain empty values")
    if len(set(normalized_task_ids)) != len(normalized_task_ids):
        raise ValueError("task_ids must be unique")

    source_records: list[dict[str, Any]] = []
    all_point_keys: list[str] = []
    label_rows: list[dict[str, Any]] = []
    for task_id in normalized_task_ids:
        candidate_rows = [
            row
            for row in audit_rows_by_task.get(task_id, [])
            if str(Path(str(row.get("report_path") or "")).expanduser().resolve())
            in eligible_paths
        ]
        if len(candidate_rows) != 1:
            raise ValueError(
                f"task {task_id} must resolve to exactly one eligible source report"
            )
        audit_row = candidate_rows[0]
        report_path = Path(str(audit_row["report_path"])).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"source report must be a JSON object: {report_path}")
        if str(report.get("task_id") or "") != task_id:
            raise ValueError(f"source report task identity mismatch: {report_path}")
        trace_path = _resolve_trace_file(report.get("trace_file"), report_path).resolve()
        events = _load_jsonl_strict(trace_path)
        steps = build_offline_prediction_steps(events)
        points = select_common_cohort_prediction_points(
            steps,
            PointSelectionConfig(scheme="middle_pair", count=2),
            max_horizon=max(_HORIZONS),
        )
        details = _trace_access_decision_details(trace_path)
        frozen_points: list[dict[str, Any]] = []
        for point in points:
            sequence_num = int(point.step.action_sequence_num)
            point_key = f"{task_id}:{sequence_num}"
            exact_step_ground_truth: list[dict[str, Any]] = []
            for horizon in _HORIZONS:
                labels = _ground_truth_labels(
                    details,
                    action_sequence_num=sequence_num,
                    horizon=horizon,
                )
                label = {
                    "horizon": horizon,
                    "available": labels["exact_step_available"],
                    "violated": labels["exact_step_violated"],
                    "violation_codes": labels["exact_step_codes"],
                }
                exact_step_ground_truth.append(label)
                label_rows.append(label)
            frozen_points.append(
                {
                    "point_num": int(point.point_num),
                    "sequence_num": sequence_num,
                    "point_key": point_key,
                    "prefix_action_count": len(point.step.trace_prefix_actions),
                    "remaining_action_count": int(point.step.remaining_action_count),
                    "exact_step_ground_truth": exact_step_ground_truth,
                }
            )
            all_point_keys.append(point_key)

        source_point_keys = [point["point_key"] for point in frozen_points]
        source_records.append(
            {
                "report_path": str(report_path),
                "report_sha256": _file_sha256(report_path),
                "task_id": task_id,
                "run_id": str(report.get("run_id") or ""),
                "trace_path": str(trace_path),
                "trace_sha256": _file_sha256(trace_path),
                "source_action_count": len(steps),
                "resource_semantics": str(
                    audit_row.get("resource_semantics_mode") or ""
                ),
                "source_audit_valid": audit_row.get("is_clean_replayable") is True,
                "resource_semantics_provenance_valid": (
                    audit_row.get("resource_semantics_provenance_valid") is True
                ),
                "point_keys": source_point_keys,
                "selected_sequence_nums": [
                    point["sequence_num"] for point in frozen_points
                ],
                "points": frozen_points,
            }
        )

    support_census = []
    for horizon in _HORIZONS:
        horizon_labels = [
            label for label in label_rows if label["horizon"] == horizon
        ]
        positive = sum(label["violated"] is True for label in horizon_labels)
        negative = sum(label["violated"] is False for label in horizon_labels)
        support_census.append(
            {
                "horizon": horizon,
                "positive": positive,
                "negative": negative,
                "available": sum(
                    label["available"] is True for label in horizon_labels
                ),
            }
        )
    violation_code_support = sorted(
        {
            str(code)
            for label in label_rows
            if label["violated"] is True
            for code in label["violation_codes"]
        }
    )
    implementation_files = _implementation_files()
    partial_audit_sha256 = _file_sha256(audit_path)
    model_endpoint = _validated_model_endpoint(
        os.environ.get("WM_BASE_URL") or _DEFAULT_MODEL_ENDPOINT
    )
    if normalized_model_id == _AGENTWORLD_MODEL_ID:
        model_endpoint = _validated_agentworld_endpoint(model_endpoint)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_frozen_manifest",
        "study_id": _study_id_for_model(normalized_model_id),
        "run_id": _deterministic_run_id(
            partial_audit_sha256=partial_audit_sha256,
            ordered_task_ids=normalized_task_ids,
            model_id=normalized_model_id,
            requested_revision_label=str(model_revision_label),
            reasoning_mode=normalized_reasoning_mode,
        ),
        "diagnostic_label": _DIAGNOSTIC_LABEL,
        "partial_audit": {
            "path": str(audit_path),
            "sha256": partial_audit_sha256,
            "cohort_digest": str(audit.get("cohort_digest") or ""),
        },
        "source_reports": source_records,
        "ordered_task_ids": normalized_task_ids,
        "point_selection": {
            "scheme": "middle_pair",
            "required_remaining_action_count": max(_HORIZONS),
            "point_keys": all_point_keys,
        },
        "support_census": support_census,
        "violation_code_support": violation_code_support,
        "model": {
            "id": normalized_model_id,
            **(
                {"provider": "local_vllm"}
                if normalized_model_id == _AGENTWORLD_MODEL_ID
                else {}
            ),
            "endpoint": model_endpoint,
            "requested_revision_label": str(model_revision_label),
            "seed": 0,
            **(
                {"reasoning_mode": normalized_reasoning_mode}
                if normalized_reasoning_mode is not None
                else {}
            ),
        },
        **(
            {"execution_provenance": execution_binding}
            if execution_binding is not None
            else {}
        ),
        "matrix": {
            "target": "exact_step",
            "horizons": _HORIZONS,
            "rollout_modes": _ROLLOUT_MODES,
            "policy_modes": _POLICY_MODES,
            "candidate_count": 1,
            "max_trace_events": 64,
            "tier3_enabled": False,
            "concurrency": 1,
        },
        "resource_semantics": _RESOURCE_SEMANTICS,
        "diagnostic": True,
        "confirmatory": False,
        "compositionality": "unannotated",
        "implementation_files": implementation_files,
        "implementation_manifest_sha256": _canonical_sha256(
            implementation_files
        ),
        "condition_grid": [
            {
                "horizon": horizon,
                "rollout_mode": rollout_mode,
                "policy_mode": policy_mode,
                "point_keys": all_point_keys,
            }
            for horizon in _HORIZONS
            for rollout_mode in _ROLLOUT_MODES
            for policy_mode in _POLICY_MODES
        ],
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    validate_frozen_manifest(manifest)
    return manifest


def _implementation_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    records = manifest.get("implementation_files")
    if not isinstance(records, list):
        raise ValueError("implementation_files must be a list")
    hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("implementation file records must be objects")
        relative_path = str(record.get("relative_path") or "")
        sha256 = str(record.get("sha256") or "")
        if (
            not relative_path
            or relative_path in hashes
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            raise ValueError("implementation file records must have unique hashes")
        hashes[relative_path] = sha256
    return hashes


def validate_model_comparison_shards(
    cohort_reference: dict[str, Any],
    implementation_reference: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that four candidate shards change only the model treatment."""

    if not isinstance(cohort_reference, dict) or not isinstance(
        implementation_reference, dict
    ):
        raise ValueError("comparison references must be manifest objects")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("model comparison requires exactly four candidate shards")
    reference_task_ids = cohort_reference.get("ordered_task_ids")
    reference_sources = cohort_reference.get("source_reports")
    if (
        not isinstance(reference_task_ids, list)
        or not isinstance(reference_sources, list)
        or len(reference_task_ids) != len(reference_sources)
        or not reference_task_ids
        or len(set(reference_task_ids)) != len(reference_task_ids)
    ):
        raise ValueError("cohort reference must contain a unique ordered cohort")
    if [
        str(source.get("task_id") or "")
        for source in reference_sources
        if isinstance(source, dict)
    ] != reference_task_ids:
        raise ValueError("cohort reference task/source order is inconsistent")

    reference_implementation = _implementation_hashes(
        implementation_reference
    )
    if _RUNNER_IMPLEMENTATION_PATH not in reference_implementation:
        raise ValueError("implementation reference is missing the resilient runner")
    scientific_implementation = {
        path: digest
        for path, digest in reference_implementation.items()
        if path != _RUNNER_IMPLEMENTATION_PATH
    }
    if not scientific_implementation:
        raise ValueError("scientific implementation reference must not be empty")

    reference_point_selection = cohort_reference.get("point_selection")
    reference_matrix = cohort_reference.get("matrix")
    reference_conditions = cohort_reference.get("condition_grid")
    if (
        not isinstance(reference_point_selection, dict)
        or not isinstance(reference_matrix, dict)
        or not isinstance(reference_conditions, list)
    ):
        raise ValueError("cohort reference experiment contract is incomplete")
    invariant_fields = (
        "partial_audit",
        "matrix",
        "resource_semantics",
        "diagnostic_label",
        "diagnostic",
        "confirmatory",
        "compositionality",
    )
    reference_model = cohort_reference.get("model")
    if not isinstance(reference_model, dict):
        raise ValueError("cohort reference model must be an object")
    reference_seed = reference_model.get("seed")

    aggregate_support: dict[int, dict[str, int]] = {}
    reference_support = cohort_reference.get("support_census")
    if not isinstance(reference_support, list):
        raise ValueError("cohort reference support census must be a list")
    for row in reference_support:
        if not isinstance(row, dict) or isinstance(row.get("horizon"), bool):
            raise ValueError("cohort reference support census is malformed")
        horizon = int(row["horizon"])
        aggregate_support[horizon] = {
            "positive": 0,
            "negative": 0,
            "available": 0,
        }

    candidate_model_binding: tuple[str, str, str, str | None] | None = None
    shard_counts: list[int] = []
    observed_task_ids: list[str] = []
    observed_violation_codes: set[str] = set()
    candidate_runner_hashes: list[str] = []
    for shard_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError("candidate shards must be manifest objects")
        expected_task_ids = reference_task_ids[shard_index::4]
        if candidate.get("ordered_task_ids") != expected_task_ids:
            raise ValueError(
                f"candidate {shard_index} is not the exact modulo shard"
            )
        expected_sources = reference_sources[shard_index::4]
        if not _json_exact(candidate.get("source_reports"), expected_sources):
            raise ValueError(
                f"candidate {shard_index} source record changed"
            )
        for field in invariant_fields:
            if not _json_exact(candidate.get(field), cohort_reference.get(field)):
                raise ValueError(
                    f"candidate {shard_index} changed invariant {field}"
                )

        candidate_model = candidate.get("model")
        model_keys = {"id", "endpoint", "requested_revision_label", "seed"}
        if (
            not isinstance(candidate_model, dict)
            or set(candidate_model) not in (model_keys, model_keys | {"reasoning_mode"})
            or candidate_model.get("seed") != reference_seed
        ):
            raise ValueError(f"candidate {shard_index} changed model seed")
        reasoning_mode = _validated_reasoning_mode(
            candidate_model.get("reasoning_mode"),
            model_id=str(candidate_model["id"]),
        )
        model_binding = (
            str(candidate_model["id"]),
            _validated_model_endpoint(candidate_model["endpoint"]),
            str(candidate_model["requested_revision_label"]),
            reasoning_mode,
        )
        if (
            model_binding[0] != _DEEPSEEK_MODEL_ID
            or _official_deepseek_root(model_binding[1]) is None
            or not model_binding[2].strip()
        ):
            raise ValueError(
                f"candidate {shard_index} has an invalid treatment model binding"
            )
        if candidate_model_binding is None:
            candidate_model_binding = model_binding
        elif model_binding != candidate_model_binding:
            raise ValueError("candidate shards do not share one model binding")

        expected_point_keys = [
            str(point_key)
            for source in expected_sources
            for point_key in source["point_keys"]
        ]
        candidate_point_selection = candidate.get("point_selection")
        expected_point_selection = {
            "scheme": reference_point_selection.get("scheme"),
            "required_remaining_action_count": reference_point_selection.get(
                "required_remaining_action_count"
            ),
            "point_keys": expected_point_keys,
        }
        if not _json_exact(
            candidate_point_selection,
            expected_point_selection,
        ):
            raise ValueError(
                f"candidate {shard_index} changed point selection"
            )
        expected_conditions = [
            {
                **{
                    key: value
                    for key, value in condition.items()
                    if key != "point_keys"
                },
                "point_keys": expected_point_keys,
            }
            for condition in reference_conditions
            if isinstance(condition, dict)
        ]
        if len(expected_conditions) != len(reference_conditions) or not _json_exact(
            candidate.get("condition_grid"),
            expected_conditions,
        ):
            raise ValueError(
                f"candidate {shard_index} changed condition grid"
            )

        candidate_implementation = _implementation_hashes(candidate)
        runner_hash = candidate_implementation.pop(
            _RUNNER_IMPLEMENTATION_PATH,
            "",
        )
        if not runner_hash or candidate_implementation != scientific_implementation:
            raise ValueError(
                f"candidate {shard_index} changed scientific implementation"
            )
        candidate_runner_hashes.append(runner_hash)

        support_rows = candidate.get("support_census")
        if not isinstance(support_rows, list):
            raise ValueError(
                f"candidate {shard_index} support census is malformed"
            )
        observed_horizons: set[int] = set()
        for row in support_rows:
            if not isinstance(row, dict) or isinstance(row.get("horizon"), bool):
                raise ValueError(
                    f"candidate {shard_index} support census is malformed"
                )
            horizon = int(row["horizon"])
            if horizon not in aggregate_support or horizon in observed_horizons:
                raise ValueError(
                    f"candidate {shard_index} support census is malformed"
                )
            observed_horizons.add(horizon)
            for field in ("positive", "negative", "available"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"candidate {shard_index} support census is malformed"
                    )
                aggregate_support[horizon][field] += value
        if observed_horizons != set(aggregate_support):
            raise ValueError(
                f"candidate {shard_index} support census is incomplete"
            )
        codes = candidate.get("violation_code_support")
        if not isinstance(codes, list) or any(
            not isinstance(code, str) for code in codes
        ):
            raise ValueError(
                f"candidate {shard_index} violation support is malformed"
            )
        observed_violation_codes.update(codes)
        shard_counts.append(len(expected_task_ids))
        observed_task_ids.extend(expected_task_ids)

    reconstructed = [
        task_id
        for index in range(len(reference_task_ids))
        for task_id in [candidates[index % 4]["ordered_task_ids"][index // 4]]
    ]
    if reconstructed != reference_task_ids or len(set(observed_task_ids)) != len(
        reference_task_ids
    ):
        raise ValueError("candidate modulo shards do not reconstruct the cohort")
    observed_support = [
        {"horizon": horizon, **aggregate_support[horizon]}
        for horizon in aggregate_support
    ]
    if not _json_exact(observed_support, reference_support):
        raise ValueError("candidate support census does not reconstruct the cohort")
    if sorted(observed_violation_codes) != cohort_reference.get(
        "violation_code_support"
    ):
        raise ValueError(
            "candidate violation support does not reconstruct the cohort"
        )

    return {
        "schema_version": 1,
        "artifact_type": "ha_predict_only_model_comparability_audit",
        "passed": True,
        "diagnostic": True,
        "requested_count": len(reference_task_ids),
        "shard_count": 4,
        "shard_counts": shard_counts,
        "reference_run_id": str(cohort_reference.get("run_id") or ""),
        "implementation_reference_run_id": str(
            implementation_reference.get("run_id") or ""
        ),
        "candidate_run_ids": [
            str(candidate.get("run_id") or "") for candidate in candidates
        ],
        "model_binding": {
            "id": candidate_model_binding[0],
            "endpoint": candidate_model_binding[1],
            "requested_revision_label": candidate_model_binding[2],
            "seed": reference_seed,
            **(
                {"reasoning_mode": candidate_model_binding[3]}
                if candidate_model_binding[3] is not None
                else {}
            ),
        },
        "scientific_implementation_sha256": _canonical_sha256(
            scientific_implementation
        ),
        "candidate_runner_sha256": candidate_runner_hashes,
        "execution_contract": {
            "parent_shards": 4,
            "per_shard_manifest_concurrency": 1,
            "child_workers": 1,
            "max_attempts": 6,
            "retry_delay_seconds": 15,
            "retry_backoff": 2,
            "terminal_failure": "persist_skip_and_continue",
        },
        "allowed_differences": [
            "study_id",
            "run_id",
            "manifest_sha256",
            "implementation_manifest_sha256",
            "model.id",
            "model.endpoint",
            "model.requested_revision_label",
            "model.reasoning_mode",
            f"implementation_files.{_RUNNER_IMPLEMENTATION_PATH}",
            "shard-projected cohort counts and point keys",
        ],
    }


def _required_source_audit_row(
    partial_audit: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_audit = partial_audit.get("source_audit")
    rows = source_audit.get("reports") if isinstance(source_audit, dict) else None
    if not isinstance(rows, list):
        raise ValueError("partial audit must contain source audit reports")
    report_path = str(Path(source["report_path"]).resolve())
    matching = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(Path(str(row.get("report_path") or "")).resolve()) == report_path
    ]
    if len(matching) != 1:
        raise ValueError(
            f"source audit must contain exactly one row for {report_path}"
        )
    return matching[0]


def _validate_legacy_diagnostic_provenance(
    report: dict[str, Any],
    trace_start: dict[str, Any],
) -> None:
    report_provenance = report.get("actor_provenance")
    trace_provenance = trace_start.get("actor_provenance")
    if not isinstance(report_provenance, dict) or not isinstance(
        trace_provenance, dict
    ):
        raise ValueError("legacy source requires report and trace actor provenance")
    observed_modes = [
        report.get("resource_semantics_mode"),
        trace_start.get("resource_semantics_mode"),
        report_provenance.get("resource_semantics_mode"),
        trace_provenance.get("resource_semantics_mode"),
    ]
    if any(mode != _RESOURCE_SEMANTICS for mode in observed_modes):
        raise ValueError(
            "source provenance must consistently use legacy_official_v1"
        )
    if report_provenance != trace_provenance:
        raise ValueError("report and trace actor provenance must match")


def _offline_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    from multi_agent.safepred_experiments.offline_prediction import (
        build_offline_prediction_steps,
    )
    from multi_agent.safepred_experiments.predict_only_pipeline import (
        PointSelectionConfig,
        _load_jsonl_strict,
        _resolve_trace_file,
        select_common_cohort_prediction_points,
        validate_selected_prefix_budget,
    )
    from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
        _ground_truth_labels,
        _trace_access_decision_details,
    )

    validated = validate_frozen_manifest(manifest)
    implementation_files = manifest.get("implementation_files")
    if not isinstance(implementation_files, list):
        raise ValueError("implementation_files must be a list")
    if _canonical_sha256(implementation_files) != manifest.get(
        "implementation_manifest_sha256"
    ):
        raise ValueError("implementation manifest SHA-256 mismatch")
    manifest_without_hash = dict(manifest)
    observed_manifest_sha256 = manifest_without_hash.pop("manifest_sha256", "")
    if (
        not observed_manifest_sha256
        or _canonical_sha256(manifest_without_hash) != observed_manifest_sha256
    ):
        raise ValueError("manifest SHA-256 mismatch")
    if implementation_files != _implementation_files():
        raise ValueError(
            "implementation_files must exactly match the deterministic current set"
        )

    partial = manifest["partial_audit"]
    if not isinstance(partial, dict):
        raise ValueError("partial_audit must be an object")
    partial_path = Path(str(partial.get("path") or ""))
    if not partial_path.is_absolute():
        raise ValueError("partial audit path must be absolute")
    if _file_sha256(partial_path) != partial.get("sha256"):
        raise ValueError("partial audit SHA-256 mismatch")
    partial_payload = json.loads(partial_path.read_text(encoding="utf-8"))
    if not isinstance(partial_payload, dict):
        raise ValueError("partial audit must be a JSON object")
    if partial_payload.get("passed") is not True:
        raise ValueError("partial audit must have passed=true")
    if partial_payload.get("cohort_digest") != partial.get("cohort_digest"):
        raise ValueError("partial audit cohort digest mismatch")
    eligible_paths = {
        str(Path(str(path)).expanduser().resolve())
        for path in partial_payload.get("eligible_report_paths") or []
    }

    for record in implementation_files:
        if not isinstance(record, dict):
            raise ValueError("implementation file records must be objects")
        implementation_path = Path(str(record.get("path") or ""))
        if not implementation_path.is_absolute():
            raise ValueError("implementation file paths must be absolute")
        if _file_sha256(implementation_path) != record.get("sha256"):
            raise ValueError(
                f"implementation SHA-256 mismatch: {implementation_path}"
            )

    observed_support = {
        horizon: {"positive": 0, "negative": 0, "available": 0}
        for horizon in _HORIZONS
    }
    observed_codes: set[str] = set()
    points_by_task = []
    observed_point_keys: list[str] = []
    for source in manifest["source_reports"]:
        task_id = str(source["task_id"])
        report_path = Path(str(source["report_path"]))
        trace_path = Path(str(source["trace_path"]))
        if not report_path.is_absolute() or not trace_path.is_absolute():
            raise ValueError("source report and trace paths must be absolute")
        if str(report_path.resolve()) not in eligible_paths:
            raise ValueError(
                f"source report is not an exact eligible_report_paths member: {report_path}"
            )
        if _file_sha256(report_path) != source.get("report_sha256"):
            raise ValueError(f"report SHA-256 mismatch: {report_path}")
        if _file_sha256(trace_path) != source.get("trace_sha256"):
            raise ValueError(f"trace SHA-256 mismatch: {trace_path}")

        source_audit_row = _required_source_audit_row(partial_payload, source)
        if source.get("source_audit_valid") is not True or source_audit_row.get(
            "is_clean_replayable"
        ) is not True:
            raise ValueError(f"source audit is invalid: {report_path}")
        if (
            source.get("resource_semantics_provenance_valid") is not True
            or source_audit_row.get("resource_semantics_provenance_valid") is not True
        ):
            raise ValueError(
                f"source resource-semantics provenance is invalid: {report_path}"
            )
        if source_audit_row.get("source_report_sha256") not in (
            None,
            "",
            source.get("report_sha256"),
        ):
            raise ValueError(f"source audit report hash mismatch: {report_path}")
        if source_audit_row.get("source_trace_sha256") not in (
            None,
            "",
            source.get("trace_sha256"),
        ):
            raise ValueError(f"source audit trace hash mismatch: {trace_path}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"source report must be a JSON object: {report_path}")
        if str(report.get("task_id") or "") != task_id:
            raise ValueError(f"source report task identity mismatch: {report_path}")
        if str(report.get("run_id") or "") != str(source.get("run_id") or ""):
            raise ValueError(f"source report run identity mismatch: {report_path}")
        resolved_trace = _resolve_trace_file(
            report.get("trace_file"),
            report_path,
        ).resolve()
        if resolved_trace != trace_path.resolve():
            raise ValueError(f"source report trace identity mismatch: {report_path}")

        events = _load_jsonl_strict(trace_path)
        trace_starts = [
            event for event in events if event.get("type") == "trace_start"
        ]
        if len(trace_starts) != 1:
            raise ValueError(f"trace must contain exactly one trace_start: {trace_path}")
        trace_start = trace_starts[0]
        if (
            str(trace_start.get("task_id") or "") != task_id
            or str(trace_start.get("trace_id") or "") != str(source.get("run_id") or "")
        ):
            raise ValueError(f"source trace identity mismatch: {trace_path}")
        _validate_legacy_diagnostic_provenance(report, trace_start)

        steps = build_offline_prediction_steps(events)
        if len(steps) != int(source.get("source_action_count") or 0):
            raise ValueError(f"source action count mismatch: {trace_path}")
        if int(report.get("n_actions") or 0) != len(steps):
            raise ValueError(f"source report action count mismatch: {report_path}")
        if int(source_audit_row.get("action_count") or 0) != len(steps):
            raise ValueError(f"source audit action count mismatch: {report_path}")

        selected = select_common_cohort_prediction_points(
            steps,
            PointSelectionConfig(scheme="middle_pair", count=2),
            max_horizon=max(_HORIZONS),
        )
        points_by_task.append((task_id, selected))
        selected_sequences = [
            int(point.step.action_sequence_num) for point in selected
        ]
        if selected_sequences != source.get("selected_sequence_nums"):
            raise ValueError(f"frozen middle_pair sequences changed: {task_id}")
        selected_point_keys = [
            f"{task_id}:{sequence_num}" for sequence_num in selected_sequences
        ]
        if selected_point_keys != source.get("point_keys"):
            raise ValueError(f"frozen point keys changed: {task_id}")
        observed_point_keys.extend(selected_point_keys)

        frozen_points = source.get("points")
        details = _trace_access_decision_details(trace_path)
        for point, frozen_point in zip(selected, frozen_points, strict=True):
            observed_point = {
                "point_num": int(point.point_num),
                "sequence_num": int(point.step.action_sequence_num),
                "point_key": f"{task_id}:{int(point.step.action_sequence_num)}",
                "prefix_action_count": len(point.step.trace_prefix_actions),
                "remaining_action_count": int(point.step.remaining_action_count),
            }
            if any(
                frozen_point.get(field) != value
                for field, value in observed_point.items()
            ):
                raise ValueError(f"frozen point metadata changed: {task_id}")
            frozen_ground_truth = frozen_point.get("exact_step_ground_truth")
            if not isinstance(frozen_ground_truth, list):
                raise ValueError(f"frozen point ground truth is missing: {task_id}")
            observed_ground_truth = []
            for horizon in _HORIZONS:
                labels = _ground_truth_labels(
                    details,
                    action_sequence_num=observed_point["sequence_num"],
                    horizon=horizon,
                )
                label = {
                    "horizon": horizon,
                    "available": labels["exact_step_available"],
                    "violated": labels["exact_step_violated"],
                    "violation_codes": labels["exact_step_codes"],
                }
                observed_ground_truth.append(label)
                if label["available"] is True:
                    observed_support[horizon]["available"] += 1
                if label["violated"] is True:
                    observed_support[horizon]["positive"] += 1
                    observed_codes.update(label["violation_codes"])
                elif label["violated"] is False:
                    observed_support[horizon]["negative"] += 1
            if frozen_ground_truth != observed_ground_truth:
                raise ValueError(f"frozen exact-step ground truth changed: {task_id}")

    longest_prefix = validate_selected_prefix_budget(
        points_by_task,
        max_trace_events=int(manifest["matrix"]["max_trace_events"]),
    )
    if observed_point_keys != manifest["point_selection"]["point_keys"]:
        raise ValueError("common frozen point keys changed")
    observed_census = [
        {"horizon": horizon, **observed_support[horizon]}
        for horizon in _HORIZONS
    ]
    if observed_census != manifest.get("support_census"):
        raise ValueError("exact-step support census changed")
    if sorted(observed_codes) != manifest.get("violation_code_support"):
        raise ValueError("violation-code support changed")
    return {
        "requested_count": validated["requested_count"],
        "eligible_count": validated["eligible_count"],
        "longest_selected_prefix": longest_prefix,
        "offline_checks_passed": True,
    }


def enumerate_aifast_models() -> dict[str, Any]:
    """Enumerate models via the bundled key-safe helper."""

    if not str(os.environ.get("WM_API_KEY") or "").strip():
        raise ValueError("Set WM_API_KEY in the environment before model enumeration")
    completed = subprocess.run(
        [
            sys.executable,
            str(_AIFAST_ENUMERATION_SCRIPT),
            "--api-key-env",
            "WM_API_KEY",
            "--base-url",
            _AIFAST_ROOT,
            "models",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    payload: Any = None
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    return {
        "exit_status": int(completed.returncode),
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
    }


def enumerate_local_vllm_models(endpoint: str) -> dict[str, Any]:
    """Enumerate one loopback vLLM service without credential material."""

    base = _validated_agentworld_endpoint(endpoint)
    request = Request(
        f"{base}/models",
        headers={"User-Agent": "world-model-agent-safety/1.0"},
        method="GET",
    )
    response_bytes = b""
    exit_status = 1
    payload: Any = None
    try:
        proxy_free_opener = build_opener(ProxyHandler({}))
        with proxy_free_opener.open(request, timeout=30) as response:
            response_bytes = response.read()
            exit_status = 0 if int(response.status) == 200 else int(response.status)
    except HTTPError as exc:
        exit_status = int(exc.code)
        response_bytes = exc.read()
    except URLError:
        exit_status = 1
    if response_bytes:
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
    return {
        "exit_status": exit_status,
        "payload": payload,
        "payload_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }


def _official_deepseek_root(endpoint: Any) -> str | None:
    parsed = urlsplit(_validated_model_endpoint(endpoint))
    if str(parsed.hostname or "").lower() != "api.deepseek.com":
        return None
    return "https://api.deepseek.com"


def _chat_completions_url(endpoint: Any) -> str:
    base = _validated_model_endpoint(endpoint)
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def probe_openai_compatible_model(
    endpoint: str,
    model_id: str,
) -> dict[str, Any]:
    """Probe one exact official model without enumerating another provider."""

    api_key = str(os.environ.get("WM_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Set WM_API_KEY in the environment before model probing")
    request_payload: dict[str, Any] = {
        "model": str(model_id),
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly OK.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    if str(model_id) == _DEEPSEEK_MODEL_ID:
        request_payload["thinking"] = {"type": "disabled"}
    request = Request(
        _chat_completions_url(endpoint),
        data=json.dumps(
            request_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "world-model-agent-safety/1.0",
        },
        method="POST",
    )
    response_bytes = b""
    exit_status = 1
    payload: Any = None
    try:
        with urlopen(request, timeout=90) as response:
            response_bytes = response.read()
            exit_status = 0 if int(response.status) == 200 else int(response.status)
    except HTTPError as exc:
        exit_status = int(exc.code)
        response_bytes = exc.read()
    except URLError:
        exit_status = 1
    if response_bytes:
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
    return {
        "exit_status": exit_status,
        "payload": payload,
        "payload_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }


def _normalize_model_enumeration(result: Any) -> tuple[int, Any, str]:
    if not isinstance(result, dict):
        raise ValueError("model enumeration must return a JSON object")
    if "payload" in result:
        exit_status = int(result.get("exit_status", 0))
        payload = result.get("payload")
        payload_sha256 = str(result.get("payload_sha256") or "")
    else:
        exit_status = int(result.get("exit_status", 0))
        payload = {
            key: value for key, value in result.items() if key != "exit_status"
        }
        payload_sha256 = ""
    if not payload_sha256:
        payload_sha256 = _canonical_sha256(payload)
    return exit_status, payload, payload_sha256


def preflight_frozen_manifest(
    manifest: dict[str, Any],
    run_root: Path,
    *,
    model_enumerator: Callable[[], Any] | None = None,
    model_prober: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Complete offline checks before checking the exact provider-bound model."""

    offline_summary = _offline_preflight(manifest)
    model_id = str(manifest["model"]["id"])
    endpoint = str(manifest["model"]["endpoint"])
    official_deepseek_root = _official_deepseek_root(endpoint)
    if model_id == _AGENTWORLD_MODEL_ID:
        enumerate_models = model_enumerator or (
            lambda: enumerate_local_vllm_models(endpoint)
        )
        exit_status, payload, payload_sha256 = _normalize_model_enumeration(
            enumerate_models()
        )
        model_rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(model_rows, list):
            model_rows = []
        model_ids = [
            str(row.get("id") or "")
            for row in model_rows
            if isinstance(row, dict) and str(row.get("id") or "")
        ]
        provider_root = _validated_agentworld_endpoint(endpoint)
    elif official_deepseek_root is not None:
        probe_model = model_prober or probe_openai_compatible_model
        exit_status, payload, payload_sha256 = _normalize_model_enumeration(
            probe_model(endpoint, model_id)
        )
        returned_model = (
            str(payload.get("model") or "")
            if isinstance(payload, dict)
            else ""
        )
        model_ids = [returned_model] if returned_model else []
        provider_root = official_deepseek_root
    else:
        enumerate_models = model_enumerator or enumerate_aifast_models
        exit_status, payload, payload_sha256 = _normalize_model_enumeration(
            enumerate_models()
        )
        model_rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(model_rows, list):
            model_rows = []
        model_ids = [
            str(row.get("id") or "")
            for row in model_rows
            if isinstance(row, dict) and str(row.get("id") or "")
        ]
        provider_root = _AIFAST_ROOT
    selected_id = model_id if model_id in model_ids else ""
    enumeration_record = {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "exit_status": exit_status,
        "endpoint": endpoint,
        "root": provider_root,
        "selected_id": selected_id,
        "model_count": len(model_ids),
        "payload_sha256": payload_sha256,
    }
    preflight = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_preflight",
        "passed": exit_status == 0 and selected_id == model_id,
        "manifest_binding": {
            "study_id": manifest["study_id"],
            "run_id": manifest["run_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "implementation_manifest_sha256": manifest[
                "implementation_manifest_sha256"
            ],
            "model_id": manifest["model"]["id"],
            "ordered_task_ids": list(manifest["ordered_task_ids"]),
        },
        "offline": offline_summary,
        "model_enumeration": enumeration_record,
    }
    _atomic_write_json(
        Path(run_root) / "preflight.json",
        preflight,
        run_root=run_root,
    )
    if exit_status != 0:
        raise ValueError(
            f"model availability check failed with exit status {exit_status}"
        )
    if not selected_id:
        raise ValueError(f"required model {model_id} was not enumerated")
    return preflight


def _validate_bound_preflight(
    manifest: dict[str, Any],
    run_root: Path,
    offline_summary: dict[str, Any],
) -> dict[str, Any]:
    preflight_path = Path(run_root) / "preflight.json"
    _safe_run_path(preflight_path, run_root=run_root)
    preflight = _read_json_object(preflight_path, label="preflight")
    expected_top_fields = {
        "schema_version",
        "artifact_type",
        "passed",
        "manifest_binding",
        "offline",
        "model_enumeration",
    }
    if set(preflight) != expected_top_fields:
        raise ValueError("preflight has missing or unknown fields")
    if (
        type(preflight.get("schema_version")) is not int
        or preflight.get("schema_version") != 1
        or preflight.get("artifact_type")
        != "ha_resilient_predict_only_preflight"
        or preflight.get("passed") is not True
    ):
        raise ValueError("run requires a safe preflight with passed=true")
    expected_binding = {
        "study_id": manifest["study_id"],
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        "model_id": manifest["model"]["id"],
        "ordered_task_ids": list(manifest["ordered_task_ids"]),
    }
    binding = preflight.get("manifest_binding")
    if (
        not isinstance(binding, dict)
        or set(binding) != set(expected_binding)
        or not _json_exact(binding, expected_binding)
    ):
        raise ValueError("preflight is not bound to the current frozen manifest")
    model_record = preflight.get("model_enumeration")
    if manifest["model"]["id"] == _AGENTWORLD_MODEL_ID:
        expected_provider_root = _validated_agentworld_endpoint(
            manifest["model"]["endpoint"]
        )
    else:
        expected_provider_root = (
            _official_deepseek_root(manifest["model"]["endpoint"])
            or _AIFAST_ROOT
        )
    expected_model_fields = {
        "timestamp_utc",
        "exit_status",
        "endpoint",
        "root",
        "selected_id",
        "model_count",
        "payload_sha256",
    }
    if (
        not isinstance(model_record, dict)
        or set(model_record) != expected_model_fields
        or not _is_timestamp(model_record.get("timestamp_utc"))
        or isinstance(model_record.get("exit_status"), bool)
        or not isinstance(model_record.get("exit_status"), int)
        or model_record.get("exit_status") != 0
        or model_record.get("selected_id") != manifest["model"]["id"]
        or model_record.get("endpoint") != manifest["model"]["endpoint"]
        or model_record.get("root") != expected_provider_root
        or isinstance(model_record.get("model_count"), bool)
        or not isinstance(model_record.get("model_count"), int)
        or model_record.get("model_count") < 1
        or not isinstance(model_record.get("payload_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(model_record.get("payload_sha256") or ""),
        )
    ):
        raise ValueError("preflight did not select the exact frozen model")
    offline = preflight.get("offline")
    if (
        not isinstance(offline, dict)
        or set(offline) != set(offline_summary)
        or not _json_exact(offline, offline_summary)
    ):
        raise ValueError("preflight offline summary does not match current validation")
    return preflight


def _append_status_event(
    path: Path,
    event: dict[str, Any],
    *,
    run_root: Path,
) -> None:
    destination = _safe_run_path(path, run_root=run_root)
    _ensure_safe_directory(destination.parent, run_root=run_root)
    _safe_run_path(destination, run_root=run_root)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                event,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _source_event_identity(
    manifest: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    source = manifest["source_reports"][index]
    return {
        "index": int(index),
        "task_id": str(source["task_id"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "run_id": manifest["run_id"],
        "source_report_path": source["report_path"],
        "source_report_sha256": source["report_sha256"],
        "source_trace_path": source["trace_path"],
        "source_trace_sha256": source["trace_sha256"],
    }


def _progress_payload(
    manifest: dict[str, Any],
    *,
    passed_task_ids: list[str],
    skipped_task_ids: list[str],
    running: int,
) -> dict[str, Any]:
    requested = len(manifest["source_reports"])
    processed = len(passed_task_ids) + len(skipped_task_ids)
    return {
        "requested": requested,
        "processed": processed,
        "passed": len(passed_task_ids),
        "skipped": len(skipped_task_ids),
        "pending": requested - processed - int(running),
        "running": int(running),
        "passed_task_ids": list(passed_task_ids),
        "skipped_task_ids": list(skipped_task_ids),
    }


def _inventory_partial_artifacts(
    run_root: Path,
    task_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = _task_paths(run_root, task_id)
    for kind, root in paths.items():
        _safe_run_path(root, run_root=run_root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if (
                kind == "diagnostic"
                and path.name == "manifest.json"
                and path.parent == root
            ):
                continue
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(mode):
                continue
            _safe_run_path(path, run_root=run_root)
            records.append(_artifact_record(path, run_root=run_root))
    return sorted(records, key=lambda record: str(record["path"]))


def _sanitized_exception(exc: BaseException) -> tuple[str, str]:
    error_type = re.sub(r"[^A-Za-z0-9_.-]+", "_", type(exc).__name__)[:200]
    message = _sanitize_persisted_text(str(exc))
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = message[:2000].strip()
    if not message:
        message = f"{error_type} without a message"
    return error_type, message


def _write_skip_diagnostic(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    *,
    stage: str,
    exception: BaseException,
    exit_code: int | None,
    elapsed_seconds: float,
    log_path: Path,
) -> tuple[dict[str, Any], Path]:
    source = manifest["source_reports"][index]
    task_id = str(source["task_id"])
    error_type, message = _sanitized_exception(exception)
    log_record = _artifact_record(log_path, run_root=run_root)
    diagnostic = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_skip_diagnostic",
        "manifest_sha256": manifest["manifest_sha256"],
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        **_source_event_identity(manifest, index),
        "stage": str(stage),
        "exception_type": error_type,
        "exception_message": message,
        "exit_code": exit_code,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "log": log_record,
        "inventory": _inventory_partial_artifacts(run_root, task_id),
    }
    diagnostic_path = _task_paths(run_root, task_id)["diagnostic"] / "manifest.json"
    _atomic_write_json(diagnostic_path, diagnostic, run_root=run_root)
    return diagnostic, diagnostic_path


def _default_child_command(
    manifest_path: Path,
    run_root: Path,
    index: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "multi_agent.safepred_experiments.ha_resilient_predict_only",
        "child",
        "--manifest",
        str(Path(manifest_path).resolve()),
        "--run-root",
        str(Path(run_root).resolve()),
        "--index",
        str(int(index)),
    ]


def run_frozen_child(
    manifest_path: Path,
    run_root: Path,
    index: int,
    *,
    pipeline_runner: Callable[..., dict[str, Path]] | None = None,
) -> dict[str, Any]:
    """Run exactly one frozen source through the authoritative HA pipeline."""

    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    manifest_file = _safe_run_path(manifest_path, run_root=root)
    manifest = _read_json_object(manifest_file, label="manifest")
    validate_frozen_manifest(manifest)
    model_id = str(manifest["model"]["id"])
    source_reports = manifest["source_reports"]
    if int(index) < 0 or int(index) >= len(source_reports):
        raise ValueError(f"source index is out of range: {index}")
    source = source_reports[int(index)]
    task_id = str(source["task_id"])
    paths = _task_paths(root, task_id)
    for path in paths.values():
        _ensure_safe_directory(path, run_root=root)
    analysis_dir = paths["diagnostic"] / "analysis"
    figure_dir = paths["diagnostic"] / "figures"
    _ensure_safe_directory(analysis_dir, run_root=root)
    _ensure_safe_directory(figure_dir, run_root=root)

    endpoint = _validated_model_endpoint(manifest["model"]["endpoint"])
    inherited_endpoint = str(os.environ.get("WM_BASE_URL") or "").strip().rstrip("/")
    if inherited_endpoint and _validated_model_endpoint(inherited_endpoint) != endpoint:
        raise ValueError("WM_BASE_URL does not match the frozen model endpoint")
    api_key = str(os.environ.get("WM_API_KEY") or "")
    child_environment = {
        "SAFEPRED_LLM_MAX_ATTEMPTS": "6",
        "SAFEPRED_LLM_RETRY_DELAY_SECONDS": "15",
        "SAFEPRED_LLM_RETRY_BACKOFF": "2",
        "SAFEPRED_REQUEST_LEDGER_PATH": str(
            paths["diagnostic"] / "request_attempts.jsonl"
        ),
        "SAFEPRED_TRAIN_POLICY_CACHE_ROOT": str(
            paths["diagnostic"] / "policy_cache"
        ),
        "SAFEPRED_POLICY_EXTRACTION_DEBUG_DIR": str(
            paths["diagnostic"] / "policy_extraction_debug"
        ),
        "SAFEPRED_RUN_DIR": str(paths["diagnostic"]),
        "SAFEPRED_WM_SEED": "0",
        "SAFEPRED_POLICY_EXTRACTION_SEED": "0",
        "SAFEPRED_RESOURCE_SEMANTICS_MODE": str(
            manifest["resource_semantics"]
        ),
        "SAFEPRED_MODEL_NAME": model_id,
        "SAFEPRED_POLICY_EXTRACTION_MODEL": model_id,
        "WM_BASE_URL": endpoint,
        "CUSTOM_API_URL": endpoint,
        "OPENAI_API_URL": endpoint,
        "OPENAI_BASE_URL": endpoint,
    }
    if api_key:
        child_environment.update(
            {
                "CUSTOM_API_KEY": api_key,
                "OPENAI_API_KEY": api_key,
            }
        )
    original_environment = os.environ.copy()
    try:
        os.environ.update(child_environment)
        _offline_preflight(manifest)

        # Import the pipeline entry/types only after the complete child-local
        # environment is installed. Importing this orchestrator never
        # constructs a client, and the parent never calls this function
        # in-process.
        from multi_agent.safepred_experiments.predict_only_pipeline import (
            ArtifactPaths,
            ModelAPIConfig,
            PointSelectionConfig,
            _default_base_config,
            run_pipeline,
        )

        runner = pipeline_runner or run_pipeline
        artifacts = runner(
            tasks=[],
            source_reports=[Path(str(source["report_path"]))],
            policy_modes=list(manifest["matrix"]["policy_modes"]),
            rollout_modes=list(manifest["matrix"]["rollout_modes"]),
            horizons=list(manifest["matrix"]["horizons"]),
            point_config=PointSelectionConfig(
                scheme="explicit",
                count=2,
                explicit_sequences=[
                    int(value) for value in source["selected_sequence_nums"]
                ],
            ),
            model=model_id,
            judge_model=model_id,
            max_turns=30,
            skip_clean_run=True,
            skip_judge=True,
            base_config=_default_base_config(),
            paths=ArtifactPaths(
                output_root=paths["result"],
                trace_root=paths["trace"],
                analysis_dir=analysis_dir,
                figure_dir=figure_dir,
            ),
            candidate_count=1,
            max_trace_events=64,
            skip_existing=False,
            run_tier3_judge=False,
            tier3_model=model_id,
            model_api=ModelAPIConfig(
                provider="custom",
                api_key=api_key or None,
                api_url=endpoint,
                model=model_id,
                model_revision=str(
                    manifest["model"]["requested_revision_label"]
                ),
                policy_extraction_model=model_id,
            ),
            clean_agent_api=None,
            fail_on_prediction_errors=True,
            fail_fast_decision_errors=True,
            workers=_frozen_child_workers(),
            gt_estimand="exact_step",
            experiment_seed=0,
            actor_trace_seed=None,
            wm_seed=0,
            formal=False,
        )
        if not isinstance(artifacts, dict):
            raise ValueError(
                "authoritative run_pipeline must return an artifact mapping"
            )
        return finalize_child_result(
            manifest,
            root,
            int(index),
            {str(name): Path(path) for name, path in artifacts.items()},
        )
    finally:
        os.environ.clear()
        os.environ.update(original_environment)


class InterruptedTrace(RuntimeError):
    """A persisted TRACE_START that never reached a terminal record."""


def _read_status_events(path: Path, *, run_root: Path) -> list[dict[str, Any]]:
    status_path = _safe_run_path(path, run_root=run_root)
    if not status_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with status_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"status history line {line_number} is blank")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"status history line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"status history line {line_number} must be a JSON object"
                )
            events.append(event)
    return events


def _validate_event_identity(
    manifest: dict[str, Any],
    index: int,
    event: dict[str, Any],
) -> None:
    expected = _source_event_identity(manifest, index)
    if any(
        not _json_exact(event.get(key), value)
        for key, value in expected.items()
    ):
        raise ValueError(
            f"status event does not match ordered frozen task/index {index}"
        )


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_status_event_schema(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    event: dict[str, Any],
    event_name: str,
) -> None:
    common = {
        "event",
        "timestamp_utc",
        *set(_source_event_identity(manifest, index)),
    }
    event_fields = {
        "TRACE_START": common | {"log_path"},
        "TRACE_PASS": common
        | {"elapsed_seconds", "exit_code", "log", "child_result"},
        "TRACE_SKIP": common
        | {
            "stage",
            "exception_type",
            "exception_message",
            "elapsed_seconds",
            "exit_code",
            "log",
            "diagnostic",
        },
    }
    expected_fields = event_fields.get(event_name)
    if expected_fields is None or set(event) != expected_fields:
        raise ValueError(f"{event_name or 'status event'} has missing or unknown fields")
    if (
        not _is_timestamp(event.get("timestamp_utc"))
    ):
        raise ValueError(f"{event_name} timestamp_utc must be a nonempty string")
    if event_name == "TRACE_START":
        source = manifest["source_reports"][index]
        expected_log = (
            _lexical_absolute(run_root)
            / "logs"
            / f"{index:04d}__{_safe_task_name(str(source['task_id']))}.log"
        )
        if event.get("log_path") != str(expected_log):
            raise ValueError("TRACE_START log_path is not the deterministic task log")
        _safe_run_path(expected_log, run_root=run_root)
        return
    if not _is_nonnegative_number(event.get("elapsed_seconds")):
        raise ValueError(f"{event_name} elapsed_seconds must be nonnegative")
    exit_code = event.get("exit_code")
    if isinstance(exit_code, bool) or (
        exit_code is not None and not isinstance(exit_code, int)
    ):
        raise ValueError(f"{event_name} exit_code must be an integer or null")
    if event_name == "TRACE_PASS" and exit_code != 0:
        raise ValueError("TRACE_PASS must have exit_code=0")
    if event_name == "TRACE_SKIP" and any(
        not isinstance(event.get(field), str) or not event.get(field)
        for field in ("stage", "exception_type", "exception_message")
    ):
        raise ValueError("TRACE_SKIP diagnostic strings must be nonempty")


def _verify_bound_record(
    raw_record: Any,
    *,
    run_root: Path,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_record, dict):
        raise ValueError(f"{label} record must be an object")
    if set(raw_record) != {"path", "size", "sha256"}:
        raise ValueError(f"{label} record has missing or unknown fields")
    if (
        not isinstance(raw_record.get("path"), str)
        or isinstance(raw_record.get("size"), bool)
        or not isinstance(raw_record.get("size"), int)
        or raw_record["size"] < 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(raw_record.get("sha256") or ""))
    ):
        raise ValueError(f"{label} record has invalid scalar fields")
    current = _artifact_record(
        Path(str(raw_record.get("path") or "")),
        run_root=run_root,
    )
    if expected_path is not None and Path(current["path"]) != _lexical_absolute(
        expected_path
    ):
        raise ValueError(f"{label} path changed")
    if current != raw_record:
        raise ValueError(f"{label} hash/size changed")
    return current


def _verify_pass_terminal(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    start: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    source = manifest["source_reports"][index]
    task_id = str(source["task_id"])
    expected_log = (
        _lexical_absolute(run_root)
        / "logs"
        / f"{index:04d}__{_safe_task_name(task_id)}.log"
    )
    _verify_bound_record(
        terminal.get("log"),
        run_root=run_root,
        label="pass log",
        expected_path=expected_log,
    )
    if terminal.get("exit_code") != 0:
        raise ValueError("TRACE_PASS must have exit_code=0")
    result_path = (
        _task_paths(run_root, task_id)["diagnostic"] / "child_result.json"
    )
    _verify_bound_record(
        terminal.get("child_result"),
        run_root=run_root,
        label="child result",
        expected_path=result_path,
    )
    validate_child_result(manifest, run_root, index)


def _verify_skip_terminal(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    start: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    source = manifest["source_reports"][index]
    task_id = str(source["task_id"])
    diagnostic_path = (
        _task_paths(run_root, task_id)["diagnostic"] / "manifest.json"
    )
    _verify_bound_record(
        terminal.get("diagnostic"),
        run_root=run_root,
        label="skip diagnostic",
        expected_path=diagnostic_path,
    )
    diagnostic = _read_json_object(diagnostic_path, label="skip diagnostic")
    expected_identity = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_skip_diagnostic",
        "implementation_manifest_sha256": manifest[
            "implementation_manifest_sha256"
        ],
        **_source_event_identity(manifest, index),
    }
    expected_diagnostic_fields = set(expected_identity) | {
        "manifest_sha256",
        "stage",
        "exception_type",
        "exception_message",
        "exit_code",
        "elapsed_seconds",
        "log",
        "inventory",
    }
    if set(diagnostic) != expected_diagnostic_fields:
        raise ValueError("skip diagnostic has missing or unknown fields")
    if any(
        not _json_exact(diagnostic.get(key), value)
        for key, value in expected_identity.items()
    ):
        raise ValueError("skip diagnostic identity changed")
    if not _is_nonnegative_number(diagnostic.get("elapsed_seconds")):
        raise ValueError("skip diagnostic elapsed_seconds must be nonnegative")
    diagnostic_exit = diagnostic.get("exit_code")
    if isinstance(diagnostic_exit, bool) or (
        diagnostic_exit is not None and not isinstance(diagnostic_exit, int)
    ):
        raise ValueError("skip diagnostic exit_code must be an integer or null")
    if any(
        not isinstance(diagnostic.get(field), str) or not diagnostic.get(field)
        for field in ("stage", "exception_type", "exception_message")
    ):
        raise ValueError("skip diagnostic strings must be nonempty")
    expected_log = (
        _lexical_absolute(run_root)
        / "logs"
        / f"{index:04d}__{_safe_task_name(task_id)}.log"
    )
    current_log = _verify_bound_record(
        diagnostic.get("log"),
        run_root=run_root,
        label="skip log",
        expected_path=expected_log,
    )
    if terminal.get("log") != current_log:
        raise ValueError("TRACE_SKIP log binding changed")
    for field in (
        "stage",
        "exception_type",
        "exception_message",
        "exit_code",
        "elapsed_seconds",
    ):
        if terminal.get(field) != diagnostic.get(field):
            raise ValueError(f"TRACE_SKIP {field} binding changed")
    inventory = diagnostic.get("inventory")
    if not isinstance(inventory, list):
        raise ValueError("skip diagnostic inventory must be a list")
    verified_inventory = [
        _verify_bound_record(
            record,
            run_root=run_root,
            label="partial artifact",
        )
        for record in inventory
    ]
    if verified_inventory != _inventory_partial_artifacts(run_root, task_id):
        raise ValueError("skip diagnostic artifact inventory changed")


def _validate_terminal_history(
    manifest: dict[str, Any],
    run_root: Path,
    events: list[dict[str, Any]],
) -> tuple[list[str], list[str], int | None, dict[str, Any] | None]:
    passed_task_ids: list[str] = []
    skipped_task_ids: list[str] = []
    expected_index = 0
    running_start: dict[str, Any] | None = None
    for event in events:
        event_name = str(event.get("event") or "")
        if running_start is None:
            if event_name != "TRACE_START":
                raise ValueError("terminal status event has no preceding TRACE_START")
            if expected_index >= len(manifest["source_reports"]):
                raise ValueError("status history contains extra tasks")
            _validate_status_event_schema(
                manifest,
                run_root,
                expected_index,
                event,
                event_name,
            )
            _validate_event_identity(manifest, expected_index, event)
            running_start = event
            continue
        if event_name not in {"TRACE_PASS", "TRACE_SKIP"}:
            raise ValueError("TRACE_START must be followed by exactly one terminal event")
        _validate_status_event_schema(
            manifest,
            run_root,
            expected_index,
            event,
            event_name,
        )
        _validate_event_identity(manifest, expected_index, event)
        if event_name == "TRACE_PASS":
            _verify_pass_terminal(
                manifest,
                run_root,
                expected_index,
                running_start,
                event,
            )
            passed_task_ids.append(str(event["task_id"]))
        else:
            _verify_skip_terminal(
                manifest,
                run_root,
                expected_index,
                running_start,
                event,
            )
            skipped_task_ids.append(str(event["task_id"]))
        expected_index += 1
        running_start = None
    return (
        passed_task_ids,
        skipped_task_ids,
        expected_index if running_start is not None else None,
        running_start,
    )


def run_frozen_batch(
    manifest_path: Path,
    run_root: Path,
    *,
    resume: bool = False,
    command_builder: Callable[[Path, Path, int], list[str]] | None = None,
) -> dict[str, Any]:
    """Run frozen sources sequentially through isolated child subprocesses."""

    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    manifest_file = _safe_run_path(manifest_path, run_root=root)
    manifest = _read_json_object(manifest_file, label="manifest")
    offline_summary = _offline_preflight(manifest)
    _validate_bound_preflight(manifest, root, offline_summary)
    status_path = root / _STATUS_FILENAME
    _safe_run_path(status_path, run_root=root)
    if not resume and status_path.is_file() and status_path.stat().st_size:
        raise ValueError("existing nonempty status history requires --resume")
    events = _read_status_events(status_path, run_root=root) if resume else []
    (
        passed_task_ids,
        skipped_task_ids,
        dangling_index,
        dangling_start,
    ) = _validate_terminal_history(manifest, root, events)
    next_index = len(passed_task_ids) + len(skipped_task_ids)
    if dangling_index is not None:
        if dangling_start is None or dangling_index != next_index:
            raise ValueError("invalid dangling TRACE_START state")
        source = manifest["source_reports"][dangling_index]
        task_id = str(source["task_id"])
        log_path = _safe_run_path(
            Path(str(dangling_start.get("log_path") or "")),
            run_root=root,
        )
        _ensure_safe_directory(log_path.parent, run_root=root)
        _safe_run_path(log_path, run_root=root)
        log_path.touch(exist_ok=True)
        interrupted = InterruptedTrace(
            "prior process ended after TRACE_START without a terminal record"
        )
        diagnostic, diagnostic_path = _write_skip_diagnostic(
            manifest,
            root,
            dangling_index,
            stage="interrupted",
            exception=interrupted,
            exit_code=None,
            elapsed_seconds=0.0,
            log_path=log_path,
        )
        skip_event = {
            "event": "TRACE_SKIP",
            "timestamp_utc": _timestamp_utc(),
            **_source_event_identity(manifest, dangling_index),
            "stage": diagnostic["stage"],
            "exception_type": diagnostic["exception_type"],
            "exception_message": diagnostic["exception_message"],
            "elapsed_seconds": diagnostic["elapsed_seconds"],
            "exit_code": None,
            "log": diagnostic["log"],
            "diagnostic": _artifact_record(diagnostic_path, run_root=root),
        }
        _append_status_event(status_path, skip_event, run_root=root)
        skipped_task_ids.append(task_id)
        next_index += 1
    progress = _progress_payload(
        manifest,
        passed_task_ids=passed_task_ids,
        skipped_task_ids=skipped_task_ids,
        running=0,
    )
    _atomic_write_json(
        root / _PROGRESS_FILENAME,
        progress,
        run_root=root,
    )
    build_command = command_builder or _default_child_command
    for index in range(next_index, len(manifest["source_reports"])):
        source = manifest["source_reports"][index]
        task_id = str(source["task_id"])
        paths = _task_paths(root, task_id)
        for path in paths.values():
            _ensure_safe_directory(path, run_root=root)
        log_path = root / "logs" / f"{index:04d}__{_safe_task_name(task_id)}.log"
        _ensure_safe_directory(log_path.parent, run_root=root)
        _safe_run_path(log_path, run_root=root)
        start_event = {
            "event": "TRACE_START",
            "timestamp_utc": _timestamp_utc(),
            **_source_event_identity(manifest, index),
            "log_path": str(log_path),
        }
        _append_status_event(status_path, start_event, run_root=root)
        _atomic_write_json(
            root / _PROGRESS_FILENAME,
            _progress_payload(
                manifest,
                passed_task_ids=passed_task_ids,
                skipped_task_ids=skipped_task_ids,
                running=1,
            ),
            run_root=root,
        )
        started = time.monotonic()
        completed: subprocess.CompletedProcess[Any] | None = None
        failure: BaseException | None = None
        operator_interrupt: KeyboardInterrupt | SystemExit | None = None
        stage = "child_launch"
        _safe_run_path(log_path, run_root=root)
        with log_path.open("wb") as log_handle:
            try:
                command = build_command(manifest_file, root, index)
                if not isinstance(command, list) or not command:
                    raise ValueError("child command builder must return a nonempty argv list")
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                if completed.returncode != 0:
                    stage = "child_exit"
                    failure = RuntimeError(
                        f"child exited with status {completed.returncode}"
                    )
                else:
                    stage = "terminal_verification"
                    validate_child_result(manifest, root, index)
            except (KeyboardInterrupt, SystemExit) as exc:
                failure = exc
                operator_interrupt = exc
            except Exception as exc:
                failure = exc
        elapsed = time.monotonic() - started
        _sanitize_log_file(log_path, run_root=root)
        log_record = _artifact_record(log_path, run_root=root)
        if failure is None:
            result_path = paths["diagnostic"] / "child_result.json"
            pass_event = {
                "event": "TRACE_PASS",
                "timestamp_utc": _timestamp_utc(),
                **_source_event_identity(manifest, index),
                "elapsed_seconds": elapsed,
                "exit_code": int(completed.returncode) if completed else 0,
                "log": log_record,
                "child_result": _artifact_record(result_path, run_root=root),
            }
            _append_status_event(status_path, pass_event, run_root=root)
            passed_task_ids.append(task_id)
        else:
            diagnostic, diagnostic_path = _write_skip_diagnostic(
                manifest,
                root,
                index,
                stage=stage,
                exception=failure,
                exit_code=(
                    int(completed.returncode)
                    if completed is not None
                    else None
                ),
                elapsed_seconds=elapsed,
                log_path=log_path,
            )
            skip_event = {
                "event": "TRACE_SKIP",
                "timestamp_utc": _timestamp_utc(),
                **_source_event_identity(manifest, index),
                "stage": diagnostic["stage"],
                "exception_type": diagnostic["exception_type"],
                "exception_message": diagnostic["exception_message"],
                "elapsed_seconds": elapsed,
                "exit_code": diagnostic["exit_code"],
                "log": log_record,
                "diagnostic": _artifact_record(diagnostic_path, run_root=root),
            }
            _append_status_event(status_path, skip_event, run_root=root)
            skipped_task_ids.append(task_id)
        progress = _progress_payload(
            manifest,
            passed_task_ids=passed_task_ids,
            skipped_task_ids=skipped_task_ids,
            running=0,
        )
        _atomic_write_json(
            root / _PROGRESS_FILENAME,
            progress,
            run_root=root,
        )
        if operator_interrupt is not None:
            raise operator_interrupt
    return progress


_RECOVERABLE_POINT_FIELDS = {
    "task_id",
    "point_num",
    "action_sequence_num",
    "horizon",
    "rollout_mode",
    "policy_mode",
    "valid_prediction",
    "decision_error",
    "skipped_by_prediction_interval",
}
_EMPTY_POINT_FIELDS = [
    "task_id",
    "point_num",
    "action_sequence_num",
    "horizon",
    "rollout_mode",
    "policy_mode",
]
_MANAGED_AGGREGATE_FILENAMES = (
    "complete_prediction_points.csv",
    "partial_diagnostic_prediction_points.csv",
    "aggregate_by_policy_horizon.csv",
    "cohort_quality.json",
)
_MANAGED_FIGURE_FILENAMES = (
    "tier1_accuracy_recall_by_horizon.png",
    "tier2_total_code_match_by_horizon.png",
    "token_cost_by_horizon.png",
    "predict_only_summary.png",
)


class _EmptyPassCohort(ValueError):
    """Expected diagnostic-only terminal state with no primary cohort."""


def _invalidate_managed_aggregate_outputs(run_root: Path) -> None:
    """Remove only known aggregate outputs after a complete safety inspection."""

    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    managed = (
        ("aggregate", _MANAGED_AGGREGATE_FILENAMES),
        ("figures", _MANAGED_FIGURE_FILENAMES),
    )
    regular_files: list[Path] = []
    for directory_name, filenames in managed:
        directory = root / directory_name
        try:
            directory_metadata = directory.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(directory_metadata.st_mode):
            raise ValueError(
                f"managed output directory must not be a symlink: {directory}"
            )
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError(
                f"managed output directory must be a real directory: {directory}"
            )
        _assert_no_symlink_components(directory)
        for filename in filenames:
            path = directory / filename
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"managed output must not be a symlink: {path}"
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"managed output must be a regular file: {path}"
                )
            regular_files.append(path)
    for path in regular_files:
        path.unlink()


def _read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fieldnames


def _ordered_csv_fields(
    rows: list[dict[str, Any]],
    *,
    empty_fields: list[str] | None = None,
) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields or list(empty_fields or _EMPTY_POINT_FIELDS)


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    run_root: Path,
    empty_fields: list[str] | None = None,
) -> None:
    destination = _safe_run_path(path, run_root=run_root)
    _ensure_safe_directory(destination.parent, run_root=run_root)
    _safe_run_path(destination, run_root=run_root)
    fieldnames = _ordered_csv_fields(rows, empty_fields=empty_fields)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _safe_run_path(destination, run_root=run_root)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _recover_skipped_point_rows(
    manifest: dict[str, Any],
    run_root: Path,
    index: int,
    terminal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_id = str(manifest["source_reports"][index]["task_id"])
    diagnostic_record = terminal["diagnostic"]
    diagnostic = _read_json_object(
        Path(str(diagnostic_record["path"])),
        label="skip diagnostic",
    )
    recovered: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    frozen_points = {
        (int(point["point_num"]), int(point["sequence_num"]))
        for point in manifest["source_reports"][index]["points"]
    }
    allowed_horizons = {
        int(horizon) for horizon in manifest["matrix"]["horizons"]
    }
    allowed_rollouts = {
        str(mode) for mode in manifest["matrix"]["rollout_modes"]
    }
    allowed_policies = {
        str(mode) for mode in manifest["matrix"]["policy_modes"]
    }
    seen_keys: set[tuple[int, int, int, str, str]] = set()
    for record in diagnostic["inventory"]:
        path = Path(str(record["path"]))
        if path.suffix.lower() != ".csv":
            continue
        try:
            rows, fieldnames = _read_csv_rows(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        if not _RECOVERABLE_POINT_FIELDS.issubset(fieldnames):
            continue
        task_rows: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("task_id") or "") != task_id:
                continue
            try:
                point_num = int(str(row.get("point_num") or ""))
                sequence_num = int(
                    str(row.get("action_sequence_num") or "")
                )
                horizon = int(str(row.get("horizon") or ""))
            except ValueError:
                continue
            rollout_mode = str(row.get("rollout_mode") or "")
            policy_mode = str(row.get("policy_mode") or "")
            key = (
                point_num,
                sequence_num,
                horizon,
                rollout_mode,
                policy_mode,
            )
            if (
                (point_num, sequence_num) not in frozen_points
                or horizon not in allowed_horizons
                or rollout_mode not in allowed_rollouts
                or policy_mode not in allowed_policies
                or key in seen_keys
            ):
                continue
            seen_keys.add(key)
            task_rows.append(
                {
                    **row,
                    "diagnostic_terminal_state": "TRACE_SKIP",
                    "diagnostic_source_artifact": str(path),
                }
            )
        if not task_rows:
            continue
        source_records.append(dict(record))
        recovered.extend(task_rows)
    return recovered, source_records


def _plot_contract(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate_row_count": len(aggregate_rows),
        "series": [
            {
                "rollout_mode": rollout_mode,
                "policy_mode": policy_mode,
            }
            for rollout_mode, policy_mode in sorted(
                {
                    (
                        str(row.get("rollout_mode") or ""),
                        str(row.get("policy_mode") or ""),
                    )
                    for row in aggregate_rows
                }
            )
        ],
        "horizons": sorted(
            {int(str(row["horizon"])) for row in aggregate_rows}
        ),
        "figures": {
            "tier1": {
                "metrics": ["t1_accuracy", "t1_recall"],
            },
            "tier2_total": {
                "metrics": ["t2_code_match_rate", "t2_mean_jaccard"],
                "estimand": (
                    "all_risk_relevant_rows_missing_pred_codes_zero"
                ),
            },
            "token": {
                "metrics": [
                    "prompt_tokens_per_point",
                    "completion_tokens_per_point",
                    "total_tokens_per_point",
                ],
            },
            "pilot_summary": {
                "metrics": [
                    "t1_accuracy",
                    "t1_recall",
                    "t2_code_match_rate",
                ],
            },
        },
    }


def _write_standard_diagnostic_figures(
    aggregate_rows: list[dict[str, Any]],
    *,
    run_root: Path,
) -> dict[str, Path]:
    from experiments import plot_results as unified_plots

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = _ensure_safe_directory(
        _lexical_absolute(run_root) / "figures",
        run_root=run_root,
    )
    plot_rows = [
        {
            **unified_plots._with_derived_values(row),
            "dataset_id": "HarnessAudit Luna diagnostic pilot",
            "benchmark": "harnessaudit",
        }
        for row in aggregate_rows
    ]
    specs = [
        (
            "tier1",
            "tier1_accuracy_recall_by_horizon.png",
            list(unified_plots.TIER1_METRICS),
            "HA Luna diagnostic predict-only\nTier 1 risk detection by horizon",
            True,
            2,
        ),
        (
            "tier2_total",
            "tier2_total_code_match_by_horizon.png",
            list(unified_plots.TIER2_METRICS),
            (
                "HA Luna diagnostic predict-only\n"
                "Tier 2 total code matching by horizon"
            ),
            True,
            2,
        ),
        (
            "token",
            "token_cost_by_horizon.png",
            list(unified_plots.TOKEN_METRICS),
            "HA Luna diagnostic predict-only\nToken cost by horizon",
            False,
            3,
        ),
        (
            "pilot_summary",
            "predict_only_summary.png",
            list(unified_plots.SUMMARY_METRICS),
            (
                "HA Luna diagnostic predict-only\n"
                "Accuracy, recall, and Tier 2 total by horizon"
            ),
            True,
            3,
        ),
    ]
    artifacts: dict[str, Path] = {}
    for name, filename, metrics, title, unit_interval, ncols in specs:
        destination = _safe_run_path(
            figure_dir / filename,
            run_root=run_root,
        )
        with tempfile.NamedTemporaryFile(
            dir=figure_dir,
            prefix=f".{destination.stem}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            unified_plots._plot_metric_grid(
                plt=plt,
                rows=plot_rows,
                metrics=metrics,
                paths=[temporary],
                title=title,
                unit_interval=unit_interval,
                ncols=ncols,
            )
            _safe_run_path(destination, run_root=run_root)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        artifacts[name] = destination
    return artifacts


def aggregate_complete_cases(
    manifest_path: Path,
    run_root: Path,
) -> dict[str, Path]:
    """Regenerate complete-case cohort outputs without invoking a child/model."""

    root = _assert_no_symlink_components(_lexical_absolute(run_root))
    manifest_file = _safe_run_path(manifest_path, run_root=root)
    _invalidate_managed_aggregate_outputs(root)
    try:
        return _aggregate_complete_cases_after_invalidation(
            manifest_file,
            root,
        )
    except _EmptyPassCohort:
        raise
    except BaseException:
        _invalidate_managed_aggregate_outputs(root)
        raise


def _aggregate_complete_cases_after_invalidation(
    manifest_file: Path,
    root: Path,
) -> dict[str, Path]:
    manifest = _read_json_object(manifest_file, label="manifest")
    offline_summary = _offline_preflight(manifest)
    _validate_bound_preflight(manifest, root, offline_summary)
    status_path = _safe_run_path(root / _STATUS_FILENAME, run_root=root)
    events = _read_status_events(status_path, run_root=root)
    (
        passed_task_ids,
        skipped_task_ids,
        dangling_index,
        _dangling_start,
    ) = _validate_terminal_history(manifest, root, events)
    if dangling_index is not None:
        raise ValueError("cannot aggregate a dangling TRACE_START")
    if len(passed_task_ids) + len(skipped_task_ids) != len(
        manifest["source_reports"]
    ):
        raise ValueError("every frozen source must have a verified terminal state")

    aggregate_dir = _ensure_safe_directory(
        root / "aggregate",
        run_root=root,
    )
    figure_dir = _ensure_safe_directory(root / "figures", run_root=root)
    del figure_dir
    terminal_events = [
        event
        for event in events
        if event["event"] in {"TRACE_PASS", "TRACE_SKIP"}
    ]

    complete_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    passed_bindings: list[dict[str, Any]] = []
    skipped_bindings: list[dict[str, Any]] = []
    ordered_skipped: list[dict[str, Any]] = []
    for index, (source, terminal) in enumerate(
        zip(manifest["source_reports"], terminal_events)
    ):
        task_id = str(source["task_id"])
        source_binding = {
            "index": index,
            "task_id": task_id,
            "report_path": source["report_path"],
            "report_sha256": source["report_sha256"],
            "trace_path": source["trace_path"],
            "trace_sha256": source["trace_sha256"],
            "point_keys": list(source["point_keys"]),
        }
        if terminal["event"] == "TRACE_PASS":
            child_result = validate_child_result(manifest, root, index)
            points_record = child_result["artifacts"]["points"]
            rows, _fieldnames = _read_csv_rows(
                Path(str(points_record["path"]))
            )
            complete_rows.extend(rows)
            passed_bindings.append(
                {
                    **source_binding,
                    "terminal": "TRACE_PASS",
                    "child_result": dict(terminal["child_result"]),
                    "points": dict(points_record),
                }
            )
            continue
        recovered_rows, recovery_records = _recover_skipped_point_rows(
            manifest,
            root,
            index,
            terminal,
        )
        diagnostic_rows.extend(recovered_rows)
        ordered_skipped.append(
            {
                "task_id": task_id,
                "stage": terminal["stage"],
                "reason_type": terminal["exception_type"],
                "reason": terminal["exception_message"],
                "exit_code": terminal["exit_code"],
            }
        )
        skipped_bindings.append(
            {
                **source_binding,
                "terminal": "TRACE_SKIP",
                "diagnostic": dict(terminal["diagnostic"]),
                "recoverable_point_artifacts": recovery_records,
            }
        )

    from multi_agent.safepred_experiments.predict_only_pipeline import (
        _sort_point_rows,
    )
    from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
        aggregate_by_policy_horizon,
    )

    complete_rows = _sort_point_rows(complete_rows)
    diagnostic_rows = _sort_point_rows(diagnostic_rows)
    complete_path = aggregate_dir / "complete_prediction_points.csv"
    diagnostic_path = (
        aggregate_dir / "partial_diagnostic_prediction_points.csv"
    )
    aggregate_path = aggregate_dir / "aggregate_by_policy_horizon.csv"
    quality_path = aggregate_dir / "cohort_quality.json"
    _atomic_write_csv(
        diagnostic_path,
        diagnostic_rows,
        run_root=root,
    )

    requested_task_ids = [
        str(source["task_id"]) for source in manifest["source_reports"]
    ]
    quality: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ha_resilient_predict_only_cohort_quality",
        "run_id": manifest["run_id"],
        "requested_count": len(requested_task_ids),
        "eligible_count": len(requested_task_ids),
        "eligible_count_definition": (
            "source records frozen in the validated manifest before outcomes; "
            "no post-outcome replacement or available-case filtering"
        ),
        "passed_count": len(passed_task_ids),
        "skipped_count": len(skipped_task_ids),
        "incomplete_count": (
            len(requested_task_ids)
            - len(passed_task_ids)
            - len(skipped_task_ids)
        ),
        "ordered_requested_task_ids": requested_task_ids,
        "ordered_eligible_task_ids": requested_task_ids,
        "ordered_passed_task_ids": passed_task_ids,
        "ordered_skipped": ordered_skipped,
        "ordered_incomplete_task_ids": [],
        "resource_semantics": _RESOURCE_SEMANTICS,
        "diagnostic": True,
        "confirmatory": False,
        "compositionality_status": "unannotated",
        "primary_complete_case": True,
        "primary_row_count": len(complete_rows),
        "partial_diagnostic_row_count": len(diagnostic_rows),
        "bindings": {
            "manifest": _artifact_record(manifest_file, run_root=root),
            "preflight": _artifact_record(
                root / "preflight.json",
                run_root=root,
            ),
            "status_history": _artifact_record(status_path, run_root=root),
            "passed_sources": passed_bindings,
            "skipped_sources": skipped_bindings,
        },
    }
    if not complete_rows:
        quality["plot_contract"] = None
        _atomic_write_json(quality_path, quality, run_root=root)
        raise _EmptyPassCohort(
            "no verified TRACE_PASS cohort; primary aggregate and figures refused"
        )

    aggregate_rows = aggregate_by_policy_horizon(complete_rows)
    quality["aggregate_condition_count"] = len(aggregate_rows)
    quality["plot_contract"] = _plot_contract(aggregate_rows)
    _atomic_write_csv(complete_path, complete_rows, run_root=root)
    _atomic_write_csv(
        aggregate_path,
        aggregate_rows,
        run_root=root,
        empty_fields=["rollout_mode", "policy_mode", "horizon"],
    )
    figures = _write_standard_diagnostic_figures(
        aggregate_rows,
        run_root=root,
    )
    _atomic_write_json(quality_path, quality, run_root=root)
    return {
        "complete_points": complete_path,
        "partial_diagnostic_points": diagnostic_path,
        "aggregate": aggregate_path,
        "cohort_quality": quality_path,
        **figures,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and preflight a resilient predict-only manifest."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="freeze a local source cohort into run-root/manifest.json",
    )
    prepare.add_argument("--partial-audit", type=Path, required=True)
    prepare.add_argument("--task-ids", nargs="+", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--model-revision-label", required=True)
    prepare.add_argument(
        "--reasoning-mode",
        choices=("provider_default", "disabled"),
    )
    prepare.add_argument("--run-root", type=Path, required=True)

    preflight = commands.add_parser(
        "preflight",
        help="run offline checks, then enumerate the exact gateway model",
    )
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--run-root", type=Path, required=True)

    child = commands.add_parser(
        "child",
        help="run exactly one indexed frozen source in an isolated process",
    )
    child.add_argument("--manifest", type=Path, required=True)
    child.add_argument("--run-root", type=Path, required=True)
    child.add_argument("--index", type=int, required=True)

    run = commands.add_parser(
        "run",
        help="run or resume the ordered frozen source cohort",
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--resume", action="store_true")

    aggregate = commands.add_parser(
        "aggregate",
        help="regenerate complete-case aggregate artifacts without model calls",
    )
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "prepare":
        manifest = build_frozen_manifest(
            arguments.partial_audit,
            list(arguments.task_ids),
            arguments.model,
            arguments.model_revision_label,
            reasoning_mode=arguments.reasoning_mode,
        )
        _atomic_write_json(
            arguments.run_root / "manifest.json",
            manifest,
            run_root=arguments.run_root,
        )
        return 0
    if arguments.command == "child":
        run_frozen_child(
            arguments.manifest,
            arguments.run_root,
            arguments.index,
        )
        return 0
    if arguments.command == "run":
        run_frozen_batch(
            arguments.manifest,
            arguments.run_root,
            resume=arguments.resume,
        )
        return 0
    if arguments.command == "aggregate":
        aggregate_complete_cases(
            arguments.manifest,
            arguments.run_root,
        )
        return 0

    manifest_path = _safe_run_path(
        arguments.manifest,
        run_root=arguments.run_root,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    preflight_frozen_manifest(manifest, arguments.run_root)
    return 0

def validate_frozen_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate count-independent frozen-manifest structure."""

    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    source_reports = manifest.get("source_reports")
    if not isinstance(source_reports, list):
        raise ValueError("source_reports must be a list")
    if not source_reports:
        raise ValueError("source_reports must not be empty")

    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must equal 1")
    if manifest.get("artifact_type") != "ha_resilient_predict_only_frozen_manifest":
        raise ValueError("unexpected manifest artifact_type")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    model_id = str(model.get("id") or "")
    expected_study_id = _study_id_for_model(model_id)
    execution_binding = manifest.get("execution_provenance")
    if execution_binding is not None:
        if (
            model_id != _AGENTWORLD_MODEL_ID
            or not isinstance(execution_binding, dict)
            or set(execution_binding)
            != {"artifact_type", "record_sha256"}
            or execution_binding.get("artifact_type")
            != "ha_agentworld_execution_provenance"
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(execution_binding.get("record_sha256") or ""),
            )
        ):
            raise ValueError("manifest execution provenance binding is invalid")
    if manifest.get("study_id") != expected_study_id:
        raise ValueError(f"study_id must equal {expected_study_id}")
    run_id = str(manifest.get("run_id") or "")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must match the frozen HA run identity form")
    if manifest.get("diagnostic_label") != _DIAGNOSTIC_LABEL:
        raise ValueError(f"diagnostic_label must equal {_DIAGNOSTIC_LABEL}")
    if manifest.get("resource_semantics") != _RESOURCE_SEMANTICS:
        raise ValueError("resource_semantics must equal legacy_official_v1")
    if manifest.get("diagnostic") is not True or manifest.get("confirmatory") is not False:
        raise ValueError("manifest must be diagnostic-only")
    if manifest.get("compositionality") != "unannotated":
        raise ValueError("compositionality must equal unannotated")

    ordered_task_ids = manifest.get("ordered_task_ids")
    observed_task_ids = [
        str(source.get("task_id") or "")
        for source in source_reports
        if isinstance(source, dict)
    ]
    if len(observed_task_ids) != len(source_reports) or any(
        not task_id for task_id in observed_task_ids
    ):
        raise ValueError("every source report must have a task_id")
    if ordered_task_ids != observed_task_ids:
        raise ValueError("ordered_task_ids must match source report order")
    if len(set(observed_task_ids)) != len(observed_task_ids):
        raise ValueError("source report task IDs must be unique")
    safe_task_names = [_safe_task_name(task_id) for task_id in observed_task_ids]
    if len(set(safe_task_names)) != len(safe_task_names):
        raise ValueError("source report safe task names must be unique")

    all_point_keys: list[str] = []
    eligible_count = 0
    for source in source_reports:
        source_point_keys = source.get("point_keys")
        points = source.get("points")
        selected_sequences = source.get("selected_sequence_nums")
        if not isinstance(source_point_keys, list) or len(source_point_keys) != 2:
            raise ValueError("every source report must freeze exactly two point keys")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("every source report must freeze exactly two points")
        point_sequences = [
            int(point.get("sequence_num") or 0)
            for point in points
            if isinstance(point, dict)
        ]
        if len(point_sequences) != 2 or selected_sequences != point_sequences:
            raise ValueError("selected sequences must match frozen points")
        if [point.get("point_key") for point in points] != source_point_keys:
            raise ValueError("source point keys must match frozen points")
        if source.get("resource_semantics") != _RESOURCE_SEMANTICS:
            raise ValueError("source resource semantics must equal legacy_official_v1")
        if source.get("source_audit_valid") is True:
            eligible_count += 1
        all_point_keys.extend(str(point_key) for point_key in source_point_keys)

    point_selection = manifest.get("point_selection")
    if not isinstance(point_selection, dict):
        raise ValueError("point_selection must be an object")
    if point_selection.get("scheme") != "middle_pair":
        raise ValueError("point selection scheme must equal middle_pair")
    if point_selection.get("required_remaining_action_count") != max(_HORIZONS):
        raise ValueError("point eligibility must require six remaining actions")
    if point_selection.get("point_keys") != all_point_keys:
        raise ValueError("point_selection point keys must match source order")

    if model_id == _AGENTWORLD_MODEL_ID:
        if model.get("provider") != "local_vllm":
            raise ValueError("AgentWorld model provider must equal local_vllm")
        _validated_agentworld_endpoint(model.get("endpoint"))
    else:
        _validated_model_endpoint(model.get("endpoint"))
    requested_revision_label = str(
        model.get("requested_revision_label") or ""
    )
    if not requested_revision_label.strip():
        raise ValueError("requested_revision_label must not be empty")
    if model.get("seed") != 0:
        raise ValueError("model seed must equal 0")
    reasoning_mode = _validated_reasoning_mode(
        model.get("reasoning_mode"),
        model_id=model_id,
    )
    partial_audit = manifest.get("partial_audit")
    if not isinstance(partial_audit, dict):
        raise ValueError("partial_audit must be an object")
    partial_audit_sha256 = str(partial_audit.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", partial_audit_sha256):
        raise ValueError("partial_audit.sha256 must be a lowercase SHA-256")
    expected_run_id = _deterministic_run_id(
        partial_audit_sha256=partial_audit_sha256,
        ordered_task_ids=observed_task_ids,
        model_id=str(model["id"]),
        requested_revision_label=requested_revision_label,
        reasoning_mode=reasoning_mode,
    )
    if run_id != expected_run_id:
        raise ValueError("run_id does not match the deterministic frozen identity")

    expected_matrix = {
        "target": "exact_step",
        "horizons": _HORIZONS,
        "rollout_modes": _ROLLOUT_MODES,
        "policy_modes": _POLICY_MODES,
        "candidate_count": 1,
        "max_trace_events": 64,
        "tier3_enabled": False,
        "concurrency": 1,
    }
    if manifest.get("matrix") != expected_matrix:
        raise ValueError("manifest matrix does not match the frozen experiment matrix")

    expected_conditions = [
        (horizon, rollout_mode, policy_mode)
        for horizon in _HORIZONS
        for rollout_mode in _ROLLOUT_MODES
        for policy_mode in _POLICY_MODES
    ]
    condition_grid = manifest.get("condition_grid")
    if not isinstance(condition_grid, list) or [
        (
            condition.get("horizon"),
            condition.get("rollout_mode"),
            condition.get("policy_mode"),
        )
        for condition in condition_grid
        if isinstance(condition, dict)
    ] != expected_conditions:
        raise ValueError("condition_grid must contain the complete ordered matrix")
    if any(condition.get("point_keys") != all_point_keys for condition in condition_grid):
        raise ValueError("every condition must share the exact same point keys")

    return {
        **manifest,
        "requested_count": len(source_reports),
        "eligible_count": eligible_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
