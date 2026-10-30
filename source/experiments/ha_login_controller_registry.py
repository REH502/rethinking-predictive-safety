#!/usr/bin/env python3
"""Strict provenance records for a HA formal login-host controller.

The Qwen actor may run as a separately scheduled vLLM service while the
controller stays on the login host in order to reach the external WM/judge
endpoint.  That is a real two-process topology, not a SLURM controller job.
This module records it explicitly so terminal postflight never has to infer
or misrepresent ownership of the actor service.

The module is intentionally model-free.  It only reads public files and a
durable matrix checkpoint; it never loads credentials, calls a model, or
contacts the scheduler.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid
from typing import Any
from urllib.parse import urlsplit


LOGIN_CONTROLLER_REGISTRY_SCHEMA = "ha-formal-login-controller-invocation-registry-v1"
LOGIN_CONTROLLER_STATUS_SCHEMA = "ha-formal-login-controller-status-v1"
LOGIN_CONTROLLER_KIND = "login_controller"
DEADLINE_SOURCE = "actor_service_slurm_end_time_v1"
EXTERNAL_SERVICE_OWNERSHIP = "external_slurm_service_not_owned_by_login_controller"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,62}")


class LoginControllerRegistryError(ValueError):
    """A stable, secret-free validation disposition."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", code):
            raise ValueError("invalid login-controller registry error code")
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise LoginControllerRegistryError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> Path:
    absolute = _lexical_absolute(path)
    candidates = [*reversed(absolute.parents), absolute]
    for index, candidate in enumerate(candidates):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(candidates) - 1:
                continue
            _fail("login_controller_path_component_missing")
        if stat.S_ISLNK(metadata.st_mode):
            _fail("login_controller_symlink_path_rejected")
    return absolute


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _regular_file(path: Path, *, code: str, allow_empty: bool = True) -> Path:
    absolute = _reject_symlink_components(path)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        _fail(code)
    if not stat.S_ISREG(metadata.st_mode):
        _fail(code)
    if not allow_empty and metadata.st_size == 0:
        _fail(code)
    return absolute


def _read_json(path: Path, *, code: str) -> Any:
    source = _regular_file(path, code=code, allow_empty=False)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)


def _read_jsonl(path: Path, *, code: str) -> list[dict[str, Any]]:
    source = _regular_file(path, code=code, allow_empty=False)
    rows: list[dict[str, Any]] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line:
                _fail(code)
            row = json.loads(line)
            if not isinstance(row, dict):
                _fail(code)
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not rows:
        _fail(code)
    return rows


def _parse_utc_timestamp(value: Any, *, code: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError:
        _fail(code)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(code)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _binding(path: Path, *, code: str, allow_empty: bool = True) -> dict[str, Any]:
    source = _regular_file(path, code=code, allow_empty=allow_empty)
    before = source.lstat()
    digest = _sha256_file(source)
    after = source.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail("login_controller_file_changed_during_hash")
    return {
        "path": str(source),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def _validate_binding(
    value: Any,
    *,
    code: str,
    allowed_roots: Sequence[Path],
    allow_empty: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "size_bytes", "sha256"}:
        _fail(code)
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        _fail(code)
    source = _regular_file(Path(raw_path), code=code, allow_empty=allow_empty)
    if not any(_is_relative_to(source, root) for root in allowed_roots):
        _fail(code)
    if (
        not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] != source.lstat().st_size
        or not isinstance(value.get("sha256"), str)
        or not _SHA256_RE.fullmatch(value["sha256"])
        or value["sha256"] != _sha256_file(source)
    ):
        _fail(code)
    return {
        "path": str(source),
        "size_bytes": int(value["size_bytes"]),
        "sha256": str(value["sha256"]),
    }


def _validate_controller_id(value: Any, *, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        _fail(code)
    if str(parsed) != value.lower():
        _fail(code)
    return str(parsed)


def _validate_service_identity(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "slurm_job_id",
        "slurm_node",
        "base_url",
        "service_end_epoch_seconds",
        "invocation_deadline_epoch_seconds",
        "invocation_safety_margin_seconds",
        "observed_state_at_start",
        "process_ownership",
    }:
        _fail(code)
    job_id = str(value.get("slurm_job_id") or "")
    node = str(value.get("slurm_node") or "")
    base_url = value.get("base_url")
    if not _JOB_ID_RE.fullmatch(job_id) or not _HOST_RE.fullmatch(node):
        _fail(code)
    if not isinstance(base_url, str):
        _fail(code)
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        _fail(code)
    if (
        parsed.scheme != "http"
        or parsed.hostname != node
        or parsed.path != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        _fail(code)
    end_epoch = value.get("service_end_epoch_seconds")
    deadline_epoch = value.get("invocation_deadline_epoch_seconds")
    margin = value.get("invocation_safety_margin_seconds")
    if (
        isinstance(end_epoch, bool)
        or not isinstance(end_epoch, int)
        or end_epoch <= 0
        or isinstance(deadline_epoch, bool)
        or not isinstance(deadline_epoch, int)
        or deadline_epoch != end_epoch
        or isinstance(margin, bool)
        or not isinstance(margin, int)
        or margin < 1
        or margin >= end_epoch
    ):
        _fail(code)
    if value.get("observed_state_at_start") != "RUNNING":
        _fail(code)
    if value.get("process_ownership") != EXTERNAL_SERVICE_OWNERSHIP:
        _fail(code)
    return {
        "slurm_job_id": job_id,
        "slurm_node": node,
        "base_url": base_url,
        "service_end_epoch_seconds": end_epoch,
        "invocation_deadline_epoch_seconds": deadline_epoch,
        "invocation_safety_margin_seconds": margin,
        "observed_state_at_start": "RUNNING",
        "process_ownership": EXTERNAL_SERVICE_OWNERSHIP,
    }


_REQUIRED_BINDINGS = frozenset(
    {
        "controller_launcher",
        "model_profile",
        "preregistration",
        "serving_contract",
        "service_info",
    }
)
_REQUIRED_PATHS = frozenset(
    {
        "controller_stdout",
        "controller_stderr",
        "invocation_status",
        "outer_credential_scan",
        "launcher_lock",
        "service_vllm_log",
        "service_checkpoint_verification",
        "service_gpu_info",
    }
)


def _validate_paths(
    value: Any,
    *,
    runs_root: Path,
    code: str,
    require_exists: bool,
    allowed_missing: frozenset[str] = frozenset(),
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _REQUIRED_PATHS:
        _fail(code)
    normalized: dict[str, str] = {}
    for name, raw_path in value.items():
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            _fail(code)
        may_be_missing = not require_exists and name in allowed_missing
        candidate = _reject_symlink_components(
            Path(raw_path), allow_missing_leaf=may_be_missing
        )
        if not _is_relative_to(candidate, runs_root):
            _fail(code)
        if require_exists or not may_be_missing:
            _regular_file(candidate, code=code)
        normalized[name] = str(candidate)
    return normalized


def validate_login_controller_registry(
    path: Path,
    *,
    runs_root: Path,
    project_root: Path,
    require_paths_exist: bool = True,
) -> list[dict[str, Any]]:
    """Validate a complete, hash-chained login-controller registry.

    A distinct registry permits a terminal postflight to prove the actual
    controller topology.  It does not accept, coerce, or upgrade legacy SLURM
    registry rows.
    """

    root = _regular_file(path, code="login_controller_registry_invalid", allow_empty=False)
    normalized_runs = _reject_symlink_components(runs_root)
    normalized_project = _reject_symlink_components(project_root)
    rows = _read_jsonl(root, code="login_controller_registry_invalid")
    previous: str | None = None
    seen_controller_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema_version") != LOGIN_CONTROLLER_REGISTRY_SCHEMA:
            _fail("login_controller_registry_schema_mismatch")
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        if (
            row.get("record_sha256") != _sha256_bytes(_canonical_json_bytes(unsigned))
            or row.get("previous_record_sha256") != previous
        ):
            _fail("login_controller_registry_hash_chain_invalid")
        controller = row.get("controller")
        if not isinstance(controller, dict) or set(controller) != {
            "kind",
            "id",
            "pid",
            "hostname",
            "started_at_utc",
            "deadline_source",
        }:
            _fail("login_controller_registry_controller_invalid")
        controller_id = _validate_controller_id(
            controller.get("id"), code="login_controller_registry_controller_invalid"
        )
        if controller_id in seen_controller_ids:
            _fail("login_controller_registry_controller_duplicate")
        seen_controller_ids.add(controller_id)
        if (
            controller.get("kind") != LOGIN_CONTROLLER_KIND
            or isinstance(controller.get("pid"), bool)
            or not isinstance(controller.get("pid"), int)
            or controller["pid"] <= 0
            or not isinstance(controller.get("hostname"), str)
            or not _HOST_RE.fullmatch(controller["hostname"])
            or controller.get("deadline_source") != DEADLINE_SOURCE
        ):
            _fail("login_controller_registry_controller_invalid")
        _parse_utc_timestamp(
            controller.get("started_at_utc"),
            code="login_controller_registry_controller_invalid",
        )
        study_id = row.get("study_id")
        if not isinstance(study_id, str) or not study_id or study_id != study_id.strip():
            _fail("login_controller_registry_study_invalid")
        service = _validate_service_identity(
            row.get("actor_service"), code="login_controller_registry_service_invalid"
        )
        bindings = row.get("immutable_input_bindings")
        if not isinstance(bindings, dict) or set(bindings) != _REQUIRED_BINDINGS:
            _fail("login_controller_registry_bindings_incomplete")
        normalized_bindings = {
            name: _validate_binding(
                value,
                code="login_controller_registry_binding_invalid",
                allowed_roots=(normalized_project, normalized_runs),
                allow_empty=False,
            )
            for name, value in bindings.items()
        }
        paths = _validate_paths(
            row.get("paths"),
            runs_root=normalized_runs,
            code="login_controller_registry_paths_invalid",
            require_exists=require_paths_exist,
            allowed_missing=(
                frozenset({"invocation_status", "outer_credential_scan"})
                if not require_paths_exist
                else frozenset()
            ),
        )
        normalized = {
            **row,
            "controller": {
                "kind": LOGIN_CONTROLLER_KIND,
                "id": controller_id,
                "pid": int(controller["pid"]),
                "hostname": str(controller["hostname"]),
                "started_at_utc": str(controller["started_at_utc"]),
                "deadline_source": DEADLINE_SOURCE,
            },
            "actor_service": service,
            "immutable_input_bindings": normalized_bindings,
            "paths": paths,
        }
        validated.append(normalized)
        previous = str(row["record_sha256"])
    return validated


def _write_new_json(path: Path, payload: Mapping[str, Any], *, code: str) -> None:
    destination = _reject_symlink_components(path, allow_missing_leaf=True)
    if destination.exists() or destination.is_symlink():
        _fail(code)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail(code)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except FileExistsError:
        _fail(code)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _append_new_registry_row(path: Path, row: Mapping[str, Any]) -> None:
    registry = _reject_symlink_components(path, allow_missing_leaf=True)
    if registry.exists() or registry.is_symlink():
        _fail("login_controller_registry_already_exists")
    registry.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(registry, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(
            fd,
            (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(fd)
    finally:
        os.close(fd)


def create_login_controller_registry(
    *,
    registry_path: Path,
    project_root: Path,
    runs_root: Path,
    study_id: str,
    controller_id: str,
    controller_pid: int,
    controller_hostname: str,
    started_at_utc: str,
    actor_service: Mapping[str, Any],
    immutable_input_paths: Mapping[str, Path],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Write one new registry record before formal runner invocation.

    The caller must create the lock/log/status placeholders first.  This lets
    the record bind every eventual evidence location before a model request is
    possible.  The deliberately one-shot registry avoids ambiguous retry rows.
    """

    normalized_project = _reject_symlink_components(project_root)
    normalized_runs = _reject_symlink_components(runs_root)
    if not isinstance(study_id, str) or not study_id or study_id != study_id.strip():
        _fail("login_controller_registry_study_invalid")
    normalized_controller_id = _validate_controller_id(
        controller_id, code="login_controller_registry_controller_invalid"
    )
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
        or not _HOST_RE.fullmatch(controller_hostname)
    ):
        _fail("login_controller_registry_controller_invalid")
    _parse_utc_timestamp(started_at_utc, code="login_controller_registry_controller_invalid")
    normalized_service = _validate_service_identity(
        actor_service, code="login_controller_registry_service_invalid"
    )
    if set(immutable_input_paths) != _REQUIRED_BINDINGS:
        _fail("login_controller_registry_bindings_incomplete")
    bindings = {
        name: _binding(Path(raw), code="login_controller_registry_binding_invalid", allow_empty=False)
        for name, raw in immutable_input_paths.items()
    }
    for binding in bindings.values():
        if not any(
            _is_relative_to(Path(str(binding["path"])), root)
            for root in (normalized_project, normalized_runs)
        ):
            _fail("login_controller_registry_binding_invalid")
    raw_paths = {name: str(Path(value)) for name, value in paths.items()}
    normalized_paths = _validate_paths(
        raw_paths,
        runs_root=normalized_runs,
        code="login_controller_registry_paths_invalid",
        require_exists=False,
        # These terminal artifacts are intentionally absent at registration;
        # every other registered path must already be a regular file.
        allowed_missing=frozenset({"invocation_status", "outer_credential_scan"}),
    )
    unsigned: dict[str, Any] = {
        "schema_version": LOGIN_CONTROLLER_REGISTRY_SCHEMA,
        "study_id": study_id,
        "controller": {
            "kind": LOGIN_CONTROLLER_KIND,
            "id": normalized_controller_id,
            "pid": controller_pid,
            "hostname": controller_hostname,
            "started_at_utc": started_at_utc,
            "deadline_source": DEADLINE_SOURCE,
        },
        "actor_service": normalized_service,
        "immutable_input_bindings": bindings,
        "paths": normalized_paths,
        "previous_record_sha256": None,
    }
    row = {**unsigned, "record_sha256": _sha256_bytes(_canonical_json_bytes(unsigned))}
    _append_new_registry_row(registry_path, row)
    return row


def write_login_controller_status(
    *,
    registry_path: Path,
    project_root: Path,
    runs_root: Path,
    formal_root: Path,
    status_path: Path,
    runner_exit_status: int,
    service_state_at_finish: str,
    finished_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write one immutable controller terminal status from the durable plan."""

    rows = validate_login_controller_registry(
        registry_path,
        runs_root=runs_root,
        project_root=project_root,
        require_paths_exist=False,
    )
    if len(rows) != 1:
        _fail("login_controller_registry_row_count_invalid")
    row = rows[0]
    # An early launcher failure can occur before the runner creates its root.
    # Preserve a terminal controller record with null progress in that case;
    # terminal postflight still requires a real completed formal root.
    formal = _reject_symlink_components(formal_root, allow_missing_leaf=True)
    if not _is_relative_to(formal, _reject_symlink_components(runs_root)):
        _fail("login_controller_status_formal_root_invalid")
    checkpoint = formal / ".ha_online_resume" / "checkpoint.json"
    if checkpoint.is_file():
        payload = _read_json(checkpoint, code="login_controller_status_checkpoint_invalid")
        if not isinstance(payload, dict):
            _fail("login_controller_status_checkpoint_invalid")
        total = payload.get("total_condition_count")
        terminal = payload.get("terminal_condition_count")
        pending = payload.get("pending_condition_count")
        status_counts = payload.get("status_counts")
    else:
        # Preserve an honest terminal controller record even if startup failed
        # before the matrix could create its first durable checkpoint.  Such a
        # record can never pass terminal postflight.
        total = terminal = pending = None
        status_counts = {}
    checkpoint_shape_valid = (
        isinstance(total, int)
        and total >= 1
        and isinstance(terminal, int)
        and terminal >= 0
        and isinstance(pending, int)
        and pending >= 0
        and isinstance(status_counts, dict)
    )
    if (
        isinstance(runner_exit_status, bool)
        or not isinstance(runner_exit_status, int)
        or runner_exit_status < 0
        or not isinstance(service_state_at_finish, str)
        or not re.fullmatch(r"[A-Z_]+", service_state_at_finish)
        or (checkpoint.is_file() and not checkpoint_shape_valid)
    ):
        _fail("login_controller_status_invalid")
    now = finished_at_utc or _utc_now()
    finished = _parse_utc_timestamp(now, code="login_controller_status_invalid")
    started = _parse_utc_timestamp(
        row["controller"]["started_at_utc"], code="login_controller_status_invalid"
    )
    if finished < started:
        _fail("login_controller_status_invalid")
    clean_counts = {
        "passed": total,
        "failed": 0,
        "failed_validation": 0,
        "skipped_task_failure": 0,
        "skipped_interrupted": 0,
    }
    # Legacy clean checkpoints intentionally omit the typed abstention key.
    # If it is present, it must be an explicit zero for a promotable run.
    if "safety_abstained" in status_counts:
        clean_counts["safety_abstained"] = 0
    raw_formal_complete = (
        checkpoint_shape_valid
        and runner_exit_status == 0
        and terminal == total
        and pending == 0
        and status_counts == clean_counts
        and service_state_at_finish == "RUNNING"
    )
    safety_abstained_count = status_counts.get("safety_abstained", 0)
    itt_counts = {
        "passed": (
            total - safety_abstained_count
            if isinstance(total, int) and isinstance(safety_abstained_count, int)
            else None
        ),
        "failed": 0,
        "failed_validation": 0,
        "skipped_task_failure": 0,
        "skipped_interrupted": 0,
        "safety_abstained": safety_abstained_count,
    }
    itt_fail_closed_complete = (
        checkpoint_shape_valid
        and isinstance(safety_abstained_count, int)
        and safety_abstained_count > 0
        and total is not None
        and status_counts == itt_counts
        and payload.get("itt_fail_closed_candidate_eligible") is True
        and runner_exit_status == 0
        and terminal == total
        and pending == 0
        and service_state_at_finish == "RUNNING"
    )
    formal_complete = raw_formal_complete or itt_fail_closed_complete
    status = {
        "schema_version": LOGIN_CONTROLLER_STATUS_SCHEMA,
        "controller_id": row["controller"]["id"],
        "controller_pid": row["controller"]["pid"],
        "controller_hostname": row["controller"]["hostname"],
        "controller_started_at_utc": row["controller"]["started_at_utc"],
        "controller_finished_at_utc": now,
        "controller_process_finished": True,
        "controller_exit_status": runner_exit_status,
        "launcher_phase": "formal_complete" if formal_complete else "formal_incomplete",
        "resumable_incomplete": not formal_complete,
        "actor_service": {
            **row["actor_service"],
            "observed_state_at_finish": service_state_at_finish,
        },
        "service_process_ownership": EXTERNAL_SERVICE_OWNERSHIP,
        "service_stopped_by_controller": False,
        "total_condition_count": total,
        "terminal_condition_count": terminal,
        "pending_condition_count": pending,
        "status_counts": status_counts,
        # Raw task-success remains ineligible under a typed safety abstention.
        # The v3 fixed-denominator endpoint has its own explicit eligibility.
        "formal_candidate_eligible": raw_formal_complete,
        **(
            {"itt_fail_closed_candidate_eligible": True}
            if itt_fail_closed_complete
            else {}
        ),
        "registry_record_sha256": row["record_sha256"],
    }
    _write_new_json(status_path, status, code="login_controller_status_already_exists")
    return status


def _parse_path_mapping(values: Sequence[str], *, code: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or not name or name in parsed:
            _fail(code)
        parsed[name] = Path(value)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--registry-path", type=Path, required=True)
    register.add_argument("--project-root", type=Path, required=True)
    register.add_argument("--runs-root", type=Path, required=True)
    register.add_argument("--study-id", required=True)
    register.add_argument("--controller-id", required=True)
    register.add_argument("--controller-pid", type=int, required=True)
    register.add_argument("--controller-hostname", required=True)
    register.add_argument("--started-at-utc", required=True)
    register.add_argument("--service-job-id", required=True)
    register.add_argument("--service-node", required=True)
    register.add_argument("--service-base-url", required=True)
    register.add_argument("--service-end-epoch-seconds", type=int, required=True)
    register.add_argument("--invocation-safety-margin-seconds", type=int, required=True)
    register.add_argument("--immutable-input", action="append", default=[])
    register.add_argument("--path", action="append", default=[])
    finish = subparsers.add_parser("finish")
    finish.add_argument("--registry-path", type=Path, required=True)
    finish.add_argument("--project-root", type=Path, required=True)
    finish.add_argument("--runs-root", type=Path, required=True)
    finish.add_argument("--formal-root", type=Path, required=True)
    finish.add_argument("--status-path", type=Path, required=True)
    finish.add_argument("--runner-exit-status", type=int, required=True)
    finish.add_argument("--service-state-at-finish", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "register":
            immutable_inputs = _parse_path_mapping(
                args.immutable_input, code="login_controller_registry_bindings_incomplete"
            )
            paths = _parse_path_mapping(args.path, code="login_controller_registry_paths_invalid")
            create_login_controller_registry(
                registry_path=args.registry_path,
                project_root=args.project_root,
                runs_root=args.runs_root,
                study_id=args.study_id,
                controller_id=args.controller_id,
                controller_pid=args.controller_pid,
                controller_hostname=args.controller_hostname,
                started_at_utc=args.started_at_utc,
                actor_service={
                    "slurm_job_id": args.service_job_id,
                    "slurm_node": args.service_node,
                    "base_url": args.service_base_url,
                    "service_end_epoch_seconds": args.service_end_epoch_seconds,
                    "invocation_deadline_epoch_seconds": args.service_end_epoch_seconds,
                    "invocation_safety_margin_seconds": args.invocation_safety_margin_seconds,
                    "observed_state_at_start": "RUNNING",
                    "process_ownership": EXTERNAL_SERVICE_OWNERSHIP,
                },
                immutable_input_paths=immutable_inputs,
                paths=paths,
            )
        else:
            write_login_controller_status(
                registry_path=args.registry_path,
                project_root=args.project_root,
                runs_root=args.runs_root,
                formal_root=args.formal_root,
                status_path=args.status_path,
                runner_exit_status=args.runner_exit_status,
                service_state_at_finish=args.service_state_at_finish,
            )
    except LoginControllerRegistryError as exc:
        raise SystemExit(f"login-controller registry failed: {exc.code}") from None


if __name__ == "__main__":
    main()
