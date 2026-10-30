"""Durable task-level policy-preparation state for formal HA online runs.

The append-only event history is authoritative.  The expected task set is
frozen as canonical bytes, while the checkpoint is only a verified convenience
snapshot.  Every begun task receives exactly one terminal status and is never
eligible for policy extraction again, including failures and starts recovered
after an interruption.

Callers must hold the formal run-root lifetime lock while model work is active;
``load_policy_preparation_plan`` treats any dangling start as evidence of an
interrupted prior invocation and closes it fail-closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator, Mapping, Sequence


EXPECTED_TASKS_SCHEMA = "ha-policy-preparation-expected-tasks-v1"
EVENT_SCHEMA = "ha-policy-preparation-event-v3"
PLAN_SCHEMA = "ha-policy-preparation-plan-v3"
TERMINAL_STATUSES = ("success", "failure", "skipped_interrupted")

_STATE_DIRECTORY = ".ha_policy_preparation"
_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PUBLIC_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_ERROR_TYPE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_COMMON_EVENT_FIELDS = {
    "schema_version",
    "event",
    "expected_tasks_sha256",
    "task_id",
    "timestamp_utc",
    "previous_event_sha256",
    "event_sha256",
}
_START_EVENT_FIELDS = _COMMON_EVENT_FIELDS | {"artifact_roots"}
_SUCCESS_EVENT_FIELDS = _COMMON_EVENT_FIELDS | {"status", "artifacts"}
_FAILURE_EVENT_FIELDS = _COMMON_EVENT_FIELDS | {
    "status",
    "failure",
    "artifacts",
}
_FAILURE_FIELDS = {"stage", "error_type", "reason_code"}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("policy preparation event timestamp must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("policy preparation event timestamp must be UTC") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("policy preparation event timestamp must be UTC")
    return value


def _validate_task_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _TASK_ID_RE.fullmatch(value)
        or "/" in value
        or "\\" in value
        or "." in value
        or Path(value).name != value
        or Path(value).stem != value
    ):
        raise ValueError(
            "formal policy preparation task must be a stable task ID, not a path"
        )
    return value


def _normalize_expected_task_ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("expected task set must be a nonempty task-ID sequence")
    try:
        task_ids = [_validate_task_id(value) for value in values]
    except TypeError as exc:
        raise ValueError("expected task set must be a nonempty task-ID sequence") from exc
    if not task_ids:
        raise ValueError("expected task set must be a nonempty task-ID sequence")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("expected task set contains duplicate task IDs")
    return sorted(task_ids)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path) -> Path:
    absolute = _lexical_absolute(path)
    for candidate in [*reversed(absolute.parents), absolute]:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("policy preparation path contains a symlink component")
    return absolute


def _run_root(path: Path) -> Path:
    root = _reject_symlink_components(path)
    if not root.is_dir():
        raise ValueError("policy preparation run_root must be an existing real directory")
    return root.resolve(strict=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_descendant(
    path: Path,
    *,
    run_root: Path,
    allow_root: bool = False,
) -> Path:
    root = _run_root(run_root)
    absolute = _reject_symlink_components(path)
    if not _is_relative_to(absolute, root) or (absolute == root and not allow_root):
        raise ValueError("policy preparation path is outside run_root")
    return absolute


def _ensure_directory(path: Path, *, run_root: Path) -> Path:
    root = _run_root(run_root)
    absolute = _safe_descendant(path, run_root=root)
    current = root
    for component in absolute.relative_to(root).parts:
        current = current / component
        _reject_symlink_components(current)
        current.mkdir(exist_ok=True, mode=0o700)
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("policy preparation state parent must be a real directory")
    return absolute


def _state_paths(run_root: Path) -> dict[str, Path]:
    root = _run_root(run_root)
    state_root = root / _STATE_DIRECTORY
    return {
        "root": state_root,
        "expected": state_root / "expected_tasks.json",
        "history": state_root / "task_history.jsonl",
        "checkpoint": state_root / "checkpoint.json",
        "lock": state_root / "active.lock",
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(
    path: Path,
    value: Any,
    *,
    run_root: Path,
    exclusive: bool,
) -> None:
    destination = _safe_descendant(path, run_root=run_root)
    _ensure_directory(destination.parent, run_root=run_root)
    encoded = _canonical_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_descendant(temporary, run_root=run_root)
        if exclusive:
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise ValueError(
                    "policy preparation frozen state already exists"
                ) from exc
            temporary.unlink()
            temporary = None
        else:
            _reject_symlink_components(destination)
            os.replace(temporary, destination)
            temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_if_changed(path: Path, value: Any, *, run_root: Path) -> None:
    destination = _safe_descendant(path, run_root=run_root)
    encoded = _canonical_bytes(value)
    if destination.exists():
        if not stat.S_ISREG(destination.lstat().st_mode):
            raise ValueError("policy preparation checkpoint must be a regular file")
        if destination.read_bytes() == encoded:
            return
    _write_atomic(destination, value, run_root=run_root, exclusive=False)


def _read_canonical(path: Path, *, label: str, run_root: Path) -> Any:
    absolute = _safe_descendant(path, run_root=run_root)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"policy preparation {label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"policy preparation {label} must be a regular file")
    raw = absolute.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"policy preparation {label} is invalid JSON") from exc
    if raw != _canonical_bytes(value):
        raise ValueError(f"policy preparation {label} canonical byte drift")
    return value


def _expected_manifest(task_ids: Sequence[str]) -> dict[str, Any]:
    normalized = list(task_ids)
    return {
        "schema_version": EXPECTED_TASKS_SCHEMA,
        "task_ids": normalized,
        "task_set_sha256": _canonical_sha256(normalized),
    }


def _validate_expected_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "task_ids",
        "task_set_sha256",
    }:
        raise ValueError("policy preparation expected task manifest schema drift")
    if raw["schema_version"] != EXPECTED_TASKS_SCHEMA:
        raise ValueError("policy preparation expected task manifest schema drift")
    if not isinstance(raw["task_ids"], list):
        raise ValueError("policy preparation expected task manifest is invalid")
    task_ids = _normalize_expected_task_ids(raw["task_ids"])
    if raw["task_ids"] != task_ids:
        raise ValueError("policy preparation expected task manifest is not canonical")
    if raw["task_set_sha256"] != _canonical_sha256(task_ids):
        raise ValueError("policy preparation expected task set hash drift")
    return _expected_manifest(task_ids)


def _event_sha256(event: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    return _canonical_sha256(unsigned)


def _validate_public_failure(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != _FAILURE_FIELDS:
        raise ValueError(
            "policy preparation public failure record has missing or unknown fields"
        )
    stage = raw.get("stage")
    error_type = raw.get("error_type")
    reason_code = raw.get("reason_code")
    if not isinstance(stage, str) or not _PUBLIC_CODE_RE.fullmatch(stage):
        raise ValueError("policy preparation public failure record stage is invalid")
    if not isinstance(error_type, str) or not _ERROR_TYPE_RE.fullmatch(error_type):
        raise ValueError(
            "policy preparation public failure record error_type is invalid"
        )
    if not isinstance(reason_code, str) or not _PUBLIC_CODE_RE.fullmatch(reason_code):
        raise ValueError(
            "policy preparation public failure record reason_code is invalid"
        )
    return {
        "stage": stage,
        "error_type": error_type,
        "reason_code": reason_code,
    }


def _artifact_binding(path: Path, *, run_root: Path) -> dict[str, Any]:
    root = _run_root(run_root)
    absolute = _safe_descendant(path, run_root=root)
    state_root = root / _STATE_DIRECTORY
    if _is_relative_to(absolute, state_root):
        raise ValueError("policy preparation artifact cannot be a state file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(absolute, flags)
    except FileNotFoundError as exc:
        raise ValueError("policy preparation artifact is missing") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("policy preparation artifact cannot be a symlink") from exc
        raise
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("policy preparation artifact must be a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError("policy preparation artifact changed while hashing")
    finally:
        os.close(fd)
    return {
        "path": str(absolute),
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_artifact_binding(raw: Any, *, run_root: Path) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
        raise ValueError("policy preparation artifact binding schema drift")
    path = raw.get("path")
    size = raw.get("size")
    digest = raw.get("sha256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("policy preparation artifact path must be absolute")
    if type(size) is not int or size < 0:
        raise ValueError("policy preparation artifact size is invalid")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("policy preparation artifact hash is invalid")
    observed = _artifact_binding(Path(path), run_root=run_root)
    if observed != raw:
        raise ValueError("policy preparation artifact bytes or hash drift")
    return observed


def _artifact_root_binding(
    path: Path,
    *,
    run_root: Path,
    task_id: str,
    require_empty: bool,
) -> dict[str, str]:
    root = _run_root(run_root)
    absolute = _safe_descendant(path, run_root=root)
    state_root = root / _STATE_DIRECTORY
    if _is_relative_to(absolute, state_root):
        raise ValueError("policy preparation artifact root cannot be a state path")
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError("policy preparation artifact root is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("policy preparation artifact root must be a real directory")
    if absolute.name != task_id:
        raise ValueError(
            "policy preparation artifact root must be task-scoped by task ID"
        )
    if require_empty and any(absolute.iterdir()):
        raise ValueError("policy preparation artifact root must be empty at begin")
    return {"path": str(absolute)}


def _validate_artifact_root_binding(
    raw: Any,
    *,
    run_root: Path,
    task_id: str,
) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != {"path"}:
        raise ValueError("policy preparation artifact root binding schema drift")
    path = raw.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("policy preparation artifact root path must be absolute")
    return _artifact_root_binding(
        Path(path),
        run_root=run_root,
        task_id=task_id,
        require_empty=False,
    )


def _validate_artifact_roots(
    raw_roots: Any,
    *,
    run_root: Path,
    task_id: str,
    require_empty: bool,
) -> list[dict[str, str]]:
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError(
            "policy preparation start must freeze one or more artifact roots"
        )
    roots = [
        (
            _artifact_root_binding(
                Path(raw["path"]),
                run_root=run_root,
                task_id=task_id,
                require_empty=require_empty,
            )
            if isinstance(raw, dict) and set(raw) == {"path"}
            else _validate_artifact_root_binding(
                raw,
                run_root=run_root,
                task_id=task_id,
            )
        )
        for raw in raw_roots
    ]
    paths = [Path(binding["path"]) for binding in roots]
    if [str(path) for path in paths] != sorted(str(path) for path in paths):
        if not require_empty:
            raise ValueError("policy preparation artifact roots are not canonical")
        roots.sort(key=lambda binding: binding["path"])
        paths = [Path(binding["path"]) for binding in roots]
    if len(paths) != len(set(paths)):
        raise ValueError("policy preparation artifact roots contain duplicates")
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if _is_relative_to(other, path) or _is_relative_to(path, other):
                raise ValueError("policy preparation artifact roots overlap")
    return roots


def _artifact_inventory(
    root_bindings: Sequence[Mapping[str, Any]],
    *,
    run_root: Path,
    task_id: str,
) -> list[dict[str, Any]]:
    roots = _validate_artifact_roots(
        [dict(binding) for binding in root_bindings],
        run_root=run_root,
        task_id=task_id,
        require_empty=False,
    )
    files: list[Path] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    "policy preparation artifact inventory contains a symlink"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
            else:
                raise ValueError(
                    "policy preparation artifact inventory contains a special file"
                )

    for binding in roots:
        visit(Path(binding["path"]))
    artifacts = [_artifact_binding(path, run_root=run_root) for path in files]
    artifacts.sort(key=lambda item: item["path"])
    return artifacts


def _base_event(
    *,
    name: str,
    task_id: str,
    expected_tasks_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA,
        "event": name,
        "expected_tasks_sha256": expected_tasks_sha256,
        "task_id": task_id,
        "timestamp_utc": _timestamp_utc(),
    }


def _validate_event(
    raw: Any,
    *,
    expected_task_ids: set[str],
    expected_tasks_sha256: str,
    previous_event_sha256: str,
    run_root: Path,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("policy preparation history event must be an object")
    event = dict(raw)
    name = event.get("event")
    status = event.get("status")
    if name == "task_started":
        fields = _START_EVENT_FIELDS
    elif name == "task_terminal" and status == "success":
        fields = _SUCCESS_EVENT_FIELDS
    elif name == "task_terminal" and status in {"failure", "skipped_interrupted"}:
        fields = _FAILURE_EVENT_FIELDS
    else:
        raise ValueError("policy preparation history event type is invalid")
    if set(event) != fields:
        raise ValueError("policy preparation history event schema drift")
    if event["schema_version"] != EVENT_SCHEMA:
        raise ValueError("policy preparation history event schema drift")
    task_id = _validate_task_id(event["task_id"])
    if task_id not in expected_task_ids:
        raise ValueError("policy preparation history contains an unknown task")
    if event["expected_tasks_sha256"] != expected_tasks_sha256:
        raise ValueError("policy preparation history expected-task hash drift")
    _validate_timestamp(event["timestamp_utc"])
    if event["previous_event_sha256"] != previous_event_sha256:
        raise ValueError("policy preparation history hash chain is broken")
    if (
        not isinstance(event["event_sha256"], str)
        or not _SHA256_RE.fullmatch(event["event_sha256"])
        or event["event_sha256"] != _event_sha256(event)
    ):
        raise ValueError("policy preparation history event hash drift")
    if name == "task_terminal":
        artifacts = event["artifacts"]
        if not isinstance(artifacts, list) or (status == "success" and not artifacts):
            raise ValueError(
                "policy preparation terminal artifact list is invalid"
            )
        event["artifacts"] = [
            _validate_artifact_binding(binding, run_root=run_root)
            for binding in artifacts
        ]
        paths = [binding["path"] for binding in event["artifacts"]]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("policy preparation artifact bindings are not canonical")
        if status != "success":
            event["failure"] = _validate_public_failure(event["failure"])
    else:
        event["artifact_roots"] = _validate_artifact_roots(
            event["artifact_roots"],
            run_root=run_root,
            task_id=task_id,
            require_empty=False,
        )
    return event


def _read_history(
    path: Path,
    *,
    expected_task_ids: Sequence[str],
    expected_tasks_sha256: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    absolute = _safe_descendant(path, run_root=run_root)
    if not absolute.exists():
        return []
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise ValueError("policy preparation history must be a regular file")
    events: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    with absolute.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(
                    f"policy preparation history line {line_number} is blank"
                )
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"policy preparation history line {line_number} is invalid JSON"
                ) from exc
            if raw_line != _canonical_bytes(value):
                raise ValueError("policy preparation history canonical byte drift")
            event = _validate_event(
                value,
                expected_task_ids=set(expected_task_ids),
                expected_tasks_sha256=expected_tasks_sha256,
                previous_event_sha256=previous,
                run_root=run_root,
            )
            events.append(event)
            previous = event["event_sha256"]
    active_start: dict[str, Any] | None = None
    for event in events:
        if event["event"] == "task_started":
            active_start = event
            continue
        if active_start is None or active_start["task_id"] != event["task_id"]:
            raise ValueError(
                "policy preparation terminal lacks its artifact-root start"
            )
        observed = _artifact_inventory(
            active_start["artifact_roots"],
            run_root=run_root,
            task_id=event["task_id"],
        )
        if event["artifacts"] != observed:
            raise ValueError(
                "policy preparation terminal artifact inventory drift"
            )
        active_start = None
    return events


def _reduce_history(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    terminal_by_task: dict[str, dict[str, Any]] = {}
    active: dict[str, Any] | None = None
    for raw_event in events:
        event = dict(raw_event)
        task_id = event["task_id"]
        if event["event"] == "task_started":
            if active is not None:
                raise ValueError("policy preparation history has concurrent task starts")
            if task_id in terminal_by_task:
                raise ValueError("policy preparation task was retried after terminal")
            active = event
            continue
        if active is None or active["task_id"] != task_id:
            raise ValueError("policy preparation terminal lacks its matching start")
        if task_id in terminal_by_task:
            raise ValueError("policy preparation task has duplicate terminals")
        terminal_by_task[task_id] = event
        active = None
    return terminal_by_task, active


def _plan(
    *,
    expected_manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    terminal_by_task: Mapping[str, Mapping[str, Any]],
    active: Mapping[str, Any] | None,
) -> dict[str, Any]:
    task_ids = list(expected_manifest["task_ids"])
    terminal = {
        task_id: dict(terminal_by_task[task_id])
        for task_id in task_ids
        if task_id in terminal_by_task
    }
    status_counts = {status: 0 for status in TERMINAL_STATUSES}
    for event in terminal.values():
        status_counts[event["status"]] += 1
    active_task_id = active["task_id"] if active is not None else None
    pending = [
        task_id
        for task_id in task_ids
        if task_id not in terminal and task_id != active_task_id
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "expected_tasks_sha256": _canonical_sha256(expected_manifest),
        "total_task_count": len(task_ids),
        "terminal_task_count": len(terminal),
        "pending_task_count": len(pending),
        "status_counts": status_counts,
        "active_task_id": active_task_id,
        "successful_task_ids": [
            task_id
            for task_id in task_ids
            if task_id in terminal and terminal[task_id]["status"] == "success"
        ],
        "failed_task_ids": [
            task_id
            for task_id in task_ids
            if task_id in terminal and terminal[task_id]["status"] == "failure"
        ],
        "interrupted_task_ids": [
            task_id
            for task_id in task_ids
            if task_id in terminal
            and terminal[task_id]["status"] == "skipped_interrupted"
        ],
        "pending_task_ids": pending,
        "terminal_by_task": terminal,
        "history_line_count": len(events),
        "history_tail_sha256": (
            events[-1]["event_sha256"] if events else _ZERO_SHA256
        ),
    }


def _append_event(
    path: Path,
    event: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    run_root: Path,
) -> dict[str, Any]:
    destination = _safe_descendant(path, run_root=run_root)
    _ensure_directory(destination.parent, run_root=run_root)
    chained = {
        **event,
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else _ZERO_SHA256
        ),
    }
    chained["event_sha256"] = _event_sha256(chained)
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as handle:
            handle.write(_canonical_bytes(chained))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    _fsync_directory(destination.parent)
    events.append(chained)
    return chained


@contextmanager
def _state_lock(run_root: Path) -> Iterator[tuple[Path, dict[str, Path]]]:
    root = _run_root(run_root)
    paths = _state_paths(root)
    _ensure_directory(paths["root"], run_root=root)
    lock_path = _safe_descendant(paths["lock"], run_root=root)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield root, paths
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _freeze_or_load_expected(
    *,
    root: Path,
    paths: Mapping[str, Path],
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    expected = _expected_manifest(expected_task_ids)
    if not paths["expected"].exists():
        _write_atomic(
            paths["expected"],
            expected,
            run_root=root,
            exclusive=True,
        )
    frozen = _validate_expected_manifest(
        _read_canonical(
            paths["expected"],
            label="expected task manifest",
            run_root=root,
        )
    )
    if frozen != expected:
        raise ValueError("policy preparation expected task set drift")
    return frozen


def _verified_state(
    *,
    root: Path,
    paths: Mapping[str, Path],
    expected_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any] | None]:
    expected_sha256 = _canonical_sha256(expected_manifest)
    events = _read_history(
        paths["history"],
        expected_task_ids=expected_manifest["task_ids"],
        expected_tasks_sha256=expected_sha256,
        run_root=root,
    )
    terminal, active = _reduce_history(events)
    if not paths["checkpoint"].exists():
        if events:
            raise ValueError("policy preparation checkpoint is missing for history")
        _write_atomic(
            paths["checkpoint"],
            _plan(
                expected_manifest=expected_manifest,
                events=events,
                terminal_by_task=terminal,
                active=active,
            ),
            run_root=root,
            exclusive=True,
        )
        return events, terminal, active
    checkpoint = _read_canonical(
        paths["checkpoint"],
        label="checkpoint",
        run_root=root,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("policy preparation checkpoint must be an object")
    checkpoint_lines = checkpoint.get("history_line_count")
    if (
        type(checkpoint_lines) is not int
        or checkpoint_lines < 0
        or checkpoint_lines > len(events)
    ):
        raise ValueError("policy preparation history rollback or checkpoint drift")
    prefix = events[:checkpoint_lines]
    prefix_terminal, prefix_active = _reduce_history(prefix)
    expected_checkpoint = _plan(
        expected_manifest=expected_manifest,
        events=prefix,
        terminal_by_task=prefix_terminal,
        active=prefix_active,
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("policy preparation checkpoint or history prefix drift")
    return events, terminal, active


def _write_checkpoint(
    *,
    root: Path,
    paths: Mapping[str, Path],
    expected_manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal, active = _reduce_history(events)
    plan = _plan(
        expected_manifest=expected_manifest,
        events=events,
        terminal_by_task=terminal,
        active=active,
    )
    _write_if_changed(paths["checkpoint"], plan, run_root=root)
    return plan


def load_policy_preparation_plan(
    *,
    run_root: Path,
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze/verify the task set and recover a dangling start fail-closed."""

    normalized = _normalize_expected_task_ids(expected_task_ids)
    with _state_lock(run_root) as (root, paths):
        expected = _freeze_or_load_expected(
            root=root,
            paths=paths,
            expected_task_ids=normalized,
        )
        events, terminal, active = _verified_state(
            root=root,
            paths=paths,
            expected_manifest=expected,
        )
        if active is not None:
            partial_artifacts = _artifact_inventory(
                active["artifact_roots"],
                run_root=root,
                task_id=active["task_id"],
            )
            interrupted = {
                **_base_event(
                    name="task_terminal",
                    task_id=active["task_id"],
                    expected_tasks_sha256=_canonical_sha256(expected),
                ),
                "status": "skipped_interrupted",
                "failure": {
                    "stage": "policy_preparation",
                    "error_type": "InterruptedPolicyPreparation",
                    "reason_code": "dangling_started_recovered",
                },
                "artifacts": partial_artifacts,
            }
            _append_event(
                paths["history"],
                interrupted,
                events=events,
                run_root=root,
            )
        return _write_checkpoint(
            root=root,
            paths=paths,
            expected_manifest=expected,
            events=events,
        )


def begin_policy_preparation(
    *,
    run_root: Path,
    expected_task_ids: Sequence[str],
    task_id: str,
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    """Durably begin one task with empty, frozen task-scoped artifact roots."""

    normalized = _normalize_expected_task_ids(expected_task_ids)
    validated_task_id = _validate_task_id(task_id)
    if validated_task_id not in set(normalized):
        raise ValueError("policy preparation requested an unknown task")
    if isinstance(artifact_roots, (str, bytes)):
        raise ValueError("policy preparation artifact roots must be a sequence")
    with _state_lock(run_root) as (root, paths):
        expected = _freeze_or_load_expected(
            root=root,
            paths=paths,
            expected_task_ids=normalized,
        )
        events, terminal, active = _verified_state(
            root=root,
            paths=paths,
            expected_manifest=expected,
        )
        if validated_task_id in terminal:
            raise ValueError("policy preparation task is already terminal")
        if active is not None:
            if active["task_id"] == validated_task_id:
                raise ValueError("policy preparation task is already active")
            raise ValueError("another policy preparation task is already active")
        roots = _validate_artifact_roots(
            [{"path": str(_lexical_absolute(Path(path)))} for path in artifact_roots],
            run_root=root,
            task_id=validated_task_id,
            require_empty=True,
        )
        event = _base_event(
            name="task_started",
            task_id=validated_task_id,
            expected_tasks_sha256=_canonical_sha256(expected),
        )
        event["artifact_roots"] = roots
        persisted = _append_event(
            paths["history"],
            event,
            events=events,
            run_root=root,
        )
        _write_checkpoint(
            root=root,
            paths=paths,
            expected_manifest=expected,
            events=events,
        )
        return persisted


def terminal_policy_preparation_success(
    *,
    run_root: Path,
    expected_task_ids: Sequence[str],
    task_id: str,
    artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    """Hash-bind one or more artifacts and terminalize the active task."""

    normalized = _normalize_expected_task_ids(expected_task_ids)
    validated_task_id = _validate_task_id(task_id)
    if validated_task_id not in set(normalized):
        raise ValueError("policy preparation requested an unknown task")
    if isinstance(artifact_paths, (str, bytes)):
        raise ValueError(
            "successful policy preparation must bind one or more artifacts"
        )
    paths_to_bind = [Path(path) for path in artifact_paths]
    if not paths_to_bind:
        raise ValueError(
            "successful policy preparation must bind one or more artifacts"
        )
    with _state_lock(run_root) as (root, paths):
        expected = _freeze_or_load_expected(
            root=root,
            paths=paths,
            expected_task_ids=normalized,
        )
        events, terminal, active = _verified_state(
            root=root,
            paths=paths,
            expected_manifest=expected,
        )
        if validated_task_id in terminal:
            raise ValueError("policy preparation task is already terminal")
        if active is None or active["task_id"] != validated_task_id:
            raise ValueError("policy preparation success lacks its matching start")
        artifacts = _artifact_inventory(
            active["artifact_roots"],
            run_root=root,
            task_id=validated_task_id,
        )
        supplied_paths = sorted(str(_lexical_absolute(path)) for path in paths_to_bind)
        observed_paths = [binding["path"] for binding in artifacts]
        if supplied_paths != observed_paths:
            raise ValueError(
                "policy preparation success must bind the complete frozen-root "
                "artifact inventory"
            )
        event = {
            **_base_event(
                name="task_terminal",
                task_id=validated_task_id,
                expected_tasks_sha256=_canonical_sha256(expected),
            ),
            "status": "success",
            "artifacts": artifacts,
        }
        persisted = _append_event(
            paths["history"],
            event,
            events=events,
            run_root=root,
        )
        _write_checkpoint(
            root=root,
            paths=paths,
            expected_manifest=expected,
            events=events,
        )
        return persisted


def terminal_policy_preparation_failure(
    *,
    run_root: Path,
    expected_task_ids: Sequence[str],
    task_id: str,
    failure: Mapping[str, Any],
    artifact_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Terminalize the active task with code-only failure and bound artifacts."""

    normalized = _normalize_expected_task_ids(expected_task_ids)
    validated_task_id = _validate_task_id(task_id)
    if validated_task_id not in set(normalized):
        raise ValueError("policy preparation requested an unknown task")
    public_failure = _validate_public_failure(failure)
    if isinstance(artifact_paths, (str, bytes)):
        raise ValueError("policy preparation artifact paths must be a sequence")
    paths_to_bind = [Path(path) for path in artifact_paths]
    with _state_lock(run_root) as (root, paths):
        expected = _freeze_or_load_expected(
            root=root,
            paths=paths,
            expected_task_ids=normalized,
        )
        events, terminal, active = _verified_state(
            root=root,
            paths=paths,
            expected_manifest=expected,
        )
        if validated_task_id in terminal:
            raise ValueError("policy preparation task is already terminal")
        if active is None or active["task_id"] != validated_task_id:
            raise ValueError("policy preparation failure lacks its matching start")
        artifacts = _artifact_inventory(
            active["artifact_roots"],
            run_root=root,
            task_id=validated_task_id,
        )
        supplied_paths = sorted(str(_lexical_absolute(path)) for path in paths_to_bind)
        observed_paths = [binding["path"] for binding in artifacts]
        if supplied_paths != observed_paths:
            raise ValueError(
                "policy preparation failure must bind the complete frozen-root "
                "artifact inventory"
            )
        event = {
            **_base_event(
                name="task_terminal",
                task_id=validated_task_id,
                expected_tasks_sha256=_canonical_sha256(expected),
            ),
            "status": "failure",
            "failure": public_failure,
            "artifacts": artifacts,
        }
        persisted = _append_event(
            paths["history"],
            event,
            events=events,
            run_root=root,
        )
        _write_checkpoint(
            root=root,
            paths=paths,
            expected_manifest=expected,
            events=events,
        )
        return persisted


__all__ = [
    "begin_policy_preparation",
    "load_policy_preparation_plan",
    "terminal_policy_preparation_failure",
    "terminal_policy_preparation_success",
]
