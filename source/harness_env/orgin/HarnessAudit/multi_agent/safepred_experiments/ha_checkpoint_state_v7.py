"""Pure, model-free replay-unit checkpoint state for HarnessAudit v7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Literal

from .ha_resilient_predict_only import _sanitize_persisted_text
from .predict_only_pipeline import _replay_job_horizons


_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROLLOUT_MODES = {"direct_multi_horizon", "iterative_wm_rollout"}
_POLICY_MODES = {"train_policy", "full_harnessaudit_rules"}
_EVENT_FIELDS = {
    "TRACE_ATTEMPT_START": set(),
    "UNIT_START": {"unit_key"},
    "UNIT_PASS": {"unit_key", "artifact"},
    "UNIT_PAUSE": {"unit_key", "failure_kind"},
    "TRACE_PAUSE": {"failure_kind"},
    "TRACE_COMPLETE": set(),
    "TRACE_TERMINAL_MODEL_INVALID": {"failure_kind"},
}
_COMMON_INPUT_FIELDS = {
    "schema_version",
    "event",
    "manifest_sha256",
    "run_id",
    "task_id",
    "source_index",
    "attempt",
    "timestamp_utc",
}

RecoveryStatus = Literal[
    "pending",
    "paused",
    "active-without-outcome",
    "complete",
    "terminal-model-invalid",
    "integrity-failed",
]


@dataclass(frozen=True)
class ReplayUnit:
    rollout_mode: str
    policy_mode: str
    horizon: int

    @property
    def key(self) -> str:
        return f"{self.rollout_mode}|{self.policy_mode}|h{self.horizon:02d}"


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not Path(self.path).is_absolute():
            raise ValueError("artifact path must be absolute")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("artifact size must be a nonnegative integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must be lowercase 64-hex")


@dataclass(frozen=True)
class TaskRecoveryState:
    manifest_sha256: str
    run_id: str
    task_id: str
    source_index: int
    status: RecoveryStatus
    attempt_count: int
    completed_unit_keys: tuple[str, ...]
    next_unit: ReplayUnit | None
    last_failure_kind: str | None
    history_line_count: int
    history_tail_sha256: str


def build_replay_unit_plan(
    *,
    rollout_modes: tuple[str, ...],
    policy_modes: tuple[str, ...],
    horizons: tuple[int, ...],
) -> tuple[ReplayUnit, ...]:
    """Expand the frozen manifest order into endpoint replay units."""

    if not rollout_modes or not policy_modes or not horizons:
        raise ValueError("replay unit inputs must not be empty")
    if len(set(rollout_modes)) != len(rollout_modes):
        raise ValueError("rollout modes must not contain duplicates")
    if len(set(policy_modes)) != len(policy_modes):
        raise ValueError("policy modes must not contain duplicates")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in horizons):
        raise ValueError("horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must not contain duplicates")
    if any(mode not in _ROLLOUT_MODES for mode in rollout_modes):
        raise ValueError("unsupported rollout mode")
    if any(mode not in _POLICY_MODES for mode in policy_modes):
        raise ValueError("unsupported policy mode")

    units: list[ReplayUnit] = []
    for rollout_mode in rollout_modes:
        job_horizons = _replay_job_horizons(rollout_mode, horizons)
        for policy_mode in policy_modes:
            units.extend(
                ReplayUnit(rollout_mode, policy_mode, horizon)
                for horizon in job_horizons
            )
    if len({unit.key for unit in units}) != len(units):
        raise ValueError("replay unit plan contains duplicate keys")
    return tuple(units)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event_sha256(event: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp_utc must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp_utc must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp_utc must be a UTC timestamp")


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
            raise ValueError("run path contains a symlink component")
    return absolute


def _run_root(run_root: Path) -> Path:
    root = _assert_no_symlink_components(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing real directory")
    return root.resolve(strict=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_run_path(path: Path, *, run_root: Path) -> Path:
    root = _run_root(run_root)
    absolute = _assert_no_symlink_components(path)
    if not _is_relative_to(absolute, root):
        raise ValueError("run path is outside run_root")
    return absolute


def _ensure_safe_parent(path: Path, *, run_root: Path) -> Path:
    root = _run_root(run_root)
    parent = _lexical_absolute(path).parent
    if not _is_relative_to(parent, root):
        raise ValueError("run path is outside run_root")
    current = root
    for part in parent.relative_to(root).parts:
        current = current / part
        _assert_no_symlink_components(current)
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError("run path parent must be a directory")
    return _safe_run_path(path, run_root=root)


def _sanitize_tree(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_persisted_text(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_tree(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_tree(item) for item in value]
    return value


def _validate_artifact(raw: Any, *, run_root: Path | None) -> ArtifactBinding:
    if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
        raise ValueError("artifact binding has missing or unknown fields")
    binding = ArtifactBinding(
        path=raw.get("path"),
        size=raw.get("size"),
        sha256=raw.get("sha256"),
    )
    if run_root is None:
        return binding
    absolute = _safe_run_path(Path(binding.path), run_root=run_root)
    if str(absolute) != binding.path:
        raise ValueError("artifact path must be canonical and absolute")
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError("artifact must be a regular file") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("artifact must be a regular non-symlink file")
    digest = hashlib.sha256(absolute.read_bytes()).hexdigest()
    if metadata.st_size != binding.size or digest != binding.sha256:
        raise ValueError("artifact binding does not match current bytes")
    return binding


def _validate_event_scalars(event: dict[str, Any]) -> None:
    if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    name = event.get("event")
    if not isinstance(name, str) or name not in _EVENT_FIELDS:
        raise ValueError("unknown event")
    if not isinstance(event.get("manifest_sha256"), str) or not _SHA256_RE.fullmatch(event["manifest_sha256"]):
        raise ValueError("manifest_sha256 must be lowercase 64-hex")
    if any(not isinstance(event.get(field), str) or not event[field] for field in ("run_id", "task_id")):
        raise ValueError("run_id and task_id must be nonempty strings")
    if isinstance(event.get("source_index"), bool) or not isinstance(event.get("source_index"), int) or event["source_index"] < 0:
        raise ValueError("source_index must be a nonnegative integer")
    if isinstance(event.get("attempt"), bool) or not isinstance(event.get("attempt"), int) or event["attempt"] <= 0:
        raise ValueError("attempt must be a positive integer")
    _validate_timestamp(event.get("timestamp_utc"))
    if "unit_key" in event and (not isinstance(event["unit_key"], str) or not event["unit_key"]):
        raise ValueError("unit_key must be a nonempty string")
    if "failure_kind" in event and (not isinstance(event["failure_kind"], str) or not event["failure_kind"]):
        raise ValueError("failure_kind must be a nonempty string")


def _validate_input_event(event: Any, *, run_root: Path) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    name = event.get("event")
    expected = _COMMON_INPUT_FIELDS | _EVENT_FIELDS.get(name, set())
    if name not in _EVENT_FIELDS or set(event) != expected:
        raise ValueError("event has missing or unknown fields")
    sanitized = _sanitize_tree(event)
    _validate_event_scalars(sanitized)
    if name == "UNIT_PASS":
        _validate_artifact(sanitized["artifact"], run_root=run_root)
    return sanitized


def _validate_persisted_event(
    event: Any,
    *,
    previous_event_sha256: str,
    run_root: Path | None,
) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("history event must be a JSON object")
    name = event.get("event")
    expected = _COMMON_INPUT_FIELDS | _EVENT_FIELDS.get(name, set()) | {
        "previous_event_sha256",
        "event_sha256",
    }
    if name not in _EVENT_FIELDS or set(event) != expected:
        raise ValueError("history event has missing or unknown fields")
    _validate_event_scalars(event)
    if event.get("previous_event_sha256") != previous_event_sha256:
        raise ValueError("history hash chain is broken")
    if not isinstance(event.get("event_sha256"), str) or not _SHA256_RE.fullmatch(event["event_sha256"]):
        raise ValueError("event_sha256 must be lowercase 64-hex")
    if _event_sha256(event) != event["event_sha256"]:
        raise ValueError("history event hash does not match content")
    if name == "UNIT_PASS":
        _validate_artifact(event["artifact"], run_root=run_root)
    return event


def _lock_path(history_path: Path) -> Path:
    return history_path.with_name(f".{history_path.name}.lock")


def _open_lock(path: Path, *, run_root: Path) -> int:
    lock_path = _ensure_safe_parent(_lock_path(path), run_root=run_root)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(lock_path, flags, 0o600)


def _read_unlocked(history_path: Path, *, run_root: Path) -> list[dict[str, Any]]:
    path = _safe_run_path(history_path, run_root=run_root)
    if not path.exists():
        return []
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("task history must be a regular non-symlink file")
    events: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"task history line {line_number} is blank")
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"task history line {line_number} is invalid JSON") from exc
            validated = _validate_persisted_event(
                event,
                previous_event_sha256=previous,
                run_root=run_root,
            )
            events.append(validated)
            previous = validated["event_sha256"]
    return events


def append_task_event(
    history_path: Path,
    event: dict[str, Any],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Validate, chain, lock, append, flush, and fsync one canonical event."""

    path = _ensure_safe_parent(history_path, run_root=run_root)
    lock_fd = _open_lock(path, run_root=run_root)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        events = _read_unlocked(path, run_root=run_root)
        return _append_task_event_unlocked(
            path,
            event,
            events=events,
            run_root=run_root,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _append_task_event_unlocked(
    history_path: Path,
    event: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    run_root: Path,
) -> dict[str, Any]:
    """Append one event while the caller holds the history exclusive lock."""

    path = _safe_run_path(history_path, run_root=run_root)
    validated = _validate_input_event(event, run_root=run_root)
    chained = {
        **validated,
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else _ZERO_SHA256
        ),
    }
    chained["event_sha256"] = _event_sha256(chained)
    _validate_persisted_event(
        chained,
        previous_event_sha256=chained["previous_event_sha256"],
        run_root=run_root,
    )
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(_canonical_bytes(chained).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    events.append(chained)
    return chained


def read_task_history(history_path: Path, *, run_root: Path) -> list[dict[str, Any]]:
    """Read and revalidate a complete hash-chained task history."""

    path = _ensure_safe_parent(history_path, run_root=run_root)
    lock_fd = _open_lock(path, run_root=run_root)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        return _read_unlocked(path, run_root=run_root)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def reduce_task_history(
    events: list[dict[str, Any]],
    *,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    source_index: int,
    run_root: Path,
) -> TaskRecoveryState:
    """Reduce a verified event stream without skipping an incomplete unit."""

    _run_root(run_root)
    if not unit_plan or len({unit.key for unit in unit_plan}) != len(unit_plan):
        raise ValueError("unit plan must be nonempty with unique keys")
    expected_identity = {
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task_id,
        "source_index": source_index,
    }
    completed: list[str] = []
    attempt_count = 0
    active_attempt = False
    active_unit: str | None = None
    unit_paused = False
    status: RecoveryStatus = "pending"
    last_failure: str | None = None
    terminal = False
    previous = _ZERO_SHA256

    for event in events:
        event = _validate_persisted_event(
            event,
            previous_event_sha256=previous,
            run_root=run_root,
        )
        previous = event["event_sha256"]
        if any(event.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("task history identity does not match frozen identity")
        if terminal:
            raise ValueError("task history contains an event after terminal trace")
        name = event["event"]
        attempt = event["attempt"]

        if name == "TRACE_ATTEMPT_START":
            if active_attempt or status not in {"pending", "paused"}:
                raise ValueError("duplicate or reordered attempt start")
            if attempt != attempt_count + 1:
                raise ValueError("attempt number is duplicate or reordered")
            attempt_count = attempt
            active_attempt = True
            unit_paused = False
            status = "active-without-outcome"
            last_failure = None
            continue
        if not active_attempt or attempt != attempt_count:
            raise ValueError("event order requires an active matching attempt start")

        if name == "UNIT_START":
            if active_unit is not None or unit_paused:
                raise ValueError("duplicate or reordered unit start")
            if len(completed) >= len(unit_plan):
                raise ValueError("unit start follows completion of the unit plan")
            expected_key = unit_plan[len(completed)].key
            if event["unit_key"] != expected_key:
                raise ValueError("unit start is reordered against the frozen plan")
            active_unit = expected_key
        elif name == "UNIT_PASS":
            if event["unit_key"] in completed:
                raise ValueError("duplicate UNIT_PASS")
            if active_unit is None or event["unit_key"] != active_unit:
                raise ValueError("UNIT_PASS requires its matching unit start")
            completed.append(active_unit)
            active_unit = None
        elif name == "UNIT_PAUSE":
            if active_unit is None or event["unit_key"] != active_unit:
                raise ValueError("UNIT_PAUSE requires its matching UNIT_START")
            active_unit = None
            unit_paused = True
            last_failure = event["failure_kind"]
        elif name == "TRACE_PAUSE":
            if active_unit is not None:
                raise ValueError("TRACE_PAUSE cannot abandon a started unit")
            if unit_paused and event["failure_kind"] != last_failure:
                raise ValueError("TRACE_PAUSE failure kind conflicts with UNIT_PAUSE")
            active_attempt = False
            unit_paused = False
            status = "paused"
            last_failure = event["failure_kind"]
        elif name == "TRACE_COMPLETE":
            if active_unit is not None or unit_paused or len(completed) != len(unit_plan):
                raise ValueError("TRACE_COMPLETE requires every unit to pass in order")
            active_attempt = False
            status = "complete"
            terminal = True
            last_failure = None
        elif name == "TRACE_TERMINAL_MODEL_INVALID":
            if active_unit is not None:
                raise ValueError("terminal model-invalid cannot abandon a started unit")
            if unit_paused and event["failure_kind"] != last_failure:
                raise ValueError("terminal failure kind conflicts with UNIT_PAUSE")
            active_attempt = False
            status = "terminal-model-invalid"
            terminal = True
            last_failure = event["failure_kind"]
        else:
            raise ValueError("event order is invalid")

    if active_attempt:
        status = "active-without-outcome"
    next_unit = unit_plan[len(completed)] if len(completed) < len(unit_plan) else None
    return TaskRecoveryState(
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        source_index=source_index,
        status=status,
        attempt_count=attempt_count,
        completed_unit_keys=tuple(completed),
        next_unit=next_unit,
        last_failure_kind=last_failure,
        history_line_count=len(events),
        history_tail_sha256=previous,
    )


def _snapshot_payload(
    state: TaskRecoveryState,
    *,
    history_line_count: int,
    history_tail_sha256: str,
) -> dict[str, Any]:
    next_unit = None
    if state.next_unit is not None:
        next_unit = {**asdict(state.next_unit), "key": state.next_unit.key}
    return {
        "schema_version": 1,
        "artifact_type": "ha_checkpoint_snapshot_v7",
        "manifest_sha256": state.manifest_sha256,
        "run_id": state.run_id,
        "task_id": state.task_id,
        "source_index": state.source_index,
        "status": state.status,
        "attempt_count": state.attempt_count,
        "completed_unit_keys": list(state.completed_unit_keys),
        "next_unit": next_unit,
        "last_failure_kind": state.last_failure_kind,
        "history_line_count": history_line_count,
        "history_tail_sha256": history_tail_sha256,
    }


def write_checkpoint_snapshot(
    path: Path,
    state: TaskRecoveryState,
    *,
    history_path: Path,
    unit_plan: tuple[ReplayUnit, ...],
    history_line_count: int,
    history_tail_sha256: str,
    run_root: Path,
) -> dict[str, Any]:
    """Atomically write deterministic convenience state bound to history."""

    destination = _ensure_safe_parent(path, run_root=run_root)
    history = _ensure_safe_parent(history_path, run_root=run_root)
    if destination == history:
        raise ValueError("checkpoint path must be distinct from task history")
    lock_fd = _open_lock(history, run_root=run_root)
    temporary: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        events = _read_unlocked(history, run_root=run_root)
        authoritative = reduce_task_history(
            events,
            unit_plan=unit_plan,
            manifest_sha256=state.manifest_sha256,
            run_id=state.run_id,
            task_id=state.task_id,
            source_index=state.source_index,
            run_root=run_root,
        )
        if authoritative != state:
            raise ValueError("checkpoint state conflicts with authoritative history")
        if isinstance(history_line_count, bool) or not isinstance(history_line_count, int) or history_line_count < 0:
            raise ValueError("history line count must be a nonnegative integer")
        if not isinstance(history_tail_sha256, str) or not _SHA256_RE.fullmatch(history_tail_sha256):
            raise ValueError("history tail sha256 must be lowercase 64-hex")
        if (
            history_line_count != authoritative.history_line_count
            or history_tail_sha256 != authoritative.history_tail_sha256
        ):
            raise ValueError("checkpoint history binding conflicts with reduced history")
        payload = _sanitize_tree(
            _snapshot_payload(
                authoritative,
                history_line_count=history_line_count,
                history_tail_sha256=history_tail_sha256,
            )
        )
        encoded = _canonical_bytes(payload) + b"\n"
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
        _safe_run_path(destination, run_root=run_root)
        _safe_run_path(temporary, run_root=run_root)
        os.replace(temporary, destination)
        temporary = None
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return payload
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def record_interrupted_attempt(
    history_path: Path,
    *,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    source_index: int,
    run_root: Path,
    timestamp_utc: str | None = None,
    failure_kind: str = "interrupted",
) -> dict[str, Any]:
    """Close a dangling attempt as paused while preserving its next unit."""

    path = _ensure_safe_parent(history_path, run_root=run_root)
    lock_fd = _open_lock(path, run_root=run_root)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        events = _read_unlocked(path, run_root=run_root)
        state = reduce_task_history(
            events,
            unit_plan=unit_plan,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            source_index=source_index,
            run_root=run_root,
        )
        if state.status != "active-without-outcome" or not events:
            raise ValueError("there is no interrupted active attempt to record")
        timestamp = timestamp_utc or _timestamp_utc()
        common: dict[str, Any] = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "task_id": task_id,
            "source_index": source_index,
            "attempt": state.attempt_count,
            "timestamp_utc": timestamp,
        }
        last = events[-1]
        if last["event"] == "UNIT_START":
            _append_task_event_unlocked(
                path,
                {
                    "event": "UNIT_PAUSE",
                    **common,
                    "unit_key": last["unit_key"],
                    "failure_kind": failure_kind,
                },
                events=events,
                run_root=run_root,
            )
        elif last["event"] == "UNIT_PAUSE":
            failure_kind = last["failure_kind"]
        return _append_task_event_unlocked(
            path,
            {"event": "TRACE_PAUSE", **common, "failure_kind": failure_kind},
            events=events,
            run_root=run_root,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


__all__ = [
    "ArtifactBinding",
    "ReplayUnit",
    "TaskRecoveryState",
    "append_task_event",
    "build_replay_unit_plan",
    "read_task_history",
    "record_interrupted_attempt",
    "reduce_task_history",
    "write_checkpoint_snapshot",
]
