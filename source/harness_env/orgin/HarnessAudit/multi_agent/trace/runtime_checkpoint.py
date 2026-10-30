"""Runtime recovery coordinator built on :mod:`multi_agent.trace.checkpoint`.

The lower-level checkpoint owns atomic step order and identity binding.  This
module binds each committed model turn to the mutable runtime surfaces that a
runner must reconstruct after interruption: the bank, normalized actions, SDK
session items, guard/hook/counter state, append-only ledgers, and trace bytes.

It intentionally imports no model SDK.  ``DurableCheckpointSession`` accepts
any object exposing synchronous or asynchronous ``clear_session`` and
``add_items`` methods.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .checkpoint import (
    BankSnapshotBinding,
    CheckpointConflictError,
    CheckpointTerminalError,
    CommittedTraceStep,
    DurableTraceCheckpoint,
    TerminalTraceState,
    TraceCheckpointSnapshot,
)


RUNTIME_SCHEMA_VERSION: Final = 2
BANK_RUNTIME_STATE_SCHEMA_VERSION: Final = 1
INITIAL_STATE: Final = "initial_state"
MODEL_TURN: Final = "model_turn"
ACTOR_COMPLETE: Final = "actor_complete"
EVALUATION_COMPLETE: Final = "evaluation_complete"


class RuntimeCheckpointError(RuntimeError):
    """Base error for runtime checkpoint coordination."""


class RuntimeCheckpointDriftError(RuntimeCheckpointError):
    """A bound runtime artifact is missing or differs from its committed prefix."""


class RuntimeCheckpointOrderError(RuntimeCheckpointError):
    """Runtime phases or turns were committed out of order."""


class RuntimeSessionDirtyError(RuntimeCheckpointError):
    """The underlying session must be prepared again before it can be extended."""


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Content-addressed runtime artifact under the condition root."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if not self.relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must be a non-empty relative path")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("artifact sha256 must be 64 lowercase hex characters")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise ValueError("artifact size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be non-negative")

    def as_json_value(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_json_value(cls, value: Any, *, label: str) -> ArtifactBinding:
        if not isinstance(value, dict) or set(value) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeCheckpointDriftError(f"{label} binding is malformed")
        try:
            return cls(
                relative_path=value["relative_path"],
                sha256=value["sha256"],
                size_bytes=value["size_bytes"],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeCheckpointDriftError(f"{label} binding is malformed") from exc


@dataclass(frozen=True, slots=True)
class LedgerOffsetBinding:
    """Hash-bound committed prefix of one append-only ledger."""

    name: str
    relative_path: str
    byte_offset: int
    prefix_sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ledger name must be non-empty")
        if not self.relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("ledger path must be a non-empty relative path")
        if not isinstance(self.byte_offset, int) or isinstance(self.byte_offset, bool):
            raise ValueError("ledger byte_offset must be an integer")
        if self.byte_offset < 0:
            raise ValueError("ledger byte_offset must be non-negative")
        if len(self.prefix_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.prefix_sha256
        ):
            raise ValueError("ledger prefix_sha256 must be 64 lowercase hex characters")

    def as_json_value(self) -> dict[str, Any]:
        return {
            "byte_offset": self.byte_offset,
            "name": self.name,
            "prefix_sha256": self.prefix_sha256,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_json_value(cls, value: Any, *, expected_name: str) -> LedgerOffsetBinding:
        if not isinstance(value, dict) or set(value) != {
            "byte_offset",
            "name",
            "prefix_sha256",
            "relative_path",
        }:
            raise RuntimeCheckpointDriftError("ledger offset binding is malformed")
        try:
            binding = cls(
                name=value["name"],
                relative_path=value["relative_path"],
                byte_offset=value["byte_offset"],
                prefix_sha256=value["prefix_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeCheckpointDriftError("ledger offset binding is malformed") from exc
        if binding.name != expected_name:
            raise RuntimeCheckpointDriftError("ledger binding name mismatch")
        return binding


@dataclass(frozen=True, slots=True)
class RuntimeCommitState:
    """Complete runtime state supplied at a model or phase boundary."""

    normalized_actions: Sequence[Mapping[str, Any]]
    guard_state: Mapping[str, Any]
    hook_state: Mapping[str, Any]
    counts_state: Mapping[str, Any]
    usage: Mapping[str, Any]
    ledger_paths: Mapping[str, Path]
    trace_bytes: bytes
    workspace: Path | None = None


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryState:
    """Verified state from the last committed runtime boundary."""

    last_step_id: int
    model_turn_count: int
    normalized_actions: tuple[Mapping[str, Any], ...]
    sdk_session_items: tuple[Any, ...]
    guard_state: Mapping[str, Any]
    hook_state: Mapping[str, Any]
    counts_state: Mapping[str, Any]
    usage: Mapping[str, Any]
    ledger_offsets: Mapping[str, LedgerOffsetBinding]
    bank_snapshot: BankSnapshotBinding | None
    bank_runtime_state: Mapping[str, Any] | None
    initial_state_state: Mapping[str, Any] | None
    trace_snapshot: ArtifactBinding | None
    workspace_snapshot: ArtifactBinding | None
    actor_complete_state: Mapping[str, Any] | None
    evaluation_complete_state: Mapping[str, Any] | None
    terminal: TerminalTraceState | None
    recovery_orphans: tuple[ArtifactBinding, ...] = ()

    @property
    def actor_complete(self) -> bool:
        return self.actor_complete_state is not None

    @property
    def evaluation_complete(self) -> bool:
        return self.evaluation_complete_state is not None


def _canonical_value(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc
    return json.loads(encoded)


def validate_oai_actor_completion_state(value: Any) -> dict[str, Any]:
    """Validate the exact legacy-v1 or safety-abstention-v2 actor state."""

    # Keep this dependency lazy: the runtime checkpoint remains independent
    # from framework registry initialization and optional model SDK adapters.
    from multi_agent.safepred_guard.terminal import (
        SAFETY_ABSTENTION_DISPOSITION,
        SAFETY_ABSTENTION_FINAL_OUTPUT,
        SAFETY_ABSTENTION_TASK_STATUS,
        validate_safety_abstention_terminal,
    )

    if not isinstance(value, Mapping):
        raise RuntimeCheckpointDriftError("actor completion state is malformed")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        state = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise RuntimeCheckpointDriftError("actor completion state is malformed") from exc
    v1_fields = {"schema_version", "framework", "final_output", "error"}
    v2_fields = {
        *v1_fields,
        "terminal_disposition",
        "task_success_status",
        "safety_abstention",
        "safety_abstention_artifact",
    }
    if set(state) == v1_fields:
        if (
            state.get("schema_version") != 1
            or state.get("framework") != "oai"
            or not isinstance(state.get("final_output"), str)
            or (
                state.get("error") is not None
                and not isinstance(state.get("error"), str)
            )
        ):
            raise RuntimeCheckpointDriftError("actor completion state is malformed")
        return state
    if set(state) != v2_fields:
        raise RuntimeCheckpointDriftError("actor completion state is malformed")
    if (
        state.get("schema_version") != 2
        or state.get("framework") != "oai"
        or state.get("final_output") != SAFETY_ABSTENTION_FINAL_OUTPUT
        or state.get("error") is not None
        or state.get("terminal_disposition") != SAFETY_ABSTENTION_DISPOSITION
        or state.get("task_success_status") != SAFETY_ABSTENTION_TASK_STATUS
    ):
        raise RuntimeCheckpointDriftError("safety abstention actor state is malformed")
    try:
        state["safety_abstention"] = validate_safety_abstention_terminal(
            state["safety_abstention"]
        )
        state["safety_abstention_artifact"] = ArtifactBinding.from_json_value(
            state["safety_abstention_artifact"],
            label="safety abstention artifact",
        ).as_json_value()
    except (TypeError, ValueError, RuntimeCheckpointDriftError) as exc:
        raise RuntimeCheckpointDriftError(
            "safety abstention actor state is malformed"
        ) from exc
    return state


def _canonical_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = _canonical_value(value, label=label)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _canonical_actions(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = _canonical_value(list(value), label="normalized_actions")
    if not isinstance(normalized, list) or any(not isinstance(item, dict) for item in normalized):
        raise ValueError("normalized_actions must be a JSON array of objects")
    return normalized


def _canonical_items(value: Sequence[Any]) -> list[Any]:
    normalized = _canonical_value(list(value), label="sdk_session_items")
    if not isinstance(normalized, list):
        raise ValueError("sdk_session_items must be a JSON array")
    return normalized



def _bank_class_identity(bank: Any) -> dict[str, str]:
    cls = type(bank)
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def _bank_runtime_field_kind(name: str) -> str | None:
    if name.endswith("_counter"):
        return "nonnegative_integer"
    if name.startswith("_current_") or name.startswith("_active_"):
        return "nullable_string"
    return None


def _validated_bank_runtime_state_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "bank_class",
        "field_types",
        "field_values",
        "schema_version",
    }:
        raise RuntimeCheckpointDriftError("bank runtime state schema is malformed")
    if value["schema_version"] != BANK_RUNTIME_STATE_SCHEMA_VERSION:
        raise RuntimeCheckpointDriftError("bank runtime state schema version drift")
    bank_class = value["bank_class"]
    if (
        not isinstance(bank_class, dict)
        or set(bank_class) != {"module", "qualname"}
        or not all(isinstance(item, str) and item for item in bank_class.values())
    ):
        raise RuntimeCheckpointDriftError("bank runtime class identity is malformed")
    field_types = value["field_types"]
    field_values = value["field_values"]
    if not isinstance(field_types, dict) or not isinstance(field_values, dict):
        raise RuntimeCheckpointDriftError("bank runtime fields are malformed")
    if set(field_types) != set(field_values):
        raise RuntimeCheckpointDriftError("bank runtime field set drift")
    for name, kind in field_types.items():
        expected_kind = _bank_runtime_field_kind(name) if isinstance(name, str) else None
        if expected_kind is None or kind != expected_kind:
            raise RuntimeCheckpointDriftError("bank runtime field schema drift")
        field_value = field_values[name]
        if kind == "nonnegative_integer":
            if type(field_value) is not int or field_value < 0:
                raise RuntimeCheckpointDriftError("bank runtime counter type drift")
        elif field_value is not None and not isinstance(field_value, str):
            raise RuntimeCheckpointDriftError("bank runtime scope type drift")
    return value


def _capture_bank_runtime_state(bank: Any) -> dict[str, Any]:
    if bank is not None and not hasattr(bank, "__dict__"):
        raise RuntimeCheckpointError("bank must expose inspectable instance state")
    field_types: dict[str, str] = {}
    field_values: dict[str, Any] = {}
    for name, value in (vars(bank).items() if bank is not None else ()):
        if not isinstance(name, str):
            raise RuntimeCheckpointDriftError("bank instance field name is malformed")
        if not name.startswith("_"):
            continue
        kind = _bank_runtime_field_kind(name)
        if kind is None:
            raise RuntimeCheckpointDriftError(
                "bank has unsupported private runtime state"
            )
        field_types[name] = kind
        field_values[name] = value
    try:
        captured = _canonical_mapping(
            {
                "bank_class": _bank_class_identity(bank),
                "field_types": field_types,
                "field_values": field_values,
                "schema_version": BANK_RUNTIME_STATE_SCHEMA_VERSION,
            },
            label="bank_runtime_state",
        )
    except ValueError as exc:
        raise RuntimeCheckpointDriftError(
            "bank private runtime state is not finite JSON data"
        ) from exc
    return _validated_bank_runtime_state_shape(captured)


def _validated_bank_restore_values(
    bank: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _validated_bank_runtime_state_shape(
        _canonical_mapping(state, label="bank_runtime_state")
    )
    if normalized["bank_class"] != _bank_class_identity(bank):
        raise RuntimeCheckpointDriftError("bank runtime class identity drift")
    current = _capture_bank_runtime_state(bank)
    if normalized["field_types"] != current["field_types"]:
        raise RuntimeCheckpointDriftError("bank runtime field/type schema drift")
    return dict(normalized["field_values"])


def _restore_bank_runtime_state(bank: Any, values: Mapping[str, Any]) -> None:
    if bank is None:
        if values:
            raise RuntimeCheckpointDriftError(
                "bankless runtime state unexpectedly contains fields"
            )
        return
    previous = {name: getattr(bank, name) for name in values}
    try:
        for name, value in values.items():
            setattr(bank, name, value)
    except Exception as exc:
        for name, value in previous.items():
            try:
                setattr(bank, name, value)
            except Exception:
                pass
        raise RuntimeCheckpointDriftError("bank runtime state restoration failed") from exc


def _validate_initial_trace_bytes(trace_bytes: bytes) -> None:
    if not isinstance(trace_bytes, bytes):
        raise TypeError("trace_bytes must be bytes")
    lines = [line for line in trace_bytes.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("initial trace must contain exactly one trace_start event")
    try:
        event = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("initial trace_start event is malformed") from exc
    if not isinstance(event, dict) or event.get("type") != "trace_start":
        raise ValueError("initial trace must contain a trace_start event")


def _validated_initial_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "run_started_at_utc",
        "schema_version",
    }:
        raise RuntimeCheckpointDriftError("initial runtime state is malformed")
    if value["schema_version"] != 1:
        raise RuntimeCheckpointDriftError("initial runtime state schema drift")
    timestamp = value["run_started_at_utc"]
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise RuntimeCheckpointDriftError("initial run timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeCheckpointDriftError("initial run timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeCheckpointDriftError("initial run timestamp is not UTC")
    return value

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeCheckpointDriftError(f"bound artifact is missing: {path.name}")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != before.st_size
    ):
        raise RuntimeCheckpointDriftError(f"bound artifact changed while reading: {path.name}")
    return data


class RuntimeCheckpointCoordinator:
    """Bind runtime artifacts to atomic ``DurableTraceCheckpoint`` steps."""

    def __init__(self, root: str | Path, checkpoint: DurableTraceCheckpoint) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint = checkpoint
        self._mutex = threading.RLock()

    def _relative_path(self, path: str | Path, *, label: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            relative = candidate.resolve().relative_to(self.root)
        except ValueError as exc:
            raise RuntimeCheckpointError(f"{label} must be inside the runtime root") from exc
        if not relative.parts:
            raise RuntimeCheckpointError(f"{label} must not be the runtime root")
        return relative.as_posix()

    def _resolve_binding_path(self, relative_path: str, *, label: str) -> Path:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise RuntimeCheckpointDriftError(f"{label} path is not safely relative")
        candidate = self.root.joinpath(*pure.parts)
        try:
            candidate.resolve().relative_to(self.root)
        except ValueError as exc:
            raise RuntimeCheckpointDriftError(f"{label} escapes the runtime root") from exc
        return candidate

    def _write_content_addressed(
        self,
        *,
        directory: str,
        data: bytes,
        suffix: str,
    ) -> ArtifactBinding:
        digest = _sha256(data)
        target_dir = self.root / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest}{suffix}"
        if target.exists():
            existing = _stable_read(target)
            if existing != data:
                raise RuntimeCheckpointDriftError("content-addressed artifact collision")
        else:
            descriptor, raw_temp = tempfile.mkstemp(prefix=".pending-", dir=target_dir)
            temp = Path(raw_temp)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
                os.chmod(target, 0o600)
                _fsync_directory(target_dir)
            except Exception:
                if temp.exists():
                    temp.unlink()
                raise
        return ArtifactBinding(
            relative_path=target.relative_to(self.root).as_posix(),
            sha256=digest,
            size_bytes=len(data),
        )

    def _read_artifact(self, binding: ArtifactBinding, *, label: str) -> bytes:
        path = self._resolve_binding_path(binding.relative_path, label=label)
        data = _stable_read(path)
        if len(data) != binding.size_bytes or _sha256(data) != binding.sha256:
            raise RuntimeCheckpointDriftError(f"{label} size/hash drift")
        return data

    def dump_bank_snapshot(self, bank: Any) -> BankSnapshotBinding:
        """Create and publish an immutable content-addressed SQLite bank dump."""

        conn = getattr(bank, "conn", None)
        dump_to_file = getattr(bank, "dump_to_file", None)
        if not isinstance(conn, sqlite3.Connection) or not callable(dump_to_file):
            raise RuntimeCheckpointError("bank must expose sqlite3 conn and dump_to_file(path)")
        snapshot_dir = self.root / "bank_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp = tempfile.mkstemp(prefix=".pending-bank-", dir=snapshot_dir)
        temp = Path(raw_temp)
        os.close(descriptor)
        os.chmod(temp, 0o600)
        try:
            conn.commit()
            dump_to_file(temp)
            with sqlite3.connect(f"file:{temp}?mode=ro", uri=True) as verifier:
                quick_check = verifier.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                raise RuntimeCheckpointDriftError("bank dump failed SQLite quick_check")
            with temp.open("rb") as handle:
                os.fsync(handle.fileno())
            data = _stable_read(temp)
            digest = _sha256(data)
            target = snapshot_dir / f"{digest}.sqlite"
            if target.exists():
                if _stable_read(target) != data:
                    raise RuntimeCheckpointDriftError("bank snapshot hash collision")
                temp.unlink()
            else:
                os.replace(temp, target)
                os.chmod(target, 0o600)
                _fsync_directory(snapshot_dir)
            return BankSnapshotBinding(
                relative_path=target.relative_to(self.root).as_posix(),
                sha256=digest,
                size_bytes=len(data),
            )
        except Exception:
            if temp.exists():
                temp.unlink()
            raise

    def restore_bank_snapshot(self, bank: Any, binding: BankSnapshotBinding) -> None:
        """Verify and restore a committed SQLite bank image into ``bank.conn``."""

        conn = getattr(bank, "conn", None)
        if not isinstance(conn, sqlite3.Connection):
            raise RuntimeCheckpointError("bank must expose sqlite3 conn")
        path = self._resolve_binding_path(binding.relative_path, label="bank snapshot")
        data = _stable_read(path)
        if len(data) != binding.size_bytes or _sha256(data) != binding.sha256:
            raise RuntimeCheckpointDriftError("bank snapshot size/hash drift")
        conn.rollback()
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source:
            quick_check = source.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                raise RuntimeCheckpointDriftError("bank snapshot failed SQLite quick_check")
            source.backup(conn)
        conn.commit()
        restored_check = conn.execute("PRAGMA quick_check").fetchone()
        if restored_check is None or restored_check[0] != "ok":
            raise RuntimeCheckpointDriftError("restored bank failed SQLite quick_check")

    def _capture_ledger_offsets(
        self,
        ledger_paths: Mapping[str, Path],
        previous: Mapping[str, LedgerOffsetBinding],
    ) -> dict[str, LedgerOffsetBinding]:
        if not set(previous).issubset(ledger_paths):
            raise RuntimeCheckpointDriftError("a previously bound ledger was omitted")
        captured: dict[str, LedgerOffsetBinding] = {}
        resolved_paths: set[Path] = set()
        for name in sorted(ledger_paths):
            if not isinstance(name, str) or not name:
                raise ValueError("ledger names must be non-empty strings")
            relative = self._relative_path(ledger_paths[name], label=f"ledger {name}")
            path = self.root / relative
            resolved = path.resolve()
            if resolved in resolved_paths:
                raise RuntimeCheckpointError("two ledger names resolve to the same file")
            resolved_paths.add(resolved)
            data = _stable_read(path)
            prior = previous.get(name)
            if prior is not None:
                if prior.relative_path != relative:
                    raise RuntimeCheckpointDriftError("ledger path changed across turns")
                if len(data) < prior.byte_offset:
                    raise RuntimeCheckpointDriftError("ledger lost a committed prefix")
                if _sha256(data[: prior.byte_offset]) != prior.prefix_sha256:
                    raise RuntimeCheckpointDriftError("ledger committed prefix drifted")
            captured[name] = LedgerOffsetBinding(
                name=name,
                relative_path=relative,
                byte_offset=len(data),
                prefix_sha256=_sha256(data),
            )
        return captured

    def _reconcile_ledgers(
        self,
        bindings: Mapping[str, LedgerOffsetBinding],
    ) -> tuple[ArtifactBinding, ...]:
        orphans: list[ArtifactBinding] = []
        for name in sorted(bindings):
            binding = bindings[name]
            path = self._resolve_binding_path(binding.relative_path, label=f"ledger {name}")
            data = _stable_read(path)
            if len(data) < binding.byte_offset:
                raise RuntimeCheckpointDriftError("ledger committed prefix is missing")
            prefix = data[: binding.byte_offset]
            if _sha256(prefix) != binding.prefix_sha256:
                raise RuntimeCheckpointDriftError("ledger committed prefix drifted")
            suffix = data[binding.byte_offset :]
            if not suffix:
                continue
            source_tag = _sha256(binding.relative_path.encode("utf-8"))[:16]
            orphan = self._write_content_addressed(
                directory=f"recovery_orphans/ledger_suffixes/{source_tag}",
                data=suffix,
                suffix=".bin",
            )
            orphans.append(orphan)
            with path.open("r+b") as handle:
                handle.truncate(binding.byte_offset)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(path.parent)
        return tuple(orphans)

    def prepare_initial_ledgers(
        self,
        ledger_paths: Mapping[str, Path],
    ) -> tuple[ArtifactBinding, ...]:
        """Archive/truncate pre-commit suffixes before the first model call.

        A process may die after writing a ledger but before its first completed
        model turn. With no committed offset yet, ordinary recovery cannot
        distinguish those bytes from a clean baseline. This preparation step
        preserves such bytes as content-addressed orphans and restores every
        declared ledger to an empty baseline. It is valid only after the
        immutable initial-state checkpoint and before the first model turn.
        """

        with self._mutex:
            state = self.inspect()
            if (
                state.initial_state_state is None
                or state.model_turn_count != 0
                or state.actor_complete
                or state.evaluation_complete
                or state.terminal is not None
            ):
                raise RuntimeCheckpointOrderError(
                    "initial ledger preparation requires a baseline-only checkpoint"
                )
            orphans: list[ArtifactBinding] = []
            resolved_paths: set[Path] = set()
            for name in sorted(ledger_paths):
                if not isinstance(name, str) or not name:
                    raise ValueError("ledger names must be non-empty strings")
                relative = self._relative_path(
                    ledger_paths[name],
                    label=f"ledger {name}",
                )
                path = self.root / relative
                resolved = path.resolve()
                if resolved in resolved_paths:
                    raise RuntimeCheckpointError(
                        "two ledger names resolve to the same file"
                    )
                resolved_paths.add(resolved)
                data = _stable_read(path)
                if not data:
                    continue
                source_tag = _sha256(relative.encode("utf-8"))[:16]
                orphans.append(
                    self._write_content_addressed(
                        directory=(
                            "recovery_orphans/initial_ledger_suffixes/"
                            f"{source_tag}"
                        ),
                        data=data,
                        suffix=".bin",
                    )
                )
                with path.open("r+b") as handle:
                    handle.truncate(0)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(path.parent)
            return tuple(orphans)

    def _bind_trace_bytes(self, trace_bytes: bytes) -> ArtifactBinding:
        if not isinstance(trace_bytes, bytes):
            raise TypeError("trace_bytes must be bytes")
        return self._write_content_addressed(
            directory="trace_snapshots",
            data=trace_bytes,
            suffix=".bin",
        )

    def archive_and_rebuild_trace(
        self,
        trace_path: str | Path,
        committed_trace: ArtifactBinding | None,
    ) -> ArtifactBinding | None:
        """Archive exact current trace bytes, then atomically rebuild committed bytes."""

        relative = self._relative_path(trace_path, label="trace path")
        path = self.root / relative
        archived: ArtifactBinding | None = None
        if path.exists():
            original = _stable_read(path)
            archived = self._write_content_addressed(
                directory="recovery_orphans/trace_archives",
                data=original,
                suffix=".original",
            )
        committed_bytes = (
            self._read_artifact(committed_trace, label="trace snapshot")
            if committed_trace is not None
            else b""
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp = tempfile.mkstemp(prefix=".rebuild-", dir=path.parent)
        temp = Path(raw_temp)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(committed_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            os.chmod(path, 0o600)
            _fsync_directory(path.parent)
        except Exception:
            if temp.exists():
                temp.unlink()
            raise
        return archived


    def snapshot_workspace(self, workspace: str | Path | None) -> ArtifactBinding | None:
        """Store an exact workspace image excluding ``.git`` metadata."""

        if workspace is None:
            return None
        relative_root = self._relative_path(workspace, label="workspace")
        root = self.root / relative_root
        if root.is_symlink() or not root.is_dir():
            raise RuntimeCheckpointDriftError("workspace must be a real directory")
        entries: list[dict[str, Any]] = []
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root)
            if ".git" in relative.parts:
                continue
            if path.is_symlink():
                raise RuntimeCheckpointDriftError("workspace contains a symbolic link")
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            if path.is_dir():
                entries.append({"mode": mode, "path": relative.as_posix(), "type": "dir"})
                continue
            if not path.is_file():
                raise RuntimeCheckpointDriftError("workspace contains a special file")
            data = _stable_read(path)
            blob = self._write_content_addressed(
                directory="workspace_blobs",
                data=data,
                suffix=".bin",
            )
            entries.append(
                {
                    "blob": blob.as_json_value(),
                    "mode": mode,
                    "path": relative.as_posix(),
                    "type": "file",
                }
            )
        manifest = json.dumps(
            {"entries": entries, "schema_version": RUNTIME_SCHEMA_VERSION},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._write_content_addressed(
            directory="workspace_manifests",
            data=manifest,
            suffix=".json",
        )

    def restore_workspace(
        self,
        workspace: str | Path,
        binding: ArtifactBinding,
    ) -> None:
        """Restore an exact workspace image while preserving ``.git`` metadata."""

        relative_root = self._relative_path(workspace, label="workspace")
        root = self.root / relative_root
        if root.is_symlink():
            raise RuntimeCheckpointDriftError("workspace must not be a symbolic link")
        root.mkdir(parents=True, exist_ok=True)
        raw_manifest = self._read_artifact(binding, label="workspace manifest")
        try:
            manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as exc:
            raise RuntimeCheckpointDriftError("workspace manifest is invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION
            or not isinstance(manifest.get("entries"), list)
        ):
            raise RuntimeCheckpointDriftError("workspace manifest schema is invalid")
        validated: list[tuple[PurePosixPath, dict[str, Any]]] = []
        seen: set[PurePosixPath] = set()
        for entry in manifest["entries"]:
            if not isinstance(entry, dict):
                raise RuntimeCheckpointDriftError("workspace manifest entry is invalid")
            relative = PurePosixPath(str(entry.get("path") or ""))
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or ".git" in relative.parts
                or relative in seen
            ):
                raise RuntimeCheckpointDriftError("workspace manifest path is unsafe")
            if entry.get("type") not in {"dir", "file"}:
                raise RuntimeCheckpointDriftError("workspace manifest type is invalid")
            mode = entry.get("mode")
            if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o7777:
                raise RuntimeCheckpointDriftError("workspace manifest mode is invalid")
            if entry["type"] == "file":
                ArtifactBinding.from_json_value(entry.get("blob"), label="workspace blob")
            seen.add(relative)
            validated.append((relative, entry))

        existing = sorted(
            (item for item in root.rglob("*") if ".git" not in item.relative_to(root).parts),
            key=lambda item: len(item.relative_to(root).parts),
            reverse=True,
        )
        if any(item.is_symlink() for item in existing):
            raise RuntimeCheckpointDriftError("workspace contains a symbolic link")
        for item in existing:
            if item.is_dir():
                item.rmdir()
            elif item.is_file():
                item.unlink()
            else:
                raise RuntimeCheckpointDriftError("workspace contains a special file")

        for relative, entry in sorted(validated, key=lambda item: len(item[0].parts)):
            target = root.joinpath(*relative.parts)
            if entry["type"] == "dir":
                target.mkdir(parents=True, exist_ok=True)
                os.chmod(target, entry["mode"])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = ArtifactBinding.from_json_value(entry["blob"], label="workspace blob")
            data = self._read_artifact(blob, label="workspace blob")
            descriptor, raw_temp = tempfile.mkstemp(prefix=".restore-", dir=target.parent)
            temp = Path(raw_temp)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
                os.chmod(target, entry["mode"])
                _fsync_directory(target.parent)
            except Exception:
                if temp.exists():
                    temp.unlink()
                raise

    def _parse_snapshot(self, snapshot: TraceCheckpointSnapshot) -> RuntimeRecoveryState:
        normalized_actions: tuple[Mapping[str, Any], ...] = ()
        guard_state: Mapping[str, Any] = {}
        hook_state: Mapping[str, Any] = {}
        counts_state: Mapping[str, Any] = {}
        usage: Mapping[str, Any] = {}
        ledger_offsets: Mapping[str, LedgerOffsetBinding] = {}
        bank_snapshot: BankSnapshotBinding | None = None
        bank_runtime_state: Mapping[str, Any] | None = None
        trace_snapshot: ArtifactBinding | None = None
        workspace_snapshot: ArtifactBinding | None = None
        initial_state: Mapping[str, Any] | None = None
        actor_state: Mapping[str, Any] | None = None
        evaluation_state: Mapping[str, Any] | None = None
        phase = -1
        model_turn_count = 0
        common_fields = {
            "bank_runtime_state",
            "counts_state",
            "hook_state",
            "ledger_offsets",
            "normalized_actions",
            "phase_state",
            "runtime_schema_version",
            "trace_snapshot",
            "workspace_snapshot",
        }

        for step in snapshot.steps:
            action = step.action
            if (
                not isinstance(action, dict)
                or set(action) != common_fields
                or action.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION
            ):
                raise RuntimeCheckpointDriftError("step runtime checkpoint schema drift")
            parsed_bank_state = _validated_bank_runtime_state_shape(
                action.get("bank_runtime_state")
            )
            if bank_runtime_state is not None and (
                parsed_bank_state["bank_class"]
                != bank_runtime_state["bank_class"]
                or parsed_bank_state["field_types"]
                != bank_runtime_state["field_types"]
            ):
                raise RuntimeCheckpointDriftError(
                    "bank runtime class/field schema changed within the trace"
                )
            bankless_state = parsed_bank_state["bank_class"] == {
                "module": "builtins",
                "qualname": "NoneType",
            }
            if bankless_state != (step.bank_snapshot is None):
                raise RuntimeCheckpointDriftError(
                    "bank snapshot/runtime-state presence mismatch"
                )
            if step.step_kind == INITIAL_STATE:
                if (
                    step.action_id != "runtime-phase:initial_state"
                    or phase != -1
                    or initial_state is not None
                    or step.step_id != 1
                ):
                    raise RuntimeCheckpointOrderError("initial runtime state is out of order")
                if step.sdk_session_items:
                    raise RuntimeCheckpointDriftError("initial runtime session is not empty")
                initial_state = _validated_initial_state(
                    self._phase_state(action, label="initial runtime")
                )
                phase = 0
            elif step.step_kind == MODEL_TURN:
                if phase != 0 or initial_state is None:
                    raise RuntimeCheckpointOrderError("model turn is out of order")
                turn_state = self._phase_state(action, label="model turn")
                if (
                    set(turn_state) != {"turn_id"}
                    or not isinstance(turn_state["turn_id"], str)
                    or not turn_state["turn_id"]
                    or step.action_id != f"runtime-turn:{turn_state['turn_id']}"
                ):
                    raise RuntimeCheckpointDriftError("model turn state is malformed")
                model_turn_count += 1
            elif step.step_kind == ACTOR_COMPLETE:
                if (
                    step.action_id != "runtime-phase:actor_complete"
                    or phase != 0
                    or initial_state is None
                    or actor_state is not None
                ):
                    raise RuntimeCheckpointOrderError("actor completion is out of order")
                if step.sdk_session_items:
                    raise RuntimeCheckpointDriftError("actor completion session delta is not empty")
                actor_state = validate_oai_actor_completion_state(
                    self._phase_state(action, label="actor completion")
                )
                phase = 1
            elif step.step_kind == EVALUATION_COMPLETE:
                if (
                    step.action_id != "runtime-phase:evaluation_complete"
                    or phase != 1
                    or evaluation_state is not None
                ):
                    raise RuntimeCheckpointOrderError("evaluation completion is out of order")
                if step.sdk_session_items:
                    raise RuntimeCheckpointDriftError(
                        "evaluation completion session delta is not empty"
                    )
                evaluation_state = self._phase_state(action, label="evaluation completion")
                if (
                    set(evaluation_state) != {
                        "metric_summary",
                        "report",
                        "report_role",
                        "run_id",
                        "schema_version",
                        "task_id",
                    }
                    or evaluation_state.get("schema_version") != 1
                    or evaluation_state.get("report_role")
                    != "harnessaudit_evaluation_report_v1"
                    or not isinstance(evaluation_state.get("run_id"), str)
                    or not evaluation_state.get("run_id")
                    or not isinstance(evaluation_state.get("task_id"), str)
                    or not evaluation_state.get("task_id")
                    or not isinstance(evaluation_state.get("metric_summary"), dict)
                ):
                    raise RuntimeCheckpointDriftError(
                        "evaluation completion state is malformed"
                    )
                ArtifactBinding.from_json_value(
                    evaluation_state.get("report"),
                    label="evaluation report",
                )
                phase = 2
            else:
                raise RuntimeCheckpointOrderError(f"unsupported runtime step: {step.step_kind}")

            actions = action.get("normalized_actions")
            hook = action.get("hook_state")
            counts = action.get("counts_state")
            if (
                not isinstance(actions, list)
                or any(not isinstance(item, dict) for item in actions)
                or not isinstance(hook, dict)
                or not isinstance(counts, dict)
                or not isinstance(step.guard_state, dict)
                or not isinstance(step.usage, dict)
            ):
                raise RuntimeCheckpointDriftError("runtime state snapshot is malformed")
            raw_ledgers = action.get("ledger_offsets")
            if not isinstance(raw_ledgers, dict) or any(
                not isinstance(name, str) or not name for name in raw_ledgers
            ):
                raise RuntimeCheckpointDriftError("ledger offset state is malformed")
            parsed_ledgers = {
                name: LedgerOffsetBinding.from_json_value(value, expected_name=name)
                for name, value in raw_ledgers.items()
            }
            if not set(ledger_offsets).issubset(parsed_ledgers):
                raise RuntimeCheckpointDriftError("a committed ledger binding was omitted")

            parsed_trace = ArtifactBinding.from_json_value(
                action.get("trace_snapshot"),
                label="trace snapshot",
            )
            raw_workspace = action.get("workspace_snapshot")
            parsed_workspace = (
                ArtifactBinding.from_json_value(raw_workspace, label="workspace snapshot")
                if raw_workspace is not None
                else None
            )
            if step.step_kind == INITIAL_STATE:
                if (
                    actions
                    or hook
                    or counts
                    or step.guard_state
                    or step.usage
                    or parsed_ledgers
                ):
                    raise RuntimeCheckpointDriftError("initial runtime baseline is not empty")
                workspace_snapshot = parsed_workspace
            elif workspace_snapshot is None:
                if parsed_workspace is not None:
                    raise RuntimeCheckpointDriftError(
                        "workspace appeared after an absent initial baseline"
                    )
            elif parsed_workspace is not None:
                workspace_snapshot = parsed_workspace
            elif step.step_kind != EVALUATION_COMPLETE:
                raise RuntimeCheckpointDriftError(
                    "active runtime step omitted the committed workspace"
                )

            normalized_actions = tuple(actions)
            hook_state = hook
            counts_state = counts
            ledger_offsets = parsed_ledgers
            trace_snapshot = parsed_trace
            guard_state = step.guard_state
            usage = step.usage
            bank_snapshot = step.bank_snapshot
            bank_runtime_state = parsed_bank_state

        if snapshot.steps and initial_state is None:
            raise RuntimeCheckpointOrderError("runtime checkpoint lacks initial state")
        terminal = snapshot.terminal
        if terminal is not None:
            if evaluation_state is None or phase != 2:
                raise RuntimeCheckpointDriftError(
                    "terminal runtime checkpoint lacks evaluation completion"
                )
            if not isinstance(terminal.state, dict) or set(terminal.state) != {
                "evaluation_step_id",
                "report",
                "runtime_schema_version",
            }:
                raise RuntimeCheckpointDriftError("terminal runtime state schema drift")
            if (
                terminal.state["runtime_schema_version"] != RUNTIME_SCHEMA_VERSION
                or terminal.state["evaluation_step_id"] != snapshot.last_step_id
            ):
                raise RuntimeCheckpointDriftError("terminal runtime state identity drift")
            terminal_report = ArtifactBinding.from_json_value(
                terminal.state.get("report"), label="terminal report"
            )
            evaluated_report = ArtifactBinding.from_json_value(
                evaluation_state.get("report"), label="evaluation report"
            )
            if terminal_report != evaluated_report:
                raise RuntimeCheckpointDriftError(
                    "terminal and evaluation report bindings differ"
                )

        return RuntimeRecoveryState(
            last_step_id=snapshot.last_step_id,
            model_turn_count=model_turn_count,
            normalized_actions=normalized_actions,
            sdk_session_items=snapshot.sdk_session_items,
            guard_state=guard_state,
            hook_state=hook_state,
            counts_state=counts_state,
            usage=usage,
            ledger_offsets=ledger_offsets,
            bank_snapshot=bank_snapshot,
            bank_runtime_state=bank_runtime_state,
            initial_state_state=initial_state,
            trace_snapshot=trace_snapshot,
            workspace_snapshot=workspace_snapshot,
            actor_complete_state=actor_state,
            evaluation_complete_state=evaluation_state,
            terminal=terminal,
        )

    @staticmethod
    def _phase_state(action: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
        state = action.get("phase_state")
        if not isinstance(state, dict):
            raise RuntimeCheckpointDriftError(f"{label} state is malformed")
        return state

    def inspect(self) -> RuntimeRecoveryState:
        """Validate and return checkpoint state without mutating runtime artifacts."""

        with self._mutex:
            return self._parse_snapshot(self.checkpoint.load())

    def has_committed_action(self, action_id: str) -> bool:
        return any(step.action_id == action_id for step in self.checkpoint.load().steps)

    def _existing_step(self, action_id: str) -> CommittedTraceStep | None:
        return next(
            (step for step in self.checkpoint.load().steps if step.action_id == action_id),
            None,
        )

    def _validate_existing_step(
        self,
        step: CommittedTraceStep,
        *,
        step_kind: str,
        sdk_session_items: Sequence[Any],
        bank: Any,
        state: RuntimeCommitState,
        phase_state: Mapping[str, Any],
    ) -> None:
        expected_actions = _canonical_actions(state.normalized_actions)
        expected_items = _canonical_items(sdk_session_items)
        expected_guard = _canonical_mapping(state.guard_state, label="guard_state")
        expected_hook = _canonical_mapping(state.hook_state, label="hook_state")
        expected_counts = _canonical_mapping(state.counts_state, label="counts_state")
        expected_usage = _canonical_mapping(state.usage, label="usage")
        expected_phase = _canonical_mapping(phase_state, label="phase_state")
        expected_trace = self._bind_trace_bytes(state.trace_bytes)
        expected_workspace = self.snapshot_workspace(state.workspace)
        expected_bank = self.dump_bank_snapshot(bank) if bank is not None else None
        expected_bank_runtime = _capture_bank_runtime_state(bank)
        expected_ledgers = self._capture_ledger_offsets(state.ledger_paths, {})
        action = step.action
        if not isinstance(action, dict):
            raise CheckpointConflictError("committed runtime action is malformed")
        try:
            stored_trace = ArtifactBinding.from_json_value(
                action.get("trace_snapshot"), label="trace snapshot"
            )
            raw_workspace = action.get("workspace_snapshot")
            stored_workspace = (
                ArtifactBinding.from_json_value(raw_workspace, label="workspace snapshot")
                if raw_workspace is not None
                else None
            )
        except RuntimeCheckpointDriftError as exc:
            raise CheckpointConflictError(
                "committed runtime action_id has malformed bindings"
            ) from exc
        if (
            step.step_kind != step_kind
            or list(step.sdk_session_items) != expected_items
            or step.bank_snapshot != expected_bank
            or step.guard_state != expected_guard
            or step.usage != expected_usage
            or action.get("bank_runtime_state") != expected_bank_runtime
            or action.get("normalized_actions") != expected_actions
            or action.get("hook_state") != expected_hook
            or action.get("counts_state") != expected_counts
            or action.get("phase_state") != expected_phase
            or action.get("ledger_offsets")
            != {
                name: binding.as_json_value()
                for name, binding in sorted(expected_ledgers.items())
            }
            or stored_trace != expected_trace
            or stored_workspace != expected_workspace
        ):
            raise CheckpointConflictError("committed runtime action_id has different input")

    def _commit_runtime_step(
        self,
        *,
        step_kind: str,
        action_id: str,
        sdk_session_items: Sequence[Any],
        bank: Any,
        state: RuntimeCommitState,
        phase_state: Mapping[str, Any],
    ) -> CommittedTraceStep:
        with self._mutex:
            existing = self._existing_step(action_id)
            if existing is not None:
                self._validate_existing_step(
                    existing,
                    step_kind=step_kind,
                    sdk_session_items=sdk_session_items,
                    bank=bank,
                    state=state,
                    phase_state=phase_state,
                )
                return existing

            snapshot = self.checkpoint.load()
            current = self._parse_snapshot(snapshot)
            if snapshot.terminal is not None:
                raise CheckpointTerminalError("terminal runtime checkpoint cannot be extended")
            if step_kind == INITIAL_STATE:
                if snapshot.steps or current.initial_state_state is not None:
                    raise RuntimeCheckpointOrderError("initial runtime state already exists")
            elif current.initial_state_state is None:
                raise RuntimeCheckpointOrderError(
                    "runtime step precedes initial-state baseline"
                )
            if step_kind == MODEL_TURN and current.actor_complete:
                raise RuntimeCheckpointOrderError("model turn follows actor completion")
            if step_kind == ACTOR_COMPLETE and current.actor_complete:
                raise RuntimeCheckpointOrderError("actor completion already exists")
            if step_kind == EVALUATION_COMPLETE:
                if not current.actor_complete:
                    raise RuntimeCheckpointOrderError("evaluation precedes actor completion")
                if current.evaluation_complete:
                    raise RuntimeCheckpointOrderError("evaluation completion already exists")

            normalized_actions = _canonical_actions(state.normalized_actions)
            if (
                len(normalized_actions) < len(current.normalized_actions)
                or normalized_actions[: len(current.normalized_actions)]
                != list(current.normalized_actions)
            ):
                raise RuntimeCheckpointDriftError(
                    "normalized actions do not extend the committed prefix"
                )
            if current.trace_snapshot is not None:
                previous_trace = self._read_artifact(
                    current.trace_snapshot,
                    label="previous trace snapshot",
                )
                if not state.trace_bytes.startswith(previous_trace):
                    raise RuntimeCheckpointDriftError(
                        "trace bytes do not extend the committed prefix"
                    )
            session_items = _canonical_items(sdk_session_items)
            hook_state = _canonical_mapping(state.hook_state, label="hook_state")
            counts_state = _canonical_mapping(state.counts_state, label="counts_state")
            phase = _canonical_mapping(phase_state, label="phase_state")
            guard_state = _canonical_mapping(state.guard_state, label="guard_state")
            usage = _canonical_mapping(state.usage, label="usage")
            bank_runtime_state = _capture_bank_runtime_state(bank)
            if current.bank_runtime_state is not None and (
                bank_runtime_state["bank_class"]
                != current.bank_runtime_state["bank_class"]
                or bank_runtime_state["field_types"]
                != current.bank_runtime_state["field_types"]
            ):
                raise RuntimeCheckpointDriftError(
                    "bank runtime class/field schema changed within the trace"
                )
            bank_snapshot = self.dump_bank_snapshot(bank) if bank is not None else None
            ledger_offsets = self._capture_ledger_offsets(
                state.ledger_paths,
                current.ledger_offsets,
            )
            trace_snapshot = self._bind_trace_bytes(state.trace_bytes)
            workspace_snapshot = self.snapshot_workspace(state.workspace)
            action = {
                "bank_runtime_state": bank_runtime_state,
                "counts_state": counts_state,
                "hook_state": hook_state,
                "ledger_offsets": {
                    name: binding.as_json_value()
                    for name, binding in sorted(ledger_offsets.items())
                },
                "normalized_actions": normalized_actions,
                "phase_state": phase,
                "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                "trace_snapshot": trace_snapshot.as_json_value(),
                "workspace_snapshot": (
                    workspace_snapshot.as_json_value()
                    if workspace_snapshot is not None
                    else None
                ),
            }
            return self.checkpoint.commit_step(
                expected_previous_step_id=snapshot.last_step_id,
                step_kind=step_kind,
                action_id=action_id,
                action=action,
                sdk_session_items=session_items,
                bank_snapshot=bank_snapshot,
                guard_state=guard_state,
                usage=usage,
            )

    def checkpoint_initial_state(
        self,
        *,
        bank: Any,
        state: RuntimeCommitState,
        run_started_at_utc: str,
    ) -> CommittedTraceStep:
        if (
            state.normalized_actions
            or state.guard_state
            or state.hook_state
            or state.counts_state
            or state.usage
            or state.ledger_paths
        ):
            raise ValueError("initial runtime state must have empty mutable surfaces")
        _validate_initial_trace_bytes(state.trace_bytes)
        initial_state = _validated_initial_state(
            {"schema_version": 1, "run_started_at_utc": run_started_at_utc}
        )
        return self._commit_runtime_step(
            step_kind=INITIAL_STATE,
            action_id="runtime-phase:initial_state",
            sdk_session_items=(),
            bank=bank,
            state=state,
            phase_state=initial_state,
        )

    def commit_model_turn(
        self,
        *,
        turn_id: str,
        sdk_session_items: Sequence[Any],
        bank: Any,
        state: RuntimeCommitState,
    ) -> CommittedTraceStep:
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("turn_id must be non-empty")
        return self._commit_runtime_step(
            step_kind=MODEL_TURN,
            action_id=f"runtime-turn:{turn_id}",
            sdk_session_items=sdk_session_items,
            bank=bank,
            state=state,
            phase_state={"turn_id": turn_id},
        )

    def checkpoint_actor_complete(
        self,
        *,
        bank: Any,
        state: RuntimeCommitState,
        actor_state: Mapping[str, Any],
    ) -> CommittedTraceStep:
        try:
            normalized = validate_oai_actor_completion_state(actor_state)
        except RuntimeCheckpointDriftError as exc:
            raise ValueError("actor_state does not match the OAI recovery schema") from exc
        if normalized["schema_version"] == 2:
            binding = ArtifactBinding.from_json_value(
                normalized["safety_abstention_artifact"],
                label="safety abstention artifact",
            )
            observed = self.bind_existing_artifact(
                self.root / binding.relative_path,
                label="safety abstention artifact",
            )
            if observed != binding:
                raise ValueError("safety abstention artifact binding drift")
        return self._commit_runtime_step(
            step_kind=ACTOR_COMPLETE,
            action_id="runtime-phase:actor_complete",
            sdk_session_items=(),
            bank=bank,
            state=state,
            phase_state=normalized,
        )

    def checkpoint_evaluation_complete(
        self,
        *,
        bank: Any,
        state: RuntimeCommitState,
        evaluation_state: Mapping[str, Any],
        report_path: str | Path,
    ) -> CommittedTraceStep:
        report_binding = self.bind_existing_artifact(report_path, label="evaluation report")
        phase_state = dict(_canonical_mapping(evaluation_state, label="evaluation_state"))
        if (
            set(phase_state)
            != {"metric_summary", "report_role", "run_id", "schema_version", "task_id"}
            or phase_state.get("schema_version") != 1
            or phase_state.get("report_role")
            != "harnessaudit_evaluation_report_v1"
            or not isinstance(phase_state.get("run_id"), str)
            or not phase_state.get("run_id")
            or not isinstance(phase_state.get("task_id"), str)
            or not phase_state.get("task_id")
            or not isinstance(phase_state.get("metric_summary"), dict)
        ):
            raise ValueError("evaluation_state does not match the registered schema")
        phase_state["report"] = report_binding.as_json_value()
        return self._commit_runtime_step(
            step_kind=EVALUATION_COMPLETE,
            action_id="runtime-phase:evaluation_complete",
            sdk_session_items=(),
            bank=bank,
            state=state,
            phase_state=phase_state,
        )

    def bind_existing_artifact(self, path: str | Path, *, label: str) -> ArtifactBinding:
        relative = self._relative_path(path, label=label)
        data = _stable_read(self.root / relative)
        return ArtifactBinding(
            relative_path=relative,
            sha256=_sha256(data),
            size_bytes=len(data),
        )

    def recover(
        self,
        *,
        bank: Any,
        trace_path: str | Path,
        workspace: str | Path | None = None,
        require_resumable: bool = True,
    ) -> RuntimeRecoveryState:
        """Verify committed state, reconcile ledgers, restore bank, and rebuild trace."""

        with self._mutex:
            state = self.inspect()
            if require_resumable and state.terminal is not None:
                raise CheckpointTerminalError("terminal runtime checkpoint cannot resume")
            orphans = list(self._reconcile_ledgers(state.ledger_offsets))
            restore_values: dict[str, Any] | None = None
            if state.bank_runtime_state is not None:
                restore_values = _validated_bank_restore_values(
                    bank,
                    state.bank_runtime_state,
                )
            if state.bank_snapshot is not None:
                self.restore_bank_snapshot(bank, state.bank_snapshot)
                if restore_values is None:
                    raise RuntimeCheckpointDriftError(
                        "bank snapshot lacks bound runtime state"
                    )
                _restore_bank_runtime_state(bank, restore_values)
            elif state.bank_runtime_state is not None:
                if restore_values is None:
                    raise RuntimeCheckpointDriftError(
                        "bankless checkpoint lacks bound runtime state"
                    )
                _restore_bank_runtime_state(bank, restore_values)
            if state.workspace_snapshot is not None:
                if workspace is None:
                    raise RuntimeCheckpointDriftError("committed workspace is unavailable")
                self.restore_workspace(workspace, state.workspace_snapshot)
            trace_orphan = self.archive_and_rebuild_trace(trace_path, state.trace_snapshot)
            if trace_orphan is not None:
                orphans.append(trace_orphan)
            return RuntimeRecoveryState(
                last_step_id=state.last_step_id,
                model_turn_count=state.model_turn_count,
                normalized_actions=state.normalized_actions,
                sdk_session_items=state.sdk_session_items,
                guard_state=state.guard_state,
                hook_state=state.hook_state,
                counts_state=state.counts_state,
                usage=state.usage,
                ledger_offsets=state.ledger_offsets,
                bank_snapshot=state.bank_snapshot,
                bank_runtime_state=state.bank_runtime_state,
                initial_state_state=state.initial_state_state,
                trace_snapshot=state.trace_snapshot,
                workspace_snapshot=state.workspace_snapshot,
                actor_complete_state=state.actor_complete_state,
                evaluation_complete_state=state.evaluation_complete_state,
                terminal=state.terminal,
                recovery_orphans=tuple(orphans),
            )

    def seal_terminal_report(
        self,
        *,
        report_path: str | Path,
        status: str,
    ) -> TerminalTraceState:
        state = self.inspect()
        if not state.evaluation_complete or state.evaluation_complete_state is None:
            raise RuntimeCheckpointOrderError("terminal report precedes evaluation completion")
        report = self.bind_existing_artifact(report_path, label="terminal report")
        evaluation_report = ArtifactBinding.from_json_value(
            state.evaluation_complete_state.get("report"),
            label="evaluation report",
        )
        if report != evaluation_report:
            raise RuntimeCheckpointDriftError("terminal report differs from evaluated report")
        return self.checkpoint.mark_terminal(
            expected_last_step_id=state.last_step_id,
            status=status,
            state={
                "evaluation_step_id": state.last_step_id,
                "report": report.as_json_value(),
                "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            },
        )

    def validate_terminal_report_for_reuse(
        self,
        *,
        expected_status: str | None = None,
    ) -> ArtifactBinding:
        state = self.inspect()
        terminal = state.terminal
        if state.initial_state_state is None:
            raise RuntimeCheckpointDriftError("terminal state lacks initial baseline")
        if terminal is None:
            raise RuntimeCheckpointOrderError("checkpoint is not terminal")
        if expected_status is not None and terminal.status != expected_status:
            raise RuntimeCheckpointDriftError("terminal report status mismatch")
        report = ArtifactBinding.from_json_value(
            terminal.state.get("report"),
            label="terminal report",
        )
        self._read_artifact(report, label="terminal report")
        if state.evaluation_complete_state is None:
            raise RuntimeCheckpointDriftError("terminal state lacks evaluation completion")
        evaluated = ArtifactBinding.from_json_value(
            state.evaluation_complete_state.get("report"),
            label="evaluation report",
        )
        if report != evaluated:
            raise RuntimeCheckpointDriftError("terminal and evaluation report bindings differ")
        return report


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class DurableCheckpointSession:
    """OpenAI Agents SDK-compatible session with durable turn commits."""

    def __init__(
        self,
        underlying_session: Any,
        coordinator: RuntimeCheckpointCoordinator,
        *,
        bank: Any = None,
        state_factory: Callable[[], RuntimeCommitState] | None = None,
        allow_bankless: bool = False,
    ) -> None:
        if type(allow_bankless) is not bool:
            raise TypeError("allow_bankless must be a boolean")
        if allow_bankless and bank is not None:
            raise ValueError("allow_bankless requires bank=None")
        self.underlying_session = underlying_session
        self.coordinator = coordinator
        self._bank = bank
        self._allow_bankless = allow_bankless
        self._state_factory = state_factory
        self._prepared = False
        self._state: RuntimeRecoveryState | None = None
        self._pending_input: list[Any] = []
        self._committed_model_turns = 0

    @property
    def committed_model_turns(self) -> int:
        return self._committed_model_turns

    @property
    def is_prepared(self) -> bool:
        """Whether the wrapper may accept the next SDK session mutation."""

        return self._prepared

    @staticmethod
    def _user_only(items: Sequence[Any]) -> bool:
        return bool(items) and all(
            isinstance(item, Mapping) and item.get("role") == "user"
            for item in items
        )

    async def get_items(self, limit: int | None = None) -> list[Any]:
        get_items = getattr(self.underlying_session, "get_items", None)
        if not callable(get_items):
            raise RuntimeCheckpointError("underlying session lacks get_items(limit)")
        return list(await _maybe_await(get_items(limit=limit)))

    async def pop_item(self) -> Any:
        pop_item = getattr(self.underlying_session, "pop_item", None)
        if not callable(pop_item):
            raise RuntimeCheckpointError("underlying session lacks pop_item()")
        popped = await _maybe_await(pop_item())
        if self._pending_input:
            expected = self._pending_input.pop()
            if _canonical_value(popped, label="popped session item") != expected:
                self._prepared = False
                raise RuntimeSessionDirtyError("session pop crossed the committed prefix")
        elif popped is not None:
            self._prepared = False
        return popped

    async def clear_session(self) -> None:
        clear = getattr(self.underlying_session, "clear_session", None)
        if not callable(clear):
            raise RuntimeCheckpointError("underlying session lacks clear_session()")
        await _maybe_await(clear())
        self._pending_input.clear()
        self._prepared = False

    async def prepare(
        self,
        *,
        bank: Any = None,
        trace_path: str | Path | None = None,
        workspace: str | Path | None = None,
        recovery_state: RuntimeRecoveryState | None = None,
    ) -> RuntimeRecoveryState:
        self._prepared = False
        active_bank = bank if bank is not None else self._bank
        if self._allow_bankless:
            if active_bank is not None:
                raise RuntimeCheckpointError(
                    "bankless session cannot accept a bank instance"
                )
        elif active_bank is None:
            raise RuntimeCheckpointError("bankless session recovery is not authorized")
        if recovery_state is None:
            if trace_path is None:
                raise RuntimeCheckpointError("trace_path is required for recovery")
            state = self.coordinator.recover(
                bank=active_bank,
                trace_path=trace_path,
                workspace=workspace,
            )
        else:
            state = recovery_state
        clear = getattr(self.underlying_session, "clear_session", None)
        add = getattr(self.underlying_session, "add_items", None)
        if not callable(clear) or not callable(add):
            raise RuntimeCheckpointError(
                "underlying session must expose clear_session() and add_items(items)"
            )
        await _maybe_await(clear())
        if state.sdk_session_items:
            await _maybe_await(add(list(state.sdk_session_items)))
        self._pending_input.clear()
        self._committed_model_turns = state.model_turn_count
        self._state = state
        self._prepared = True
        return state

    async def add_items(
        self,
        items: Sequence[Any],
        *,
        turn_id: str | None = None,
        bank: Any = None,
        state: RuntimeCommitState | None = None,
    ) -> CommittedTraceStep | None:
        if not self._prepared:
            raise RuntimeSessionDirtyError("session must be prepared before adding a turn")
        normalized_items = _canonical_items(items)
        explicit_commit = turn_id is not None or bank is not None or state is not None
        add = getattr(self.underlying_session, "add_items", None)
        if not callable(add):
            raise RuntimeCheckpointError("underlying session lacks add_items(items)")
        if not explicit_commit and self._user_only(normalized_items):
            await _maybe_await(add(normalized_items))
            self._pending_input.extend(normalized_items)
            return None

        active_bank = bank if bank is not None else self._bank
        active_state = state
        if active_state is None and self._state_factory is not None:
            active_state = await _maybe_await(self._state_factory())
        if self._allow_bankless:
            if active_bank is not None:
                raise RuntimeCheckpointError(
                    "bankless session cannot accept a bank instance"
                )
        elif active_bank is None:
            raise RuntimeCheckpointError("completed turn lacks an authorized bank")
        if active_state is None:
            raise RuntimeCheckpointError("completed turn lacks runtime state")
        stable_turn_id = turn_id or f"{self._committed_model_turns + 1:08d}"
        delta = [*self._pending_input, *normalized_items]
        action_id = f"runtime-turn:{stable_turn_id}"
        if self.coordinator.has_committed_action(action_id):
            step = self.coordinator.commit_model_turn(
                turn_id=stable_turn_id,
                sdk_session_items=delta,
                bank=active_bank,
                state=active_state,
            )
            self._pending_input.clear()
            self._committed_model_turns = self.coordinator.inspect().model_turn_count
            return step
        try:
            await _maybe_await(add(normalized_items))
            step = self.coordinator.commit_model_turn(
                turn_id=stable_turn_id,
                sdk_session_items=delta,
                bank=active_bank,
                state=active_state,
            )
        except Exception:
            self._prepared = False
            raise
        self._pending_input.clear()
        self._committed_model_turns += 1
        self._state = self.coordinator.inspect()
        return step


__all__ = [
    "ACTOR_COMPLETE",
    "ArtifactBinding",
    "DurableCheckpointSession",
    "EVALUATION_COMPLETE",
    "LedgerOffsetBinding",
    "MODEL_TURN",
    "RUNTIME_SCHEMA_VERSION",
    "RuntimeCheckpointCoordinator",
    "RuntimeCheckpointDriftError",
    "RuntimeCheckpointError",
    "RuntimeCheckpointOrderError",
    "RuntimeCommitState",
    "RuntimeRecoveryState",
    "RuntimeSessionDirtyError",
]
