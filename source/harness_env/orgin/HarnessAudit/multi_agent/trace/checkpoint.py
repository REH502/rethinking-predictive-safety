"""Durable, identity-bound checkpoints for resumable HarnessAudit traces.

This module is deliberately independent of the runner and model SDK.  It
provides the persistence primitive that those layers can later integrate at
complete model/tool boundaries.  A committed step contains the append-only
SDK-session delta and the latest bindings needed to reconstruct execution
state.  SQLite supplies atomicity; the explicit hash chain detects accidental
or out-of-band mutation.

The caller owns two important responsibilities:

* ``action_id`` must be a stable logical id, reused when retrying the same
  action after an uncertain process failure.
* A bank snapshot named by ``BankSnapshotBinding`` must be immutable and
  content-addressed.  This store records the binding but does not copy it.

No model or network code is imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final


SCHEMA_VERSION: Final = 1
GENESIS_DOMAIN: Final = "harnessaudit.trace-checkpoint.genesis.v1"
STEP_DOMAIN: Final = "harnessaudit.trace-checkpoint.step.v1"
TERMINAL_DOMAIN: Final = "harnessaudit.trace-checkpoint.terminal.v1"

STEP_BEFORE_COMMIT: Final = "step.before_commit"
STEP_AFTER_COMMIT: Final = "step.after_commit"
TERMINAL_BEFORE_COMMIT: Final = "terminal.before_commit"
TERMINAL_AFTER_COMMIT: Final = "terminal.after_commit"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TraceCheckpointError(RuntimeError):
    """Base class for durable trace-checkpoint failures."""


class CheckpointIdentityError(TraceCheckpointError):
    """The requested run identity does not match the frozen checkpoint."""


class CheckpointCorruptionError(TraceCheckpointError):
    """The checkpoint is malformed or fails its integrity checks."""


class CheckpointConflictError(TraceCheckpointError):
    """A caller attempted a non-monotonic or non-idempotent commit."""


class CheckpointTerminalError(TraceCheckpointError):
    """A terminal checkpoint cannot be resumed or extended."""


class CheckpointLockError(TraceCheckpointError):
    """Another writer held the SQLite write lock past the configured timeout."""


@dataclass(frozen=True, slots=True)
class BankSnapshotBinding:
    """Content-addressed reference to an immutable, checkpoint-local bank image."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if not self.relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("bank snapshot path must be a non-empty relative path")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("bank snapshot sha256 must be 64 lowercase hex characters")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("bank snapshot size_bytes must be a non-negative integer")

    def as_json_value(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_json_value(cls, value: Mapping[str, Any]) -> BankSnapshotBinding:
        if set(value) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError("bank snapshot binding has unexpected fields")
        relative_path = value["relative_path"]
        sha256 = value["sha256"]
        size_bytes = value["size_bytes"]
        if not isinstance(relative_path, str) or not isinstance(sha256, str):
            raise ValueError("bank snapshot path and sha256 must be strings")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError("bank snapshot size_bytes must be an integer")
        return cls(relative_path=relative_path, sha256=sha256, size_bytes=size_bytes)


@dataclass(frozen=True, slots=True)
class CommittedTraceStep:
    """One atomically committed logical model/tool boundary."""

    step_id: int
    step_kind: str
    action_id: str
    action: Any
    sdk_session_items: tuple[Any, ...]
    bank_snapshot: BankSnapshotBinding | None
    guard_state: Any
    usage: Any
    payload_sha256: str
    previous_state_sha256: str
    state_sha256: str
    committed_at_utc: str


@dataclass(frozen=True, slots=True)
class TerminalTraceState:
    """Immutable terminal marker bound to the last committed step."""

    status: str
    state: Any
    final_step_id: int
    previous_state_sha256: str
    terminal_sha256: str
    committed_at_utc: str


@dataclass(frozen=True, slots=True)
class TraceCheckpointSnapshot:
    """Fully verified restart state."""

    schema_version: int
    identity: Mapping[str, Any]
    identity_sha256: str
    run_id: str
    attempt_id: str
    genesis_state_sha256: str
    steps: tuple[CommittedTraceStep, ...]
    sdk_session_items: tuple[Any, ...]
    terminal: TerminalTraceState | None

    @property
    def last_step_id(self) -> int:
        return self.steps[-1].step_id if self.steps else 0

    @property
    def last_state_sha256(self) -> str:
        if self.steps:
            return self.steps[-1].state_sha256
        return self.genesis_state_sha256

    @property
    def resumable(self) -> bool:
        return self.terminal is None


FailureInjector = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite, JSON-serializable data") from exc
    return encoded


def _canonical_mapping(value: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], str]:
    encoded = _canonical_json(value, label=label)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return decoded, encoded


def _canonical_sequence(value: Sequence[Any], *, label: str) -> tuple[list[Any], str]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a JSON array")
    encoded = _canonical_json(list(value), label=label)
    decoded = json.loads(encoded)
    if not isinstance(decoded, list):
        raise ValueError(f"{label} must be a JSON array")
    return decoded, encoded


def _decode_canonical_json(encoded: str, *, label: str) -> Any:
    try:
        decoded = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(f"invalid {label} JSON") from exc
    if _canonical_json(decoded, label=label) != encoded:
        raise CheckpointCorruptionError(f"non-canonical {label} JSON")
    return decoded


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_payload(domain: str, value: Mapping[str, Any]) -> str:
    return _sha256_text(f"{domain}\n{_canonical_json(value, label=domain)}")


def _require_nonempty_text(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableTraceCheckpoint:
    """SQLite-backed, append-only trace checkpoint.

    ``run_id`` and ``attempt_id`` are mandatory and never generated here.  A
    caller therefore cannot accidentally create a new logical trace while it
    believes it is resuming an old one.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any],
        run_id: str,
        attempt_id: str,
        busy_timeout_ms: int = 5_000,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        if isinstance(busy_timeout_ms, bool) or busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be a non-negative integer")
        self.path = Path(path)
        self.run_id = _require_nonempty_text(run_id, label="run_id")
        self.attempt_id = _require_nonempty_text(attempt_id, label="attempt_id")
        self.identity, self._identity_json = _canonical_mapping(identity, label="identity")
        self.identity_sha256 = _sha256_text(self._identity_json)
        self._busy_timeout_ms = busy_timeout_ms
        self._failure_injector = failure_injector
        self._mutex = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._known_checkpoint_files: set[Path] = set()

        existed_nonempty = self.path.exists() and self.path.stat().st_size > 0
        self._prepare_database_file()
        try:
            self._conn = sqlite3.connect(
                self.path,
                timeout=busy_timeout_ms / 1_000,
                isolation_level=None,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._configure_connection()
            self._initialise_or_validate(existed_nonempty=existed_nonempty)
            self._secure_checkpoint_files(sync_new_names=True)
        except TraceCheckpointError:
            self._close_after_initialisation_failure()
            raise
        except sqlite3.DatabaseError as exc:
            self._close_after_initialisation_failure()
            raise CheckpointCorruptionError("checkpoint is not a valid SQLite database") from exc
        except Exception:
            self._close_after_initialisation_failure()
            raise

    def _prepare_database_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise TraceCheckpointError("checkpoint path must not be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise TraceCheckpointError("checkpoint path must be a regular file")
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise TraceCheckpointError("checkpoint is closed")
        return self._conn

    def _configure_connection(self) -> None:
        conn = self._connection()
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        journal_mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
        if journal_mode != "wal":
            raise CheckpointCorruptionError("SQLite refused WAL journal mode")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA trusted_schema = OFF")

    def _begin_immediate(self) -> None:
        try:
            self._connection().execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise CheckpointLockError("checkpoint writer lock timed out") from exc
            raise

    def _initialise_or_validate(self, *, existed_nonempty: bool) -> None:
        conn = self._connection()
        with self._mutex:
            self._begin_immediate()
            try:
                self._create_schema()
                row = conn.execute("SELECT * FROM checkpoint_metadata WHERE singleton = 1").fetchone()
                if row is None:
                    if existed_nonempty:
                        raise CheckpointCorruptionError(
                            "non-empty checkpoint has no frozen metadata"
                        )
                    genesis = _hash_payload(
                        GENESIS_DOMAIN,
                        {
                            "attempt_id": self.attempt_id,
                            "identity_sha256": self.identity_sha256,
                            "run_id": self.run_id,
                            "schema_version": SCHEMA_VERSION,
                        },
                    )
                    conn.execute(
                        """
                        INSERT INTO checkpoint_metadata (
                            singleton, schema_version, identity_json, identity_sha256,
                            run_id, attempt_id, genesis_state_sha256, last_step_id,
                            terminal
                        ) VALUES (1, ?, ?, ?, ?, ?, ?, 0, 0)
                        """,
                        (
                            SCHEMA_VERSION,
                            self._identity_json,
                            self.identity_sha256,
                            self.run_id,
                            self.attempt_id,
                            genesis,
                        ),
                    )
                else:
                    self._validate_frozen_metadata(row)
                self._validate_locked()
                conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise

    def _create_schema(self) -> None:
        conn = self._connection()
        statements = (
            """
            CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                identity_json TEXT NOT NULL,
                identity_sha256 TEXT NOT NULL,
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                genesis_state_sha256 TEXT NOT NULL,
                last_step_id INTEGER NOT NULL CHECK (last_step_id >= 0),
                terminal INTEGER NOT NULL CHECK (terminal IN (0, 1))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS committed_steps (
                step_id INTEGER PRIMARY KEY CHECK (step_id > 0),
                step_kind TEXT NOT NULL,
                action_id TEXT NOT NULL UNIQUE,
                action_json TEXT NOT NULL,
                sdk_session_items_json TEXT NOT NULL,
                bank_snapshot_json TEXT NOT NULL,
                guard_state_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                previous_state_sha256 TEXT NOT NULL,
                state_sha256 TEXT NOT NULL UNIQUE,
                committed_at_utc TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS terminal_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                status TEXT NOT NULL,
                state_json TEXT NOT NULL,
                final_step_id INTEGER NOT NULL CHECK (final_step_id >= 0),
                previous_state_sha256 TEXT NOT NULL,
                terminal_sha256 TEXT NOT NULL UNIQUE,
                committed_at_utc TEXT NOT NULL
            )
            """,
            """
            CREATE TRIGGER IF NOT EXISTS checkpoint_metadata_no_delete
            BEFORE DELETE ON checkpoint_metadata
            BEGIN SELECT RAISE(ABORT, 'checkpoint metadata is immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS checkpoint_metadata_identity_immutable
            BEFORE UPDATE OF schema_version, identity_json, identity_sha256,
                             run_id, attempt_id, genesis_state_sha256
            ON checkpoint_metadata
            BEGIN SELECT RAISE(ABORT, 'checkpoint identity is immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS committed_steps_no_update
            BEFORE UPDATE ON committed_steps
            BEGIN SELECT RAISE(ABORT, 'committed steps are immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS committed_steps_no_delete
            BEFORE DELETE ON committed_steps
            BEGIN SELECT RAISE(ABORT, 'committed steps are immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS committed_steps_after_terminal
            BEFORE INSERT ON committed_steps
            WHEN EXISTS (SELECT 1 FROM terminal_state WHERE singleton = 1)
            BEGIN SELECT RAISE(ABORT, 'terminal checkpoint cannot be extended'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS terminal_state_no_update
            BEFORE UPDATE ON terminal_state
            BEGIN SELECT RAISE(ABORT, 'terminal state is immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS terminal_state_no_delete
            BEFORE DELETE ON terminal_state
            BEGIN SELECT RAISE(ABORT, 'terminal state is immutable'); END
            """,
        )
        for statement in statements:
            conn.execute(statement)

    def _validate_frozen_metadata(self, row: sqlite3.Row) -> None:
        try:
            schema_version = int(row["schema_version"])
            stored_identity_json = str(row["identity_json"])
            stored_identity_sha256 = str(row["identity_sha256"])
            stored_run_id = str(row["run_id"])
            stored_attempt_id = str(row["attempt_id"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptionError("checkpoint metadata schema is malformed") from exc
        if schema_version != SCHEMA_VERSION:
            raise CheckpointIdentityError(
                f"checkpoint schema {schema_version} is not supported by {SCHEMA_VERSION}"
            )
        decoded_identity = _decode_canonical_json(
            stored_identity_json,
            label="stored identity",
        )
        if not isinstance(decoded_identity, dict):
            raise CheckpointCorruptionError("stored identity is not a JSON object")
        if _sha256_text(stored_identity_json) != stored_identity_sha256:
            raise CheckpointCorruptionError("stored identity hash does not match its payload")
        if stored_identity_sha256 != self.identity_sha256:
            raise CheckpointIdentityError("checkpoint identity hash mismatch")
        if stored_run_id != self.run_id:
            raise CheckpointIdentityError("checkpoint run_id mismatch")
        if stored_attempt_id != self.attempt_id:
            raise CheckpointIdentityError("checkpoint attempt_id mismatch")

    def _normalise_step_payload(
        self,
        *,
        step_kind: str,
        action: Mapping[str, Any] | None,
        sdk_session_items: Sequence[Any],
        bank_snapshot: BankSnapshotBinding | None,
        guard_state: Mapping[str, Any] | None,
        usage: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        step_kind = _require_nonempty_text(step_kind, label="step_kind")
        action_value, action_json = _canonical_mapping(action or {}, label="action")
        sdk_value, sdk_json = _canonical_sequence(
            sdk_session_items,
            label="sdk_session_items",
        )
        guard_value, guard_json = _canonical_mapping(guard_state or {}, label="guard_state")
        usage_value, usage_json = _canonical_mapping(usage or {}, label="usage")
        bank_value = bank_snapshot.as_json_value() if bank_snapshot is not None else None
        bank_json = _canonical_json(bank_value, label="bank_snapshot")
        payload = {
            "action": action_value,
            "bank_snapshot": bank_value,
            "guard_state": guard_value,
            "sdk_session_items": sdk_value,
            "step_kind": step_kind,
            "usage": usage_value,
        }
        encoded = {
            "action_json": action_json,
            "bank_snapshot_json": bank_json,
            "guard_state_json": guard_json,
            "sdk_session_items_json": sdk_json,
            "usage_json": usage_json,
        }
        return payload, encoded

    def commit_step(
        self,
        *,
        expected_previous_step_id: int,
        step_kind: str,
        action_id: str,
        action: Mapping[str, Any] | None = None,
        sdk_session_items: Sequence[Any] = (),
        bank_snapshot: BankSnapshotBinding | None = None,
        guard_state: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> CommittedTraceStep:
        """Atomically append one logical step.

        A retry with the same ``action_id`` and exact payload is idempotent,
        including when the first caller crashed after SQLite committed but
        before it observed success.  Reusing an id with different content is a
        hard conflict.
        """

        if isinstance(expected_previous_step_id, bool) or expected_previous_step_id < 0:
            raise ValueError("expected_previous_step_id must be a non-negative integer")
        action_id = _require_nonempty_text(action_id, label="action_id")
        payload, encoded = self._normalise_step_payload(
            step_kind=step_kind,
            action=action,
            sdk_session_items=sdk_session_items,
            bank_snapshot=bank_snapshot,
            guard_state=guard_state,
            usage=usage,
        )
        payload_sha256 = _hash_payload(STEP_DOMAIN, payload)
        conn = self._connection()

        with self._mutex:
            self._begin_immediate()
            try:
                metadata = self._metadata_row_locked()
                if int(metadata["terminal"]):
                    raise CheckpointTerminalError("terminal checkpoint cannot be extended")
                existing = conn.execute(
                    "SELECT * FROM committed_steps WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if existing is not None:
                    step = self._step_from_row(existing)
                    if step.payload_sha256 != payload_sha256:
                        raise CheckpointConflictError(
                            "action_id already exists with a different payload"
                        )
                    conn.commit()
                    return step

                last_step_id = int(metadata["last_step_id"])
                if expected_previous_step_id != last_step_id:
                    raise CheckpointConflictError(
                        "stale step predecessor: expected "
                        f"{expected_previous_step_id}, durable state is {last_step_id}"
                    )
                previous_state_sha256 = self._last_state_sha256_locked(metadata)
                step_id = last_step_id + 1
                state_sha256 = _hash_payload(
                    STEP_DOMAIN,
                    {
                        "action_id": action_id,
                        "payload_sha256": payload_sha256,
                        "previous_state_sha256": previous_state_sha256,
                        "step_id": step_id,
                    },
                )
                committed_at_utc = _utc_now()
                conn.execute(
                    """
                    INSERT INTO committed_steps (
                        step_id, step_kind, action_id, action_json,
                        sdk_session_items_json, bank_snapshot_json,
                        guard_state_json, usage_json, payload_sha256,
                        previous_state_sha256, state_sha256, committed_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        step_id,
                        payload["step_kind"],
                        action_id,
                        encoded["action_json"],
                        encoded["sdk_session_items_json"],
                        encoded["bank_snapshot_json"],
                        encoded["guard_state_json"],
                        encoded["usage_json"],
                        payload_sha256,
                        previous_state_sha256,
                        state_sha256,
                        committed_at_utc,
                    ),
                )
                conn.execute(
                    "UPDATE checkpoint_metadata SET last_step_id = ? WHERE singleton = 1",
                    (step_id,),
                )
                self._inject(STEP_BEFORE_COMMIT)
                conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise

            self._secure_checkpoint_files(sync_new_names=True)
            self._inject(STEP_AFTER_COMMIT)
            row = conn.execute(
                "SELECT * FROM committed_steps WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            if row is None:
                raise CheckpointCorruptionError("committed step disappeared")
            return self._step_from_row(row)

    def mark_terminal(
        self,
        *,
        expected_last_step_id: int,
        status: str,
        state: Mapping[str, Any] | None = None,
    ) -> TerminalTraceState:
        """Atomically seal the checkpoint at its current last step."""

        if isinstance(expected_last_step_id, bool) or expected_last_step_id < 0:
            raise ValueError("expected_last_step_id must be a non-negative integer")
        status = _require_nonempty_text(status, label="status")
        state_value, state_json = _canonical_mapping(state or {}, label="terminal state")
        conn = self._connection()

        with self._mutex:
            self._begin_immediate()
            try:
                metadata = self._metadata_row_locked()
                existing = conn.execute(
                    "SELECT * FROM terminal_state WHERE singleton = 1"
                ).fetchone()
                if existing is not None:
                    terminal = self._terminal_from_row(existing)
                    if (
                        terminal.status == status
                        and terminal.state == state_value
                        and terminal.final_step_id == expected_last_step_id
                    ):
                        conn.commit()
                        return terminal
                    raise CheckpointTerminalError("checkpoint already has a terminal state")

                last_step_id = int(metadata["last_step_id"])
                if expected_last_step_id != last_step_id:
                    raise CheckpointConflictError(
                        "stale terminal predecessor: expected "
                        f"{expected_last_step_id}, durable state is {last_step_id}"
                    )
                previous_state_sha256 = self._last_state_sha256_locked(metadata)
                terminal_sha256 = _hash_payload(
                    TERMINAL_DOMAIN,
                    {
                        "final_step_id": last_step_id,
                        "previous_state_sha256": previous_state_sha256,
                        "state": state_value,
                        "status": status,
                    },
                )
                committed_at_utc = _utc_now()
                conn.execute(
                    """
                    INSERT INTO terminal_state (
                        singleton, status, state_json, final_step_id,
                        previous_state_sha256, terminal_sha256, committed_at_utc
                    ) VALUES (1, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        status,
                        state_json,
                        last_step_id,
                        previous_state_sha256,
                        terminal_sha256,
                        committed_at_utc,
                    ),
                )
                conn.execute(
                    "UPDATE checkpoint_metadata SET terminal = 1 WHERE singleton = 1"
                )
                self._inject(TERMINAL_BEFORE_COMMIT)
                conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise

            self._secure_checkpoint_files(sync_new_names=True)
            self._inject(TERMINAL_AFTER_COMMIT)
            row = conn.execute("SELECT * FROM terminal_state WHERE singleton = 1").fetchone()
            if row is None:
                raise CheckpointCorruptionError("terminal state disappeared")
            return self._terminal_from_row(row)

    def load(self) -> TraceCheckpointSnapshot:
        """Read and verify the complete committed prefix.

        SQLite never exposes an uncommitted transaction, so an interrupted
        pre-commit step is absent here.  Hash-chain and metadata checks reject
        any committed but internally inconsistent state.
        """

        conn = self._connection()
        with self._mutex:
            try:
                conn.execute("BEGIN")
                snapshot = self._validate_locked()
                conn.commit()
                return snapshot
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise

    def load_for_resume(self) -> TraceCheckpointSnapshot:
        """Return restart state, refusing a checkpoint already marked terminal."""

        snapshot = self.load()
        if snapshot.terminal is not None:
            raise CheckpointTerminalError(
                f"checkpoint is terminal ({snapshot.terminal.status}) and cannot resume"
            )
        return snapshot

    def _validate_locked(self) -> TraceCheckpointSnapshot:
        conn = self._connection()
        metadata = self._metadata_row_locked()
        self._validate_frozen_metadata(metadata)
        stored_identity = _decode_canonical_json(
            str(metadata["identity_json"]),
            label="stored identity",
        )
        genesis = _hash_payload(
            GENESIS_DOMAIN,
            {
                "attempt_id": self.attempt_id,
                "identity_sha256": self.identity_sha256,
                "run_id": self.run_id,
                "schema_version": SCHEMA_VERSION,
            },
        )
        if str(metadata["genesis_state_sha256"]) != genesis:
            raise CheckpointCorruptionError("genesis state hash mismatch")

        rows = conn.execute("SELECT * FROM committed_steps ORDER BY step_id").fetchall()
        steps: list[CommittedTraceStep] = []
        all_session_items: list[Any] = []
        previous_hash = genesis
        for expected_step_id, row in enumerate(rows, start=1):
            step = self._step_from_row(row)
            if step.step_id != expected_step_id:
                raise CheckpointCorruptionError("committed step ids are not contiguous")
            if step.previous_state_sha256 != previous_hash:
                raise CheckpointCorruptionError("committed step hash chain is broken")
            previous_hash = step.state_sha256
            steps.append(step)
            all_session_items.extend(step.sdk_session_items)
        if int(metadata["last_step_id"]) != len(steps):
            raise CheckpointCorruptionError("metadata last_step_id does not match step rows")

        terminal_row = conn.execute(
            "SELECT * FROM terminal_state WHERE singleton = 1"
        ).fetchone()
        terminal = self._terminal_from_row(terminal_row) if terminal_row is not None else None
        terminal_flag = bool(int(metadata["terminal"]))
        if terminal_flag != (terminal is not None):
            raise CheckpointCorruptionError("metadata terminal flag does not match terminal row")
        if terminal is not None:
            if terminal.final_step_id != len(steps):
                raise CheckpointCorruptionError("terminal state is bound to the wrong step")
            if terminal.previous_state_sha256 != previous_hash:
                raise CheckpointCorruptionError("terminal state hash chain is broken")

        return TraceCheckpointSnapshot(
            schema_version=SCHEMA_VERSION,
            identity=stored_identity,
            identity_sha256=self.identity_sha256,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            genesis_state_sha256=genesis,
            steps=tuple(steps),
            sdk_session_items=tuple(all_session_items),
            terminal=terminal,
        )

    def _metadata_row_locked(self) -> sqlite3.Row:
        rows = self._connection().execute("SELECT * FROM checkpoint_metadata").fetchall()
        if len(rows) != 1 or int(rows[0]["singleton"]) != 1:
            raise CheckpointCorruptionError("checkpoint must contain exactly one metadata row")
        return rows[0]

    def _last_state_sha256_locked(self, metadata: sqlite3.Row) -> str:
        last_step_id = int(metadata["last_step_id"])
        if last_step_id == 0:
            return str(metadata["genesis_state_sha256"])
        row = self._connection().execute(
            "SELECT state_sha256 FROM committed_steps WHERE step_id = ?",
            (last_step_id,),
        ).fetchone()
        if row is None:
            raise CheckpointCorruptionError("metadata refers to a missing last step")
        return str(row["state_sha256"])

    def _step_from_row(self, row: sqlite3.Row) -> CommittedTraceStep:
        try:
            step_id = int(row["step_id"])
            step_kind = str(row["step_kind"])
            action_id = str(row["action_id"])
            action = _decode_canonical_json(str(row["action_json"]), label="action")
            session_items = _decode_canonical_json(
                str(row["sdk_session_items_json"]),
                label="sdk_session_items",
            )
            bank_value = _decode_canonical_json(
                str(row["bank_snapshot_json"]),
                label="bank_snapshot",
            )
            guard_state = _decode_canonical_json(
                str(row["guard_state_json"]),
                label="guard_state",
            )
            usage = _decode_canonical_json(str(row["usage_json"]), label="usage")
            payload_sha256 = str(row["payload_sha256"])
            previous_state_sha256 = str(row["previous_state_sha256"])
            state_sha256 = str(row["state_sha256"])
            committed_at_utc = str(row["committed_at_utc"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptionError("committed step row is malformed") from exc
        if not isinstance(action, dict) or not isinstance(session_items, list):
            raise CheckpointCorruptionError("committed action/session shape is invalid")
        if not isinstance(guard_state, dict) or not isinstance(usage, dict):
            raise CheckpointCorruptionError("committed guard/usage shape is invalid")
        bank_snapshot: BankSnapshotBinding | None
        if bank_value is None:
            bank_snapshot = None
        elif isinstance(bank_value, dict):
            try:
                bank_snapshot = BankSnapshotBinding.from_json_value(bank_value)
            except ValueError as exc:
                raise CheckpointCorruptionError("bank snapshot binding is malformed") from exc
        else:
            raise CheckpointCorruptionError("bank snapshot binding is not an object")
        payload = {
            "action": action,
            "bank_snapshot": bank_value,
            "guard_state": guard_state,
            "sdk_session_items": session_items,
            "step_kind": step_kind,
            "usage": usage,
        }
        expected_payload_sha256 = _hash_payload(STEP_DOMAIN, payload)
        if payload_sha256 != expected_payload_sha256:
            raise CheckpointCorruptionError("committed step payload hash mismatch")
        expected_state_sha256 = _hash_payload(
            STEP_DOMAIN,
            {
                "action_id": action_id,
                "payload_sha256": payload_sha256,
                "previous_state_sha256": previous_state_sha256,
                "step_id": step_id,
            },
        )
        if state_sha256 != expected_state_sha256:
            raise CheckpointCorruptionError("committed step state hash mismatch")
        return CommittedTraceStep(
            step_id=step_id,
            step_kind=step_kind,
            action_id=action_id,
            action=action,
            sdk_session_items=tuple(session_items),
            bank_snapshot=bank_snapshot,
            guard_state=guard_state,
            usage=usage,
            payload_sha256=payload_sha256,
            previous_state_sha256=previous_state_sha256,
            state_sha256=state_sha256,
            committed_at_utc=committed_at_utc,
        )

    def _terminal_from_row(self, row: sqlite3.Row) -> TerminalTraceState:
        try:
            status = str(row["status"])
            state = _decode_canonical_json(str(row["state_json"]), label="terminal state")
            final_step_id = int(row["final_step_id"])
            previous_state_sha256 = str(row["previous_state_sha256"])
            terminal_sha256 = str(row["terminal_sha256"])
            committed_at_utc = str(row["committed_at_utc"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptionError("terminal state row is malformed") from exc
        if not isinstance(state, dict):
            raise CheckpointCorruptionError("terminal state is not a JSON object")
        expected_hash = _hash_payload(
            TERMINAL_DOMAIN,
            {
                "final_step_id": final_step_id,
                "previous_state_sha256": previous_state_sha256,
                "state": state,
                "status": status,
            },
        )
        if terminal_sha256 != expected_hash:
            raise CheckpointCorruptionError("terminal state hash mismatch")
        return TerminalTraceState(
            status=status,
            state=state,
            final_step_id=final_step_id,
            previous_state_sha256=previous_state_sha256,
            terminal_sha256=terminal_sha256,
            committed_at_utc=committed_at_utc,
        )

    def durability_settings(self) -> dict[str, Any]:
        """Return effective SQLite settings for audit evidence and tests."""

        with self._mutex:
            conn = self._connection()
            return {
                "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
                "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
                "busy_timeout_ms": int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
            }

    def _inject(self, phase: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(phase)

    def _secure_checkpoint_files(self, *, sync_new_names: bool) -> None:
        candidates = (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        )
        present: set[Path] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise TraceCheckpointError("checkpoint SQLite files must be regular files")
            os.chmod(candidate, 0o600)
            present.add(candidate)
        if sync_new_names and present - self._known_checkpoint_files:
            _fsync_directory(self.path.parent)
        self._known_checkpoint_files = present

    def _close_after_initialisation_failure(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def close(self) -> None:
        """Close the SQLite handle; committed state remains restartable."""

        with self._mutex:
            if self._conn is None:
                return
            self._conn.close()
            self._conn = None
            self._secure_checkpoint_files(sync_new_names=False)

    def __enter__(self) -> DurableTraceCheckpoint:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "BankSnapshotBinding",
    "CheckpointConflictError",
    "CheckpointCorruptionError",
    "CheckpointIdentityError",
    "CheckpointLockError",
    "CheckpointTerminalError",
    "CommittedTraceStep",
    "DurableTraceCheckpoint",
    "SCHEMA_VERSION",
    "STEP_AFTER_COMMIT",
    "STEP_BEFORE_COMMIT",
    "TERMINAL_AFTER_COMMIT",
    "TERMINAL_BEFORE_COMMIT",
    "TerminalTraceState",
    "TraceCheckpointError",
    "TraceCheckpointSnapshot",
]
