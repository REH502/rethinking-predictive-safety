"""HarnessAudit v7 replay-unit validation and child failure boundary.

The module is versioned so historical v6 implementation hashes stay intact.
It contains no scheduler; non-prefix batch scheduling is Task 3.
"""

from __future__ import annotations

from contextlib import contextmanager
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Iterable, Literal, Mapping

import requests

from .ha_checkpoint_state_v7 import (
    ArtifactBinding,
    ReplayUnit,
    TaskRecoveryState,
    append_task_event,
    build_replay_unit_plan,
    read_task_history,
    record_interrupted_attempt,
    reduce_task_history,
    write_checkpoint_snapshot,
)
from .ha_transport_observer_v7 import (
    CompletedResponseSemanticInvalid,
    TransportExhausted,
    V7TransportObserver,
)


ROLLOUT_ALIAS = {
    "direct_multi_horizon": "direct_multi",
    "iterative_wm_rollout": "iterative_wm",
}
POLICY_ALIAS = {
    "full_harnessaudit_rules": "full",
    "train_policy": "train",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")
_RETRYABLE_HTTP = {404, 408, 409, 425, 429}
_LEDGER_OUTCOMES = {
    "transport_error",
    "retryable_http",
    "completed_response",
    "semantic_invalid",
    "nonretryable_http",
}
_TRANSPORT_EXCEPTION_TYPES = {
    "Timeout",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectionError",
    "ProxyError",
    "SSLError",
}
_TERMINAL_EXCEPTION_TYPES = {
    "CompletedResponseSemanticInvalid",
    "LLMInvalidContentError",
    "PredictionDecisionError",
    "ModelOutputValidationError",
}
_DECISION_OUTCOME = "prediction_recorded_original_executed"
_TRANSPORT_LEDGER_FIELDS = {
    "schema_version",
    "artifact_type",
    "semantic_group",
    "transport_attempt",
    "body_sha256",
    "body_size",
    "outcome",
    "http_status",
    "exception_type",
    "timestamp_utc",
}


class IntegrityFailure(RuntimeError):
    """A path, identity, schema, hash, or programming failure."""


class UnitState(str, Enum):
    ABSENT = "ABSENT"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class UnitExpectation:
    unit: ReplayUnit
    run_root: Path
    task_id: str
    run_id: str
    result_dir: Path
    result_path: Path
    guard_run_dir: Path
    decision_path: Path
    selected_point_ids: tuple[int, ...]
    selected_sequence_nums: tuple[int, ...]
    point_signature: str
    cache_signature: str
    formal_annotation_horizon: int
    source_report_path: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.run_id:
            raise ValueError("unit expectation identity must be nonempty")
        if self.unit.rollout_mode not in ROLLOUT_ALIAS:
            raise ValueError("unsupported rollout mode")
        if self.unit.policy_mode not in POLICY_ALIAS:
            raise ValueError("unsupported policy mode")
        if self.selected_point_ids != tuple(range(1, len(self.selected_point_ids) + 1)):
            raise ValueError("selected point IDs must be consecutive from one")
        if len(self.selected_point_ids) != len(self.selected_sequence_nums):
            raise ValueError("point IDs and sequence numbers differ in length")
        if not self.selected_point_ids:
            raise ValueError("a replay unit requires prediction points")
        if not re.fullmatch(r"[0-9a-f]{8}", self.point_signature):
            raise ValueError("point signature must be lowercase 8-hex")
        if not re.fullmatch(r"[0-9a-f]{12}", self.cache_signature):
            raise ValueError("cache signature must be lowercase 12-hex")
        for path in (
            self.run_root,
            self.result_dir,
            self.result_path,
            self.guard_run_dir,
            self.decision_path,
        ):
            if not Path(path).is_absolute():
                raise ValueError("unit paths must be absolute")

@dataclass(frozen=True)
class UnitBinding:
    unit_key: str
    task_id: str
    run_id: str
    rollout_mode: str
    policy_mode: str
    horizon: int
    selected_point_ids: tuple[int, ...]
    selected_sequence_nums: tuple[int, ...]
    cache_signature: str
    result: ArtifactBinding
    decisions: ArtifactBinding
    decision_count: int


@dataclass(frozen=True)
class UnitDiscovery:
    state: UnitState
    expectation: UnitExpectation
    binding: UnitBinding | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PathArtifactBinding(ArtifactBinding):
    @property
    def path_obj(self) -> Path:
        return Path(self.path)


@dataclass(frozen=True)
class PartialArchive:
    path: Path
    inventory: PathArtifactBinding


@dataclass(frozen=True)
class DanglingUnitReconciliation:
    kind: Literal["reconciled-complete", "needs-pause"]
    binding: UnitBinding | None
    policy_cache_fingerprint: ArtifactBinding

    def __post_init__(self) -> None:
        if self.kind not in {"reconciled-complete", "needs-pause"}:
            raise ValueError("unknown dangling-unit reconciliation kind")
        if not isinstance(self.policy_cache_fingerprint, ArtifactBinding):
            raise ValueError("dangling reconciliation requires policy fingerprint evidence")
        if self.binding is not None and not isinstance(self.binding, UnitBinding):
            raise ValueError("dangling reconciliation binding is invalid")
        if self.kind == "reconciled-complete" and self.binding is None:
            raise ValueError("reconciled complete unit requires a binding")
        if self.kind == "needs-pause" and self.binding is not None:
            raise ValueError("needs-pause reconciliation cannot carry a binding")


FailureKind = Literal[
    "recoverable_transport", "terminal_model_invalid", "integrity_failure"
]


@dataclass(frozen=True)
class FailureClassification:
    kind: FailureKind
    exception_type: str
    message: str
    http_status: int | None

    def __post_init__(self) -> None:
        if self.kind not in {
            "recoverable_transport",
            "terminal_model_invalid",
            "integrity_failure",
        }:
            raise ValueError("unknown failure kind")
        if not _TOKEN_RE.fullmatch(self.exception_type):
            raise ValueError("exception_type must be a sanitized token")
        if not _TOKEN_RE.fullmatch(self.message):
            raise ValueError("message must be a sanitized token")
        if self.http_status is not None and (
            isinstance(self.http_status, bool)
            or not isinstance(self.http_status, int)
            or not 100 <= self.http_status <= 599
        ):
            raise ValueError("http_status must be null or a valid status")


@dataclass(frozen=True)
class ChildOutcome:
    task_id: str
    index: int
    attempt: int
    kind: Literal["complete", "paused", "terminal-model-invalid"]
    completed_unit_keys: tuple[str, ...]
    child_result: ArtifactBinding | None
    failure: FailureClassification | None
    policy_cache_fingerprint: ArtifactBinding

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not _TOKEN_RE.fullmatch(self.task_id):
            raise ValueError("task_id must be a nonempty token")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.completed_unit_keys, tuple):
            raise ValueError("completed unit keys must be a tuple")
        canonical_keys = tuple(
            unit.key
            for unit in build_replay_unit_plan(
                rollout_modes=("direct_multi_horizon", "iterative_wm_rollout"),
                policy_modes=("train_policy", "full_harnessaudit_rules"),
                horizons=(1, 2, 3, 4, 5, 6),
            )
        )
        if (
            len(set(self.completed_unit_keys)) != len(self.completed_unit_keys)
            or self.completed_unit_keys != canonical_keys[: len(self.completed_unit_keys)]
        ):
            raise ValueError("completed units must be a strict valid plan prefix")
        if not isinstance(self.policy_cache_fingerprint, ArtifactBinding):
            raise ValueError("child outcome requires a policy-cache fingerprint")
        if self.failure is not None and not isinstance(self.failure, FailureClassification):
            raise ValueError("failure must be a FailureClassification")
        if self.child_result is not None and not isinstance(self.child_result, ArtifactBinding):
            raise ValueError("child_result must be an ArtifactBinding")
        if self.kind not in {"complete", "paused", "terminal-model-invalid"}:
            raise ValueError("unknown child outcome")
        if self.kind == "complete":
            if self.failure is not None:
                raise ValueError("complete child cannot carry a failure")
            if not isinstance(self.child_result, ArtifactBinding):
                raise ValueError("complete child requires child_result evidence")
            if self.completed_unit_keys != canonical_keys:
                raise ValueError("complete child requires the full replay-unit plan")
        elif self.failure is None or self.failure.kind == "integrity_failure":
            raise ValueError("non-complete child requires a non-integrity failure")
        elif self.child_result is not None:
            raise ValueError("non-complete child cannot carry child_result evidence")
        if self.kind == "paused" and self.failure is not None and self.failure.kind != "recoverable_transport":
            raise ValueError("paused child requires recoverable transport failure")
        if self.kind == "terminal-model-invalid" and self.failure is not None and self.failure.kind != "terminal_model_invalid":
            raise ValueError("terminal model-invalid child requires model failure")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_symlink_components(path: Path) -> None:
    absolute = _absolute(path)
    for candidate in [*reversed(absolute.parents), absolute]:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise IntegrityFailure("path contains a symlink")


def _safe_path(path: Path, *, run_root: Path, must_exist: bool = False) -> Path:
    root = _absolute(run_root)
    _assert_no_symlink_components(root)
    if not root.is_dir():
        raise IntegrityFailure("run_root is not a real directory")
    candidate = _absolute(path)
    if not _is_relative_to(candidate, root):
        raise IntegrityFailure("path is outside run_root")
    _assert_no_symlink_components(candidate)
    if must_exist and not candidate.exists():
        raise IntegrityFailure("required artifact is missing")
    return candidate


def _ensure_directory(path: Path, *, run_root: Path) -> Path:
    root = _safe_path(run_root, run_root=run_root)
    destination = _absolute(path)
    if not _is_relative_to(destination, root):
        raise IntegrityFailure("directory is outside run_root")
    current = root
    for part in destination.relative_to(root).parts:
        current = current / part
        _assert_no_symlink_components(current)
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise IntegrityFailure("path component is not a directory")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, run_root: Path) -> ArtifactBinding:
    candidate = _safe_path(path, run_root=run_root, must_exist=True)
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise IntegrityFailure("artifact must be a regular file")
    return ArtifactBinding(
        path=str(candidate),
        size=metadata.st_size,
        sha256=_sha256(candidate),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityFailure("result JSON is malformed") from exc
    if not isinstance(value, dict):
        raise IntegrityFailure("result JSON must be an object")
    return value


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IntegrityFailure("decision JSONL contains a blank line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IntegrityFailure("decision JSONL is malformed") from exc
                if not isinstance(value, dict):
                    raise IntegrityFailure("decision JSONL row must be an object")
                rows.append(value)
    except OSError as exc:
        raise IntegrityFailure("decision JSONL cannot be read") from exc
    if not rows:
        raise IntegrityFailure("decision JSONL is empty")
    return rows


def _invalid(expectation: UnitExpectation, reason: str) -> UnitDiscovery:
    return UnitDiscovery(UnitState.INVALID, expectation, reason=reason)


def inspect_unit(expectation: UnitExpectation) -> UnitDiscovery:
    """Classify deterministic replay evidence without trusting the legacy bool."""

    try:
        root = _safe_path(expectation.run_root, run_root=expectation.run_root)
        result_dir = _safe_path(expectation.result_dir, run_root=root)
        result_path = _safe_path(expectation.result_path, run_root=root)
        guard_dir = _safe_path(expectation.guard_run_dir, run_root=root)
        decision_path = _safe_path(expectation.decision_path, run_root=root)
        if result_path.parent != result_dir:
            return _invalid(expectation, "result path is not in deterministic result directory")
        if decision_path.parent != guard_dir:
            return _invalid(expectation, "decision path is not in deterministic guard directory")
        if result_path.name != f"{expectation.run_id}.json":
            return _invalid(expectation, "result filename does not match run ID")
        if guard_dir.name != expectation.run_id:
            return _invalid(expectation, "guard directory does not match run ID")
        if not expectation.run_id.endswith(f"-{expectation.cache_signature}"):
            return _invalid(expectation, "run ID does not bind cache signature")
        if f"-{expectation.point_signature}-" not in expectation.run_id:
            return _invalid(expectation, "run ID does not bind point signature")

        candidates: list[Path] = []
        if result_dir.exists():
            if not result_dir.is_dir():
                return _invalid(expectation, "result directory is not a directory")
            for candidate in sorted(result_dir.glob("*.json")):
                _assert_no_symlink_components(candidate)
                if not stat.S_ISREG(candidate.lstat().st_mode):
                    return _invalid(expectation, "result candidate is not a regular file")
                candidates.append(candidate)
        guard_parent = guard_dir.parent
        guard_candidates: list[Path] = []
        if guard_parent.exists():
            for candidate in sorted(guard_parent.iterdir()):
                _assert_no_symlink_components(candidate)
                if candidate.is_dir():
                    guard_candidates.append(candidate)
                elif candidate.exists():
                    return _invalid(expectation, "guard candidate is not a directory")
        any_expected = result_path.exists() or guard_dir.exists() or decision_path.exists()
        if not candidates and not guard_candidates and not any_expected:
            return UnitDiscovery(UnitState.ABSENT, expectation)
        if len(candidates) > 1 or any(candidate != result_path for candidate in candidates):
            return _invalid(expectation, "multiple or unexpected result candidates")
        if len(guard_candidates) > 1 or any(candidate != guard_dir for candidate in guard_candidates):
            return _invalid(expectation, "multiple or unexpected guard candidates")
        if not result_path.exists() or not decision_path.exists():
            return UnitDiscovery(UnitState.PARTIAL, expectation, reason="unit artifacts are incomplete")
        if not stat.S_ISREG(result_path.lstat().st_mode) or not stat.S_ISREG(decision_path.lstat().st_mode):
            return _invalid(expectation, "unit artifacts must be regular files")

        # The unchanged predicate is intentionally only a cheap prefilter.  A
        # true result never bypasses the stronger identity/path/hash checks
        # below.
        from .predict_only_pipeline import _cached_ha_replay_is_complete

        if not _cached_ha_replay_is_complete(
            result_path=result_path,
            decision_path=decision_path,
            run_id=expectation.run_id,
            task_id=expectation.task_id,
            expected_decision_count=len(expectation.selected_point_ids),
        ):
            return _invalid(expectation, "legacy completeness prefilter rejected unit evidence")

        report = _read_json_object(result_path)
        decisions = _read_jsonl_strict(decision_path)
        replay = report.get("safepred_offline_replay")
        if not isinstance(replay, dict):
            return _invalid(expectation, "missing replay identity")
        expected_replay = {
            "rollout_mode": expectation.unit.rollout_mode,
            "policy_mode": expectation.unit.policy_mode,
            "intervention_mode": "prediction_only",
            "prediction_steps": expectation.unit.horizon,
            "formal_annotation_horizon": expectation.formal_annotation_horizon,
            "prediction_point_scheme": "explicit",
            "prediction_point_count": len(expectation.selected_point_ids),
            "replayed_prediction_points": len(expectation.selected_point_ids),
            "expected_decision_count": len(expectation.selected_point_ids),
            "point_signature": expectation.point_signature,
            "decision_file": str(decision_path),
        }
        if str(report.get("run_id") or "") != expectation.run_id:
            return _invalid(expectation, "result run ID mismatch")
        if str(report.get("task_id") or "") != expectation.task_id:
            return _invalid(expectation, "result task ID mismatch")
        if str(report.get("decision_file") or "") != str(decision_path):
            return _invalid(expectation, "top-level decision path mismatch")
        if any(replay.get(key) != value for key, value in expected_replay.items()):
            return _invalid(expectation, "replay identity/config mismatch")
        if expectation.source_report_path is not None and replay.get("source_report") != expectation.source_report_path:
            return _invalid(expectation, "source report identity mismatch")
        expected_direct = (
            [expectation.unit.horizon]
            if expectation.unit.rollout_mode == "direct_multi_horizon"
            else [1, 2, 3, 4, 5, 6]
        )
        if replay.get("direct_multi_horizons") != expected_direct:
            return _invalid(expectation, "endpoint horizon identity mismatch")
        selected = replay.get("selected_prediction_points")
        if not isinstance(selected, list) or len(selected) != len(expectation.selected_point_ids):
            return _invalid(expectation, "selected point count mismatch")
        observed_points = tuple(item.get("point_num") for item in selected if isinstance(item, dict))
        observed_sequences = tuple(item.get("sequence_num") for item in selected if isinstance(item, dict))
        if observed_points != expectation.selected_point_ids or observed_sequences != expectation.selected_sequence_nums:
            return _invalid(expectation, "selected point identity mismatch")

        if len(decisions) != len(expectation.selected_point_ids):
            return _invalid(expectation, "decision count is truncated")
        if tuple(row.get("decision_id") for row in decisions) != expectation.selected_point_ids:
            return _invalid(expectation, "decision IDs are not exact and consecutive")
        for row in decisions:
            checks = {
                "run_id": expectation.run_id,
                "task_id": expectation.task_id,
                "policy_mode": expectation.unit.policy_mode,
                "intervention_mode": "prediction_only",
                "horizon": expectation.unit.horizon,
                "prediction_steps": expectation.unit.horizon,
                "skipped_by_prediction_interval": False,
                "intervention_outcome": _DECISION_OUTCOME,
            }
            if any(row.get(key) != value for key, value in checks.items()):
                return _invalid(expectation, "decision identity/config mismatch")
            if row.get("error") not in (None, False, ""):
                return _invalid(expectation, "decision contains model/validator error")

        binding = UnitBinding(
            unit_key=expectation.unit.key,
            task_id=expectation.task_id,
            run_id=expectation.run_id,
            rollout_mode=expectation.unit.rollout_mode,
            policy_mode=expectation.unit.policy_mode,
            horizon=expectation.unit.horizon,
            selected_point_ids=expectation.selected_point_ids,
            selected_sequence_nums=expectation.selected_sequence_nums,
            cache_signature=expectation.cache_signature,
            result=_artifact(result_path, run_root=root),
            decisions=_artifact(decision_path, run_root=root),
            decision_count=len(decisions),
        )
        return UnitDiscovery(UnitState.COMPLETE, expectation, binding=binding)
    except IntegrityFailure as exc:
        return _invalid(expectation, str(exc))
    except (OSError, ValueError, TypeError) as exc:
        return _invalid(expectation, f"{type(exc).__name__} during unit validation")


def validate_unit_binding(binding: UnitBinding, expectation: UnitExpectation) -> UnitBinding:
    """Rehash a bound unit and require exact independently rediscovered identity."""

    if binding.unit_key != expectation.unit.key:
        raise IntegrityFailure("unit binding key mismatch")
    for artifact, expected_path in (
        (binding.result, expectation.result_path),
        (binding.decisions, expectation.decision_path),
    ):
        current = _artifact(Path(artifact.path), run_root=expectation.run_root)
        if Path(artifact.path) != expected_path or current != artifact:
            raise IntegrityFailure("unit binding artifact hash/size/path mismatch")
    discovery = inspect_unit(expectation)
    if discovery.state is not UnitState.COMPLETE or discovery.binding != binding:
        raise IntegrityFailure("unit binding no longer has complete strict identity")
    return binding


def discover_complete_units(
    expectations: Iterable[UnitExpectation],
) -> tuple[UnitBinding, ...]:
    bindings: list[UnitBinding] = []
    for expectation in expectations:
        discovery = inspect_unit(expectation)
        if discovery.state is UnitState.INVALID:
            raise IntegrityFailure(discovery.reason or "unit evidence is invalid")
        if discovery.state is UnitState.COMPLETE:
            assert discovery.binding is not None
            bindings.append(discovery.binding)
    return tuple(bindings)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _snapshot_regular_file(source: Path, destination: Path) -> None:
    """Copy one prevalidated regular file and durably flush the snapshot."""

    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    write_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        write_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, read_flags)
    destination_fd: int | None = None
    try:
        source_mode = os.fstat(source_fd).st_mode
        if not stat.S_ISREG(source_mode):
            raise IntegrityFailure("partial evidence changed before snapshot")
        destination_fd = os.open(destination, write_flags, 0o600)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("partial snapshot write made no progress")
                view = view[written:]
        os.fsync(destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def _publish_partial_archive(staging: Path, final: Path, parent: Path) -> None:
    os.replace(staging, final)
    _fsync_directory(parent)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, payload: dict[str, Any], *, run_root: Path) -> None:
    destination = _safe_path(path, run_root=run_root)
    _ensure_directory(destination.parent, run_root=run_root)
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
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def archive_partial_unit(
    expectation: UnitExpectation,
    *,
    attempt_dir: Path,
    unit_ordinal: int,
) -> PartialArchive:
    """Publish a crash-safe snapshot of a whole ambiguous replay unit."""

    if isinstance(unit_ordinal, bool) or not isinstance(unit_ordinal, int) or unit_ordinal < 0:
        raise ValueError("unit_ordinal must be a nonnegative integer")
    root = expectation.run_root
    attempt = _ensure_directory(attempt_dir, run_root=root)
    unit_dir = _ensure_directory(attempt / f"unit-{unit_ordinal:02d}", run_root=root)
    final = unit_dir / "partial"
    if final.exists():
        raise IntegrityFailure("partial archive already exists")
    orphaned = sorted(
        path for path in unit_dir.iterdir() if path.name.startswith(".partial.")
    )
    if orphaned:
        raise IntegrityFailure("orphan partial staging evidence requires inspection")
    guard = _safe_path(expectation.guard_run_dir, run_root=root)
    result = _safe_path(expectation.result_path, run_root=root)
    guard_directories: list[Path] = []
    guard_files: list[Path] = []
    # Validate every source entry before creating the transaction.  Originals
    # are never renamed or deleted, including after successful publication.
    if guard.exists():
        if not guard.is_dir():
            raise IntegrityFailure("partial guard evidence is not a directory")
        for path in [guard, *sorted(guard.rglob("*"))]:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise IntegrityFailure("partial guard evidence contains a symlink")
            if path != guard and not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise IntegrityFailure("partial guard evidence contains a special file")
            if path != guard:
                (guard_directories if stat.S_ISDIR(mode) else guard_files).append(path)
    if result.exists() and not stat.S_ISREG(result.lstat().st_mode):
        raise IntegrityFailure("partial result is not a regular file")
    staging = Path(tempfile.mkdtemp(prefix=".partial.", dir=unit_dir))
    if guard.exists():
        guard_snapshot = staging / "guard_run"
        guard_snapshot.mkdir()
        for source_directory in guard_directories:
            (guard_snapshot / source_directory.relative_to(guard)).mkdir()
        for source_file in guard_files:
            _snapshot_regular_file(
                source_file,
                guard_snapshot / source_file.relative_to(guard),
            )
    if result.exists():
        _snapshot_regular_file(result, staging / "result.json")
    files: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*")):
        _assert_no_symlink_components(path)
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityFailure("partial evidence contains a non-regular file")
        files.append(
            {
                "relative_path": str(path.relative_to(staging)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    inventory = {
        "schema_version": 1,
        "artifact_type": "ha_v7_partial_unit_inventory",
        "unit_key": expectation.unit.key,
        "unit_ordinal": unit_ordinal,
        "files": files,
    }
    inventory_path = staging / "inventory.json"
    with inventory_path.open("xb") as handle:
        handle.write(_canonical_bytes(inventory) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_file(inventory_path)
    directories = [path for path in staging.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(staging)
    _publish_partial_archive(staging, final, unit_dir)
    bound = _artifact(final / "inventory.json", run_root=root)
    return PartialArchive(
        path=final,
        inventory=PathArtifactBinding(**asdict(bound)),
    )


def validate_and_fingerprint_train_policy_cache(
    cache_root: Path,
    *,
    run_root: Path,
    config_paths: Iterable[Path] = (),
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly fingerprint shared policy/config inputs without moving them."""

    root = _safe_path(cache_root, run_root=run_root)
    if not root.is_dir():
        raise IntegrityFailure("train-policy cache is missing or not a directory")
    paths: list[tuple[str, str, Path]] = []
    for path in sorted(root.rglob("*")):
        _assert_no_symlink_components(path)
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityFailure("train-policy cache contains a non-regular file")
        paths.append(("cache", f"cache/{path.relative_to(root)}", path))
    configs = tuple(sorted({_absolute(Path(path)) for path in config_paths}))
    for ordinal, config in enumerate(configs):
        _assert_no_symlink_components(config)
        if not config.exists():
            raise IntegrityFailure("train-policy config is missing")
        if not stat.S_ISREG(config.lstat().st_mode):
            raise IntegrityFailure("train-policy config is not a regular file")
        paths.append(("config", f"config/{ordinal:03d}-{config.name}", config))
    if len({name for _, name, _ in paths}) != len(paths):
        raise IntegrityFailure("train-policy fingerprint contains duplicate paths")
    files = [
        {
            "kind": kind,
            "source_path": str(path),
            "relative_path": name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for kind, name, path in paths
    ]
    digest = hashlib.sha256(_canonical_bytes({"files": files})).hexdigest()
    fingerprint = {
        "schema_version": 1,
        "artifact_type": "ha_v7_train_policy_cache_fingerprint",
        "cache_root": str(root),
        "files": files,
        "sha256": digest,
    }
    if expected is not None and dict(expected) != fingerprint:
        raise IntegrityFailure("train-policy cache fingerprint mismatch")
    return fingerprint


def policy_cache_fingerprint_evidence_path(
    *,
    run_root: Path,
    task_id: str,
    attempt: int,
) -> Path:
    if not isinstance(task_id, str) or not _TOKEN_RE.fullmatch(task_id):
        raise ValueError("task_id must be a nonempty token")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    root = _safe_path(run_root, run_root=run_root)
    return _absolute(
        root
        / "diagnostics"
        / task_id
        / "attempts"
        / f"attempt-{attempt}"
        / "policy_cache_fingerprint.json"
    )


def _validate_policy_fingerprint_mapping(value: Any) -> dict[str, Any]:
    fields = {"schema_version", "artifact_type", "cache_root", "files", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise IntegrityFailure("policy fingerprint schema mismatch")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise IntegrityFailure("policy fingerprint schema version is invalid")
    if value.get("artifact_type") != "ha_v7_train_policy_cache_fingerprint":
        raise IntegrityFailure("policy fingerprint artifact type is invalid")
    cache_root = value.get("cache_root")
    if not isinstance(cache_root, str) or not Path(cache_root).is_absolute():
        raise IntegrityFailure("policy fingerprint cache root is invalid")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise IntegrityFailure("policy fingerprint digest is invalid")
    files = value.get("files")
    if not isinstance(files, list):
        raise IntegrityFailure("policy fingerprint files are invalid")
    validated_files: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "source_path",
            "relative_path",
            "size",
            "sha256",
        }:
            raise IntegrityFailure("policy fingerprint file schema mismatch")
        kind = row.get("kind")
        source_path = row.get("source_path")
        relative_path = row.get("relative_path")
        size = row.get("size")
        sha = row.get("sha256")
        if kind not in {"cache", "config"}:
            raise IntegrityFailure("policy fingerprint file kind is invalid")
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise IntegrityFailure("policy fingerprint source path is invalid")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise IntegrityFailure("policy fingerprint relative path is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise IntegrityFailure("policy fingerprint file size is invalid")
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise IntegrityFailure("policy fingerprint file hash is invalid")
        validated_files.append(dict(row))
    if len({row["relative_path"] for row in validated_files}) != len(validated_files):
        raise IntegrityFailure("policy fingerprint paths are duplicated")
    calculated = hashlib.sha256(_canonical_bytes({"files": validated_files})).hexdigest()
    if calculated != digest:
        raise IntegrityFailure("policy fingerprint digest does not bind its files")
    return dict(value)


def _validate_policy_evidence_payload(
    payload: Any,
    *,
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "artifact_type",
        "manifest_sha256",
        "run_id",
        "task_id",
        "index",
        "attempt",
        "fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise IntegrityFailure("policy fingerprint evidence schema mismatch")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise IntegrityFailure("policy fingerprint evidence version is invalid")
    if payload.get("artifact_type") != "ha_v7_policy_cache_fingerprint_evidence":
        raise IntegrityFailure("policy fingerprint evidence type is invalid")
    if not isinstance(payload.get("manifest_sha256"), str) or not _SHA256_RE.fullmatch(
        payload["manifest_sha256"]
    ):
        raise IntegrityFailure("policy fingerprint manifest hash is invalid")
    for name in ("run_id", "task_id"):
        if not isinstance(payload.get(name), str) or not _TOKEN_RE.fullmatch(payload[name]):
            raise IntegrityFailure(f"policy fingerprint {name} is invalid")
    for name, minimum in (("index", 0), ("attempt", 1)):
        scalar = payload.get(name)
        if isinstance(scalar, bool) or not isinstance(scalar, int) or scalar < minimum:
            raise IntegrityFailure(f"policy fingerprint {name} is invalid")
    identity = {
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task_id,
        "index": index,
        "attempt": attempt,
    }
    if any(payload.get(name) != expected for name, expected in identity.items()):
        raise IntegrityFailure("policy fingerprint evidence identity mismatch")
    return _validate_policy_fingerprint_mapping(payload.get("fingerprint"))


def prepare_policy_cache_fingerprint_evidence(
    *,
    run_root: Path,
    cache_root: Path,
    config_paths: Iterable[Path],
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
    require_existing: bool,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ArtifactBinding]:
    """Create once or strictly revalidate deterministic policy input evidence."""

    if not isinstance(require_existing, bool):
        raise ValueError("require_existing must be a boolean")
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest sha256 must be lowercase 64-hex")
    if not isinstance(run_id, str) or not _TOKEN_RE.fullmatch(run_id):
        raise ValueError("run_id must be a nonempty token")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    evidence_path = policy_cache_fingerprint_evidence_path(
        run_root=run_root,
        task_id=task_id,
        attempt=attempt,
    )
    current = validate_and_fingerprint_train_policy_cache(
        cache_root,
        run_root=run_root,
        config_paths=config_paths,
    )
    if expected is not None:
        validated_expected = _validate_policy_fingerprint_mapping(dict(expected))
        if validated_expected != current:
            raise IntegrityFailure("policy fingerprint differs from expected resume evidence")
    if evidence_path.exists():
        evidence = _safe_path(evidence_path, run_root=run_root, must_exist=True)
        binding = _artifact(evidence, run_root=run_root)
        payload = _read_json_object(evidence)
        persisted = _validate_policy_evidence_payload(
            payload,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            index=index,
            attempt=attempt,
        )
        if persisted != current:
            raise IntegrityFailure("persisted policy fingerprint drifted from current inputs")
        return persisted, binding
    if require_existing:
        raise IntegrityFailure("required policy fingerprint evidence is missing")
    _ensure_directory(evidence_path.parent, run_root=run_root)
    payload = {
        "schema_version": 1,
        "artifact_type": "ha_v7_policy_cache_fingerprint_evidence",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task_id,
        "index": index,
        "attempt": attempt,
        "fingerprint": current,
    }
    _atomic_json(evidence_path, payload, run_root=run_root)
    binding = _artifact(evidence_path, run_root=run_root)
    persisted = _validate_policy_evidence_payload(
        _read_json_object(evidence_path),
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        index=index,
        attempt=attempt,
    )
    if persisted != current:
        raise IntegrityFailure("persisted policy fingerprint differs after publication")
    return persisted, binding


def validate_policy_cache_fingerprint_binding(
    binding: ArtifactBinding,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
) -> dict[str, Any]:
    """Strictly bind an evidence artifact and rehash all referenced inputs."""

    if not isinstance(binding, ArtifactBinding):
        raise IntegrityFailure("policy fingerprint artifact binding is invalid")
    expected_path = policy_cache_fingerprint_evidence_path(
        run_root=run_root,
        task_id=task_id,
        attempt=attempt,
    )
    if Path(binding.path) != expected_path or _artifact(expected_path, run_root=run_root) != binding:
        raise IntegrityFailure("policy fingerprint artifact path/hash mismatch")
    payload = _read_json_object(expected_path)
    fingerprint = _validate_policy_evidence_payload(
        payload,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        index=index,
        attempt=attempt,
    )
    config_paths = tuple(
        Path(row["source_path"])
        for row in fingerprint["files"]
        if row["kind"] == "config"
    )
    current = validate_and_fingerprint_train_policy_cache(
        Path(fingerprint["cache_root"]),
        run_root=run_root,
        config_paths=config_paths,
        expected=fingerprint,
    )
    return current


def _status_from_exception(error: BaseException) -> int | None:
    status = getattr(error, "http_status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _exception_graph(error: BaseException) -> list[BaseException]:
    queue = [error]
    seen: set[int] = set()
    values: list[BaseException] = []
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        values.append(current)
        for candidate in (
            getattr(current, "last_error", None),
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(candidate, BaseException) and id(candidate) not in seen:
                queue.append(candidate)
    return values


def _safe_type_name(error: BaseException) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", type(error).__name__)[:200]
    return token or "UnknownException"


def classify_child_failure(error: BaseException) -> FailureClassification:
    """Classify from typed exception structure, never from log/message text."""

    nodes = _exception_graph(error)
    from .predict_only_pipeline import PredictionDecisionError

    for node in nodes:
        guard_error = getattr(node, "guard_error", None)
        if (
            isinstance(node, PredictionDecisionError)
            and isinstance(guard_error, dict)
            and set(guard_error) == {"type", "message"}
            and guard_error.get("type") == "TransportExhausted"
            and guard_error.get("message") == "transport_attempts_exhausted"
        ):
            return FailureClassification(
                "recoverable_transport",
                "TransportExhausted",
                "transport_attempts_exhausted",
                None,
            )
    for node in nodes:
        if (
            isinstance(node, CompletedResponseSemanticInvalid)
            or type(node).__name__ in _TERMINAL_EXCEPTION_TYPES
        ):
            return FailureClassification(
                "terminal_model_invalid",
                _safe_type_name(node),
                "completed_model_output_invalid",
                _status_from_exception(node),
            )
    for node in nodes:
        status = _status_from_exception(node)
        if isinstance(node, TransportExhausted):
            return FailureClassification(
                "recoverable_transport",
                _safe_type_name(node),
                "transport_attempts_exhausted",
                status,
            )
        if isinstance(node, requests.exceptions.HTTPError):
            if status in _RETRYABLE_HTTP or (status is not None and status >= 500):
                return FailureClassification(
                    "recoverable_transport",
                    _safe_type_name(node),
                    "retryable_http_transport_failure",
                    status,
                )
        if isinstance(node, (requests.Timeout, requests.ConnectionError, requests.exceptions.SSLError)):
            return FailureClassification(
                "recoverable_transport",
                _safe_type_name(node),
                "retryable_transport_failure",
                status,
            )
    status = next((_status_from_exception(node) for node in nodes if _status_from_exception(node) is not None), None)
    return FailureClassification(
        "integrity_failure",
        _safe_type_name(error),
        "integrity_failure",
        status,
    )


def _active_secrets(extra: Iterable[str] = ()) -> tuple[str, ...]:
    values = {
        str(value)
        for name, value in os.environ.items()
        if value
        and any(token in name.upper() for token in ("API_KEY", "ACCESS_TOKEN", "PASSWORD", "SECRET"))
    }
    values.update(str(value) for value in extra if value)
    return tuple(sorted(values, key=len, reverse=True))


def _validate_failure_mapping(value: Any) -> FailureClassification:
    if not isinstance(value, dict) or set(value) != {"kind", "exception_type", "message", "http_status"}:
        raise IntegrityFailure("failure sidecar classification schema mismatch")
    try:
        return FailureClassification(**value)
    except (TypeError, ValueError) as exc:
        raise IntegrityFailure("failure sidecar classification is invalid") from exc


def _read_transport_ledger(path: Path, *, run_root: Path) -> list[dict[str, Any]]:
    ledger = _safe_path(path, run_root=run_root, must_exist=True)
    if not stat.S_ISREG(ledger.lstat().st_mode):
        raise IntegrityFailure("transport ledger is not a regular file")
    rows: list[dict[str, Any]] = []
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line:
                raise IntegrityFailure("transport ledger contains a blank line")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != _TRANSPORT_LEDGER_FIELDS:
                raise IntegrityFailure("transport ledger row schema mismatch")
            if (
                type(row.get("schema_version")) is not int
                or row.get("schema_version") != 2
                or row.get("artifact_type") != "ha_v7_transport_attempt"
            ):
                raise IntegrityFailure("transport ledger version mismatch")
            group = row.get("semantic_group")
            attempt = row.get("transport_attempt")
            body_sha = row.get("body_sha256")
            body_size = row.get("body_size")
            outcome = row.get("outcome")
            status_code = row.get("http_status")
            exception_type = row.get("exception_type")
            timestamp = row.get("timestamp_utc")
            if not isinstance(group, str) or not re.fullmatch(r"group-[0-9]{6}", group):
                raise IntegrityFailure("transport ledger semantic group is invalid")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 4:
                raise IntegrityFailure("transport ledger attempt is invalid")
            if not isinstance(body_sha, str) or not _SHA256_RE.fullmatch(body_sha):
                raise IntegrityFailure("transport ledger body hash is invalid")
            if isinstance(body_size, bool) or not isinstance(body_size, int) or body_size <= 0:
                raise IntegrityFailure("transport ledger body size is invalid")
            if outcome not in _LEDGER_OUTCOMES:
                raise IntegrityFailure("transport ledger outcome is invalid")
            if status_code is not None and (
                isinstance(status_code, bool)
                or not isinstance(status_code, int)
                or not 100 <= status_code <= 599
            ):
                raise IntegrityFailure("transport ledger HTTP status is invalid")
            if exception_type is not None and (
                not isinstance(exception_type, str)
                or not _TOKEN_RE.fullmatch(exception_type)
            ):
                raise IntegrityFailure("transport ledger exception type is invalid")
            if not isinstance(timestamp, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[^\s]+Z",
                timestamp,
            ):
                raise IntegrityFailure("transport ledger timestamp is invalid")
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise IntegrityFailure("transport ledger timestamp is invalid") from exc
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise IntegrityFailure("transport ledger timestamp is not UTC")
            retryable = isinstance(status_code, int) and (
                status_code in _RETRYABLE_HTTP or 500 <= status_code <= 599
            )
            if outcome == "transport_error" and not (
                status_code is None and exception_type in _TRANSPORT_EXCEPTION_TYPES
            ):
                raise IntegrityFailure("transport ledger transport-error fields conflict")
            if outcome == "retryable_http" and not (
                retryable and exception_type in {"HTTPStatus", "HTTPError"}
            ):
                raise IntegrityFailure("transport ledger retryable-HTTP fields conflict")
            if outcome == "completed_response" and not (
                isinstance(status_code, int)
                and 200 <= status_code <= 299
                and exception_type is None
            ):
                raise IntegrityFailure("transport ledger completed-response fields conflict")
            if outcome == "semantic_invalid" and not (
                isinstance(status_code, int)
                and 200 <= status_code <= 299
                and exception_type == "JSONDecodeError"
            ):
                raise IntegrityFailure("transport ledger semantic-invalid fields conflict")
            if outcome == "nonretryable_http" and not (
                isinstance(status_code, int)
                and not retryable
                and not 200 <= status_code <= 299
                and exception_type in {"HTTPStatus", "HTTPError"}
            ):
                raise IntegrityFailure("transport ledger nonretryable-HTTP fields conflict")
            if rows:
                prior = rows[-1]
                expected_attempt = (
                    int(prior["transport_attempt"]) + 1
                    if prior["semantic_group"] == group
                    else 1
                )
                if attempt != expected_attempt:
                    raise IntegrityFailure("transport ledger attempts are not contiguous")
                if prior["semantic_group"] == group and (
                    prior["body_sha256"] != body_sha
                    or prior["body_size"] != body_size
                ):
                    raise IntegrityFailure("transport ledger request body identity changed")
                if prior["semantic_group"] != group and any(
                    existing["semantic_group"] == group for existing in rows
                ):
                    raise IntegrityFailure("transport ledger semantic group reappears")
            elif attempt != 1:
                raise IntegrityFailure("transport ledger first attempt is not one")
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityFailure("transport ledger is malformed") from exc
    return rows


def _bind_recoverable_transport_status(
    failure: FailureClassification,
    *,
    ledger_path: Path,
    run_root: Path,
) -> FailureClassification:
    """Recover the status hidden by the guard's sanitized error boundary."""

    if (
        failure.kind != "recoverable_transport"
        or failure.exception_type != "TransportExhausted"
        or failure.http_status is not None
    ):
        return failure
    rows = _read_transport_ledger(ledger_path, run_root=run_root)
    if not rows:
        return failure
    last_group = rows[-1]["semantic_group"]
    status = next(
        (
            row["http_status"]
            for row in reversed(rows)
            if row["semantic_group"] == last_group
            and row["http_status"] is not None
        ),
        None,
    )
    return FailureClassification(
        kind=failure.kind,
        exception_type=failure.exception_type,
        message=failure.message,
        http_status=status,
    )


def write_failure_sidecar(
    path: Path,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
    unit_key: str,
    child_kind: Literal["paused", "terminal-model-invalid"],
    failure: FailureClassification,
    ledger_path: Path,
    policy_cache_fingerprint: ArtifactBinding,
    secret_values: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest sha256 must be lowercase 64-hex")
    if not isinstance(run_id, str) or not _TOKEN_RE.fullmatch(run_id):
        raise ValueError("run_id must be a nonempty token")
    if not isinstance(task_id, str) or not _TOKEN_RE.fullmatch(task_id):
        raise ValueError("task_id must be a nonempty token")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    if not isinstance(unit_key, str) or not re.fullmatch(
        r"(?:direct_multi_horizon|iterative_wm_rollout)\|"
        r"(?:train_policy|full_harnessaudit_rules)\|h0[1-6]",
        unit_key,
    ):
        raise ValueError("unit_key is invalid")
    if child_kind == "paused" and failure.kind != "recoverable_transport":
        raise ValueError("paused sidecar requires recoverable transport failure")
    if child_kind == "terminal-model-invalid" and failure.kind != "terminal_model_invalid":
        raise ValueError("terminal sidecar requires model-invalid failure")
    validate_policy_cache_fingerprint_binding(
        policy_cache_fingerprint,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        index=index,
        attempt=attempt,
    )
    ledger = _artifact(ledger_path, run_root=run_root)
    payload = {
        "schema_version": 1,
        "artifact_type": "ha_v7_child_failure",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task_id,
        "index": index,
        "attempt": attempt,
        "unit_key": unit_key,
        "child_kind": child_kind,
        "failure": asdict(failure),
        "ledger": asdict(ledger),
        "policy_cache_fingerprint": asdict(policy_cache_fingerprint),
    }
    encoded = _canonical_bytes(payload)
    if any(secret.encode("utf-8") in encoded for secret in _active_secrets(secret_values)):
        raise IntegrityFailure("failure sidecar contains a credential value")
    _atomic_json(path, payload, run_root=run_root)
    return payload


def validate_failure_sidecar(
    path: Path,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
    unit_key: str,
    ledger_path: Path,
) -> dict[str, Any]:
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
        raise IntegrityFailure("expected manifest hash is invalid")
    for name, value in (("run_id", run_id), ("task_id", task_id)):
        if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
            raise IntegrityFailure(f"expected {name} is invalid")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise IntegrityFailure("expected index is invalid")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise IntegrityFailure("expected attempt is invalid")
    if not isinstance(unit_key, str) or not re.fullmatch(
        r"(?:direct_multi_horizon|iterative_wm_rollout)\|"
        r"(?:train_policy|full_harnessaudit_rules)\|h0[1-6]",
        unit_key,
    ):
        raise IntegrityFailure("expected unit key is invalid")
    sidecar = _safe_path(path, run_root=run_root, must_exist=True)
    if not stat.S_ISREG(sidecar.lstat().st_mode):
        raise IntegrityFailure("failure sidecar is not a regular file")
    payload = _read_json_object(sidecar)
    expected_fields = {
        "schema_version",
        "artifact_type",
        "manifest_sha256",
        "run_id",
        "task_id",
        "index",
        "attempt",
        "unit_key",
        "child_kind",
        "failure",
        "ledger",
        "policy_cache_fingerprint",
    }
    if set(payload) != expected_fields:
        raise IntegrityFailure("failure sidecar schema mismatch")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise IntegrityFailure("failure sidecar schema version is invalid")
    if payload.get("artifact_type") != "ha_v7_child_failure":
        raise IntegrityFailure("failure sidecar artifact type is invalid")
    manifest_value = payload.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not _SHA256_RE.fullmatch(manifest_value):
        raise IntegrityFailure("failure sidecar manifest hash is invalid")
    for name in ("run_id", "task_id"):
        if not isinstance(payload.get(name), str) or not _TOKEN_RE.fullmatch(payload[name]):
            raise IntegrityFailure(f"failure sidecar {name} is invalid")
    for name, minimum in (("index", 0), ("attempt", 1)):
        scalar = payload.get(name)
        if isinstance(scalar, bool) or not isinstance(scalar, int) or scalar < minimum:
            raise IntegrityFailure(f"failure sidecar {name} type/value is invalid")
    unit_value = payload.get("unit_key")
    if not isinstance(unit_value, str) or not re.fullmatch(
        r"(?:direct_multi_horizon|iterative_wm_rollout)\|"
        r"(?:train_policy|full_harnessaudit_rules)\|h0[1-6]",
        unit_value,
    ):
        raise IntegrityFailure("failure sidecar unit key is invalid")
    identity = {
        "schema_version": 1,
        "artifact_type": "ha_v7_child_failure",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task_id,
        "index": index,
        "attempt": attempt,
        "unit_key": unit_key,
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise IntegrityFailure("failure sidecar identity mismatch")
    failure = _validate_failure_mapping(payload.get("failure"))
    child_kind = payload.get("child_kind")
    if child_kind not in {"paused", "terminal-model-invalid"}:
        raise IntegrityFailure("failure sidecar child kind is invalid")
    if child_kind == "paused" and failure.kind != "recoverable_transport":
        raise IntegrityFailure("paused sidecar failure kind mismatch")
    if child_kind == "terminal-model-invalid" and failure.kind != "terminal_model_invalid":
        raise IntegrityFailure("terminal sidecar failure kind mismatch")
    ledger_binding_raw = payload.get("ledger")
    if not isinstance(ledger_binding_raw, dict) or set(ledger_binding_raw) != {"path", "size", "sha256"}:
        raise IntegrityFailure("failure sidecar ledger binding schema mismatch")
    try:
        ledger_binding = ArtifactBinding(**ledger_binding_raw)
    except (TypeError, ValueError) as exc:
        raise IntegrityFailure("failure sidecar ledger binding is invalid") from exc
    current = _artifact(ledger_path, run_root=run_root)
    if Path(ledger_binding.path) != _absolute(ledger_path) or ledger_binding != current:
        raise IntegrityFailure("failure sidecar ledger hash/path mismatch")
    fingerprint_raw = payload.get("policy_cache_fingerprint")
    if not isinstance(fingerprint_raw, dict) or set(fingerprint_raw) != {
        "path",
        "size",
        "sha256",
    }:
        raise IntegrityFailure("failure sidecar policy fingerprint binding is invalid")
    try:
        fingerprint_binding = ArtifactBinding(**fingerprint_raw)
    except (TypeError, ValueError) as exc:
        raise IntegrityFailure("failure sidecar policy fingerprint binding is invalid") from exc
    validate_policy_cache_fingerprint_binding(
        fingerprint_binding,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        index=index,
        attempt=attempt,
    )
    rows = _read_transport_ledger(ledger_path, run_root=run_root)
    if child_kind == "paused":
        if not rows:
            raise IntegrityFailure("paused ledger is empty")
        last_group = rows[-1].get("semantic_group")
        group_start = len(rows) - 1
        while group_start > 0 and rows[group_start - 1].get("semantic_group") == last_group:
            group_start -= 1
        failed_group = rows[group_start:]
        if len(failed_group) != 4 or [row.get("transport_attempt") for row in failed_group] != [1, 2, 3, 4]:
            raise IntegrityFailure("paused ledger does not prove four transport attempts")
        if len({row.get("semantic_group") for row in failed_group}) != 1 or len({row.get("body_sha256") for row in failed_group}) != 1:
            raise IntegrityFailure("paused ledger request group/hash mismatch")
        if any(
            row["outcome"] not in {"transport_error", "retryable_http"}
            for row in failed_group
        ):
            raise IntegrityFailure("paused ledger contains a nonretryable outcome")
        failed_status = next(
            (
                row["http_status"]
                for row in reversed(failed_group)
                if row["http_status"] is not None
            ),
            None,
        )
        if (
            failure.exception_type != "TransportExhausted"
            or failure.http_status != failed_status
        ):
            raise IntegrityFailure("paused sidecar status/exception conflicts with ledger")
    else:
        if not rows or rows[-1]["outcome"] not in {
            "semantic_invalid",
            "completed_response",
        }:
            raise IntegrityFailure("terminal sidecar lacks completed-response evidence")
        if failure.exception_type not in _TERMINAL_EXCEPTION_TYPES:
            raise IntegrityFailure("terminal sidecar exception is not explicitly semantic")
        if failure.http_status not in {None, rows[-1]["http_status"]}:
            raise IntegrityFailure("terminal sidecar status conflicts with ledger")
    return payload


def enumerate_replay_units(scientific_manifest: Mapping[str, Any]) -> tuple[ReplayUnit, ...]:
    matrix = scientific_manifest.get("matrix")
    if not isinstance(matrix, Mapping):
        raise IntegrityFailure("scientific manifest matrix is missing")
    try:
        return build_replay_unit_plan(
            rollout_modes=tuple(matrix["rollout_modes"]),
            policy_modes=tuple(matrix["policy_modes"]),
            horizons=tuple(matrix["horizons"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityFailure("scientific manifest replay-unit plan is invalid") from exc


def capture_v6_protection_inventory(
    manifest_paths: Iterable[Path],
) -> dict[str, Any]:
    """Hash every frozen v6 implementation and existing shard control file.

    The returned object is in-memory evidence only; callers decide where a v7
    report may persist it.  Nothing is ever written beneath a v6 tree.
    """

    manifests = tuple(_absolute(path) for path in manifest_paths)
    if len(manifests) != 4 or len(set(manifests)) != 4:
        raise IntegrityFailure("v6 protection inventory requires four unique manifests")
    protected: set[Path] = set()
    for manifest_path in manifests:
        _assert_no_symlink_components(manifest_path)
        if not manifest_path.is_file() or not stat.S_ISREG(manifest_path.lstat().st_mode):
            raise IntegrityFailure("v6 manifest is not a regular file")
        manifest = _read_json_object(manifest_path)
        implementation_files = manifest.get("implementation_files")
        if not isinstance(implementation_files, list) or not implementation_files:
            raise IntegrityFailure("v6 manifest implementation inventory is missing")
        for item in implementation_files:
            if not isinstance(item, dict):
                raise IntegrityFailure("v6 implementation record is not an object")
            path_value = item.get("path")
            declared_sha = item.get("sha256")
            if not isinstance(path_value, str) or not _SHA256_RE.fullmatch(str(declared_sha or "")):
                raise IntegrityFailure("v6 implementation record is malformed")
            implementation = _absolute(Path(path_value))
            _assert_no_symlink_components(implementation)
            if not implementation.is_file() or not stat.S_ISREG(implementation.lstat().st_mode):
                raise IntegrityFailure("v6 implementation file is missing or unsafe")
            if _sha256(implementation) != declared_sha:
                raise IntegrityFailure("v6 implementation hash differs from its manifest")
            protected.add(implementation)
        protected.add(manifest_path)
        for filename in ("batch.status.jsonl", "progress.json"):
            candidate = manifest_path.parent / filename
            if candidate.exists():
                _assert_no_symlink_components(candidate)
                if not stat.S_ISREG(candidate.lstat().st_mode):
                    raise IntegrityFailure("v6 shard control artifact is not a regular file")
                protected.add(candidate)
    files = [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(protected)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "ha_v7_v6_protection_inventory",
        "file_count": len(files),
        "files": files,
        "sha256": hashlib.sha256(_canonical_bytes({"files": files})).hexdigest(),
    }


def _expectation_from_replay_kwargs(kwargs: Mapping[str, Any], *, run_root: Path) -> UnitExpectation:
    """Recompute the unchanged pipeline's deterministic cache/run identity."""

    from . import predict_only_pipeline as pipeline

    report = kwargs["report"]
    task_id = str(report.get("task_id") or "")
    points = list(kwargs["points"])
    horizon = int(kwargs["prediction_steps"])
    rollout = str(kwargs["rollout_mode"])
    policy = str(kwargs["policy_mode"])
    identity = pipeline._source_report_actor_identity(
        report,
        legacy_fallback_seed=int(kwargs.get("experiment_seed", 0)),
        expected_actor_protocol_profile=kwargs.get("actor_protocol_profile"),
        expected_actor_replicate_id=kwargs.get("actor_replicate_id"),
        expected_actor_trace_seed=kwargs.get("actor_trace_seed"),
    )
    point_signature = pipeline._point_signature(points)
    cache_signature = pipeline._ha_replay_cache_signature(
        report_path=Path(kwargs["report_path"]),
        trace_file=Path(kwargs["trace_file"]),
        base_config=Path(kwargs["base_config"]),
        model=str(kwargs["model"]),
        model_api=kwargs["model_api"],
        policy_mode=policy,
        rollout_mode=rollout,
        prediction_steps=horizon,
        point_signature=point_signature,
        candidate_count=int(kwargs["candidate_count"]),
        max_trace_events=int(kwargs["max_trace_events"]),
        direct_multi_horizons=kwargs.get("direct_multi_horizons"),
        formal_annotation_horizon=int(kwargs["formal_annotation_horizon"]),
        policy_inputs_signature=str(kwargs.get("policy_inputs_signature") or ""),
        experiment_seed=int(kwargs.get("experiment_seed", 0)),
        actor_protocol_profile=identity.actor_protocol_profile,
        actor_replicate_id=identity.actor_replicate_id,
        actor_trace_seed=identity.actor_trace_seed,
        wm_seed=int(kwargs.get("wm_seed", kwargs.get("experiment_seed", 0))),
    )
    run_id = (
        f"po-{ROLLOUT_ALIAS[rollout]}-h{horizon:02d}-{task_id}-"
        f"{POLICY_ALIAS[policy]}-{kwargs['point_config'].scheme}-"
        f"{point_signature}-{cache_signature}"
    )
    result_dir = Path(kwargs["output_root"]) / ROLLOUT_ALIAS[rollout] / POLICY_ALIAS[policy] / "prediction_only" / task_id / f"h{horizon:02d}"
    guard_dir = Path(kwargs["trace_root"]) / "_g" / ROLLOUT_ALIAS[rollout] / POLICY_ALIAS[policy] / "prediction_only" / task_id / f"h{horizon:02d}" / run_id
    return UnitExpectation(
        unit=ReplayUnit(rollout, policy, horizon),
        run_root=_absolute(run_root),
        task_id=task_id,
        run_id=run_id,
        result_dir=_absolute(result_dir),
        result_path=_absolute(result_dir / f"{run_id}.json"),
        guard_run_dir=_absolute(guard_dir),
        decision_path=_absolute(guard_dir / "01_guard_decisions.jsonl"),
        selected_point_ids=tuple(int(point.point_num) for point in points),
        selected_sequence_nums=tuple(int(point.step.action_sequence_num) for point in points),
        point_signature=point_signature,
        cache_signature=cache_signature,
        formal_annotation_horizon=int(kwargs["formal_annotation_horizon"]),
        source_report_path=str(Path(kwargs["report_path"])),
    )


def _binding_payload(
    binding: UnitBinding,
    policy_cache_fingerprint: ArtifactBinding | None = None,
) -> dict[str, Any]:
    value = asdict(binding)
    value["selected_point_ids"] = list(binding.selected_point_ids)
    value["selected_sequence_nums"] = list(binding.selected_sequence_nums)
    return {
        "schema_version": 1,
        "artifact_type": "ha_v7_unit_binding",
        "binding": value,
        "policy_cache_fingerprint": (
            asdict(policy_cache_fingerprint)
            if policy_cache_fingerprint is not None
            else None
        ),
    }


@contextmanager
def _retry_environment(overrides: Mapping[str, str]):
    original = os.environ.copy()
    try:
        os.environ.update({str(key): str(value) for key, value in overrides.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _require_authoritative_open_unit(
    history_path: Path | None,
    *,
    run_root: Path,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
    unit_key: str,
) -> None:
    """Require a verified current-attempt UNIT_START before recovering a pass."""

    if history_path is None:
        raise IntegrityFailure("complete unit lacks authoritative UNIT_START history")
    try:
        events = read_task_history(history_path, run_root=run_root)
        state = reduce_task_history(
            events,
            unit_plan=unit_plan,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            source_index=index,
            run_root=run_root,
        )
    except (OSError, ValueError) as exc:
        raise IntegrityFailure("complete unit history is not authoritative") from exc
    if (
        state.status != "active-without-outcome"
        or state.attempt_count != attempt
        or state.next_unit is None
        or state.next_unit.key != unit_key
        or not events
        or events[-1].get("event") != "UNIT_START"
        or events[-1].get("unit_key") != unit_key
        or events[-1].get("attempt") != attempt
    ):
        raise IntegrityFailure("complete unit lacks matching authoritative UNIT_START")


def _history_requires_existing_policy_evidence(
    history_path: Path | None,
    *,
    run_root: Path,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
    task_id: str,
    index: int,
    attempt: int,
) -> bool:
    if history_path is None:
        return attempt > 1
    try:
        events = read_task_history(history_path, run_root=run_root)
        state = reduce_task_history(
            events,
            unit_plan=unit_plan,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            source_index=index,
            run_root=run_root,
        )
    except (OSError, ValueError) as exc:
        raise IntegrityFailure("child history is not authoritative") from exc
    if (
        state.status != "active-without-outcome"
        or state.attempt_count != attempt
        or not events
    ):
        raise IntegrityFailure("child requires one active matching attempt")
    return events[-1].get("event") != "TRACE_ATTEMPT_START"


def reconcile_dangling_unit(
    *,
    expectation: UnitExpectation,
    history_path: Path,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
    index: int,
    attempt: int,
    cache_root: Path,
    config_paths: Iterable[Path],
) -> DanglingUnitReconciliation:
    """Reconcile crash evidence before Task 1 closes a dangling UNIT_START."""

    root = _safe_path(expectation.run_root, run_root=expectation.run_root)
    if not unit_plan or expectation.unit not in unit_plan:
        raise IntegrityFailure("dangling unit is outside the supplied replay plan")
    if expectation.task_id == "" or not _TOKEN_RE.fullmatch(expectation.task_id):
        raise IntegrityFailure("dangling unit task identity is invalid")
    _require_authoritative_open_unit(
        history_path,
        run_root=root,
        unit_plan=unit_plan,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=expectation.task_id,
        index=index,
        attempt=attempt,
        unit_key=expectation.unit.key,
    )
    fingerprint, fingerprint_binding = prepare_policy_cache_fingerprint_evidence(
        run_root=root,
        cache_root=cache_root,
        config_paths=config_paths,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=expectation.task_id,
        index=index,
        attempt=attempt,
        require_existing=True,
    )
    if not fingerprint:
        raise IntegrityFailure("dangling unit policy fingerprint is empty")
    discovery = inspect_unit(expectation)
    if discovery.state is UnitState.INVALID:
        raise IntegrityFailure(discovery.reason or "dangling unit evidence is invalid")
    if discovery.state in {UnitState.ABSENT, UnitState.PARTIAL}:
        return DanglingUnitReconciliation(
            kind="needs-pause",
            binding=None,
            policy_cache_fingerprint=fingerprint_binding,
        )
    if discovery.binding is None:
        raise IntegrityFailure("complete dangling unit lacks a strict binding")
    ordinal = unit_plan.index(expectation.unit)
    binding_path = (
        root
        / "diagnostics"
        / expectation.task_id
        / "unit_bindings"
        / f"unit-{ordinal:02d}.json"
    )
    binding_payload = _binding_payload(discovery.binding, fingerprint_binding)
    if binding_path.exists():
        existing = _read_json_object(
            _safe_path(binding_path, run_root=root, must_exist=True)
        )
        if existing != binding_payload:
            raise IntegrityFailure("dangling unit binding evidence conflicts")
    else:
        _atomic_json(binding_path, binding_payload, run_root=root)
    binding_artifact = _artifact(binding_path, run_root=root)
    append_task_event(
        history_path,
        {
            "schema_version": 1,
            "event": "UNIT_PASS",
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "task_id": expectation.task_id,
            "source_index": index,
            "attempt": attempt,
            "timestamp_utc": _timestamp_utc(),
            "unit_key": expectation.unit.key,
            "artifact": asdict(binding_artifact),
        },
        run_root=root,
    )
    return DanglingUnitReconciliation(
        kind="reconciled-complete",
        binding=discovery.binding,
        policy_cache_fingerprint=fingerprint_binding,
    )


def run_v7_child(
    *,
    scientific_manifest: Mapping[str, Any],
    manifest_sha256: str,
    run_id: str,
    run_root: Path,
    index: int,
    attempt: int,
    pipeline_runner: Callable[..., Mapping[str, Path]] | None = None,
    history_path: Path | None = None,
    verified_bindings: Iterable[UnitBinding] = (),
    expected_policy_fingerprint: Mapping[str, Any] | None = None,
    transport_observer_factory: Callable[..., Any] = V7TransportObserver,
) -> ChildOutcome:
    """Run one unchanged scientific pipeline under v7 unit/transport guards.

    The scheduler is responsible for ``TRACE_ATTEMPT_START`` and the trace-level
    terminal event.  This child appends only unit start/pass/pause events.
    """

    from . import predict_only_pipeline as pipeline

    root = _safe_path(run_root, run_root=run_root)
    sources = scientific_manifest.get("source_reports")
    if not isinstance(sources, list) or index < 0 or index >= len(sources):
        raise IntegrityFailure("source index is outside scientific manifest")
    source = sources[index]
    task_id = str(source.get("task_id") or "")
    if not task_id:
        raise IntegrityFailure("source task ID is missing")
    unit_plan = enumerate_replay_units(scientific_manifest)
    if len(unit_plan) != 14:
        raise IntegrityFailure("v7 HarnessAudit child requires exactly 14 replay units")
    matrix = scientific_manifest["matrix"]
    model_record = scientific_manifest.get("model")
    if not isinstance(model_record, Mapping):
        raise IntegrityFailure("scientific manifest model record is missing")
    model_id = str(model_record.get("id") or "")
    endpoint = str(model_record.get("endpoint") or "").rstrip("/")
    if not model_id or not endpoint:
        raise IntegrityFailure("scientific manifest model identity is incomplete")

    result_root = _ensure_directory(root / "results" / task_id, run_root=root)
    trace_root = _ensure_directory(root / "traces" / task_id, run_root=root)
    diagnostic = _ensure_directory(root / "diagnostics" / task_id, run_root=root)
    attempt_dir = _ensure_directory(diagnostic / "attempts" / f"attempt-{attempt}", run_root=root)
    analysis_dir = _ensure_directory(diagnostic / "analysis", run_root=root)
    figure_dir = _ensure_directory(diagnostic / "figures", run_root=root)
    ledger_path = attempt_dir / "transport.jsonl"
    cache_root = _ensure_directory(diagnostic / "policy_cache", run_root=root)
    api_key = str(os.environ.get("WM_API_KEY") or "")
    environment = {
        "SAFEPRED_LLM_MAX_ATTEMPTS": "4",
        "SAFEPRED_LLM_RETRY_DELAY_SECONDS": "15",
        "SAFEPRED_LLM_RETRY_BACKOFF": "2",
        "SAFEPRED_REQUEST_LEDGER_PATH": str(attempt_dir / "semantic_requests.jsonl"),
        "SAFEPRED_TRAIN_POLICY_CACHE_ROOT": str(cache_root),
        "SAFEPRED_POLICY_EXTRACTION_DEBUG_DIR": str(diagnostic / "policy_extraction_debug"),
        "SAFEPRED_RUN_DIR": str(diagnostic),
        "SAFEPRED_WM_SEED": "0",
        "SAFEPRED_POLICY_EXTRACTION_SEED": "0",
        "SAFEPRED_RESOURCE_SEMANTICS_MODE": str(scientific_manifest.get("resource_semantics") or ""),
        "SAFEPRED_MODEL_NAME": model_id,
        "SAFEPRED_POLICY_EXTRACTION_MODEL": model_id,
        "WM_BASE_URL": endpoint,
        "CUSTOM_API_URL": endpoint,
        "OPENAI_API_URL": endpoint,
        "OPENAI_BASE_URL": endpoint,
    }
    if api_key:
        environment.update({"CUSTOM_API_KEY": api_key, "OPENAI_API_KEY": api_key})

    point_config = pipeline.PointSelectionConfig(
        scheme="explicit",
        count=2,
        explicit_sequences=[int(value) for value in source["selected_sequence_nums"]],
    )
    pipeline_kwargs = {
        "tasks": [],
        "source_reports": [Path(str(source["report_path"]))],
        "policy_modes": list(matrix["policy_modes"]),
        "rollout_modes": list(matrix["rollout_modes"]),
        "horizons": list(matrix["horizons"]),
        "point_config": point_config,
        "model": model_id,
        "judge_model": model_id,
        "max_turns": 30,
        "skip_clean_run": True,
        "skip_judge": True,
        "base_config": pipeline._default_base_config(),
        "paths": pipeline.ArtifactPaths(
            output_root=result_root,
            trace_root=trace_root,
            analysis_dir=analysis_dir,
            figure_dir=figure_dir,
        ),
        "candidate_count": int(matrix["candidate_count"]),
        "max_trace_events": int(matrix["max_trace_events"]),
        "skip_existing": True,
        "run_tier3_judge": False,
        "tier3_model": model_id,
        "model_api": pipeline.ModelAPIConfig(
            provider="custom",
            api_key=api_key or None,
            api_url=endpoint,
            model=model_id,
            model_revision=str(model_record.get("requested_revision_label") or ""),
            policy_extraction_model=model_id,
        ),
        "clean_agent_api": None,
        "fail_on_prediction_errors": True,
        "fail_fast_decision_errors": True,
        "workers": 1,
        "gt_estimand": "exact_step",
        "experiment_seed": int(model_record.get("seed") or 0),
        "actor_trace_seed": None,
        "wm_seed": int(model_record.get("seed") or 0),
        "formal": False,
    }
    verified = tuple(verified_bindings)
    known = {binding.unit_key: binding for binding in verified}
    if len(known) != len(verified):
        raise IntegrityFailure("verified unit bindings contain duplicates")
    planned_keys = {unit.key for unit in unit_plan}
    if not set(known).issubset(planned_keys):
        raise IntegrityFailure("verified unit binding is outside the frozen plan")
    expected_units = {unit.key: unit for unit in unit_plan}
    expected_sequences = tuple(int(value) for value in source["selected_sequence_nums"])
    # This deliberately precedes policy preparation and the pipeline entry.
    # Full semantic identity is checked again when the deterministic replay job
    # supplies its independently recomputed expectation.
    for binding in verified:
        unit = expected_units[binding.unit_key]
        if (
            binding.task_id != task_id
            or binding.rollout_mode != unit.rollout_mode
            or binding.policy_mode != unit.policy_mode
            or binding.horizon != unit.horizon
            or binding.selected_point_ids != (1, 2)
            or binding.selected_sequence_nums != expected_sequences
            or binding.decision_count != 2
        ):
            raise IntegrityFailure("verified unit binding identity differs from the frozen task plan")
        for bound in (binding.result, binding.decisions):
            current = _artifact(Path(bound.path), run_root=root)
            if current != bound:
                raise IntegrityFailure("verified unit binding artifact hash/size changed")
    requires_policy_fingerprint = attempt > 1 or bool(verified)
    if requires_policy_fingerprint and expected_policy_fingerprint is None:
        raise IntegrityFailure("resumed child requires an expected policy-cache fingerprint")
    policy_fingerprint_path = policy_cache_fingerprint_evidence_path(
        run_root=root,
        task_id=task_id,
        attempt=attempt,
    )
    evidence_existed = policy_fingerprint_path.exists()
    history_requires_evidence = _history_requires_existing_policy_evidence(
        history_path,
        run_root=root,
        unit_plan=unit_plan,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task_id,
        index=index,
        attempt=attempt,
    )
    require_existing_evidence = (
        history_requires_evidence or evidence_existed or bool(verified) or attempt > 1
    )
    policy_fingerprint, policy_fingerprint_binding = (
        prepare_policy_cache_fingerprint_evidence(
            run_root=root,
            cache_root=cache_root,
            config_paths=(Path(pipeline_kwargs["base_config"]),),
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            index=index,
            attempt=attempt,
            require_existing=require_existing_evidence,
            expected=expected_policy_fingerprint,
        )
    )
    if history_path is not None:
        current_history = read_task_history(history_path, run_root=root)
        if current_history and current_history[-1].get("event") == "UNIT_START":
            raise IntegrityFailure(
                "dangling UNIT_START requires reconcile_dangling_unit before child launch"
            )
    may_refresh_before_first_unit = (
        not evidence_existed
        and not require_existing_evidence
        and expected_policy_fingerprint is None
    )
    completed: dict[str, UnitBinding] = dict(known)
    observed: list[str] = []
    current_expectation: UnitExpectation | None = None
    current_unit_committed_during_error = False
    cache_checked = False
    original_replay = pipeline.run_predict_only_replay_for_report
    runner = pipeline_runner or pipeline.run_pipeline

    def append_unit_event(event: str, **fields: Any) -> None:
        if history_path is None:
            return
        append_task_event(
            history_path,
            {
                "schema_version": 1,
                "event": event,
                "manifest_sha256": manifest_sha256,
                "run_id": run_id,
                "task_id": task_id,
                "source_index": index,
                "attempt": attempt,
                "timestamp_utc": _timestamp_utc(),
                **fields,
            },
            run_root=root,
        )

    def commit_complete_unit(
        expectation: UnitExpectation,
        binding: UnitBinding,
        *,
        require_prior_start: bool,
    ) -> None:
        if require_prior_start:
            _require_authoritative_open_unit(
                history_path,
                run_root=root,
                unit_plan=unit_plan,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                task_id=task_id,
                index=index,
                attempt=attempt,
                unit_key=expectation.unit.key,
            )
        binding_path = diagnostic / "unit_bindings" / f"unit-{len(observed)-1:02d}.json"
        _atomic_json(
            binding_path,
            _binding_payload(binding, policy_fingerprint_binding),
            run_root=root,
        )
        append_unit_event(
            "UNIT_PASS",
            unit_key=expectation.unit.key,
            artifact=asdict(_artifact(binding_path, run_root=root)),
        )
        completed[expectation.unit.key] = binding

    def observed_replay(**kwargs: Any) -> Path:
        nonlocal current_expectation, cache_checked
        nonlocal current_unit_committed_during_error, policy_fingerprint_binding
        nonlocal policy_fingerprint, may_refresh_before_first_unit
        expectation = _expectation_from_replay_kwargs(kwargs, run_root=root)
        current_expectation = expectation
        if len(observed) >= len(unit_plan) or expectation.unit != unit_plan[len(observed)]:
            raise IntegrityFailure("pipeline replay order differs from frozen unit plan")
        observed.append(expectation.unit.key)
        if not cache_checked:
            checked_fingerprint = validate_and_fingerprint_train_policy_cache(
                cache_root,
                run_root=root,
                config_paths=(Path(pipeline_kwargs["base_config"]),),
            )
            if checked_fingerprint != policy_fingerprint:
                if not may_refresh_before_first_unit:
                    raise IntegrityFailure(
                        "policy fingerprint drifted before the first scored unit"
                    )
                refreshed_payload = {
                    "schema_version": 1,
                    "artifact_type": "ha_v7_policy_cache_fingerprint_evidence",
                    "manifest_sha256": manifest_sha256,
                    "run_id": run_id,
                    "task_id": task_id,
                    "index": index,
                    "attempt": attempt,
                    "fingerprint": checked_fingerprint,
                }
                _atomic_json(policy_fingerprint_path, refreshed_payload, run_root=root)
                policy_fingerprint = checked_fingerprint
                policy_fingerprint_binding = _artifact(
                    policy_fingerprint_path,
                    run_root=root,
                )
                validate_policy_cache_fingerprint_binding(
                    policy_fingerprint_binding,
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    task_id=task_id,
                    index=index,
                    attempt=attempt,
                )
            may_refresh_before_first_unit = False
            cache_checked = True
        if expectation.unit.key in known:
            validate_unit_binding(known[expectation.unit.key], expectation)
        discovery = inspect_unit(expectation)
        if discovery.state is UnitState.INVALID:
            raise IntegrityFailure(discovery.reason or "invalid existing unit")
        if discovery.state is UnitState.COMPLETE:
            assert discovery.binding is not None
            if expectation.unit.key not in known:
                commit_complete_unit(
                    expectation,
                    discovery.binding,
                    require_prior_start=True,
                )
            else:
                completed[expectation.unit.key] = discovery.binding
            return expectation.result_path
        if discovery.state is UnitState.PARTIAL:
            archive_partial_unit(
                expectation,
                attempt_dir=attempt_dir,
                unit_ordinal=len(observed) - 1,
            )
        append_unit_event("UNIT_START", unit_key=expectation.unit.key)
        try:
            result = original_replay(**kwargs)
        except BaseException:
            after_error = inspect_unit(expectation)
            if after_error.state is UnitState.COMPLETE and after_error.binding is not None:
                commit_complete_unit(
                    expectation,
                    after_error.binding,
                    require_prior_start=False,
                )
                current_unit_committed_during_error = True
            raise
        after = inspect_unit(expectation)
        if after.state is not UnitState.COMPLETE or after.binding is None:
            raise IntegrityFailure("pipeline returned without a complete strict unit")
        commit_complete_unit(
            expectation,
            after.binding,
            require_prior_start=False,
        )
        return result

    pipeline.run_predict_only_replay_for_report = observed_replay
    try:
        with _retry_environment(environment):
            with transport_observer_factory(ledger_path=ledger_path, run_root=root):
                artifacts = runner(**pipeline_kwargs)
        if tuple(observed) != tuple(unit.key for unit in unit_plan):
            raise IntegrityFailure("pipeline did not visit the complete 14-unit plan")
        if set(completed) != {unit.key for unit in unit_plan}:
            raise IntegrityFailure("pipeline did not produce the complete 14-unit bindings")
        child_payload = {
            "schema_version": 1,
            "artifact_type": "ha_v7_child_result",
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "task_id": task_id,
            "index": index,
            "attempt": attempt,
            "completed_unit_keys": [unit.key for unit in unit_plan],
            "policy_cache_fingerprint": asdict(policy_fingerprint_binding),
            "artifacts": {
                str(name): asdict(_artifact(Path(path), run_root=root))
                for name, path in sorted(dict(artifacts).items())
            },
        }
        child_path = attempt_dir / "child_result.json"
        _atomic_json(child_path, child_payload, run_root=root)
        return ChildOutcome(
            task_id=task_id,
            index=index,
            attempt=attempt,
            kind="complete",
            completed_unit_keys=tuple(unit.key for unit in unit_plan),
            child_result=_artifact(child_path, run_root=root),
            failure=None,
            policy_cache_fingerprint=policy_fingerprint_binding,
        )
    except BaseException as error:
        classification = classify_child_failure(error)
        if classification.kind == "recoverable_transport":
            classification = _bind_recoverable_transport_status(
                classification,
                ledger_path=ledger_path,
                run_root=root,
            )
        if classification.kind == "integrity_failure":
            if isinstance(error, IntegrityFailure):
                raise
            raise IntegrityFailure(classification.message) from error
        if current_unit_committed_during_error:
            unit_key = (
                unit_plan[len(completed)].key
                if len(completed) < len(unit_plan)
                else unit_plan[-1].key
            )
        else:
            unit_key = (
                current_expectation.unit.key
                if current_expectation is not None
                else unit_plan[len(completed)].key
            )
        child_kind: Literal["paused", "terminal-model-invalid"] = (
            "paused" if classification.kind == "recoverable_transport" else "terminal-model-invalid"
        )
        if (
            history_path is not None
            and current_expectation is not None
            and not current_unit_committed_during_error
        ):
            append_unit_event(
                "UNIT_PAUSE",
                unit_key=unit_key,
                failure_kind=classification.kind,
            )
        failure_path = attempt_dir / "failure.json"
        write_failure_sidecar(
            failure_path,
            run_root=root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            index=index,
            attempt=attempt,
            unit_key=unit_key,
            child_kind=child_kind,
            failure=classification,
            ledger_path=ledger_path,
            policy_cache_fingerprint=policy_fingerprint_binding,
        )
        validate_failure_sidecar(
            failure_path,
            run_root=root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task_id,
            index=index,
            attempt=attempt,
            unit_key=unit_key,
            ledger_path=ledger_path,
        )
        return ChildOutcome(
            task_id=task_id,
            index=index,
            attempt=attempt,
            kind=child_kind,
            completed_unit_keys=tuple(
                unit.key for unit in unit_plan if unit.key in completed
            ),
            child_result=None,
            failure=classification,
            policy_cache_fingerprint=policy_fingerprint_binding,
        )
    finally:
        pipeline.run_predict_only_replay_for_report = original_replay


# ---------------------------------------------------------------------------
# Task 3: shard scheduler and derived control artifacts

_PROGRESS_FIELDS = {
    "schema_version",
    "artifact_type",
    "manifest_sha256",
    "run_id",
    "shard_id",
    "requested_count",
    "requested_ids",
    "complete_count",
    "complete_ids",
    "terminal_model_invalid_count",
    "terminal_model_invalid_ids",
    "paused_count",
    "paused_ids",
    "pending_count",
    "pending_ids",
    "active_count",
    "active_ids",
    "integrity_failed",
    "integrity_failed_task_id",
    "completed_replay_units",
    "total_replay_units",
    "task_history_schema_version",
    "checkpoint_schema_version",
    "transport_total_attempts",
    "transport_retries_after_initial",
    "transport_retry_delays_seconds",
}
_TRANSPORT_TOTAL_ATTEMPTS = 4
_TRANSPORT_RETRIES_AFTER_INITIAL = 3
_TRANSPORT_RETRY_DELAYS_SECONDS = [15, 30, 60]
_REQUIRED_CHILD_ARTIFACTS = frozenset(
    {
        "aggregate",
        "label_support_preflight",
        "manifest",
        "points",
        "provenance_manifest",
        "quality_report",
        "source_audit_csv",
        "source_audit_json",
        "split_audit_csv",
        "split_audit_json",
        "summary",
        "summary_plot",
        "tier1_plot",
        "tier2_plot",
        "token_plot",
    }
)
_ATTEMPT_EVIDENCE_FIELDS = {
    "schema_version",
    "artifact_type",
    "manifest_sha256",
    "run_id",
    "task_id",
    "source_index",
    "attempt",
    "outcome",
    "unit_key",
    "attempt_start_history_sha256",
    "pre_outcome_history_tail_sha256",
    "failure_kind",
    "policy_cache_fingerprint",
    "failure_sidecar",
    "transport_ledger",
    "child_result",
    "child_artifacts",
    "previous_evidence_sha256",
    "evidence_sha256",
}
_VALIDATION_GATE_FIELDS = {
    "schema_version",
    "artifact_type",
    "manifest_sha256",
    "run_id",
    "shard_id",
    "status",
    "ready",
    "integrity_failed",
    "integrity_failed_task_id",
    "progress",
    "validation_sha256",
}
_TERMINAL_PROJECTION_FIELDS = {
    "schema_version",
    "artifact_type",
    "manifest_sha256",
    "run_id",
    "shard_id",
    "requested_count",
    "terminal_count",
    "entries",
}
_UNIT_BINDING_FIELDS = {
    "unit_key",
    "task_id",
    "run_id",
    "rollout_mode",
    "policy_mode",
    "horizon",
    "selected_point_ids",
    "selected_sequence_nums",
    "cache_signature",
    "result",
    "decisions",
    "decision_count",
}


@dataclass(frozen=True)
class V7TaskDescriptor:
    """All deterministic scheduler paths and unit expectations for one task."""

    task_id: str
    source_index: int
    history_path: Path
    checkpoint_path: Path
    expectations: tuple[UnitExpectation, ...]
    cache_root: Path
    config_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not _TOKEN_RE.fullmatch(self.task_id):
            raise ValueError("task descriptor ID must be a nonempty token")
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("task descriptor source index must be a nonnegative integer")
        if not isinstance(self.expectations, tuple) or len(self.expectations) != 14:
            raise ValueError("task descriptor requires exactly fourteen unit expectations")
        if not isinstance(self.config_paths, tuple) or not self.config_paths:
            raise ValueError("task descriptor requires explicit config paths")
        for path in (
            self.history_path,
            self.checkpoint_path,
            self.cache_root,
            *self.config_paths,
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("task descriptor paths must be absolute Path values")


@dataclass(frozen=True)
class ValidatedV7Task:
    descriptor: V7TaskDescriptor
    state: TaskRecoveryState
    bindings: tuple[UnitBinding, ...]
    policy_cache_fingerprint: Mapping[str, Any] | None
    attempt_evidence: tuple[Mapping[str, Any], ...] = ()
    pending_attempt_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedV7Batch:
    tasks: tuple[ValidatedV7Task, ...]
    unit_plan: tuple[ReplayUnit, ...]
    manifest_sha256: str
    run_id: str


def validate_descriptor_policy_fingerprint(
    fingerprint: Mapping[str, Any],
    *,
    task: V7TaskDescriptor,
) -> dict[str, Any]:
    """Require fingerprint provenance to match the frozen task descriptor."""

    validated = _validate_policy_fingerprint_mapping(dict(fingerprint))
    descriptor_cache = _absolute(task.cache_root)
    if _absolute(Path(validated["cache_root"])) != descriptor_cache:
        raise IntegrityFailure("policy fingerprint cache root differs from descriptor")
    observed_configs = tuple(
        _absolute(Path(row["source_path"]))
        for row in validated["files"]
        if row["kind"] == "config"
    )
    expected_configs = tuple(sorted({_absolute(path) for path in task.config_paths}))
    if observed_configs != expected_configs:
        raise IntegrityFailure("policy fingerprint config paths differ from descriptor")
    observed_cache_rows = [
        row for row in validated["files"] if row["kind"] == "cache"
    ]
    for row in observed_cache_rows:
        source = _absolute(Path(row["source_path"]))
        if not _is_relative_to(source, descriptor_cache):
            raise IntegrityFailure("policy fingerprint cache file is outside descriptor cache")
        expected_relative = f"cache/{source.relative_to(descriptor_cache)}"
        if row["relative_path"] != expected_relative:
            raise IntegrityFailure("policy fingerprint cache relative path is invalid")
    return validated


def v7_batch_lock_path(run_root: Path) -> Path:
    root = _safe_path(run_root, run_root=run_root)
    return root / ".ha-v7-batch.lock"


@contextmanager
def _v7_batch_lock(run_root: Path):
    path = v7_batch_lock_path(run_root)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise IntegrityFailure("v7 batch lock path is unsafe") from exc
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise IntegrityFailure("another v7 batch scheduler is active for this root") from exc
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _validate_scheduler_identity(
    *,
    manifest_sha256: str,
    run_id: str,
    shard_id: int,
) -> None:
    if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
        raise IntegrityFailure("scheduler manifest hash is invalid")
    if not isinstance(run_id, str) or not _TOKEN_RE.fullmatch(run_id):
        raise IntegrityFailure("scheduler run ID is invalid")
    if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
        raise IntegrityFailure("scheduler shard ID is invalid")


def _validate_task_descriptor(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    unit_plan: tuple[ReplayUnit, ...],
) -> V7TaskDescriptor:
    if type(task) is not V7TaskDescriptor:
        raise IntegrityFailure("scheduler task descriptor type is invalid")
    root = _safe_path(run_root, run_root=run_root)
    diagnostic = root / "diagnostics" / task.task_id
    expected_paths = {
        "history": diagnostic / "task_history.jsonl",
        "checkpoint": diagnostic / "checkpoint.json",
        "cache": diagnostic / "policy_cache",
    }
    if (
        _absolute(task.history_path) != expected_paths["history"]
        or _absolute(task.checkpoint_path) != expected_paths["checkpoint"]
        or _absolute(task.cache_root) != expected_paths["cache"]
    ):
        raise IntegrityFailure("task descriptor recovery paths are not deterministic")
    _safe_path(task.history_path, run_root=root)
    _safe_path(task.checkpoint_path, run_root=root)
    cache = _ensure_directory(task.cache_root, run_root=root)
    if not cache.is_dir():
        raise IntegrityFailure("task descriptor cache root is not a directory")
    for path in task.config_paths:
        _assert_no_symlink_components(path)
        if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityFailure("task descriptor config is missing or unsafe")
    if len(unit_plan) != 14 or tuple(value.unit for value in task.expectations) != unit_plan:
        raise IntegrityFailure("task expectations differ from the exact fourteen-unit plan")
    for expectation in task.expectations:
        if (
            expectation.task_id != task.task_id
            or _absolute(expectation.run_root) != root
        ):
            raise IntegrityFailure("task expectation identity differs from its descriptor")
        for path in (
            expectation.result_dir,
            expectation.result_path,
            expectation.guard_run_dir,
            expectation.decision_path,
        ):
            _safe_path(path, run_root=root)
    return task


def _artifact_binding_from_mapping(value: Any, *, label: str) -> ArtifactBinding:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise IntegrityFailure(f"{label} artifact binding schema is invalid")
    try:
        return ArtifactBinding(**value)
    except (TypeError, ValueError) as exc:
        raise IntegrityFailure(f"{label} artifact binding is invalid") from exc


def _unit_binding_from_mapping(value: Any) -> UnitBinding:
    if not isinstance(value, dict) or set(value) != _UNIT_BINDING_FIELDS:
        raise IntegrityFailure("unit binding payload schema is invalid")
    for name in ("selected_point_ids", "selected_sequence_nums"):
        items = value.get(name)
        if not isinstance(items, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in items
        ):
            raise IntegrityFailure("unit binding point identities are invalid")
    for name in ("horizon", "decision_count"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise IntegrityFailure("unit binding integer identity is invalid")
    try:
        return UnitBinding(
            unit_key=value["unit_key"],
            task_id=value["task_id"],
            run_id=value["run_id"],
            rollout_mode=value["rollout_mode"],
            policy_mode=value["policy_mode"],
            horizon=value["horizon"],
            selected_point_ids=tuple(value["selected_point_ids"]),
            selected_sequence_nums=tuple(value["selected_sequence_nums"]),
            cache_signature=value["cache_signature"],
            result=_artifact_binding_from_mapping(value["result"], label="unit result"),
            decisions=_artifact_binding_from_mapping(value["decisions"], label="unit decisions"),
            decision_count=value["decision_count"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityFailure("unit binding payload is invalid") from exc


def _validate_unit_pass_event(
    event: Mapping[str, Any],
    *,
    task: V7TaskDescriptor,
    expectation: UnitExpectation,
    ordinal: int,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
) -> tuple[UnitBinding, dict[str, Any]]:
    binding_artifact = _artifact_binding_from_mapping(event.get("artifact"), label="UNIT_PASS")
    expected_path = (
        _absolute(run_root)
        / "diagnostics"
        / task.task_id
        / "unit_bindings"
        / f"unit-{ordinal:02d}.json"
    )
    if Path(binding_artifact.path) != expected_path or _artifact(expected_path, run_root=run_root) != binding_artifact:
        raise IntegrityFailure("UNIT_PASS binding artifact path/hash mismatch")
    payload = _read_json_object(expected_path)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "artifact_type",
        "binding",
        "policy_cache_fingerprint",
    }:
        raise IntegrityFailure("UNIT_PASS binding evidence schema mismatch")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise IntegrityFailure("UNIT_PASS binding schema version is invalid")
    if payload.get("artifact_type") != "ha_v7_unit_binding":
        raise IntegrityFailure("UNIT_PASS binding artifact type is invalid")
    binding = _unit_binding_from_mapping(payload.get("binding"))
    if binding.unit_key != event.get("unit_key"):
        raise IntegrityFailure("UNIT_PASS event and binding unit identities differ")
    validate_unit_binding(binding, expectation)
    fingerprint_binding = _artifact_binding_from_mapping(
        payload.get("policy_cache_fingerprint"),
        label="policy fingerprint",
    )
    fingerprint = validate_policy_cache_fingerprint_binding(
        fingerprint_binding,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        index=task.source_index,
        attempt=int(event["attempt"]),
    )
    fingerprint = validate_descriptor_policy_fingerprint(fingerprint, task=task)
    return binding, fingerprint


def _fingerprint_from_attempt_evidence(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    attempt: int,
) -> dict[str, Any] | None:
    path = policy_cache_fingerprint_evidence_path(
        run_root=run_root,
        task_id=task.task_id,
        attempt=attempt,
    )
    if not path.exists():
        return None
    binding = _artifact(path, run_root=run_root)
    fingerprint = validate_policy_cache_fingerprint_binding(
        binding,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        index=task.source_index,
        attempt=attempt,
    )
    return validate_descriptor_policy_fingerprint(fingerprint, task=task)


def _validate_failure_evidence_for_attempt(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    attempt: int,
    unit_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_dir = (
        _absolute(run_root)
        / "diagnostics"
        / task.task_id
        / "attempts"
        / f"attempt-{attempt}"
    )
    payload = validate_failure_sidecar(
        attempt_dir / "failure.json",
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        index=task.source_index,
        attempt=attempt,
        unit_key=unit_key,
        ledger_path=attempt_dir / "transport.jsonl",
    )
    fingerprint_binding = _artifact_binding_from_mapping(
        payload.get("policy_cache_fingerprint"),
        label="failure policy fingerprint",
    )
    fingerprint = validate_policy_cache_fingerprint_binding(
        fingerprint_binding,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        index=task.source_index,
        attempt=attempt,
    )
    fingerprint = validate_descriptor_policy_fingerprint(fingerprint, task=task)
    return payload, fingerprint


def v7_attempt_evidence_path(task: V7TaskDescriptor) -> Path:
    return _absolute(task.history_path.parent / "attempt_evidence.jsonl")


def _attempt_evidence_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("evidence_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _optional_artifact_binding(value: Any, *, label: str) -> ArtifactBinding | None:
    if value is None:
        return None
    return _artifact_binding_from_mapping(value, label=label)


def _read_attempt_evidence(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    events: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    """Validate the immutable per-attempt evidence ledger and Task 1 pairing."""

    path = v7_attempt_evidence_path(task)
    if not path.exists():
        if any(
            event["event"]
            in {"TRACE_PAUSE", "TRACE_COMPLETE", "TRACE_TERMINAL_MODEL_INVALID"}
            for event in events
        ):
            raise IntegrityFailure("Task 1 outcome is missing durable attempt evidence")
        return (), None
    safe = _safe_path(path, run_root=run_root, must_exist=True)
    if not stat.S_ISREG(safe.lstat().st_mode):
        raise IntegrityFailure("attempt evidence ledger must be a regular file")
    rows = _read_jsonl_strict(safe)
    previous = "0" * 64
    starts = {
        event["attempt"]: event
        for event in events
        if event["event"] == "TRACE_ATTEMPT_START"
    }
    outcomes = {
        event["attempt"]: event
        for event in events
        if event["event"]
        in {"TRACE_PAUSE", "TRACE_COMPLETE", "TRACE_TERMINAL_MODEL_INVALID"}
    }
    if len(rows) > len(starts):
        raise IntegrityFailure("attempt evidence has no Task 1 attempt start")
    pending: dict[str, Any] | None = None
    validated: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        if set(row) != _ATTEMPT_EVIDENCE_FIELDS:
            raise IntegrityFailure("attempt evidence schema has missing or unknown fields")
        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise IntegrityFailure("attempt evidence schema version is invalid")
        identity = {
            "artifact_type": "ha_v7_attempt_evidence_event",
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "task_id": task.task_id,
            "source_index": task.source_index,
            "attempt": ordinal,
        }
        if any(row.get(name) != value for name, value in identity.items()):
            raise IntegrityFailure("attempt evidence identity is invalid")
        if row.get("previous_evidence_sha256") != previous:
            raise IntegrityFailure("attempt evidence hash chain is broken")
        digest = row.get("evidence_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise IntegrityFailure("attempt evidence hash is invalid")
        if _attempt_evidence_digest(row) != digest:
            raise IntegrityFailure("attempt evidence hash does not match content")
        previous = digest
        outcome = row.get("outcome")
        if outcome not in {"complete", "paused", "terminal-model-invalid", "interrupted"}:
            raise IntegrityFailure("attempt evidence outcome is invalid")
        for field in ("unit_key", "attempt_start_history_sha256", "pre_outcome_history_tail_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or (
                field.endswith("sha256") and not _SHA256_RE.fullmatch(value)
            ):
                raise IntegrityFailure("attempt evidence scalar is invalid")
        start = starts.get(ordinal)
        if start is None or row["attempt_start_history_sha256"] != start["event_sha256"]:
            raise IntegrityFailure("attempt evidence does not bind its Task 1 start")
        trace = outcomes.get(ordinal)
        expected_trace = {
            "complete": "TRACE_COMPLETE",
            "paused": "TRACE_PAUSE",
            "terminal-model-invalid": "TRACE_TERMINAL_MODEL_INVALID",
            "interrupted": "TRACE_PAUSE",
        }[outcome]
        if trace is None:
            if ordinal != len(rows) or ordinal != len(starts):
                raise IntegrityFailure("historical attempt evidence lacks Task 1 outcome")
            pending = row
        elif (
            trace["event"] != expected_trace
            or trace["previous_event_sha256"] != row["pre_outcome_history_tail_sha256"]
        ):
            raise IntegrityFailure("attempt evidence and Task 1 outcome differ")
        supporting_event = next(
            (
                event
                for event in events
                if event["event_sha256"] == row["pre_outcome_history_tail_sha256"]
            ),
            None,
        )
        if supporting_event is None:
            raise IntegrityFailure("attempt evidence history support is missing")
        if outcome == "complete":
            finalizer_only = (
                supporting_event["event"] == "TRACE_ATTEMPT_START"
                and len(
                    [event for event in events if event["event"] == "UNIT_PASS"]
                )
                == 14
            )
            final_unit_pass = (
                supporting_event["event"] == "UNIT_PASS"
                and supporting_event.get("unit_key") == row["unit_key"]
            )
            if not (finalizer_only or final_unit_pass):
                raise IntegrityFailure("complete evidence lacks full-unit history support")
        if outcome in {"paused", "terminal-model-invalid"} and (
            supporting_event["event"] != "UNIT_PAUSE"
            or supporting_event.get("unit_key") != row["unit_key"]
        ):
            raise IntegrityFailure("failure evidence lacks its UNIT_PAUSE support")
        if outcome == "interrupted" and supporting_event["event"] not in {
            "TRACE_ATTEMPT_START",
            "UNIT_PASS",
            "UNIT_PAUSE",
        }:
            raise IntegrityFailure("interrupted evidence history support is invalid")
        failure_kind = row.get("failure_kind")
        if outcome == "complete":
            if failure_kind is not None:
                raise IntegrityFailure("complete attempt evidence cannot carry failure kind")
        else:
            expected_failure = "interrupted" if outcome == "interrupted" else (
                "recoverable_transport" if outcome == "paused" else "terminal_model_invalid"
            )
            if failure_kind != expected_failure:
                raise IntegrityFailure("attempt evidence failure kind is invalid")
            if trace is not None and trace.get("failure_kind") != expected_failure:
                raise IntegrityFailure("attempt evidence failure differs from Task 1 outcome")

        fingerprint_binding = _optional_artifact_binding(
            row.get("policy_cache_fingerprint"), label="attempt policy fingerprint"
        )
        failure_sidecar = _optional_artifact_binding(
            row.get("failure_sidecar"), label="attempt failure sidecar"
        )
        transport_ledger = _optional_artifact_binding(
            row.get("transport_ledger"), label="attempt transport ledger"
        )
        child_result = _optional_artifact_binding(
            row.get("child_result"), label="attempt child result"
        )
        artifacts_raw = row.get("child_artifacts")
        if not isinstance(artifacts_raw, dict):
            raise IntegrityFailure("attempt evidence child artifact map is invalid")
        child_artifacts = {
            name: _artifact_binding_from_mapping(value, label="attempt child artifact")
            for name, value in artifacts_raw.items()
            if isinstance(name, str) and name
        }
        if len(child_artifacts) != len(artifacts_raw):
            raise IntegrityFailure("attempt evidence child artifact name is invalid")

        fingerprint: dict[str, Any] | None = None
        if fingerprint_binding is not None:
            fingerprint = validate_policy_cache_fingerprint_binding(
                fingerprint_binding,
                run_root=run_root,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                task_id=task.task_id,
                index=task.source_index,
                attempt=ordinal,
            )
            validate_descriptor_policy_fingerprint(fingerprint, task=task)
        if outcome in {"paused", "terminal-model-invalid"}:
            if fingerprint_binding is None or failure_sidecar is None or transport_ledger is None:
                raise IntegrityFailure("failure attempt evidence is incomplete")
            attempt_dir = path.parent / "attempts" / f"attempt-{ordinal}"
            if (
                Path(failure_sidecar.path) != attempt_dir / "failure.json"
                or Path(transport_ledger.path) != attempt_dir / "transport.jsonl"
                or _artifact(Path(failure_sidecar.path), run_root=run_root) != failure_sidecar
                or _artifact(Path(transport_ledger.path), run_root=run_root) != transport_ledger
            ):
                raise IntegrityFailure("failure attempt evidence artifact binding differs")
            payload, observed = _validate_failure_evidence_for_attempt(
                task,
                run_root=run_root,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                attempt=ordinal,
                unit_key=row["unit_key"],
            )
            if payload.get("child_kind") != outcome or observed != fingerprint:
                raise IntegrityFailure("failure attempt evidence sidecar differs")
            if child_result is not None or child_artifacts:
                raise IntegrityFailure("failure attempt evidence carries child outputs")
        elif outcome == "complete":
            if fingerprint_binding is None or child_result is None:
                raise IntegrityFailure("complete attempt evidence is incomplete")
            if failure_sidecar is not None or transport_ledger is not None:
                raise IntegrityFailure("complete attempt evidence carries failure artifacts")
            _validate_complete_child_evidence(
                child_result=child_result,
                child_artifacts=child_artifacts,
                policy_cache_fingerprint=fingerprint_binding,
                task=task,
                attempt=ordinal,
                run_root=run_root,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
            )
        else:
            if any(value is not None for value in (failure_sidecar, transport_ledger, child_result)) or child_artifacts:
                raise IntegrityFailure("interrupted attempt evidence carries child outcome artifacts")
        validated.append(dict(row))

    for attempt, trace in outcomes.items():
        if attempt > len(validated):
            raise IntegrityFailure("Task 1 outcome is missing durable attempt evidence")
    return tuple(validated), pending


def _append_attempt_evidence(
    *,
    task: V7TaskDescriptor,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    attempt: int,
    outcome: str,
    unit_key: str,
    attempt_start_history_sha256: str,
    pre_outcome_history_tail_sha256: str,
    failure_kind: str | None,
    policy_cache_fingerprint: ArtifactBinding | None,
    failure_sidecar: ArtifactBinding | None,
    transport_ledger: ArtifactBinding | None,
    child_result: ArtifactBinding | None,
    child_artifacts: Mapping[str, ArtifactBinding],
) -> dict[str, Any]:
    """Durably append one outcome evidence record before its Task 1 event."""

    path = v7_attempt_evidence_path(task)
    _ensure_directory(path.parent, run_root=run_root)
    existing = _read_jsonl_strict(path) if path.exists() else []
    if attempt != len(existing) + 1:
        raise IntegrityFailure("attempt evidence append is duplicate or reordered")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ha_v7_attempt_evidence_event",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task.task_id,
        "source_index": task.source_index,
        "attempt": attempt,
        "outcome": outcome,
        "unit_key": unit_key,
        "attempt_start_history_sha256": attempt_start_history_sha256,
        "pre_outcome_history_tail_sha256": pre_outcome_history_tail_sha256,
        "failure_kind": failure_kind,
        "policy_cache_fingerprint": (
            asdict(policy_cache_fingerprint) if policy_cache_fingerprint is not None else None
        ),
        "failure_sidecar": asdict(failure_sidecar) if failure_sidecar is not None else None,
        "transport_ledger": asdict(transport_ledger) if transport_ledger is not None else None,
        "child_result": asdict(child_result) if child_result is not None else None,
        "child_artifacts": {
            name: asdict(binding) for name, binding in sorted(child_artifacts.items())
        },
        "previous_evidence_sha256": (
            existing[-1].get("evidence_sha256") if existing else "0" * 64
        ),
        "evidence_sha256": "",
    }
    payload["evidence_sha256"] = _attempt_evidence_digest(payload)
    data = _canonical_bytes(payload) + b"\n"
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(data):
            count = os.write(fd, data[written:])
            if count <= 0:
                raise OSError("attempt evidence append made no progress")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)
    return payload


def _validate_one_task_history(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    unit_plan: tuple[ReplayUnit, ...],
    manifest_sha256: str,
    run_id: str,
) -> ValidatedV7Task:
    try:
        events = read_task_history(task.history_path, run_root=run_root)
        state = reduce_task_history(
            events,
            unit_plan=unit_plan,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            task_id=task.task_id,
            source_index=task.source_index,
            run_root=run_root,
        )
    except (OSError, ValueError) as exc:
        raise IntegrityFailure(f"task history is invalid for {task.task_id}") from exc
    pass_events = [event for event in events if event["event"] == "UNIT_PASS"]
    if len(pass_events) != len(state.completed_unit_keys):
        raise IntegrityFailure("history reducer and UNIT_PASS support differ")
    bindings: list[UnitBinding] = []
    fingerprint: dict[str, Any] | None = None
    for ordinal, event in enumerate(pass_events):
        binding, observed_fingerprint = _validate_unit_pass_event(
            event,
            task=task,
            expectation=task.expectations[ordinal],
            ordinal=ordinal,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
        )
        if fingerprint is not None and observed_fingerprint != fingerprint:
            raise IntegrityFailure("policy fingerprint changed across completed units")
        fingerprint = observed_fingerprint
        bindings.append(binding)

    # Recover prior authoritative input identity even when no unit passed.  A
    # transport pause has a strict sidecar; an abrupt pre-unit crash may have
    # only the per-attempt fingerprint evidence.
    for attempt in range(1, state.attempt_count + 1):
        observed = _fingerprint_from_attempt_evidence(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            attempt=attempt,
        )
        if observed is not None:
            if fingerprint is not None and observed != fingerprint:
                raise IntegrityFailure("policy fingerprint changed across task attempts")
            fingerprint = observed

    attempt_evidence, pending_attempt_evidence = _read_attempt_evidence(
        task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        events=events,
    )
    if pending_attempt_evidence is not None and state.status != "active-without-outcome":
        raise IntegrityFailure("pending attempt evidence has no active Task 1 attempt")

    if state.status == "paused" and state.last_failure_kind == "recoverable_transport":
        if state.next_unit is None:
            raise IntegrityFailure("transport pause has no incomplete replay unit")
        payload, observed = _validate_failure_evidence_for_attempt(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            attempt=state.attempt_count,
            unit_key=state.next_unit.key,
        )
        if payload.get("child_kind") != "paused":
            raise IntegrityFailure("paused task sidecar has the wrong child kind")
        if fingerprint is not None and observed != fingerprint:
            raise IntegrityFailure("paused sidecar policy fingerprint changed")
        fingerprint = observed
    if state.status == "terminal-model-invalid":
        if state.next_unit is None:
            raise IntegrityFailure("terminal model-invalid task has no failed replay unit")
        payload, observed = _validate_failure_evidence_for_attempt(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            attempt=state.attempt_count,
            unit_key=state.next_unit.key,
        )
        if payload.get("child_kind") != "terminal-model-invalid":
            raise IntegrityFailure("terminal model-invalid sidecar kind differs")
        if fingerprint is not None and observed != fingerprint:
            raise IntegrityFailure("terminal sidecar policy fingerprint changed")
        fingerprint = observed
    if state.status == "complete" and len(bindings) != len(unit_plan):
        raise IntegrityFailure("complete task lacks fourteen verified unit bindings")
    return ValidatedV7Task(
        descriptor=task,
        state=state,
        bindings=tuple(bindings),
        policy_cache_fingerprint=fingerprint,
        attempt_evidence=attempt_evidence,
        pending_attempt_evidence=pending_attempt_evidence,
    )


def validate_batch_histories(
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    tasks: tuple[V7TaskDescriptor, ...],
    unit_plan: tuple[ReplayUnit, ...],
) -> ValidatedV7Batch:
    """Rebuild every task from hash-chained history and rehash all bindings."""

    root = _safe_path(run_root, run_root=run_root)
    _validate_scheduler_identity(
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        shard_id=0,
    )
    if not isinstance(tasks, tuple) or not tasks:
        raise IntegrityFailure("scheduler requires a nonempty ordered task tuple")
    if not isinstance(unit_plan, tuple) or len(unit_plan) != 14:
        raise IntegrityFailure("scheduler requires the exact fourteen-unit tuple")
    if len({unit.key for unit in unit_plan}) != 14:
        raise IntegrityFailure("scheduler unit plan contains duplicates")
    canonical_plan = build_replay_unit_plan(
        rollout_modes=("direct_multi_horizon", "iterative_wm_rollout"),
        policy_modes=("train_policy", "full_harnessaudit_rules"),
        horizons=(1, 2, 3, 4, 5, 6),
    )
    if unit_plan != canonical_plan:
        raise IntegrityFailure("scheduler unit plan differs from the frozen v7 order")
    task_ids = [task.task_id for task in tasks if isinstance(task, V7TaskDescriptor)]
    source_indices = [task.source_index for task in tasks if isinstance(task, V7TaskDescriptor)]
    if len(task_ids) != len(tasks) or len(set(task_ids)) != len(tasks):
        raise IntegrityFailure("scheduler task IDs are missing or duplicated")
    if len(set(source_indices)) != len(tasks):
        raise IntegrityFailure("scheduler source indices are duplicated")
    validated: list[ValidatedV7Task] = []
    for task in tasks:
        try:
            _validate_task_descriptor(task, run_root=root, unit_plan=unit_plan)
            validated.append(
                _validate_one_task_history(
                    task,
                    run_root=root,
                    unit_plan=unit_plan,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                )
            )
        except IntegrityFailure as exc:
            setattr(exc, "task_id", task.task_id)
            raise
    if sum(item.state.status == "active-without-outcome" for item in validated) > 1:
        raise IntegrityFailure("single-child scheduler cannot have multiple active tasks")
    return ValidatedV7Batch(
        tasks=tuple(validated),
        unit_plan=unit_plan,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
    )


def _validate_progress_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _PROGRESS_FIELDS:
        raise IntegrityFailure("progress schema has missing or unknown fields")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 7:
        raise IntegrityFailure("progress schema version is invalid")
    if payload.get("artifact_type") != "ha_v7_batch_progress":
        raise IntegrityFailure("progress artifact type is invalid")
    if not isinstance(payload.get("manifest_sha256"), str) or not _SHA256_RE.fullmatch(payload["manifest_sha256"]):
        raise IntegrityFailure("progress manifest identity is invalid")
    if not isinstance(payload.get("run_id"), str) or not _TOKEN_RE.fullmatch(payload["run_id"]):
        raise IntegrityFailure("progress run identity is invalid")
    if isinstance(payload.get("shard_id"), bool) or not isinstance(payload.get("shard_id"), int) or payload["shard_id"] < 0:
        raise IntegrityFailure("progress shard identity is invalid")
    list_fields = (
        "requested_ids",
        "complete_ids",
        "terminal_model_invalid_ids",
        "paused_ids",
        "pending_ids",
        "active_ids",
    )
    for field in list_fields:
        value = payload.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not _TOKEN_RE.fullmatch(item) for item in value
        ):
            raise IntegrityFailure("progress ID list is invalid")
        if len(set(value)) != len(value):
            raise IntegrityFailure("progress ID list contains duplicates")
    pairs = (
        ("requested_count", "requested_ids"),
        ("complete_count", "complete_ids"),
        ("terminal_model_invalid_count", "terminal_model_invalid_ids"),
        ("paused_count", "paused_ids"),
        ("pending_count", "pending_ids"),
        ("active_count", "active_ids"),
    )
    for count_field, list_field in pairs:
        count = payload.get(count_field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise IntegrityFailure("progress count is invalid")
        if count != len(payload[list_field]):
            raise IntegrityFailure("progress count differs from its ID list")
    requested = payload["requested_ids"]
    categories = [
        payload["complete_ids"],
        payload["terminal_model_invalid_ids"],
        payload["paused_ids"],
        payload["pending_ids"],
        payload["active_ids"],
    ]
    flattened = [item for category in categories for item in category]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(requested):
        raise IntegrityFailure("progress task categories are not an exact disjoint partition")
    requested_position = {task_id: index for index, task_id in enumerate(requested)}
    if any(
        category != sorted(category, key=requested_position.__getitem__)
        for category in categories
    ):
        raise IntegrityFailure("progress category lists are not in manifest order")
    if type(payload.get("integrity_failed")) is not bool:
        raise IntegrityFailure("progress integrity flag must be a boolean")
    failed_id = payload.get("integrity_failed_task_id")
    if payload["integrity_failed"]:
        if not isinstance(failed_id, str) or failed_id not in requested:
            raise IntegrityFailure("progress integrity failure ID is invalid")
    elif failed_id is not None:
        raise IntegrityFailure("progress clean state cannot name an integrity failure")
    for field in ("completed_replay_units", "total_replay_units"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IntegrityFailure("progress replay-unit count is invalid")
    if payload["completed_replay_units"] > payload["total_replay_units"]:
        raise IntegrityFailure("progress completed units exceed total units")
    if payload["total_replay_units"] != payload["requested_count"] * 14:
        raise IntegrityFailure("progress total units differ from the fourteen-unit plan")
    for field in ("task_history_schema_version", "checkpoint_schema_version"):
        if type(payload.get(field)) is not int or payload[field] != 1:
            raise IntegrityFailure("progress recovery schema version is invalid")
    if (
        type(payload.get("transport_total_attempts")) is not int
        or payload["transport_total_attempts"] != _TRANSPORT_TOTAL_ATTEMPTS
        or type(payload.get("transport_retries_after_initial")) is not int
        or payload["transport_retries_after_initial"] != _TRANSPORT_RETRIES_AFTER_INITIAL
        or payload.get("transport_retry_delays_seconds")
        != _TRANSPORT_RETRY_DELAYS_SECONDS
    ):
        raise IntegrityFailure("progress transport attempt contract is invalid")
    return dict(payload)


def write_v7_progress(
    path: Path,
    *,
    payload: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    """Strictly validate and atomically publish a derived progress snapshot."""

    validated = _validate_progress_payload(dict(payload))
    _atomic_json(path, validated, run_root=run_root)
    return validated


def _build_progress_payload(
    validation: ValidatedV7Batch,
    *,
    shard_id: int,
    integrity_failed_task_id: str | None,
) -> dict[str, Any]:
    requested = [task.descriptor.task_id for task in validation.tasks]
    by_status = {
        "complete": [],
        "terminal-model-invalid": [],
        "paused": [],
        "pending": [],
        "active-without-outcome": [],
    }
    for task in validation.tasks:
        if task.state.status not in by_status:
            raise IntegrityFailure("history reducer returned an unsupported scheduler status")
        by_status[task.state.status].append(task.descriptor.task_id)
    payload = {
        "schema_version": 7,
        "artifact_type": "ha_v7_batch_progress",
        "manifest_sha256": validation.manifest_sha256,
        "run_id": validation.run_id,
        "shard_id": shard_id,
        "requested_count": len(requested),
        "requested_ids": requested,
        "complete_count": len(by_status["complete"]),
        "complete_ids": by_status["complete"],
        "terminal_model_invalid_count": len(by_status["terminal-model-invalid"]),
        "terminal_model_invalid_ids": by_status["terminal-model-invalid"],
        "paused_count": len(by_status["paused"]),
        "paused_ids": by_status["paused"],
        "pending_count": len(by_status["pending"]),
        "pending_ids": by_status["pending"],
        "active_count": len(by_status["active-without-outcome"]),
        "active_ids": by_status["active-without-outcome"],
        "integrity_failed": integrity_failed_task_id is not None,
        "integrity_failed_task_id": integrity_failed_task_id,
        "completed_replay_units": sum(len(task.bindings) for task in validation.tasks),
        "total_replay_units": len(validation.tasks) * len(validation.unit_plan),
        "task_history_schema_version": 1,
        "checkpoint_schema_version": 1,
        "transport_total_attempts": _TRANSPORT_TOTAL_ATTEMPTS,
        "transport_retries_after_initial": _TRANSPORT_RETRIES_AFTER_INITIAL,
        "transport_retry_delays_seconds": list(_TRANSPORT_RETRY_DELAYS_SECONDS),
    }
    return _validate_progress_payload(payload)


def _build_conservative_integrity_progress(
    *,
    tasks: tuple[V7TaskDescriptor, ...],
    manifest_sha256: str,
    run_id: str,
    shard_id: int,
    failed_task_id: str,
) -> dict[str, Any]:
    requested = [task.task_id for task in tasks]
    payload = {
        "schema_version": 7,
        "artifact_type": "ha_v7_batch_progress",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "shard_id": shard_id,
        "requested_count": len(requested),
        "requested_ids": requested,
        "complete_count": 0,
        "complete_ids": [],
        "terminal_model_invalid_count": 0,
        "terminal_model_invalid_ids": [],
        "paused_count": 0,
        "paused_ids": [],
        "pending_count": len(requested),
        "pending_ids": requested,
        "active_count": 0,
        "active_ids": [],
        "integrity_failed": True,
        "integrity_failed_task_id": failed_task_id,
        "completed_replay_units": 0,
        "total_replay_units": len(requested) * 14,
        "task_history_schema_version": 1,
        "checkpoint_schema_version": 1,
        "transport_total_attempts": _TRANSPORT_TOTAL_ATTEMPTS,
        "transport_retries_after_initial": _TRANSPORT_RETRIES_AFTER_INITIAL,
        "transport_retry_delays_seconds": list(_TRANSPORT_RETRY_DELAYS_SECONDS),
    }
    return _validate_progress_payload(payload)


def v7_validation_gate_path(run_root: Path) -> Path:
    root = _safe_path(run_root, run_root=run_root)
    return root / "validation-gate.v7.json"


def _validation_sha256(validation: ValidatedV7Batch) -> str:
    payload = {
        "manifest_sha256": validation.manifest_sha256,
        "run_id": validation.run_id,
        "tasks": [
            {
                "task_id": item.descriptor.task_id,
                "status": item.state.status,
                "history_tail_sha256": item.state.history_tail_sha256,
                "attempt_evidence_tail_sha256": (
                    item.attempt_evidence[-1]["evidence_sha256"]
                    if item.attempt_evidence
                    else "0" * 64
                ),
                "unit_artifacts": [
                    {
                        "result": binding.result.sha256,
                        "decisions": binding.decisions.sha256,
                    }
                    for binding in item.bindings
                ],
            }
            for item in validation.tasks
        ],
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_validation_gate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _VALIDATION_GATE_FIELDS:
        raise IntegrityFailure("validation gate schema is invalid")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 7:
        raise IntegrityFailure("validation gate version is invalid")
    if payload.get("artifact_type") != "ha_v7_validation_gate":
        raise IntegrityFailure("validation gate artifact type is invalid")
    _validate_scheduler_identity(
        manifest_sha256=payload.get("manifest_sha256"),
        run_id=payload.get("run_id"),
        shard_id=payload.get("shard_id"),
    )
    if payload.get("status") not in {"validating", "ready", "incomplete", "integrity-failed"}:
        raise IntegrityFailure("validation gate status is invalid")
    if type(payload.get("ready")) is not bool or type(payload.get("integrity_failed")) is not bool:
        raise IntegrityFailure("validation gate flags are invalid")
    if payload["ready"] != (payload["status"] == "ready"):
        raise IntegrityFailure("validation gate ready flag differs from status")
    if payload["integrity_failed"] != (payload["status"] == "integrity-failed"):
        raise IntegrityFailure("validation gate integrity flag differs from status")
    failed = payload.get("integrity_failed_task_id")
    if payload["integrity_failed"]:
        if not isinstance(failed, str) or not _TOKEN_RE.fullmatch(failed):
            raise IntegrityFailure("validation gate failure task is invalid")
    elif failed is not None:
        raise IntegrityFailure("clean validation gate names a failed task")
    progress = payload.get("progress")
    if progress is not None:
        _artifact_binding_from_mapping(progress, label="validation gate progress")
    signature = payload.get("validation_sha256")
    if signature is not None and (
        not isinstance(signature, str) or not _SHA256_RE.fullmatch(signature)
    ):
        raise IntegrityFailure("validation gate signature is invalid")
    if payload["status"] in {"ready", "incomplete"} and (progress is None or signature is None):
        raise IntegrityFailure("validated gate lacks progress/signature context")
    if payload["status"] == "validating" and (progress is not None or signature is not None):
        raise IntegrityFailure("validating gate cannot claim validated context")
    return dict(payload)


def _write_validation_gate(
    *,
    run_root: Path,
    shard_id: int,
    manifest_sha256: str,
    run_id: str,
    status: str,
    progress: ArtifactBinding | None,
    validation_sha256: str | None,
    integrity_failed_task_id: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 7,
        "artifact_type": "ha_v7_validation_gate",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "shard_id": shard_id,
        "status": status,
        "ready": status == "ready",
        "integrity_failed": status == "integrity-failed",
        "integrity_failed_task_id": integrity_failed_task_id,
        "progress": asdict(progress) if progress is not None else None,
        "validation_sha256": validation_sha256,
    }
    validated = _validate_validation_gate(payload)
    _atomic_json(v7_validation_gate_path(run_root), validated, run_root=run_root)
    return validated


def batch_ready_for_aggregation(
    progress: Mapping[str, Any] | None = None,
    *,
    gate: Mapping[str, Any] | None = None,
    validation: ValidatedV7Batch | None = None,
    run_root: Path | None = None,
    progress_path: Path | None = None,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    shard_id: int | None = None,
    tasks: tuple[V7TaskDescriptor, ...] | None = None,
    unit_plan: tuple[ReplayUnit, ...] | None = None,
) -> bool:
    """Rebuild readiness only from locked, current on-disk authority.

    The first three arguments remain compatibility placeholders and are
    deliberately ignored: callers cannot promote stale in-memory snapshots.
    """

    del progress, gate, validation
    if (
        run_root is None
        or progress_path is None
        or manifest_sha256 is None
        or run_id is None
        or shard_id is None
        or tasks is None
        or unit_plan is None
    ):
        return False
    root = _safe_path(run_root, run_root=run_root)
    with _v7_batch_lock(root):
        return _batch_ready_from_disk(
            run_root=root,
            progress_path=progress_path,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            shard_id=shard_id,
            tasks=tasks,
            unit_plan=unit_plan,
        )


def _batch_ready_from_disk(
    *,
    run_root: Path,
    progress_path: Path,
    manifest_sha256: str,
    run_id: str,
    shard_id: int,
    tasks: tuple[V7TaskDescriptor, ...],
    unit_plan: tuple[ReplayUnit, ...],
) -> bool:
    """Readiness check for a caller that already holds the root batch lock."""

    root = _safe_path(run_root, run_root=run_root)
    expected_progress_path = _safe_path(progress_path, run_root=root, must_exist=True)
    gate_path = v7_validation_gate_path(root)
    if not gate_path.exists():
        return False
    validated_gate = _validate_validation_gate(
        _read_json_object(_safe_path(gate_path, run_root=root, must_exist=True))
    )
    if not validated_gate["ready"]:
        return False
    fresh_validation = validate_batch_histories(
        run_root=root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        tasks=tasks,
        unit_plan=unit_plan,
    )
    validated = _validate_progress_payload(_read_json_object(expected_progress_path))
    if (
        validated_gate["manifest_sha256"] != manifest_sha256
        or validated_gate["run_id"] != run_id
        or validated_gate["shard_id"] != shard_id
        or validated["manifest_sha256"] != manifest_sha256
        or validated["run_id"] != run_id
        or validated["shard_id"] != shard_id
        or validated_gate["validation_sha256"] != _validation_sha256(fresh_validation)
        or validated["requested_ids"]
        != [item.descriptor.task_id for item in fresh_validation.tasks]
    ):
        return False
    progress_binding = _artifact_binding_from_mapping(
        validated_gate["progress"], label="validation gate progress"
    )
    if (
        Path(progress_binding.path) != expected_progress_path
        or _artifact(expected_progress_path, run_root=root) != progress_binding
    ):
        return False
    return (
        not validated["integrity_failed"]
        and validated["paused_count"] == 0
        and validated["pending_count"] == 0
        and validated["active_count"] == 0
        and validated["complete_count"]
        + validated["terminal_model_invalid_count"]
        == validated["requested_count"]
    )


def build_terminal_status_projection(
    validation: ValidatedV7Batch,
    *,
    shard_id: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for task in validation.tasks:
        if task.state.status not in {"complete", "terminal-model-invalid"}:
            raise IntegrityFailure("terminal projection requires every task to be terminal")
        entries.append(
            {
                "index": task.descriptor.source_index,
                "task_id": task.descriptor.task_id,
                "event": (
                    "TRACE_COMPLETE"
                    if task.state.status == "complete"
                    else "TRACE_TERMINAL_MODEL_INVALID"
                ),
                "completed_unit_count": len(task.bindings),
            }
        )
    return {
        "schema_version": 7,
        "artifact_type": "ha_v7_terminal_status_projection",
        "manifest_sha256": validation.manifest_sha256,
        "run_id": validation.run_id,
        "shard_id": shard_id,
        "requested_count": len(validation.tasks),
        "terminal_count": len(entries),
        "entries": entries,
    }


def _validate_terminal_projection(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _TERMINAL_PROJECTION_FIELDS:
        raise IntegrityFailure("terminal projection schema is invalid")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 7:
        raise IntegrityFailure("terminal projection version is invalid")
    if payload.get("artifact_type") != "ha_v7_terminal_status_projection":
        raise IntegrityFailure("terminal projection type is invalid")
    _validate_scheduler_identity(
        manifest_sha256=payload.get("manifest_sha256"),
        run_id=payload.get("run_id"),
        shard_id=payload.get("shard_id"),
    )
    for field in ("requested_count", "terminal_count"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IntegrityFailure("terminal projection count is invalid")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != payload["terminal_count"]:
        raise IntegrityFailure("terminal projection entries/count differ")
    if payload["terminal_count"] != payload["requested_count"]:
        raise IntegrityFailure("terminal projection is incomplete")
    task_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "index",
            "task_id",
            "event",
            "completed_unit_count",
        }:
            raise IntegrityFailure("terminal projection entry schema is invalid")
        if (
            isinstance(entry.get("index"), bool)
            or not isinstance(entry.get("index"), int)
            or entry["index"] < 0
            or isinstance(entry.get("completed_unit_count"), bool)
            or not isinstance(entry.get("completed_unit_count"), int)
            or not 0 <= entry["completed_unit_count"] <= 14
        ):
            raise IntegrityFailure("terminal projection entry integer is invalid")
        if not isinstance(entry.get("task_id"), str) or not _TOKEN_RE.fullmatch(entry["task_id"]):
            raise IntegrityFailure("terminal projection task ID is invalid")
        if entry.get("event") not in {"TRACE_COMPLETE", "TRACE_TERMINAL_MODEL_INVALID"}:
            raise IntegrityFailure("terminal projection event is invalid")
        if entry["event"] == "TRACE_COMPLETE" and entry["completed_unit_count"] != 14:
            raise IntegrityFailure("complete terminal projection lacks fourteen units")
        task_ids.append(entry["task_id"])
    if len(set(task_ids)) != len(task_ids):
        raise IntegrityFailure("terminal projection contains duplicate tasks")
    return dict(payload)


def write_terminal_status_projection(
    path: Path,
    *,
    payload: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    validated = _validate_terminal_projection(dict(payload))
    _atomic_json(path, validated, run_root=run_root)
    return validated


def _sync_task_checkpoint(
    task: ValidatedV7Task,
    *,
    run_root: Path,
    unit_plan: tuple[ReplayUnit, ...],
) -> None:
    write_checkpoint_snapshot(
        task.descriptor.checkpoint_path,
        task.state,
        history_path=task.descriptor.history_path,
        unit_plan=unit_plan,
        history_line_count=task.state.history_line_count,
        history_tail_sha256=task.state.history_tail_sha256,
        run_root=run_root,
    )


def _append_trace_event(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    attempt: int,
    event: str,
    failure_kind: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "event": event,
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task.task_id,
        "source_index": task.source_index,
        "attempt": attempt,
        "timestamp_utc": _timestamp_utc(),
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind
    append_task_event(task.history_path, payload, run_root=run_root)


def _append_unit_pause_event(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    attempt: int,
    unit_key: str,
    failure_kind: str,
) -> None:
    append_task_event(
        task.history_path,
        {
            "schema_version": 1,
            "event": "UNIT_PAUSE",
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "task_id": task.task_id,
            "source_index": task.source_index,
            "attempt": attempt,
            "timestamp_utc": _timestamp_utc(),
            "unit_key": unit_key,
            "failure_kind": failure_kind,
        },
        run_root=run_root,
    )


def _append_trace_outcome_from_evidence(
    task: V7TaskDescriptor,
    evidence: Mapping[str, Any],
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
) -> None:
    event = {
        "complete": "TRACE_COMPLETE",
        "paused": "TRACE_PAUSE",
        "terminal-model-invalid": "TRACE_TERMINAL_MODEL_INVALID",
        "interrupted": "TRACE_PAUSE",
    }[str(evidence["outcome"])]
    _append_trace_event(
        task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        attempt=int(evidence["attempt"]),
        event=event,
        failure_kind=(
            str(evidence["failure_kind"]) if evidence["failure_kind"] is not None else None
        ),
    )


def _record_interrupted_attempt_with_evidence(
    task: V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    unit_plan: tuple[ReplayUnit, ...],
) -> None:
    events = read_task_history(task.history_path, run_root=run_root)
    state = reduce_task_history(
        events,
        unit_plan=unit_plan,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        source_index=task.source_index,
        run_root=run_root,
    )
    if state.status != "active-without-outcome" or not events:
        raise IntegrityFailure("there is no interrupted active attempt to record")
    attempt = state.attempt_count
    last = events[-1]
    if last["event"] == "UNIT_START":
        _append_unit_pause_event(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            attempt=attempt,
            unit_key=last["unit_key"],
            failure_kind="interrupted",
        )
        events = read_task_history(task.history_path, run_root=run_root)
        last = events[-1]
    if last["event"] not in {"TRACE_ATTEMPT_START", "UNIT_PASS", "UNIT_PAUSE"}:
        raise IntegrityFailure("interrupted history tail is not closable")
    start = next(
        event
        for event in events
        if event["event"] == "TRACE_ATTEMPT_START" and event["attempt"] == attempt
    )
    unit_key = (
        last.get("unit_key")
        or (state.next_unit.key if state.next_unit is not None else unit_plan[-1].key)
    )
    fingerprint_path = policy_cache_fingerprint_evidence_path(
        run_root=run_root,
        task_id=task.task_id,
        attempt=attempt,
    )
    fingerprint_binding = (
        _artifact(fingerprint_path, run_root=run_root) if fingerprint_path.exists() else None
    )
    _append_attempt_evidence(
        task=task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        attempt=attempt,
        outcome="interrupted",
        unit_key=unit_key,
        attempt_start_history_sha256=start["event_sha256"],
        pre_outcome_history_tail_sha256=last["event_sha256"],
        failure_kind="interrupted",
        policy_cache_fingerprint=fingerprint_binding,
        failure_sidecar=None,
        transport_ledger=None,
        child_result=None,
        child_artifacts={},
    )
    _append_trace_event(
        task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        attempt=attempt,
        event="TRACE_PAUSE",
        failure_kind="interrupted",
    )


def _validate_child_result(
    outcome: ChildOutcome,
    *,
    task: V7TaskDescriptor,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
) -> dict[str, ArtifactBinding]:
    if outcome.child_result is None:
        raise IntegrityFailure("complete child result binding is missing")
    artifacts = _validate_complete_child_evidence(
        child_result=outcome.child_result,
        child_artifacts=None,
        policy_cache_fingerprint=outcome.policy_cache_fingerprint,
        task=task,
        attempt=outcome.attempt,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
    )
    return artifacts


def _validate_complete_child_evidence(
    *,
    child_result: ArtifactBinding,
    child_artifacts: Mapping[str, ArtifactBinding] | None,
    policy_cache_fingerprint: ArtifactBinding,
    task: V7TaskDescriptor,
    attempt: int,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
) -> dict[str, ArtifactBinding]:
    expected_path = (
        _absolute(run_root)
        / "diagnostics"
        / task.task_id
        / "attempts"
        / f"attempt-{attempt}"
        / "child_result.json"
    )
    if Path(child_result.path) != expected_path or _artifact(expected_path, run_root=run_root) != child_result:
        raise IntegrityFailure("child result path/hash mismatch")
    payload = _read_json_object(expected_path)
    fields = {
        "schema_version",
        "artifact_type",
        "manifest_sha256",
        "run_id",
        "task_id",
        "index",
        "attempt",
        "completed_unit_keys",
        "policy_cache_fingerprint",
        "artifacts",
    }
    if set(payload) != fields:
        raise IntegrityFailure("child result schema is invalid")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise IntegrityFailure("child result schema version is invalid")
    expected_identity = {
        "artifact_type": "ha_v7_child_result",
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "task_id": task.task_id,
        "index": task.source_index,
        "attempt": attempt,
        "completed_unit_keys": [expectation.unit.key for expectation in task.expectations],
        "policy_cache_fingerprint": asdict(policy_cache_fingerprint),
    }
    if any(payload.get(name) != value for name, value in expected_identity.items()):
        raise IntegrityFailure("child result identity differs from typed outcome")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _REQUIRED_CHILD_ARTIFACTS or any(
        not isinstance(name, str) or not name for name in artifacts
    ):
        raise IntegrityFailure("child result artifact map is not the exact required set")
    validated_artifacts: dict[str, ArtifactBinding] = {}
    normalized_paths: set[Path] = set()
    file_identities: set[tuple[int, int]] = set()
    for name, value in artifacts.items():
        binding = _artifact_binding_from_mapping(value, label="child result")
        normalized = _absolute(Path(binding.path))
        if _artifact(normalized, run_root=run_root) != binding:
            raise IntegrityFailure("child result artifact hash/path mismatch")
        metadata = normalized.lstat()
        if metadata.st_size <= 0:
            raise IntegrityFailure("child result artifact must be nonempty")
        file_identity = (metadata.st_dev, metadata.st_ino)
        if normalized in normalized_paths or file_identity in file_identities:
            raise IntegrityFailure("child result artifacts must have distinct files")
        normalized_paths.add(normalized)
        file_identities.add(file_identity)
        validated_artifacts[name] = binding
    if len(normalized_paths) != len(_REQUIRED_CHILD_ARTIFACTS):
        raise IntegrityFailure("child result artifact paths are aliased")
    if child_artifacts is not None and dict(child_artifacts) != validated_artifacts:
        raise IntegrityFailure("attempt evidence child artifacts differ from child result")
    return validated_artifacts


def validate_child_outcome_for_task(
    outcome: ChildOutcome,
    *,
    before: ValidatedV7Task,
    task: V7TaskDescriptor,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    unit_plan: tuple[ReplayUnit, ...],
) -> ValidatedV7Task:
    """Accept only typed child evidence bound to the active task attempt."""

    if type(outcome) is not ChildOutcome:
        raise IntegrityFailure("child returned an unclassified outcome type")
    if (
        outcome.task_id != task.task_id
        or outcome.index != task.source_index
        or outcome.attempt != before.state.attempt_count
    ):
        raise IntegrityFailure("child outcome identity differs from active task")
    observed_fingerprint = validate_policy_cache_fingerprint_binding(
        outcome.policy_cache_fingerprint,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        task_id=task.task_id,
        index=task.source_index,
        attempt=outcome.attempt,
    )
    observed_fingerprint = validate_descriptor_policy_fingerprint(
        observed_fingerprint,
        task=task,
    )
    if (
        before.policy_cache_fingerprint is not None
        and observed_fingerprint != before.policy_cache_fingerprint
    ):
        raise IntegrityFailure("child policy fingerprint differs from resume evidence")
    after = _validate_one_task_history(
        task,
        run_root=run_root,
        unit_plan=unit_plan,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
    )
    if after.state.status != "active-without-outcome" or after.state.attempt_count != outcome.attempt:
        raise IntegrityFailure("child history is not one active matching attempt")
    if after.state.completed_unit_keys != outcome.completed_unit_keys:
        raise IntegrityFailure("child completed prefix differs from authoritative history")
    if outcome.completed_unit_keys[: len(before.state.completed_unit_keys)] != before.state.completed_unit_keys:
        raise IntegrityFailure("child completed prefix discarded a verified unit")
    if tuple(binding.unit_key for binding in after.bindings) != outcome.completed_unit_keys:
        raise IntegrityFailure("child bindings differ from its completed prefix")
    if outcome.kind == "complete":
        if outcome.completed_unit_keys != tuple(unit.key for unit in unit_plan):
            raise IntegrityFailure("complete child lacks the exact fourteen-unit prefix")
        _validate_child_result(
            outcome,
            task=task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
        )
    else:
        if outcome.failure is None or len(outcome.completed_unit_keys) >= len(unit_plan):
            raise IntegrityFailure("non-complete child outcome is malformed")
        unit_key = unit_plan[len(outcome.completed_unit_keys)].key
        payload, sidecar_fingerprint = _validate_failure_evidence_for_attempt(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            attempt=outcome.attempt,
            unit_key=unit_key,
        )
        expected_kind = "paused" if outcome.kind == "paused" else "terminal-model-invalid"
        if (
            payload.get("child_kind") != expected_kind
            or payload.get("failure") != asdict(outcome.failure)
            or payload.get("policy_cache_fingerprint")
            != asdict(outcome.policy_cache_fingerprint)
            or sidecar_fingerprint != observed_fingerprint
        ):
            raise IntegrityFailure("child outcome differs from strict failure sidecar")
    return after


def _append_child_outcome_evidence(
    outcome: ChildOutcome,
    *,
    task: V7TaskDescriptor,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    unit_plan: tuple[ReplayUnit, ...],
) -> dict[str, Any]:
    events = read_task_history(task.history_path, run_root=run_root)
    if not events:
        raise IntegrityFailure("child outcome has no Task 1 history")
    start = next(
        (
            event
            for event in events
            if event["event"] == "TRACE_ATTEMPT_START"
            and event["attempt"] == outcome.attempt
        ),
        None,
    )
    if start is None:
        raise IntegrityFailure("child outcome attempt start is missing")
    tail = events[-1]
    failure_sidecar: ArtifactBinding | None = None
    transport_ledger: ArtifactBinding | None = None
    child_artifacts: dict[str, ArtifactBinding] = {}
    if outcome.kind == "complete":
        child_artifacts = _validate_child_result(
            outcome,
            task=task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
        )
        unit_key = unit_plan[-1].key
        failure_kind = None
    else:
        unit_key = unit_plan[len(outcome.completed_unit_keys)].key
        attempt_dir = (
            _absolute(run_root)
            / "diagnostics"
            / task.task_id
            / "attempts"
            / f"attempt-{outcome.attempt}"
        )
        failure_sidecar = _artifact(attempt_dir / "failure.json", run_root=run_root)
        transport_ledger = _artifact(attempt_dir / "transport.jsonl", run_root=run_root)
        if outcome.failure is None:
            raise IntegrityFailure("failed child outcome has no classification")
        failure_kind = outcome.failure.kind
    return _append_attempt_evidence(
        task=task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        attempt=outcome.attempt,
        outcome=outcome.kind,
        unit_key=unit_key,
        attempt_start_history_sha256=start["event_sha256"],
        pre_outcome_history_tail_sha256=tail["event_sha256"],
        failure_kind=failure_kind,
        policy_cache_fingerprint=outcome.policy_cache_fingerprint,
        failure_sidecar=failure_sidecar,
        transport_ledger=transport_ledger,
        child_result=outcome.child_result,
        child_artifacts=child_artifacts,
    )


def _reconcile_dangling_histories(
    validation: ValidatedV7Batch,
    *,
    run_root: Path,
) -> ValidatedV7Batch:
    current = validation
    for ordered in tuple(current.tasks):
        while True:
            refreshed = next(
                item for item in current.tasks if item.descriptor.task_id == ordered.descriptor.task_id
            )
            if refreshed.state.status != "active-without-outcome":
                break
            task = refreshed.descriptor
            if refreshed.pending_attempt_evidence is not None:
                _append_trace_outcome_from_evidence(
                    task,
                    refreshed.pending_attempt_evidence,
                    run_root=run_root,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                )
                current = validate_batch_histories(
                    run_root=run_root,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                    tasks=tuple(item.descriptor for item in current.tasks),
                    unit_plan=current.unit_plan,
                )
                _sync_task_checkpoint(
                    next(item for item in current.tasks if item.descriptor.task_id == task.task_id),
                    run_root=run_root,
                    unit_plan=current.unit_plan,
                )
                break
            events = read_task_history(task.history_path, run_root=run_root)
            if not events:
                raise IntegrityFailure("active task has no authoritative history")
            last = events[-1]
            if last["event"] == "UNIT_START":
                if refreshed.state.next_unit is None:
                    raise IntegrityFailure("dangling UNIT_START has no matching next unit")
                expectation = task.expectations[len(refreshed.bindings)]
                reconciliation = reconcile_dangling_unit(
                    expectation=expectation,
                    history_path=task.history_path,
                    unit_plan=current.unit_plan,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                    index=task.source_index,
                    attempt=refreshed.state.attempt_count,
                    cache_root=task.cache_root,
                    config_paths=task.config_paths,
                )
                if reconciliation.kind not in {"reconciled-complete", "needs-pause"}:
                    raise IntegrityFailure("dangling reconciliation returned an unknown kind")
                current = validate_batch_histories(
                    run_root=run_root,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                    tasks=tuple(item.descriptor for item in current.tasks),
                    unit_plan=current.unit_plan,
                )
                refreshed = next(
                    item for item in current.tasks if item.descriptor.task_id == task.task_id
                )
                # Unit reconciliation never proves final pipeline provenance.
                # Close the crashed attempt and let a later child use the
                # rehashed prefix; a full prefix becomes a finalizer-only run.
                _record_interrupted_attempt_with_evidence(
                    task,
                    run_root=run_root,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                    unit_plan=current.unit_plan,
                )
            elif last["event"] in {"TRACE_ATTEMPT_START", "UNIT_PASS"}:
                _record_interrupted_attempt_with_evidence(
                    task,
                    run_root=run_root,
                    manifest_sha256=current.manifest_sha256,
                    run_id=current.run_id,
                    unit_plan=current.unit_plan,
                )
            else:
                raise IntegrityFailure("dangling history tail is not reconcilable")
            current = validate_batch_histories(
                run_root=run_root,
                manifest_sha256=current.manifest_sha256,
                run_id=current.run_id,
                tasks=tuple(item.descriptor for item in current.tasks),
                unit_plan=current.unit_plan,
            )
            _sync_task_checkpoint(
                next(item for item in current.tasks if item.descriptor.task_id == task.task_id),
                run_root=run_root,
                unit_plan=current.unit_plan,
            )
            break
    return current


def run_v7_batch(
    *,
    run_root: Path,
    shard_id: int,
    manifest_sha256: str,
    run_id: str,
    tasks: tuple[V7TaskDescriptor, ...],
    unit_plan: tuple[ReplayUnit, ...],
    child_callback: Callable[..., ChildOutcome],
    progress_path: Path,
    terminal_status_path: Path,
) -> dict[str, Any]:
    """Run one child at a time with non-prefix pause/resume semantics."""

    _validate_scheduler_identity(
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        shard_id=shard_id,
    )
    root = _safe_path(run_root, run_root=run_root)
    _safe_path(progress_path, run_root=root)
    _safe_path(terminal_status_path, run_root=root)
    if not callable(child_callback):
        raise IntegrityFailure("scheduler child callback is not callable")
    last_validation: ValidatedV7Batch | None = None
    current_task_id: str | None = None
    with _v7_batch_lock(root):
        if terminal_status_path.exists():
            _safe_path(terminal_status_path, run_root=root, must_exist=True).unlink()
            _fsync_directory(terminal_status_path.parent)
        _write_validation_gate(
            run_root=root,
            shard_id=shard_id,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            status="validating",
            progress=None,
            validation_sha256=None,
        )
        try:
            last_validation = validate_batch_histories(
                run_root=root,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                tasks=tasks,
                unit_plan=unit_plan,
            )
            active_before_reconciliation = [
                item.descriptor.task_id
                for item in last_validation.tasks
                if item.state.status == "active-without-outcome"
            ]
            current_task_id = (
                active_before_reconciliation[0]
                if active_before_reconciliation
                else None
            )
            last_validation = _reconcile_dangling_histories(last_validation, run_root=root)
            current_task_id = None
            initial_progress = _build_progress_payload(
                last_validation,
                shard_id=shard_id,
                integrity_failed_task_id=None,
            )
            write_v7_progress(progress_path, payload=initial_progress, run_root=root)

            paused = [
                item.descriptor.task_id
                for item in last_validation.tasks
                if item.state.status == "paused"
            ]
            pending = [
                item.descriptor.task_id
                for item in last_validation.tasks
                if item.state.status == "pending"
            ]
            schedule = paused + pending
            for task_id in schedule:
                current_task_id = task_id
                # Revalidate every history and bound artifact immediately before
                # each child launch.
                last_validation = validate_batch_histories(
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    tasks=tasks,
                    unit_plan=unit_plan,
                )
                before = next(
                    item for item in last_validation.tasks if item.descriptor.task_id == task_id
                )
                if before.state.status in {"complete", "terminal-model-invalid"}:
                    continue
                if before.state.status not in {"paused", "pending"}:
                    raise IntegrityFailure("scheduled task is neither paused nor pending")
                if before.state.status == "paused" and before.policy_cache_fingerprint is None:
                    raise IntegrityFailure(
                        "paused task lacks an authoritative policy fingerprint for resume"
                    )
                attempt = before.state.attempt_count + 1
                _append_trace_event(
                    before.descriptor,
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    attempt=attempt,
                    event="TRACE_ATTEMPT_START",
                )
                last_validation = validate_batch_histories(
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    tasks=tasks,
                    unit_plan=unit_plan,
                )
                active = next(
                    item for item in last_validation.tasks if item.descriptor.task_id == task_id
                )
                _sync_task_checkpoint(active, run_root=root, unit_plan=unit_plan)
                write_v7_progress(
                    progress_path,
                    payload=_build_progress_payload(
                        last_validation,
                        shard_id=shard_id,
                        integrity_failed_task_id=None,
                    ),
                    run_root=root,
                )
                if attempt > 1:
                    if before.policy_cache_fingerprint is None:
                        raise IntegrityFailure("resume attempt has no prior policy fingerprint")
                    prepare_policy_cache_fingerprint_evidence(
                        run_root=root,
                        cache_root=before.descriptor.cache_root,
                        config_paths=before.descriptor.config_paths,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        task_id=before.descriptor.task_id,
                        index=before.descriptor.source_index,
                        attempt=attempt,
                        require_existing=False,
                        expected=before.policy_cache_fingerprint,
                    )
                outcome = child_callback(
                    task=before.descriptor,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    run_root=root,
                    index=before.descriptor.source_index,
                    attempt=attempt,
                    history_path=before.descriptor.history_path,
                    unit_plan=unit_plan,
                    completed_unit_keys=before.state.completed_unit_keys,
                    verified_bindings=before.bindings,
                    expected_policy_fingerprint=before.policy_cache_fingerprint,
                )
                active = validate_child_outcome_for_task(
                    outcome,
                    before=active,
                    task=before.descriptor,
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    unit_plan=unit_plan,
                )
                _append_child_outcome_evidence(
                    outcome,
                    task=before.descriptor,
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    unit_plan=unit_plan,
                )
                if outcome.kind == "complete":
                    _append_trace_event(
                        before.descriptor,
                        run_root=root,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        attempt=attempt,
                        event="TRACE_COMPLETE",
                    )
                elif outcome.kind == "paused":
                    assert outcome.failure is not None
                    _append_trace_event(
                        before.descriptor,
                        run_root=root,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        attempt=attempt,
                        event="TRACE_PAUSE",
                        failure_kind=outcome.failure.kind,
                    )
                elif outcome.kind == "terminal-model-invalid":
                    assert outcome.failure is not None
                    _append_trace_event(
                        before.descriptor,
                        run_root=root,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        attempt=attempt,
                        event="TRACE_TERMINAL_MODEL_INVALID",
                        failure_kind=outcome.failure.kind,
                    )
                else:
                    raise IntegrityFailure("child returned an unclassified outcome kind")
                last_validation = validate_batch_histories(
                    run_root=root,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    tasks=tasks,
                    unit_plan=unit_plan,
                )
                terminal = next(
                    item for item in last_validation.tasks if item.descriptor.task_id == task_id
                )
                _sync_task_checkpoint(terminal, run_root=root, unit_plan=unit_plan)
                write_v7_progress(
                    progress_path,
                    payload=_build_progress_payload(
                        last_validation,
                        shard_id=shard_id,
                        integrity_failed_task_id=None,
                    ),
                    run_root=root,
                )

            last_validation = validate_batch_histories(
                run_root=root,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                tasks=tasks,
                unit_plan=unit_plan,
            )
            progress = _build_progress_payload(
                last_validation,
                shard_id=shard_id,
                integrity_failed_task_id=None,
            )
            write_v7_progress(progress_path, payload=progress, run_root=root)
            progress_binding = _artifact(progress_path, run_root=root)
            terminal_ready = (
                not progress["integrity_failed"]
                and progress["paused_count"] == 0
                and progress["pending_count"] == 0
                and progress["active_count"] == 0
                and progress["complete_count"]
                + progress["terminal_model_invalid_count"]
                == progress["requested_count"]
            )
            gate = _write_validation_gate(
                run_root=root,
                shard_id=shard_id,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                status="ready" if terminal_ready else "incomplete",
                progress=progress_binding,
                validation_sha256=_validation_sha256(last_validation),
            )
            if _batch_ready_from_disk(
                run_root=root,
                progress_path=progress_path,
                manifest_sha256=manifest_sha256,
                run_id=run_id,
                shard_id=shard_id,
                tasks=tasks,
                unit_plan=unit_plan,
            ):
                write_terminal_status_projection(
                    terminal_status_path,
                    payload=build_terminal_status_projection(
                        last_validation,
                        shard_id=shard_id,
                    ),
                    run_root=root,
                )
            elif terminal_status_path.exists():
                raise IntegrityFailure("nonterminal batch has a stale terminal projection")
            return progress
        except BaseException as exc:
            failure = exc if isinstance(exc, IntegrityFailure) else IntegrityFailure(
                "unclassified child or scheduler failure"
            )
            failed_task_id = (
                current_task_id
                or getattr(exc, "task_id", None)
                or (tasks[0].task_id if tasks and isinstance(tasks[0], V7TaskDescriptor) else None)
            )
            progress_binding: ArtifactBinding | None = None
            if last_validation is not None and failed_task_id is not None:
                try:
                    current = validate_batch_histories(
                        run_root=root,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        tasks=tasks,
                        unit_plan=unit_plan,
                    )
                except IntegrityFailure:
                    current = last_validation
                write_v7_progress(
                    progress_path,
                    payload=_build_progress_payload(
                        current,
                        shard_id=shard_id,
                        integrity_failed_task_id=failed_task_id,
                    ),
                    run_root=root,
                )
                progress_binding = _artifact(progress_path, run_root=root)
            elif failed_task_id is not None and tasks and all(
                isinstance(task, V7TaskDescriptor) for task in tasks
            ):
                write_v7_progress(
                    progress_path,
                    payload=_build_conservative_integrity_progress(
                        tasks=tasks,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,
                        shard_id=shard_id,
                        failed_task_id=failed_task_id,
                    ),
                    run_root=root,
                )
                progress_binding = _artifact(progress_path, run_root=root)
            if failed_task_id is not None:
                _write_validation_gate(
                    run_root=root,
                    shard_id=shard_id,
                    manifest_sha256=manifest_sha256,
                    run_id=run_id,
                    status="integrity-failed",
                    progress=progress_binding,
                    validation_sha256=None,
                    integrity_failed_task_id=failed_task_id,
                )
            if failure is exc:
                raise
            raise failure from exc


_V7_MANIFEST_ARTIFACT = "ha_checkpoint_predict_only_frozen_manifest_v7"
_V7_VERIFIED_PASS_IDS = ("dl-t12", "ec-t46", "ec-t21", "ec-t8", "ec-t42")
_V7_EXECUTION_CONTRACT = {
    "transport_total_attempts": 4,
    "transport_retries_after_initial": 3,
    "retry_delay_seconds": 15,
    "retry_backoff": 2,
    "per_shard_manifest_concurrency": 1,
    "array_concurrency": 2,
    "terminal_transport_failure": "pause_and_continue",
}
_V7_CORE_IMPLEMENTATION_RELATIVE_PATHS = (
    "harness_env/orgin/HarnessAudit/multi_agent/safepred_experiments/ha_checkpoint_state_v7.py",
    "harness_env/orgin/HarnessAudit/multi_agent/safepred_experiments/ha_transport_observer_v7.py",
    "harness_env/orgin/HarnessAudit/multi_agent/safepred_experiments/ha_resilient_predict_only_v7.py",
)
_V7_DEFAULT_RUN_ENTRY_ROOT = "runs/qwen35_plus_predict_only_checkpoint_v7_20260805"
_V7_RUN_ENTRY_FILENAMES = (
    "ha_checkpoint_array.sbatch",
    "prepare_ha_shards.py",
    "submit_ha_from_credentials.sh",
)
_V7_POLICY_ENVIRONMENT = {
    "SAFEPRED_TRAIN_POLICY_SPLIT_FILE": "",
    "SAFEPRED_TRAIN_POLICY_SPLIT_ID": "",
    "SAFEPRED_TRAIN_POLICY_TEST_TASKS": "",
    "SAFEPRED_TRAIN_POLICY_STRICT_SPLIT": "",
    "SAFEPRED_TRAIN_POLICY_WRITE_CACHE": "",
    "SAFEPRED_ALLOW_TRAIN_POLICY_FALLBACK": "",
}


def _v7_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "harness_env").is_dir() and (parent / "runs").is_dir():
            return parent
    raise IntegrityFailure("cannot resolve project root for v7 manifest")


def _v7_relative(path: Path, *, project_root: Path) -> str:
    root = _absolute(project_root)
    candidate = _absolute(path)
    _assert_no_symlink_components(candidate)
    if not _is_relative_to(candidate, root):
        raise IntegrityFailure("bound path is outside project root")
    relative = candidate.relative_to(root)
    if not relative.parts or ".." in relative.parts:
        raise IntegrityFailure("bound project-relative path is unsafe")
    return relative.as_posix()


def _v7_project_file_record(
    path: Path,
    *,
    project_root: Path,
    kind: str,
    allow_missing: bool = False,
) -> dict[str, Any]:
    candidate = _absolute(path)
    relative = _v7_relative(candidate, project_root=project_root)
    if not candidate.exists():
        if not allow_missing:
            raise IntegrityFailure(f"required {kind} file is missing")
        return {
            "kind": kind,
            "relative_path": relative,
            "exists": False,
            "size": 0,
            "sha256": None,
        }
    if stat.S_ISLNK(candidate.lstat().st_mode) or not stat.S_ISREG(candidate.lstat().st_mode):
        raise IntegrityFailure(f"bound {kind} path is not a regular nonsymlink file")
    return {
        "kind": kind,
        "relative_path": relative,
        "exists": True,
        "size": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _v7_validate_project_file_record(
    raw: Any,
    *,
    project_root: Path,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    fields = {"kind", "relative_path", "exists", "size", "sha256"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise IntegrityFailure("v7 project file record schema mismatch")
    kind = raw.get("kind")
    relative = raw.get("relative_path")
    exists = raw.get("exists")
    size = raw.get("size")
    digest = raw.get("sha256")
    if not isinstance(kind, str) or not kind or (expected_kind and kind != expected_kind):
        raise IntegrityFailure("v7 project file kind mismatch")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or Path(relative).as_posix() != relative
    ):
        raise IntegrityFailure("v7 project file relative path is unsafe")
    if type(exists) is not bool or type(size) is not int or size < 0:
        raise IntegrityFailure("v7 project file scalar type is invalid")
    if exists:
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise IntegrityFailure("v7 project file digest is invalid")
    elif size != 0 or digest is not None:
        raise IntegrityFailure("absent v7 project file record is malformed")
    path = _absolute(project_root) / relative
    current = _v7_project_file_record(
        path,
        project_root=project_root,
        kind=kind,
        allow_missing=True,
    )
    if current != raw:
        raise IntegrityFailure("bound v7 project file changed")
    return dict(raw)


def reconstruct_v6_continuation_cohort(
    *,
    parent_manifest_path: Path,
    v6_manifest_paths: Iterable[Path],
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly reconstruct the exact 152 v6 non-PASS task cohort."""

    from . import ha_resilient_predict_only as legacy

    root = _absolute(project_root or _v7_project_root())
    parent_path = _absolute(parent_manifest_path)
    parent = _read_json_object(parent_path)
    legacy.validate_frozen_manifest(parent)
    parent_ids = parent.get("ordered_task_ids")
    if (
        not isinstance(parent_ids, list)
        or len(parent_ids) != 182
        or len(set(parent_ids)) != 182
        or any(not isinstance(task_id, str) or not task_id for task_id in parent_ids)
    ):
        raise IntegrityFailure("parent scientific cohort must contain 182 unique tasks")
    parent_index = {task_id: index for index, task_id in enumerate(parent_ids)}
    manifests_by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    predecessor_files: list[dict[str, Any]] = []
    v6_ids: list[str] = []
    verified: list[str] = []
    paths = tuple(_absolute(path) for path in v6_manifest_paths)
    if len(paths) != 4 or len(set(paths)) != 4:
        raise IntegrityFailure("v6 continuation reconstruction requires four manifests")
    for path in paths:
        manifest = _read_json_object(path)
        legacy.validate_frozen_manifest(manifest)
        shard = manifest.get("shard")
        if not isinstance(shard, dict) or set(shard) != {
            "index", "count", "source_indices", "parent_manifest_sha256"
        }:
            raise IntegrityFailure("v6 shard record schema mismatch")
        shard_index = shard.get("index")
        if type(shard_index) is not int or shard_index not in range(4) or shard.get("count") != 4:
            raise IntegrityFailure("v6 shard index/count is invalid")
        if shard_index in manifests_by_index:
            raise IntegrityFailure("duplicate v6 shard index")
        if shard.get("parent_manifest_sha256") != parent.get("manifest_sha256"):
            raise IntegrityFailure("v6 shard parent scientific identity mismatch")
        local_ids = manifest.get("ordered_task_ids")
        local_indices = shard.get("source_indices")
        if (
            not isinstance(local_ids, list)
            or not isinstance(local_indices, list)
            or len(local_ids) != len(local_indices)
            or any(type(value) is not int for value in local_indices)
            or [parent_ids[value] for value in local_indices] != local_ids
        ):
            raise IntegrityFailure("v6 shard source mapping differs from parent order")
        for field in ("model", "matrix", "partial_audit", "resource_semantics"):
            if manifest.get(field) != parent.get(field):
                raise IntegrityFailure(f"v6 shard changed parent scientific {field}")
        manifests_by_index[shard_index] = (path, manifest)
    if set(manifests_by_index) != set(range(4)):
        raise IntegrityFailure("v6 shard set is incomplete")
    for shard_index in range(4):
        path, manifest = manifests_by_index[shard_index]
        local_ids = list(manifest["ordered_task_ids"])
        if set(v6_ids).intersection(local_ids):
            raise IntegrityFailure("v6 task appears in multiple shards")
        v6_ids.extend(local_ids)
        status_path = path.parent / "batch.status.jsonl"
        events = legacy._read_status_events(status_path, run_root=path.parent)
        passed, _skipped, _dangling_index, _dangling_start = legacy._validate_terminal_history(
            manifest,
            path.parent,
            events,
        )
        verified.extend(passed)
        predecessor_files.extend(
            [
                _v7_project_file_record(
                    path,
                    project_root=root,
                    kind="v6_manifest",
                ),
                _v7_project_file_record(
                    status_path,
                    project_root=root,
                    kind="v6_status",
                    allow_missing=True,
                ),
                _v7_project_file_record(
                    path.parent / "progress.json",
                    project_root=root,
                    kind="v6_progress",
                    allow_missing=True,
                ),
            ]
        )
    if len(v6_ids) != 157 or len(set(v6_ids)) != 157:
        raise IntegrityFailure("v6 source cohort must contain exactly 157 unique tasks")
    if any(task_id not in parent_index for task_id in v6_ids):
        raise IntegrityFailure("v6 cohort contains a task outside the 182-task parent")
    if set(verified) != set(_V7_VERIFIED_PASS_IDS) or len(verified) != 5:
        raise IntegrityFailure("v6 verified PASS set differs from the approved five tasks")
    v6_set = set(v6_ids)
    continuation_ids = [
        task_id
        for task_id in parent_ids
        if task_id in v6_set and task_id not in set(_V7_VERIFIED_PASS_IDS)
    ]
    if len(continuation_ids) != 152 or len(set(continuation_ids)) != 152:
        raise IntegrityFailure("v7 continuation cohort must contain exactly 152 unique tasks")
    predecessor_files = sorted(
        predecessor_files,
        key=lambda row: (row["relative_path"], row["kind"]),
    )
    parent_record = _v7_project_file_record(
        parent_path,
        project_root=root,
        kind="parent_scientific_manifest",
    )
    return {
        "parent_count": 182,
        "parent_task_ids": list(parent_ids),
        "parent_task_digest": hashlib.sha256(
            _canonical_bytes({"ordered_task_ids": parent_ids})
        ).hexdigest(),
        "parent_manifest": parent_record,
        "v6_count": 157,
        "v6_task_digest": hashlib.sha256(
            _canonical_bytes({"ordered_task_ids": [task_id for task_id in parent_ids if task_id in v6_set]})
        ).hexdigest(),
        "continuation_count": 152,
        "continuation_task_ids": continuation_ids,
        "continuation_task_digest": hashlib.sha256(
            _canonical_bytes({"ordered_task_ids": continuation_ids})
        ).hexdigest(),
        "continuation_source_indices": [parent_index[task_id] for task_id in continuation_ids],
        "verified_pass_ids": list(_V7_VERIFIED_PASS_IDS),
        "predecessor_files": predecessor_files,
        "predecessor_digest": hashlib.sha256(
            _canonical_bytes({"files": predecessor_files})
        ).hexdigest(),
    }


def _v7_checkpoint_contract(scientific_manifest: Mapping[str, Any]) -> dict[str, Any]:
    units = enumerate_replay_units(scientific_manifest)
    return {
        "unit_count_per_task": 14,
        "ordered_unit_keys": [unit.key for unit in units],
        "task_history_schema_version": 1,
        "checkpoint_schema_version": 1,
        "progress_schema_version": 7,
        "transport_ledger_schema_version": 2,
        "attempt_evidence_schema_version": 1,
        "root_lock": "ha_v7_nonblocking_flock",
        "cross_run_cache_import": False,
    }


def _v7_manifest_run_id(payload: Mapping[str, Any]) -> str:
    identity = {
        key: payload[key]
        for key in (
            "scientific_manifest",
            "parent_scientific_manifest",
            "implementation_manifest_sha256",
            "predecessor_digest",
            "execution_contract",
            "checkpoint_contract",
            "cohort",
            "shard",
            "model_contract",
            "credential_contract",
        )
    }
    return "ha-v7-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:24]


def _v7_credential_contract(
    *,
    credential_selector_name: str,
    gateway_group: str,
    credential_profile_relative_path: str,
    run_entry_relative_path: str,
) -> dict[str, Any]:
    for label, value in (
        ("credential selector", credential_selector_name),
        ("gateway group", gateway_group),
        ("credential profile", credential_profile_relative_path),
        ("run entry root", run_entry_relative_path),
    ):
        if not isinstance(value, str) or not value:
            raise IntegrityFailure(f"v7 {label} is invalid")
    if credential_selector_name != "AIFAST_QWEN_API_KEY" or gateway_group != "qwen":
        raise IntegrityFailure("v7 Qwen credential selector/group changed")
    for label, value in (
        ("credential profile", credential_profile_relative_path),
        ("run entry root", run_entry_relative_path),
    ):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise IntegrityFailure(f"v7 {label} path is unsafe")
    if credential_profile_relative_path != "config/experiment.env":
        raise IntegrityFailure("v7 credential profile identity changed")
    if not run_entry_relative_path.startswith("runs/"):
        raise IntegrityFailure("v7 run entry root must remain project-run relative")
    return {
        "api_key_env": credential_selector_name,
        "gateway_group": gateway_group,
        "gateway_root": "https://aifast.site",
        "openai_compatible_endpoint": "https://aifast.site/v1",
        "profile_relative_path": credential_profile_relative_path,
        "required_profile_mode": "600",
        "loader": "project_experiment_env",
        "secret_persistence": "environment_only",
        "run_entry_root": run_entry_relative_path,
    }


def build_v7_execution_manifest(
    scientific_manifest_path: Path,
    *,
    run_root: Path,
    project_root: Path | None,
    cohort: Mapping[str, Any],
    shard_index: int,
    shard_mapping: Iterable[Mapping[str, Any]],
    proxy_variable_name: str = "QWEN_RELAY_PROXY_URL",
    implementation_paths: Iterable[Path] | None = None,
    credential_selector_name: str = "AIFAST_QWEN_API_KEY",
    gateway_group: str = "qwen",
    credential_profile_relative_path: str = "config/experiment.env",
    run_entry_relative_path: str = _V7_DEFAULT_RUN_ENTRY_ROOT,
) -> dict[str, Any]:
    """Build one strict schema-v7 execution wrapper around schema-v1 science."""

    from . import ha_resilient_predict_only as legacy

    project = _absolute(project_root or _v7_project_root())
    root = _absolute(run_root)
    science_path = _safe_path(scientific_manifest_path, run_root=root, must_exist=True)
    science = _read_json_object(science_path)
    legacy.validate_frozen_manifest(science)
    mapping = [dict(row) for row in shard_mapping]
    if type(shard_index) is not int or shard_index not in range(4) or len(mapping) != 4:
        raise IntegrityFailure("v7 shard mapping must contain indices 0..3")
    if [row.get("shard_index") for row in mapping] != [0, 1, 2, 3]:
        raise IntegrityFailure("v7 shard mapping order is invalid")
    local = mapping[shard_index]
    if local.get("task_ids") != science.get("ordered_task_ids"):
        raise IntegrityFailure("scientific manifest is not the declared v7 shard projection")
    predecessor_files = list(cohort.get("predecessor_files") or [])
    credential_contract = _v7_credential_contract(
        credential_selector_name=credential_selector_name,
        gateway_group=gateway_group,
        credential_profile_relative_path=credential_profile_relative_path,
        run_entry_relative_path=run_entry_relative_path,
    )
    default_implementation_paths = (
        *(project / relative for relative in _V7_CORE_IMPLEMENTATION_RELATIVE_PATHS),
        *(project / run_entry_relative_path / filename for filename in _V7_RUN_ENTRY_FILENAMES),
    )
    implementation_candidates = tuple(
        implementation_paths or default_implementation_paths
    )
    implementation = [
        _v7_project_file_record(
            path,
            project_root=project,
            kind="v7_implementation",
        )
        for path in implementation_candidates
    ]
    implementation = sorted(implementation, key=lambda row: row["relative_path"])
    model = science.get("model")
    matrix = science.get("matrix")
    if not isinstance(model, dict) or not isinstance(matrix, dict):
        raise IntegrityFailure("scientific model/matrix identity is incomplete")
    science_binding = {
        "relative_path": science_path.relative_to(root).as_posix(),
        "size": science_path.stat().st_size,
        "sha256": _sha256(science_path),
        "manifest_sha256": science.get("manifest_sha256"),
    }
    if science_binding["relative_path"] != "scientific_manifest.json":
        raise IntegrityFailure("v7 scientific manifest must use scientific_manifest.json")
    cohort_record = {
        "parent_count": cohort.get("parent_count"),
        "parent_task_digest": cohort.get("parent_task_digest"),
        "v6_count": cohort.get("v6_count"),
        "v6_task_digest": cohort.get("v6_task_digest"),
        "continuation_count": cohort.get("continuation_count"),
        "continuation_task_ids": cohort.get("continuation_task_ids"),
        "continuation_task_digest": cohort.get("continuation_task_digest"),
        "verified_pass_ids": cohort.get("verified_pass_ids"),
    }
    payload: dict[str, Any] = {
        "schema_version": 7,
        "artifact_type": _V7_MANIFEST_ARTIFACT,
        "scientific_manifest": science_binding,
        "parent_scientific_manifest": dict(cohort.get("parent_manifest") or {}),
        "implementation_files": implementation,
        "implementation_manifest_sha256": hashlib.sha256(
            _canonical_bytes({"files": implementation})
        ).hexdigest(),
        "predecessor_files": predecessor_files,
        "predecessor_digest": cohort.get("predecessor_digest"),
        "execution_contract": dict(_V7_EXECUTION_CONTRACT),
        "checkpoint_contract": _v7_checkpoint_contract(science),
        "cohort": cohort_record,
        "shard": {
            "index": shard_index,
            "count": 4,
            "mapping": mapping,
        },
        "model_contract": {
            "id": model.get("id"),
            "endpoint": model.get("endpoint"),
            "requested_revision_label": model.get("requested_revision_label"),
            "reasoning_mode": model.get("reasoning_mode", "provider_default"),
            "seed": model.get("seed"),
            "proxy_variable_name": proxy_variable_name,
            "target": matrix.get("target"),
            "horizons": matrix.get("horizons"),
            "rollout_modes": matrix.get("rollout_modes"),
            "policy_modes": matrix.get("policy_modes"),
        },
        "credential_contract": credential_contract,
    }
    payload["run_id"] = _v7_manifest_run_id(payload)
    payload["manifest_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    validate_v7_execution_manifest(payload, run_root=root, project_root=project)
    return payload


def validate_v7_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    run_root: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly rehash a schema-v7 manifest and all frozen dependencies."""

    from . import ha_resilient_predict_only as legacy

    fields = {
        "schema_version", "artifact_type", "run_id", "scientific_manifest",
        "parent_scientific_manifest", "implementation_files",
        "implementation_manifest_sha256", "predecessor_files", "predecessor_digest",
        "execution_contract", "checkpoint_contract", "cohort", "shard",
        "model_contract", "credential_contract", "manifest_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise IntegrityFailure("v7 execution manifest schema mismatch")
    value = dict(manifest)
    if type(value.get("schema_version")) is not int or value["schema_version"] != 7:
        raise IntegrityFailure("v7 execution manifest version is invalid")
    if value.get("artifact_type") != _V7_MANIFEST_ARTIFACT:
        raise IntegrityFailure("v7 execution manifest artifact type is invalid")
    digest = value.get("manifest_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise IntegrityFailure("v7 execution manifest hash is invalid")
    without_hash = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() != digest:
        raise IntegrityFailure("v7 execution manifest self hash mismatch")
    if value.get("execution_contract") != _V7_EXECUTION_CONTRACT:
        raise IntegrityFailure("v7 execution contract changed")
    project = _absolute(project_root or _v7_project_root())
    root = _safe_path(run_root, run_root=run_root)
    science_record = value.get("scientific_manifest")
    if not isinstance(science_record, dict) or set(science_record) != {
        "relative_path", "size", "sha256", "manifest_sha256"
    }:
        raise IntegrityFailure("v7 scientific manifest binding schema mismatch")
    if science_record.get("relative_path") != "scientific_manifest.json":
        raise IntegrityFailure("v7 scientific manifest relative path changed")
    if type(science_record.get("size")) is not int or science_record["size"] <= 0:
        raise IntegrityFailure("v7 scientific manifest size is invalid")
    if not isinstance(science_record.get("sha256"), str) or not _SHA256_RE.fullmatch(science_record["sha256"]):
        raise IntegrityFailure("v7 scientific manifest digest is invalid")
    if not isinstance(science_record.get("manifest_sha256"), str) or not _SHA256_RE.fullmatch(science_record["manifest_sha256"]):
        raise IntegrityFailure("v7 scientific manifest self digest is invalid")
    science_path = _safe_path(root / science_record["relative_path"], run_root=root, must_exist=True)
    if science_path.stat().st_size != science_record["size"] or _sha256(science_path) != science_record["sha256"]:
        raise IntegrityFailure("v7 scientific manifest binding changed")
    science = _read_json_object(science_path)
    legacy.validate_frozen_manifest(science)
    if science.get("manifest_sha256") != science_record["manifest_sha256"]:
        raise IntegrityFailure("v7 scientific manifest self identity mismatch")
    parent_record = _v7_validate_project_file_record(
        value.get("parent_scientific_manifest"),
        project_root=project,
        expected_kind="parent_scientific_manifest",
    )
    parent = _read_json_object(project / parent_record["relative_path"])
    legacy.validate_frozen_manifest(parent)
    credential = value.get("credential_contract")
    if not isinstance(credential, dict):
        raise IntegrityFailure("v7 credential contract schema mismatch")
    expected_credential = _v7_credential_contract(
        credential_selector_name=credential.get("api_key_env"),
        gateway_group=credential.get("gateway_group"),
        credential_profile_relative_path=credential.get("profile_relative_path"),
        run_entry_relative_path=credential.get("run_entry_root"),
    )
    if credential != expected_credential:
        raise IntegrityFailure("v7 credential contract changed")
    profile_path = project / credential["profile_relative_path"]
    _assert_no_symlink_components(profile_path)
    if not profile_path.exists() or not stat.S_ISREG(profile_path.lstat().st_mode):
        raise IntegrityFailure("v7 credential profile is missing or unsafe")
    if stat.S_IMODE(profile_path.stat().st_mode) != 0o600:
        raise IntegrityFailure("v7 credential profile permission must be 600")
    expected_implementation_paths = sorted(
        (
            *_V7_CORE_IMPLEMENTATION_RELATIVE_PATHS,
            *(
                f"{credential['run_entry_root']}/{filename}"
                for filename in _V7_RUN_ENTRY_FILENAMES
            ),
        )
    )
    implementations = value.get("implementation_files")
    if not isinstance(implementations, list) or len(implementations) != len(
        expected_implementation_paths
    ):
        raise IntegrityFailure("v7 implementation inventory has the wrong file count")
    validated_implementations = [
        _v7_validate_project_file_record(
            row,
            project_root=project,
            expected_kind="v7_implementation",
        )
        for row in implementations
    ]
    if [row["relative_path"] for row in validated_implementations] != expected_implementation_paths:
        raise IntegrityFailure("v7 implementation file set changed")
    implementation_digest = hashlib.sha256(
        _canonical_bytes({"files": validated_implementations})
    ).hexdigest()
    if value.get("implementation_manifest_sha256") != implementation_digest:
        raise IntegrityFailure("v7 implementation manifest digest mismatch")
    predecessors = value.get("predecessor_files")
    if not isinstance(predecessors, list) or len(predecessors) != 12:
        raise IntegrityFailure("v7 predecessor inventory must contain twelve records")
    validated_predecessors = [
        _v7_validate_project_file_record(row, project_root=project)
        for row in predecessors
    ]
    if [row["relative_path"] for row in validated_predecessors] != sorted(
        row["relative_path"] for row in validated_predecessors
    ):
        raise IntegrityFailure("v7 predecessor inventory order changed")
    predecessor_digest = hashlib.sha256(
        _canonical_bytes({"files": validated_predecessors})
    ).hexdigest()
    if value.get("predecessor_digest") != predecessor_digest:
        raise IntegrityFailure("v7 predecessor digest mismatch")
    v6_paths = [
        project / row["relative_path"]
        for row in validated_predecessors
        if row["kind"] == "v6_manifest"
    ]
    cohort_current = reconstruct_v6_continuation_cohort(
        parent_manifest_path=project / parent_record["relative_path"],
        v6_manifest_paths=v6_paths,
        project_root=project,
    )
    cohort = value.get("cohort")
    expected_cohort_fields = {
        "parent_count", "parent_task_digest", "v6_count", "v6_task_digest",
        "continuation_count", "continuation_task_ids", "continuation_task_digest",
        "verified_pass_ids",
    }
    if not isinstance(cohort, dict) or set(cohort) != expected_cohort_fields:
        raise IntegrityFailure("v7 cohort schema mismatch")
    for count_name, expected in (("parent_count", 182), ("v6_count", 157), ("continuation_count", 152)):
        if type(cohort.get(count_name)) is not int or cohort[count_name] != expected:
            raise IntegrityFailure("v7 cohort count changed")
    expected_cohort = {
        field: cohort_current[field]
        for field in expected_cohort_fields
    }
    if cohort != expected_cohort:
        raise IntegrityFailure("v7 continuation cohort differs from frozen v6 evidence")
    shard = value.get("shard")
    if not isinstance(shard, dict) or set(shard) != {"index", "count", "mapping"}:
        raise IntegrityFailure("v7 shard schema mismatch")
    if type(shard.get("index")) is not int or shard["index"] not in range(4) or shard.get("count") != 4:
        raise IntegrityFailure("v7 shard index/count is invalid")
    mapping = shard.get("mapping")
    if not isinstance(mapping, list) or len(mapping) != 4:
        raise IntegrityFailure("v7 shard mapping is incomplete")
    reconstructed: list[str] = []
    all_ids: list[str] = []
    continuation_ids = cohort["continuation_task_ids"]
    parent_ids = cohort_current["parent_task_ids"]
    for index, row in enumerate(mapping):
        if not isinstance(row, dict) or set(row) != {"shard_index", "task_ids", "source_indices"}:
            raise IntegrityFailure("v7 shard mapping row schema mismatch")
        if type(row.get("shard_index")) is not int or row["shard_index"] != index:
            raise IntegrityFailure("v7 shard mapping index changed")
        expected_ids = continuation_ids[index::4]
        expected_indices = [parent_ids.index(task_id) for task_id in expected_ids]
        if row.get("task_ids") != expected_ids or row.get("source_indices") != expected_indices:
            raise IntegrityFailure("v7 shard mapping differs from deterministic stride")
        all_ids.extend(row["task_ids"])
    for offset in range(len(continuation_ids)):
        reconstructed.append(mapping[offset % 4]["task_ids"][offset // 4])
    if reconstructed != continuation_ids or len(set(all_ids)) != 152:
        raise IntegrityFailure("v7 shard mapping duplicates, omits, or reshards a task")
    if science.get("ordered_task_ids") != mapping[shard["index"]]["task_ids"]:
        raise IntegrityFailure("v7 local scientific cohort differs from shard mapping")
    parent_sources = {row["task_id"]: row for row in parent["source_reports"]}
    if science.get("source_reports") != [parent_sources[task_id] for task_id in science["ordered_task_ids"]]:
        raise IntegrityFailure("v7 local source records differ from the 182-task parent")
    checkpoint = value.get("checkpoint_contract")
    if checkpoint != _v7_checkpoint_contract(science):
        raise IntegrityFailure("v7 checkpoint/unit contract changed")
    if type(checkpoint.get("cross_run_cache_import")) is not bool or checkpoint["cross_run_cache_import"]:
        raise IntegrityFailure("v7 cross-run cache import must remain false")
    model = science["model"]
    matrix = science["matrix"]
    expected_model = {
        "id": model.get("id"),
        "endpoint": model.get("endpoint"),
        "requested_revision_label": model.get("requested_revision_label"),
        "reasoning_mode": model.get("reasoning_mode", "provider_default"),
        "seed": model.get("seed"),
        "proxy_variable_name": "QWEN_RELAY_PROXY_URL",
        "target": matrix.get("target"),
        "horizons": matrix.get("horizons"),
        "rollout_modes": matrix.get("rollout_modes"),
        "policy_modes": matrix.get("policy_modes"),
    }
    if value.get("model_contract") != expected_model:
        raise IntegrityFailure("v7 model/endpoint/reasoning/science identity changed")
    if credential["openai_compatible_endpoint"] != expected_model["endpoint"]:
        raise IntegrityFailure("v7 credential endpoint differs from model endpoint")
    if value.get("run_id") != _v7_manifest_run_id(value):
        raise IntegrityFailure("v7 deterministic run ID mismatch")
    return value


@contextmanager
def _v7_frozen_policy_environment(model_id: str):
    names = set(_V7_POLICY_ENVIRONMENT) | {
        "SAFEPRED_POLICY_EXTRACTION_MODEL",
        "SAFEPRED_POLICY_EXTRACTION_SEED",
    }
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(_V7_POLICY_ENVIRONMENT)
        os.environ["SAFEPRED_POLICY_EXTRACTION_MODEL"] = model_id
        os.environ["SAFEPRED_POLICY_EXTRACTION_SEED"] = "0"
        yield
    finally:
        for name, prior in original.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior


def build_v7_task_descriptors(
    scientific_manifest: Mapping[str, Any],
    *,
    run_root: Path,
) -> tuple[V7TaskDescriptor, ...]:
    """Recompute deterministic unit paths without model or policy calls."""

    from . import predict_only_pipeline as pipeline

    root = _absolute(run_root)
    matrix = scientific_manifest["matrix"]
    model = scientific_manifest["model"]
    model_id = str(model["id"])
    endpoint = str(model["endpoint"])
    seed = int(model["seed"])
    base_config = pipeline._default_base_config().resolve()
    model_api = pipeline.ModelAPIConfig(
        provider="custom",
        api_key=None,
        api_url=endpoint,
        model=model_id,
        model_revision=str(model["requested_revision_label"]),
        policy_extraction_model=model_id,
    )
    descriptors: list[V7TaskDescriptor] = []
    with _v7_frozen_policy_environment(model_id):
        policy_signature = pipeline._ha_policy_inputs_signature()
        for source_index, source in enumerate(scientific_manifest["source_reports"]):
            task_id = str(source["task_id"])
            report_path = Path(str(source["report_path"])).resolve()
            report = _read_json_object(report_path)
            trace_path = Path(str(source["trace_path"])).resolve()
            sequences = tuple(int(value) for value in source["selected_sequence_nums"])
            point_signature = hashlib.sha1(
                ";".join(str(value) for value in sequences).encode("utf-8")
            ).hexdigest()[:8]
            actor = pipeline._source_report_actor_identity(
                report,
                legacy_fallback_seed=seed,
            )
            result_root = root / "results" / task_id
            trace_root = root / "traces" / task_id
            expectations: list[UnitExpectation] = []
            for unit in enumerate_replay_units(scientific_manifest):
                direct_horizons = (
                    [unit.horizon]
                    if unit.rollout_mode == "direct_multi_horizon"
                    else list(matrix["horizons"])
                )
                cache_signature = pipeline._ha_replay_cache_signature(
                    report_path=report_path,
                    trace_file=trace_path,
                    base_config=base_config,
                    model=model_id,
                    model_api=model_api,
                    policy_mode=unit.policy_mode,
                    rollout_mode=unit.rollout_mode,
                    prediction_steps=unit.horizon,
                    point_signature=point_signature,
                    candidate_count=int(matrix["candidate_count"]),
                    max_trace_events=int(matrix["max_trace_events"]),
                    direct_multi_horizons=direct_horizons,
                    formal_annotation_horizon=max(matrix["horizons"]),
                    policy_inputs_signature=policy_signature,
                    experiment_seed=seed,
                    actor_protocol_profile=actor.actor_protocol_profile,
                    actor_replicate_id=actor.actor_replicate_id,
                    actor_trace_seed=actor.actor_trace_seed,
                    wm_seed=seed,
                )
                run_id = (
                    f"po-{ROLLOUT_ALIAS[unit.rollout_mode]}-h{unit.horizon:02d}-{task_id}-"
                    f"{POLICY_ALIAS[unit.policy_mode]}-explicit-{point_signature}-{cache_signature}"
                )
                result_dir = result_root / ROLLOUT_ALIAS[unit.rollout_mode] / POLICY_ALIAS[unit.policy_mode] / "prediction_only" / task_id / f"h{unit.horizon:02d}"
                guard_dir = trace_root / "_g" / ROLLOUT_ALIAS[unit.rollout_mode] / POLICY_ALIAS[unit.policy_mode] / "prediction_only" / task_id / f"h{unit.horizon:02d}" / run_id
                expectations.append(
                    UnitExpectation(
                        unit=unit,
                        run_root=root,
                        task_id=task_id,
                        run_id=run_id,
                        result_dir=result_dir,
                        result_path=result_dir / f"{run_id}.json",
                        guard_run_dir=guard_dir,
                        decision_path=guard_dir / "01_guard_decisions.jsonl",
                        selected_point_ids=(1, 2),
                        selected_sequence_nums=sequences,
                        point_signature=point_signature,
                        cache_signature=cache_signature,
                        formal_annotation_horizon=max(matrix["horizons"]),
                        source_report_path=str(report_path),
                    )
                )
            diagnostic = root / "diagnostics" / task_id
            descriptors.append(
                V7TaskDescriptor(
                    task_id=task_id,
                    source_index=source_index,
                    history_path=(diagnostic / "task_history.jsonl").resolve(),
                    checkpoint_path=(diagnostic / "checkpoint.json").resolve(),
                    expectations=tuple(expectations),
                    cache_root=(diagnostic / "policy_cache").resolve(),
                    config_paths=(base_config,),
                )
            )
    return tuple(descriptors)


def _v7_lock_is_active(run_root: Path) -> bool:
    path = v7_batch_lock_path(run_root)
    if not path.exists():
        return False
    _safe_path(path, run_root=run_root, must_exist=True)
    fd = os.open(path, os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _v7_scan_credentials(run_root: Path) -> None:
    pattern = re.compile(rb"(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|authorization)\s*[:=]\s*[^\s\"']{12,})", re.I)
    for path in sorted(_absolute(run_root).rglob("*")):
        if path.is_dir():
            continue
        _assert_no_symlink_components(path)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityFailure("v7 root contains an unsafe special file")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if pattern.search(chunk):
                    raise IntegrityFailure("credential-looking bytes are persisted in v7 root")


def preflight_v7(
    manifest_path: Path,
    *,
    run_root: Path,
    project_root: Path | None = None,
    mode: Literal["launch", "resume", "aggregate"] = "launch",
    provider_probe: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Perform every offline integrity gate before the optional provider probe."""

    from . import ha_resilient_predict_only as legacy

    if mode not in {"launch", "resume", "aggregate"}:
        raise IntegrityFailure("v7 preflight mode is invalid")
    root = _safe_path(run_root, run_root=run_root)
    manifest_file = _safe_path(manifest_path, run_root=root, must_exist=True)
    manifest = validate_v7_execution_manifest(
        _read_json_object(manifest_file),
        run_root=root,
        project_root=project_root,
    )
    science_path = root / manifest["scientific_manifest"]["relative_path"]
    science = _read_json_object(science_path)
    offline = legacy._offline_preflight(science)
    if _v7_lock_is_active(root):
        raise IntegrityFailure("v7 shard has an active batch lock")
    _v7_scan_credentials(root)
    prediction_files = [
        path
        for directory_name in ("results", "traces")
        for path in (root / directory_name).rglob("*")
        if path.is_file()
    ]
    if mode == "launch":
        allowed = {"manifest.json", "scientific_manifest.json", "task_ids.txt"}
        unexpected = [
            path
            for path in root.iterdir()
            if path.name not in allowed and path.name != ".ha-v7-batch.lock"
        ]
        if unexpected or prediction_files:
            raise IntegrityFailure("launch requires a fresh registered v7 root")
    unit_plan = enumerate_replay_units(science)
    history_candidates = list((root / "diagnostics").glob("*/task_history.jsonl")) if (root / "diagnostics").is_dir() else []
    descriptors: tuple[V7TaskDescriptor, ...] = ()
    histories_exist = bool(history_candidates)
    validation: ValidatedV7Batch | None = None
    if histories_exist or mode == "aggregate":
        descriptors = build_v7_task_descriptors(science, run_root=root)
        validation = validate_batch_histories(
            run_root=root,
            manifest_sha256=manifest["manifest_sha256"],
            run_id=manifest["run_id"],
            tasks=descriptors,
            unit_plan=unit_plan,
        )
    progress_path = root / "progress.v7.json"
    if progress_path.exists():
        progress = _validate_progress_payload(_read_json_object(progress_path))
        if (
            progress["manifest_sha256"] != manifest["manifest_sha256"]
            or progress["run_id"] != manifest["run_id"]
            or progress["shard_id"] != manifest["shard"]["index"]
        ):
            raise IntegrityFailure("v7 progress identity differs from manifest")
    if mode == "aggregate":
        if validation is None or not _batch_ready_from_disk(
            run_root=root,
            progress_path=progress_path,
            manifest_sha256=manifest["manifest_sha256"],
            run_id=manifest["run_id"],
            shard_id=manifest["shard"]["index"],
            tasks=descriptors,
            unit_plan=unit_plan,
        ):
            raise IntegrityFailure("v7 aggregate preflight requires every task terminal and revalidated")
    if provider_probe is not None:
        result = provider_probe(manifest)
        if result is False:
            raise IntegrityFailure("v7 optional provider probe failed")
    return {
        "schema_version": 7,
        "artifact_type": "ha_v7_preflight_summary",
        "passed": True,
        "mode": mode,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_index": manifest["shard"]["index"],
        "requested_count": len(science["ordered_task_ids"]),
        "offline": offline,
        "provider_probe_enabled": provider_probe is not None,
    }


def _run_v7_registered_shard(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    root = _absolute(run_root)
    manifest = validate_v7_execution_manifest(
        _read_json_object(manifest_path),
        run_root=root,
        project_root=_v7_project_root(),
    )
    preflight_v7(
        manifest_path,
        run_root=root,
        project_root=_v7_project_root(),
        mode="resume",
    )
    science = _read_json_object(root / "scientific_manifest.json")
    tasks = build_v7_task_descriptors(science, run_root=root)
    units = enumerate_replay_units(science)

    def child(**kwargs: Any) -> ChildOutcome:
        return run_v7_child(
            scientific_manifest=science,
            manifest_sha256=manifest["manifest_sha256"],
            run_id=manifest["run_id"],
            run_root=root,
            index=kwargs["index"],
            attempt=kwargs["attempt"],
            history_path=kwargs["history_path"],
            verified_bindings=kwargs["verified_bindings"],
            expected_policy_fingerprint=kwargs["expected_policy_fingerprint"],
        )

    with _v7_frozen_policy_environment(str(science["model"]["id"])):
        return run_v7_batch(
            run_root=root,
            shard_id=manifest["shard"]["index"],
            manifest_sha256=manifest["manifest_sha256"],
            run_id=manifest["run_id"],
            tasks=tasks,
            unit_plan=units,
            child_callback=child,
            progress_path=root / "progress.v7.json",
            terminal_status_path=root / "terminal-status.v7.json",
        )


def _build_v7_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HarnessAudit v7 checkpoint runner")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--run-root", type=Path, required=True)
    preflight.add_argument("--mode", choices=("launch", "resume", "aggregate"), default="launch")
    resume = commands.add_parser("resume")
    resume.add_argument("--manifest", type=Path, required=True)
    resume.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_v7_argument_parser().parse_args(argv)
    if arguments.command == "preflight":
        summary = preflight_v7(
            arguments.manifest,
            run_root=arguments.run_root,
            project_root=_v7_project_root(),
            mode=arguments.mode,
        )
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    progress = _run_v7_registered_shard(arguments.manifest, arguments.run_root)
    print(json.dumps(progress, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = [
    "ValidatedV7Batch",
    "ValidatedV7Task",
    "V7TaskDescriptor",
    "ChildOutcome",
    "DanglingUnitReconciliation",
    "FailureClassification",
    "IntegrityFailure",
    "PartialArchive",
    "UnitBinding",
    "UnitDiscovery",
    "UnitExpectation",
    "UnitState",
    "archive_partial_unit",
    "capture_v6_protection_inventory",
    "batch_ready_for_aggregation",
    "build_terminal_status_projection",
    "classify_child_failure",
    "discover_complete_units",
    "enumerate_replay_units",
    "inspect_unit",
    "policy_cache_fingerprint_evidence_path",
    "prepare_policy_cache_fingerprint_evidence",
    "reconcile_dangling_unit",
    "run_v7_batch",
    "run_v7_child",
    "validate_and_fingerprint_train_policy_cache",
    "validate_failure_sidecar",
    "validate_batch_histories",
    "validate_child_outcome_for_task",
    "validate_descriptor_policy_fingerprint",
    "validate_policy_cache_fingerprint_binding",
    "validate_unit_binding",
    "write_failure_sidecar",
    "write_terminal_status_projection",
    "write_v7_progress",
    "build_v7_execution_manifest",
    "build_v7_task_descriptors",
    "preflight_v7",
    "reconstruct_v6_continuation_cohort",
    "validate_v7_execution_manifest",
    "v7_batch_lock_path",
    "v7_attempt_evidence_path",
    "v7_validation_gate_path",
]


if __name__ == "__main__":
    raise SystemExit(main())
