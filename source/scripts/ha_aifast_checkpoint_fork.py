"""Fork a verified HA trace prefix into a NEW recovery runtime root.

This does not call models, change native HA code, repair source evidence, or
reinterpret a failed terminal as successful. The caller must register the new
recovery lineage and preserve the original condition/model/CLI identity.
``judge_only`` retains ACTOR_COMPLETE, so native --trace-resume reuses the
actor outcome. ``wm`` retains the last COMPLETE SDK turn before the earliest
ledger-proven AIFast transport failure; this is not exact-request replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness_env/orgin/HarnessAudit"))
from multi_agent.trace.checkpoint import DurableTraceCheckpoint
from multi_agent.trace.runtime_checkpoint import RuntimeCheckpointCoordinator


class UnsupportedRecovery(ValueError):
    """Available evidence does not justify the requested recovery boundary."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("artifact path is not safely relative")
    value = root.joinpath(*pure.parts)
    if value.is_symlink() or not value.resolve().is_relative_to(root):
        raise ValueError("artifact symlink or root escape")
    # Reject internal symlink directories too, even when they resolve in-root.
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact path traverses a symlink")
    return value


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact missing or not a regular file")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source artifact changed during read")
    return data


def _is_aifast_transport_error(row: dict) -> bool:
    if row.get("status") != "error" or row.get("component") != "world_model":
        return False
    if urlparse(str(row.get("endpoint", ""))).hostname != "aifast.site":
        return False
    kind = str(row.get("exception_type", ""))
    if kind in {"SSLError", "SSLEOFError", "ProxyError", "ConnectionError", "ConnectError", "ReadError", "RemoteProtocolError", "Timeout", "ReadTimeout", "ConnectTimeout", "APITimeoutError", "APIConnectionError"}:
        return True
    status = row.get("http_status", row.get("status_code"))
    return isinstance(status, int) and (status in {408, 429} or 500 <= status < 600)


def _transport_failure(ledger: Path) -> tuple[int, dict]:
    data = _read(ledger)
    offset = 0
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise UnsupportedRecovery("WM ledger has an incomplete final row")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise UnsupportedRecovery("WM ledger row is malformed") from exc
        if not isinstance(row, dict):
            raise UnsupportedRecovery("WM ledger row is not an object")
        if row.get("status") == "error":
            if not _is_aifast_transport_error(row):
                raise UnsupportedRecovery("earlier WM failure lacks AIFast transport attribution")
            return offset, {"line_number": number, "byte_offset": offset, "ledger_sha256": _sha(data), "ledger_size_bytes": len(data), "exception_type": row.get("exception_type"), "transport_attempt_count": row.get("transport_attempt_count")}
        offset += len(line)
    raise UnsupportedRecovery("no ledger-proven AIFast WM transport failure")


def _verified_snapshot(source_checkpoint: Path, temporary_root: Path):
    """Validate a stable DB/WAL byte copy, never opening source in SQLite.

    Even SQLite mode=ro may create source WAL/SHM sidecars. Read ordinary
    files twice for stability, then let SQLite open only the temporary copy.
    """
    backup_path = temporary_root / "source.sqlite"
    capture = {}
    for suffix in ("", "-wal"):
        source_path = Path(str(source_checkpoint) + suffix)
        capture[suffix] = _read(source_path) if source_path.exists() else None
    for suffix, data in capture.items():
        source_path = Path(str(source_checkpoint) + suffix)
        if (source_path.exists() != (data is not None)) or (data is not None and _read(source_path) != data):
            raise ValueError("source checkpoint changed during snapshot")
        if data is not None:
            Path(str(backup_path) + suffix).write_bytes(data)
    source_digest = _sha(capture[""])
    with sqlite3.connect(backup_path) as backup:
        backup.row_factory = sqlite3.Row
        metadata = dict(backup.execute("SELECT * FROM checkpoint_metadata").fetchone())
        rows = [dict(row) for row in backup.execute("SELECT * FROM committed_steps ORDER BY step_id")]
    checkpoint = DurableTraceCheckpoint(backup_path, identity=json.loads(metadata["identity_json"]), run_id=metadata["run_id"], attempt_id=metadata["attempt_id"])
    try:
        snapshot = checkpoint.load()
        coordinator = RuntimeCheckpointCoordinator(temporary_root, checkpoint)
        coordinator.inspect()
    finally:
        checkpoint.close()
    return snapshot, rows, source_digest


def _plan(source: Path, checkpoint_path: Path, temporary_root: Path, phase: str, ledger: Path | None):
    snapshot, raw_rows, source_digest = _verified_snapshot(checkpoint_path, temporary_root)
    failure = None
    if phase == "judge_only":
        actor = [step for step in snapshot.steps if step.step_kind == "actor_complete"]
        if len(actor) != 1:
            raise UnsupportedRecovery("judge-only recovery requires exactly one ACTOR_COMPLETE checkpoint")
        actor_state = actor[0].action["phase_state"]
        if actor_state.get("error") is not None or actor_state.get("schema_version") != 1:
            raise UnsupportedRecovery("judge-only requires a normally completed actor, not actor failure or safety abstention")
        keep = actor[0].step_id
    elif phase == "wm":
        if ledger is None:
            raise UnsupportedRecovery("WM recovery requires the exact condition request ledger")
        ledger = ledger.resolve()
        if not ledger.is_relative_to(source):
            raise ValueError("WM ledger outside source runtime root")
        relative = ledger.relative_to(source).as_posix()
        failure_start, failure = _transport_failure(ledger)
        failure["ledger_relative_path"] = relative
        keep = 0
        bound = False
        previous = 0
        for step in snapshot.steps:
            bindings = step.action.get("ledger_offsets", {})
            request = bindings.get("request")
            if request is not None:
                if request["relative_path"] != relative:
                    raise UnsupportedRecovery("WM failure ledger does not match checkpoint binding")
                bound = True
                offset = request["byte_offset"]
                if offset < previous:
                    raise ValueError("non-monotonic request ledger binding")
                data = _read(ledger)[:offset]
                if len(data) != offset or _sha(data) != request["prefix_sha256"]:
                    raise ValueError("WM ledger prefix hash drift")
                previous = offset
            else:
                offset = 0
            if step.step_kind in {"initial_state", "model_turn"} and offset <= failure_start:
                keep = step.step_id
            elif offset > failure_start:
                break
        # Initial-state-only failures have no ledger binding yet. Exact ledger
        # ownership then cannot be proven by this utility, so do not guess.
        if not keep or not bound:
            raise UnsupportedRecovery("no checkpoint-bound pre-failure SDK turn")
    else:
        raise ValueError("phase must be judge_only or wm")
    selected = snapshot.steps[:keep]
    prefix = replace(snapshot, steps=selected, sdk_session_items=tuple(item for step in selected for item in step.sdk_session_items), terminal=None)
    # Only invokes the model-free structural parser on the new prefix.
    parser = object.__new__(RuntimeCheckpointCoordinator)
    recovered = parser._parse_snapshot(prefix)
    blobs: dict[str, bytes] = {}

    def artifact(binding):
        if binding is None:
            return
        if set(binding) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError("artifact binding schema drift")
        relative = binding["relative_path"]
        data = _read(_path(source, relative))
        if len(data) != binding["size_bytes"] or _sha(data) != binding["sha256"]:
            raise ValueError("bound artifact hash/size drift")
        if relative in blobs and blobs[relative] != data:
            raise ValueError("artifact collision")
        blobs[relative] = data
        if relative.startswith("workspace_manifests/"):
            manifest = json.loads(data)
            for entry in manifest["entries"]:
                if entry.get("type") == "file":
                    artifact(entry["blob"])

    for step in selected:
        artifact(step.bank_snapshot.as_json_value() if step.bank_snapshot else None)
        artifact(step.action["trace_snapshot"])
        artifact(step.action["workspace_snapshot"])
        artifact(step.action["phase_state"].get("safety_abstention_artifact"))
    # Native judge-only resume bypasses guard initialization. Keep the exact
    # condition's frozen policy/strategy/runtime sidecars for output validation.
    # Do not copy stale reports, decisions beyond the boundary, or raw logs.
    guard_binding = recovered.ledger_offsets.get("guard_decisions")
    if guard_binding is not None:
        guard_parent = _path(source, guard_binding.relative_path).parent
        names = ["00_safepred_policies.raw.json", "00_guard_strategy.raw.json"]
        native_h1 = (recovered.guard_state or {}).get("schema_version") == "environment-h1-preview-review-commit-v1"
        if not native_h1:
            names.insert(1, "00_safepred_runtime_config.yaml")
        elif (guard_parent / "00_safepred_runtime_config.yaml").exists():
            raise ValueError("unexpected learned-WM config for environment h1")
        for name in names:
            sidecar = guard_parent / name
            if not sidecar.is_file():
                raise ValueError("required frozen guard metadata sidecar missing")
            relative = sidecar.relative_to(source).as_posix()
            blobs[relative] = _read(_path(source, relative))
    ledger_manifest = {}
    for name, binding in recovered.ledger_offsets.items():
        data = _read(_path(source, binding.relative_path))[:binding.byte_offset]
        if len(data) != binding.byte_offset or _sha(data) != binding.prefix_sha256:
            raise ValueError("committed ledger prefix hash drift")
        blobs[binding.relative_path] = data
        ledger_manifest[name] = binding.as_json_value()
    return snapshot, raw_rows[:keep], recovered, blobs, {
        "schema_version": "ha-aifast-checkpoint-prefix-fork-v1",
        "classification": "diagnostic-only",
        "source_runtime_root": str(source),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": source_digest,
        "source_checkpoint_last_state_sha256": snapshot.last_state_sha256,
        "source_terminal_sha256": snapshot.terminal.terminal_sha256 if snapshot.terminal else None,
        "identity_sha256": snapshot.identity_sha256,
        "run_id": snapshot.run_id,
        "attempt_id": snapshot.attempt_id,
        "phase": phase,
        "source_steps": len(snapshot.steps),
        "retained_steps": keep,
        "model_turn_count": recovered.model_turn_count,
        "actor_complete": recovered.actor_complete,
        "prefix_state_sha256": prefix.last_state_sha256,
        "normalized_action_count": len(recovered.normalized_actions),
        "trace_snapshot": recovered.trace_snapshot.as_json_value() if recovered.trace_snapshot else None,
        "ledger_prefixes": ledger_manifest,
        "failure_evidence": failure,
        "recovery_boundary": "actor_complete_evaluation_only" if phase == "judge_only" else "complete_sdk_turn_before_first_wm_transport_failure",
        "exact_request_replay": False,
        "actor_calls_expected": phase != "judge_only",
        "world_model_calls_expected": phase != "judge_only",
    }


def plan_checkpoint_fork(source_runtime_root: Path, source_checkpoint_path: Path, *, phase: str = "judge_only", wm_failure_ledger: Path | None = None) -> dict:
    """Return a verified recovery plan without creating experiment state."""
    source = Path(source_runtime_root).resolve(strict=True)
    checkpoint = Path(source_checkpoint_path).resolve(strict=True)
    if not checkpoint.is_relative_to(source):
        raise ValueError("checkpoint outside source runtime root")
    with tempfile.TemporaryDirectory(prefix="ha-checkpoint-plan-") as temp:
        return _plan(source, checkpoint, Path(temp), phase, wm_failure_ledger)[-1]


def fork_checkpoint(source_runtime_root: Path, source_checkpoint_path: Path, destination_runtime_root: Path, *, phase: str = "judge_only", wm_failure_ledger: Path | None = None) -> dict:
    """Copy a validated immutable prefix; refuse any pre-existing destination.

    Runtime roots are the original CLI's --trace-runtime-root (normally
    ``run/harnessaudit``), not its ``traces`` child. The relative artifact and
    checkpoint paths are preserved. Do not change original scientific flags
    when launching native CLI --trace-resume against the destination.
    """
    source = Path(source_runtime_root).resolve(strict=True)
    checkpoint = Path(source_checkpoint_path).resolve(strict=True)
    destination = Path(destination_runtime_root).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("recovery destination already exists")
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("recovery and source roots must be disjoint")
    relative_checkpoint = checkpoint.relative_to(source)
    _path(source, relative_checkpoint.as_posix())
    with tempfile.TemporaryDirectory(prefix="ha-checkpoint-fork-") as temp:
        snapshot, rows, recovered, blobs, manifest = _plan(source, checkpoint, Path(temp), phase, wm_failure_ledger)
        # All source/state validation is complete BEFORE creating the new root.
        destination.mkdir(parents=True, exist_ok=False)
        for relative, data in blobs.items():
            path = _path(destination, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        target = destination / relative_checkpoint
        copied = DurableTraceCheckpoint(target, identity=snapshot.identity, run_id=snapshot.run_id, attempt_id=snapshot.attempt_id)
        try:
            # Insert unmodified verified rows into a newly initialized store:
            # original timestamps AND every payload/state hash remain exact.
            conn = copied._connection()
            conn.execute("BEGIN IMMEDIATE")
            for row in rows:
                columns = tuple(row)
                conn.execute("INSERT INTO committed_steps (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row.values()))
            conn.execute("UPDATE checkpoint_metadata SET last_step_id=? WHERE singleton=1", (len(rows),))
            conn.commit()
            cloned = copied.load_for_resume()
            restored = RuntimeCheckpointCoordinator(destination, copied).inspect()
            if cloned.last_state_sha256 != manifest["prefix_state_sha256"] or restored != recovered:
                raise ValueError("fork validation does not equal exact source prefix")
        finally:
            copied.close()
        manifest.update({"destination_runtime_root": str(destination), "destination_checkpoint_path": str(target), "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_bindings": [{"relative_path": rel, "sha256": _sha(data), "size_bytes": len(data)} for rel, data in sorted(blobs.items())]})
        lineage = destination / "checkpoint_fork_lineage.json"
        with lineage.open("x") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest["lineage_path"] = str(lineage)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runtime-root", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--destination-runtime-root", type=Path)
    parser.add_argument("--phase", choices=("judge_only", "wm"), default="judge_only")
    parser.add_argument("--wm-failure-ledger", type=Path)
    args = parser.parse_args()
    if args.destination_runtime_root is None:
        result = plan_checkpoint_fork(args.source_runtime_root, args.source_checkpoint, phase=args.phase, wm_failure_ledger=args.wm_failure_ledger)
    else:
        result = fork_checkpoint(args.source_runtime_root, args.source_checkpoint, args.destination_runtime_root, phase=args.phase, wm_failure_ledger=args.wm_failure_ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()



