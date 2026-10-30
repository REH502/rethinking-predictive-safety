from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from multi_agent.trace.checkpoint import (
    STEP_AFTER_COMMIT,
    STEP_BEFORE_COMMIT,
    BankSnapshotBinding,
    CheckpointConflictError,
    CheckpointIdentityError,
    CheckpointLockError,
    CheckpointTerminalError,
    DurableTraceCheckpoint,
)


IDENTITY = {
    "actor": {"model": "fixture-actor", "revision": "actor-r1"},
    "condition_id": "task-1__enforced_guard__filler",
    "perturbation": "filler",
    "prompt_sha256": "1" * 64,
    "seed": 0,
    "task_sha256": "2" * 64,
}


def _open(
    path: Path,
    *,
    identity=IDENTITY,
    run_id: str = "run-fixed",
    attempt_id: str = "attempt-0001",
    busy_timeout_ms: int = 5_000,
    failure_injector=None,
) -> DurableTraceCheckpoint:
    return DurableTraceCheckpoint(
        path,
        identity=identity,
        run_id=run_id,
        attempt_id=attempt_id,
        busy_timeout_ms=busy_timeout_ms,
        failure_injector=failure_injector,
    )


def _commit_step(
    checkpoint: DurableTraceCheckpoint,
    *,
    expected_previous_step_id: int = 0,
    action_id: str = "action-0001",
    tool_name: str = "lookup_order",
):
    return checkpoint.commit_step(
        expected_previous_step_id=expected_previous_step_id,
        step_kind="tool_result",
        action_id=action_id,
        action={"arguments": {"order_id": "fixture-order"}, "tool": tool_name},
        sdk_session_items=[
            {"role": "assistant", "type": "tool_call"},
            {"role": "tool", "type": "tool_result"},
        ],
        bank_snapshot=BankSnapshotBinding(
            relative_path="bank/step-0001.sqlite",
            sha256="a" * 64,
            size_bytes=4096,
        ),
        guard_state={"decision_counter": 1, "pending_consultations": []},
        usage={"input_tokens": 17, "output_tokens": 5, "requests": 1},
    )


def test_restart_restores_all_bound_state_and_full_durability(tmp_path):
    path = tmp_path / "checkpoint" / "trace.sqlite"
    with _open(path) as checkpoint:
        settings = checkpoint.durability_settings()
        assert settings["journal_mode"] == "wal"
        assert settings["synchronous"] == 2  # SQLite FULL
        assert settings["foreign_keys"] == 1
        committed = _commit_step(checkpoint)
        assert committed.step_id == 1
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if candidate.exists():
                assert stat.S_IMODE(candidate.stat().st_mode) == 0o600

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with _open(path) as restarted:
        snapshot = restarted.load_for_resume()

    assert snapshot.run_id == "run-fixed"
    assert snapshot.attempt_id == "attempt-0001"
    assert snapshot.identity == IDENTITY
    assert snapshot.last_step_id == 1
    assert snapshot.steps[0].action["tool"] == "lookup_order"
    assert snapshot.steps[0].bank_snapshot == BankSnapshotBinding(
        relative_path="bank/step-0001.sqlite",
        sha256="a" * 64,
        size_bytes=4096,
    )
    assert snapshot.steps[0].guard_state["decision_counter"] == 1
    assert snapshot.steps[0].usage["input_tokens"] == 17
    assert len(snapshot.sdk_session_items) == 2


def test_failure_before_commit_leaves_no_visible_partial_step(tmp_path):
    path = tmp_path / "trace.sqlite"

    def fail_before_commit(phase: str) -> None:
        if phase == STEP_BEFORE_COMMIT:
            raise RuntimeError("injected pre-commit crash")

    checkpoint = _open(path, failure_injector=fail_before_commit)
    with pytest.raises(RuntimeError, match="pre-commit"):
        _commit_step(checkpoint)
    checkpoint.close()

    with _open(path) as restarted:
        snapshot = restarted.load_for_resume()
    assert snapshot.last_step_id == 0
    assert snapshot.steps == ()
    assert snapshot.sdk_session_items == ()


def test_failure_after_commit_is_visible_and_retry_is_idempotent(tmp_path):
    path = tmp_path / "trace.sqlite"

    def fail_after_commit(phase: str) -> None:
        if phase == STEP_AFTER_COMMIT:
            raise RuntimeError("injected post-commit crash")

    checkpoint = _open(path, failure_injector=fail_after_commit)
    with pytest.raises(RuntimeError, match="post-commit"):
        _commit_step(checkpoint)
    checkpoint.close()

    with _open(path) as restarted:
        durable = restarted.load_for_resume()
        retried = _commit_step(restarted, expected_previous_step_id=0)
        after_retry = restarted.load_for_resume()

    assert durable.last_step_id == 1
    assert retried.step_id == 1
    assert after_retry.last_step_id == 1
    assert [step.action_id for step in after_retry.steps] == ["action-0001"]


def test_duplicate_action_id_with_different_payload_fails_closed(tmp_path):
    with _open(tmp_path / "trace.sqlite") as checkpoint:
        _commit_step(checkpoint)
        with pytest.raises(CheckpointConflictError, match="different payload"):
            _commit_step(checkpoint, tool_name="delete_order")


def test_monotonic_commit_rejects_stale_predecessor(tmp_path):
    with _open(tmp_path / "trace.sqlite") as checkpoint:
        _commit_step(checkpoint)
        with pytest.raises(CheckpointConflictError, match="stale step predecessor"):
            _commit_step(
                checkpoint,
                expected_previous_step_id=0,
                action_id="action-0002",
            )


@pytest.mark.parametrize(
    ("identity", "run_id", "attempt_id", "message"),
    [
        ({**IDENTITY, "seed": 1}, "run-fixed", "attempt-0001", "identity hash"),
        (IDENTITY, "run-drifted", "attempt-0001", "run_id"),
        (IDENTITY, "run-fixed", "attempt-0002", "attempt_id"),
    ],
)
def test_restart_rejects_identity_run_or_attempt_drift(
    tmp_path,
    identity,
    run_id,
    attempt_id,
    message,
):
    path = tmp_path / "trace.sqlite"
    with _open(path):
        pass

    with pytest.raises(CheckpointIdentityError, match=message):
        _open(path, identity=identity, run_id=run_id, attempt_id=attempt_id)


def test_terminal_state_is_restartable_for_read_but_refuses_resume(tmp_path):
    path = tmp_path / "trace.sqlite"
    with _open(path) as checkpoint:
        _commit_step(checkpoint)
        terminal = checkpoint.mark_terminal(
            expected_last_step_id=1,
            status="passed",
            state={"report_sha256": "b" * 64},
        )
        assert terminal.final_step_id == 1
        same_terminal = checkpoint.mark_terminal(
            expected_last_step_id=1,
            status="passed",
            state={"report_sha256": "b" * 64},
        )
        assert same_terminal.terminal_sha256 == terminal.terminal_sha256
        with pytest.raises(CheckpointTerminalError, match="cannot be extended"):
            _commit_step(
                checkpoint,
                expected_previous_step_id=1,
                action_id="action-0002",
            )

    with _open(path) as restarted:
        snapshot = restarted.load()
        assert snapshot.terminal is not None
        assert snapshot.terminal.status == "passed"
        with pytest.raises(CheckpointTerminalError, match="cannot resume"):
            restarted.load_for_resume()


def test_concurrent_writer_lock_times_out_without_partial_state(tmp_path):
    path = tmp_path / "trace.sqlite"
    writer_has_lock = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def hold_before_commit(phase: str) -> None:
        if phase == STEP_BEFORE_COMMIT:
            writer_has_lock.set()
            if not release_writer.wait(timeout=5):
                raise RuntimeError("test failed to release writer")

    first = _open(path, failure_injector=hold_before_commit)
    second = _open(path, busy_timeout_ms=25)

    def write_first() -> None:
        try:
            _commit_step(first)
        except BaseException as exc:  # pragma: no cover - only aids thread diagnostics
            writer_errors.append(exc)

    thread = threading.Thread(target=write_first, daemon=True)
    thread.start()
    assert writer_has_lock.wait(timeout=5)
    try:
        with pytest.raises(CheckpointLockError, match="lock timed out"):
            _commit_step(
                second,
                expected_previous_step_id=0,
                action_id="action-0002",
            )
    finally:
        release_writer.set()
        thread.join(timeout=5)
        first.close()
        second.close()

    assert not thread.is_alive()
    assert writer_errors == []
    with _open(path) as restarted:
        snapshot = restarted.load_for_resume()
    assert snapshot.last_step_id == 1
    assert [step.action_id for step in snapshot.steps] == ["action-0001"]
