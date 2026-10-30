from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from multi_agent.trace.checkpoint import (
    STEP_AFTER_COMMIT,
    STEP_BEFORE_COMMIT,
    CheckpointTerminalError,
    DurableTraceCheckpoint,
)
from multi_agent.trace.runtime_checkpoint import (
    DurableCheckpointSession,
    RuntimeCheckpointCoordinator,
    RuntimeCheckpointDriftError,
    RuntimeCommitState,
    RuntimeSessionDirtyError,
)


IDENTITY = {
    "condition_id": "fixture-task__distributed_enforced__filler",
    "model_revision": "fixture-r1",
    "perturbation": "filler",
    "seed": 0,
}
TRACE_START = b'{"type":"trace_start"}\n'


class FakeBank:
    def __init__(self, value: str) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.execute("INSERT INTO state (id, value) VALUES (1, ?)", (value,))
        self.conn.commit()
        self._order_counter = 10
        self._active_request_id: str | None = None

    @property
    def value(self) -> str:
        return str(self.conn.execute("SELECT value FROM state WHERE id = 1").fetchone()[0])

    @value.setter
    def value(self, value: str) -> None:
        self.conn.execute("UPDATE state SET value = ? WHERE id = 1", (value,))
        self.conn.commit()

    def dump_to_file(self, path: Path) -> None:
        destination = sqlite3.connect(path)
        try:
            self.conn.backup(destination)
        finally:
            destination.close()

    def close(self) -> None:
        self.conn.close()


class FakeSession:
    def __init__(self, items=None) -> None:
        self.items = list(items or [])
        self.clear_count = 0
        self.add_count = 0

    async def clear_session(self) -> None:
        self.clear_count += 1
        self.items.clear()

    async def add_items(self, items) -> None:
        self.add_count += 1
        self.items.extend(items)


def _open_checkpoint(root: Path, failure_injector=None) -> DurableTraceCheckpoint:
    return DurableTraceCheckpoint(
        root / "checkpoint.sqlite",
        identity=IDENTITY,
        run_id="run-fixed",
        attempt_id="attempt-0001",
        failure_injector=failure_injector,
    )


def _state(
    *,
    ledger: Path,
    trace_bytes: bytes,
    action_count: int,
) -> RuntimeCommitState:
    return RuntimeCommitState(
        normalized_actions=[
            {"sequence_num": index, "tool": f"tool_{index}"}
            for index in range(1, action_count + 1)
        ],
        guard_state={"decision_counter": action_count},
        hook_state={"pending_hooks": []},
        counts_state={"model_turns": action_count, "tool_calls": action_count},
        usage={"input_tokens": action_count * 10, "output_tokens": action_count * 2},
        ledger_paths={"wm_requests": ledger},
        trace_bytes=trace_bytes,
    )


def _baseline(coordinator, *, bank, trace_path, workspace=None):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not trace_path.exists():
        trace_path.write_bytes(TRACE_START)
    return coordinator.checkpoint_initial_state(
        bank=bank,
        state=RuntimeCommitState(
            normalized_actions=(),
            guard_state={},
            hook_state={},
            counts_state={},
            usage={},
            ledger_paths={},
            trace_bytes=trace_path.read_bytes(),
            workspace=workspace,
        ),
        run_started_at_utc="2026-09-03T00:00:00Z",
    )


def _prepare(wrapper, *, bank, trace_path):
    return asyncio.run(wrapper.prepare(bank=bank, trace_path=trace_path))


def _add(wrapper, items, *, turn_id, bank, state):
    return asyncio.run(
        wrapper.add_items(items, turn_id=turn_id, bank=bank, state=state)
    )


def test_restart_restores_bank_actions_session_and_archives_uncommitted_suffixes(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "ledgers" / "wm.jsonl"
    ledger.parent.mkdir()
    ledger.write_bytes(b'{"request":1}\n')
    trace = root / "trace" / "trace.jsonl"
    bank = FakeBank("committed-one")
    checkpoint = _open_checkpoint(root)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    session = FakeSession()
    wrapper = DurableCheckpointSession(session, coordinator)
    _prepare(wrapper, bank=bank, trace_path=trace)

    trace_one = b'{"type":"trace_start"}\n{"turn":1}\n'
    trace.parent.mkdir(exist_ok=True)
    trace.write_bytes(trace_one)
    _add(
        wrapper,
        [{"role": "assistant", "content": "turn one"}],
        turn_id="0001",
        bank=bank,
        state=_state(ledger=ledger, trace_bytes=trace_one, action_count=1),
    )

    bank.value = "uncommitted-two"
    ledger.write_bytes(ledger.read_bytes() + b'{"request":2}\n')
    uncommitted_trace = trace_one + b'{"turn":2,"uncommitted":true}\n'
    trace.write_bytes(uncommitted_trace)
    session.items.append({"role": "assistant", "content": "uncommitted"})
    checkpoint.close()
    bank.close()

    recovered_bank = FakeBank("wrong-seed")
    recovered_session = FakeSession([{"role": "assistant", "content": "dirty"}])
    restarted_checkpoint = _open_checkpoint(root)
    restarted = RuntimeCheckpointCoordinator(root, restarted_checkpoint)
    restarted_wrapper = DurableCheckpointSession(recovered_session, restarted)
    recovery = _prepare(restarted_wrapper, bank=recovered_bank, trace_path=trace)

    assert recovered_bank.value == "committed-one"
    assert recovery.normalized_actions == ({"sequence_num": 1, "tool": "tool_1"},)
    assert recovered_session.items == [{"content": "turn one", "role": "assistant"}]
    assert recovered_session.clear_count == 1
    assert ledger.read_bytes() == b'{"request":1}\n'
    assert trace.read_bytes() == trace_one
    orphan_bytes = [
        (root / binding.relative_path).read_bytes()
        for binding in recovery.recovery_orphans
    ]
    assert b'{"request":2}\n' in orphan_bytes
    assert uncommitted_trace in orphan_bytes

    restarted_checkpoint.close()
    recovered_bank.close()


def test_precommit_failure_dirties_session_and_restart_discards_second_turn(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "wm.jsonl"
    ledger.write_bytes(b"one\n")
    trace = root / "trace.jsonl"
    fail_second = {"enabled": False}

    def injector(phase: str) -> None:
        if fail_second["enabled"] and phase == STEP_BEFORE_COMMIT:
            raise RuntimeError("injected before commit")

    bank = FakeBank("one")
    checkpoint = _open_checkpoint(root, failure_injector=injector)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    session = FakeSession()
    wrapper = DurableCheckpointSession(session, coordinator)
    _prepare(wrapper, bank=bank, trace_path=trace)
    trace_one = TRACE_START + b"turn-one\n"
    trace.write_bytes(trace_one)
    _add(
        wrapper,
        [{"turn": 1}],
        turn_id="0001",
        bank=bank,
        state=_state(ledger=ledger, trace_bytes=trace_one, action_count=1),
    )

    fail_second["enabled"] = True
    bank.value = "two"
    ledger.write_bytes(b"one\ntwo\n")
    trace_two = TRACE_START + b"turn-one\nturn-two\n"
    trace.write_bytes(trace_two)
    with pytest.raises(RuntimeError, match="before commit"):
        _add(
            wrapper,
            [{"turn": 2}],
            turn_id="0002",
            bank=bank,
            state=_state(ledger=ledger, trace_bytes=trace_two, action_count=2),
        )
    assert session.items == [{"turn": 1}, {"turn": 2}]
    with pytest.raises(RuntimeSessionDirtyError, match="must be prepared"):
        _add(
            wrapper,
            [{"turn": 3}],
            turn_id="0003",
            bank=bank,
            state=_state(ledger=ledger, trace_bytes=trace_two, action_count=3),
        )
    checkpoint.close()
    bank.close()

    recovered_bank = FakeBank("wrong")
    recovered_session = FakeSession([{"dirty": True}])
    restarted_checkpoint = _open_checkpoint(root)
    restarted = RuntimeCheckpointCoordinator(root, restarted_checkpoint)
    state = _prepare(
        DurableCheckpointSession(recovered_session, restarted),
        bank=recovered_bank,
        trace_path=trace,
    )
    assert state.last_step_id == 2
    assert recovered_bank.value == "one"
    assert recovered_session.items == [{"turn": 1}]
    assert ledger.read_bytes() == b"one\n"
    assert trace.read_bytes() == trace_one

    restarted_checkpoint.close()
    recovered_bank.close()


def test_postcommit_failure_rebuilds_then_retries_without_duplicate_session_item(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "wm.jsonl"
    ledger.write_bytes(b"one\n")
    trace = root / "trace.jsonl"
    fail_after = {"enabled": False}

    def injector(phase: str) -> None:
        if fail_after["enabled"] and phase == STEP_AFTER_COMMIT:
            raise RuntimeError("injected after commit")

    bank = FakeBank("one")
    checkpoint = _open_checkpoint(root, failure_injector=injector)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    session = FakeSession()
    wrapper = DurableCheckpointSession(session, coordinator)
    _prepare(wrapper, bank=bank, trace_path=trace)
    trace_one = TRACE_START + b"turn-one\n"
    trace.write_bytes(trace_one)
    _add(
        wrapper,
        [{"turn": 1}],
        turn_id="0001",
        bank=bank,
        state=_state(ledger=ledger, trace_bytes=trace_one, action_count=1),
    )

    fail_after["enabled"] = True
    bank.value = "two"
    ledger.write_bytes(b"one\ntwo\n")
    trace_two = TRACE_START + b"turn-one\nturn-two\n"
    trace.write_bytes(trace_two)
    second_state = _state(ledger=ledger, trace_bytes=trace_two, action_count=2)
    with pytest.raises(RuntimeError, match="after commit"):
        _add(
            wrapper,
            [{"turn": 2}],
            turn_id="0002",
            bank=bank,
            state=second_state,
        )
    checkpoint.close()
    bank.close()

    recovered_bank = FakeBank("wrong")
    recovered_session = FakeSession([{"dirty": True}])
    restarted_checkpoint = _open_checkpoint(root)
    restarted = RuntimeCheckpointCoordinator(root, restarted_checkpoint)
    restarted_wrapper = DurableCheckpointSession(recovered_session, restarted)
    recovered = _prepare(restarted_wrapper, bank=recovered_bank, trace_path=trace)
    assert recovered.last_step_id == 3
    assert recovered_session.items == [{"turn": 1}, {"turn": 2}]
    add_count_before_retry = recovered_session.add_count

    retried = _add(
        restarted_wrapper,
        [{"turn": 2}],
        turn_id="0002",
        bank=recovered_bank,
        state=second_state,
    )
    assert retried.step_id == 3
    assert recovered_session.add_count == add_count_before_retry
    assert recovered_session.items == [{"turn": 1}, {"turn": 2}]

    restarted_checkpoint.close()
    recovered_bank.close()


@pytest.mark.parametrize("failure", ["missing", "drift"])
def test_recovery_fails_closed_when_ledger_prefix_is_missing_or_drifted(tmp_path, failure):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "wm.jsonl"
    ledger.write_bytes(b"committed\n")
    trace = root / "trace.jsonl"
    trace.write_bytes(TRACE_START)
    bank = FakeBank("one")
    checkpoint = _open_checkpoint(root)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    coordinator.commit_model_turn(
        turn_id="0001",
        sdk_session_items=[{"turn": 1}],
        bank=bank,
        state=_state(ledger=ledger, trace_bytes=TRACE_START, action_count=1),
    )
    checkpoint.close()
    bank.close()

    if failure == "missing":
        ledger.unlink()
    else:
        ledger.write_bytes(b"drifted!!\n")
    restarted_bank = FakeBank("wrong")
    restarted_checkpoint = _open_checkpoint(root)
    restarted = RuntimeCheckpointCoordinator(root, restarted_checkpoint)
    with pytest.raises(RuntimeCheckpointDriftError, match="ledger|bound artifact"):
        restarted.recover(bank=restarted_bank, trace_path=trace)
    restarted_checkpoint.close()
    restarted_bank.close()


def test_bank_snapshot_drift_fails_closed(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "wm.jsonl"
    ledger.write_bytes(b"committed\n")
    trace = root / "trace.jsonl"
    trace.write_bytes(TRACE_START)
    bank = FakeBank("one")
    checkpoint = _open_checkpoint(root)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    coordinator.commit_model_turn(
        turn_id="0001",
        sdk_session_items=[{"turn": 1}],
        bank=bank,
        state=_state(ledger=ledger, trace_bytes=TRACE_START, action_count=1),
    )
    binding = coordinator.inspect().bank_snapshot
    assert binding is not None
    snapshot_path = root / binding.relative_path
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeCheckpointDriftError, match="bank snapshot size/hash"):
        coordinator.recover(bank=bank, trace_path=trace)
    checkpoint.close()
    bank.close()


def test_actor_evaluation_and_terminal_report_are_idempotent_and_hash_bound(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "wm.jsonl"
    ledger.write_bytes(b"committed\n")
    trace = root / "trace.jsonl"
    trace.write_bytes(TRACE_START)
    report = root / "report.json"
    report.write_bytes(b'{"score":1}\n')
    bank = FakeBank("one")
    checkpoint = _open_checkpoint(root)
    coordinator = RuntimeCheckpointCoordinator(root, checkpoint)
    _baseline(coordinator, bank=bank, trace_path=trace)
    state = _state(ledger=ledger, trace_bytes=TRACE_START, action_count=1)
    coordinator.commit_model_turn(
        turn_id="0001",
        sdk_session_items=[{"turn": 1}],
        bank=bank,
        state=state,
    )
    actor = coordinator.checkpoint_actor_complete(
        bank=bank,
        state=state,
        actor_state={
            "schema_version": 1,
            "framework": "oai",
            "final_output": "done",
            "error": None,
        },
    )
    assert coordinator.checkpoint_actor_complete(
        bank=bank,
        state=state,
        actor_state={
            "schema_version": 1,
            "framework": "oai",
            "final_output": "done",
            "error": None,
        },
    ).step_id == actor.step_id
    evaluation = coordinator.checkpoint_evaluation_complete(
        bank=bank,
        state=state,
        evaluation_state={
            "schema_version": 1,
            "run_id": "run-fixed",
            "task_id": "fixture-task",
            "report_role": "harnessaudit_evaluation_report_v1",
            "metric_summary": {"score": 1},
        },
        report_path=report,
    )
    assert coordinator.checkpoint_evaluation_complete(
        bank=bank,
        state=state,
        evaluation_state={
            "schema_version": 1,
            "run_id": "run-fixed",
            "task_id": "fixture-task",
            "report_role": "harnessaudit_evaluation_report_v1",
            "metric_summary": {"score": 1},
        },
        report_path=report,
    ).step_id == evaluation.step_id
    terminal = coordinator.seal_terminal_report(report_path=report, status="passed")
    assert terminal.status == "passed"
    binding = coordinator.validate_terminal_report_for_reuse(expected_status="passed")
    assert binding.size_bytes == len(b'{"score":1}\n')
    with pytest.raises(CheckpointTerminalError, match="terminal"):
        coordinator.commit_model_turn(
            turn_id="0002",
            sdk_session_items=[{"turn": 2}],
            bank=bank,
            state=state,
        )

    checkpoint.close()
    restarted_checkpoint = _open_checkpoint(root)
    restarted = RuntimeCheckpointCoordinator(root, restarted_checkpoint)
    assert restarted.validate_terminal_report_for_reuse().sha256 == binding.sha256
    report.write_bytes(b'{"score":0}\n')
    with pytest.raises(RuntimeCheckpointDriftError, match="terminal report size/hash"):
        restarted.validate_terminal_report_for_reuse()

    restarted_checkpoint.close()
    bank.close()
