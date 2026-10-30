"""HarnessAudit v8 compatibility fix for pre-unit model failures.

The v7 checkpoint schema intentionally permits an attempt to pause before a
replay unit starts.  This happens when policy preparation or another pipeline
stage calls the model before ``run_predict_only_replay_for_report``.  The v7
attempt-evidence validator nevertheless required every paused/terminal outcome
to be supported by ``UNIT_PAUSE``.  Such an event cannot exist when no unit was
opened, so a valid ``TRACE_ATTEMPT_START -> TRACE_PAUSE`` history was rejected.

This module keeps the v7 on-disk schema and scheduler, but installs one narrow
validator compatibility rule.  Both this adapter and the complete v7 base
implementation are included in the frozen v8 implementation inventory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import ha_resilient_predict_only_v7 as _v7


_ORIGINAL_READ_ATTEMPT_EVIDENCE = _v7._read_attempt_evidence
_PREUNIT_SUPPORT_ERROR = "failure evidence lacks its UNIT_PAUSE support"
_V8_RELATIVE_PATH = (
    "harness_env/orgin/HarnessAudit/multi_agent/safepred_experiments/"
    "ha_resilient_predict_only_v8.py"
)
_V8_CORE_IMPLEMENTATION_RELATIVE_PATHS = (
    *_v7._V7_CORE_IMPLEMENTATION_RELATIVE_PATHS,
    _V8_RELATIVE_PATH,
)


def _read_attempt_evidence_v8(
    task: _v7.V7TaskDescriptor,
    *,
    run_root: Path,
    manifest_sha256: str,
    run_id: str,
    events: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    """Accept a strictly proven failure before the first unit of an attempt.

    The base validator remains authoritative for every schema, hash, identity,
    sidecar, transport-ledger, and artifact check.  We only provide a synthetic
    validation view for an attempt whose persisted history contains exactly a
    start followed by its trace outcome, with no unit event in between.  The
    synthetic row is never persisted and cannot weaken the history reducer.
    """

    try:
        return _ORIGINAL_READ_ATTEMPT_EVIDENCE(
            task,
            run_root=run_root,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            events=events,
        )
    except _v7.IntegrityFailure as exc:
        if str(exc) != _PREUNIT_SUPPORT_ERROR:
            raise
        original_error = exc

    path = _v7.v7_attempt_evidence_path(task)
    if not path.exists():
        raise original_error
    rows = _v7._read_jsonl_strict(path)
    synthetic_support: list[dict[str, Any]] = []
    recovered = 0
    for row in rows:
        outcome = row.get("outcome")
        if outcome not in {"paused", "terminal-model-invalid"}:
            continue
        support_sha = row.get("pre_outcome_history_tail_sha256")
        supporting_event = next(
            (event for event in events if event.get("event_sha256") == support_sha),
            None,
        )
        if supporting_event is None or supporting_event.get("event") == "UNIT_PAUSE":
            continue
        if supporting_event.get("event") != "TRACE_ATTEMPT_START":
            raise original_error

        attempt = row.get("attempt")
        attempt_events = [event for event in events if event.get("attempt") == attempt]
        expected_trace = {
            "paused": "TRACE_PAUSE",
            "terminal-model-invalid": "TRACE_TERMINAL_MODEL_INVALID",
        }[outcome]
        if [event.get("event") for event in attempt_events] not in (
            ["TRACE_ATTEMPT_START"],
            ["TRACE_ATTEMPT_START", expected_trace],
        ):
            raise original_error
        if attempt_events[0].get("event_sha256") != support_sha:
            raise original_error
        if len(attempt_events) == 2 and (
            attempt_events[1].get("previous_event_sha256") != support_sha
        ):
            raise original_error

        start_index = events.index(supporting_event)
        prior_passes = sum(
            event.get("event") == "UNIT_PASS" for event in events[:start_index]
        )
        if prior_passes >= len(task.expectations):
            raise original_error
        expected_unit_key = task.expectations[prior_passes].unit.key
        if row.get("unit_key") != expected_unit_key:
            raise original_error

        support_view = dict(supporting_event)
        support_view["event"] = "UNIT_PAUSE"
        support_view["unit_key"] = expected_unit_key
        support_view["failure_kind"] = row.get("failure_kind")
        synthetic_support.append(support_view)
        recovered += 1

    if recovered == 0:
        raise original_error
    return _ORIGINAL_READ_ATTEMPT_EVIDENCE(
        task,
        run_root=run_root,
        manifest_sha256=manifest_sha256,
        run_id=run_id,
        events=[*synthetic_support, *events],
    )


def install_v8_runtime() -> None:
    """Install the compatibility validator and freeze both v7 and v8 sources."""

    _v7._read_attempt_evidence = _read_attempt_evidence_v8
    _v7._V7_CORE_IMPLEMENTATION_RELATIVE_PATHS = (
        _V8_CORE_IMPLEMENTATION_RELATIVE_PATHS
    )


def __getattr__(name: str) -> Any:
    return getattr(_v7, name)


def main(argv: list[str] | None = None) -> int:
    install_v8_runtime()
    return _v7.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
