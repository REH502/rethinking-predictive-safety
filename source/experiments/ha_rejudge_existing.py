#!/usr/bin/env python3
"""Re-evaluate persisted HarnessAudit traces with an external judge only.

The source actor/WM trajectories are immutable inputs. This diagnostic tool
reconstructs ObservableAction records from trace JSONL, re-runs the pooled TCR
LLM component and AVS operational judge, and preserves source failures in the
fixed denominator. It never writes into a source run and never persists judge
reasoning or response bodies.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.model_stack import load_model_stack_profile
from experiments.workflows import load_experiment_credentials
from multi_agent.completion_judge import (
    _LLM_SYSTEM_PROMPT,
    _LLM_USER_PROMPT_TEMPLATE,
    format_trajectory_evidence,
)
from multi_agent.loader import load_task_with_tools
from multi_agent.openai_json import OpenAIJSONError, create_json_response_async
from multi_agent.operational_judge import (
    _SYSTEM_PROMPT as AVS_SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE as AVS_USER_PROMPT_TEMPLATE,
    _extract_scope_constraints,
    _extract_tool_calls_by_role,
    _format_scope_section,
)
from multi_agent.safepred_experiments.predict_only_pipeline import (
    _trace_event_to_observable_action,
)
from multi_agent.safepred_experiments.run_matrix import (
    _active_credential_scan,
    _preflight_openai_model_recorded,
)


ARMS = ("no_guard", "advisory_only", "enforced_guard")
RESULT_SCHEMA_VERSION = "ha-existing-trace-external-rejudge-result-v2"
SUMMARY_SCHEMA_VERSION = "ha-existing-trace-external-rejudge-summary-v2"
PROVENANCE_SCHEMA_VERSION = "ha-existing-trace-external-rejudge-provenance-v2"
CONDITION_FIELDS = (
    "experiment_seed",
    "guidance_mode",
    "intervention_mode",
    "observation_perturbation",
    "policy_mode",
    "policy_perturbation",
    "risk_noise_rate",
    "risk_signal_mode",
    "task",
)
_STATUS_PATTERN = re.compile(r"status=(\d{3})")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_key(condition: dict[str, Any]) -> str:
    payload = {field: condition.get(field) for field in CONDITION_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at line {line_number}")
        rows.append(row)
    return rows


def _safe_source_file(raw: str, source_root: Path, *, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    root = source_root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes source root") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _arm_from_report_path(path: Path) -> str | None:
    rendered = path.as_posix()
    if "/harnessaudit/results/none/n/" in rendered:
        return "no_guard"
    if "/harnessaudit/results/train/a/" in rendered:
        return "advisory_only"
    if "/harnessaudit/results/train/e/" in rendered:
        return "enforced_guard"
    return None


def _build_report_index(
    terminal_rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], Path]:
    roots = sorted(
        {
            str(Path(str(row.get("source_root") or "")).expanduser().resolve())
            for row in terminal_rows
        }
    )
    index: dict[tuple[str, str, str], Path] = {}
    for root_text in roots:
        root = Path(root_text)
        if not root.is_dir():
            raise ValueError(f"source root is missing: {root}")
        for path in root.rglob("*.json"):
            arm = _arm_from_report_path(path)
            if arm is None:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "trace_file" not in payload:
                continue
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                continue
            key = (root_text, arm, task_id)
            if key in index:
                raise ValueError(f"multiple source reports for {arm}/{task_id}")
            index[key] = path.resolve()
    return index


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _usage_zero() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def _judge_call(
    *,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    condition_key: str,
    call_kind: str,
    role: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.monotonic()
    ledger = {
        "schema_version": "ha-existing-trace-rejudge-request-v1",
        "condition_key": condition_key,
        "call_kind": call_kind,
        "role": role,
        "model": judge_model,
        "semantic_attempt_count": 1,
    }
    try:
        async with semaphore:
            response = await create_json_response_async(
                model=judge_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                include_metadata=True,
            )
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("judge payload is not an object")
        ledger.update(
            {
                "status": "accepted",
                "http_status": 200,
                "transport_attempt_count": int(response.get("attempt_count") or 1),
                "token_usage": dict(response.get("usage") or _usage_zero()),
                "api_mode": str(response.get("api_mode") or ""),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
        return payload, ledger
    except (OpenAIJSONError, ValueError, TypeError, json.JSONDecodeError) as exc:
        status_match = _STATUS_PATTERN.search(str(exc))
        ledger.update(
            {
                "status": "failed",
                "http_status": int(status_match.group(1)) if status_match else None,
                "transport_attempt_count": 1,
                "token_usage": _usage_zero(),
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
        return None, ledger


def _score_payload(payload: dict[str, Any] | None) -> float | None:
    if payload is None or not _finite(payload.get("score")):
        return None
    return max(0.0, min(1.0, float(payload["score"])))


async def _evaluate_condition(
    source_row: dict[str, Any],
    *,
    report_path: Path,
    judge_model: str,
    call_semaphore: asyncio.Semaphore,
    condition_semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    async with condition_semaphore:
        condition = dict(source_row["condition"])
        condition_key = _condition_key(condition)
        source_root = Path(source_row["source_root"]).resolve()
        report_path = _safe_source_file(str(report_path), source_root, label="report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = _safe_source_file(
            str(report.get("trace_file") or ""),
            source_root,
            label="trace",
        )
        trace_rows = _read_jsonl(trace_path)
        run_id = str(report.get("run_id") or "")
        actions = [
            _trace_event_to_observable_action(event, run_id=run_id)
            for event in trace_rows
            if event.get("type") in {"tool_call", "communication"}
        ]
        task_id = str(condition["task"])
        task, _catalog = load_task_with_tools(f"{task_id}.yaml")
        hub_role = task.metadata.get("hub_role")
        if not isinstance(hub_role, str):
            hub_role = task.agents[0].role

        call_specs: list[tuple[str, str | None, Any]] = []
        llm_weight = float(report.get("llm_completion_weight") or 0.0)
        deterministic_weight = float(
            report.get("deterministic_completion_weight") or 0.0
        )
        deterministic_contribution = float(
            report.get("deterministic_completion_contribution") or 0.0
        )
        if llm_weight > 0.0:
            completion_prompt = _LLM_USER_PROMPT_TEMPLATE.format(
                task_goal=task.goal,
                trajectory_evidence=format_trajectory_evidence(actions),
            )
            call_specs.append(
                (
                    "completion",
                    None,
                    _judge_call(
                        semaphore=call_semaphore,
                        judge_model=judge_model,
                        system_prompt=_LLM_SYSTEM_PROMPT,
                        user_prompt=completion_prompt,
                        max_output_tokens=512,
                        condition_key=condition_key,
                        call_kind="completion",
                    ),
                )
            )

        ground_truth = task.ground_truth_tool_paths or {}
        roles_to_score = set(ground_truth)
        exclude_hub = roles_to_score != {hub_role}
        actual_by_role = _extract_tool_calls_by_role(
            actions,
            hub_role,
            exclude_hub=exclude_hub,
        )
        scope_by_role = _extract_scope_constraints(task.access_rules)
        if ground_truth and actual_by_role:
            for role, valid_sets in ground_truth.items():
                prompt = AVS_USER_PROMPT_TEMPLATE.format(
                    role=role,
                    ground_truth_json=json.dumps(valid_sets, indent=2),
                    scope_section=_format_scope_section(role, scope_by_role),
                    actual_tools_json=json.dumps(actual_by_role.get(role, []), indent=2),
                )
                call_specs.append(
                    (
                        "avs",
                        str(role),
                        _judge_call(
                            semaphore=call_semaphore,
                            judge_model=judge_model,
                            system_prompt=AVS_SYSTEM_PROMPT,
                            user_prompt=prompt,
                            max_output_tokens=512,
                            condition_key=condition_key,
                            call_kind="avs_role",
                            role=str(role),
                        ),
                    )
                )

        outcomes = await asyncio.gather(*(spec[2] for spec in call_specs))
        ledger_rows: list[dict[str, Any]] = []
        completion_score: float | None = 0.0 if llm_weight == 0.0 else None
        avs_scores: list[float] = []
        evaluator_failures: list[str] = []
        for (kind, role, _coroutine), (payload, ledger) in zip(call_specs, outcomes):
            score = _score_payload(payload)
            if score is None:
                if ledger.get("status") == "accepted":
                    ledger["status"] = "failed_validation"
                ledger["semantic_validation_passed"] = False
                evaluator_failures.append(f"{kind}:{role or 'pooled'}")
            else:
                ledger["semantic_validation_passed"] = True
                if kind == "completion":
                    completion_score = score
                else:
                    avs_scores.append(score)
            ledger_rows.append(ledger)

        expected_avs_calls = len(ground_truth) if ground_truth and actual_by_role else 0
        if expected_avs_calls and len(avs_scores) != expected_avs_calls:
            evaluator_failures.append("avs:incomplete_role_set")
        rejudged_avs = (
            sum(avs_scores) / len(avs_scores)
            if expected_avs_calls and len(avs_scores) == expected_avs_calls
            else None
        )
        source_error = report.get("error")
        completion_denominator = deterministic_weight + llm_weight
        rejudged_tcr = None
        if (
            source_error in (None, "")
            and completion_score is not None
            and completion_denominator > 0.0
        ):
            rejudged_tcr = max(
                0.0,
                min(
                    1.0,
                    (
                        deterministic_contribution
                        + completion_score * llm_weight
                    )
                    / completion_denominator,
                ),
            )

        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "condition_key": condition_key,
            "condition": condition,
            "source_terminal_status": str(source_row.get("status") or ""),
            "scientific_eligibility": (
                "source_passed_primary_rejudge"
                if source_row.get("status") == "passed"
                else "source_invalid_partial_trajectory_diagnostic"
            ),
            "evaluator_status": "failed" if evaluator_failures else "passed",
            "evaluator_failure_components": sorted(set(evaluator_failures)),
            "source_report_path": str(report_path),
            "source_report_sha256": _sha256_file(report_path),
            "source_trace_path": str(trace_path),
            "source_trace_sha256": _sha256_file(trace_path),
            "source_trace_error_present": source_error not in (None, ""),
            "n_actions": len(actions),
            "original_tcr": report.get("tcr"),
            "original_avs": report.get("avs"),
            "deterministic_completion_weight": deterministic_weight,
            "deterministic_completion_contribution": deterministic_contribution,
            "llm_completion_weight": llm_weight,
            "rejudged_llm_completion_score": completion_score,
            "rejudged_tcr": rejudged_tcr,
            "rejudged_avs": rejudged_avs,
            "avs_role_count": expected_avs_calls,
            "judge_call_count": len(ledger_rows),
        }
        return result, ledger_rows


def summarize_results(
    results: list[dict[str, Any]],
    *,
    expected_per_arm: int,
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in results if row["condition"]["intervention_mode"] == arm]
        primary = [
            row
            for row in rows
            if row["source_terminal_status"] == "passed"
            and row["evaluator_status"] == "passed"
            and _finite(row.get("rejudged_tcr"))
            and _finite(row.get("rejudged_avs"))
        ]
        by_arm[arm] = {
            "fixed_denominator_n": expected_per_arm,
            "observed_n": len(rows),
            "source_status_counts": dict(
                sorted(Counter(row["source_terminal_status"] for row in rows).items())
            ),
            "evaluator_status_counts": dict(
                sorted(Counter(row["evaluator_status"] for row in rows).items())
            ),
            "primary_rejudged_n": len(primary),
            "primary_rejudged_tcr_mean": (
                statistics.mean(float(row["rejudged_tcr"]) for row in primary)
                if primary
                else None
            ),
            "primary_rejudged_avs_mean": (
                statistics.mean(float(row["rejudged_avs"]) for row in primary)
                if primary
                else None
            ),
        }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_class": "diagnostic_rejudge_unblinded_not_formal_evidence",
        "expected_total": expected_per_arm * len(ARMS),
        "observed_total": len(results),
        "complete": len(results) == expected_per_arm * len(ARMS),
        "by_arm": by_arm,
    }


async def _run(args: argparse.Namespace) -> int:
    terminal_manifest = args.terminal_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    profile_path = args.model_profile.expanduser().resolve()
    if not terminal_manifest.is_file():
        raise ValueError("terminal manifest is missing")
    terminal_rows = _read_jsonl(terminal_manifest)
    if len(terminal_rows) != args.expected_per_arm * len(ARMS):
        raise ValueError("terminal manifest does not match the fixed denominator")
    if Counter(
        row["condition"]["intervention_mode"] for row in terminal_rows
    ) != Counter({arm: args.expected_per_arm for arm in ARMS}):
        raise ValueError("terminal manifest arm counts differ from the fixed design")

    results_path = output_root / "rejudge_results.jsonl"
    ledger_path = output_root / "judge_request_ledger.jsonl"
    provenance_path = output_root / "rejudge_provenance.json"
    if output_root.exists() and not args.resume:
        raise ValueError("refusing to overwrite existing rejudge root")
    output_root.mkdir(parents=True, exist_ok=args.resume)

    loaded = load_experiment_credentials(dict(os.environ))
    stack = load_model_stack_profile(
        profile_path,
        environ=loaded,
        require_credentials=True,
    )
    judge = stack.roles["judge"]
    os.environ.update(stack.runtime_environment())
    os.environ["MASP_OPENAI_RETRY_BASE_DELAY"] = "0"

    provenance_identity = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_class": "diagnostic_rejudge_unblinded_not_formal_evidence",
        "source_terminal_manifest": str(terminal_manifest),
        "source_terminal_manifest_sha256": _sha256_file(terminal_manifest),
        "model_profile": str(profile_path),
        "model_profile_sha256": stack.source_sha256,
        "judge_identity": judge.public_dict(),
        "judge_alias_immutable": False,
        "rejudge_script_sha256": _sha256_file(Path(__file__).resolve()),
        "completion_judge_source_sha256": _sha256_file(
            Path(__file__).resolve().parents[1]
            / "harness_env/orgin/HarnessAudit/multi_agent/completion_judge.py"
        ),
        "operational_judge_source_sha256": _sha256_file(
            Path(__file__).resolve().parents[1]
            / "harness_env/orgin/HarnessAudit/multi_agent/operational_judge.py"
        ),
        "response_bodies_persisted": False,
        "reasoning_persisted": False,
        "transport_max_attempts": judge.transport_max_attempts,
    }
    if provenance_path.exists():
        observed_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed_identity = dict(observed_provenance)
        observed_identity.pop("created_at_utc", None)
        if observed_identity != provenance_identity:
            raise ValueError("resume provenance differs")
    else:
        provenance = {
            **provenance_identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(provenance_path, provenance)

    _preflight_openai_model_recorded(
        artifact_path=output_root / "endpoint_preflight.jsonl",
        base_url=judge.base_url,
        model=judge.model,
        api_key=judge.api_key,
        label="HarnessAudit persisted-trace evaluator",
        timeout=judge.timeout_seconds,
        api_mode=judge.api_mode,
        request_extra_body=judge.request_extra_body,
        temperature=judge.temperature,
    )

    completed: dict[str, dict[str, Any]] = {}
    if results_path.is_file():
        for row in _read_jsonl(results_path):
            key = str(row.get("condition_key") or "")
            if not key or key in completed:
                raise ValueError("invalid or duplicate resume result key")
            completed[key] = row
    report_index = _build_report_index(terminal_rows)
    pending = [
        row
        for row in terminal_rows
        if _condition_key(row["condition"]) not in completed
    ]
    call_semaphore = asyncio.Semaphore(max(1, args.judge_workers))
    condition_semaphore = asyncio.Semaphore(max(1, args.condition_workers))
    coroutines = []
    for row in pending:
        condition = row["condition"]
        index_key = (
            str(Path(row["source_root"]).resolve()),
            str(condition["intervention_mode"]),
            str(condition["task"]),
        )
        report_path = report_index.get(index_key)
        if report_path is None:
            raise ValueError(f"missing source report for {index_key[1:]}")
        coroutines.append(
            _evaluate_condition(
                row,
                report_path=report_path,
                judge_model=judge.model,
                call_semaphore=call_semaphore,
                condition_semaphore=condition_semaphore,
            )
        )

    newly_completed = 0
    for future in asyncio.as_completed(coroutines):
        result, ledger_rows = await future
        for ledger in ledger_rows:
            _append_jsonl_durable(ledger_path, ledger)
        _append_jsonl_durable(results_path, result)
        completed[result["condition_key"]] = result
        newly_completed += 1
        if newly_completed % 10 == 0 or len(completed) == len(terminal_rows):
            print(f"rejudge progress: {len(completed)}/{len(terminal_rows)}", flush=True)

    results = list(completed.values())
    summary = summarize_results(results, expected_per_arm=args.expected_per_arm)
    summary["judge_identity"] = judge.public_dict()
    summary["source_terminal_manifest_sha256"] = _sha256_file(terminal_manifest)
    ledger_rows = _read_jsonl(ledger_path) if ledger_path.is_file() else []
    summary["judge_request_summary"] = {
        "call_count": len(ledger_rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in ledger_rows).items())),
        "transport_attempt_count": sum(
            int(row.get("transport_attempt_count") or 0) for row in ledger_rows
        ),
        "token_usage": {
            field: sum(int((row.get("token_usage") or {}).get(field) or 0) for row in ledger_rows)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    _write_json_atomic(output_root / "rejudge_summary.json", summary)

    credential_scan_paths = [output_root, *args.credential_scan_path]
    credential_scan = _active_credential_scan(credential_scan_paths)
    credential_scan.update(
        {
            "type": "ha_existing_trace_rejudge_active_credential_scan",
            "scope": "rejudge_output_root_plus_declared_paths_v1",
        }
    )
    _write_json_atomic(output_root / "active_credential_scan.json", credential_scan)
    if credential_scan.get("status") != "passed":
        return 7
    if not summary["complete"]:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--expected-per-arm", type=int, default=192)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--condition-workers", type=int, default=2)
    parser.add_argument(
        "--credential-scan-path",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
