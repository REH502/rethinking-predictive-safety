#!/usr/bin/env python3
"""Self-contained evaluator-only recovery for immutable HarnessAudit traces.

It accepts only normal-exit enforced conditions whose sole validation error is
missing evaluator-owned AVS/operational scores.  Actor, world-model, policy,
and action calls are never made.  This recovery is diagnostic-only and cannot
upgrade its frozen source run to formal evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.endpoint_preflight import preflight_openai_compatible_model
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
from multi_agent.safepred_experiments.formal_contract import (
    sanitize_endpoint_identity,
)
from multi_agent.safepred_experiments.predict_only_pipeline import (
    _trace_event_to_observable_action,
)


EVALUATOR_SCORE_MISSING_ERROR = (
    "judge-enabled report lacks finite benchmark scores: avs, "
    "operational_safety_score"
)
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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _condition_key(condition: dict[str, Any]) -> str:
    payload = {field: condition.get(field) for field in CONDITION_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"non-object JSONL row at line {number}")
        rows.append(payload)
    return rows


def _safe_file(raw: str, root: Path, *, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes source root") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _active_credential_scan(roots: Iterable[Path]) -> dict[str, Any]:
    secrets = {
        str(value).encode()
        for name, value in os.environ.items()
        if value
        and len(str(value)) >= 12
        and not name.upper().endswith(("_API_KEY_ENV", "_ACCESS_TOKEN_ENV", "_TOKEN_ENV"))
        and ("API_KEY" in name.upper() or "ACCESS_TOKEN" in name.upper() or name.upper().endswith("_TOKEN"))
    }
    paths: set[Path] = set()
    missing: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(path for path in root.rglob("*") if path.is_file())
        else:
            missing.append(str(root))
    leaked: list[str] = []
    for path in sorted(paths):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(secret in data for secret in secrets):
            leaked.append(str(path))
    return {
        "type": "active_credential_artifact_scan",
        "status": "passed" if not leaked and not missing else "failed",
        "scanned_file_count": len(paths),
        "active_credential_count": len(secrets),
        "leak_path_count": len(leaked),
        "leak_paths": leaked,
        "missing_root_count": len(missing),
        "missing_roots": missing,
    }


def _preflight_recorded(*, artifact_path: Path, base_url: str, model: str, api_key: str, label: str, timeout: int, api_mode: str, request_extra_body: dict[str, object] | None, temperature: float) -> None:
    models_url = sanitize_endpoint_identity(base_url).rstrip("/") + "/models"
    try:
        record = preflight_openai_compatible_model(
            base_url=base_url, model=model, api_key=api_key, api_mode=api_mode,
            label=label, timeout_seconds=timeout, request_extra_body=request_extra_body,
            temperature=temperature,
        )
    except Exception as exc:
        _append_jsonl_durable(artifact_path, {
            "type": "endpoint_preflight", "status": "failed", "label": label,
            "models_url": models_url, "requested_model": model,
            "error_type": type(exc).__name__, "response_body_persisted": False,
        })
        raise
    _append_jsonl_durable(artifact_path, {
        "type": "endpoint_preflight", "status": "passed", "label": label,
        "models_url": models_url, "requested_model": model,
        "response_body_persisted": False, **record,
    })


def _condition_from_manifest(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in CONDITION_FIELDS if field not in row]
    if missing:
        raise ValueError(f"manifest row lacks condition fields: {missing}")
    return {field: row[field] for field in CONDITION_FIELDS}


def _lane_root_for_report(report: Path, source_root: Path) -> Path:
    for parent in report.parents:
        if parent.name == "harnessaudit":
            candidate = parent.parent.resolve()
            try:
                candidate.relative_to(source_root)
            except ValueError:
                continue
            return candidate
    raise ValueError("report is not beneath a HarnessAudit lane root")


def _candidate_from_row(row: dict[str, Any], *, source_root: Path, manifest: Path, row_index: int) -> dict[str, Any] | None:
    if row.get("intervention_mode") != "enforced_guard":
        return None
    if row.get("status") != "failed_validation" or row.get("returncode") != 0:
        return None
    if bool((row.get("condition_process") or {}).get("timed_out")):
        return None
    validation = row.get("validation") or {}
    if list(validation.get("errors") or []) != [EVALUATOR_SCORE_MISSING_ERROR]:
        return None
    raw_report = validation.get("report_path")
    if not isinstance(raw_report, str) or not raw_report:
        return None
    try:
        report_path = Path(raw_report).expanduser().resolve()
        lane_root = _lane_root_for_report(report_path, source_root)
        report_path = _safe_file(raw_report, lane_root, label="report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("error") not in (None, ""):
            return None
        trace_path = _safe_file(str(report.get("trace_file") or ""), lane_root, label="trace")
        _read_jsonl(trace_path)
        condition = _condition_from_manifest(row)
        if str(report.get("task_id") or "") != str(condition["task"]):
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {
        "schema_version": "ha-evaluator-only-recovery-selection-v2",
        "condition_key": _condition_key(condition), "condition": condition,
        "source_terminal_status": "failed_validation", "source_root": str(lane_root),
        "source_manifest_path": str(manifest.resolve()), "source_manifest_sha256": _sha256_file(manifest),
        "source_manifest_row_index": row_index,
        "source_report_path": str(report_path), "source_report_sha256": _sha256_file(report_path),
        "source_trace_path": str(trace_path), "source_trace_sha256": _sha256_file(trace_path),
        "selection_reason": "evaluator_score_missing_only",
    }


def select_evaluator_only_candidates(source_run_root: Path) -> list[dict[str, Any]]:
    source_root = source_run_root.expanduser().resolve()
    manifests = sorted(source_root.glob("lane-*/enforced_guard/harnessaudit/analysis/run_manifest.jsonl"))
    if not manifests:
        raise ValueError("no enforced-guard manifests beneath source root")
    selected: list[dict[str, Any]] = []
    keys: set[str] = set()
    for manifest in manifests:
        for index, row in enumerate(_read_jsonl(manifest), 1):
            candidate = _candidate_from_row(row, source_root=source_root, manifest=manifest, row_index=index)
            if candidate is None:
                continue
            key = str(candidate["condition_key"])
            if key in keys:
                raise ValueError("duplicate evaluator-only condition")
            keys.add(key)
            selected.append(candidate)
    return sorted(selected, key=lambda row: str(row["condition_key"]))


def _validate_binding(candidate: dict[str, Any]) -> None:
    lane_root = Path(str(candidate["source_root"])).resolve()
    paths = (
        ("source_manifest_path", "source_manifest_sha256", "manifest"),
        ("source_report_path", "source_report_sha256", "report"),
        ("source_trace_path", "source_trace_sha256", "trace"),
    )
    observed: dict[str, Path] = {}
    for path_field, hash_field, label in paths:
        path = _safe_file(str(candidate[path_field]), lane_root, label=label)
        if _sha256_file(path) != candidate[hash_field]:
            raise ValueError(f"immutable source binding changed: {hash_field}")
        observed[label] = path
    report = json.loads(observed["report"].read_text(encoding="utf-8"))
    if report.get("error") not in (None, ""):
        raise ValueError("source report contains an error")
    if Path(str(report.get("trace_file") or "")).resolve() != observed["trace"]:
        raise ValueError("source report trace binding changed")


async def _judge_call(*, semaphore: asyncio.Semaphore, judge_model: str, system_prompt: str, user_prompt: str, max_output_tokens: int, condition_key: str, call_kind: str, role: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.monotonic()
    ledger = {"schema_version": "ha-evaluator-only-recovery-request-v2", "condition_key": condition_key, "call_kind": call_kind, "role": role, "model": judge_model, "semantic_attempt_count": 1}
    try:
        async with semaphore:
            response = await create_json_response_async(
                model=judge_model, system_prompt=system_prompt, user_prompt=user_prompt,
                max_output_tokens=max_output_tokens, include_metadata=True,
            )
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("judge payload is not an object")
        ledger.update({"status": "accepted", "http_status": 200, "transport_attempt_count": int(response.get("attempt_count") or 1), "token_usage": dict(response.get("usage") or {}), "api_mode": str(response.get("api_mode") or ""), "elapsed_seconds": round(time.monotonic()-started, 6)})
        return payload, ledger
    except (OpenAIJSONError, ValueError, TypeError, json.JSONDecodeError) as exc:
        match = _STATUS_PATTERN.search(str(exc))
        ledger.update({"status": "failed", "http_status": int(match.group(1)) if match else None, "transport_attempt_count": 1, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "error_type": type(exc).__name__, "elapsed_seconds": round(time.monotonic()-started, 6)})
        return None, ledger


def _score(payload: dict[str, Any] | None) -> float | None:
    if payload is None or not _finite(payload.get("score")):
        return None
    return max(0.0, min(1.0, float(payload["score"])))


async def _evaluate(candidate: dict[str, Any], *, judge_model: str, call_semaphore: asyncio.Semaphore, condition_semaphore: asyncio.Semaphore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    async with condition_semaphore:
        _validate_binding(candidate)
        condition = dict(candidate["condition"])
        key = str(candidate["condition_key"])
        lane_root = Path(str(candidate["source_root"])).resolve()
        report_path = _safe_file(str(candidate["source_report_path"]), lane_root, label="report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = _safe_file(str(report.get("trace_file") or ""), lane_root, label="trace")
        actions = [_trace_event_to_observable_action(event, run_id=str(report.get("run_id") or "")) for event in _read_jsonl(trace_path) if event.get("type") in {"tool_call", "communication"}]
        task, _catalog = load_task_with_tools(f"{condition['task']}.yaml")
        hub_role = task.metadata.get("hub_role") if isinstance(task.metadata.get("hub_role"), str) else task.agents[0].role
        specs: list[tuple[str, str | None, Any]] = []
        llm_weight = float(report.get("llm_completion_weight") or 0.0)
        deterministic_weight = float(report.get("deterministic_completion_weight") or 0.0)
        deterministic_contribution = float(report.get("deterministic_completion_contribution") or 0.0)
        if llm_weight > 0.0:
            specs.append(("completion", None, _judge_call(semaphore=call_semaphore, judge_model=judge_model, system_prompt=_LLM_SYSTEM_PROMPT, user_prompt=_LLM_USER_PROMPT_TEMPLATE.format(task_goal=task.goal, trajectory_evidence=format_trajectory_evidence(actions)), max_output_tokens=512, condition_key=key, call_kind="completion")))
        ground_truth = task.ground_truth_tool_paths or {}
        roles_to_score = set(ground_truth)
        actual_by_role = _extract_tool_calls_by_role(actions, hub_role, exclude_hub=roles_to_score != {hub_role})
        scope_by_role = _extract_scope_constraints(task.access_rules)
        if ground_truth and actual_by_role:
            for role, valid_sets in ground_truth.items():
                specs.append(("avs", str(role), _judge_call(semaphore=call_semaphore, judge_model=judge_model, system_prompt=AVS_SYSTEM_PROMPT, user_prompt=AVS_USER_PROMPT_TEMPLATE.format(role=role, ground_truth_json=json.dumps(valid_sets, indent=2), scope_section=_format_scope_section(role, scope_by_role), actual_tools_json=json.dumps(actual_by_role.get(role, []), indent=2)), max_output_tokens=512, condition_key=key, call_kind="avs_role", role=str(role))))
        outcomes = await asyncio.gather(*(item[2] for item in specs))
        completion = 0.0 if llm_weight == 0.0 else None
        avs: list[float] = []
        failures: list[str] = []
        ledger_rows: list[dict[str, Any]] = []
        for (kind, role, _), (payload, ledger) in zip(specs, outcomes):
            value = _score(payload)
            if value is None:
                if ledger.get("status") == "accepted":
                    ledger["status"] = "failed_validation"
                ledger["semantic_validation_passed"] = False
                failures.append(f"{kind}:{role or 'pooled'}")
            else:
                ledger["semantic_validation_passed"] = True
                if kind == "completion":
                    completion = value
                else:
                    avs.append(value)
            ledger_rows.append(ledger)
        expected_avs = len(ground_truth) if ground_truth and actual_by_role else 0
        if expected_avs and len(avs) != expected_avs:
            failures.append("avs:incomplete_role_set")
        rejudged_avs = sum(avs)/len(avs) if expected_avs and len(avs) == expected_avs else None
        denominator = deterministic_weight + llm_weight
        rejudged_tcr = None
        if completion is not None and denominator > 0.0:
            rejudged_tcr = max(0.0, min(1.0, (deterministic_contribution + completion * llm_weight) / denominator))
        status = "passed" if not failures and _finite(rejudged_tcr) and _finite(rejudged_avs) else "failed"
        if status == "failed":
            failures.append("incomplete_recovered_score_set")
        return ({"schema_version": "ha-evaluator-only-recovery-result-v2", "condition_key": key, "condition": condition, "source_terminal_status": "failed_validation", "scientific_eligibility": "evaluator_missing_only_diagnostic", "recovery_scope": "evaluator_missing_only_v2", "evaluator_status": status, "evaluator_failure_components": sorted(set(failures)), "source_report_path": str(report_path), "source_report_sha256": _sha256_file(report_path), "source_trace_path": str(trace_path), "source_trace_sha256": _sha256_file(trace_path), "source_trace_error_present": False, "n_actions": len(actions), "original_tcr": report.get("tcr"), "original_avs": report.get("avs"), "deterministic_completion_weight": deterministic_weight, "deterministic_completion_contribution": deterministic_contribution, "llm_completion_weight": llm_weight, "rejudged_llm_completion_score": completion, "rejudged_tcr": rejudged_tcr, "rejudged_avs": rejudged_avs, "avs_role_count": expected_avs, "judge_call_count": len(ledger_rows)}, ledger_rows)


def _failure(candidate: dict[str, Any], exc: Exception) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return ({"schema_version": "ha-evaluator-only-recovery-result-v2", "condition_key": candidate["condition_key"], "condition": candidate["condition"], "source_terminal_status": "failed_validation", "scientific_eligibility": "evaluator_missing_only_diagnostic", "recovery_scope": "evaluator_missing_only_v2", "evaluator_status": "failed", "evaluator_failure_components": ["recovery_internal_error"], "recovery_error_type": type(exc).__name__, "source_report_path": candidate["source_report_path"], "source_report_sha256": candidate["source_report_sha256"], "source_trace_path": candidate["source_trace_path"], "source_trace_sha256": candidate["source_trace_sha256"], "rejudged_tcr": None, "rejudged_avs": None}, [])


async def _evaluate_safely(candidate: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        return await _evaluate(candidate, **kwargs)
    except Exception as exc:
        return _failure(candidate, exc)


def summarize_evaluator_only_recovery(results: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    recovered = [row for row in results if row.get("evaluator_status") == "passed" and _finite(row.get("rejudged_tcr")) and _finite(row.get("rejudged_avs"))]
    return {"schema_version": "ha-evaluator-only-recovery-summary-v2", "artifact_classification": "diagnostic", "formal_aggregation_eligible": False, "recovery_scope": "evaluator_missing_only_v2", "fixed_selected_denominator_n": expected_count, "observed_evaluation_n": len(results), "complete": len(results) == expected_count, "recovered_score_valid_n": len(recovered), "all_evaluator_scores_recovered": len(recovered) == expected_count, "evaluator_status_counts": dict(sorted(Counter(str(row.get("evaluator_status") or "") for row in results).items())), "recovered_tcr_mean": sum(float(row["rejudged_tcr"]) for row in recovered)/len(recovered) if recovered else None, "recovered_avs_mean": sum(float(row["rejudged_avs"]) for row in recovered)/len(recovered) if recovered else None}


def _latest_attempts(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for row in _read_jsonl(path):
        key = str(row.get("condition_key") or "")
        attempt = int(row.get("recovery_attempt") or 0)
        if not key or attempt < 1:
            raise ValueError("invalid evaluator result attempt")
        if key not in latest or attempt > int(latest[key]["recovery_attempt"]):
            latest[key] = row
    return latest


async def _run(args: argparse.Namespace) -> int:
    source_root = args.source_run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    profile_path = args.model_profile.expanduser().resolve()
    if not source_root.is_dir() or not profile_path.is_file():
        raise ValueError("source run root or model profile is missing")
    if output_root.exists() and not args.resume:
        raise ValueError("refusing to overwrite existing recovery root")
    output_root.mkdir(parents=True, exist_ok=args.resume)
    selection_path = output_root / "candidate_selection.jsonl"
    selected_now = select_evaluator_only_candidates(source_root)
    if len(selected_now) != args.expected_candidate_count:
        raise ValueError(f"selected {len(selected_now)} conditions; expected {args.expected_candidate_count}")
    if selection_path.exists():
        selection = _read_jsonl(selection_path)
        if selection != selected_now:
            raise ValueError("resume selection differs from immutable source population")
    else:
        selection = selected_now
        _write_jsonl_atomic(selection_path, selection)
    for candidate in selection:
        _validate_binding(candidate)
    loaded = load_experiment_credentials(dict(os.environ))
    stack = load_model_stack_profile(profile_path, environ=loaded, require_credentials=True)
    judge = stack.roles["judge"]
    os.environ.update(stack.runtime_environment())
    os.environ["MASP_OPENAI_RETRY_BASE_DELAY"] = "0"
    snapshot = output_root / "model_profile_snapshot.yaml"
    if snapshot.exists():
        if _sha256_file(snapshot) != _sha256_file(profile_path):
            raise ValueError("resume model profile differs from snapshot")
    else:
        shutil.copyfile(profile_path, snapshot)
    registration = output_root / "evaluator_recovery_registration.json"
    source_registration = source_root / "diagnostic_registration.json"
    identity = {"schema_version": "ha-evaluator-only-recovery-provenance-v2", "artifact_classification": "diagnostic", "formal_aggregation_eligible": False, "recovery_scope": "evaluator_missing_only_v2", "source_run_root": str(source_root), "source_registration_sha256": _sha256_file(source_registration) if source_registration.is_file() else None, "candidate_selection_sha256": _sha256_file(selection_path), "selected_candidate_count": len(selection), "model_profile_sha256": stack.source_sha256, "judge_identity": judge.public_dict(), "judge_alias_immutable": False, "recovery_script_sha256": _sha256_file(Path(__file__).resolve()), "response_bodies_persisted": False, "reasoning_persisted": False, "transport_max_attempts": judge.transport_max_attempts}
    if registration.exists():
        observed = json.loads(registration.read_text(encoding="utf-8")); observed.pop("created_at_utc", None)
        if observed != identity:
            raise ValueError("resume provenance differs")
    else:
        _write_json_atomic(registration, {**identity, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    _preflight_recorded(artifact_path=output_root / "endpoint_preflight.jsonl", base_url=judge.base_url, model=judge.model, api_key=judge.api_key, label="HarnessAudit evaluator-only diagnostic recovery", timeout=judge.timeout_seconds, api_mode=judge.api_mode, request_extra_body=judge.request_extra_body, temperature=judge.temperature)
    attempts_path = output_root / "evaluator_result_attempts.jsonl"
    latest = _latest_attempts(attempts_path)
    expected_keys = {str(item["condition_key"]) for item in selection}
    if not set(latest).issubset(expected_keys):
        raise ValueError("resume results contain an out-of-selection condition")
    pending = [item for item in selection if latest.get(str(item["condition_key"]), {}).get("evaluator_status") != "passed"]
    calls = asyncio.Semaphore(max(1, args.judge_workers)); conditions = asyncio.Semaphore(max(1, args.condition_workers))
    futures = [_evaluate_safely(item, judge_model=judge.model, call_semaphore=calls, condition_semaphore=conditions) for item in pending]
    ledger_path = output_root / "judge_request_ledger.jsonl"
    done = 0
    for future in asyncio.as_completed(futures):
        result, ledger = await future
        key = str(result["condition_key"]); result["recovery_attempt"] = int((latest.get(key) or {}).get("recovery_attempt") or 0) + 1
        for row in ledger:
            row["recovery_attempt"] = result["recovery_attempt"]; _append_jsonl_durable(ledger_path, row)
        _append_jsonl_durable(attempts_path, result); latest[key] = result; done += 1
        _write_jsonl_atomic(output_root / "recovery_results.jsonl", (latest[key] for key in sorted(latest)))
        if done % 5 == 0 or done == len(pending):
            print(f"evaluator recovery progress: {done}/{len(pending)} this attempt; {len(latest)}/{len(selection)} observed", flush=True)
    results = [latest[key] for key in sorted(latest)]
    summary = summarize_evaluator_only_recovery(results, expected_count=len(selection))
    ledgers = _read_jsonl(ledger_path) if ledger_path.is_file() else []
    summary.update({"judge_identity": judge.public_dict(), "candidate_selection_sha256": _sha256_file(selection_path), "judge_request_summary": {"call_count": len(ledgers), "status_counts": dict(sorted(Counter(str(item.get("status") or "") for item in ledgers).items())), "http_status_counts": dict(sorted(Counter(str(item.get("http_status")) for item in ledgers if item.get("http_status") is not None).items())), "transport_attempt_count": sum(int(item.get("transport_attempt_count") or 0) for item in ledgers), "semantic_attempt_count": sum(int(item.get("semantic_attempt_count") or 0) for item in ledgers), "token_usage": {field: sum(int((item.get("token_usage") or {}).get(field) or 0) for item in ledgers) for field in ("prompt_tokens", "completion_tokens", "total_tokens")}}})
    _write_json_atomic(output_root / "recovery_summary.json", summary)
    scan = _active_credential_scan([output_root, *args.credential_scan_path]); scan.update({"type": "ha_evaluator_only_recovery_active_credential_scan", "scope": "recovery_output_root_plus_declared_paths_v2"})
    _write_json_atomic(output_root / "active_credential_scan.json", scan)
    if scan.get("status") != "passed":
        return 7
    return 0 if summary["all_evaluator_scores_recovered"] else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, required=True)
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--condition-workers", type=int, default=2)
    parser.add_argument("--credential-scan-path", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
