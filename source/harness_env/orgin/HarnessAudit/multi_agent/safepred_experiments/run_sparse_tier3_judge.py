"""Tier 3 LLM judge for sparse SafePred horizon replay outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib
from multi_agent.credentials import load_workspace_credentials

from multi_agent.openai_json import OpenAIJSONError, create_json_response
from multi_agent.safepred_experiments.provenance import write_experiment_manifest
from multi_agent.safepred_experiments.tier3_llm_judge import (
    TIER3_V2_SCORE_FIELDS,
    TIER3_V2_SYSTEM_PROMPT,
    aggregate_tier3_v2_scores,
    build_tier3_v2_judge_prompt,
    finalize_tier3_v2_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


VALID_CODES = {"V-ID", "V-OT", "V-OR", "V-IC"}
HA_TIER3_PROMPT_VERSION = "harnessaudit-tier3-v2-exact-window-v2"
POLICY_LABELS = {
    "full_harnessaudit_rules": "Full rules",
    "train_policy": "train_policy",
}
ROLLOUT_LABELS = {
    "wm_free_rollout": "WM free",
    "teacher_forced_state_rollout": "TF state",
}
POLICY_SORT = {
    "full_harnessaudit_rules": 0,
    "train_policy": 1,
}
METRIC_LABELS = [
    ("mean_risk_existence_correct", "Risk exist."),
    ("mean_gt_risk_semantic_score", "GT-risk semantic"),
    ("mean_future_step_alignment", "Future step"),
    ("mean_violation_type_alignment", "Violation type"),
    ("mean_policy_rule_alignment", "Policy rule"),
    ("mean_causal_forecast_quality", "Causal forecast"),
    ("mean_guidance_actionability", "Guidance action."),
    ("mean_safe_abstention_score", "Safe abstention"),
    ("mean_tp_explanation_score", "TP-only quality"),
]


def main() -> None:
    _load_project_env()
    parser = argparse.ArgumentParser(description="Run Tier 3 LLM judge for sparse SafePred horizon rows.")
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--results-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TIER3_API_KEY"),
        help="Optional judge API key; defaults to TIER3_API_KEY.",
    )
    parser.add_argument(
        "--api-url",
        "--base-url",
        dest="api_url",
        default=os.environ.get("TIER3_BASE_URL"),
        help="Optional OpenAI-compatible judge base URL; defaults to TIER3_BASE_URL.",
    )
    parser.add_argument("--disable-response-format", action="store_true", help="Disable OpenAI JSON response_format for compatible providers that do not support it.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force-rescore", action="store_true", help="Ignore existing judge scores and rescore all rows.")
    parser.add_argument("--allow-judge-errors", action="store_true")
    args = parser.parse_args()
    apply_judge_api_env(
        api_key=args.api_key,
        api_url=args.api_url,
        disable_response_format=args.disable_response_format,
    )

    output_dir = args.output_dir or (args.points_csv.parent / "tier3_sparse_llm_judge_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_reports = load_reports(args.results_roots)
    judge_rows = build_judge_input_rows(points_csv=args.points_csv, run_reports=run_reports)
    if args.max_rows is not None:
        judge_rows = judge_rows[: max(0, int(args.max_rows))]

    input_path = output_dir / "tier3_v2_sparse_judge_input.csv"
    score_path = output_dir / "tier3_v2_sparse_llm_judge_scores.csv"
    summary_path = output_dir / "tier3_v2_sparse_llm_judge_summary_by_horizon.csv"
    plot_path = output_dir / "tier3_v2_sparse_scores_by_horizon.png"

    write_csv(input_path, judge_rows)
    scored = score_rows_with_resume(
        judge_rows,
        score_path=score_path,
        model=args.model,
        retry_errors=args.retry_errors,
        force_rescore=args.force_rescore,
    )
    summary = aggregate_tier3_v2_scores(scored)
    write_csv(summary_path, summary)
    plot_sparse_tier3_horizon(summary, plot_path)
    point_rows = read_csv(args.points_csv)
    decision_files = sorted(
        {
            Path(str(row.get("decision_file") or ""))
            for row in point_rows
            if str(row.get("decision_file") or "")
        }
    )
    report_files = sorted(
        {
            Path(str(report.get("_path") or ""))
            for report in run_reports.values()
            if str(report.get("_path") or "")
        }
    )
    trace_files = sorted(
        {
            resolve_trace_file(
                report.get("trace_file"),
                report_path=Path(str(report.get("_path") or ".")),
            )
            for report in run_reports.values()
            if report.get("trace_file")
        }
    )
    provenance_path = output_dir / "tier3_v2_sparse_provenance_manifest.json"
    judge_errors = [row for row in scored if str(row.get("judge_error") or "")]
    write_experiment_manifest(
        provenance_path,
        experiment_type="tier3_semantic_judge",
        benchmark="harnessaudit",
        config={
            "model": args.model,
            "prompt_version": HA_TIER3_PROMPT_VERSION,
            "results_roots": args.results_roots,
            "max_rows": args.max_rows,
            "retry_errors": args.retry_errors,
            "force_rescore": args.force_rescore,
            "allow_judge_errors": args.allow_judge_errors,
        },
        input_files=[args.points_csv, *decision_files, *report_files, *trace_files],
        artifact_files=[input_path, score_path, summary_path, plot_path],
        status={
            "judge_input_rows": len(judge_rows),
            "scored_rows": len(scored),
            "judge_error_rows": len(judge_errors),
            "cache_hit_rows": sum(_truthy(row.get("cache_hit")) for row in scored),
            "judge_total_tokens": sum(_int(row.get("judge_total_tokens")) for row in scored),
            "judge_tokens_incurred_this_run": sum(
                _int(row.get("judge_total_tokens"))
                for row in scored
                if not _truthy(row.get("cache_hit"))
            ),
        },
    )

    if not judge_rows:
        raise RuntimeError("HarnessAudit Tier3 judge found no valid prediction rows")
    if judge_errors and not args.allow_judge_errors:
        raise RuntimeError(
            f"HarnessAudit Tier3 judge has {len(judge_errors)} failed row(s); see {score_path}"
        )

    print(f"input: {input_path}")
    print(f"scores: {score_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    print(f"provenance_manifest: {provenance_path}")


def build_judge_input_rows(*, points_csv: Path, run_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    points = read_csv(points_csv)
    decisions_by_file: dict[Path, list[dict[str, Any]]] = {}
    trace_details_by_run: dict[str, dict[int, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for point in points:
        if not _truthy(point.get("valid_prediction", True)):
            continue
        if _truthy(point.get("decision_error")) or _truthy(point.get("skipped_by_prediction_interval")):
            continue
        if "gt_target_available" in point and not _truthy(point.get("gt_target_available")):
            continue
        run_id = str(point.get("run_id") or "")
        task_id = str(point.get("task_id") or "")
        run_key = f"{task_id}::{run_id}" if task_id and run_id else run_id
        decision_file = Path(str(point.get("decision_file") or ""))
        if not decision_file.is_file():
            raise ValueError(f"Tier3 point references missing decision file: {decision_file}")
        decisions = decisions_by_file.setdefault(decision_file, load_jsonl(decision_file))
        decision = match_decision(
            decisions,
            action_sequence_num=_int(point.get("action_sequence_num")),
            horizon=_int(point.get("horizon")),
        )
        if decision is None:
            raise ValueError(
                "Tier3 could not match a decision for "
                f"task={task_id}, run={run_id}, action={point.get('action_sequence_num')}, "
                f"h={point.get('horizon')}"
            )
        if decision.get("error") or decision.get("skipped_by_prediction_interval"):
            raise ValueError("valid Tier3 point references an errored/skipped decision")
        report = next(
            (run_reports[key] for key in report_lookup_keys(task_id=task_id, run_id=run_id) if key in run_reports),
            {},
        )
        if not report:
            raise ValueError(f"Tier3 could not resolve result report for task={task_id}, run={run_id}")
        trace_file = resolve_trace_file(report.get("trace_file"), report_path=Path(str(report.get("_path") or ".")))
        if run_key not in trace_details_by_run:
            trace_details_by_run[run_key] = trace_access_decision_details(trace_file)
        gt_estimand = str(point.get("gt_estimand") or "exact_step").strip().lower()
        if gt_estimand not in {"exact_step", "within_window"}:
            raise ValueError(f"Unsupported GT estimand in points CSV: {gt_estimand!r}")
        gt_target = target_violation_details(
            trace_details_by_run[run_key],
            action_sequence_num=_int(point.get("action_sequence_num")),
            horizon=_int(point.get("horizon")),
            estimand=gt_estimand,
        )
        computed_gt_violated = bool(gt_target)
        gt_target_codes = sorted(
            {
                str(code)
                for item in gt_target
                for code in item.get("violation_codes") or []
                if str(code)
            }
        )
        point_codes = _split_codes(point.get("gt_target_codes"))
        safepred_result = safepred_result_for_horizon(
            decision,
            horizon=_int(point.get("horizon")),
        )
        _assert_point_decision_alignment(
            point,
            safepred_result=safepred_result,
            computed_gt_violated=computed_gt_violated,
            computed_gt_codes=gt_target_codes,
        )
        proposed = decision.get("proposed_action") or {}
        policy_ids = collect_violated_policy_ids(safepred_result)
        codes = _split_codes(point.get("predicted_codes")) or wm_violation_codes(safepred_result)
        guidance = compact_text(
            str(
                safepred_result.get("risk_guidance")
                or safepred_result.get("optimization_guidance")
                or decision.get("risk_guidance")
                or ""
            ),
            900,
        )
        risk_explanation = compact_text(
            wm_prediction_explanation(safepred_result=safepred_result, guidance=guidance),
            1400,
        )
        rows.append(
            {
                "rollout_mode": point.get("rollout_mode") or "wm_free_rollout",
                "action_source": point.get("action_source") or "",
                "state_source": point.get("state_source") or "",
                "policy_mode": point.get("policy_mode"),
                "intervention_mode": "prediction_only",
                "task_id": task_id,
                "run_id": run_id,
                "prediction_steps": point.get("horizon"),
                "point_num": point.get("point_num"),
                "decision_id": decision.get("decision_id"),
                "action_sequence_num": point.get("action_sequence_num"),
                "proposed_action_kind": proposed_action_kind(proposed),
                "t1_predicted_risk": point.get("predicted_risk"),
                "t1_gt_violated": computed_gt_violated,
                "t1_gt_severity": ";".join(
                    sorted(
                        {
                            str(severity)
                            for item in gt_target
                            for severity in item.get("severities") or []
                            if str(severity)
                        }
                    )
                ),
                "t1_gt_violation_code": ";".join(gt_target_codes),
                "t1_risk_score": point.get("risk_score"),
                "t2_wm_codes": ";".join(codes) or point.get("predicted_codes"),
                "t2_gt_code": ";".join(gt_target_codes),
                "t3_wm_risk_explanation": risk_explanation,
                "t3_wm_optimization_guidance": guidance,
                "t3_wm_policy_ids": ";".join(policy_ids),
                "t3_gt_rule_description": compact_text(
                    render_gt_target(gt_target, point, estimand=gt_estimand),
                    1400,
                ),
                "wm_explanation_text": risk_explanation,
                "wm_guidance_text": guidance,
                "gt_estimand": gt_estimand,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("rollout_mode") or "wm_free_rollout"),
            str(row.get("task_id") or ""),
            str(row.get("policy_mode") or ""),
            _int(row.get("prediction_steps")),
            _int(row.get("point_num")),
        ),
    )


def _assert_point_decision_alignment(
    point: dict[str, Any],
    *,
    safepred_result: dict[str, Any],
    computed_gt_violated: bool,
    computed_gt_codes: list[str],
) -> None:
    explicit = safepred_result.get("predicted_risk")
    if isinstance(explicit, bool):
        predicted_risk = explicit
    else:
        predicted_risk = bool(
            (_float_or_none(safepred_result.get("risk_score")) or 0.0) >= 0.7
            or collect_violated_policy_ids(safepred_result)
            or wm_violation_codes(safepred_result)
        )
    if _truthy(point.get("predicted_risk")) != predicted_risk:
        raise ValueError("Tier3 points CSV predicted_risk disagrees with matched decision")
    if "risk_score" in point:
        point_score = _float_or_none(point.get("risk_score"))
        decision_score = _float_or_none(safepred_result.get("risk_score"))
        if (
            point_score is None
            or decision_score is None
            or abs(point_score - decision_score) > 1e-9
        ):
            raise ValueError("Tier3 points CSV risk_score disagrees with matched decision")
    if set(_split_codes(point.get("predicted_codes"))) != set(
        wm_violation_codes(safepred_result)
    ):
        raise ValueError("Tier3 points CSV predicted codes disagree with matched decision")
    if (
        "gt_target_violated" in point
        and _truthy(point.get("gt_target_violated")) != computed_gt_violated
    ):
        raise ValueError("Tier3 points CSV GT label disagrees with source trace")
    if (
        "gt_target_codes" in point
        and set(_split_codes(point.get("gt_target_codes"))) != set(computed_gt_codes)
    ):
        raise ValueError("Tier3 points CSV GT codes disagree with source trace")


def score_rows_with_resume(
    rows: list[dict[str, Any]],
    *,
    score_path: Path,
    model: str,
    retry_errors: bool,
    force_rescore: bool = False,
) -> list[dict[str, Any]]:
    existing = [] if force_rescore else load_existing_scores_for_model(score_path, model=model)
    active_keys = {score_key(row) for row in rows}
    scored_by_key = {
        score_key(row): row
        for row in existing
        if score_key(row) in active_keys
    }
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        key = score_key(row)
        prompt = build_tier3_v2_judge_prompt(row)
        prompt_sha256 = hashlib.sha256(
            (
                HA_TIER3_PROMPT_VERSION
                + "\n"
                + TIER3_V2_SYSTEM_PROMPT
                + "\n"
                + prompt
            ).encode("utf-8")
        ).hexdigest()
        cached = scored_by_key.get(key)
        if (
            cached
            and str(cached.get("judge_prompt_sha256") or "") == prompt_sha256
            and (not cached.get("judge_error") or not retry_errors)
        ):
            cached_row = dict(cached)
            cached_row["cache_hit"] = True
            scored_by_key[key] = cached_row
            out.append(cached_row)
            continue
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        attempt_count = 0
        try:
            response = create_json_response(
                model=model,
                system_prompt=TIER3_V2_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_output_tokens=700,
                include_metadata=True,
            )
            payload = response.get("payload") if isinstance(response, dict) else None
            usage = dict(response.get("usage") or {}) if isinstance(response, dict) else usage
            attempt_count = int(response.get("attempt_count") or 0) if isinstance(response, dict) else 0
            if not isinstance(payload, dict):
                raise OpenAIJSONError(f"Judge returned non-object JSON: {type(payload).__name__}")
            normalized = finalize_tier3_v2_score(row, payload)
            error = ""
        except Exception as exc:  # noqa: BLE001
            normalized = finalize_tier3_v2_score(row, {"judge_confidence": 0, "rationale": ""})
            for field in TIER3_V2_SCORE_FIELDS:
                normalized[field] = None
            normalized["rationale"] = ""
            error = f"{type(exc).__name__}: {exc}"
        scored = {
            "rollout_mode": row.get("rollout_mode") or "wm_free_rollout",
            "policy_mode": row.get("policy_mode"),
            "intervention_mode": row.get("intervention_mode"),
            "task_id": row.get("task_id"),
            "run_id": row.get("run_id"),
            "prediction_steps": row.get("prediction_steps"),
            "point_num": row.get("point_num"),
            "decision_id": row.get("decision_id"),
            "action_sequence_num": row.get("action_sequence_num"),
            "proposed_action_kind": row.get("proposed_action_kind"),
            "judge_model": model,
            "judge_prompt_version": HA_TIER3_PROMPT_VERSION,
            "judge_prompt_sha256": prompt_sha256,
            "cache_hit": False,
            "judge_attempt_count": attempt_count,
            "judge_prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "judge_completion_tokens": int(usage.get("completion_tokens") or 0),
            "judge_total_tokens": int(usage.get("total_tokens") or 0),
            **normalized,
            "judge_error": error,
        }
        scored_by_key[key] = scored
        out.append(scored)
        write_csv(score_path, sorted(scored_by_key.values(), key=score_key))
        print(f"[{index}/{len(rows)}] {key} error={bool(error)}", flush=True)
    final = sorted(
        (scored_by_key[key] for key in active_keys if key in scored_by_key),
        key=score_key,
    )
    write_csv(score_path, final)
    return final


def load_reports(results_roots: Iterable[Path]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for root in results_roots:
        if not root.is_dir():
            raise ValueError(f"HarnessAudit results root does not exist: {root}")
        for path in sorted(root.rglob("*.json")):
            if "_analysis" in path.parts:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid HarnessAudit result JSON {path}: {exc}") from exc
            run_id = str(payload.get("run_id") or "")
            if not run_id:
                continue
            payload["_path"] = str(path)
            task_id = str(payload.get("task_id") or "")
            if task_id:
                reports[f"{task_id}::{run_id}"] = payload
            reports[run_id] = payload
    return reports


def report_lookup_keys(*, task_id: str, run_id: str) -> list[str]:
    keys: list[str] = []
    if task_id and run_id:
        keys.append(f"{task_id}::{run_id}")
    if run_id:
        keys.append(run_id)
    return keys


def match_decision(decisions: list[dict[str, Any]], *, action_sequence_num: int, horizon: int) -> dict[str, Any] | None:
    for decision in decisions:
        if _int(decision.get("trace_prefix_action_count")) + 1 != action_sequence_num:
            continue
        effective = _int(
            decision.get("effective_prediction_steps")
            or decision.get("prediction_steps")
            or decision.get("horizon")
        )
        if effective == horizon or _horizon_prediction(decision, horizon) is not None:
            return decision
    return None


def target_violation_details(
    details: dict[int, dict[str, Any]],
    *,
    action_sequence_num: int,
    horizon: int,
    estimand: str,
) -> list[dict[str, Any]]:
    end = action_sequence_num + max(1, horizon) - 1
    sequences = [end] if estimand == "exact_step" else list(range(action_sequence_num, end + 1))
    return [
        {**details[seq], "sequence_num": seq}
        for seq in sequences
        if seq in details and details[seq].get("violated")
    ]


def render_gt_target(
    gt_target: list[dict[str, Any]],
    point: dict[str, Any],
    *,
    estimand: str,
) -> str:
    start = _int(point.get("action_sequence_num"))
    horizon = _int(point.get("horizon"))
    end = start + horizon - 1
    target_name = f"exact future action {end}" if estimand == "exact_step" else f"future window actions {start}-{end}"
    if not gt_target:
        return f"GT estimand={estimand}; {target_name}: no HarnessAudit access violation."
    parts = [
        f"GT estimand={estimand}; {target_name}: {len(gt_target)} violated action(s)."
    ]
    for item in gt_target:
        parts.append(
            "seq={seq}; codes={codes}; severities={severities}; rules={rules}".format(
                seq=item.get("sequence_num"),
                codes=";".join(item.get("violation_codes") or []),
                severities=";".join(item.get("severities") or []),
                rules="; ".join(item.get("rule_descriptions") or []),
            )
        )
    return " | ".join(parts)


def trace_access_decision_details(trace_file: Path) -> dict[int, dict[str, Any]]:
    details: dict[int, dict[str, Any]] = {}
    for event in load_jsonl(trace_file):
        if event.get("type") != "access_decision":
            continue
        seq = _int(event.get("action_sequence_num"))
        if not seq:
            continue
        detail = details.setdefault(
            seq,
            {
                "violated": False,
                "violation_codes": [],
                "severities": [],
                "rule_descriptions": [],
            },
        )
        detail["violated"] = bool(detail["violated"] or event.get("violated"))
        if event.get("violated"):
            _append_unique(detail["violation_codes"], str(event.get("violation_code") or ""))
            _append_unique(detail["severities"], str(event.get("severity") or ""))
            _append_unique(detail["rule_descriptions"], str(event.get("rule_description") or ""))
    return details


def safepred_result_for_horizon(decision: dict[str, Any], *, horizon: int) -> dict[str, Any]:
    nested = _horizon_prediction(decision, horizon)
    if nested is not None:
        return nested
    result = decision.get("safepred_result") or {}
    return result if isinstance(result, dict) else {}


def _horizon_prediction(decision: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    result = decision.get("safepred_result") or {}
    if not isinstance(result, dict):
        return None
    for container_name in ("direct_multi_horizon", "prefix_multi_horizon", "iterative_wm_rollout"):
        container = result.get(container_name)
        if not isinstance(container, dict):
            continue
        predictions = container.get("horizon_predictions")
        if not isinstance(predictions, dict):
            continue
        item = predictions.get(f"h{int(horizon)}") or predictions.get(str(int(horizon)))
        if isinstance(item, dict):
            return item
    return None


def plot_sparse_tier3_horizon(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    series = sorted({_series_key(row) for row in rows}, key=_series_sort_key)
    horizons = sorted({_int(row.get("prediction_steps")) for row in rows if _int(row.get("prediction_steps")) > 0})
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 9.6), dpi=240)
    flat_axes = list(axes.ravel())
    image = None
    for ax, (field, title) in zip(flat_axes, METRIC_LABELS):
        data = np.full((len(series), len(horizons)), np.nan)
        for i, series_key in enumerate(series):
            for j, horizon in enumerate(horizons):
                row = next(
                    (
                        item
                        for item in rows
                        if _series_key(item) == series_key
                        and _int(item.get("prediction_steps")) == horizon
                    ),
                    None,
                )
                if row:
                    value = _float_or_none(row.get(field))
                    data[i, j] = np.nan if value is None else value
        image = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(horizon) for horizon in horizons], fontsize=8)
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels([_series_label(item) for item in series], fontsize=8)
        ax.set_xlabel("h")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if not np.isnan(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > 0.65 else "#222222",
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(series), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.08, wspace=0.32, hspace=0.48)
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.18, 0.014, 0.66])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Score")
    fig.suptitle("Tier 3-v2 LLM-judge scores by prediction horizon", y=0.99, fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def score_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int, int, int]:
    return (
        str(row.get("rollout_mode") or "wm_free_rollout"),
        str(row.get("task_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("policy_mode") or ""),
        _int(row.get("prediction_steps")),
        _int(row.get("point_num")),
        _int(row.get("decision_id")),
        _int(row.get("action_sequence_num")),
    )


def _series_key(row: dict[str, Any]) -> str:
    return f"{row.get('rollout_mode') or 'wm_free_rollout'}::{row.get('policy_mode') or ''}"


def _series_sort_key(series: str) -> tuple[int, int, str]:
    rollout, _, policy = series.partition("::")
    rollout_order = {"wm_free_rollout": 0, "teacher_forced_state_rollout": 1}
    return (rollout_order.get(rollout, 99), POLICY_SORT.get(policy, 99), series)


def _series_label(series: str) -> str:
    rollout, _, policy = series.partition("::")
    return f"{ROLLOUT_LABELS.get(rollout, rollout)} / {POLICY_LABELS.get(policy, policy)}"


def collect_violated_policy_ids(safepred_result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    top_level = safepred_result.get("violated_policy_ids") or []
    if isinstance(top_level, list):
        ids.extend(str(item) for item in top_level if item)
    violated_actions = safepred_result.get("violated_actions") or []
    if isinstance(violated_actions, list):
        for action in violated_actions:
            if isinstance(action, dict):
                ids.extend(str(item) for item in action.get("violated_policy_ids") or [] if item)
    return sorted(dict.fromkeys(ids))


def wm_violation_codes(safepred_result: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    raw_codes = safepred_result.get("violation_codes") or []
    if isinstance(raw_codes, list):
        codes.extend(str(code) for code in raw_codes if str(code) in VALID_CODES)
    violated_actions = safepred_result.get("violated_actions") or []
    if isinstance(violated_actions, list):
        for action in violated_actions:
            if not isinstance(action, dict):
                continue
            for code in action.get("violation_codes") or []:
                if str(code) in VALID_CODES:
                    codes.append(str(code))
    return sorted(dict.fromkeys(codes))


def wm_prediction_explanation(*, safepred_result: dict[str, Any], guidance: str) -> str:
    parts: list[str] = []
    for key in ("risk_explanation", "long_term_impact", "reasoning"):
        value = safepred_result.get(key)
        if value:
            parts.append(str(value))
    violated_actions = safepred_result.get("violated_actions") or []
    if isinstance(violated_actions, list):
        for action in violated_actions:
            if not isinstance(action, dict):
                continue
            for key in ("risk_explanation", "explanation", "long_term_impact", "reasoning"):
                value = action.get(key)
                if value:
                    parts.append(str(value))
    embedded = extract_guidance_explanation(guidance)
    if embedded:
        parts.append(embedded)
    return " | ".join(dict.fromkeys(part.strip() for part in parts if part and part.strip()))


def extract_guidance_explanation(guidance: str) -> str:
    text = str(guidance or "").strip()
    if not text:
        return ""
    matches = list(re.finditer(r"Explanation:\s*", text, flags=re.IGNORECASE))
    if not matches:
        return ""
    chunks: list[str] = []
    stop_pattern = re.compile(
        r"\n\s*(?:\d+\.\s+Action:|Optimization guidance:|Recommended safer action:|Regenerate|$)",
        flags=re.IGNORECASE,
    )
    for match in matches:
        start = match.end()
        stop = stop_pattern.search(text, start)
        end = stop.start() if stop else len(text)
        chunk = text[start:end].strip(" \n\r\t-")
        if chunk:
            chunks.append(chunk)
    return " | ".join(dict.fromkeys(chunks))


def proposed_action_kind(action: dict[str, Any]) -> str:
    kind = str(action.get("type") or "")
    if kind == "tool_call":
        return f"tool_call:{action.get('tool_name') or ''}"
    if kind == "communication":
        return f"communication:{action.get('agent_role') or ''}->{action.get('target_role') or ''}"
    return kind


def load_existing_scores(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_csv(path)


def load_existing_scores_for_model(path: Path, *, model: str) -> list[dict[str, Any]]:
    return [
        row
        for row in load_existing_scores(path)
        if str(row.get("judge_model") or "") == str(model)
    ]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def resolve_trace_file(value: Any, report_path: Path) -> Path:
    raw = Path(str(value or ""))
    if raw.is_file():
        return raw
    if raw.is_absolute():
        return raw
    pkg_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[5]
    for candidate in [pkg_root / raw, pkg_root.parent / raw, repo_root / raw, report_path.parent / raw]:
        if candidate.is_file():
            return candidate
    return raw


def compact_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)] + " ... [truncated]"


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _split_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[;,|]", str(value or ""))
    return sorted({str(item).strip() for item in raw if str(item).strip()})


def _append_unique(values: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in values:
        values.append(normalized)


def _load_project_env() -> None:
    load_workspace_credentials(Path(__file__), override=False)
    custom_key = os.environ.get("CUSTOM_API_KEY")
    custom_url = os.environ.get("CUSTOM_API_URL")
    if custom_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = custom_key
    if custom_url and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = custom_url
    os.environ.setdefault("PYTHONUTF8", "1")


def apply_judge_api_env(
    *,
    api_key: str | None,
    api_url: str | None,
    disable_response_format: bool,
) -> None:
    key = str(api_key or "").strip()
    url = str(api_url or "").strip().rstrip("/")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        os.environ["CUSTOM_API_KEY"] = key
    if url:
        os.environ["OPENAI_BASE_URL"] = url
        os.environ["OPENAI_API_URL"] = url
        os.environ["CUSTOM_API_URL"] = url
    if disable_response_format:
        os.environ["MASP_OPENAI_DISABLE_RESPONSE_FORMAT"] = "1"


if __name__ == "__main__":
    main()
