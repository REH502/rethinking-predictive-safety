"""Formal ISBench x SafePred direct-vision predict-only runner.

This thin entry point wires together:
1. report audit / unique-task selection,
2. optional VLM endpoint probe,
3. the ISBench predict-only pipeline.

It does not change the experimental prompt or scoring logic.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from multi_agent.safepred_experiments.isbench_predict_only_pipeline import (
    DEFAULT_DIRECT_VISION_MODEL,
    DEFAULT_POLICY_MODES,
    DEFAULT_ROLLOUT_MODES,
    VALID_POLICY_MODES,
    VALID_ROLLOUT_MODES,
    ISBenchPipelineConfig,
    PointSelectionConfig,
    parse_horizons,
    preflight_formal_label_support,
    prepare_formal_offline_contract,
    run_pipeline,
)
from multi_agent.safepred_experiments.isbench_probe_vision_models import probe_models
from multi_agent.safepred_experiments.isbench_report_audit import audit_reports
from multi_agent.safepred_experiments.isbench_tier3_judge import (
    apply_judge_api_env,
    run_isbench_tier3_judge,
)


def select_recommended_report_paths(
    audit_payload: dict[str, Any],
    *,
    require_unique_tasks: int,
) -> list[str]:
    if not audit_payload.get("meets_unique_task_requirement", True):
        found = int(audit_payload.get("unique_replayable_task_count") or 0)
        raise ValueError(
            f"Formal ISBench run requires {require_unique_tasks} unique replayable tasks, "
            f"but report audit found {found}."
        )
    reports = audit_payload.get("recommended_reports") or []
    selected = [str(item.get("path") or "") for item in reports if str(item.get("path") or "").strip()]
    selected = selected[: int(require_unique_tasks)]
    if len(selected) < int(require_unique_tasks):
        raise ValueError(
            f"Formal ISBench run requires {require_unique_tasks} recommended reports, "
            f"but audit selected {len(selected)}."
        )
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run formal ISBench x SafePred direct-vision predict-only experiment.")
    parser.add_argument("--report-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--require-unique-tasks", type=int, default=3)
    parser.add_argument(
        "--policy-modes",
        nargs="+",
        default=list(DEFAULT_POLICY_MODES),
        choices=sorted(VALID_POLICY_MODES),
    )
    parser.add_argument("--rollout-modes", nargs="+", default=list(DEFAULT_ROLLOUT_MODES), choices=sorted(VALID_ROLLOUT_MODES))
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--max-h", type=int, default=6)
    parser.add_argument("--point-scheme", choices=["start_mid", "all", "first", "interval", "explicit"], default="all")
    parser.add_argument("--point-count", type=int, default=4)
    parser.add_argument("--point-interval", type=int, default=1)
    parser.add_argument("--explicit-sequences", nargs="*", type=int, default=[])
    parser.add_argument("--model", "--wm-model", dest="model", default=DEFAULT_DIRECT_VISION_MODEL)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Immutable WM weights or deployment revision recorded in the contract.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-url", "--base-url", dest="api_url", default=None)
    parser.add_argument(
        "--wm-max-images",
        type=int,
        default=0,
        help=(
            "Maximum current-observation views sent to the WM; 0 is the formal default "
            "and sends every frozen current view. A positive cap must cover every "
            "selected point or the run fails before model access."
        ),
    )
    parser.add_argument("--wm-image-max-side", type=int, default=768)
    parser.add_argument("--wm-image-quality", type=int, default=85)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--allow-cross-domain-train-policy", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--actor-trace-seed", type=int, default=0)
    parser.add_argument("--experiment-seed", type=int, default=0)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--compositionality-manifest", type=Path, required=True)
    parser.add_argument(
        "--faithfulness-adjudication-protocol", type=Path, required=True
    )
    parser.add_argument("--adjudicator-blinding-protocol", type=Path, required=True)
    parser.add_argument("--run-tier3-judge", action="store_true")
    parser.add_argument("--tier3-model", "--judge-model", dest="tier3_model", default=None)
    parser.add_argument("--tier3-api-key", default=None)
    parser.add_argument("--tier3-api-url", "--tier3-base-url", dest="tier3_api_url", default=None)
    parser.add_argument("--tier3-disable-response-format", action="store_true")
    parser.add_argument("--tier3-force-rescore", action="store_true")
    parser.add_argument("--tier3-retry-errors", action="store_true")
    parser.add_argument("--tier3-max-rows", type=int, default=None)
    parser.add_argument(
        "--allow-decision-errors",
        action="store_true",
        help="Write aggregate artifacts even if some WM decisions error; metrics still exclude them.",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or Path("multi_agent/results") / f"isbench_formal_directwm_{timestamp}"
    trace_root = args.trace_root or Path("multi_agent/traces") / f"isbench_formal_directwm_{timestamp}"
    analysis_dir = args.analysis_dir or output_root / "_analysis"
    figure_dir = args.figure_dir or analysis_dir / "figures"

    audit_payload = audit_reports(
        args.report_roots,
        require_unique_tasks=args.require_unique_tasks,
        recommended_count=args.require_unique_tasks,
        tasks_dir=args.tasks_dir,
        require_complete_safety_labels=True,
        require_explicit_process_timing=True,
        require_image_quality=True,
        require_successful=False,
        require_generation_provenance=True,
    )
    selected_reports = select_recommended_report_paths(
        audit_payload,
        require_unique_tasks=args.require_unique_tasks,
    )
    api_key = args.api_key or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    pipeline_config = ISBenchPipelineConfig(
        reports=[Path(path) for path in selected_reports],
        tasks_dir=args.tasks_dir,
        policy_modes=list(args.policy_modes),
        rollout_modes=list(args.rollout_modes),
        horizons=parse_horizons(args.horizons, min_h=args.min_h, max_h=args.max_h),
        point_config=PointSelectionConfig(
            scheme=args.point_scheme,
            count=args.point_count,
            interval=args.point_interval,
            explicit_sequences=list(args.explicit_sequences),
        ),
        model=args.model,
        model_revision=args.model_revision,
        api_key=api_key,
        api_url=args.api_url,
        wm_input_mode="direct_vision",
        wm_disable_response_format=False,
        wm_max_images=args.wm_max_images,
        wm_image_max_side=args.wm_image_max_side,
        wm_image_quality=args.wm_image_quality,
        output_root=output_root,
        trace_root=trace_root,
        analysis_dir=analysis_dir,
        figure_dir=figure_dir,
        allow_missing_captions=True,
        train_policy_same_domain_only=not args.allow_cross_domain_train_policy,
        require_unique_tasks=args.require_unique_tasks,
        skip_existing=args.skip_existing,
        fail_on_decision_error=not args.allow_decision_errors,
        actor_trace_seed=args.actor_trace_seed,
        experiment_seed=args.experiment_seed,
        formal=True,
        preregistration_path=args.preregistration,
        compositionality_manifest_path=args.compositionality_manifest,
        faithfulness_adjudication_protocol_path=(
            args.faithfulness_adjudication_protocol
        ),
        adjudicator_blinding_protocol_path=args.adjudicator_blinding_protocol,
    )
    # This must precede audit artifact writes and the endpoint probe: populated
    # unregistered roots may not be retroactively labeled formal.
    prepare_formal_offline_contract(pipeline_config)
    # Complete the label-only cohort gate before the optional paid endpoint
    # probe. The core pipeline repeats and persists this census for provenance.
    preflight_formal_label_support(pipeline_config)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    audit_path = analysis_dir / "isbench_report_audit.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path = analysis_dir / "isbench_selected_reports.txt"
    selected_path.write_text("\n".join(selected_reports) + "\n", encoding="utf-8")

    if not args.skip_probe:
        probe_payload = probe_models(
            api_key=api_key,
            api_url=args.api_url,
            image_path=_first_report_image(Path(selected_reports[0])),
            models=[args.model],
            timeout_seconds=60.0,
        )
        probe_path = analysis_dir / "isbench_vision_model_probe.json"
        probe_path.write_text(json.dumps(probe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(probe_payload.get("ok_count") or 0) < 1:
            raise RuntimeError(f"Vision model probe failed; see {probe_path}")

    artifacts = run_pipeline(pipeline_config)
    if args.run_tier3_judge:
        tier3_model = args.tier3_model or args.model
        apply_judge_api_env(
            api_key=args.tier3_api_key or api_key,
            api_url=args.tier3_api_url or args.api_url,
            disable_response_format=args.tier3_disable_response_format,
        )
        tier3_artifacts = run_isbench_tier3_judge(
            points_csv=artifacts["points"],
            output_dir=analysis_dir / "tier3_isbench_llm_judge_v2",
            model=tier3_model,
            max_rows=args.tier3_max_rows,
            retry_errors=args.tier3_retry_errors,
            force_rescore=args.tier3_force_rescore,
        )
        artifacts.update({f"tier3_{name}": path for name, path in tier3_artifacts.items()})
    print(f"audit: {audit_path}")
    print(f"selected_reports: {selected_path}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


def _first_report_image(report_path: Path) -> Path:
    images = sorted(Path(report_path).parent.rglob("*.png"))
    if not images:
        raise ValueError(f"No observation images found for report: {report_path}")
    return images[0]


if __name__ == "__main__":
    main()
