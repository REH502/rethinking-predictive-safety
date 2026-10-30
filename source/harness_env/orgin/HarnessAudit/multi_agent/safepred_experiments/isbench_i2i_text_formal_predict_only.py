"""Formal ISBench x SafePred i2i+text WM predict-only runner."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from multi_agent.safepred_experiments.isbench_formal_predict_only import (
    _first_report_image,
    select_recommended_report_paths,
)
from multi_agent.safepred_experiments.isbench_i2i_text_wm_pipeline import (
    DEFAULT_I2I_MODEL,
    DEFAULT_POLICY_MODES,
    ISBenchI2ITextPipelineConfig,
    PointSelectionConfig,
    parse_horizons,
    run_pipeline,
)
from multi_agent.safepred_experiments.isbench_probe_vision_models import probe_models
from multi_agent.safepred_experiments.isbench_report_audit import audit_reports
from multi_agent.safepred_experiments.isbench_tier3_judge import (
    apply_judge_api_env,
    run_isbench_tier3_judge,
)
from multi_agent.safepred_experiments.isbench_predict_only_pipeline import (
    DEFAULT_DIRECT_VISION_MODEL,
    DIRECT_MULTI_HORIZON_ROLLOUT,
    VALID_ROLLOUT_MODES,
    prepare_formal_offline_contract,
)


I2I_FORMAL_ROLLOUT_MODES = sorted(
    VALID_ROLLOUT_MODES - {DIRECT_MULTI_HORIZON_ROLLOUT}
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run formal ISBench x SafePred i2i+text WM predict-only experiment.")
    parser.add_argument("--report-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--require-unique-tasks", type=int, default=3)
    parser.add_argument("--policy-modes", nargs="+", default=list(DEFAULT_POLICY_MODES))
    parser.add_argument(
        "--rollout-modes",
        nargs="+",
        default=["wm_free_rollout"],
        choices=I2I_FORMAL_ROLLOUT_MODES,
    )
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--max-h", type=int, default=6)
    parser.add_argument("--point-scheme", choices=["start_mid", "all", "first", "interval", "explicit"], default="start_mid")
    parser.add_argument("--point-count", type=int, default=3)
    parser.add_argument("--point-interval", type=int, default=1)
    parser.add_argument("--explicit-sequences", nargs="*", type=int, default=[])
    parser.add_argument("--model", "--text-model", dest="text_model", default=DEFAULT_DIRECT_VISION_MODEL)
    parser.add_argument("--i2i-model", default=DEFAULT_I2I_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-url", "--base-url", dest="api_url", default=None)
    parser.add_argument("--wm-disable-response-format", action="store_true")
    parser.add_argument("--wm-max-images", type=int, default=1)
    parser.add_argument("--wm-image-max-side", type=int, default=768)
    parser.add_argument("--wm-image-quality", type=int, default=85)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--allow-cross-domain-train-policy", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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
    output_root = args.output_root or Path("multi_agent/results") / f"isbench_i2i_text_wm_{timestamp}"
    trace_root = args.trace_root or Path("multi_agent/traces") / f"isbench_i2i_text_wm_{timestamp}"
    analysis_dir = args.analysis_dir or output_root / "_analysis"
    figure_dir = args.figure_dir or analysis_dir / "figures"

    audit_payload = audit_reports(
        args.report_roots,
        require_unique_tasks=args.require_unique_tasks,
        recommended_count=args.require_unique_tasks,
    )
    selected_reports = select_recommended_report_paths(
        audit_payload,
        require_unique_tasks=args.require_unique_tasks,
    )
    api_key = args.api_key or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    pipeline_config = ISBenchI2ITextPipelineConfig(
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
        model=args.text_model,
        text_model=args.text_model,
        i2i_model=args.i2i_model,
        api_key=api_key,
        api_url=args.api_url,
        wm_disable_response_format=args.wm_disable_response_format,
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
        experiment_seed=args.experiment_seed,
        formal=True,
        preregistration_path=args.preregistration,
        compositionality_manifest_path=args.compositionality_manifest,
        faithfulness_adjudication_protocol_path=(
            args.faithfulness_adjudication_protocol
        ),
        adjudicator_blinding_protocol_path=args.adjudicator_blinding_protocol,
    )
    # Freeze before any wrapper-owned artifact or endpoint probe.
    prepare_formal_offline_contract(pipeline_config)
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
            models=[args.text_model],
            timeout_seconds=60.0,
        )
        probe_path = analysis_dir / "isbench_i2i_text_model_probe.json"
        probe_path.write_text(json.dumps(probe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(probe_payload.get("ok_count") or 0) < 1:
            raise RuntimeError(f"Text/VLM model probe failed; see {probe_path}")

    artifacts = run_pipeline(pipeline_config)
    if args.run_tier3_judge:
        tier3_model = args.tier3_model or args.text_model
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


if __name__ == "__main__":
    main()
