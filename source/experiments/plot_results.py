"""Unified plotting for HarnessAudit and IS-Bench predict-only aggregates.

This module intentionally consumes aggregate CSV artifacts instead of importing
the benchmark pipelines.  Plotting therefore stays independent of actor,
simulator, and model-serving dependencies and can combine multiple completed
runs in one report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
MIN_AGGREGATE_SCHEMA_VERSION = 3
CANONICAL_AGGREGATE_NAMES = {
    "predict_only_aggregate_by_policy_horizon.csv",
    "isbench_predict_only_aggregate_by_policy_horizon.csv",
}
REQUIRED_COLUMNS = {"rollout_mode", "policy_mode", "horizon"}
TIER1_METRICS = (
    ("t1_accuracy", "Accuracy"),
    ("t1_recall", "Recall"),
)
TIER2_METRICS = (
    ("t2_code_match_rate", "Code match (all relevant rows)"),
    ("t2_mean_jaccard", "Mean Jaccard (all relevant rows)"),
)
CALIBRATION_METRICS = (
    ("brier_score", "Brier score"),
    ("ece", "Expected calibration error"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
)
SUPPORT_METRICS = (
    ("selected_prediction_points", "Valid prediction points"),
    ("task_count", "Tasks"),
    ("positive_support", "Positive GT support"),
    ("invalid_decisions", "Invalid decisions"),
)
TOKEN_METRICS = (
    ("prompt_tokens_per_point", "Input tokens / prediction"),
    ("completion_tokens_per_point", "Output tokens / prediction"),
    ("total_tokens_per_point", "Total tokens / prediction"),
)
SUMMARY_METRICS = (
    ("t1_accuracy", "Tier 1 Accuracy"),
    ("t1_recall", "Tier 1 Recall"),
    ("t2_code_match_rate", "Tier 2 Code match (all relevant rows)"),
)
FOUR_PANEL_METRICS = (
    ("t1_accuracy", "Accuracy"),
    ("t1_precision", "Precision"),
    ("t1_recall", "Recall"),
    ("t2_code_match_rate", "Tier 2 Code match (all relevant rows)"),
)
POLICY_STYLES = {
    "full_harnessaudit_rules": ("Full policy", "#1f77b4"),
    "full_isbench_policy_text": ("Full policy", "#1f77b4"),
    "train_policy": ("Train policy", "#ff7f0e"),
    "generic_embodied_policy": ("Generic policy", "#2ca02c"),
}
ROLLOUT_STYLES = {
    "direct_multi_horizon": ("Direct", "-", "o"),
    "direct_future_state_rollout": ("Direct", "-", "o"),
    "prefix_multi_horizon": ("Prefix", "--", "s"),
    "wm_free_rollout": ("WM-free", "--", "s"),
    "iterative_wm_rollout": ("Iterative WM", "--", "s"),
}
PRUNED_DISCOVERY_DIRS = {
    ".git",
    "__pycache__",
    "attempts",
    "benchmark",
    "conditions",
    "figures",
    "logs",
    "traces",
}


@dataclass(frozen=True)
class AggregateDataset:
    dataset_id: str
    benchmark: str
    source_csv: Path
    rows: list[dict[str, Any]]
    target_estimand: str
    formal_input: bool
    validation_warnings: tuple[str, ...]
    quality_path: Path | None = None


def discover_aggregate_csvs(inputs: Iterable[Path]) -> list[Path]:
    """Resolve files or directories to predict-only aggregate CSV artifacts."""

    resolved: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Plot input must be a CSV file or directory: {path}")
            resolved.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"Plot input does not exist: {path}")
        canonical: list[Path] = []
        fallback: list[Path] = []
        for root, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name not in PRUNED_DISCOVERY_DIRS]
            root_path = Path(root)
            for filename in filenames:
                if filename in CANONICAL_AGGREGATE_NAMES:
                    canonical.append(root_path / filename)
                elif filename.endswith("aggregate_by_policy_horizon.csv"):
                    fallback.append(root_path / filename)
        discovered = canonical or fallback
        if not discovered:
            raise ValueError(
                "No predict-only aggregate CSV found below "
                f"{path}; expected *aggregate_by_policy_horizon.csv"
            )
        resolved.extend(sorted(discovered))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        canonical_path = path.resolve()
        if canonical_path not in seen:
            seen.add(canonical_path)
            unique.append(canonical_path)
    if not unique:
        raise ValueError("At least one predict-only aggregate CSV is required")
    return unique


def load_datasets(
    sources: Iterable[Path],
    *,
    single_label: str | None = None,
    allow_nonformal_input: bool = False,
) -> list[AggregateDataset]:
    source_paths = list(sources)
    if single_label and len(source_paths) != 1:
        raise ValueError("--label can only be used when exactly one aggregate CSV is resolved")

    datasets: list[AggregateDataset] = []
    used_ids: set[str] = set()
    for source_path in source_paths:
        rows, fieldnames = _read_csv(source_path)
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"{source_path} is missing required columns: {missing}")
        if not rows:
            raise ValueError(f"Predict-only aggregate CSV contains no rows: {source_path}")
        if not any(_number(row.get("horizon")) is not None for row in rows):
            raise ValueError(f"Predict-only aggregate CSV has no numeric horizons: {source_path}")
        if not any(
            _number(row.get(metric)) is not None
            for row in rows
            for metric, _label in TIER1_METRICS
        ):
            raise ValueError(f"Predict-only aggregate CSV has no Tier-1 metrics: {source_path}")

        benchmark = _infer_benchmark(source_path, rows)
        base_id = single_label or f"{benchmark}:{_run_name(source_path)}"
        dataset_id = base_id
        suffix = 2
        while dataset_id in used_ids:
            dataset_id = f"{base_id}#{suffix}"
            suffix += 1
        used_ids.add(dataset_id)
        normalized = [_with_derived_values(row) for row in rows]
        target_estimand, validation_warnings, quality_path = _validate_aggregate_input(
            source_path,
            normalized,
        )
        if validation_warnings and not allow_nonformal_input:
            details = "; ".join(validation_warnings)
            raise ValueError(
                f"Non-formal predict-only aggregate rejected: {source_path}: {details}. "
                "Use --allow-nonformal-input only for explicitly diagnostic plots."
            )
        datasets.append(
            AggregateDataset(
                dataset_id=dataset_id,
                benchmark=benchmark,
                source_csv=source_path,
                rows=normalized,
                target_estimand=target_estimand,
                formal_input=not validation_warnings,
                validation_warnings=tuple(validation_warnings),
                quality_path=quality_path,
            )
        )
    estimands = {dataset.target_estimand for dataset in datasets if dataset.target_estimand != "unknown"}
    if len(estimands) > 1:
        message = "mixed target estimands across datasets: " + ", ".join(sorted(estimands))
        if not allow_nonformal_input:
            raise ValueError(message)
        datasets = [
            replace(
                dataset,
                formal_input=False,
                validation_warnings=(*dataset.validation_warnings, message),
            )
            for dataset in datasets
        ]
    return datasets


def generate_standard_plots(
    *,
    inputs: Iterable[Path],
    output_dir: Path,
    title: str = "World Model Safety: Predict-only Results",
    formats: Iterable[str] = ("png",),
    label: str | None = None,
    overwrite: bool = False,
    allow_nonformal_input: bool = False,
    include_four_panel: bool = False,
) -> dict[str, Any]:
    sources = discover_aggregate_csvs(inputs)
    datasets = load_datasets(
        sources,
        single_label=label,
        allow_nonformal_input=allow_nonformal_input,
    )
    output_dir = Path(output_dir).expanduser().resolve()
    selected_formats = list(dict.fromkeys(str(item).lower() for item in formats))
    unsupported = sorted(set(selected_formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported plot format(s): {unsupported}")
    if not selected_formats:
        raise ValueError("At least one plot format is required")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace standard artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = [
        {
            **row,
            "dataset_id": dataset.dataset_id,
            "benchmark": dataset.benchmark,
            "source_csv": str(dataset.source_csv),
        }
        for dataset in datasets
        for row in dataset.rows
    ]
    figure_specs = (
        ("tier1_by_horizon", TIER1_METRICS, "Tier 1 risk detection by horizon", True, 2),
        ("tier2_by_horizon", TIER2_METRICS, "Tier 2 violation-code matching by horizon", True, 2),
        ("calibration_by_horizon", CALIBRATION_METRICS, "Calibration by horizon", True, 2),
        ("support_by_horizon", SUPPORT_METRICS, "Evaluation support by horizon", False, 2),
        ("token_cost_by_horizon", TOKEN_METRICS, "Token cost by horizon", False, 3),
        (
            "predict_only_summary",
            SUMMARY_METRICS,
            "Accuracy, recall, and Tier 2 by horizon",
            True,
            3,
        ),
    )
    if include_four_panel:
        figure_specs += (
            (
                "accuracy_precision_recall_tier2",
                FOUR_PANEL_METRICS,
                "Accuracy, precision, recall, and Tier 2 by horizon",
                True,
                2,
            ),
        )
    artifacts: dict[str, list[str]] = {}
    warnings = list(
        dict.fromkeys(
            warning
            for dataset in datasets
            for warning in dataset.validation_warnings
        )
    )
    for stem, metrics, subtitle, unit_interval, ncols in figure_specs:
        available_metrics = [metric for metric in metrics if _metric_available(rows, metric[0])]
        if not available_metrics:
            warnings.append(f"Skipped {stem}: none of its metrics are available")
            continue
        paths = [output_dir / f"{stem}.{extension}" for extension in selected_formats]
        _plot_metric_grid(
            plt=plt,
            rows=rows,
            metrics=available_metrics,
            paths=paths,
            title=f"{title}\n{subtitle}",
            unit_interval=unit_interval,
            ncols=ncols,
        )
        artifacts[stem] = [str(path) for path in paths]

    unified_csv = output_dir / "unified_predict_only_aggregate.csv"
    _write_rows(unified_csv, rows)
    manifest_path = output_dir / "plot_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "unified_predict_only_plots",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "output_dir": str(output_dir),
        "formats": selected_formats,
        "formal_input_validation_enabled": not allow_nonformal_input,
        "all_inputs_formal": all(dataset.formal_input for dataset in datasets),
        "target_estimands": sorted({dataset.target_estimand for dataset in datasets}),
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "benchmark": dataset.benchmark,
                "source_csv": str(dataset.source_csv),
                "target_estimand": dataset.target_estimand,
                "formal_input": dataset.formal_input,
                "quality_path": str(dataset.quality_path) if dataset.quality_path else None,
                "validation_warnings": list(dataset.validation_warnings),
                "row_count": len(dataset.rows),
                "horizons": sorted(
                    {
                        int(value)
                        for row in dataset.rows
                        if (value := _number(row.get("horizon"))) is not None
                    }
                ),
            }
            for dataset in datasets
        ],
        "figures": artifacts,
        "unified_aggregate_csv": str(unified_csv),
        "warnings": warnings,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    return manifest


def _plot_metric_grid(
    *,
    plt: Any,
    rows: list[dict[str, Any]],
    metrics: list[tuple[str, str]],
    paths: list[Path],
    title: str,
    unit_interval: bool | None,
    ncols: int,
) -> None:
    ncols = max(1, min(int(ncols), len(metrics)))
    nrows = max(1, math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.7 * nrows), squeeze=False)
    flat_axes = list(axes.flatten())
    series = _series_rows(rows)
    fallback_policies = sorted({series_id[2] for series_id in series})
    fallback_colors = {
        policy: plt.get_cmap("tab10")(index % 10)
        for index, policy in enumerate(fallback_policies)
    }
    single_dataset = len({series_id[0] for series_id in series}) == 1
    single_rollout = len({series_id[1] for series_id in series}) == 1
    for ax, (metric, metric_label) in zip(flat_axes, metrics):
        for series_id, series_rows in series.items():
            dataset_id, rollout_mode, policy_mode = series_id
            policy_label, color = POLICY_STYLES.get(
                policy_mode,
                (_humanize(policy_mode or "Policy"), fallback_colors[policy_mode]),
            )
            rollout_label, line_style, marker = ROLLOUT_STYLES.get(
                rollout_mode,
                (_humanize(rollout_mode or "Rollout"), "-", "o"),
            )
            if single_dataset and single_rollout:
                display_label = policy_label
            elif single_dataset:
                display_label = f"{rollout_label} · {policy_label}"
            else:
                display_label = f"{dataset_id} · {rollout_label} · {policy_label}"
            points = [
                (int(horizon), _number(row.get(metric)), row)
                for row in series_rows
                if (horizon := _number(row.get("horizon"))) is not None
            ]
            if not points or not any(value is not None for _horizon, value, _row in points):
                continue
            points.sort(key=lambda item: item[0])
            horizons = [item[0] for item in points]
            # Preserve missing metric rows as NaN so matplotlib breaks the line
            # instead of connecting non-adjacent horizons across undefined values.
            values = [float(item[1]) if item[1] is not None else math.nan for item in points]
            ax.plot(
                horizons,
                values,
                marker=marker,
                markersize=5.0,
                linewidth=2.0,
                linestyle=line_style,
                color=color,
                label=display_label,
            )
            lows = [_number(item[2].get(f"{metric}_ci_low")) for item in points]
            highs = [_number(item[2].get(f"{metric}_ci_high")) for item in points]
            if any(low is not None and high is not None for low, high in zip(lows, highs)):
                ax.fill_between(
                    horizons,
                    [float(value) if value is not None else math.nan for value in lows],
                    [float(value) if value is not None else math.nan for value in highs],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        ax.set_title(metric_label)
        ax.set_xlabel("Prediction horizon h")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)
        ax.set_xticks(
            sorted(
                {
                    int(value)
                    for row in rows
                    if (value := _number(row.get("horizon"))) is not None
                }
            )
        )
        if unit_interval is True or (unit_interval is None and metric != "total_tokens_per_point"):
            ax.set_ylim(0, 1.05)
        elif unit_interval is False:
            ax.set_ylim(bottom=0)
    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")

    handles: list[Any] = []
    labels: list[str] = []
    for ax in flat_axes[: len(metrics)]:
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(3, len(labels)),
            fontsize=8,
            frameon=True,
        )
    fig.suptitle(title, fontsize=15, y=0.99)
    # Fixed margins are more stable than tight_layout for long two-line titles,
    # bottom legends, and wide confidence bands.  tight_layout can push subplot
    # titles into the suptitle even in a single-process render.
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.78 if nrows == 1 else 0.88,
        bottom=0.21 if handles and nrows == 1 else (0.11 if handles else 0.07),
        wspace=0.28,
        hspace=0.42,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The layout already reserves title and legend space.  A second
        # bbox_inches="tight" pass can nondeterministically crop long titles,
        # axes labels, or legends when several reports render concurrently.
        fig.savefig(path, dpi=220)
    plt.close(fig)


def _series_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        series_id = (
            str(row.get("dataset_id") or "dataset"),
            str(row.get("rollout_mode") or "wm_free_rollout"),
            str(row.get("policy_mode") or "policy"),
        )
        grouped.setdefault(series_id, []).append(row)
    return dict(sorted(grouped.items()))


def _humanize(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value).replace("-", "_").split("_") if part)


def _with_derived_values(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    selected = _number(row.get("selected_prediction_points"))
    output["positive_support"] = _sum_numbers(row.get("tp"), row.get("fn"))
    output["negative_support"] = _sum_numbers(row.get("tn"), row.get("fp"))
    for token_field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        tokens = _number(row.get(token_field))
        output[f"{token_field}_per_point"] = (
            tokens / selected
            if tokens is not None and selected is not None and selected > 0
            else None
        )
    return output


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fieldnames


def _validate_aggregate_input(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[str, list[str], Path | None]:
    """Return the estimand and every reason this aggregate is non-formal."""

    warnings: list[str] = []
    required_text_fields = (
        "rollout_mode",
        "policy_mode",
        "target_estimand",
        "horizon_cohort",
        "cohort_signature",
        "tier2_estimand",
        "experiment_seed",
        "artifact_status",
    )
    for field in required_text_fields:
        missing_rows = [
            index
            for index, row in enumerate(rows, start=2)
            if row.get(field) is None or not str(row.get(field)).strip()
        ]
        if missing_rows:
            warnings.append(f"{field} is missing on CSV row(s) {missing_rows}")
    count_fields = (
        "task_count",
        "bootstrap_cluster_count",
        "positive_task_cluster_count",
        "negative_task_cluster_count",
        "selected_prediction_points",
        "tp",
        "fp",
        "tn",
        "fn",
        "invalid_decisions",
    )
    for index, row in enumerate(rows, start=2):
        schema_version = _number(row.get("aggregate_schema_version"))
        horizon = _number(row.get("horizon"))
        if (
            schema_version is None
            or not schema_version.is_integer()
            or schema_version < MIN_AGGREGATE_SCHEMA_VERSION
        ):
            warnings.append(
                f"aggregate_schema_version must be an integer >= "
                f"{MIN_AGGREGATE_SCHEMA_VERSION} on CSV row {index}"
            )
        if horizon is None or not horizon.is_integer() or horizon < 1:
            warnings.append(f"horizon must be a positive integer on CSV row {index}")
        for field in count_fields:
            value = _number(row.get(field))
            if value is None or not value.is_integer() or value < 0:
                warnings.append(
                    f"{field} must be a non-negative integer on CSV row {index}"
                )
    estimands = {
        str(row.get("target_estimand") or "").strip().lower()
        for row in rows
        if str(row.get("target_estimand") or "").strip()
    }
    if len(estimands) != 1:
        warnings.append(
            "target_estimand must be present and constant"
            if not estimands
            else f"target_estimand is mixed within one aggregate: {sorted(estimands)}"
        )
        target_estimand = "unknown" if not estimands else "+".join(sorted(estimands))
    else:
        target_estimand = next(iter(estimands))
    if target_estimand not in {"exact_step", "within_window"}:
        warnings.append(f"unsupported target_estimand={target_estimand!r}")

    horizon_cohorts = {
        str(row.get("horizon_cohort") or "").strip().lower()
        for row in rows
        if str(row.get("horizon_cohort") or "").strip()
    }
    if horizon_cohorts != {"common_max_horizon_support"}:
        warnings.append(
            "horizon_cohort must be common_max_horizon_support, found "
            f"{sorted(horizon_cohorts) or ['missing']}"
        )
    cohort_signatures = {
        str(row.get("cohort_signature") or "").strip()
        for row in rows
        if str(row.get("cohort_signature") or "").strip()
    }
    if len(cohort_signatures) != 1:
        warnings.append(
            "cohort_signature must be present and identical across all horizons/conditions"
        )
    tier2_estimands = {
        str(row.get("tier2_estimand") or "").strip()
        for row in rows
        if str(row.get("tier2_estimand") or "").strip()
    }
    if tier2_estimands != {"all_risk_relevant_rows_missing_pred_codes_zero"}:
        warnings.append(
            "tier2_estimand must declare all risk-relevant rows with missing predicted codes scored zero"
        )
    experiment_seeds = {
        str(row.get("experiment_seed")).strip()
        for row in rows
        if row.get("experiment_seed") is not None
        and str(row.get("experiment_seed")).strip()
    }
    if len(experiment_seeds) != 1:
        warnings.append("experiment_seed must be present and constant within an aggregate")

    schema_versions = {_number(row.get("aggregate_schema_version")) for row in rows}
    schema_versions.discard(None)
    if not schema_versions or min(schema_versions) < MIN_AGGREGATE_SCHEMA_VERSION:
        warnings.append(
            f"aggregate_schema_version must be >= {MIN_AGGREGATE_SCHEMA_VERSION}"
        )
    elif len(schema_versions) != 1:
        warnings.append(f"aggregate_schema_version is mixed: {sorted(schema_versions)}")
    statuses = {
        str(row.get("artifact_status") or "").strip().lower()
        for row in rows
        if str(row.get("artifact_status") or "").strip()
    }
    if statuses != {"complete"}:
        warnings.append(f"artifact_status must be complete, found {sorted(statuses) or ['missing']}")
    if any(
        value is not None and value != 0.0
        for value in (_number(row.get("invalid_decisions")) for row in rows)
    ):
        warnings.append("invalid_decisions is non-zero")

    seen_keys: set[tuple[str, str, int]] = set()
    duplicate_keys: set[tuple[str, str, int]] = set()
    support_by_series: dict[tuple[str, str], set[float]] = {}
    tasks_by_series: dict[tuple[str, str], set[float]] = {}
    for row in rows:
        horizon = _number(row.get("horizon"))
        if horizon is None:
            continue
        key = (
            str(row.get("rollout_mode") or ""),
            str(row.get("policy_mode") or ""),
            int(horizon),
        )
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
        support = _number(row.get("selected_prediction_points"))
        confusion = _sum_numbers(row.get("tp"), row.get("fp"), row.get("tn"), row.get("fn"))
        if support is None or confusion is None:
            warnings.append(f"missing selected support or confusion counts for {key}")
        elif support != confusion:
            warnings.append(f"confusion counts do not equal selected support for {key}")
        else:
            support_by_series.setdefault(key[:2], set()).add(support)
            positive_support = _sum_numbers(row.get("tp"), row.get("fn"))
            negative_support = _sum_numbers(row.get("tn"), row.get("fp"))
            if not positive_support or not negative_support:
                warnings.append(
                    f"both positive and negative GT support are required for formal metrics at {key}"
                )
        task_count = _number(row.get("task_count"))
        bootstrap_count = _number(row.get("bootstrap_cluster_count"))
        positive_cluster_count = _number(row.get("positive_task_cluster_count"))
        negative_cluster_count = _number(row.get("negative_task_cluster_count"))
        if task_count is None:
            warnings.append(f"missing task_count for {key}")
        elif task_count < 2:
            warnings.append(f"at least two task clusters are required for formal metrics at {key}")
        else:
            tasks_by_series.setdefault(key[:2], set()).add(task_count)
        if bootstrap_count is None or task_count is None or bootstrap_count != task_count:
            warnings.append(f"bootstrap_cluster_count must equal task_count for {key}")
        if positive_cluster_count is None or positive_cluster_count < 2:
            warnings.append(
                f"at least two positive task clusters are required for formal metrics at {key}"
            )
        if negative_cluster_count is None or negative_cluster_count < 2:
            warnings.append(
                f"at least two negative task clusters are required for formal metrics at {key}"
            )
        if (
            task_count is not None
            and positive_cluster_count is not None
            and positive_cluster_count > task_count
        ):
            warnings.append(f"positive task clusters exceed task_count for {key}")
        if (
            task_count is not None
            and negative_cluster_count is not None
            and negative_cluster_count > task_count
        ):
            warnings.append(f"negative task clusters exceed task_count for {key}")
    if duplicate_keys:
        warnings.append(f"duplicate rollout/policy/horizon rows: {sorted(duplicate_keys)}")
    changing_support = {
        series: sorted(values)
        for series, values in support_by_series.items()
        if len(values) > 1
    }
    if changing_support:
        warnings.append(f"selected support changes across horizons: {changing_support}")
    changing_task_count = {
        series: sorted(values)
        for series, values in tasks_by_series.items()
        if len(values) > 1
    }
    if changing_task_count:
        warnings.append(f"task_count changes across horizons: {changing_task_count}")

    quality_path: Path | None = None
    for name in (
        "predict_only_quality_report.json",
        "isbench_predict_only_run_quality.json",
    ):
        candidate = path.parent / name
        if not candidate.is_file():
            continue
        quality_path = candidate
        try:
            quality = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"cannot read quality sidecar {candidate.name}: {exc}")
            break
        if quality.get("complete") is not True:
            warnings.append(f"quality sidecar {candidate.name} is incomplete")
        quality_estimand = str(quality.get("target_estimand") or "").strip().lower()
        if quality_estimand and quality_estimand != target_estimand:
            warnings.append(
                f"quality sidecar target_estimand={quality_estimand!r} disagrees with aggregate"
            )
        quality_cohort = str(quality.get("horizon_cohort") or "").strip().lower()
        if quality_cohort and quality_cohort != "common_max_horizon_support":
            warnings.append(
                f"quality sidecar horizon_cohort={quality_cohort!r} is non-formal"
            )
        break
    return target_estimand, list(dict.fromkeys(warnings)), quality_path


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = ("dataset_id", "benchmark", "source_csv", "rollout_mode", "policy_mode", "horizon")
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _infer_benchmark(path: Path, rows: list[dict[str, Any]]) -> str:
    path_text = str(path).lower()
    policies = " ".join(str(row.get("policy_mode") or "").lower() for row in rows)
    if "isbench" in path_text or "isbench" in policies or "embodied" in policies:
        return "isbench"
    return "harnessaudit"


def _run_name(path: Path) -> str:
    ignored = {
        "_analysis",
        "analysis",
        "harnessaudit",
        "isbench",
        "prediction",
        "results",
        "result",
    }
    for parent in path.parents:
        if parent.name and parent.name.lower() not in ignored:
            return parent.name
    return path.stem


def _metric_available(rows: Iterable[dict[str, Any]], metric: str) -> bool:
    return any(_number(row.get(metric)) is not None for row in rows)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sum_numbers(*values: Any) -> float | None:
    numbers = [_number(value) for value in values]
    if any(value is None for value in numbers):
        return None
    return sum(float(value) for value in numbers)


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[1] / "reports" / "predict_only_plots" / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one standard predict-only plot bundle from HA/IS-Bench aggregates."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Aggregate CSV file(s), or run directories containing them.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default="World Model Safety: Predict-only Results")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--label", help="Custom dataset label; valid only for one resolved CSV.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--four-panel",
        action="store_true",
        help="Also render a four-panel Accuracy/Precision/Recall/Tier-2 figure.",
    )
    parser.add_argument(
        "--allow-nonformal-input",
        action="store_true",
        help="Allow legacy, partial, or mixed-estimand inputs for diagnostics only.",
    )
    parser.add_argument("--json", action="store_true", help="Print the generated manifest as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_standard_plots(
            inputs=args.inputs,
            output_dir=args.output_dir or _default_output_dir(),
            title=args.title,
            formats=args.formats,
            label=args.label,
            overwrite=args.overwrite,
            allow_nonformal_input=args.allow_nonformal_input,
            include_four_panel=args.four_panel,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"plot-results: {exc}") from exc
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {sum(len(paths) for paths in manifest['figures'].values())} figure files")
        print(manifest["unified_aggregate_csv"])
        print(manifest["manifest"])
        for warning in manifest["warnings"]:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
