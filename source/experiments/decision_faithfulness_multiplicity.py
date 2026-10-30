"""Generate fail-closed multiplicity-adjusted Decision Faithfulness effects.

The online matrix runners intentionally emit only raw, unadjusted paired
effects.  This module is the separately frozen confirmatory analysis step.  It
reconstructs the exact preregistered task x seed pairs from raw score files and
uses a joint task-cluster bootstrap to produce simultaneous confidence bounds
for every primary policy x endpoint family member.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from multi_agent.safepred_experiments.decision_faithfulness_postflight import (
    _file_set_sha256,
    _validate_matrix_provenance,
    build_evidence_bindings,
)
from multi_agent.safepred_experiments.formal_contract import load_preregistration


BENCHMARKS = ("harnessaudit", "isbench")
ANALYSIS_VERSION = "joint-task-cluster-max-deviation-bootstrap-v1"
MULTIPLICITY_METHOD = "joint_task_cluster_max_deviation"


def generate_adjusted_effects(
    *,
    benchmark: str,
    preregistration_path: Path,
    matrix_provenance_path: Path,
    points_path: Path,
    paired_score_input_paths: Iterable[Path],
    output_path: Path,
    bootstrap_seed: int = 0,
    analysis_code_path: Path | None = None,
) -> dict[str, Any]:
    """Validate inputs and atomically write the exact adjusted primary family."""

    normalized_benchmark = str(benchmark).strip().lower()
    if normalized_benchmark not in BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark!r}")
    preregistration = load_preregistration(
        Path(preregistration_path), benchmark=normalized_benchmark
    )
    payload = preregistration["payload"]
    statistics = payload["statistics"]
    method = str(statistics["multiplicity_method"]).strip()
    if method != MULTIPLICITY_METHOD:
        raise ValueError(
            "this frozen analysis implements only "
            f"statistics.multiplicity_method={MULTIPLICITY_METHOD!r}"
        )
    alpha = _finite_number(statistics["alpha"], label="statistics.alpha")
    bootstrap_samples = _strict_int(
        statistics["bootstrap_samples"], label="statistics.bootstrap_samples"
    )
    if not 0.0 < alpha < 0.5:
        raise ValueError("statistics.alpha must be in (0, 0.5)")
    if bootstrap_samples < 1000:
        raise ValueError("statistics.bootstrap_samples must be >= 1000")

    code_path = Path(analysis_code_path or __file__).expanduser().resolve()
    code_sha256 = _sha256_file(code_path)
    frozen_code_sha256 = str(
        payload["annotation_contract"]["multiplicity_analysis_code_sha256"]
    ).strip().lower()
    if code_sha256 != frozen_code_sha256:
        raise ValueError(
            "multiplicity analysis code does not match the preregistered SHA256"
        )

    points_path = Path(points_path).expanduser().resolve()
    score_paths = [Path(path).expanduser().resolve() for path in paired_score_input_paths]
    if not score_paths:
        raise ValueError("at least one paired score input is required")
    point_rows, _point_columns = _read_csv(points_path, label="points")
    score_rows: list[dict[str, str]] = []
    for path in score_paths:
        rows, _columns = _read_csv(path, label="paired score input")
        score_rows.extend(rows)

    score_inputs_sha256, score_hash_errors = _file_set_sha256(
        score_paths, label="paired score input"
    )
    if score_hash_errors or score_inputs_sha256 is None:
        raise ValueError("; ".join(score_hash_errors))
    matrix_provenance_path = Path(matrix_provenance_path).expanduser().resolve()
    provenance_errors = _validate_matrix_provenance(
        matrix_provenance_path=matrix_provenance_path,
        benchmark=normalized_benchmark,
        preregistration_sha256=preregistration["sha256"],
        preregistration_payload=payload,
        points_path=points_path,
        paired_score_input_paths=score_paths,
    )
    if provenance_errors:
        raise ValueError("invalid matrix provenance: " + "; ".join(provenance_errors))

    matrix_sha256 = _sha256_file(matrix_provenance_path)
    bindings = build_evidence_bindings(
        benchmark=normalized_benchmark,
        preregistration_payload=payload,
        point_rows=point_rows,
        matrix_provenance_sha256=matrix_sha256,
        paired_score_inputs_sha256=score_inputs_sha256,
        multiplicity_analysis_code_sha256=code_sha256,
    )
    family = _complete_primary_family(
        benchmark=normalized_benchmark,
        payload=payload,
        score_rows=score_rows,
    )
    adjusted = _joint_cluster_bootstrap(
        family,
        samples=bootstrap_samples,
        alpha=alpha,
        seed=int(bootstrap_seed),
    )
    rows: list[dict[str, Any]] = []
    primary_intervention = str(
        payload["decision_faithfulness"]["primary_intervention"]
    ).strip()
    for key in sorted(family):
        policy_mode, metric = key
        cluster_values = family[key]
        interval = adjusted[key]
        rows.append(
            {
                **bindings,
                "policy_mode": policy_mode,
                "intervention_mode": primary_intervention,
                "metric": metric,
                "effect_definition": "condition_minus_paired_no_guard",
                "mean_paired_effect": interval["mean_effect"],
                "adjusted_ci_low": interval["adjusted_ci_low"],
                "adjusted_ci_high": interval["adjusted_ci_high"],
                "confidence_level": 1.0 - alpha,
                "task_clusters": len(cluster_values),
                "task_seed_pairs": sum(item["pair_count"] for item in cluster_values.values()),
                "bootstrap_samples": bootstrap_samples,
                "multiplicity_method": method,
                "multiplicity_adjusted": "true",
                "analysis_version": ANALYSIS_VERSION,
                "bootstrap_seed": int(bootstrap_seed),
                "simultaneous_family_size": len(family),
            }
        )

    output_path = Path(output_path).expanduser().resolve()
    _write_csv_atomic(output_path, rows)
    return {
        "schema_version": "decision-faithfulness-multiplicity-result-v1",
        "valid": True,
        "benchmark": normalized_benchmark,
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "family_size": len(rows),
        "task_clusters": len(next(iter(family.values()))),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": int(bootstrap_seed),
        "multiplicity_method": method,
        "analysis_version": ANALYSIS_VERSION,
        "preregistration_sha256": preregistration["sha256"],
        "matrix_provenance_sha256": matrix_sha256,
        "paired_score_inputs_sha256": score_inputs_sha256,
        "multiplicity_analysis_code_sha256": code_sha256,
    }


def _complete_primary_family(
    *,
    benchmark: str,
    payload: dict[str, Any],
    score_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """Return task-cluster means only when the exact primary cohort is complete."""

    plan = payload["run_plans"][benchmark]
    task_field = "tasks" if benchmark == "harnessaudit" else "task_ids"
    tasks = [Path(str(item)).stem for item in plan[task_field]]
    seeds = [int(item) for item in plan["experiment_seeds"]]
    policies = [str(item).strip() for item in plan["policy_modes"]]
    intervention = str(
        payload["decision_faithfulness"]["primary_intervention"]
    ).strip()
    endpoints = [
        str(payload["endpoints"]["primary_safety"]).strip(),
        str(payload["endpoints"]["primary_utility"]).strip(),
    ]
    expected_pairs = {(task, seed) for task in tasks for seed in seeds}

    baselines: dict[tuple[str, int], dict[str, str]] = {}
    treatments: dict[tuple[str, str, int], dict[str, str]] = {}
    errors: list[str] = []
    for row in score_rows:
        row_intervention = str(row.get("intervention_mode") or "").strip()
        row_policy = str(row.get("policy_mode") or "").strip()
        relevant = row_intervention == "no_guard" or (
            row_intervention == intervention and row_policy in policies
        )
        if not relevant:
            continue
        task = Path(str(row.get("task_id") or "").strip()).stem
        try:
            seed = _strict_int(row.get("experiment_seed"), label="experiment_seed")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        pair = (task, seed)
        if pair not in expected_pairs:
            errors.append(f"unregistered task x seed row: {pair!r}")
            continue
        if row_intervention == "no_guard":
            if pair in baselines:
                errors.append(f"duplicate no_guard row: {pair!r}")
            baselines[pair] = row
        else:
            key = (row_policy, task, seed)
            if key in treatments:
                errors.append(f"duplicate primary intervention row: {key!r}")
            treatments[key] = row

    missing_baselines = sorted(expected_pairs - set(baselines))
    if missing_baselines:
        errors.append(f"missing no_guard rows: {missing_baselines[:12]!r}")
    expected_treatments = {
        (policy, task, seed)
        for policy in policies
        for task, seed in expected_pairs
    }
    missing_treatments = sorted(expected_treatments - set(treatments))
    if missing_treatments:
        errors.append(f"missing primary intervention rows: {missing_treatments[:12]!r}")

    family: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for policy in policies:
        for metric in endpoints:
            by_task: dict[str, dict[str, Any]] = {}
            for task in tasks:
                differences: list[float] = []
                for seed in seeds:
                    baseline = baselines.get((task, seed))
                    treatment = treatments.get((policy, task, seed))
                    if baseline is None or treatment is None:
                        continue
                    try:
                        baseline_value = _finite_number(
                            baseline.get(metric),
                            label=f"no_guard/{task}/{seed}/{metric}",
                        )
                        treatment_value = _finite_number(
                            treatment.get(metric),
                            label=f"{policy}/{intervention}/{task}/{seed}/{metric}",
                        )
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    differences.append(treatment_value - baseline_value)
                if len(differences) != len(seeds):
                    errors.append(
                        f"incomplete evaluable endpoint cohort for "
                        f"{policy}/{metric}/{task}: {len(differences)} != {len(seeds)}"
                    )
                    continue
                by_task[task] = {
                    "mean": sum(differences) / len(differences),
                    "pair_count": len(differences),
                }
            if set(by_task) != set(tasks):
                errors.append(f"incomplete task clusters for {policy}/{metric}")
            family[(policy, metric)] = by_task
    if len(set(endpoints)) != 2:
        errors.append("primary safety and utility endpoints must be distinct")
    if errors:
        raise ValueError("invalid paired score cohort: " + "; ".join(_ordered_unique(errors)))
    return family


def _joint_cluster_bootstrap(
    family: dict[tuple[str, str], dict[str, dict[str, Any]]],
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> dict[tuple[str, str], dict[str, float]]:
    """Construct simultaneous basic-bootstrap CIs over the complete family."""

    if not family:
        raise ValueError("primary hypothesis family is empty")
    task_sets = {tuple(sorted(values)) for values in family.values()}
    if len(task_sets) != 1:
        raise ValueError("family members do not share the same task clusters")
    tasks = list(next(iter(task_sets)))
    if len(tasks) < 2:
        raise ValueError("at least two task clusters are required")
    means = {
        key: sum(values[task]["mean"] for task in tasks) / len(tasks)
        for key, values in family.items()
    }
    rng = random.Random(int(seed))
    max_deviations: list[float] = []
    for _ in range(int(samples)):
        draw = [rng.choice(tasks) for _task in tasks]
        deviations = []
        for key, values in family.items():
            bootstrap_mean = sum(values[task]["mean"] for task in draw) / len(draw)
            deviations.append(abs(bootstrap_mean - means[key]))
        max_deviations.append(max(deviations))
    max_deviations.sort()
    critical_index = min(
        len(max_deviations) - 1,
        max(0, math.ceil((1.0 - alpha) * len(max_deviations)) - 1),
    )
    critical = max_deviations[critical_index]
    return {
        key: {
            "mean_effect": mean,
            "adjusted_ci_low": mean - critical,
            "adjusted_ci_high": mean + critical,
        }
        for key, mean in means.items()
    }


def _read_csv(path: Path, *, label: str) -> tuple[list[dict[str, str]], list[str]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{label} CSV contains no data rows: {path}")
    return rows, columns


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty multiplicity artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        raise ValueError(f"{label} is not evaluable")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer")
    text = str(value).strip()
    digits = text[1:] if text[:1] in {"+", "-"} else text
    if not text or not digits.isdigit():
        raise ValueError(f"{label} is not an integer")
    return int(text)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen multiplicity-adjusted Decision Faithfulness effects"
    )
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--matrix-provenance", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--paired-score-inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_adjusted_effects(
        benchmark=args.benchmark,
        preregistration_path=args.preregistration,
        matrix_provenance_path=args.matrix_provenance,
        points_path=args.points,
        paired_score_input_paths=args.paired_score_inputs,
        output_path=args.output,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
