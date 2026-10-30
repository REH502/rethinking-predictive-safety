"""Joint fail-closed postflight for preregistered predict-only replicates.

Per-run pipelines intentionally emit diagnostic figures.  This module is the
only path that can label an offline result set confirmatory: it validates the
exact actor/WM seed matrix, verifies every run and condition, rejects nominally
different actor seeds only when their complete task-cohort trajectories are
identical, combines points, clusters uncertainty by task x actor seed, and then
renders the same established predict-only figure style.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from multi_agent.safepred_experiments.formal_contract import (
    ActorReplicateIdentity,
    load_preregistration,
    normalize_actor_replicate_identity,
    validate_preregistered_offline_replicates,
)
from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
    aggregate_support_is_formal,
    plot_pilot_summary,
    plot_tier1_horizon,
    plot_tier2_horizon,
    plot_token_split_horizon,
)


QUALITY_FILENAMES = {
    "harnessaudit": "predict_only_quality_report.json",
    "isbench": "isbench_predict_only_run_quality.json",
}
POINT_FILENAMES = {
    "harnessaudit": "predict_only_prediction_points.csv",
    "isbench": "isbench_predict_only_prediction_points.csv",
}
PROVENANCE_FILENAMES = {
    "harnessaudit": "predict_only_provenance_manifest.json",
    "isbench": "isbench_predict_only_provenance_manifest.json",
}


def run_postflight(
    *,
    benchmark: str,
    preregistration_path: Path,
    analysis_dirs: Iterable[Path],
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, Path]:
    benchmark = str(benchmark).strip().lower()
    if benchmark not in QUALITY_FILENAMES:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}")
    preregistration = load_preregistration(preregistration_path, benchmark=benchmark)
    directories = [Path(path).expanduser().resolve() for path in analysis_dirs]
    if not directories:
        raise ValueError("At least one replicate analysis directory is required")

    gates: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    cohort_by_actor: dict[
        tuple[str, str], set[tuple[str, str, str, int, str]]
    ] = {}
    immutable_rows_by_actor: dict[
        tuple[str, str], dict[tuple[str, str, str, int, str], str]
    ] = {}
    trajectory_digests_by_actor: dict[tuple[str, str], dict[str, str]] = {}
    seen_pairs: set[tuple[str, str, int | None, int]] = set()
    seen_point_keys: set[tuple[Any, ...]] = set()

    plan = preregistration["payload"]["run_plans"][benchmark]
    offline = preregistration["payload"]["offline_prediction"]
    protocol = offline["protocol_by_benchmark"][benchmark]
    expected_implementation_sha256 = str(
        protocol["implementation_manifest_sha256"]
    ).strip().lower()
    current_implementation_sha256 = _current_implementation_manifest_sha256(
        benchmark
    )
    if current_implementation_sha256 != expected_implementation_sha256:
        raise ValueError(
            "Current postflight/runtime implementation differs from the frozen "
            f"{benchmark} protocol: current={current_implementation_sha256}, "
            f"preregistered={expected_implementation_sha256}"
        )
    task_field = "tasks" if benchmark == "harnessaudit" else "task_ids"
    expected_conditions = {
        (Path(str(task)).stem, rollout, policy, int(horizon))
        for task in plan[task_field]
        for rollout in offline["rollout_modes_by_benchmark"][benchmark]
        for policy in plan["policy_modes"]
        for horizon in offline["horizons"]
    }
    expected_tasks = {Path(str(task)).stem for task in plan[task_field]}
    if benchmark == "harnessaudit":
        from multi_agent.safepred_experiments.run_sparse_prediction_horizons import (
            aggregate_by_policy_horizon,
        )
    else:
        from multi_agent.safepred_experiments.isbench_predict_only_pipeline import (
            aggregate_by_policy_horizon,
        )

    for directory in directories:
        quality_path = directory / QUALITY_FILENAMES[benchmark]
        points_path = directory / POINT_FILENAMES[benchmark]
        provenance_path = directory / PROVENANCE_FILENAMES[benchmark]
        quality = _load_json(quality_path)
        if quality.get("formal") is not True:
            raise ValueError(f"Replicate is not marked formal: {quality_path}")
        if quality.get("complete") is not True:
            raise ValueError(f"Replicate is incomplete: {quality_path}")
        if quality.get("formal_replicate_ready") is not True:
            raise ValueError(f"Replicate did not pass formal gates: {quality_path}")
        if quality.get("formal_plot_ready") is True:
            raise ValueError(
                f"Single replicate was incorrectly marked plot-ready: {quality_path}"
            )
        if str(quality.get("implementation_manifest_sha256") or "").lower() != (
            expected_implementation_sha256
        ):
            raise ValueError(
                "Replicate implementation manifest differs from the frozen/current "
                f"postflight implementation: {quality_path}"
            )
        gate = quality.get("offline_preregistration_gate")
        if not isinstance(gate, dict):
            raise ValueError(f"Replicate lacks an offline contract gate: {quality_path}")
        actor_identity = _mapping_actor_identity(gate, benchmark=benchmark)
        wm_seed = _strict_int(gate.get("wm_seed"), "wm_seed")
        pair = (*actor_identity.full_key, wm_seed)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate normalized replicate pair: {pair}")
        seen_pairs.add(pair)
        gates.append(dict(gate))

        raw_trajectory_digests = quality.get(
            "source_action_trajectory_sha256_by_task"
        )
        if not isinstance(raw_trajectory_digests, dict):
            raise ValueError(
                "Formal replicate lacks source action-trajectory digests: "
                f"{quality_path}"
            )
        trajectory_digests: dict[str, str] = {}
        for raw_task_id, raw_digest in raw_trajectory_digests.items():
            task_id = Path(str(raw_task_id)).stem
            digest = str(raw_digest).strip().lower()
            if task_id in trajectory_digests:
                raise ValueError(
                    "Formal replicate has duplicate normalized trajectory task id: "
                    f"task={task_id}, quality={quality_path}"
                )
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(
                    "Formal replicate has an invalid action-trajectory SHA256: "
                    f"task={task_id}, quality={quality_path}"
                )
            trajectory_digests[task_id] = digest
        if set(trajectory_digests) != expected_tasks:
            raise ValueError(
                "Formal replicate action-trajectory task set differs from the "
                f"preregistered tasks: quality={quality_path}"
            )
        actor_cluster = actor_identity.cluster_key
        prior_actor_digests = trajectory_digests_by_actor.get(actor_cluster)
        if prior_actor_digests is not None and trajectory_digests != prior_actor_digests:
            raise ValueError(
                "WM replicates for one actor identity disagree on the frozen action "
                f"trajectories: actor_identity={actor_identity.full_key}"
            )
        trajectory_digests_by_actor[actor_cluster] = trajectory_digests

        identity_path = directory / "formal_replicate_identity.json"
        identity = _load_json(identity_path)
        marker_identity = _formal_identity_marker_actor_identity(
            identity,
            benchmark=benchmark,
        )
        if marker_identity != actor_identity:
            raise ValueError(
                f"Formal replicate identity mismatch for actor identity: {identity_path}"
            )
        for field, expected in (
            ("benchmark", benchmark),
            ("wm_seed", wm_seed),
            ("preregistration_sha256", gate.get("preregistration_sha256")),
            ("protocol_sha256", gate.get("protocol_sha256")),
            ("source_input_manifest_sha256", gate.get("source_input_manifest_sha256")),
        ):
            if identity.get(field) != expected:
                raise ValueError(
                    f"Formal replicate identity mismatch for {field}: {identity_path}"
                )
        _verify_provenance_artifacts(
            provenance_path,
            required_paths=(quality_path, points_path, identity_path),
        )

        points = _read_csv(points_path)
        if not points:
            raise ValueError(f"Replicate contains no prediction points: {points_path}")
        observed_conditions: set[tuple[str, str, str, int]] = set()
        actor_cohort: set[tuple[str, str, str, int, str]] = set()
        immutable_snapshot: dict[tuple[str, str, str, int, str], str] = {}
        point_ids_by_condition: dict[tuple[str, str, str, int], set[str]] = {}
        for row in points:
            row_identity = _mapping_actor_identity(
                row,
                benchmark=benchmark,
                allow_csv_seed=True,
            )
            row_wm = _strict_int(row.get("wm_seed"), "point.wm_seed")
            if (*row_identity.full_key, row_wm) != pair:
                raise ValueError(
                    "Point row actor/WM identity mismatch in "
                    f"{points_path}: row={(*row_identity.full_key, row_wm)}, gate={pair}"
                )
            task_id = Path(str(row.get("task_id") or "")).stem
            rollout = str(row.get("rollout_mode") or "")
            policy = str(row.get("policy_mode") or "")
            horizon = _strict_int(row.get("horizon"), "point.horizon")
            point_id = "::".join(
                str(value or "")
                for value in (
                    row.get("action_sequence_num"),
                    row.get("example_id"),
                    row.get("point_num"),
                )
            )
            condition = (task_id, rollout, policy, horizon)
            observed_conditions.add(condition)
            actor_cohort.add((task_id, rollout, policy, horizon, point_id))
            point_ids_by_condition.setdefault(condition, set()).add(point_id)
            if not _truthy(row.get("valid_prediction")):
                raise ValueError(f"Invalid prediction row in formal replicate: {points_path}")
            if _truthy(row.get("decision_error")) or _truthy(
                row.get("skipped_by_prediction_interval")
            ):
                raise ValueError(f"Errored/skipped row in formal replicate: {points_path}")
            gt_status = (
                row.get("gt_target_available")
                if "gt_target_available" in row
                else row.get("gt_target_evaluable")
            )
            if not _truthy(gt_status):
                raise ValueError(f"Non-evaluable GT row in formal replicate: {points_path}")
            if str(row.get("gt_estimand") or row.get("target_estimand") or "") != "exact_step":
                raise ValueError(f"Non-exact estimand row in formal replicate: {points_path}")
            immutable_key = (task_id, rollout, policy, horizon, point_id)
            immutable_snapshot[immutable_key] = hashlib.sha256(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "source_report",
                            "action_sequence_num",
                            "example_id",
                            "point_num",
                            "gt_estimand",
                            "gt_target_violated",
                            "gt_target_positive",
                            "gt_target_codes",
                            "gt_target_available",
                            "gt_target_evaluable",
                            "gt_exact_violated",
                            "gt_exact_positive",
                            "gt_exact_codes",
                            "gt_exact_available",
                            "gt_exact_evaluable",
                            "horizon_cohort",
                        )
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            point_key = (*actor_identity.full_key, wm_seed, *condition, point_id)
            if point_key in seen_point_keys:
                raise ValueError(f"Duplicate point identity across artifacts: {point_key}")
            seen_point_keys.add(point_key)
            row["actor_protocol_profile"] = actor_identity.actor_protocol_profile
            row["actor_replicate_id"] = actor_identity.actor_replicate_id
            row["actor_trace_seed"] = actor_identity.actor_trace_seed
            row["replicate_id"] = (
                f"{actor_identity.actor_protocol_profile}:"
                f"{actor_identity.actor_replicate_id}-wm{wm_seed}"
            )
            # WM seeds are repeated measurements inside one actor trace, not
            # independent traces.  Cluster at task x actor seed to preserve the
            # outer dependence structure and avoid artificially narrow CIs.
            row["bootstrap_cluster_id"] = (
                f"{task_id}::actor{actor_identity.actor_trace_seed}"
                if actor_identity.actor_protocol_profile == "seeded_sampling_v1"
                else (
                    f"{task_id}::{actor_identity.actor_protocol_profile}::"
                    f"{actor_identity.actor_replicate_id}"
                )
            )
            all_points.append(row)
        missing_conditions = sorted(expected_conditions - observed_conditions)
        unexpected_conditions = sorted(observed_conditions - expected_conditions)
        if missing_conditions or unexpected_conditions:
            raise ValueError(
                f"Replicate condition grid mismatch for {pair}: "
                f"missing={missing_conditions[:10]}, unexpected={unexpected_conditions[:10]}"
            )
        for task_id in {condition[0] for condition in observed_conditions}:
            task_point_sets = {
                frozenset(point_ids)
                for condition, point_ids in point_ids_by_condition.items()
                if condition[0] == task_id
            }
            if len(task_point_sets) != 1:
                raise ValueError(
                    f"Replicate does not use one exact point cohort across all conditions: "
                    f"pair={pair}, task={task_id}"
                )
        replicate_aggregate = aggregate_by_policy_horizon(points)
        if not aggregate_support_is_formal(replicate_aggregate):
            raise ValueError(
                f"Recomputed formal metric support failed for replicate {pair}: {points_path}"
            )
        prior_cohort = cohort_by_actor.get(actor_cluster)
        if prior_cohort is not None and actor_cohort != prior_cohort:
            raise ValueError(
                "WM replicates for the same actor trace seed do not share an exact point cohort: "
                f"actor_identity={actor_identity.full_key}"
            )
        cohort_by_actor[actor_cluster] = actor_cohort
        prior_immutable = immutable_rows_by_actor.get(actor_cluster)
        if prior_immutable is not None and immutable_snapshot != prior_immutable:
            raise ValueError(
                "WM replicates for one actor trace disagree on source/GT immutable fields: "
                f"actor_identity={actor_identity.full_key}"
            )
        immutable_rows_by_actor[actor_cluster] = immutable_snapshot
        input_hashes[str(quality_path)] = _sha256_file(quality_path)
        input_hashes[str(points_path)] = _sha256_file(points_path)
        input_hashes[str(identity_path)] = _sha256_file(identity_path)
        input_hashes[str(provenance_path)] = _sha256_file(provenance_path)

    trajectory_independence = _audit_actor_trajectory_independence(
        trajectory_digests_by_actor,
        actor_protocol_profile=next(
            iter({profile for profile, _replicate_id in trajectory_digests_by_actor}),
            "",
        ),
    )
    fatal_collisions = [
        collision
        for collision in trajectory_independence["identical_complete_cohorts"]
        if str(collision.get("disposition") or "").startswith("failure_")
    ]
    if fatal_collisions:
        first = fatal_collisions[0]
        raise ValueError(
            "Distinct seeded actor identities produced an identical complete "
            "observable action-trajectory cohort, so seed transport is not "
            f"accepted as effective: actor_replicate_ids={first['actor_replicate_ids']}"
        )

    joint_gate = validate_preregistered_offline_replicates(
        preregistration,
        benchmark=benchmark,
        replicate_gates=gates,
    )
    task_actor_clusters = {
        (
            str(row.get("task_id") or ""),
            str(row.get("actor_protocol_profile") or ""),
            str(row.get("actor_replicate_id") or ""),
        )
        for row in all_points
    }
    required_clusters = int(
        preregistration["payload"]["support"]["min_task_seed_clusters"]
    )
    if len(task_actor_clusters) < required_clusters:
        raise ValueError(
            "Formal replicate set lacks independent task x actor-trace clusters: "
            f"observed={len(task_actor_clusters)}, required={required_clusters}"
        )
    aggregate_rows = aggregate_by_policy_horizon(all_points)
    if not aggregate_rows:
        raise ValueError("Joint postflight produced no aggregate rows")
    for row in aggregate_rows:
        row["aggregate_scope"] = "complete_preregistered_replicate_set"
        row["confirmatory_complete"] = True
        row["replicate_count"] = joint_gate["observed_replicate_count"]
        row["preregistration_sha256"] = joint_gate["preregistration_sha256"]
        row["protocol_sha256"] = joint_gate["protocol_sha256"]

    output_dir = Path(output_dir).expanduser().resolve()
    figure_dir = Path(figure_dir).expanduser().resolve()
    _require_empty_output_roots((output_dir, figure_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    points_out = output_dir / f"{benchmark}_formal_combined_prediction_points.csv"
    aggregate_out = output_dir / f"{benchmark}_formal_aggregate_by_policy_horizon.csv"
    quality_out = output_dir / f"{benchmark}_formal_postflight.json"
    _write_csv(points_out, all_points)
    _write_csv(aggregate_out, aggregate_rows)
    audit = {
        **joint_gate,
        "formal_plot_ready": True,
        "point_row_count": len(all_points),
        "analysis_dirs": [str(path) for path in directories],
        "input_sha256": input_hashes,
        "actor_trajectory_independence": trajectory_independence,
        "combined_points_sha256": _sha256_file(points_out),
        "combined_aggregate_sha256": _sha256_file(aggregate_out),
    }
    prefix = "isbench_" if benchmark == "isbench" else ""
    title = (
        "SafePred-ISBench prediction-only"
        if benchmark == "isbench"
        else "SafePred-HarnessAudit prediction-only"
    )
    tier1 = figure_dir / f"{prefix}tier1_risk_detection_by_horizon.png"
    tier2 = figure_dir / f"{prefix}tier2_code_match_by_horizon.png"
    tokens = figure_dir / f"{prefix}token_cost_input_output_by_horizon.png"
    summary = figure_dir / f"{prefix}predict_only_summary_by_horizon.png"
    plot_tier1_horizon(aggregate_rows, tier1, experiment_title=title)
    plot_tier2_horizon(aggregate_rows, tier2, experiment_title=title)
    plot_token_split_horizon(aggregate_rows, tokens, experiment_title=title)
    plot_pilot_summary(
        aggregate_rows,
        summary,
        experiment_title=f"{title} sparse horizon pilot",
    )
    temporary_quality = quality_out.with_name(f".{quality_out.name}.tmp")
    temporary_quality.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_quality.replace(quality_out)
    return {
        "points": points_out,
        "aggregate": aggregate_out,
        "postflight": quality_out,
        "tier1_plot": tier1,
        "tier2_plot": tier2,
        "token_plot": tokens,
        "summary_plot": summary,
    }


def _audit_actor_trajectory_independence(
    trajectory_digests_by_actor: dict[tuple[str, str], dict[str, str]],
    *,
    actor_protocol_profile: str,
) -> dict[str, Any]:
    """Diagnose task collisions and reject only an identical full cohort.

    A short or constrained task can legitimately yield the same action path
    for two independent seeds.  Rejecting that observation would encourage
    behavior-dependent seed replacement.  A complete cohort match is the
    fail-closed signal that seed transport may have been ignored.
    """

    complete_seen: dict[str, tuple[str, str]] = {}
    identical_complete: list[dict[str, Any]] = []
    task_seen: dict[tuple[str, str], list[tuple[str, str]]] = {}
    cohort_sha256_by_actor: dict[str, str] = {}
    for actor_identity, task_digests in sorted(trajectory_digests_by_actor.items()):
        profile, replicate_id = actor_identity
        canonical = json.dumps(
            dict(sorted(task_digests.items())),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cohort_digest = hashlib.sha256(canonical).hexdigest()
        cohort_sha256_by_actor[f"{profile}:{replicate_id}"] = cohort_digest
        prior_actor = complete_seen.get(cohort_digest)
        if prior_actor is not None and prior_actor != actor_identity:
            collision_ids = [prior_actor[1], replicate_id]
            identical_complete.append(
                {
                    "cohort_trajectory_sha256": cohort_digest,
                    "actor_replicate_ids": collision_ids,
                    "disposition": (
                        "failure_seeded_sampling_collision"
                        if actor_protocol_profile == "seeded_sampling_v1"
                        else "diagnostic_deterministic_close_source"
                    ),
                }
            )
        else:
            complete_seen[cohort_digest] = actor_identity
        for task_id, digest in sorted(task_digests.items()):
            task_seen.setdefault((task_id, digest), []).append(actor_identity)
    task_collisions = [
        {
            "task_id": task_id,
            "trajectory_sha256": digest,
            "actor_replicate_ids": sorted(
                {replicate_id for _profile, replicate_id in actor_identities}
            ),
            "disposition": "diagnostic_per_task_collision",
        }
        for (task_id, digest), actor_identities in sorted(task_seen.items())
        if len(set(actor_identities)) > 1
    ]
    return {
        "criterion": "profile_aware_complete_cohort_collision_v2",
        "cohort_trajectory_sha256_by_actor_identity": cohort_sha256_by_actor,
        "identical_complete_cohorts": identical_complete,
        "task_trajectory_collisions": task_collisions,
    }


def _mapping_actor_identity(
    value: Mapping[str, Any],
    *,
    benchmark: str,
    allow_csv_seed: bool = False,
) -> ActorReplicateIdentity:
    profile = value.get("actor_protocol_profile")
    replicate_id = value.get("actor_replicate_id")
    raw_seed = value.get("actor_trace_seed")
    if isinstance(raw_seed, str) and not raw_seed.strip():
        raw_seed = None
    elif allow_csv_seed and isinstance(raw_seed, str):
        raw_seed = _strict_int(raw_seed, "point.actor_trace_seed")
    return normalize_actor_replicate_identity(
        benchmark=benchmark,
        actor_protocol_profile=profile,
        actor_replicate_id=replicate_id,
        actor_trace_seed=raw_seed,
    )


def _formal_identity_marker_actor_identity(
    value: Mapping[str, Any],
    *,
    benchmark: str,
) -> ActorReplicateIdentity:
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError(
            "formal_replicate_identity schema_version must equal 1 or 2"
        )
    if schema_version == 1:
        if (
            "actor_protocol_profile" in value
            or "actor_replicate_id" in value
            or type(value.get("actor_trace_seed")) is not int
        ):
            raise ValueError(
                "formal_replicate_identity schema-1 is seeded-only and requires "
                "an exact integer actor_trace_seed"
            )
        return normalize_actor_replicate_identity(
            benchmark=benchmark,
            actor_trace_seed=value["actor_trace_seed"],
        )
    required = {
        "actor_protocol_profile",
        "actor_replicate_id",
        "actor_trace_seed",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            "formal_replicate_identity schema-2 lacks explicit actor identity "
            f"fields: missing={missing}"
        )
    return _mapping_actor_identity(value, benchmark=benchmark)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read postflight JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Postflight JSON must be an object: {path}")
    return payload


def _current_implementation_manifest_sha256(benchmark: str) -> str:
    """Recompute the code identity used to create and aggregate formal runs."""

    if benchmark == "harnessaudit":
        from multi_agent.safepred_experiments.predict_only_pipeline import (
            _ha_implementation_manifest_digest,
        )

        return _ha_implementation_manifest_digest()
    if benchmark == "isbench":
        from multi_agent.safepred_experiments.isbench_predict_only_pipeline import (
            _isbench_implementation_manifest_sha256,
        )

        return _isbench_implementation_manifest_sha256()
    raise ValueError(f"Unsupported benchmark: {benchmark!r}")


def _require_empty_output_roots(paths: Iterable[Path]) -> None:
    """Prevent stale formal-ready files from surviving or being overwritten."""

    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            raise ValueError(f"Formal postflight output root is a file: {path}")
        if path.is_dir() and any(path.iterdir()):
            raise ValueError(
                "Formal postflight output root already contains artifacts; use a "
                f"fresh directory so stale plot-ready state cannot survive: {path}"
            )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ValueError(f"Cannot read postflight CSV {path}: {exc}") from exc


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _verify_provenance_artifacts(
    provenance_path: Path,
    *,
    required_paths: Iterable[Path],
) -> None:
    provenance = _load_json(provenance_path)
    recorded_payload_sha = str(provenance.pop("manifest_payload_sha256", ""))
    canonical = json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != recorded_payload_sha:
        raise ValueError(f"Provenance payload SHA mismatch: {provenance_path}")
    artifacts = {
        str(Path(row.get("path") or "").expanduser().absolute()): row
        for row in provenance.get("artifacts") or []
        if isinstance(row, dict)
    }
    for raw_path in required_paths:
        path = Path(raw_path).expanduser().absolute()
        record = artifacts.get(str(path))
        if not record or record.get("exists") is not True:
            raise ValueError(
                f"Provenance does not bind required formal artifact {path}: {provenance_path}"
            )
        if str(record.get("sha256") or "") != _sha256_file(path):
            raise ValueError(
                f"Formal artifact differs from provenance SHA: {path}"
            )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _strict_int(value: Any, field: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, observed={value!r}") from exc
    if text not in {str(parsed), f"+{parsed}"}:
        raise ValueError(f"{field} must be an exact integer, observed={value!r}")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(QUALITY_FILENAMES))
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--analysis-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    artifacts = run_postflight(
        benchmark=args.benchmark,
        preregistration_path=args.preregistration,
        analysis_dirs=args.analysis_dirs,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
