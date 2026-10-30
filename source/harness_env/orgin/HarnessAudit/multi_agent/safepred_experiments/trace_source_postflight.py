"""Fail-fast joint audit for frozen actor-trace source replicates.

Run this after all actor seeds have produced source traces and before any
world-model calls.  It moves the existing joint action-trajectory independence
check from the expensive WM postflight to the source-generation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from multi_agent.safepred_experiments.isbench_report_audit import audit_reports
from multi_agent.safepred_experiments.isbench_predict_only_pipeline import (
    _isbench_source_action_trajectory_digests,
)
from multi_agent.safepred_experiments.formal_contract import (
    ActorReplicateIdentity,
    normalize_actor_replicate_identity,
    read_ha_source_replicate_identity,
)
from multi_agent.safepred_experiments.predict_only_pipeline import (
    _ha_source_action_trajectory_digests,
    audit_clean_source_reports,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report root must be an object: {path}")
    return value


def _read_report_list(path: Path) -> list[Path]:
    base = Path(path).expanduser().resolve().parent
    reports: list[Path] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve()
        if not candidate.is_file() or (
            candidate.name != "report.json" and candidate.suffix != ".json"
        ):
            raise ValueError(f"source report is missing or not JSON: {candidate}")
        reports.append(candidate)
    unique = list(dict.fromkeys(reports))
    if not unique:
        raise ValueError(f"source report list is empty: {path}")
    return unique


def _parse_replicate(value: str) -> tuple[int | None, Path]:
    raw_seed, separator, raw_path = str(value).partition("=")
    if not separator:
        path = Path(str(value)).expanduser().resolve()
        return None, path
    if not raw_seed.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "replicate must be /path/to/reports.txt or "
            "SEED=/path/to/reports.txt"
        )
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid actor seed: {raw_seed!r}") from exc
    return seed, Path(raw_path).expanduser().resolve()


def _stable_protocol(benchmark: str, report: dict[str, Any]) -> dict[str, Any]:
    if benchmark == "isbench":
        metadata = report.get("experiment_metadata") or {}
        protocol = dict(metadata.get("generation_protocol") or {})
        protocol.pop("task_config_sha256", None)
        return protocol
    provenance = report.get("actor_provenance") or {}
    hashes = provenance.get("hashes") or {}
    return {
        key: provenance.get(key)
        for key in (
            "schema_version",
            "framework",
            "harness",
            "model",
            "model_revision",
            "model_revision_source",
            "runtime_revision",
            "runtime_revision_source",
            "runtime_versions",
            "endpoint",
            "actor_protocol_profile",
            "actor_temperature",
            "actor_temperature_requested",
            "actor_parallel_tool_calls",
            "actor_seed_required",
            "seed_transport",
            "max_turns",
        )
    } | {
        "implementation_manifest_sha256": hashes.get(
            "implementation_manifest_sha256"
        )
    }


def _stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actor_seed_transport_mode(
    benchmark: str,
    report: dict[str, Any],
) -> str:
    """Classify whether actor request seeds provably affected generation."""

    if benchmark == "isbench":
        metadata = report.get("experiment_metadata")
        if not isinstance(metadata, dict):
            return "unknown"
        protocol = metadata.get("generation_protocol")
        attempts = report.get("actor_generation_attempts")
        if (
            not isinstance(protocol, dict)
            or not isinstance(attempts, list)
            or not attempts
            or any(not isinstance(attempt, dict) for attempt in attempts)
            or any("request_seed" not in attempt for attempt in attempts)
        ):
            return "unknown"
        provider_mode = str(protocol.get("provider_mode") or "").strip().lower()
        retry_policy = str(
            protocol.get("actor_retry_seed_policy") or ""
        ).strip()
        request_seeds = [attempt.get("request_seed") for attempt in attempts]
        if (
            provider_mode == "local"
            and retry_policy == "base_plus_validation_attempt_v1"
            and all(
                isinstance(seed, int) and not isinstance(seed, bool)
                for seed in request_seeds
            )
        ):
            return "seeded"
        try:
            temperature = float(protocol.get("actor_temperature"))
        except (TypeError, ValueError):
            return "unknown"
        if (
            provider_mode == "close_source"
            and retry_policy == "none"
            and temperature == 0.0
            and all(seed is None for seed in request_seeds)
        ):
            return "seedless"
        return "unknown"

    provenance = report.get("actor_provenance")
    if not isinstance(provenance, dict):
        return "unknown"
    seed_required = provenance.get("actor_seed_required")
    seed_transport = str(provenance.get("seed_transport") or "").strip().lower()
    if seed_required is True and seed_transport not in {"", "none"}:
        return "seeded"
    if seed_required is False and seed_transport in {"", "none"}:
        return "seedless"
    return "unknown"


def _task_id(benchmark: str, report: dict[str, Any]) -> str:
    return str(report.get("task") if benchmark == "isbench" else report.get("task_id") or "").strip()


def _audit_replicate(
    *,
    benchmark: str,
    seed: int | None,
    reports: list[Path],
    tasks_dir: Path | None,
) -> dict[str, Any]:
    loaded = [_load_json(path) for path in reports]
    task_ids = [_task_id(benchmark, report) for report in loaded]
    if any(not task for task in task_ids):
        raise ValueError(f"replicate contains a report without task id: {reports}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"replicate contains duplicate tasks: {reports}")

    if benchmark == "isbench":
        if seed is None:
            raise ValueError(
                "ISBench remains seeded; use SEED=/path/to/reports.txt"
            )
        identity = normalize_actor_replicate_identity(
            benchmark="isbench",
            actor_trace_seed=seed,
        )
        if tasks_dir is None:
            raise ValueError("--tasks-dir is required for ISBench postflight")
        audit = audit_reports(
            reports,
            recommended_count=0,
            tasks_dir=tasks_dir,
            require_complete_safety_labels=True,
            require_explicit_process_timing=True,
            require_image_quality=True,
            require_successful=False,
            require_generation_provenance=True,
            expected_experiment_seed=seed,
        )
        invalid = [row for row in audit["reports"] if not row.get("is_eligible")]
        if invalid:
            first = invalid[0]
            raise ValueError(
                f"ISBench actor_seed={seed} source audit failed: "
                f"{first.get('path')}: {first.get('generation_provenance_errors') or first.get('safety_label_errors') or first.get('image_quality_errors') or first.get('termination_reason')}"
            )
        digests = _isbench_source_action_trajectory_digests(reports)
    else:
        identity = read_ha_source_replicate_identity(
            reports,
            expected_actor_trace_seed=seed,
        )
        audit = audit_clean_source_reports(reports)
        invalid = [row for row in audit["reports"] if not row.get("is_clean_replayable")]
        if invalid:
            first = invalid[0]
            raise ValueError(
                f"HarnessAudit actor_seed={seed} source audit failed: "
                f"{first.get('report_path')}: {first.get('errors')}"
            )
        digests = _ha_source_action_trajectory_digests(reports)

    protocols = {_stable_json_digest(_stable_protocol(benchmark, report)) for report in loaded}
    if len(protocols) != 1:
        raise ValueError(
            f"actor_seed={seed} reports do not share one frozen actor protocol"
        )
    seed_transport_modes = {
        _actor_seed_transport_mode(benchmark, report)
        for report in loaded
    }
    if len(seed_transport_modes) != 1:
        raise ValueError(
            f"actor_seed={seed} reports use mixed actor seed transport modes: "
            f"{sorted(seed_transport_modes)}"
        )
    seed_transport_mode = next(iter(seed_transport_modes))
    if seed_transport_mode == "unknown":
        raise ValueError(
            f"actor_seed={seed} reports cannot prove actor seed transport "
            "from generation protocol and request provenance"
        )
    return {
        "actor_protocol_profile": identity.actor_protocol_profile,
        "actor_replicate_id": identity.actor_replicate_id,
        "actor_trace_seed": identity.actor_trace_seed,
        "actor_seed_transport_mode": seed_transport_mode,
        "task_ids": sorted(task_ids),
        "scene_by_task": (
            {
                _task_id(benchmark, report): str(report.get("scene") or "").strip()
                for report in loaded
            }
            if benchmark == "isbench"
            else {}
        ),
        "task_bound_protocol_sha256_by_task": {
            _task_id(benchmark, report): str(
                (
                    (report.get("experiment_metadata") or {}).get(
                        "generation_protocol_sha256"
                    )
                    if benchmark == "isbench"
                    else (report.get("actor_provenance") or {}).get(
                        "actor_stable_protocol_sha256"
                    )
                )
                or ""
            )
            for report in loaded
        },
        "trajectory_sha256_by_task": dict(sorted(digests.items())),
        "stable_actor_protocol_sha256": next(iter(protocols)),
        "report_paths": [str(path) for path in reports],
    }


def validate_joint_sources(replicates: Iterable[dict[str, Any]], *, min_seeds: int) -> dict[str, Any]:
    rows = [dict(row) for row in replicates]
    identities = [_row_actor_identity(row) for row in rows]
    for row, identity in zip(rows, identities):
        row["actor_protocol_profile"] = identity.actor_protocol_profile
        row["actor_replicate_id"] = identity.actor_replicate_id
        row["actor_trace_seed"] = identity.actor_trace_seed
    identity_keys = [identity.full_key for identity in identities]
    profiles = {identity.actor_protocol_profile for identity in identities}
    seeds = [
        identity.actor_trace_seed
        for identity in identities
        if identity.actor_trace_seed is not None
    ]
    errors: list[str] = []
    if len(identity_keys) != len(set(identity_keys)):
        errors.append("normalized actor replicate identities are not unique")
    if len(identity_keys) < int(min_seeds):
        errors.append(
            f"observed actor identities={len(identity_keys)} required={int(min_seeds)}"
        )
    if len(profiles) != 1:
        errors.append(
            f"mixed actor protocol profiles are not allowed: {sorted(profiles)}"
        )
    if profiles == {"seeded_sampling_v1"} and len(seeds) != len(set(seeds)):
        errors.append("transported actor trace seeds are not unique")
    task_sets = {tuple(row["task_ids"]) for row in rows}
    if len(task_sets) != 1:
        errors.append("actor-seed replicates do not contain the same exact task set")
    scene_maps = {
        _stable_json_digest(dict(sorted((row.get("scene_by_task") or {}).items())))
        for row in rows
    }
    if len(scene_maps) != 1:
        errors.append(
            "actor-seed replicates do not contain the same exact task-to-scene pairing"
        )
    task_bound_protocols = {
        _stable_json_digest(
            dict(
                sorted(
                    (row.get("task_bound_protocol_sha256_by_task") or {}).items()
                )
            )
        )
        for row in rows
    }
    if len(task_bound_protocols) != 1:
        errors.append(
            "actor-seed replicates do not share identical task-bound actor "
            "inputs/provenance"
        )
    protocols = {str(row["stable_actor_protocol_sha256"]) for row in rows}
    if len(protocols) != 1:
        errors.append("actor-seed replicates do not share one frozen actor protocol")
    seed_transport_modes = {
        str(row.get("actor_seed_transport_mode") or "unknown")
        for row in rows
    }
    if "unknown" in seed_transport_modes:
        errors.append(
            "cannot prove actor seed transport for every replicate from report "
            "metadata/provenance"
        )
    if len(seed_transport_modes) != 1:
        errors.append(
            "replicates use mixed actor seed transport modes: "
            f"{sorted(seed_transport_modes)}"
        )

    # A single task can legitimately choose the same action path under two
    # independent seeds, especially for short or strongly constrained tasks.
    # Treat those collisions as diagnostics, not as a reason to re-sample
    # seeds after seeing behavior.  Only an identical *complete cohort* is a
    # fail-closed indication that the actor seed likely did not take effect.
    cohort_seen: dict[str, list[ActorReplicateIdentity]] = {}
    per_task_seen: dict[
        tuple[str, str], list[ActorReplicateIdentity]
    ] = {}
    for row, identity in zip(rows, identities):
        cohort_digest = _stable_json_digest(
            dict(sorted(row["trajectory_sha256_by_task"].items()))
        )
        row["cohort_trajectory_sha256"] = cohort_digest
        cohort_seen.setdefault(cohort_digest, []).append(identity)
        for task, digest in row["trajectory_sha256_by_task"].items():
            key = (str(task), str(digest))
            per_task_seen.setdefault(key, []).append(identity)
    cohort_collisions = []
    for digest, collision_identities in sorted(cohort_seen.items()):
        unique_identities = sorted(
            set(collision_identities), key=lambda item: item.full_key
        )
        if len(unique_identities) < 2:
            continue
        if profiles == {"deterministic_close_source_v1"}:
            disposition = "diagnostic_deterministic_close_source"
            collision = {
                "actor_replicate_ids": [
                    identity.actor_replicate_id for identity in unique_identities
                ],
                "disposition": disposition,
                "trajectory_sha256": digest,
            }
        else:
            disposition = "failure_seeded_sampling_collision"
            unique_seeds = [
                int(identity.actor_trace_seed)
                for identity in unique_identities
                if identity.actor_trace_seed is not None
            ]
            errors.append(
                "distinct seeded actor identities produced an identical complete "
                "action-trajectory cohort: "
                f"actor_seeds={','.join(str(seed) for seed in unique_seeds)}"
            )
            collision = {
                "actor_seeds": unique_seeds,
                "disposition": disposition,
                "trajectory_sha256": digest,
            }
        cohort_collisions.append(collision)
    task_collisions = []
    for (task, digest), collision_identities in sorted(per_task_seen.items()):
        unique_identities = sorted(
            set(collision_identities), key=lambda item: item.full_key
        )
        if len(unique_identities) < 2:
            continue
        collision: dict[str, Any] = {
            "task": task,
            "trajectory_sha256": digest,
        }
        if profiles == {"deterministic_close_source_v1"}:
            collision["actor_replicate_ids"] = [
                identity.actor_replicate_id for identity in unique_identities
            ]
        else:
            collision["actor_seeds"] = [
                int(identity.actor_trace_seed)
                for identity in unique_identities
                if identity.actor_trace_seed is not None
            ]
        task_collisions.append(collision)
    return {
        "artifact_type": "trace_source_joint_postflight",
        "passed": not errors,
        "actor_identity_count": len(set(identity_keys)),
        "actor_seed_count": len(set(seeds)),
        "task_count": len(next(iter(task_sets), ())),
        "errors": errors,
        "cohort_trajectory_collisions": cohort_collisions,
        "task_trajectory_collisions": task_collisions,
        "replicates": rows,
    }


def _row_actor_identity(row: dict[str, Any]) -> ActorReplicateIdentity:
    expanded = (
        "actor_protocol_profile" in row or "actor_replicate_id" in row
    )
    return normalize_actor_replicate_identity(
        benchmark="harnessaudit",
        actor_protocol_profile=(
            row.get("actor_protocol_profile") if expanded else None
        ),
        actor_replicate_id=(
            row.get("actor_replicate_id") if expanded else None
        ),
        actor_trace_seed=row.get("actor_trace_seed"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("harnessaudit", "isbench"), required=True)
    parser.add_argument(
        "--replicate",
        action="append",
        type=_parse_replicate,
        required=True,
        help=(
            "Repeat /path/to/source_reports.txt for report-derived deterministic "
            "identity, or SEED=/path/to/source_reports.txt for a legacy seeded assertion."
        ),
    )
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    replicate_rows = [
        _audit_replicate(
            benchmark=args.benchmark,
            seed=seed,
            reports=_read_report_list(path),
            tasks_dir=args.tasks_dir,
        )
        for seed, path in args.replicate
    ]
    payload = validate_joint_sources(replicate_rows, min_seeds=args.min_seeds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise SystemExit("Trace source joint postflight failed: " + "; ".join(payload["errors"]))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
