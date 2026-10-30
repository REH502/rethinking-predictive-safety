"""Fail-closed audit for paired HarnessAudit perturbation policy caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prepare_ha_shared_policy_cache import SCHEMA_VERSION, verify


EXPECTED_LANES = (
    "empty",
    "corrupt",
    "irrelevant",
    "filler",
    "corrupt_key_random_chars",
    "oracle_correct_h1",
)
EXPECTED_SOURCE_POLICY = "local_empty_irrelevant_oracle_h1_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _verify_exact_cache_tree(
    *, cache_root: Path, manifest_path: Path, expected_relative_paths: set[str]
) -> dict[str, Any]:
    summary = verify(
        SimpleNamespace(
            manifest_path=str(manifest_path),
            cache_root=str(cache_root),
        )
    )
    observed = {
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    if observed != expected_relative_paths:
        raise ValueError("Policy cache contains missing or unexpected files")
    writable = [
        path
        for path in cache_root.rglob("*")
        if path.is_file() and path.stat().st_mode & 0o222
    ]
    if writable:
        raise ValueError("Policy cache contains writable files")
    return summary


def audit(run_root: Path, expected_task_count: int) -> dict[str, Any]:
    root = run_root.resolve()
    manifest_path = root / "shared_policy_cache_manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise ValueError("Shared policy-cache manifest did not pass")
    entries = manifest.get("entries") or []
    if not isinstance(entries, list) or len(entries) != expected_task_count:
        raise ValueError("Shared policy-cache task denominator mismatch")
    task_ids = [str(row.get("task_id") or "") for row in entries]
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Shared policy-cache task ids are invalid")
    entries_by_task = {str(row["task_id"]): row for row in entries}
    expected_relative_paths = {str(row["relative_path"]) for row in entries}
    expected_identity = str(manifest.get("cache_identity_sha256") or "")
    if manifest.get("destination_root") != str((root / "shared_policy_cache").resolve()):
        raise ValueError("Shared policy-cache destination provenance mismatch")
    master_summary = _verify_exact_cache_tree(
        cache_root=root / "shared_policy_cache",
        manifest_path=manifest_path,
        expected_relative_paths=expected_relative_paths,
    )
    if master_summary.get("cache_identity_sha256") != expected_identity:
        raise ValueError("Shared policy-cache master identity mismatch")

    design = _load(root / "design.json")
    design_cache = design.get("shared_policy_cache") or {}
    if (
        design_cache.get("status") != "passed"
        or design_cache.get("task_count") != expected_task_count
        or design_cache.get("cache_identity_sha256") != expected_identity
        or design_cache.get("manifest_sha256") != _sha256(manifest_path)
        or design_cache.get("lane_cache_contract")
        != "byte_identical_private_copies_read_only_v1"
    ):
        raise ValueError("Experiment design is not bound to the shared cache")

    if (
        design.get("lane_count") != len(EXPECTED_LANES)
        or design.get("max_concurrent_lanes") != 4
        or design.get("lane_execution_schedule_version")
        != "work_conserving_max4_v1"
        or design.get("expected_condition_denominator")
        != expected_task_count * len(EXPECTED_LANES)
    ):
        raise ValueError("Experiment design six-lane denominator mismatch")

    lanes_root = root / "lanes"
    observed_lanes = {path.name for path in lanes_root.iterdir() if path.is_dir()}
    if observed_lanes != set(EXPECTED_LANES):
        raise ValueError("Perturbation lane set mismatch")

    lane_summaries: dict[str, Any] = {}
    for lane in EXPECTED_LANES:
        lane_root = lanes_root / lane
        analysis = lane_root / "harnessaudit" / "analysis"
        cache_root = analysis / "policy_cache"
        cache_summary = _verify_exact_cache_tree(
            cache_root=cache_root,
            manifest_path=manifest_path,
            expected_relative_paths=expected_relative_paths,
        )
        invocation = _load(lane_root / "lane_invocation.json")
        verification = _load(lane_root / "policy_cache_verification.json")
        if any(
            payload.get("status") != "passed"
            or payload.get("task_count") != expected_task_count
            or payload.get("cache_identity_sha256") != expected_identity
            for payload in (cache_summary, verification)
        ):
            raise ValueError(f"Lane cache verification mismatch: {lane}")
        if (
            invocation.get("shared_policy_cache_identity_sha256") != expected_identity
            or invocation.get("shared_policy_cache_task_count") != expected_task_count
        ):
            raise ValueError(f"Lane invocation cache identity mismatch: {lane}")
        if (
            invocation.get("lane") != lane
            or invocation.get("signal_source_policy") != EXPECTED_SOURCE_POLICY
            or invocation.get("intervention_modes") != ["distributed_enforced"]
            or invocation.get("wm_return_perturbations") != [lane]
            or invocation.get("workers") != 1
        ):
            raise ValueError(f"Lane invocation scientific identity mismatch: {lane}")

        manifest_rows = [
            json.loads(line)
            for line in (analysis / "run_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        observed_tasks = [Path(str(row.get("task") or "")).stem for row in manifest_rows]
        if (
            len(manifest_rows) != expected_task_count
            or len(observed_tasks) != len(set(observed_tasks))
            or set(observed_tasks) != set(task_ids)
        ):
            raise ValueError(f"Lane manifest denominator mismatch: {lane}")
        for row, task_id in zip(manifest_rows, observed_tasks):
            expected = entries_by_task[task_id]
            binding = row.get("policy_cache_binding") or {}
            if (
                binding.get("sha256") != expected.get("sha256")
                or binding.get("size") != expected.get("size")
            ):
                raise ValueError(f"Condition cache binding mismatch: {lane}/{task_id}")
            binding_path = Path(str(binding.get("path") or "")).resolve()
            try:
                relative = binding_path.relative_to(cache_root.resolve()).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"Condition cache escaped its lane root: {lane}/{task_id}"
                ) from exc
            if relative != expected.get("relative_path"):
                raise ValueError(f"Condition cache path mismatch: {lane}/{task_id}")

        provenance = _load(analysis / "online_matrix_provenance_manifest.json")
        config = provenance.get("config") or {}
        policy_contract = (config.get("online_advisory_contract") or {}).get(
            "policy_extraction_contract"
        ) or {}
        if (
            config.get("require_prepared_policy_cache") is not True
            or policy_contract.get("require_precomputed_cache") is not True
            or policy_contract.get("write_cache") is not False
        ):
            raise ValueError(f"Lane did not enforce prepared-cache-only mode: {lane}")
        policy_request_rows = _nonempty_jsonl_rows(
            analysis / "policy_extraction_request_attempts.jsonl"
        )
        policy_request_rows += sum(
            _nonempty_jsonl_rows(path)
            for path in (analysis / "policy_preparation").glob(
                "*/request_attempts.jsonl"
            )
        )
        if policy_request_rows:
            raise ValueError(f"Lane regenerated policy data online: {lane}")
        lane_summaries[lane] = {
            "manifest_row_count": len(manifest_rows),
            "unique_task_count": len(set(observed_tasks)),
            "cache_identity_sha256": expected_identity,
            "policy_extraction_request_row_count": 0,
            "require_prepared_policy_cache": True,
        }

    return {
        "schema_version": "ha-shared-policy-cache-run-audit-v1",
        "status": "passed",
        "task_count": expected_task_count,
        "lane_count": len(EXPECTED_LANES),
        "cache_identity_sha256": expected_identity,
        "lane_summaries": lane_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-task-count", required=True, type=int)
    args = parser.parse_args()
    result = audit(Path(args.run_root), args.expected_task_count)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
