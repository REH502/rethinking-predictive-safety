"""Prepare and verify one immutable HarnessAudit policy cache for paired lanes.

The perturbation experiment is diagnostic, but its four lanes are paired on
task.  They must therefore consume byte-identical policy extractions; allowing
each lane to sample its own extraction changes more than the WM-return
perturbation and invalidates the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = "ha-shared-policy-cache-v1"
EXPECTED_CACHE_SCHEMA_VERSION = 5
EXPECTED_PROMPT_VERSION = (
    "harnessaudit-train-policy-extraction-v2-no-test-tool-necessity"
)
EXPECTED_MODEL_INPUT_BOUNDARY = (
    "harnessaudit-model-input-boundary-v2-ground-truth-only-tool-necessity"
)
POLICY_ROLE_FIELDS = (
    "endpoint",
    "model",
    "revision",
    "runtime_revision",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "request_extra_body",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a model profile mapping: {path}")
    return payload


def _policy_role_contract(profile: dict[str, Any]) -> dict[str, Any]:
    roles = profile.get("roles") or {}
    role = roles.get("policy_extractor") or {}
    if not isinstance(role, dict):
        raise ValueError("Model profile lacks a policy_extractor role")
    contract = {field: role.get(field) for field in POLICY_ROLE_FIELDS}
    if not contract.get("model") or not contract.get("revision"):
        raise ValueError("Policy extractor lacks model or revision provenance")
    return contract


def _selected_tasks(cohort_path: Path | None, tasks: Iterable[str]) -> list[str]:
    explicit = [str(task) for task in tasks]
    if cohort_path is not None and explicit:
        raise ValueError("Use either --cohort-manifest or --task, not both")
    if cohort_path is not None:
        payload = _load_json(cohort_path)
        selected = payload.get("selected") or {}
        explicit = [str(task) for task in selected.get("task_ids") or []]
        if selected.get("task_count") != len(explicit):
            raise ValueError("Cohort task_count does not match task_ids")
    if not explicit or len(explicit) != len(set(explicit)):
        raise ValueError("The selected task set must be non-empty and unique")
    if any(Path(task).stem != task for task in explicit):
        raise ValueError("Task selectors must be canonical task ids")
    return explicit


def _source_cache_path(source_root: Path, split_id: str, task_id: str) -> Path:
    candidates = [
        source_root
        / task_id
        / "robosafe_cache"
        / split_id
        / f"{task_id}.json",
        source_root / "robosafe_cache" / split_id / f"{task_id}.json",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one frozen source cache for {task_id}; found {len(matches)}"
        )
    return matches[0].resolve()


def _validate_cache(
    payload: dict[str, Any],
    *,
    task_id: str,
    split: dict[str, Any],
    policy_model: str,
) -> None:
    expected = {
        "status": "success",
        "schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "policy_cache_schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "test_task_id": task_id,
        "split_id": split.get("split_id"),
        "split_strategy": split.get("split_strategy"),
        "split_label": split.get("split_label"),
        "method": split.get("method"),
        "evaluation_cohort_id": split.get("evaluation_cohort_id"),
        "evaluation_cohort_manifest_path": split.get(
            "evaluation_cohort_manifest"
        ),
        "evaluation_cohort_manifest_sha256": split.get(
            "evaluation_cohort_manifest_sha256"
        ),
        "evaluation_task_ids_sha256": split.get("evaluation_task_ids_sha256"),
        "policy_extraction_model": policy_model,
        "policy_extraction_seed": 0,
        "policy_extraction_prompt_version": EXPECTED_PROMPT_VERSION,
        "model_input_boundary_version": EXPECTED_MODEL_INPUT_BOUNDARY,
        "resource_semantics_mode": "legacy_official_v1",
        "include_test_tool_necessity_policies": False,
        "input_includes_test_access_rules": False,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Frozen cache compatibility mismatch for {task_id}: {','.join(mismatches)}"
        )
    access_rules = payload.get("access_rules")
    policies = payload.get("safepred_policies")
    training_ids = payload.get("training_task_ids")
    if (
        not isinstance(access_rules, list)
        or not access_rules
        or not isinstance(policies, list)
        or len(policies) != len(access_rules)
        or not isinstance(training_ids, list)
        or task_id in training_ids
        or payload.get("training_task_count") != len(training_ids)
    ):
        raise ValueError(f"Frozen cache structure is invalid for {task_id}")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    destination_root = Path(args.destination_root).resolve()
    recorded_destination_root = (
        Path(args.recorded_destination_root).resolve()
        if args.recorded_destination_root
        else destination_root
    )
    manifest_path = Path(args.manifest_path).resolve()
    split_path = Path(args.cross_fit_split).resolve()
    current_profile_path = Path(args.current_model_profile).resolve()
    source_profile_path = Path(args.source_model_profile).resolve()
    source_serving_path = Path(args.source_serving_contract).resolve()
    current_service_path = Path(args.current_service_info).resolve()
    cohort_path = (
        Path(args.cohort_manifest).resolve() if args.cohort_manifest else None
    )
    for path in (
        source_root,
        split_path,
        current_profile_path,
        source_profile_path,
        source_serving_path,
        current_service_path,
    ):
        if not path.exists():
            raise ValueError(f"Required cache provenance input is missing: {path}")
    if destination_root.exists():
        raise ValueError(f"Refusing an existing shared cache root: {destination_root}")
    if manifest_path.exists():
        raise ValueError(f"Refusing an existing shared cache manifest: {manifest_path}")

    tasks = _selected_tasks(cohort_path, args.task)
    split = _load_json(split_path)
    current_profile = _load_profile(current_profile_path)
    source_profile = _load_profile(source_profile_path)
    current_role = _policy_role_contract(current_profile)
    source_role = _policy_role_contract(source_profile)
    if current_role != source_role:
        raise ValueError("Cache source policy-extractor contract differs from current role")

    source_serving = _load_json(source_serving_path)
    current_service = _load_json(current_service_path)
    for source_key, current_key in (
        ("model_path", "model_path"),
        ("served_model_name", "served_model_name"),
    ):
        if source_serving.get(source_key) != current_service.get(current_key):
            raise ValueError(f"Cache source serving identity mismatch: {source_key}")
    if source_serving.get("backend") != "vllm":
        raise ValueError("Cache source was not generated through vLLM")

    split_id = str(split.get("split_id") or "")
    if not split_id:
        raise ValueError("Cross-fit split lacks split_id")
    destination_data_root = destination_root / "robosafe_cache" / split_id
    destination_data_root.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for task_id in tasks:
        source_path = _source_cache_path(source_root, split_id, task_id)
        payload = _load_json(source_path)
        _validate_cache(
            payload,
            task_id=task_id,
            split=split,
            policy_model=str(current_role["model"]),
        )
        prompt_hash = str(payload.get("prompt_hash") or "")
        if len(prompt_hash) != 64:
            raise ValueError(f"Frozen cache prompt hash is invalid for {task_id}")
        prompt_hashes.add(prompt_hash)
        destination_path = destination_data_root / f"{task_id}.json"
        shutil.copy2(source_path, destination_path)
        source_sha = _sha256_file(source_path)
        if _sha256_file(destination_path) != source_sha:
            raise ValueError(f"Frozen cache copy drift for {task_id}")
        destination_path.chmod(0o444)
        entries.append(
            {
                "task_id": task_id,
                "relative_path": destination_path.relative_to(
                    destination_root
                ).as_posix(),
                "source_path": str(source_path),
                "size": destination_path.stat().st_size,
                "sha256": source_sha,
                "access_rules_sha256": _canonical_sha256(payload["access_rules"]),
                "safepred_policies_sha256": _canonical_sha256(
                    payload["safepred_policies"]
                ),
            }
        )
    for directory in sorted(
        (path for path in destination_root.rglob("*") if path.is_dir()),
        reverse=True,
    ):
        directory.chmod(0o555)
    destination_root.chmod(0o555)

    entry_identity = [
        {"task_id": row["task_id"], "size": row["size"], "sha256": row["sha256"]}
        for row in entries
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "task_count": len(tasks),
        "task_ids_sha256": _canonical_sha256(tasks),
        "cache_identity_sha256": _canonical_sha256(entry_identity),
        "split_id": split_id,
        "policy_extractor_contract": current_role,
        "policy_extractor_contract_sha256": _canonical_sha256(current_role),
        "prompt_hashes": sorted(prompt_hashes),
        "source_root": str(source_root),
        "destination_root": str(recorded_destination_root),
        "source_model_profile": {
            "path": str(source_profile_path),
            "sha256": _sha256_file(source_profile_path),
        },
        "current_model_profile": {
            "path": str(current_profile_path),
            "sha256": _sha256_file(current_profile_path),
        },
        "source_serving_contract": {
            "path": str(source_serving_path),
            "sha256": _sha256_file(source_serving_path),
        },
        "current_service_info": {
            "path": str(current_service_path),
            "sha256": _sha256_file(current_service_path),
        },
        "entries": entries,
        "lane_cache_contract": "byte_identical_private_copies_read_only_v1",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_path).resolve()
    cache_root = Path(args.cache_root).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise ValueError("Shared policy-cache manifest is not valid")
    entries = manifest.get("entries") or []
    if not isinstance(entries, list) or len(entries) != manifest.get("task_count"):
        raise ValueError("Shared policy-cache manifest has an invalid denominator")
    observed_identity = []
    for row in entries:
        path = cache_root / str(row.get("relative_path") or "")
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"Shared policy-cache identity drift: {row.get('task_id')}")
        observed_identity.append(
            {
                "task_id": row.get("task_id"),
                "size": row.get("size"),
                "sha256": row.get("sha256"),
            }
        )
    if _canonical_sha256(observed_identity) != manifest.get("cache_identity_sha256"):
        raise ValueError("Shared policy-cache aggregate identity drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "task_count": len(entries),
        "cache_identity_sha256": manifest["cache_identity_sha256"],
        "cache_root": str(cache_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.add_argument("--destination-root", required=True)
    prepare_parser.add_argument("--recorded-destination-root")
    prepare_parser.add_argument("--manifest-path", required=True)
    prepare_parser.add_argument("--cross-fit-split", required=True)
    prepare_parser.add_argument("--current-model-profile", required=True)
    prepare_parser.add_argument("--source-model-profile", required=True)
    prepare_parser.add_argument("--source-serving-contract", required=True)
    prepare_parser.add_argument("--current-service-info", required=True)
    selection = prepare_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cohort-manifest")
    selection.add_argument("--task", action="append", default=[])

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest-path", required=True)
    verify_parser.add_argument("--cache-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    else:
        result = verify(args)
    summary = {
        key: result[key]
        for key in ("schema_version", "status", "task_count", "cache_identity_sha256")
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
