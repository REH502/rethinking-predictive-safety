"""Native, pre-execution provenance for HarnessAudit actor traces.

The provenance payload built here is deliberately independent of report
post-processing.  ``BenchmarkRunner`` creates it after the effective task is
known but before the framework adapter is invoked, writes that exact payload
to ``trace_start``, and passes the same runtime values to the adapter.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from multi_agent.resource_semantics import normalize_resource_semantics_mode


PROVENANCE_SCHEMA_VERSION = 1
REQUIRED_RUNTIME_DISTRIBUTIONS = ("openai-agents", "openai", "pydantic")
SEEDED_SAMPLING_PROFILE = "seeded_sampling_v1"
DETERMINISTIC_CLOSE_SOURCE_PROFILE = "deterministic_close_source_v1"


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_manifest(path: Path, *, patterns: tuple[str, ...] = ("*",)) -> dict[str, Any]:
    """Return a content-addressed manifest without absolute-path dependence."""

    root = Path(path)
    if not root.exists():
        payload: dict[str, Any] = {"exists": False, "files": []}
        return {**payload, "sha256": stable_json_sha256(payload)}
    if root.is_file():
        files = [{"path": root.name, "sha256": sha256_file(root)}]
    else:
        candidates: set[Path] = set()
        for pattern in patterns:
            candidates.update(item for item in root.rglob(pattern) if item.is_file())
        files = [
            {
                "path": item.relative_to(root).as_posix(),
                "sha256": sha256_file(item),
            }
            for item in sorted(candidates)
        ]
    payload = {"exists": True, "files": files}
    return {**payload, "sha256": stable_json_sha256(payload)}


def implementation_manifest(package_root: Path) -> dict[str, Any]:
    """Hash runtime source while never traversing generated artifact trees."""

    package_root = Path(package_root)
    candidates = set(package_root.glob("*.py"))
    directory_names = [
        # SafePred package roots.
        "adapters",
        "agent",
        "config",
        "core",
        "models",
        "utils",
        # HarnessAudit multi_agent package roots.
        "banks",
        "frameworks",
        "safepred_experiments",
        "safepred_guard",
        "schemas",
        "trace",
    ]
    if package_root.name == "og_ego_prim":
        # IS-Bench execution, evaluation, and primitive semantics live outside
        # models/utils and must be bound by the formal implementation digest.
        directory_names.extend(("benchmark", "cli", "primitives"))
    for directory_name in directory_names:
        source_dir = package_root / directory_name
        if source_dir.is_dir():
            candidates.update(source_dir.rglob("*.py"))
    files = [
        {
            "path": item.relative_to(package_root).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in sorted(candidates)
        if item.is_file() and "__pycache__" not in item.parts
    ]
    payload = {"exists": package_root.is_dir(), "files": files}
    return {**payload, "sha256": stable_json_sha256(payload)}


def runtime_versions() -> dict[str, Any]:
    """Read versions from the interpreter that will execute the actor call."""

    packages: dict[str, str | None] = {}
    for distribution in REQUIRED_RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
    }


def sanitize_endpoint(endpoint: str | None) -> str:
    """Record an endpoint while stripping credentials, query, and fragment."""

    value = str(endpoint or "https://api.openai.com/v1").strip()
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Actor endpoint must be an absolute URL: {value!r}")
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{hostname}{port}"
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def build_actor_prompt_payload(task: Any, framework: str) -> dict[str, Any]:
    """Build the canonical actor input and system-prompt payload."""

    from multi_agent.frameworks.core.task_input import build_task_input

    initial_input = build_task_input(task)
    payload: dict[str, Any] = {
        "initial_input": initial_input,
        "framework": framework,
    }
    if framework == "oai":
        from multi_agent.frameworks.core.task_input import extract_input_text
        from multi_agent.frameworks.oai.prompts import (
            build_oai_hub_prompt,
            build_oai_spoke_prompt,
        )

        payload["initial_input"] = extract_input_text(initial_input) or task.goal
        hub = task.agents[0]
        spokes = task.agents[1:]
        payload["hub_prompt"] = build_oai_hub_prompt(
            hub,
            spokes,
            task.goal,
        )
        payload["spoke_prompts"] = {
            agent.role: build_oai_spoke_prompt(agent, hub.role)
            for agent in spokes
        }
    return payload


def build_actor_provenance(
    *,
    task: Any,
    catalog: Any,
    task_path: Path,
    repo_root: Path,
    package_root: Path,
    framework: str,
    harness: str,
    model: str,
    max_turns: int,
    runtime: dict[str, Any],
    prompt_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable actor protocol recorded before the actor call."""

    task_path = Path(task_path).resolve()
    repo_root = Path(repo_root).resolve()
    package_root = Path(package_root).resolve()

    fixture_roots: list[dict[str, Any]] = []
    real_fixture = str(getattr(task, "fixture", "") or "").strip()
    if real_fixture:
        fixture_path = repo_root / "fixtures" / real_fixture
        fixture_roots.append(
            {
                "kind": "workspace_fixture",
                "name": real_fixture,
                **path_manifest(fixture_path),
            }
        )
    mock_fixture_names = sorted(
        {
            str(getattr(tool, "fixture", "") or "")
            for tool in getattr(catalog, "tools", [])
            if str(getattr(tool, "fixture", "") or "").strip()
        }
    )
    for fixture_name in mock_fixture_names:
        fixture_path = package_root / "fixtures" / str(task.domain) / fixture_name
        fixture_roots.append(
            {
                "kind": "tool_fixture",
                "name": fixture_name,
                **path_manifest(fixture_path),
            }
        )

    assets: list[dict[str, Any]] = []
    for asset in getattr(task, "input_assets", []) or []:
        resolved = str(getattr(asset, "resolved_path", "") or "").strip()
        if resolved:
            asset_path = Path(resolved)
            assets.append(
                {
                    "name": str(getattr(asset, "path", "") or asset_path.name),
                    "sha256": sha256_file(asset_path),
                }
            )

    implementation = implementation_manifest(package_root)
    effective_task = _model_dump(task)
    effective_catalog = _model_dump(catalog)
    actor_trace_seed = (
        int(runtime["actor_trace_seed"])
        if runtime.get("actor_trace_seed") is not None
        else None
    )
    actor_protocol_profile = str(
        runtime.get("actor_protocol_profile") or SEEDED_SAMPLING_PROFILE
    )
    actor_replicate_id = str(runtime.get("actor_replicate_id") or "").strip()
    if (
        not actor_replicate_id
        and actor_protocol_profile == SEEDED_SAMPLING_PROFILE
        and actor_trace_seed is not None
    ):
        actor_replicate_id = f"seed-{actor_trace_seed}"
    hashes = {
        "task_source_sha256": sha256_file(task_path),
        "effective_task_sha256": stable_json_sha256(effective_task),
        "tool_catalog_sha256": stable_json_sha256(effective_catalog),
        "fixture_manifest_sha256": stable_json_sha256(fixture_roots),
        "domain_fixture_manifest_sha256": str(
            path_manifest(
                package_root / "fixtures" / str(task.domain)
            )["sha256"]
        ),
        "input_asset_manifest_sha256": stable_json_sha256(assets),
        "prompt_protocol_sha256": stable_json_sha256(prompt_payload),
        "implementation_manifest_sha256": str(implementation["sha256"]),
    }
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "recorded_before_actor_call": True,
        "framework": framework,
        "harness": harness,
        "model": model,
        "model_revision": str(runtime.get("model_revision") or model),
        "model_revision_source": (
            "explicit" if runtime.get("model_revision_explicit") else "model_argument"
        ),
        "runtime_revision": str(
            runtime.get("runtime_revision") or "unversioned-runtime"
        ),
        "runtime_revision_source": (
            "explicit" if runtime.get("runtime_revision_explicit") else "default"
        ),
        "runtime_versions": runtime_versions(),
        "endpoint": sanitize_endpoint(runtime.get("endpoint")),
        "actor_protocol_profile": actor_protocol_profile,
        "actor_replicate_id": actor_replicate_id,
        "actor_trace_seed": actor_trace_seed,
        "actor_temperature": runtime.get("actor_temperature"),
        "actor_temperature_requested": runtime.get(
            "actor_temperature_requested", runtime.get("actor_temperature")
        ),
        "actor_parallel_tool_calls": bool(
            runtime.get("actor_parallel_tool_calls", False)
        ),
        "actor_seed_required": bool(runtime.get("actor_seed_required", False)),
        "seed_transport": str(runtime.get("seed_transport") or "none"),
        "max_turns": int(max_turns),
        "resource_semantics_mode": normalize_resource_semantics_mode(
            runtime.get("resource_semantics_mode")
        ),
        "hashes": hashes,
    }
    stable_payload = dict(payload)
    stable_payload["actor_trace_seed"] = None
    stable_payload["actor_replicate_id"] = None
    payload["actor_stable_protocol_sha256"] = stable_json_sha256(stable_payload)
    payload["actor_protocol_sha256"] = stable_json_sha256(payload)
    return payload


def validate_actor_provenance(value: Any, *, formal: bool = False) -> list[str]:
    """Validate integrity and the stronger formal trace-generation contract."""

    if not isinstance(value, dict):
        return ["missing native actor_provenance object"]
    errors: list[str] = []
    if value.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        errors.append("unsupported actor_provenance schema_version")
    if value.get("recorded_before_actor_call") is not True:
        errors.append("actor provenance was not recorded before the actor call")
    try:
        normalize_resource_semantics_mode(value.get("resource_semantics_mode"))
    except ValueError as exc:
        errors.append(str(exc))
    claimed_hash = str(value.get("actor_protocol_sha256") or "")
    payload = dict(value)
    payload.pop("actor_protocol_sha256", None)
    if claimed_hash != stable_json_sha256(payload):
        errors.append("actor_protocol_sha256 does not match the provenance payload")
    claimed_stable_hash = str(value.get("actor_stable_protocol_sha256") or "")
    stable_payload = dict(value)
    stable_payload.pop("actor_protocol_sha256", None)
    stable_payload.pop("actor_stable_protocol_sha256", None)
    stable_payload["actor_trace_seed"] = None
    stable_payload["actor_replicate_id"] = None
    if claimed_stable_hash != stable_json_sha256(stable_payload):
        errors.append(
            "actor_stable_protocol_sha256 does not match the "
            "seed-and-replicate-invariant payload"
        )
    hashes = value.get("hashes")
    required_hashes = {
        "task_source_sha256",
        "effective_task_sha256",
        "tool_catalog_sha256",
        "fixture_manifest_sha256",
        "domain_fixture_manifest_sha256",
        "input_asset_manifest_sha256",
        "prompt_protocol_sha256",
        "implementation_manifest_sha256",
    }
    if not isinstance(hashes, dict):
        errors.append("missing actor provenance hashes")
    else:
        for name in sorted(required_hashes):
            digest = str(hashes.get(name) or "")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                errors.append(f"invalid {name}")
    if formal:
        profile = value.get("actor_protocol_profile")
        if not str(value.get("actor_replicate_id") or "").strip():
            errors.append("formal trace requires actor_replicate_id")
        if value.get("actor_parallel_tool_calls") is not False:
            errors.append("formal trace requires serial actor tool calls")
        if profile == SEEDED_SAMPLING_PROFILE:
            actor_seed = value.get("actor_trace_seed")
            if not isinstance(actor_seed, int) or isinstance(actor_seed, bool):
                errors.append("formal seeded trace requires integer actor_trace_seed")
            temperature = value.get("actor_temperature")
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or float(temperature) <= 0
            ):
                errors.append("formal seeded trace requires actor_temperature > 0")
            if value.get("actor_seed_required") is not True:
                errors.append(
                    "formal seeded trace must fail closed when seed transport "
                    "is unsupported"
                )
            if value.get("seed_transport") != "extra_body.seed":
                errors.append(
                    "formal seeded trace requires extra_body.seed transport"
                )
        elif profile == DETERMINISTIC_CLOSE_SOURCE_PROFILE:
            if value.get("actor_trace_seed") is not None:
                errors.append(
                    "formal deterministic trace must not include actor_trace_seed"
                )
            requested_temperature = value.get("actor_temperature_requested")
            if (
                not isinstance(requested_temperature, (int, float))
                or isinstance(requested_temperature, bool)
                or float(requested_temperature) != 0.0
            ):
                errors.append(
                    "formal deterministic trace requires "
                    "actor_temperature_requested == 0.0"
                )
            if value.get("actor_seed_required") is not False:
                errors.append(
                    "formal deterministic trace must not require actor seed transport"
                )
            if value.get("seed_transport") != "none":
                errors.append(
                    "formal deterministic trace requires seed_transport none"
                )
        else:
            errors.append(f"unknown formal actor protocol profile: {profile!r}")
        if not str(value.get("model_revision") or "").strip():
            errors.append("formal trace requires an explicit actor model revision")
        if value.get("model_revision_source") != "explicit":
            errors.append("formal trace actor model revision was not explicit")
        if (
            not str(value.get("runtime_revision") or "").strip()
            or value.get("runtime_revision") == "unversioned-runtime"
        ):
            errors.append("formal trace requires an explicit actor runtime revision")
        if value.get("runtime_revision_source") != "explicit":
            errors.append("formal trace actor runtime revision was not explicit")
        versions = value.get("runtime_versions")
        python_version = (
            versions.get("python") if isinstance(versions, dict) else None
        )
        packages = (
            versions.get("packages") if isinstance(versions, dict) else None
        )
        if not isinstance(python_version, dict) or not all(
            str(python_version.get(name) or "").strip()
            for name in ("implementation", "version")
        ):
            errors.append("formal trace lacks the executing Python runtime version")
        if not isinstance(packages, dict):
            errors.append("formal trace lacks actor dependency versions")
        else:
            for distribution in REQUIRED_RUNTIME_DISTRIBUTIONS:
                if not str(packages.get(distribution) or "").strip():
                    errors.append(
                        f"formal trace lacks runtime version for {distribution}"
                    )
        endpoint = str(value.get("endpoint") or "").strip()
        if not endpoint:
            errors.append("formal trace requires a non-empty actor endpoint")
        else:
            try:
                sanitize_endpoint(endpoint)
            except ValueError as exc:
                errors.append(str(exc))
    return errors


__all__ = [
    "DETERMINISTIC_CLOSE_SOURCE_PROFILE",
    "PROVENANCE_SCHEMA_VERSION",
    "SEEDED_SAMPLING_PROFILE",
    "build_actor_prompt_payload",
    "build_actor_provenance",
    "implementation_manifest",
    "path_manifest",
    "sanitize_endpoint",
    "sha256_file",
    "stable_json_sha256",
    "runtime_versions",
    "validate_actor_provenance",
]
