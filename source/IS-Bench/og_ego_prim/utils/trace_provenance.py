"""Pure helpers for binding an ISBench trace to its generation protocol."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


TRACE_PROVENANCE_SCHEMA_VERSION = 2
TRACE_DEPENDENCY_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("omnigibson", "omnigibson"),
    ("isaacsim", "isaacsim"),
    ("torch", "torch"),
    ("openai", "openai"),
    ("google-genai", "google-genai"),
    ("numpy", "numpy"),
    ("Pillow", "Pillow"),
)

# Keep this list explicit: a resume is invalidated only by code that can change
# actor decisions, execution, observations, or safety labels.  Paths are
# relative to the ``og_ego_prim`` package root.
TRACE_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "benchmark/__init__.py",
    "benchmark/base_benchmark.py",
    "benchmark/custom_behavior_task.py",
    "benchmark/data_utils.py",
    "benchmark/online_benchmark.py",
    "benchmark/evaluator/__init__.py",
    "benchmark/evaluator/bddl_goal_condition.py",
    "benchmark/evaluator/evaluator.py",
    "benchmark/tracker/__init__.py",
    "benchmark/tracker/base_tracker.py",
    "benchmark/tracker/online_tracker.py",
    "cli/online_benchmark_once.py",
    "cli/online_benchmark_all.py",
    "models/plan_agent.py",
    "models/safepred_safeguard.py",
    "models/server_inference.py",
    "primitives/__init__.py",
    "primitives/attachment.py",
    "primitives/ego_primitives.py",
    "primitives/executor.py",
    "primitives/object_states_utils.py",
    "primitives/primitive_utils.py",
    "utils/action_validation.py",
    "utils/action_recovery.py",
    "utils/episode_checkpoint.py",
    "utils/simulation_replay.py",
    "utils/monkey_patch.py",
    "utils/prompts.py",
    "utils/trace_provenance.py",
)

GENERATION_PROTOCOL_FIELDS: tuple[str, ...] = (
    "benchmark",
    "model_requested",
    "model_resolved",
    "provider_mode",
    "model_revision",
    "api_base",
    "runtime_revision",
    "python_version",
    "dependency_versions",
    "prompt_setting",
    "actor_temperature",
    "actor_top_p",
    "actor_max_completion_tokens",
    "actor_max_images",
    "actor_retry_seed_policy",
    "actor_transport_retry_policy",
    "actor_transport_max_retries",
    "actor_transport_retry_base_delay_seconds",
    "actor_transport_compat",
    "actor_output_contract",
    "action_resume_contract",
    "disable_thinking",
    "use_initial_setup",
    "use_self_caption",
    "draw_bbox_2d",
    "skip_viewer_obs",
    "online_object_sampling",
    "eval_process_safety",
    "eval_termination_safety",
    "eval_awareness",
    "eval_execution",
    "task_config_sha256",
    "camera_config_sha256",
    "implementation_sha256",
)

AIFAST_TRANSPORT_COMPAT = "aifast_noncalling_dummy_tool_v1"
NO_TRANSPORT_COMPAT = "none"
TRANSIENT_TRANSPORT_RETRY_POLICY = "same_request_transient_http_v1"
NO_TRANSPORT_RETRY_POLICY = "none"
UPSTREAM_OUTPUT_CONTRACT = "upstream_lowercase_fenced_json_v1"
AIFAST_OUTPUT_CONTRACT = "upstream_fenced_json_or_strict_whole_json_v1"
PROVIDER_REASONING_CAPTURE = "openai_message_reasoning_content_v1"


def is_aifast_endpoint(value: str | None) -> bool:
    """Return whether an endpoint is the explicitly supported AIFast host."""

    try:
        parsed = urlsplit(str(value or "").strip())
        return (
            parsed.scheme.lower() in {"http", "https"}
            and str(parsed.hostname or "").lower() == "aifast.site"
        )
    except ValueError:
        return False


def actor_transport_compat(value: str | None) -> str:
    return (
        AIFAST_TRANSPORT_COMPAT
        if is_aifast_endpoint(value)
        else NO_TRANSPORT_COMPAT
    )


def actor_output_contract(value: str | None) -> str:
    explicit = os.environ.get("ISBENCH_ACTOR_OUTPUT_CONTRACT")
    if explicit is not None:
        if explicit not in {UPSTREAM_OUTPUT_CONTRACT, AIFAST_OUTPUT_CONTRACT}:
            raise ValueError("unsupported ISBENCH_ACTOR_OUTPUT_CONTRACT")
        return explicit
    return (
        AIFAST_OUTPUT_CONTRACT
        if is_aifast_endpoint(value)
        else UPSTREAM_OUTPUT_CONTRACT
    )


def actor_transport_max_retries() -> int:
    value = int(os.environ.get("ISBENCH_ACTOR_TRANSPORT_MAX_RETRIES", "0"))
    if value < 0:
        raise ValueError("ISBENCH_ACTOR_TRANSPORT_MAX_RETRIES must be nonnegative")
    return value


def actor_transport_retry_base_delay_seconds() -> float:
    value = float(
        os.environ.get("ISBENCH_ACTOR_TRANSPORT_RETRY_BASE_DELAY", "1.0")
    )
    if value < 0:
        raise ValueError(
            "ISBENCH_ACTOR_TRANSPORT_RETRY_BASE_DELAY must be nonnegative"
        )
    return value


def actor_transport_retry_policy(max_retries: int | None = None) -> str:
    retries = (
        actor_transport_max_retries()
        if max_retries is None
        else int(max_retries)
    )
    return (
        TRANSIENT_TRANSPORT_RETRY_POLICY
        if retries > 0
        else NO_TRANSPORT_RETRY_POLICY
    )


def derive_task_seed(experiment_seed: int, task: str, scene: str) -> int:
    """Derive the deterministic per-task actor seed used by ISBench."""

    base_seed = int(experiment_seed)
    digest = hashlib.sha256(
        f"{base_seed}:{task}:{scene}".encode("utf-8")
    ).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256(
    package_root: str | Path,
    *,
    relative_paths: Sequence[str] = TRACE_IMPLEMENTATION_FILES,
) -> str:
    """Hash the named trace-producing sources, including their relative paths."""

    root = Path(package_root).resolve()
    relatives = tuple(sorted(str(item) for item in relative_paths))
    source_paths: list[tuple[str, Path]] = []
    for relative in relatives:
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"implementation source escapes package root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"missing trace implementation source: {path}")
        source_paths.append((relative, path))
    digest = hashlib.sha256()
    for relative, path in source_paths:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def normalize_api_base(value: str | None) -> str:
    """Normalize an endpoint while dropping credentials, query, and fragment."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname.lower()
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", "")
    )


def resolve_openai_api_base(explicit: str | None = None) -> str:
    """Resolve the OpenAI-compatible endpoint without import-time env binding."""

    for value in (
        explicit,
        os.environ.get("OPENAI_API_BASE"),
        os.environ.get("OPENAI_BASE_URL"),
    ):
        resolved = str(value or "").strip()
        if resolved:
            return resolved
    return ""


def strict_http_endpoint_error(value: str | None, *, field_name: str) -> str:
    """Return a strict-trace validation error for an ambiguous endpoint."""

    raw = str(value or "").strip()
    try:
        endpoint = urlsplit(raw)
        endpoint.port
        valid = (
            not any(character.isspace() for character in raw)
            and endpoint.scheme in {"http", "https"}
            and bool(endpoint.hostname)
            and endpoint.username is None
            and endpoint.password is None
            and "?" not in raw
            and "#" not in raw
            and not endpoint.query
            and not endpoint.fragment
        )
    except ValueError:
        valid = False
    if valid:
        return ""
    return (
        f"{field_name} must be an absolute HTTP(S) URL without credentials, "
        "query, or fragment"
    )


def build_generation_protocol(
    *,
    task_config_path: str | Path,
    camera_config_path: str | Path,
    package_root: str | Path,
    model_requested: str,
    model_resolved: str,
    provider_mode: str,
    model_revision: str,
    api_base: str | None,
    runtime_revision: str,
    prompt_setting: str,
    actor_temperature: float,
    actor_top_p: float,
    actor_max_completion_tokens: int,
    actor_max_images: int,
    actor_retry_seed_policy: str,
    disable_thinking: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    draw_bbox_2d: bool,
    skip_viewer_obs: bool,
    online_object_sampling: bool,
    eval_process_safety: bool,
    eval_termination_safety: bool,
    eval_awareness: bool,
    eval_execution: bool,
    actor_transport_retry_policy: str = "none",
    actor_transport_max_retries: int = 0,
    actor_transport_retry_base_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build the canonical protocol object embedded in every report."""

    normalized_api_base = normalize_api_base(
        api_base or "https://api.openai.com/v1"
    )
    action_contract = 'none'
    if os.environ.get('ISBENCH_ACTION_RESUME', '0') == '1':
        from og_ego_prim.utils.action_recovery import active_contract
        action_contract = active_contract()
    return {
        "benchmark": "isbench",
        "model_requested": str(model_requested),
        "model_resolved": str(model_resolved),
        "provider_mode": str(provider_mode),
        "model_revision": str(model_revision),
        "api_base": normalized_api_base,
        "runtime_revision": str(runtime_revision),
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions(),
        "prompt_setting": str(prompt_setting),
        "actor_temperature": float(actor_temperature),
        "actor_top_p": float(actor_top_p),
        "actor_max_completion_tokens": int(actor_max_completion_tokens),
        "actor_max_images": int(actor_max_images),
        "actor_retry_seed_policy": str(actor_retry_seed_policy),
        "actor_transport_retry_policy": str(actor_transport_retry_policy),
        "actor_transport_max_retries": int(actor_transport_max_retries),
        "actor_transport_retry_base_delay_seconds": float(
            actor_transport_retry_base_delay_seconds
        ),
        "actor_transport_compat": actor_transport_compat(normalized_api_base),
        "actor_output_contract": actor_output_contract(normalized_api_base),
        "action_resume_contract": action_contract,
        "provider_reasoning_capture": PROVIDER_REASONING_CAPTURE,
        "disable_thinking": bool(disable_thinking),
        "use_initial_setup": bool(use_initial_setup),
        "use_self_caption": bool(use_self_caption),
        "draw_bbox_2d": bool(draw_bbox_2d),
        "skip_viewer_obs": bool(skip_viewer_obs),
        "online_object_sampling": bool(online_object_sampling),
        "eval_process_safety": bool(eval_process_safety),
        "eval_termination_safety": bool(eval_termination_safety),
        "eval_awareness": bool(eval_awareness),
        "eval_execution": bool(eval_execution),
        "task_config_sha256": sha256_file(task_config_path),
        "camera_config_sha256": sha256_file(camera_config_path),
        "implementation_sha256": implementation_sha256(package_root),
    }


def dependency_versions() -> dict[str, str]:
    # OmniGibson can prepend simulator/vendor site-packages while importing
    # native plugins.  That can make ``importlib.metadata`` select a different
    # duplicate distribution record in the actor child than in the parent
    # validator.  A launch-bound JSON snapshot, captured before simulator
    # startup, gives both processes one immutable dependency identity.
    frozen = os.environ.get("ISBENCH_DEPENDENCY_VERSIONS_JSON", "").strip()
    if frozen:
        try:
            payload = json.loads(frozen)
        except json.JSONDecodeError:
            payload = None
        expected_keys = [key for key, _ in TRACE_DEPENDENCY_DISTRIBUTIONS]
        if (
            isinstance(payload, dict)
            and all(str(payload.get(key) or "").strip() for key in expected_keys)
        ):
            return {key: str(payload[key]) for key in expected_keys}

    versions: dict[str, str] = {}
    for key, distribution in TRACE_DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def build_experiment_metadata(
    *,
    experiment_seed: int,
    task: str,
    scene: str,
    generation_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    task_seed = derive_task_seed(experiment_seed, task, scene)
    protocol = dict(generation_protocol)
    return {
        "schema_version": TRACE_PROVENANCE_SCHEMA_VERSION,
        "recorded_before_actor_call": True,
        "experiment_seed": int(experiment_seed),
        "actor_trace_seed": int(experiment_seed),
        "task_seed": task_seed,
        "actor_seed": task_seed,
        "safepred_seed": task_seed,
        "generation_protocol": protocol,
        "generation_protocol_sha256": protocol_sha256(protocol),
    }


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(protocol), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
