"""HarnessAudit CLI — run tasks and evaluate access control violations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.model_stack import load_model_stack_from_environment

from multi_agent.checker import (
    EXPLICIT_RESOURCE_SEMANTICS,
    RESOURCE_SEMANTICS_MODES,
    check_trace,
    compute_metrics,
    normalize_resource_semantics_mode,
    validate_resource_rule_set,
)
from multi_agent.credentials import load_workspace_credentials
from multi_agent.completion_judge import COMPLETION_SCORE_MODE
from multi_agent.frameworks.clawteam.backbone import canonicalize_clawteam_artifacts
from multi_agent.safepred_guard.terminal import (
    SAFETY_ABSTENTION_DISPOSITION,
    SAFETY_ABSTENTION_TASK_STATUS,
    SAFETY_ABSTENTION_UNSCORED_SCORE_FIELDS,
    validate_artifact_binding,
    validate_safety_abstention_terminal,
)
from multi_agent.loader import load_task_with_tools, resolve_task_path
from multi_agent.operational_judge import evaluate_operational_governance
from multi_agent.perturbation import (
    PerturbationConfig,
    default_index_path,
    load_variant_by_id,
    source_task_sha256,
)
from multi_agent.perturbation_eval import PerturbationEvaluation, evaluate_perturbation
from multi_agent.runner import (
    BenchmarkRunner,
    framework_run_id_slug,
    next_run_id,
)
from multi_agent.trace import AccessDecisionEvent, TraceEnd, TraceWriter
from multi_agent.trace.provenance import (
    DETERMINISTIC_CLOSE_SOURCE_PROFILE,
    SEEDED_SAMPLING_PROFILE,
    sanitize_endpoint,
    validate_actor_provenance,
)
from multi_agent.workspace import WorkspaceManager


# A safety-abstained task is terminal and reusable, but it is never an ordinary
# completed-task pass.  Keep the checkpoint status distinct from the runner's
# task-success status so resume cannot silently promote it.
SAFETY_ABSTENTION_CHECKPOINT_STATUS = "safety_abstained"


def _terminal_checkpoint_status(
    *,
    terminal_disposition: Any,
    task_success_status: Any,
) -> str:
    """Return the only valid checkpoint terminal status for one task result."""

    if terminal_disposition == SAFETY_ABSTENTION_DISPOSITION:
        if task_success_status != SAFETY_ABSTENTION_TASK_STATUS:
            raise RuntimeError("Safety-abstention terminal has an invalid task status")
        return SAFETY_ABSTENTION_CHECKPOINT_STATUS
    if terminal_disposition is not None:
        raise RuntimeError("Unsupported task terminal disposition")
    if task_success_status not in (None, "scored"):
        raise RuntimeError("Ordinary task terminal has an invalid task status")
    return "passed"


def _model_dir_name(model: str) -> str:
    """Return a filesystem-safe model directory name.

    Keep the model argument readable, but prevent provider-prefixed names such
    as ``anthropic/claude-sonnet-4.6`` from becoming unintended nested paths.
    """
    model_name = (model or "unknown-model").strip() or "unknown-model"
    return model_name.replace("/", "__").replace(os.sep, "__")


def _artifact_scoped_dir(base_dir: Path, harness: str, model: str) -> Path:
    """Store run artifacts under harness/model child directories.

    Callers may pass the artifact root (``results``), a harness-scoped
    directory (``results/openclaw``), or a fully scoped directory
    (``results/oai/deepseek-v4-flash``); avoid appending path components twice.
    """
    model_dir = _model_dir_name(model)
    if base_dir.name == model_dir and base_dir.parent.name == harness:
        return base_dir
    if base_dir.name == model_dir:
        return base_dir
    if base_dir.name == harness:
        return base_dir / model_dir
    return base_dir / harness / model_dir


def _effective_harness(args: argparse.Namespace) -> str:
    """Return the artifact/run harness label implied by the CLI args."""
    if args.harness:
        return args.harness
    if args.framework == "oai":
        return "oai"
    if args.framework == "adk":
        return "adk"
    return "claude"


def _artifact_prefix(task_id: str, perturbation: PerturbationConfig | None) -> str:
    if perturbation is None:
        return task_id
    return f"{task_id}_{perturbation.variant_id}"


TRACE_RESUME_IDENTITY_SCHEMA = "harnessaudit-trace-resume-identity-v2"
TRACE_RESUME_IMPLEMENTATION_SHA256_ENV = (
    "HARNESS_TRACE_RESUME_IMPLEMENTATION_SHA256"
)


def _trace_resume_implementation_sha256() -> str:
    """Return the controller-verified implementation digest fail-closed.

    The matrix controller computes the full online implementation manifest once
    and passes its frozen public digest to each condition. Re-hashing the full
    source and registered-input trees in every CLI child is both redundant and
    a material harness throughput bottleneck.
    """

    digest = os.environ.get(TRACE_RESUME_IMPLEMENTATION_SHA256_ENV)
    if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "Trace resume requires a canonical controller-verified "
            f"{TRACE_RESUME_IMPLEMENTATION_SHA256_ENV} digest"
        )
    return digest


_TRACE_RESUME_RUNTIME_ENV_KEYS = (
    "HARNESS_ACTOR_REASONING_LEDGER_REQUIRED",
    "HARNESS_DISABLE_PARALLEL_TOOL_CALLS",
    "HARNESS_SAFEPRED_MAX_CONSULTATIONS_PER_CALLER",
    "HARNESS_SAFEPRED_MAX_CONSULTATIONS_PER_EPISODE",
    "HARNESS_WM_RETURN_SIGNAL_SOURCE_POLICY",
    "SAFEPRED_ALLOW_TRAIN_POLICY_FALLBACK",
    "SAFEPRED_LLM_MAX_ATTEMPTS",
    "SAFEPRED_LLM_RETRY_BACKOFF",
    "SAFEPRED_LLM_RETRY_DELAY_SECONDS",
    "SAFEPRED_POLICY_EXTRACTION_RETRIES",
    "SAFEPRED_POLICY_EXTRACTION_SEED",
    "SAFEPRED_POLICY_THINKING_MAX_TOKENS",
    "SAFEPRED_TRAIN_POLICY_REQUIRE_CACHE",
    "SAFEPRED_TRAIN_POLICY_WRITE_CACHE",
)


def _trace_resume_runtime_environment_identity() -> dict[str, str | None]:
    """Return only public, trajectory-affecting runtime controls.

    Credential variables are deliberately absent. Missing values remain explicit
    so an explicit override cannot pass as the same resumable trajectory.
    """

    return {key: os.environ.get(key) for key in _TRACE_RESUME_RUNTIME_ENV_KEYS}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_descendant(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    bounded_root = root.expanduser().resolve()
    if bounded_root == Path(bounded_root.anchor):
        raise ValueError("Trace runtime root must be a bounded directory")
    if resolved != bounded_root and bounded_root not in resolved.parents:
        raise ValueError(f"{label} must be inside --trace-runtime-root")
    return resolved


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish one private JSON artifact without exposing partial bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _trace_resume_identity(
    *,
    args: argparse.Namespace,
    task_path: Path,
    task: Any,
    harness: str,
    perturbation: PerturbationConfig | None,
    actor_runtime: dict[str, Any],
    guard_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Freeze every experiment control that may change a resumed trajectory."""

    resolved_model_stack = load_model_stack_from_environment(
        os.environ,
        require_credentials=False,
    )
    model_stack_identity = (
        resolved_model_stack.public_identity
        if resolved_model_stack is not None
        else None
    )

    config_path_value = str((guard_config or {}).get("config_path") or "").strip()
    config_path = (
        Path(config_path_value).expanduser().resolve()
        if config_path_value
        else None
    )
    config_binding = None
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"SafePred config is missing: {config_path}")
        config_binding = {
            "path": str(config_path),
            "sha256": _sha256_path(config_path),
        }
    role_endpoints: dict[str, dict[str, Any]] = {}
    fallback_endpoint = actor_runtime.get("endpoint")
    for role in (
        "ACTOR",
        "WORLD_MODEL",
        "POLICY_EXTRACTOR",
        "ACTION_AGENT",
        "JUDGE",
    ):
        endpoint = (
            os.environ.get(f"WM_SAFETY_{role}_BASE_URL")
            or (os.environ.get("CUSTOM_API_URL") if role != "ACTOR" else None)
            or fallback_endpoint
        )
        default_model = (
            args.model
            if role == "ACTOR"
            else args.judge_model if role == "JUDGE" else ""
        )
        role_endpoints[role.lower()] = {
            "endpoint": sanitize_endpoint(endpoint),
            "model": str(
                os.environ.get(f"WM_SAFETY_{role}_MODEL") or default_model
            ),
            "api_mode": str(
                os.environ.get(f"WM_SAFETY_{role}_API_MODE")
                or os.environ.get("WM_SAFETY_MODEL_API_MODE")
                or "chat_completions"
            ),
        }
    guard_identity = dict(guard_config or {})
    guard_identity.pop("artifact_dir", None)
    guard_identity["config_binding"] = config_binding
    return {
        "schema_version": TRACE_RESUME_IDENTITY_SCHEMA,
        "task_id": str(task.task_id),
        "task_sha256": _sha256_path(task_path),
        "domain": str(task.domain),
        "fixture": str(getattr(task, "fixture", "") or ""),
        "framework": str(args.framework),
        "harness": str(harness),
        "model": str(args.model),
        "judge_model": str(args.judge_model),
        "max_turns": int(args.max_turns),
        "agent_timeout": int(getattr(args, "agent_timeout", 300)),
        "judge_workers": int(args.judge_workers),
        "skip_judge": bool(args.skip_judge),
        "formal_trace": bool(getattr(args, "formal_trace", False)),
        "actor_runtime": dict(actor_runtime),
        "active_online_implementation_sha256": (
            _trace_resume_implementation_sha256()
        ),
        "guard": guard_identity,
        "model_stack": model_stack_identity,
        "perturbation": perturbation.trace_metadata() if perturbation else None,
        "role_endpoints": role_endpoints,
        "runtime_environment": _trace_resume_runtime_environment_identity(),
    }


def _formal_actor_scope_parts(runtime: dict[str, Any]) -> tuple[str, str]:
    """Return collision-resistant profile/replicate formal artifact directories."""

    profile = runtime.get("actor_protocol_profile")
    if profile not in {
        SEEDED_SAMPLING_PROFILE,
        DETERMINISTIC_CLOSE_SOURCE_PROFILE,
    }:
        raise ValueError(f"Unknown formal actor protocol profile: {profile!r}")
    replicate_id = str(runtime.get("actor_replicate_id") or "").strip()
    if not replicate_id:
        raise ValueError("Formal artifact scope requires actor_replicate_id")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", replicate_id).strip("._")
    sanitized = sanitized or "replicate"
    digest = hashlib.sha256(replicate_id.encode("utf-8")).hexdigest()[:8]
    return str(profile), f"{sanitized}-{digest}"


def _uses_deterministic_close_source_profile(model: str) -> bool:
    """Match supported model families without accepting arbitrary substrings."""

    model_name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return any(
        model_name == family
        or model_name.startswith(f"{family}-")
        or model_name.startswith(f"{family}.")
        for family in ("gpt-5", "o3", "o4")
    )


def _resolve_actor_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve one actor protocol used by both provenance and model calls."""

    formal = bool(getattr(args, "formal_trace", False))
    model_lower = str(args.model).lower()
    deterministic_model = _uses_deterministic_close_source_profile(args.model)
    requested_profile = str(
        getattr(args, "actor_protocol_profile", None) or ""
    ).strip()
    actor_protocol_profile = requested_profile or (
        DETERMINISTIC_CLOSE_SOURCE_PROFILE
        if formal and deterministic_model
        else SEEDED_SAMPLING_PROFILE
    )

    raw_seed = getattr(args, "actor_seed", None)
    if raw_seed is None:
        raw_seed = os.environ.get("HARNESS_ACTOR_SEED")
    actor_seed = (
        int(raw_seed) if raw_seed is not None and str(raw_seed).strip() else None
    )

    raw_temperature = getattr(args, "actor_temperature", None)
    if raw_temperature is None:
        raw_temperature = os.environ.get("HARNESS_ACTOR_TEMPERATURE", "0")
    requested_temperature = float(raw_temperature)
    temperature = (
        None
        if deterministic_model
        else requested_temperature
    )

    explicit_parallel = getattr(args, "actor_parallel_tool_calls", None)
    guard_config = _safepred_guard_config_from_args(args)
    guard_mode = str((guard_config or {}).get("intervention_mode") or "")
    guard_allows_parallel = (
        not guard_config or guard_mode.replace("-", "_") == "prediction_only"
    )
    if explicit_parallel is None:
        parallel_tool_calls = (
            guard_allows_parallel
            and os.environ.get("HARNESS_DISABLE_PARALLEL_TOOL_CALLS", "0") != "1"
        )
    else:
        parallel_tool_calls = bool(explicit_parallel) and guard_allows_parallel

    requested_revision = getattr(args, "actor_model_revision", None)
    env_revision = os.environ.get("HARNESS_ACTOR_MODEL_REVISION")
    revision_is_explicit = bool(
        str(requested_revision or env_revision or "").strip()
    )
    model_revision = str(requested_revision or env_revision or args.model).strip()
    requested_runtime_revision = getattr(args, "actor_runtime_revision", None)
    env_runtime_revision = os.environ.get("HARNESS_ACTOR_RUNTIME_REVISION")
    runtime_revision_is_explicit = bool(
        str(requested_runtime_revision or env_runtime_revision or "").strip()
    )
    runtime_revision = str(
        requested_runtime_revision or env_runtime_revision or "unversioned-runtime"
    ).strip()
    requested_endpoint = getattr(args, "actor_endpoint", None)
    endpoint = sanitize_endpoint(
        requested_endpoint or os.environ.get("OPENAI_BASE_URL")
    )
    require_seed = actor_protocol_profile == SEEDED_SAMPLING_PROFILE and (
        formal or os.environ.get("HARNESS_ACTOR_REQUIRE_SEED") == "1"
    )
    actor_replicate_id = str(
        getattr(args, "actor_replicate_id", None) or ""
    ).strip()
    resource_semantics_mode = normalize_resource_semantics_mode(
        getattr(args, "resource_semantics_mode", None)
    )
    if (
        not actor_replicate_id
        and actor_protocol_profile == SEEDED_SAMPLING_PROFILE
        and actor_seed is not None
    ):
        actor_replicate_id = f"seed-{actor_seed}"
    return {
        "actor_protocol_profile": actor_protocol_profile,
        "actor_replicate_id": actor_replicate_id,
        "actor_trace_seed": actor_seed,
        "actor_temperature": temperature,
        "actor_temperature_requested": requested_temperature,
        "actor_parallel_tool_calls": parallel_tool_calls,
        "actor_seed_required": require_seed,
        "seed_transport": "extra_body.seed" if actor_seed is not None else "none",
        "model_revision": model_revision,
        "model_revision_explicit": revision_is_explicit,
        "runtime_revision": runtime_revision,
        "runtime_revision_explicit": runtime_revision_is_explicit,
        "endpoint": endpoint,
        "resource_semantics_mode": resource_semantics_mode,
    }


def _validate_formal_trace_args(
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> None:
    if not getattr(args, "formal_trace", False):
        return
    errors: list[str] = []
    if args.framework != "oai" or _effective_harness(args) != "oai":
        errors.append("--formal-trace currently requires --framework oai --harness oai")
    if getattr(args, "safepred_guard", False):
        errors.append("formal source traces must be generated without a SafePred guard")
    profile = runtime.get("actor_protocol_profile")
    replicate_id = str(runtime.get("actor_replicate_id") or "").strip()
    if profile == SEEDED_SAMPLING_PROFILE:
        if runtime.get("actor_trace_seed") is None:
            errors.append("--formal-trace requires --actor-seed")
        if float(runtime.get("actor_temperature") or 0.0) <= 0:
            errors.append("--formal-trace requires --actor-temperature > 0")
        if runtime.get("actor_seed_required") is not True:
            errors.append("--formal-trace seeded profile requires enforced seed transport")
        if runtime.get("seed_transport") != "extra_body.seed":
            errors.append("--formal-trace seeded profile requires extra_body.seed transport")
    elif profile == DETERMINISTIC_CLOSE_SOURCE_PROFILE:
        if runtime.get("actor_trace_seed") is not None:
            errors.append(
                "--formal-trace deterministic profile must not set --actor-seed"
            )
        if float(runtime.get("actor_temperature_requested") or 0.0) != 0.0:
            errors.append(
                "--formal-trace deterministic profile requires --actor-temperature 0"
            )
        if runtime.get("actor_seed_required") is not False:
            errors.append(
                "--formal-trace deterministic profile must not require seed transport"
            )
        if runtime.get("seed_transport") != "none":
            errors.append(
                "--formal-trace deterministic profile requires seed transport none"
            )
    else:
        errors.append(f"--formal-trace has unknown actor protocol profile: {profile!r}")
    if not replicate_id:
        errors.append("--formal-trace requires --actor-replicate-id")
    model_lower = str(args.model).lower()
    if model_lower.startswith("litellm/") or any(
        name in model_lower for name in ("claude", "gemini")
    ):
        errors.append(
            "--formal-trace requires the OpenAI-compatible provider path; "
            "LiteLLM-routed actor models do not bind the recorded endpoint/seed transport"
        )
    if (
        profile == SEEDED_SAMPLING_PROFILE
        and _uses_deterministic_close_source_profile(args.model)
    ):
        errors.append(
            "the selected model path does not accept actor temperature; "
            "use a sampling-capable actor model for independent seeded traces"
        )
    if runtime.get("actor_parallel_tool_calls") is not False:
        errors.append("--formal-trace requires --no-actor-parallel-tool-calls")
    if not runtime.get("model_revision_explicit"):
        errors.append(
            "--formal-trace requires --actor-model-revision (or "
            "HARNESS_ACTOR_MODEL_REVISION)"
        )
    if not runtime.get("runtime_revision_explicit"):
        errors.append(
            "--formal-trace requires --actor-runtime-revision (or "
            "HARNESS_ACTOR_RUNTIME_REVISION)"
        )
    if int(getattr(args, "repeat", 1)) != 1:
        errors.append(
            "--formal-trace requires --repeat 1; use one fresh "
            "profile/replicate root per run"
        )
    if not getattr(args, "output", None):
        errors.append("--formal-trace requires --output so a source report is audited")
    if not getattr(args, "skip_judge", False):
        errors.append(
            "--formal-trace requires --skip-judge; source GT is deterministic and "
            "must not depend on an unfrozen post-actor LLM judge"
        )
    if errors:
        raise ValueError("Invalid formal trace protocol: " + "; ".join(errors))


def _perturbation_trace_payload(
    perturbation: PerturbationConfig | None,
    perturbation_result: PerturbationEvaluation | None,
) -> dict[str, Any] | None:
    if perturbation is None:
        return None
    metadata = perturbation.trace_metadata()
    if perturbation_result is None:
        return {
            **metadata,
            "delivered": None,
            "stable": None,
            "stability_score": None,
        }
    return {**metadata, **perturbation_result.as_dict()}


def _load_perturbation_config(
    args: argparse.Namespace,
    *,
    task_id: str,
    domain: str,
    task_path: Path,
) -> PerturbationConfig | None:
    variant_id = getattr(args, "perturbation_id", None)
    if not variant_id:
        return None
    if args.framework != "clawteam":
        raise ValueError("Perturbations are only supported for --framework clawteam")

    index_path = Path(args.perturbation_index) if args.perturbation_index else default_index_path()
    variant, row = load_variant_by_id(variant_id, index_path=index_path)
    if variant.task_id != task_id or variant.domain != domain:
        raise ValueError(
            f"Perturbation {variant_id!r} is for {variant.domain}/{variant.task_id}, "
            f"but loaded task is {domain}/{task_id}"
        )
    current_hash = source_task_sha256(task_path)
    if (
        variant.source_task_sha256
        and variant.source_task_sha256 != current_hash
        and not args.allow_stale_perturbation
    ):
        raise ValueError(
            f"Perturbation {variant_id!r} was generated for a different task file hash. "
            "Pass --allow-stale-perturbation to override."
        )
    return PerturbationConfig(
        variant=variant,
        spec_path=str(row.get("variant_path") or row.get("spec_path") or ""),
        index_path=str(index_path),
    )


def _final_output_from_trace_actions(actions: list) -> str:
    for action in reversed(actions):
        if (
            str(getattr(action, "surface", "")) == "communication"
            and getattr(action, "target_role", "") == "user"
        ):
            return str(getattr(action, "content", "") or "")
    return ""


def _is_degenerate_empty_trace(trace: object) -> bool:
    """Detect harness runs that produced no usable native agent transcript."""

    if (
        getattr(trace, "terminal_disposition", None)
        == SAFETY_ABSTENTION_DISPOSITION
        and getattr(trace, "task_success_status", None)
        == SAFETY_ABSTENTION_TASK_STATUS
        and getattr(trace, "safety_abstention", None) is not None
    ):
        return False
    if getattr(trace, "error", None):
        return False
    if int(getattr(trace, "n_actions", 0) or 0) > 1:
        return False
    if int(getattr(trace, "n_tool_calls", 0) or 0) != 0:
        return False
    final_output = _final_output_from_trace_actions(getattr(trace, "actions", []) or [])
    return not final_output.strip()


@dataclass
class RunContextBundle:
    args: argparse.Namespace
    task_path: Path
    task: Any
    catalog: Any
    perturbation: PerturbationConfig | None
    harness: str
    trace_dir: Path
    output_dir: Path | None
    run_id: str
    artifact_prefix: str
    trace_path: Path
    writer: TraceWriter
    runner: BenchmarkRunner
    trace_runtime_root: Path | None = None
    trace_checkpoint_path: Path | None = None
    trace_checkpoint_identity: dict[str, Any] | None = None
    trace_checkpoint_attempt_id: str | None = None


@dataclass
class RunResultBundle:
    trace: Any | None = None
    metrics: Any | None = None
    violations: list[Any] = field(default_factory=list)
    perturbation_result: PerturbationEvaluation | None = None
    trace_path: Path | None = None


def build_run_context(
    args: argparse.Namespace,
    *,
    task_path: Path,
    pkg_root: Path,
    repo_root: Path,
    harness: str,
) -> RunContextBundle:
    """Load task state and construct the runner for one concrete run."""

    task, catalog = load_task_with_tools(task_path)
    perturbation = _load_perturbation_config(
        args,
        task_id=task.task_id,
        domain=task.domain,
        task_path=task_path,
    )
    actor_runtime = getattr(args, "_resolved_actor_runtime", None)
    if not isinstance(actor_runtime, dict):
        actor_runtime = _resolve_actor_runtime(args)
    resource_semantics_mode = normalize_resource_semantics_mode(
        actor_runtime.get("resource_semantics_mode")
    )
    resource_rule_errors = validate_resource_rule_set(
        task.access_rules,
        resource_semantics_mode,
    )
    if resource_rule_errors:
        raise ValueError(
            "Task access rules are invalid for "
            f"resource_semantics_mode={resource_semantics_mode!r}: "
            + "; ".join(resource_rule_errors)
        )
    trace_dir = _artifact_scoped_dir(Path(args.trace_dir), harness, args.model)
    output_dir = (
        _artifact_scoped_dir(Path(args.output), harness, args.model)
        if args.output else None
    )
    if getattr(args, "formal_trace", False):
        profile_scope, replicate_scope = _formal_actor_scope_parts(actor_runtime)
        trace_dir = trace_dir / profile_scope / replicate_scope
        output_dir = (
            output_dir / profile_scope / replicate_scope
            if output_dir is not None
            else None
        )
    trace_resume = bool(getattr(args, "trace_resume", False))
    requested_run_id = str(getattr(args, "run_id", None) or "").strip()
    if requested_run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", requested_run_id):
            raise ValueError(
                "--run-id must be 1-128 characters from [A-Za-z0-9._-] "
                "and start with an alphanumeric character"
            )
        run_id = requested_run_id
    elif trace_resume:
        raise ValueError("--trace-resume requires an explicit stable --run-id")
    else:
        run_id = next_run_id(
            model=args.model,
            framework_slug=framework_run_id_slug(args.framework),
            harness_slug=harness,
        )
    artifact_prefix = _artifact_prefix(task.task_id, perturbation)
    trace_path = trace_dir / f"{artifact_prefix}_{run_id}.jsonl"
    if getattr(args, "formal_trace", False):
        existing_traces = list(trace_dir.glob(f"{artifact_prefix}_*.jsonl"))
        existing_reports = (
            list(output_dir.glob(f"{artifact_prefix}_*.json"))
            if output_dir is not None and output_dir.exists()
            else []
        )
        expected_report = (
            output_dir / f"{artifact_prefix}_{run_id}.json"
            if output_dir is not None
            else None
        )
        if trace_resume:
            unexpected = [path for path in existing_traces if path != trace_path]
            unexpected.extend(
                path
                for path in existing_reports
                if expected_report is None or path != expected_report
            )
            if unexpected:
                raise FileExistsError(
                    "Formal resume root contains an artifact for a different run id"
                )
        elif existing_traces or existing_reports:
            raise FileExistsError(
                "Formal trace target is not fresh for this task/profile/replicate. "
                f"Use a new --trace-dir/--output root: {trace_dir}"
            )
        trace_dir.mkdir(parents=True, exist_ok=True)
        claim_path = trace_dir / f".{artifact_prefix}.formal-generation-claimed"
        claim_payload = {
            "task_id": task.task_id,
            "actor_protocol_profile": actor_runtime["actor_protocol_profile"],
            "actor_replicate_id": actor_runtime["actor_replicate_id"],
            "run_id": run_id,
        }
        if claim_path.exists():
            if not trace_resume:
                raise FileExistsError(
                    "Formal trace generation is already claimed for this "
                    f"task/profile/replicate: {claim_path}"
                )
            try:
                observed_claim = json.loads(claim_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Formal resume claim is unreadable") from exc
            if observed_claim != claim_payload:
                raise ValueError("Formal resume claim identity drift")
        else:
            with claim_path.open("x", encoding="utf-8") as claim:
                os.fchmod(claim.fileno(), 0o600)
                json.dump(claim_payload, claim, sort_keys=True)
                claim.write("\n")
                claim.flush()
                os.fsync(claim.fileno())
    writer = TraceWriter(trace_path)

    fixtures_root = repo_root / "fixtures"
    workspaces_root = Path(os.environ.get(
        "MASP_WORKSPACES_ROOT", str(pkg_root / "workspaces"),
    ))
    workspace_manager = WorkspaceManager(
        fixtures_root=fixtures_root,
        workspaces_root=workspaces_root,
    )
    guard_config = _safepred_guard_config_from_args(args)
    trace_runtime_root: Path | None = None
    trace_checkpoint_path: Path | None = None
    trace_checkpoint_identity: dict[str, Any] | None = None
    trace_checkpoint_attempt_id: str | None = None
    if trace_resume:
        if args.framework != "oai":
            raise ValueError("Internal trace resume is currently supported only for oai")
        if output_dir is None:
            raise ValueError("--trace-resume requires --output for terminal report binding")
        runtime_root_value = str(
            getattr(args, "trace_runtime_root", None) or ""
        ).strip()
        if not runtime_root_value:
            raise ValueError("--trace-resume requires --trace-runtime-root")
        trace_runtime_root = Path(runtime_root_value).expanduser().resolve()
        if trace_runtime_root == Path(trace_runtime_root.anchor):
            raise ValueError("--trace-runtime-root must not be a filesystem root")
        _assert_descendant(trace_path, trace_runtime_root, label="trace path")
        _assert_descendant(output_dir, trace_runtime_root, label="output directory")
        if guard_config and guard_config.get("artifact_dir"):
            _assert_descendant(
                Path(str(guard_config["artifact_dir"])),
                trace_runtime_root,
                label="SafePred artifact directory",
            )
        checkpoint_value = str(
            getattr(args, "trace_checkpoint_path", None) or ""
        ).strip()
        if checkpoint_value:
            trace_checkpoint_path = _assert_descendant(
                Path(checkpoint_value),
                trace_runtime_root,
                label="trace checkpoint path",
            )
        else:
            checkpoint_scope = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
            trace_checkpoint_path = (
                trace_runtime_root
                / "_trace_checkpoints"
                / checkpoint_scope
                / "checkpoint.sqlite"
            )
        trace_checkpoint_identity = _trace_resume_identity(
            args=args,
            task_path=task_path,
            task=task,
            harness=harness,
            perturbation=perturbation,
            actor_runtime=actor_runtime,
            guard_config=guard_config,
        )
        attempt_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        trace_checkpoint_attempt_id = f"trace-resume-v1-{attempt_digest}"
        workspace_manager = WorkspaceManager(
            fixtures_root=fixtures_root,
            workspaces_root=trace_runtime_root / "_trace_workspaces",
        )
    runner = BenchmarkRunner(
        task,
        catalog,
        model=args.model,
        framework=args.framework,
        harness=harness,
        trace_writer=writer,
        max_turns=args.max_turns,
        workspace_manager=workspace_manager,
        agent_timeout=getattr(args, "agent_timeout", 300),
        judge_model=args.judge_model,
        skip_judge=args.skip_judge,
        judge_workers=args.judge_workers,
        perturbation=perturbation,
        safepred_guard_config=guard_config,
        actor_runtime=actor_runtime,
        task_path=task_path,
        repo_root=repo_root,
        package_root=pkg_root,
        trace_resume=trace_resume,
        trace_runtime_root=trace_runtime_root,
        trace_checkpoint_path=trace_checkpoint_path,
        trace_checkpoint_identity=trace_checkpoint_identity,
        trace_checkpoint_attempt_id=trace_checkpoint_attempt_id,
    )
    return RunContextBundle(
        args=args,
        task_path=task_path,
        task=task,
        catalog=catalog,
        perturbation=perturbation,
        harness=harness,
        trace_dir=trace_dir,
        output_dir=output_dir,
        run_id=run_id,
        artifact_prefix=artifact_prefix,
        trace_path=trace_path,
        writer=writer,
        runner=runner,
        trace_runtime_root=trace_runtime_root,
        trace_checkpoint_path=trace_checkpoint_path,
        trace_checkpoint_identity=trace_checkpoint_identity,
        trace_checkpoint_attempt_id=trace_checkpoint_attempt_id,
    )


def _validate_reusable_report(ctx: RunContextBundle, report_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Checkpoint-bound report is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint-bound report is not a JSON object")
    if payload.get("run_id") != ctx.run_id or payload.get("task_id") != ctx.task.task_id:
        raise RuntimeError("Checkpoint-bound report identity mismatch")
    if payload.get("error") not in (None, ""):
        raise RuntimeError("A failed checkpoint report cannot be reused as passed")
    if payload.get("terminal_disposition") == SAFETY_ABSTENTION_DISPOSITION:
        if payload.get("task_success_status") != SAFETY_ABSTENTION_TASK_STATUS:
            raise RuntimeError("Checkpoint-bound safety abstention status mismatch")
        try:
            validate_safety_abstention_terminal(payload.get("safety_abstention"))
            validate_artifact_binding(payload.get("safety_abstention_artifact"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Checkpoint-bound safety abstention is malformed") from exc
        for field in (
            "tcr",
            "task_completion_score",
            "trajectory_tcr",
            "deterministic_completion_score",
            "deterministic_completion_weight",
            "deterministic_completion_contribution",
            "llm_completion_score",
            "llm_completion_weight",
            "llm_completion_contribution",
            "completion_weight_total",
        ):
            if payload.get(field) is not None:
                raise RuntimeError("Checkpoint-bound safety abstention has a task score")
    else:
        if payload.get("task_success_status") not in (None, "scored"):
            raise RuntimeError("Checkpoint-bound report has an invalid task status")
        for metric in ("sar_avg", "tcr"):
            value = payload.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeError("Checkpoint-bound report lacks required metrics")
    trace_value = str(payload.get("trace_file") or "").strip()
    if not trace_value:
        raise RuntimeError("Checkpoint-bound report lacks trace_file")
    trace_path = Path(trace_value)
    if not trace_path.is_absolute():
        trace_path = report_path.parent / trace_path
    if ctx.trace_runtime_root is None:
        raise RuntimeError("Trace runtime root is unavailable")
    _assert_descendant(trace_path, ctx.trace_runtime_root, label="report trace_file")
    if not trace_path.is_file():
        raise RuntimeError("Checkpoint-bound trace file is missing")
    return payload


def _terminal_report_for_resume(ctx: RunContextBundle) -> Path | None:
    """Validate and reuse a terminal report without any actor or judge call."""

    if not bool(getattr(ctx.args, "trace_resume", False)):
        return None
    checkpoint_path = ctx.trace_checkpoint_path
    runtime_root = ctx.trace_runtime_root
    identity = ctx.trace_checkpoint_identity
    attempt_id = ctx.trace_checkpoint_attempt_id
    if (
        checkpoint_path is None
        or runtime_root is None
        or identity is None
        or attempt_id is None
        or not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size == 0
    ):
        return None
    from multi_agent.trace.checkpoint import DurableTraceCheckpoint
    from multi_agent.trace.runtime_checkpoint import (
        ArtifactBinding,
        RuntimeCheckpointCoordinator,
    )

    checkpoint = DurableTraceCheckpoint(
        checkpoint_path,
        identity=identity,
        run_id=ctx.run_id,
        attempt_id=attempt_id,
    )
    try:
        coordinator = RuntimeCheckpointCoordinator(runtime_root, checkpoint)
        state = coordinator.inspect()
        if state.evaluation_complete and state.terminal is None:
            evaluation = state.evaluation_complete_state or {}
            binding = ArtifactBinding.from_json_value(
                evaluation.get("report"),
                label="evaluation report",
            )
            report_path = _assert_descendant(
                runtime_root / binding.relative_path,
                runtime_root,
                label="evaluation report",
            )
            if coordinator.bind_existing_artifact(
                report_path,
                label="evaluation report",
            ) != binding:
                raise RuntimeError("Evaluation report binding drift")
            report = _validate_reusable_report(ctx, report_path)
            status = _terminal_checkpoint_status(
                terminal_disposition=report.get("terminal_disposition"),
                task_success_status=report.get("task_success_status"),
            )
            coordinator.seal_terminal_report(report_path=report_path, status=status)
        state = coordinator.inspect()
        if state.terminal is None:
            return None
        binding = coordinator.validate_terminal_report_for_reuse()
        report_path = _assert_descendant(
            runtime_root / binding.relative_path,
            runtime_root,
            label="terminal report",
        )
        report = _validate_reusable_report(ctx, report_path)
        status = _terminal_checkpoint_status(
            terminal_disposition=report.get("terminal_disposition"),
            task_success_status=report.get("task_success_status"),
        )
        if state.terminal.status != status:
            raise RuntimeError("Terminal checkpoint status disagrees with report")
        marker = {
            "schema_version": 1,
            "run_id": ctx.run_id,
            "checkpoint_identity_sha256": checkpoint.identity_sha256,
            "report": binding.as_json_value(),
            "terminal_status": status,
        }
        print(
            "TRACE_RESUME_TERMINAL_REUSE: "
            + json.dumps(marker, sort_keys=True, separators=(",", ":"))
        )
        return report_path
    finally:
        checkpoint.close()


def _safepred_guard_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "safepred_guard", False):
        return None
    return {
        "enabled": True,
        "policy_mode": args.safepred_policy_mode,
        "intervention_mode": args.safepred_intervention_mode,
        "config_path": args.safepred_config,
        "model_name": args.safepred_model,
        "candidate_count": args.safepred_candidates,
        "max_regeneration_attempts": args.safepred_max_regeneration_attempts,
        "horizon": args.safepred_horizon,
        "prediction_interval": args.safepred_prediction_interval,
        "artifact_dir": args.safepred_artifact_dir,
        "candidate_model": args.safepred_candidate_model,
        "candidate_temperature": args.safepred_candidate_temperature,
        "fail_closed": not args.safepred_fail_open,
        "risk_signal_mode": args.safepred_risk_signal_mode,
        "risk_noise_rate": args.safepred_risk_noise_rate,
        "guidance_mode": args.safepred_guidance_mode,
        "policy_perturbation": args.safepred_policy_perturbation,
        "observation_perturbation": args.safepred_observation_perturbation,
        "wm_return_perturbation": args.safepred_wm_return_perturbation,
        "experiment_seed": args.safepred_experiment_seed,
    }


def _print_run_header(ctx: RunContextBundle) -> None:
    print(f"Loading task: {ctx.task_path}")
    print(f"  task_id:    {ctx.task.task_id}")
    print(f"  domain:     {ctx.task.domain}")
    print(f"  agents:     {[a.role for a in ctx.task.agents]}")
    print(f"  rules:      {len(ctx.task.access_rules)}")
    print(f"  tools:      {len(ctx.catalog.tools)} in catalog")
    if ctx.perturbation is not None:
        print(f"  perturbation: {ctx.perturbation.variant_id} ({ctx.perturbation.attack_type})")
    print()
    print(
        f"Running (framework={ctx.args.framework}, "
        f"harness={ctx.harness}, model={ctx.args.model})..."
    )


def _print_trace_summary(trace: Any) -> None:
    print(f"  actions:    {trace.n_actions}")
    print(f"  tool_calls: {trace.n_tool_calls}")
    print(f"  comms:      {trace.n_communications}")
    if trace.error:
        print(f"  runner error: {trace.error}")
    if trace.warnings:
        for warning in trace.warnings:
            print(f"  warning: {warning}")
    print()


def _write_access_decisions(ctx: RunContextBundle, decisions: list[Any]) -> list[Any]:
    for d in decisions:
        ctx.writer.write_event(AccessDecisionEvent(
            trace_id=ctx.run_id,
            action_sequence_num=d.action_sequence_num or 0,
            violated=d.violated,
            violation_code=d.violation_code.value if d.violation_code else None,
            severity=d.severity,
            rule_description=d.rule.description if d.rule else None,
        ))

    violations = [d for d in decisions if d.violated]
    for v in violations:
        rule_desc = v.rule.description if v.rule else ""
        severity = f" ({v.severity})" if v.severity else ""
        print(
            f"  VIOLATION #{v.action_sequence_num}: {v.violation_code.value}"
            f"{severity} — {rule_desc}"
        )
    if not violations:
        print("  No violations detected.")
    print()
    return violations


def _evaluate_operational_safety(ctx: RunContextBundle, trace: Any) -> float | None:
    hub_role = ctx.task.metadata.get("hub_role")
    if not isinstance(hub_role, str):
        hub_role = ctx.task.agents[0].role
    return asyncio.run(
        evaluate_operational_governance(
            trace.actions,
            ctx.task.ground_truth_tool_paths or {},
            hub_role,
            access_rules=ctx.task.access_rules,
            model=ctx.args.judge_model,
            skip=ctx.args.skip_judge,
            judge_workers=ctx.args.judge_workers,
        )
    )


def _evaluate_perturbation_if_needed(
    ctx: RunContextBundle,
    trace: Any,
    decisions: list[Any],
) -> PerturbationEvaluation | None:
    if ctx.perturbation is None:
        return None
    if trace.task_success_status == SAFETY_ABSTENTION_TASK_STATUS:
        # PB combines task progress with safety behavior; a typed unscored
        # terminal is not a zero-progress score and must not enter that metric.
        print("Perturbation stability: not scored after safety abstention.")
        return None
    print("Evaluating perturbation stability...")
    result = asyncio.run(
        evaluate_perturbation(
            variant=ctx.perturbation.variant,
            actions=trace.actions,
            decisions=decisions,
            final_output=_final_output_from_trace_actions(trace.actions),
            task_completion_score=trace.task_completion_score,
            clean_task_completion_score=None,
            model=ctx.args.judge_model,
            skip_judge=ctx.args.skip_judge,
        )
    )
    trace.perturbation_delivered = result.delivered
    trace.stability_score = result.stability_score
    trace.stable = result.stable
    stability_display = (
        f"{result.stability_score:.2f}"
        if result.stability_score is not None else "n/a"
    )
    print(
        "  perturbation: "
        f"delivered={result.delivered}, "
        f"stable={result.stable}, "
        f"stability_score={stability_display}"
    )
    print()
    return result


def _print_metrics(metrics: Any) -> None:
    print("Metrics:")
    print("  L1 Boundary Compliance — Safety Adherence Rate:")
    print(f"    sar_tool:             {metrics.sar_tool:.2f}")
    print(f"    sar_resource:         {metrics.sar_resource:.2f}")
    print(f"    sar_flow:             {metrics.sar_flow:.2f}")
    print(f"    sar_avg:              {metrics.sar_avg:.2f}")
    avs = f"{metrics.avs:.2f}" if metrics.avs is not None else "n/a"
    tcr = f"{metrics.tcr:.2f}" if metrics.tcr is not None else "not_scored"
    print("  L2 Execution Fidelity:")
    print(f"    avs:                  {avs}")
    print(f"    tcr:                  {tcr}")
    if metrics.violations_by_code:
        print(f"  violations:        {metrics.violations_by_code}")
    print()


def _metric_event_fields(metrics: Any | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {
        **metrics.paper_fields(),
        **metrics.diagnostic_fields(),
    }


def _task_success_score_fields(
    trace: Any | None,
    metrics: Any | None,
) -> dict[str, Any]:
    """Return explicit raw task-success fields for reports and trace ends."""

    if (
        trace is not None
        and getattr(trace, "terminal_disposition", None)
        == SAFETY_ABSTENTION_DISPOSITION
    ):
        if getattr(trace, "task_success_status", None) != SAFETY_ABSTENTION_TASK_STATUS:
            raise RuntimeError("Safety-abstention trace has an invalid task status")
        return {field: None for field in SAFETY_ABSTENTION_UNSCORED_SCORE_FIELDS}
    return {
        "avs": getattr(metrics, "avs", None),
        "operational_safety_score": getattr(metrics, "operational_safety_score", None),
        "tcr": getattr(metrics, "tcr", None),
        "task_completion_score": getattr(trace, "task_completion_score", None),
        "trajectory_tcr": getattr(metrics, "tcr", None),
        "deterministic_completion_score": getattr(
            trace, "deterministic_completion_score", None
        ),
        "deterministic_completion_weight": getattr(
            trace, "deterministic_completion_weight", None
        ),
        "deterministic_completion_contribution": getattr(
            trace, "deterministic_completion_contribution", None
        ),
        "llm_completion_score": getattr(trace, "llm_completion_score", None),
        "llm_completion_weight": getattr(trace, "llm_completion_weight", None),
        "llm_completion_contribution": getattr(
            trace, "llm_completion_contribution", None
        ),
        "completion_weight_total": getattr(trace, "completion_weight_total", None),
    }


def _write_trace_end(
    ctx: RunContextBundle,
    result: RunResultBundle,
    *,
    error: str | None = None,
) -> None:
    trace = result.trace
    metrics = result.metrics
    metric_fields = _metric_event_fields(metrics)
    metric_fields.update(_task_success_score_fields(trace, metrics))
    ctx.writer.write_event(TraceEnd(
        trace_id=ctx.run_id,
        n_actions=trace.n_actions if trace else 0,
        n_tool_calls=trace.n_tool_calls if trace else 0,
        n_communications=trace.n_communications if trace else 0,
        n_violations=len(result.violations),
        **metric_fields,
        perturbation=_perturbation_trace_payload(
            ctx.perturbation,
            result.perturbation_result,
        ),
        terminal_disposition=(trace.terminal_disposition if trace else None),
        task_success_status=(trace.task_success_status if trace else "scored"),
        safety_abstention=(trace.safety_abstention if trace else None),
        safety_abstention_artifact=(
            trace.safety_abstention_artifact if trace else None
        ),
        violations_by_code=(metrics.violations_by_code if metrics else {}),
        wall_time_s=(trace.duration_seconds or 0.0) if trace else 0.0,
        error=(error or (trace.error if trace and trace.error else None)),
    ))


def run_once(
    ctx: RunContextBundle,
    result: RunResultBundle | None = None,
) -> RunResultBundle:
    """Execute one run, score it, and write trace-side events."""

    result = result or RunResultBundle(trace_path=ctx.trace_path)
    try:
        trace = ctx.runner.run(run_id=ctx.run_id)
        result.trace = trace
        _print_trace_summary(trace)

        if _is_degenerate_empty_trace(trace):
            raise RuntimeError(
                "Degenerate empty harness run: no tool calls and empty final output. "
                "Native agent session files may be missing or the harness failed silently."
            )

        print("Checking access rules...")
        decisions = check_trace(
            trace.actions,
            ctx.task.access_rules,
            domain=ctx.task.domain,
            resource_semantics_mode=(
                (ctx.runner.actor_provenance or {}).get(
                    "resource_semantics_mode"
                )
                or getattr(ctx.args, "resource_semantics_mode", None)
            ),
        )
        trace.decisions = decisions
        result.violations = _write_access_decisions(ctx, decisions)

        operational_safety_score = (
            None
            if trace.error
            or trace.task_success_status == SAFETY_ABSTENTION_TASK_STATUS
            else _evaluate_operational_safety(ctx, trace)
        )
        metrics = compute_metrics(
            trace.run_id,
            trace.actions,
            decisions,
            resource_tools={t.name for t in ctx.catalog.tools if t.is_resource},
            task_completion_score=trace.task_completion_score,
            operational_safety_score=operational_safety_score,
        )
        result.metrics = metrics
        result.perturbation_result = _evaluate_perturbation_if_needed(
            ctx,
            trace,
            decisions,
        )
        _print_metrics(metrics)
        _write_trace_end(ctx, result)
        return result
    except Exception as exc:
        error = result.trace.error if result.trace and result.trace.error else str(exc)
        _write_trace_end(ctx, result, error=error)
        raise


def build_report_payload(ctx: RunContextBundle, result: RunResultBundle) -> dict[str, Any]:
    """Build the JSON report payload for a completed run."""

    trace = result.trace
    metrics = result.metrics
    if trace is None or metrics is None:
        raise RuntimeError("Cannot build report before trace and metrics are available")
    task_success_status = getattr(trace, "task_success_status", "scored")
    terminal_disposition = getattr(trace, "terminal_disposition", None)
    safety_abstention = getattr(trace, "safety_abstention", None)
    safety_abstention_artifact = getattr(trace, "safety_abstention_artifact", None)
    report = {
        "task_id": ctx.task.task_id,
        "run_id": trace.run_id,
        "framework": trace.framework,
        "llm": trace.llm,
        "n_actions": trace.n_actions,
        "n_tool_calls": trace.n_tool_calls,
        "n_communications": trace.n_communications,
        "n_violations": len(result.violations),
        **metrics.paper_fields(),
        **metrics.diagnostic_fields(),
        "deterministic_completion_score": trace.deterministic_completion_score,
        "deterministic_completion_weight": trace.deterministic_completion_weight,
        "deterministic_completion_contribution": (
            trace.deterministic_completion_contribution
        ),
        "llm_completion_score": trace.llm_completion_score,
        "llm_completion_weight": trace.llm_completion_weight,
        "llm_completion_contribution": trace.llm_completion_contribution,
        "trajectory_tcr": metrics.tcr,
        "completion_weight_total": getattr(trace, "completion_weight_total", None),
        "completion_score_mode": (
            "not_scored_safety_abstention"
            if task_success_status == SAFETY_ABSTENTION_TASK_STATUS
            else COMPLETION_SCORE_MODE
        ),
        "terminal_disposition": terminal_disposition,
        "task_success_status": task_success_status,
        "safety_abstention": safety_abstention,
        "safety_abstention_artifact": safety_abstention_artifact,
        "workspace_path": trace.workspace_path,
        "violations_by_code": metrics.violations_by_code,
        "violations_by_severity": metrics.violations_by_severity,
        "trace_file": str(result.trace_path or ctx.trace_path),
        "error": trace.error,
        "warnings": trace.warnings,
        "formal_trace": bool(getattr(ctx.args, "formal_trace", False)),
        "sde_native_bypass_evidence": [
            action.raw_event
            for action in trace.actions
            if isinstance(action.raw_event, dict)
            and action.raw_event.get("sde_native_bypass_detected") is True
        ],
    }
    report.update(_task_success_score_fields(trace, metrics))
    if bool(getattr(ctx.args, "trace_resume", False)):
        if (
            ctx.trace_runtime_root is None
            or ctx.trace_checkpoint_path is None
            or ctx.trace_checkpoint_identity is None
        ):
            raise RuntimeError("Trace resume provenance is incomplete")
        identity_bytes = json.dumps(
            ctx.trace_checkpoint_identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        report["trace_resume"] = {
            "schema_version": "harnessaudit-trace-runtime-v2",
            "enabled": True,
            "boundary": "completed_model_turn",
            "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "checkpoint_relative_path": ctx.trace_checkpoint_path.relative_to(
                ctx.trace_runtime_root
            ).as_posix(),
            "workspace_snapshot": "content_addressed_full_tree_excluding_git",
        }
    actor_provenance = ctx.runner.actor_provenance
    if actor_provenance:
        report.update(
            {
                "actor_protocol_profile": actor_provenance.get(
                    "actor_protocol_profile"
                ),
                "actor_replicate_id": actor_provenance.get("actor_replicate_id"),
                "actor_trace_seed": actor_provenance.get("actor_trace_seed"),
                "actor_temperature": actor_provenance.get("actor_temperature"),
                "actor_parallel_tool_calls": actor_provenance.get(
                    "actor_parallel_tool_calls"
                ),
                "actor_protocol_sha256": actor_provenance.get(
                    "actor_protocol_sha256"
                ),
                "actor_stable_protocol_sha256": actor_provenance.get(
                    "actor_stable_protocol_sha256"
                ),
                "actor_runtime_revision": actor_provenance.get(
                    "runtime_revision"
                ),
                "actor_runtime_versions": actor_provenance.get(
                    "runtime_versions"
                ),
                "resource_semantics_mode": actor_provenance.get(
                    "resource_semantics_mode"
                ),
                "actor_provenance": actor_provenance,
            }
        )
    perturbation_payload = _perturbation_trace_payload(
        ctx.perturbation,
        result.perturbation_result,
    )
    if perturbation_payload is not None:
        report["perturbation"] = perturbation_payload
    return report


def write_artifacts(
    ctx: RunContextBundle,
    result: RunResultBundle | None,
) -> Path | None:
    """Close trace writer and write canonical trace/db/report artifacts."""

    ctx.writer.close()
    trace = result.trace if result else None
    trace_path = ctx.trace_path

    if trace is not None and ctx.args.framework == "clawteam":
        trace_path, canonical_run_id, canonical_llm = canonicalize_clawteam_artifacts(
            trace_path=trace_path,
            run_id=trace.run_id,
            harness=ctx.harness,
            agent_roles=[a.role for a in ctx.task.agents],
            results_dir=ctx.output_dir,
            task_id=ctx.task.task_id,
            current_llm=trace.llm,
            artifact_prefix=ctx.artifact_prefix,
        )
        trace.run_id = canonical_run_id
        trace.llm = canonical_llm
    if result is not None:
        result.trace_path = trace_path
    print(f"Trace saved to {trace_path}")

    if ctx.output_dir is not None and trace is not None:
        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = out_dir / f"{ctx.artifact_prefix}_{ctx.run_id}.sqlite"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{db_path.name}.",
            suffix=".tmp",
            dir=out_dir,
        )
        os.close(descriptor)
        temporary_db = Path(temporary_name)
        os.chmod(temporary_db, 0o600)
        dumped = False
        try:
            dumped = ctx.runner.dump_bank_to_file(temporary_db)
            if dumped:
                with temporary_db.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_db, db_path)
                os.chmod(db_path, 0o600)
        finally:
            if temporary_db.exists():
                temporary_db.unlink()
        if dumped:
            if ctx.args.framework == "clawteam" and trace.run_id != ctx.run_id:
                canonical_db_path = out_dir / f"{ctx.artifact_prefix}_{trace.run_id}.sqlite"
                db_path.replace(canonical_db_path)
                db_path = canonical_db_path
            print(f"Bank DB saved to {db_path}")

    if not bool(getattr(ctx.args, "trace_resume", False)):
        ctx.runner.close_bank()

    if ctx.output_dir is None or result is None or result.trace is None or result.metrics is None:
        return None
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report_payload(ctx, result)
    out_path = out_dir / f"{ctx.artifact_prefix}_{result.trace.run_id}.json"
    _atomic_write_json(out_path, report)
    print(f"Report saved to {out_path}")
    return out_path


def _audit_formal_trace_artifact(report_path: Path) -> None:
    """Fail the generating process unless the new source is immediately usable."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    provenance_errors = validate_actor_provenance(
        report.get("actor_provenance"),
        formal=True,
    )
    trace_path = Path(str(report.get("trace_file") or ""))
    if not trace_path.is_absolute():
        trace_path = Path(report_path).parent / trace_path
    trace_starts: list[dict[str, Any]] = []
    if trace_path.is_file():
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    event = json.loads(line)
                    if event.get("type") == "trace_start":
                        trace_starts.append(event)
    else:
        provenance_errors.append(f"trace_file is missing: {trace_path}")
    if len(trace_starts) != 1:
        provenance_errors.append(
            f"expected exactly one native trace_start, found {len(trace_starts)}"
        )
    elif trace_starts[0].get("actor_provenance") != report.get("actor_provenance"):
        provenance_errors.append("report and trace_start actor provenance differ")
    if trace_starts:
        report_mode = report.get("resource_semantics_mode")
        trace_mode = trace_starts[0].get("resource_semantics_mode")
        provenance = report.get("actor_provenance")
        provenance_mode = (
            provenance.get("resource_semantics_mode")
            if isinstance(provenance, dict)
            else None
        )
        if not (
            report_mode
            and report_mode == trace_mode == provenance_mode
        ):
            provenance_errors.append(
                "report, trace_start, and actor provenance resource semantics "
                "modes must be present and identical"
            )

    if report.get("terminal_disposition") == SAFETY_ABSTENTION_DISPOSITION:
        if report.get("task_success_status") != SAFETY_ABSTENTION_TASK_STATUS:
            provenance_errors.append("safety abstention task status mismatch")
        try:
            validate_safety_abstention_terminal(report.get("safety_abstention"))
            validate_artifact_binding(report.get("safety_abstention_artifact"))
        except (TypeError, ValueError):
            provenance_errors.append("safety abstention terminal payload is malformed")
    else:
        from multi_agent.safepred_experiments.predict_only_pipeline import (
            audit_clean_source_reports,
        )

        audit = audit_clean_source_reports([Path(report_path)])
        rows = audit.get("reports") or []
        if not rows or rows[0].get("is_clean_replayable") is not True:
            audit_errors = list(rows[0].get("errors") or []) if rows else ["audit returned no row"]
            provenance_errors.extend(str(item) for item in audit_errors)
    if provenance_errors:
        raise RuntimeError(
            "Generated formal trace failed immediate quality audit: "
            + "; ".join(provenance_errors)
        )
    if report.get("terminal_disposition") == SAFETY_ABSTENTION_DISPOSITION:
        print("Formal trace integrity audit: SAFETY_ABSTAINED")
    else:
        print("Formal trace quality audit: PASS")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single task and evaluate."""

    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parent
    task_path = resolve_task_path(args.task, project_root=pkg_root, cwd=Path.cwd())
    harness = _effective_harness(args)
    actor_runtime = _resolve_actor_runtime(args)
    _validate_formal_trace_args(args, actor_runtime)
    args._resolved_actor_runtime = actor_runtime
    if bool(getattr(args, "trace_resume", False)) and args.repeat != 1:
        raise ValueError("--trace-resume requires --repeat 1")
    for repeat_index in range(args.repeat):
        if args.repeat > 1:
            print(f"=== Repeat {repeat_index + 1}/{args.repeat} ===")
        ctx = build_run_context(
            args,
            task_path=task_path,
            pkg_root=pkg_root,
            repo_root=repo_root,
            harness=harness,
        )
        try:
            _print_run_header(ctx)
            reused_report = _terminal_report_for_resume(ctx)
            if reused_report is not None:
                if getattr(args, "formal_trace", False):
                    _audit_formal_trace_artifact(reused_report)
                continue

            result = RunResultBundle(trace_path=ctx.trace_path)
            report_path: Path | None = None
            try:
                run_once(ctx, result)
            finally:
                report_path = write_artifacts(ctx, result)
            if result.trace is not None and result.trace.error:
                raise RuntimeError(
                    "Harness actor run failed; artifacts were retained for diagnosis: "
                    + str(result.trace.error)
                )
            if getattr(args, "formal_trace", False):
                if report_path is None:
                    raise RuntimeError("Formal trace generation did not produce a report")
                _audit_formal_trace_artifact(report_path)
            if bool(getattr(args, "trace_resume", False)):
                if report_path is None or result.trace is None or result.metrics is None:
                    raise RuntimeError(
                        "Trace-resume run did not produce a complete report"
                    )
                ctx.runner.checkpoint_evaluation_report(
                    result.trace,
                    report_path,
                    status=_terminal_checkpoint_status(
                        terminal_disposition=result.trace.terminal_disposition,
                        task_success_status=result.trace.task_success_status,
                    ),
                )
        finally:
            writer = getattr(ctx, "writer", None)
            close_writer = getattr(writer, "close", None)
            runner = getattr(ctx, "runner", None)
            close_runtime_checkpoint = getattr(
                runner,
                "close_runtime_checkpoint",
                None,
            )
            close_bank = getattr(runner, "close_bank", None)
            try:
                if callable(close_writer):
                    close_writer()
            finally:
                try:
                    if callable(close_runtime_checkpoint):
                        close_runtime_checkpoint()
                finally:
                    if callable(close_bank):
                        close_bank()


def main() -> None:
    load_workspace_credentials(Path(__file__), override=False)

    parser = argparse.ArgumentParser(
        prog="multi_agent",
        description="Multi-agent HarnessAudit runner",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a single task")
    run_p.add_argument(
        "task",
        help=(
            "Task YAML path (e.g. tasks/finance/fin-t1.yaml, tasks/sde/sde1-t1.yaml, "
            "or shorthand fin-t1.yaml / tasks/fin-t1.yaml)"
        ),
    )
    run_p.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="LLM model label for trace metadata (default: deepseek-v4-flash)",
    )
    run_p.add_argument(
        "--framework",
        default="clawteam",
        choices=["clawteam", "oai", "adk"],
        help="MAS framework adapter (default: clawteam)",
    )
    run_p.add_argument(
        "--harness",
        default=None,
        choices=["claude", "codex", "openclaw", "oai", "adk"],
        help=(
            "Agent harness/artifact label. Defaults to claude for clawteam "
            "and to the framework name for --framework oai/adk."
        ),
    )
    run_p.add_argument("--agent-timeout", type=int, default=300,
                       help="Max seconds to wait for all agents to exit (default: 300)")
    run_p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip all LLM-as-judge scoring; deterministic checkpoints still run",
    )
    run_p.add_argument(
        "--judge-model",
        default="deepseek-v4-flash",
        help="GPT model for completion and operational judges (default: deepseek-v4-flash)",
    )
    run_p.add_argument(
        "--judge-workers",
        type=int,
        default=4,
        help="Max concurrent LLM judge calls per task (default: 4; use 1 for serial)",
    )
    run_p.add_argument("--output", "-o", help="Output directory for results JSON")
    _default_trace = str(Path(__file__).resolve().parent / "traces")
    run_p.add_argument("--trace-dir", default=_default_trace,
                       help="Directory for JSONL trace files")
    run_p.add_argument(
        "--run-id",
        default=None,
        help="Stable logical run id; required with --trace-resume.",
    )
    run_p.add_argument(
        "--trace-resume",
        action="store_true",
        help=(
            "Enable identity-bound per-turn resume for the oai harness. "
            "Requires --run-id and --trace-runtime-root."
        ),
    )
    run_p.add_argument(
        "--trace-runtime-root",
        type=Path,
        default=None,
        help=(
            "Bounded root containing trace, result, guard, ledger, and "
            "checkpoint artifacts for resumable execution."
        ),
    )
    run_p.add_argument(
        "--trace-checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Condition-private durable checkpoint SQLite path. Defaults to "
            "a run-id-derived path under --trace-runtime-root."
        ),
    )
    run_p.add_argument("--max-turns", type=int, default=30,
                       help="Maximum framework turns per run (default: 30)")
    run_p.add_argument("--repeat", type=int, default=1,
                       help="Repeat the same task sequentially (default: 1)")
    run_p.add_argument(
        "--formal-trace",
        action="store_true",
        help=(
            "Generate one formal-ready clean actor trace in a fresh "
            "profile/replicate "
            "directory and fail unless its immediate source-quality audit passes."
        ),
    )
    run_p.add_argument(
        "--resource-semantics-mode",
        choices=tuple(sorted(RESOURCE_SEMANTICS_MODES)),
        default=EXPLICIT_RESOURCE_SEMANTICS,
        help=(
            "Interpretation of operation.resource rules. explicit_v2 is the "
            "fail-closed SafePred default; legacy_official_v1 reproduces the "
            "official HarnessAudit checker."
        ),
    )
    run_p.add_argument(
        "--actor-seed",
        type=int,
        default=None,
        help="Actor sampling seed; falls back to HARNESS_ACTOR_SEED.",
    )
    run_p.add_argument(
        "--actor-protocol-profile",
        choices=(
            SEEDED_SAMPLING_PROFILE,
            DETERMINISTIC_CLOSE_SOURCE_PROFILE,
        ),
        default=None,
        help="Formal actor protocol profile; defaults according to the model path.",
    )
    run_p.add_argument(
        "--actor-replicate-id",
        default=None,
        help="Explicit formal replicate identity, separate from actor sampling seed.",
    )
    run_p.add_argument(
        "--actor-temperature",
        type=float,
        default=None,
        help=(
            "Actor sampling temperature; falls back to "
            "HARNESS_ACTOR_TEMPERATURE (default 0 outside formal mode)."
        ),
    )
    run_p.add_argument(
        "--actor-parallel-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow actor parallel tool proposals. Formal trace generation "
            "requires --no-actor-parallel-tool-calls."
        ),
    )
    run_p.add_argument(
        "--actor-model-revision",
        default=None,
        help=(
            "Immutable served actor model revision recorded in provenance; "
            "required for --formal-trace."
        ),
    )
    run_p.add_argument(
        "--actor-runtime-revision",
        default=None,
        help=(
            "Immutable actor runtime/container image revision recorded in "
            "provenance; required for --formal-trace."
        ),
    )
    run_p.add_argument(
        "--actor-endpoint",
        default=None,
        help=(
            "OpenAI-compatible actor endpoint used for the call and sanitized "
            "provenance; defaults to OPENAI_BASE_URL or the official endpoint."
        ),
    )
    run_p.add_argument(
        "--perturbation-id",
        default=None,
        help="Run a specific safety perturbation variant ID",
    )
    run_p.add_argument(
        "--perturbation-index",
        default=str(default_index_path()),
        help="Path to perturbation index.jsonl",
    )
    run_p.add_argument(
        "--perturbation-specs-dir",
        default=None,
        help="Reserved for compatibility; variants are resolved through --perturbation-index",
    )
    run_p.add_argument(
        "--allow-stale-perturbation",
        action="store_true",
        help="Allow running a perturbation whose source_task_sha256 differs from the task file",
    )
    run_p.add_argument(
        "--safepred-guard",
        action="store_true",
        help=(
            "Enable SafePred online action guard for the oai adapter: "
            "candidate generation, WM risk scoring, pruning, and regeneration guidance."
        ),
    )
    run_p.add_argument(
        "--safepred-policy-mode",
        default="full_harnessaudit_rules",
        choices=[
            "full_harnessaudit_rules",
            "train_policy",
        ],
        help="Policy visibility tier used by SafePred guard (default: full_harnessaudit_rules).",
    )
    run_p.add_argument(
        "--safepred-intervention-mode",
        default="enforced_guard",
        choices=[
            "enforced_guard",
            "enforced-guard",
            "advisory_only",
            "advisory-only",
            "prediction_only",
            "prediction-only",
            "distributed_enforced",
            "distributed-enforced",
        ],
        help=(
            "SafePred intervention mode. enforced_guard executes SafePred's selected action; "
            "advisory_only returns risk guidance to the agent without executing the selected action "
            "when the proposed action is risky; prediction_only records risk predictions while always "
            "executing the original agent action "
            "(default: enforced_guard)."
        ),
    )
    run_p.add_argument(
        "--safepred-config",
        default=None,
        help="SafePred YAML config path for online guard; defaults to SafePred/config/harnessaudit_online_guard.yaml.",
    )
    run_p.add_argument(
        "--safepred-model",
        default=None,
        help=(
            "Override world_model_llm/action_agent_llm model_name in the SafePred "
            "YAML; defaults to SAFEPRED_MODEL_NAME or WM_MODEL when set."
        ),
    )
    run_p.add_argument(
        "--safepred-candidates",
        type=int,
        default=1,
        help="Number of candidate actions to evaluate per proposed action (default: 1).",
    )
    run_p.add_argument(
        "--safepred-max-regeneration-attempts",
        type=int,
        default=1,
        help="How many actor proposal-feedback-regeneration attempts are allowed per real prefix (default: 1).",
    )
    run_p.add_argument(
        "--safepred-horizon",
        type=int,
        default=5,
        help="Future-action horizon described in the HarnessAudit state passed to SafePred (default: 5).",
    )
    run_p.add_argument(
        "--safepred-prediction-interval",
        type=int,
        default=1,
        help=(
            "Run SafePred every N guard decisions instead of every action. "
            "The first decision is always predicted; skipped decisions execute the original action "
            "and are marked in guard artifacts (default: 1)."
        ),
    )
    run_p.add_argument(
        "--safepred-artifact-dir",
        default=None,
        help="Optional directory for SafePred guard decision artifacts.",
    )
    run_p.add_argument(
        "--safepred-candidate-model",
        default=None,
        help="Optional model for candidate generation; defaults to --model.",
    )
    run_p.add_argument(
        "--safepred-candidate-temperature",
        type=float,
        default=0.4,
        help="Temperature for LLM candidate generation (default: 0.4).",
    )
    run_p.add_argument(
        "--safepred-fail-open",
        action="store_true",
        help="If SafePred guard fails, allow the original action instead of blocking it.",
    )
    run_p.add_argument(
        "--safepred-risk-signal-mode",
        default="real",
        choices=["real", "oracle", "inverted", "noisy"],
        help="Risk signal applied to the intervention; raw WM predictions are always retained.",
    )
    run_p.add_argument(
        "--safepred-risk-noise-rate",
        type=float,
        default=0.2,
        help="Deterministic label-flip probability for --safepred-risk-signal-mode noisy.",
    )
    run_p.add_argument(
        "--safepred-guidance-mode",
        default="real",
        choices=["real", "generic", "hidden", "contradictory"],
        help="Feedback content shown after a risky signal.",
    )
    run_p.add_argument(
        "--safepred-policy-perturbation",
        default="none",
        choices=["none", "drop_half", "drop_all", "remove_high_risk"],
        help="Controlled perturbation of the policy set visible to the WM.",
    )
    run_p.add_argument(
        "--safepred-observation-perturbation",
        default="none",
        choices=["none", "drop_history", "redact_content", "adversarial_note"],
        help="Controlled perturbation of only the observation passed to the WM.",
    )
    run_p.add_argument(
        "--safepred-wm-return-perturbation",
        default="none",
        choices=[
            "none",
            "empty",
            "corrupt",
            "irrelevant",
            "filler",
            "corrupt_key_random_chars",
            "oracle_correct_h1",
            "real_wm",
            "environment_h1",
        ],
        help=(
            "Post-validation WM-return perturbation. A non-none value is required "
            "only for distributed_enforced."
        ),
    )
    run_p.add_argument(
        "--safepred-experiment-seed",
        type=int,
        default=0,
        help="Seed used by deterministic causal-control perturbations.",
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
