"""One stable command-line entry point for all experiment workflows."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from .model_stack import ResolvedModelStack, load_model_stack_profile
from .run_catalog import write_catalog
from .workflows import (
    HARNESS_ROOT,
    ISBENCH_ROOT,
    PROJECT_ROOT,
    RUNS_ROOT,
    SAFEPRED_ROOT,
    WORKFLOWS,
    Workflow,
    workflow_environment,
)


SECRET_OPTIONS = {
    "--api-key",
    "--agent-api-key",
    "--tier3-api-key",
    "--vlm-api-key",
    "--local-serve-key",
}

FORMAL_NATIVE_MODEL_OPTIONS = {
    "ha-online": {
        "--model",
        "--judge-model",
        "--safepred-model",
    },
    "isbench-online": {
        "--agent-model",
        "--agent-base-url",
        "--safepred-model",
        "--safepred-base-url",
    },
}

EXTERNAL_CONFIG_PATH_OPTIONS = {
    "--safepred-config",
    "--preregistration",
    "--compositionality-manifest",
    "--faithfulness-adjudication-protocol",
    "--adjudicator-blinding-protocol",
    # Native workflows execute from benchmark-specific working directories;
    # bind caller-relative artifact and IS-Bench input paths first.
    "--is-root",
    "--task-list",
    "--output-root",
    "--trace-root",
    "--analysis-dir",
}


def _split_forwarded(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    separator = argv.index("--")
    return argv[:separator], argv[separator + 1 :]


def _redacted(command: list[str]) -> str:
    output: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            output.append("[REDACTED]")
            redact_next = False
            continue
        option = argument.split("=", 1)[0]
        if option in SECRET_OPTIONS:
            if "=" in argument:
                output.append(f"{option}=[REDACTED]")
            else:
                output.append(argument)
                redact_next = True
            continue
        output.append(argument)
    return shlex.join(output)


def _resolve_python(workflow: Workflow, override: str | None) -> Path:
    return Path(override).expanduser() if override else workflow.python


def _forwarded_option(forwarded: list[str], name: str) -> str | None:
    for index, argument in enumerate(forwarded):
        if argument == name and index + 1 < len(forwarded):
            return forwarded[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _run_environment(workflow: Workflow, forwarded: list[str]) -> dict[str, str]:
    env = workflow_environment(workflow)
    agent_base_url = (
        env.get("AGENT_BASE_URL", "").strip()
        or env.get("HARNESSAUDIT_BASE_URL", "").strip()
    )
    if agent_base_url:
        env["OPENAI_API_BASE"] = agent_base_url
        env["OPENAI_BASE_URL"] = agent_base_url
    agent_api_key = (
        env.get("AGENT_API_KEY", "").strip()
        or env.get("HARNESSAUDIT_API_KEY", "").strip()
    )
    if agent_api_key:
        env["OPENAI_API_KEY"] = agent_api_key
    if workflow.name == "isbench-trace":
        env["ISBENCH_TRACE_STRICT"] = "1"
        # Strict resume never blindly trusts a report: the native launcher
        # re-audits seed, cohort, protocol, labels, images, and source hashes
        # before reuse.  Enable that safe resume for every provider, not only
        # the local-vLLM transport, so rerunning a partial formal cohort cannot
        # resample already accepted actor behavior.
        env["ISBENCH_RESUME_VALID_REPORTS"] = "1"
    if workflow.name == "isbench-awareness":
        env["ISBENCH_RESUME_VALID_REPORTS"] = "1"
    if workflow.name == "isbench-trace" and "--local_llm_serve" in forwarded:
        local_url = _forwarded_option(forwarded, "--local_serve_ip")
        model = _forwarded_option(forwarded, "--model")
        if local_url:
            env["OPENAI_API_BASE"] = local_url
            env["OPENAI_BASE_URL"] = local_url
            env["OPENAI_API_URL"] = local_url
        env["OPENAI_API_KEY"] = env.get("LOCAL_SERVE_KEY", "local")
        if model:
            env["ISBENCH_JUDGER_MODEL"] = model
    return env


def _native_option_values(forwarded: list[str], name: str) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(forwarded):
        if argument == name:
            if index + 1 >= len(forwarded):
                raise ValueError(f"{name} requires a value")
            values.append(forwarded[index + 1])
        elif argument.startswith(f"{name}="):
            values.append(argument.split("=", 1)[1])
    return values


def _resolve_external_config_paths(forwarded: list[str]) -> list[str]:
    """Bind caller-relative external config paths before changing workflow cwd."""

    caller_cwd = Path.cwd()
    resolved: list[str] = []
    index = 0
    while index < len(forwarded):
        argument = forwarded[index]
        option = argument.split("=", 1)[0]
        if option not in EXTERNAL_CONFIG_PATH_OPTIONS:
            resolved.append(argument)
            index += 1
            continue
        if "=" in argument:
            value = argument.split("=", 1)[1]
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = caller_cwd / path
            resolved.append(f"{option}={path.resolve()}")
            index += 1
            continue
        if index + 1 >= len(forwarded):
            raise ValueError(f"{option} requires a path")
        path = Path(forwarded[index + 1]).expanduser()
        if not path.is_absolute():
            path = caller_cwd / path
        resolved.extend((option, str(path.resolve())))
        index += 2
    return resolved


def _bind_native_model_option(
    forwarded: list[str],
    *,
    option: str,
    expected: str,
) -> None:
    observed = _native_option_values(forwarded, option)
    if any(value != expected for value in observed):
        raise ValueError(
            f"native {option}={observed!r} conflicts with model stack value "
            f"{expected!r}"
        )
    if not observed:
        # Native workflows may expose an argparse.REMAINDER boundary for
        # child-only options. Profile-bound model identity belongs to the
        # outer workflow and must therefore be inserted before that boundary.
        boundary = (
            forwarded.index("--extra-args")
            if "--extra-args" in forwarded
            else len(forwarded)
        )
        forwarded[boundary:boundary] = (option, expected)


def _apply_model_stack_profile(
    workflow: Workflow,
    forwarded: list[str],
    *,
    profile_path: Path,
    environment: dict[str, str],
    require_credentials: bool = True,
) -> tuple[list[str], dict[str, str], ResolvedModelStack]:
    """Resolve one profile and map it to a workflow without provider branches."""

    if workflow.name not in {
        "ha-online",
        "ha-trace",
        "isbench-online",
        "isbench-awareness",
    }:
        raise ValueError(
            f"workflow {workflow.name!r} has not adopted model-stack-profile-v1"
        )
    stack = load_model_stack_profile(
        profile_path,
        environ=environment,
        require_credentials=require_credentials,
    )
    env = dict(environment)
    env.update(stack.runtime_environment())

    actor = stack.roles["actor"]
    judge = stack.roles["judge"]
    world_model = stack.roles["world_model"]
    values = list(forwarded)
    if workflow.name in {"ha-online", "ha-trace"}:
        env.update(
            {
                # Compatibility variables are written once at the experiment
                # boundary. Internal model code consumes only resolved values.
                "OPENAI_API_KEY": actor.api_key,
                "OPENAI_BASE_URL": actor.base_url,
                "OPENAI_API_BASE": actor.base_url,
                "HARNESS_ACTOR_TEMPERATURE": str(actor.temperature),
                "HARNESS_ACTOR_MODEL_REVISION": actor.revision,
                "HARNESS_ACTOR_RUNTIME_REVISION": actor.runtime_revision,
                "CUSTOM_API_KEY": world_model.api_key,
                "CUSTOM_API_URL": world_model.base_url,
                "WM_API_KEY": world_model.api_key,
                "WM_BASE_URL": world_model.base_url,
                "WM_MODEL": world_model.model,
                "SAFEPRED_MODEL_NAME": world_model.model,
                "SAFEPRED_MODEL_REVISION": world_model.revision,
                "SAFEPRED_POLICY_EXTRACTION_MODEL": stack.roles[
                    "policy_extractor"
                ].model,
                "WM_SAFETY_POLICY_EXTRACTOR_MODEL": stack.roles[
                    "policy_extractor"
                ].model,
                "WM_SAFETY_POLICY_EXTRACTOR_MAX_TOKENS": str(
                    stack.roles["policy_extractor"].max_tokens
                ),
                "WM_SAFETY_POLICY_EXTRACTOR_TIMEOUT_SECONDS": str(
                    stack.roles["policy_extractor"].timeout_seconds
                ),
                "WM_SAFETY_POLICY_EXTRACTOR_TRANSPORT_MAX_ATTEMPTS": str(
                    stack.roles["policy_extractor"].transport_max_attempts
                ),
                "WM_SAFETY_POLICY_EXTRACTOR_REQUEST_EXTRA_BODY_JSON": json.dumps(
                    stack.roles["policy_extractor"].request_extra_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        _bind_native_model_option(values, option="--model", expected=actor.model)
        _bind_native_model_option(
            values,
            option="--judge-model",
            expected=judge.model,
        )
    else:
        # IS-Bench's actor is passed an explicit local/OpenAI-compatible
        # endpoint and key, while its evaluator consumes OPENAI_* variables.
        # This keeps actor and judge identities independent even when they are
        # served by different providers.
        env.update(
            {
                "LOCAL_SERVE_KEY": actor.api_key,
                "AGENT_API_KEY": actor.api_key,
                "OPENAI_API_KEY": judge.api_key,
                "OPENAI_BASE_URL": judge.base_url,
                "OPENAI_API_BASE": judge.base_url,
                "ISBENCH_JUDGER_MODEL": judge.model,
                "ISBENCH_JUDGE_API_KEY": judge.api_key,
                "ISBENCH_JUDGE_BASE_URL": judge.base_url,
                "ISBENCH_JUDGE_TEMPERATURE": str(judge.temperature),
                "ISBENCH_JUDGE_MAX_TOKENS": str(judge.max_tokens),
                "ISBENCH_JUDGE_REQUEST_TIMEOUT": str(judge.timeout_seconds),
                "ISBENCH_JUDGE_TRANSPORT_MAX_ATTEMPTS": str(
                    judge.transport_max_attempts
                ),
                "ISBENCH_ACTOR_TEMPERATURE": str(actor.temperature),
                "ISBENCH_AGENT_MAX_COMPLETION_TOKENS": str(actor.max_tokens),
                "ISBENCH_ACTOR_REQUEST_TIMEOUT": str(actor.timeout_seconds),
                "ISBENCH_ACTOR_MODEL_REVISION": actor.revision,
                "ISBENCH_ACTOR_RUNTIME_REVISION": actor.runtime_revision,
                "ISBENCH_ACTOR_TRANSPORT_MAX_RETRIES": str(
                    max(0, actor.transport_max_attempts - 1)
                ),
                "ISBENCH_SAFEPRED_API_KEY": world_model.api_key,
                "ISBENCH_SAFEPRED_MODEL_REVISION": world_model.revision,
                "ISBENCH_SAFEPRED_RUNTIME_REVISION": (
                    world_model.runtime_revision
                ),
                "ISBENCH_SAFEPRED_TEMPERATURE": str(
                    world_model.temperature
                ),
                "ISBENCH_SAFEPRED_MAX_TOKENS": str(world_model.max_tokens),
                "ISBENCH_SAFEPRED_REQUEST_TIMEOUT": str(
                    world_model.timeout_seconds
                ),
                "ISBENCH_SAFEPRED_TRANSPORT_MAX_ATTEMPTS": str(
                    world_model.transport_max_attempts
                ),
                "WM_API_KEY": world_model.api_key,
                "WM_BASE_URL": world_model.base_url,
                "WM_MODEL": world_model.model,
            }
        )
        # The native actor and its generation-provenance builder read the
        # ISBENCH-prefixed value, not WM_SAFETY_ACTOR_TOP_P. Keep an explicit
        # profile value authoritative, as for temperature and token limits.
        if actor.top_p is not None:
            env["ISBENCH_ACTOR_TOP_P"] = str(actor.top_p)
        if workflow.name == "isbench-awareness":
            _bind_native_model_option(
                values,
                option="--model",
                expected=actor.model,
            )
            _bind_native_model_option(
                values,
                option="--local_serve_ip",
                expected=actor.base_url,
            )
            if "--local_llm_serve" not in values:
                values.append("--local_llm_serve")
        else:
            _bind_native_model_option(
                values,
                option="--agent-model",
                expected=actor.model,
            )
            _bind_native_model_option(
                values,
                option="--agent-base-url",
                expected=actor.base_url,
            )

    if workflow.name in {"ha-online", "isbench-online"}:
        _bind_native_model_option(
            values,
            option="--safepred-model",
            expected=world_model.model,
        )
    if workflow.name == "isbench-online":
        _bind_native_model_option(
            values,
            option="--safepred-base-url",
            expected=world_model.base_url,
        )
    return values, env, stack


def _strict_trace_forwarded(workflow: Workflow, forwarded: list[str]) -> list[str]:
    """Make the canonical trace workflows fail-closed by default."""

    values = list(forwarded)
    if any(item in {"-h", "--help"} for item in values):
        return values
    if workflow.name == "isbench-trace" and "--strict_trace" not in values:
        values.append("--strict_trace")
    if (
        workflow.name == "ha-trace"
        and values
        and values[0] == "run"
        and "--formal-trace" not in values
    ):
        values.append("--formal-trace")
    return values


def _run_workflow(args: argparse.Namespace, forwarded: list[str]) -> int:
    workflow = WORKFLOWS[args.workflow]
    forwarded = _resolve_external_config_paths(forwarded)
    forwarded = _strict_trace_forwarded(workflow, forwarded)
    model_profile = getattr(args, "model_profile", None)
    forwarded_options = {argument.split("=", 1)[0] for argument in forwarded}
    if (
        workflow.name in FORMAL_NATIVE_MODEL_OPTIONS
        and "--formal" in forwarded_options
    ):
        if model_profile is None:
            raise SystemExit(
                f"Formal {workflow.benchmark} online execution requires "
                "--model-profile; "
                "model and endpoint identity cannot be supplied through native defaults"
            )
        secret_options = sorted(forwarded_options & SECRET_OPTIONS)
        if secret_options:
            raise SystemExit(
                f"Formal {workflow.benchmark} online execution rejects credential "
                "CLI options: "
                + ", ".join(secret_options)
            )
        native_model_options = sorted(
            forwarded_options & FORMAL_NATIVE_MODEL_OPTIONS[workflow.name]
        )
        if native_model_options:
            raise SystemExit(
                f"Formal {workflow.benchmark} online execution requires the model "
                "profile to "
                "be the sole model identity source; remove native option(s): "
                + ", ".join(native_model_options)
            )
    environment = _run_environment(workflow, forwarded)
    if model_profile is not None:
        inspection_only = bool(
            args.print_command
            or any(item in {"-h", "--help", "--dry-run"} for item in forwarded)
        )
        try:
            forwarded, environment, _stack = _apply_model_stack_profile(
                workflow,
                forwarded,
                profile_path=model_profile,
                environment=environment,
                require_credentials=not inspection_only,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid model stack profile: {exc}") from exc
    python = _resolve_python(workflow, args.python)
    command = [str(python), "-m", workflow.module, *forwarded]
    if args.print_command:
        print(_redacted(command), flush=True)
        return 0
    print(_redacted(command), file=sys.stderr, flush=True)
    if not python.is_file():
        raise SystemExit(f"Python interpreter not found: {python}")
    if not workflow.cwd.is_dir():
        raise SystemExit(f"Workflow directory not found: {workflow.cwd}")
    if workflow.requires_gpu and not _gpu_execution_allowed(
        forwarded, environment=environment
    ):
        raise SystemExit(
            f"Workflow '{workflow.name}' requires an allocated compute-node GPU. "
            "Submit scripts/hpc/experiment.sbatch with an explicit --gres, or use "
            "--print-command/--help for inspection."
        )
    pass_fds: tuple[int, ...] = ()
    inherited_lock_fd = environment.get("HA_FORMAL_ROOT_LOCK_FD", "").strip()
    if inherited_lock_fd:
        try:
            descriptor = int(inherited_lock_fd)
        except ValueError as exc:
            raise SystemExit("HA_FORMAL_ROOT_LOCK_FD must be an integer") from exc
        if descriptor < 3:
            raise SystemExit("HA_FORMAL_ROOT_LOCK_FD must identify a non-standard fd")
        pass_fds = (descriptor,)
    completed = subprocess.run(
        command,
        cwd=workflow.cwd,
        env=environment,
        pass_fds=pass_fds,
        check=False,
    )
    return int(completed.returncode)


def _doctor(as_json: bool) -> int:
    checks: list[dict[str, object]] = []

    def add(name: str, path: Path, kind: str = "directory") -> None:
        ok = path.is_file() if kind == "file" else path.is_dir()
        checks.append({"name": name, "ok": ok, "path": str(path), "kind": kind})

    add("project_root", PROJECT_ROOT)
    add("safepred", SAFEPRED_ROOT)
    add("harnessaudit", HARNESS_ROOT)
    add("isbench", ISBENCH_ROOT)
    add("runs", RUNS_ROOT)
    seen_python: set[Path] = set()
    for workflow in WORKFLOWS.values():
        if workflow.python not in seen_python:
            add(f"python:{workflow.python.parent.parent.name}", workflow.python, "file")
            seen_python.add(workflow.python)
    for workflow in WORKFLOWS.values():
        module_path = workflow.cwd / (workflow.module.replace(".", "/") + ".py")
        add(f"workflow:{workflow.name}", module_path, "file")
    dependency_groups = {
        "zhliu": (
            WORKFLOWS["ha-offline"],
            ("yaml", "dotenv", "pydantic", "openai", "agents", "matplotlib", "numpy", "PIL"),
        ),
        "isbench": (
            WORKFLOWS["isbench-online"],
            ("yaml", "pydantic", "openai", "torch", "omnigibson", "numpy", "PIL"),
        ),
    }
    for label, (workflow, modules) in dependency_groups.items():
        checks.append(_dependency_probe(label=label, workflow=workflow, modules=modules))
    ok = all(bool(item["ok"]) for item in checks)
    payload = {"ok": ok, "checks": checks}
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for item in checks:
            marker = "OK" if item["ok"] else "MISSING"
            print(f"[{marker:7}] {item['name']}: {item['path']}")
            if item.get("detail") and not item["ok"]:
                print(f"          {item['detail']}")
    return 0 if ok else 1


def _dependency_probe(
    *,
    label: str,
    workflow: Workflow,
    modules: tuple[str, ...],
) -> dict[str, object]:
    script = (
        "import importlib.util,json; "
        f"mods={list(modules)!r}; "
        "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
        "print(json.dumps({'missing': missing}))"
    )
    try:
        completed = subprocess.run(
            [str(workflow.python), "-c", script],
            cwd=workflow.cwd,
            env=workflow_environment(workflow),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        payload = json.loads(completed.stdout.strip() or "{}")
        missing = payload.get("missing") or []
        ok = completed.returncode == 0 and not missing
        detail = "ok" if ok else f"missing={missing}; stderr={completed.stderr.strip()[:400]}"
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "name": f"dependencies:{label}",
        "ok": ok,
        "path": str(workflow.python),
        "kind": "module_probe",
        "detail": detail,
    }


def _gpu_execution_allowed(
    forwarded: list[str], *, environment: Mapping[str, str] | None = None
) -> bool:
    """Allow a GPU workflow only in an allocation or an explicit runtime env.

    ``workflow_environment`` merges the mode-0600 external runtime file before
    the native child process is created.  Inspect that resolved environment,
    rather than only this launcher's raw process environment, so an authorized
    external GPU host can opt in without putting configuration on the command
    line.  The same environment is then passed unchanged to the child.
    """

    resolved_environment = os.environ if environment is None else environment
    if any(item in {"-h", "--help", "--dry-run"} for item in forwarded):
        return True
    if resolved_environment.get("WM_SAFETY_ALLOW_LOCAL_GPU") == "1":
        return True
    if not resolved_environment.get("SLURM_JOB_ID"):
        return False
    allocation = (
        resolved_environment.get("SLURM_JOB_GPUS")
        or resolved_environment.get("SLURM_STEP_GPUS")
        or resolved_environment.get("CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()
    return allocation not in {"", "-1", "NoDevFiles", "none"}


def _list_workflows(as_json: bool) -> int:
    rows = [
        {
            "name": workflow.name,
            "benchmark": workflow.benchmark,
            "stage": workflow.stage,
            "module": workflow.module,
            "python": str(workflow.python),
            "requires_gpu": workflow.requires_gpu,
            "description": workflow.description,
        }
        for workflow in WORKFLOWS.values()
    ]
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            gpu = " [GPU]" if row["requires_gpu"] else ""
            print(f"{row['name']:<16} {row['benchmark']:<12} {row['stage']}{gpu}")
            print(f"{'':16} {row['description']}")
    return 0


def _run_plot_results(args: argparse.Namespace) -> int:
    forwarded = [str(path) for path in args.inputs]
    if args.output_dir is not None:
        forwarded.extend(("--output-dir", str(args.output_dir)))
    if args.title:
        forwarded.extend(("--title", args.title))
    if args.formats:
        forwarded.append("--formats")
        forwarded.extend(args.formats)
    if args.label:
        forwarded.extend(("--label", args.label))
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.allow_nonformal_input:
        forwarded.append("--allow-nonformal-input")
    if args.json:
        forwarded.append("--json")
    workflow_args = argparse.Namespace(
        workflow="plot-results",
        python=args.python,
        print_command=args.print_command,
    )
    return _run_workflow(workflow_args, forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wm-safety",
        description="Unified HarnessAudit / IS-Bench / SafePred experiment launcher.",
        epilog="Pass native workflow arguments after a literal -- separator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List canonical workflows.")
    list_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Check project paths and interpreters.")
    doctor_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run one canonical workflow.")
    run_parser.add_argument("workflow", choices=tuple(WORKFLOWS))
    run_parser.add_argument("--python", help="Override this workflow's Python interpreter.")
    run_parser.add_argument(
        "--model-profile",
        type=Path,
        help=(
            "Strict model-stack-profile-v1 YAML. Official APIs, relays, and "
            "local OpenAI-compatible servers use the same experiment path."
        ),
    )
    run_parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the redacted native command without executing it.",
    )

    catalog_parser = subparsers.add_parser(
        "index-runs",
        help="Rebuild runs/index.json, runs/catalog.csv, and runs/README.md.",
    )
    catalog_parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    catalog_parser.add_argument("--json", action="store_true")

    plot_parser = subparsers.add_parser(
        "plot-results",
        help="Generate standard figures from HA/IS-Bench predict-only aggregate CSVs.",
    )
    plot_parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Aggregate CSV file(s), or run directories containing them.",
    )
    plot_parser.add_argument("--output-dir", type=Path, default=None)
    plot_parser.add_argument("--title", default="World Model Safety: Predict-only Results")
    plot_parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
    )
    plot_parser.add_argument("--label", help="Custom label when exactly one CSV is resolved.")
    plot_parser.add_argument("--overwrite", action="store_true")
    plot_parser.add_argument(
        "--allow-nonformal-input",
        action="store_true",
        help="Allow legacy/partial inputs for explicitly diagnostic plots.",
    )
    plot_parser.add_argument("--json", action="store_true")
    plot_parser.add_argument("--python", help="Override the plotting Python interpreter.")
    plot_parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the plotting command without executing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    launcher_args, forwarded = _split_forwarded(raw)
    args = build_parser().parse_args(launcher_args)
    if args.command != "run" and forwarded:
        raise SystemExit("Only 'run' accepts arguments after --")

    if args.command == "list":
        code = _list_workflows(args.json)
    elif args.command == "doctor":
        code = _doctor(args.json)
    elif args.command == "index-runs":
        result = write_catalog(args.runs_root)
        if args.json:
            print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
        else:
            print(f"Indexed {result['run_count']} run directories")
            print(result["json"])
            print(result["csv"])
            print(result["readme"])
        code = 0
    elif args.command == "plot-results":
        code = _run_plot_results(args)
    else:
        code = _run_workflow(args, forwarded)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
