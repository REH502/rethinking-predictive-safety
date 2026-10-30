"""Canonical workflow registry used by the top-level experiment launcher."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = PROJECT_ROOT / "harness_env" / "orgin" / "HarnessAudit"
ISBENCH_ROOT = PROJECT_ROOT / "IS-Bench"
SAFEPRED_ROOT = PROJECT_ROOT / "SafePred"
RUNS_ROOT = PROJECT_ROOT / "runs"
CREDENTIALS_FILE_ENV = "WM_SAFETY_ENV_FILE"
DEFAULT_CREDENTIALS_FILE = PROJECT_ROOT / "config" / "experiment.env"

def _runtime_python(*, environment_name: str, legacy_path: str) -> Path:
    override = os.environ.get(environment_name)
    if override:
        return Path(override).expanduser()
    portable = PROJECT_ROOT / ".venvs" / (
        "isbench" if environment_name == "WM_SAFETY_ISBENCH_PYTHON" else "zhliu"
    ) / "bin" / "python"
    if portable.is_file():
        return portable
    legacy = Path(legacy_path)
    return legacy if legacy.is_file() else Path(sys.executable)


DEFAULT_HA_PYTHON = _runtime_python(
    environment_name="WM_SAFETY_HA_PYTHON",
    legacy_path="/home/zhliu/Environment/miniconda3/envs/zhliu/bin/python",
)
DEFAULT_ISBENCH_PYTHON = _runtime_python(
    environment_name="WM_SAFETY_ISBENCH_PYTHON",
    legacy_path="/home/zhliu/Environment/miniconda3/envs/isbench/bin/python",
)


@dataclass(frozen=True)
class Workflow:
    name: str
    benchmark: str
    stage: str
    module: str
    cwd: Path
    python: Path
    python_paths: tuple[Path, ...]
    description: str
    requires_gpu: bool = False


WORKFLOWS: dict[str, Workflow] = {
    "ha-trace": Workflow(
        name="ha-trace",
        benchmark="HarnessAudit",
        stage="trace generation",
        module="multi_agent.cli",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT),
        description="Run a native HarnessAudit agent task and emit a clean trace.",
    ),
    "ha-offline": Workflow(
        name="ha-offline",
        benchmark="HarnessAudit",
        stage="offline prediction",
        module="multi_agent.safepred_experiments.predict_only_pipeline",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT),
        description="Replay immutable HarnessAudit traces for SafePred prediction evaluation.",
    ),
    "ha-online": Workflow(
        name="ha-online",
        benchmark="HarnessAudit",
        stage="online safeguard",
        module="multi_agent.safepred_experiments.run_matrix",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT),
        description=(
            "Run the HarnessAudit baseline/SafePred matrix, including the "
            "optional four-condition distributed-enforced WM-return perturbation arm."
        ),
    ),
    "isbench-trace": Workflow(
        name="isbench-trace",
        benchmark="IS-Bench",
        stage="trace generation",
        module="og_ego_prim.cli.online_benchmark_all",
        cwd=ISBENCH_ROOT,
        python=DEFAULT_ISBENCH_PYTHON,
        python_paths=(ISBENCH_ROOT, HARNESS_ROOT, SAFEPRED_ROOT),
        description="Generate native IS-Bench simulator reports without SafePred intervention.",
        requires_gpu=True,
    ),
    "isbench-awareness": Workflow(
        name="isbench-awareness",
        benchmark="IS-Bench",
        stage="diagnostic safety-awareness evaluation",
        module="og_ego_prim.cli.online_benchmark_all",
        cwd=ISBENCH_ROOT,
        python=DEFAULT_ISBENCH_PYTHON,
        python_paths=(PROJECT_ROOT, ISBENCH_ROOT, HARNESS_ROOT, SAFEPRED_ROOT),
        description=(
            "Generate and judge IS-Bench safety awareness without executing "
            "environment actions."
        ),
        requires_gpu=True,
    ),
    "isbench-offline": Workflow(
        name="isbench-offline",
        benchmark="IS-Bench",
        stage="offline prediction",
        module="multi_agent.safepred_experiments.isbench_predict_only_pipeline",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description="Evaluate SafePred predictions on immutable IS-Bench reports.",
    ),
    "isbench-online": Workflow(
        name="isbench-online",
        benchmark="IS-Bench",
        stage="online safeguard",
        module="multi_agent.safepred_experiments.isbench_online_matrix",
        cwd=HARNESS_ROOT,
        python=DEFAULT_ISBENCH_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Run the IS-Bench baseline/SafePred matrix, including the optional "
            "four-condition distributed-enforced WM-return perturbation arm."
        ),
        requires_gpu=True,
    ),
    "isbench-terminal-postflight": Workflow(
        name="isbench-terminal-postflight",
        benchmark="IS-Bench",
        stage="terminal formal audit",
        module="experiments.isbench_formal_terminal_postflight",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Independently rehash the completed registered IS-Bench denominator, "
            "validate lifecycle/provenance, and run the terminal credential scan."
        ),
    ),
    "ha-terminal-postflight": Workflow(
        name="ha-terminal-postflight",
        benchmark="HarnessAudit",
        stage="terminal formal audit",
        module="experiments.ha_formal_terminal_postflight",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT),
        description=(
            "Independently rehash the completed registered HarnessAudit "
            "denominator, validate lifecycle/provenance, and run the terminal "
            "credential scan."
        ),
    ),
    "faithfulness-postflight": Workflow(
        name="faithfulness-postflight",
        benchmark="combined",
        stage="confirmatory postflight",
        module=(
            "multi_agent.safepred_experiments."
            "decision_faithfulness_postflight"
        ),
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Join blinded Decision Faithfulness adjudications with "
            "multiplicity-adjusted paired endpoint evidence."
        ),
    ),
    "faithfulness-multiplicity": Workflow(
        name="faithfulness-multiplicity",
        benchmark="combined",
        stage="confirmatory multiplicity analysis",
        module="experiments.decision_faithfulness_multiplicity",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Generate jointly adjusted paired endpoint evidence for HA or IS-Bench."
        ),
    ),
    "formal-readiness": Workflow(
        name="formal-readiness",
        benchmark="combined",
        stage="joint pre-execution gate",
        module="experiments.formal_readiness",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Fail closed unless one frozen study is ready for both HA and IS-Bench."
        ),
    ),
    "freeze-formal-preregistration": Workflow(
        name="freeze-formal-preregistration",
        benchmark="combined",
        stage="joint preregistration freeze",
        module="experiments.freeze_formal_preregistration",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT, HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Bind one draft to the exact model stack, code, and protocol identities."
        ),
    ),
    "ha-analyze": Workflow(
        name="ha-analyze",
        benchmark="HarnessAudit",
        stage="analysis",
        module="multi_agent.safepred_experiments.analysis",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT),
        description="Recompute HarnessAudit metrics and publication-style figures from artifacts.",
    ),
    "ha-tier3": Workflow(
        name="ha-tier3",
        benchmark="HarnessAudit",
        stage="optional Tier-3 judge",
        module="multi_agent.safepred_experiments.run_sparse_tier3_judge",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT),
        description="Run the optional semantic consequence judge on HA prediction rows.",
    ),
    "isbench-audit": Workflow(
        name="isbench-audit",
        benchmark="IS-Bench",
        stage="input validation",
        module="multi_agent.safepred_experiments.isbench_report_audit",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description="Validate and enumerate replayable IS-Bench report.json inputs.",
    ),
    "trace-postflight": Workflow(
        name="trace-postflight",
        benchmark="combined",
        stage="joint trace-source validation",
        module="multi_agent.safepred_experiments.trace_source_postflight",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Validate complete actor-seed trace cohorts and reject identical "
            "cross-seed action trajectories before world-model calls."
        ),
    ),
    "offline-postflight": Workflow(
        name="offline-postflight",
        benchmark="combined",
        stage="joint offline replicate validation",
        module=(
            "multi_agent.safepred_experiments."
            "offline_replicate_postflight"
        ),
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description=(
            "Validate and combine the complete preregistered predict-only "
            "replicate set before formal plotting."
        ),
    ),
    "isbench-tier3": Workflow(
        name="isbench-tier3",
        benchmark="IS-Bench",
        stage="optional Tier-3 judge",
        module="multi_agent.safepred_experiments.isbench_tier3_judge",
        cwd=HARNESS_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(HARNESS_ROOT, SAFEPRED_ROOT, ISBENCH_ROOT),
        description="Run the optional semantic consequence judge on IS-Bench predictions.",
    ),
    "plot-results": Workflow(
        name="plot-results",
        benchmark="combined",
        stage="predict-only plotting",
        module="experiments.plot_results",
        cwd=PROJECT_ROOT,
        python=DEFAULT_HA_PYTHON,
        python_paths=(PROJECT_ROOT,),
        description="Generate one standard plot bundle from HA/IS-Bench predict-only aggregates.",
    ),
}


def load_experiment_credentials(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment augmented from the private workspace credentials file.

    Existing process values win, so schedulers and CI can inject short-lived
    credentials without being silently replaced by the persistent local file.
    """

    env = dict(os.environ if environ is None else environ)
    credentials_path = Path(
        env.get(CREDENTIALS_FILE_ENV) or DEFAULT_CREDENTIALS_FILE
    ).expanduser().resolve()
    env[CREDENTIALS_FILE_ENV] = str(credentials_path)
    if not credentials_path.exists():
        return env
    if not credentials_path.is_file():
        raise ValueError(f"Credentials path is not a file: {credentials_path}")
    if credentials_path.stat().st_mode & 0o077:
        raise PermissionError(
            f"Credentials file is accessible by group or other users: "
            f"{credentials_path}. Run: chmod 600 {credentials_path}"
        )
    for name, value in dotenv_values(
        dotenv_path=credentials_path,
    ).items():
        if value is not None:
            env.setdefault(str(name), str(value))
    return env


def workflow_environment(workflow: Workflow) -> dict[str, str]:
    env = load_experiment_credentials()
    existing = env.get("PYTHONPATH", "")
    prefixes = [str(path) for path in workflow.python_paths]
    if existing:
        prefixes.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(prefixes)
    env["PYTHONUNBUFFERED"] = "1"
    # User-site packages must not override the benchmark's pinned environment.
    # This is especially important for OmniGibson's compiled numpy/h5py stack.
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    loopback_hosts = ("127.0.0.1", "localhost")
    existing_no_proxy = env.get("NO_PROXY") or env.get("no_proxy", "")
    no_proxy_entries = [entry.strip() for entry in existing_no_proxy.split(",") if entry.strip()]
    for host in loopback_hosts:
        if host not in no_proxy_entries:
            no_proxy_entries.append(host)
    no_proxy = ",".join(no_proxy_entries)
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env
