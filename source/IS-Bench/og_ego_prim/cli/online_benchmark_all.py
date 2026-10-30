import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import datetime
from functools import wraps
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

try:
    import fcntl
except ImportError:  # pragma: no cover - formal generation runs on Linux/HPC.
    fcntl = None

from tqdm import tqdm

from og_ego_prim.utils.constants import CAMERAS, SCENES, TASKS
from og_ego_prim.utils.metric import Metric, read_benchmark_report
from og_ego_prim.utils.action_progress import run_supervised
from og_ego_prim.utils.task_failure import (
    _active_credential_values,
    _redact_active_credentials,
)
from og_ego_prim.utils.model_request_options import (
    model_specific_extra_body,
    request_extra_body_disables_thinking,
)
from og_ego_prim.utils.trace_provenance import (
    actor_transport_max_retries,
    actor_transport_retry_base_delay_seconds,
    actor_transport_retry_policy,
    build_generation_protocol,
    protocol_sha256,
    resolve_openai_api_base,
    sha256_file,
    strict_http_endpoint_error,
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument('--action-resume', action='store_true',
                    help='Persist and restore native high-level action transactions (opt-in diagnostic contract)')
parser.add_argument('--task_list', type=str, default=None)
parser.add_argument('--not_use_default_scene_model', action='store_true')
parser.add_argument('--data_parallel', type=int, default=1)
parser.add_argument('--online_object_sampling', type=bool, default=None)
parser.add_argument('--num_retry', type=int, default=3)
parser.add_argument('--task_timeout', type=int, default=None)
parser.add_argument('--local_llm_serve', action='store_true')
parser.add_argument('--local_serve_ip', type=str, default=None)
parser.add_argument('--actor_api_base', type=str, default=None)
parser.add_argument('--draw_bbox_2d', action='store_true')
parser.add_argument('--skip_viewer_obs', action='store_true')
parser.add_argument('--use_initial_setup', action='store_true')
parser.add_argument('--use_self_caption', action='store_true')
parser.add_argument('--safepred_safeguard', action='store_true')
parser.add_argument('--safepred_max_regenerations', type=int, default=None)
parser.add_argument('--safepred_policy_mode', choices=['generic_embodied_policy', 'train_policy', 'full_isbench_policy_text'], default=None)
parser.add_argument('--safepred_intervention_mode', choices=['prediction_only', 'advisory_only', 'enforced_guard', 'distributed_enforced'], default=None)
parser.add_argument('--safepred_fail_open', action='store_true')
parser.add_argument('--safepred_model', type=str, default=None)
parser.add_argument('--safepred_base_url', type=str, default=None)
parser.add_argument('--safepred_risk_threshold', type=float, default=None)
parser.add_argument('--safepred_max_images', type=int, default=None)
parser.add_argument('--safepred_disable_response_format', action='store_true')
parser.add_argument('--prompt_setting', type=str, default='default')
parser.add_argument('--strict_trace', '--strict-trace', action='store_true')
parser.add_argument('--online_strict', '--online-strict', action='store_true')
parser.add_argument(
    '--resume',
    action='store_true',
    help=(
        'Reuse complete, provenance-compatible terminal task reports and run '
        'only tasks without a terminal report.'
    ),
)
parser.add_argument(
    '--experiment_seed',
    type=int,
    default=(
        int(os.environ['ISBENCH_EXPERIMENT_SEED'])
        if 'ISBENCH_EXPERIMENT_SEED' in os.environ
        else None
    ),
)
parser.add_argument(
    '--actor_temperature',
    type=float,
    default=float(os.environ.get('ISBENCH_ACTOR_TEMPERATURE', '0.0')),
)
parser.add_argument(
    '--actor_top_p',
    type=float,
    default=float(os.environ.get('ISBENCH_ACTOR_TOP_P', '1.0')),
)
parser.add_argument(
    '--actor_max_completion_tokens',
    type=int,
    default=int(os.environ.get('ISBENCH_AGENT_MAX_COMPLETION_TOKENS', '512')),
)
parser.add_argument(
    '--actor_model_revision',
    default=os.environ.get('ISBENCH_ACTOR_MODEL_REVISION', ''),
)
parser.add_argument(
    '--actor_runtime_revision',
    default=os.environ.get('ISBENCH_ACTOR_RUNTIME_REVISION', ''),
)

parser.add_argument('--not_eval_process_safety', action='store_true')
parser.add_argument('--not_eval_termination_safety', action='store_true')
parser.add_argument('--not_eval_awareness', action='store_true')
parser.add_argument('--not_eval_execution', action='store_true')

def get_time_tag() -> str:
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_model_tag(model: str) -> str:
    return model.replace('/', '__') if model is not None else 'example'


def get_report_path(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
) -> str:
    benchmark_tag = f'{task_name}___{scene_name}'
    return os.path.join(
        work_dir,
        'benchmark',
        benchmark_tag,
        get_model_tag(model),
        'report.json',
    )


def get_awareness_report_path(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
) -> str:
    benchmark_tag = f'{task_name}___{scene_name}'
    return os.path.join(
        work_dir,
        'benchmark',
        benchmark_tag,
        get_model_tag(model),
        'report_awareness.json',
    )


def _awareness_only_mode(
    *,
    not_eval_process_safety: bool,
    not_eval_termination_safety: bool,
    not_eval_awareness: bool,
    not_eval_execution: bool,
) -> bool:
    return (
        bool(not_eval_process_safety)
        and bool(not_eval_termination_safety)
        and not bool(not_eval_awareness)
        and bool(not_eval_execution)
    )


def _awareness_report_validation(
    report_path: str | Path,
    *,
    task_name: str,
    scene_name: str,
    model: str,
    expected_experiment_seed: int,
    expected_generation_protocol: dict,
) -> tuple[bool, str]:
    """Validate a completed awareness-only artifact before promotion/reuse."""

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable awareness report: {type(exc).__name__}: {exc}"
    if not isinstance(report, dict):
        return False, "awareness report root must be an object"
    for field, expected in (
        ("task", task_name),
        ("scene", scene_name),
        ("model", model),
    ):
        if report.get(field) != expected:
            return False, f"{field}={report.get(field)!r} expected={expected!r}"

    awareness = report.get("awareness")
    if not isinstance(awareness, dict):
        return False, "awareness must be an object"
    content = awareness.get("content")
    if not isinstance(content, str) or not content.strip():
        return False, "awareness.content must be a non-empty string"
    results = awareness.get("eval_results")
    if not isinstance(results, list) or not results:
        return False, "awareness.eval_results must be a non-empty list"
    try:
        task_config = json.loads(
            (Path(TASKS) / f"{task_name}.json").read_text(encoding="utf-8")
        )
        expected_caution_count = len(task_config.get("evaluation_cautions") or [])
        goal_conditions = task_config.get("evaluation_goal_conditions") or {}
        expected_caution_count += len(
            goal_conditions.get("process_safety_goal_condition") or []
        )
        expected_caution_count += len(
            goal_conditions.get("termination_safety_goal_condition") or []
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return False, f"cannot resolve expected caution count: {exc}"
    if len(results) != expected_caution_count:
        return False, (
            f"awareness judge returned {len(results)} rows; "
            f"expected {expected_caution_count}"
        )
    for index, row in enumerate(results):
        if not isinstance(row, dict) or not isinstance(row.get("eval"), bool):
            return False, f"awareness.eval_results[{index}].eval must be boolean"

    metadata = report.get("experiment_metadata")
    if not isinstance(metadata, dict):
        return False, "experiment_metadata must be an object"
    if metadata.get("recorded_before_actor_call") is not True:
        return False, "experiment metadata was not recorded before the actor call"
    if metadata.get("experiment_seed") != int(expected_experiment_seed):
        return False, "experiment_seed mismatch"
    if metadata.get("generation_protocol") != expected_generation_protocol:
        return False, "generation_protocol mismatch"
    if metadata.get("generation_protocol_sha256") != protocol_sha256(
        expected_generation_protocol
    ):
        return False, "generation_protocol_sha256 mismatch"
    return True, ""


def _ingest_awareness_report_metrics(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
    metric: Metric,
) -> None:
    report_path = get_awareness_report_path(
        task_name, scene_name, model, work_dir
    )
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    results = report["awareness"]["eval_results"]
    metric.num_pred_cautions += sum(
        1 for row in results if row["eval"] is True
    )
    metric.num_total_cautions += len(results)
    metric.num_tasks += 1


def _first_physical_gpu_id(value):
    if value is None:
        return None
    candidate = str(value).split(",", 1)[0].strip()
    if ":" in candidate:
        candidate = candidate.rsplit(":", 1)[-1].strip()
    return candidate if candidate.isdigit() else None


def _normalize_simulator_environment(environment, *, worker_slot=None):
    """Keep SLURM's logical CUDA mapping coherent with OmniGibson.

    SLURM commonly exposes a physical allocation (``SLURM_JOB_GPUS=4``) while
    remapping that device to logical CUDA ordinal 0.  Clearing
    ``CUDA_VISIBLE_DEVICES`` in that situation makes Isaac's renderer probe
    an unavailable GPU0.  An explicit ``ISBENCH_SIM_GPU_ID`` remains a
    physical-device override and therefore clears the inherited remapping.
    """

    normalized = dict(environment)
    explicit_gpu_id = _first_physical_gpu_id(
        normalized.get("ISBENCH_SIM_GPU_ID")
    )
    if explicit_gpu_id is not None:
        normalized.pop("CUDA_VISIBLE_DEVICES", None)
        normalized["OMNIGIBSON_GPU_ID"] = explicit_gpu_id
        return normalized

    visible_devices = [
        item.strip()
        for item in str(normalized.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if item.strip()
    ]
    if worker_slot is not None and len(visible_devices) > 1:
        if type(worker_slot) is not int or worker_slot < 0:
            raise ValueError("worker_slot must be a non-negative exact integer")
        # Keep the complete Slurm-visible allocation in every child.  Isaac's
        # Vulkan renderer enumerates that allocation independently; narrowing
        # CUDA_VISIBLE_DEVICES per child remaps CUDA to logical zero while
        # Vulkan still selects a different allocation index.  Select the same
        # logical ordinal for OmniGibson's renderer and physics instead.
        normalized["OMNIGIBSON_GPU_ID"] = str(
            worker_slot % len(visible_devices)
        )
        return normalized

    logical_gpu_id = _first_physical_gpu_id(
        normalized.get("CUDA_VISIBLE_DEVICES")
    )
    if logical_gpu_id is not None:
        normalized["OMNIGIBSON_GPU_ID"] = logical_gpu_id
        return normalized

    physical_gpu_id = _first_physical_gpu_id(normalized.get("SLURM_JOB_GPUS"))
    if physical_gpu_id is not None:
        normalized["OMNIGIBSON_GPU_ID"] = physical_gpu_id
    return normalized


def _ingest_report_metrics(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
    metric: Metric,
) -> None:
    placement_failure = read_benchmark_report(
        task_name,
        scene_name,
        model,
        work_dir,
        metric,
    )
    if placement_failure:
        metric.failure_placement.append((task_name, scene_name))
        metric.num_safe_success_evaluable_tasks += 1
    metric.num_tasks += 1


def _seed_scoped_work_dir(work_dir: str | Path, experiment_seed: int) -> Path:
    root = Path(work_dir).expanduser().resolve()
    tag = f"actor-seed-{int(experiment_seed)}"
    return root if root.name == tag else root / tag


def _strict_trace_enabled() -> bool:
    return os.environ.get("ISBENCH_TRACE_STRICT", "0").lower() in {
        "1", "true", "yes",
    }


CLEAN_STRICT_TRACE_PROFILE = "clean_strict_trace_v1"
ONLINE_STRICT_PROFILE = "online_strict_v1"


def _online_strict_enabled() -> bool:
    return os.environ.get("ISBENCH_ONLINE_STRICT", "0").lower() in {
        "1", "true", "yes",
    }


def _strict_generation_profile() -> str | None:
    clean = _strict_trace_enabled()
    online = _online_strict_enabled()
    if clean and online:
        raise ValueError(
            "ISBENCH_TRACE_STRICT and ISBENCH_ONLINE_STRICT are mutually exclusive"
        )
    if clean:
        return CLEAN_STRICT_TRACE_PROFILE
    if online:
        return ONLINE_STRICT_PROFILE
    return None


def _strict_generation_enabled() -> bool:
    return _strict_generation_profile() is not None


def _strict_report_validation(
    report_path: str | Path,
    *,
    task_name: str,
    scene_name: str,
    expected_experiment_seed: int,
    expected_generation_protocol: dict,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    allow_action_errors: bool = False,
) -> tuple[bool, str]:
    try:
        from multi_agent.safepred_experiments.isbench_report_audit import audit_reports

        expected_intervention_mode = None
        expected_policy_mode = None
        if generation_validation_profile == ONLINE_STRICT_PROFILE:
            expected_intervention_mode = str(
                os.environ.get("ISBENCH_ONLINE_STRICT_INTERVENTION_MODE") or ""
            ).strip()
            expected_policy_mode = str(
                os.environ.get("ISBENCH_ONLINE_STRICT_POLICY_MODE") or ""
            ).strip() or None

        audit = audit_reports(
            [Path(report_path)],
            recommended_count=0,
            tasks_dir=Path(TASKS),
            require_complete_safety_labels=True,
            require_explicit_process_timing=True,
            require_image_quality=True,
            require_successful=False,
            require_generation_provenance=True,
            expected_experiment_seed=expected_experiment_seed,
            expected_generation_protocol=expected_generation_protocol,
            generation_validation_profile=generation_validation_profile,
            expected_intervention_mode=expected_intervention_mode,
            expected_policy_mode=expected_policy_mode,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    reports = audit.get("reports") or []
    if len(reports) != 1:
        return False, f"strict audit found {len(reports)} reports, expected 1"
    row = reports[0]
    identity_matches = (
        row.get("task") == task_name and row.get("scene") == scene_name
    )
    if allow_action_errors:
        # Resume is an idempotence operation, not a success filter.  A complete
        # action-error trajectory is a terminal scientific outcome and must
        # remain in the denominator instead of being sampled again.
        profile_eligible = (
            row.get("is_structurally_replayable")
            and not row.get("has_infrastructure_errors")
        )
    else:
        profile_eligible = (
            row.get("is_eligible")
            if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
            else (
                row.get("is_structurally_replayable")
                and not row.get("has_infrastructure_errors")
            )
        )
    if profile_eligible and identity_matches:
        return True, ""
    details = [
        f"task={row.get('task')!r} expected={task_name!r}",
        f"scene={row.get('scene')!r} expected={scene_name!r}",
        f"termination={row.get('termination_reason')!r}",
    ]
    for field in (
        "parse_error",
        "safety_label_errors",
        "image_quality_errors",
        "generation_provenance_errors",
        "action_error_types",
        "infrastructure_error_types",
    ):
        if row.get(field):
            details.append(f"{field}={row[field]}")
    return False, "; ".join(details)


def _promote_attempt_report(
    *,
    staged_report_path: str | Path,
    final_report_path: str | Path,
) -> None:
    """Atomically replace one task output with a complete isolated attempt."""

    staged_path = Path(staged_report_path).resolve()
    staged_dir = staged_path.parent
    final_dir = Path(final_report_path).resolve().parent
    if not staged_path.is_file():
        raise FileNotFoundError(
            f"attempt did not produce {staged_path.name}: {staged_dir}"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = staged_dir.with_name(
        f".{final_dir.name}.previous-{os.getpid()}-{time.time_ns()}"
    )
    had_previous = final_dir.exists() or final_dir.is_symlink()
    if had_previous:
        os.replace(final_dir, backup_dir)
    try:
        os.replace(staged_dir, final_dir)
    except Exception:
        if had_previous and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    if had_previous and backup_dir.exists():
        if backup_dir.is_dir() and not backup_dir.is_symlink():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.unlink()


def _atomic_write_json(path: str | Path, payload: dict, *, root: str | Path) -> None:
    """Durably replace one JSON artifact without exposing a truncated target."""

    destination = Path(path).resolve()
    allowed_root = Path(root).resolve()
    try:
        destination.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"atomic JSON destination escapes its registered root: {destination}"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=4, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _redact_file_credentials(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        return
    original = path.read_bytes()
    redacted = _redact_active_credentials(original)
    if redacted == original:
        return
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".redacting",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(redacted)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _redact_tree_credentials(root: Path) -> None:
    """Sanitize generated attempt artifacts before validation or promotion."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"credential-redaction root is not a real directory: {root}")
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and not artifact.is_symlink():
            _redact_file_credentials(artifact)


def _diagnostic_tag(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    ).strip("._")
    return normalized or "unnamed"


def _preserve_failed_attempt(
    *,
    attempt_root: str | Path,
    work_dir: str | Path,
    task_name: str,
    scene_name: str,
    model: str,
    retry: int,
    returncode: int,
    stage: str,
    detail: str,
    log_file: str | Path,
) -> Path:
    """Atomically retain one failed attempt without creating discoverable reports."""

    source = Path(attempt_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"failed attempt directory is missing: {source}")
    work_path = Path(work_dir).resolve()
    diagnostic_parent = (
        work_path
        / "failed_attempts"
        / f"{_diagnostic_tag(task_name)}___{_diagnostic_tag(scene_name)}"
        / _diagnostic_tag(get_model_tag(model))
    )
    diagnostic_parent.mkdir(parents=True, exist_ok=True)
    destination = diagnostic_parent / f"retry-{int(retry)}-{time.time_ns()}"

    for report_path in sorted(source.rglob("report.json")):
        if report_path.is_symlink() or not report_path.is_file():
            continue
        failed_name = report_path.with_name("report.failed.json")
        if failed_name.exists():
            raise FileExistsError(
                f"failed-attempt report destination already exists: {failed_name}"
            )
        os.replace(report_path, failed_name)

    log_path = Path(log_file).resolve()
    if log_path.is_file() and not log_path.is_symlink():
        copied_log = source / "worker.log"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=source,
            prefix=".worker.log.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_log = Path(handle.name)
            handle.write(log_path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary_log, copied_log)
        finally:
            temporary_log.unlink(missing_ok=True)

    _redact_tree_credentials(source)
    if log_path.is_file() and not log_path.is_symlink():
        _redact_file_credentials(log_path)

    inventory = []
    for artifact in sorted(source.rglob("*")):
        if artifact.is_file() and not artifact.is_symlink():
            inventory.append(
                {
                    "path": artifact.relative_to(source).as_posix(),
                    "size": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            )
    record = {
        "schema_version": 1,
        "artifact_type": "isbench_failed_attempt_terminal",
        "task": task_name,
        "scene": scene_name,
        "model": model,
        "retry": int(retry),
        "returncode": int(returncode),
        "stage": stage,
        "detail": _redact_active_credentials(str(detail).encode("utf-8")).decode(
            "utf-8", errors="replace"
        ),
        "partial_artifacts": inventory,
    }
    original_failure_path = source / "task_failure.json"
    if original_failure_path.is_file():
        try:
            original_failure = json.loads(original_failure_path.read_text())
            if original_failure.get("artifact_type") == "isbench_worker_failure":
                record["original_error_type"] = original_failure.get("error_type")
                record["original_failure_stage"] = original_failure.get("stage")
                record["http_status"] = original_failure.get("http_status")
        except (OSError, ValueError, AttributeError):
            record["original_failure_unreadable"] = True
    _atomic_write_json(source / "terminal_failure.json", record, root=source)
    os.replace(source, destination)
    directory_fd = os.open(diagnostic_parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination / "terminal_failure.json"


def _worker_log_path(
    *,
    work_dir: str | Path,
    benchmark_tag: str,
    model_tag: str,
    retry: int,
) -> Path:
    log_dir = Path(work_dir).resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    seed = os.environ.get("ISBENCH_EXPERIMENT_SEED", "unseeded")
    return log_dir / (
        f"benchmark_{benchmark_tag}_{model_tag}_seed-{seed}_retry-{int(retry)}_"
        f"pid-{os.getpid()}_{time.time_ns()}.log"
    )


def _attempt_parent(work_path: Path) -> Path:
    """Keep strict attempts outside the user-visible trace discovery root."""

    resolved = Path(work_path).resolve()
    if _strict_generation_enabled() and resolved.name.startswith("actor-seed-"):
        return resolved.parent.parent
    return resolved.parent


def _attempt_prefix(work_path: Path) -> str:
    resolved = Path(work_path).resolve()
    return (
        f".{resolved.parent.name}-{resolved.name}-"
        "isbench-attempt-"
    )


def _recover_interrupted_attempts(
    *,
    work_dir: str | Path,
    all_tasks: List[Tuple[str, str]],
    model: str,
) -> list[dict]:
    """Preserve orphan worker sandboxes before an idempotent resume.

    A hard scheduler stop can kill the batch between worker sandbox creation
    and terminal promotion.  These sandboxes are never treated as completed
    reports.  They are moved into the failure ledger before the missing task is
    allowed to run again, so the interrupted attempt remains auditable.
    """

    work_path = Path(work_dir).resolve()
    attempt_parent = _attempt_parent(work_path)
    if not attempt_parent.is_dir():
        return []
    expected_tasks = set(all_tasks)
    recovered = []
    for attempt_root in sorted(attempt_parent.glob(f"{_attempt_prefix(work_path)}*")):
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError(
                f"interrupted attempt is not a real directory: {attempt_root}"
            )
        identity_path = attempt_root / "attempt_identity.json"
        identified = False
        task_name = "__unidentified__"
        scene_name = "unknown"
        attempt_model = model
        retry = 0
        if identity_path.is_file() and not identity_path.is_symlink():
            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f"Cannot read interrupted attempt identity {identity_path}: {exc}"
                )
            else:
                candidate_task = identity.get("task")
                candidate_scene = identity.get("scene")
                candidate_model = identity.get("model")
                candidate_retry = identity.get("retry")
                if (
                    identity.get("schema_version") != 1
                    or identity.get("artifact_type")
                    != "isbench_worker_attempt_identity"
                    or identity.get("work_dir") != str(work_path)
                    or (candidate_task, candidate_scene) not in expected_tasks
                    or candidate_model != model
                    or isinstance(candidate_retry, bool)
                    or not isinstance(candidate_retry, int)
                    or candidate_retry < 0
                ):
                    raise ValueError(
                        "interrupted attempt identity does not match the current "
                        f"task cohort: {identity_path}"
                    )
                task_name = candidate_task
                scene_name = candidate_scene
                attempt_model = candidate_model
                retry = candidate_retry
                identified = True
        if not identified:
            warnings.warn(
                "Preserving a legacy interrupted attempt without task identity; "
                f"the missing terminal task remains eligible for resume: {attempt_root}"
            )
        terminal_path = _preserve_failed_attempt(
            attempt_root=attempt_root,
            work_dir=work_path,
            task_name=task_name,
            scene_name=scene_name,
            model=attempt_model,
            retry=retry,
            returncode=130,
            stage="interrupted_before_resume",
            detail=(
                "worker sandbox existed without a promoted terminal report when "
                "the native resume acquired the work root"
            ),
            log_file=attempt_root / "__missing_worker.log",
        )
        recovered.append(
            {
                "task": task_name,
                "scene": scene_name,
                "identified": identified,
                "terminal_path": str(terminal_path),
            }
        )
    return recovered


def existing_report_is_replayable(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
    *,
    expected_experiment_seed: int,
    expected_generation_protocol: dict,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    resume_enabled: bool | None = None,
) -> bool:
    if resume_enabled is None:
        resume_enabled = os.environ.get(
            "ISBENCH_RESUME_VALID_REPORTS", "0"
        ).lower() in {"1", "true", "yes"}
    if not resume_enabled:
        return False
    report_path = get_report_path(task_name, scene_name, model, work_dir)
    if not os.path.isfile(report_path):
        return False
    valid, reason = _strict_report_validation(
        report_path,
        task_name=task_name,
        scene_name=scene_name,
        expected_experiment_seed=expected_experiment_seed,
        expected_generation_protocol=expected_generation_protocol,
        generation_validation_profile=generation_validation_profile,
        allow_action_errors=True,
    )
    if not valid:
        warnings.warn(f'Cannot reuse existing report "{report_path}": {reason}')
    return valid


def existing_awareness_report_is_replayable(
    task_name: str,
    scene_name: str,
    model: str,
    work_dir: str,
    *,
    expected_experiment_seed: int,
    expected_generation_protocol: dict,
    resume_enabled: bool | None = None,
) -> bool:
    if resume_enabled is None:
        resume_enabled = os.environ.get(
            "ISBENCH_RESUME_VALID_REPORTS", "0"
        ).lower() in {"1", "true", "yes"}
    if not resume_enabled:
        return False
    report_path = get_awareness_report_path(
        task_name, scene_name, model, work_dir
    )
    if not os.path.isfile(report_path):
        return False
    valid, reason = _awareness_report_validation(
        report_path,
        task_name=task_name,
        scene_name=scene_name,
        model=model,
        expected_experiment_seed=expected_experiment_seed,
        expected_generation_protocol=expected_generation_protocol,
    )
    if not valid:
        warnings.warn(
            f'Cannot reuse existing awareness report "{report_path}": {reason}'
        )
    return valid


def expected_generation_protocol(
    *,
    task_name: str,
    model: str,
    local_llm_serve: bool,
    local_serve_ip: str,
    actor_api_base: str | None = None,
    prompt_setting: str,
    draw_bbox_2d: bool,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    online_object_sampling: bool,
    not_eval_process_safety: bool,
    not_eval_termination_safety: bool,
    not_eval_awareness: bool,
    not_eval_execution: bool,
) -> dict:
    return build_generation_protocol(
        task_config_path=Path(TASKS) / f"{task_name}.json",
        camera_config_path=Path(CAMERAS) / "camera.json",
        package_root=Path(__file__).resolve().parents[1],
        model_requested=model,
        model_resolved=model,
        provider_mode="local" if local_llm_serve else "close_source",
        model_revision=os.environ.get("ISBENCH_ACTOR_MODEL_REVISION", ""),
        api_base=(
            local_serve_ip
            if local_llm_serve
            else resolve_openai_api_base(actor_api_base)
        ),
        runtime_revision=os.environ.get("ISBENCH_ACTOR_RUNTIME_REVISION", ""),
        prompt_setting=prompt_setting,
        actor_temperature=float(
            os.environ.get("ISBENCH_ACTOR_TEMPERATURE", "0.0")
        ),
        actor_top_p=(
            float(os.environ.get("ISBENCH_ACTOR_TOP_P", "1.0"))
            if local_llm_serve
            else 1.0
        ),
        actor_max_completion_tokens=(
            int(os.environ.get("ISBENCH_AGENT_MAX_COMPLETION_TOKENS", "512"))
            if local_llm_serve
            else 512
        ),
        actor_max_images=int(os.environ.get("ISBENCH_ACTOR_MAX_IMAGES", "0")),
        actor_retry_seed_policy=(
            "base_plus_validation_attempt_v1"
            if local_llm_serve
            else "none"
        ),
        actor_transport_retry_policy=actor_transport_retry_policy(),
        actor_transport_max_retries=actor_transport_max_retries(),
        actor_transport_retry_base_delay_seconds=(
            actor_transport_retry_base_delay_seconds()
        ),
        disable_thinking=request_extra_body_disables_thinking(
            model_specific_extra_body(model, role="actor")
        ),
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        draw_bbox_2d=draw_bbox_2d,
        skip_viewer_obs=skip_viewer_obs,
        online_object_sampling=bool(online_object_sampling),
        eval_process_safety=not not_eval_process_safety,
        eval_termination_safety=not not_eval_termination_safety,
        eval_awareness=not not_eval_awareness,
        eval_execution=not not_eval_execution,
    )


def _strict_scene_source_archive_binding() -> dict:
    data_root = Path(SCENES).expanduser().resolve().parent
    archive = data_root / "scenes.tar.gz"
    if archive.is_symlink() or not archive.is_file() or archive.resolve() != archive:
        raise ValueError(f"scene source archive is missing or not a regular file: {archive}")
    return {
        "relative_path": str(archive.relative_to(data_root)),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
    }


def _strict_scene_template_binding(task_name: str, scene_name: str) -> dict:
    """Resolve one exact fixed-scene template without importing OmniGibson."""

    if not isinstance(task_name, str) or not isinstance(scene_name, str):
        raise ValueError("scene-template task and scene ids must be strings")
    task_id = task_name
    selected_scene = scene_name
    if (
        not task_id
        or task_id != task_id.strip()
        or task_id in {".", ".."}
        or Path(task_id).name != task_id
        or "\\" in task_id
    ):
        raise ValueError(f"invalid scene-template task id: {task_name!r}")
    if (
        not selected_scene
        or selected_scene != selected_scene.strip()
        or selected_scene in {".", ".."}
        or Path(selected_scene).name != selected_scene
        or "\\" in selected_scene
    ):
        raise ValueError(f"invalid scene-template scene id: {scene_name!r}")

    task_path = Path(TASKS).expanduser().resolve() / f"{task_id}.json"
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load scene-template task config {task_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(task, dict):
        raise ValueError(f"scene-template task config is not an object: {task_path}")
    task_info = task.get("task_info")
    scene_info = task.get("scene_info")
    if not isinstance(task_info, dict) or not isinstance(scene_info, dict):
        raise ValueError(f"scene-template task config is incomplete: {task_path}")
    declared_task = task_info.get("task_name")
    if declared_task != task_id:
        raise ValueError(
            f"scene-template task identity is ambiguous: file={task_id!r} "
            f"declared={declared_task!r}"
        )
    default_scene = scene_info.get("default_scene_model")
    scene_models = scene_info.get("scene_models")
    if (
        not isinstance(default_scene, str)
        or default_scene != selected_scene
        or not isinstance(scene_models, list)
        or scene_models.count(default_scene) != 1
    ):
        raise ValueError(
            f"scene-template default scene is missing or ambiguous for task "
            f"{task_id!r}: selected={selected_scene!r} default={default_scene!r}"
        )
    if scene_info.get("online_object_sampling") is not False:
        raise ValueError(
            f"scene-template task must set online_object_sampling=false: {task_path}"
        )
    activity_definition_id = task_info.get("activity_definition_id")
    activity_instance_id = task_info.get("activity_instance_id")
    for field, value in (
        ("activity_definition_id", activity_definition_id),
        ("activity_instance_id", activity_instance_id),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"scene-template {field} must be a non-negative integer: {task_path}"
            )

    expected_name = (
        f"{selected_scene}_task_{task_id}_{activity_definition_id}_"
        f"{activity_instance_id}_template.json"
    )
    scenes_root = Path(SCENES).expanduser().resolve()
    template_dir = scenes_root / selected_scene / "json"
    if not template_dir.is_dir():
        raise ValueError(
            f"scene template is missing for task={task_id!r} scene={selected_scene!r}: "
            f"{template_dir / expected_name}"
        )
    matches = sorted(
        path
        for path in template_dir.iterdir()
        if path.name.casefold() == expected_name.casefold()
    )
    if not matches:
        raise ValueError(
            f"scene template is missing for task={task_id!r} scene={selected_scene!r}: "
            f"{template_dir / expected_name}"
        )
    if len(matches) != 1 or matches[0].name != expected_name:
        raise ValueError(
            f"scene template is ambiguous for task={task_id!r} scene={selected_scene!r}: "
            f"{[path.name for path in matches]}"
        )
    template = matches[0]
    if (
        template.is_symlink()
        or not template.is_file()
        or template.resolve() != template
    ):
        raise ValueError(f"scene template is not a regular file: {template}")
    try:
        template_payload = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"scene template is invalid JSON: {template}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(template_payload, dict):
        raise ValueError(f"scene template is not a JSON object: {template}")
    return {
        "task": task_id,
        "scene": selected_scene,
        "activity_definition_id": activity_definition_id,
        "activity_instance_id": activity_instance_id,
        "relative_path": str(template.relative_to(scenes_root.parent)),
        "size_bytes": template.stat().st_size,
        "sha256": sha256_file(template),
    }


def _strict_scene_template_bindings(
    all_tasks: List[Tuple[str, str]],
) -> dict:
    templates = [
        _strict_scene_template_binding(task_name, scene_name)
        for task_name, scene_name in all_tasks
    ]
    paths = [row["relative_path"] for row in templates]
    if len(paths) != len(set(paths)):
        raise ValueError("scene template census resolves multiple tasks to one template")
    archive = Path(SCENES).expanduser().resolve().parent / "scenes.tar.gz"
    required_members = set(paths)
    archived_hashes: dict[str, list[str]] = {
        member_name: [] for member_name in required_members
    }
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                member_name = member.name.removeprefix("./")
                if member_name not in required_members:
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"scene source archive member is not a regular file: {member_name}"
                    )
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"cannot read scene source archive member: {member_name}"
                    )
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                archived_hashes[member_name].append(digest.hexdigest())
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(
            f"cannot validate scene source archive {archive}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for row in templates:
        member_name = row["relative_path"]
        member_hashes = archived_hashes[member_name]
        if not member_hashes:
            raise ValueError(
                f"scene source archive member is missing: {member_name}"
            )
        if len(member_hashes) != 1:
            raise ValueError(
                f"scene source archive member is ambiguous: {member_name}"
            )
        if member_hashes[0] != row["sha256"]:
            raise ValueError(
                f"scene template differs from source archive member: {member_name}"
            )
        row["source_archive_member_sha256"] = member_hashes[0]
    payload = {
        "scene_template_count": len(templates),
        "source_archive": _strict_scene_source_archive_binding(),
        "templates": templates,
    }
    payload["bindings_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _strict_cohort_manifest(
    *,
    task_list: str,
    all_tasks: List[Tuple[str, str]],
    experiment_seed: int,
    model: str,
    local_llm_serve: bool,
    local_serve_ip: str,
    actor_api_base: str | None = None,
    prompt_setting: str,
    draw_bbox_2d: bool,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    online_object_sampling: bool,
    not_eval_process_safety: bool,
    not_eval_termination_safety: bool,
    not_eval_awareness: bool,
    not_eval_execution: bool,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
    safepred_safeguard: bool = False,
) -> dict:
    """Bind one seed root to one exact task/scene/protocol cohort."""

    rows = []
    for task_name, scene_name in all_tasks:
        protocol = expected_generation_protocol(
            task_name=task_name,
            model=model,
            local_llm_serve=local_llm_serve,
            local_serve_ip=local_serve_ip,
            actor_api_base=actor_api_base,
            prompt_setting=prompt_setting,
            draw_bbox_2d=draw_bbox_2d,
            skip_viewer_obs=skip_viewer_obs,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            online_object_sampling=online_object_sampling,
            not_eval_process_safety=not_eval_process_safety,
            not_eval_termination_safety=not_eval_termination_safety,
            not_eval_awareness=not_eval_awareness,
            not_eval_execution=not_eval_execution,
        )
        rows.append(
            {
                "task": task_name,
                "scene": scene_name,
                "generation_protocol_sha256": protocol_sha256(protocol),
            }
        )
    scene_templates = _strict_scene_template_bindings(all_tasks)
    payload = {
        "schema_version": 1,
        "artifact_type": (
            "isbench_strict_trace_cohort"
            if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
            else "isbench_online_strict_cohort"
        ),
        "generation_validation_profile": generation_validation_profile,
        "experiment_seed": int(experiment_seed),
        "task_list_sha256": sha256_file(Path(task_list).expanduser().resolve()),
        "tasks": rows,
        "scene_templates": scene_templates,
    }
    if generation_validation_profile == ONLINE_STRICT_PROFILE:
        payload["online_condition"] = {
            "safepred_safeguard": bool(safepred_safeguard),
            "policy_mode": (
                str(os.environ.get("ISBENCH_ONLINE_STRICT_POLICY_MODE") or "")
                or None
            ),
            "intervention_mode": str(
                os.environ.get("ISBENCH_ONLINE_STRICT_INTERVENTION_MODE") or ""
            ),
        }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _expected_cohort_report_paths(
    *,
    work_dir: str | Path,
    all_tasks: List[Tuple[str, str]],
    model: str,
) -> set[Path]:
    return {
        Path(get_report_path(task, scene, model, str(work_dir))).resolve()
        for task, scene in all_tasks
    }


def _bind_strict_cohort_root(
    *,
    work_dir: str | Path,
    manifest: dict,
    expected_reports: set[Path],
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
) -> Path:
    """Create/verify the immutable cohort contract and reject stale reports."""

    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / (
        "trace_cohort_manifest.json"
        if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
        else "online_strict_cohort_manifest.json"
    )
    discovered_reports = {path.resolve() for path in root.rglob("report.json")}
    unexpected = sorted(discovered_reports - expected_reports)
    if unexpected:
        raise ValueError(
            "Strict ISBench seed root contains report(s) outside the requested "
            f"cohort: {[str(path) for path in unexpected[:10]]}"
        )
    if manifest_path.exists():
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read strict trace cohort manifest {manifest_path}: {exc}"
            ) from exc
        if observed != manifest:
            raise ValueError(
                "Strict ISBench seed root is already bound to a different "
                f"task/scene/protocol cohort: {manifest_path}"
            )
        return manifest_path
    if discovered_reports:
        raise ValueError(
            "Strict ISBench seed root contains reports but no cohort manifest; "
            f"refusing legacy/stale resume: {root}"
        )
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        # A concurrent launcher may win between our existence check and
        # promotion.  Never overwrite its cohort declaration.
        os.link(temporary, manifest_path)
    except FileExistsError:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise ValueError(
                "Concurrent strict launcher bound the seed root to a different "
                f"cohort: {manifest_path}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def _acquire_strict_root_lock(
    work_dir: str | Path,
    *,
    generation_validation_profile: str = CLEAN_STRICT_TRACE_PROFILE,
):
    """Prevent two batch launchers from generating into one seed root."""

    if fcntl is None:
        raise RuntimeError(
            "Strict trace generation requires POSIX fcntl locking; run the "
            "formal generator in the supported Linux/HPC environment"
        )
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / (
        ".strict_trace_generation.lock"
        if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
        else ".online_strict_generation.lock"
    )
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another strict trace launcher is active for seed root: {root}"
        ) from exc
    return handle


def _acquire_resume_root_lock(work_dir: str | Path):
    """Reject concurrent native writers for the same diagnostic task root."""

    if fcntl is None:
        raise RuntimeError(
            "Native ISBench resume requires POSIX fcntl locking; run it in the "
            "supported Linux/HPC environment"
        )
    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / ".native_resume.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another native benchmark/resume launcher is active for work root: {root}"
        ) from exc
    return handle


def _strict_seed_root_locked(function):
    """Scope and lock the formal seed root with guaranteed release."""

    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        generation_validation_profile = _strict_generation_profile()
        if (
            generation_validation_profile is None
            or "ISBENCH_EXPERIMENT_SEED" not in os.environ
        ):
            # Non-formal runs previously had no writer exclusion, allowing a
            # resume to race the still-live original process.  Use the same
            # root lock for fresh and resumed native batches.
            lock_handle = _acquire_resume_root_lock(bound.arguments["work_dir"])
            try:
                return function(*bound.args, **bound.kwargs)
            finally:
                assert fcntl is not None
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
        _validate_strict_trace_batch(
            model=bound.arguments["model"],
            local_llm_serve=bound.arguments["local_llm_serve"],
            local_serve_ip=bound.arguments["local_serve_ip"],
            actor_api_base=bound.arguments["actor_api_base"],
            task_list=bound.arguments["task_list"],
            use_default_scene_model=bound.arguments["use_default_scene_model"],
            prompt_setting=bound.arguments["prompt_setting"],
            skip_viewer_obs=bound.arguments["skip_viewer_obs"],
            use_initial_setup=bound.arguments["use_initial_setup"],
            use_self_caption=bound.arguments["use_self_caption"],
            safepred_safeguard=bound.arguments["safepred_safeguard"],
            online_object_sampling=bound.arguments["online_object_sampling"],
            not_eval_process_safety=bound.arguments["not_eval_process_safety"],
            not_eval_termination_safety=bound.arguments[
                "not_eval_termination_safety"
            ],
            not_eval_awareness=bound.arguments["not_eval_awareness"],
            not_eval_execution=bound.arguments["not_eval_execution"],
        )
        scoped_work_dir = (
            _seed_scoped_work_dir(
                bound.arguments["work_dir"],
                int(os.environ["ISBENCH_EXPERIMENT_SEED"]),
            )
            if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
            else Path(bound.arguments["work_dir"]).expanduser().resolve()
        )
        bound.arguments["work_dir"] = str(scoped_work_dir)
        lock_handle = _acquire_strict_root_lock(
            scoped_work_dir,
            generation_validation_profile=generation_validation_profile,
        )
        print(
            f'[{get_time_tag()}][strict] profile={generation_validation_profile} '
            f'work_dir={scoped_work_dir}'
        )
        try:
            return function(*bound.args, **bound.kwargs)
        finally:
            assert fcntl is not None
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    return wrapped


def _validate_strict_trace_batch(
    *,
    model: str,
    local_llm_serve: bool,
    local_serve_ip: str,
    actor_api_base: str | None = None,
    task_list: str,
    use_default_scene_model: bool,
    prompt_setting: str,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    safepred_safeguard: bool,
    online_object_sampling: bool,
    not_eval_process_safety: bool,
    not_eval_termination_safety: bool,
    not_eval_awareness: bool,
    not_eval_execution: bool,
) -> None:
    generation_validation_profile = _strict_generation_profile()
    if generation_validation_profile is None:
        return
    errors = []
    if "ISBENCH_EXPERIMENT_SEED" not in os.environ:
        errors.append("ISBENCH_EXPERIMENT_SEED must be explicit")
    elif int(os.environ["ISBENCH_EXPERIMENT_SEED"]) < 0:
        errors.append("ISBENCH_EXPERIMENT_SEED must be non-negative")
    if not str(model or "").strip():
        errors.append("model must be explicit")
    if str(model or "").strip().lower() == "local":
        errors.append(
            "model must be the immutable served model id, not the 'local' alias"
        )
    if "gemini_direct" in str(model or "").lower():
        errors.append(
            "gemini_direct bypasses the recorded OpenAI-compatible endpoint and "
            "is not supported for strict trace generation"
        )
    if local_llm_serve:
        endpoint = urlsplit(str(local_serve_ip or "").strip())
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or bool(endpoint.query)
            or bool(endpoint.fragment)
        ):
            errors.append(
                "local_serve_ip must be an absolute HTTP(S) URL without "
                "credentials, query, or fragment"
            )
    else:
        resolved_actor_api_base = resolve_openai_api_base(actor_api_base)
        if not resolved_actor_api_base:
            errors.append(
                "actor_api_base is required; pass --actor_api_base or set "
                "OPENAI_API_BASE/OPENAI_BASE_URL"
            )
        else:
            endpoint_error = strict_http_endpoint_error(
                resolved_actor_api_base,
                field_name="actor_api_base",
            )
            if endpoint_error:
                errors.append(endpoint_error)
    if not str(os.environ.get("ISBENCH_ACTOR_MODEL_REVISION") or "").strip():
        errors.append("ISBENCH_ACTOR_MODEL_REVISION must be explicit")
    if not str(os.environ.get("ISBENCH_ACTOR_RUNTIME_REVISION") or "").strip():
        errors.append("ISBENCH_ACTOR_RUNTIME_REVISION must be explicit")
    temperature = float(os.environ.get("ISBENCH_ACTOR_TEMPERATURE", "0.0"))
    if local_llm_serve:
        if not 0 < temperature <= 2:
            errors.append("ISBENCH_ACTOR_TEMPERATURE must be in (0, 2]")
    elif temperature != 0.0:
        errors.append(
            "ISBENCH_ACTOR_TEMPERATURE must be exactly 0.0 for the "
            "upstream-compatible close-source profile"
        )
    top_p = float(os.environ.get("ISBENCH_ACTOR_TOP_P", "1.0"))
    if local_llm_serve:
        if not 0 < top_p <= 1:
            errors.append("ISBENCH_ACTOR_TOP_P must be in (0, 1]")
    elif top_p != 1.0:
        errors.append(
            "ISBENCH_ACTOR_TOP_P must be exactly 1.0 for the "
            "upstream-compatible close-source profile"
        )
    max_tokens = int(
        os.environ.get("ISBENCH_AGENT_MAX_COMPLETION_TOKENS", "512")
    )
    if local_llm_serve:
        if max_tokens < 1:
            errors.append(
                "ISBENCH_AGENT_MAX_COMPLETION_TOKENS must be positive"
            )
    elif max_tokens != 512:
        errors.append(
            "ISBENCH_AGENT_MAX_COMPLETION_TOKENS must be exactly 512 for the "
            "upstream-compatible close-source profile"
        )
    if prompt_setting != "v1":
        errors.append("prompt_setting must be 'v1'")
    if skip_viewer_obs:
        errors.append("skip_viewer_obs is forbidden")
    if use_initial_setup or use_self_caption:
        errors.append("initial-setup/self-caption actor inputs are forbidden")
    if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE:
        if safepred_safeguard:
            errors.append("SafePred safeguard must be disabled")
    else:
        errors.extend(
            _online_strict_condition_errors(
                safepred_safeguard=safepred_safeguard,
            )
        )
    if online_object_sampling:
        errors.append("online_object_sampling is forbidden")
    if (
        not_eval_process_safety
        or not_eval_termination_safety
        or not_eval_execution
    ):
        errors.append("process, termination, and execution evaluators are required")
    if not not_eval_awareness:
        errors.append("awareness evaluation must be disabled for clean v1 traces")
    if not use_default_scene_model:
        errors.append("non-default scene models are forbidden for the fixed cohort")
    errors.extend(_strict_task_list_errors(task_list))
    if errors:
        label = (
            "Strict ISBench trace"
            if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
            else "ISBench online-strict"
        )
        raise ValueError(f"{label} request is invalid: " + "; ".join(errors))


def _online_strict_condition_errors(*, safepred_safeguard: bool) -> list[str]:
    errors: list[str] = []
    expected_intervention = str(
        os.environ.get("ISBENCH_ONLINE_STRICT_INTERVENTION_MODE") or ""
    ).strip()
    expected_policy = str(
        os.environ.get("ISBENCH_ONLINE_STRICT_POLICY_MODE") or ""
    ).strip()
    guarded_modes = {
        "prediction_only",
        "advisory_only",
        "enforced_guard",
        "distributed_enforced",
    }
    if expected_intervention not in {"no_guard", *guarded_modes}:
        errors.append(
            "ISBENCH_ONLINE_STRICT_INTERVENTION_MODE must identify the assigned arm"
        )
        return errors
    if expected_intervention == "no_guard":
        if safepred_safeguard:
            errors.append("online-strict no_guard must not enable SafePred safeguard")
        if expected_policy:
            errors.append("online-strict no_guard must not bind a policy mode")
        if str(os.environ.get("ISBENCH_SAFEPRED_INTERVENTION_MODE") or "").strip():
            errors.append("online-strict no_guard inherited a safeguard intervention mode")
        if str(os.environ.get("ISBENCH_SAFEPRED_POLICY_MODE") or "").strip():
            errors.append("online-strict no_guard inherited a safeguard policy mode")
        return errors

    if not safepred_safeguard:
        errors.append(
            f"online-strict {expected_intervention} must enable SafePred safeguard"
        )
    if expected_policy not in {
        "generic_embodied_policy",
        "train_policy",
        "full_isbench_policy_text",
    }:
        errors.append("ISBENCH_ONLINE_STRICT_POLICY_MODE is missing or unsupported")
    if str(os.environ.get("ISBENCH_SAFEPRED_INTERVENTION_MODE") or "").strip() != expected_intervention:
        errors.append("online-strict safeguard intervention mode differs from assignment")
    if str(os.environ.get("ISBENCH_SAFEPRED_POLICY_MODE") or "").strip() != expected_policy:
        errors.append("online-strict safeguard policy mode differs from assignment")
    if os.environ.get("ISBENCH_SAFEPRED_FAIL_OPEN", "0").lower() in {
        "1", "true", "yes",
    }:
        errors.append("online-strict safeguard must fail closed")
    model = str(os.environ.get("ISBENCH_SAFEPRED_MODEL") or "").strip()
    if not model or model.lower() == "local":
        errors.append("ISBENCH_SAFEPRED_MODEL must be an explicit served model id")
    revision = str(
        os.environ.get("ISBENCH_SAFEPRED_MODEL_REVISION")
        or os.environ.get("WM_MODEL_REVISION")
        or ""
    ).strip()
    if not revision:
        errors.append("ISBENCH_SAFEPRED_MODEL_REVISION must be explicit")
    endpoint_error = strict_http_endpoint_error(
        str(os.environ.get("ISBENCH_SAFEPRED_BASE_URL") or ""),
        field_name="ISBENCH_SAFEPRED_BASE_URL",
    )
    if endpoint_error:
        errors.append(endpoint_error)
    return errors


def _strict_task_list_errors(task_list: str | None) -> list[str]:
    if not task_list:
        return ["task_list must be explicit"]
    path = Path(task_list).expanduser()
    if not path.is_file():
        return [f"task_list must be an existing file: {path}"]
    try:
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return [f"cannot read task_list {path}: {type(exc).__name__}: {exc}"]
    errors: list[str] = []
    if not raw_lines:
        return [f"task_list is empty: {path}"]
    blank_lines = [index for index, line in enumerate(raw_lines, start=1) if not line.strip()]
    if blank_lines:
        errors.append(f"task_list has blank lines: {blank_lines}")
    normalized = [
        value if value.endswith(".json") else f"{value}.json"
        for value in (line.strip() for line in raw_lines)
        if value
    ]
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        errors.append(f"task_list has duplicate tasks: {duplicates}")
    supported = {
        item
        for item in os.listdir(TASKS)
        if item.endswith(".json") and "test" not in item
    }
    unknown = sorted(set(normalized) - supported)
    if unknown:
        errors.append(f"task_list has unknown tasks: {unknown}")
    if not normalized:
        errors.append(f"task_list has no tasks: {path}")
    try:
        with (Path(CAMERAS) / "camera.json").open("r", encoding="utf-8") as handle:
            camera_config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            f"cannot load camera config: {type(exc).__name__}: {exc}"
        )
        camera_config = {}
    try:
        _strict_scene_source_archive_binding()
    except ValueError as exc:
        errors.append(str(exc))
    declared_task_ids: list[str] = []
    for filename in normalized:
        if filename not in supported:
            continue
        task_path = Path(TASKS) / filename
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"cannot load task config {task_path}: {type(exc).__name__}: {exc}"
            )
            continue
        task_info = task.get("task_info") if isinstance(task, dict) else None
        task_id = str(
            task_info.get("task_name") if isinstance(task_info, dict) else ""
        ).strip()
        if not task_id:
            errors.append(f"task config lacks task_info.task_name: {task_path}")
        else:
            declared_task_ids.append(task_id)
        expected_task_id = Path(filename).stem
        if task_id != expected_task_id:
            errors.append(
                "task config task_info.task_name differs from its filename: "
                f"{task_path}"
            )
            continue
        scene_info = task.get("scene_info") if isinstance(task, dict) else None
        if not isinstance(scene_info, dict):
            errors.append(f"task config lacks scene_info: {task_path}")
            continue
        scene = str(scene_info.get("default_scene_model") or "").strip()
        scene_models = scene_info.get("scene_models")
        if not scene or not isinstance(scene_models, list) or scene not in scene_models:
            errors.append(f"task config has invalid default_scene_model: {task_path}")
            continue
        room = str(scene_info.get("room") or "").strip()
        if not room or f"{room}__{scene}" not in camera_config:
            errors.append(
                f"task default scene lacks camera configuration: {task_path}"
            )
        try:
            _strict_scene_template_binding(task_id, scene)
        except ValueError as exc:
            errors.append(str(exc))
    duplicate_ids = sorted(
        {item for item in declared_task_ids if declared_task_ids.count(item) > 1}
    )
    if duplicate_ids:
        errors.append(f"task_list resolves to duplicate task ids: {duplicate_ids}")
    return errors


def get_all_tasks(
    task_list: Optional[str] = None, 
    use_default_scene_model: bool = True,
) -> List[Tuple[str, str]]:

    all_supported_tasks = [task for task in os.listdir(TASKS) 
                                if task.endswith('.json') and not 'test' in task]
    if task_list is None:
        task_list = all_supported_tasks
    elif not os.path.exists(task_list):
        warnings.warn(f'args.task_list "{task_list}" not exists, using all tasks under "data/tasks".')
        task_list = all_supported_tasks
    else:
        with open(task_list, 'r') as f:
            task_list = [task.strip().lstrip('\ufeff') for task in f.readlines() if task.strip()]
        task_list = list(map(lambda task: task if task.endswith('.json') else f'{task}.json', task_list))
        task_list = list(filter(lambda task: task in all_supported_tasks, task_list))

    with open(os.path.join(CAMERAS, 'camera.json'), 'r') as f:
        camera_config = json.load(f)
    
    print(f"[INFO]######\n{task_list}")
    all_tasks = []
    for task in task_list:
        with open(os.path.join(TASKS, task), 'r') as f:
            task_config = json.load(f)
        task_name = task_config['task_info']['task_name']
        scene_info = task_config['scene_info']

        # prepare scene_models
        if use_default_scene_model:
            if 'default_scene_model' not in scene_info:
                warnings.warn(f'task "{task_name}" has not key "default_scene_model", skip')
                continue
            scene_model = scene_info['default_scene_model']

            if scene_model not in scene_info['scene_models']:
                warnings.warn(f'default_scene_model "{scene_model}" not supported in task "{task_name}", skip')
                continue
            scene_models = [scene_model]
        else:
            scene_models = scene_info['scene_models']
        
        # check camera_config
        for scene_model in scene_models:
            room = f'{scene_info["room"]}__{scene_model}'
            if room not in camera_config:
                warnings.warn(f'room "{room}" in scene_model "{scene_model}" has no camera config, skip')
                continue

            all_tasks.append((task_name, scene_model))

    print(f'[{get_time_tag()}][tasks] Totally {len(all_tasks)} requests')
    return all_tasks


def get_launcher(
    task_name: str, 
    scene_name: str, 
    model: str = None, 
    work_dir: str = None,
    online_object_sampling: bool=None,
    draw_bbox_2d: bool=None,
    skip_viewer_obs: bool=None,
    use_initial_setup: bool=None,
    use_self_caption: bool=None,
    safepred_safeguard: bool=None,
    safepred_max_regenerations: int=None,
    local_llm_serve: bool=None,
    local_serve_ip: str=None,
    prompt_setting: str=None,
    not_eval_process_safety: bool=None,
    not_eval_termination_safety: bool=None,
    not_eval_awareness: bool=None,
    not_eval_execution: bool=None,
    actor_api_base: str=None,
) -> List[str]:
    entrypoint = []

    if 'PARTITION' in os.environ:
        partition = os.environ.get('PARTITION', None)
        assert partition is not None
        num_gpu_per_task = os.environ.get('NUM_GPUS', 1)
        entrypoint.extend([
            'srun', '-p', partition,
            # '--debug',
            f'--gres=gpu:{num_gpu_per_task}',
            '-N', f'{num_gpu_per_task}'
        ])

    if 'APPTAINER_IMAGE' in os.environ:
        entrypoint.extend(['apptainer', 'run', '--nv'])
        path_binding = os.environ.get('BINDING')
        if path_binding:
            entrypoint.extend(['--bind', path_binding])
        image = os.environ['APPTAINER_IMAGE']
        entrypoint.append(image)
    
    entrypoint.extend([
        sys.executable, '-m', 'og_ego_prim.cli.online_benchmark_once',
        '--task', task_name, 
        '--scene', scene_name,
    ])
    if model is not None:
        entrypoint.extend(['--model', model])
    if work_dir is not None:
        entrypoint.extend(['--work_dir', work_dir])
    if online_object_sampling is not None and online_object_sampling:
        entrypoint.extend(['--online_object_sampling', 'True'])
    if draw_bbox_2d is not None and draw_bbox_2d:
        entrypoint.extend(['--draw_bbox_2d'])
    if skip_viewer_obs is not None and skip_viewer_obs:
        entrypoint.extend(['--skip_viewer_obs'])
    if use_initial_setup is not None and use_initial_setup:
        entrypoint.extend(['--use_initial_setup'])
    if use_self_caption is not None and use_self_caption:
        entrypoint.extend(['--use_self_caption'])
    if safepred_safeguard is not None and safepred_safeguard:
        entrypoint.extend(['--safepred_safeguard'])
    if safepred_max_regenerations is not None:
        entrypoint.extend(['--safepred_max_regenerations', str(safepred_max_regenerations)])
    if local_llm_serve is not None and local_llm_serve:
        entrypoint.extend(['--local_llm_serve'])
    if local_serve_ip is not None:
        entrypoint.extend(['--local_serve_ip', local_serve_ip])
    if actor_api_base is not None:
        entrypoint.extend(['--actor_api_base', actor_api_base])
    if prompt_setting is not None:
        entrypoint.extend(['--prompt_setting', prompt_setting])
    if not_eval_process_safety is not None and not_eval_process_safety:
        entrypoint.extend(['--not_eval_process_safety'])
    if not_eval_termination_safety is not None and not_eval_termination_safety:
        entrypoint.extend(['--not_eval_termination_safety'])
    if not_eval_awareness is not None and not_eval_awareness:
        entrypoint.extend(['--not_eval_awareness'])
    if not_eval_execution is not None and not_eval_execution:
        entrypoint.extend(['--not_eval_execution'])

    return entrypoint


def worker(task_name: str, scene_name: str, model: str, work_dir: str, online_object_sampling: bool, retry: int, draw_bbox_2d: bool, skip_viewer_obs: bool, use_initial_setup: bool, use_self_caption: bool, safepred_safeguard: bool, safepred_max_regenerations: int, local_llm_serve: bool, local_serve_ip: str, prompt_setting: str, not_eval_process_safety: bool, not_eval_termination_safety: bool, not_eval_awareness: bool, not_eval_execution: bool, task_timeout: int, actor_api_base: str=None):
    # Opt-in operational boundary: existing frozen callers keep their limit.
    action_timeout = int(os.environ.get("ISBENCH_ACTION_STALL_TIMEOUT_SECONDS", "0"))
    if action_timeout < 0:
        raise ValueError("ISBENCH_ACTION_STALL_TIMEOUT_SECONDS must be non-negative")
    action_timeout_mode = os.environ.get("ISBENCH_ACTION_TIMEOUT_MODE", "absolute")
    if action_timeout_mode not in {"absolute", "no_progress"}:
        raise ValueError("ISBENCH_ACTION_TIMEOUT_MODE must be absolute or no_progress")
    startup_timeout = int(os.environ.get("ISBENCH_SIM_STARTUP_TIMEOUT_SECONDS", "0"))
    if startup_timeout < 0:
        raise ValueError("ISBENCH_SIM_STARTUP_TIMEOUT_SECONDS must be non-negative")
    worker_id = multiprocessing.current_process()._identity[0]
    time.sleep(worker_id * 0.5)
    print(f'[{get_time_tag()}][worker_{worker_id}] Processing "{task_name}___{scene_name}"')

    benchmark_tag = f'{task_name}___{scene_name}'
    model_tag = get_model_tag(model)
    log_file = _worker_log_path(
        work_dir=work_dir,
        benchmark_tag=benchmark_tag,
        model_tag=model_tag,
        retry=retry,
    )

    work_path = Path(work_dir).resolve()
    attempt_parent = _attempt_parent(work_path)
    attempt_parent.mkdir(parents=True, exist_ok=True)
    attempt_root = Path(
        tempfile.mkdtemp(
            prefix=f"{_attempt_prefix(work_path)}{retry}-",
            dir=str(attempt_parent),
        )
    )
    _atomic_write_json(
        attempt_root / "attempt_identity.json",
        {
            "schema_version": 1,
            "artifact_type": "isbench_worker_attempt_identity",
            "task": task_name,
            "scene": scene_name,
            "model": model,
            "retry": int(retry),
            "work_dir": str(work_path),
            "action_timeout_seconds": action_timeout,
            "action_timeout_mode": action_timeout_mode,
            "startup_timeout_seconds": startup_timeout,
            "task_timeout_seconds": task_timeout,
            "created_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        },
        root=attempt_root,
    )
    launcher = get_launcher(
        task_name,
        scene_name,
        model,
        str(attempt_root),
        online_object_sampling,
        draw_bbox_2d,
        skip_viewer_obs,
        use_initial_setup,
        use_self_caption,
        safepred_safeguard,
        safepred_max_regenerations,
        local_llm_serve,
        local_serve_ip,
        prompt_setting,
        not_eval_process_safety,
        not_eval_termination_safety,
        not_eval_awareness,
        not_eval_execution,
        actor_api_base,
    )
    envs = _normalize_simulator_environment(
        os.environ.copy(), worker_slot=worker_id - 1
    )
    envs['OMNIGIBSON_HEADLESS'] = '1'
    if os.environ.get('ISBENCH_ACTION_RESUME', '0') == '1':
        # Stable across child attempt sandboxes, but isolated by condition/task.
        episode_id = hashlib.sha256(
            json.dumps([task_name, scene_name, model], ensure_ascii=False).encode('utf-8')
        ).hexdigest()
        checkpoint_root = Path(os.environ.get('ISBENCH_ACTION_CHECKPOINT_ROOT') or work_path / '.action_checkpoints')
        if not checkpoint_root.is_absolute():
            raise ValueError('persistent action checkpoint root must be absolute')
        envs['ISBENCH_ACTION_CHECKPOINT_DIR'] = str(checkpoint_root / episode_id)
    else:
        envs.pop('ISBENCH_ACTION_CHECKPOINT_DIR', None)

    awareness_only = _awareness_only_mode(
        not_eval_process_safety=not_eval_process_safety,
        not_eval_termination_safety=not_eval_termination_safety,
        not_eval_awareness=not_eval_awareness,
        not_eval_execution=not_eval_execution,
    )
    report_path_factory = (
        get_awareness_report_path if awareness_only else get_report_path
    )
    report_path = report_path_factory(task_name, scene_name, model, work_dir)
    staged_report_path = report_path_factory(
        task_name, scene_name, model, str(attempt_root)
    )
    progress_path = Path(staged_report_path).parent / "action_progress.json"
    envs["ISBENCH_ACTION_PROGRESS_PATH"] = str(progress_path)
    startup_state_path = attempt_root / "startup_state.json"
    if startup_timeout > 0:
        envs["ISBENCH_SIM_STARTUP_STATE_PATH"] = str(startup_state_path)
        envs["ISBENCH_SIM_RUNTIME_ROOT"] = str(attempt_root / "simulator_runtime")
    else:
        envs.pop("ISBENCH_SIM_STARTUP_STATE_PATH", None)
        envs.pop("ISBENCH_SIM_RUNTIME_ROOT", None)
    supervision_failure = None

    with open(log_file, 'w') as outfile:
        try:
            if action_timeout > 0 or startup_timeout > 0:
                result, supervision_failure = run_supervised(
                    launcher, env=envs, output=outfile,
                    task_timeout=task_timeout, action_timeout=action_timeout,
                    action_timeout_mode=action_timeout_mode,
                    progress_path=progress_path,
                    startup_timeout=startup_timeout,
                    startup_state_path=startup_state_path,
                )
                if supervision_failure is not None:
                    _atomic_write_json(
                        attempt_root / "supervision_failure.json",
                        supervision_failure, root=attempt_root,
                    )
                    outfile.write("\n[TIMEOUT] episode-local supervision limit reached; continuing.\n")
            else:
                result = subprocess.run(
                    launcher,
                    env=envs,
                    stdout=outfile,
                    stderr=outfile,
                    text=True,
                    check=False,
                    timeout=task_timeout,
                )
        except subprocess.TimeoutExpired:
            outfile.write(f"\n[TIMEOUT] task exceeded {task_timeout} seconds; marked failed and continuing.\n")
            result = subprocess.CompletedProcess(launcher, 124)

    _redact_tree_credentials(attempt_root)
    _redact_file_credentials(log_file)

    report_updated = False
    report_validation_error = ""
    promotion_error = None
    try:
        if result.returncode == 0 and os.path.isfile(staged_report_path):
            valid = True
            generation_validation_profile = _strict_generation_profile()
            protocol = expected_generation_protocol(
                task_name=task_name,
                model=model,
                local_llm_serve=local_llm_serve,
                local_serve_ip=local_serve_ip,
                actor_api_base=actor_api_base,
                prompt_setting=prompt_setting,
                draw_bbox_2d=draw_bbox_2d,
                skip_viewer_obs=skip_viewer_obs,
                use_initial_setup=use_initial_setup,
                use_self_caption=use_self_caption,
                online_object_sampling=online_object_sampling,
                not_eval_process_safety=not_eval_process_safety,
                not_eval_termination_safety=not_eval_termination_safety,
                not_eval_awareness=not_eval_awareness,
                not_eval_execution=not_eval_execution,
            )
            if awareness_only:
                valid, report_validation_error = _awareness_report_validation(
                    staged_report_path,
                    task_name=task_name,
                    scene_name=scene_name,
                    model=model,
                    expected_experiment_seed=int(
                        os.environ.get("ISBENCH_EXPERIMENT_SEED", "0")
                    ),
                    expected_generation_protocol=protocol,
                )
            elif generation_validation_profile is not None:
                valid, report_validation_error = _strict_report_validation(
                    staged_report_path,
                    task_name=task_name,
                    scene_name=scene_name,
                    expected_experiment_seed=int(
                        os.environ["ISBENCH_EXPERIMENT_SEED"]
                    ),
                    expected_generation_protocol=protocol,
                    generation_validation_profile=generation_validation_profile,
                )
            if valid:
                if startup_state_path.is_file():
                    # Retain the small startup handshake, not simulator caches.
                    shutil.copyfile(
                        startup_state_path,
                        Path(staged_report_path).parent / "startup_state.json",
                    )
                _promote_attempt_report(
                    staged_report_path=staged_report_path,
                    final_report_path=report_path,
                )
                report_updated = True
    except Exception as exc:
        promotion_error = exc
    if report_updated:
        shutil.rmtree(attempt_root, ignore_errors=True)
    else:
        if promotion_error is not None:
            failure_stage = "report_promotion"
            failure_detail = f"{type(promotion_error).__name__}: {promotion_error}"
        elif report_validation_error:
            failure_stage = "strict_report_validation"
            failure_detail = report_validation_error
        elif supervision_failure is not None:
            failure_stage = supervision_failure["stage"]
            if failure_stage == "primitive_action_timeout":
                failure_detail = f"primitive exceeded {action_timeout} seconds ({action_timeout_mode})"
            elif failure_stage == "simulator_startup_timeout":
                failure_detail = f"simulator startup exceeded {startup_timeout} seconds"
            else:
                failure_detail = f"task exceeded {task_timeout} seconds"
        elif result.returncode == 124:
            failure_stage = "subprocess_timeout"
            failure_detail = f"task exceeded {task_timeout} seconds"
        elif result.returncode != 0:
            failure_stage = "actor_subprocess"
            failure_detail = f"actor subprocess returned {result.returncode}"
        else:
            failure_stage = "missing_report"
            failure_detail = "actor returned success without a promotable report"
        _preserve_failed_attempt(
            attempt_root=attempt_root,
            work_dir=work_path,
            task_name=task_name,
            scene_name=scene_name,
            model=model,
            retry=retry,
            returncode=result.returncode,
            stage=failure_stage,
            detail=failure_detail,
            log_file=log_file,
        )
    if promotion_error is not None:
        raise promotion_error
    return (
        task_name,
        scene_name,
        result,
        retry,
        report_updated,
        report_validation_error,
    )


@_strict_seed_root_locked
def benchmark_all(
    model: str, 
    work_dir: str, 
    task_list: str,
    use_default_scene_model: bool,
    data_parallel: int,
    online_object_sampling: bool,
    num_retry: int,
    draw_bbox_2d: bool,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool, 
    safepred_safeguard: bool,
    safepred_max_regenerations: int,
    local_llm_serve: bool, 
    local_serve_ip: str,
    prompt_setting: str,
    not_eval_process_safety: bool, 
    not_eval_termination_safety: bool, 
    not_eval_awareness: bool, 
    not_eval_execution: bool,
    task_timeout: int,
    actor_api_base: str | None = None,
    resume: bool = False,
):
    _validate_strict_trace_batch(
        model=model,
        local_llm_serve=local_llm_serve,
        local_serve_ip=local_serve_ip,
        actor_api_base=actor_api_base,
        task_list=task_list,
        use_default_scene_model=use_default_scene_model,
        prompt_setting=prompt_setting,
        skip_viewer_obs=skip_viewer_obs,
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        safepred_safeguard=safepred_safeguard,
        online_object_sampling=online_object_sampling,
        not_eval_process_safety=not_eval_process_safety,
        not_eval_termination_safety=not_eval_termination_safety,
        not_eval_awareness=not_eval_awareness,
        not_eval_execution=not_eval_execution,
    )
    if data_parallel < 1:
        warnings.warn(f'data_parallel can not be set < 1, set to 1 by default.')
        data_parallel = 1

    resume_enabled = bool(resume) or os.environ.get(
        "ISBENCH_RESUME_VALID_REPORTS", "0"
    ).lower() in {"1", "true", "yes"}

    all_tasks = get_all_tasks(task_list, use_default_scene_model)
    recovered_interrupted_attempts = (
        _recover_interrupted_attempts(
            work_dir=work_dir,
            all_tasks=all_tasks,
            model=model,
        )
        if resume_enabled
        else []
    )
    awareness_only = _awareness_only_mode(
        not_eval_process_safety=not_eval_process_safety,
        not_eval_termination_safety=not_eval_termination_safety,
        not_eval_awareness=not_eval_awareness,
        not_eval_execution=not_eval_execution,
    )
    generation_validation_profile = _strict_generation_profile()
    if generation_validation_profile is not None:
        experiment_seed = int(os.environ["ISBENCH_EXPERIMENT_SEED"])
        expected_reports = _expected_cohort_report_paths(
            work_dir=work_dir,
            all_tasks=all_tasks,
            model=model,
        )
        cohort_manifest = _strict_cohort_manifest(
            task_list=task_list,
            all_tasks=all_tasks,
            experiment_seed=experiment_seed,
            model=model,
            local_llm_serve=local_llm_serve,
            local_serve_ip=local_serve_ip,
            actor_api_base=actor_api_base,
            prompt_setting=prompt_setting,
            draw_bbox_2d=draw_bbox_2d,
            skip_viewer_obs=skip_viewer_obs,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            online_object_sampling=online_object_sampling,
            not_eval_process_safety=not_eval_process_safety,
            not_eval_termination_safety=not_eval_termination_safety,
            not_eval_awareness=not_eval_awareness,
            not_eval_execution=not_eval_execution,
            generation_validation_profile=generation_validation_profile,
            safepred_safeguard=safepred_safeguard,
        )
        _bind_strict_cohort_root(
            work_dir=work_dir,
            manifest=cohort_manifest,
            expected_reports=expected_reports,
            generation_validation_profile=generation_validation_profile,
        )
    metric = Metric()
    unrecoverable_failures = []

    pending_tasks = []
    resumed_tasks = []
    resume_ingestion_failures = []
    experiment_seed = int(os.environ.get("ISBENCH_EXPERIMENT_SEED", "0"))
    for task_name, scene_name in all_tasks:
        expected_protocol = expected_generation_protocol(
            task_name=task_name,
            model=model,
            local_llm_serve=local_llm_serve,
            local_serve_ip=local_serve_ip,
            actor_api_base=actor_api_base,
            prompt_setting=prompt_setting,
            draw_bbox_2d=draw_bbox_2d,
            skip_viewer_obs=skip_viewer_obs,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            online_object_sampling=online_object_sampling,
            not_eval_process_safety=not_eval_process_safety,
            not_eval_termination_safety=not_eval_termination_safety,
            not_eval_awareness=not_eval_awareness,
            not_eval_execution=not_eval_execution,
        )
        replayable = (
            existing_awareness_report_is_replayable(
                task_name,
                scene_name,
                model,
                work_dir,
                expected_experiment_seed=experiment_seed,
                expected_generation_protocol=expected_protocol,
                resume_enabled=resume_enabled,
            )
            if awareness_only
            else existing_report_is_replayable(
                task_name,
                scene_name,
                model,
                work_dir,
                expected_experiment_seed=experiment_seed,
                expected_generation_protocol=expected_protocol,
                generation_validation_profile=(
                    generation_validation_profile
                    or CLEAN_STRICT_TRACE_PROFILE
                ),
                resume_enabled=resume_enabled,
            )
        )
        if replayable:
            try:
                if awareness_only:
                    _ingest_awareness_report_metrics(
                        task_name, scene_name, model, work_dir, metric
                    )
                else:
                    _ingest_report_metrics(
                        task_name, scene_name, model, work_dir, metric
                    )
            except Exception as exc:
                failure = {
                    'task': task_name,
                    'scene': scene_name,
                    'error': 'metric_ingestion_failure',
                    'detail': f'{type(exc).__name__}: {exc}',
                    'retryable': False,
                    'source': 'replayable_resume',
                }
                resume_ingestion_failures.append((task_name, scene_name))
                unrecoverable_failures.append(failure)
                warnings.warn(
                    f'Existing replayable report metric ingestion failed for '
                    f'"{task_name}"; actor will not be rerun: {exc}'
                )
            else:
                resumed_tasks.append((task_name, scene_name))
        else:
            pending_tasks.append((task_name, scene_name))
    if resumed_tasks:
        print(
            f'[{get_time_tag()}][resume] Reusing {len(resumed_tasks)} replayable reports; '
            f'{len(pending_tasks)} tasks remain.'
        )

    task_queue = deque([(task, 0) for task in pending_tasks])
    pbar = tqdm(
        total=len(all_tasks),
        initial=len(resumed_tasks) + len(resume_ingestion_failures),
        desc='IS-Bench',
    )

    with ProcessPoolExecutor(max_workers=data_parallel) as executor:
        while len(task_queue) > 0:
            dispatched = {}
            for i in range(len(task_queue)):
                task, retry = task_queue.popleft()
                task_name, scene_name = task
                future = executor.submit(
                    worker, 
                    task_name, 
                    scene_name, 
                    model, 
                    work_dir, 
                    online_object_sampling, 
                    retry, 
                    draw_bbox_2d,
                    skip_viewer_obs,
                    use_initial_setup,
                    use_self_caption, 
                    safepred_safeguard,
                    safepred_max_regenerations,
                    local_llm_serve, 
                    local_serve_ip, 
                    prompt_setting,
                    not_eval_process_safety, 
                    not_eval_termination_safety, 
                    not_eval_awareness,
                    not_eval_execution,
                    task_timeout,
                    actor_api_base,
                )
                dispatched[future] = (task, retry)

            for future in as_completed(dispatched):
                try:
                    (
                        task_name,
                        scene_name,
                        result,
                        retry,
                        report_updated,
                        report_validation_error,
                    ) = future.result()
                except Exception as exc:
                    (task_name, scene_name), retry = dispatched[future]
                    if retry < num_retry:
                        task_queue.append(((task_name, scene_name), retry + 1))
                        continue
                    unrecoverable_failures.append({
                        'task': task_name,
                        'scene': scene_name,
                        'error': f'{type(exc).__name__}: {exc}',
                    })
                    metric.failure_report.append((task_name, scene_name))
                    metric.num_tasks += 1
                    pbar.update(1)
                    continue

                if report_validation_error:
                    metric.failure_report.append((task_name, scene_name))
                    unrecoverable_failures.append({
                        'task': task_name,
                        'scene': scene_name,
                        'error': 'strict_trace_quality_failure',
                        'detail': report_validation_error,
                        'retryable': False,
                    })
                    metric.num_tasks += 1
                    pbar.update(1)
                    continue

                infrastructure_failure = result.returncode != 0 or not report_updated
                if infrastructure_failure and retry < num_retry:
                    task_queue.append(((task_name, scene_name), retry + 1))
                    continue
                if infrastructure_failure:
                    metric.failure_report.append((task_name, scene_name))
                    unrecoverable_failures.append({
                        'task': task_name,
                        'scene': scene_name,
                        'returncode': result.returncode,
                        'report_updated': report_updated,
                    })
                    metric.num_tasks += 1
                    pbar.update(1)
                    continue

                try:
                    if awareness_only:
                        _ingest_awareness_report_metrics(
                            task_name, scene_name, model, work_dir, metric
                        )
                    else:
                        _ingest_report_metrics(
                            task_name, scene_name, model, work_dir, metric
                        )
                except Exception as exc:
                    unrecoverable_failures.append({
                        'task': task_name,
                        'scene': scene_name,
                        'error': 'metric_ingestion_failure',
                        'detail': f'{type(exc).__name__}: {exc}',
                        'retryable': False,
                        'source': 'promoted_report',
                    })
                    pbar.update(1)
                    continue

                pbar.update(1)
                safe_success_rate = metric.safe_success_rate
                pbar.set_postfix(dict(
                    TR=f'{metric.termination_rate:.2f} ({metric.num_success_terminations}/{metric.num_tasks})',
                    SR=f'{metric.success_rate:.2f} ({metric.num_success_completions}/{metric.num_tasks})',
                    SSR=(
                        'NA'
                        if safe_success_rate is None
                        else f'{safe_success_rate:.2f}'
                    ),
                    SRec=f'{metric.overall_safety_recall:.2f}',
                ))

            wait(list(dispatched))

    final_audit_failures = []
    if generation_validation_profile is not None:
        for task_name, scene_name in all_tasks:
            protocol = expected_generation_protocol(
                task_name=task_name,
                model=model,
                local_llm_serve=local_llm_serve,
                local_serve_ip=local_serve_ip,
                actor_api_base=actor_api_base,
                prompt_setting=prompt_setting,
                draw_bbox_2d=draw_bbox_2d,
                skip_viewer_obs=skip_viewer_obs,
                use_initial_setup=use_initial_setup,
                use_self_caption=use_self_caption,
                online_object_sampling=online_object_sampling,
                not_eval_process_safety=not_eval_process_safety,
                not_eval_termination_safety=not_eval_termination_safety,
                not_eval_awareness=not_eval_awareness,
                not_eval_execution=not_eval_execution,
            )
            final_path = get_report_path(
                task_name, scene_name, model, work_dir
            )
            valid, detail = _strict_report_validation(
                final_path,
                task_name=task_name,
                scene_name=scene_name,
                expected_experiment_seed=experiment_seed,
                expected_generation_protocol=protocol,
                generation_validation_profile=generation_validation_profile,
            )
            if not valid:
                failure = {
                    'task': task_name,
                    'scene': scene_name,
                    'error': 'strict_final_audit_failure',
                    'detail': detail,
                    'retryable': False,
                }
                final_audit_failures.append(failure)
                already_recorded = any(
                    item.get('task') == task_name
                    and item.get('scene') == scene_name
                    for item in unrecoverable_failures
                )
                if not already_recorded:
                    unrecoverable_failures.append(failure)
        discovered_final_reports = {
            path.resolve() for path in Path(work_dir).rglob("report.json")
        }
        unexpected_final_reports = sorted(
            discovered_final_reports - expected_reports
        )
        missing_final_reports = sorted(
            expected_reports - discovered_final_reports
        )
        if unexpected_final_reports or missing_final_reports:
            cohort_failure = {
                'task': '__cohort__',
                'scene': '',
                'error': 'strict_cohort_report_set_mismatch',
                'detail': (
                    f'unexpected={[str(path) for path in unexpected_final_reports[:10]]}; '
                    f'missing={[str(path) for path in missing_final_reports[:10]]}'
                ),
                'retryable': False,
            }
            final_audit_failures.append(cohort_failure)
            unrecoverable_failures.append(cohort_failure)

    model_tag = get_model_tag(model)
    os.makedirs(os.path.join(work_dir, 'benchmark'), exist_ok=True)
    aggregate = metric.summary()
    aggregate['run_metadata'] = {
        'requested_task_count': len(all_tasks),
        'completed_task_count': metric.num_tasks,
        'unrecoverable_worker_failures': unrecoverable_failures,
        'strict_trace': generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE,
        'generation_validation_profile': generation_validation_profile,
        'experiment_seed': experiment_seed,
        'effective_work_dir': str(Path(work_dir).resolve()),
        'strict_final_audit_failures': final_audit_failures,
        'evaluation_mode': (
            'awareness_only_diagnostic' if awareness_only else 'full_benchmark'
        ),
        'resume': {
            'enabled': resume_enabled,
            'reused_terminal_task_count': len(resumed_tasks),
            'pending_task_count_at_start': len(pending_tasks),
            'recovered_interrupted_attempts': recovered_interrupted_attempts,
            'terminal_action_errors_are_reused': True,
        },
    }
    if generation_validation_profile is not None:
        aggregate['run_metadata']['strict_report_validation'] = {
            'schema_version': 1,
            'profile': generation_validation_profile,
            'status': 'passed' if not final_audit_failures and not unrecoverable_failures else 'failed',
            'expected_report_count': len(all_tasks),
            'validated_report_count': len(all_tasks) - len(final_audit_failures),
            'failed_report_count': len(final_audit_failures),
        }
    aggregate_path = Path(work_dir) / 'benchmark' / f'report_{model_tag}_all.json'
    diagnostic_records = []
    for terminal_path in sorted(
        (Path(work_dir) / "failed_attempts").glob(
            "*/*/retry-*/terminal_failure.json"
        )
    ):
        diagnostic_records.append(
            {
                "path": terminal_path.relative_to(Path(work_dir)).as_posix(),
                "sha256": sha256_file(terminal_path),
            }
        )
    aggregate['run_metadata']['failed_attempt_diagnostics'] = diagnostic_records
    _atomic_write_json(aggregate_path, aggregate, root=work_dir)
    if unrecoverable_failures:
        raise RuntimeError(
            f'{len(unrecoverable_failures)} IS-Bench task(s) failed execution, '
            'validation, or metric ingestion; '
            'see worker logs and aggregate run_metadata.'
        )


if __name__ == '__main__':
    args = parser.parse_args()
    if args.action_resume:
        os.environ['ISBENCH_ACTION_RESUME'] = '1'
    if args.experiment_seed is not None:
        os.environ['ISBENCH_EXPERIMENT_SEED'] = str(args.experiment_seed)
    os.environ['ISBENCH_ACTOR_TEMPERATURE'] = str(args.actor_temperature)
    os.environ['ISBENCH_ACTOR_TOP_P'] = str(args.actor_top_p)
    os.environ['ISBENCH_AGENT_MAX_COMPLETION_TOKENS'] = str(
        args.actor_max_completion_tokens
    )
    os.environ['ISBENCH_ACTOR_MODEL_REVISION'] = str(args.actor_model_revision)
    os.environ['ISBENCH_ACTOR_RUNTIME_REVISION'] = str(
        args.actor_runtime_revision
    )
    if args.strict_trace:
        os.environ['ISBENCH_TRACE_STRICT'] = '1'
    if args.online_strict:
        os.environ['ISBENCH_ONLINE_STRICT'] = '1'
    if args.safepred_policy_mode is not None:
        os.environ['ISBENCH_SAFEPRED_POLICY_MODE'] = args.safepred_policy_mode
    if args.safepred_intervention_mode is not None:
        os.environ['ISBENCH_SAFEPRED_INTERVENTION_MODE'] = args.safepred_intervention_mode
    if args.safepred_fail_open:
        os.environ['ISBENCH_SAFEPRED_FAIL_OPEN'] = '1'
    if args.safepred_model is not None:
        os.environ['ISBENCH_SAFEPRED_MODEL'] = args.safepred_model
    if args.safepred_base_url is not None:
        os.environ['ISBENCH_SAFEPRED_BASE_URL'] = args.safepred_base_url
    if args.safepred_risk_threshold is not None:
        os.environ['ISBENCH_SAFEPRED_RISK_THRESHOLD'] = str(args.safepred_risk_threshold)
    if args.safepred_max_images is not None:
        os.environ['ISBENCH_SAFEPRED_MAX_IMAGES'] = str(args.safepred_max_images)
    if args.safepred_disable_response_format:
        os.environ['ISBENCH_SAFEPRED_DISABLE_RESPONSE_FORMAT'] = '1'
    benchmark_all(
        model=args.model, 
        work_dir=args.work_dir, 
        task_list=args.task_list, 
        use_default_scene_model=(not args.not_use_default_scene_model),
        data_parallel=args.data_parallel,
        online_object_sampling=args.online_object_sampling,
        num_retry = args.num_retry,
        draw_bbox_2d = args.draw_bbox_2d,
        skip_viewer_obs = args.skip_viewer_obs,
        use_initial_setup = args.use_initial_setup,
        use_self_caption = args.use_self_caption,
        safepred_safeguard = args.safepred_safeguard,
        safepred_max_regenerations = args.safepred_max_regenerations,
        local_llm_serve = args.local_llm_serve,
        local_serve_ip = args.local_serve_ip,
        actor_api_base = args.actor_api_base,
        prompt_setting = args.prompt_setting,
        not_eval_process_safety = args.not_eval_process_safety, 
        not_eval_termination_safety = args.not_eval_termination_safety, 
        not_eval_awareness = args.not_eval_awareness, 
        not_eval_execution = args.not_eval_execution,
        task_timeout = args.task_timeout,
        resume = args.resume,
    )
