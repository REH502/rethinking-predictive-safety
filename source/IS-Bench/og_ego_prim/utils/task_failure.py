"""Persist worker failures before native simulator teardown can mask them."""

import datetime
import json
import os
from pathlib import Path
import sys
import tempfile


def _active_credential_values() -> tuple[str, ...]:
    """Keep the native batch runner's active-credential selection contract."""
    values: list[str] = []
    for name, value in os.environ.items():
        normalized_name = name.upper()
        if normalized_name.startswith("SLURM_"):
            continue
        if normalized_name.endswith(("_API_KEY_ENV", "_ACCESS_TOKEN_ENV", "_TOKEN_ENV")):
            continue
        if not (normalized_name.endswith(
            ("_API_KEY", "_SERVE_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
        ) or "_SECRET_" in normalized_name):
            continue
        if value and len(value) >= 12 and value not in values:
            values.append(value)
    return tuple(values)


def _redact_active_credentials(data: bytes) -> bytes:
    for value in _active_credential_values():
        data = data.replace(value.encode("utf-8"), b"[REDACTED]")
    return data


def persist_task_failure(exc, *, work_dir, task, scene, model, context):
    """Store structural evidence only, never messages, responses, or locals.

    This is an interrupted-attempt diagnostic, not a completed report or a
    simulator checkpoint. The batch parent retains it with partial artifacts.
    """
    frames = []
    traceback = exc.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append({"file": code.co_filename, "function": code.co_name,
                       "line": traceback.tb_lineno})
        traceback = traceback.tb_next
    tracker = context.get("tracker")
    status = getattr(exc, "status_code", None)
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    record = {
        "schema_version": 1,
        "artifact_type": "isbench_worker_failure",
        "status": "failed",
        "task": task,
        "scene": scene,
        "model": model,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "http_status": status,
        "stage": context.get("stage", "worker_initialization"),
        "traceback_frames": frames,
        "progress": {
            "plans": len(getattr(tracker, "plans", ())),
            "action_outcomes": len(getattr(tracker, "action_outcomes", ())),
        },
        "resume_supported": False,
    }
    recovery = context.get('action_recovery')
    if recovery is not None:
        try:
            checkpoint = recovery.store.load()
            if checkpoint is not None:
                restore_failed = context.get('stage') == 'action_checkpoint_restore'
                record['action_checkpoint'] = {
                    'contract': recovery.store.identity['contract'],
                    'step': checkpoint['step'], 'phase': checkpoint['phase'],
                    'restore_validation': ('failed' if restore_failed else 'required_before_next_model_call'),
                }
                record['resume_supported'] = not restore_failed
        except Exception:
            record['action_checkpoint'] = {'status': 'unreadable_resume_refused'}
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "task_failure.json"
    payload = _redact_active_credentials(
        (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    with tempfile.NamedTemporaryFile(mode="wb", dir=root, prefix=".task_failure.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # An existing failure is immutable attempt evidence, not an overwrite target.
        os.link(temporary, destination)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_with_failure_evidence(worker, **kwargs):
    """CLI boundary: preserve the error and bypass crashing native atexit hooks."""
    context = {"stage": "worker_initialization"}
    try:
        return worker(_failure_context=context, **kwargs)
    except Exception as exc:
        try:
            persist_task_failure(
                exc, work_dir=kwargs["work_dir"], task=kwargs.get("task"),
                scene=kwargs.get("scene"), model=kwargs.get("model"), context=context,
            )
            message = "[isbench] worker failed; structural evidence: task_failure.json\n"
        except Exception:
            message = "[isbench] worker failed; structural evidence could not be persisted\n"
        try:
            sys.stderr.write(message)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Preserve rc=1; Isaac Sim plugin teardown can otherwise replace the
            # original Python/provider exception with SIGSEGV or hang forever.
            os._exit(1)
