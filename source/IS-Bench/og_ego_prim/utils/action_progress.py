"""Body-free primitive telemetry and parent-side episode supervision.

The legacy optional action limit bounds one whole primitive. An explicitly
selected no-progress limit is extended only by completed simulation steps,
never by a heartbeat or a repeated phase. The episode deadline still applies.
"""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class ActionProgress:
    def __init__(self):
        configured = os.environ.get("ISBENCH_ACTION_PROGRESS_PATH")
        self.path = Path(configured) if configured else None
        self.record = {}
        self.last_write = 0.0
        self.ordinal = 0

    def start(self, primitive):
        self.ordinal += 1
        started = time.monotonic()
        self.record = {
            "schema_version": 1, "pid": os.getpid(), "active": True,
            "action_ordinal": self.ordinal,
            # Never persist action arguments, model text, or object metadata.
            "primitive": primitive if re.fullmatch(r"[a-z_]+", primitive) else "unknown",
            "action_started_monotonic": started,
            "last_simulation_progress_monotonic": started,
            "simulation_steps": 0,
        }
        self.update("action_parse", 0, force=True)

    def update(self, phase, simulation_steps, *, force=False):
        if self.path is None:
            return
        now = time.monotonic()
        if simulation_steps > self.record.get("simulation_steps", 0):
            self.record["last_simulation_progress_monotonic"] = now
        self.record.update(phase=phase, simulation_steps=simulation_steps,
                           updated_monotonic=now, updated_at_unix=time.time())
        if not force and now - self.last_write < 5.0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.record, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                if force:
                    os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_write = now

    def finish(self, *, failed=False):
        self.record["active"] = False
        self.update("action_failed" if failed else "action_complete",
                    self.record.get("simulation_steps", 0), force=True)


def action_timeout_record(path, *, pid, now, timeout, mode="absolute"):
    if mode not in {"absolute", "no_progress"}:
        raise ValueError("unknown action timeout mode")
    if timeout <= 0:
        return None
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        started = record["action_started_monotonic"]
        if (record.get("schema_version") != 1 or record.get("pid") != pid
                or record.get("active") is not True
                or not isinstance(started, (int, float))
                or not math.isfinite(started) or started > now):
            return None
        clock = started
        if mode == "no_progress":
            candidate = record.get("last_simulation_progress_monotonic", started)
            if type(candidate) in (int, float) and math.isfinite(candidate) and started <= candidate <= now:
                clock = candidate
        return record if now - clock >= timeout else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def simulator_ready(path, *, pid, started, now):
    """Accept only this newly launched episode's typed, fresh ready handshake."""
    if path is None:
        return False
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        updated = record.get("updated_monotonic")
        return (
            type(record.get("schema_version")) is int
            and record["schema_version"] == 1
            and type(record.get("pid")) is int and record["pid"] == pid
            and record.get("ready") is True and record.get("stage") == "ready"
            and type(updated) in (int, float) and math.isfinite(updated)
            and started <= updated <= now
        )
    except (OSError, ValueError, TypeError):
        return False


def _process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[19]), fields[0]
    except (OSError, ValueError, IndexError):
        return None


def terminate_episode_tree(process, *, grace_seconds=5.0):
    """Terminate only this child tree, never the shared worker process group."""
    if process.poll() is not None:
        return
    root_identity = _process_identity(process.pid)
    identities = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            identity = _process_identity(int(entry.name))
            if identity is not None:
                identities[int(entry.name)] = identity
    targets = {process.pid: root_identity} if root_identity else {}
    while True:
        added = {pid: ident for pid, ident in identities.items()
                 if pid not in targets and ident[0] in targets}
        if not added:
            break
        targets.update(added)

    def alive(pid, identity):
        current = _process_identity(pid)
        return (current is not None and identity is not None
                and current[1] == identity[1] and current[2] != "Z")

    def send(sig):
        for pid, identity in reversed(list(targets.items())):
            if alive(pid, identity):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(alive(*item) for item in targets.items()):
        time.sleep(0.05)
    send(signal.SIGKILL)
    process.wait(timeout=max(grace_seconds, 1.0))


def run_supervised(command, *, env, output, task_timeout, action_timeout,
                   progress_path, poll_seconds=1.0, terminate_grace_seconds=5.0,
                   startup_timeout=0, startup_state_path=None, action_timeout_mode="absolute"):
    if action_timeout_mode not in {"absolute", "no_progress"}:
        raise ValueError("unknown action timeout mode")
    started = time.monotonic()
    failure = None
    startup_complete = startup_timeout <= 0
    # Share the matrix's process group so its cancellation catches this child.
    process = subprocess.Popen(command, env=env, stdout=output, stderr=output,
                               text=True)
    try:
        while process.poll() is None:
            now = time.monotonic()
            if not startup_complete:
                startup_complete = simulator_ready(
                    startup_state_path, pid=process.pid, started=started,
                    now=min(now, started + startup_timeout),
                )
            progress = (
                action_timeout_record(progress_path, pid=process.pid,
                                      now=now, timeout=action_timeout, mode=action_timeout_mode)
                if startup_complete else None
            )
            if not startup_complete and now - started >= startup_timeout:
                failure = {
                    "stage": "simulator_startup_timeout",
                    "startup_timeout_seconds": startup_timeout,
                    "startup_elapsed_seconds": now - started,
                    "pid": process.pid,
                }
            elif progress is not None:
                failure = {
                    "stage": "primitive_action_timeout",
                    "action_timeout_seconds": action_timeout,
                    "action_timeout_mode": action_timeout_mode,
                    "action_progress": progress,
                }
            elif task_timeout is not None and now - started >= task_timeout:
                failure = {"stage": "subprocess_timeout",
                           "task_timeout_seconds": task_timeout}
            if failure is not None:
                terminate_episode_tree(process, grace_seconds=terminate_grace_seconds)
                break
            time.sleep(poll_seconds)
    finally:
        if process.poll() is None:
            terminate_episode_tree(process, grace_seconds=terminate_grace_seconds)
    return subprocess.CompletedProcess(command, 124 if failure else process.returncode), failure
