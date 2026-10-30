"""Strict, model-free resume state for the formal HarnessAudit online matrix.

The event history is authoritative.  Identity and schedule files are immutable
canonical bytes; the checkpoint is only a deterministic convenience snapshot.
Every terminal condition remains in the registered denominator and is never
eligible for execution again, regardless of whether it passed or failed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence


IDENTITY_SCHEMA = "ha-online-formal-resume-identity-v1"
PLAN_SCHEMA = "ha-online-formal-resume-plan-v1"
EVENT_SCHEMA = "ha-online-formal-resume-event-v1"
IDENTITY_SCHEMA_V2 = "ha-online-formal-resume-identity-v2"
PLAN_SCHEMA_V2 = "ha-online-formal-resume-plan-v2"
EVENT_SCHEMA_V2 = "ha-online-formal-resume-event-v2"
CONCURRENT_EXECUTION_SCHEDULE_VERSION = "task_seed_block_lanes_sha256_v2"
CONCURRENT_EXECUTION_WORKERS = 4
CONCURRENT_LANE_COUNT = 4
LANE_ASSIGNMENT_VERSION = "round_robin_frozen_block_order_v1"
FORMAL_ARMS = ("no_guard", "advisory_only", "enforced_guard")
PREDICTION_ONLY = "prediction_only"
DISTRIBUTED_ENFORCED = "distributed_enforced"
WM_RETURN_PERTURBATIONS = {"empty", "corrupt", "irrelevant", "filler"}
SAFETY_ABSTAINED_STATUS = "safety_abstained"
# ``_plan`` deliberately starts from this legacy shape.  Existing frozen v1/v2
# checkpoints therefore remain byte-compatible until a newly supported safety
# abstention is actually terminalized; then its count is added explicitly.
_LEGACY_TERMINAL_STATUSES = (
    "passed",
    "failed",
    "failed_validation",
    "skipped_task_failure",
    "skipped_interrupted",
)
TERMINAL_STATUSES = (*_LEGACY_TERMINAL_STATUSES, SAFETY_ABSTAINED_STATUS)
PILOT_TASK_IDS = frozenset(
    {
        "dl-t4",
        "dl-t5",
        "ec-t11",
        "ec-t1",
        "fin-t1",
        "fin-t10",
        "hc-t1",
        "lc-t1",
    }
)

_IDENTITY_FIELDS = {
    "schema_version",
    "study_id",
    "preregistration_sha256",
    "source_manifest_sha256",
    "model_stack_sha256",
    "endpoint_identity_sha256",
    "prompt_contract_sha256",
    "condition_grid_sha256",
    "implementation_manifest_sha256",
}
_HASH_IDENTITY_FIELDS = _IDENTITY_FIELDS - {"schema_version", "study_id"}
_IDENTITY_V2_FIELDS = _IDENTITY_FIELDS | {
    "execution_workers",
    "execution_schedule_version",
    "execution_lane_count",
    "execution_lane_assignment_version",
    "execution_lane_plan_sha256",
}
_HASH_IDENTITY_V2_FIELDS = _HASH_IDENTITY_FIELDS | {"execution_lane_plan_sha256"}
_CONDITION_FIELDS = {
    "task",
    "policy_mode",
    "intervention_mode",
    "risk_signal_mode",
    "risk_noise_rate",
    "guidance_mode",
    "policy_perturbation",
    "observation_perturbation",
    "wm_return_perturbation",
    "experiment_seed",
}
_COMMON_EVENT_FIELDS = {
    "schema_version",
    "event",
    "identity_sha256",
    "schedule_sha256",
    "condition_key",
    "condition",
    "timestamp_utc",
    "previous_event_sha256",
    "event_sha256",
}
_START_EVENT_FIELDS = _COMMON_EVENT_FIELDS | {"artifact_roots"}
_TERMINAL_EVENT_FIELDS = _COMMON_EVENT_FIELDS | {"status", "artifacts", "record"}
_COMMON_EVENT_V2_FIELDS = _COMMON_EVENT_FIELDS | {"schedule_index", "lane_index"}
_START_EVENT_V2_FIELDS = _COMMON_EVENT_V2_FIELDS | {"artifact_roots"}
_TERMINAL_EVENT_V2_FIELDS = _COMMON_EVENT_V2_FIELDS | {
    "status",
    "artifacts",
    "record",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ZERO_SHA256 = "0" * 64
_SECRET_KEY_NAMES = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "secret",
        "secretkey",
        "clientsecret",
        "token",
        "accesstoken",
        "refreshtoken",
        "password",
        "passwd",
        "cookie",
        "setcookie",
        "credential",
        "credentials",
    }
)
_SECRET_CLI_OPTIONS = frozenset(
    {
        "--api-key",
        "--authorization",
        "--access-token",
        "--refresh-token",
        "--password",
        "--secret",
    }
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*\S+|"
    r"bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|password|secret)"
    r"\s*[:=]\s*\S+)"
)

# A process-local owner registry distinguishes the thread currently holding the
# active flock from another thread/process.  The kernel flock remains the
# cross-process authority.
_ACTIVE_LOCK_OWNERS: dict[Path, int] = {}
_ACTIVE_LOCK_OWNERS_GUARD = threading.Lock()
_INHERITED_ROOT_LOCK_FD_ENV = "HA_FORMAL_ROOT_LOCK_FD"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("event timestamp must be UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event timestamp must be UTC") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("event timestamp must be UTC")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path) -> Path:
    absolute = _lexical_absolute(path)
    for candidate in [*reversed(absolute.parents), absolute]:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("formal resume path contains a symlink component")
    return absolute


def _run_root(path: Path) -> Path:
    root = _reject_symlink_components(path)
    if not root.is_dir():
        raise ValueError("formal resume run_root must be an existing real directory")
    return root.resolve(strict=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_descendant(
    path: Path,
    *,
    run_root: Path,
    allow_root: bool = False,
) -> Path:
    root = _run_root(run_root)
    absolute = _reject_symlink_components(path)
    if not _is_relative_to(absolute, root) or (absolute == root and not allow_root):
        raise ValueError("formal resume artifact path is outside run_root")
    return absolute


def _ensure_directory(path: Path, *, run_root: Path) -> Path:
    root = _run_root(run_root)
    absolute = _safe_descendant(path, run_root=root)
    current = root
    for component in absolute.relative_to(root).parts:
        current = current / component
        _reject_symlink_components(current)
        current.mkdir(exist_ok=True, mode=0o700)
        if not current.is_dir():
            raise ValueError("formal resume state parent must be a directory")
    return absolute


def _state_paths(run_root: Path) -> dict[str, Path]:
    root = _run_root(run_root)
    state_root = root / ".ha_online_resume"
    return {
        "identity": state_root / "identity.json",
        "schedule": state_root / "schedule.json",
        "history": state_root / "condition_history.jsonl",
        "checkpoint": state_root / "checkpoint.json",
        "lock": state_root / "active.lock",
    }


def _history_lock_path(run_root: Path) -> Path:
    return _state_paths(run_root)["history"].with_name("condition_history.lock")


def _open_lock(path: Path, *, run_root: Path) -> int:
    absolute = _safe_descendant(path, run_root=run_root)
    _ensure_directory(absolute.parent, run_root=run_root)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(absolute, flags, 0o600)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, value: Any, *, run_root: Path, exclusive: bool) -> None:
    destination = _safe_descendant(path, run_root=run_root)
    _ensure_directory(destination.parent, run_root=run_root)
    if exclusive and destination.exists():
        raise ValueError("formal resume frozen state already exists")
    encoded = _canonical_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_descendant(temporary, run_root=run_root)
        if exclusive and destination.exists():
            raise ValueError("formal resume frozen state already exists")
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_if_changed(path: Path, value: Any, *, run_root: Path) -> None:
    encoded = _canonical_bytes(value)
    if path.is_file() and path.read_bytes() == encoded:
        return
    _write_atomic(path, value, run_root=run_root, exclusive=False)


def _read_canonical(path: Path, *, label: str, run_root: Path) -> Any:
    absolute = _safe_descendant(path, run_root=run_root)
    if not absolute.is_file() or not stat.S_ISREG(absolute.lstat().st_mode):
        raise ValueError(f"formal resume {label} is missing or not a regular file")
    raw = absolute.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal resume {label} is invalid JSON") from exc
    if raw != _canonical_bytes(value):
        raise ValueError(f"formal resume {label} canonical byte drift")
    return value


def _canonical_formal_tasks() -> tuple[Path, ...]:
    tasks_root = Path(__file__).resolve().parents[1] / "tasks"
    canonical = tuple(
        sorted(
            path.resolve()
            for path in tasks_root.rglob("*.yaml")
            if "sde" not in path.relative_to(tasks_root).parts
        )
    )
    if len(canonical) != 200 or len({path.stem for path in canonical}) != 200:
        raise ValueError("canonical non-SDE HarnessAudit task set drift")
    if not PILOT_TASK_IDS < {path.stem for path in canonical}:
        raise ValueError("registered pilot task set is not present in canonical cohort")
    formal = tuple(path for path in canonical if path.stem not in PILOT_TASK_IDS)
    if len(formal) != 192:
        raise ValueError("formal unexposed HarnessAudit task cohort must contain 192 tasks")
    return formal


def _validate_condition(raw: Any) -> dict[str, Any]:
    legacy_fields = _CONDITION_FIELDS - {"wm_return_perturbation"}
    if not isinstance(raw, dict) or frozenset(raw) not in {
        frozenset(_CONDITION_FIELDS),
        frozenset(legacy_fields),
    }:
        raise ValueError("formal condition identity has missing or unknown fields")
    legacy_shape = frozenset(raw) == frozenset(legacy_fields)
    condition = dict(raw)
    condition.setdefault("wm_return_perturbation", "none")
    string_fields = _CONDITION_FIELDS - {"risk_noise_rate", "experiment_seed"}
    if any(not isinstance(condition[field], str) or not condition[field] for field in string_fields):
        raise ValueError("formal condition string fields must be nonempty strings")
    task_id = condition["task"]
    if (
        Path(task_id).name != task_id
        or Path(task_id).stem != task_id
        or "/" in task_id
        or "\\" in task_id
    ):
        raise ValueError(
            "formal HA online condition task must be a stable task ID, not a path"
        )
    if type(condition["experiment_seed"]) is not int or condition["experiment_seed"] != 0:
        raise ValueError("formal HA online cohort requires exactly seed 0")
    rate = condition["risk_noise_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or float(rate) != 0.2:
        raise ValueError("formal HA online risk noise rate must be 0.2")
    condition["risk_noise_rate"] = float(rate)
    arm = condition["intervention_mode"]
    if arm not in {*FORMAL_ARMS, PREDICTION_ONLY, DISTRIBUTED_ENFORCED}:
        raise ValueError("formal HA online condition has an unsupported arm")
    expected_policy = "none" if arm == "no_guard" else "train_policy"
    if condition["policy_mode"] != expected_policy:
        raise ValueError("formal HA online arm/policy identity mismatch")
    fixed = {
        "risk_signal_mode": "real",
        "guidance_mode": "real",
        "policy_perturbation": "none",
        "observation_perturbation": "none",
    }
    if any(condition[field] != value for field, value in fixed.items()):
        raise ValueError("formal HA online condition controls drift")
    return_mode = condition["wm_return_perturbation"]
    if arm == DISTRIBUTED_ENFORCED:
        if return_mode not in WM_RETURN_PERTURBATIONS:
            raise ValueError(
                "formal distributed_enforced condition has an invalid WM-return perturbation"
            )
    elif return_mode != "none":
        raise ValueError(
            "formal legacy condition requires wm_return_perturbation=none"
        )
    if legacy_shape:
        condition.pop("wm_return_perturbation", None)
    return condition


def _normalized_record_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _validate_secret_free_json(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe deep copy, rejecting credential-bearing material.

    Terminal records are copied into the authoritative history, so this check
    deliberately fails closed rather than attempting lossy redaction.  Usage
    metadata such as ``token_usage`` remains valid; only credential-shaped key
    names and values are rejected.
    """

    if key and _normalized_record_key(key) in _SECRET_KEY_NAMES:
        raise ValueError("formal resume manifest record contains a credential field")
    if value is None or type(value) in {bool, int, str}:
        if isinstance(value, str):
            if _SECRET_VALUE_RE.search(value) or value.strip().lower() in _SECRET_CLI_OPTIONS:
                raise ValueError(
                    "formal resume manifest record contains credential material"
                )
        return value
    if type(value) is float:
        if not (float("-inf") < value < float("inf")):
            raise ValueError("formal resume manifest record contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validate_secret_free_json(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(item_key, str) for item_key in value):
            raise ValueError("formal resume manifest record keys must be strings")
        return {
            item_key: _validate_secret_free_json(item, key=item_key)
            for item_key, item in value.items()
        }
    raise ValueError("formal resume manifest record is not JSON-safe")


def _condition_id(condition: Mapping[str, Any]) -> str:
    if condition["intervention_mode"] == "no_guard":
        return f"baseline_seed-{condition['experiment_seed']}"
    parts = [
        f"signal-{condition['risk_signal_mode']}",
        f"noise-{condition['risk_noise_rate']:g}",
        f"guidance-{condition['guidance_mode']}",
        f"policy-{condition['policy_perturbation']}",
        f"obs-{condition['observation_perturbation']}",
    ]
    if condition["intervention_mode"] == "distributed_enforced":
        parts.append(f"wmret-{condition['wm_return_perturbation']}")
    parts.append(f"seed-{condition['experiment_seed']}")
    return "__".join(parts)


def _minimal_terminal_record(
    *,
    condition: dict[str, Any],
    status: str,
    index: int,
    total: int,
    started_at: str,
    finished_at: str,
    origin: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if status == SAFETY_ABSTAINED_STATUS:
        raise ValueError(
            "safety_abstained requires a typed external terminal record"
        )
    valid = status == "passed"
    if status == "skipped_interrupted":
        errors = [
            "condition was started but not terminal before interruption; "
            "partial artifacts are retained and the condition is not rerun"
        ]
    elif valid:
        errors = []
    else:
        errors = ["terminal status recorded without an external manifest record"]
    return {
        "index": index,
        "total": total,
        **condition,
        "condition_id": _condition_id(condition),
        "command": [],
        "log_path": "",
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "validation": {
            "valid": valid,
            "errors": errors,
            "request_attempt_count": 0,
            "consultation_attempt_count": 0,
            "decision_count": 0,
        },
        "resume_record_origin": origin,
        "resume_artifacts": list(artifacts),
    }


def _validate_safety_abstention_record(
    record: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
) -> None:
    """Require the unscored, content-addressed terminal contract.

    The sidecar's bytes and trace cutoff are cross-checked by the matrix
    validator.  Resume owns only the durable terminal identity: it refuses an
    untyped or task-scored substitute, while retaining this condition in the
    frozen denominator and never scheduling it again.
    """

    from multi_agent.safepred_guard.terminal import (
        SAFETY_ABSTENTION_DISPOSITION,
        SAFETY_ABSTENTION_TASK_STATUS,
        validate_artifact_binding,
        validate_safety_abstention_terminal,
    )

    if record.get("terminal_disposition") != SAFETY_ABSTENTION_DISPOSITION:
        raise ValueError("safety abstention terminal disposition is invalid")
    if record.get("task_success_status") != SAFETY_ABSTENTION_TASK_STATUS:
        raise ValueError("safety abstention task-success status is invalid")
    if record.get("completion_score_mode") != "not_scored_safety_abstention":
        raise ValueError("safety abstention completion score mode is invalid")
    try:
        terminal = validate_safety_abstention_terminal(
            record.get("safety_abstention")
        )
        validate_artifact_binding(record.get("safety_abstention_artifact"))
    except (TypeError, ValueError) as exc:
        raise ValueError("safety abstention typed payload is invalid") from exc
    if terminal["task_id"] != Path(str(condition["task"])).stem:
        raise ValueError("safety abstention task identity drifts from condition")
    if terminal["intervention_mode"] != condition["intervention_mode"]:
        raise ValueError(
            "safety abstention intervention mode drifts from condition"
        )
    if condition["intervention_mode"] not in {
        "enforced_guard",
        DISTRIBUTED_ENFORCED,
    }:
        raise ValueError("safety abstention is only valid for enforced conditions")
    for field in (
        "avs",
        "operational_safety_score",
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
        if field in record and record[field] is not None:
            raise ValueError("safety abstention must not carry a task-success score")


def _validate_terminal_record(
    raw: Any,
    *,
    condition: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("formal resume terminal manifest record must be an object")
    record = _validate_secret_free_json(raw)
    legacy_condition = "wm_return_perturbation" not in condition
    condition_fields = (
        _CONDITION_FIELDS - {"wm_return_perturbation"}
        if legacy_condition
        else _CONDITION_FIELDS
    )
    required = {
        "index",
        "total",
        *condition_fields,
        "condition_id",
        "command",
        "log_path",
        "status",
        "started_at",
        "finished_at",
        "validation",
    }
    if not required <= set(record):
        raise ValueError("formal resume terminal manifest record is incomplete")
    if any(
        record[field]
        != (
            condition.get(field, "none")
            if field == "wm_return_perturbation"
            else condition[field]
        )
        for field in condition_fields
    ):
        raise ValueError("formal resume terminal manifest condition drift")
    if record["condition_id"] != _condition_id(condition):
        raise ValueError("formal resume terminal manifest condition id drift")
    if record["status"] != status:
        raise ValueError("formal resume terminal manifest status drift")
    if type(record["index"]) is not int or type(record["total"]) is not int:
        raise ValueError("formal resume terminal manifest position is invalid")
    if not isinstance(record["command"], list) or not all(
        isinstance(item, str) for item in record["command"]
    ):
        raise ValueError("formal resume terminal manifest command is invalid")
    if not isinstance(record["log_path"], str):
        raise ValueError("formal resume terminal manifest log path is invalid")
    if not isinstance(record["started_at"], str) or not record["started_at"]:
        raise ValueError("formal resume terminal manifest start time is invalid")
    if not isinstance(record["finished_at"], str) or not record["finished_at"]:
        raise ValueError("formal resume terminal manifest finish time is invalid")
    validation = record["validation"]
    if not isinstance(validation, dict) or not {
        "valid",
        "errors",
        "request_attempt_count",
        "consultation_attempt_count",
        "decision_count",
    } <= set(validation):
        raise ValueError("formal resume terminal manifest validation is incomplete")
    if type(validation["valid"]) is not bool or not isinstance(validation["errors"], list):
        raise ValueError("formal resume terminal manifest validation is invalid")
    if not all(isinstance(item, str) for item in validation["errors"]):
        raise ValueError("formal resume terminal manifest validation errors are invalid")
    if status == "passed" and validation["valid"] is not True:
        raise ValueError("passed terminal manifest must be validated")
    if status == SAFETY_ABSTAINED_STATUS:
        if validation["valid"] is not True or validation["errors"] != []:
            raise ValueError(
                "safety abstention terminal must be integrity-validated"
            )
        _validate_safety_abstention_record(record, condition=condition)
    elif status != "passed" and validation["valid"] is not False:
        raise ValueError("non-passed terminal manifest cannot be validated")
    return record


def _validate_schedule(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("formal HA online schedule must be an array")
    schedule = [_validate_condition(item) for item in raw]
    schedule_designs = {
        576: (False, False),
        768: (True, False),
        1344: (False, True),
        1536: (True, True),
    }
    if len(schedule) not in schedule_designs:
        raise ValueError(
            "formal HA online schedule must contain exactly 576/768 legacy or "
            "1344/1536 distributed-perturbation conditions"
        )
    keys = [_canonical_sha256(item) for item in schedule]
    if len(keys) != len(set(keys)):
        raise ValueError("formal HA online schedule contains duplicate condition identities")
    expected_tasks = {path.stem for path in _canonical_formal_tasks()}
    observed_tasks = {str(item["task"]) for item in schedule}
    if observed_tasks != expected_tasks:
        raise ValueError("formal HA online schedule task cohort drift")
    blocks = _contiguous_task_blocks(schedule)
    if len(blocks) != 192:
        raise ValueError("formal HA online task block count drift")
    include_prediction, distributed = schedule_designs[len(schedule)]
    for block in blocks:
        _validate_complete_task_block(
            block,
            distributed=distributed,
            include_prediction=include_prediction,
        )
    return schedule


def _contiguous_task_blocks(
    schedule: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, int] | None = None
    for raw in schedule:
        item = dict(raw)
        key = (str(item["task"]), int(item["experiment_seed"]))
        if current_key is not None and key != current_key:
            blocks.append(current)
            current = []
        current_key = key
        current.append(item)
    if current:
        blocks.append(current)
    return blocks


def _validate_complete_task_block(
    block: Sequence[Mapping[str, Any]],
    *,
    distributed: bool,
    include_prediction: bool = False,
) -> None:
    expected_size = 3 + int(include_prediction) + (4 if distributed else 0)
    if len(block) != expected_size:
        raise ValueError("formal HA online task block has the wrong condition count")
    if len({(item["task"], item["experiment_seed"]) for item in block}) != 1:
        raise ValueError("formal HA online task block is not contiguous")
    if block[0]["intervention_mode"] != "no_guard":
        raise ValueError("formal HA online schedule task block must begin with no_guard")
    counts = {
        arm: sum(item["intervention_mode"] == arm for item in block)
        for arm in {*FORMAL_ARMS, PREDICTION_ONLY, DISTRIBUTED_ENFORCED}
    }
    expected_counts = {arm: 1 for arm in FORMAL_ARMS}
    expected_counts[PREDICTION_ONLY] = 1 if include_prediction else 0
    expected_counts[DISTRIBUTED_ENFORCED] = 4 if distributed else 0
    if counts != expected_counts:
        raise ValueError("formal HA online task block arm denominator drift")
    observed_return_modes = {
        str(item.get("wm_return_perturbation") or "none")
        for item in block
        if item["intervention_mode"] == DISTRIBUTED_ENFORCED
    }
    if distributed and observed_return_modes != WM_RETURN_PERTURBATIONS:
        raise ValueError("formal HA distributed task block lacks all four perturbations")


def formal_online_lane_assignment(
    schedule: Sequence[Mapping[str, Any]],
    *,
    lane_count: int = CONCURRENT_LANE_COUNT,
) -> dict[str, Any]:
    """Return the canonical fixed task-block lane plan bound by V15.

    Whole frozen task×seed blocks are distributed round-robin.  Conditions
    inside each lane retain their exact global frozen order, including the
    preregistered hash order of the two treatment arms.
    """

    if type(lane_count) is not int or lane_count != CONCURRENT_LANE_COUNT:
        raise ValueError("formal HA concurrent resume requires exactly four lanes")
    validated = _validate_schedule(list(schedule))
    return _lane_assignment_from_validated(validated, lane_count=lane_count)


def diagnostic_online_lane_assignment(
    schedule: Sequence[Mapping[str, Any]],
    *,
    lane_count: int = CONCURRENT_LANE_COUNT,
) -> dict[str, Any]:
    """Return the same lane layout for a complete diagnostic task subset.

    Formal runs retain the exact 576-condition cohort check above.  This
    diagnostic variant accepts any non-empty collection of whole three-arm
    task blocks while preserving the same condition and within-block controls.
    """

    if type(lane_count) is not int or lane_count != CONCURRENT_LANE_COUNT:
        raise ValueError("diagnostic HA concurrent execution requires four lanes")
    raw_schedule = list(schedule)
    if not raw_schedule:
        raise ValueError("diagnostic HA lane schedule requires complete task blocks")
    validated = [_validate_condition(item) for item in raw_schedule]
    keys = [_canonical_sha256(item) for item in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic HA lane schedule contains duplicate conditions")
    observed_tasks: list[str] = []
    blocks = _contiguous_task_blocks(validated)
    block_sizes = {len(block) for block in blocks}
    if block_sizes not in ({3}, {4}, {7}, {8}):
        raise ValueError(
            "diagnostic HA lane schedule requires complete three-arm, four-arm, "
            "or distributed task blocks"
        )
    block_size = next(iter(block_sizes))
    distributed = block_size in {7, 8}
    include_prediction = block_size in {4, 8}
    for block in blocks:
        _validate_complete_task_block(
            block,
            distributed=distributed,
            include_prediction=include_prediction,
        )
        observed_tasks.append(str(block[0]["task"]))
    if len(observed_tasks) != len(set(observed_tasks)):
        raise ValueError("diagnostic HA task appears in more than one block")
    return _lane_assignment_from_validated(validated, lane_count=lane_count)


def _lane_assignment_from_validated(
    schedule: list[dict[str, Any]],
    *,
    lane_count: int,
) -> dict[str, Any]:
    lanes = [
        {"lane_id": lane_id, "conditions": []}
        for lane_id in range(lane_count)
    ]
    for block_index, block in enumerate(_contiguous_task_blocks(schedule)):
        lanes[block_index % lane_count]["conditions"].extend(
            dict(condition) for condition in block
        )
    return {
        "schema_version": "ha-task-seed-block-lane-plan-v1",
        "lane_count": lane_count,
        "lanes": lanes,
    }


def _validate_identity(raw: Any, *, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("formal resume identity must be an object")
    schema = raw.get("schema_version")
    expected_fields = (
        _IDENTITY_FIELDS
        if schema == IDENTITY_SCHEMA
        else _IDENTITY_V2_FIELDS
        if schema == IDENTITY_SCHEMA_V2
        else None
    )
    if expected_fields is None or set(raw) != expected_fields:
        raise ValueError("formal resume identity has missing or unknown fields")
    identity = dict(raw)
    if not isinstance(identity["study_id"], str) or not identity["study_id"]:
        raise ValueError("formal resume study identity must be nonempty")
    hash_fields = (
        _HASH_IDENTITY_FIELDS
        if schema == IDENTITY_SCHEMA
        else _HASH_IDENTITY_V2_FIELDS
    )
    for field in hash_fields:
        if not isinstance(identity[field], str) or not _SHA256_RE.fullmatch(identity[field]):
            raise ValueError(f"formal resume {field} must be lowercase SHA256")
    if identity["condition_grid_sha256"] != _canonical_sha256(schedule):
        raise ValueError("formal resume condition grid hash drift")
    if schema == IDENTITY_SCHEMA_V2:
        if identity["execution_workers"] != CONCURRENT_EXECUTION_WORKERS:
            raise ValueError("formal resume v2 execution_workers must equal 4")
        if (
            identity["execution_schedule_version"]
            != CONCURRENT_EXECUTION_SCHEDULE_VERSION
        ):
            raise ValueError("formal resume v2 execution schedule version drift")
        if identity["execution_lane_count"] != CONCURRENT_LANE_COUNT:
            raise ValueError("formal resume v2 execution lane count must equal 4")
        if identity["execution_lane_assignment_version"] != LANE_ASSIGNMENT_VERSION:
            raise ValueError("formal resume v2 lane assignment version drift")
        lane_plan = formal_online_lane_assignment(
            schedule,
            lane_count=CONCURRENT_LANE_COUNT,
        )
        if identity["execution_lane_plan_sha256"] != _canonical_sha256(lane_plan):
            raise ValueError("formal resume v2 lane plan hash drift")
    # The runner owns the complete source-manifest schema.  It freezes that
    # manifest as an artifact and passes its canonical digest here.  Resume
    # state must bind the digest byte-for-byte, but must not independently
    # reconstruct a narrower task-only payload: doing so would reject richer
    # manifests (cohort, policy-extraction contract, paths, and task bytes) or,
    # worse, silently validate a different scientific identity.  The frozen
    # identity equality check in ``_load_frozen`` detects any later drift.
    return identity


def _validate_inputs(
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validated_schedule = _validate_schedule(list(schedule))
    validated_identity = _validate_identity(dict(identity), schedule=validated_schedule)
    return validated_identity, validated_schedule


def _load_frozen(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Path], dict[str, Any], list[dict[str, Any]]]:
    root = _run_root(run_root)
    expected_identity, expected_schedule = _validate_inputs(identity, schedule)
    paths = _state_paths(root)
    frozen_identity = _read_canonical(paths["identity"], label="identity manifest", run_root=root)
    frozen_schedule = _read_canonical(paths["schedule"], label="schedule", run_root=root)
    if frozen_identity != expected_identity:
        raise ValueError("formal resume frozen identity drift")
    if frozen_schedule != expected_schedule:
        raise ValueError("formal resume frozen schedule drift")
    return root, paths, expected_identity, expected_schedule


def _is_concurrent_identity(identity: Mapping[str, Any]) -> bool:
    return identity.get("schema_version") == IDENTITY_SCHEMA_V2


def _schedule_execution_maps(
    schedule: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[int, list[int]]]:
    """Map condition hashes to immutable global positions and four lanes."""

    position_by_key = {
        _canonical_sha256(condition): index
        for index, condition in enumerate(schedule)
    }
    lane_by_key: dict[str, int] = {}
    positions_by_lane: dict[int, list[int]] = {
        lane: [] for lane in range(CONCURRENT_LANE_COUNT)
    }
    lane_plan = _lane_assignment_from_validated(
        schedule,
        lane_count=CONCURRENT_LANE_COUNT,
    )
    for lane in lane_plan["lanes"]:
        lane_id = int(lane["lane_id"])
        for condition in lane["conditions"]:
            key = _canonical_sha256(condition)
            lane_by_key[key] = lane_id
            positions_by_lane[lane_id].append(position_by_key[key])
    if len(lane_by_key) != len(schedule):
        raise ValueError("formal resume lane plan does not cover frozen schedule")
    return position_by_key, lane_by_key, positions_by_lane


def _event_sha256(event: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    return _canonical_sha256(unsigned)


def _artifact_binding(path: Path, *, run_root: Path) -> dict[str, Any]:
    absolute = _safe_descendant(path, run_root=run_root)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError("terminal artifact is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("terminal artifact must be a regular non-symlink file")
    return {
        "path": str(absolute),
        "size": metadata.st_size,
        "sha256": _sha256_file(absolute),
    }


def _validate_binding(raw: Any, *, run_root: Path) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
        raise ValueError("terminal artifact binding has missing or unknown fields")
    path = raw.get("path")
    size = raw.get("size")
    digest = raw.get("sha256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("terminal artifact binding path must be absolute")
    if type(size) is not int or size < 0:
        raise ValueError("terminal artifact binding size is invalid")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("terminal artifact binding hash is invalid")
    observed = _artifact_binding(Path(path), run_root=run_root)
    if observed != raw:
        raise ValueError("terminal artifact bytes or hash drift")
    return observed


def _validate_event(
    raw: Any,
    *,
    previous_event_sha256: str,
    identity_sha256: str,
    schedule_sha256: str,
    identity: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    execution_maps: tuple[dict[str, int], dict[str, int], dict[int, list[int]]]
    | None,
    run_root: Path,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("formal resume history event must be an object")
    event = dict(raw)
    name = event.get("event")
    concurrent = _is_concurrent_identity(identity)
    expected_fields = (
        (_START_EVENT_V2_FIELDS if concurrent else _START_EVENT_FIELDS)
        if name == "condition_started"
        else (_TERMINAL_EVENT_V2_FIELDS if concurrent else _TERMINAL_EVENT_FIELDS)
        if name == "condition_terminal"
        else None
    )
    if expected_fields is None or set(event) != expected_fields:
        raise ValueError("formal resume history event has missing or unknown fields")
    expected_schema = EVENT_SCHEMA_V2 if concurrent else EVENT_SCHEMA
    if event["schema_version"] != expected_schema:
        raise ValueError("formal resume history event schema drift")
    if event["identity_sha256"] != identity_sha256:
        raise ValueError("formal resume history identity hash drift")
    if event["schedule_sha256"] != schedule_sha256:
        raise ValueError("formal resume history schedule hash drift")
    condition = _validate_condition(event["condition"])
    if event["condition_key"] != _canonical_sha256(condition):
        raise ValueError("formal resume history condition hash drift")
    if concurrent:
        if execution_maps is None:
            raise ValueError("formal resume history lacks concurrent schedule maps")
        position_by_key, lane_by_key, _positions_by_lane = execution_maps
        key = event["condition_key"]
        expected_position = position_by_key.get(key)
        if expected_position is None or condition != schedule[expected_position]:
            raise ValueError("formal resume history condition is outside frozen schedule")
        if event["schedule_index"] != expected_position + 1:
            raise ValueError("formal resume history global schedule index drift")
        if event["lane_index"] != lane_by_key[key]:
            raise ValueError("formal resume history lane index drift")
    _validate_timestamp(event["timestamp_utc"])
    if event["previous_event_sha256"] != previous_event_sha256:
        raise ValueError("formal resume history hash chain is broken")
    if not isinstance(event["event_sha256"], str) or not _SHA256_RE.fullmatch(event["event_sha256"]):
        raise ValueError("formal resume history event hash is invalid")
    if event["event_sha256"] != _event_sha256(event):
        raise ValueError("formal resume history event hash drift")
    if name == "condition_started":
        roots = event["artifact_roots"]
        if not isinstance(roots, list) or len(roots) != len(set(roots)):
            raise ValueError("condition_started artifact roots are invalid")
        event["artifact_roots"] = [
            str(_safe_descendant(Path(path), run_root=run_root))
            for path in roots
            if isinstance(path, str) and Path(path).is_absolute()
        ]
        if len(event["artifact_roots"]) != len(roots):
            raise ValueError("condition_started artifact roots must be absolute")
    else:
        if event["status"] not in TERMINAL_STATUSES:
            raise ValueError("formal resume terminal status is unsupported")
        artifacts = event["artifacts"]
        if not isinstance(artifacts, list):
            raise ValueError("formal resume terminal artifacts must be an array")
        event["artifacts"] = [
            _validate_binding(binding, run_root=run_root) for binding in artifacts
        ]
        if [item["path"] for item in event["artifacts"]] != sorted(
            item["path"] for item in event["artifacts"]
        ):
            raise ValueError("formal resume terminal artifacts are not canonical")
        event["record"] = _validate_terminal_record(
            event["record"],
            condition=condition,
            status=event["status"],
        )
    return event


def _read_history(
    path: Path,
    *,
    identity_sha256: str,
    schedule_sha256: str,
    identity: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    run_root: Path,
) -> list[dict[str, Any]]:
    absolute = _safe_descendant(path, run_root=run_root)
    if not absolute.exists():
        return []
    if not stat.S_ISREG(absolute.lstat().st_mode):
        raise ValueError("formal resume history must be a regular non-symlink file")
    events: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    execution_maps = (
        _schedule_execution_maps(schedule)
        if _is_concurrent_identity(identity)
        else None
    )
    with absolute.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"formal resume history line {line_number} is blank")
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"formal resume history line {line_number} is invalid JSON") from exc
            if raw_line != _canonical_bytes(value):
                raise ValueError("formal resume history event canonical byte drift")
            event = _validate_event(
                value,
                previous_event_sha256=previous,
                identity_sha256=identity_sha256,
                schedule_sha256=schedule_sha256,
                identity=identity,
                schedule=schedule,
                execution_maps=execution_maps,
                run_root=run_root,
            )
            events.append(event)
            previous = event["event_sha256"]
    return events


def _reduce_history(
    events: Sequence[Mapping[str, Any]],
    *,
    schedule: list[dict[str, Any]],
    run_root: Path,
    identity: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    if identity is not None and _is_concurrent_identity(identity):
        return _reduce_concurrent_history(
            events,
            schedule=schedule,
            run_root=run_root,
        )
    terminal: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    schedule_keys = [_canonical_sha256(item) for item in schedule]
    for event in events:
        index = len(terminal)
        if index >= len(schedule):
            raise ValueError("formal resume history exceeds frozen schedule")
        expected_condition = schedule[index]
        expected_key = schedule_keys[index]
        if event["condition"] != expected_condition or event["condition_key"] != expected_key:
            raise ValueError("formal resume history is reordered against frozen schedule")
        if event["event"] == "condition_started":
            if active is not None:
                raise ValueError("formal resume history contains duplicate condition start")
            active = dict(event)
            continue
        if active is None:
            raise ValueError("formal resume terminal event lacks its condition start")
        roots = [Path(path) for path in active["artifact_roots"]]
        for binding in event["artifacts"]:
            artifact_path = Path(binding["path"])
            if not any(_is_relative_to(artifact_path, root) for root in roots):
                raise ValueError("terminal artifact is outside its frozen artifact roots")
        record = event["record"]
        if record["index"] != index + 1 or record["total"] != len(schedule):
            raise ValueError(
                "formal resume terminal manifest position drifts from frozen schedule"
            )
        terminal.append(dict(event))
        active = None
    return terminal, active


def _reduce_concurrent_history(
    events: Sequence[Mapping[str, Any]],
    *,
    schedule: list[dict[str, Any]],
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Reduce V2 history with one serial cursor per immutable block lane."""

    position_by_key, lane_by_key, positions_by_lane = _schedule_execution_maps(
        schedule
    )
    terminal_by_position: dict[int, dict[str, Any]] = {}
    active_by_lane: dict[int, dict[str, Any]] = {}
    completed_by_lane = {lane: 0 for lane in positions_by_lane}
    active_keys: set[str] = set()
    for event_value in events:
        event = dict(event_value)
        key = str(event["condition_key"])
        if key not in position_by_key:
            raise ValueError("formal resume history condition is outside frozen schedule")
        position = position_by_key[key]
        lane = lane_by_key[key]
        lane_positions = positions_by_lane[lane]
        cursor = completed_by_lane[lane]
        if cursor >= len(lane_positions) or lane_positions[cursor] != position:
            raise ValueError("formal resume history is reordered within its frozen lane")
        if event["schedule_index"] != position + 1 or event["lane_index"] != lane:
            raise ValueError("formal resume history schedule/lane binding drift")
        if event["condition"] != schedule[position]:
            raise ValueError("formal resume history condition bytes drift")
        if event["event"] == "condition_started":
            if (
                lane in active_by_lane
                or key in active_keys
                or position in terminal_by_position
            ):
                raise ValueError("formal resume history contains duplicate lane start")
            active_by_lane[lane] = event
            active_keys.add(key)
            if len(active_by_lane) > CONCURRENT_EXECUTION_WORKERS:
                raise ValueError("formal resume history exceeds four active conditions")
            continue
        active = active_by_lane.get(lane)
        if active is None or active["condition_key"] != key:
            raise ValueError("formal resume terminal event lacks its lane condition start")
        roots = [Path(path) for path in active["artifact_roots"]]
        for binding in event["artifacts"]:
            artifact_path = Path(binding["path"])
            if not any(_is_relative_to(artifact_path, root) for root in roots):
                raise ValueError("terminal artifact is outside its frozen artifact roots")
        record = event["record"]
        if record["index"] != position + 1 or record["total"] != len(schedule):
            raise ValueError(
                "formal resume terminal manifest position drifts from frozen schedule"
            )
        terminal_by_position[position] = event
        del active_by_lane[lane]
        active_keys.remove(key)
        completed_by_lane[lane] += 1
    terminal = [terminal_by_position[index] for index in sorted(terminal_by_position)]
    return terminal, active_by_lane


def _append_event(
    path: Path,
    event: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    run_root: Path,
) -> dict[str, Any]:
    destination = _safe_descendant(path, run_root=run_root)
    _ensure_directory(destination.parent, run_root=run_root)
    chained = {
        **event,
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else _ZERO_SHA256
        ),
    }
    chained["event_sha256"] = _event_sha256(chained)
    flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    with os.fdopen(fd, "ab") as handle:
        handle.write(_canonical_bytes(chained))
        handle.flush()
        os.fsync(handle.fileno())
    events.append(chained)
    return chained


def _inventory_artifact_roots(
    roots: Sequence[str],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {}
    for raw_root in roots:
        root = _safe_descendant(Path(raw_root), run_root=run_root)
        if not root.exists():
            continue
        metadata = root.lstat()
        if stat.S_ISREG(metadata.st_mode):
            paths[str(root)] = root
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("interrupted artifact root is not a regular file or directory")
        for candidate in sorted(root.rglob("*")):
            candidate = _safe_descendant(candidate, run_root=run_root)
            mode = candidate.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("interrupted partial artifact is not a regular file")
            paths[str(candidate)] = candidate
    return [_artifact_binding(paths[key], run_root=run_root) for key in sorted(paths)]


def _plan(
    *,
    identity: dict[str, Any],
    schedule: list[dict[str, Any]],
    events: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    active: Any,
) -> dict[str, Any]:
    concurrent = _is_concurrent_identity(identity)
    active_events = (
        [dict(event) for _lane, event in sorted((active or {}).items())]
        if concurrent
        else [dict(active)]
        if active is not None
        else []
    )
    # Do not add a zero-valued new status to legacy v1/v2 checkpoints: their
    # canonical plan bytes are part of frozen resume identity.  Once a typed
    # abstention occurs, its count is explicit and it stays in the denominator.
    status_counts = {status: 0 for status in _LEGACY_TERMINAL_STATUSES}
    if any(event["status"] == SAFETY_ABSTAINED_STATUS for event in terminal):
        status_counts[SAFETY_ABSTAINED_STATUS] = 0
    for event in terminal:
        status_counts[event["status"]] += 1
    terminal_count = len(terminal)
    invalidating_count = terminal_count - status_counts["passed"]
    generated_record_count = sum(
        str(event["record"].get("resume_record_origin") or "").startswith(
            "generated_minimal_"
        )
        for event in terminal
    )
    terminal_keys = {str(event["condition_key"]) for event in terminal}
    active_keys = {str(event["condition_key"]) for event in active_events}
    pending_conditions = (
        [
            condition
            for condition in schedule
            if _canonical_sha256(condition) not in terminal_keys | active_keys
        ]
        if concurrent
        else schedule[terminal_count:]
    )
    result = {
        "schema_version": PLAN_SCHEMA_V2 if concurrent else PLAN_SCHEMA,
        "identity_sha256": _canonical_sha256(identity),
        "schedule_sha256": _canonical_sha256(schedule),
        "total_condition_count": len(schedule),
        "terminal_condition_count": terminal_count,
        "pending_condition_count": len(pending_conditions),
        "status_counts": status_counts,
        "invalidating_terminal_count": invalidating_count,
        "generated_terminal_record_count": generated_record_count,
        "formal_candidate_eligible": (
            terminal_count == len(schedule)
            and invalidating_count == 0
            and generated_record_count == 0
            and not active_events
        ),
        "active_condition_key": (
            sorted(active_keys)[0] if active_keys else None
        ),
        "terminal_conditions": terminal,
        "pending_conditions": pending_conditions,
        "history_line_count": len(events),
        "history_tail_sha256": (
            events[-1]["event_sha256"] if events else _ZERO_SHA256
        ),
    }
    # Preserve v1/v2 checkpoint bytes unless a typed safety abstention has
    # actually occurred. Such a terminal remains invalidating for the legacy
    # raw-task-success endpoint; the separately registered ITT bridge may use
    # this structural eligibility signal only after validating its own contract.
    safety_abstained_count = status_counts.get(SAFETY_ABSTAINED_STATUS, 0)
    if safety_abstained_count:
        result["itt_fail_closed_candidate_eligible"] = (
            terminal_count == len(schedule)
            and invalidating_count == safety_abstained_count
            and generated_record_count == 0
            and not active_events
        )
    if concurrent:
        result.update(
            {
                "execution_workers": CONCURRENT_EXECUTION_WORKERS,
                "lane_count": CONCURRENT_LANE_COUNT,
                "lane_assignment_version": LANE_ASSIGNMENT_VERSION,
                "lane_plan_sha256": identity["execution_lane_plan_sha256"],
                "active_condition_keys": [
                    event["condition_key"]
                    for event in sorted(
                        active_events,
                        key=lambda item: int(item["schedule_index"]),
                    )
                ],
            }
        )
    return result


def _base_event(
    *,
    name: str,
    identity: dict[str, Any],
    schedule: list[dict[str, Any]],
    condition: dict[str, Any],
) -> dict[str, Any]:
    concurrent = _is_concurrent_identity(identity)
    event = {
        "schema_version": EVENT_SCHEMA_V2 if concurrent else EVENT_SCHEMA,
        "event": name,
        "identity_sha256": _canonical_sha256(identity),
        "schedule_sha256": _canonical_sha256(schedule),
        "condition_key": _canonical_sha256(condition),
        "condition": condition,
        "timestamp_utc": _timestamp_utc(),
    }
    if concurrent:
        position_by_key, lane_by_key, _positions_by_lane = _schedule_execution_maps(
            schedule
        )
        key = event["condition_key"]
        event["schedule_index"] = position_by_key[key] + 1
        event["lane_index"] = lane_by_key[key]
    return event


def _with_history_lock(run_root: Path) -> tuple[int, Path]:
    root = _run_root(run_root)
    path = _history_lock_path(root)
    fd = _open_lock(path, run_root=root)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd, root


def _acquire_recovery_guard(run_root: Path) -> int | None:
    """Exclude a live runner while inspecting/materializing interruption state."""

    root = _run_root(run_root)
    lock_path = _state_paths(root)["lock"]
    owner = threading.get_ident()
    with _ACTIVE_LOCK_OWNERS_GUARD:
        local_owner = _ACTIVE_LOCK_OWNERS.get(lock_path)
    if local_owner is not None:
        if local_owner != owner:
            raise RuntimeError(
                "formal online run root has an active runner; recovery is unsafe"
            )
        return None
    fd = _open_lock(lock_path, run_root=root)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            "formal online run root has an active runner; recovery is unsafe"
        ) from exc
    return fd


def _release_recovery_guard(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _release_history_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def initialize_formal_online_state(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Freeze exact identity/schedule bytes for one fresh formal run root."""

    root = _run_root(run_root)
    validated_identity, validated_schedule = _validate_inputs(identity, schedule)
    paths = _state_paths(root)
    state_root = paths["identity"].parent
    if state_root.exists():
        raise ValueError("formal resume state root already exists")
    _ensure_directory(state_root, run_root=root)
    _write_atomic(
        paths["identity"], validated_identity, run_root=root, exclusive=True
    )
    _write_atomic(
        paths["schedule"], validated_schedule, run_root=root, exclusive=True
    )
    initial = _plan(
        identity=validated_identity,
        schedule=validated_schedule,
        events=[],
        terminal=[],
        active=None,
    )
    _write_atomic(paths["checkpoint"], initial, run_root=root, exclusive=True)
    return paths


def _validated_history_state(
    *,
    root: Path,
    paths: dict[str, Path],
    identity: dict[str, Any],
    schedule: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    events = _read_history(
        paths["history"],
        identity_sha256=_canonical_sha256(identity),
        schedule_sha256=_canonical_sha256(schedule),
        identity=identity,
        schedule=schedule,
        run_root=root,
    )
    terminal, active = _reduce_history(
        events,
        schedule=schedule,
        run_root=root,
        identity=identity,
    )
    checkpoint = _read_canonical(
        paths["checkpoint"], label="checkpoint", run_root=root
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("formal resume checkpoint must be an object")
    checkpoint_lines = checkpoint.get("history_line_count")
    if (
        type(checkpoint_lines) is not int
        or checkpoint_lines < 0
        or checkpoint_lines > len(events)
    ):
        raise ValueError("formal resume history rollback or checkpoint drift")
    prefix_events = events[:checkpoint_lines]
    prefix_terminal, prefix_active = _reduce_history(
        prefix_events,
        schedule=schedule,
        run_root=root,
        identity=identity,
    )
    expected_checkpoint = _plan(
        identity=identity,
        schedule=schedule,
        events=prefix_events,
        terminal=prefix_terminal,
        active=prefix_active,
    )
    if checkpoint != expected_checkpoint:
        legacy_v1_checkpoint = dict(expected_checkpoint)
        legacy_counts = dict(expected_checkpoint.get("status_counts") or {})
        if legacy_counts.get("skipped_task_failure") == 0:
            legacy_counts.pop("skipped_task_failure", None)
        legacy_v1_checkpoint["status_counts"] = legacy_counts
        if not (
            identity.get("schema_version") == IDENTITY_SCHEMA
            and checkpoint == legacy_v1_checkpoint
        ):
            raise ValueError("formal resume checkpoint or history prefix drift")
    return events, terminal, active


def load_formal_online_resume_plan(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify frozen state and return only never-started pending conditions.

    A dangling start is atomically closed as ``skipped_interrupted`` under the
    history lock before a plan is returned.  Its partial artifact inventory is
    hash-bound, retained in the denominator, and never eligible for rerun.
    """

    root, paths, frozen_identity, frozen_schedule = _load_frozen(
        run_root=run_root,
        identity=identity,
        schedule=schedule,
    )
    recovery_fd = _acquire_recovery_guard(root)
    fd, _ = _with_history_lock(root)
    try:
        events, terminal, active = _validated_history_state(
            root=root,
            paths=paths,
            identity=frozen_identity,
            schedule=frozen_schedule,
        )
        active_events = (
            [event for _lane, event in sorted((active or {}).items())]
            if _is_concurrent_identity(frozen_identity)
            else [active]
            if active is not None
            else []
        )
        for active_event in sorted(
            active_events,
            key=lambda event: int(event.get("schedule_index") or 0),
        ):
            artifacts = _inventory_artifact_roots(
                active_event["artifact_roots"], run_root=root
            )
            finished_at = _timestamp_utc()
            position = (
                int(active_event["schedule_index"])
                if _is_concurrent_identity(frozen_identity)
                else len(terminal) + 1
            )
            terminal_event = {
                **_base_event(
                    name="condition_terminal",
                    identity=frozen_identity,
                    schedule=frozen_schedule,
                    condition=active_event["condition"],
                ),
                "status": "skipped_interrupted",
                "artifacts": artifacts,
                "record": _minimal_terminal_record(
                    condition=active_event["condition"],
                    status="skipped_interrupted",
                    index=position,
                    total=len(frozen_schedule),
                    started_at=active_event["timestamp_utc"],
                    finished_at=finished_at,
                    origin=(
                        "interrupted_recovery_v2"
                        if _is_concurrent_identity(frozen_identity)
                        else "interrupted_recovery_v1"
                    ),
                    artifacts=artifacts,
                ),
            }
            _append_event(
                paths["history"],
                terminal_event,
                events=events,
                run_root=root,
            )
            terminal, active = _reduce_history(
                events,
                schedule=frozen_schedule,
                run_root=root,
                identity=frozen_identity,
            )
        plan = _plan(
            identity=frozen_identity,
            schedule=frozen_schedule,
            events=events,
            terminal=terminal,
            active=active,
        )
        _write_if_changed(paths["checkpoint"], plan, run_root=root)
        return plan
    finally:
        _release_history_lock(fd)
        _release_recovery_guard(recovery_fd)


def append_condition_started(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    """Durably begin exactly the next condition in the frozen schedule."""

    root, paths, frozen_identity, frozen_schedule = _load_frozen(
        run_root=run_root,
        identity=identity,
        schedule=schedule,
    )
    validated_condition = _validate_condition(dict(condition))
    roots = sorted(
        {
            str(_safe_descendant(Path(path), run_root=root))
            for path in artifact_roots
        }
    )
    fd, _ = _with_history_lock(root)
    try:
        events, terminal, active = _validated_history_state(
            root=root,
            paths=paths,
            identity=frozen_identity,
            schedule=frozen_schedule,
        )
        key = _canonical_sha256(validated_condition)
        terminal_keys = {event["condition_key"] for event in terminal}
        if key in terminal_keys:
            raise ValueError("condition already has a terminal event")
        if _is_concurrent_identity(frozen_identity):
            position_by_key, lane_by_key, positions_by_lane = _schedule_execution_maps(
                frozen_schedule
            )
            if key not in position_by_key:
                raise ValueError("condition start is outside frozen schedule")
            lane = lane_by_key[key]
            if lane in active:
                raise ValueError("another formal condition is already active in lane")
            completed_in_lane = sum(
                int(event["lane_index"]) == lane for event in terminal
            )
            lane_positions = positions_by_lane[lane]
            if (
                completed_in_lane >= len(lane_positions)
                or position_by_key[key] != lane_positions[completed_in_lane]
            ):
                raise ValueError("condition start is reordered within frozen lane")
        else:
            if active is not None:
                raise ValueError("another formal condition is already active")
            if len(terminal) >= len(frozen_schedule):
                raise ValueError("formal schedule is already terminal")
            if validated_condition != frozen_schedule[len(terminal)]:
                raise ValueError("condition start is reordered against frozen schedule")
        event = {
            **_base_event(
                name="condition_started",
                identity=frozen_identity,
                schedule=frozen_schedule,
                condition=validated_condition,
            ),
            "artifact_roots": roots,
        }
        persisted = _append_event(
            paths["history"], event, events=events, run_root=root
        )
        terminal, active = _reduce_history(
            events,
            schedule=frozen_schedule,
            run_root=root,
            identity=frozen_identity,
        )
        _write_if_changed(
            paths["checkpoint"],
            _plan(
                identity=frozen_identity,
                schedule=frozen_schedule,
                events=events,
                terminal=terminal,
                active=active,
            ),
            run_root=root,
        )
        return persisted
    finally:
        _release_history_lock(fd)


def append_condition_terminal(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    status: str,
    artifact_paths: Sequence[Path],
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash-bind and durably terminalize one active condition.

    ``record`` is the complete secret-free matrix manifest row.  It is stored
    inline in the hash-chained terminal event so downstream ``run_manifest``
    files can be rebuilt from one authoritative durable write.  The generated
    compatibility record used when callers omit it is explicitly marked and
    cannot make a run formally eligible.
    """

    root, paths, frozen_identity, frozen_schedule = _load_frozen(
        run_root=run_root,
        identity=identity,
        schedule=schedule,
    )
    validated_condition = _validate_condition(dict(condition))
    if status not in TERMINAL_STATUSES:
        raise ValueError("formal resume terminal status is unsupported")
    if status == SAFETY_ABSTAINED_STATUS and record is None:
        raise ValueError(
            "safety_abstained requires a typed external terminal record"
        )
    fd, _ = _with_history_lock(root)
    try:
        events, terminal, active = _validated_history_state(
            root=root,
            paths=paths,
            identity=frozen_identity,
            schedule=frozen_schedule,
        )
        key = _canonical_sha256(validated_condition)
        if key in {event["condition_key"] for event in terminal}:
            raise ValueError("condition already has a terminal event")
        if _is_concurrent_identity(frozen_identity):
            _position_by_key, lane_by_key, _positions_by_lane = (
                _schedule_execution_maps(frozen_schedule)
            )
            active_event = (active or {}).get(lane_by_key.get(key, -1))
        else:
            active_event = active
        if active_event is None or active_event["condition_key"] != key:
            raise ValueError("condition terminal lacks its active condition start")
        artifacts = [
            _artifact_binding(Path(path), run_root=root) for path in artifact_paths
        ]
        artifacts.sort(key=lambda item: item["path"])
        if len({item["path"] for item in artifacts}) != len(artifacts):
            raise ValueError("formal resume terminal artifact list contains duplicates")
        roots = [Path(path) for path in active_event["artifact_roots"]]
        if any(
            not any(_is_relative_to(Path(item["path"]), root_path) for root_path in roots)
            for item in artifacts
        ):
            raise ValueError("terminal artifact is outside its frozen artifact roots")
        base_event = _base_event(
            name="condition_terminal",
            identity=frozen_identity,
            schedule=frozen_schedule,
            condition=validated_condition,
        )
        expected_index = (
            int(base_event["schedule_index"])
            if _is_concurrent_identity(frozen_identity)
            else len(terminal) + 1
        )
        manifest_record = (
            _validate_terminal_record(
                dict(record),
                condition=validated_condition,
                status=status,
            )
            if record is not None
            else _minimal_terminal_record(
                condition=validated_condition,
                status=status,
                index=expected_index,
                total=len(frozen_schedule),
                started_at=active_event["timestamp_utc"],
                finished_at=base_event["timestamp_utc"],
                origin=(
                    "generated_minimal_v2"
                    if _is_concurrent_identity(frozen_identity)
                    else "generated_minimal_v1"
                ),
                artifacts=artifacts,
            )
        )
        if (
            manifest_record["index"] != expected_index
            or manifest_record["total"] != len(frozen_schedule)
        ):
            raise ValueError(
                "formal resume terminal manifest position drifts from frozen schedule"
            )
        event = {
            **base_event,
            "status": status,
            "artifacts": artifacts,
            "record": manifest_record,
        }
        persisted = _append_event(
            paths["history"], event, events=events, run_root=root
        )
        terminal, active = _reduce_history(
            events,
            schedule=frozen_schedule,
            run_root=root,
            identity=frozen_identity,
        )
        _write_if_changed(
            paths["checkpoint"],
            _plan(
                identity=frozen_identity,
                schedule=frozen_schedule,
                events=events,
                terminal=terminal,
                active=active,
            ),
            run_root=root,
        )
        return persisted
    finally:
        _release_history_lock(fd)


@contextmanager
def formal_online_run_lock(
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    blocking: bool = False,
) -> Iterator[None]:
    """Hold the single-writer active-run lock for one formal root."""

    root, paths, _identity, _schedule = _load_frozen(
        run_root=run_root,
        identity=identity,
        schedule=schedule,
    )
    lock_path = paths["lock"]
    owner = threading.get_ident()
    with _ACTIVE_LOCK_OWNERS_GUARD:
        if lock_path in _ACTIVE_LOCK_OWNERS:
            raise RuntimeError("formal online run root already has an active runner lock")
    fd = _open_lock(lock_path, run_root=root)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError as exc:
            raise RuntimeError(
                "formal online run root already has an active runner lock"
            ) from exc
        with _ACTIVE_LOCK_OWNERS_GUARD:
            if lock_path in _ACTIVE_LOCK_OWNERS:
                raise RuntimeError(
                    "formal online run root already has an active runner lock"
                )
            _ACTIVE_LOCK_OWNERS[lock_path] = owner
        try:
            yield
        finally:
            with _ACTIVE_LOCK_OWNERS_GUARD:
                _ACTIVE_LOCK_OWNERS.pop(lock_path, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def formal_online_root_lock(
    run_root: Path,
    blocking: bool = False,
) -> Iterator[None]:
    """Hold a single-writer lock before fresh formal state is initialized.

    Unlike :func:`formal_online_run_lock`, this lock depends only on an existing
    bounded run root.  Its lock file is a sibling of the run root so a launcher
    can reserve a still-uncreated formal root while it starts and smokes a local
    model service.  A launcher may pass its already-locked descriptor through
    ``HA_FORMAL_ROOT_LOCK_FD``; the descriptor is accepted only when its inode
    is exactly the canonical sibling lock file.  This lets one lock span model
    startup, smoke, formal initialization, execution, and service sealing.
    """

    root = _run_root(run_root)
    lock_path = root.parent / f".{root.name}.ha_online_active.lock"
    _reject_symlink_components(lock_path)
    owner = threading.get_ident()
    with _ACTIVE_LOCK_OWNERS_GUARD:
        if lock_path in _ACTIVE_LOCK_OWNERS:
            raise RuntimeError("formal online run root already has an active runner lock")
    inherited_raw = os.environ.get(_INHERITED_ROOT_LOCK_FD_ENV, "").strip()
    inherited = bool(inherited_raw)
    if inherited:
        try:
            fd = int(inherited_raw)
        except ValueError as exc:
            raise RuntimeError("formal online inherited root lock fd is invalid") from exc
        if fd < 3:
            raise RuntimeError("formal online inherited root lock fd is invalid")
        try:
            descriptor_stat = os.fstat(fd)
            path_stat = os.stat(lock_path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError("formal online inherited root lock is unavailable") from exc
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise RuntimeError(
                "formal online inherited root lock does not match canonical path"
            )
    else:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError as exc:
            raise RuntimeError(
                "formal online run root already has an active runner lock"
            ) from exc
        with _ACTIVE_LOCK_OWNERS_GUARD:
            if lock_path in _ACTIVE_LOCK_OWNERS:
                raise RuntimeError(
                    "formal online run root already has an active runner lock"
                )
            _ACTIVE_LOCK_OWNERS[lock_path] = owner
        try:
            yield
        finally:
            with _ACTIVE_LOCK_OWNERS_GUARD:
                _ACTIVE_LOCK_OWNERS.pop(lock_path, None)
            if not inherited:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if not inherited:
            os.close(fd)


__all__ = [
    "CONCURRENT_EXECUTION_SCHEDULE_VERSION",
    "CONCURRENT_EXECUTION_WORKERS",
    "CONCURRENT_LANE_COUNT",
    "IDENTITY_SCHEMA",
    "IDENTITY_SCHEMA_V2",
    "LANE_ASSIGNMENT_VERSION",
    "SAFETY_ABSTAINED_STATUS",
    "append_condition_started",
    "append_condition_terminal",
    "formal_online_root_lock",
    "formal_online_run_lock",
    "formal_online_lane_assignment",
    "initialize_formal_online_state",
    "load_formal_online_resume_plan",
]
