"""Immutable committed-SDK-turn recovery; never claims exact HTTP replay.

Reuses the existing private fork writer and all snapshot/hash/ledger checks.
No native HA files are modified. Semantic and budget failures are not retried.
"""
from contextvars import ContextVar
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from urllib.parse import urlsplit

spec = spec_from_file_location('_uuapi_private_fork', Path(__file__).with_name('ha_aifast_checkpoint_fork.py'))
impl = module_from_spec(spec)
spec.loader.exec_module(impl)
legacy_plan = impl._plan
UnsupportedRecovery = impl.UnsupportedRecovery
CONTEXT = ContextVar('uuapi_recovery_context')
ACTOR_TRANSPORT = frozenset({'InternalServerError', 'RateLimitError', 'PermissionDeniedError',
    'APIConnectionError', 'APITimeoutError', 'ConnectionError', 'TimeoutError'})


def eligible_wm_failure(row):
    if (row.get('status') != 'error' or row.get('component') != 'world_model'
        or urlsplit(str(row.get('endpoint', ''))).hostname != 'uuapi.io'
        or row.get('model') != 'glm-5.3-flash'):
        return False
    status = row.get('http_status', row.get('status_code'))
    return (isinstance(status, int) and (status in {403, 408, 429} or 500 <= status < 600)
            or row.get('exception_type') in {'APIConnectionError', 'APITimeoutError', 'ConnectionError',
                'ConnectError', 'ReadError', 'RemoteProtocolError', 'ReadTimeout', 'ConnectTimeout', 'SSLError'})


impl._is_aifast_transport_error = eligible_wm_failure


def _plan(source, checkpoint_path, temporary_root, phase, ledger):
    context = CONTEXT.get()
    failure_type = context['failure_type']
    if failure_type == 'MaxTurnsExceeded':
        raise UnsupportedRecovery('budget_exhausted_preserved')
    if phase == 'judge_only':
        return legacy_plan(source, checkpoint_path, temporary_root, phase, ledger)
    if ledger is not None:
        result = legacy_plan(source, checkpoint_path, temporary_root, 'wm', ledger)
        if result[2].model_turn_count >= context['max_turns']:
            raise UnsupportedRecovery('no_remaining_model_turn_budget')
        result[-1].update(provider='uuapi.io', failure_type=failure_type)
        return result
    if failure_type not in ACTOR_TRANSPORT:
        raise UnsupportedRecovery('no_positive_actor_transport_evidence')
    snapshot, raw_rows, source_digest = impl._verified_snapshot(checkpoint_path, temporary_root)
    actor = [step for step in snapshot.steps if step.step_kind == 'actor_complete']
    if actor and actor[-1].action['phase_state'].get('error') is None:
        raise UnsupportedRecovery('completed_actor_must_not_be_rerun')
    keep = max((s.step_id for s in snapshot.steps if s.step_kind in {'initial_state', 'model_turn'}), default=0)
    if not keep:
        raise UnsupportedRecovery('no_committed_initial_state')
    selected = snapshot.steps[:keep]
    prefix = replace(snapshot, steps=selected,
        sdk_session_items=tuple(item for step in selected for item in step.sdk_session_items), terminal=None)
    recovered = object.__new__(impl.RuntimeCheckpointCoordinator)._parse_snapshot(prefix)
    if recovered.model_turn_count >= context['max_turns']:
        raise UnsupportedRecovery('no_remaining_model_turn_budget')
    blobs = {}

    def artifact(binding):
        if binding is None:
            return
        if set(binding) != {'relative_path', 'sha256', 'size_bytes'}:
            raise ValueError('artifact_binding_schema_drift')
        relative = binding['relative_path']
        data = impl._read(impl._path(source, relative))
        if len(data) != binding['size_bytes'] or impl._sha(data) != binding['sha256']:
            raise ValueError('bound_artifact_drift')
        if relative in blobs and blobs[relative] != data:
            raise ValueError('artifact_collision')
        blobs[relative] = data
        if relative.startswith('workspace_manifests/'):
            for entry in json.loads(data)['entries']:
                if entry.get('type') == 'file':
                    artifact(entry['blob'])

    for step in selected:
        artifact(step.bank_snapshot.as_json_value() if step.bank_snapshot else None)
        artifact(step.action['trace_snapshot'])
        artifact(step.action['workspace_snapshot'])
        artifact(step.action['phase_state'].get('safety_abstention_artifact'))
    guard = recovered.ledger_offsets.get('guard_decisions')
    if guard is not None:
        parent = impl._path(source, guard.relative_path).parent
        names = ['00_safepred_policies.raw.json', '00_guard_strategy.raw.json']
        native_h1 = (recovered.guard_state or {}).get('schema_version') == 'environment-h1-preview-review-commit-v1'
        if not native_h1:
            names.insert(1, '00_safepred_runtime_config.yaml')
        elif (parent/'00_safepred_runtime_config.yaml').exists():
            raise ValueError('unexpected_learned_wm_config_for_environment_h1')
        for name in names:
            path = parent/name
            relative = path.relative_to(source).as_posix()
            blobs[relative] = impl._read(impl._path(source, relative))
    ledgers = {}
    for name, binding in recovered.ledger_offsets.items():
        data = impl._read(impl._path(source, binding.relative_path))[:binding.byte_offset]
        if len(data) != binding.byte_offset or impl._sha(data) != binding.prefix_sha256:
            raise ValueError('committed_ledger_prefix_drift')
        if name == 'request' and any(json.loads(line).get('status') == 'error' for line in data.splitlines() if line.strip()):
            raise UnsupportedRecovery('earlier_wm_error_requires_ledger_bound_recovery')
        blobs[binding.relative_path] = data
        ledgers[name] = binding.as_json_value()
    manifest = dict(schema_version='ha-uuapi-committed-turn-fork-v1', classification='diagnostic-only',
        source_runtime_root=str(source), source_checkpoint_path=str(checkpoint_path),
        source_checkpoint_sha256=source_digest, source_checkpoint_last_state_sha256=snapshot.last_state_sha256,
        source_terminal_sha256=snapshot.terminal.terminal_sha256 if snapshot.terminal else None,
        identity_sha256=snapshot.identity_sha256, run_id=snapshot.run_id, attempt_id=snapshot.attempt_id,
        phase=phase, source_steps=len(snapshot.steps), retained_steps=keep,
        model_turn_count=recovered.model_turn_count, actor_complete=False,
        prefix_state_sha256=prefix.last_state_sha256, normalized_action_count=len(recovered.normalized_actions),
        trace_snapshot=recovered.trace_snapshot.as_json_value() if recovered.trace_snapshot else None,
        ledger_prefixes=ledgers, failure_evidence=dict(failure_type=failure_type),
        recovery_boundary='last_committed_sdk_turn_before_actor_transport_failure',
        exact_request_replay=False, uncommitted_actions_may_repeat=True,
        actor_calls_expected=True, world_model_calls_expected=(recovered.guard_state or {}).get('schema_version') != 'environment-h1-preview-review-commit-v1')
    return snapshot, raw_rows[:keep], recovered, blobs, manifest


impl._plan = _plan


def plan_checkpoint_fork(source, checkpoint, *, failure_type, max_turns=24, wm_failure_ledger=None):
    token = CONTEXT.set(dict(failure_type=failure_type, max_turns=max_turns))
    try:
        return impl.plan_checkpoint_fork(source, checkpoint, phase='actor', wm_failure_ledger=wm_failure_ledger)
    finally:
        CONTEXT.reset(token)


def fork_checkpoint(source, checkpoint, destination, *, failure_type, max_turns=24, wm_failure_ledger=None):
    token = CONTEXT.set(dict(failure_type=failure_type, max_turns=max_turns))
    try:
        return impl.fork_checkpoint(source, checkpoint, destination, phase='actor', wm_failure_ledger=wm_failure_ledger)
    finally:
        CONTEXT.reset(token)
