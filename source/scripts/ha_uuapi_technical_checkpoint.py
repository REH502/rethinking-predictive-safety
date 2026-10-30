"""Bounded technical recovery extension; original frozen helpers stay unchanged.

Never resets a completed model-turn budget or retries validated model-output
failures. HTTP-envelope JSON decoding is distinct from model-content parsing.
Every result is an immutable, diagnostic recovery fork, not exact HTTP replay.
"""
from contextvars import ContextVar
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.parse import urlsplit

spec = spec_from_file_location('_technical_uuapi_fork', Path(__file__).with_name('ha_uuapi_checkpoint_recovery.py'))
base = module_from_spec(spec)
spec.loader.exec_module(base)
UnsupportedRecovery = base.UnsupportedRecovery
DETAIL = ContextVar('technical_recovery_detail')
_old_plan = base.impl._plan
_old_eligible = base.eligible_wm_failure
# These two are execution/interface exceptions, not a schema-repair budget.
# Unknown UserError is deliberately not treated as a transport exception.
base.ACTOR_TRANSPORT = base.ACTOR_TRANSPORT | {'AttributeError', 'BadRequestError'}


def envelope_failure(row, *, http_envelope_decode_proven=False):
    if _old_eligible(row):
        return True
    return bool(http_envelope_decode_proven
        and row.get('status') == 'error' and row.get('component') == 'world_model'
        and urlsplit(str(row.get('endpoint', ''))).hostname == 'uuapi.io'
        and row.get('model') == 'glm-5.3-flash'
        and row.get('exception_type') == 'JSONDecodeError'
        and row.get('http_status', row.get('status_code')) == 200)


def _eligible(row):
    return envelope_failure(row, http_envelope_decode_proven=DETAIL.get()['http_envelope_decode_proven'])


base.impl._is_aifast_transport_error = _eligible


def _plan(source, checkpoint, temporary, phase, ledger):
    details = DETAIL.get()
    if phase == 'judge_only':
        return _old_plan(source, checkpoint, temporary, phase, ledger)
    try:
        result = _old_plan(source, checkpoint, temporary, phase, ledger)
    except UnsupportedRecovery as exc:
        if (str(exc) != 'no checkpoint-bound pre-failure SDK turn'
                or not details['ledger_ownership_proven'] or ledger is None):
            raise
        # Ledger ownership is supplied by the hash-bound receipt + native
        # validation record. Only an initial-state-only snapshot may use this.
        snapshot, _, _ = base.impl._verified_snapshot(checkpoint, temporary)
        if any(s.step_kind == 'model_turn' for s in snapshot.steps):
            raise UnsupportedRecovery('initial_fallback_would_discard_committed_turns') from None
        base.impl._transport_failure(ledger)
        context = dict(base.CONTEXT.get())
        token = base.CONTEXT.set(dict(context, failure_type='APIConnectionError'))
        try:
            result = _old_plan(source, checkpoint, temporary, phase, None)
        finally:
            base.CONTEXT.reset(token)
        result[-1]['failure_evidence'] = dict(failure_type=context['failure_type'],
            ledger_ownership='verified_receipt_and_native_validation')
    snapshot, _, recovered, _, manifest = result
    prior_turns = sum(s.step_kind == 'model_turn' for s in snapshot.steps)
    discarded = prior_turns - recovered.model_turn_count
    if discarded:
        raise UnsupportedRecovery('rollback_would_reset_committed_turn_budget')
    manifest.update(classification='diagnostic-only',
        failure_type=base.CONTEXT.get()['failure_type'],
        recovery_boundary='verified_complete_sdk_turn_before_technical_failure',
        prior_model_turn_count=prior_turns, discarded_model_turn_count=0,
        exact_request_replay=False, uncommitted_actions_may_repeat=True,
        semantic_regeneration_budget_reset=False)
    return result


base.impl._plan = _plan


def plan_checkpoint_fork(source, checkpoint, *, failure_type, max_turns=24,
                         wm_failure_ledger=None, http_envelope_decode_proven=False,
                         ledger_ownership_proven=False):
    token = DETAIL.set(dict(http_envelope_decode_proven=http_envelope_decode_proven,
                           ledger_ownership_proven=ledger_ownership_proven))
    try:
        return base.plan_checkpoint_fork(source, checkpoint, failure_type=failure_type,
            max_turns=max_turns, wm_failure_ledger=wm_failure_ledger)
    finally:
        DETAIL.reset(token)


def fork_checkpoint(source, checkpoint, destination, **kwargs):
    token = DETAIL.set(dict(http_envelope_decode_proven=kwargs.pop('http_envelope_decode_proven', False),
                           ledger_ownership_proven=kwargs.pop('ledger_ownership_proven', False)))
    try:
        return base.fork_checkpoint(source, checkpoint, destination, **kwargs)
    finally:
        DETAIL.reset(token)
