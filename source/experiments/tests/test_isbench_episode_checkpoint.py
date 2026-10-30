"""CPU contracts: durable intra-trace boundaries, not completed-trace skipping."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[2] / 'IS-Bench/og_ego_prim/utils/episode_checkpoint.py'


def helper():
    spec = importlib.util.spec_from_file_location('episode_checkpoint_test', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(step=1):
    return {'step': step, 'tracker': {'plans': [{'execution_outcome': {'status': 'completed'}} for _ in range(step)],
            'action_outcomes': [{'status': 'completed'} for _ in range(step)]}, 'scene': [{'state': {'x': 1}}]}


def test_checkpoint_roundtrip_and_identity_mismatch(tmp_path):
    m = helper()
    store = m.CheckpointStore(tmp_path, {'task': 't', 'arm': 'advisory_only', 'seed': 0})
    assert store.load() is None
    store.save(state())
    assert store.load() == state()
    with pytest.raises(ValueError, match='identity'):
        m.CheckpointStore(tmp_path, {'task': 't', 'arm': 'enforced_guard', 'seed': 0}).load()


def test_checkpoint_corruption_fails_closed(tmp_path):
    store = helper().CheckpointStore(tmp_path, {'task': 't'})
    store.save(state())
    path = tmp_path / 'checkpoint.json.gz'
    body = json.loads(gzip.decompress(path.read_bytes()))
    body['payload']['step'] = 9
    path.write_bytes(gzip.compress(json.dumps(body).encode()))
    with pytest.raises(ValueError, match='digest'):
        store.load()


def test_incomplete_or_infrastructure_action_cannot_be_checkpointed(tmp_path):
    store = helper().CheckpointStore(tmp_path, {'task': 't'})
    data = state()
    data['tracker']['action_outcomes'] = []
    with pytest.raises(ValueError, match='boundary'):
        store.save(data)
    data = state()
    data['tracker']['action_outcomes'][0]['status'] = 'infrastructure_error'
    with pytest.raises(ValueError, match='infrastructure'):
        store.save(data)
    assert store.load() is None


def test_checkpoint_cannot_move_backwards(tmp_path):
    store = helper().CheckpointStore(tmp_path, {'task': 't'})
    store.save(state(2))
    with pytest.raises(ValueError, match='backwards'):
        store.save(state(1))
    assert store.load()['step'] == 2


def test_checkpoint_rejects_active_secret_without_exposing_it(tmp_path, monkeypatch):
    value = 'private-test-value-unique-123'
    monkeypatch.setenv('CHECKPOINT_TEST_API_KEY', value)
    store = helper().CheckpointStore(tmp_path, {'task': 't'})
    data = state(); data['unexpected'] = value
    with pytest.raises(ValueError, match='credential') as error:
        store.save(data)
    assert value not in str(error.value)
    assert not (tmp_path / 'checkpoint.json.gz').exists()


class Reply:
    def __init__(self, text): self.text = text
    def model_dump(self, **kwargs): return {'text': self.text}


def test_journal_reuses_completed_actor_and_wm_calls_after_restart(tmp_path):
    m = helper(); calls = []
    def send(**kwargs): calls.append(kwargs); return Reply('saved')
    journal = m.RequestJournal(tmp_path, {'task': 't'}, response_factory=lambda x: Reply(x['text']))
    for role in ['actor', 'wm']:
        assert journal.create(role, 'https://example.invalid/v1', send, model='m', messages=[]).text == 'saved'
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, response_factory=lambda x: Reply(x['text']))
    for role in ['actor', 'wm']:
        assert restarted.create(role, 'https://example.invalid/v1', send, model='m', messages=[]).text == 'saved'
    assert len(calls) == 2
    assert restarted.cursor == 2


def test_journal_request_mismatch_never_calls_model(tmp_path):
    m = helper(); journal = m.RequestJournal(tmp_path, {'task': 't'}, response_factory=lambda x: x)
    journal.create('actor', 'https://example.invalid', lambda **kw: Reply('ok'), model='original')
    journal.cursor = 0
    with pytest.raises(ValueError, match='request identity'):
        journal.create('actor', 'https://example.invalid', lambda **kw: pytest.fail('model called'), model='changed')


def test_failed_request_keeps_evidence_and_enforces_attempt_budget(tmp_path):
    m = helper(); journal = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=1)
    def fail(**kw): raise TimeoutError('untrusted response body')
    with pytest.raises(TimeoutError): journal.create('actor', 'https://example.invalid', fail, model='m')
    assert journal.cursor == 0
    journal = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=1)
    with pytest.raises(ValueError, match='attempt budget'):
        journal.create('actor', 'https://example.invalid', lambda **kw: pytest.fail('model called'), model='m')
    assert 'untrusted response body' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))


def test_journal_resumes_at_saved_cursor_not_from_first_request(tmp_path):
    m = helper(); journal = m.RequestJournal(tmp_path, {'task': 't'}, response_factory=lambda x: x)
    journal.create('actor', 'https://example.invalid', lambda **kw: Reply('a'), model='m', step=1)
    journal.create('actor', 'https://example.invalid', lambda **kw: Reply('b'), model='m', step=2)
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, cursor=1, response_factory=lambda x: x)
    assert restarted.create('actor', 'https://example.invalid', lambda **kw: pytest.fail('model called'), model='m', step=2)['text'] == 'b'


def test_authorized_two_retries_are_identical_and_durable(tmp_path):
    m = helper(); calls = []
    journal = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, backoff_seconds=0)
    def send(**kw):
        calls.append(kw.copy())
        if len(calls) < 3:
            raise TimeoutError('do not persist exception body')
        return Reply('ok')
    journal.create('wm', 'https://example.invalid', send, model='m', messages=[])
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    record = json.loads(next(tmp_path.glob('request-*.json')).read_text())['payload']
    assert [a['status'] for a in record['attempts']] == ['failed', 'failed', 'completed']
    assert journal.cursor == 1


def test_exhausted_three_attempts_do_not_reset_after_restart(tmp_path):
    m = helper(); calls = []
    def send(**kw):
        calls.append(1)
        raise TimeoutError('private')
    journal = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, backoff_seconds=0)
    with pytest.raises(TimeoutError):
        journal.create('actor', 'https://example.invalid', send, model='m')
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, backoff_seconds=0)
    with pytest.raises(ValueError, match='attempt budget'):
        restarted.create('actor', 'https://example.invalid', send, model='m')
    assert len(calls) == 3


def test_auth_errors_are_not_retried_even_after_restart(tmp_path):
    m = helper(); calls = []
    class AuthError(Exception):
        status_code = 401
    def send(**kw):
        calls.append(1)
        raise AuthError('private')
    journal = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, backoff_seconds=0)
    with pytest.raises(AuthError):
        journal.create('actor', 'https://example.invalid', send, model='m')
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, backoff_seconds=0)
    with pytest.raises(ValueError, match='nonretryable'):
        restarted.create('actor', 'https://example.invalid', send, model='m')
    assert len(calls) == 1


def action_state(step=1, phase='boundary'):
    data = state(step)
    data['phase'] = phase
    for i, plan in enumerate(data['tracker']['plans'], 1):
        plan.update(step=i, plan={'action': 'open(cabinet)'})
        data['tracker']['action_outcomes'][i - 1].update(
            sequence_num=i, action='open(cabinet)')
    if phase in {'pending', 'prepared'}:
        data['tracker']['plans'][-1]['execution_outcome'] = None
        data['tracker']['action_outcomes'].pop()
    return data


def test_action_phases_keep_selected_plan_without_reexecuting_committed_prefix(tmp_path):
    store = helper().ActionCheckpointStore(tmp_path, {'task': 't'})
    store.save(action_state(0))
    for step in (1, 2):
        for phase in ('pending', 'prepared', 'executed', 'evaluated', 'boundary'):
            store.save(action_state(step, phase))
            assert store.load() == action_state(step, phase)
    with pytest.raises(ValueError, match='backwards'):
        store.save(action_state(1, 'pending'))


def test_invalid_pending_action_and_outcome_mismatch_fail_closed(tmp_path):
    store = helper().ActionCheckpointStore(tmp_path, {'task': 't'})
    data = action_state(2, 'pending')
    data['tracker']['plans'][0]['execution_outcome'] = None
    with pytest.raises(ValueError, match='boundary'):
        store.save(data)
    data = action_state()
    data['tracker']['action_outcomes'][0]['status'] = 'failed'
    with pytest.raises(ValueError, match='outcome'):
        store.save(data)


def test_action_checkpoint_exclusive_lock(tmp_path):
    module = helper()
    with module.ActionCheckpointStore(tmp_path, {'task': 't'}).locked():
        with pytest.raises(RuntimeError, match='locked'):
            with module.ActionCheckpointStore(tmp_path, {'task': 't'}).locked():
                pytest.fail('second writer entered')


def test_action_checkpoint_requires_posix_lock_without_breaking_import(tmp_path, monkeypatch):
    module = helper()
    monkeypatch.setattr(module, 'fcntl', None)
    with pytest.raises(RuntimeError, match='POSIX'):
        with module.ActionCheckpointStore(tmp_path, {}).locked():
            pytest.fail('unlocked recovery')


def test_action_checkpoint_preserves_previous_generation_on_failed_commit(tmp_path, monkeypatch):
    module = helper()
    store = module.ActionCheckpointStore(tmp_path, {'task': 't'})
    store.save(action_state(1, 'pending'))
    monkeypatch.setattr(module.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):
        store.save(action_state(1, 'executed'))
    assert store.load()['phase'] == 'pending'


def test_native_attempt_wrapper_does_not_multiply_transport_budget(tmp_path):
    m = helper()
    calls = []
    def fail(**request):
        calls.append(request)
        raise TimeoutError('not saved')
    j = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            j.create_once('actor', 'https://example.invalid', fail, model='m')
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3)
    with pytest.raises(ValueError, match='budget exhausted'):
        restarted.create_once('actor', 'https://example.invalid', fail, model='m')
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]


def test_native_attempt_wrapper_reuses_success_without_calling_sdk(tmp_path):
    m = helper()
    j = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, response_factory=lambda x: x)
    j.create_once('wm', 'https://example.invalid', lambda **kw: Reply('ok'), model='m')
    restarted = m.RequestJournal(tmp_path, {'task': 't'}, max_attempts=3, response_factory=lambda x: x)
    assert restarted.create_once('wm', 'https://example.invalid', lambda **kw: pytest.fail('duplicate call'), model='m') == {'text': 'ok'}
    assert restarted.last_replayed and restarted.cursor == 1


def test_exhausted_native_budget_keeps_original_http_failure_classification(tmp_path):
    m = helper()
    class GatewayError(Exception):
        status_code = 524
    def fail(**kw): raise GatewayError('not stored')
    journal = m.RequestJournal(tmp_path, {}, max_attempts=1)
    with pytest.raises(GatewayError):
        journal.create_once('actor', 'https://example.invalid', fail, model='m')
    restarted = m.RequestJournal(tmp_path, {}, max_attempts=1)
    with pytest.raises(m.PersistedRequestFailure) as caught:
        restarted.create_once('actor', 'https://example.invalid', lambda **kw: pytest.fail('extra call'), model='m')
    assert caught.value.status_code == 524 and caught.value.retry_budget_closed
    assert restarted.last_replayed
