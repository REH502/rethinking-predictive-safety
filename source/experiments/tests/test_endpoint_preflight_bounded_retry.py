import io
import json
import urllib.error

import pytest

from experiments import endpoint_preflight as m


def test_429_preflight_retries_exact_request_and_records_each_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv('ISBENCH_ENDPOINT_PREFLIGHT_MAX_ATTEMPTS', '3')
    monkeypatch.setenv('ISBENCH_ENDPOINT_PREFLIGHT_ATTEMPT_LEDGER', str(tmp_path/'ledger.jsonl'))
    calls = []
    sleeps = []
    class Response(io.BytesIO):
        headers = {}
    def send(req, **kw):
        calls.append(req)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, 'private-body', {}, io.BytesIO(b'private-body'))
        return Response(b'{"ok":true}')
    monkeypatch.setattr(m.urllib.request, 'urlopen', send)
    monkeypatch.setattr(m.time, 'sleep', sleeps.append)
    result, _ = m._request_json(url='https://example.invalid/v1/chat/completions', api_key='secret-key-value-123',
        timeout_seconds=5, method='POST', body={'model':'m','messages':[]}, label='actor', stage='inference')
    assert result == {'ok': True}
    assert len(calls) == 3 and calls[0] is calls[1] is calls[2]
    assert sleeps == [5, 10]
    raw = (tmp_path/'ledger.jsonl').read_text()
    assert 'private-body' not in raw and 'secret-key-value' not in raw
    rows = [json.loads(x) for x in raw.splitlines()]
    assert len([x for x in rows if x['status']=='started']) == 3
    assert len({x['request_id'] for x in rows}) == 1


def test_401_and_completed_invalid_json_never_retry(monkeypatch):
    monkeypatch.setenv('ISBENCH_ENDPOINT_PREFLIGHT_MAX_ATTEMPTS', '3')
    monkeypatch.delenv('ISBENCH_ENDPOINT_PREFLIGHT_ATTEMPT_LEDGER', raising=False)
    monkeypatch.setattr(m.time, 'sleep', lambda _: pytest.fail('unexpected retry'))
    for error in (urllib.error.HTTPError('https://example.invalid', 401, 'private', {}, io.BytesIO()), None):
        calls = []
        def send(*a, **kw):
            calls.append(1)
            if error: raise error
            return io.BytesIO(b'not-json')
        monkeypatch.setattr(m.urllib.request, 'urlopen', send)
        with pytest.raises(RuntimeError):
            m._request_json(url='https://example.invalid', api_key='key', timeout_seconds=1,
                            method='GET', label='actor', stage='listing')
        assert len(calls) == 1
