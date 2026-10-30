"""Private durable recovery primitives; simulator integration is a separate gate.

Never pass checkpoint payloads to models. Request replay requires an identical
ordered request stream and a caller-owned exclusive episode lock.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import copy
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Preserve non-recovery imports on non-POSIX platforms.
    fcntl = None


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _check_credentials(value):
    content = _encode(value).decode('utf-8')
    for name, secret in os.environ.items():
        if any(marker in name.upper() for marker in ('API_KEY', 'TOKEN', 'PASSWORD', 'SECRET')):
            if len(secret) >= 8 and (secret in content or json.dumps(secret)[1:-1] in content):
                raise ValueError('active credential in recovery payload')


def _write(path, value, compressed=False):
    _check_credentials(value)
    data = _encode(value)
    if compressed:
        data = gzip.compress(data)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _envelope(identity, payload):
    body = {'schema': 1, 'identity': identity, 'payload': payload}
    return dict(body, digest=_digest(body))


def _read(path, identity, compressed=False):
    data = path.read_bytes()
    body = json.loads(gzip.decompress(data) if compressed else data)
    digest = body.pop('digest', None)
    if digest != _digest(body):
        raise ValueError('recovery digest mismatch')
    if body.get('schema') != 1 or body.get('identity') != identity:
        raise ValueError('recovery identity mismatch')
    return body['payload']


class CheckpointStore:
    def __init__(self, root, identity):
        self.path = Path(root) / 'checkpoint.json.gz'
        self.identity = copy.deepcopy(identity)

    def load(self):
        if not self.path.exists():
            return None
        return _read(self.path, self.identity, compressed=True)

    def save(self, state):
        step = state.get('step')
        tracker = state.get('tracker', {})
        plans = tracker.get('plans', [])
        outcomes = tracker.get('action_outcomes', [])
        if (type(step) is not int or step < 0 or len(plans) != step
                or len(outcomes) != step or any(not p.get('execution_outcome') for p in plans)):
            raise ValueError('incomplete action boundary')
        statuses = [o.get('status') for o in outcomes]
        statuses += [p['execution_outcome'].get('status') for p in plans]
        if 'infrastructure_error' in statuses:
            raise ValueError('infrastructure outcome is not a recoverable boundary')
        if any(s not in {'completed', 'failed', 'terminal_noop', 'navigation_noop'} for s in statuses):
            raise ValueError('unknown action boundary outcome')
        previous = self.load()
        if previous is not None and step < previous['step']:
            raise ValueError('checkpoint cannot move backwards')
        _write(self.path, _envelope(self.identity, state), compressed=True)


class ActionCheckpointStore(CheckpointStore):
    """Atomic high-level action transaction, never a physics-generator pickle.

    A pending/prepared action is replayed from its saved pre-action state. An
    executed action only needs post-action evaluation and observations. The
    last durable generation remains readable if writing its successor fails.
    Callers must hold ``locked()`` throughout the episode, including restore.
    """
    phases = ('pending', 'prepared', 'executed', 'evaluated', 'boundary', 'terminal')

    @contextmanager
    def locked(self):
        if fcntl is None:
            raise RuntimeError('action recovery requires POSIX file locking')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path.parent / '.action.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('action checkpoint is locked') from None
            yield self
        finally:
            os.close(descriptor)

    @classmethod
    def validate(cls, state):
        phase, step = state.get('phase'), state.get('step')
        if phase not in cls.phases or type(step) is not int or step < 0:
            raise ValueError('invalid action checkpoint phase or step')
        tracker = state.get('tracker', {})
        plans, outcomes = tracker.get('plans', []), tracker.get('action_outcomes', [])
        pending = phase in {'pending', 'prepared'}
        complete = step - int(pending)
        if complete < 0 or len(plans) != step or len(outcomes) != complete:
            raise ValueError('incomplete action boundary')
        if pending and plans[-1].get('execution_outcome') is not None:
            raise ValueError('pending boundary already has an outcome')
        for index, (plan, outcome) in enumerate(zip(plans[:complete], outcomes), 1):
            bound = plan.get('execution_outcome')
            if not bound:
                raise ValueError('incomplete action boundary')
            if outcome.get('status') not in {'completed', 'failed', 'terminal_noop', 'navigation_noop'}:
                raise ValueError('unrecoverable action outcome')
            if (any(outcome.get(k) != v for k, v in bound.items())
                    or outcome.get('sequence_num') != index
                    or outcome.get('action') != (plan.get('plan') or {}).get('action')):
                raise ValueError('action outcome identity mismatch')
        if any(not (p.get('plan') or {}).get('action') for p in plans):
            raise ValueError('missing action identity')
        return step * len(cls.phases) + cls.phases.index(phase)

    def load(self):
        state = super().load()
        if state is not None:
            self.validate(state)
        return state

    def save(self, state):
        position = self.validate(state)
        previous = self.load()
        if previous is not None and position < self.validate(previous):
            raise ValueError('checkpoint cannot move backwards')
        _write(self.path, _envelope(self.identity, state), compressed=True)


def _response(body):
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate(body)


def _transient(error):
    status = getattr(error, 'status_code', None)
    return (status in {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
            or isinstance(error, (TimeoutError, ConnectionError))
            or type(error).__name__ in {'APITimeoutError', 'APIConnectionError'})


class PersistedRequestFailure(ValueError):
    """Keep the last HTTP status observable without making another request."""
    retry_budget_closed = True

    def __init__(self, message, record):
        super().__init__(message)
        attempts = record.get('attempts') or [{}]
        self.status_code = attempts[-1].get('http_status')
        self.original_error_type = attempts[-1].get('error_type')


class RequestJournal:
    """At most three total attempts, including attempts interrupted by a crash.

    Underlying SDK/relay retries MUST be disabled. Persisting a started attempt
    before sending bounds ambiguous deliveries too; it cannot prevent a provider
    charging for a response lost before its durable commit.
    """
    def __init__(self, root, identity, response_factory=None, cursor=0, max_attempts=1,
                 backoff_seconds=5):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError('attempt budget must be between one and three')
        if type(cursor) is not int or cursor < 0:
            raise ValueError('invalid journal cursor')
        self.root = Path(root)
        self.identity = copy.deepcopy(identity)
        self.response_factory = response_factory or _response
        self.cursor = cursor
        self.max_attempts = max_attempts
        self.backoff_seconds = max(0, min(float(backoff_seconds), 30))
        self.last_replayed = False

    def create_once(self, role, endpoint, send, **request):
        """One SDK attempt; the native caller owns backoff and retry loops.

        Unlike create(), this cannot multiply the native transport budget.
        Completed replies are replayed; failed/ambiguous attempts consume the
        same durable budget across process restarts. No exception body is saved.
        """
        request = copy.deepcopy(request)
        _check_credentials(request)
        binding = _digest({'role': role, 'endpoint': endpoint, 'request': request})
        path = self.root / ('request-%08d.json' % self.cursor)
        self.last_replayed = False
        if path.exists():
            record = _read(path, self.identity)
            if record['request_digest'] != binding:
                raise ValueError('request identity mismatch; replay refused')
            if record['status'] == 'completed':
                self.last_replayed = True
                self.cursor += 1
                return self.response_factory(record['response'])
            if record['status'] == 'nonretryable':
                self.last_replayed = True
                raise PersistedRequestFailure('nonretryable persisted request failure', record)
        else:
            record = {'request_digest': binding, 'role': role, 'attempts': [], 'status': 'pending'}
        if len(record['attempts']) >= self.max_attempts:
            self.last_replayed = True
            raise PersistedRequestFailure('persisted attempt budget exhausted', record)
        attempt = {'number': len(record['attempts']) + 1, 'status': 'started', 'started_at_unix': time.time()}
        record['attempts'].append(attempt)
        _write(path, _envelope(self.identity, record))
        try:
            result = send(**copy.deepcopy(request))
        except Exception as error:
            attempt.update(status='failed', error_type=type(error).__name__,
                           http_status=getattr(error, 'status_code', None), finished_at_unix=time.time())
            record['status'] = 'pending' if _transient(error) else 'nonretryable'
            _write(path, _envelope(self.identity, record))
            raise
        attempt.update(status='completed', finished_at_unix=time.time())
        record.update(status='completed', response=result.model_dump(mode='json'))
        _write(path, _envelope(self.identity, record))
        self.cursor += 1
        return result

    def create(self, role, endpoint, send, **request):
        # Keep an isolated copy so a client cannot mutate the next attempt.
        request = copy.deepcopy(request)
        _check_credentials(request)
        binding = _digest({'role': role, 'endpoint': endpoint, 'request': request})
        path = self.root / ('request-%08d.json' % self.cursor)
        if path.exists():
            record = _read(path, self.identity)
            if record['request_digest'] != binding:
                raise ValueError('request identity mismatch; replay refused')
            if record.get('status') == 'completed':
                result = self.response_factory(record['response'])
                self.cursor += 1
                return result
            if record.get('status') == 'nonretryable':
                raise ValueError('nonretryable persisted request failure')
        else:
            record = {'request_digest': binding, 'role': role, 'attempts': [], 'status': 'pending'}
        while len(record['attempts']) < self.max_attempts:
            attempt = {'number': len(record['attempts']) + 1, 'status': 'started',
                       'started_at_unix': time.time()}
            record['attempts'].append(attempt)
            _write(path, _envelope(self.identity, record))
            try:
                result = send(**copy.deepcopy(request))
            except Exception as error:
                attempt.update(status='failed', error_type=type(error).__name__,
                               http_status=getattr(error, 'status_code', None),
                               finished_at_unix=time.time())
                retryable = _transient(error)
                record['status'] = 'pending' if retryable else 'nonretryable'
                _write(path, _envelope(self.identity, record))
                if not retryable or len(record['attempts']) >= self.max_attempts:
                    raise
                time.sleep(self.backoff_seconds * len(record['attempts']))
                continue
            # Commit before exposing the response to the agent. A commit failure
            # must not trigger a second call within this invocation.
            response = result.model_dump(mode='json')
            attempt.update(status='completed', finished_at_unix=time.time())
            record.update(status='completed', response=response)
            _write(path, _envelope(self.identity, record))
            self.cursor += 1
            return result
        raise ValueError('persisted attempt budget exhausted')
