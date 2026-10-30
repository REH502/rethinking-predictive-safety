"""Private native-operation WAL; models are never called during reconstruction.

This candidate replaces approximate PhysX setter restoration with native prefix
execution. It does NOT repeat actor decisions, count replay as a research action,
or qualify itself for formal use. A caller owns the exclusive episode lock.
"""
import copy
import json
from pathlib import Path
import random

from og_ego_prim.utils.episode_checkpoint import _digest, _envelope, _read, _write


CONTRACT = 'native_operation_replay_v2'
FORMAL_READY = False
KINDS = {'primitive', 'observation', 'process', 'execution_goal', 'remaining_process', 'termination_goal'}


class SimulationReplayMismatch(RuntimeError):
    pass


class SimulationOperationJournal:
    def __init__(self, root, identity, *, capture, get_rng, set_rng,
                 native_errors=(), get_context=lambda: None, set_context=lambda _: None):
        self.root, self.identity = Path(root), copy.deepcopy(identity)
        self.capture, self.get_rng, self.set_rng = capture, get_rng, set_rng
        self.native_errors = native_errors
        self.get_context, self.set_context = get_context, set_context
        self.baseline = _digest(capture())
        self.events, self.callbacks = [], {}
        self.replaying = False

    def bind(self, kind, callback):
        if kind not in KINDS or kind in self.callbacks:
            raise ValueError('unsupported or duplicate native replay operation')
        self.callbacks[kind] = callback

        def call(*args, **kwargs):
            if self.replaying:
                return callback(*args, **kwargs)
            index = len(self.events)
            path = self.root / ('event-%08d.json.gz' % index)
            data = {'kind': kind, 'args': list(copy.deepcopy(args)), 'kwargs': copy.deepcopy(kwargs),
                    'context': copy.deepcopy(self.get_context())}
            existing = _read(path, self.identity, compressed=True) if path.exists() else None
            if existing is not None:
                if any(existing[k] != v for k, v in data.items()):
                    raise SimulationReplayMismatch('uncommitted native event identity drift')
                # A completed native operation can outlive the outer transaction.
                # Redo it with its original inputs/RNG, never replace its evidence.
                self.set_rng(existing['rng'])
            data['rng'] = copy.deepcopy(self.get_rng())
            result, error = None, None
            try:
                result = callback(*args, **kwargs)
            except self.native_errors as exc:
                error = exc
            # Infrastructure exceptions escape without committing partial work.
            data['error_type'] = None if error is None else type(error).__name__
            data['error_digest'] = None if error is None else _digest(str(error))
            data['physical_digest'] = _digest(self.capture())
            if existing is not None and data != existing:
                raise SimulationReplayMismatch('uncommitted native physical result drift')
            if existing is None:
                _write(path, _envelope(self.identity, data), compressed=True)
            self.events.append({'index': index, 'digest': _digest(data)})
            if error is not None:
                raise error
            return result
        return call

    def snapshot(self):
        return {'contract': CONTRACT, 'baseline': self.baseline, 'events': copy.deepcopy(self.events)}

    def restore(self, prefix):
        if self.events or prefix.get('contract') != CONTRACT or prefix.get('baseline') != self.baseline:
            raise SimulationReplayMismatch('native baseline or contract mismatch')
        rows = []
        # Validate ALL bound files and operation identities before touching physics.
        for index, binding in enumerate(prefix['events']):
            if binding.get('index') != index:
                raise ValueError('native operation prefix order mismatch')
            row = _read(self.root / ('event-%08d.json.gz' % index), self.identity, compressed=True)
            if _digest(row) != binding.get('digest') or row.get('kind') not in self.callbacks:
                raise ValueError('native operation binding mismatch')
            rows.append(row)
        self.replaying = True
        try:
            for row in rows:
                self.set_context(copy.deepcopy(row['context']))
                self.set_rng(copy.deepcopy(row['rng']))
                error_type, error_digest = None, None
                try:
                    self.callbacks[row['kind']](*row['args'], **row['kwargs'])
                except self.native_errors as error:
                    error_type = type(error).__name__
                    error_digest = _digest(str(error))
                if error_type != row['error_type'] or error_digest != row['error_digest']:
                    raise SimulationReplayMismatch('native outcome changed during reconstruction')
                if _digest(self.capture()) != row['physical_digest']:
                    raise SimulationReplayMismatch('native physical state changed during reconstruction')
            self.events = copy.deepcopy(prefix['events'])
        finally:
            self.replaying = False


def get_rng():
    import numpy as np
    import torch
    from og_ego_prim.utils.action_recovery import _pack
    return _pack({'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                  'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None})


def set_rng(value):
    import numpy as np
    import torch
    from og_ego_prim.utils.action_recovery import _unpack
    value = _unpack(value)
    random.setstate(value['python_rng'])
    np.random.set_state(value['numpy_rng'])
    torch.set_rng_state(value['torch_rng'])
    if value['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(value['cuda_rng'])


def replay_adapter_class(base):
    """Candidate adapter; no simulator setters, stop/play, or extra physics steps."""
    class ReplayOmniStateAdapter(base):
        def __init__(self, benchmark, sim, root, identity):
            super().__init__(benchmark, sim)
            from omnigibson.action_primitives.action_primitive_set_base import (
                ActionPrimitiveError, ActionPrimitiveErrorGroup)
            self.journal = SimulationOperationJournal(root, identity,
                capture=self.physical_state, get_rng=get_rng, set_rng=set_rng,
                native_errors=(ActionPrimitiveError, ActionPrimitiveErrorGroup),
                get_context=lambda: {'plans': benchmark.tracker.plans},
                set_context=lambda value: setattr(benchmark.tracker, 'plans', value['plans']))
            benchmark.executor.execute_plan = self.journal.bind('primitive', benchmark.executor.execute_plan)
            observe = benchmark.get_surrounding_viewer_obs
            # Paths belong to the new attempt, not the physical operation identity.
            pending_destination = [None]
            def observe_native():
                return observe(save_img=None if self.journal.replaying else pending_destination[0])
            observed = self.journal.bind('observation', observe_native)
            def observe_recorded(save_img=None):
                pending_destination[0] = save_img
                try:
                    return observed()
                finally:
                    pending_destination[0] = None
            benchmark.get_surrounding_viewer_obs = observe_recorded
            for kind, method in (
                ('process', 'evaluate_process_safety_goal_condition'),
                ('execution_goal', 'evaluate_execution_goal_condition'),
                ('remaining_process', 'evaluate_non_executed_process_safety_goal_condition'),
                ('termination_goal', 'evaluate_termination_safety_goal_condition')):
                setattr(benchmark.evaluator, method, self.journal.bind(kind, getattr(benchmark.evaluator, method)))

        def _scenes(self):
            # Replay recreates inventory/cloth through native actions, not setters.
            result = self.sim.save()
            if not result:
                raise RuntimeError('simulator did not provide a complete snapshot')
            return [json.loads(row) for row in result]

        def physical_state(self):
            state = super().capture()
            for key in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                state['runtime'].pop(key, None)
            return state

        def capture(self):
            return dict(super().capture(), replay_prefix=self.journal.snapshot())

        def restore(self, snapshot):
            from og_ego_prim.utils.action_recovery import _same_numeric, _difference_paths, NativeStateMismatch
            self.journal.restore(snapshot['replay_prefix'])
            expected = copy.deepcopy(snapshot)
            expected.pop('replay_prefix')
            saved_rng = {key: expected['runtime'].pop(key) for key in
                         ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')}
            actual = self.physical_state()
            if not _same_numeric(expected, actual):
                raise NativeStateMismatch(_difference_paths(expected, actual))
            # Model transport may consume RNG between native operations. Restore
            # the committed Python RNG only AFTER physical equivalence is proven.
            set_rng(saved_rng)
    return ReplayOmniStateAdapter
