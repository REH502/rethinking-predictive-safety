"""Opt-in, private action-boundary recovery for the native online runner.

No checkpoint data enters a prompt. No physics settling step is inserted.
The Omni adapter deliberately rejects unsupported state rather than pretending
that an action log is a simulator checkpoint. Real GPU restart validation is a
separate qualification gate; enabling this changes the generation identity.
"""
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from functools import wraps

from og_ego_prim.utils.episode_checkpoint import ActionCheckpointStore, RequestJournal


CONTRACT = 'native_action_transaction_v1'
TRACKER_FIELDS = ('task', 'scene', 'model', 'plans', 'raw_outputs',
    'actor_generation_attempts', 'awareness', 'caption', 'safepred_safeguard',
    'safepred_configuration', 'experiment_metadata', 'goal_condition',
    'termination', 'error_stack', 'action_outcomes')
AGENT_FIELDS = ('current_step', 'safepred_feedback', 'safepred_consultation_id',
    'safepred_consultations_this_step', 'safepred_consultations_this_episode')
WM_FIELDS = ('model', 'api_base', 'policy_mode', 'intervention_mode',
    'wm_return_perturbation', 'policy_sha256', 'risk_threshold', 'max_tokens',
    'temperature', 'fail_closed', 'max_images', 'image_max_side', 'image_quality',
    'disable_response_format', 'experiment_seed', 'inference_seed',
    'prediction_horizon', 'request_timeout_seconds', 'semantic_max_attempts',
    'transport_max_attempts', 'transport_retry_base_delay_seconds')


def active_contract():
    from og_ego_prim.utils.simulation_replay import CONTRACT as replay_contract
    selected = os.environ.get('ISBENCH_ACTION_RESUME_CONTRACT', replay_contract)
    if selected not in (CONTRACT, replay_contract):
        raise ValueError('unknown action recovery contract')
    return selected


def enabled():
    value = os.environ.get('ISBENCH_ACTION_RESUME', '0')
    if value not in ('0', '1'):
        raise ValueError('ISBENCH_ACTION_RESUME must be 0 or 1')
    return value == '1'


def action_resume_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not enabled():
            return function(*args, **kwargs)
        root = os.environ.get('ISBENCH_ACTION_CHECKPOINT_DIR', '')
        if not root or not Path(root).is_absolute():
            raise ValueError('action resume requires an absolute episode checkpoint directory')
        with ActionCheckpointStore(root, {}).locked():
            return function(*args, **kwargs)
    return wrapped


def _select(obj, fields):
    return {key: copy.deepcopy(getattr(obj, key)) for key in fields}


def _assign(obj, values):
    for key, value in values.items():
        setattr(obj, key, copy.deepcopy(value))


def _pack(value):
    """Lossless numeric data only. Never pickle objects, clients or callables."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return {'$tensor': value.detach().cpu().tolist(), 'dtype': str(value.dtype).split('.')[-1],
                'device': str(value.device)}
    if isinstance(value, np.ndarray):
        return {'$numpy': value.tolist(), 'dtype': str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'$tuple': [_pack(x) for x in value]}
    if isinstance(value, list):
        return [_pack(x) for x in value]
    if isinstance(value, dict) and all(isinstance(k, str) and not k.startswith('$') for k in value):
        return {k: _pack(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('unsupported private checkpoint state type')


def _unpack(value):
    import numpy as np
    import torch
    if isinstance(value, dict):
        if '$tensor' in value:
            return torch.tensor(value['$tensor'], dtype=getattr(torch, value['dtype']), device=value['device'])
        if '$numpy' in value:
            return np.asarray(value['$numpy'], dtype=value['dtype'])
        if '$tuple' in value:
            return tuple(_unpack(x) for x in value['$tuple'])
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(x) for x in value]
    return value


def _same_numeric(expected, actual):
    # This verifies the serialized native state, not future trajectory identity.
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_same_numeric(v, actual[k]) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_same_numeric(a, b) for a, b in zip(expected, actual)))
    if type(expected) is float and type(actual) in (int, float):
        return math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)
    return type(expected) is type(actual) and expected == actual


def _difference_paths(expected, actual, path=(), limit=8):
    """Structural locations only; no observation/label/response values."""
    if _same_numeric(expected, actual):
        return []
    if isinstance(expected, dict) and isinstance(actual, dict) and expected.keys() == actual.keys():
        result = []
        for key in expected:
            result.extend(_difference_paths(expected[key], actual[key], path + (str(key),), limit - len(result)))
            if len(result) >= limit: break
        return result
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        result = []
        for index, (a, b) in enumerate(zip(expected, actual)):
            result.extend(_difference_paths(a, b, path + (str(index),), limit - len(result)))
            if len(result) >= limit: break
        return result
    return [list(path)]


class NativeStateMismatch(RuntimeError):
    def __init__(self, paths):
        super().__init__('native simulator state roundtrip mismatch')
        self.state_difference_paths = paths


def _restore_particle_bookkeeping(scene, snapshot):
    """Restore exactly what native ModifiedParticles._dump_state() recorded.

    The installed native _load_state() drops zero-count system entries, whereas
    _dump_state() includes them and state_size depends on their presence. Do not
    ignore the difference or change physical particle counts: put back the
    recorded dictionary, including its zeros, then validate the entire state.
    """
    for name, obj_state in snapshot.get('object_registry', {}).items():
        saved = obj_state.get('non_kin', {}).get('ModifiedParticles')
        if saved is None:
            continue
        counts = {key: value for key, value in saved.items() if key != 'n_systems'}
        if (type(saved.get('n_systems')) is not int or saved['n_systems'] != len(counts)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in counts.values())):
            raise ValueError('invalid native particle bookkeeping checkpoint')
        obj = scene.object_registry('name', name)
        states = [value for cls, value in obj.states.items() if cls.__name__ == 'ModifiedParticles']
        if len(states) != 1:
            raise ValueError('native particle state component mismatch')
        states[0].particle_counts = copy.deepcopy(counts)


class OmniStateAdapter:
    """Native rigid-object state, controller, RNG and evaluator bookkeeping.

    Changed object inventory and cloth are not supported by this adapter. In
    particular, silently rebinding BDDL entities after object replacement is
    unsafe. Unsupported scenes fail before the next actor request.
    """
    task_fields = ('_reward', '_done', '_success', '_info', 'instruction_order',
                   'currently_viewed_index', 'currently_viewed_instruction')

    def __init__(self, benchmark, sim):
        self.benchmark, self.sim = benchmark, sim

    def _scenes(self):
        result = self.sim.save()
        if not result:
            raise RuntimeError('simulator did not provide a complete snapshot')
        for scene in self.sim.scenes:
            if any('cloth' in str(obj.prim_type).lower() for obj in scene.objects):
                raise RuntimeError('cloth action recovery is not qualified')
        return [json.loads(row) for row in result]

    def _components(self):
        b = self.benchmark
        components = {}
        for name, obj in b.env.task._termination_conditions.items():
            if type(obj).__name__ not in ('Timeout', 'PredicateGoal'):
                raise RuntimeError('unsupported task termination recovery state')
            components['termination:' + name] = obj
        for name, obj in b.env.task._reward_functions.items():
            if type(obj).__name__ not in ('PotentialReward', 'PointGoalReward'):
                raise RuntimeError('unsupported task reward recovery state')
            components['reward:' + name] = obj
        e = b.evaluator
        for key, rows in (e.process_safety_goal_condition or {}).items():
            for row in rows:
                components['process:' + str(row['task_config_index'])] = row['bddl_evaluator']
        for i, row in enumerate(e.termination_safety_goal_condition or []):
            components['end:' + str(i)] = row['bddl_evaluator']
        if e.execution_goal_condition:
            components['goal'] = e.execution_goal_condition['bddl_evaluator']
        return components

    def capture(self):
        import numpy as np
        import torch
        b = self.benchmark
        controller = b.executor.controller
        if getattr(controller, 'attachments', []):
            raise RuntimeError('live primitive attachment is not a recoverable boundary')
        task = {k: getattr(b.env.task, k) for k in self.task_fields if hasattr(b.env.task, k)}
        components = {name: {k: v for k, v in vars(obj).items() if not callable(v)}
                      for name, obj in self._components().items()}
        tracking = getattr(controller, '_tracking_object', None)
        return {'scenes': self._scenes(), 'runtime': _pack({
            'task': task, 'components': components,
            'episode': b.env._current_episode, 'env_step': b.env._current_step,
            'native_clock': {'_current_time': self.sim._current_time,
                             '_number_of_steps': self.sim._number_of_steps},
            'process_keys': list((b.evaluator.process_safety_goal_condition or {}).keys()),
            'arm_targets': controller._arm_targets,
            'tracking_object': None if tracking is None else tracking.name,
            'viewer_pose': self.sim.viewer_camera.get_position_orientation(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        })}

    def restore(self, snapshot):
        import numpy as np
        import torch
        saved, current = snapshot['scenes'], self._scenes()
        if len(saved) != len(current):
            raise RuntimeError('checkpoint scene count mismatch')
        for old, now in zip(saved, current):
            if (old['init_info'] != now['init_info']
                    or old['objects_info'] != now['objects_info']):
                raise RuntimeError('checkpoint object inventory mismatch; restore refused')
        # Object inventory is identical. Native Simulator.restore() stops and
        # restarts physics to recreate objects; this invalidates the primitive
        # controller's copied robot articulation. Use the native in-place state
        # loader, keeping those live controller handles and adding no sim step.
        from omnigibson.utils.python_utils import recursively_convert_to_torch
        for scene, snapshot_scene in zip(self.sim.scenes, saved):
            scene.load_state(recursively_convert_to_torch(copy.deepcopy(snapshot_scene['state'])), serialized=False)
            _restore_particle_bookkeeping(scene, snapshot_scene['state'])
        actual = self._scenes()
        if not all(_same_numeric(old['state'], now['state']) for old, now in zip(saved, actual)):
            raise NativeStateMismatch(_difference_paths(
                [s['state'] for s in saved], [s['state'] for s in actual]))
        b = self.benchmark
        runtime = _unpack(snapshot['runtime'])
        _assign(self.sim, runtime['native_clock'])
        _assign(b.env.task, runtime['task'])
        b.env._current_episode, b.env._current_step = runtime['episode'], runtime['env_step']
        keys = runtime['process_keys']
        remaining = b.evaluator.process_safety_goal_condition
        if remaining is not None:
            if not set(keys).issubset(remaining):
                raise RuntimeError('checkpoint evaluator trigger mismatch')
            b.evaluator.process_safety_goal_condition = {key: remaining[key] for key in keys}
        components = self._components()
        if set(components) != set(runtime['components']):
            raise RuntimeError('checkpoint evaluator component mismatch')
        for name, data in runtime['components'].items():
            _assign(components[name], data)
        b.executor.controller._arm_targets = runtime['arm_targets']
        name = runtime['tracking_object']
        tracked = None if name is None else b.env.scene.object_registry('name', name)
        if name is not None and tracked is None:
            raise RuntimeError('checkpoint controller object missing')
        b.executor.controller._tracking_object = tracked
        # Camera pose is restored directly: reset_viewer_camera would add steps.
        self.sim.viewer_camera.set_position_orientation(*runtime['viewer_pose'])
        random.setstate(runtime['python_rng'])
        np.random.set_state(runtime['numpy_rng'])
        torch.set_rng_state(runtime['torch_rng'])
        if runtime['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(runtime['cuda_rng'])


class ActionRecovery:
    def __init__(self, store, benchmark, agent, adapter, output_dir):
        self.store, self.benchmark, self.agent = store, benchmark, agent
        self.adapter, self.output_dir = adapter, Path(output_dir)
        self.phase = None
        self.restored_step = None
        self.events = []
        self.journals = {}

    @classmethod
    def for_native(cls, root, benchmark, agent, sim, output_dir):
        wm = agent.safepred_safeguard
        selected = active_contract()
        identity = {'contract': selected,
                    'experiment': benchmark.tracker.experiment_metadata,
                    'safeguard': benchmark.tracker.safepred_configuration,
                    'wm': None if wm is None else _select(wm, WM_FIELDS)}
        store = ActionCheckpointStore(root, identity)
        # Reject incompatible old checkpoints before capturing physics or
        # connecting adapters to any model client.
        store.load()
        if selected == CONTRACT:
            adapter = OmniStateAdapter(benchmark, sim)
        else:
            from og_ego_prim.utils.simulation_replay import replay_adapter_class
            adapter = replay_adapter_class(OmniStateAdapter)(
                benchmark, sim, store.path.parent / 'native_operations', identity)
        recovery = cls(store, benchmark, agent, adapter, output_dir)
        recovery.bind_client('actor', agent.client.client, agent.client._actor_transport_max_retries() + 1)
        if wm is not None:
            recovery.bind_client('wm', wm.client, wm.transport_max_attempts)
        return recovery

    def bind_client(self, role, client, max_attempts):
        if not hasattr(client, 'chat') or not hasattr(client, 'base_url') or client.max_retries != 0:
            raise ValueError('action replay requires an OpenAI-compatible client with SDK retries disabled')
        journal = RequestJournal(self.store.path.parent / 'requests' / role, self.store.identity,
                                 max_attempts=max_attempts)
        send = client.chat.completions.create
        endpoint = str(client.base_url)
        def create(**request):
            if request.get('stream'):
                raise ValueError('native streaming request recovery is not supported')
            return journal.create_once(role, endpoint, send, **request)
        client.chat.completions.create = create
        client._action_recovery_journal = journal
        self.journals[role] = journal

    def _images(self):
        # Retain the exact prefix images; re-rendering them would advance physics.
        images = {}
        for path in sorted(self.output_dir.glob('*/*.png')):
            if path.is_symlink():
                raise ValueError('checkpoint image symlink refused')
            raw = path.read_bytes()
            images[path.relative_to(self.output_dir).as_posix()] = {
                'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode('ascii')}
        return images

    def commit(self, phase):
        t, a = self.benchmark.tracker, self.agent
        if t.video_cache:
            raise RuntimeError('video checkpoint recovery is not supported')
        if a.safepred_pending_consultation is not None:
            raise RuntimeError('cannot commit an unresolved planning consultation as an action boundary')
        wm = a.safepred_safeguard
        state = {'step': len(t.plans), 'phase': phase,
                 'tracker': _pack(_select(t, TRACKER_FIELDS)), 'agent': _select(a, AGENT_FIELDS),
                 'wm_counter': None if wm is None else wm._semantic_request_counter,
                 'simulation': self.adapter.capture(), 'images': self._images(),
                 'recovery_events': self.events,
                 'request_cursors': {name: journal.cursor for name, journal in self.journals.items()}}
        self.store.save(state)
        self.phase = phase

    def restore(self):
        state = self.store.load()
        if state is None:
            return False
        if set(state['request_cursors']) != set(self.journals):
            raise ValueError('checkpoint model request journals mismatch')
        # Check every artifact before modifying the environment or making calls.
        images = {}
        for name, row in state['images'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.suffix != '.png':
                raise ValueError('checkpoint image path refused')
            raw = base64.b64decode(row['data'], validate=True)
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('checkpoint image digest mismatch')
            destination = self.output_dir / name
            if not destination.resolve().is_relative_to(self.output_dir.resolve()):
                raise ValueError('checkpoint image destination escapes output root')
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError('restore destination contains conflicting images')
            images[name] = raw
        self.adapter.restore(state['simulation'])
        _assign(self.benchmark.tracker, _unpack(state['tracker']))
        _assign(self.agent, state['agent'])
        if self.agent.safepred_safeguard is not None:
            self.agent.safepred_safeguard._semantic_request_counter = state['wm_counter']
        self.agent.safepred_pending_consultation = None
        for name, cursor in state['request_cursors'].items():
            self.journals[name].cursor = cursor
        for name, raw in images.items():
            path = self.output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != raw:
                raise ValueError('restore destination contains conflicting images')
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(raw)
        self.phase, self.restored_step = state['phase'], state['step']
        self.events = list(state['recovery_events']) + [{
            'step': self.restored_step, 'phase': self.phase,
            'checkpoint_sha256': hashlib.sha256(self.store.path.read_bytes()).hexdigest(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'serialized_state_validation': 'passed',
            'simulation_restore_method': ('native_operation_replay' if hasattr(self.adapter, 'journal')
                                          else 'native_state_setters'),
            'replayed_native_operations': (len(self.adapter.journal.events)
                                           if hasattr(self.adapter, 'journal') else 0),
        }]
        return True

    def finish_action(self, observe, failure_context=None):
        context = {} if failure_context is None else failure_context
        context['stage'] = 'action_execution_and_evaluation'
        plan = self.benchmark.tracker.plans[-1]['plan']
        if self.phase in ('pending', 'prepared', 'executed'):
            self.benchmark.execute_plan(plan, recovery=self, resume_phase=self.phase)
        if self.phase != 'evaluated':
            raise RuntimeError('unexpected action recovery phase')
        context['stage'] = 'observation_capture'
        observe(len(self.benchmark.tracker.plans), plan)
        context['stage'] = 'action_checkpoint_persistence'
        self.commit('boundary')

    def run(self, observe, use_obs, max_step, *, failure_context=None):
        context = {} if failure_context is None else failure_context
        if self.phase == 'terminal':
            return
        if self.phase != 'boundary':
            self.finish_action(observe, context)
        context['stage'] = 'actor_planning'
        planner = self.agent.step(use_obs=use_obs, max_step=max_step,
                                  resume_after_action=bool(self.benchmark.tracker.plans))
        for plan in planner:
            context['stage'] = 'action_checkpoint_persistence'
            self.commit('pending')
            self.finish_action(observe, context)
            context['stage'] = 'actor_planning'
        context['stage'] = 'termination_evaluation'
        self.benchmark.termination_evaluation()
        context['stage'] = 'action_checkpoint_persistence'
        self.commit('terminal')
