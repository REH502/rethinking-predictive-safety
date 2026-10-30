import os
import sys
import ctypes
import json
import random
import argparse
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit


def _register_isaacsim_dll_dirs() -> None:
    """Register IsaacSim DLL dirs before OmniGibson imports load native plugins."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    env_dirs = [
        item
        for item in os.environ.get("ISBENCH_EXTRA_DLL_DIRS", "").split(os.pathsep)
        if item
    ]
    default_dirs = [
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "isaacsim"
        / "extscache"
        / "omni.sensors.nv.common-1.2.2-isaac+wx64.r.cp310"
        / "bin",
        Path(sys.prefix) / "Library" / "bin",
    ]
    dll_dirs = [Path(item) for item in env_dirs] + default_dirs
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        for name in ("hdf5.dll", "hdf5_hl.dll", "hdf5_cpp.dll", "hdf5_hl_cpp.dll"):
            dll_path = dll_dir / name
            if dll_path.exists():
                try:
                    ctypes.WinDLL(str(dll_path))
                except OSError:
                    pass


_register_isaacsim_dll_dirs()
from og_ego_prim.utils.constants import CAMERAS, TASKS
from og_ego_prim.utils.task_failure import run_with_failure_evidence
from og_ego_prim.utils.action_recovery import action_resume_scope, enabled as action_resume_enabled
from og_ego_prim.utils.trace_provenance import (
    actor_transport_max_retries,
    actor_transport_retry_base_delay_seconds,
    actor_transport_retry_policy,
    build_experiment_metadata,
    build_generation_protocol,
    dependency_versions,
    derive_task_seed,
    resolve_openai_api_base,
    strict_http_endpoint_error,
)

# OmniGibson prepends simulator/vendor site-packages while it imports native
# plugins. Freeze the dependency identity before that import so the actor child
# and the batch parent build the same generation-protocol provenance.
os.environ.setdefault(
    "ISBENCH_DEPENDENCY_VERSIONS_JSON",
    json.dumps(dependency_versions(), sort_keys=True, separators=(",", ":")),
)


def _reject_inline_actor_api_key(_value: str) -> str:
    """Keep actor credentials out of process arguments and child logs."""

    raise argparse.ArgumentTypeError(
        "inline actor credentials are disabled; configure LOCAL_SERVE_KEY "
        "or AGENT_API_KEY in the process environment"
    )


def _actor_api_key_from_environment() -> str:
    """Resolve the actor key only at the final client-construction boundary."""

    return (
        os.environ.get("LOCAL_SERVE_KEY")
        or os.environ.get("AGENT_API_KEY")
        or "sk-123456"
    )


parser = argparse.ArgumentParser()
parser.add_argument('--try_id', type=str, default=None)
# parser.add_argument('--try_id', type=bool, default=True)

parser.add_argument('--task', type=str, default=None)
parser.add_argument('--scene', type=str, default=None)
parser.add_argument('--model', type=str, default=None, help="If not local llm, referece to the model_id, if local_llm, referece to the local model path.")
parser.add_argument('--local_llm_serve', action='store_true')
parser.add_argument('--local_serve_ip', type=str, default="")
parser.add_argument('--actor_api_base', type=str, default=None)
parser.add_argument(
    '--local_serve_key',
    type=_reject_inline_actor_api_key,
    default=None,
    metavar="ENV_ONLY",
    help=argparse.SUPPRESS,
)
parser.add_argument('--work_dir', type=str, default='./work_dir')

parser.add_argument('--draw_bbox_2d', action='store_true')
parser.add_argument('--skip_viewer_obs', action='store_true')
parser.add_argument('--use_initial_setup', action='store_true')
parser.add_argument('--use_self_caption', action='store_true')
parser.add_argument('--safepred_safeguard', action='store_true')
parser.add_argument('--safepred_max_regenerations', type=int, default=None)
parser.add_argument('--safepred_policy_mode', choices=['generic_embodied_policy', 'train_policy', 'full_isbench_policy_text'], default=None)
parser.add_argument('--safepred_intervention_mode', choices=['prediction_only', 'advisory_only', 'enforced_guard', 'distributed_enforced'], default=None)
parser.add_argument('--safepred_fail_open', action='store_true')
parser.add_argument('--safepred_model', type=str, default=None)
parser.add_argument('--safepred_base_url', type=str, default=None)
parser.add_argument('--safepred_risk_threshold', type=float, default=None)
parser.add_argument('--safepred_max_images', type=int, default=None)
parser.add_argument('--safepred_disable_response_format', action='store_true')
parser.add_argument('--online_object_sampling', type=bool, default=None)
parser.add_argument('--debug', action='store_true')

parser.add_argument('--not_eval_process_safety', action='store_true')
parser.add_argument('--not_eval_termination_safety', action='store_true')
parser.add_argument('--not_eval_awareness', action='store_true')
parser.add_argument('--not_eval_execution', action='store_true')
parser.add_argument('--prompt_setting', type=str, default='default')
parser.add_argument(
    '--experiment_seed',
    type=int,
    default=(
        int(os.environ['ISBENCH_EXPERIMENT_SEED'])
        if 'ISBENCH_EXPERIMENT_SEED' in os.environ
        else None
    ),
)
parser.add_argument(
    '--actor_temperature',
    type=float,
    default=float(os.environ.get('ISBENCH_ACTOR_TEMPERATURE', '0.0')),
)
parser.add_argument(
    '--actor_top_p',
    type=float,
    default=float(os.environ.get('ISBENCH_ACTOR_TOP_P', '1.0')),
)
parser.add_argument(
    '--actor_max_completion_tokens',
    type=int,
    default=int(os.environ.get('ISBENCH_AGENT_MAX_COMPLETION_TOKENS', '512')),
)
parser.add_argument(
    '--actor_model_revision',
    default=os.environ.get('ISBENCH_ACTOR_MODEL_REVISION', ''),
)
parser.add_argument(
    '--actor_runtime_revision',
    default=os.environ.get('ISBENCH_ACTOR_RUNTIME_REVISION', ''),
)


def _set_paired_task_seed(task: str, scene: str) -> tuple[int, int]:
    import numpy as np
    import torch

    experiment_seed = int(os.environ.get("ISBENCH_EXPERIMENT_SEED", "0"))
    task_seed = derive_task_seed(experiment_seed, task, scene)
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(task_seed)
    os.environ["ISBENCH_ACTOR_SEED"] = str(task_seed)
    os.environ["ISBENCH_SAFEPRED_SEED"] = str(task_seed)
    return experiment_seed, task_seed


def _annotate_report_seed(
    path: str,
    *,
    experiment_seed: int,
    task: str,
    scene: str,
    generation_protocol: dict,
) -> None:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ISBench report is not a JSON object: {report_path}")
    metadata = build_experiment_metadata(
        experiment_seed=experiment_seed,
        task=task,
        scene=scene,
        generation_protocol=generation_protocol,
    )
    existing = payload.get("experiment_metadata")
    if existing is not None and existing != metadata:
        raise ValueError(
            f"ISBench report generation provenance changed during the run: {report_path}"
        )
    if existing == metadata:
        return
    if _strict_trace_requested():
        raise ValueError(
            "Strict ISBench provenance must be embedded by the tracker before "
            f"the first actor request; post-hoc metadata is forbidden: {report_path}"
        )
    # Legacy/example diagnostic generation can still annotate a report after
    # the fact, but such reports fail the formal native-provenance contract.
    metadata["recorded_before_actor_call"] = False
    payload["experiment_metadata"] = metadata
    temporary = report_path.with_name(f".{report_path.name}.metadata.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(report_path)


def _strict_trace_requested() -> bool:
    return os.environ.get('ISBENCH_TRACE_STRICT', '0').lower() in {
        '1', 'true', 'yes',
    }


CLEAN_STRICT_TRACE_PROFILE = 'clean_strict_trace_v1'
ONLINE_STRICT_PROFILE = 'online_strict_v1'


def _online_strict_requested() -> bool:
    return os.environ.get('ISBENCH_ONLINE_STRICT', '0').lower() in {
        '1', 'true', 'yes',
    }


def _strict_generation_profile() -> str | None:
    clean = _strict_trace_requested()
    online = _online_strict_requested()
    if clean and online:
        raise ValueError(
            'ISBENCH_TRACE_STRICT and ISBENCH_ONLINE_STRICT are mutually exclusive'
        )
    if clean:
        return CLEAN_STRICT_TRACE_PROFILE
    if online:
        return ONLINE_STRICT_PROFILE
    return None


def _validate_strict_trace_request(
    *,
    task: str,
    scene: str,
    model: str,
    local_llm_serve: bool,
    local_serve_ip: str,
    actor_api_base: str | None = None,
    prompt_setting: str,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    safepred_safeguard: bool,
    online_object_sampling: bool,
    eval_process_safety: bool,
    eval_termination_safety: bool,
    eval_execution: bool,
    eval_awareness: bool,
) -> None:
    """Fail before simulator/model work when a formal trace is under-specified."""

    generation_validation_profile = _strict_generation_profile()
    if generation_validation_profile is None:
        return
    errors: list[str] = []
    if 'ISBENCH_EXPERIMENT_SEED' not in os.environ:
        errors.append('ISBENCH_EXPERIMENT_SEED must be explicit')
    elif int(os.environ['ISBENCH_EXPERIMENT_SEED']) < 0:
        errors.append('ISBENCH_EXPERIMENT_SEED must be non-negative')
    if not str(model or '').strip():
        errors.append('model must be explicit')
    if local_llm_serve:
        endpoint = urlsplit(str(local_serve_ip or '').strip())
        if (
            endpoint.scheme not in {'http', 'https'}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or bool(endpoint.query)
            or bool(endpoint.fragment)
        ):
            errors.append(
                'local_serve_ip must be an absolute HTTP(S) URL without '
                'credentials, query, or fragment'
            )
    else:
        resolved_actor_api_base = resolve_openai_api_base(actor_api_base)
        if not resolved_actor_api_base:
            errors.append(
                'actor_api_base is required; pass --actor_api_base or set '
                'OPENAI_API_BASE/OPENAI_BASE_URL'
            )
        else:
            endpoint_error = strict_http_endpoint_error(
                resolved_actor_api_base,
                field_name='actor_api_base',
            )
            if endpoint_error:
                errors.append(endpoint_error)
    task_path = Path(TASKS) / f'{task}.json'
    try:
        task_config = json.loads(task_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            f'cannot load strict task config {task_path}: '
            f'{type(exc).__name__}: {exc}'
        )
        task_config = {}
    scene_info = task_config.get('scene_info') if isinstance(task_config, dict) else None
    default_scene = (
        str(scene_info.get('default_scene_model') or '').strip()
        if isinstance(scene_info, dict)
        else ''
    )
    if not default_scene or scene != default_scene:
        errors.append(
            f'scene={scene!r} must equal the task default scene={default_scene!r}'
        )
    if not str(os.environ.get('ISBENCH_ACTOR_MODEL_REVISION') or '').strip():
        errors.append('ISBENCH_ACTOR_MODEL_REVISION must be explicit')
    if not str(os.environ.get('ISBENCH_ACTOR_RUNTIME_REVISION') or '').strip():
        errors.append('ISBENCH_ACTOR_RUNTIME_REVISION must be explicit')
    temperature = float(os.environ.get('ISBENCH_ACTOR_TEMPERATURE', '0.0'))
    if local_llm_serve:
        if not 0.0 < temperature <= 2.0:
            errors.append(
                'ISBENCH_ACTOR_TEMPERATURE must be in (0, 2] for '
                'independent actor seeds'
            )
    elif temperature != 0.0:
        errors.append(
            'ISBENCH_ACTOR_TEMPERATURE must be exactly 0.0 for the '
            'upstream-compatible close-source profile'
        )
    top_p = float(os.environ.get('ISBENCH_ACTOR_TOP_P', '1.0'))
    if local_llm_serve:
        if not 0.0 < top_p <= 1.0:
            errors.append('ISBENCH_ACTOR_TOP_P must be in (0, 1]')
    elif top_p != 1.0:
        errors.append(
            'ISBENCH_ACTOR_TOP_P must be exactly 1.0 for the '
            'upstream-compatible close-source profile'
        )
    max_tokens = int(
        os.environ.get('ISBENCH_AGENT_MAX_COMPLETION_TOKENS', '512')
    )
    if local_llm_serve:
        if max_tokens < 1:
            errors.append(
                'ISBENCH_AGENT_MAX_COMPLETION_TOKENS must be positive'
            )
    elif max_tokens != 512:
        errors.append(
            'ISBENCH_AGENT_MAX_COMPLETION_TOKENS must be exactly 512 for the '
            'upstream-compatible close-source profile'
        )
    if prompt_setting != 'v1':
        errors.append("prompt_setting must be 'v1'")
    if skip_viewer_obs:
        errors.append('skip_viewer_obs is forbidden')
    if use_initial_setup or use_self_caption:
        errors.append('initial-setup/self-caption actor inputs are forbidden')
    if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE:
        if safepred_safeguard:
            errors.append('SafePred safeguard must be disabled for clean source traces')
    else:
        expected_intervention = str(
            os.environ.get('ISBENCH_ONLINE_STRICT_INTERVENTION_MODE') or ''
        ).strip()
        expected_policy = str(
            os.environ.get('ISBENCH_ONLINE_STRICT_POLICY_MODE') or ''
        ).strip()
        guarded_modes = {
            'prediction_only',
            'advisory_only',
            'enforced_guard',
            'distributed_enforced',
        }
        if expected_intervention == 'no_guard':
            if safepred_safeguard:
                errors.append('online-strict no_guard must not enable SafePred safeguard')
            if expected_policy:
                errors.append('online-strict no_guard must not bind a policy mode')
            if str(os.environ.get('ISBENCH_SAFEPRED_INTERVENTION_MODE') or '').strip():
                errors.append(
                    'online-strict no_guard inherited a safeguard intervention mode'
                )
            if str(os.environ.get('ISBENCH_SAFEPRED_POLICY_MODE') or '').strip():
                errors.append('online-strict no_guard inherited a safeguard policy mode')
        elif expected_intervention in guarded_modes:
            if not safepred_safeguard:
                errors.append(
                    f'online-strict {expected_intervention} must enable SafePred safeguard'
                )
            if expected_policy not in {
                'generic_embodied_policy',
                'train_policy',
                'full_isbench_policy_text',
            }:
                errors.append('ISBENCH_ONLINE_STRICT_POLICY_MODE is unsupported')
            if str(os.environ.get('ISBENCH_SAFEPRED_INTERVENTION_MODE') or '').strip() != expected_intervention:
                errors.append('online-strict safeguard intervention differs from assignment')
            if str(os.environ.get('ISBENCH_SAFEPRED_POLICY_MODE') or '').strip() != expected_policy:
                errors.append('online-strict safeguard policy differs from assignment')
            if os.environ.get('ISBENCH_SAFEPRED_FAIL_OPEN', '0').lower() in {
                '1', 'true', 'yes',
            }:
                errors.append('online-strict safeguard must fail closed')
            safepred_model = str(
                os.environ.get('ISBENCH_SAFEPRED_MODEL') or ''
            ).strip()
            if not safepred_model or safepred_model.lower() == 'local':
                errors.append(
                    'ISBENCH_SAFEPRED_MODEL must be an explicit served model id'
                )
            if not str(
                os.environ.get('ISBENCH_SAFEPRED_MODEL_REVISION')
                or os.environ.get('WM_MODEL_REVISION')
                or ''
            ).strip():
                errors.append('ISBENCH_SAFEPRED_MODEL_REVISION must be explicit')
            safepred_endpoint_error = strict_http_endpoint_error(
                str(os.environ.get('ISBENCH_SAFEPRED_BASE_URL') or ''),
                field_name='ISBENCH_SAFEPRED_BASE_URL',
            )
            if safepred_endpoint_error:
                errors.append(safepred_endpoint_error)
        else:
            errors.append(
                'ISBENCH_ONLINE_STRICT_INTERVENTION_MODE must identify the assigned arm'
            )
    if online_object_sampling:
        errors.append('online_object_sampling is forbidden for the fixed trace cohort')
    if not (eval_process_safety and eval_termination_safety and eval_execution):
        errors.append('all native safety/execution evaluators must be enabled')
    if eval_awareness:
        errors.append('awareness evaluation must be disabled for clean v1 traces')
    if errors:
        label = (
            'Strict ISBench trace'
            if generation_validation_profile == CLEAN_STRICT_TRACE_PROFILE
            else 'ISBench online-strict'
        )
        raise ValueError(f'{label} request is invalid: ' + '; '.join(errors))


@action_resume_scope
def online_benchmark_once(
    try_id,
    task: str,
    scene: str,
    model: str,
    local_llm_serve: str, 
    local_serve_ip: str,
    local_serve_key: str,
    work_dir: str,
    draw_bbox_2d: bool,
    skip_viewer_obs: bool,
    use_initial_setup: bool,
    use_self_caption: bool,
    safepred_safeguard: bool,
    safepred_max_regenerations: int | None,
    online_object_sampling: bool,
    debug: bool,
    eval_process_safety: bool,
    eval_termination_safety: bool,
    eval_awareness: bool,
    eval_execution: bool,
    prompt_setting: str,
    actor_api_base: str | None = None,
    _failure_context: dict | None = None,
):
    failure_context = _failure_context if _failure_context is not None else {}
    if action_resume_enabled() and (
        online_object_sampling or eval_awareness or use_self_caption
        or prompt_setting == 'v2' or not (model or local_llm_serve)
        or not (eval_process_safety and eval_termination_safety and eval_execution)
    ):
        raise ValueError('action recovery currently requires fixed-scene online actions with awareness/caption off')
    failure_context['stage'] = 'strict_request_validation'
    _validate_strict_trace_request(
        task=task,
        scene=scene,
        model=model,
        local_llm_serve=bool(local_llm_serve),
        local_serve_ip=local_serve_ip,
        actor_api_base=actor_api_base,
        prompt_setting=prompt_setting,
        skip_viewer_obs=skip_viewer_obs,
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        safepred_safeguard=safepred_safeguard,
        online_object_sampling=bool(online_object_sampling),
        eval_process_safety=eval_process_safety,
        eval_termination_safety=eval_termination_safety,
        eval_execution=eval_execution,
        eval_awareness=eval_awareness,
    )
    # Keep strict request validation ahead of simulator, benchmark, and model
    # imports so malformed formal assignments fail without touching GPU/native
    # runtime or constructing any endpoint client.
    failure_context['stage'] = 'simulator_initialization'
    from og_ego_prim.utils.simulator_runtime import configure_simulator_runtime, mark_simulator_ready

    configure_simulator_runtime()
    from og_ego_prim.utils.monkey_patch import add_monkey_patch

    add_monkey_patch()
    import omnigibson as og
    from omnigibson.macros import gm
    from og_ego_prim.benchmark import build_benchmark
    from og_ego_prim.models import PlanningAgent

    # Don't use GPU dynamics and use flatcache for performance boost.
    gm.USE_GPU_DYNAMICS = True
    experiment_seed, task_seed = _set_paired_task_seed(task, scene)
    benchmark = build_benchmark(
        task=task, 
        scene=scene, 
        ego_view=True,
        draw_bbox_2d=draw_bbox_2d,
        use_initial_setup=use_initial_setup,
        use_self_caption=use_self_caption,
        online_object_sampling=online_object_sampling, 
        debug=debug,
        eval_process_safety=eval_process_safety,
        eval_termination_safety=eval_termination_safety,
        eval_awareness=eval_awareness,
        eval_execution=eval_execution,
    )
    mark_simulator_ready()
    failure_context['tracker'] = benchmark.tracker
    if debug and gm.HEADLESS is False:
        og.sim.enable_viewer_camera_teleoperation()

    benchmark_tag = f'{benchmark.task_name}___{benchmark.scene_name}'
    model_tag = model.replace('/', '__') if model is not None else 'example'
    
    if not try_id:
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, model_tag)
    else: 
        output_dir = os.path.join(work_dir, 'benchmark', benchmark_tag, f"{try_id}_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)

    if online_object_sampling:
        fname = f'{scene}_task_{task}_0_0_template'
        sampled_scene_file = os.path.join(output_dir, f'{fname}.json')
        benchmark.env.task.save_task(path=sampled_scene_file)

    generation_protocol = None
    failure_context['stage'] = 'actor_initialization'
    if model or local_llm_serve:
        if safepred_max_regenerations is not None:
            os.environ['ISBENCH_SAFEPRED_MAX_REGENERATIONS'] = str(safepred_max_regenerations)
        use_obs = not skip_viewer_obs
        agent = PlanningAgent(
            task_name=task, 
            scene_name=scene, 
            agent_name=model,
            work_dir=work_dir,
            local_llm_serve=local_llm_serve, 
            local_serve_ip=local_serve_ip,  
            actor_api_base=actor_api_base,
            local_serve_key=local_serve_key, 
            debug=debug,
            prompt_setting=prompt_setting,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            safepred_safeguard=safepred_safeguard,
        )
        agent.set_tracker(benchmark.tracker)
        actor_protocol = agent.client.generation_protocol()
        generation_protocol = build_generation_protocol(
            task_config_path=Path(TASKS) / f'{task}.json',
            camera_config_path=Path(CAMERAS) / 'camera.json',
            package_root=Path(__file__).resolve().parents[1],
            model_requested=model,
            model_resolved=actor_protocol['model_resolved'],
            provider_mode=actor_protocol['provider_mode'],
            model_revision=os.environ.get('ISBENCH_ACTOR_MODEL_REVISION', ''),
            api_base=(
                local_serve_ip
                if local_llm_serve
                else resolve_openai_api_base(actor_api_base)
            ),
            runtime_revision=os.environ.get('ISBENCH_ACTOR_RUNTIME_REVISION', ''),
            prompt_setting=prompt_setting,
            actor_temperature=actor_protocol['actor_temperature'],
            actor_top_p=actor_protocol['actor_top_p'],
            actor_max_completion_tokens=actor_protocol[
                'actor_max_completion_tokens'
            ],
            actor_max_images=actor_protocol['actor_max_images'],
            actor_retry_seed_policy=actor_protocol[
                'actor_retry_seed_policy'
            ],
            actor_transport_retry_policy=actor_protocol[
                'actor_transport_retry_policy'
            ],
            actor_transport_max_retries=actor_protocol[
                'actor_transport_max_retries'
            ],
            actor_transport_retry_base_delay_seconds=actor_protocol[
                'actor_transport_retry_base_delay_seconds'
            ],
            disable_thinking=actor_protocol['disable_thinking'],
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            draw_bbox_2d=draw_bbox_2d,
            skip_viewer_obs=skip_viewer_obs,
            online_object_sampling=bool(online_object_sampling),
            eval_process_safety=eval_process_safety,
            eval_termination_safety=eval_termination_safety,
            eval_awareness=eval_awareness,
            eval_execution=eval_execution,
        )
        benchmark.tracker.set_experiment_metadata(
            build_experiment_metadata(
                experiment_seed=experiment_seed,
                task=task,
                scene=scene,
                generation_protocol=generation_protocol,
            )
        )
        planner = agent.step(use_obs=use_obs, max_step=(len(benchmark._example_planning) + 10))
    else:
        use_obs = False
        planner = benchmark.get_example_planning()

    recovery = None
    restored = False
    if action_resume_enabled():
        from og_ego_prim.utils.action_recovery import ActionRecovery
        failure_context['stage'] = 'action_checkpoint_restore'
        recovery = ActionRecovery.for_native(
            os.environ['ISBENCH_ACTION_CHECKPOINT_DIR'], benchmark, agent, og.sim, output_dir)
        restored = recovery.restore()
        failure_context['action_recovery'] = recovery

    failure_context['stage'] = 'initial_observation'
    if not skip_viewer_obs and not restored:
        benchmark.get_surrounding_viewer_obs(save_img=os.path.join(output_dir, '0_init'))
    if use_self_caption:
        caption = agent.generate_caption(use_obs=use_obs)
        benchmark.tracker.track_caption(
            content=caption
        )
    if eval_awareness and (model or local_llm_serve):
        failure_context['stage'] = 'awareness_evaluation'
        awareness = agent.generate_awareness(use_obs=use_obs)
        benchmark.evaluate_awareness(awareness)
    elif prompt_setting == 'v2':
        awareness = agent.generate_awareness(use_obs=use_obs)
        benchmark.tracker.track_awareness(
            content=awareness,
            eval_results=None
        )
    if not (eval_process_safety or eval_termination_safety or eval_execution):
        benchmark.tracker.save_tracking(os.path.join(output_dir, 'report_awareness.json'))
        time.sleep(3)
        if os.environ.get("ISBENCH_SKIP_OG_CLEAR_AFTER_REPORT") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        og.clear()
        return

    def observe_action(step, plan):
        step_tag = f'{step}_' + plan['action'].replace('(', '__').replace(')', '__')
        failure_context['stage'] = 'observation_capture'
        if not skip_viewer_obs:
            benchmark.get_surrounding_viewer_obs(save_img=os.path.join(output_dir, step_tag))
        failure_context['stage'] = 'actor_planning'

    failure_context['stage'] = 'actor_planning'
    if recovery is not None:
        if not restored:
            recovery.commit('boundary')
        recovery.run(observe_action, use_obs, len(benchmark._example_planning) + 10,
                     failure_context=failure_context)
    else:
        for i, plan in enumerate(planner):
            failure_context['stage'] = 'action_execution_and_evaluation'
            if benchmark.execute_plan(plan) is False:
                break
            observe_action(i + 1, plan)
        failure_context['stage'] = 'termination_evaluation'
        benchmark.termination_evaluation()
    failure_context['stage'] = 'report_persistence'
    report_path = os.path.join(output_dir, 'report.json')
    if recovery is not None:
        benchmark.tracker.action_recovery = {
            'contract': recovery.store.identity['contract'],
            'restores': recovery.events,
            'saved_phase': recovery.phase,
            'physics_trajectory_equivalence': ('operation_prefix_validated_on_restore'
                if recovery.events and hasattr(recovery.adapter, 'journal') else 'not_qualified'),
        }
    benchmark.tracker.save_tracking(report_path)
    if generation_protocol is None:
        generation_protocol = build_generation_protocol(
            task_config_path=Path(TASKS) / f'{task}.json',
            camera_config_path=Path(CAMERAS) / 'camera.json',
            package_root=Path(__file__).resolve().parents[1],
            model_requested=model or 'example',
            model_resolved=model or 'example',
            provider_mode='example',
            model_revision=os.environ.get('ISBENCH_ACTOR_MODEL_REVISION', ''),
            api_base=(
                local_serve_ip
                if local_llm_serve
                else resolve_openai_api_base(actor_api_base)
            ),
            runtime_revision=os.environ.get('ISBENCH_ACTOR_RUNTIME_REVISION', ''),
            prompt_setting=prompt_setting,
            actor_temperature=float(
                os.environ.get('ISBENCH_ACTOR_TEMPERATURE', '0.0')
            ),
            actor_top_p=float(os.environ.get('ISBENCH_ACTOR_TOP_P', '1.0')),
            actor_max_completion_tokens=int(
                os.environ.get('ISBENCH_AGENT_MAX_COMPLETION_TOKENS', '512')
            ),
            actor_max_images=int(
                os.environ.get('ISBENCH_ACTOR_MAX_IMAGES', '0')
            ),
            actor_retry_seed_policy='none',
            actor_transport_retry_policy=actor_transport_retry_policy(),
            actor_transport_max_retries=actor_transport_max_retries(),
            actor_transport_retry_base_delay_seconds=(
                actor_transport_retry_base_delay_seconds()
            ),
            disable_thinking=False,
            use_initial_setup=use_initial_setup,
            use_self_caption=use_self_caption,
            draw_bbox_2d=draw_bbox_2d,
            skip_viewer_obs=skip_viewer_obs,
            online_object_sampling=bool(online_object_sampling),
            eval_process_safety=eval_process_safety,
            eval_termination_safety=eval_termination_safety,
            eval_awareness=eval_awareness,
            eval_execution=eval_execution,
        )
    _annotate_report_seed(
        report_path,
        experiment_seed=experiment_seed,
        task=task,
        scene=scene,
        generation_protocol=generation_protocol,
    )
    
    if online_object_sampling:
        if benchmark.tracker.termination['reason'] == 'done' and benchmark.tracker.goal_condition['execution_goal_condition']['eval']: 
            normal_scene_file = os.path.join(work_dir, "..", "data", "scenes", scene, "json", f'{fname}.json')
            shutil.copyfile(sampled_scene_file, normal_scene_file)
        else:
            os.remove(sampled_scene_file)

    # Isaac Sim / OmniGibson can segfault while unloading native plugins during
    # ``og.clear()`` after a fully written report.  In strict trace mode this
    # cleanup is outside the scientific observation/actor/evaluation path: the
    # parent process owns the attempt directory and removes it after validating
    # and promoting the report.  Allow the launcher to request an immediate,
    # clean process exit after the report has been atomically annotated, while
    # retaining the historical graceful cleanup by default.
    if os.environ.get("ISBENCH_SKIP_OG_CLEAR_AFTER_REPORT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    time.sleep(3)
    og.clear()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.experiment_seed is not None:
        os.environ['ISBENCH_EXPERIMENT_SEED'] = str(args.experiment_seed)
    elif _strict_generation_profile() is None:
        os.environ.setdefault('ISBENCH_EXPERIMENT_SEED', '0')
    os.environ['ISBENCH_ACTOR_TEMPERATURE'] = str(args.actor_temperature)
    os.environ['ISBENCH_ACTOR_TOP_P'] = str(args.actor_top_p)
    os.environ['ISBENCH_AGENT_MAX_COMPLETION_TOKENS'] = str(
        args.actor_max_completion_tokens
    )
    os.environ['ISBENCH_ACTOR_MODEL_REVISION'] = str(args.actor_model_revision)
    os.environ['ISBENCH_ACTOR_RUNTIME_REVISION'] = str(
        args.actor_runtime_revision
    )
    if args.safepred_policy_mode is not None:
        os.environ['ISBENCH_SAFEPRED_POLICY_MODE'] = args.safepred_policy_mode
    if args.safepred_intervention_mode is not None:
        os.environ['ISBENCH_SAFEPRED_INTERVENTION_MODE'] = args.safepred_intervention_mode
    if args.safepred_fail_open:
        os.environ['ISBENCH_SAFEPRED_FAIL_OPEN'] = '1'
    if args.safepred_model is not None:
        os.environ['ISBENCH_SAFEPRED_MODEL'] = args.safepred_model
    if args.safepred_base_url is not None:
        os.environ['ISBENCH_SAFEPRED_BASE_URL'] = args.safepred_base_url
    if args.safepred_risk_threshold is not None:
        os.environ['ISBENCH_SAFEPRED_RISK_THRESHOLD'] = str(args.safepred_risk_threshold)
    if args.safepred_max_images is not None:
        os.environ['ISBENCH_SAFEPRED_MAX_IMAGES'] = str(args.safepred_max_images)
    if args.safepred_disable_response_format:
        os.environ['ISBENCH_SAFEPRED_DISABLE_RESPONSE_FORMAT'] = '1'
    run_with_failure_evidence(
        online_benchmark_once,
        try_id=args.try_id,
        task=args.task,
        scene=args.scene,
        model=args.model,
        local_llm_serve=args.local_llm_serve,
        local_serve_ip=args.local_serve_ip,
        actor_api_base=args.actor_api_base,
        local_serve_key=_actor_api_key_from_environment(),
        prompt_setting=args.prompt_setting,
        work_dir=args.work_dir,
        draw_bbox_2d=args.draw_bbox_2d,
        skip_viewer_obs=args.skip_viewer_obs,
        use_initial_setup=args.use_initial_setup,
        use_self_caption=args.use_self_caption,
        safepred_safeguard=args.safepred_safeguard,
        safepred_max_regenerations=args.safepred_max_regenerations,
        online_object_sampling=args.online_object_sampling,
        debug=args.debug,
        eval_process_safety=(not args.not_eval_process_safety),
        eval_termination_safety=(not args.not_eval_termination_safety),
        eval_awareness=(not args.not_eval_awareness),
        eval_execution=(not args.not_eval_execution),
    )
