"""Explicit IS-Bench perturbation study; never expands the old formal grid.

The sixth condition is a specification, not a fake perfect WM prediction.
It remains non-launchable until native isolated-preview qualification exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

TEXT_MODES = ('empty', 'irrelevant', 'filler', 'corrupt', 'corrupt_key_random_chars')
SIX_MODES = (*TEXT_MODES, 'environment_h1')


def validate_runtime_mode(mode: str) -> None:
    if mode == 'environment_h1':
        raise ValueError('IS-Bench environment_h1 requires a qualified isolated simulator preview/review/commit adapter; ordinary checkpoint restore is not sufficient')
    if mode not in ('none', *TEXT_MODES):
        raise ValueError('unsupported IS-Bench perturbation; historical HA checker-oracle is not transferable')


def build_plan(tasks: list[str], *, model_profile: str, baseline_settings: dict) -> dict:
    if not tasks or len(tasks) != len(set(tasks)) or any(not isinstance(t, str) or not t for t in tasks):
        raise ValueError('nonempty unique task IDs required')
    if not model_profile or not baseline_settings:
        raise ValueError('bind model profile and actual IS-Bench baseline settings explicitly')
    # Preserve IS-Bench settings, not HA's unrelated action budgets or physics.
    settings = json.loads(json.dumps(baseline_settings, sort_keys=True))
    cells = [{'task': task, 'seed': 0, 'mode': mode, 'intervention':'distributed_enforced',
              'execution_ready': mode != 'environment_h1'} for task in tasks for mode in SIX_MODES]
    return dict(schema_version='isbench-sixperturb-design-v1', classification='unevaluable',
        task_count=len(tasks), condition_count=len(cells), seeds=[0], modes=list(SIX_MODES),
        model_profile=model_profile, baseline_settings=settings,
        baseline_settings_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        source_policy='canonical_wm_all_v1',
        gate_policy='preserve canonical risk for empty/irrelevant/filler/key; invert risk only for corrupt',
        feedback_sources={m: ('isolated native environment' if m=='environment_h1' else 'validated learned WM') for m in SIX_MODES},
        environment_h1=dict(status='requires_native_implementation_and_GPU_qualification', horizon=1,
            sequence=['capture main state and RNG', 'execute candidate in independent simulator state',
                'return observed action result only, without hidden evaluator labels',
                'actor keeps or revises candidate', 'execute chosen action once in main environment'],
            required_gates=['main state and RNG unchanged by preview', 'no preview actions in real trajectory or score',
                'identical initial scene, objects, controller and physics state',
                'crash-safe preview/review/commit checkpoint identity',
                'unsupported scenes fail explicitly, never substitute a safety label'],
            comparison_limit='mandatory actor review and preview compute differ from learned-WM baseline'),
        failure_policy='isolate per episode; preserve failed attempts; no cross-task skip',
        resume_policy='same-identity native operation checkpoints only; evaluator-only recovery when actor complete; no budget reset',
        scoring='native IS-Bench metrics only; valid-only and paired-valid summaries alongside full planned denominator and exclusions',
        analysis=['perturbation fidelity and risk-gate counts', 'actor changes after feedback',
            'observed safety and task completion, not inferred from WM confidence', 'tokens by actor/WM/evaluator',
            'transport failures and evaluation coverage', 'paired comparisons and uncertainty; do not select tasks for desired scores'],
        cells=cells)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', type=Path, help='Sealed existing IS-Bench baseline plan')
    parser.add_argument('--stack', default='gemini')
    parser.add_argument('--task-file', type=Path, help='JSON list of exact baseline task IDs')
    parser.add_argument('--baseline-settings', type=Path, help='JSON object of frozen baseline runtime settings')
    parser.add_argument('--model-profile', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.parent_plan:
        if any((args.task_file,args.baseline_settings,args.model_profile)):
            parser.error('parent-plan cannot be combined with manual baseline overrides')
        seal_path=args.parent_plan.parent/'plan_seal.json'
        seal=json.loads(seal_path.read_text())
        if (str(args.parent_plan.resolve()) != seal['path'] or
            hashlib.sha256(args.parent_plan.read_bytes()).hexdigest()!=seal['sha256']):
            raise ValueError('baseline plan seal mismatch')
        parent=json.loads(args.parent_plan.read_text()); stack=parent['stacks'][args.stack]
        profile=Path(stack['profile']).resolve(strict=True)
        if hashlib.sha256(profile.read_bytes()).hexdigest()!=stack['profile_binding']['sha256']:
            raise ValueError('baseline model profile drift')
        settings={'shared_settings':parent['shared_settings'],
            **{k:parent[k] for k in ('actor_output_contract','task_timeout_seconds',
                'action_resume_contract','action_timeout_mode','action_stall_timeout_seconds',
                'native_transport_max_attempts','relay_transport_max_attempts') if k in parent}}
        plan=build_plan(stack['tasks'],model_profile=str(profile),baseline_settings=settings)
        plan['reference_runtime_binding']=stack['runtime_binding']
        plan['reference_only_not_runtime_readiness']=True
        inputs=(args.parent_plan,seal_path,profile,Path(__file__))
    else:
        if not all((args.task_file,args.baseline_settings,args.model_profile)):
            parser.error('provide parent-plan or all three manual baseline inputs')
        profile=args.model_profile.resolve(strict=True)
        plan=build_plan(json.loads(args.task_file.read_text()), model_profile=str(profile),
            baseline_settings=json.loads(args.baseline_settings.read_text()))
        inputs=(args.task_file,args.baseline_settings,profile,Path(__file__))
    plan['input_bindings']=[{'path':str(p.resolve()), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in inputs]
    with args.output.open('x') as handle:
        json.dump(plan,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'conditions':plan['condition_count'],'learned_wm_conditions':5*plan['task_count'],
        'environment_h1_planned_not_launchable':plan['task_count']}))


if __name__ == '__main__': main()
