#!/usr/bin/env python3
"""Exercise real entrypoints, native dry-run matrices and deterministic regression tests; zero model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'source'
    ha = source/'harness_env/orgin/HarnessAudit'
    with tempfile.TemporaryDirectory(prefix='wm-bundle-smoke-') as temporary:
        env = dict(os.environ)
        # No preexisting API credentials or project dotenv can affect the check.
        for key in list(env):
            if any(s in key.upper() for s in ('API_KEY', 'ACCESS_TOKEN')) or key.startswith(('HARNESS_', 'SAFEPRED_', 'WM_SAFETY_')):
                env.pop(key)
        env.update(PYTHONPATH=os.pathsep.join(map(str, [source, ha, source/'SafePred', source/'IS-Bench'])),
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=temporary,
                   WM_SAFETY_ENV_FILE=str(Path(temporary)/'does-not-exist.env'), OPENAI_AGENTS_DISABLE_TRACING='1',
                   SAFEPRED_LOG_ROOT=str(Path(temporary)/'logs'))
        profile = Path(temporary)/'generated-profile.yaml'
        subprocess.run([sys.executable, '-B', str(root/'configure.py'), '--actor-model', 'offline-smoke-actor',
                        '--wm-model', 'offline-smoke-wm', '--judge-model', 'offline-smoke-judge',
                        '--base-url', 'https://example.invalid/v1', '--output', str(profile)],
                       env=env, check=True, stdout=subprocess.DEVNULL, timeout=20)
        subprocess.run([sys.executable, '-B', '-c',
                        'import sys; from experiments.model_stack import load_model_stack_profile; '
                        'p=load_model_stack_profile(sys.argv[1], environ={}, require_credentials=False); '
                        'assert len(p.roles)==5', str(profile)], env=env, check=True, timeout=20)
        for rq in ('rq1', 'rq2', 'rq3'):
            for benchmark in ('ha', 'isbench'):
                subprocess.run([sys.executable, '-B', str(root/'run.py'), rq, benchmark, '--', '--help'], env=env, stdout=subprocess.DEVNULL, check=True, timeout=90)
        for rq, count in [('rq1', 3), ('rq3', 6)]:
            out = Path(temporary)/rq
            subprocess.run([sys.executable, '-B', str(root/'quickstart.py'), rq, '--dry-run', '--profile', str(Path(temporary)/'absent.yaml'), '--output', str(out)], env=env, check=True, stdout=subprocess.DEVNULL, timeout=90)
            matches = list(out.rglob('run_manifest.jsonl'))
            if not matches:
                raise AssertionError('Native dry-run did not produce a manifest')
            rows = [json.loads(line) for line in matches[0].read_text().splitlines() if line.strip()]
            assert len(rows) == count and all(row['status'] == 'dry_run' for row in rows), 'Unexpected dry-run condition inventory'
            print(f'PASS: {rq} native dry-run planned {count} conditions; no model calls')
        names = ['test_model_stack', 'test_endpoint_preflight', 'test_endpoint_preflight_bounded_retry',
                 'test_wm_return_perturbations', 'test_isbench_sixperturb_plan', 'test_isbench_episode_checkpoint']
        tests = [str(source/'experiments/tests'/(name+'.py')) for name in names]
        tests += [str(ha/'tests'/(name+'.py')) for name in ['test_trace_checkpoint', 'test_runtime_checkpoint', 'test_environment_h1', 'test_environment_h1_report_delivery', 'test_tool_necessity_model_input_boundary']]
        subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], env=env, cwd=source, check=True, timeout=240)
    print('PASS: offline smoke suite; temporary test artifacts removed; no scientific results included')


if __name__ == '__main__':
    main()
