#!/usr/bin/env python3
"""Convenience layer over the unchanged native HA matrix (no hidden model calls)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rq', choices=('rq1', 'rq3'))
    p.add_argument('--profile', type=Path, default=root/'local/profile.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['dl-t4'])
    p.add_argument('--all-nonsde', action='store_true', help='Use all 200 non-SDE tasks in this source snapshot (not a historical 192-task cohort)')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    output = a.output.expanduser().resolve()
    if output == root or output.is_relative_to(root):
        p.error('Choose an output directory outside this immutable distribution')
    tasks = (root/'inputs/ha_nonsde_tasks.txt').read_text().split() if a.all_nonsde else a.tasks
    env = dict(os.environ)
    env['HARNESS_ENFORCED_CONTROL_CONTRACT'] = 'legacy_online_v16'
    env['SAFEPRED_LOG_ROOT'] = str(output/'logs')
    args = [sys.executable, '-B', str(root/'run.py'), a.rq, 'ha']
    if not a.dry_run or a.profile.exists():
        args += ['--model-profile', str(a.profile.resolve())]
    args += ['--', '--tasks', *tasks, '--policy-modes', 'full_harnessaudit_rules',
             '--intervention-modes', *(['no_guard', 'advisory_only', 'enforced_guard'] if a.rq == 'rq1' else ['distributed_enforced']),
             '--experiment-seeds', '0', '--prediction-horizon', '1', '--candidate-count', '1',
             '--max-regeneration-attempts', '1', '--max-turns', '24', '--workers', str(a.workers),
             '--continue-on-error', '--trace-resume', '--resource-semantics-mode', 'legacy_official_v1',
             '--safepred-config', str(root/'source/SafePred/config/harnessaudit_online_openai_compatible.yaml'),
             '--output-root', str(output/'results'), '--trace-root', str(output/'traces'), '--analysis-dir', str(output/'analysis')]
    if a.rq == 'rq3':
        args += ['--wm-return-perturbations', 'empty', 'irrelevant', 'filler', 'corrupt', 'corrupt_key_random_chars', 'environment_h1',
                 '--wm-return-signal-source-policy', 'canonical_wm_all_v1']
    if a.dry_run:
        args += ['--dry-run']
    return subprocess.call(args, env=env, cwd=root)


if __name__ == '__main__':
    raise SystemExit(main())
