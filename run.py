#!/usr/bin/env python3
"""Route RQ / benchmark to unchanged native experiments, with logs outside the distribution."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rq', choices=('rq1', 'rq2', 'rq3'))
    parser.add_argument('benchmark', choices=('ha', 'isbench'))
    parser.add_argument('--model-profile', type=Path)
    parser.add_argument('--python', default=sys.executable, help='Native benchmark interpreter')
    parser.add_argument('--print-command', action='store_true', help='Print a redacted launch plan without running')
    argv = sys.argv[1:]
    index = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:index])
    forwarded = argv[index+1:] if index < len(argv) else []
    if args.rq == 'rq2' and args.model_profile is not None:
        parser.error('predict-only uses native --model / --api-url and environment credentials, not --model-profile')
    root = Path(__file__).resolve().parent
    source = root/'source'
    workflow = args.benchmark + ('-offline' if args.rq == 'rq2' else '-online')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(source)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('SAFEPRED_LOG_ROOT', str(root.parent/'wm-runtime-logs'))
    env.setdefault('WM_SAFETY_HA_PYTHON', args.python)
    env.setdefault('WM_SAFETY_ISBENCH_PYTHON', args.python)
    command = [sys.executable, '-m', 'experiments', 'run', workflow, '--python', args.python]
    if args.model_profile:
        command += ['--model-profile', str(args.model_profile.expanduser().resolve())]
    if args.print_command:
        command += ['--print-command']
    command += ['--', *forwarded]
    return subprocess.call(command, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
