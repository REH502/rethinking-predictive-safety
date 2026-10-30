#!/usr/bin/env python3
"""Collect newly generated HA reports, never select by experimental score."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    found = []
    for path in sorted(a.results.resolve().rglob('*.json')):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get('task_id') and data.get('trace_file'):
            trace = Path(data['trace_file'])
            trace = trace if trace.is_absolute() else path.parent/trace
            if not trace.is_file():
                p.error('A source report references a missing trace; fix the input before collecting reports')
            found.append(str(path))
    if not found:
        p.error('No native reports with task_id and trace_file; supply the results directory of a fresh run')
    with a.output.open('x', encoding='utf-8') as f:
        f.write('\n'.join(found)+'\n')
    print(f'Listed {len(found)} source reports without score filtering. Audit trace completeness before prediction.')


if __name__ == '__main__':
    main()
