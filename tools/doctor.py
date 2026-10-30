#!/usr/bin/env python3
"""Offline input/import checks. Does not contact model endpoints or start a simulator."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source'
HA = SOURCE/'harness_env/orgin/HarnessAudit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isbench', action='store_true', help='Also check optional simulator dependencies without launching a scene')
    args = parser.parse_args()
    sys.path[:0] = [str(SOURCE), str(HA), str(SOURCE/'SafePred')]
    missing = []
    for module in ('yaml', 'dotenv', 'openai', 'agents', 'httpx', 'pydantic', 'numpy', 'pandas', 'matplotlib', 'PIL', 'pytest'):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        print('Missing Python dependencies: '+', '.join(missing))
        return 1
    from multi_agent.loader import load_task, load_tool_catalog
    tasks = sorted((HA/'multi_agent/tasks').rglob('*.yaml'))
    domains = set()
    for path in tasks:
        task = load_task(path)
        domains.add(task.domain)
    for domain in domains:
        catalog = load_tool_catalog(domain)
        for tool in catalog.tools:
            if tool.fixture and not (HA/'multi_agent/fixtures'/domain/tool.fixture).is_file():
                raise FileNotFoundError('Missing static fixture: '+tool.fixture)
    assert len(tasks) == 210, 'HA task inventory differs from packaged snapshot'
    isroot = SOURCE/'IS-Bench'
    istasks = sorted((isroot/'data/tasks').glob('*.json'))
    assert len(istasks) == 161, 'IS-Bench task inventory differs'
    for p in istasks:
        json.loads(p.read_text())
    assert len(list((isroot/'data/scenes').rglob('*.json'))) == 563
    assert (isroot/'og_ego_prim/benchmark/OpenSans-Bold.ttf').is_file()
    print(f'PASS: HA {len(tasks)} task definitions/assets, {len(domains)} tool catalogs; IS-Bench {len(istasks)} task definitions + 563 scene templates')
    if args.isbench:
        from importlib.util import find_spec
        absent = [m for m in ('omnigibson', 'bddl', 'torch', 'transformers') if find_spec(m) is None]
        if absent:
            print('IS-Bench environment incomplete: '+', '.join(absent))
            return 1
        print('IS-Bench modules found; GPU/driver/scene startup still require a real simulator check.')
    print('No API credentials or network needed for these checks. This is not a live inference validation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
