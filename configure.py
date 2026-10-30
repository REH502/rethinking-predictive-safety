#!/usr/bin/env python3
"""Generate a credential-free five-role profile; never request or save keys."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor-model')
    p.add_argument('--wm-model')
    p.add_argument('--judge-model')
    p.add_argument('--base-url', help='OpenAI-compatible base URL, including /v1 when required')
    p.add_argument('--wm-base-url')
    p.add_argument('--judge-base-url')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'local/profile.yaml')
    a = p.parse_args()
    actor = a.actor_model or input('Actor exact model ID: ').strip()
    wm = a.wm_model or input('WM exact model ID [same as actor]: ').strip() or actor
    judge = a.judge_model or input('Evaluator exact model ID [same as actor]: ').strip() or actor
    url = a.base_url or input('API base URL (no key): ').strip()
    endpoints = {}
    for role, value in [('actor', url), ('wm', a.wm_base_url or url), ('judge', a.judge_base_url or url)]:
        parsed = urlsplit(value)
        if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            p.error('Use a plain HTTP(S) endpoint without credentials, query, or fragment')
        endpoints[role] = {'base_url': value.rstrip('/'), 'api_key_env': role.upper()+'_API_KEY', 'api_mode': 'chat_completions'}
    if not all((actor, wm, judge)):
        p.error('Exact model IDs are required')
    roles = {}
    for role, endpoint, model in [('actor', 'actor', actor), ('world_model', 'wm', wm), ('judge', 'judge', judge), ('policy_extractor', 'wm', wm), ('action_agent', 'actor', actor)]:
        roles[role] = {'endpoint': endpoint, 'model': model, 'revision': 'user-supplied-unverified', 'runtime_revision': 'openai-compatible-api'}
    config = {'schema_version': 'model-stack-profile-v1', 'defaults': {'temperature': 0.0, 'max_tokens': 8192, 'timeout_seconds': 360, 'transport_max_attempts': 1}, 'endpoints': endpoints, 'roles': roles}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print('Saved a profile without credentials. Set ACTOR_API_KEY, WM_API_KEY, JUDGE_API_KEY externally.')
    print('Before formal use: verify exact model revision, sampling parameters and budgets in this profile.')


if __name__ == '__main__':
    main()
