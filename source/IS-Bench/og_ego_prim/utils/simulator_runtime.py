"""Opt-in, per-attempt writable Kit paths and body-free startup handshake.

Call after request validation but before any OmniGibson import: SimulationApp
forwards unknown command-line arguments to Kit, and Warp reads WARP_CACHE_PATH
when initialized. This changes no renderer, GPU, asset, or benchmark setting.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time


_TOKENS = (
    'data', 'cache', 'omni_cache', 'logs', 'omni_global_data',
    'omni_global_cache', 'omni_global_config', 'omni_global_logs',
)
_SETTINGS = tuple(f'--/app/tokens/{token}=' for token in _TOKENS) + (
    '--/app/userConfigPath=', '--/log/file=',
)


def _write_startup_state(environment, *, ready):
    path_value = environment.get('ISBENCH_SIM_STARTUP_STATE_PATH')
    if not environment.get('ISBENCH_SIM_RUNTIME_ROOT') or not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = {
        'schema_version': 1,
        'pid': os.getpid(),
        'stage': 'ready' if ready else 'starting',
        'ready': ready,
        'updated_monotonic': time.monotonic(),
        'updated_at_unix': time.time(),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configure_simulator_runtime(*, environment=None, argv=None):
    """Configure isolated writable paths when explicitly enabled; otherwise no-op."""
    environment = os.environ if environment is None else environment
    argv = sys.argv if argv is None else argv
    root_value = environment.get('ISBENCH_SIM_RUNTIME_ROOT')
    if not root_value:
        return
    root = Path(root_value).resolve()
    paths = {token: root / token for token in _TOKENS}
    # Kit's ${config} means build configuration (e.g. release), not a directory.
    paths['config'] = root / 'config'
    paths['warp'] = root / 'warp'
    paths['state'] = root / 'state'
    cache_root_value = environment.get('ISBENCH_SIM_CACHE_ROOT')
    if cache_root_value:
        # The runner assigns one cache root per sequential worker. Task data,
        # configuration, logs and startup evidence remain attempt-local.
        cache_root = Path(cache_root_value).resolve()
        for token in ('cache', 'omni_global_cache', 'warp'):
            paths[token] = cache_root / token
    # GPU foundation uses ${omni_cache} for both writable RTX shader stores.
    # The general ${cache} token alone does not override that portable alias.
    paths['omni_cache'] = paths['cache']
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Replace only managed writable-path options, making repeated calls stable.
    # In particular, preserve every task, scene, seed, renderer and GPU argument.
    retained = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == '--portable-root':
            index += 2
            continue
        if not (argument == '--portable' or argument.startswith('--portable-root=')
                or argument.startswith(_SETTINGS)):
            retained.append(argument)
        index += 1
    # SimulationApp detects this two-argument form and does not append --portable.
    retained.extend(['--portable-root', str(root)])
    retained.extend(f'--/app/tokens/{token}={paths[token]}' for token in _TOKENS)
    retained.extend([
        f'--/app/userConfigPath={paths["config"] / "user.config.json"}',
        f'--/log/file={paths["logs"] / f"kit-{os.getpid()}.log"}',
    ])
    extension_seed = environment.get('ISBENCH_SIM_EXTENSION_SEED')
    if extension_seed:
        seed = Path(extension_seed).resolve()
        if not (seed / 'extension_seed.json').is_file():
            raise ValueError('verified offline extension seed is missing')
        # Packages are byte-verified by the launcher. This adds the exact
        # previously loaded versions, without replacing installed core packages.
        # No missing dependency may silently trigger a remote download.
        retained = [x for x in retained if not x.startswith('--/app/extensions/registryEnabled=')]
        if not any(retained[i:i + 2] == ['--ext-folder', str(seed)] for i in range(len(retained))):
            retained.extend(['--ext-folder', str(seed)])
        retained.append('--/app/extensions/registryEnabled=false')
    argv[:] = retained
    environment.update({
        'WARP_CACHE_PATH': str(paths['warp']),
        'XDG_CACHE_HOME': str(paths['cache']),
        'XDG_CONFIG_HOME': str(paths['config']),
        'XDG_DATA_HOME': str(paths['data']),
        'XDG_STATE_HOME': str(paths['state']),
    })
    _write_startup_state(environment, ready=False)


def mark_simulator_ready(*, environment=None):
    """Publish readiness only after the full benchmark environment was built."""
    _write_startup_state(os.environ if environment is None else environment, ready=True)
