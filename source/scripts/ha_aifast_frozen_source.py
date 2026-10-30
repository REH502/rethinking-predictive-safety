"""Materialize an existing verified HA source ZIP; never alter the worktree."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _destination(logical):
    path = PurePosixPath(logical)
    if path.is_absolute() or '..' in path.parts or '\\' in logical:
        raise ValueError('Unsafe source bundle path')
    mapping = (
        ('harnessaudit/registered_inputs/', 'harness_env/orgin/HarnessAudit/multi_agent/'),
        ('harnessaudit/multi_agent/', 'harness_env/orgin/HarnessAudit/multi_agent/'),
        ('SafePred/registered_inputs/', 'SafePred/'),
        ('SafePred/', 'SafePred/'), ('experiments/', 'experiments/'), ('scripts/', 'scripts/'))
    for prefix, target in mapping:
        if logical.startswith(prefix):
            result = target + logical[len(prefix):]
            if not result.endswith('/'):
                return result
    raise ValueError('Unexpected source bundle namespace')


def _compute_impl(root):
    root = Path(root).resolve()
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in (root, root / 'harness_env/orgin/HarnessAudit', root / 'SafePred'))
    env['PYTHONNOUSERSITE'] = '1'
    code = ('import json,sys; from pathlib import Path; '
            'from multi_agent.safepred_experiments import formal_contract as f; '
            'assert Path(f.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()); '
            'print(json.dumps({"implementation_sha256":f.current_online_implementation_manifest_sha256("harnessaudit")}))')
    result = subprocess.run([sys.executable, '-c', code, str(root)], cwd=root,
                            env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError('Isolated source digest computation failed')
    records = []
    for line in result.stdout.splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and set(item) == {'implementation_sha256'}:
                records.append(item)
        except ValueError:
            pass
    if len(records) != 1:
        raise ValueError('Isolated source digest result is ambiguous')
    return records[0]['implementation_sha256']


def materialize(bundle: Path, destination: Path):
    bundle = Path(bundle).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or destination != destination.resolve():
        raise ValueError('Frozen source destination must be fresh and non-symlink')
    contents = {}
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
        if manifest.get('schema_version') != 'formal-online-source-bundle-v1' or manifest.get('benchmark') != 'harnessaudit':
            raise ValueError('Unsupported source bundle contract')
        expected = manifest['implementation_manifest_sha256']
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Source implementation digest malformed')
        rows = manifest['files']
        names = [r['path'] for r in rows]
        if len(set(names)) != len(names) or set(archive.namelist()) != set(names) | {'SOURCE_MANIFEST.json'}:
            raise ValueError('Source bundle inventory differs from manifest')
        for row in rows:
            info = archive.getinfo(row['path'])
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Source bundle symlinks forbidden')
            data = archive.read(row['path'])
            if len(data) != row['size_bytes'] or _sha(data) != row['sha256']:
                raise ValueError('Source bundle file hash drift')
            relative = _destination(row['path'])
            if relative in contents and contents[relative] != data:
                raise ValueError('Conflicting duplicate source mapping')
            contents[relative] = data
    destination.mkdir(parents=True, mode=0o700)
    bindings = []
    for relative, data in sorted(contents.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
        bindings.append(dict(relative_path=relative, sha256=_sha(data), size_bytes=len(data)))
    observed = _compute_impl(destination)
    if observed != expected:
        raise ValueError('Materialized source does not reproduce its frozen digest')
    result = dict(schema_version='ha-aifast-frozen-native-v1', status='passed',
                  frozen_project_root=str(destination), implementation_sha256=observed,
                  expected_implementation_sha256=expected, bundle_path=str(bundle),
                  bundle_sha256=_sha(bundle.read_bytes()), file_count=len(bindings), files=bindings)
    manifest_path = destination / 'MATERIALIZATION.json'
    with manifest_path.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write('\n')
    result['manifest_path'] = str(manifest_path)
    result['manifest_sha256'] = _sha(manifest_path.read_bytes())
    return result


def verify_materialization(record):
    root = Path(record['frozen_project_root']).resolve(strict=True)
    manifest = Path(record['manifest_path'])
    if manifest.parent != root or _sha(manifest.read_bytes()) != record['manifest_sha256']:
        raise ValueError('Materialization manifest drift')
    expected = json.loads(manifest.read_text())
    if {k: v for k, v in record.items() if k not in {'manifest_path', 'manifest_sha256'}} != expected:
        raise ValueError('Materialization registration differs')
    if _sha(Path(record['bundle_path']).read_bytes()) != record['bundle_sha256']:
        raise ValueError('Original source bundle drift')
    for row in expected['files']:
        path = root / row['relative_path']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Materialized source path escape')
        data = path.read_bytes()
        if len(data) != row['size_bytes'] or _sha(data) != row['sha256']:
            raise ValueError('Materialized source file drift')
    observed = _compute_impl(root)
    if observed != expected['implementation_sha256']:
        raise ValueError('Isolated native implementation drift')
    return {'status': 'passed', 'implementation_sha256': observed, 'frozen_project_root': str(root)}
