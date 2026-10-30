#!/usr/bin/env python3
"""Verify immutable distribution contents, excluding user-created local/ and virtualenvs."""
import ast
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    seen = set()
    pycount = 0
    for line in (root/'BUNDLE_MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or name in seen:
            raise ValueError('Unsafe or repeated manifest entry')
        seen.add(name)
        path = root/relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked file: '+name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Modified file: '+name)
        if path.suffix == '.py':
            ast.parse(data, filename=name)
            pycount += 1
    print(f'PASS: integrity of {len(seen)} distribution files ({pycount} Python sources)')


if __name__ == '__main__':
    main()
