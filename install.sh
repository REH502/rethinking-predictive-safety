#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for the tested HA environment"'
"$python_bin" -c 'import platform; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Bundled wheels target Linux x86_64"; lib, v = platform.libc_ver(); assert lib == "glibc" and tuple(map(int, v.split(".")[:2])) >= (2, 34), "Bundled wheels require glibc >= 2.34 (e.g. Ubuntu 22.04+)"'
"$python_bin" -B "$package_root/tools/verify_bundle.py"
"$python_bin" -m venv "$package_root/.venv-ha"
if [ -d "$package_root/wheels" ]; then
  "$package_root/.venv-ha/bin/python" -m pip install --no-index --find-links "$package_root/wheels" -r "$package_root/requirements-ha.lock.txt"
else
  "$package_root/.venv-ha/bin/python" -m pip install -r "$package_root/source/requirements-ha.txt"
fi
"$package_root/.venv-ha/bin/python" -B "$package_root/tools/doctor.py"
"$package_root/.venv-ha/bin/python" -B "$package_root/tools/smoke_test.py"
echo 'HA installation verified. Next: .venv-ha/bin/python configure.py (keys stay outside the package).'
