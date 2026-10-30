"""Reproducibility manifests for SafePred experiment artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_NON_SOURCE_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "generated_future_images",
    "logs",
    "reports",
    "results",
    "runs",
    "tmp",
    "traces",
    "workspaces",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_path in paths:
        # Avoid one network-filesystem realpath lookup per source file. Content
        # reads still follow symlinks, and the SHA-256 remains the authority.
        path = Path(raw_path).expanduser().absolute()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            rows.append({"path": str(path), "exists": False})
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def python_source_paths(root: Path) -> list[Path]:
    """Return executable Python sources without traversing run/data trees.

    Experiment outputs can contain tens of thousands of nested files on the shared
    filesystem. They are fingerprinted separately as inputs/artifacts and must not
    be traversed while constructing the code provenance set.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name not in _NON_SOURCE_DIRECTORY_NAMES and not name.startswith(".")
        )
        directory_path = Path(directory)
        paths.extend(
            directory_path / filename
            for filename in sorted(filenames)
            if filename.endswith(".py")
        )
    return paths


def default_code_paths(*, benchmark: str) -> list[Path]:
    project_root = Path(__file__).resolve().parents[5]
    harness_root = Path(__file__).resolve().parents[2]
    common = [
        *python_source_paths(project_root / "SafePred"),
        *python_source_paths(harness_root / "multi_agent"),
        project_root / "experiments" / "plot_results.py",
    ]
    if benchmark == "harnessaudit":
        pass
    elif benchmark == "isbench":
        common.extend(python_source_paths(project_root / "IS-Bench" / "og_ego_prim"))
    else:
        raise ValueError(f"Unsupported benchmark for provenance: {benchmark}")
    return list(dict.fromkeys(common))


def sanitize(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if lowered in {
        "api_key",
        "wm_api_key",
        "vlm_api_key",
        "tier3_api_key",
        "agent_api_key",
        "local_serve_key",
        "api_token",
        "access_token",
        "authorization",
        "password",
        "secret",
    } or lowered.endswith("_api_key"):
        return "[REDACTED]" if value else None
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(item_key): sanitize(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_experiment_manifest(
    path: Path,
    *,
    experiment_type: str,
    benchmark: str,
    config: dict[str, Any] | Any,
    input_files: Iterable[Path],
    artifact_files: Iterable[Path],
    status: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "schema_version": 1,
        "experiment_type": experiment_type,
        "benchmark": benchmark,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "config": sanitize(config),
        "status": sanitize(status or {}),
        "inputs": fingerprint_paths(input_files),
        "code": fingerprint_paths(default_code_paths(benchmark=benchmark)),
        "artifacts": fingerprint_paths(artifact_files),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
