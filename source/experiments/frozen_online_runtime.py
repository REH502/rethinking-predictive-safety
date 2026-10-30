"""Create and verify an immutable IS-Bench execution tree.

Long-running online jobs must not import a mixture of source revisions from the
active workspace.  This module turns the existing provenance source bundle
into a runnable project-shaped tree and verifies every frozen source byte on
resume.  It deliberately leaves large, unregistered simulator assets as
read-only symlink targets while materializing the registered task, camera, and
BDDL inputs from the source bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from multi_agent.safepred_experiments.formal_contract import (
    current_online_implementation_manifest_sha256,
    freeze_online_implementation_source_bundle,
)


SOURCE_BUNDLE_SCHEMA = "formal-online-source-bundle-v1"
RUNTIME_SCHEMA = "isbench-frozen-online-runtime-v2"
REGISTERED_DATA_DIRS = ("tasks", "cameras", "bddl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(raw: str | Path, *, label: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe path in {label}: {raw}")
    return relative


def _regular_path_binding(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"materialized data path is a symlink: {path}")
    if path.is_file():
        return {
            "kind": "file",
            "file_count": 1,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if not path.is_dir():
        raise ValueError(f"materialized data path is not regular: {path}")
    rows: list[dict[str, Any]] = []
    total_size = 0
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"materialized data tree contains a symlink: {item}")
        if not item.is_file():
            continue
        size = item.stat().st_size
        rows.append({
            "path": item.relative_to(path).as_posix(),
            "size_bytes": size,
            "sha256": _sha256(item),
        })
        total_size += size
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "directory",
        "file_count": len(rows),
        "size_bytes": total_size,
        "sha256": digest,
    }


def _copy_reflink_or_bytes(source: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cp", "-a", "--reflink=auto", "--", str(source), str(target)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise ValueError(f"cannot materialize IS-Bench data: {detail}")


def _validated_bundle(
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    bundle = Path(bundle)
    if not bundle.is_file() or bundle.is_symlink():
        raise ValueError("frozen implementation source bundle is missing or a symlink")
    with zipfile.ZipFile(bundle) as archive:
        infos: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            _safe_relative(info.filename, label="source bundle")
            if info.filename in infos:
                raise ValueError(f"duplicate source bundle member: {info.filename}")
            unix_mode = info.external_attr >> 16
            if info.is_dir() or (unix_mode and not stat.S_ISREG(unix_mode)):
                raise ValueError(f"non-regular source bundle member: {info.filename}")
            infos[info.filename] = info
        manifest_info = infos.get("SOURCE_MANIFEST.json")
        if manifest_info is None:
            raise ValueError("source bundle manifest is missing")
        try:
            manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("source bundle manifest is invalid") from exc
        if not (
            manifest.get("schema_version") == SOURCE_BUNDLE_SCHEMA
            and manifest.get("benchmark") == "isbench"
        ):
            raise ValueError("source bundle identity is not IS-Bench v1")
        implementation = str(
            manifest.get("implementation_manifest_sha256") or ""
        ).lower()
        if len(implementation) != 64 or any(
            character not in "0123456789abcdef" for character in implementation
        ):
            raise ValueError("source bundle implementation digest is invalid")
        rows = manifest.get("files")
        if not isinstance(rows, list):
            raise ValueError("source bundle file manifest is invalid")
        expected_names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("source bundle file row is invalid")
            name = str(row.get("path") or "")
            _safe_relative(name, label="source bundle manifest")
            if name == "SOURCE_MANIFEST.json" or name in expected_names:
                raise ValueError(f"duplicate source bundle manifest path: {name}")
            expected_names.add(name)
            info = infos.get(name)
            if info is None:
                raise ValueError(f"source bundle member is missing: {name}")
            payload = archive.read(info)
            expected_size = row.get("size_bytes")
            if not isinstance(expected_size, int) or expected_size != len(payload):
                raise ValueError(f"source bundle size mismatch: {name}")
            if str(row.get("sha256") or "").lower() != hashlib.sha256(
                payload
            ).hexdigest():
                raise ValueError(f"source bundle digest mismatch: {name}")
        if set(infos) != expected_names | {"SOURCE_MANIFEST.json"}:
            raise ValueError("source bundle contains unregistered members")
    return manifest, infos


def _extract_verified_source_bundle(bundle: Path, destination: Path) -> dict[str, Any]:
    """Safely extract a byte-verified source bundle into a new directory."""

    bundle = Path(bundle)
    destination = Path(destination)
    manifest, _infos = _validated_bundle(bundle)
    if destination.exists():
        raise ValueError(f"frozen runtime destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    try:
        with zipfile.ZipFile(bundle) as archive:
            for info in archive.infolist():
                relative = _safe_relative(info.filename, label="source bundle")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = archive.read(info)
                with target.open("xb") as handle:
                    handle.write(payload)
        return manifest
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _materialize_isbench_layout(
    runtime_root: Path,
    *,
    project_root: Path,
    model_profile: Path,
    model_profile_relative: Path,
) -> dict[str, Any]:
    """Turn archive paths into the project layout expected by frozen imports."""

    runtime_root = Path(runtime_root).resolve()
    project_root = Path(project_root).resolve()
    model_profile = Path(model_profile).resolve()
    profile_relative = _safe_relative(
        model_profile_relative, label="model profile target"
    )
    if not model_profile.is_file() or model_profile.is_symlink():
        raise ValueError("frozen model profile source is missing or a symlink")

    archived_harness = runtime_root / "harnessaudit"
    harness_target = runtime_root / "harness_env/orgin/HarnessAudit"
    if not (archived_harness / "multi_agent").is_dir():
        raise ValueError("source bundle does not contain HarnessAudit")
    harness_target.parent.mkdir(parents=True, exist_ok=True)
    archived_harness.replace(harness_target)

    registered_root = runtime_root / "IS-Bench/registered_inputs"
    frozen_data = runtime_root / "IS-Bench/data"
    project_data = project_root / "IS-Bench/data"
    if not project_data.is_dir():
        raise ValueError("active IS-Bench data root is missing")
    frozen_data.mkdir(parents=True)
    for name in REGISTERED_DATA_DIRS:
        source = registered_root / name
        if not source.is_dir():
            raise ValueError(f"source bundle registered input is missing: {name}")
        source.replace(frozen_data / name)
    registered_root.rmdir()

    materialized_data_bindings: dict[str, dict[str, Any]] = {}
    for source in sorted(project_data.iterdir()):
        if source.name in REGISTERED_DATA_DIRS:
            continue
        target = frozen_data / source.name
        _copy_reflink_or_bytes(source, target)
        materialized_data_bindings[source.name] = _regular_path_binding(target)

    font_source = (
        project_root
        / "IS-Bench/og_ego_prim/benchmark/OpenSans-Bold.ttf"
    )
    if not font_source.is_file() or font_source.is_symlink():
        raise ValueError("IS-Bench bbox font asset is missing or a symlink")
    font_target = (
        runtime_root
        / "IS-Bench/og_ego_prim/benchmark/OpenSans-Bold.ttf"
    )
    font_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(font_source, font_target)

    profile_target = runtime_root / profile_relative
    profile_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_profile, profile_target)
    return {
        "model_profile_relative_path": profile_relative.as_posix(),
        "model_profile_sha256": _sha256(profile_target),
        "bbox_font_relative_path": font_target.relative_to(runtime_root).as_posix(),
        "bbox_font_sha256": _sha256(font_target),
        "materialized_data_bindings": materialized_data_bindings,
    }


def _runtime_source_path(runtime_root: Path, archive_path: str) -> Path:
    relative = _safe_relative(archive_path, label="runtime source manifest")
    if relative.parts[0] == "harnessaudit":
        relative = Path("harness_env/orgin/HarnessAudit", *relative.parts[1:])
    elif relative.parts[:2] == ("IS-Bench", "registered_inputs"):
        relative = Path("IS-Bench/data", *relative.parts[2:])
    return runtime_root / relative


def verify_frozen_isbench_runtime(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    frozen_root = run_root / "frozen"
    bundle = frozen_root / "implementation_source_bundle.zip"
    runtime_root = frozen_root / "runtime"
    marker_path = runtime_root / ".frozen_runtime.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ValueError("frozen runtime marker is missing or a symlink")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("frozen runtime marker schema is invalid")
    manifest, _infos = _validated_bundle(bundle)
    if marker.get("source_bundle_sha256") != _sha256(bundle):
        raise ValueError("frozen runtime source bundle identity drift")
    if marker.get("implementation_manifest_sha256") != manifest.get(
        "implementation_manifest_sha256"
    ):
        raise ValueError("frozen runtime implementation identity drift")
    with zipfile.ZipFile(bundle) as archive:
        for row in manifest["files"]:
            source_path = _runtime_source_path(runtime_root, str(row["path"]))
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"frozen runtime source is missing: {row['path']}")
            if source_path.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"frozen runtime source size drift: {row['path']}")
            if _sha256(source_path) != str(row["sha256"]):
                raise ValueError(f"frozen runtime source digest drift: {row['path']}")
        source_manifest = runtime_root / "SOURCE_MANIFEST.json"
        if source_manifest.read_bytes() != archive.read("SOURCE_MANIFEST.json"):
            raise ValueError("frozen runtime source manifest drift")
    profile = runtime_root / str(marker["model_profile_relative_path"])
    if not profile.is_file() or profile.is_symlink() or _sha256(profile) != marker.get(
        "model_profile_sha256"
    ):
        raise ValueError("frozen runtime model profile drift")
    font = runtime_root / str(marker["bbox_font_relative_path"])
    if not font.is_file() or font.is_symlink() or _sha256(font) != marker.get(
        "bbox_font_sha256"
    ):
        raise ValueError("frozen runtime bbox font drift")
    for name in REGISTERED_DATA_DIRS:
        path = runtime_root / "IS-Bench/data" / name
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"frozen registered input drift: {name}")
    materialized = marker.get("materialized_data_bindings")
    if not isinstance(materialized, dict):
        raise ValueError("frozen runtime materialized data bindings are missing")
    for name, expected_binding in materialized.items():
        path = runtime_root / "IS-Bench/data" / name
        if _regular_path_binding(path) != expected_binding:
            raise ValueError(f"materialized IS-Bench data binding drift: {name}")
    observed_data_names = {
        path.name for path in (runtime_root / "IS-Bench/data").iterdir()
    }
    if observed_data_names != set(REGISTERED_DATA_DIRS) | set(materialized):
        raise ValueError("frozen runtime data directory membership drift")
    return marker


def prepare_frozen_isbench_runtime(
    *,
    project_root: Path,
    run_root: Path,
    model_profile: Path,
    model_profile_relative: Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    run_root = Path(run_root).resolve()
    frozen_root = run_root / "frozen"
    frozen_root.mkdir(parents=True, exist_ok=True)
    bundle = frozen_root / "implementation_source_bundle.zip"
    runtime_root = frozen_root / "runtime"
    if runtime_root.exists():
        return verify_frozen_isbench_runtime(run_root)
    if bundle.exists():
        manifest, _infos = _validated_bundle(bundle)
    else:
        stale_temporary = bundle.with_name(bundle.name + ".tmp")
        if stale_temporary.exists():
            quarantine = frozen_root / "interrupted_source_bundle.tmp"
            if quarantine.exists():
                raise ValueError("multiple interrupted source bundle preparations")
            stale_temporary.replace(quarantine)
        implementation = current_online_implementation_manifest_sha256("isbench")
        freeze_online_implementation_source_bundle(
            benchmark="isbench",
            output_path=bundle,
            expected_implementation_sha256=implementation,
        )
        manifest, _infos = _validated_bundle(bundle)

    for stale in sorted(frozen_root.glob(".runtime.tmp-*")):
        quarantine_root = frozen_root / "interrupted_runtime_preparations"
        quarantine_root.mkdir(exist_ok=True)
        destination = quarantine_root / stale.name
        if destination.exists():
            raise ValueError("interrupted runtime quarantine collision")
        stale.replace(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".runtime.tmp-", dir=frozen_root))
    shutil.rmtree(temporary)
    try:
        extracted_manifest = _extract_verified_source_bundle(bundle, temporary)
        binding = _materialize_isbench_layout(
            temporary,
            project_root=project_root,
            model_profile=model_profile,
            model_profile_relative=model_profile_relative,
        )
        marker = {
            "schema_version": RUNTIME_SCHEMA,
            "benchmark": "isbench",
            "source_bundle_sha256": _sha256(bundle),
            "implementation_manifest_sha256": extracted_manifest[
                "implementation_manifest_sha256"
            ],
            **binding,
        }
        _atomic_json(temporary / ".frozen_runtime.json", marker)
        temporary.replace(runtime_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_frozen_isbench_runtime(run_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--model-profile", type=Path, required=True)
    prepare.add_argument("--model-profile-relative", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_frozen_isbench_runtime(
            project_root=args.project_root,
            run_root=args.run_root,
            model_profile=args.model_profile,
            model_profile_relative=args.model_profile_relative,
        )
    else:
        payload = verify_frozen_isbench_runtime(args.run_root)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
