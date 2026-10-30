#!/usr/bin/env python3
"""Verify a git/Git-LFS model checkpoint and serving-runtime identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _run(arguments: list[str], *, cwd: Path | None = None) -> bytes:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({arguments[0]} exit={completed.returncode})"
        )
    return completed.stdout


def _git(model_dir: Path, *arguments: str) -> bytes:
    # The shared checkpoint is intentionally owned by root.  Some compute
    # nodes apply stricter repository-ownership policy than the login node and
    # can reject even a command-scoped safe.directory override.  Supplying the
    # repository and work tree explicitly keeps this verifier read-only and
    # avoids depending on node-local/global Git configuration.
    return _run(
        [
            "git",
            "-c",
            f"safe.directory={model_dir}",
            f"--git-dir={model_dir / '.git'}",
            f"--work-tree={model_dir}",
            *arguments,
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_remote(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if parsed.scheme not in {"http", "https", "ssh"}:
        return ""
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname + (f":{parsed.port}" if parsed.port is not None else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _lfs_pointer(blob: bytes) -> tuple[str, int] | None:
    try:
        lines = blob.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    oid = next(
        (line.removeprefix("oid sha256:") for line in lines if line.startswith("oid sha256:")),
        "",
    )
    size_text = next(
        (line.removeprefix("size ") for line in lines if line.startswith("size ")),
        "",
    )
    if len(oid) != 64 or any(char not in "0123456789abcdef" for char in oid):
        raise ValueError("invalid Git-LFS SHA256 pointer")
    if not size_text.isdigit():
        raise ValueError("invalid Git-LFS size pointer")
    return oid, int(size_text)


def _runtime_freeze_sha256(python: Path) -> str:
    raw = _run([str(python), "-m", "pip", "freeze", "--all"])
    rows = sorted(line for line in raw.decode("utf-8").splitlines() if line)
    canonical = ("\n".join(rows) + "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_checkpoint(
    model_dir: Path,
    *,
    verify_lfs_content: bool,
    runtime_python: Path | None,
) -> dict[str, object]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    commit = _git(model_dir, "rev-parse", "HEAD").decode("ascii").strip()
    names = [
        line
        for line in _git(model_dir, "ls-tree", "-r", "--name-only", "HEAD")
        .decode("utf-8")
        .splitlines()
        if line
    ]
    lfs_entries: list[dict[str, object]] = []
    regular_entries: list[dict[str, object]] = []
    for name in sorted(names):
        blob = _git(model_dir, "cat-file", "blob", f"HEAD:{name}")
        pointer = _lfs_pointer(blob)
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"tracked checkpoint file is missing: {name}")
        if pointer is not None:
            expected_sha256, expected_size = pointer
            observed_size = path.stat().st_size
            if observed_size != expected_size:
                raise ValueError(
                    f"Git-LFS size mismatch for {name}: "
                    f"observed={observed_size}, expected={expected_size}"
                )
            if verify_lfs_content:
                observed_sha256 = _sha256_file(path)
                if observed_sha256 != expected_sha256:
                    raise ValueError(f"Git-LFS content SHA256 mismatch for {name}")
            lfs_entries.append(
                {
                    "path": name,
                    "sha256": expected_sha256,
                    "size": expected_size,
                }
            )
            continue
        expected_sha256 = hashlib.sha256(blob).hexdigest()
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"tracked checkpoint file differs from HEAD: {name}")
        regular_entries.append(
            {"path": name, "sha256": expected_sha256, "size": len(blob)}
        )

    lfs_lines = [
        f"{entry['path']}\tsha256:{entry['sha256']}\t{entry['size']}"
        for entry in lfs_entries
    ]
    lfs_manifest_sha256 = hashlib.sha256(
        (("\n".join(lfs_lines) + "\n") if lfs_lines else "").encode("utf-8")
    ).hexdigest()
    remote_raw = _git(model_dir, "remote", "get-url", "origin").decode(
        "utf-8"
    ).strip()
    runtime_sha256 = (
        _runtime_freeze_sha256(runtime_python.expanduser().resolve())
        if runtime_python is not None
        else None
    )
    return {
        "schema_version": "model-checkpoint-verification-v1",
        "status": "verified",
        "repository": _sanitize_remote(remote_raw),
        "commit": commit,
        "lfs_manifest_sha256": lfs_manifest_sha256,
        "lfs_content_verified": bool(verify_lfs_content),
        "lfs_entries": lfs_entries,
        "regular_entries": regular_entries,
        "runtime_pip_freeze_sha256": runtime_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-lfs-content", action="store_true")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-lfs-manifest-sha256")
    parser.add_argument("--expected-runtime-freeze-sha256")
    args = parser.parse_args()

    result = verify_checkpoint(
        args.model_dir,
        verify_lfs_content=args.verify_lfs_content,
        runtime_python=args.runtime_python,
    )
    expected = {
        "commit": args.expected_commit,
        "lfs_manifest_sha256": args.expected_lfs_manifest_sha256,
        "runtime_pip_freeze_sha256": args.expected_runtime_freeze_sha256,
    }
    for field, expected_value in expected.items():
        if expected_value and result.get(field) != expected_value:
            raise SystemExit(f"{field} differs from the registered value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
