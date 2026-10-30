"""Load persistent experiment credentials from one workspace-level file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


CREDENTIALS_FILE_ENV = "WM_SAFETY_ENV_FILE"


def _workspace_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "SafePred").is_dir()
            and (candidate / "IS-Bench").is_dir()
            and (candidate / "harness_env" / "orgin" / "HarnessAudit").is_dir()
        ):
            return candidate
    raise RuntimeError(f"Cannot locate world-model safety workspace from {start}")


def workspace_credentials_path(start: Path) -> Path:
    configured = str(os.environ.get(CREDENTIALS_FILE_ENV) or "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else _workspace_root(start) / "config" / "experiment.env"
    )
    return path.resolve()


def load_workspace_credentials(start: Path, *, override: bool = False) -> Path:
    """Load the private workspace file while keeping process values authoritative."""

    path = workspace_credentials_path(start)
    os.environ[CREDENTIALS_FILE_ENV] = str(path)
    if not path.exists():
        return path
    if not path.is_file():
        raise ValueError(f"Credentials path is not a file: {path}")
    if path.stat().st_mode & 0o077:
        raise PermissionError(
            f"Credentials file is accessible by group or other users: {path}. "
            f"Run: chmod 600 {path}"
        )
    load_dotenv(path, override=override)
    if not str(os.environ.get("OPENAI_API_KEY") or "").strip():
        actor_key = str(
            os.environ.get("AGENT_API_KEY")
            or os.environ.get("HARNESSAUDIT_API_KEY")
            or ""
        ).strip()
        if actor_key:
            os.environ["OPENAI_API_KEY"] = actor_key
    if not str(os.environ.get("OPENAI_BASE_URL") or "").strip():
        actor_url = str(
            os.environ.get("AGENT_BASE_URL")
            or os.environ.get("HARNESSAUDIT_BASE_URL")
            or ""
        ).strip()
        if actor_url:
            os.environ["OPENAI_BASE_URL"] = actor_url
            os.environ.setdefault("OPENAI_API_BASE", actor_url)
    return path
