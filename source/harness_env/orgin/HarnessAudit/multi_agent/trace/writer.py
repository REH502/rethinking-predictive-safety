"""Append-only JSONL trace writer."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO

from .events import (
    AccessDecisionEvent,
    CommunicationEvent,
    SafetyAbstentionEvent,
    ToolCallEvent,
    TraceEnd,
    TraceStart,
)

_AnyEvent = (
    TraceStart | ToolCallEvent | CommunicationEvent | SafetyAbstentionEvent
    | AccessDecisionEvent | TraceEnd
)


class TraceWriter:
    """Writes trace events as JSONL (one JSON object per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def _ensure_open(self) -> IO[str]:
        if self._fh is None or self._fh.closed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existed = self.path.exists()
            flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("Trace path must be a regular file")
            os.fchmod(descriptor, 0o600)
            self._fh = os.fdopen(descriptor, "a", encoding="utf-8")
            if not existed:
                directory_flags = os.O_RDONLY
                if hasattr(os, "O_DIRECTORY"):
                    directory_flags |= os.O_DIRECTORY
                directory_fd = os.open(self.path.parent, directory_flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        return self._fh

    def write_event(self, event: _AnyEvent) -> None:
        fh = self._ensure_open()
        fh.write(event.model_dump_json() + "\n")
        fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
