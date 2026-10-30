"""Credential-safe HTTP retry evidence for the HarnessAudit v7 child.

This module deliberately owns the low-level retry loop.  Exhaustion raises a
non-``requests`` exception so the unchanged SafePred client's outer retry loop
cannot multiply the registered four-attempt budget.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable

import requests


_RETRYABLE_STATUS = {404, 408, 409, 425, 429}
_LEDGER_FIELDS = {
    "schema_version",
    "artifact_type",
    "semantic_group",
    "transport_attempt",
    "body_sha256",
    "body_size",
    "outcome",
    "http_status",
    "exception_type",
    "timestamp_utc",
}
_ACTIVE_LOCK = threading.Lock()


class CompletedResponseSemanticInvalid(ValueError):
    """A transported response completed, but its JSON/content was invalid."""

    def __init__(self, token: str, *, http_status: int = 200) -> None:
        self.token = str(token)
        self.http_status = int(http_status)
        super().__init__(self.token)


class TransportExhausted(RuntimeError):
    """The registered identical-request transport budget was exhausted."""

    def __init__(
        self,
        token: str = "transport_attempts_exhausted",
        *,
        http_status: int | None = None,
        last_error: BaseException | None = None,
    ) -> None:
        self.token = str(token)
        self.http_status = http_status
        self.last_error = last_error
        super().__init__(self.token)


class NonRetryableHTTPResponse(RuntimeError):
    """A completed non-retryable HTTP response rejected before legacy resend."""

    def __init__(self, *, http_status: int) -> None:
        self.http_status = int(http_status)
        super().__init__("nonretryable_http_response")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _contains(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute(path)
    for candidate in [*reversed(absolute.parents), absolute]:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("transport ledger path contains a symlink")


def _safe_ledger_path(path: Path, *, run_root: Path) -> Path:
    root = _absolute(run_root)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    destination = _absolute(path)
    if not _contains(destination, root):
        raise ValueError("transport ledger is outside run_root")
    current = root
    for part in destination.parent.relative_to(root).parts:
        current = current / part
        _reject_symlink_components(current)
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError("transport ledger parent is not a directory")
    _reject_symlink_components(destination)
    if destination.exists() and not stat.S_ISREG(destination.lstat().st_mode):
        raise ValueError("transport ledger must be a regular file")
    return destination


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_STATUS or 500 <= status <= 599


class _ObservedResponse:
    def __init__(
        self,
        response: Any,
        observer: "V7TransportObserver",
        row_base: dict[str, Any],
    ) -> None:
        self._response = response
        self._observer = observer
        self._row_base = row_base
        self._recorded = False

    def _record(self, *, outcome: str, exception_type: str | None) -> None:
        if self._recorded:
            return
        self._observer._append_attempt(
            {
                **self._row_base,
                "outcome": outcome,
                "exception_type": exception_type,
            }
        )
        self._recorded = True
        self._observer._pending_responses.discard(self)

    def json(self, *args: Any, **kwargs: Any) -> Any:
        try:
            value = self._response.json(*args, **kwargs)
        except json.JSONDecodeError as exc:
            self._record(outcome="semantic_invalid", exception_type=type(exc).__name__)
            raise CompletedResponseSemanticInvalid(
                "completed_response_json_invalid",
                http_status=int(getattr(self._response, "status_code", 200)),
            ) from exc
        self._record(outcome="completed_response", exception_type=None)
        return value

    def raise_for_status(self) -> Any:
        try:
            return self._response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            self._record(outcome="nonretryable_http", exception_type=type(exc).__name__)
            raise NonRetryableHTTPResponse(
                http_status=int(getattr(self._response, "status_code", 0))
            ) from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


class V7TransportObserver:
    """Process-local requests interceptor with a fixed four-attempt budget."""

    def __init__(
        self,
        *,
        ledger_path: Path,
        run_root: Path,
        total_attempts: int = 4,
        retry_delay_seconds: float = 15.0,
        retry_backoff: float = 2.0,
        sleep: Callable[[float], Any] = time.sleep,
        timestamp: Callable[[], str] = _timestamp_utc,
    ) -> None:
        if total_attempts != 4:
            raise ValueError("v7 transport_total_attempts must equal 4")
        if retry_delay_seconds != 15.0 or retry_backoff != 2.0:
            raise ValueError("v7 retry schedule must equal 15/30/60 with backoff 2")
        self.ledger_path = _safe_ledger_path(ledger_path, run_root=run_root)
        self.run_root = _absolute(run_root)
        self.total_attempts = total_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_backoff = retry_backoff
        self.sleep = sleep
        self.timestamp = timestamp
        self._original_post: Any = None
        self._original_session_post: Any = None
        self._owner_thread: int | None = None
        self._group_counter = 0
        self._pending_responses: set[_ObservedResponse] = set()
        self._entered = False

    def __enter__(self) -> "V7TransportObserver":
        if self._entered or not _ACTIVE_LOCK.acquire(blocking=False):
            raise RuntimeError("a v7 transport observer is already active; nesting/concurrency is forbidden")
        self._entered = True
        self._owner_thread = threading.get_ident()
        self._original_post = requests.post
        self._original_session_post = requests.Session.post

        def observed_post(url: str, *args: Any, **kwargs: Any) -> Any:
            return self._execute(self._original_post, None, url, args, kwargs)

        def observed_session_post(session: requests.Session, url: str, *args: Any, **kwargs: Any) -> Any:
            return self._execute(self._original_session_post, session, url, args, kwargs)

        requests.post = observed_post  # type: ignore[assignment]
        requests.Session.post = observed_session_post  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            for response in tuple(self._pending_responses):
                response._record(outcome="completed_response", exception_type=None)
        finally:
            requests.post = self._original_post
            requests.Session.post = self._original_session_post
            self._owner_thread = None
            self._entered = False
            _ACTIVE_LOCK.release()

    def _assert_owner(self) -> None:
        if not self._entered or threading.get_ident() != self._owner_thread:
            raise RuntimeError("v7 transport observer forbids concurrent request execution")

    def _next_group(self) -> str:
        self._group_counter += 1
        return f"group-{self._group_counter:06d}"

    def _append_attempt(self, row: dict[str, Any]) -> None:
        payload = {
            "schema_version": 2,
            "artifact_type": "ha_v7_transport_attempt",
            **row,
        }
        if set(payload) != _LEDGER_FIELDS:
            raise RuntimeError("internal transport ledger schema mismatch")
        encoded = _canonical_json_bytes(payload) + b"\n"
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.ledger_path, flags, 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("transport ledger write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _execute(
        self,
        original: Callable[..., Any],
        receiver: Any,
        url: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self._assert_owner()
        if "json" not in kwargs or "data" in kwargs:
            raise ValueError("v7 observed POST requires json= and forbids simultaneous data=")
        body = _canonical_json_bytes(kwargs["json"])
        body_sha256 = hashlib.sha256(body).hexdigest()
        call_kwargs = dict(kwargs)
        call_kwargs.pop("json")
        call_kwargs["data"] = body
        headers = dict(call_kwargs.get("headers") or {})
        headers["Content-Type"] = "application/json"
        call_kwargs["headers"] = headers
        semantic_group = self._next_group()
        delay = self.retry_delay_seconds
        last_error: BaseException | None = None
        last_status: int | None = None

        for attempt in range(1, self.total_attempts + 1):
            row_base = {
                "semantic_group": semantic_group,
                "transport_attempt": attempt,
                "body_sha256": body_sha256,
                "body_size": len(body),
                "http_status": None,
                "timestamp_utc": self.timestamp(),
            }
            try:
                if receiver is None:
                    response = original(url, *args, **call_kwargs)
                else:
                    response = original(receiver, url, *args, **call_kwargs)
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.SSLError) as error:
                last_error = error
                self._append_attempt(
                    {
                        **row_base,
                        "outcome": "transport_error",
                        "exception_type": type(error).__name__,
                    }
                )
            except requests.exceptions.HTTPError as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if isinstance(status, int) and _is_retryable_status(status):
                    last_error = error
                    last_status = status
                    self._append_attempt(
                        {
                            **row_base,
                            "http_status": status,
                            "outcome": "retryable_http",
                            "exception_type": type(error).__name__,
                        }
                    )
                else:
                    self._append_attempt(
                        {
                            **row_base,
                            "http_status": status,
                            "outcome": "nonretryable_http",
                            "exception_type": type(error).__name__,
                        }
                    )
                    raise NonRetryableHTTPResponse(http_status=int(status or 0)) from error
            else:
                status = int(getattr(response, "status_code", 0))
                if _is_retryable_status(status):
                    last_status = status
                    self._append_attempt(
                        {
                            **row_base,
                            "http_status": status,
                            "outcome": "retryable_http",
                            "exception_type": "HTTPStatus",
                        }
                    )
                elif status < 200 or status >= 300:
                    self._append_attempt(
                        {
                            **row_base,
                            "http_status": status,
                            "outcome": "nonretryable_http",
                            "exception_type": "HTTPStatus",
                        }
                    )
                    raise NonRetryableHTTPResponse(http_status=status)
                else:
                    observed = _ObservedResponse(
                        response,
                        self,
                        {**row_base, "http_status": status},
                    )
                    self._pending_responses.add(observed)
                    return observed

            if attempt < self.total_attempts:
                self.sleep(float(delay))
                delay *= self.retry_backoff

        raise TransportExhausted(
            http_status=last_status,
            last_error=last_error,
        ) from last_error


__all__ = [
    "CompletedResponseSemanticInvalid",
    "NonRetryableHTTPResponse",
    "TransportExhausted",
    "V7TransportObserver",
]
