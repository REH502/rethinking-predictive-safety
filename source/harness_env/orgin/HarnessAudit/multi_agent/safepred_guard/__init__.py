"""SafePred online guard integration for HarnessAudit.

The public classes are loaded lazily so foundational terminal-schema imports do
not initialize optional guard/provider machinery or create framework cycles.
"""

from __future__ import annotations

from typing import Any


__all__ = ["SafePredGuardSettings", "SafePredHarnessGuard"]


def __getattr__(name: str) -> Any:
    if name == "SafePredGuardSettings":
        from .config import SafePredGuardSettings

        return SafePredGuardSettings
    if name == "SafePredHarnessGuard":
        from .guard import SafePredHarnessGuard

        return SafePredHarnessGuard
    raise AttributeError(name)
