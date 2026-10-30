"""Shared names and validation for resource-rule interpretation modes."""

from __future__ import annotations


EXPLICIT_RESOURCE_SEMANTICS = "explicit_v2"
LEGACY_OFFICIAL_RESOURCE_SEMANTICS = "legacy_official_v1"
RESOURCE_SEMANTICS_MODES = frozenset(
    {
        EXPLICIT_RESOURCE_SEMANTICS,
        LEGACY_OFFICIAL_RESOURCE_SEMANTICS,
    }
)


def normalize_resource_semantics_mode(value: str | None) -> str:
    """Resolve and validate the resource-rule interpretation mode."""

    mode = str(value or EXPLICIT_RESOURCE_SEMANTICS).strip()
    if mode not in RESOURCE_SEMANTICS_MODES:
        raise ValueError(f"Unknown resource semantics mode: {mode!r}")
    return mode
