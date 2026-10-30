"""Model-free monotonic deadline gate for resumable HA invocations.

The absolute deadline is deliberately based on :func:`time.monotonic`, not a
wall clock.  It is therefore local to one process invocation and must not be
persisted as scientific identity.  Callers may construct a fresh gate from an
invocation budget, then query it immediately before starting a bounded unit of
work such as policy extraction or one experiment condition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Any


def _finite_non_negative_seconds(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return normalized


def _monotonic_now(value: float | None) -> float:
    return _finite_non_negative_seconds(
        time.monotonic() if value is None else value,
        label="now_monotonic",
    )


@dataclass(frozen=True, slots=True)
class InvocationDeadlineDecision:
    """Secret-free decision returned at one prospective work boundary."""

    may_start: bool
    remaining_seconds: float
    available_work_seconds: float
    required_work_seconds: float
    safety_margin_seconds: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for lifecycle metadata."""

        return {
            "may_start": self.may_start,
            "remaining_seconds": self.remaining_seconds,
            "available_work_seconds": self.available_work_seconds,
            "required_work_seconds": self.required_work_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class InvocationDeadline:
    """Fail-closed gate for starting bounded work before an invocation deadline.

    ``safety_margin_seconds`` is reserved after the prospective work finishes,
    for example for final artifact persistence and process cleanup.  A start is
    allowed exactly when::

        deadline - now >= required_work_seconds + safety_margin_seconds
    """

    deadline_monotonic: float
    safety_margin_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deadline_monotonic",
            _finite_non_negative_seconds(
                self.deadline_monotonic,
                label="deadline_monotonic",
            ),
        )
        object.__setattr__(
            self,
            "safety_margin_seconds",
            _finite_non_negative_seconds(
                self.safety_margin_seconds,
                label="safety_margin_seconds",
            ),
        )

    @classmethod
    def from_budget_seconds(
        cls,
        budget_seconds: float,
        *,
        safety_margin_seconds: float,
        now_monotonic: float | None = None,
    ) -> "InvocationDeadline":
        """Anchor a non-negative invocation budget to a monotonic instant."""

        budget = _finite_non_negative_seconds(
            budget_seconds,
            label="budget_seconds",
        )
        now = _monotonic_now(now_monotonic)
        deadline = _finite_non_negative_seconds(
            now + budget,
            label="deadline_monotonic",
        )
        return cls(
            deadline_monotonic=deadline,
            safety_margin_seconds=safety_margin_seconds,
        )

    @classmethod
    def from_deadline_monotonic(
        cls,
        deadline_monotonic: float,
        *,
        safety_margin_seconds: float,
    ) -> "InvocationDeadline":
        """Construct a gate from an already-derived monotonic deadline."""

        return cls(
            deadline_monotonic=deadline_monotonic,
            safety_margin_seconds=safety_margin_seconds,
        )

    def decide(
        self,
        *,
        required_work_seconds: float = 0.0,
        now_monotonic: float | None = None,
    ) -> InvocationDeadlineDecision:
        """Return whether one bounded work unit may start now.

        Invalid, non-finite, boolean, or negative timing inputs raise
        :class:`ValueError`; callers must not silently convert them into an
        unbounded invocation.
        """

        required = _finite_non_negative_seconds(
            required_work_seconds,
            label="required_work_seconds",
        )
        now = _monotonic_now(now_monotonic)
        required_with_margin = _finite_non_negative_seconds(
            required + self.safety_margin_seconds,
            label="required work plus safety margin",
        )
        remaining = max(0.0, self.deadline_monotonic - now)
        available_work = max(0.0, remaining - self.safety_margin_seconds)
        deadline_expired = now >= self.deadline_monotonic
        may_start = not deadline_expired and remaining >= required_with_margin
        if deadline_expired:
            reason = "deadline_expired"
        elif may_start:
            reason = "sufficient_budget"
        else:
            reason = "insufficient_budget"
        return InvocationDeadlineDecision(
            may_start=may_start,
            remaining_seconds=remaining,
            available_work_seconds=available_work,
            required_work_seconds=required,
            safety_margin_seconds=self.safety_margin_seconds,
            reason=reason,
        )

    def can_start(
        self,
        *,
        required_work_seconds: float = 0.0,
        now_monotonic: float | None = None,
    ) -> bool:
        """Convenience boolean equivalent to ``decide(...).may_start``."""

        return self.decide(
            required_work_seconds=required_work_seconds,
            now_monotonic=now_monotonic,
        ).may_start


__all__ = ["InvocationDeadline", "InvocationDeadlineDecision"]
