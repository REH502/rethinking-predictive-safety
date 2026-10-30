"""Pure validation for one ISBench actor action proposal."""

from __future__ import annotations

import re
from collections.abc import Mapping, Set
from typing import Any


# Keep the actor, executor, prompt checks, and SafePred alternative-action
# validator on one pure schema that does not import OmniGibson.
ISBENCH_PRIMITIVE_ARITIES = {
    "PLACE_ON_TOP": 2,
    "PLACE_INSIDE": 2,
    "OPEN": 1,
    "CLOSE": 1,
    "TOGGLE_ON": 1,
    "TOGGLE_OFF": 1,
    "WIPE": 2,
    "CUT": 2,
    "SOAK_INSIDE": 2,
    "SOAK_UNDER": 2,
    "FILL_WITH": 2,
    "POUR_INTO": 2,
    "SPREAD": 2,
    "WAIT": 1,
    "WAIT_FOR_COOKED": 1,
    "WAIT_FOR_WASHED": 1,
    "WAIT_FOR_FROZEN": 2,
}


def validate_stepwise_plan(
    plan: Any,
    *,
    valid_primitives: Mapping[str, int],
    allowed_object_ids: Set[str],
) -> tuple[str, str, Any] | None:
    """Return a canonical action tuple only when the entire proposal is valid."""

    if not isinstance(plan, dict) or not isinstance(plan.get("action"), str):
        return None
    action = plan["action"].strip()
    if re.fullmatch(r"(?:\d+\.\s*)?DONE\s*\(\s*\)", action, re.IGNORECASE):
        return "done", "", plan.get("caution")

    match = re.fullmatch(
        r"(?:\d+\.\s*)?([a-zA-Z_]+)\s*\(([^()]*)\)\s*",
        action,
    )
    if match is None:
        return None
    operator, raw_params = match.groups()
    operator_key = operator.strip().upper()
    if operator_key not in valid_primitives:
        return None
    objects = [item.strip().lower() for item in raw_params.split(",")]
    if len(objects) != int(valid_primitives[operator_key]):
        return None
    allowed = {str(item).strip().lower() for item in allowed_object_ids}
    if any(not item or item not in allowed for item in objects):
        return None
    return operator_key.lower(), ",".join(objects), plan.get("caution")
