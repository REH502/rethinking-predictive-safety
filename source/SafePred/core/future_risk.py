"""Structured output helpers for SafePred future-risk predictions."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


WM_RISK_PREDICTION_EVENT_TYPE = "wm_risk_prediction"


FUTURE_RISK_OUTPUT_SCHEMA: Dict[str, Any] = {
    "predicted_risk": "boolean",
    "risk_score": "number from 0.0 to 1.0",
    "confidence": "number from 0.0 to 1.0",
    "predicted_violation_codes": ["V-ID", "V-OT", "V-OR", "V-IC"],
    "predicted_policy_ids": ["policy ids from SafePred policies"],
    "predicted_future_step_offset": "integer offset within horizon, or null",
    "predicted_agent_role": "agent role expected to cause the risk, or null",
    "predicted_target_role": "receiver role for information-flow risk, or null",
    "predicted_tool_name": "tool name for tool/resource risk, or null",
    "risk_explanation": "short explanation grounded in the trace prefix",
    "risk_guidance": "actionable guidance for avoiding the predicted risk",
}


def normalize_violation_codes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_codes = [value]
    elif isinstance(value, list):
        raw_codes = value
    else:
        raw_codes = [str(value)]
    codes: List[str] = []
    for code in raw_codes:
        text = str(code).strip().upper()
        if not text:
            continue
        if text not in codes:
            codes.append(text)
    return codes


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_float(value: Any, *, field: str, default: float | None = None) -> float:
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field} is required")
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def build_wm_risk_prediction_event(
    prediction: Dict[str, Any],
    *,
    prediction_sequence_num: Optional[int] = None,
    horizon: int = 5,
    predictor: str = "safepred_wm",
    risk_threshold: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize a SafePred WM output into a HarnessAudit trace event.

    The event is intentionally compatible with HarnessAudit's JSONL trace style:
    it has a ``type`` field, can be written as one JSONL row, and later can be
    matched against ``access_decision.action_sequence_num``.
    """
    metadata = metadata or {}
    risk_score = _to_float(prediction.get("risk_score"), field="risk_score")
    confidence = _to_float(
        prediction.get("confidence"),
        field="confidence",
        default=risk_score,
    )
    predicted_violation_codes = normalize_violation_codes(
        prediction.get("predicted_violation_codes")
        or prediction.get("violation_codes")
        or prediction.get("predicted_violation_code")
    )
    predicted_policy_ids = [str(item) for item in _as_list(prediction.get("predicted_policy_ids") or prediction.get("violated_policy_ids")) if str(item).strip()]
    explicit_risk = _to_optional_bool(prediction.get("predicted_risk"))
    if prediction.get("predicted_risk") is not None and explicit_risk is None:
        raise ValueError(
            "WM risk prediction field 'predicted_risk' must be a JSON boolean"
        )
    inferred_risk = risk_score >= float(risk_threshold) or bool(
        predicted_violation_codes or predicted_policy_ids
    )
    predicted_risk = explicit_risk if explicit_risk is not None else inferred_risk
    future_offset = _to_optional_int(
        prediction.get("predicted_future_step_offset")
        or prediction.get("future_step_offset")
    )
    explicit_future_seq = _to_optional_int(
        prediction.get("predicted_future_action_sequence_num")
        or prediction.get("future_action_sequence_num")
    )
    predicted_future_action_sequence_num = explicit_future_seq
    if predicted_future_action_sequence_num is None and prediction_sequence_num is not None and future_offset is not None:
        predicted_future_action_sequence_num = prediction_sequence_num + future_offset

    risk_window = None
    if prediction_sequence_num is not None:
        risk_window = {
            "start_sequence_num": prediction_sequence_num + 1,
            "end_sequence_num": prediction_sequence_num + int(horizon),
        }

    return {
        "type": WM_RISK_PREDICTION_EVENT_TYPE,
        "predictor": predictor,
        "prediction_sequence_num": prediction_sequence_num,
        "horizon": int(horizon),
        "risk_window": risk_window,
        "predicted_risk": predicted_risk,
        "risk_threshold": float(risk_threshold),
        "risk_score": risk_score,
        "confidence": confidence,
        "predicted_violation_codes": predicted_violation_codes,
        "predicted_policy_ids": predicted_policy_ids,
        "predicted_future_step_offset": future_offset,
        "predicted_future_action_sequence_num": predicted_future_action_sequence_num,
        "predicted_agent_role": prediction.get("predicted_agent_role"),
        "predicted_target_role": prediction.get("predicted_target_role"),
        "predicted_tool_name": prediction.get("predicted_tool_name"),
        "risk_explanation": prediction.get("risk_explanation", "") or "",
        "risk_guidance": prediction.get("risk_guidance", "") or "",
        "metadata": metadata,
        "raw_prediction": prediction,
    }
