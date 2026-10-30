"""Configuration for the SafePred online guard."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from experiments.wm_return_perturbations import (
    CANONICAL_WM_ALL_V1,
    WM_RETURN_SIGNAL_SOURCE_POLICY_ENV,
    normalize_wm_return_signal_source_policy,
    validate_arm_perturbation,
    wm_return_signal_source_provenance,
)

from .intervention import (
    ENFORCED_GUARD,
    REAL_GUIDANCE,
    REAL_SIGNAL,
    normalize_guidance_mode,
    normalize_intervention_mode,
    normalize_risk_signal_mode,
)


SUPPORTED_POLICY_PERTURBATIONS = {
    "none",
    "drop_half",
    "drop_all",
    "remove_high_risk",
}
SUPPORTED_OBSERVATION_PERTURBATIONS = {
    "none",
    "drop_history",
    "redact_content",
    "adversarial_note",
}


def _normalize_choice(value: Any, *, default: str, supported: set[str], label: str) -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    if normalized not in supported:
        raise ValueError(
            f"Unsupported {label} {value!r}. Expected one of {sorted(supported)}."
        )
    return normalized


@dataclass(frozen=True)
class SafePredGuardSettings:
    enabled: bool = False
    policy_mode: str = "full_harnessaudit_rules"
    intervention_mode: str = ENFORCED_GUARD
    config_path: str | None = None
    model_name: str | None = None
    candidate_count: int = 1
    max_regeneration_attempts: int = 1
    horizon: int = 5
    prediction_interval: int = 1
    artifact_dir: str | None = None
    candidate_model: str | None = None
    candidate_temperature: float = 0.4
    fail_closed: bool = True
    max_trace_events: int = 24
    risk_signal_mode: str = REAL_SIGNAL
    risk_noise_rate: float = 0.2
    guidance_mode: str = REAL_GUIDANCE
    policy_perturbation: str = "none"
    observation_perturbation: str = "none"
    wm_return_perturbation: str = "none"
    wm_return_signal_source_policy: str = CANONICAL_WM_ALL_V1
    experiment_seed: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SafePredGuardSettings":
        if not raw:
            return cls(enabled=False)
        candidate_count = max(1, int(raw.get("candidate_count") or 1))
        max_regeneration_attempts = max(
            0,
            int(raw.get("max_regeneration_attempts") or 0),
        )
        horizon = max(1, int(raw.get("horizon") or 5))
        prediction_interval = max(1, int(raw.get("prediction_interval") or 1))
        max_trace_events = max(1, int(raw.get("max_trace_events") or 24))
        risk_noise_rate = float(raw.get("risk_noise_rate", 0.2))
        if not 0.0 <= risk_noise_rate <= 1.0:
            raise ValueError("risk_noise_rate must be in [0, 1]")
        intervention_mode = normalize_intervention_mode(raw.get("intervention_mode"))
        _, wm_return_perturbation = validate_arm_perturbation(
            intervention_mode,
            raw.get("wm_return_perturbation"),
        )
        wm_return_signal_source_policy = normalize_wm_return_signal_source_policy(
            raw.get("wm_return_signal_source_policy")
            if "wm_return_signal_source_policy" in raw
            else os.environ.get(WM_RETURN_SIGNAL_SOURCE_POLICY_ENV)
        )
        source_provenance = wm_return_signal_source_provenance(
            source_policy=wm_return_signal_source_policy,
            perturbation=wm_return_perturbation,
        )
        if (
            source_provenance["signal_source"] in {"ground_truth_oracle", "isolated_environment_h1"}
            and horizon != 1
        ):
            raise ValueError(f"{wm_return_perturbation} requires horizon=1")
        if wm_return_perturbation == "environment_h1" and (
            prediction_interval != 1 or candidate_count != 1
            or max_regeneration_attempts != 1
        ):
            raise ValueError("environment_h1 requires interval=1, candidate_count=1, regeneration=1")
        risk_signal_mode = normalize_risk_signal_mode(raw.get("risk_signal_mode"))
        guidance_mode = normalize_guidance_mode(raw.get("guidance_mode"))
        policy_perturbation = _normalize_choice(
            raw.get("policy_perturbation"),
            default="none",
            supported=SUPPORTED_POLICY_PERTURBATIONS,
            label="policy perturbation",
        )
        observation_perturbation = _normalize_choice(
            raw.get("observation_perturbation"),
            default="none",
            supported=SUPPORTED_OBSERVATION_PERTURBATIONS,
            label="observation perturbation",
        )
        if intervention_mode == "distributed_enforced" and (
            risk_signal_mode != REAL_SIGNAL
            or guidance_mode != REAL_GUIDANCE
            or policy_perturbation != "none"
            or observation_perturbation != "none"
        ):
            raise ValueError(
                "distributed_enforced cannot be combined with legacy risk, guidance, "
                "policy, or observation perturbations"
            )
        if (
            intervention_mode == "distributed_enforced"
            and wm_return_signal_source_policy != CANONICAL_WM_ALL_V1
            and candidate_count != 1
        ):
            raise ValueError(
                "The diagnostic WM-return signal source split requires "
                "candidate_count=1"
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            policy_mode=str(raw.get("policy_mode") or "full_harnessaudit_rules"),
            intervention_mode=intervention_mode,
            config_path=raw.get("config_path") or None,
            model_name=raw.get("model_name") or None,
            candidate_count=candidate_count,
            max_regeneration_attempts=max_regeneration_attempts,
            horizon=horizon,
            prediction_interval=prediction_interval,
            artifact_dir=raw.get("artifact_dir") or None,
            candidate_model=raw.get("candidate_model") or None,
            candidate_temperature=float(raw.get("candidate_temperature") or 0.4),
            fail_closed=bool(raw.get("fail_closed", True)),
            max_trace_events=max_trace_events,
            risk_signal_mode=risk_signal_mode,
            risk_noise_rate=risk_noise_rate,
            guidance_mode=guidance_mode,
            policy_perturbation=policy_perturbation,
            observation_perturbation=observation_perturbation,
            wm_return_perturbation=wm_return_perturbation,
            wm_return_signal_source_policy=wm_return_signal_source_policy,
            experiment_seed=int(raw.get("experiment_seed") or 0),
        )


def find_workspace_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "SafePred").exists() and (candidate / "harness_env").exists():
            return candidate
    raise RuntimeError(f"Cannot locate workspace root from {start}")


def default_safepred_config_path(start: Path) -> Path:
    root = find_workspace_root(start)
    return root / "SafePred" / "config" / "harnessaudit_online_guard.yaml"
