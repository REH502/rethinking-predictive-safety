"""Fail-closed preregistration contract for confirmatory experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from multi_agent.trace.provenance import (
    DETERMINISTIC_CLOSE_SOURCE_PROFILE,
    SEEDED_SAMPLING_PROFILE,
    implementation_manifest,
    sha256_file,
    stable_json_sha256,
    validate_actor_provenance,
)


SCHEMA_VERSION = 3
BENCHMARKS = {"harnessaudit", "isbench"}
POLICY_MODES_BY_BENCHMARK = {
    "harnessaudit": {"full_harnessaudit_rules", "train_policy"},
    "isbench": {
        "generic_embodied_policy",
        "train_policy",
        "full_isbench_policy_text",
    },
}
ISBENCH_TRAIN_POLICY_ONLY_MODES = {"train_policy"}
ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS = (
    POLICY_MODES_BY_BENCHMARK["isbench"],
    ISBENCH_TRAIN_POLICY_ONLY_MODES,
)
POINT_SELECTION_SCHEMES = {"start_mid", "all", "first", "interval", "explicit"}
MULTIPLICITY_METHODS = {
    "holm",
    "fdr_bh",
    "hierarchical_confirmatory",
    "joint_task_cluster_max_deviation",
}
FAITHFULNESS_STAGES = [
    "correct_positive_prediction",
    "signal_exposed_to_actor",
    "risk_acknowledged",
    "action_materially_revised",
    "revision_independently_safer",
    "terminal_revision_executed",
    "paired_task_safety_improved_within_utility_margin",
]
ANNOTATION_CONTRACT_FILE_FIELDS = (
    "compositionality_manifest_sha256",
    "faithfulness_adjudication_protocol_sha256",
    "adjudicator_blinding_protocol_sha256",
)
ONLINE_INTERVENTION_MODES = {
    "no_guard",
    "prediction_only",
    "advisory_only",
    "enforced_guard",
}
ISBENCH_THREE_ARM_INTERVENTION_MODES = {
    "no_guard",
    "advisory_only",
    "enforced_guard",
}
ISBENCH_DISTRIBUTED_FIVE_ARM_INTERVENTION_MODES = {
    *ONLINE_INTERVENTION_MODES,
    "distributed_enforced",
}
ISBENCH_DISTRIBUTED_FOUR_ARM_INTERVENTION_MODES = {
    *ISBENCH_THREE_ARM_INTERVENTION_MODES,
    "distributed_enforced",
}
ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS = (
    ONLINE_INTERVENTION_MODES,
    ISBENCH_THREE_ARM_INTERVENTION_MODES,
    ISBENCH_DISTRIBUTED_FIVE_ARM_INTERVENTION_MODES,
    ISBENCH_DISTRIBUTED_FOUR_ARM_INTERVENTION_MODES,
)
ISBENCH_FORMAL_EXECUTION_CONTRACT = {
    "schema_version": "isbench-online-execution-contract-v1",
    "data_parallel": 1,
    "task_timeout_seconds": 1800,
    "condition_timeout_seconds": 151200,
    "condition_termination_grace_seconds": 60,
    "condition_start_margin_seconds": 3600,
    "deadline_policy_version": "condition_start_budget_v1",
    "prompt_setting": "v1",
    "draw_bbox_2d": True,
    "skip_viewer_obs": False,
    "continue_on_error": True,
    "endpoint_preflight": True,
    "run_mode": "formal_full_cohort",
    "report_validation_profile": "online_strict_v1",
}
ISBENCH_FORMAL_DATA_PARALLEL_RANGE = range(1, 5)
ISBENCH_FORMAL_CONDITION_TIMEOUT_SECONDS = (43200, 72000, 151200)
ISBENCH_FORMAL_SCHEDULE_VERSION = "isbench_seed_block_sha256_v1"
ISBENCH_FORMAL_SCHEDULE_SEED = 20260815


# One immutable contract binds condition-internal resume across the formal
# preregistration, runtime identity, and provenance.  The runner imports this
# mapping rather than maintaining a second independent set of literals.
HA_TRACE_RESUME_CONTRACT = {
    "schema_version": "harnessaudit-matrix-trace-resume-v1",
    "enabled": True,
    "run_id_strategy": "full-condition-actor-judge-sha256-v1",
    "runtime_root_strategy": "managed-roots-commonpath-bounded-v1",
    "checkpoint_strategy": "condition-scoped-sqlite-v1",
    "checkpoint_layout": (
        "_trace_checkpoints/<policy>/<arm>/<condition>/<task>/checkpoint.sqlite"
    ),
    "attempt_archive_schema_version": "ha-condition-attempt-archive-v1",
    "attempt_archive_strategy": "copy-before-retry-hash-bound-v1",
    "attempt_archive_layout": (
        "_attempt_orphans/<policy>/<arm>/<condition>/<task>/attempt-NNNNNN"
    ),
    "matrix_resume_orthogonal": True,
}

# A v3-only utility endpoint for an integrity-validated enforced terminal.
# It never relabels the terminal as task success and never imputes ordinary
# failures: raw TCR/AVS remain unscored whenever a terminal is present.
HA_SAFETY_ABSTENTION_ITT_CONTRACT_V1 = {
    "schema_version": "ha-safety-abstention-itt-v1",
    "valid_terminal_status": "safety_abstained",
    "valid_terminal_task_status": "not_scored",
    "raw_tcr_status": "not_evaluable",
    "itt_endpoint": "tcr_itt_fail_closed_v1",
    "safety_abstention_tcr": 0.0,
    "failure_policy": "invalid_matrix_no_imputation_v1",
    "avs_policy": "unevaluable_if_any_safety_abstention_v1",
    "denominator": "all_preregistered_task_seed_conditions_v1",
}


def isbench_formal_execution_contract_errors(value: Any) -> list[str]:
    """Validate a preregistered IS-Bench execution envelope.

    Task-level parallelism and the whole-condition infrastructure timeout are
    execution parameters, not treatments.  Keep every scientific and
    validation field fixed while allowing bounded preregistered values that
    match the available simulator allocation.  Runtime validation still
    requires the observed values to equal the frozen preregistration exactly.
    """

    if not isinstance(value, dict):
        return ["must be an object"]
    expected_keys = set(ISBENCH_FORMAL_EXECUTION_CONTRACT)
    if set(value) != expected_keys:
        return [f"must contain exactly {sorted(expected_keys)}"]
    errors: list[str] = []
    data_parallel = value.get("data_parallel")
    if type(data_parallel) is not int or data_parallel not in (
        ISBENCH_FORMAL_DATA_PARALLEL_RANGE
    ):
        errors.append("data_parallel must be an exact integer from 1 through 4")
    condition_timeout = value.get("condition_timeout_seconds")
    if type(condition_timeout) is not int or condition_timeout not in (
        ISBENCH_FORMAL_CONDITION_TIMEOUT_SECONDS
    ):
        errors.append(
            "condition_timeout_seconds must equal 43200, 72000, or 151200"
        )
    for field, expected in ISBENCH_FORMAL_EXECUTION_CONTRACT.items():
        if field in {"data_parallel", "condition_timeout_seconds"}:
            continue
        if value.get(field) != expected:
            errors.append(f"{field} must equal {expected!r}")
    return errors
ISBENCH_HARDWARE_PROFILE_SCHEMA_VERSION = (
    "omnigibson-empirical-hardware-profile-v1"
)
ISBENCH_A10040_HARDWARE_PROFILE_ID = "omnigibson_a10040_empirical_v1"
ISBENCH_A10040_HARDWARE_PROFILE_ARTIFACT_CLASS = (
    "prospective_infrastructure_qualification_contract_not_experiment_evidence"
)
ISBENCH_A10040_PRIMARY_CLAIM_SCOPE = "within_hardware_arm_contrasts_only"
ISBENCH_A10040_HARDWARE_CLAIMS = {
    "vendor_supported": False,
    "rt_cores": False,
    "external_benchmark_comparability": False,
    "claim_scope": ISBENCH_A10040_PRIMARY_CLAIM_SCOPE,
}
ISBENCH_A10040_SLURM_IDENTITY = {
    "node": "gpu16",
    "gres": "gpu:a100-pcie-40gb:1",
    "gpu_count": 1,
}
ISBENCH_A10040_OBSERVED_HARDWARE_CONSTRAINTS = {
    "exact_nvidia_smi_name": "NVIDIA A100-PCIE-40GB",
    "minimum_memory_total_mib": 40000,
}
ISBENCH_A10040_REGISTERED_WORKLOAD = {
    "task": "removing_lint_from_dryer",
    "scene": "Beechwood_0_garden",
    "initial_view_count": 5,
    "physics_steps": 60,
    "final_render_required": True,
    "isolated_cold_starts": 2,
    "maximum_peak_memory_fraction": 0.90,
}
ISBENCH_A10040_QUALIFICATION = {
    "success_status": "empirical_hardware_qualified_for_registered_workload_smoke",
    "eligible_for_formal_experiment": False,
    "required_next_evidence": "registered_full_task_compatible_online_smoke",
    "prospective_only": True,
    "prior_artifact_reuse": False,
}
ISBENCH_RTX_A6000_HARDWARE_PROFILE_ID = (
    "omnigibson_rtx_a6000_empirical_v1"
)
ISBENCH_RTX_A6000_THREE_GPU_HARDWARE_PROFILE_ID = (
    "omnigibson_rtx_a6000_empirical_v2"
)
ISBENCH_RTX_A6000_HARDWARE_PROFILE_ARTIFACT_CLASS = (
    "prospective_infrastructure_qualification_contract_not_experiment_evidence"
)
ISBENCH_RTX_A6000_HARDWARE_CLAIMS = {
    "vendor_supported": True,
    "rt_cores": True,
    "external_benchmark_comparability": False,
    "claim_scope": "within_hardware_arm_contrasts_only",
}
ISBENCH_RTX_A6000_SLURM_IDENTITY = {
    "node": "gpu07",
    "gres": "gpu:nvidia_rtx_a6000:1",
    "gpu_count": 1,
}
ISBENCH_RTX_A6000_THREE_GPU_SLURM_IDENTITY = {
    "node": "gpu07",
    "gres": "gpu:nvidia_rtx_a6000:3",
    "gpu_count": 3,
}
ISBENCH_RTX_A6000_OBSERVED_HARDWARE_CONSTRAINTS = {
    "exact_nvidia_smi_name": "NVIDIA RTX A6000",
    "minimum_memory_total_mib": 48000,
}
ISBENCH_RTX_A6000_REGISTERED_WORKLOAD = dict(
    ISBENCH_A10040_REGISTERED_WORKLOAD
)
ISBENCH_RTX_A6000_QUALIFICATION = dict(ISBENCH_A10040_QUALIFICATION)
ISBENCH_RTX_A6000_PROBE_CONTRACT = {
    "launcher_source_path": (
        "scripts/slurm/omnigibson_a6000_empirical_probe.sbatch"
    ),
    "probe_python_source_path": "scripts/probe_omnigibson_gpu.py",
    "run_schema_version": "omnigibson-empirical-hardware-probe-run-v1",
    "summary_schema_version": (
        "omnigibson-empirical-hardware-qualification-v1"
    ),
    "run_root_pattern": (
        "runs/omnigibson_gpu_probe/rtx_a6000_empirical_v1/"
        "job-${SLURM_JOB_ID}"
    ),
    "gpu_label": "nvidia-rtx-a6000",
    "required_repetitions": 2,
}
ISBENCH_REGISTERED_SINGLE_SEED_HARDWARE_PROFILE_IDS = {
    ISBENCH_A10040_HARDWARE_PROFILE_ID,
    ISBENCH_RTX_A6000_HARDWARE_PROFILE_ID,
    ISBENCH_RTX_A6000_THREE_GPU_HARDWARE_PROFILE_ID,
}
ISBENCH_FULL161_ORDERED_TASK_IDS_SHA256 = (
    "4f61946f1c37192db85307077b1ba8d7cf4f9392d5aab71f8ac5e378d31f1a75"
)
ISBENCH_REGISTERED_HARDWARE_PROFILE_SOURCE_PATHS = {
    ISBENCH_A10040_HARDWARE_PROFILE_ID: (
        "config/hardware_profiles/omnigibson_a10040_empirical_v1.json"
    ),
    ISBENCH_RTX_A6000_HARDWARE_PROFILE_ID: (
        "config/hardware_profiles/omnigibson_rtx_a6000_empirical_v1.json"
    ),
    ISBENCH_RTX_A6000_THREE_GPU_HARDWARE_PROFILE_ID: (
        "config/hardware_profiles/omnigibson_rtx_a6000_empirical_v2.json"
    ),
}
ISBENCH_HARDWARE_PROFILE_SOURCE_KEYS = {
    "schema_version",
    "profile_id",
    "artifact_class",
    "slurm",
    "observed_hardware_constraints",
    "hardware_claims",
    "registered_workload",
    "qualification",
    "cost",
    "probe_contract",
}
_ISBENCH_POLICY_ALIASES = {
    "generic_embodied_policy": "generic",
    "train_policy": "train",
    "full_isbench_policy_text": "full_text",
}
_ISBENCH_INTERVENTION_ALIASES = {
    "prediction_only": "prediction",
    "advisory_only": "advisory",
    "enforced_guard": "enforced",
    "distributed_enforced": "distributed",
}


def isbench_formal_condition_order(
    *,
    experiment_seeds: Iterable[int],
    policy_modes: Iterable[str],
    intervention_modes: Iterable[str],
    wm_return_perturbations: Iterable[str] = (
        "empty",
        "corrupt",
        "irrelevant",
        "filler",
    ),
    schedule_seed: int = ISBENCH_FORMAL_SCHEDULE_SEED,
    version: str = ISBENCH_FORMAL_SCHEDULE_VERSION,
) -> list[str]:
    """Return the registered seed-blocked deterministic condition order.

    Hash ordering avoids dependence on Python's random implementation.  The
    schedule seed is explicit even though the experiment seeds are already
    frozen, so a future ordering change requires a new registration.
    """

    seeds = sorted(dict.fromkeys(int(item) for item in experiment_seeds))
    policies = list(dict.fromkeys(str(item) for item in policy_modes))
    interventions = [
        str(item)
        for item in dict.fromkeys(str(item) for item in intervention_modes)
        if str(item) != "no_guard"
    ]
    return_perturbations = list(
        dict.fromkeys(str(item) for item in wm_return_perturbations)
    )
    if "distributed_enforced" in interventions and set(return_perturbations) != {
        "empty",
        "corrupt",
        "irrelevant",
        "filler",
    }:
        raise ValueError(
            "formal IS-Bench distributed_enforced requires exactly four WM-return perturbations"
        )
    if set(policies) not in ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS:
        raise ValueError(
            "formal IS-Bench schedule requires either all three policy tiers "
            "or the registered train_policy-only tier"
        )
    registered_modes = {"no_guard", *interventions}
    if registered_modes not in ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS:
        raise ValueError(
            "formal IS-Bench schedule requires a registered legacy or "
            "distributed intervention arm set"
        )
    if not seeds or version != ISBENCH_FORMAL_SCHEDULE_VERSION:
        raise ValueError("formal IS-Bench schedule identity is invalid")
    ordered: list[str] = []
    for seed in seeds:
        candidates = [f"baseline__no_guard__seed-{seed}"]
        for policy in policies:
            for intervention in interventions:
                if intervention == "distributed_enforced":
                    candidates.extend(
                        f"{_ISBENCH_POLICY_ALIASES[policy]}__"
                        f"{_ISBENCH_INTERVENTION_ALIASES[intervention]}__"
                        f"wmret-{mode}__seed-{seed}"
                        for mode in return_perturbations
                    )
                else:
                    candidates.append(
                        f"{_ISBENCH_POLICY_ALIASES[policy]}__"
                        f"{_ISBENCH_INTERVENTION_ALIASES[intervention]}__seed-{seed}"
                    )
        ordered.extend(
            sorted(
                candidates,
                key=lambda condition_id: hashlib.sha256(
                    (
                        f"{version}\0{int(schedule_seed)}\0{seed}\0{condition_id}"
                    ).encode("utf-8")
                ).digest(),
            )
        )
    return ordered


def isbench_formal_condition_order_sha256(condition_order: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(condition_order),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
HA_THREE_ARM_INTERVENTION_MODES = {
    "no_guard",
    "advisory_only",
    "enforced_guard",
}
HA_DISTRIBUTED_FIVE_ARM_INTERVENTION_MODES = {
    *ONLINE_INTERVENTION_MODES,
    "distributed_enforced",
}
HA_DISTRIBUTED_FOUR_ARM_INTERVENTION_MODES = {
    *HA_THREE_ARM_INTERVENTION_MODES,
    "distributed_enforced",
}
HA_ALLOWED_ONLINE_INTERVENTION_MODE_SETS = (
    ONLINE_INTERVENTION_MODES,
    HA_THREE_ARM_INTERVENTION_MODES,
    HA_DISTRIBUTED_FIVE_ARM_INTERVENTION_MODES,
    HA_DISTRIBUTED_FOUR_ARM_INTERVENTION_MODES,
)
HA_FAILURE_POLICIES = {
    "continue_and_retain_all_conditions_v1",
    "continue_and_retain_failed_task_conditions_v2",
}
HA_ONLINE_TASK_COHORT_SCHEMA_VERSION = "harnessaudit-online-task-cohort-v1"
HA_ONLINE_TASK_COHORT_SELECTION_RULE = (
    "canonical_non_sde_minus_excluded_pilot_v1"
)
HA_ONLINE_TASK_COHORT_SDE_POLICY = "separate_registered_cohort_required"
HA_ONLINE_TASK_COHORT_EXCLUDED_PILOT_IDS = (
    "dl-t4",
    "dl-t5",
    "ec-t1",
    "ec-t11",
    "fin-t1",
    "fin-t10",
    "hc-t1",
    "lc-t1",
)
HA_ONLINE_TASK_COHORT_CONTRACT_KEYS = {
    "source_path",
    "source_sha256",
    "schema_version",
    "cohort_id",
    "selection_rule",
    "resource_semantics_mode",
    "canonical_non_sde_task_count",
    "excluded_pilot_task_count",
    "selected_task_count",
    "sde_task_count",
    "canonical_task_ids_sha256",
    "excluded_task_ids_sha256",
    "selected_task_ids_sha256",
    "sde_task_ids_sha256",
    "canonical_task_files_sha256",
    "selected_task_files_sha256",
    "sde_task_files_sha256",
    "sde_registration_policy",
}
HA_SINGLE_FIXED_SEED_SAMPLING_CONTRACT = {
    "schema_version": "ha-complete-task-census-single-fixed-seed-v1",
    "fixed_seed": 0,
    "seed_count": 1,
    "inference_scope": "task_census_conditional_on_fixed_seed",
    "model_sampling_variance_claim": False,
}
HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION = (
    "ha-policy-extraction-contract-v2"
)
HA_LEGACY_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION = (
    "ha-policy-extraction-contract-v1"
)
HA_MODEL_INPUT_BOUNDARY_VERSION = (
    "harnessaudit-model-input-boundary-v2-ground-truth-only-tool-necessity"
)
HA_POLICY_EXTRACTION_PROMPT_VERSION = (
    "harnessaudit-train-policy-extraction-v2-no-test-tool-necessity"
)
HA_POLICY_CACHE_SCHEMA_VERSION = 5
HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS = (
    "task_id",
    "domain",
    "goal",
    "agents.role",
    "agents.description",
)
HA_POLICY_SOURCE_IDENTITY = (
    "harnessaudit_registered_policy_sources_v2_no_test_tool_necessity"
)
HA_TRAIN_POLICY_CONSTRUCTION_VERSION = (
    "harnessaudit-train-policy-v2-no-test-tool-necessity"
)
HA_DIRECT_PROMPT_VERSION = (
    "safepred-direct-single-horizon-v4-no-test-tool-necessity"
)
HA_ITERATIVE_PROMPT_VERSION = (
    "safepred-iterative-world-model-v3-no-test-tool-necessity"
)


@dataclass(frozen=True)
class ActorReplicateIdentity:
    actor_protocol_profile: str
    actor_replicate_id: str
    actor_trace_seed: int | None

    @property
    def full_key(self) -> tuple[str, str, int | None]:
        return (
            self.actor_protocol_profile,
            self.actor_replicate_id,
            self.actor_trace_seed,
        )

    @property
    def cluster_key(self) -> tuple[str, str]:
        return (self.actor_protocol_profile, self.actor_replicate_id)


def normalize_actor_replicate_identity(
    *,
    benchmark: str,
    actor_protocol_profile: Any = None,
    actor_replicate_id: Any = None,
    actor_trace_seed: Any = None,
) -> ActorReplicateIdentity:
    """Normalize legacy seeded or expanded actor replicate identity fields."""

    normalized_benchmark = str(benchmark).strip().lower()
    if normalized_benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported preregistration benchmark: {benchmark!r}")
    profile_present = actor_protocol_profile is not None
    replicate_present = actor_replicate_id is not None
    seed = (
        actor_trace_seed
        if type(actor_trace_seed) is int
        else None
    )
    if not profile_present and not replicate_present:
        if seed is None:
            raise ValueError(
                "Legacy actor replicate identity requires actor_trace_seed "
                "to be an exact integer"
            )
        return ActorReplicateIdentity(
            SEEDED_SAMPLING_PROFILE,
            f"seed-{seed}",
            seed,
        )
    if not profile_present or not replicate_present:
        raise ValueError(
            "Expanded actor replicate identity requires both "
            "actor_protocol_profile and actor_replicate_id"
        )
    if type(actor_protocol_profile) is not str:
        raise ValueError("actor_protocol_profile must be a string")
    if type(actor_replicate_id) is not str:
        raise ValueError("actor_replicate_id must be a string")
    profile = actor_protocol_profile.strip()
    replicate_id = actor_replicate_id.strip()
    if not replicate_id:
        raise ValueError("actor_replicate_id must be non-empty")
    if profile == SEEDED_SAMPLING_PROFILE:
        if seed is None:
            raise ValueError(
                "seeded_sampling_v1 requires actor_trace_seed to be an exact integer"
            )
        return ActorReplicateIdentity(profile, replicate_id, seed)
    if profile == DETERMINISTIC_CLOSE_SOURCE_PROFILE:
        if normalized_benchmark != "harnessaudit":
            raise ValueError(
                "ISBench remains seeded and rejects deterministic actor identities"
            )
        if actor_trace_seed is not None:
            raise ValueError(
                "deterministic_close_source_v1 requires null actor_trace_seed"
            )
        return ActorReplicateIdentity(profile, replicate_id, None)
    raise ValueError(f"Unknown actor_protocol_profile: {profile!r}")


def _require_explicit_actor_identity_fields(
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    required = {
        "actor_protocol_profile",
        "actor_replicate_id",
        "actor_trace_seed",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"{label} must explicitly contain actor profile, replicate, and "
            f"nullable seed fields; missing={missing}"
        )


def read_ha_source_replicate_identity(
    report_paths: Iterable[Path],
    *,
    expected_actor_trace_seed: int | None = None,
) -> ActorReplicateIdentity:
    """Read one authoritative formal HA identity from reports and native traces."""

    identities: list[ActorReplicateIdentity] = []
    for raw_report_path in report_paths:
        report_path = Path(raw_report_path).expanduser().resolve()
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read formal HA source report {report_path}: {exc}") from exc
        if not isinstance(report, dict):
            raise ValueError(f"Formal HA source report root must be an object: {report_path}")
        identity_fields = {
            "actor_protocol_profile",
            "actor_replicate_id",
            "actor_trace_seed",
        }
        missing_report_fields = sorted(identity_fields - set(report))
        if missing_report_fields:
            raise ValueError(
                "Formal HA report top-level must explicitly contain actor identity "
                f"fields, including nullable actor_trace_seed: "
                f"report={report_path}, missing={missing_report_fields}"
            )
        provenance = report.get("actor_provenance")
        provenance_errors = validate_actor_provenance(provenance, formal=True)
        if provenance_errors:
            raise ValueError(
                "Formal HA report lacks valid native pre-call actor provenance: "
                f"report={report_path}, errors={provenance_errors}"
            )
        assert isinstance(provenance, dict)
        missing_provenance_fields = sorted(identity_fields - set(provenance))
        if missing_provenance_fields:
            raise ValueError(
                "Formal HA actor provenance must explicitly contain profile, "
                f"replicate, and nullable seed: report={report_path}, "
                f"missing={missing_provenance_fields}"
            )
        identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_protocol_profile=report.get("actor_protocol_profile"),
            actor_replicate_id=report.get("actor_replicate_id"),
            actor_trace_seed=report.get("actor_trace_seed"),
        )
        nested_identity = normalize_actor_replicate_identity(
            benchmark="harnessaudit",
            actor_protocol_profile=provenance.get("actor_protocol_profile"),
            actor_replicate_id=provenance.get("actor_replicate_id"),
            actor_trace_seed=provenance.get("actor_trace_seed"),
        )
        if identity != nested_identity:
            raise ValueError(
                "Formal HA report top-level actor identity differs from "
                f"actor provenance: {report_path}"
            )
        trace_value = Path(str(report.get("trace_file") or ""))
        trace_path = (
            trace_value
            if trace_value.is_absolute()
            else (report_path.parent / trace_value)
        ).resolve()
        try:
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read formal HA native trace {trace_path}: {exc}") from exc
        trace_starts = [
            event
            for event in events
            if isinstance(event, dict) and event.get("type") == "trace_start"
        ]
        if len(trace_starts) != 1:
            raise ValueError(
                "Formal HA trace must contain exactly one trace_start: "
                f"trace={trace_path}, observed={len(trace_starts)}"
            )
        trace_start = trace_starts[0]
        if "actor_trace_seed" not in trace_start:
            raise ValueError(
                "Formal HA trace_start must explicitly contain nullable "
                f"actor_trace_seed: trace={trace_path}"
            )
        if trace_start.get("actor_provenance") != provenance:
            raise ValueError(
                "Formal HA report and native trace_start actor provenance differ: "
                f"report={report_path}, trace={trace_path}"
            )
        trace_seed = trace_start.get("actor_trace_seed")
        if identity.actor_trace_seed is None:
            seed_matches = trace_seed is None
        else:
            seed_matches = type(trace_seed) is int and trace_seed == identity.actor_trace_seed
        if not seed_matches:
            raise ValueError(
                "Formal HA trace actor_trace_seed provenance is missing or mismatched: "
                f"trace={trace_path}, embedded={trace_seed!r}, "
                f"report={identity.actor_trace_seed!r}"
            )
        identities.append(identity)
    if not identities:
        raise ValueError("Formal HA source set must contain at least one report")
    unique = {identity.full_key for identity in identities}
    if len(unique) != 1:
        raise ValueError(
            "Formal HA source reports do not share one normalized actor identity: "
            f"{sorted(unique, key=str)}"
        )
    identity = identities[0]
    if expected_actor_trace_seed is not None:
        expected = _integer(expected_actor_trace_seed)
        if expected is None or identity.actor_trace_seed != expected:
            raise ValueError(
                "--actor-trace-seed assertion does not match authoritative "
                f"source identity: expected={expected_actor_trace_seed!r}, "
                f"observed={identity.actor_trace_seed!r}"
            )
    return identity


def canonical_preregistration_sha256(payload: dict[str, Any]) -> str:
    """Return the single canonical preregistration identity used everywhere."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def sanitize_endpoint_identity(value: Any) -> str:
    """Return a credential- and query-free endpoint identity for provenance."""

    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        raise ValueError("Endpoint identity must be an absolute URL") from None
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Endpoint identity must be an absolute URL")
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def validate_formal_endpoint_identity(value: Any, *, label: str) -> str:
    """Reject secret-bearing formal endpoints without echoing their raw value."""

    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        raise ValueError(f"{label} formal endpoint must be an absolute URL") from None
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"{label} formal endpoint must be an absolute URL")
    has_query_delimiter = "?" in raw.split("#", 1)[0]
    if "@" in parsed.netloc or has_query_delimiter or "#" in raw:
        raise ValueError(
            f"{label} formal endpoint must not contain userinfo, query, or fragment"
        )
    return sanitize_endpoint_identity(raw)


def _ha_task_file_identity(
    paths: Iterable[Path],
    *,
    task_root: Path,
) -> tuple[list[str], str]:
    """Return sorted task IDs and one path-and-byte-bound digest."""

    rows: dict[str, str] = {}
    task_ids: list[str] = []
    for path in sorted(Path(item).resolve() for item in paths):
        relative = path.relative_to(task_root).as_posix()
        task_ids.append(path.stem)
        rows[relative] = sha256_file(path)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("HarnessAudit task cohort contains duplicate task IDs")
    return sorted(task_ids), stable_json_sha256(rows)


def _ha_task_cohort_identity(
    *,
    manifest_path: Path,
    payload: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    canonical = payload["canonical_non_sde"]
    excluded = payload["excluded_pilot"]
    selected = payload["selected"]
    sde = payload["sde"]
    try:
        source_path = manifest_path.relative_to(project_root).as_posix()
    except ValueError:
        source_path = str(manifest_path)
    return {
        "source_path": source_path,
        "source_sha256": sha256_file(manifest_path),
        "schema_version": payload["schema_version"],
        "cohort_id": payload["cohort_id"],
        "selection_rule": payload["selection_rule"],
        "resource_semantics_mode": payload["resource_semantics_mode"],
        "canonical_non_sde_task_count": canonical["task_count"],
        "excluded_pilot_task_count": excluded["task_count"],
        "selected_task_count": selected["task_count"],
        "sde_task_count": sde["task_count"],
        "canonical_task_ids_sha256": canonical["task_ids_sha256"],
        "excluded_task_ids_sha256": excluded["task_ids_sha256"],
        "selected_task_ids_sha256": selected["task_ids_sha256"],
        "sde_task_ids_sha256": sde["task_ids_sha256"],
        "canonical_task_files_sha256": canonical["task_files_sha256"],
        "selected_task_files_sha256": selected["task_files_sha256"],
        "sde_task_files_sha256": sde["task_files_sha256"],
        "sde_registration_policy": sde["registration_policy"],
    }


def validate_ha_online_task_cohort_manifest(
    manifest_path: Path,
    *,
    project_root: Path | None = None,
    expected_contract: Mapping[str, Any] | None = None,
    require_project_relative: bool = False,
) -> dict[str, Any]:
    """Validate the exact canonical-200-minus-pilot-8 online HA cohort.

    The validator independently rediscovers all current task YAML files, binds
    their path and byte identities, proves that the selected 192 tasks are the
    set difference, and keeps the 10 SDE tasks outside this scientific identity.
    It performs no model or endpoint access.
    """

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[5]
    )
    path = Path(manifest_path).expanduser().resolve()
    if require_project_relative:
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError(
                "HarnessAudit task cohort manifest must be inside the project"
            ) from None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read HarnessAudit task cohort manifest: {type(exc).__name__}"
        ) from None
    if not isinstance(payload, dict):
        raise ValueError("HarnessAudit task cohort manifest root must be an object")
    top_keys = {
        "schema_version",
        "cohort_id",
        "benchmark",
        "selection_rule",
        "resource_semantics_mode",
        "canonical_non_sde",
        "excluded_pilot",
        "selected",
        "sde",
    }
    if set(payload) != top_keys:
        raise ValueError(
            "HarnessAudit task cohort manifest must contain exactly "
            f"{sorted(top_keys)}"
        )
    if payload.get("schema_version") != HA_ONLINE_TASK_COHORT_SCHEMA_VERSION:
        raise ValueError("HarnessAudit task cohort manifest schema_version is invalid")
    if payload.get("benchmark") != "harnessaudit":
        raise ValueError("HarnessAudit task cohort manifest benchmark is invalid")
    if not str(payload.get("cohort_id") or "").strip():
        raise ValueError("HarnessAudit task cohort manifest cohort_id is required")
    if payload.get("selection_rule") != HA_ONLINE_TASK_COHORT_SELECTION_RULE:
        raise ValueError("HarnessAudit task cohort manifest selection_rule is invalid")
    if payload.get("resource_semantics_mode") != "legacy_official_v1":
        raise ValueError(
            "HarnessAudit remaining cohort must use legacy_official_v1"
        )

    section_keys = {
        "canonical_non_sde": {"task_count", "task_ids_sha256", "task_files_sha256"},
        "excluded_pilot": {
            "task_count",
            "task_ids",
            "task_ids_sha256",
            "exclusion_reason",
            "source_preregistration_path",
            "source_preregistration_sha256",
        },
        "selected": {
            "task_count",
            "task_ids",
            "task_ids_sha256",
            "task_files_sha256",
        },
        "sde": {
            "registration_policy",
            "task_count",
            "task_ids",
            "task_ids_sha256",
            "task_files_sha256",
        },
    }
    for name, expected_keys in section_keys.items():
        section = payload.get(name)
        if not isinstance(section, dict) or set(section) != expected_keys:
            raise ValueError(
                f"HarnessAudit task cohort {name} must contain exactly "
                f"{sorted(expected_keys)}"
            )

    task_root = (
        root
        / "harness_env"
        / "orgin"
        / "HarnessAudit"
        / "multi_agent"
        / "tasks"
    ).resolve()
    if not task_root.is_dir():
        raise ValueError("HarnessAudit canonical task root is missing")
    all_paths = sorted(path for path in task_root.rglob("*.yaml") if path.is_file())
    canonical_paths = [
        task_path
        for task_path in all_paths
        if "sde" not in task_path.relative_to(task_root).parts
    ]
    sde_paths = [
        task_path
        for task_path in all_paths
        if "sde" in task_path.relative_to(task_root).parts
    ]
    canonical_ids, canonical_files_sha256 = _ha_task_file_identity(
        canonical_paths, task_root=task_root
    )
    sde_ids, sde_files_sha256 = _ha_task_file_identity(
        sde_paths, task_root=task_root
    )
    canonical = payload["canonical_non_sde"]
    if len(canonical_ids) != 200 or canonical.get("task_count") != 200:
        raise ValueError("HarnessAudit canonical non-SDE task count must equal 200")
    if canonical.get("task_ids_sha256") != stable_json_sha256(canonical_ids):
        raise ValueError("HarnessAudit canonical non-SDE task ID digest differs")
    if canonical.get("task_files_sha256") != canonical_files_sha256:
        raise ValueError("HarnessAudit canonical non-SDE task byte digest differs")

    excluded = payload["excluded_pilot"]
    excluded_ids = excluded.get("task_ids")
    expected_excluded = list(HA_ONLINE_TASK_COHORT_EXCLUDED_PILOT_IDS)
    if excluded_ids != expected_excluded:
        raise ValueError("HarnessAudit excluded pilot task IDs differ")
    if excluded.get("task_count") != 8:
        raise ValueError("HarnessAudit excluded pilot task count must equal 8")
    if excluded.get("task_ids_sha256") != stable_json_sha256(expected_excluded):
        raise ValueError("HarnessAudit excluded pilot task ID digest differs")
    if excluded.get("exclusion_reason") != (
        "previously_registered_pilot_excluded_from_remaining_cohort"
    ):
        raise ValueError("HarnessAudit excluded pilot reason is invalid")
    if not set(expected_excluded) < set(canonical_ids):
        raise ValueError("HarnessAudit excluded pilot is not a strict canonical subset")

    source_relative = str(excluded.get("source_preregistration_path") or "")
    source_path = (root / source_relative).resolve()
    try:
        source_path.relative_to(root)
    except ValueError:
        raise ValueError("HarnessAudit excluded pilot source escapes the project") from None
    if Path(source_relative).is_absolute() or not source_path.is_file():
        raise ValueError("HarnessAudit excluded pilot source is missing")
    if sha256_file(source_path) != str(
        excluded.get("source_preregistration_sha256") or ""
    ).lower():
        raise ValueError("HarnessAudit excluded pilot preregistration digest differs")
    try:
        excluded_registration = json.loads(source_path.read_text(encoding="utf-8"))
        registered_pilot_ids = sorted(
            Path(str(item)).stem
            for item in excluded_registration["run_plans"]["harnessaudit"]["tasks"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        raise ValueError(
            "HarnessAudit excluded pilot preregistration is malformed"
        ) from None
    if registered_pilot_ids != expected_excluded:
        raise ValueError(
            "HarnessAudit excluded pilot IDs differ from source preregistration"
        )

    expected_selected = sorted(set(canonical_ids) - set(expected_excluded))
    selected = payload["selected"]
    selected_ids = selected.get("task_ids")
    if selected_ids != expected_selected:
        raise ValueError(
            "HarnessAudit selected task IDs must equal canonical non-SDE minus pilot"
        )
    if selected.get("task_count") != 192 or len(expected_selected) != 192:
        raise ValueError("HarnessAudit selected task count must equal 192")
    if selected.get("task_ids_sha256") != stable_json_sha256(expected_selected):
        raise ValueError("HarnessAudit selected task ID digest differs")
    selected_set = set(expected_selected)
    selected_paths = [item for item in canonical_paths if item.stem in selected_set]
    _, selected_files_sha256 = _ha_task_file_identity(
        selected_paths, task_root=task_root
    )
    if selected.get("task_files_sha256") != selected_files_sha256:
        raise ValueError("HarnessAudit selected task byte digest differs")

    sde = payload["sde"]
    if sde.get("registration_policy") != HA_ONLINE_TASK_COHORT_SDE_POLICY:
        raise ValueError("HarnessAudit SDE cohort must be registered separately")
    if sde.get("task_count") != 10 or len(sde_ids) != 10:
        raise ValueError("HarnessAudit SDE task count must equal 10")
    if sde.get("task_ids") != sde_ids:
        raise ValueError("HarnessAudit SDE task IDs differ")
    if sde.get("task_ids_sha256") != stable_json_sha256(sde_ids):
        raise ValueError("HarnessAudit SDE task ID digest differs")
    if sde.get("task_files_sha256") != sde_files_sha256:
        raise ValueError("HarnessAudit SDE task byte digest differs")

    identity = _ha_task_cohort_identity(
        manifest_path=path,
        payload=payload,
        project_root=root,
    )
    if expected_contract is not None and dict(expected_contract) != identity:
        raise ValueError(
            "HarnessAudit task cohort manifest differs from preregistered identity"
        )
    return {
        "schema_version": HA_ONLINE_TASK_COHORT_SCHEMA_VERSION,
        "status": "validated",
        "canonical_non_sde_task_count": len(canonical_ids),
        "excluded_pilot_task_ids": expected_excluded,
        "selected_task_ids": expected_selected,
        "sde_task_ids": sde_ids,
        "identity": identity,
    }


def _registered_input_files(*roots: Path) -> list[tuple[Path, Path, Path]]:
    """Return (root, relative, path) rows for immutable registered inputs."""

    files: list[tuple[Path, Path, Path]] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            if any(
                part in {".git", "__pycache__", ".pytest_cache"}
                for part in relative.parts
            ):
                continue
            if (
                root.name == "train_policies"
                and relative.parts
                and relative.parts[0] in {"debug", "robosafe_cache"}
            ):
                continue
            files.append((root, relative, path))
    return files


def _registered_input_tree_digest(*roots: Path) -> str:
    """Hash frozen input bytes while excluding runtime-generated artifacts."""

    rows = {
        f"{root.name}/{relative.as_posix()}": sha256_file(path)
        for root, relative, path in _registered_input_files(*roots)
    }
    return stable_json_sha256(rows)


def current_online_implementation_manifest_sha256(benchmark: str) -> str:
    """Hash the active online implementation via existing manifest helpers."""

    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARKS:
        raise ValueError(f"Unsupported online benchmark: {benchmark!r}")
    multi_agent_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[5]

    components = {
        "harnessaudit_multi_agent": implementation_manifest(multi_agent_root)[
            "sha256"
        ],
        "safepred": implementation_manifest(project_root / "SafePred")["sha256"],
        "experiment_model_boundary": implementation_manifest(
            project_root / "experiments"
        )["sha256"],
        "checkpoint_verifier": sha256_file(
            project_root / "scripts" / "verify_model_checkpoint.py"
        ),
    }
    if normalized == "harnessaudit":
        components["harnessaudit_registered_inputs"] = _registered_input_tree_digest(
            multi_agent_root / "tasks",
            multi_agent_root / "tools",
            multi_agent_root / "fixtures",
            multi_agent_root / "perturbation_specs",
            multi_agent_root / "train_policies",
            project_root / "SafePred" / "config",
        )
    else:
        from multi_agent.safepred_experiments.isbench_report_audit import (
            _implementation_sha256,
        )

        isbench_package_root = project_root / "IS-Bench" / "og_ego_prim"
        safeguard_path = isbench_package_root / "models" / "safepred_safeguard.py"
        if not safeguard_path.is_file():
            raise ValueError(
                "ISBench online implementation manifest is missing an active source"
            )
        components.update(
            {
                "isbench_trace_implementation": _implementation_sha256(
                    isbench_package_root
                ),
                "isbench_safepred_safeguard": sha256_file(safeguard_path),
                "isbench_registered_inputs": _registered_input_tree_digest(
                    project_root / "IS-Bench" / "data" / "tasks",
                    project_root / "IS-Bench" / "data" / "cameras",
                    project_root / "IS-Bench" / "data" / "bddl",
                ),
            }
        )
    return stable_json_sha256(
        {
            "schema_version": 1,
            "benchmark": normalized,
            "components": components,
        }
    )


def freeze_online_implementation_source_bundle(
    *,
    benchmark: str,
    output_path: Path,
    expected_implementation_sha256: str,
) -> Path:
    """Archive exact formal source bytes after verifying the frozen digest.

    Hash-only provenance cannot recover an implementation in a workspace with
    no durable VCS snapshot. The deterministic ZIP is therefore written before
    model calls and later included in the matrix provenance artifact set.
    """

    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARKS:
        raise ValueError(f"Unsupported online benchmark: {benchmark!r}")
    expected = str(expected_implementation_sha256 or "").strip().lower()
    if not _is_sha256(expected):
        raise ValueError(f"Formal {normalized} source bundle digest is invalid")

    multi_agent_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[5]
    source_entries: dict[str, bytes] = {}

    def add_manifest(prefix: str, root: Path) -> None:
        resolved_root = root.resolve()
        for row in implementation_manifest(resolved_root)["files"]:
            relative = str(row["path"])
            path = (resolved_root / relative).resolve()
            if not path.is_file() or resolved_root not in path.parents:
                raise ValueError(f"implementation source escapes root: {path}")
            source_entries[f"{prefix}/{relative}"] = path.read_bytes()

    add_manifest("harnessaudit/multi_agent", multi_agent_root)
    add_manifest("SafePred", project_root / "SafePred")
    add_manifest("experiments", project_root / "experiments")
    checkpoint_verifier = project_root / "scripts" / "verify_model_checkpoint.py"
    source_entries["scripts/verify_model_checkpoint.py"] = (
        checkpoint_verifier.read_bytes()
    )
    if normalized == "isbench":
        isbench_root = (project_root / "IS-Bench" / "og_ego_prim").resolve()
        for path in sorted(isbench_root.rglob("*.py")):
            resolved = path.resolve()
            if not resolved.is_file() or isbench_root not in resolved.parents:
                raise ValueError(f"ISBench implementation source escapes root: {path}")
            relative = resolved.relative_to(isbench_root).as_posix()
            source_entries[f"IS-Bench/og_ego_prim/{relative}"] = resolved.read_bytes()
        registered_roots = (
            project_root / "IS-Bench" / "data" / "tasks",
            project_root / "IS-Bench" / "data" / "cameras",
            project_root / "IS-Bench" / "data" / "bddl",
        )
        registered_prefix = "IS-Bench/registered_inputs"
    else:
        registered_roots = (
            multi_agent_root / "tasks",
            multi_agent_root / "tools",
            multi_agent_root / "fixtures",
            multi_agent_root / "perturbation_specs",
            multi_agent_root / "train_policies",
            project_root / "SafePred" / "config",
        )
        registered_prefix = "harnessaudit/registered_inputs"
    for root, relative, path in _registered_input_files(*registered_roots):
        prefix = (
            "SafePred/registered_inputs"
            if root == project_root / "SafePred" / "config"
            else registered_prefix
        )
        source_entries[
            f"{prefix}/{root.name}/{relative.as_posix()}"
        ] = path.read_bytes()

    after = current_online_implementation_manifest_sha256(normalized)
    if after != expected:
        raise ValueError(
            "implementation changed after preregistration validation and while "
            "freezing the source bundle"
        )
    manifest_rows = [
        {
            "path": logical,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for logical, content in sorted(source_entries.items())
    ]
    manifest_payload = {
        "schema_version": "formal-online-source-bundle-v1",
        "benchmark": normalized,
        "implementation_manifest_sha256": expected,
        "files": manifest_rows,
    }
    source_entries["SOURCE_MANIFEST.json"] = (
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite formal source bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_STORED) as archive:
            for logical, content in sorted(source_entries.items()):
                info = zipfile.ZipInfo(logical, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_online_implementation_manifest(
    preregistration: Mapping[str, Any] | None,
    *,
    benchmark: str,
    observed_sha256: str,
) -> None:
    """Dereference and verify the frozen online implementation digest."""

    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARKS:
        raise ValueError(f"Unsupported online benchmark: {benchmark!r}")
    if not isinstance(preregistration, Mapping):
        raise ValueError(
            f"Formal {normalized} online implementation manifest requires preregistration"
        )
    try:
        payload = preregistration["payload"]
        contract = payload["run_plans"][normalized]["online_advisory_contract"]
        source = contract["provenance"][
            "implementation_manifest_sha256_source"
        ]
    except (KeyError, TypeError):
        raise ValueError(
            f"Formal {normalized} online implementation manifest source is missing"
        ) from None
    cursor: Any = payload
    for component in str(source).split("."):
        if not component or not isinstance(cursor, Mapping) or component not in cursor:
            raise ValueError(
                f"Formal {normalized} online implementation manifest source is unresolved"
            )
        cursor = cursor[component]
    expected_sha256 = str(cursor or "").strip().lower()
    current_sha256 = str(observed_sha256 or "").strip().lower()
    if not _is_sha256(expected_sha256) or not _is_sha256(current_sha256):
        raise ValueError(
            f"Formal {normalized} online implementation manifest must be SHA256"
        )
    if current_sha256 != expected_sha256:
        raise ValueError(
            f"Formal {normalized} online implementation manifest differs from preregistration"
        )


def canonical_source_input_manifest_sha256(
    file_hashes: Mapping[str, str],
) -> str:
    """Hash a stable logical-name -> file-SHA256 source-input manifest.

    Logical names, rather than absolute paths, keep the same frozen input set
    portable across machines.  The digest binds both membership and content;
    adding, removing, renaming, or changing any source report/trace changes the
    result.
    """
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("Source input manifest must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in file_hashes.items():
        name = str(raw_name).strip()
        digest = str(raw_digest).strip().lower()
        if not name or name.upper().startswith("REPLACE"):
            raise ValueError("Source input manifest contains an unresolved logical name")
        if name in normalized:
            raise ValueError(f"Source input manifest contains duplicate name {name!r}")
        if not _is_sha256(digest):
            raise ValueError(
                f"Source input manifest entry {name!r} is not a SHA256 digest"
            )
        normalized[name] = digest
    canonical = json.dumps(
        {"schema_version": 1, "files": normalized},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def source_input_manifest_sha256(files: Mapping[str, Path]) -> str:
    """Hash source files into a portable canonical manifest digest."""
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Source input files must be a non-empty mapping")
    hashes: dict[str, str] = {}
    for raw_name, raw_path in files.items():
        name = str(raw_name).strip()
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Source input file does not exist for {name!r}: {path}")
        hashes[name] = _sha256_file(path)
    return canonical_source_input_manifest_sha256(hashes)


def load_preregistration(path: Path, *, benchmark: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Preregistration file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Preregistration is not valid JSON: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Preregistration must be a JSON object")
    validate_preregistration(payload, benchmark=benchmark)
    return {
        "path": str(path),
        "sha256": canonical_preregistration_sha256(payload),
        "payload": payload,
    }


def freeze_preregistration(
    preregistration: dict[str, Any],
    destination: Path,
) -> Path:
    """Persist the validated contract before any model or endpoint call.

    Resuming into a directory that already contains a different contract is a
    hard error.  This prevents a late analysis step from silently replacing the
    decision contract under which observations were collected.
    """
    destination = Path(destination).resolve()
    payload = preregistration["payload"]
    expected_hash = str(preregistration["sha256"])
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        observed_hash = canonical_preregistration_sha256(existing)
        if observed_hash != expected_hash:
            raise ValueError(
                "Output directory already contains a different formal preregistration: "
                f"{destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def freeze_formal_replicate_identity(
    gate: Mapping[str, Any],
    destination: Path,
) -> Path:
    """Bind a formal artifact directory to one normalized actor/WM replicate."""

    if gate.get("status") != "validated_replicate":
        raise ValueError("Cannot freeze identity for an unvalidated formal replicate")
    _require_explicit_actor_identity_fields(
        gate,
        label="Formal replicate gate",
    )
    if type(gate.get("wm_seed")) is not int:
        raise ValueError("Formal replicate gate wm_seed must be an exact integer")
    identity = normalize_actor_replicate_identity(
        benchmark=str(gate.get("benchmark") or ""),
        actor_protocol_profile=gate.get("actor_protocol_profile"),
        actor_replicate_id=gate.get("actor_replicate_id"),
        actor_trace_seed=gate.get("actor_trace_seed"),
    )
    payload = {
        "schema_version": 2,
        "benchmark": str(gate.get("benchmark") or ""),
        "actor_protocol_profile": identity.actor_protocol_profile,
        "actor_replicate_id": identity.actor_replicate_id,
        "actor_trace_seed": identity.actor_trace_seed,
        "wm_seed": gate["wm_seed"],
        "preregistration_sha256": str(gate.get("preregistration_sha256") or ""),
        "protocol_sha256": str(gate.get("protocol_sha256") or ""),
        "source_input_manifest_sha256": str(
            gate.get("source_input_manifest_sha256") or ""
        ),
    }
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read existing formal replicate identity {destination}: {exc}"
            ) from exc
        normalized_existing = _normalize_formal_replicate_marker(existing)
        if _canonical_json(normalized_existing) != _canonical_json(payload):
            raise ValueError(
                "Formal analysis directory is already bound to a different "
                f"replicate identity: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _normalize_formal_replicate_marker(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Formal replicate identity marker must be an object")
    schema_version = _integer(value.get("schema_version"))
    if schema_version == 2:
        _require_explicit_actor_identity_fields(
            value,
            label="Schema-2 formal replicate identity marker",
        )
        identity = normalize_actor_replicate_identity(
            benchmark=value.get("benchmark"),
            actor_protocol_profile=value.get("actor_protocol_profile"),
            actor_replicate_id=value.get("actor_replicate_id"),
            actor_trace_seed=value.get("actor_trace_seed"),
        )
        if type(value.get("wm_seed")) is not int:
            raise ValueError(
                "Schema-2 formal replicate identity wm_seed must be an exact integer"
            )
        return {
            "schema_version": 2,
            "benchmark": value.get("benchmark"),
            "actor_protocol_profile": identity.actor_protocol_profile,
            "actor_replicate_id": identity.actor_replicate_id,
            "actor_trace_seed": identity.actor_trace_seed,
            "wm_seed": value.get("wm_seed"),
            "preregistration_sha256": value.get("preregistration_sha256"),
            "protocol_sha256": value.get("protocol_sha256"),
            "source_input_manifest_sha256": value.get(
                "source_input_manifest_sha256"
            ),
        }
    if schema_version != 1:
        raise ValueError(
            "Formal replicate identity marker schema_version must equal 1 or 2"
        )
    seed = value.get("actor_trace_seed")
    if type(seed) is not int:
        raise ValueError(
            "Schema-1 formal replicate identity requires actor_trace_seed "
            "to be an exact integer"
        )
    if type(value.get("wm_seed")) is not int:
        raise ValueError(
            "Schema-1 formal replicate identity wm_seed must be an exact integer"
        )
    return {
        "schema_version": 2,
        "benchmark": str(value.get("benchmark") or ""),
        "actor_protocol_profile": SEEDED_SAMPLING_PROFILE,
        "actor_replicate_id": f"seed-{seed}",
        "actor_trace_seed": seed,
        "wm_seed": value.get("wm_seed"),
        "preregistration_sha256": value.get("preregistration_sha256"),
        "protocol_sha256": value.get("protocol_sha256"),
        "source_input_manifest_sha256": value.get(
            "source_input_manifest_sha256"
        ),
    }


def validate_isbench_hardware_contract(
    hardware: Any,
    *,
    prefix: str = "run_plans.isbench.online_advisory_contract.runtime.hardware_contract",
) -> list[str]:
    """Validate one exact, registered OmniGibson hardware identity.

    The A100-40 profile remains internal-only.  The RTX A6000 profile binds the
    supported-RTX simulator arm, but neither profile substitutes its prospective
    infrastructure contract for the separately required online qualification.
    """

    errors: list[str] = []
    expected_keys = {
        "source_path",
        "source_sha256",
        *ISBENCH_HARDWARE_PROFILE_SOURCE_KEYS,
    }
    if not isinstance(hardware, dict) or set(hardware) != expected_keys:
        errors.append(f"{prefix} must contain exactly {sorted(expected_keys)}")
        return errors
    source_relative = Path(str(hardware.get("source_path") or ""))
    if (
        not str(source_relative)
        or source_relative.is_absolute()
        or ".." in source_relative.parts
    ):
        errors.append(f"{prefix}.source_path must be a safe project-relative path")
    if not _is_sha256(str(hardware.get("source_sha256") or "")):
        errors.append(f"{prefix}.source_sha256 must be SHA256")
    if hardware.get("schema_version") != ISBENCH_HARDWARE_PROFILE_SCHEMA_VERSION:
        errors.append(f"{prefix}.schema_version is invalid")
    profile_id = hardware.get("profile_id")
    profile_specs = {
        ISBENCH_A10040_HARDWARE_PROFILE_ID: {
            "artifact_class": ISBENCH_A10040_HARDWARE_PROFILE_ARTIFACT_CLASS,
            "slurm": ISBENCH_A10040_SLURM_IDENTITY,
            "observed_hardware_constraints": (
                ISBENCH_A10040_OBSERVED_HARDWARE_CONSTRAINTS
            ),
            "hardware_claims": ISBENCH_A10040_HARDWARE_CLAIMS,
            "registered_workload": ISBENCH_A10040_REGISTERED_WORKLOAD,
            "qualification": ISBENCH_A10040_QUALIFICATION,
            "listed_gpu_cost_units_per_hour": 10,
            "probe_contract": {
                "launcher_source_path": (
                    "scripts/slurm/omnigibson_a10040_empirical_probe.sbatch"
                ),
                "probe_python_source_path": "scripts/probe_omnigibson_gpu.py",
                "run_schema_version": (
                    "omnigibson-empirical-hardware-probe-run-v1"
                ),
                "summary_schema_version": (
                    "omnigibson-empirical-hardware-qualification-v1"
                ),
                "run_root_pattern": (
                    "runs/omnigibson_gpu_probe/a100_pcie_40gb_empirical_v2/"
                    "job-${SLURM_JOB_ID}"
                ),
                "gpu_label": "a100-pcie-40gb",
                "required_repetitions": 2,
            },
        },
        ISBENCH_RTX_A6000_HARDWARE_PROFILE_ID: {
            "artifact_class": (
                ISBENCH_RTX_A6000_HARDWARE_PROFILE_ARTIFACT_CLASS
            ),
            "slurm": ISBENCH_RTX_A6000_SLURM_IDENTITY,
            "observed_hardware_constraints": (
                ISBENCH_RTX_A6000_OBSERVED_HARDWARE_CONSTRAINTS
            ),
            "hardware_claims": ISBENCH_RTX_A6000_HARDWARE_CLAIMS,
            "registered_workload": ISBENCH_RTX_A6000_REGISTERED_WORKLOAD,
            "qualification": ISBENCH_RTX_A6000_QUALIFICATION,
            "listed_gpu_cost_units_per_hour": 12,
            "probe_contract": ISBENCH_RTX_A6000_PROBE_CONTRACT,
        },
        ISBENCH_RTX_A6000_THREE_GPU_HARDWARE_PROFILE_ID: {
            "artifact_class": (
                ISBENCH_RTX_A6000_HARDWARE_PROFILE_ARTIFACT_CLASS
            ),
            "slurm": ISBENCH_RTX_A6000_THREE_GPU_SLURM_IDENTITY,
            "observed_hardware_constraints": (
                ISBENCH_RTX_A6000_OBSERVED_HARDWARE_CONSTRAINTS
            ),
            "hardware_claims": ISBENCH_RTX_A6000_HARDWARE_CLAIMS,
            "registered_workload": ISBENCH_RTX_A6000_REGISTERED_WORKLOAD,
            "qualification": ISBENCH_RTX_A6000_QUALIFICATION,
            "listed_gpu_cost_units_per_hour": 36,
            "probe_contract": ISBENCH_RTX_A6000_PROBE_CONTRACT,
        },
    }
    profile_spec = profile_specs.get(profile_id)
    if profile_spec is None:
        errors.append(f"{prefix}.profile_id is invalid")
        return errors
    if hardware.get("artifact_class") != profile_spec["artifact_class"]:
        errors.append(f"{prefix}.artifact_class is invalid")

    exact_groups = {
        field: profile_spec[field]
        for field in (
            "slurm",
            "observed_hardware_constraints",
            "hardware_claims",
            "registered_workload",
            "qualification",
        )
    }
    for field, expected in exact_groups.items():
        observed = hardware.get(field)
        if not isinstance(observed, dict):
            errors.append(f"{prefix}.{field} must be an object")
            continue
        if set(observed) != set(expected):
            errors.append(
                f"{prefix}.{field} must contain exactly {sorted(expected)}"
            )
        for nested_field, expected_value in expected.items():
            if observed.get(nested_field) != expected_value:
                errors.append(f"{prefix}.{field}.{nested_field} drift")

    cost = hardware.get("cost")
    cost_keys = {
        "listed_gpu_cost_units_per_hour",
        "inventory_source",
        "inventory_observed_at_utc",
    }
    if not isinstance(cost, dict) or set(cost) != cost_keys:
        errors.append(f"{prefix}.cost must contain exactly {sorted(cost_keys)}")
    else:
        if cost.get("listed_gpu_cost_units_per_hour") != profile_spec[
            "listed_gpu_cost_units_per_hour"
        ]:
            errors.append(f"{prefix}.cost.listed_gpu_cost_units_per_hour drift")
        if cost.get("inventory_source") != "scir-watch -s":
            errors.append(f"{prefix}.cost.inventory_source drift")
        try:
            observed_at = datetime.fromisoformat(
                str(cost.get("inventory_observed_at_utc") or "").replace(
                    "Z", "+00:00"
                )
            )
        except ValueError:
            observed_at = None
        if observed_at is None or observed_at.tzinfo is None:
            errors.append(
                f"{prefix}.cost.inventory_observed_at_utc must be timezone-aware"
            )

    probe = hardware.get("probe_contract")
    expected_probe = profile_spec["probe_contract"]
    probe_keys = {
        *expected_probe,
        "launcher_source_sha256",
        "probe_python_source_sha256",
    }
    if not isinstance(probe, dict) or set(probe) != probe_keys:
        errors.append(
            f"{prefix}.probe_contract must contain exactly {sorted(probe_keys)}"
        )
    else:
        for field, expected_value in expected_probe.items():
            if probe.get(field) != expected_value:
                errors.append(f"{prefix}.probe_contract.{field} drift")
        for field in ("launcher_source_sha256", "probe_python_source_sha256"):
            if not _is_sha256(str(probe.get(field) or "")):
                errors.append(f"{prefix}.probe_contract.{field} must be SHA256")
    return errors


def isbench_uses_registered_single_seed_one_pass_contract(plan: Any) -> bool:
    """Return whether ``plan`` is the exact registered 161-task one-pass design.

    This is deliberately a narrow exception to the ordinary five-seed formal
    gate.  A merely well-formed or self-declared hardware object is not enough:
    its profile must be in the explicit one-pass registry and every cohort,
    replicate, policy, and arm identity below must match exactly.
    """

    if not isinstance(plan, dict):
        return False
    runtime = (
        ((plan.get("online_advisory_contract") or {}).get("runtime") or {})
        if isinstance(plan.get("online_advisory_contract"), dict)
        else {}
    )
    hardware = runtime.get("hardware_contract")
    if not isinstance(hardware, dict):
        return False
    if (
        hardware.get("profile_id")
        not in ISBENCH_REGISTERED_SINGLE_SEED_HARDWARE_PROFILE_IDS
        or hardware.get("source_path")
        != ISBENCH_REGISTERED_HARDWARE_PROFILE_SOURCE_PATHS.get(
            hardware.get("profile_id")
        )
        or validate_isbench_hardware_contract(hardware)
    ):
        return False
    task_ids = [str(item) for item in (plan.get("task_ids") or [])]
    policies = list(plan.get("policy_modes") or [])
    modes = list(plan.get("intervention_modes") or [])
    return bool(
        len(task_ids) == 161
        and len(set(task_ids)) == 161
        and stable_json_sha256(task_ids)
        == ISBENCH_FULL161_ORDERED_TASK_IDS_SHA256
        and list(plan.get("experiment_seeds") or []) == [0]
        and plan.get("offline_replicates")
        == [{"actor_trace_seed": 0, "wm_seed": 0}]
        and set(policies) in ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS
        and len(policies) == len(set(policies))
        and set(modes) in ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS
        and len(modes) == len(set(modes))
    )


def isbench_omnigibson_hardware_identity(
    hardware: Any,
) -> dict[str, Any]:
    """Return the exact runtime identity after validating the frozen profile."""

    errors = validate_isbench_hardware_contract(hardware)
    if errors:
        raise ValueError("; ".join(errors))
    slurm = hardware["slurm"]
    observed = hardware["observed_hardware_constraints"]
    return {
        "profile_id": hardware["profile_id"],
        "source_path": hardware["source_path"],
        "source_sha256": hardware["source_sha256"],
        "node": slurm["node"],
        "gres": slurm["gres"],
        "gpu_count": slurm["gpu_count"],
        "exact_nvidia_smi_name": observed["exact_nvidia_smi_name"],
        "minimum_memory_total_mib": observed["minimum_memory_total_mib"],
        "hardware_claims": dict(hardware["hardware_claims"]),
    }


def _validate_isbench_serving_contract(
    serving: Any,
    *,
    actor: Any,
    world_model: Any,
    model_stack: Any = None,
    hardware_contract: Any = None,
    errors: list[str],
    prefix: str,
) -> None:
    source_keys = {
        "schema_version",
        "backend",
        "backend_version",
        "model_path",
        "host",
        "advertised_host",
        "port",
        "base_url",
        "served_model_name",
        "slurm_node",
        "slurm_gpu_type",
        "slurm_gpu_count",
        "checkpoint_revision",
        "checkpoint_commit",
        "lfs_manifest_sha256",
        "runtime_freeze_sha256",
        "max_model_len",
        "dtype",
        "gpu_memory_utilization",
        "launcher_source_path",
        "launcher_source_sha256",
        "environment",
        "launch_arguments",
    }
    schema_version = (
        serving.get("schema_version") if isinstance(serving, dict) else None
    )
    if schema_version in {
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        source_keys |= {
            "hardware_profile_source_path",
            "hardware_profile_source_sha256",
        }
    if schema_version == "isbench-vllm-serving-contract-v3":
        source_keys.add("ingress")
    if schema_version == "isbench-vllm-serving-contract-v4":
        source_keys |= {
            "submitter_source_path",
            "submitter_source_sha256",
        }
    expected_keys = {"source_path", "source_sha256", *source_keys}
    if not isinstance(serving, dict) or set(serving) != expected_keys:
        errors.append(f"{prefix} must contain exactly {sorted(expected_keys)}")
        return
    if serving.get("schema_version") not in {
        "isbench-vllm-serving-contract-v1",
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        errors.append(f"{prefix}.schema_version is invalid")
    if schema_version in {
        "isbench-vllm-serving-contract-v2",
        "isbench-vllm-serving-contract-v3",
        "isbench-vllm-serving-contract-v4",
    }:
        hardware_errors = validate_isbench_hardware_contract(
            hardware_contract,
            prefix=(
                "run_plans.isbench.online_advisory_contract.runtime."
                "hardware_contract"
            ),
        )
        errors.extend(hardware_errors)
        if not hardware_errors:
            if serving.get("hardware_profile_source_path") != hardware_contract.get(
                "source_path"
            ):
                errors.append(f"{prefix} hardware profile source path differs")
            if serving.get("hardware_profile_source_sha256") != hardware_contract.get(
                "source_sha256"
            ):
                errors.append(f"{prefix} hardware profile source SHA256 differs")
    elif hardware_contract is not None:
        errors.append(
            f"{prefix} v1 cannot bind an empirical OmniGibson hardware profile"
        )
    if serving.get("backend") != "vllm" or serving.get("backend_version") != "0.23.0":
        errors.append(f"{prefix} must bind vllm 0.23.0")
    required_path_fields = (
        "source_path",
        "model_path",
        "advertised_host",
        "base_url",
        "served_model_name",
        "slurm_node",
        "slurm_gpu_type",
        "checkpoint_revision",
        "launcher_source_path",
    )
    required_sha_fields = (
        "source_sha256",
        "launcher_source_sha256",
        "lfs_manifest_sha256",
        "runtime_freeze_sha256",
    )
    if schema_version == "isbench-vllm-serving-contract-v4":
        required_path_fields += ("submitter_source_path",)
        required_sha_fields += ("submitter_source_sha256",)
    for field in required_path_fields:
        if not str(serving.get(field) or "").strip():
            errors.append(f"{prefix}.{field} is required")
    for field in required_sha_fields:
        if not _is_sha256(str(serving.get(field) or "")):
            errors.append(f"{prefix}.{field} must be SHA256")
    if not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
        str(serving.get("checkpoint_commit") or ""),
    ):
        errors.append(f"{prefix}.checkpoint_commit must be an immutable git digest")
    port = _integer(serving.get("port"))
    if port is None or not 1 <= port <= 65535:
        errors.append(f"{prefix}.port is invalid")
    advertised_host = str(serving.get("advertised_host") or "")
    if serving.get("host") != "0.0.0.0":
        errors.append(f"{prefix}.host must equal 0.0.0.0")
    if serving.get("slurm_node") != advertised_host:
        errors.append(f"{prefix}.slurm_node must equal advertised_host")
    expected_base_url = (
        f"http://{advertised_host}:{port}/v1" if port is not None else ""
    )
    if schema_version == "isbench-vllm-serving-contract-v3":
        expected_base_url = _validate_isbench_https_ingress(
            serving.get("ingress"),
            serving=serving,
            advertised_host=advertised_host,
            upstream_port=port,
            errors=errors,
            prefix=f"{prefix}.ingress",
        )
    elif str(serving.get("base_url") or "").rstrip("/") != expected_base_url:
        errors.append(f"{prefix}.base_url differs from advertised host/port")
    roles = model_stack.get("roles") if isinstance(model_stack, dict) else None
    if isinstance(roles, dict):
        identities = [
            item
            for item in roles.values()
            if isinstance(item, dict)
            and str(item.get("base_url") or "").rstrip("/") == expected_base_url
        ]
        if not identities:
            errors.append(
                f"{prefix} is not used by any frozen model-stack role"
            )
    else:
        # Legacy registrations predate per-role model-stack identities and bind
        # both top-level identities to the single local serving contract.
        identities = [item for item in (actor, world_model) if isinstance(item, dict)]
        if len(identities) != 2:
            errors.append(
                f"{prefix} requires frozen actor and world-model identities"
            )
            identities = []
    if identities:
        if any(
            str(item.get("base_url", item.get("endpoint")) or "").rstrip("/")
            != expected_base_url
            for item in identities
        ):
            errors.append(f"{prefix}.base_url differs from model endpoint identity")
        if any(
            str(item.get("model") or "") != serving.get("served_model_name")
            for item in identities
        ):
            errors.append(f"{prefix}.served_model_name differs from model identity")
        if any(
            str(item.get("revision") or "") != serving.get("checkpoint_revision")
            for item in identities
        ):
            errors.append(f"{prefix}.checkpoint_revision differs from model identity")
    revision = str(serving.get("checkpoint_revision") or "")
    if f"@{serving.get('checkpoint_commit')}" not in revision or not revision.endswith(
        f"lfs-manifest:{serving.get('lfs_manifest_sha256')}"
    ):
        errors.append(f"{prefix}.checkpoint identity is internally inconsistent")
    if serving.get("slurm_gpu_count") != 1:
        errors.append(f"{prefix}.slurm_gpu_count must equal 1")
    if serving.get("max_model_len") != 65536:
        errors.append(f"{prefix}.max_model_len must equal 65536")
    if serving.get("dtype") != "bfloat16":
        errors.append(f"{prefix}.dtype must equal bfloat16")
    utilization = _number(serving.get("gpu_memory_utilization"))
    if utilization is None or not 0.0 < utilization <= 1.0:
        errors.append(f"{prefix}.gpu_memory_utilization is invalid")
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    if serving.get("environment") != expected_environment:
        errors.append(f"{prefix}.environment differs from the exact whitelist")
    arguments = serving.get("launch_arguments")
    if not isinstance(arguments, list) or any(
        not isinstance(item, str) or not item for item in arguments
    ):
        errors.append(f"{prefix}.launch_arguments is invalid")
        return
    if any(
        fragment in item.lower()
        for item in arguments
        for fragment in (
            "authorization",
            "api-key",
            "api_key",
            "access-token",
            "access_token",
            "password",
            "secret",
        )
    ):
        errors.append(f"{prefix}.launch_arguments contains a credential field")
    if any(
        item.split("=", 1)[0]
        in {"--host", "--port", "--served-model-name", "--model"}
        for item in arguments
    ):
        errors.append(f"{prefix}.launch_arguments overrides an authoritative field")

    def argument_value(name: str) -> str | None:
        for index, item in enumerate(arguments):
            if item == name and index + 1 < len(arguments):
                return arguments[index + 1]
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
        return None

    expected_arguments = {
        "--tensor-parallel-size": "1",
        "--max-model-len": str(serving.get("max_model_len")),
        "--gpu-memory-utilization": str(serving.get("gpu_memory_utilization")),
        "--max-num-seqs": "4",
        "--dtype": str(serving.get("dtype")),
        "--generation-config": "vllm",
    }
    for option, expected in expected_arguments.items():
        if argument_value(option) != expected:
            errors.append(f"{prefix}.launch_arguments {option} drift")


def _validate_isbench_https_ingress(
    ingress: Any,
    *,
    serving: dict[str, Any],
    advertised_host: str,
    upstream_port: int | None,
    errors: list[str],
    prefix: str,
) -> str:
    """Validate a public HTTPS gateway separately from the private vLLM node.

    The v3 serving identity is deliberately split: actor and world-model calls
    use a gateway HTTPS endpoint, while vLLM remains bound to the Slurm node.
    Gateway configuration and launcher identities are later rehashed at freeze
    and runtime; only non-secret identity fields are accepted here.
    """

    expected_keys = {
        "schema_version",
        "ingress_id",
        "public_base_url",
        "gateway_hostname",
        "gateway_port",
        "upstream_scheme",
        "upstream_host",
        "upstream_port",
        "gateway_source_path",
        "gateway_source_sha256",
        "gateway_launcher_source_path",
        "gateway_launcher_source_sha256",
        "gateway_runtime_freeze_sha256",
    }
    if not isinstance(ingress, dict) or set(ingress) != expected_keys:
        errors.append(f"{prefix} must contain exactly {sorted(expected_keys)}")
        return ""
    if ingress.get("schema_version") != "hpc-https-ingress-v1":
        errors.append(f"{prefix}.schema_version is invalid")
    for field in (
        "ingress_id",
        "public_base_url",
        "gateway_hostname",
        "upstream_host",
        "gateway_source_path",
        "gateway_launcher_source_path",
    ):
        if not str(ingress.get(field) or "").strip():
            errors.append(f"{prefix}.{field} is required")
    for field in (
        "gateway_source_sha256",
        "gateway_launcher_source_sha256",
        "gateway_runtime_freeze_sha256",
    ):
        if not _is_sha256(str(ingress.get(field) or "")):
            errors.append(f"{prefix}.{field} must be SHA256")
    gateway_port = _integer(ingress.get("gateway_port"))
    if gateway_port is None or not 1 <= gateway_port <= 65535:
        errors.append(f"{prefix}.gateway_port is invalid")
    if ingress.get("upstream_scheme") != "http":
        errors.append(f"{prefix}.upstream_scheme must equal http")
    if ingress.get("upstream_host") != advertised_host:
        errors.append(f"{prefix}.upstream_host must equal advertised_host")
    if ingress.get("upstream_port") != upstream_port:
        errors.append(f"{prefix}.upstream_port must equal serving port")

    public_base_url = str(ingress.get("public_base_url") or "").rstrip("/")
    try:
        parsed = urlsplit(public_base_url)
        parsed_port = parsed.port
    except ValueError:
        parsed = None
        parsed_port = None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        errors.append(f"{prefix}.public_base_url must be an HTTPS /v1 endpoint")
    else:
        effective_port = 443 if parsed_port is None else parsed_port
        if parsed.hostname.lower() != str(ingress.get("gateway_hostname") or "").lower():
            errors.append(f"{prefix}.gateway_hostname differs from public_base_url")
        if gateway_port is not None and effective_port != gateway_port:
            errors.append(f"{prefix}.gateway_port differs from public_base_url")
    if str(serving.get("base_url") or "").rstrip("/") != public_base_url:
        errors.append(f"{prefix}.public_base_url differs from serving base_url")
    try:
        isbench_https_ingress_runtime_identity(ingress)
    except ValueError as exc:
        errors.append(f"{prefix} runtime identity is invalid: {exc}")
    return public_base_url


def isbench_https_ingress_runtime_identity(ingress: Any) -> dict[str, str]:
    """Return the two non-secret values a v3 gateway must attest on ``/models``.

    The client deliberately reads only the fixed allowlisted identity headers;
    gateway authentication headers/cookies are neither inspected nor persisted.
    """

    if not isinstance(ingress, Mapping):
        raise ValueError("ingress is missing")
    ingress_id = ingress.get("ingress_id")
    runtime_digest = ingress.get("gateway_runtime_freeze_sha256")
    if (
        not isinstance(ingress_id, str)
        or not ingress_id.strip()
        or "\n" in ingress_id
        or "\r" in ingress_id
    ):
        raise ValueError("ingress_id is invalid")
    if not isinstance(runtime_digest, str) or not _is_sha256(runtime_digest):
        raise ValueError("gateway_runtime_freeze_sha256 must be SHA256")
    return {
        "ingress_id": ingress_id,
        "gateway_runtime_freeze_sha256": runtime_digest,
    }


def _validate_online_advisory_contract(
    contract: Any,
    *,
    benchmark: str,
    policy_modes: set[str],
    intervention_modes: set[str],
    prediction_horizon: int | None,
    errors: list[str],
    prefix: str,
    experiment_seeds: Iterable[int] = (),
) -> None:
    if not isinstance(contract, dict):
        errors.append(f"{prefix} must be an object")
        return

    expected_top = {
        "semantics",
        "interface",
        "budgets",
        "prediction",
        "runtime",
        "actor",
        "world_model",
        "provenance",
    }
    # New HA contracts bind the exact SafePred algorithm config. Keep the
    # fields optional only so pre-contract archives remain readable. A present
    # legacy policy-extraction contract fails closed below, and exact observed-
    # vs-frozen comparison prevents every archive from being reused as a new
    # formal run.
    corrected_ha_boundary = bool(
        benchmark == "harnessaudit"
        and isinstance(contract.get("policy_extraction_contract"), dict)
        and contract["policy_extraction_contract"].get("schema_version")
        == HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION
    )
    legacy_ha_boundary = bool(
        benchmark == "harnessaudit"
        and isinstance(contract.get("policy_extraction_contract"), dict)
        and contract["policy_extraction_contract"].get("schema_version")
        == HA_LEGACY_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION
    )
    if benchmark == "harnessaudit" and (
        "safepred_config" in contract or corrected_ha_boundary
    ):
        expected_top.add("safepred_config")
    if benchmark == "harnessaudit" and "policy_extraction_contract" in contract:
        expected_top.add("policy_extraction_contract")
    if corrected_ha_boundary or "model_stack" in contract:
        expected_top.add("model_stack")
    if set(contract) != expected_top:
        errors.append(f"{prefix} must contain exactly {sorted(expected_top)}")

    if benchmark == "harnessaudit" and (
        "safepred_config" in contract or corrected_ha_boundary
    ):
        safepred_config = contract.get("safepred_config")
        expected_safepred_config_keys = {
            "source_path",
            "source_sha256",
            "world_model_max_semantic_attempts",
            "world_model_transport_max_attempts",
        }
        if not legacy_ha_boundary:
            expected_safepred_config_keys.add("experience_replay")
        if (
            not isinstance(safepred_config, dict)
            or set(safepred_config) != expected_safepred_config_keys
        ):
            errors.append(
                f"{prefix}.safepred_config must contain exactly "
                f"{sorted(expected_safepred_config_keys)}"
            )
        else:
            if not str(safepred_config.get("source_path") or "").strip():
                errors.append(f"{prefix}.safepred_config.source_path is required")
            if not _is_sha256(
                str(safepred_config.get("source_sha256") or "").strip().lower()
            ):
                errors.append(f"{prefix}.safepred_config.source_sha256 is invalid")
            for field in (
                "world_model_max_semantic_attempts",
                "world_model_transport_max_attempts",
            ):
                value = _integer(safepred_config.get(field))
                if value is None or value < 1:
                    errors.append(
                        f"{prefix}.safepred_config.{field} must be a positive integer"
                    )
            if (
                not legacy_ha_boundary
                and safepred_config.get("experience_replay") != {"enable": False}
            ):
                errors.append(
                    f"{prefix}.safepred_config.experience_replay must equal "
                    "{'enable': False}"
                )

    if benchmark == "harnessaudit" and "policy_extraction_contract" in contract:
        policy_contract = contract.get("policy_extraction_contract")
        policy_keys = {
            "schema_version",
            "model_input_boundary_version",
            "policy_extraction_prompt_version",
            "policy_cache_schema_version",
            "test_visible_fields",
            "input_includes_test_access_rules",
            "input_includes_test_tool_necessity",
            "input_includes_tool_necessity_derivatives",
            "include_test_tool_necessity_policies",
            "split_source_path",
            "split_source_sha256",
            "split_id",
            "test_task_ids",
            "train_selection",
            "strict_split",
            "policy_extraction_seed",
            "semantic_max_attempts",
            "transport_max_attempts",
            "retry_base_delay_seconds",
            "retry_backoff",
            "max_tokens",
            "timeout_seconds",
            "thinking_max_tokens",
            "bounded_thinking_enabled",
            "generic_fallback_allowed",
            "require_precomputed_cache",
            "write_cache",
            "request_extra_body_json",
        }
        if legacy_ha_boundary:
            policy_keys = {
                "schema_version",
                "split_source_path",
                "split_source_sha256",
                "split_id",
                "test_task_ids",
                "train_selection",
                "strict_split",
                "policy_extraction_seed",
                "semantic_max_attempts",
                "transport_max_attempts",
                "retry_base_delay_seconds",
                "retry_backoff",
                "max_tokens",
                "timeout_seconds",
                "thinking_max_tokens",
                "bounded_thinking_enabled",
                "generic_fallback_allowed",
                "require_precomputed_cache",
                "write_cache",
                "request_extra_body_json",
            }
        crossfit_keys = {
            "evaluation_task_ids",
            "evaluation_task_ids_sha256",
            "holdout_unit",
            "estimand",
        }
        if isinstance(policy_contract, dict) and any(
            field in policy_contract for field in crossfit_keys
        ):
            policy_keys |= crossfit_keys
        if not isinstance(policy_contract, dict) or set(policy_contract) != policy_keys:
            errors.append(
                f"{prefix}.policy_extraction_contract must contain exactly "
                f"{sorted(policy_keys)}"
            )
        else:
            if (
                policy_contract.get("schema_version")
                != HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION
            ):
                errors.append(
                    f"{prefix}.policy_extraction_contract.schema_version is invalid"
                )
            expected_policy_boundary = (
                {
                    "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
                    "policy_extraction_prompt_version": (
                        HA_POLICY_EXTRACTION_PROMPT_VERSION
                    ),
                    "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
                    "test_visible_fields": list(
                        HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS
                    ),
                    "input_includes_test_access_rules": False,
                    "input_includes_test_tool_necessity": False,
                    "input_includes_tool_necessity_derivatives": False,
                    "include_test_tool_necessity_policies": False,
                }
                if not legacy_ha_boundary
                else {}
            )
            for field, expected_value in expected_policy_boundary.items():
                if policy_contract.get(field) != expected_value:
                    errors.append(
                        f"{prefix}.policy_extraction_contract.{field} is invalid"
                    )
            if not _is_sha256(
                str(policy_contract.get("split_source_sha256") or "")
            ):
                errors.append(
                    f"{prefix}.policy_extraction_contract.split_source_sha256 is invalid"
                )
            for field in ("split_source_path", "split_id", "train_selection"):
                if not str(policy_contract.get(field) or "").strip():
                    errors.append(
                        f"{prefix}.policy_extraction_contract.{field} is required"
                    )
            test_ids = policy_contract.get("test_task_ids")
            evaluation_ids = policy_contract.get("evaluation_task_ids")
            if (
                not isinstance(test_ids, list)
                or (not test_ids and not evaluation_ids)
                or any(
                    not isinstance(item, str) or not item
                    for item in (test_ids if isinstance(test_ids, list) else [])
                )
            ):
                errors.append(
                    f"{prefix}.policy_extraction_contract.test_task_ids is invalid"
                )
            if crossfit_keys <= set(policy_contract):
                if (
                    not isinstance(evaluation_ids, list)
                    or len(evaluation_ids) != 192
                    or len(set(evaluation_ids)) != 192
                ):
                    errors.append(
                        f"{prefix}.policy_extraction_contract.evaluation_task_ids "
                        "must contain exactly 192 unique tasks"
                    )
                elif policy_contract.get("evaluation_task_ids_sha256") != stable_json_sha256(
                    evaluation_ids
                ):
                    errors.append(
                        f"{prefix}.policy_extraction_contract.evaluation_task_ids_sha256 "
                        "must bind evaluation_task_ids"
                    )
                if policy_contract.get("holdout_unit") != (
                    "evaluated_task_and_entire_category"
                ):
                    errors.append(
                        f"{prefix}.policy_extraction_contract.holdout_unit is invalid"
                    )
                if policy_contract.get("estimand") != (
                    "per_task_category_held_out_cross_fit"
                ):
                    errors.append(
                        f"{prefix}.policy_extraction_contract.estimand is invalid"
                    )
            boolean_fields = [
                "strict_split",
                "bounded_thinking_enabled",
                "generic_fallback_allowed",
                "require_precomputed_cache",
                "write_cache",
            ]
            if not legacy_ha_boundary:
                boolean_fields.extend(
                    [
                        "input_includes_test_access_rules",
                        "input_includes_test_tool_necessity",
                        "input_includes_tool_necessity_derivatives",
                        "include_test_tool_necessity_policies",
                    ]
                )
            for field in boolean_fields:
                if type(policy_contract.get(field)) is not bool:
                    errors.append(
                        f"{prefix}.policy_extraction_contract.{field} must be boolean"
                    )
            if policy_contract.get("generic_fallback_allowed") is not False:
                errors.append(
                    f"{prefix}.policy_extraction_contract.generic_fallback_allowed "
                    "must be false for formal runs"
                )
            request_extra_body_raw = policy_contract.get("request_extra_body_json")
            if not isinstance(request_extra_body_raw, str):
                errors.append(
                    f"{prefix}.policy_extraction_contract.request_extra_body_json "
                    "must be a string"
                )
            elif request_extra_body_raw:
                try:
                    request_extra_body = json.loads(request_extra_body_raw)
                except json.JSONDecodeError:
                    errors.append(
                        f"{prefix}.policy_extraction_contract.request_extra_body_json "
                        "must be valid JSON"
                    )
                else:
                    if not isinstance(request_extra_body, dict):
                        errors.append(
                            f"{prefix}.policy_extraction_contract."
                            "request_extra_body_json must encode an object"
                        )
                    elif _contains_credential_material(request_extra_body):
                        errors.append(
                            f"{prefix}.policy_extraction_contract."
                            "request_extra_body_json contains a credential field"
                        )
            for field, minimum in (
                ("semantic_max_attempts", 1),
                ("transport_max_attempts", 1),
                ("max_tokens", 1),
                ("thinking_max_tokens", 1),
            ):
                value = _integer(policy_contract.get(field))
                if value is None or value < minimum:
                    errors.append(
                        f"{prefix}.policy_extraction_contract.{field} is invalid"
                    )
            if _integer(policy_contract.get("policy_extraction_seed")) is None:
                errors.append(
                    f"{prefix}.policy_extraction_contract.policy_extraction_seed "
                    "must be an integer"
                )
            for field, strictly_positive in (
                ("retry_base_delay_seconds", False),
                ("retry_backoff", True),
                ("timeout_seconds", True),
            ):
                value = _number(policy_contract.get(field))
                if value is None or (value <= 0 if strictly_positive else value < 0):
                    errors.append(
                        f"{prefix}.policy_extraction_contract.{field} is invalid"
                    )
            if (
                policy_contract.get("bounded_thinking_enabled") is True
                and _integer(policy_contract.get("thinking_max_tokens")) is not None
                and _integer(policy_contract.get("max_tokens")) is not None
                and int(policy_contract["thinking_max_tokens"])
                >= int(policy_contract["max_tokens"])
            ):
                errors.append(
                    f"{prefix}.policy_extraction_contract.thinking_max_tokens "
                    "must be smaller than max_tokens"
                )

    semantics = contract.get("semantics")
    expected_semantics = {
        "version": "independent_actor_initiated_pull_consultation_v1",
        "zero_uptake_valid": True,
        "primary_itt_comparator": "no_guard",
        "forecast_only_arm": (
            "prediction_only" if "prediction_only" in intervention_modes else None
        ),
    }
    if semantics != expected_semantics:
        errors.append(
            f"{prefix}.semantics must bind independent pull consultation, "
            "zero uptake, no_guard ITT, and prediction_only forecast semantics"
        )

    interface = contract.get("interface")
    expected_interface_keys = {
        "tool_or_command_name",
        "candidate_schema_version",
        "prompt_version",
        "parser_version",
        "decision_ledger_schema_version",
    }
    if not isinstance(interface, dict) or set(interface) != expected_interface_keys:
        errors.append(
            f"{prefix}.interface must contain exactly {sorted(expected_interface_keys)}"
        )
    else:
        if interface.get("tool_or_command_name") != "consult_safepred":
            errors.append(f"{prefix}.interface.tool_or_command_name must be consult_safepred")
        for field in expected_interface_keys - {"tool_or_command_name"}:
            value = str(interface.get(field) or "").strip()
            if not value or value.upper().startswith("REPLACE"):
                errors.append(f"{prefix}.interface.{field} is required")

    budgets = contract.get("budgets")
    expected_budget_keys = {"per_decision", "per_caller", "per_episode"}
    if not isinstance(budgets, dict) or set(budgets) != expected_budget_keys:
        errors.append(
            f"{prefix}.budgets must contain exactly {sorted(expected_budget_keys)}"
        )
    else:
        for field in ("per_decision", "per_episode"):
            value = _integer(budgets.get(field))
            if value is None or value < 1:
                errors.append(f"{prefix}.budgets.{field} must be an integer >= 1")
        per_caller = budgets.get("per_caller")
        if benchmark == "harnessaudit":
            if _integer(per_caller) is None or int(per_caller) < 1:
                errors.append(f"{prefix}.budgets.per_caller must be an integer >= 1")
        elif per_caller is not None:
            errors.append(f"{prefix}.budgets.per_caller must be null for isbench")

    prediction = contract.get("prediction")
    if not isinstance(prediction, dict) or set(prediction) != {
        "risk_threshold",
        "horizon",
    }:
        errors.append(f"{prefix}.prediction must contain exactly risk_threshold and horizon")
    else:
        threshold = _number(prediction.get("risk_threshold"))
        if threshold is None or not 0.0 <= threshold <= 1.0:
            errors.append(f"{prefix}.prediction.risk_threshold must be in [0, 1]")
        horizon = _integer(prediction.get("horizon"))
        if horizon is None or horizon < 1:
            errors.append(f"{prefix}.prediction.horizon must be an integer >= 1")
        elif prediction_horizon is not None and horizon != prediction_horizon:
            errors.append(f"{prefix}.prediction.horizon must equal prediction_horizon")

    runtime = contract.get("runtime")
    if benchmark == "harnessaudit":
        runtime_keys = {
            "candidate_count",
            "prediction_interval",
            "regeneration_budget",
            "execution_schedule_version",
            "execution_schedule_seed",
            "max_turns",
            "agent_timeout_seconds",
            "execution_workers",
            "failure_policy",
            "condition_controls",
        }
        if isinstance(runtime, dict) and "resource_semantics_mode" in runtime:
            runtime_keys.add("resource_semantics_mode")
        if isinstance(runtime, dict) and "model_context_window_tokens" in runtime:
            runtime_keys.add("model_context_window_tokens")
        if isinstance(runtime, dict) and "serving_contract" in runtime:
            runtime_keys.add("serving_contract")
        if isinstance(runtime, dict) and "credential_scan_scope" in runtime:
            runtime_keys.add("credential_scan_scope")
        if isinstance(runtime, dict) and "wm_return_perturbation_version" in runtime:
            runtime_keys.add("wm_return_perturbation_version")
        # Optional only for archived preregistrations. New HA runners freeze
        # this bounded judge semantic-retry budget in their runtime identity.
        if isinstance(runtime, dict) and "judge_semantic_max_attempts" in runtime:
            runtime_keys.add("judge_semantic_max_attempts")
        if isinstance(runtime, dict) and "trace_resume" in runtime:
            runtime_keys.add("trace_resume")
        if isinstance(runtime, dict) and "terminal_scoring_contract" in runtime:
            runtime_keys.add("terminal_scoring_contract")
        lane_runtime_keys = {
            "execution_lane_count",
            "execution_lane_assignment_version",
            "execution_lane_plan_sha256",
        }
        if isinstance(runtime, dict) and lane_runtime_keys & set(runtime):
            runtime_keys.update(lane_runtime_keys)
        # Optional only for archived preregistrations. New formal HA runner
        # identities always bind this parent process-group wall-clock cap.
        if isinstance(runtime, dict) and "condition_hard_timeout_seconds" in runtime:
            runtime_keys.add("condition_hard_timeout_seconds")
        if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
            errors.append(f"{prefix}.runtime must contain exactly {sorted(runtime_keys)}")
        else:
            distributed_registered = "distributed_enforced" in intervention_modes
            if distributed_registered and "wm_return_perturbation_version" not in runtime:
                errors.append(
                    f"{prefix}.runtime.wm_return_perturbation_version is required"
                )
            if not distributed_registered and "wm_return_perturbation_version" in runtime:
                errors.append(
                    f"{prefix}.runtime.wm_return_perturbation_version is only valid "
                    "for distributed_enforced"
                )
            for field, minimum in (
                ("candidate_count", 1),
                ("prediction_interval", 1),
                ("regeneration_budget", 0),
                ("max_turns", 1),
                ("agent_timeout_seconds", 1),
                ("execution_workers", 1),
            ):
                value = _integer(runtime.get(field))
                if value is None or value < minimum:
                    errors.append(f"{prefix}.runtime.{field} must be an integer >= {minimum}")
            if "judge_semantic_max_attempts" in runtime:
                judge_semantic_max_attempts = runtime.get(
                    "judge_semantic_max_attempts"
                )
                if (
                    type(judge_semantic_max_attempts) is not int
                    or not 1 <= judge_semantic_max_attempts <= 8
                ):
                    errors.append(
                        f"{prefix}.runtime.judge_semantic_max_attempts must be an "
                        "integer in [1, 8]"
                    )
            if "wm_return_perturbation_version" in runtime and runtime.get(
                "wm_return_perturbation_version"
            ) != "wm-return-perturbation-v1":
                errors.append(
                    f"{prefix}.runtime.wm_return_perturbation_version is invalid"
                )
            schedule_version = runtime.get("execution_schedule_version")
            if schedule_version not in {
                "task_seed_block_sha256_v1",
                "task_seed_block_lanes_sha256_v2",
            }:
                errors.append(
                    f"{prefix}.runtime.execution_schedule_version must be a "
                    "registered task-block schedule"
                )
            if _integer(runtime.get("execution_schedule_seed")) is None:
                errors.append(
                    f"{prefix}.runtime.execution_schedule_seed must be an integer"
                )
            execution_workers = runtime.get("execution_workers")
            if execution_workers not in {1, 4}:
                errors.append(
                    f"{prefix}.runtime.execution_workers must equal 1 or 4"
                )
            elif execution_workers == 1:
                if schedule_version != "task_seed_block_sha256_v1":
                    errors.append(
                        f"{prefix}.runtime workers=1 must use "
                        "task_seed_block_sha256_v1"
                    )
                if lane_runtime_keys & set(runtime):
                    errors.append(
                        f"{prefix}.runtime workers=1 cannot bind concurrent lanes"
                    )
            else:
                if schedule_version != "task_seed_block_lanes_sha256_v2":
                    errors.append(
                        f"{prefix}.runtime workers=4 must use "
                        "task_seed_block_lanes_sha256_v2"
                    )
                if runtime.get("execution_lane_count") != 4:
                    errors.append(
                        f"{prefix}.runtime.execution_lane_count must equal 4"
                    )
                if (
                    runtime.get("execution_lane_assignment_version")
                    != "round_robin_frozen_block_order_v1"
                ):
                    errors.append(
                        f"{prefix}.runtime.execution_lane_assignment_version drift"
                    )
                lane_digest = runtime.get("execution_lane_plan_sha256")
                if not isinstance(lane_digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", lane_digest
                ):
                    errors.append(
                        f"{prefix}.runtime.execution_lane_plan_sha256 must be SHA256"
                    )
            if runtime.get("failure_policy") not in HA_FAILURE_POLICIES:
                errors.append(
                    f"{prefix}.runtime.failure_policy must be one of "
                    f"{sorted(HA_FAILURE_POLICIES)}"
                )
            if "condition_hard_timeout_seconds" in runtime:
                hard_timeout = _integer(
                    runtime.get("condition_hard_timeout_seconds")
                )
                if hard_timeout is None or hard_timeout < 1:
                    errors.append(
                        f"{prefix}.runtime.condition_hard_timeout_seconds "
                        "must be an integer >= 1"
                    )
            if (
                "trace_resume" in runtime
                and runtime.get("trace_resume") != HA_TRACE_RESUME_CONTRACT
            ):
                errors.append(
                    f"{prefix}.runtime.trace_resume must equal the frozen "
                    "HarnessAudit trace-resume contract"
                )
            if "terminal_scoring_contract" in runtime:
                try:
                    terminal_contract_matches = _canonical_json(
                        runtime.get("terminal_scoring_contract")
                    ) == _canonical_json(HA_SAFETY_ABSTENTION_ITT_CONTRACT_V1)
                except (TypeError, ValueError):
                    terminal_contract_matches = False
                if not terminal_contract_matches:
                    errors.append(
                        f"{prefix}.runtime.terminal_scoring_contract must equal "
                        "the frozen HA safety-abstention ITT contract"
                    )
            if "model_context_window_tokens" in runtime:
                context_window = _integer(runtime.get("model_context_window_tokens"))
                if context_window is None or context_window < 32768:
                    errors.append(
                        f"{prefix}.runtime.model_context_window_tokens must be an "
                        "integer >= 32768"
                    )
            if "serving_contract" in runtime:
                serving = runtime.get("serving_contract")
                serving_keys = {
                    "source_path",
                    "source_sha256",
                    "schema_version",
                    "backend",
                    "backend_version",
                    "model_path",
                    "host",
                    "port",
                    "served_model_name",
                    "launcher_source_path",
                    "launcher_source_sha256",
                    "environment",
                    "launch_arguments",
                }
                if not isinstance(serving, dict) or set(serving) != serving_keys:
                    errors.append(
                        f"{prefix}.runtime.serving_contract must contain exactly "
                        f"{sorted(serving_keys)}"
                    )
                else:
                    if not _is_sha256(str(serving.get("source_sha256") or "")):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.source_sha256 is invalid"
                        )
                    if serving.get("schema_version") != "vllm-serving-contract-v1":
                        errors.append(
                            f"{prefix}.runtime.serving_contract.schema_version is invalid"
                        )
                    if serving.get("backend") != "vllm":
                        errors.append(
                            f"{prefix}.runtime.serving_contract.backend must be vllm"
                        )
                    for field in (
                        "source_path",
                        "backend_version",
                        "model_path",
                        "host",
                        "served_model_name",
                        "launcher_source_path",
                    ):
                        if not str(serving.get(field) or "").strip():
                            errors.append(
                                f"{prefix}.runtime.serving_contract.{field} is required"
                            )
                    if not _is_sha256(
                        str(serving.get("launcher_source_sha256") or "")
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract."
                            "launcher_source_sha256 is invalid"
                        )
                    port = _integer(serving.get("port"))
                    if port is None or not 1 <= port <= 65535:
                        errors.append(
                            f"{prefix}.runtime.serving_contract.port is invalid"
                        )
                    if not isinstance(serving.get("environment"), dict) or not serving.get(
                        "environment"
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.environment is required"
                        )
                    elif any(
                        not isinstance(name, str)
                        or not name
                        or not (name[0].isalpha() or name[0] == "_")
                        or not name.replace("_", "").isalnum()
                        or not isinstance(value, str)
                        or "\n" in name
                        or "\n" in value
                        for name, value in serving["environment"].items()
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.environment entries "
                            "must be single-line string assignments"
                        )
                    elif any(
                        fragment in str(name).lower()
                        for name in serving["environment"]
                        for fragment in (
                            "api_key",
                            "api-key",
                            "token",
                            "password",
                            "secret",
                            "authorization",
                        )
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.environment contains "
                            "a credential field"
                        )
                    arguments = serving.get("launch_arguments")
                    if not isinstance(arguments, list) or not arguments or any(
                        not isinstance(item, str) or not item for item in arguments
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.launch_arguments is invalid"
                        )
                    elif any(
                        fragment in item.lower()
                        for item in arguments
                        for fragment in (
                            "authorization",
                            "api-key",
                            "api_key",
                            "access-token",
                            "access_token",
                            "password",
                            "secret",
                        )
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.launch_arguments "
                            "contains a credential field"
                        )
                    elif any(
                        item.split("=", 1)[0]
                        in {"--host", "--port", "--served-model-name", "--model"}
                        for item in arguments
                    ):
                        errors.append(
                            f"{prefix}.runtime.serving_contract.launch_arguments "
                            "override an authoritative identity field"
                        )
            if "credential_scan_scope" in runtime and runtime.get(
                "credential_scan_scope"
            ) != "managed_roots_plus_declared_paths_v1":
                errors.append(
                    f"{prefix}.runtime.credential_scan_scope is invalid"
                )
            controls = runtime.get("condition_controls")
            control_keys = {
                "policy_mode",
                "intervention_mode",
                "risk_signal_mode",
                "risk_noise_rate",
                "guidance_mode",
                "policy_perturbation",
                "observation_perturbation",
                "wm_return_perturbation",
                "experiment_seed",
            }
            legacy_control_keys = control_keys - {"wm_return_perturbation"}
            if not isinstance(controls, list) or not controls:
                errors.append(f"{prefix}.runtime.condition_controls must be non-empty")
            else:
                for index, control in enumerate(controls):
                    control_prefix = f"{prefix}.runtime.condition_controls[{index}]"
                    observed_control_keys = (
                        set(control) if isinstance(control, dict) else set()
                    )
                    legacy_control = (
                        observed_control_keys == legacy_control_keys
                        and "distributed_enforced" not in intervention_modes
                    )
                    if not isinstance(control, dict) or (
                        observed_control_keys != control_keys and not legacy_control
                    ):
                        errors.append(
                            f"{control_prefix} must contain exactly {sorted(control_keys)}"
                        )
                        continue
                    noise = _number(control.get("risk_noise_rate"))
                    if noise is None or not 0.0 <= noise <= 1.0:
                        errors.append(f"{control_prefix}.risk_noise_rate must be in [0, 1]")
                    if _integer(control.get("experiment_seed")) is None:
                        errors.append(f"{control_prefix}.experiment_seed must be an integer")
                    required_string_fields = observed_control_keys - {
                        "risk_noise_rate",
                        "experiment_seed",
                    }
                    for field in required_string_fields:
                        if not str(control.get(field) or "").strip():
                            errors.append(f"{control_prefix}.{field} is required")
                    arm = str(control.get("intervention_mode") or "")
                    return_mode = str(
                        control.get("wm_return_perturbation") or "none"
                    )
                    if arm == "distributed_enforced":
                        if return_mode not in {
                            "empty",
                            "corrupt",
                            "irrelevant",
                            "filler",
                        }:
                            errors.append(
                                f"{control_prefix}.wm_return_perturbation is invalid "
                                "for distributed_enforced"
                            )
                    elif return_mode != "none":
                        errors.append(
                            f"{control_prefix}.wm_return_perturbation must be none "
                            "outside distributed_enforced"
                        )
            if "resource_semantics_mode" in runtime and runtime.get(
                "resource_semantics_mode"
            ) not in {"legacy_official_v1", "explicit_v2"}:
                errors.append(
                    f"{prefix}.runtime.resource_semantics_mode is invalid"
                )
    else:
        runtime_keys = {
            "regeneration_budget",
            "max_tokens",
            "temperature",
            "response_format_mode",
            "max_images",
            "image_max_side",
            "image_quality",
            "train_policy_max_tasks",
            "train_policy_same_domain_only",
            "inference_seed_policy",
            "inference_seed_bindings",
        }
        for optional_contract in (
            "serving_contract",
            "hardware_contract",
            "execution_contract",
            "execution_schedule",
            "condition_controls",
            "wm_return_perturbation_version",
        ):
            if isinstance(runtime, dict) and optional_contract in runtime:
                runtime_keys.add(optional_contract)
        transport_runtime_keys = {
            "request_timeout_seconds",
            "semantic_max_attempts",
            "transport_max_attempts",
            "transport_retry_base_delay_seconds",
        }
        if isinstance(runtime, dict) and (
            transport_runtime_keys & set(runtime)
            or "model_stack" in contract
        ):
            runtime_keys.update(transport_runtime_keys)
        if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
            errors.append(f"{prefix}.runtime must contain exactly {sorted(runtime_keys)}")
        else:
            distributed_registered = "distributed_enforced" in intervention_modes
            distributed_runtime_fields = {
                "condition_controls",
                "wm_return_perturbation_version",
            }
            observed_distributed_runtime_fields = distributed_runtime_fields & set(
                runtime
            )
            if distributed_registered and observed_distributed_runtime_fields != (
                distributed_runtime_fields
            ):
                errors.append(
                    f"{prefix}.runtime must bind both distributed condition controls "
                    "and the WM-return perturbation version"
                )
            if not distributed_registered and observed_distributed_runtime_fields:
                errors.append(
                    f"{prefix}.runtime distributed perturbation fields are only "
                    "valid for distributed_enforced"
                )
            for field, minimum in (
                ("regeneration_budget", 0),
                ("max_tokens", 1),
                ("max_images", 0),
                ("image_max_side", 64),
                ("image_quality", 1),
                ("train_policy_max_tasks", 1),
            ):
                value = _integer(runtime.get(field))
                if value is None or value < minimum:
                    errors.append(f"{prefix}.runtime.{field} must be an integer >= {minimum}")
            if transport_runtime_keys <= set(runtime):
                for field in (
                    "request_timeout_seconds",
                    "semantic_max_attempts",
                    "transport_max_attempts",
                ):
                    value = _integer(runtime.get(field))
                    if value is None or value < 1:
                        errors.append(
                            f"{prefix}.runtime.{field} must be an integer >= 1"
                        )
                retry_delay = _number(
                    runtime.get("transport_retry_base_delay_seconds")
                )
                if retry_delay is None or retry_delay < 0:
                    errors.append(
                        f"{prefix}.runtime.transport_retry_base_delay_seconds "
                        "must be nonnegative"
                    )
            temperature = _number(runtime.get("temperature"))
            if temperature is None or temperature < 0:
                errors.append(f"{prefix}.runtime.temperature must be nonnegative")
            if runtime.get("response_format_mode") not in {"json_object", "disabled"}:
                errors.append(f"{prefix}.runtime.response_format_mode is invalid")
            if type(runtime.get("train_policy_same_domain_only")) is not bool:
                errors.append(
                    f"{prefix}.runtime.train_policy_same_domain_only must be boolean"
                )
            if runtime.get("inference_seed_policy") != "task_scene_sha256_v1":
                errors.append(
                    f"{prefix}.runtime.inference_seed_policy must bind task-derived seeds"
                )
            distributed_controls = runtime.get("condition_controls")
            if "condition_controls" in runtime:
                control_keys = {
                    "policy_mode",
                    "intervention_mode",
                    "wm_return_perturbation",
                    "experiment_seed",
                }
                if (
                    runtime.get("wm_return_perturbation_version")
                    != "wm-return-perturbation-v1"
                ):
                    errors.append(
                        f"{prefix}.runtime.wm_return_perturbation_version is invalid"
                    )
                if not isinstance(distributed_controls, list) or not distributed_controls:
                    errors.append(
                        f"{prefix}.runtime.condition_controls must be non-empty"
                    )
                else:
                    for index, control in enumerate(distributed_controls):
                        control_prefix = (
                            f"{prefix}.runtime.condition_controls[{index}]"
                        )
                        if not isinstance(control, dict) or set(control) != control_keys:
                            errors.append(
                                f"{control_prefix} must contain exactly "
                                f"{sorted(control_keys)}"
                            )
                            continue
                        if _integer(control.get("experiment_seed")) is None:
                            errors.append(
                                f"{control_prefix}.experiment_seed must be an integer"
                            )
                        arm = str(control.get("intervention_mode") or "")
                        mode = str(control.get("wm_return_perturbation") or "")
                        if arm == "distributed_enforced":
                            if mode not in {
                                "empty",
                                "corrupt",
                                "irrelevant",
                                "filler",
                            }:
                                errors.append(
                                    f"{control_prefix}.wm_return_perturbation is invalid"
                                )
                        elif mode != "none":
                            errors.append(
                                f"{control_prefix}.wm_return_perturbation must be none"
                            )
            bindings = runtime.get("inference_seed_bindings")
            binding_keys = {
                "task_id",
                "scene",
                "experiment_seed",
                "inference_seed",
            }
            if not isinstance(bindings, list) or not bindings:
                errors.append(
                    f"{prefix}.runtime.inference_seed_bindings must be non-empty"
                )
            else:
                identities = set()
                for index, binding in enumerate(bindings):
                    binding_prefix = (
                        f"{prefix}.runtime.inference_seed_bindings[{index}]"
                    )
                    if not isinstance(binding, dict) or set(binding) != binding_keys:
                        errors.append(
                            f"{binding_prefix} must contain exactly {sorted(binding_keys)}"
                        )
                        continue
                    if not str(binding.get("task_id") or "").strip():
                        errors.append(f"{binding_prefix}.task_id is required")
                    if not str(binding.get("scene") or "").strip():
                        errors.append(f"{binding_prefix}.scene is required")
                    if _integer(binding.get("experiment_seed")) is None:
                        errors.append(f"{binding_prefix}.experiment_seed must be an integer")
                    if _integer(binding.get("inference_seed")) is None:
                        errors.append(f"{binding_prefix}.inference_seed must be an integer")
                    identity = (
                        str(binding.get("task_id") or ""),
                        str(binding.get("scene") or ""),
                        binding.get("experiment_seed"),
                    )
                    if identity in identities:
                        errors.append(f"{binding_prefix} duplicates a task/scene/seed identity")
                    identities.add(identity)
            if "execution_contract" in runtime:
                execution_errors = isbench_formal_execution_contract_errors(
                    runtime.get("execution_contract")
                )
                if execution_errors:
                    errors.append(
                        f"{prefix}.runtime.execution_contract must equal the "
                        "registered formal execution contract except for a "
                        "preregistered data_parallel value from 1 through 4 "
                        "and condition timeout of 43200, 72000, or 151200 seconds: "
                        + "; ".join(execution_errors)
                    )
            if "execution_schedule" in runtime:
                schedule = runtime.get("execution_schedule")
                expected_schedule_keys = {
                    "version",
                    "schedule_seed",
                    "condition_order",
                    "condition_order_sha256",
                }
                if not isinstance(schedule, dict) or set(schedule) != expected_schedule_keys:
                    errors.append(
                        f"{prefix}.runtime.execution_schedule must contain exactly "
                        f"{sorted(expected_schedule_keys)}"
                    )
                else:
                    try:
                        expected_order = isbench_formal_condition_order(
                            experiment_seeds=experiment_seeds,
                            policy_modes=policy_modes,
                            intervention_modes=intervention_modes,
                            schedule_seed=int(schedule.get("schedule_seed")),
                            version=str(schedule.get("version") or ""),
                        )
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{prefix}.runtime.execution_schedule is invalid: {exc}"
                        )
                    else:
                        if schedule.get("schedule_seed") != ISBENCH_FORMAL_SCHEDULE_SEED:
                            errors.append(
                                f"{prefix}.runtime.execution_schedule.schedule_seed drift"
                            )
                        if schedule.get("condition_order") != expected_order:
                            errors.append(
                                f"{prefix}.runtime.execution_schedule.condition_order drift"
                            )
                        expected_digest = isbench_formal_condition_order_sha256(
                            expected_order
                        )
                        if schedule.get("condition_order_sha256") != expected_digest:
                            errors.append(
                                f"{prefix}.runtime.execution_schedule digest drift"
                            )
            if "serving_contract" in runtime:
                _validate_isbench_serving_contract(
                    runtime.get("serving_contract"),
                    actor=contract.get("actor"),
                    world_model=contract.get("world_model"),
                    model_stack=contract.get("model_stack"),
                    hardware_contract=runtime.get("hardware_contract"),
                    errors=errors,
                    prefix=f"{prefix}.runtime.serving_contract",
                )
            elif "hardware_contract" in runtime:
                errors.extend(
                    validate_isbench_hardware_contract(
                        runtime.get("hardware_contract"),
                        prefix=f"{prefix}.runtime.hardware_contract",
                    )
                )

    for identity_field in ("actor", "world_model"):
        identity = contract.get(identity_field)
        identity_prefix = f"{prefix}.{identity_field}"
        identity_keys = {"provider", "model", "endpoint", "revision"}
        if not isinstance(identity, dict) or set(identity) != identity_keys:
            errors.append(f"{identity_prefix} must contain exactly {sorted(identity_keys)}")
            continue
        for field in identity_keys:
            value = str(identity.get(field) or "").strip()
            if not value or value.upper().startswith("REPLACE"):
                errors.append(f"{identity_prefix}.{field} is required")
        if str(identity.get("revision") or "").strip() == str(
            identity.get("model") or ""
        ).strip():
            errors.append(
                f"{identity_prefix}.revision must bind an immutable provider/checkpoint "
                "revision, not repeat the mutable model selector"
            )
        endpoint = str(identity.get("endpoint") or "")
        try:
            sanitized = sanitize_endpoint_identity(endpoint)
        except ValueError as exc:
            errors.append(f"{identity_prefix}.endpoint is invalid: {exc}")
        else:
                if endpoint != sanitized:
                    errors.append(
                        f"{identity_prefix}.endpoint must not contain credentials, query, or fragment"
                    )

    model_stack = contract.get("model_stack")
    if corrected_ha_boundary and model_stack is None:
        errors.append(
            f"{prefix}.model_stack is required for corrected HarnessAudit "
            "formal contracts"
        )
    if model_stack is not None:
        stack_prefix = f"{prefix}.model_stack"
        stack_keys = {"schema_version", "source_path", "source_sha256", "roles"}
        if not isinstance(model_stack, dict) or set(model_stack) != stack_keys:
            errors.append(
                f"{stack_prefix} must contain exactly {sorted(stack_keys)}"
            )
        else:
            if model_stack.get("schema_version") != "model-stack-profile-v1":
                errors.append(
                    f"{stack_prefix}.schema_version must be model-stack-profile-v1"
                )
            if not str(model_stack.get("source_path") or "").strip():
                errors.append(f"{stack_prefix}.source_path is required")
            if not _is_sha256(str(model_stack.get("source_sha256") or "").lower()):
                errors.append(f"{stack_prefix}.source_sha256 must be SHA256")
            roles = model_stack.get("roles")
            expected_roles = {
                "actor",
                "judge",
                "world_model",
                "policy_extractor",
                "action_agent",
            }
            if not isinstance(roles, dict) or set(roles) != expected_roles:
                errors.append(
                    f"{stack_prefix}.roles must contain exactly {sorted(expected_roles)}"
                )
            else:
                base_role_keys = {
                    "transport",
                    "endpoint",
                    "model",
                    "revision",
                    "runtime_revision",
                    "base_url",
                    "api_mode",
                    "base_url_env",
                    "auth",
                    "api_key_env",
                    "temperature",
                    "max_tokens",
                    "timeout_seconds",
                    "transport_max_attempts",
                    "request_extra_body",
                }
                for role_name in sorted(expected_roles):
                    role = roles.get(role_name)
                    role_prefix = f"{stack_prefix}.roles.{role_name}"
                    role_keys = set(role) if isinstance(role, dict) else set()
                    sampling_keys = {"top_p", "top_k"}
                    role_shape_valid = bool(
                        isinstance(role, dict)
                        and base_role_keys <= role_keys
                        and role_keys <= (
                            base_role_keys | sampling_keys
                            if role_name == "actor"
                            else base_role_keys
                        )
                    )
                    if not role_shape_valid:
                        errors.append(
                            f"{role_prefix} must contain the base role fields and "
                            "only optional actor top_p/top_k sampling fields"
                        )
                        continue
                    if role.get("transport") != "openai_compatible":
                        errors.append(
                            f"{role_prefix}.transport must be openai_compatible"
                        )
                    model = str(role.get("model") or "").strip()
                    revision = str(role.get("revision") or "").strip()
                    runtime_revision = str(
                        role.get("runtime_revision") or ""
                    ).strip()
                    if not model:
                        errors.append(f"{role_prefix}.model is required")
                    if not revision or revision == model:
                        errors.append(
                            f"{role_prefix}.revision must be an immutable revision"
                        )
                    if (
                        not runtime_revision
                        or runtime_revision == "unversioned-runtime"
                    ):
                        errors.append(
                            f"{role_prefix}.runtime_revision must be immutable and explicit"
                        )
                    endpoint = str(role.get("base_url") or "")
                    try:
                        sanitized = sanitize_endpoint_identity(endpoint)
                    except ValueError as exc:
                        errors.append(f"{role_prefix}.base_url is invalid: {exc}")
                    else:
                        if endpoint != sanitized:
                            errors.append(
                                f"{role_prefix}.base_url must be credential-free"
                            )
                    if role.get("api_mode") not in {
                        "chat_completions",
                        "responses",
                    }:
                        errors.append(f"{role_prefix}.api_mode is invalid")
                    if role.get("auth") not in {"none", "api_key"}:
                        errors.append(f"{role_prefix}.auth is invalid")
                    temperature = _number(role.get("temperature"))
                    if temperature is None or temperature < 0:
                        errors.append(
                            f"{role_prefix}.temperature must be nonnegative"
                        )
                    elif (
                        benchmark == "isbench"
                        and role_name == "actor"
                        and temperature <= 0
                    ):
                        errors.append(
                            f"{role_prefix}.temperature must be > 0 so "
                            "preregistered ISBench seeds are genuine actor replicates"
                        )
                    if role_name == "actor" and "top_p" in role_keys:
                        top_p = _number(role.get("top_p"))
                        if top_p is None or not 0.0 < top_p <= 1.0:
                            errors.append(
                                f"{role_prefix}.top_p must be a finite number in (0, 1]"
                            )
                    if role_name == "actor" and "top_k" in role_keys:
                        top_k = _integer(role.get("top_k"))
                        if top_k is None or top_k < 0:
                            errors.append(
                                f"{role_prefix}.top_k must be a nonnegative integer"
                            )
                    if _integer(role.get("max_tokens")) is None or int(
                        role.get("max_tokens") or 0
                    ) < 1:
                        errors.append(f"{role_prefix}.max_tokens must be positive")
                    if _integer(role.get("timeout_seconds")) is None or int(
                        role.get("timeout_seconds") or 0
                    ) < 1:
                        errors.append(
                            f"{role_prefix}.timeout_seconds must be positive"
                        )
                    if _integer(role.get("transport_max_attempts")) is None or int(
                        role.get("transport_max_attempts") or 0
                    ) < 1:
                        errors.append(
                            f"{role_prefix}.transport_max_attempts must be positive"
                        )
                    if not isinstance(role.get("request_extra_body"), dict):
                        errors.append(
                            f"{role_prefix}.request_extra_body must be an object"
                        )

    provenance = contract.get("provenance")
    provenance_keys = {
        "policy_modes",
        "policy_source_identity",
        "implementation_manifest_sha256_source",
        "implementation_manifest_sha256",
    }
    if isinstance(provenance, dict) and "trace_resume" in provenance:
        provenance_keys.add("trace_resume")
    if not isinstance(provenance, dict) or set(provenance) != provenance_keys:
        errors.append(f"{prefix}.provenance must contain exactly {sorted(provenance_keys)}")
    else:
        runtime_trace_resume = (
            runtime.get("trace_resume")
            if isinstance(runtime, dict) and "trace_resume" in runtime
            else None
        )
        if runtime_trace_resume is not None:
            if provenance.get("trace_resume") != HA_TRACE_RESUME_CONTRACT:
                errors.append(
                    f"{prefix}.provenance.trace_resume must exactly bind the "
                    "runtime trace-resume contract"
                )
        elif "trace_resume" in provenance:
            errors.append(
                f"{prefix}.provenance.trace_resume is invalid without runtime.trace_resume"
            )
        if provenance.get("policy_modes") != sorted(policy_modes):
            errors.append(f"{prefix}.provenance.policy_modes must equal sorted run-plan modes")
        ha_policy_contract = contract.get("policy_extraction_contract")
        expected_policy_source = (
            HA_POLICY_SOURCE_IDENTITY
            if benchmark == "harnessaudit"
            and isinstance(ha_policy_contract, dict)
            and ha_policy_contract.get("schema_version")
            == HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION
            else "harnessaudit_registered_policy_sources_v1"
            if benchmark == "harnessaudit"
            else "isbench_registered_policy_tiers_v1"
        )
        if provenance.get("policy_source_identity") != expected_policy_source:
            errors.append(
                f"{prefix}.provenance.policy_source_identity must equal {expected_policy_source}"
            )
        expected_manifest_source = (
            f"run_plans.{benchmark}.online_advisory_contract.provenance."
            "implementation_manifest_sha256"
        )
        if (
            provenance.get("implementation_manifest_sha256_source")
            != expected_manifest_source
        ):
            errors.append(
                f"{prefix}.provenance.implementation_manifest_sha256_source "
                f"must equal {expected_manifest_source}"
            )
        if not _is_sha256(
            str(provenance.get("implementation_manifest_sha256") or "").lower()
        ):
            errors.append(
                f"{prefix}.provenance.implementation_manifest_sha256 "
                "must be SHA256"
            )


def _validate_ha_online_task_cohort_contract(
    plan: Mapping[str, Any],
    *,
    errors: list[str],
    prefix: str,
    require_contract: bool,
) -> None:
    contract = plan.get("task_cohort_contract")
    online = plan.get("online_advisory_contract")
    runtime = online.get("runtime") if isinstance(online, Mapping) else None
    runtime_semantics = (
        str(runtime.get("resource_semantics_mode") or "")
        if isinstance(runtime, Mapping)
        else ""
    )
    if contract is None:
        if require_contract:
            errors.append(
                f"{prefix}.task_cohort_contract is required for HA formal "
                "preregistration version 5 or later"
            )
        return
    cohort_prefix = f"{prefix}.task_cohort_contract"
    if not isinstance(contract, dict) or set(contract) != (
        HA_ONLINE_TASK_COHORT_CONTRACT_KEYS
    ):
        errors.append(
            f"{cohort_prefix} must contain exactly "
            f"{sorted(HA_ONLINE_TASK_COHORT_CONTRACT_KEYS)}"
        )
        return
    expected_values = {
        "schema_version": HA_ONLINE_TASK_COHORT_SCHEMA_VERSION,
        "cohort_id": "ha-canonical-nonsde200-minus-v4-pilot8-v1",
        "selection_rule": HA_ONLINE_TASK_COHORT_SELECTION_RULE,
        "resource_semantics_mode": "legacy_official_v1",
        "canonical_non_sde_task_count": 200,
        "excluded_pilot_task_count": 8,
        "selected_task_count": 192,
        "sde_task_count": 10,
        "sde_registration_policy": HA_ONLINE_TASK_COHORT_SDE_POLICY,
    }
    for field, expected in expected_values.items():
        if contract.get(field) != expected:
            errors.append(f"{cohort_prefix}.{field} must equal {expected!r}")
    source_path = str(contract.get("source_path") or "").strip()
    if not source_path or Path(source_path).is_absolute():
        errors.append(f"{cohort_prefix}.source_path must be project-relative")
    for field in (
        "source_sha256",
        "canonical_task_ids_sha256",
        "excluded_task_ids_sha256",
        "selected_task_ids_sha256",
        "sde_task_ids_sha256",
        "canonical_task_files_sha256",
        "selected_task_files_sha256",
        "sde_task_files_sha256",
    ):
        if not _is_sha256(str(contract.get(field) or "").lower()):
            errors.append(f"{cohort_prefix}.{field} must be SHA256")
    expected_excluded_sha256 = stable_json_sha256(
        list(HA_ONLINE_TASK_COHORT_EXCLUDED_PILOT_IDS)
    )
    if contract.get("excluded_task_ids_sha256") != expected_excluded_sha256:
        errors.append(
            f"{cohort_prefix}.excluded_task_ids_sha256 must bind the frozen pilot 8"
        )
    task_ids = sorted(
        Path(str(item)).stem for item in plan.get("tasks") or [] if str(item).strip()
    )
    if len(task_ids) != 192:
        errors.append(f"{prefix}.tasks must contain exactly 192 selected tasks")
    if contract.get("selected_task_ids_sha256") != stable_json_sha256(task_ids):
        errors.append(
            f"{cohort_prefix}.selected_task_ids_sha256 must bind run-plan tasks"
        )
    if list(plan.get("experiment_seeds") or []) != [0]:
        errors.append(
            f"{prefix}.experiment_seeds must equal [0] for the one-run-per-task cohort"
        )
    sampling_contract = plan.get("sampling_contract")
    if sampling_contract != HA_SINGLE_FIXED_SEED_SAMPLING_CONTRACT:
        errors.append(
            f"{prefix}.sampling_contract must exactly bind the fixed-seed task "
            "census inference scope and prohibit a model-sampling variance claim"
        )
    if runtime_semantics != "legacy_official_v1":
        errors.append(
            f"{prefix}.online_advisory_contract.runtime.resource_semantics_mode "
            "must equal legacy_official_v1 for the remaining cohort"
        )


def validate_preregistration(payload: dict[str, Any], *, benchmark: str) -> None:
    normalized_benchmark = str(benchmark).strip().lower()
    if normalized_benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported preregistration benchmark: {benchmark!r}")
    errors: list[str] = []
    study_id = str(payload.get("study_id") or "").strip()
    study_version_suffix = study_id.rsplit("-v", 1)
    ha_task_cohort_required = (
        len(study_version_suffix) == 2
        and study_version_suffix[1].isdigit()
        and int(study_version_suffix[1]) >= 5
    )

    if _integer(payload.get("schema_version")) != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")
    for field in ("study_id", "frozen_at_utc"):
        value = str(payload.get(field) or "").strip()
        if not value or value.upper().startswith("REPLACE"):
            errors.append(f"{field} is required")
    frozen_at = str(payload.get("frozen_at_utc") or "").strip()
    try:
        parsed_frozen_at = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError:
        parsed_frozen_at = None
    if parsed_frozen_at is None or parsed_frozen_at.tzinfo is None:
        errors.append("frozen_at_utc must be an ISO-8601 timestamp with timezone")
    scope = payload.get("benchmark_scope")
    if not isinstance(scope, list) or normalized_benchmark not in {
        str(item).strip().lower() for item in scope
    }:
        errors.append(f"benchmark_scope must include {normalized_benchmark}")
    if str(payload.get("primary_prediction_estimand") or "") != "exact_step":
        errors.append("primary_prediction_estimand must be exact_step")

    run_plans = _mapping(payload, "run_plans", errors)
    for scoped_benchmark in sorted(
        {str(item).strip().lower() for item in scope or []}
    ):
        if scoped_benchmark not in BENCHMARKS:
            errors.append(f"benchmark_scope contains unsupported value {scoped_benchmark!r}")
            continue
        plan = _mapping(run_plans, scoped_benchmark, errors, prefix="run_plans")
        task_field = "tasks" if scoped_benchmark == "harnessaudit" else "task_ids"
        _nonempty_unique_strings(plan, task_field, errors, prefix=f"run_plans.{scoped_benchmark}")
        _nonempty_unique_integers(
            plan,
            "experiment_seeds",
            errors,
            prefix=f"run_plans.{scoped_benchmark}",
        )
        _offline_replicates(
            plan,
            errors,
            benchmark=scoped_benchmark,
            prefix=f"run_plans.{scoped_benchmark}",
        )
        _nonempty_unique_strings(
            plan,
            "policy_modes",
            errors,
            prefix=f"run_plans.{scoped_benchmark}",
        )
        observed_policy_modes = {
            str(item).strip()
            for item in plan.get("policy_modes") or []
            if str(item).strip()
        }
        unsupported_policy_modes = sorted(
            observed_policy_modes - POLICY_MODES_BY_BENCHMARK[scoped_benchmark]
        )
        if unsupported_policy_modes:
            errors.append(
                f"run_plans.{scoped_benchmark}.policy_modes contains unsupported "
                f"or legacy modes: {unsupported_policy_modes}"
            )
        if (
            scoped_benchmark == "isbench"
            and observed_policy_modes
            not in ISBENCH_ALLOWED_ONLINE_POLICY_MODE_SETS
        ):
            errors.append(
                "run_plans.isbench.policy_modes must equal either all three "
                "registered policy tiers or train_policy only"
            )
        _nonempty_unique_strings(
            plan,
            "intervention_modes",
            errors,
            prefix=f"run_plans.{scoped_benchmark}",
        )
        _minimum_integer(
            plan,
            "prediction_horizon",
            1 if scoped_benchmark == "harnessaudit" else 5,
            errors,
            prefix=f"run_plans.{scoped_benchmark}",
        )
        observed_intervention_modes = {
            str(item).strip()
            for item in plan.get("intervention_modes") or []
            if str(item).strip()
        }
        if scoped_benchmark == "harnessaudit":
            _validate_ha_online_task_cohort_contract(
                plan,
                errors=errors,
                prefix="run_plans.harnessaudit",
                require_contract=ha_task_cohort_required,
            )
            if observed_intervention_modes not in HA_ALLOWED_ONLINE_INTERVENTION_MODE_SETS:
                errors.append(
                    "run_plans.harnessaudit.intervention_modes must equal a "
                    "registered legacy or distributed intervention arm set"
                )
        elif (
            observed_intervention_modes
            not in ISBENCH_ALLOWED_ONLINE_INTERVENTION_MODE_SETS
        ):
            errors.append(
                f"run_plans.{scoped_benchmark}.intervention_modes must equal the "
                "registered legacy or distributed intervention arm set"
            )
        _validate_online_advisory_contract(
            plan.get("online_advisory_contract"),
            benchmark=scoped_benchmark,
            policy_modes=observed_policy_modes,
            intervention_modes=observed_intervention_modes,
            experiment_seeds=plan.get("experiment_seeds") or [],
            prediction_horizon=_integer(plan.get("prediction_horizon")),
            errors=errors,
            prefix=f"run_plans.{scoped_benchmark}.online_advisory_contract",
        )

    offline = _mapping(payload, "offline_prediction", errors)
    horizons = _positive_integer_list(
        offline.get("horizons"),
        errors,
        label="offline_prediction.horizons",
    )
    max_horizon = _integer(offline.get("max_formal_horizon"))
    if not horizons or max_horizon != max(horizons):
        errors.append(
            "offline_prediction.max_formal_horizon must equal max(horizons)"
        )
    if str(offline.get("gt_horizon_mode") or "") != "exact_step":
        errors.append("offline_prediction.gt_horizon_mode must be exact_step")
    if _integer(offline.get("lead_time_offset_base")) != 0:
        errors.append("offline_prediction.lead_time_offset_base must equal 0")
    rollout_modes_by_benchmark = offline.get("rollout_modes_by_benchmark")
    if not isinstance(rollout_modes_by_benchmark, dict):
        errors.append("offline_prediction.rollout_modes_by_benchmark must be an object")
        rollout_modes_by_benchmark = {}
    for scoped_benchmark in sorted(
        {str(item).strip().lower() for item in scope or []}
    ):
        _nonempty_unique_strings(
            {"rollout_modes": rollout_modes_by_benchmark.get(scoped_benchmark)},
            "rollout_modes",
            errors,
            prefix=(
                "offline_prediction.rollout_modes_by_benchmark."
                + scoped_benchmark
            ),
        )

    protocols = _mapping(offline, "protocol_by_benchmark", errors)
    for scoped_benchmark in sorted(
        {str(item).strip().lower() for item in scope or []}
    ):
        protocol = _mapping(
            protocols,
            scoped_benchmark,
            errors,
            prefix="offline_prediction.protocol_by_benchmark",
        )
        scoped_plan = run_plans.get(scoped_benchmark)
        online_contract = (
            scoped_plan.get("online_advisory_contract")
            if isinstance(scoped_plan, dict)
            else None
        )
        policy_contract = (
            online_contract.get("policy_extraction_contract")
            if isinstance(online_contract, dict)
            else None
        )
        _validate_offline_protocol(
            protocol,
            benchmark=scoped_benchmark,
            rollout_modes=rollout_modes_by_benchmark.get(scoped_benchmark),
            offline_replicates=(
                scoped_plan.get("offline_replicates")
                if isinstance(scoped_plan, dict)
                else None
            ),
            require_ha_model_input_boundary=(
                scoped_benchmark == "harnessaudit"
                and isinstance(policy_contract, dict)
                and policy_contract.get("schema_version")
                == HA_POLICY_EXTRACTION_CONTRACT_SCHEMA_VERSION
            ),
            errors=errors,
            prefix=(
                "offline_prediction.protocol_by_benchmark."
                + scoped_benchmark
            ),
        )

    gt_contract = _mapping(payload, "ground_truth_contract", errors)
    ha_resource_semantics = str(
        gt_contract.get("harnessaudit_resource_semantics") or ""
    )
    if ha_resource_semantics not in {"explicit_effect_v1", "legacy_official_v1"}:
        errors.append(
            "ground_truth_contract.harnessaudit_resource_semantics must be one of "
            "explicit_effect_v1 or legacy_official_v1"
        )
    if "harnessaudit" in {
        str(item).strip().lower() for item in scope or []
    }:
        ha_plan = run_plans.get("harnessaudit")
        ha_runtime = (
            (ha_plan.get("online_advisory_contract") or {}).get("runtime")
            if isinstance(ha_plan, dict)
            else None
        )
        observed_resource_semantics = (
            str(ha_runtime.get("resource_semantics_mode") or "")
            if isinstance(ha_runtime, dict)
            else ""
        )
        if not observed_resource_semantics:
            errors.append(
                "run_plans.harnessaudit.online_advisory_contract.runtime."
                "resource_semantics_mode is required"
            )
        expected_gt_semantics = {
            "legacy_official_v1": "legacy_official_v1",
            "explicit_v2": "explicit_effect_v1",
        }.get(observed_resource_semantics)
        if observed_resource_semantics and expected_gt_semantics != ha_resource_semantics:
            errors.append(
                "HarnessAudit online runtime resource_semantics_mode must map to "
                "ground_truth_contract.harnessaudit_resource_semantics"
            )
    if str(gt_contract.get("isbench_null_eval") or "") != "not_evaluable":
        errors.append("ground_truth_contract.isbench_null_eval must be not_evaluable")

    faithfulness = _mapping(payload, "decision_faithfulness", errors)
    if faithfulness.get("stages") != FAITHFULNESS_STAGES:
        errors.append("decision_faithfulness.stages must equal the frozen seven-stage funnel")
    if str(faithfulness.get("primary_intervention") or "") != "advisory_only":
        errors.append("decision_faithfulness.primary_intervention must be advisory_only")
    if _integer(faithfulness.get("min_blinded_adjudicators")) is None or _integer(
        faithfulness.get("min_blinded_adjudicators")
    ) < 2:
        errors.append("decision_faithfulness.min_blinded_adjudicators must be >= 2")

    annotations = _mapping(payload, "annotation_contract", errors)
    for field in (
        "compositionality_manifest_sha256",
        "faithfulness_adjudication_protocol_sha256",
        "adjudicator_blinding_protocol_sha256",
        "multiplicity_analysis_code_sha256",
    ):
        _sha256_field(annotations, field, errors, prefix="annotation_contract")

    support = _mapping(payload, "support", errors)
    ha_plan_for_support = run_plans.get("harnessaudit")
    ha_cohort_for_support = (
        ha_plan_for_support.get("task_cohort_contract")
        if isinstance(ha_plan_for_support, dict)
        else None
    )
    ha_sampling_for_support = (
        ha_plan_for_support.get("sampling_contract")
        if isinstance(ha_plan_for_support, dict)
        else None
    )
    ha_intervention_modes_for_support = {
        str(item).strip()
        for item in (
            ha_plan_for_support.get("intervention_modes")
            if isinstance(ha_plan_for_support, dict)
            else []
        )
        or []
        if str(item).strip()
    }
    scope_set = {str(item).strip().lower() for item in scope or []}
    fixed_seed_census_contract = (
        scope_set == {"harnessaudit"}
        and isinstance(ha_cohort_for_support, dict)
        and ha_cohort_for_support.get("selected_task_count") == 192
        and ha_sampling_for_support == HA_SINGLE_FIXED_SEED_SAMPLING_CONTRACT
    )
    ha_single_seed_census_arm_sets = (
        HA_THREE_ARM_INTERVENTION_MODES,
        HA_DISTRIBUTED_FOUR_ARM_INTERVENTION_MODES,
        HA_DISTRIBUTED_FIVE_ARM_INTERVENTION_MODES,
    )
    if (
        fixed_seed_census_contract
        and ha_intervention_modes_for_support not in ha_single_seed_census_arm_sets
    ):
        errors.append(
            "the HarnessAudit complete-task-census single-fixed-seed exception "
            "requires exactly no_guard/advisory_only/enforced_guard or a "
            "registered distributed perturbation arm set"
        )
    single_fixed_seed_census = (
        fixed_seed_census_contract
        and ha_intervention_modes_for_support in ha_single_seed_census_arm_sets
    )
    isbench_plan_for_support = run_plans.get("isbench")
    isbench_one_pass_internal = bool(
        scope_set == {"isbench"}
        and isbench_uses_registered_single_seed_one_pass_contract(
            isbench_plan_for_support
        )
    )
    _minimum_integer(
        support,
        "min_seeds",
        1 if single_fixed_seed_census or isbench_one_pass_internal else 5,
        errors,
        prefix="support",
    )
    _minimum_integer(support, "min_tasks", 2, errors, prefix="support")
    _minimum_integer(
        support,
        "min_task_seed_clusters",
        2,
        errors,
        prefix="support",
    )
    if isinstance(ha_cohort_for_support, dict):
        expected_remaining_support = {
            "min_seeds": 1,
            "min_tasks": 192,
            "min_task_seed_clusters": 192,
        }
        for field, expected in expected_remaining_support.items():
            if support.get(field) != expected:
                errors.append(
                    f"support.{field} must equal {expected} for the "
                    "one-run-per-task remaining HA cohort"
                )
    if isbench_one_pass_internal:
        expected_one_pass_support = {
            "min_seeds": 1,
            "min_tasks": 161,
            "min_task_seed_clusters": 161,
        }
        for field, expected in expected_one_pass_support.items():
            if support.get(field) != expected:
                errors.append(
                    f"support.{field} must equal {expected} for the internal "
                    "registered-hardware single-seed one-pass IS-Bench cohort"
                )
        if isbench_plan_for_support.get("offline_replicates") != [
            {"actor_trace_seed": 0, "wm_seed": 0}
        ]:
            errors.append(
                "run_plans.isbench.offline_replicates must contain only seed 0 "
                "for the registered-hardware single-seed one-pass cohort"
            )
    min_seeds = _integer(support.get("min_seeds"))
    if min_seeds is not None:
        for scoped_benchmark in sorted(
            {str(item).strip().lower() for item in scope or []}
        ):
            scoped_plan = run_plans.get(scoped_benchmark)
            replicates = (
                scoped_plan.get("offline_replicates")
                if isinstance(scoped_plan, dict)
                else None
            )
            if isinstance(replicates, list) and len(replicates) < min_seeds:
                errors.append(
                    f"run_plans.{scoped_benchmark}.offline_replicates must "
                    f"contain at least support.min_seeds={min_seeds} entries"
                )
            actor_identities: set[tuple[str, str, int | None]] = set()
            for item in replicates or []:
                if not isinstance(item, dict):
                    continue
                expanded = (
                    "actor_protocol_profile" in item
                    or "actor_replicate_id" in item
                )
                try:
                    identity = normalize_actor_replicate_identity(
                        benchmark=scoped_benchmark,
                        actor_protocol_profile=(
                            item.get("actor_protocol_profile") if expanded else None
                        ),
                        actor_replicate_id=(
                            item.get("actor_replicate_id") if expanded else None
                        ),
                        actor_trace_seed=item.get("actor_trace_seed"),
                    )
                except ValueError:
                    continue
                actor_identities.add(identity.full_key)
            if len(actor_identities) < min_seeds:
                errors.append(
                    f"run_plans.{scoped_benchmark}.offline_replicates must contain "
                    f"at least support.min_seeds={min_seeds} distinct normalized "
                    "actor identities so repeated WM calls are not treated as "
                    "independent traces"
                )
            declared_seeds = (
                scoped_plan.get("experiment_seeds")
                if isinstance(scoped_plan, dict)
                else None
            )
            if isinstance(declared_seeds, list) and len(set(declared_seeds)) < min_seeds:
                errors.append(
                    f"run_plans.{scoped_benchmark}.experiment_seeds must contain "
                    f"at least support.min_seeds={min_seeds} unique WM seeds"
                )
    _minimum_integer(
        support,
        "min_correct_exposed_points",
        1,
        errors,
        prefix="support",
    )
    _minimum_integer(
        support,
        "min_blinded_adjudicators",
        2,
        errors,
        prefix="support",
    )

    endpoints = _mapping(payload, "endpoints", errors)
    for field in ("primary_safety", "primary_utility"):
        value = str(endpoints.get(field) or "").strip()
        if not value or value.upper().startswith("REPLACE"):
            errors.append(f"endpoints.{field} is required")

    hypotheses = _mapping(payload, "hypotheses", errors)
    delayed = _mapping(hypotheses, "delayed_degradation", errors, prefix="hypotheses")
    _positive_number(
        delayed,
        "minimum_negative_slope",
        errors,
        prefix="hypotheses.delayed_degradation",
    )
    ignored = _mapping(hypotheses, "predict_but_ignore", errors, prefix="hypotheses")
    _unit_interval(
        ignored,
        "minimum_acknowledgement_rate",
        errors,
        prefix="hypotheses.predict_but_ignore",
    )
    _unit_interval(
        ignored,
        "maximum_independently_safer_revision_rate",
        errors,
        prefix="hypotheses.predict_but_ignore",
    )
    integration = _mapping(
        hypotheses,
        "integration_bottleneck",
        errors,
        prefix="hypotheses",
    )
    _positive_number(
        integration,
        "minimum_prediction_gain",
        errors,
        prefix="hypotheses.integration_bottleneck",
    )
    _nonnegative_number(
        integration,
        "maximum_safety_gain",
        errors,
        prefix="hypotheses.integration_bottleneck",
    )

    decision = _mapping(payload, "decision_rules", errors)
    _positive_number(
        decision,
        "minimum_paired_safety_effect",
        errors,
        prefix="decision_rules",
    )
    _nonnegative_number(
        decision,
        "utility_noninferiority_margin",
        errors,
        prefix="decision_rules",
    )

    statistics = _mapping(payload, "statistics", errors)
    alpha = _number(statistics.get("alpha"))
    if alpha is None or not 0.0 < alpha < 0.5:
        errors.append("statistics.alpha must be in (0, 0.5)")
    _minimum_integer(statistics, "bootstrap_samples", 1000, errors, prefix="statistics")
    if str(statistics.get("multiplicity_method") or "") not in MULTIPLICITY_METHODS:
        errors.append(
            "statistics.multiplicity_method must be one of "
            + ", ".join(sorted(MULTIPLICITY_METHODS))
        )

    if errors:
        raise ValueError("Invalid preregistration: " + "; ".join(errors))


def validate_preregistered_run_plan(
    preregistration: dict[str, Any],
    *,
    benchmark: str,
    tasks: Iterable[str],
    experiment_seeds: Iterable[int],
    policy_modes: Iterable[str],
    intervention_modes: Iterable[str],
    prediction_horizon: int,
    observed_online_advisory_contract: Mapping[str, Any] | None = None,
) -> None:
    """Require exact agreement with the frozen core run-plan dimensions."""
    normalized = str(benchmark).strip().lower()
    plan = preregistration["payload"]["run_plans"][normalized]
    task_field = "tasks" if normalized == "harnessaudit" else "task_ids"
    observed_tasks = {Path(str(item)).stem for item in tasks}
    expected_tasks = {Path(str(item)).stem for item in plan[task_field]}
    comparisons = {
        "tasks": (observed_tasks, expected_tasks),
        "experiment_seeds": (
            {int(item) for item in experiment_seeds},
            {int(item) for item in plan["experiment_seeds"]},
        ),
        "policy_modes": (
            {str(item) for item in policy_modes if item is not None},
            {str(item) for item in plan["policy_modes"]},
        ),
        "intervention_modes": (
            {str(item) for item in intervention_modes},
            {str(item) for item in plan["intervention_modes"]},
        ),
    }
    errors = [
        f"{field}: observed={sorted(observed)}, preregistered={sorted(expected)}"
        for field, (observed, expected) in comparisons.items()
        if observed != expected
    ]
    if int(prediction_horizon) != int(plan["prediction_horizon"]):
        errors.append(
            f"prediction_horizon: observed={prediction_horizon}, "
            f"preregistered={plan['prediction_horizon']}"
        )
    expected_advisory_contract = plan.get("online_advisory_contract")
    if not isinstance(observed_online_advisory_contract, Mapping):
        errors.append("online_advisory_contract: observed contract is required")
    elif _canonical_json(dict(observed_online_advisory_contract)) != _canonical_json(
        expected_advisory_contract
    ):
        differences = _mapping_differences(
            observed_online_advisory_contract,
            expected_advisory_contract,
            prefix="online_advisory_contract",
        )
        errors.append(
            "online_advisory_contract differs: "
            + ("; ".join(differences) if differences else "canonical JSON differs")
        )
    if errors:
        raise ValueError(
            f"Formal {normalized} run plan differs from preregistration: "
            + "; ".join(errors)
        )


def validate_preregistered_offline_plan(
    preregistration: dict[str, Any],
    *,
    benchmark: str,
    tasks: Iterable[str],
    policy_modes: Iterable[str],
    rollout_modes: Iterable[str],
    horizons: Iterable[int],
    max_formal_horizon: int,
    gt_horizon_mode: str,
    lead_time_offset_base: int,
    actor_trace_seed: int | None,
    wm_seed: int,
    observed_protocol: Mapping[str, Any],
    source_input_manifest_sha256: str,
    actor_protocol_profile: str | None = None,
    actor_replicate_id: str | None = None,
) -> dict[str, Any]:
    """Validate one preregistered offline replicate, never the full study.

    Every result-affecting protocol field is compared as canonical JSON.  The
    source manifest is selected by ``actor_trace_seed`` and independently
    checked.  A successful single-run gate deliberately reports
    ``confirmatory_complete=False``; only
    :func:`validate_preregistered_offline_replicates` can certify the complete
    preregistered replicate set.
    """
    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARKS:
        raise ValueError(f"Unsupported preregistration benchmark: {benchmark!r}")
    if type(wm_seed) is not int:
        raise ValueError("wm_seed must be an exact integer")
    payload = preregistration["payload"]
    plan = payload["run_plans"][normalized]
    offline = payload["offline_prediction"]
    task_field = "tasks" if normalized == "harnessaudit" else "task_ids"

    observed_tasks = {Path(str(item)).stem for item in tasks if str(item).strip()}
    expected_tasks = {Path(str(item)).stem for item in plan[task_field]}
    observed_policies = {str(item).strip() for item in policy_modes if str(item).strip()}
    expected_policies = {str(item).strip() for item in plan["policy_modes"]}
    observed_horizons = sorted({int(item) for item in horizons})
    expected_horizons = sorted(int(item) for item in offline["horizons"])
    observed_rollouts = sorted(
        {str(item).strip() for item in rollout_modes if str(item).strip()}
    )

    errors: list[str] = []
    if observed_tasks != expected_tasks:
        errors.append(
            f"tasks: observed={sorted(observed_tasks)}, "
            f"preregistered={sorted(expected_tasks)}"
        )
    if observed_policies != expected_policies:
        errors.append(
            f"policy_modes: observed={sorted(observed_policies)}, "
            f"preregistered={sorted(expected_policies)}"
        )
    if observed_horizons != expected_horizons:
        errors.append(
            f"horizons: observed={observed_horizons}, "
            f"preregistered={expected_horizons}"
        )
    expected_max = int(offline["max_formal_horizon"])
    if int(max_formal_horizon) != expected_max:
        errors.append(
            f"max_formal_horizon: observed={max_formal_horizon}, "
            f"preregistered={expected_max}"
        )
    expected_gt_mode = str(offline["gt_horizon_mode"])
    if str(gt_horizon_mode) != expected_gt_mode:
        errors.append(
            f"gt_horizon_mode: observed={gt_horizon_mode!r}, "
            f"preregistered={expected_gt_mode!r}"
        )
    expected_offset = int(offline["lead_time_offset_base"])
    if int(lead_time_offset_base) != expected_offset:
        errors.append(
            f"lead_time_offset_base: observed={lead_time_offset_base}, "
            f"preregistered={expected_offset}"
        )
    observed_identity = normalize_actor_replicate_identity(
        benchmark=normalized,
        actor_protocol_profile=actor_protocol_profile,
        actor_replicate_id=actor_replicate_id,
        actor_trace_seed=actor_trace_seed,
    )
    observed_pair = (*observed_identity.full_key, wm_seed)
    expected_pairs = _replicate_pairs(
        plan["offline_replicates"],
        benchmark=normalized,
    )
    if observed_pair not in expected_pairs:
        errors.append(
            "offline_replicate: observed="
            f"{_replicate_dicts({observed_pair})[0]}, "
            f"not in preregistered={_replicate_dicts(expected_pairs)}"
        )

    rollout_modes_by_benchmark = offline.get("rollout_modes_by_benchmark")
    preregistered_rollouts = rollout_modes_by_benchmark[normalized]
    expected_rollouts = sorted(
        {str(item).strip() for item in preregistered_rollouts if str(item).strip()}
    )
    if observed_rollouts != expected_rollouts:
        errors.append(
            f"rollout_modes: observed={observed_rollouts}, "
            f"preregistered={expected_rollouts}"
        )

    expected_protocol = offline["protocol_by_benchmark"][normalized]
    if not isinstance(observed_protocol, Mapping):
        errors.append("observed_protocol must be an object")
        observed_protocol_dict: dict[str, Any] = {}
    else:
        observed_protocol_dict = dict(observed_protocol)
    if _canonical_json(observed_protocol_dict) != _canonical_json(expected_protocol):
        differences = _mapping_differences(
            observed_protocol_dict,
            expected_protocol,
            prefix="protocol",
        )
        errors.append(
            "offline protocol differs from preregistration: "
            + ("; ".join(differences) if differences else "canonical JSON differs")
        )

    observed_source_digest = str(source_input_manifest_sha256 or "").strip().lower()
    if not _is_sha256(observed_source_digest):
        errors.append("source_input_manifest_sha256 must be a resolved SHA256 digest")
    source_contract = expected_protocol["source_inputs"]
    if observed_identity.actor_protocol_profile == SEEDED_SAMPLING_PROFILE:
        manifest_field = "manifest_sha256_by_actor_trace_seed"
        manifest_key = str(observed_identity.actor_trace_seed)
    else:
        manifest_field = "manifest_sha256_by_actor_replicate_id"
        manifest_key = observed_identity.actor_replicate_id
    expected_source_map = source_contract.get(manifest_field)
    expected_source_digest = str(
        expected_source_map.get(manifest_key)
        if isinstance(expected_source_map, dict)
        else ""
    ).strip().lower()
    if not expected_source_digest:
        errors.append(
            "source input manifest is not preregistered for normalized actor "
            f"identity={observed_identity.full_key}"
        )
    elif observed_source_digest != expected_source_digest:
        errors.append(
            "source_input_manifest_sha256: observed="
            f"{observed_source_digest}, preregistered={expected_source_digest}"
        )

    if errors:
        raise ValueError(
            f"Formal {normalized} offline plan differs from preregistration: "
            + "; ".join(errors)
        )
    return {
        "status": "validated_replicate",
        "benchmark": normalized,
        "tasks": sorted(observed_tasks),
        "policy_modes": sorted(observed_policies),
        "horizons": observed_horizons,
        "max_formal_horizon": int(max_formal_horizon),
        "gt_horizon_mode": str(gt_horizon_mode),
        "lead_time_offset_base": int(lead_time_offset_base),
        "actor_protocol_profile": observed_identity.actor_protocol_profile,
        "actor_replicate_id": observed_identity.actor_replicate_id,
        "actor_trace_seed": observed_identity.actor_trace_seed,
        "wm_seed": wm_seed,
        "rollout_modes": {
            "status": "preregistered_exact",
            "observed": observed_rollouts,
            "preregistered": expected_rollouts,
            "contract_source": (
                f"offline_prediction.rollout_modes_by_benchmark.{normalized}"
            ),
        },
        "protocol_sha256": _canonical_value_sha256(expected_protocol),
        "source_input_manifest_sha256": observed_source_digest,
        "preregistration_sha256": str(
            preregistration.get("sha256")
            or canonical_preregistration_sha256(payload)
        ),
        "observed_replicate_count": 1,
        "required_replicate_count": len(expected_pairs),
        "replicate_set_complete": False,
        "confirmatory_complete": False,
    }


def validate_preregistered_offline_replicates(
    preregistration: dict[str, Any],
    *,
    benchmark: str,
    replicate_gates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Certify an exact, unmixed set of preregistered offline replicates."""
    normalized = str(benchmark).strip().lower()
    if normalized not in BENCHMARKS:
        raise ValueError(f"Unsupported preregistration benchmark: {benchmark!r}")
    payload = preregistration["payload"]
    plan = payload["run_plans"][normalized]
    expected_pairs = _replicate_pairs(
        plan["offline_replicates"],
        benchmark=normalized,
    )
    expected_protocol = payload["offline_prediction"]["protocol_by_benchmark"][normalized]
    expected_protocol_sha = _canonical_value_sha256(expected_protocol)
    expected_prereg_sha = str(
        preregistration.get("sha256") or canonical_preregistration_sha256(payload)
    )

    errors: list[str] = []
    observed_pairs: list[tuple[str, str, int | None, int]] = []
    for index, raw_gate in enumerate(replicate_gates):
        gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else {}
        label = f"replicate_gates[{index}]"
        if gate.get("status") != "validated_replicate":
            errors.append(f"{label}.status must equal validated_replicate")
        if str(gate.get("benchmark") or "").strip().lower() != normalized:
            errors.append(f"{label}.benchmark does not match {normalized}")
        try:
            _require_explicit_actor_identity_fields(gate, label=label)
        except ValueError as exc:
            errors.append(str(exc))
            explicit_identity = False
        else:
            explicit_identity = True
        raw_wm_seed = gate.get("wm_seed")
        wm_seed = raw_wm_seed if type(raw_wm_seed) is int else None
        try:
            identity = (
                normalize_actor_replicate_identity(
                    benchmark=normalized,
                    actor_protocol_profile=gate.get("actor_protocol_profile"),
                    actor_replicate_id=gate.get("actor_replicate_id"),
                    actor_trace_seed=gate.get("actor_trace_seed"),
                )
                if explicit_identity
                else None
            )
        except ValueError as exc:
            errors.append(f"{label} has invalid actor identity: {exc}")
            identity = None
        if wm_seed is None:
            errors.append(f"{label}.wm_seed must be an exact integer")
        elif identity is not None:
            observed_pairs.append((*identity.full_key, wm_seed))
        if str(gate.get("protocol_sha256") or "") != expected_protocol_sha:
            errors.append(f"{label}.protocol_sha256 differs from preregistration")
        if str(gate.get("preregistration_sha256") or "") != expected_prereg_sha:
            errors.append(f"{label}.preregistration_sha256 differs from preregistration")
        if gate.get("confirmatory_complete") is not False:
            errors.append(
                f"{label}.confirmatory_complete must be false before joint postflight"
            )

    observed_set = set(observed_pairs)
    if len(observed_pairs) != len(observed_set):
        errors.append("replicate_gates contains duplicate normalized actor/WM pairs")
    if observed_set != expected_pairs:
        missing = sorted(expected_pairs - observed_set)
        unexpected = sorted(observed_set - expected_pairs)
        if missing:
            errors.append(f"missing preregistered replicates={_replicate_dicts(set(missing))}")
        if unexpected:
            errors.append(f"unexpected replicates={_replicate_dicts(set(unexpected))}")
    if errors:
        raise ValueError(
            f"Formal {normalized} offline replicate postflight failed: "
            + "; ".join(errors)
        )
    return {
        "status": "confirmatory_complete",
        "benchmark": normalized,
        "observed_replicate_count": len(observed_set),
        "required_replicate_count": len(expected_pairs),
        "replicates": _replicate_dicts(observed_set),
        "replicate_set_complete": True,
        "confirmatory_complete": True,
        "protocol_sha256": expected_protocol_sha,
        "preregistration_sha256": expected_prereg_sha,
    }


def validate_formal_output_registration(
    *,
    marker_path: Path,
    artifact_roots: Iterable[Path],
    expected_payload: Mapping[str, Any] | None = None,
) -> None:
    """Reject post-hoc formalization of an already populated artifact tree."""
    marker = Path(marker_path).expanduser().resolve()
    if marker.is_file():
        if expected_payload is None:
            return
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError(
                "Formal registration marker is not valid JSON"
            ) from None
        if not isinstance(existing, dict) or canonical_preregistration_sha256(
            existing
        ) != canonical_preregistration_sha256(dict(expected_payload)):
            raise ValueError(
                "Formal registration marker does not match loaded preregistration"
            )
        return
    populated: list[str] = []
    seen: set[Path] = set()
    for raw_root in artifact_roots:
        root = Path(raw_root).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        if root.is_file():
            populated.append(str(root))
        elif root.is_dir() and any(root.iterdir()):
            populated.append(str(root))
    if populated:
        raise ValueError(
            "Formal run cannot adopt existing unregistered artifacts "
            "(including --skip-existing/post-hoc formalization): "
            + ", ".join(populated)
        )


def validate_annotation_contract_files(
    preregistration: dict[str, Any],
    *,
    compositionality_manifest_path: Path | None,
    faithfulness_adjudication_protocol_path: Path | None,
    adjudicator_blinding_protocol_path: Path | None,
) -> dict[str, Path]:
    """Bind preregistered annotation/protocol digests to real input files."""
    supplied = {
        "compositionality_manifest_sha256": compositionality_manifest_path,
        "faithfulness_adjudication_protocol_sha256": (
            faithfulness_adjudication_protocol_path
        ),
        "adjudicator_blinding_protocol_sha256": adjudicator_blinding_protocol_path,
    }
    expected = preregistration["payload"]["annotation_contract"]
    resolved: dict[str, Path] = {}
    errors: list[str] = []
    for field in ANNOTATION_CONTRACT_FILE_FIELDS:
        raw_path = supplied[field]
        if raw_path is None:
            errors.append(f"{field}: matching input file path is required")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            errors.append(f"{field}: input file does not exist: {path}")
            continue
        observed = _sha256_file(path)
        preregistered = str(expected.get(field) or "").strip().lower()
        if observed != preregistered:
            errors.append(
                f"{field}: file SHA256 {observed} does not match "
                f"preregistered {preregistered}"
            )
            continue
        resolved[field] = path
    if errors:
        raise ValueError(
            "Formal annotation contract file validation failed: "
            + "; ".join(errors)
        )
    return resolved


def _validate_offline_protocol(
    protocol: dict[str, Any],
    *,
    benchmark: str,
    rollout_modes: Any,
    offline_replicates: Any,
    require_ha_model_input_boundary: bool = False,
    errors: list[str],
    prefix: str,
) -> None:
    try:
        _canonical_json(protocol)
    except (TypeError, ValueError):
        errors.append(f"{prefix} must contain only finite JSON values")

    point = _mapping(protocol, "point_selection", errors, prefix=prefix)
    scheme = str(point.get("scheme") or "").strip()
    if scheme not in POINT_SELECTION_SCHEMES:
        errors.append(
            f"{prefix}.point_selection.scheme must be one of "
            + ", ".join(sorted(POINT_SELECTION_SCHEMES))
        )
    _minimum_integer(point, "count", 1, errors, prefix=f"{prefix}.point_selection")
    _minimum_integer(point, "interval", 1, errors, prefix=f"{prefix}.point_selection")
    explicit = point.get("explicit_sequences")
    explicit_values = (
        [_integer(item) for item in explicit] if isinstance(explicit, list) else []
    )
    if not isinstance(explicit, list) or any(
        item is None or item < 0 for item in explicit_values
    ):
        errors.append(
            f"{prefix}.point_selection.explicit_sequences must be a list of "
            "non-negative integers"
        )
    elif len(set(explicit_values)) != len(explicit_values):
        errors.append(
            f"{prefix}.point_selection.explicit_sequences must not contain duplicates"
        )
    if scheme == "explicit" and not explicit_values:
        errors.append(
            f"{prefix}.point_selection.explicit_sequences must be non-empty "
            "when scheme=explicit"
        )

    execution = _mapping(
        protocol,
        "rollout_execution_semantics",
        errors,
        prefix=prefix,
    )
    expected_rollouts = {
        str(item).strip() for item in rollout_modes or [] if str(item).strip()
    }
    observed_execution_rollouts = {
        str(item).strip() for item in execution if str(item).strip()
    }
    if observed_execution_rollouts != expected_rollouts:
        errors.append(
            f"{prefix}.rollout_execution_semantics keys must exactly match "
            "rollout modes"
        )
    for rollout_mode, semantic in execution.items():
        value = str(semantic or "").strip()
        if not value or value.upper().startswith("REPLACE"):
            errors.append(
                f"{prefix}.rollout_execution_semantics.{rollout_mode} must be resolved"
            )
    if "direct_multi_horizon" in expected_rollouts and execution.get(
        "direct_multi_horizon"
    ) != "single_horizon_per_call":
        errors.append(
            f"{prefix}.rollout_execution_semantics.direct_multi_horizon must "
            "equal single_horizon_per_call"
        )
    for iterative_mode in ("iterative_wm_rollout", "wm_free_rollout"):
        if iterative_mode in expected_rollouts and execution.get(
            iterative_mode
        ) != "single_hmax_nested_rollout":
            errors.append(
                f"{prefix}.rollout_execution_semantics.{iterative_mode} must "
                "equal single_hmax_nested_rollout"
            )

    model = _mapping(protocol, "model", errors, prefix=prefix)
    for field in ("provider", "model", "revision", "input_mode", "api_base_url"):
        _resolved_string(model, field, errors, prefix=f"{prefix}.model")
    if not isinstance(model.get("response_format_json_object"), bool):
        errors.append(
            f"{prefix}.model.response_format_json_object must be boolean"
        )
    elif model.get("response_format_json_object") is not True:
        errors.append(
            f"{prefix}.model.response_format_json_object must be true for formal prediction"
        )
    _unit_interval(model, "risk_threshold", errors, prefix=f"{prefix}.model")
    prompt_versions = _mapping(model, "prompt_versions", errors, prefix=f"{prefix}.model")
    observed_prompt_rollouts = {
        str(item).strip() for item in prompt_versions if str(item).strip()
    }
    if observed_prompt_rollouts != expected_rollouts:
        errors.append(
            f"{prefix}.model.prompt_versions keys must exactly match rollout modes"
        )
    for rollout_mode, version in prompt_versions.items():
        value = str(version or "").strip()
        if not value or value.upper().startswith("REPLACE"):
            errors.append(
                f"{prefix}.model.prompt_versions.{rollout_mode} must be resolved"
            )
    if benchmark == "harnessaudit" and require_ha_model_input_boundary:
        corrected_prompt_versions = {
            "direct_multi_horizon": HA_DIRECT_PROMPT_VERSION,
            "iterative_wm_rollout": HA_ITERATIVE_PROMPT_VERSION,
        }
        for rollout_mode in expected_rollouts:
            expected_version = corrected_prompt_versions.get(rollout_mode)
            if (
                expected_version is not None
                and prompt_versions.get(rollout_mode) != expected_version
            ):
                errors.append(
                    f"{prefix}.model.prompt_versions.{rollout_mode} must equal "
                    f"{expected_version} for the corrected HA model-input boundary"
                )
        if protocol.get("experience_replay") != {"enable": False}:
            errors.append(
                f"{prefix}.experience_replay must equal {{'enable': False}} "
                "for the corrected HA model-input boundary"
            )

    train_policy = _mapping(protocol, "train_policy", errors, prefix=prefix)
    _sha256_field(
        protocol,
        "implementation_manifest_sha256",
        errors,
        prefix=prefix,
    )
    _resolved_string(
        train_policy,
        "construction_version",
        errors,
        prefix=f"{prefix}.train_policy",
    )
    if benchmark == "harnessaudit":
        if require_ha_model_input_boundary:
            expected_train_policy_boundary = {
                "construction_version": HA_TRAIN_POLICY_CONSTRUCTION_VERSION,
                "model_input_boundary_version": HA_MODEL_INPUT_BOUNDARY_VERSION,
                "policy_extraction_prompt_version": (
                    HA_POLICY_EXTRACTION_PROMPT_VERSION
                ),
                "policy_cache_schema_version": HA_POLICY_CACHE_SCHEMA_VERSION,
                "test_visible_fields": list(
                    HA_POLICY_EXTRACTION_TEST_VISIBLE_FIELDS
                ),
                "input_includes_test_access_rules": False,
                "input_includes_test_tool_necessity": False,
                "input_includes_tool_necessity_derivatives": False,
                "include_test_tool_necessity_policies": False,
            }
            for field, expected_value in expected_train_policy_boundary.items():
                if train_policy.get(field) != expected_value:
                    errors.append(
                        f"{prefix}.train_policy.{field} is invalid for the "
                        "corrected HA model-input boundary"
                    )
        _sha256_field(protocol, "base_config_sha256", errors, prefix=prefix)
        _resolved_string(
            train_policy,
            "policy_extraction_model",
            errors,
            prefix=f"{prefix}.train_policy",
        )
        if _integer(train_policy.get("policy_extraction_seed")) is None:
            errors.append(
                f"{prefix}.train_policy.policy_extraction_seed must be an integer"
            )
        _sha256_field(
            train_policy,
            "policy_inputs_manifest_sha256",
            errors,
            prefix=f"{prefix}.train_policy",
        )
        _minimum_integer(
            protocol,
            "max_trace_events",
            1,
            errors,
            prefix=prefix,
        )
        _sha256_field(protocol, "approved_task_manifest_sha256", errors, prefix=prefix)
    elif benchmark == "isbench":
        if str(model.get("input_mode") or "") != "direct_vision":
            errors.append(
                f"{prefix}.model.input_mode must equal direct_vision for formal ISBench"
            )
        _minimum_integer(
            train_policy,
            "max_tasks",
            1,
            errors,
            prefix=f"{prefix}.train_policy",
        )
        if not isinstance(train_policy.get("same_domain_only"), bool):
            errors.append(f"{prefix}.train_policy.same_domain_only must be boolean")
        _resolved_string(protocol, "gt_label_mode", errors, prefix=prefix)
        visual = _mapping(protocol, "visual", errors, prefix=prefix)
        # Zero is the explicit, frozen sentinel for "all current camera views".
        _minimum_integer(visual, "max_images", 0, errors, prefix=f"{prefix}.visual")
        _minimum_integer(
            visual,
            "image_max_side",
            1,
            errors,
            prefix=f"{prefix}.visual",
        )
        image_quality = _integer(visual.get("image_quality"))
        if image_quality is None or not 1 <= image_quality <= 100:
            errors.append(f"{prefix}.visual.image_quality must be an integer in [1, 100]")

    source_inputs = _mapping(protocol, "source_inputs", errors, prefix=prefix)
    _resolved_string(
        source_inputs,
        "manifest_kind",
        errors,
        prefix=f"{prefix}.source_inputs",
    )
    seed_map_name = "manifest_sha256_by_actor_trace_seed"
    replicate_map_name = "manifest_sha256_by_actor_replicate_id"
    present_maps = [
        name
        for name in (seed_map_name, replicate_map_name)
        if name in source_inputs
    ]
    if len(present_maps) != 1:
        errors.append(
            f"{prefix}.source_inputs must contain exactly one of "
            f"{seed_map_name} or {replicate_map_name}"
        )
        manifests: dict[str, Any] = {}
    else:
        raw_manifests = source_inputs.get(present_maps[0])
        if not isinstance(raw_manifests, dict):
            errors.append(f"{prefix}.source_inputs.{present_maps[0]} must be an object")
            manifests = {}
        else:
            manifests = raw_manifests
    if not manifests:
        errors.append(
            f"{prefix}.source_inputs manifest map must not be empty"
        )
    identities: list[ActorReplicateIdentity] = []
    if isinstance(offline_replicates, list):
        for item in offline_replicates:
            if not isinstance(item, dict):
                continue
            expanded = (
                "actor_protocol_profile" in item or "actor_replicate_id" in item
            )
            try:
                identities.append(
                    normalize_actor_replicate_identity(
                        benchmark=benchmark,
                        actor_protocol_profile=(
                            item.get("actor_protocol_profile") if expanded else None
                        ),
                        actor_replicate_id=(
                            item.get("actor_replicate_id") if expanded else None
                        ),
                        actor_trace_seed=item.get("actor_trace_seed"),
                    )
                )
            except ValueError:
                continue
    profiles = {identity.actor_protocol_profile for identity in identities}
    if profiles == {SEEDED_SAMPLING_PROFILE}:
        expected_map_name = seed_map_name
        expected_keys = {
            str(identity.actor_trace_seed) for identity in identities
        }
    elif profiles == {DETERMINISTIC_CLOSE_SOURCE_PROFILE}:
        expected_map_name = replicate_map_name
        expected_keys = {
            identity.actor_replicate_id for identity in identities
        }
    else:
        expected_map_name = ""
        expected_keys = set()
    if present_maps and expected_map_name and present_maps[0] != expected_map_name:
        errors.append(
            f"{prefix}.source_inputs uses wrong-profile manifest map: "
            f"observed={present_maps[0]}, required={expected_map_name}"
        )
    observed_keys = {str(key) for key in manifests}
    if expected_keys and observed_keys != expected_keys:
        errors.append(
            f"{prefix}.source_inputs manifest keys must exactly match normalized "
            f"actor identities: observed={sorted(observed_keys)}, "
            f"expected={sorted(expected_keys)}"
        )
    for key, digest in manifests.items():
        if expected_map_name == seed_map_name and _integer(key) is None:
            errors.append(
                f"{prefix}.source_inputs manifest key {key!r} must be an integer seed"
            )
        if not _is_sha256(str(digest or "").strip().lower()):
            errors.append(
                f"{prefix}.source_inputs manifest for key {key!r} must be a "
                "resolved SHA256 digest"
            )


def _offline_replicates(
    plan: dict[str, Any],
    errors: list[str],
    *,
    benchmark: str,
    prefix: str,
) -> None:
    replicates = plan.get("offline_replicates")
    if not isinstance(replicates, list) or not replicates:
        errors.append(f"{prefix}.offline_replicates must be a non-empty list")
        return
    records: list[tuple[ActorReplicateIdentity, int]] = []
    legacy_keys = {"actor_trace_seed", "wm_seed"}
    expanded_keys = {
        "actor_protocol_profile",
        "actor_replicate_id",
        "actor_trace_seed",
        "wm_seed",
    }
    for index, raw_replicate in enumerate(replicates):
        label = f"{prefix}.offline_replicates[{index}]"
        if not isinstance(raw_replicate, dict):
            errors.append(f"{label} must be an object")
            continue
        keys = set(raw_replicate)
        if keys == legacy_keys:
            profile = replicate_id = None
        elif keys == expanded_keys:
            profile = raw_replicate.get("actor_protocol_profile")
            replicate_id = raw_replicate.get("actor_replicate_id")
        else:
            errors.append(
                f"{label} must contain exactly the legacy seeded identity fields "
                "(actor_trace_seed, wm_seed) or the expanded identity fields "
                "(actor_protocol_profile, actor_replicate_id, actor_trace_seed, wm_seed)"
            )
            continue
        raw_wm_seed = raw_replicate.get("wm_seed")
        if type(raw_wm_seed) is not int:
            errors.append(f"{label}.wm_seed must be an exact integer")
            continue
        wm_seed = raw_wm_seed
        try:
            identity = normalize_actor_replicate_identity(
                benchmark=benchmark,
                actor_protocol_profile=profile,
                actor_replicate_id=replicate_id,
                actor_trace_seed=raw_replicate.get("actor_trace_seed"),
            )
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        records.append((identity, wm_seed))
    normalized_pairs = [(identity.full_key, wm_seed) for identity, wm_seed in records]
    if len(normalized_pairs) != len(set(normalized_pairs)):
        errors.append(f"{prefix}.offline_replicates must not contain duplicates")
    identities = [identity.full_key for identity, _wm_seed in records]
    if len(identities) != len(set(identities)):
        errors.append(
            f"{prefix}.offline_replicates must use each normalized actor identity "
            "exactly once; repeated WM measurements may not reweight the formal point estimate"
        )
    profiles = {identity.actor_protocol_profile for identity, _wm_seed in records}
    if len(profiles) > 1:
        errors.append(
            f"{prefix}.offline_replicates must not contain mixed actor protocol profiles"
        )
    actor_seeds = [
        identity.actor_trace_seed
        for identity, _wm_seed in records
        if identity.actor_protocol_profile == SEEDED_SAMPLING_PROFILE
    ]
    if len(actor_seeds) != len(set(actor_seeds)):
        errors.append(
            f"{prefix}.offline_replicates must use each actor_trace_seed exactly "
            "once; repeated WM measurements may not reweight the formal point estimate"
        )
    wm_seeds = [wm_seed for _identity, wm_seed in records]
    if len(wm_seeds) != len(set(wm_seeds)):
        errors.append(
            f"{prefix}.offline_replicates must use each wm_seed exactly once"
        )
    declared_wm_seeds = {
        value
        for value in (_integer(item) for item in plan.get("experiment_seeds") or [])
        if value is not None
    }
    observed_wm_seeds = set(wm_seeds)
    if records and observed_wm_seeds != declared_wm_seeds:
        errors.append(
            f"{prefix}.offline_replicates wm_seed values must exactly equal "
            f"experiment_seeds: observed={sorted(observed_wm_seeds)}, "
            f"declared={sorted(declared_wm_seeds)}"
        )


def _replicate_pairs(
    replicates: Iterable[Mapping[str, Any]],
    *,
    benchmark: str,
) -> set[tuple[str, str, int | None, int]]:
    pairs: set[tuple[str, str, int | None, int]] = set()
    for item in replicates:
        if type(item.get("wm_seed")) is not int:
            raise ValueError("offline replicate wm_seed must be an exact integer")
        keys = set(item)
        expanded = "actor_protocol_profile" in keys or "actor_replicate_id" in keys
        identity = normalize_actor_replicate_identity(
            benchmark=benchmark,
            actor_protocol_profile=(
                item.get("actor_protocol_profile") if expanded else None
            ),
            actor_replicate_id=(
                item.get("actor_replicate_id") if expanded else None
            ),
            actor_trace_seed=item.get("actor_trace_seed"),
        )
        pairs.add((*identity.full_key, item["wm_seed"]))
    return pairs


def _replicate_dicts(
    pairs: set[tuple[str, str, int | None, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "actor_protocol_profile": profile,
            "actor_replicate_id": replicate_id,
            "actor_trace_seed": actor_seed,
            "wm_seed": wm_seed,
        }
        for profile, replicate_id, actor_seed, wm_seed in sorted(
            pairs, key=lambda item: (item[0], item[1], str(item[2]), item[3])
        )
    ]


def _resolved_string(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = str(mapping.get(key) or "").strip()
    if not value or value.upper().startswith("REPLACE"):
        errors.append(f"{prefix}.{key} must be a resolved string")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping_differences(
    observed: Any,
    expected: Any,
    *,
    prefix: str,
) -> list[str]:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        differences: list[str] = []
        for key in sorted(set(observed) | set(expected), key=str):
            label = f"{prefix}.{key}"
            if key not in observed:
                differences.append(f"{label}: missing (preregistered={expected[key]!r})")
            elif key not in expected:
                differences.append(f"{label}: unexpected (observed={observed[key]!r})")
            else:
                differences.extend(
                    _mapping_differences(observed[key], expected[key], prefix=label)
                )
        return differences
    if observed != expected:
        return [f"{prefix}: observed={observed!r}, preregistered={expected!r}"]
    return []


def _mapping(
    parent: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    value = parent.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _minimum_integer(
    mapping: dict[str, Any],
    key: str,
    minimum: int,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = _integer(mapping.get(key))
    if value is None or value < minimum:
        errors.append(f"{prefix}.{key} must be an integer >= {minimum}")


def _positive_number(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = _number(mapping.get(key))
    if value is None or value <= 0.0:
        errors.append(f"{prefix}.{key} must be > 0")


def _nonnegative_number(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = _number(mapping.get(key))
    if value is None or value < 0.0:
        errors.append(f"{prefix}.{key} must be >= 0")


def _unit_interval(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = _number(mapping.get(key))
    if value is None or not 0.0 <= value <= 1.0:
        errors.append(f"{prefix}.{key} must be in [0, 1]")


def _nonempty_unique_strings(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = mapping.get(key)
    normalized = [str(item).strip() for item in value] if isinstance(value, list) else []
    if not normalized or any(not item or item.upper().startswith("REPLACE") for item in normalized):
        errors.append(f"{prefix}.{key} must be a non-empty list of resolved strings")
    elif len(set(normalized)) != len(normalized):
        errors.append(f"{prefix}.{key} must not contain duplicates")


def _nonempty_unique_integers(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = mapping.get(key)
    integers = [_integer(item) for item in value] if isinstance(value, list) else []
    if not integers or any(item is None for item in integers):
        errors.append(f"{prefix}.{key} must be a non-empty integer list")
    elif len(set(integers)) != len(integers):
        errors.append(f"{prefix}.{key} must not contain duplicates")


def _positive_integer_list(
    value: Any,
    errors: list[str],
    *,
    label: str,
) -> list[int]:
    integers = [_integer(item) for item in value] if isinstance(value, list) else []
    if not integers or any(item is None or item < 1 for item in integers):
        errors.append(f"{label} must be a non-empty list of positive integers")
        return []
    result = [int(item) for item in integers if item is not None]
    if len(set(result)) != len(result):
        errors.append(f"{label} must not contain duplicates")
    return result


def _sha256_field(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str,
) -> None:
    value = str(mapping.get(key) or "").strip().lower()
    if not _is_sha256(value):
        errors.append(f"{prefix}.{key} must be a resolved SHA256 hex digest")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _contains_credential_material(value: Any, *, key: str = "") -> bool:
    """Reject credential-bearing structured config without exposing values."""

    normalized_key = key.lower().replace("-", "_")
    if any(
        fragment in normalized_key
        for fragment in ("authorization", "api_key", "access_token", "password", "secret")
    ):
        return True
    if isinstance(value, dict):
        return any(
            _contains_credential_material(item, key=str(item_key))
            for item_key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_credential_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return "authorization:" in lowered or "bearer " in lowered
    return False
