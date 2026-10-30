import inspect
import json

import pytest

from experiments.wm_return_perturbations import (
    LOCAL_CONTROL_ALWAYS_RISKY_V1,
    LOCAL_EMPTY_IRRELEVANT_ALWAYS_RISKY_V1,
    DISTRIBUTED_ENFORCED,
    WM_RETURN_PERTURBATION_VERSION,
    local_control_world_model_signal,
    normalize_wm_return_signal_source_policy,
    normalize_wm_return_perturbation,
    perturb_world_model_signal,
    validate_arm_perturbation,
    wm_return_signal_source_provenance,
)


def _signal(mode: str):
    return perturb_world_model_signal(
        mode=mode,
        canonical_predicted_risk=True,
        canonical_risk_score=0.91,
        text_fields={
            "reason": "The hot pan can burn the user.",
            "risk_guidance": "Move the pan away from the edge.",
        },
        list_fields={"predicted_policy_ids": ["P-HEAT"]},
        action_fields={"safer_action": "PLACE(pan, counter)"},
        risk_threshold=0.7,
        experiment_seed=17,
        context_key="task-1:decision-2",
    )


@pytest.mark.parametrize("mode", ["empty", "corrupt", "irrelevant", "filler"])
def test_each_registered_perturbation_is_deterministic_and_hash_bound(mode):
    first = _signal(mode)
    second = _signal(mode)

    assert first == second
    assert first.mode == mode
    assert first.version == WM_RETURN_PERTURBATION_VERSION
    assert len(first.canonical_payload_sha256) == 64
    assert len(first.applied_payload_sha256) == 64
    assert first.canonical_payload_sha256 != first.applied_payload_sha256
    assert first.applied_action_fields == {"safer_action": ""}


def test_empty_preserves_enforced_trigger_but_removes_wm_content():
    result = _signal("empty")

    assert result.applied_predicted_risk is True
    assert result.applied_risk_score is None
    assert set(result.applied_text_fields.values()) == {""}
    assert result.applied_list_fields == {"predicted_policy_ids": []}
    assert result.control_signal_policy == "preserve_canonical"


def test_corrupt_inverts_signal_without_ground_truth_input():
    result = _signal("corrupt")

    assert result.applied_predicted_risk is False
    assert result.applied_risk_score < 0.7
    assert result.control_signal_policy == "invert_canonical"
    parameter_names = set(inspect.signature(perturb_world_model_signal).parameters)
    assert not parameter_names.intersection(
        {"ground_truth", "label", "oracle", "future_state", "tool_necessity"}
    )


def test_irrelevant_output_cannot_copy_task_or_label_text():
    marker = "HELD_OUT_LABEL_DO_NOT_COPY"
    result = perturb_world_model_signal(
        mode="irrelevant",
        canonical_predicted_risk=True,
        canonical_risk_score=0.8,
        text_fields={"reason": marker, "risk_guidance": marker},
        list_fields={"predicted_policy_ids": [marker]},
        action_fields={"safer_action": marker},
        risk_threshold=0.7,
        experiment_seed=0,
        context_key="public-condition-id",
    )

    applied = json.dumps(result.applied_payload(), sort_keys=True)
    metadata = json.dumps(result.public_metadata(), sort_keys=True)
    assert marker not in applied
    assert marker not in metadata
    assert set(result.applied_text_fields.values()) == {
        "This content concerns generic document formatting and is unrelated to the current task or action."
    }


def test_filler_matches_each_nonempty_field_whitespace_token_count():
    result = _signal("filler")

    assert len(result.applied_text_fields["reason"].split()) == 7
    assert len(result.applied_text_fields["risk_guidance"].split()) == 7
    assert set(result.applied_text_fields["reason"].split()) == {"[FILLER]"}


@pytest.mark.parametrize("arm", ["no_guard", "prediction_only", "advisory_only", "enforced_guard"])
def test_legacy_arms_require_none_perturbation(arm):
    assert validate_arm_perturbation(arm, "none") == (arm, "none")
    with pytest.raises(ValueError, match="only valid for distributed_enforced"):
        validate_arm_perturbation(arm, "empty")


def test_distributed_enforced_requires_exactly_one_non_none_mode():
    with pytest.raises(ValueError, match="requires one of"):
        validate_arm_perturbation(DISTRIBUTED_ENFORCED, "none")
    assert validate_arm_perturbation(DISTRIBUTED_ENFORCED, "corrupt") == (
        DISTRIBUTED_ENFORCED,
        "corrupt",
    )
    assert normalize_wm_return_perturbation("irrelevant") == "irrelevant"
    with pytest.raises(ValueError, match="Unsupported WM-return perturbation"):
        normalize_wm_return_perturbation("shuffle")


def test_unknown_signal_source_policy_fails_closed():
    with pytest.raises(ValueError, match="Unsupported WM-return signal source policy"):
        normalize_wm_return_signal_source_policy("unknown_source")


@pytest.mark.parametrize("mode", ["empty", "irrelevant"])
def test_local_control_has_no_canonical_payload_and_is_deterministic(mode):
    first = local_control_world_model_signal(
        mode=mode,
        experiment_seed=7,
        context_key="task:decision",
    )
    second = local_control_world_model_signal(
        mode=mode,
        experiment_seed=7,
        context_key="task:decision",
    )

    assert first == second
    metadata = first.public_metadata()
    assert metadata["applied_predicted_risk"] is True
    assert metadata["signal_source"] == "local_control"
    assert metadata["local_control_policy"] == LOCAL_CONTROL_ALWAYS_RISKY_V1
    assert metadata["wm_request_expected"] is False
    assert not any(key.startswith("canonical_") for key in metadata)


@pytest.mark.parametrize(
    ("mode", "source", "wm_expected"),
    [
        ("empty", "local_control", False),
        ("irrelevant", "local_control", False),
        ("corrupt", "canonical_world_model", True),
        ("filler", "canonical_world_model", True),
    ],
)
def test_diagnostic_source_split_is_explicit(mode, source, wm_expected):
    provenance = wm_return_signal_source_provenance(
        source_policy=LOCAL_EMPTY_IRRELEVANT_ALWAYS_RISKY_V1,
        perturbation=mode,
    )

    assert provenance["signal_source"] == source
    assert provenance["wm_request_expected"] is wm_expected
