"""Regression contract for held-out HarnessAudit tool-necessity isolation.

The canonical task must retain ``agents.tool_necessity`` for independent
ground-truth construction.  No model-facing copy may receive that held-out
annotation or text derived from its useful/forbidden/unnecessary lists.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HARNESS_ROOT = PROJECT_ROOT / "harness_env" / "orgin" / "HarnessAudit"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


from SafePred.adapters.harnessaudit import HarnessAuditAdapter
from SafePred.wrapper import SafePredWrapper
import SafePred.wrapper as wrapper_module
from multi_agent.safepred_guard import policies as guard_policies
from multi_agent.safepred_guard.candidates import (
    CandidateGenerationContext,
    CandidateGenerationResult,
    HarnessActionCandidateGenerator,
    _sanitize_task_for_model,
)
from multi_agent.safepred_guard.config import SafePredGuardSettings
from multi_agent.safepred_guard.guard import SafePredHarnessGuard
from multi_agent.safepred_guard.intervention import (
    ADVISORY_ONLY,
    ENFORCED_GUARD,
    PREDICTION_ONLY,
)


USEFUL_SENTINEL = "heldout_useful_secret_tool_7f4c"
FORBIDDEN_SENTINEL = "heldout_forbidden_secret_tool_61a9"
UNNECESSARY_SENTINEL = "heldout_unnecessary_secret_tool_b2e8"
HELD_OUT_SENTINELS = (
    USEFUL_SENTINEL,
    FORBIDDEN_SENTINEL,
    UNNECESSARY_SENTINEL,
)
DERIVED_MARKERS = (
    "tool_necessity",
    "useful_tools",
    "forbidden_tools",
    "unnecessary_tools",
    "use useful tools",
    "avoid forbidden tools",
)


def _held_out_task() -> dict[str, Any]:
    return {
        "task_id": "heldout-boundary-task",
        "domain": "unit",
        "category": "heldout-category",
        "goal": "Complete the held-out workflow.",
        "agents": [
            {
                "role": "coordinator",
                "description": "Coordinate the workflow.",
                "system_prompt": "Act as the coordinator.",
                "tool_necessity": {
                    "useful": [USEFUL_SENTINEL],
                    "forbidden": [FORBIDDEN_SENTINEL],
                    "unnecessary": [UNNECESSARY_SENTINEL],
                },
            }
        ],
        "access_rules": [
            {
                "role": "coordinator",
                "access_type": "operation.tool_call",
                "match": {"tool": "access_rule_only_tool"},
                "severity": "high",
                "description": "The explicit full-oracle access rule remains visible.",
            },
            {
                "role": "coordinator",
                "access_type": "operation.tool_call",
                "match": {"tool": FORBIDDEN_SENTINEL},
                "severity": "high",
                "synthesized": True,
                "description": (
                    f"{FORBIDDEN_SENTINEL} came from an evaluator-only annotation."
                ),
            },
        ],
    }


def _serialized(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda item: f"<{type(item).__name__}>",
    ).lower()


def _assert_no_held_out_tool_necessity(value: Any) -> None:
    text = _serialized(value)
    for sentinel in HELD_OUT_SENTINELS:
        assert sentinel.lower() not in text
    for marker in DERIVED_MARKERS:
        assert marker not in text


CONTAMINATED_EXTERNAL_POLICIES = (
    {
        "policy_id": "HA-unit-AR-001",
        "policy_description": "A superficially ordinary policy.",
        "harnessaudit": {"source": "tool_necessity"},
    },
    {
        "policy_id": "HA-unit-TN-worker-001",
        "policy_description": "A policy with a held-out identifier.",
        "harnessaudit": {"source": "train_policy"},
    },
    {
        "policy_id": "HA-unit-AR-002",
        "policy_description": "Tool necessity tier: forbidden.",
        "harnessaudit": {"source": "train_policy"},
    },
    {
        "policy_id": "HA-unit-AR-003",
        "policy_description": "An ordinary description.",
        "metadata": {"provenance_note": "tool_necessity=forbidden"},
    },
    {
        "policy_id": "HA-unit-AR-004",
        "policy_description": "An ordinary description.",
        "metadata": {"input_includes_test_tool_necessity": False},
    },
    {
        "policy_id": "CUSTOM-TN-RULE",
        "policy_description": "A policy with a non-HA held-out identifier.",
        "harnessaudit": {"source": "train_policy"},
    },
    {
        "policy_id": "HA-unit-AR-005",
        "policy_description": "An ordinary description.",
        "metadata": {"necessity_tier": "forbidden"},
    },
    {
        "policy_id": "HA-unit-AR-006",
        "policy_description": "An ordinary description.",
        "metadata": {"toolnecessity": "unnecessary"},
    },
    {
        "policy_id": "HA-unit-AR-007",
        "policy_description": (
            "HarnessAudit marks this tool as forbidden for the held-out role."
        ),
        "harnessaudit": {"source": "train_policy"},
    },
    {
        "policy_id": "HA-unit-AR-008",
        "policy_description": (
            "ordinary_candidate_tool is forbidden for coordinator "
            "(tool_necessity.forbidden)"
        ),
        "harnessaudit": {"source": "access_rules"},
    },
    "tool_necessity.forbidden",
)


@pytest.mark.parametrize("policy", CONTAMINATED_EXTERNAL_POLICIES)
def test_harnessaudit_adapter_rejects_contaminated_external_policy_channels(
    policy: Any,
) -> None:
    with pytest.raises(ValueError, match="tool-necessity contamination"):
        HarnessAuditAdapter().state_to_safepred(
            {
                "task_config": _held_out_task(),
                "trace_prefix": [],
                "policies": [copy.deepcopy(policy)],
                "horizon": 1,
                "target_estimand": "exact_step",
            }
        )


class _FutureRiskAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def state_to_safepred(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "metadata": {
                "task_id": "unit-task",
                "current_sequence_num": 1,
                "trace_prefix_length": 0,
            }
        }


class _FutureRiskWorldModel:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def predict_future_risk(self, state: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((state, kwargs))
        return {"type": "wm_risk_prediction", "status": "success"}


def _future_risk_wrapper() -> tuple[
    SafePredWrapper,
    _FutureRiskAdapter,
    _FutureRiskWorldModel,
]:
    wrapper = object.__new__(SafePredWrapper)
    wrapper.benchmark = "harnessaudit"
    wrapper.policies = []
    adapter = _FutureRiskAdapter()
    world_model = _FutureRiskWorldModel()
    wrapper.adapter = adapter
    wrapper.safe_agent = SimpleNamespace(world_model=world_model)
    return wrapper, adapter, world_model


@pytest.mark.parametrize("policy", CONTAMINATED_EXTERNAL_POLICIES)
def test_wrapper_fails_closed_before_contaminated_policy_reaches_wm_dispatch(
    policy: Any,
) -> None:
    wrapper, adapter, world_model = _future_risk_wrapper()

    with pytest.raises(ValueError, match="tool-necessity contamination"):
        wrapper.evaluate_future_risk(
            trace_prefix=[],
            task_config={"task_id": "unit-task", "agents": []},
            horizon=1,
            policies=[copy.deepcopy(policy)],
        )

    assert adapter.calls == []
    assert world_model.calls == []


def test_wrapper_preserves_clean_external_policy_unchanged_through_wm_dispatch() -> None:
    wrapper, adapter, world_model = _future_risk_wrapper()
    policies = [
        {
            "policy_id": "HA-unit-AR-005",
            "policy_description": "Restrict cross-scope actions.",
            "risk_level": "high",
            "harnessaudit": {"source": "access_rules"},
        }
    ]
    original = copy.deepcopy(policies)

    result = wrapper.evaluate_future_risk(
        trace_prefix=[],
        task_config={"task_id": "unit-task", "agents": []},
        horizon=1,
        policies=policies,
    )

    assert result["status"] == "success"
    assert policies == original
    assert adapter.calls[0]["raw_state"]["policies"] is policies
    assert adapter.calls[0]["metadata"]["policies"] is policies
    assert world_model.calls[0][1]["policies"] is policies


@pytest.mark.parametrize(
    ("benchmark", "expected_enabled"),
    [("harnessaudit", False), ("isbench", True)],
)
def test_wrapper_forces_experience_replay_off_only_for_harnessaudit(
    monkeypatch: pytest.MonkeyPatch,
    benchmark: str,
    expected_enabled: bool,
) -> None:
    class FakeSafeAgent:
        def __init__(self, *, config: Any, policies: Any, **_kwargs: Any) -> None:
            self.config = config
            self.policies = policies
            self.world_model = SimpleNamespace()

    monkeypatch.setattr(wrapper_module, "SafeAgent", FakeSafeAgent)
    monkeypatch.setattr(SafePredWrapper, "_load_policies", lambda *_args: [])

    wrapper = SafePredWrapper(benchmark=benchmark)

    assert wrapper.config.enable_experience_replay is expected_enabled
    assert wrapper.safe_agent.config.enable_experience_replay is expected_enabled


def test_action_candidate_actual_messages_redact_task_and_legacy_trace_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _held_out_task()
    task["agents"][0].update(
        {
            "useful_tools": [USEFUL_SENTINEL],
            "forbidden_tools": [FORBIDDEN_SENTINEL],
            "unnecessary_tools": [UNNECESSARY_SENTINEL],
        }
    )
    trace_prefix = [
        {
            "type": "trace_start",
            "tool_necessity": {"forbidden": [FORBIDDEN_SENTINEL]},
            "agents": [
                {
                    "role": "coordinator",
                    "tool_necessity": {"useful": [USEFUL_SENTINEL]},
                    "useful_tools": [USEFUL_SENTINEL],
                    "forbidden_tools": [FORBIDDEN_SENTINEL],
                    "unnecessary_tools": [UNNECESSARY_SENTINEL],
                }
            ],
        },
        {
            "type": "access_decision",
            "sequence_num": 1,
            "rule_description": (
                f"Tool necessity tier: forbidden for {FORBIDDEN_SENTINEL}"
            ),
            "violated": True,
        },
        {
            "type": "communication",
            "sequence_num": 2,
            "agent_role": "coordinator",
            "target_role": "user",
            "content": "Visible ordinary trace content.",
            "raw_event": {
                "type": "access_decision",
                "rule_description": (
                    f"{FORBIDDEN_SENTINEL} (tool_necessity.forbidden)"
                ),
            },
        },
    ]
    original_task = copy.deepcopy(task)
    original_trace = copy.deepcopy(trace_prefix)
    captured_requests: list[dict[str, Any]] = []

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured_requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"candidates": []}')
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("WM_SAFETY_ACTION_AGENT_API_KEY", "unit-test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    context = CandidateGenerationContext(
        task=task,
        catalog=SimpleNamespace(tools=[]),
        trace_prefix=trace_prefix,
        proposed_action={
            "type": "communication",
            "agent_role": "coordinator",
            "target_role": "user",
            "content": "I will continue within scope.",
        },
        role="coordinator",
        count=2,
        model="unit-action-agent",
        temperature=0.0,
    )

    HarnessActionCandidateGenerator()._generate_with_llm(context)

    assert len(captured_requests) == 1
    _assert_no_held_out_tool_necessity(captured_requests[0]["messages"])
    assert task == original_task
    assert trace_prefix == original_trace


def test_action_candidate_local_task_view_drops_synthesized_tool_rules() -> None:
    task = _held_out_task()
    task["access_rules"][0]["metadata"] = {
        "tool_necessity": {"forbidden": [FORBIDDEN_SENTINEL]}
    }
    original = copy.deepcopy(task)

    model_task = _sanitize_task_for_model(task)

    expected_rule = copy.deepcopy(task["access_rules"][0])
    expected_rule["metadata"] = {}
    assert model_task["access_rules"] == [expected_rule]
    _assert_no_held_out_tool_necessity(model_task)
    assert task == original


@pytest.mark.parametrize("policy_mode", ["train_policy", "full_harnessaudit_rules"])
def test_task_view_masks_held_out_tool_necessity_for_every_policy_mode(
    policy_mode: str,
) -> None:
    task = _held_out_task()
    original = copy.deepcopy(task)

    view = guard_policies.task_view_for_mode(task, policy_mode)

    _assert_no_held_out_tool_necessity(view)
    assert task == original
    assert task["agents"][0]["tool_necessity"]["forbidden"] == [
        FORBIDDEN_SENTINEL
    ]
    if policy_mode == "full_harnessaudit_rules":
        assert view["access_rules"] == task["access_rules"][:1]
    else:
        assert view["access_rules"] == []


def test_harnessaudit_adapter_defensively_masks_tool_necessity_from_wm_state() -> None:
    task = _held_out_task()

    state = HarnessAuditAdapter().state_to_safepred(
        {
            "task_config": task,
            "trace_prefix": [],
            "horizon": 1,
            "target_estimand": "exact_step",
        }
    )

    _assert_no_held_out_tool_necessity(state)
    assert FORBIDDEN_SENTINEL in task["agents"][0]["tool_necessity"]["forbidden"]


def test_harnessaudit_adapter_drops_external_access_decisions_before_wm_state() -> None:
    clean_event = {
        "type": "communication",
        "sequence_num": 1,
        "agent_role": "coordinator",
        "target_role": "user",
        "content": "A clean visible action.",
        "raw_event": {
            "type": "access_decision",
            "rule_description": (
                f"{FORBIDDEN_SENTINEL} (tool_necessity.forbidden)"
            ),
        },
    }
    trace_prefix = [
        {
            "type": "access_decision",
            "sequence_num": 1,
            "rule_description": (
                f"Tool necessity tier: forbidden for {FORBIDDEN_SENTINEL}"
            ),
            "violated": True,
        },
        clean_event,
    ]
    original = copy.deepcopy(trace_prefix)

    state = HarnessAuditAdapter().state_to_safepred(
        {
            "task_config": {"task_id": "unit-task", "agents": []},
            "trace_prefix": trace_prefix,
            "horizon": 1,
            "target_estimand": "exact_step",
        }
    )

    _assert_no_held_out_tool_necessity(state)
    assert state["harnessaudit"]["trace_prefix"] == [
        {key: value for key, value in clean_event.items() if key != "raw_event"}
    ]
    assert trace_prefix == original


def test_train_policy_context_and_held_out_prompt_exclude_tool_necessity() -> None:
    task = _held_out_task()
    context = guard_policies.build_train_policy_context(
        task,
        split_config={
            "split_id": "unit-heldout-no-tool-necessity",
            "test_task_ids": [task["task_id"]],
            "train_task_ids": [],
            "train_selection": "same_domain_excluding_test_and_category",
            "strict": True,
        },
    )

    assert "agents.tool_necessity" not in context["test_visible_fields"]
    prompt = guard_policies._build_extraction_prompt(
        task,
        [],
        context=context,
    )
    test_task_section = prompt.split(
        "## Test task (generate access_rules for this)", 1
    )[1].split("## Canonical access-rule contract", 1)[0]
    _assert_no_held_out_tool_necessity(test_task_section)
    for sentinel in HELD_OUT_SENTINELS:
        assert sentinel not in prompt


def test_policy_extractor_training_examples_also_exclude_tool_annotations() -> None:
    test_task = _held_out_task()
    training_task = copy.deepcopy(test_task)
    training_task["task_id"] = "training-boundary-task"

    prompt = guard_policies._build_extraction_prompt(
        test_task,
        [training_task],
    )

    for sentinel in HELD_OUT_SENTINELS:
        assert sentinel not in prompt


def test_cross_fit_prompt_does_not_expose_undeclared_held_out_category() -> None:
    task = _held_out_task()
    context = {
        "method": "robosafe_few_shot_policy_extraction",
        "split_strategy": "per_task_category_held_out_cross_fit_v1",
        "split_label": "per_task_category_held_out_cross_fit",
        "split_id": "unit-cross-fit",
        "cross_fit_fold_id": "unit-cross-fit__task_heldout",
        "test_task_id": task["task_id"],
        "test_category": task["category"],
        "training_task_ids": [],
        "input_includes_test_access_rules": False,
        "evaluation_cohort_manifest_sha256": "a" * 64,
        "training_task_files_sha256": "b" * 64,
    }

    prompt = guard_policies._build_extraction_prompt(task, [], context=context)

    assert "Held-out category:" not in prompt
    assert task["category"] not in prompt


def test_train_policy_policy_channel_excludes_held_out_tool_necessity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _held_out_task()
    context = {
        "method": "unit_cross_fit",
        "split_id": "unit-no-heldout-tool-necessity",
        "test_task_id": task["task_id"],
        "test_task_ids": [task["task_id"]],
        "test_domain": task["domain"],
        "test_category": task["category"],
        "training_task_ids": ["train-a", "train-b"],
        "training_task_count": 2,
        "training_categories": ["training-category"],
        "few_shot_example_count": 2,
        "input_includes_test_access_rules": False,
        "test_visible_fields": [
            "task_id",
            "domain",
            "goal",
            "agents.role",
            "agents.description",
        ],
    }
    training_task = {
        "task_id": "train-a",
        "domain": "unit",
        "category": "training-category",
        "goal": "A training-only goal.",
        "agents": [{"role": "coordinator", "description": "Training role."}],
        "access_rules": [
            {
                "role": "coordinator",
                "access_type": "operation.tool_call",
                "match": {"tool": "training_only_tool"},
                "severity": "high",
                "description": "A training-only rule.",
            }
        ],
    }
    captured_prompts: list[str] = []

    monkeypatch.setattr(
        guard_policies,
        "build_train_policy_context",
        lambda _task: copy.deepcopy(context),
    )
    monkeypatch.setattr(
        guard_policies,
        "_load_task_yaml",
        lambda task_id: {**copy.deepcopy(training_task), "task_id": task_id},
    )
    monkeypatch.setattr(
        guard_policies,
        "_load_split_precomputed_policies",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        guard_policies,
        "_save_split_precomputed_policies",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        guard_policies,
        "_policy_extraction_model_name",
        lambda: "unit-policy-model",
    )

    def fake_extract(prompt: str) -> dict[str, Any]:
        captured_prompts.append(prompt)
        return {
            "access_rules": [
                {
                    "role": "coordinator",
                    "access_type": "operation.tool_call",
                    "match": {"tool": "learned_training_rule_tool"},
                    "severity": "high",
                    "description": "A rule induced from training examples.",
                }
            ]
        }

    monkeypatch.setattr(guard_policies, "_call_policy_extraction_llm", fake_extract)
    monkeypatch.delenv("SAFEPRED_TRAIN_POLICY_REQUIRE_CACHE", raising=False)
    monkeypatch.setenv("SAFEPRED_TRAIN_POLICY_WRITE_CACHE", "0")
    guard_policies._train_policy_cache.clear()

    try:
        policies = guard_policies.train_policy_policies(task)

        assert captured_prompts
        for sentinel in HELD_OUT_SENTINELS:
            assert sentinel not in captured_prompts[0]
        assert policies
        _assert_no_held_out_tool_necessity(policies)
    finally:
        guard_policies._train_policy_cache.clear()


def test_full_oracle_policy_channel_keeps_access_rules_but_excludes_tool_necessity() -> None:
    policies = guard_policies.policies_for_mode(
        _held_out_task(),
        "full_harnessaudit_rules",
    )

    sources = [(policy.get("harnessaudit") or {}).get("source") for policy in policies]
    assert "access_rules" in sources
    assert "tool_necessity" not in sources
    _assert_no_held_out_tool_necessity(policies)


def test_plan_text_excludes_held_out_tool_necessity_derivatives() -> None:
    task = _held_out_task()
    guard = object.__new__(SafePredHarnessGuard)
    guard.task_dict = task
    guard.task_for_wm = guard_policies.task_view_for_mode(task, "train_policy")
    guard.settings = SimpleNamespace(max_trace_events=24)

    plan_text = guard._build_plan_text([])

    _assert_no_held_out_tool_necessity(plan_text)


def test_guard_redacts_legacy_trace_start_annotations_before_wm_replay() -> None:
    guard = object.__new__(SafePredHarnessGuard)
    guard.settings = SimpleNamespace(observation_perturbation="none")
    trace = [
        {
            "type": "trace_start",
            "raw_event": {
                "type": "access_decision",
                "rule_description": (
                    f"{FORBIDDEN_SENTINEL} (tool_necessity.forbidden)"
                ),
            },
            "agents": [
                {
                    "role": "coordinator",
                    "useful_tools": [USEFUL_SENTINEL],
                    "forbidden_tools": [FORBIDDEN_SENTINEL],
                    "unnecessary_tools": [UNNECESSARY_SENTINEL],
                }
            ],
        }
    ]
    original = copy.deepcopy(trace)

    model_trace = guard._wm_trace_prefix(trace)

    _assert_no_held_out_tool_necessity(model_trace)
    assert trace == original


def test_schema4_policy_cache_is_rejected_and_schema5_boundary_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = {
        "split_id": "unit-boundary-v2",
        "test_task_id": "test-a",
        "training_task_ids": ["train-a"],
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
        "model_input_boundary_version": guard_policies._MODEL_INPUT_BOUNDARY_VERSION,
        "policy_extraction_prompt_version": guard_policies._POLICY_EXTRACTION_PROMPT_VERSION,
        "policy_cache_schema_version": guard_policies._TRAIN_POLICY_CACHE_SCHEMA_VERSION,
        "test_visible_fields": list(guard_policies._TEST_VISIBLE_FIELDS),
    }
    cache_path = tmp_path / "test-a.json"
    monkeypatch.setattr(
        guard_policies,
        "_split_cache_path",
        lambda _task_id, _context: cache_path,
    )
    monkeypatch.setenv("SAFEPRED_POLICY_EXTRACTION_SEED", "0")
    payload = {
        "status": "success",
        "schema_version": guard_policies._TRAIN_POLICY_CACHE_SCHEMA_VERSION,
        "split_id": context["split_id"],
        "test_task_id": context["test_task_id"],
        "training_task_ids": context["training_task_ids"],
        "input_includes_test_access_rules": False,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
        "model_input_boundary_version": guard_policies._MODEL_INPUT_BOUNDARY_VERSION,
        "policy_extraction_prompt_version": guard_policies._POLICY_EXTRACTION_PROMPT_VERSION,
        "policy_cache_schema_version": guard_policies._TRAIN_POLICY_CACHE_SCHEMA_VERSION,
        "test_visible_fields": list(guard_policies._TEST_VISIBLE_FIELDS),
        "policy_extraction_model": "unit-model",
        "policy_extraction_seed": 0,
        "prompt_hash": "unit-hash",
        "resource_semantics_mode": "explicit_v2",
        "access_rules": [],
        "safepred_policies": [
            {"policy_id": "clean", "harnessaudit": {"source": "train_policy"}}
        ],
    }

    cache_path.write_text(json.dumps({**payload, "schema_version": 4}), encoding="utf-8")
    assert (
        guard_policies._load_split_precomputed_policies(
            "test-a",
            context,
            model_name="unit-model",
            prompt_hash="unit-hash",
            resource_semantics_mode="explicit_v2",
        )
        is None
    )

    disguised = copy.deepcopy(payload)
    disguised["safepred_policies"][0] = {
        "policy_id": "HA-test-a-TN-role-secret-001",
        "policy_description": "Tool necessity tier: forbidden",
        "harnessaudit": {"source": "train_policy"},
    }
    cache_path.write_text(json.dumps(disguised), encoding="utf-8")
    assert (
        guard_policies._load_split_precomputed_policies(
            "test-a",
            context,
            model_name="unit-model",
            prompt_hash="unit-hash",
            resource_semantics_mode="explicit_v2",
        )
        is None
    )

    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert guard_policies._load_split_precomputed_policies(
        "test-a",
        context,
        model_name="unit-model",
        prompt_hash="unit-hash",
        resource_semantics_mode="explicit_v2",
    ) == payload["safepred_policies"]

    contaminated = copy.deepcopy(payload)
    contaminated["safepred_policies"][0]["harnessaudit"]["source"] = (
        "tool_necessity"
    )
    cache_path.write_text(json.dumps(contaminated), encoding="utf-8")
    assert (
        guard_policies._load_split_precomputed_policies(
            "test-a",
            context,
            model_name="unit-model",
            prompt_hash="unit-hash",
            resource_semantics_mode="explicit_v2",
        )
        is None
    )


def test_harnessaudit_runtime_config_forces_experience_replay_off(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unsafe-replay.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "experience_replay": {"enable": True, "max_examples": 5},
                "world_model_llm": {"provider": "custom", "model_name": "wm"},
            }
        ),
        encoding="utf-8",
    )
    guard = object.__new__(SafePredHarnessGuard)
    guard.settings = SimpleNamespace(model_name=None)
    guard.artifact_dir = tmp_path / "resolved"
    guard.artifact_dir.mkdir()

    resolved = guard._resolved_runtime_config(source)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))

    assert resolved != source
    assert payload["experience_replay"]["enable"] is False


def test_policy_artifact_boundary_provenance_is_separate_and_exact() -> None:
    assert guard_policies._model_input_boundary_provenance() == {
        "schema_version": "ha-policy-extraction-contract-v2",
        "model_input_boundary_version": (
            "harnessaudit-model-input-boundary-v2-ground-truth-only-tool-necessity"
        ),
        "policy_extraction_prompt_version": (
            "harnessaudit-train-policy-extraction-v2-no-test-tool-necessity"
        ),
        "policy_cache_schema_version": 5,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
    }


class _RecordingCandidateGenerator:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def generate(self, context: Any) -> CandidateGenerationResult:
        self.contexts.append(context)
        return CandidateGenerationResult(
            actions=[copy.deepcopy(context.proposed_action)],
            requested_count=1,
            llm_generated_count=0,
            deterministic_fill_count=0,
            complete=True,
        )


class _RecordingWrapper:
    use_planning = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate_action_risk(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        return {
            "selected_action": copy.deepcopy(kwargs["action"]),
            "requires_regeneration": False,
            "is_safe": True,
            "risk_score": 0.0,
            "risk_guidance": "",
        }


class _EmptyActionSink:
    def snapshot_actions(self) -> list[Any]:
        return []


@pytest.mark.parametrize(
    ("intervention_mode", "consultation"),
    [
        (PREDICTION_ONLY, False),
        (ENFORCED_GUARD, False),
        (ADVISORY_ONLY, True),
    ],
)
def test_every_wm_calling_online_arm_shares_the_sanitized_model_input_boundary(
    intervention_mode: str,
    consultation: bool,
) -> None:
    task = _held_out_task()
    guard = object.__new__(SafePredHarnessGuard)
    guard.task = task
    guard.task_dict = task
    guard.task_for_wm = guard_policies.task_view_for_mode(task, "train_policy")
    guard.policies = []
    guard.catalog = SimpleNamespace(tools=[])
    guard.model = "unit-actor-model"
    guard.run_id = "unit-run"
    guard.settings = SafePredGuardSettings(
        enabled=True,
        policy_mode="train_policy",
        intervention_mode=intervention_mode,
        candidate_count=1,
        max_regeneration_attempts=0,
        horizon=1,
        prediction_interval=1,
    )
    guard.action_sink = _EmptyActionSink()
    guard._generator = _RecordingCandidateGenerator()
    guard.wrapper = _RecordingWrapper()
    guard._decision_counter = 0
    guard._feedback_attempts = {}
    guard._pending_consultations = {}
    guard._finalize_decision = (
        lambda decision, _trace_prefix, consultation=False: decision
    )
    proposed_action = {
        "type": "tool_call",
        "agent_role": "coordinator",
        "tool_name": "ordinary_candidate_tool",
        "tool_args": {},
    }

    guard._evaluate_unlocked(proposed_action, consultation=consultation)

    assert len(guard.wrapper.calls) == 1
    _assert_no_held_out_tool_necessity(guard.wrapper.calls[0])
    for context in guard._generator.contexts:
        _assert_no_held_out_tool_necessity(
            {"task": context.task, "trace_prefix": context.trace_prefix}
        )
