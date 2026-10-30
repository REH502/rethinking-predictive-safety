import json
from pathlib import Path

import pytest
import yaml

from experiments.model_stack import (
    MODEL_STACK_ROLES,
    load_model_stack_from_environment,
    load_model_stack_profile,
)


def _write_profile(path: Path, *, actor_endpoint: str = "remote") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "model-stack-profile-v1",
                "defaults": {
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "timeout_seconds": 180,
                    "transport_max_attempts": 2,
                    "request_extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                },
                "endpoints": {
                    "remote": {
                        "base_url": "https://relay.example.test/v1",
                        "api_key_env": "TEST_RELAY_API_KEY",
                        "api_mode": "chat_completions",
                    },
                    "local": {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "auth": "none",
                    },
                },
                "roles": {
                    "actor": {
                        "endpoint": actor_endpoint,
                        "model": "actor-model",
                        "revision": "actor-rev-1",
                        "runtime_revision": "runtime-image@sha256:actor",
                        "temperature": 0.2,
                        "max_tokens": 4096,
                        "timeout_seconds": 300,
                        "transport_max_attempts": 1,
                    },
                    "judge": {"endpoint": "remote", "model": "judge-model"},
                    "world_model": {"endpoint": "local", "model": "wm-model"},
                    "policy_extractor": {
                        "endpoint": "local",
                        "model": "policy-model",
                    },
                    "action_agent": {
                        "endpoint": "local",
                        "model": "action-model",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_model_stack_resolves_all_roles_without_persisting_key_values(tmp_path):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)

    stack = load_model_stack_profile(
        profile_path,
        environ={"TEST_RELAY_API_KEY": "super-secret-value"},
    )

    assert tuple(stack.roles) == MODEL_STACK_ROLES
    assert stack.roles["actor"].model == "actor-model"
    assert stack.roles["actor"].temperature == 0.2
    assert stack.roles["actor"].runtime_revision == (
        "runtime-image@sha256:actor"
    )
    assert stack.roles["judge"].runtime_revision == "unversioned-runtime"
    assert stack.roles["actor"].transport_max_attempts == 1
    assert stack.roles["judge"].transport_max_attempts == 2
    assert stack.roles["actor"].request_extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert stack.roles["world_model"].base_url == "http://127.0.0.1:8000/v1"
    assert stack.roles["world_model"].api_key == "local"
    public_json = json.dumps(stack.public_identity, sort_keys=True)
    assert "super-secret-value" not in public_json
    assert stack.public_identity["roles"]["actor"]["api_key_env"] == (
        "TEST_RELAY_API_KEY"
    )
    assert stack.public_identity["roles"]["world_model"]["auth"] == "none"
    assert stack.public_identity["roles"]["actor"]["api_mode"] == (
        "chat_completions"
    )
    assert stack.public_identity["source_sha256"]

    runtime_env = stack.runtime_environment()
    assert runtime_env["WM_SAFETY_ACTOR_API_KEY"] == "super-secret-value"
    assert runtime_env["WM_SAFETY_WORLD_MODEL_API_KEY"] == "local"
    assert runtime_env["WM_SAFETY_POLICY_EXTRACTOR_MODEL"] == "policy-model"
    assert runtime_env["WM_SAFETY_ACTOR_API_MODE"] == "chat_completions"
    assert runtime_env["WM_SAFETY_ACTOR_TRANSPORT_MAX_ATTEMPTS"] == "1"
    assert runtime_env["WM_SAFETY_ACTOR_RUNTIME_REVISION"] == (
        "runtime-image@sha256:actor"
    )
    assert runtime_env["WM_SAFETY_JUDGE_TRANSPORT_MAX_ATTEMPTS"] == "2"
    assert json.loads(runtime_env["WM_SAFETY_ACTOR_REQUEST_EXTRA_BODY_JSON"]) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "super-secret-value" not in runtime_env["WM_SAFETY_MODEL_STACK_PUBLIC_JSON"]


def test_model_stack_binds_actor_sampling_controls_as_public_runtime_identity(
    tmp_path,
):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    payload["roles"]["actor"].update({"top_p": 0.95, "top_k": 20})
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    stack = load_model_stack_profile(
        profile_path,
        environ={"TEST_RELAY_API_KEY": "super-secret-value"},
    )

    actor = stack.roles["actor"]
    assert actor.top_p == 0.95
    assert actor.top_k == 20
    assert stack.public_identity["roles"]["actor"]["top_p"] == 0.95
    assert stack.public_identity["roles"]["actor"]["top_k"] == 20
    runtime_env = stack.runtime_environment()
    assert runtime_env["WM_SAFETY_ACTOR_TOP_P"] == "0.95"
    assert runtime_env["WM_SAFETY_ACTOR_TOP_K"] == "20"
    assert "WM_SAFETY_JUDGE_TOP_P" not in runtime_env
    assert "super-secret-value" not in json.dumps(
        stack.public_identity, sort_keys=True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["roles"]["actor"].update({"top_p": 0}),
            "top_p",
        ),
        (
            lambda payload: payload["roles"]["actor"].update({"top_k": -1}),
            "top_k",
        ),
        (
            lambda payload: payload["roles"]["judge"].update(
                {"top_p": 0.95, "top_k": 20}
            ),
            "only supported for roles.actor",
        ),
        (
            lambda payload: (
                payload["roles"]["actor"].update({"top_p": 0.95, "top_k": 20}),
                payload["endpoints"]["remote"].update({"api_mode": "responses"}),
            ),
            "require a chat_completions endpoint",
        ),
        (
            lambda payload: payload["roles"]["actor"].update(
                {"request_extra_body": {"top_p": 0.95}}
            ),
            "reserved",
        ),
    ],
)
def test_model_stack_rejects_unbound_or_opaque_sampling_controls(
    tmp_path,
    mutation,
    message,
):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    mutation(payload)
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model_stack_profile(
            profile_path,
            environ={"TEST_RELAY_API_KEY": "super-secret-value"},
        )


def test_model_stack_accepts_standard_top_p_without_provider_specific_top_k(
    tmp_path,
):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    payload["roles"]["actor"].update({"top_p": 0.95})
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    stack = load_model_stack_profile(
        profile_path,
        environ={"TEST_RELAY_API_KEY": "super-secret-value"},
    )

    actor = stack.roles["actor"]
    assert actor.top_p == 0.95
    assert actor.top_k is None
    assert stack.runtime_environment()["WM_SAFETY_ACTOR_TOP_P"] == "0.95"
    assert "WM_SAFETY_ACTOR_TOP_K" not in stack.runtime_environment()


def test_model_stack_accepts_base_url_from_environment(tmp_path):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    payload["endpoints"]["remote"] = {
        "base_url_env": "TEST_RELAY_BASE_URL",
        "api_key_env": "TEST_RELAY_API_KEY",
    }
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    stack = load_model_stack_profile(
        profile_path,
        environ={
            "TEST_RELAY_API_KEY": "secret",
            "TEST_RELAY_BASE_URL": "https://official.example.test/v1/",
        },
    )

    assert stack.roles["actor"].base_url == "https://official.example.test/v1"
    assert stack.public_identity["roles"]["actor"]["base_url_env"] == (
        "TEST_RELAY_BASE_URL"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update({"api_key": "literal-secret"}), "unknown field"),
        (lambda p: p["roles"].pop("judge"), "missing roles"),
        (
            lambda p: p["endpoints"]["remote"].update(
                {"base_url": "https://user:pw@relay.example/v1?key=x"}
            ),
            "userinfo, query, or fragment",
        ),
        (
            lambda p: p["roles"]["actor"].update({"temperature": -0.1}),
            "temperature",
        ),
        (
            lambda p: p["roles"]["actor"].update(
                {"request_extra_body": {"authorization": "secret"}}
            ),
            "credential-bearing",
        ),
        (
            lambda p: p["roles"]["actor"].update(
                {"request_extra_body": {"seed": 7}}
            ),
            "reserved",
        ),
    ],
)
def test_model_stack_rejects_ambiguous_or_secret_bearing_profiles(
    tmp_path,
    mutation,
    message,
):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    mutation(payload)
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model_stack_profile(
            profile_path,
            environ={"TEST_RELAY_API_KEY": "secret"},
        )


def test_model_stack_requires_selected_key_environment_variable(tmp_path):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)

    with pytest.raises(ValueError, match="TEST_RELAY_API_KEY"):
        load_model_stack_profile(profile_path, environ={})


def test_model_stack_structural_inspection_allows_missing_key_without_fabricating_one(
    tmp_path,
):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)

    stack = load_model_stack_profile(
        profile_path,
        environ={},
        require_credentials=False,
    )

    assert stack.roles["actor"].api_key == ""
    assert stack.roles["actor"].api_key_env == "TEST_RELAY_API_KEY"
    assert stack.runtime_environment()["WM_SAFETY_ACTOR_API_KEY"] == ""
    assert "TEST_RELAY_API_KEY" in json.dumps(stack.public_identity)


def test_model_stack_environment_identity_is_revalidated(tmp_path):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    stack = load_model_stack_profile(
        profile_path,
        environ={"TEST_RELAY_API_KEY": "secret"},
    )
    env = {
        "TEST_RELAY_API_KEY": "secret",
        **stack.runtime_environment(),
    }

    reloaded = load_model_stack_from_environment(env)
    assert reloaded is not None
    assert reloaded.source_sha256 == stack.source_sha256

    public = json.loads(env["WM_SAFETY_MODEL_STACK_PUBLIC_JSON"])
    public["roles"]["actor"]["model"] = "tampered"
    env["WM_SAFETY_MODEL_STACK_PUBLIC_JSON"] = json.dumps(public)
    with pytest.raises(ValueError, match="public identity"):
        load_model_stack_from_environment(env)


def test_model_stack_environment_revalidation_supports_structural_dry_run(tmp_path):
    profile_path = tmp_path / "models.yaml"
    _write_profile(profile_path)
    stack = load_model_stack_profile(
        profile_path,
        environ={},
        require_credentials=False,
    )

    reloaded = load_model_stack_from_environment(
        stack.runtime_environment(),
        require_credentials=False,
    )

    assert reloaded is not None
    assert reloaded.public_identity == stack.public_identity
    assert reloaded.roles["actor"].api_key == ""








def test_ha_online_algorithm_config_contains_no_embedded_model_identity():
    project_root = Path(__file__).resolve().parents[2]
    config_path = (
        project_root
        / "SafePred"
        / "config"
        / "harnessaudit_online_openai_compatible.yaml"
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for section in (
        "world_model_llm",
        "rule_extractor_llm",
        "action_agent_llm",
    ):
        assert "model_name" not in payload[section]
        assert payload[section]["provider"] == "custom"






