import base64
import io
import json
import urllib.error

import pytest

from experiments.endpoint_preflight import (
    _PREFLIGHT_IMAGE_DATA_URL,
    preflight_openai_compatible_model,
)


def test_preflight_image_fixture_is_strictly_decodable_png():
    from PIL import Image

    encoded = _PREFLIGHT_IMAGE_DATA_URL.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()

    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (2, 2)


class _Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = dict(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


@pytest.mark.parametrize("status", [520, 524])
def test_preflight_retries_identical_gateway_error_with_recorded_attempts(monkeypatch, tmp_path, status):
    from experiments.endpoint_preflight import _request_json
    ledger = tmp_path / 'attempts.jsonl'
    monkeypatch.setenv('ISBENCH_ENDPOINT_PREFLIGHT_MAX_ATTEMPTS', '3')
    monkeypatch.setenv('ISBENCH_ENDPOINT_PREFLIGHT_ATTEMPT_LEDGER', str(ledger))
    monkeypatch.setattr('experiments.endpoint_preflight.time.sleep', lambda _: None)
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        if len(requests) < 3:
            raise urllib.error.HTTPError(request.full_url, status, 'gateway error', {}, io.BytesIO(b''))
        return _Response({'ok': True})
    monkeypatch.setattr('experiments.endpoint_preflight.urllib.request.urlopen', open_request)
    data, _ = _request_json(url='http://relay/v1/chat/completions', api_key='local-dummy',
        timeout_seconds=10, method='POST', body={'model':'actor','messages':[]}, label='actor', stage='inference')
    assert data == {'ok': True}
    assert len(requests) == 3 and requests[0] is requests[1] is requests[2]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [x['transport_attempt'] for x in rows if x['status']=='failed'] == [1, 2]
    assert len({x['request_sha256'] for x in rows}) == 1


def test_preflight_requires_exact_model_and_real_chat_inference(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "qwen3.5-27b"}]})
        assert request.full_url.endswith("/chat/completions")
        body = json.loads(request.data)
        assert body["model"] == "qwen3.5-27b"
        return _Response(
            {"choices": [{"message": {"content": "OK", "tool_calls": []}}]}
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)

    record = preflight_openai_compatible_model(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3.5-27b",
        api_key="local-secret",
        api_mode="chat_completions",
        label="actor",
        request_extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )

    assert len(requests) == 2
    assert requests[0].headers["User-agent"] == "world-model-agent-safety-preflight/1.0"
    assert record["status"] == "passed"
    assert record["model_listing"] == "exact_match"
    assert record["inference_probe"] == "passed"
    assert record["request_extra_body_present"] is True
    assert "chat_template_kwargs" not in json.dumps(record, sort_keys=True)
    assert json.loads(requests[1].data)["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "local-secret" not in json.dumps(record, sort_keys=True)


def test_preflight_sends_frozen_sampling_controls_in_actor_probe(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "qwen3.5-27b"}]})
        return _Response(
            {"choices": [{"message": {"content": "OK", "tool_calls": []}}]}
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)

    record = preflight_openai_compatible_model(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3.5-27b",
        api_key="local-secret",
        api_mode="chat_completions",
        label="actor",
        temperature=0.2,
        top_p=0.95,
        top_k=20,
    )

    body = json.loads(requests[1].data)
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.95
    assert body["top_k"] == 20
    assert record["sampling_controls"] == {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_preflight_supports_standard_top_p_without_top_k(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "glm-5.3-flash"}]})
        return _Response(
            {"choices": [{"message": {"content": "OK", "tool_calls": []}}]}
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)

    record = preflight_openai_compatible_model(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-5.3-flash",
        api_key="local-secret",
        api_mode="chat_completions",
        label="actor",
        temperature=1.0,
        top_p=0.95,
    )

    body = json.loads(requests[1].data)
    assert body["top_p"] == 0.95
    assert "top_k" not in body
    assert record["sampling_controls"]["top_k"] is None


def test_preflight_tool_probe_requires_an_actual_tool_call(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "actor-model"}]})
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "preflight_healthcheck",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)

    record = preflight_openai_compatible_model(
        base_url="https://relay.example/v1",
        model="actor-model",
        api_key="secret",
        api_mode="chat_completions",
        label="HA actor",
        require_tool_call=True,
    )

    assert record["capability_probe"] == "tool_call"
    assert record["tool_call_name"] == "preflight_healthcheck"


def test_preflight_image_probe_is_capability_checked_without_persisting_payload(
    monkeypatch,
):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/models"):
            return _Response({"data": [{"id": "qwen3.5-27b"}]})
        return _Response(
            {"choices": [{"message": {"content": "IMAGE_OK", "tool_calls": []}}]}
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)
    secret = "image-preflight-secret"

    record = preflight_openai_compatible_model(
        base_url="https://qwen-gateway.example.test/v1",
        model="qwen3.5-27b",
        api_key=secret,
        api_mode="chat_completions",
        label="ISBench actor",
        require_image_input=True,
    )

    body = json.loads(requests[1].data)
    assert isinstance(body["messages"][0]["content"], list)
    assert [item["type"] for item in body["messages"][0]["content"]] == [
        "text",
        "image_url",
    ]
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert record["capability_probe"] == "image"
    serialized = json.dumps(record, sort_keys=True)
    assert "data:image" not in serialized
    assert secret not in serialized


def test_preflight_v3_gateway_identity_uses_only_allowlisted_models_headers(
    monkeypatch,
):
    requests = []
    ingress_id = "approved-ingress-20260818"
    digest = "a" * 64

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/models"):
            return _Response(
                {"data": [{"id": "qwen3.5-27b"}]},
                headers={
                    "X-ISBench-Ingress-ID": ingress_id,
                    "X-ISBench-Gateway-Runtime-Freeze-SHA256": digest,
                    "Authorization": "Bearer gateway-response-secret",
                    "Set-Cookie": "session=gateway-response-secret",
                },
            )
        return _Response(
            {"choices": [{"message": {"content": "OK", "tool_calls": []}}]}
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)
    record = preflight_openai_compatible_model(
        base_url="https://gateway.example.test/v1",
        model="qwen3.5-27b",
        api_key="client-secret",
        api_mode="chat_completions",
        label="ISBench actor",
        expected_gateway_runtime_identity={
            "ingress_id": ingress_id,
            "gateway_runtime_freeze_sha256": digest,
        },
    )

    assert len(requests) == 2
    assert record["gateway_runtime_identity"] == {
        "ingress_id": ingress_id,
        "gateway_runtime_freeze_sha256": digest,
    }
    serialized = json.dumps(record, sort_keys=True)
    assert "gateway-response-secret" not in serialized
    assert "client-secret" not in serialized
    assert "Authorization" not in serialized
    assert "Set-Cookie" not in serialized


def test_preflight_v3_gateway_identity_fails_closed_without_recording_headers(
    monkeypatch,
):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return _Response(
                {"data": [{"id": "qwen3.5-27b"}]},
                headers={
                    "X-ISBench-Ingress-ID": "other-ingress",
                    "X-ISBench-Gateway-Runtime-Freeze-SHA256": "b" * 64,
                    "Authorization": "Bearer response-secret",
                },
            )
        raise AssertionError("mismatched gateway identity must stop before inference")

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="identity ingress_id is missing or mismatched") as raised:
        preflight_openai_compatible_model(
            base_url="https://gateway.example.test/v1",
            model="qwen3.5-27b",
            api_key="client-secret",
            api_mode="chat_completions",
            label="ISBench actor",
            expected_gateway_runtime_identity={
                "ingress_id": "approved-ingress",
                "gateway_runtime_freeze_sha256": "a" * 64,
            },
        )
    assert "other-ingress" not in str(raised.value)
    assert "response-secret" not in str(raised.value)


def test_preflight_rejects_sampling_controls_for_responses_before_io(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("responses preflight must fail before network I/O")

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="require chat_completions"):
        preflight_openai_compatible_model(
            base_url="https://relay.example/v1",
            model="actor-model",
            api_key="secret",
            api_mode="responses",
            label="actor",
            top_p=0.95,
            top_k=20,
        )


def test_preflight_failure_is_bounded_and_does_not_leak_credentials(monkeypatch):
    calls = 0

    def fail_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            502,
            "gateway failed secret-key",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("experiments.endpoint_preflight.urllib.request.urlopen", fail_urlopen)

    with pytest.raises(RuntimeError) as raised:
        preflight_openai_compatible_model(
            base_url="https://user:pw@relay.example/v1?token=x",
            model="actor-model",
            api_key="secret-key",
            api_mode="chat_completions",
            label="actor",
        )

    assert calls == 1
    message = str(raised.value)
    assert "secret-key" not in message
    assert "pw" not in message
    assert "token=x" not in message
    assert "HTTPError" in message
    assert "status=502" in message
