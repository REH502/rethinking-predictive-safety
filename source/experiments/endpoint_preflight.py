"""Bounded, credential-safe OpenAI-compatible endpoint capability checks."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
import uuid
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_PREFLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "preflight_healthcheck",
        "description": "Return a transport capability healthcheck.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

# A fixed, strictly decodable 2x2 RGB PNG exercises the OpenAI-compatible image
# message schema.  The earlier one-pixel fixture had a valid PNG signature but
# a truncated pixel stream, which recent Pillow/vLLM versions correctly reject.
# It is sent only on the wire, never persisted in a preflight record or error
# artifact.
_PREFLIGHT_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91Jpz"
    "AAAAEklEQVR4nGP4z8DAAMIM/4EAAB/uBfsL2WiLAAAAAElFTkSuQmCC"
)

# Only these two explicitly non-secret headers are ever inspected for an
# administrator-managed IS-Bench gateway.  Do not enumerate response headers:
# that could accidentally persist an authorization/cookie value supplied by a
# proxy.  Their names are part of the frozen v3 client implementation.
_GATEWAY_RUNTIME_IDENTITY_HEADERS = {
    "ingress_id": "X-ISBench-Ingress-ID",
    "gateway_runtime_freeze_sha256": "X-ISBench-Gateway-Runtime-Freeze-SHA256",
}


def sanitize_endpoint(value: str) -> str:
    """Return an absolute credential/query-free endpoint identity."""

    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        raise ValueError("endpoint must be an absolute HTTP(S) URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme.lower(), netloc, (parsed.path or "").rstrip("/"), "", "")
    )


def _request_json(
    *,
    url: str,
    api_key: str,
    timeout_seconds: int,
    method: str,
    body: dict[str, Any] | None = None,
    label: str,
    stage: str,
    expected_gateway_runtime_identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    headers = {
        "Accept": "application/json",
        # Some OpenAI-compatible gateways reject anonymous/default urllib
        # clients at the model-listing endpoint. This contains no credential
        # and is persisted only as transport provenance if needed.
        "User-Agent": "world-model-agent-safety-preflight/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, headers=headers, data=data, method=method)
    attempts = int(os.environ.get('ISBENCH_ENDPOINT_PREFLIGHT_MAX_ATTEMPTS', '1'))
    if not 1 <= attempts <= 3:
        raise ValueError('endpoint preflight attempts must be between one and three')
    request_id = uuid.uuid4().hex
    digest = hashlib.sha256(method.encode() + sanitize_endpoint(url).encode() + (data or b'')).hexdigest()
    ledger = os.environ.get('ISBENCH_ENDPOINT_PREFLIGHT_ATTEMPT_LEDGER')
    def record(attempt, status, **fields):
        if not ledger:
            return
        row = {'request_id': request_id, 'request_sha256': digest, 'method': method,
               'stage': stage, 'label': label, 'endpoint': sanitize_endpoint(url),
               'transport_attempt': attempt, 'status': status, 'at_unix': time.time(), **fields}
        raw = (json.dumps(row, sort_keys=True) + '\n').encode()
        if api_key and api_key.encode() in raw:
            raise ValueError('credential material in preflight telemetry')
        fd = os.open(ledger, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(fd, raw[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
    for attempt in range(1, attempts + 1):
        record(attempt, 'started')
        try:
            with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
                response_bytes = response.read()
                headers = getattr(response, 'headers', None)
            record(attempt, 'transport_completed', http_status=200)
            payload = json.loads(response_bytes.decode('utf-8'))
            break
        except Exception as exc:
            status = getattr(exc, 'code', None)
            record(attempt, 'failed', error_type=type(exc).__name__, http_status=status)
            retryable = (status in {408, 409, 429, 500, 502, 503, 504, 520, 524}
                         or isinstance(exc, (TimeoutError, ConnectionError))
                         or isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError))
            if retryable and attempt < attempts:
                time.sleep(5 * attempt)
                continue
            status_suffix = f' status={status}' if isinstance(status, int) else ''
            raise RuntimeError(
                f'{label} endpoint preflight {stage} failed at '
                f'{sanitize_endpoint(url)}: {type(exc).__name__}{status_suffix}'
            ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{label} endpoint preflight {stage} returned non-object JSON at "
            f"{sanitize_endpoint(url)}"
        )
    observed_identity: dict[str, str] | None = None
    if expected_gateway_runtime_identity is not None:
        observed_identity = {}
        for field, header_name in _GATEWAY_RUNTIME_IDENTITY_HEADERS.items():
            value = headers.get(header_name) if headers is not None else None
            if not isinstance(value, str) or value != expected_gateway_runtime_identity[field]:
                raise RuntimeError(
                    f"{label} endpoint preflight model_listing gateway runtime "
                    f"identity {field} is missing or mismatched"
                )
            # Persist only allowlisted, already-matched non-secret values.
            observed_identity[field] = value
    return payload, observed_identity


def _merge_request_extra_body(
    body: dict[str, Any],
    request_extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extension = dict(request_extra_body or {})
    collisions = sorted(set(body) & set(extension))
    if collisions:
        raise ValueError(
            "preflight request_extra_body collides with probe-owned field(s): "
            f"{collisions}"
        )
    body.update(extension)
    return body


def _chat_probe_body(
    model: str,
    *,
    require_tool_call: bool,
    require_image_input: bool,
    request_extra_body: Mapping[str, Any] | None,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> dict[str, Any]:
    text = (
        "Call preflight_healthcheck exactly once with an empty object."
        if require_tool_call
        else "Reply with exactly IMAGE_OK."
        if require_image_input
        else "Reply with exactly OK."
    )
    content: str | list[dict[str, Any]] = text
    if require_image_input:
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": _PREFLIGHT_IMAGE_DATA_URL},
            },
        ]
    body: dict[str, Any] = {
        "model": str(model),
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if require_tool_call:
        body["tools"] = [_PREFLIGHT_TOOL]
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": "preflight_healthcheck"},
        }
    return _merge_request_extra_body(body, request_extra_body)


def _validate_chat_probe(
    payload: dict[str, Any],
    *,
    require_tool_call: bool,
    label: str,
) -> str | None:
    choices = payload.get("choices")
    message = (
        choices[0].get("message")
        if isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
        else None
    )
    if not isinstance(message, dict):
        raise RuntimeError(f"{label} inference probe returned no assistant message")
    if require_tool_call:
        calls = message.get("tool_calls")
        function = (
            calls[0].get("function")
            if isinstance(calls, list)
            and calls
            and isinstance(calls[0], dict)
            else None
        )
        name = function.get("name") if isinstance(function, dict) else None
        if name != "preflight_healthcheck":
            raise RuntimeError(
                f"{label} inference probe did not emit the required tool call"
            )
        return str(name)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{label} inference probe returned empty content")
    return None


def _validate_responses_probe(payload: dict[str, Any], *, label: str) -> None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return
    output = payload.get("output")
    if isinstance(output, list) and output:
        return
    raise RuntimeError(f"{label} responses inference probe returned empty output")


def _normalized_gateway_runtime_identity(
    value: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    if set(value) != set(_GATEWAY_RUNTIME_IDENTITY_HEADERS):
        raise ValueError("gateway runtime identity must contain exactly ingress_id and gateway_runtime_freeze_sha256")
    ingress_id = value.get("ingress_id")
    runtime_digest = value.get("gateway_runtime_freeze_sha256")
    if not isinstance(ingress_id, str) or not ingress_id.strip() or "\n" in ingress_id or "\r" in ingress_id:
        raise ValueError("gateway runtime ingress_id is invalid")
    if not isinstance(runtime_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime_digest):
        raise ValueError("gateway runtime freeze digest must be a lowercase SHA256")
    return {
        "ingress_id": ingress_id,
        "gateway_runtime_freeze_sha256": runtime_digest,
    }


def preflight_openai_compatible_model(
    *,
    base_url: str,
    model: str,
    api_key: str,
    api_mode: str,
    label: str,
    timeout_seconds: int = 30,
    require_tool_call: bool = False,
    require_image_input: bool = False,
    request_extra_body: Mapping[str, Any] | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    expected_gateway_runtime_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check exact model enumeration and one real inference, with no retries."""

    endpoint = sanitize_endpoint(base_url)
    requested_model = str(model or "").strip()
    if not requested_model:
        raise ValueError("preflight model must be non-empty")
    if api_mode not in {"chat_completions", "responses"}:
        raise ValueError(f"unsupported OpenAI-compatible api_mode: {api_mode!r}")
    if require_tool_call and api_mode != "chat_completions":
        raise ValueError("tool-call preflight requires chat_completions api_mode")
    if require_image_input and api_mode != "chat_completions":
        raise ValueError("image preflight requires chat_completions api_mode")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("preflight temperature must be a finite nonnegative number")
    normalized_temperature = float(temperature)
    if not math.isfinite(normalized_temperature) or normalized_temperature < 0:
        raise ValueError("preflight temperature must be a finite nonnegative number")
    normalized_top_p: float | None = None
    normalized_top_k: int | None = None
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
            raise ValueError("preflight top_p must be a finite number in (0, 1]")
        normalized_top_p = float(top_p)
        if not math.isfinite(normalized_top_p) or not 0.0 < normalized_top_p <= 1.0:
            raise ValueError("preflight top_p must be a finite number in (0, 1]")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("preflight top_k must be a nonnegative integer")
        normalized_top_k = top_k
    if api_mode == "responses" and normalized_top_p is not None:
        raise ValueError(
            "preflight top_p/top_k controls require chat_completions api_mode"
        )

    gateway_runtime_identity = _normalized_gateway_runtime_identity(
        expected_gateway_runtime_identity
    )
    models_url = endpoint + "/models"
    listing, observed_gateway_runtime_identity = _request_json(
        url=models_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        method="GET",
        label=label,
        stage="model_listing",
        expected_gateway_runtime_identity=gateway_runtime_identity,
    )
    rows = listing.get("data")
    model_ids = sorted(
        {
            str(item.get("id") or "")
            for item in rows or []
            if isinstance(item, dict) and str(item.get("id") or "")
        }
    )
    if requested_model not in model_ids:
        raise RuntimeError(
            f"{label} exact model {requested_model!r} is not served by {models_url}"
        )

    if api_mode == "chat_completions":
        inference_url = endpoint + "/chat/completions"
        response, _ = _request_json(
            url=inference_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            method="POST",
            body=_chat_probe_body(
                requested_model,
                require_tool_call=require_tool_call,
                require_image_input=require_image_input,
                request_extra_body=request_extra_body,
                temperature=normalized_temperature,
                top_p=normalized_top_p,
                top_k=normalized_top_k,
            ),
            label=label,
            stage="inference",
        )
        tool_call_name = _validate_chat_probe(
            response,
            require_tool_call=require_tool_call,
            label=label,
        )
    else:
        inference_url = endpoint + "/responses"
        response, _ = _request_json(
            url=inference_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            method="POST",
            body=_merge_request_extra_body(
                {
                    "model": requested_model,
                    "input": "Reply with exactly OK.",
                    "temperature": 0,
                    "max_output_tokens": 32,
                },
                request_extra_body,
            ),
            label=label,
            stage="inference",
        )
        _validate_responses_probe(response, label=label)
        tool_call_name = None

    record: dict[str, Any] = {
        "type": "endpoint_preflight",
        "status": "passed",
        "label": str(label),
        "base_url": endpoint,
        "requested_model": requested_model,
        "api_mode": api_mode,
        "model_listing": "exact_match",
        "available_model_count": len(model_ids),
        "inference_probe": "passed",
        "capability_probe": (
            "tool_call_and_image"
            if require_tool_call and require_image_input
            else "tool_call"
            if require_tool_call
            else "image"
            if require_image_input
            else "text"
        ),
        # Do not persist a request body (or fragments of one).  The frozen
        # model profile binds any extensions separately.
        "request_extra_body_present": bool(request_extra_body),
        "sampling_controls": {
            "temperature": normalized_temperature,
            "top_p": normalized_top_p,
            "top_k": normalized_top_k,
        },
    }
    if tool_call_name is not None:
        record["tool_call_name"] = tool_call_name
    if observed_gateway_runtime_identity is not None:
        record["gateway_runtime_identity"] = observed_gateway_runtime_identity
    return record
