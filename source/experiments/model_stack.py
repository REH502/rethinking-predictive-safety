"""Strict, transport-agnostic model-stack profiles for experiment workflows.

Every runtime is represented as an OpenAI-compatible endpoint.  A profile may
therefore point at an official API, a relay, or a local server without adding a
provider branch inside the experiment implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

import yaml


MODEL_STACK_SCHEMA_VERSION = "model-stack-profile-v1"
MODEL_STACK_ROLES = (
    "actor",
    "judge",
    "world_model",
    "policy_extractor",
    "action_agent",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOP_LEVEL_FIELDS = {"schema_version", "defaults", "endpoints", "roles"}
_REQUIRED_DEFAULT_FIELDS = {"temperature", "max_tokens", "timeout_seconds"}
_DEFAULT_FIELDS = {
    *_REQUIRED_DEFAULT_FIELDS,
    "transport_max_attempts",
    "request_extra_body",
}
_ENDPOINT_FIELDS = {
    "base_url",
    "base_url_env",
    "api_key_env",
    "auth",
    "api_mode",
}
_ROLE_FIELDS = {
    "endpoint",
    "model",
    "revision",
    "runtime_revision",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "timeout_seconds",
    "transport_max_attempts",
    "request_extra_body",
}

_FORBIDDEN_REQUEST_EXTRA_BODY_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "input",
    "instructions",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "model",
    "seed",
    "temperature",
    "top_p",
    "top_k",
}


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {unknown}")


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _env_name(value: object, *, label: str) -> str:
    name = _required_text(value, label=label)
    if not _ENV_NAME.fullmatch(name):
        raise ValueError(f"{label} is not a valid environment variable name: {name!r}")
    return name


def _nonnegative_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nonnegative number") from exc
    if number < 0:
        raise ValueError(f"{label} must be a nonnegative number")
    return number


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number in (0, 1]") from exc
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    return number


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number <= 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nonnegative integer") from exc
    if number < 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ValueError(f"{label} must be a nonnegative integer")
    return number


def _request_extra_body(value: object, *, label: str) -> dict[str, object]:
    """Validate a credential-free, JSON-compatible provider extension mapping.

    Per-trial fields remain owned by the experiment implementation.  In
    particular, a profile cannot override the model, prompt, sampling controls,
    output budget, or trial seed through an opaque provider extension.
    """

    if value is None:
        return {}
    mapping = _mapping(value, label=label)

    def normalize(item: object, *, path: str) -> object:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} must contain only finite JSON numbers")
            return item
        if isinstance(item, list):
            return [normalize(child, path=f"{path}[]") for child in item]
        if isinstance(item, dict):
            normalized: dict[str, object] = {}
            for raw_key, child in item.items():
                if not isinstance(raw_key, str) or not raw_key.strip():
                    raise ValueError(f"{path} keys must be non-empty strings")
                key = raw_key.strip()
                if key.lower().replace("-", "_") in _FORBIDDEN_REQUEST_EXTRA_BODY_KEYS:
                    raise ValueError(
                        f"{path}.{key} is reserved or credential-bearing and cannot "
                        "be set through request_extra_body"
                    )
                normalized[key] = normalize(child, path=f"{path}.{key}")
            return normalized
        raise ValueError(f"{path} must contain only JSON-compatible values")

    normalized: dict[str, object] = {}
    for key, item in mapping.items():
        if key.lower().replace("-", "_") in _FORBIDDEN_REQUEST_EXTRA_BODY_KEYS:
            raise ValueError(
                f"{label}.{key} is reserved or credential-bearing and cannot "
                "be set through request_extra_body"
            )
        normalized[key] = normalize(item, path=f"{label}.{key}")
    return normalized


def _sanitize_base_url(value: object, *, label: str) -> str:
    raw = _required_text(value, label=label)
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not contain userinfo, query, or fragment")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    path = (parsed.path or "").rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


@dataclass(frozen=True)
class ResolvedModelRole:
    name: str
    endpoint: str
    model: str
    revision: str
    runtime_revision: str
    base_url: str
    api_mode: str
    auth: str
    api_key_env: str | None
    base_url_env: str | None
    temperature: float
    top_p: float | None
    top_k: int | None
    max_tokens: int
    timeout_seconds: int
    transport_max_attempts: int
    request_extra_body: dict[str, object]
    api_key: str = field(repr=False)

    def public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "transport": "openai_compatible",
            "endpoint": self.endpoint,
            "model": self.model,
            "revision": self.revision,
            "runtime_revision": self.runtime_revision,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "base_url_env": self.base_url_env,
            "auth": self.auth,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "transport_max_attempts": self.transport_max_attempts,
            "request_extra_body": self.request_extra_body,
        }
        # These controls are actor-only and optional for compatibility with
        # existing frozen profiles.  When present they must be carried through
        # the public identity and child environment as first-class sampling
        # parameters, never as opaque provider extensions.
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        return payload


@dataclass(frozen=True)
class ResolvedModelStack:
    source_path: Path
    source_sha256: str
    roles: dict[str, ResolvedModelRole]

    @property
    def public_identity(self) -> dict[str, object]:
        try:
            source_identity = self.source_path.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            source_identity = str(self.source_path)
        return {
            "schema_version": MODEL_STACK_SCHEMA_VERSION,
            "source_path": source_identity,
            "source_sha256": self.source_sha256,
            "roles": {
                role: self.roles[role].public_dict() for role in MODEL_STACK_ROLES
            },
        }

    def runtime_environment(self) -> dict[str, str]:
        env = {
            "WM_SAFETY_MODEL_STACK_PROFILE": str(self.source_path),
            "WM_SAFETY_MODEL_STACK_SHA256": self.source_sha256,
            "WM_SAFETY_MODEL_STACK_PUBLIC_JSON": json.dumps(
                self.public_identity,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for role_name in MODEL_STACK_ROLES:
            role = self.roles[role_name]
            prefix = f"WM_SAFETY_{role_name.upper()}"
            role_env = {
                    f"{prefix}_MODEL": role.model,
                    f"{prefix}_REVISION": role.revision,
                    f"{prefix}_RUNTIME_REVISION": role.runtime_revision,
                    f"{prefix}_BASE_URL": role.base_url,
                    f"{prefix}_API_MODE": role.api_mode,
                    f"{prefix}_API_KEY": role.api_key,
                    f"{prefix}_API_KEY_ENV": role.api_key_env or "",
                    f"{prefix}_TEMPERATURE": str(role.temperature),
                    f"{prefix}_MAX_TOKENS": str(role.max_tokens),
                    f"{prefix}_TIMEOUT_SECONDS": str(role.timeout_seconds),
                    f"{prefix}_TRANSPORT_MAX_ATTEMPTS": str(
                        role.transport_max_attempts
                    ),
                    f"{prefix}_REQUEST_EXTRA_BODY_JSON": json.dumps(
                        role.request_extra_body,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
            }
            if role_name == "actor" and role.top_p is not None:
                role_env[f"{prefix}_TOP_P"] = str(role.top_p)
            if role_name == "actor" and role.top_k is not None:
                role_env[f"{prefix}_TOP_K"] = str(role.top_k)
            env.update(role_env)
        return env


def load_model_stack_profile(
    path: str | Path,
    *,
    environ: Mapping[str, str],
    require_credentials: bool = True,
) -> ResolvedModelStack:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"Model stack profile is not a file: {source_path}")
    source_bytes = source_path.read_bytes()
    payload = yaml.safe_load(source_bytes) or {}
    profile = _mapping(payload, label="model stack profile")
    _reject_unknown(profile, _TOP_LEVEL_FIELDS, label="model stack profile")
    if profile.get("schema_version") != MODEL_STACK_SCHEMA_VERSION:
        raise ValueError(
            "model stack profile schema_version must be "
            f"{MODEL_STACK_SCHEMA_VERSION!r}"
        )

    defaults = _mapping(profile.get("defaults"), label="defaults")
    _reject_unknown(defaults, _DEFAULT_FIELDS, label="defaults")
    missing_defaults = sorted(_REQUIRED_DEFAULT_FIELDS - set(defaults))
    if missing_defaults:
        raise ValueError(f"defaults missing fields: {missing_defaults}")
    resolved_defaults = {
        "temperature": _nonnegative_float(
            defaults["temperature"], label="defaults.temperature"
        ),
        "max_tokens": _positive_int(
            defaults["max_tokens"], label="defaults.max_tokens"
        ),
        "timeout_seconds": _positive_int(
            defaults["timeout_seconds"], label="defaults.timeout_seconds"
        ),
        "transport_max_attempts": _positive_int(
            defaults.get("transport_max_attempts", 1),
            label="defaults.transport_max_attempts",
        ),
        "request_extra_body": _request_extra_body(
            defaults.get("request_extra_body"),
            label="defaults.request_extra_body",
        ),
    }

    endpoint_payloads = _mapping(profile.get("endpoints"), label="endpoints")
    if not endpoint_payloads:
        raise ValueError("endpoints must define at least one endpoint")
    endpoints: dict[str, dict[str, object]] = {}
    for endpoint_name, raw_endpoint in endpoint_payloads.items():
        endpoint = _mapping(raw_endpoint, label=f"endpoints.{endpoint_name}")
        _reject_unknown(
            endpoint,
            _ENDPOINT_FIELDS,
            label=f"endpoints.{endpoint_name}",
        )
        has_literal_url = bool(str(endpoint.get("base_url") or "").strip())
        has_url_env = bool(str(endpoint.get("base_url_env") or "").strip())
        if has_literal_url == has_url_env:
            raise ValueError(
                f"endpoints.{endpoint_name} must set exactly one of base_url or base_url_env"
            )
        base_url_env = (
            _env_name(
                endpoint.get("base_url_env"),
                label=f"endpoints.{endpoint_name}.base_url_env",
            )
            if has_url_env
            else None
        )
        raw_base_url = (
            environ.get(base_url_env or "") if base_url_env else endpoint.get("base_url")
        )
        if base_url_env and not str(raw_base_url or "").strip():
            raise ValueError(
                f"endpoints.{endpoint_name} selected missing environment variable "
                f"{base_url_env}"
            )
        base_url = _sanitize_base_url(
            raw_base_url,
            label=f"endpoints.{endpoint_name}.base_url",
        )

        api_key_env = (
            _env_name(
                endpoint.get("api_key_env"),
                label=f"endpoints.{endpoint_name}.api_key_env",
            )
            if str(endpoint.get("api_key_env") or "").strip()
            else None
        )
        auth = str(endpoint.get("auth") or ("api_key" if api_key_env else "none"))
        if auth not in {"api_key", "none"}:
            raise ValueError(f"endpoints.{endpoint_name}.auth must be api_key or none")
        if auth == "api_key" and not api_key_env:
            raise ValueError(
                f"endpoints.{endpoint_name}.api_key_env is required for api_key auth"
            )
        if auth == "none" and api_key_env:
            raise ValueError(
                f"endpoints.{endpoint_name} cannot combine auth none with api_key_env"
            )
        api_key = "local"
        if api_key_env:
            api_key = str(environ.get(api_key_env) or "").strip()
            if not api_key and require_credentials:
                raise ValueError(
                    f"endpoints.{endpoint_name} selected missing environment variable "
                    f"{api_key_env}"
                )
        api_mode = str(endpoint.get("api_mode") or "chat_completions").strip()
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(
                f"endpoints.{endpoint_name}.api_mode must be chat_completions or responses"
            )
        endpoints[endpoint_name] = {
            "base_url": base_url,
            "api_mode": api_mode,
            "base_url_env": base_url_env,
            "auth": auth,
            "api_key_env": api_key_env,
            "api_key": api_key,
        }

    role_payloads = _mapping(profile.get("roles"), label="roles")
    missing_roles = sorted(set(MODEL_STACK_ROLES) - set(role_payloads))
    extra_roles = sorted(set(role_payloads) - set(MODEL_STACK_ROLES))
    if missing_roles:
        raise ValueError(f"model stack profile missing roles: {missing_roles}")
    if extra_roles:
        raise ValueError(f"model stack profile has unknown roles: {extra_roles}")
    roles: dict[str, ResolvedModelRole] = {}
    for role_name in MODEL_STACK_ROLES:
        role = _mapping(role_payloads[role_name], label=f"roles.{role_name}")
        _reject_unknown(role, _ROLE_FIELDS, label=f"roles.{role_name}")
        has_top_p = "top_p" in role
        has_top_k = "top_k" in role
        if role_name != "actor" and (has_top_p or has_top_k):
            raise ValueError(
                "top_p and top_k are only supported for roles.actor"
            )
        endpoint_name = _required_text(
            role.get("endpoint"), label=f"roles.{role_name}.endpoint"
        )
        if endpoint_name not in endpoints:
            raise ValueError(
                f"roles.{role_name}.endpoint references unknown endpoint {endpoint_name!r}"
            )
        model = _required_text(role.get("model"), label=f"roles.{role_name}.model")
        endpoint = endpoints[endpoint_name]
        if (has_top_p or has_top_k) and endpoint["api_mode"] != "chat_completions":
            raise ValueError(
                "roles.actor.top_p/top_k require a chat_completions endpoint"
            )
        if (
            role_name in {"world_model", "policy_extractor", "action_agent"}
            and endpoint["api_mode"] != "chat_completions"
        ):
            raise ValueError(
                f"roles.{role_name} requires a chat_completions endpoint"
            )
        roles[role_name] = ResolvedModelRole(
            name=role_name,
            endpoint=endpoint_name,
            model=model,
            revision=_required_text(
                role.get("revision") or model,
                label=f"roles.{role_name}.revision",
            ),
            runtime_revision=_required_text(
                role.get("runtime_revision") or "unversioned-runtime",
                label=f"roles.{role_name}.runtime_revision",
            ),
            base_url=str(endpoint["base_url"]),
            api_mode=str(endpoint["api_mode"]),
            base_url_env=(
                str(endpoint["base_url_env"])
                if endpoint["base_url_env"] is not None
                else None
            ),
            auth=str(endpoint["auth"]),
            api_key_env=(
                str(endpoint["api_key_env"])
                if endpoint["api_key_env"] is not None
                else None
            ),
            api_key=str(endpoint["api_key"]),
            temperature=_nonnegative_float(
                role.get("temperature", resolved_defaults["temperature"]),
                label=f"roles.{role_name}.temperature",
            ),
            top_p=(
                _probability(role["top_p"], label="roles.actor.top_p")
                if has_top_p
                else None
            ),
            top_k=(
                _nonnegative_int(role["top_k"], label="roles.actor.top_k")
                if has_top_k
                else None
            ),
            max_tokens=_positive_int(
                role.get("max_tokens", resolved_defaults["max_tokens"]),
                label=f"roles.{role_name}.max_tokens",
            ),
            timeout_seconds=_positive_int(
                role.get("timeout_seconds", resolved_defaults["timeout_seconds"]),
                label=f"roles.{role_name}.timeout_seconds",
            ),
            transport_max_attempts=_positive_int(
                role.get(
                    "transport_max_attempts",
                    resolved_defaults["transport_max_attempts"],
                ),
                label=f"roles.{role_name}.transport_max_attempts",
            ),
            request_extra_body=_request_extra_body(
                role.get(
                    "request_extra_body",
                    resolved_defaults["request_extra_body"],
                ),
                label=f"roles.{role_name}.request_extra_body",
            ),
        )

    return ResolvedModelStack(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        roles=roles,
    )


def load_model_stack_from_environment(
    environ: Mapping[str, str],
    *,
    require_credentials: bool = True,
) -> ResolvedModelStack | None:
    """Revalidate the parent-resolved profile at a subprocess boundary."""

    profile_path = str(environ.get("WM_SAFETY_MODEL_STACK_PROFILE") or "").strip()
    if not profile_path:
        return None
    stack = load_model_stack_profile(
        profile_path,
        environ=environ,
        require_credentials=require_credentials,
    )
    expected_sha = str(environ.get("WM_SAFETY_MODEL_STACK_SHA256") or "").strip()
    if expected_sha != stack.source_sha256:
        raise ValueError("model stack profile SHA256 changed after parent resolution")
    public_raw = str(
        environ.get("WM_SAFETY_MODEL_STACK_PUBLIC_JSON") or ""
    ).strip()
    try:
        observed_public = json.loads(public_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model stack public identity is not valid JSON") from exc
    if observed_public != stack.public_identity:
        raise ValueError("model stack public identity changed after parent resolution")
    return stack
