import os
import hashlib
import re
import sys
import time
from typing import Any, List
from google import genai
from google.genai import types
from openai import OpenAI 

from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.utils.model_request_options import (
    model_specific_extra_body,
    request_extra_body_disables_thinking,
)
from og_ego_prim.models.image_utils import (
    encode_image, 
    guess_image_type_from_base64,
)
from og_ego_prim.utils.trace_provenance import (
    AIFAST_OUTPUT_CONTRACT,
    actor_output_contract,
    actor_transport_max_retries,
    actor_transport_compat,
    actor_transport_retry_base_delay_seconds,
    actor_transport_retry_policy,
    is_aifast_endpoint,
    PROVIDER_REASONING_CAPTURE,
    resolve_openai_api_base,
)


AIFAST_DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "aifast_transport_compat_noop",
        "description": "Transport compatibility placeholder; never call this function.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

TRANSIENT_ACTOR_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 520, 524}
TRANSIENT_ACTOR_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionTimeout",
    "ReadTimeout",
    "Timeout",
}
PROVIDER_REASONING_SOURCE_FIELD = "choices[0].message.reasoning_content"


def read_image(image_path: str):
  with open(image_path, "rb") as f:
     return f.read()


def _require_nonempty_actor_content(content, *, provider: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{provider} actor response contained empty content")
    return content


def empty_provider_reasoning_record() -> dict[str, Any]:
    """Return the fixed report shape for a response without exposed reasoning."""

    return {
        "capture_contract": PROVIDER_REASONING_CAPTURE,
        "source_field": PROVIDER_REASONING_SOURCE_FIELD,
        "present": False,
        "content": "",
        "char_count": 0,
        "sha256": None,
    }


def _provider_reasoning_value(message: Any) -> Any:
    if message is None:
        return None
    if isinstance(message, dict) and "reasoning_content" in message:
        return message.get("reasoning_content")
    value = getattr(message, "reasoning_content", None)
    if value is not None:
        return value
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
        return model_extra.get("reasoning_content")
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped.get("reasoning_content")
    return None


def _sanitize_provider_reasoning_content(
    value: Any,
    *,
    secrets: tuple[str, ...] = (),
) -> str:
    if value is None:
        return ""
    content = value if isinstance(value, str) else str(value)
    for secret in secrets:
        if secret:
            content = content.replace(secret, "[REDACTED]")
    return re.sub(
        r"data:image/[^\s,;]+(?:;base64)?,[^\s\"']+",
        "[REDACTED_IMAGE]",
        content,
        flags=re.IGNORECASE,
    )


def provider_reasoning_record(
    message: Any,
    *,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Capture provider-exposed reasoning without feeding it into control."""

    content = _sanitize_provider_reasoning_content(
        _provider_reasoning_value(message),
        secrets=secrets,
    )
    if not content:
        return empty_provider_reasoning_record()
    return {
        "capture_contract": PROVIDER_REASONING_CAPTURE,
        "source_field": PROVIDER_REASONING_SOURCE_FIELD,
        "present": True,
        "content": content,
        "char_count": len(content),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
  
class ServerClient(BaseClient):

    @staticmethod
    def _actor_request_timeout() -> float:
        value = float(os.environ.get("ISBENCH_ACTOR_REQUEST_TIMEOUT", "120"))
        if value <= 0:
            raise ValueError("ISBENCH_ACTOR_REQUEST_TIMEOUT must be positive")
        return value

    @staticmethod
    def _actor_transport_max_retries() -> int:
        return actor_transport_max_retries()

    @staticmethod
    def _actor_transport_retry_base_delay() -> float:
        return actor_transport_retry_base_delay_seconds()

    def __init__(
        self,
        model_type,
        model_name,
        api_key=None,
        api_base=None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._provider_reasoning_secrets = (
            (str(resolved_api_key),) if resolved_api_key else ()
        )
        self._last_provider_reasoning_record = empty_provider_reasoning_record()
        self.model_type = model_type
        self.api_base = str(api_base or "").strip()
        self.model_name = str(model_name)
        self.model_extra_body = model_specific_extra_body(
            self.model_name,
            role="actor",
        )
        self.disable_thinking = request_extra_body_disables_thinking(
            self.model_extra_body
        )
        if model_type == "local":
            for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
                os.environ.pop(key, None)
        # Any remote proxy is deployment configuration supplied through the
        # process environment; it is intentionally absent from source code.

        if 'gemini_direct' in model_name.lower():
            self.client = genai.Client(
                vertexai=True,
                # project="czby-gemini-250612",
                location="global",
            )
            self.gemini_safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
                ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
                )]
            self.generate_content_config = types.GenerateContentConfig(
                temperature=self._actor_temperature(),
                top_p=self._actor_top_p(),
                seed=self._actor_seed(),
                max_output_tokens=self._actor_max_completion_tokens(),
                safety_settings=self.gemini_safety_settings,
            )
        else:
            if model_type == "close_source":
                api_base = resolve_openai_api_base(api_base)
                if not api_base:
                    raise ValueError(
                        "OpenAI API base is required; set OPENAI_API_BASE "
                        "or OPENAI_BASE_URL"
                    )
            self.api_base = str(api_base or "").strip()
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=api_base,
                # Transport retries are explicit below so their policy is
                # frozen in trace provenance instead of hidden inside the SDK.
                max_retries=0,
                timeout=self._actor_request_timeout(),
            )
        
        if model_name == "local":
            model_name = self.client.models.list().data[0].id
        self.model_name = model_name
        print(f"MODEL NAME: {self.model_name}")

    @staticmethod
    def _actor_seed() -> int:
        return int(os.environ.get("ISBENCH_ACTOR_SEED", "0"))

    @staticmethod
    def _actor_temperature() -> float:
        value = float(os.environ.get("ISBENCH_ACTOR_TEMPERATURE", "0.0"))
        if not 0.0 <= value <= 2.0:
            raise ValueError("ISBENCH_ACTOR_TEMPERATURE must be in [0, 2]")
        return value

    @staticmethod
    def _actor_top_p() -> float:
        value = float(os.environ.get("ISBENCH_ACTOR_TOP_P", "1.0"))
        if not 0.0 < value <= 1.0:
            raise ValueError("ISBENCH_ACTOR_TOP_P must be in (0, 1]")
        return value

    @staticmethod
    def _actor_max_completion_tokens() -> int:
        value = int(os.environ.get("ISBENCH_AGENT_MAX_COMPLETION_TOKENS", "512"))
        if value < 1:
            raise ValueError("ISBENCH_AGENT_MAX_COMPLETION_TOKENS must be positive")
        return value

    @staticmethod
    def _actor_max_images() -> int:
        """Return the frozen per-request actor image budget.

        Zero preserves the upstream unlimited behavior.  A positive value is
        applied deterministically to the camera-config order before encoding,
        which lets a shared serving endpoint expose a smaller multimodal limit
        without causing a provider-side 400 after simulator startup.
        """

        value = int(os.environ.get("ISBENCH_ACTOR_MAX_IMAGES", "0"))
        if value < 0:
            raise ValueError("ISBENCH_ACTOR_MAX_IMAGES must be non-negative")
        return value

    @classmethod
    def _select_actor_images(cls, image_files: List[str]) -> List[str]:
        limit = cls._actor_max_images()
        selected = list(image_files)
        return selected if limit == 0 else selected[:limit]

    @staticmethod
    def _transport_status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @classmethod
    def _is_transient_transport_error(cls, exc: BaseException) -> bool:
        if getattr(exc, 'retry_budget_closed', False):
            return False
        if cls._transport_status_code(exc) in TRANSIENT_ACTOR_STATUS_CODES:
            return True
        current: BaseException | None = exc
        while current is not None:
            if current.__class__.__name__ in TRANSIENT_ACTOR_ERROR_NAMES:
                return True
            current = current.__cause__
        return False

    def _chat_completions_create(self, **request):
        max_retries = self._actor_transport_max_retries()
        base_delay = self._actor_transport_retry_base_delay()
        for retry_index in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                journal = getattr(self.client, '_action_recovery_journal', None)
                if journal is not None:
                    self._last_checkpoint_request_replay = journal.last_replayed
                return response
            except Exception as exc:
                if (
                    retry_index >= max_retries
                    or not self._is_transient_transport_error(exc)
                ):
                    raise
                delay = base_delay * (2 ** retry_index)
                status = self._transport_status_code(exc)
                status_label = status if status is not None else "connection"
                print(
                    "Actor transport retry "
                    f"{retry_index + 1}/{max_retries} after status="
                    f"{status_label}; waiting {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if delay:
                    time.sleep(delay)

    def _uses_upstream_close_source_protocol(self) -> bool:
        return (
            self.model_type == "close_source"
            and "gemini_direct" not in str(getattr(self, "model_name", "")).lower()
        )

    def _uses_aifast_transport_compat(self) -> bool:
        return is_aifast_endpoint(getattr(self, "api_base", ""))

    def accepts_strict_whole_json_output(self) -> bool:
        return (
            actor_output_contract(getattr(self, "api_base", ""))
            == AIFAST_OUTPUT_CONTRACT
        )

    def generation_protocol(self) -> dict:
        """Return the effective actor decoding configuration for provenance."""

        clean_close_source = self._uses_upstream_close_source_protocol()
        return {
            "model_resolved": str(self.model_name),
            "actor_seed": None if clean_close_source else self._actor_seed(),
            "actor_temperature": (
                0.0 if clean_close_source else self._actor_temperature()
            ),
            "actor_top_p": (
                1.0 if clean_close_source else self._actor_top_p()
            ),
            "actor_max_completion_tokens": (
                512
                if clean_close_source
                else self._actor_max_completion_tokens()
            ),
            "actor_max_images": self._actor_max_images(),
            "actor_retry_seed_policy": (
                "none"
                if clean_close_source
                else "base_plus_validation_attempt_v1"
            ),
            "actor_transport_retry_policy": (
                actor_transport_retry_policy(
                    self._actor_transport_max_retries()
                )
            ),
            "actor_transport_max_retries": self._actor_transport_max_retries(),
            "actor_transport_retry_base_delay_seconds": (
                self._actor_transport_retry_base_delay()
            ),
            "actor_transport_compat": actor_transport_compat(
                getattr(self, "api_base", "")
            ),
            "actor_output_contract": actor_output_contract(
                getattr(self, "api_base", "")
            ),
            "provider_reasoning_capture": PROVIDER_REASONING_CAPTURE,
            "disable_thinking": bool(self.disable_thinking),
            "provider_mode": (
                "gemini_direct"
                if "gemini_direct" in str(self.model_name).lower()
                else self.model_type
            ),
        }

    def _reset_provider_reasoning_record(self) -> None:
        self._last_provider_reasoning_record = empty_provider_reasoning_record()

    def _capture_provider_reasoning_record(self, message: Any) -> None:
        self._last_provider_reasoning_record = provider_reasoning_record(
            message,
            secrets=tuple(
                str(item)
                for item in getattr(self, "_provider_reasoning_secrets", ())
                if str(item)
            ),
        )

    def consume_provider_reasoning_record(self) -> dict[str, Any]:
        record = dict(
            getattr(
                self,
                "_last_provider_reasoning_record",
                empty_provider_reasoning_record(),
            )
        )
        self._reset_provider_reasoning_record()
        return record

    def _prepare_gen_args(self, gen_args=None):
        prepared = dict(gen_args or {})
        if self._uses_upstream_close_source_protocol():
            prepared.pop("seed", None)
            prepared["temperature"] = 0.0
            prepared["top_p"] = 1.0
            prepared["max_completion_tokens"] = 512
            if self._uses_aifast_transport_compat():
                prepared["tools"] = [AIFAST_DUMMY_TOOL]
                prepared["tool_choice"] = "none"
            return prepared
        prepared.setdefault("seed", self._actor_seed())
        if "ISBENCH_ACTOR_TEMPERATURE" in os.environ:
            prepared["temperature"] = self._actor_temperature()
        else:
            prepared.setdefault("temperature", self._actor_temperature())
        if "ISBENCH_ACTOR_TOP_P" in os.environ:
            prepared["top_p"] = self._actor_top_p()
        else:
            prepared.setdefault("top_p", self._actor_top_p())
        if "ISBENCH_AGENT_MAX_COMPLETION_TOKENS" in os.environ:
            prepared["max_completion_tokens"] = self._actor_max_completion_tokens()
        else:
            prepared.setdefault(
                "max_completion_tokens", self._actor_max_completion_tokens()
            )
        model_extra_body = dict(getattr(self, "model_extra_body", {}) or {})
        if model_extra_body:
            extra_body = dict(prepared.get("extra_body") or {})
            extra_body.update(model_extra_body)
            prepared["extra_body"] = extra_body
        if self._uses_aifast_transport_compat():
            prepared["tools"] = [AIFAST_DUMMY_TOOL]
            prepared["tool_choice"] = "none"
        return prepared
        
    def model(self, prompt, image_file: List[str] | str = None, gen_args=None): 
        self._reset_provider_reasoning_record()
        gen_args = self._prepare_gen_args(gen_args)
        if not image_file: 
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            chat_completion = self._chat_completions_create(
                    messages=messages,
                    model=self.model_name,
                    **gen_args
                )
            self._capture_provider_reasoning_record(
                chat_completion.choices[0].message
            )
            return _require_nonempty_actor_content(
                chat_completion.choices[0].message.content,
                provider="OpenAI-compatible",
            )
        else:
            if isinstance(image_file, str):  # support single and multi image
                image_file = [image_file]   
            image_file = self._select_actor_images(list(image_file))
            if 'gemini_direct' in self.model_name.lower(): # support gemini api
                parts=[
                        types.Part.from_text(text=prompt)
                    ]
                for image in image_file:
                    image_base64 = read_image(image)
                    image_type = "image/png"
                    image_content = types.Part.from_bytes(
                        data=image_base64,
                        mime_type=image_type,
                    )
                    parts.append(image_content)
                contents = [
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ]
                request_config = types.GenerateContentConfig(
                    temperature=float(gen_args["temperature"]),
                    top_p=float(gen_args["top_p"]),
                    seed=int(gen_args["seed"]),
                    max_output_tokens=int(gen_args["max_completion_tokens"]),
                    safety_settings=self.gemini_safety_settings,
                )

                result = ""
                for chunk in self.client.models.generate_content_stream(
                    model = self.model_name,
                    contents = contents,
                    config=request_config,
                    ):
                    if isinstance(chunk.text, str):
                        result += chunk.text
                return _require_nonempty_actor_content(
                    result,
                    provider="Gemini",
                )
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ]
                for image in image_file:
                    image_base64 = encode_image(image)
                    image_type = guess_image_type_from_base64(image_base64)
                    messages[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base64}"
                            },
                        }
                    )
                chat_completion = self._chat_completions_create(
                        messages=messages,
                        model=self.model_name,
                        **gen_args
                    )
                self._capture_provider_reasoning_record(
                    chat_completion.choices[0].message
                )
                return _require_nonempty_actor_content(
                    chat_completion.choices[0].message.content,
                    provider="OpenAI-compatible",
                )
