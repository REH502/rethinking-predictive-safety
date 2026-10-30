"""Probe OpenAI-compatible VLM endpoints for ISBench direct-vision WM use.

This is a diagnostic helper only. It does not score predictions and should not
be mixed into Tier1/Tier2/Tier3 metrics.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable


EXCLUDED_VISION_MODEL_MARKERS = (
    "embedding",
    "seedream",
    "seedance",
    "i2v",
    "t2i",
    "t2v",
    "i2i",
    "3d",
)
KNOWN_MULTIMODAL_CHAT_MARKERS = (
    "vision",
    "doubao-seed-2-0",
    "doubao-seed-code-preview",
)


def select_vision_chat_candidates(model_ids: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    for model_id in model_ids:
        normalized = str(model_id).lower()
        if not any(marker in normalized for marker in KNOWN_MULTIMODAL_CHAT_MARKERS):
            continue
        if any(marker in normalized for marker in EXCLUDED_VISION_MODEL_MARKERS):
            continue
        candidates.append(str(model_id))
    return sorted(candidates)


def probe_models(
    *,
    api_key: str | None,
    api_url: str | None,
    image_path: Path,
    models: list[str] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Probe requires --api-key or ARK_API_KEY/OPENAI_API_KEY.")
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
    if api_url:
        client_kwargs["base_url"] = api_url
    client = OpenAI(**client_kwargs)
    if models is None:
        listed = client.models.list()
        model_ids = [item.id for item in listed.data]
        models = select_vision_chat_candidates(model_ids)
    image_url = _image_data_url(image_path)
    results: list[dict[str, Any]] = []
    for model in models:
        result = {
            "model": model,
            "api_url": api_url or "",
            "ok": False,
            "error_type": "",
            "error_message": "",
            "response_text": "",
            "usage": {},
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": 'Return JSON {"ok": true} if you can process this image.',
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=64,
            )
            result["ok"] = True
            result["response_text"] = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            result["usage"] = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
        results.append(result)
    return {
        "api_url": api_url or "",
        "image_path": str(image_path),
        "candidate_count": len(models),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe VLM models for ISBench direct-vision WM.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-url", "--base-url", dest="api_url", default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)

    api_key = args.api_key or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    payload = probe_models(
        api_key=api_key,
        api_url=args.api_url,
        image_path=args.image,
        models=list(args.models) if args.models else None,
        timeout_seconds=args.timeout_seconds,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


def _image_data_url(path: Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


if __name__ == "__main__":
    main()
