"""Pre-generate RoboSafe-style split-aware train-policies for HarnessAudit tasks.

For each held-out test task, uses only training split tasks as few-shot examples,
calls an LLM to generate access_rules from visible task fields, validates format,
and saves split-aware cache files. The held-out test task's ground-truth
access_rules are not used as generation input.

Usage:
    python -m safepred_guard.generate_train_policies
    python -m safepred_guard.generate_train_policies --task fin-t4
    python -m safepred_guard.generate_train_policies --domain finance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_agent.credentials import load_workspace_credentials
from multi_agent.resource_semantics import (
    LEGACY_OFFICIAL_RESOURCE_SEMANTICS,
    RESOURCE_SEMANTICS_MODES,
    normalize_resource_semantics_mode,
)

from .config import find_workspace_root
from .policies import (
    _MODEL_INPUT_BOUNDARY_VERSION,
    _POLICY_EXTRACTION_PROMPT_VERSION,
    _TRAIN_POLICY_CACHE_SCHEMA_VERSION,
    _build_extraction_prompt,
    _find_tasks_dir,
    _load_task_yaml,
    _prompt_hash,
    _sanitize_task_for_model_input,
    _save_split_precomputed_policies,
    build_train_policy_context,
)

WORKSPACE_ROOT = find_workspace_root(Path(__file__).resolve())
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "train_policies"


def _load_env() -> None:
    credentials_path = load_workspace_credentials(Path(__file__), override=False)
    if credentials_path.is_file():
        print(f"  Loaded credentials from {credentials_path}")
    # Ensure CUSTOM_ variants are set (SafePred convention)
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("CUSTOM_API_KEY"):
        os.environ["CUSTOM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("CUSTOM_API_URL"):
        base = os.environ["OPENAI_BASE_URL"].rstrip("/")
        os.environ["CUSTOM_API_URL"] = base if base.endswith("/v1") else f"{base}/v1"


def _repair_json(text: str) -> dict[str, Any]:
    """Attempt to repair malformed JSON from LLM output."""
    import re

    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try straight parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix unterminated strings: find lines with odd number of unescaped quotes and close them
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        # Count unescaped quotes
        in_string = False
        escaped = False
        quote_count = 0
        for ch in stripped:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                quote_count += 1
        if quote_count % 2 != 0:
            # Unterminated string — add closing quote
            if stripped.endswith(","):
                stripped = stripped[:-1] + '",'
            else:
                stripped = stripped + '"'
        fixed_lines.append(stripped)
    text = "\n".join(fixed_lines)

    # Try again
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Count braces/brackets and close them
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    # Remove trailing commas before closing
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Close open structures
    text = text.rstrip()
    while open_brackets > 0 or open_braces > 0:
        if open_brackets > open_braces:
            text += "]"
            open_brackets -= 1
        elif open_braces > open_brackets:
            text += "}"
            open_braces -= 1
        else:
            # Both open: need to close inner first
            if text.rstrip().endswith("["):
                text += "]"
                open_brackets -= 1
            elif text.rstrip().endswith("{"):
                text += "}"
                open_braces -= 1
            else:
                text += "}"
                open_braces -= 1
                if open_brackets:
                    text += "]"
                    open_brackets -= 1

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract just the access_rules array
    match = re.search(r'"access_rules"\s*:\s*\[', text)
    if match:
        start = match.start()
        # Find the matching closing bracket
        depth = 0
        i = match.end() - 1  # position of [
        for j in range(i, len(text)):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    # Try to extract just this portion
                    fragment = text[start: j + 1]
                    try:
                        return json.loads("{" + fragment + "}")
                    except json.JSONDecodeError:
                        break
        # If that fails, try to extract each rule individually
        rules_text = text[match.end():]
        rules: list[dict[str, Any]] = []
        brace_depth = 0
        current_rule = ""
        for ch in rules_text:
            current_rule += ch
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    try:
                        rule = json.loads(current_rule.strip().rstrip(","))
                        rules.append(rule)
                    except json.JSONDecodeError:
                        pass
                    current_rule = ""
        if rules:
            return {"access_rules": rules}

    raise ValueError(f"Unable to parse LLM output as JSON. Raw output saved for debugging.")


def _call_extraction_llm(prompt: str, model: str, debug_dir: Path | None = None) -> tuple[dict[str, Any], str]:
    api_key = os.environ.get("CUSTOM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    api_url = (os.environ.get("CUSTOM_API_URL") or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not api_url.endswith("/v1"):
        api_url = f"{api_url}/v1"

    if not api_key or not api_url:
        raise RuntimeError("API credentials not configured")

    import requests

    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a safety policy extraction system. Output ONLY valid JSON. Do NOT use markdown fences. Ensure all strings are properly closed and all braces/brackets are balanced.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    model_lower = model.lower()
    if "qwen3" in model_lower:
        request_payload["chat_template_kwargs"] = {"enable_thinking": False}
    if "deepseek-v4" in model_lower:
        request_payload["thinking"] = {"type": "disabled"}

    response = requests.post(
        f"{api_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request_payload,
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    raw_content: str = result["choices"][0]["message"]["content"]

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "raw_response.txt").write_text(raw_content, encoding="utf-8")

    return _repair_json(raw_content), raw_content


def _validate_access_rules(
    rules: list[dict[str, Any]],
    *,
    resource_semantics_mode: str | None = None,
) -> tuple[bool, list[str]]:
    semantics_mode = normalize_resource_semantics_mode(resource_semantics_mode)
    errors: list[str] = []
    valid_access_types = {
        "information.data_leak",
        "operation.resource",
        "operation.tool_call",
        "information.communication",
    }
    for i, rule in enumerate(rules):
        prefix = f"Rule[{i}]"
        if not isinstance(rule, dict):
            errors.append(f"{prefix}: not a dict")
            continue
        for key in ("role", "access_type", "match"):
            if key not in rule:
                errors.append(f"{prefix}: missing '{key}'")
        access_type = rule.get("access_type", "")
        if access_type not in valid_access_types:
            errors.append(f"{prefix}: invalid access_type '{access_type}'")
        if access_type == "operation.resource":
            effect = rule.get("resource_effect")
            if semantics_mode == LEGACY_OFFICIAL_RESOURCE_SEMANTICS:
                if "resource_effect" in rule:
                    errors.append(
                        f"{prefix}: resource_effect must be omitted for legacy_official_v1"
                    )
            elif effect not in {"allow_scope", "deny_match", "deny_all"}:
                errors.append(
                    f"{prefix}: operation.resource requires explicit resource_effect"
                )
        match = rule.get("match", {})
        if not isinstance(match, dict):
            errors.append(f"{prefix}: match is not a dict")
        elif access_type == "operation.resource":
            tool = match.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                errors.append(f"{prefix}: operation.resource requires a non-empty tool")
            allowed_args = match.get("allowed_args")
            if (
                semantics_mode == LEGACY_OFFICIAL_RESOURCE_SEMANTICS
                and allowed_args is not None
                and not isinstance(allowed_args, dict)
            ):
                errors.append(
                    f"{prefix}: legacy allowed_args must be an object, null, or absent"
                )
        severity = rule.get("severity", "")
        if severity not in ("high", "medium", "low", ""):
            errors.append(f"{prefix}: invalid severity '{severity}'")
    return len(errors) == 0, errors


def _convert_to_safepred_policies(
    task: dict[str, Any],
    access_rules: list[dict[str, Any]],
    *,
    resource_semantics_mode: str | None = None,
) -> list[dict[str, Any]]:
    semantics_mode = normalize_resource_semantics_mode(resource_semantics_mode)
    root_text = str(WORKSPACE_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from SafePred.utils.harnessaudit_policy import convert_harnessaudit_task_to_policies

    synthetic_task = _sanitize_task_for_model_input(task)
    synthetic_task["access_rules"] = access_rules
    policies = convert_harnessaudit_task_to_policies(
        synthetic_task,
        include_access_rules=True,
        include_tool_necessity=False,
        resource_semantics_mode=semantics_mode,
    )
    for p in policies:
        harness = p.setdefault("harnessaudit", {})
        harness["source"] = "train_policy"
        harness["resource_semantics_mode"] = semantics_mode
    return policies


def _validate_safepred_policies(policies: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required_keys = {"policy_id", "name", "policy_description", "risk_level", "violation_code"}
    valid_risk = {"high", "medium", "low"}
    valid_codes = {"V-ID", "V-IC", "V-OT", "V-OR", "V-UNK"}
    seen_ids: set[str] = set()

    for i, p in enumerate(policies):
        prefix = f"Policy[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{prefix}: not a dict")
            continue
        pid = p.get("policy_id", "")
        if pid in seen_ids:
            errors.append(f"{prefix}: duplicate policy_id '{pid}'")
        seen_ids.add(pid)
        for key in required_keys:
            if key not in p:
                errors.append(f"{prefix}: missing '{key}'")
        if p.get("risk_level") not in valid_risk:
            errors.append(f"{prefix}: invalid risk_level '{p.get('risk_level')}'")
        if p.get("violation_code") not in valid_codes:
            errors.append(f"{prefix}: invalid violation_code '{p.get('violation_code')}'")
        harness = p.get("harnessaudit", {})
        if not isinstance(harness, dict):
            errors.append(f"{prefix}: harnessaudit is not a dict")

    return len(errors) == 0, errors


def _discover_all_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    tasks_dir = _find_tasks_dir()
    for yaml_file in sorted(tasks_dir.rglob("*.yaml")):
        import yaml

        task = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        if task.get("task_id"):
            tasks.append(task)
    return tasks


def _generate_for_task(
    task: dict[str, Any],
    model: str,
    dry_run: bool = False,
    *,
    resource_semantics_mode: str | None = None,
) -> dict[str, Any] | None:
    task_id = task.get("task_id", "unknown")
    domain = task.get("domain", "")
    category = task.get("category", "")
    semantics_mode = normalize_resource_semantics_mode(resource_semantics_mode)

    context = build_train_policy_context(task)
    train_ids = context["training_task_ids"]
    training_tasks: list[dict[str, Any]] = []
    for tid in train_ids:
        t = _load_task_yaml(tid)
        if t and t.get("access_rules"):
            training_tasks.append(t)

    result = {
        "task_id": task_id,
        "domain": domain,
        "category": category,
        "schema_version": 3,
        "method": context["method"],
        "split_id": context["split_id"],
        "test_task_id": task_id,
        "training_task_ids": context["training_task_ids"],
        "training_task_count": context["training_task_count"],
        "few_shot_training_task_ids": [t.get("task_id") for t in training_tasks],
        "few_shot_training_task_count": len(training_tasks),
        "training_categories": context["training_categories"],
        "input_includes_test_access_rules": False,
        "input_includes_test_tool_necessity": False,
        "input_includes_tool_necessity_derivatives": False,
        "include_test_tool_necessity_policies": False,
        "model_input_boundary_version": _MODEL_INPUT_BOUNDARY_VERSION,
        "policy_extraction_prompt_version": _POLICY_EXTRACTION_PROMPT_VERSION,
        "policy_cache_schema_version": _TRAIN_POLICY_CACHE_SCHEMA_VERSION,
        "test_visible_fields": context["test_visible_fields"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "resource_semantics_mode": semantics_mode,
    }

    if len(training_tasks) < 2:
        result["status"] = "skipped"
        result["error"] = f"Only {len(training_tasks)} training tasks available (need >= 2)"
        return result

    if dry_run:
        result["status"] = "dry_run"
        result["prompt_size_chars"] = len(
            _build_extraction_prompt(
                task,
                training_tasks,
                context=context,
                resource_semantics_mode=semantics_mode,
            )
        )
        return result

    # Call LLM with retries
    print(f"    Training tasks (first 5): {[t.get('task_id') for t in training_tasks[:5]]}")
    print(f"    Calling LLM ({model})...", end=" ", flush=True)
    t0 = time.time()

    debug_dir = OUTPUT_DIR / "debug" / task_id
    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"\n    Retry {attempt}/{max_retries}...", end=" ", flush=True)
            prompt = _build_extraction_prompt(
                task,
                training_tasks,
                context=context,
                resource_semantics_mode=semantics_mode,
            )
            extracted, raw_response = _call_extraction_llm(prompt, model, debug_dir=debug_dir)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            access_rules = extracted.get("access_rules", [])
            print(f"    Generated {len(access_rules)} access_rules")

            # Validate access_rules format
            valid, errors = _validate_access_rules(
                access_rules,
                resource_semantics_mode=semantics_mode,
            )
            result["access_rules_valid"] = valid
            result["access_rules_errors"] = errors
            result["access_rules_count"] = len(access_rules)

            if not valid:
                print(f"    WARNING: access_rules validation errors: {errors[:5]}")

            # Convert to SafePred policies
            safepred_policies = _convert_to_safepred_policies(
                task,
                access_rules,
                resource_semantics_mode=semantics_mode,
            )
            valid_sp, sp_errors = _validate_safepred_policies(safepred_policies)
            result["safepred_policy_count"] = len(safepred_policies)
            result["safepred_policies_valid"] = valid_sp
            result["safepred_policies_errors"] = sp_errors

            if not valid_sp:
                print(f"    WARNING: SafePred policy validation errors: {sp_errors[:5]}")

            # Store results
            result["access_rules"] = access_rules
            result["safepred_policies"] = safepred_policies
            result["status"] = "success"
            _save_split_precomputed_policies(
                task,
                context,
                extracted,
                safepred_policies,
                model_name=model,
                prompt_hash=_prompt_hash(prompt),
                resource_semantics_mode=semantics_mode,
            )

            # Compute coverage comparison
            result["has_data_leak_rules"] = any(
                r.get("access_type") == "information.data_leak" for r in access_rules
            )
            result["has_resource_rules"] = any(
                r.get("access_type") == "operation.resource" for r in access_rules
            )

            # Save raw response for debugging
            if debug_dir.exists():
                (debug_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")

            break  # Success — exit retry loop

        except Exception as exc:
            last_error = exc
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            if attempt < max_retries:
                time.sleep(2)  # Brief pause before retry
            else:
                result["status"] = "error"
                result["error"] = str(last_error)

    return result


def _collect_ground_truth_stats(task: dict[str, Any]) -> dict[str, Any]:
    rules = task.get("access_rules", [])
    return {
        "gt_access_rules_count": len(rules),
        "gt_data_leak_count": sum(1 for r in rules if r.get("access_type") == "information.data_leak"),
        "gt_resource_count": sum(1 for r in rules if r.get("access_type") == "operation.resource"),
        "gt_tool_call_count": sum(1 for r in rules if r.get("access_type") == "operation.tool_call"),
        "gt_communication_count": sum(1 for r in rules if r.get("access_type") == "information.communication"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate train-policies for HarnessAudit tasks")
    parser.add_argument("--task", type=str, help="Generate for a specific task ID only")
    parser.add_argument("--domain", type=str, help="Generate for a specific domain only")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="LLM model for policy extraction")
    parser.add_argument(
        "--resource-semantics-mode",
        choices=sorted(RESOURCE_SEMANTICS_MODES),
        default="explicit_v2",
        help="Resource-rule semantics shared by extraction, policy conversion, and cache identity",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show training splits without calling LLM")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to process")
    args = parser.parse_args()

    _load_env()
    resource_semantics_mode = normalize_resource_semantics_mode(
        args.resource_semantics_mode
    )
    os.environ["SAFEPRED_RESOURCE_SEMANTICS_MODE"] = resource_semantics_mode

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = _discover_all_tasks()
    print(f"Discovered {len(all_tasks)} tasks total")

    # Filter
    if args.task:
        all_tasks = [t for t in all_tasks if t.get("task_id") == args.task]
        if not all_tasks:
            print(f"Task '{args.task}' not found")
            return
    if args.domain:
        all_tasks = [t for t in all_tasks if t.get("domain") == args.domain]

    print(f"Processing {len(all_tasks)} tasks")

    manifest: list[dict[str, Any]] = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, task in enumerate(all_tasks):
        if args.limit and i >= args.limit:
            break
        task_id = task.get("task_id")
        domain = task.get("domain")
        category = task.get("category")
        print(f"\n[{i+1}/{len(all_tasks)}] {task_id} (domain={domain}, category={category})")

        gt_stats = _collect_ground_truth_stats(task)

        result = _generate_for_task(
            task,
            model=args.model,
            dry_run=args.dry_run,
            resource_semantics_mode=resource_semantics_mode,
        )
        if result is None:
            continue

        result.update(gt_stats)

        # Save individual result
        out_path = OUTPUT_DIR / f"{task_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    Saved to {out_path}")

        manifest.append({
            "task_id": task_id,
            "domain": domain,
            "category": category,
            "status": result.get("status"),
            "training_task_ids": result.get("training_task_ids", []),
            "access_rules_count": result.get("access_rules_count", 0),
            "safepred_policy_count": result.get("safepred_policy_count", 0),
            "access_rules_valid": result.get("access_rules_valid"),
            "safepred_policies_valid": result.get("safepred_policies_valid"),
            "error": result.get("error"),
            "resource_semantics_mode": result.get("resource_semantics_mode"),
        })

        status = result.get("status")
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skip_count += 1
        elif status == "dry_run":
            pass  # don't count dry runs
        else:
            error_count += 1

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "resource_semantics_mode": resource_semantics_mode,
        "total_tasks": len(manifest),
        "success_count": success_count,
        "skip_count": skip_count,
        "error_count": error_count,
        "tasks": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved to {manifest_path}")
    print(f"Summary: {success_count} success, {skip_count} skipped, {error_count} errors")


if __name__ == "__main__":
    main()
