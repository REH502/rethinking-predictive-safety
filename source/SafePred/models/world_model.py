"""
World Model Module for Safety-TS-LMA.

Simulates state transitions for internal tree search without executing
actions in the real environment. Supports multiple simulation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import copy
import json
import math
import os
import re
from ..utils.logger import get_logger
from ..utils.parsers import parse_json_from_text, parse_json_array_from_text, normalize_action
from ..utils.formatters import format_object_to_string
from ..utils.xml_parser import parse_xml_output, convert_xml_to_world_model_format
from ..utils.structured_text_parser import parse_structured_text, convert_structured_text_to_world_model_format

logger = get_logger("SafePred.WorldModel")


def _get_state_summary(linearized_accessibility_tree: str, max_chars: int = 200) -> str:
    """
    Generate a summary of the state for debugging purposes.
    
    Args:
        linearized_accessibility_tree: The linearized accessibility tree string
        max_chars: Maximum number of characters to include in summary
        
    Returns:
        A summary string with key information
    """
    if not linearized_accessibility_tree:
        return "empty"
    
    # Get first few lines
    lines = linearized_accessibility_tree.split('\n')
    first_lines = '\n'.join(lines[:3]) if len(lines) > 3 else linearized_accessibility_tree[:max_chars]
    summary = f"first_lines={first_lines[:max_chars]}"
    
    # Check for key UI elements that might indicate state
    key_elements = []
    if "push-button\tOK" in linearized_accessibility_tree:
        key_elements.append("OK_button")
    if "push-button\tCancel" in linearized_accessibility_tree:
        key_elements.append("Cancel_button")
    if "push-button\tReset" in linearized_accessibility_tree:
        key_elements.append("Reset_button")
    if "menu-item\tParagraph" in linearized_accessibility_tree:
        key_elements.append("Paragraph_menu")
    if "menu\tFormat" in linearized_accessibility_tree:
        key_elements.append("Format_menu")
    
    if key_elements:
        summary += f", key_elements={','.join(key_elements)}"
    
    return summary


class BaseWorldModel(ABC):
    """
    Abstract base class for world models.
    
    World models simulate state transitions: s' = simulate(s, a)
    without actually executing actions in the real environment.
    """
    
    @abstractmethod
    def simulate(self, state: Any, action: Any) -> Any:
        """
        Simulate the next state given current state and action.
        
        Args:
            state: Current state representation
            action: Action to simulate
        
        Returns:
            Predicted next state
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: Any) -> List[Any]:
        """
        Get valid actions for a given state.
        
        Args:
            state: Current state representation
        
        Returns:
            List of valid actions
        """
        pass
    
    def simulate_batch(self, states: List[Any], actions: List[Any]) -> List[Any]:
        """
        Simulate state transitions for a batch of state-action pairs.
        
        Args:
            states: List of current states
            actions: List of actions
        
        Returns:
            List of predicted next states
        """
        return [self.simulate(s, a) for s, a in zip(states, actions)]


class LLMBasedWorldModel(BaseWorldModel):
    """
    LLM-based world model using language models to predict state transitions.
    
    Uses LLMs (e.g., Qwen, GPT) to generate predictions about next states
    given current state and action.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: str = "cuda",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        use_few_shot: bool = False,
        use_chain_of_thought: bool = False,
        require_json_output: bool = True,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        use_state_delta: bool = True,
        prediction_steps: int = 1,
        risk_threshold: float = 0.7,
    ):
        """
        Initialize LLM-based world model.
        
        Args:
            model: Pre-loaded model object
            model_name: Name/path of the model to load
            device: Device to run model on
            temperature: Temperature for generation (overrides config if provided)
            max_tokens: Maximum tokens for generation (overrides config if provided)
            api_key: LLM API key (overrides config if provided)
            api_url: LLM API URL (overrides config if provided)
            provider: LLM provider type (overrides config if provided)
            timeout: Request timeout in seconds (overrides config if provided)
            llm_config: Dictionary with LLM configuration (from SafetyConfig.get_llm_config())
            use_few_shot: Whether to use few-shot learning in prompts (default: False)
            use_chain_of_thought: Whether to use chain-of-thought reasoning (default: False)
            require_json_output: Whether to require JSON format output (default: True)
            few_shot_examples: List of example dicts with keys: 'state', 'action', 'next_state'
            use_state_delta: Whether to use state delta mode - predicts only changes instead of full state (default: True)
            prediction_steps: Number of steps to predict ahead (1 = single-step, >1 = multi-step) (default: 1)
        """
        self.model = model
        self.device = device
        self._model_loaded = model is not None
        self.risk_threshold = self._validate_risk_threshold(risk_threshold)
        
        # Load LLM configuration from llm_config or individual parameters
        if llm_config:
            self.api_key = api_key or llm_config.get("api_key")
            self.api_url = api_url or llm_config.get("api_url")
            self.model_name = model_name or llm_config.get("model_name")
            # Provider from llm_config takes precedence, then explicit parameter, then default
            self.provider = llm_config.get("provider") or provider or "openai"
            self.temperature = temperature if temperature is not None else llm_config.get("temperature", 0.7)
            self.max_tokens = max_tokens if max_tokens is not None else llm_config.get("max_tokens", 512)
            self.timeout = timeout if timeout is not None else llm_config.get("timeout", 30)
        else:
            # Fallback to individual parameters or defaults
            self.api_key = api_key
            self.api_url = api_url
            self.model_name = model_name
            self.provider = provider or "openai"
            self.temperature = temperature if temperature is not None else 0.7
            self.max_tokens = max_tokens if max_tokens is not None else 512
            self.timeout = timeout if timeout is not None else 30
        
        # Prompt optimization settings: directly use passed parameters (not from config)
        self.use_few_shot = use_few_shot
        self.use_chain_of_thought = use_chain_of_thought
        self.require_json_output = require_json_output
        self.few_shot_examples = few_shot_examples or []
        
        # State delta mode: predict only changes instead of full state (more efficient)
        self.use_state_delta = use_state_delta if use_state_delta is not None else True  # Default to state delta mode for efficiency
        
        # Initialize token usage tracking
        self._last_token_usage = None  # Store token usage from last LLM API call
        
        # Multi-step prediction configuration
        if llm_config:
            self.prediction_steps = llm_config.get("prediction_steps", prediction_steps)
            self.log_prompt = llm_config.get("log_prompt", False)  # Read from config
            # Policy reference configuration
            self.show_policy_references = llm_config.get("show_policy_references", False)  # Default: False (do not show references)
            # Risk score calculation configuration
            self.risk_score_violation_penalty = llm_config.get("risk_score_violation_penalty", 0.5)
            # Prediction type configuration
            self.enable_short_term_prediction = llm_config.get("enable_short_term_prediction", True)  # Default: True
            self.enable_long_term_prediction = llm_config.get("enable_long_term_prediction", True)  # Default: True
            self.strict_state_delta_output = bool(
                llm_config.get("strict_state_delta_output", False)
            )
            self.max_semantic_attempts = int(
                llm_config.get("max_semantic_attempts", 3)
            )
            self.transport_max_attempts = int(
                llm_config.get("transport_max_attempts", 3)
            )
        else:
            self.prediction_steps = prediction_steps
            self.log_prompt = False  # Default to False
            self.show_policy_references = False  # Default: False (do not show references)
            # Default risk score values
            self.risk_score_violation_penalty = 0.5
            # Default prediction type values
            self.enable_short_term_prediction = True  # Default: True
            self.enable_long_term_prediction = True  # Default: True
            self.strict_state_delta_output = False
            self.max_semantic_attempts = 3
            self.transport_max_attempts = 3

        if self.max_semantic_attempts < 1:
            raise ValueError("max_semantic_attempts must be at least 1")
        if self.transport_max_attempts < 1:
            raise ValueError("transport_max_attempts must be at least 1")
        
        # Log prediction type configuration
        logger.info(f"[World Model] Prediction type configuration: short_term={'enabled' if self.enable_short_term_prediction else 'disabled'}, long_term={'enabled' if self.enable_long_term_prediction else 'disabled'}")
        
        if model is None and self.model_name:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the LLM model configuration.
        
        Supports API-based models (openai, qwen, custom).
        Configuration is loaded from config.yaml via SafetyConfig.
        """
        # API-based model (OpenAI, Qwen, custom)
        # Model will be called via API, no need to load locally
        # API credentials must be set, otherwise will raise error when used
        if not self.api_key or not self.api_url:
            self._use_api = False
        else:
            self._use_api = True
        self._model_loaded = True
    
    def _create_prompt(self, state: Any, action: Any, policies: Optional[List[Dict[str, Any]]] = None, plan_text: Optional[str] = None) -> str:
        """
        Create a prompt for LLM to predict next state.
        
        Uses unified prompt templates from prompts module.
        Prefers state delta mode for efficiency (predicts only changes).
        
        Args:
            state: Current state (already contains complete chat_history)
            action: Action to simulate
            policies: Optional list of policy dictionaries for risk evaluation
            plan_text: Optional execution plan text (only used when violation detected for optimization_guidance generation, not for progress tracking)
        
        Returns:
            Formatted prompt string
        """
        from .prompts import PromptTemplates
        
        action_str = self._to_string(action)
        
        # Use state delta mode by default (more efficient)
        if self.use_state_delta:
            # Pass few-shot examples if available (for experience replay)
            examples = self.few_shot_examples if (self.use_few_shot and self.few_shot_examples) else None
            # Pass compact state dict directly (not string) to world_model_state_delta
            # Current State already contains complete chat_history
            return PromptTemplates.world_model_state_delta(
                state=state,  # Pass compact state dict directly
                action=action_str,
                examples=examples,
                policies=policies,
                show_policy_references=self.show_policy_references,  # Read from config
                plan_text=plan_text,  # Pass plan_text only when violation detected (for optimization_guidance generation)
                enable_short_term_prediction=self.enable_short_term_prediction,  # Read from config
                enable_long_term_prediction=self.enable_long_term_prediction  # Read from config
            )
        
        # Fallback to other modes if state delta is disabled
        # Convert compact state to string for legacy prompts
        # Use full axtree_txt if available, otherwise fall back to key_elements
        if isinstance(state, dict):
            if 'axtree_txt' in state and state.get('axtree_txt'):
                state_str = PromptTemplates._format_state_with_full_tree(state)
            elif 'key_elements' in state:
                state_str = PromptTemplates._format_compact_state(state)
            else:
                state_str = self._to_string(state)
        else:
            state_str = self._to_string(state)
        
        if self.use_few_shot and self.few_shot_examples:
            return PromptTemplates.world_model_few_shot(
                state=state_str,
                action=action_str,
                examples=self.few_shot_examples
            )
        elif self.use_chain_of_thought:
            return PromptTemplates.world_model_chain_of_thought(
                state=state_str,
                action=action_str,
                require_json=self.require_json_output
            )
        elif self.require_json_output:
            return PromptTemplates.world_model_json(
                state=state_str,
                action=action_str
            )
        else:
            return PromptTemplates.world_model_basic(
                state=state_str,
                action=action_str
            )
    
    def predict_direct_multi_horizon(
        self,
        state: Any,
        action: Any,
        policies: Optional[List[Dict[str, Any]]] = None,
        max_horizon: int = 6,
        target_horizon: Optional[int] = None,
        max_tokens: int = 1800,
        state_scope: str = "initial",
    ) -> Dict[str, Any]:
        """Predict direct risk in one JSON-only LLM call.

        The call returns only ``target_horizon``.  For compatibility, callers
        that omit it get the requested maximum horizon, never a joint h1..hN
        response.
        """
        from .prompts import PromptTemplates
        from ..utils.llm_client import LLMClient, LLMInvalidContentError

        max_horizon = max(1, int(max_horizon))
        target_horizon = max_horizon if target_horizon is None else max(1, int(target_horizon))
        if target_horizon > max_horizon:
            raise ValueError(
                "target_horizon cannot exceed max_horizon: "
                f"target={target_horizon}, max={max_horizon}"
            )
        prompt = PromptTemplates.world_model_direct_multi_horizon(
            state=state,
            action=self._to_string(action),
            policies=policies,
            max_horizon=max_horizon,
            target_horizon=target_horizon,
            show_policy_references=self.show_policy_references,
            state_scope=state_scope,
        )

        if not hasattr(self, "_llm_client"):
            self._llm_client = LLMClient(
                api_key=self.api_key,
                api_url=self.api_url,
                model_name=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=getattr(self, "transport_max_attempts", 3),
            )

        normalized = None
        validation_error: Exception | None = None
        for attempt in range(3):
            request_prompt = prompt
            if validation_error is not None:
                request_prompt += self._strict_json_retry_instruction(
                    validation_error,
                    scope=(
                        f"the complete h{target_horizon} single-horizon direct prediction object"
                        if target_horizon is not None
                        else f"the complete h1..h{max_horizon} direct prediction object"
                    ),
                )
            try:
                enable_thinking = attempt < 2
                generated_text = self._llm_client.generate(
                    request_prompt,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    enable_thinking=enable_thinking,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                    request_ledger_attempt=attempt + 1,
                )
            except LLMInvalidContentError as exc:
                validation_error = exc
                if attempt == 2:
                    raise
                logger.warning(
                    "[WorldModel] Direct multi-horizon response had invalid "
                    "assistant content; regenerating within the semantic "
                    "validation budget: %s",
                    exc,
                )
                continue
            except Exception as exc:
                if "response_format" not in str(exc).lower():
                    raise
                if os.environ.get("SAFEPRED_REQUIRE_RESPONSE_FORMAT") == "1":
                    raise RuntimeError(
                        "Formal prediction requires response_format=json_object; "
                        "the configured endpoint rejected it"
                    ) from exc
                logger.warning(
                    "[WorldModel] response_format unsupported for direct multi-horizon; retrying without it: %s",
                    exc,
                )
                generated_text = self._llm_client.generate(
                    request_prompt,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                    request_ledger_attempt=attempt + 1,
                )

            self._last_generated_text = generated_text
            if hasattr(self._llm_client, "_last_token_usage") and self._llm_client._last_token_usage:
                self._last_token_usage = self._llm_client._last_token_usage.copy()
            try:
                parsed = self._parse_json_output_with_required_keys(
                    generated_text,
                    {"component", "mode", "horizon_predictions"},
                )
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "[WorldModel] Failed to parse direct multi-horizon JSON. "
                        f"Raw text: {generated_text[:500]}"
                    )
                normalized = self._coerce_direct_multi_horizon(
                    parsed,
                    max_horizon,
                    target_horizon=target_horizon,
                    known_policy_ids=self._known_policy_ids(policies),
                )
                break
            except ValueError as exc:
                validation_error = exc
                if attempt == 2:
                    raise
                logger.warning(
                    "[WorldModel] Direct multi-horizon output failed strict validation; "
                    "regenerating with strict feedback: %s",
                    exc,
                )
        if normalized is None:
            raise RuntimeError("Direct multi-horizon prediction produced no validated output")
        self._last_direct_multi_horizon = normalized

        scores = [
            self._safe_float(item.get("risk_score"), default=0.0)
            for item in normalized["horizon_predictions"].values()
        ]
        violated_policy_ids = sorted(
            {
                str(policy_id)
                for item in normalized["horizon_predictions"].values()
                for policy_id in item.get("violated_policy_ids", [])
                if policy_id
            }
        )
        violation_codes = sorted(
            {
                str(code)
                for item in normalized["horizon_predictions"].values()
                for code in item.get("violation_codes", [])
                if str(code) in {"V-ID", "V-OT", "V-OR", "V-IC"}
            }
        )
        self._last_predicted_delta = {
            "risk_score": max(scores) if scores else 0.0,
            "risk_explanation": "Direct single-horizon prediction; see horizon_predictions.",
            "violated_policy_ids": violated_policy_ids,
            "violation_codes": violation_codes,
            "optimization_guidance": None,
            "direct_multi_horizon": normalized,
        }
        return normalized

    def predict_iterative_multi_horizon(
        self,
        state: Any,
        action: Any,
        policies: Optional[List[Dict[str, Any]]] = None,
        max_horizon: int = 6,
        max_tokens_per_step: int = 2400,
    ) -> Dict[str, Any]:
        """Roll the WM forward without reading real future actions or states."""
        from .prompts import PromptTemplates
        from ..utils.llm_client import LLMClient, LLMInvalidContentError

        if not isinstance(state, dict):
            raise ValueError("Iterative multi-horizon prediction requires a compact state dict")
        max_horizon = max(1, int(max_horizon))
        if not hasattr(self, "_llm_client"):
            self._llm_client = LLMClient(
                api_key=self.api_key,
                api_url=self.api_url,
                model_name=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=getattr(self, "transport_max_attempts", 3),
            )

        current_state = copy.deepcopy(state)
        current_action = copy.deepcopy(action)
        rollout_history: List[Dict[str, Any]] = []
        horizon_predictions: Dict[str, Dict[str, Any]] = {}
        cumulative_usage = {
            "call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for step_index in range(1, max_horizon + 1):
            prompt = PromptTemplates.world_model_iterative_rollout_step(
                state=current_state,
                action=current_action,
                policies=policies,
                step_index=step_index,
                max_horizon=max_horizon,
                rollout_history=rollout_history,
                show_policy_references=self.show_policy_references,
            )
            step_usage = {key: 0 for key in cumulative_usage}
            normalized = None
            validation_error: Exception | None = None
            semantic_attempts = max(
                1,
                int(getattr(self, "max_semantic_attempts", 3)),
            )
            token_budget = min(
                int(max_tokens_per_step),
                int(getattr(self, "max_tokens", max_tokens_per_step) or max_tokens_per_step),
            )
            for attempt in range(semantic_attempts):
                request_prompt = prompt
                if validation_error is not None:
                    request_prompt += self._strict_json_retry_instruction(
                        validation_error,
                        scope=f"the h{step_index} iterative rollout object",
                    )
                usage_before = self._client_usage_snapshot(self._llm_client)
                try:
                    enable_thinking = (
                        False
                        if getattr(self, "strict_state_delta_output", False)
                        else attempt < 2
                    )
                    generated_text = self._llm_client.generate(
                        request_prompt,
                        max_tokens=token_budget,
                        response_format={"type": "json_object"},
                        enable_thinking=enable_thinking,
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                        request_ledger_attempt=attempt + 1,
                    )
                except LLMInvalidContentError as exc:
                    usage_after = self._client_usage_snapshot(
                        self._llm_client
                    )
                    for usage_key in cumulative_usage:
                        value = int(
                            usage_after.get(usage_key, 0) or 0
                        ) - int(usage_before.get(usage_key, 0) or 0)
                        step_usage[usage_key] += value
                        cumulative_usage[usage_key] += value
                    validation_error = exc
                    if attempt == semantic_attempts - 1:
                        raise
                    logger.warning(
                        "[WorldModel] Iterative h%s response had invalid "
                        "assistant content; regenerating within the semantic "
                        "validation budget: %s",
                        step_index,
                        exc,
                    )
                    continue
                except Exception as exc:
                    if "response_format" not in str(exc).lower():
                        raise
                    if os.environ.get("SAFEPRED_REQUIRE_RESPONSE_FORMAT") == "1":
                        raise RuntimeError(
                            "Formal prediction requires response_format=json_object; "
                            "the configured endpoint rejected it"
                        ) from exc
                    generated_text = self._llm_client.generate(
                        request_prompt,
                        max_tokens=token_budget,
                        enable_thinking=enable_thinking,
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                        request_ledger_attempt=attempt + 1,
                    )
                usage_after = self._client_usage_snapshot(self._llm_client)
                for usage_key in cumulative_usage:
                    value = int(usage_after.get(usage_key, 0) or 0) - int(
                        usage_before.get(usage_key, 0) or 0
                    )
                    step_usage[usage_key] += value
                    cumulative_usage[usage_key] += value

                self._last_generated_text = generated_text
                try:
                    parsed = self._parse_json_output_with_required_keys(
                        generated_text,
                        {
                            "predicted_state_delta",
                            "predicted_next_action",
                            "predicted_risk",
                            "risk_score",
                            "confidence",
                            "violated_policy_ids",
                            "violation_codes",
                            "risk_explanation",
                        },
                    )
                    if not isinstance(parsed, dict):
                        raise ValueError(
                            f"[WorldModel] Iterative rollout h{step_index} returned invalid JSON: "
                            f"{generated_text[:500]}"
                        )
                    normalized = self._coerce_iterative_horizon_step(
                        parsed,
                        step_index=step_index,
                        max_horizon=max_horizon,
                        current_action=current_action,
                        known_policy_ids=self._known_policy_ids(policies),
                    )
                    break
                except ValueError as exc:
                    validation_error = exc
                    if attempt == semantic_attempts - 1:
                        raise
                    logger.warning(
                        "[WorldModel] Iterative h%s output failed strict validation; "
                        "regenerating with strict feedback: %s",
                        step_index,
                        exc,
                    )
            if normalized is None:
                raise RuntimeError(f"Iterative rollout h{step_index} produced no validated output")
            normalized["step_token_usage"] = step_usage
            normalized["cumulative_token_usage"] = copy.deepcopy(cumulative_usage)
            key = f"h{step_index}"
            horizon_predictions[key] = normalized
            rollout_history.append(
                {
                    "horizon": key,
                    "action": copy.deepcopy(current_action),
                    "predicted_state_delta": normalized["predicted_state_delta"],
                    "predicted_next_action": copy.deepcopy(normalized["predicted_next_action"]),
                    "predicted_risk": normalized["predicted_risk"],
                    "risk_score": normalized["risk_score"],
                    "violated_policy_ids": list(normalized["violated_policy_ids"]),
                    "violation_codes": list(normalized["violation_codes"]),
                    "step_token_usage": copy.deepcopy(step_usage),
                    "cumulative_token_usage": copy.deepcopy(cumulative_usage),
                }
            )
            if step_index < max_horizon:
                current_state = self._state_for_next_prediction(
                    previous_state=current_state,
                    predicted_state=normalized["predicted_state_delta"],
                    delta={
                        "semantic_delta": normalized["predicted_state_delta"],
                        "risk_score": normalized["risk_score"],
                        "violated_policy_ids": normalized["violated_policy_ids"],
                        "violation_codes": normalized["violation_codes"],
                    },
                )
                current_action = copy.deepcopy(normalized["predicted_next_action"])

        result = {
            "component": "World Model",
            "mode": "iterative_wm_rollout",
            "valid_prediction": True,
            "horizon_predictions": horizon_predictions,
            "rollout_history": rollout_history,
        }
        self._last_direct_multi_horizon = result
        self._last_predicted_delta = {
            "risk_score": max(item["risk_score"] for item in horizon_predictions.values()),
            "risk_explanation": "Iterative WM rollout; see horizon_predictions.",
            "violated_policy_ids": sorted(
                {
                    policy_id
                    for item in horizon_predictions.values()
                    for policy_id in item["violated_policy_ids"]
                }
            ),
            "violation_codes": sorted(
                {
                    code
                    for item in horizon_predictions.values()
                    for code in item["violation_codes"]
                }
            ),
            "iterative_wm_rollout": result,
        }
        return result

    def _coerce_direct_multi_horizon(
        self,
        parsed: Dict[str, Any],
        max_horizon: int,
        *,
        target_horizon: int | None = None,
        known_policy_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        raw_predictions = parsed.get("horizon_predictions") or {}
        if not isinstance(raw_predictions, dict):
            raise ValueError("Direct multi-horizon output is missing object field 'horizon_predictions'")
        if set(parsed) != {"component", "mode", "horizon_predictions"}:
            raise ValueError(
                "Direct single-horizon output must contain exactly component, mode, "
                "and horizon_predictions"
            )
        if str(parsed.get("component")) != "World Model":
            raise ValueError("Direct single-horizon component must equal 'World Model'")
        if str(parsed.get("mode")) != "direct_single_horizon":
            raise ValueError(
                "Direct single-horizon mode must equal 'direct_single_horizon'"
            )

        normalized: Dict[str, Any] = {
            "component": str(parsed.get("component") or "World Model"),
            "mode": "direct_single_horizon",
            "valid_prediction": True,
            "horizon_predictions": {},
        }
        valid_codes = {"V-ID", "V-OT", "V-OR", "V-IC"}
        target_horizon = max_horizon if target_horizon is None else int(target_horizon)
        requested_horizons = [target_horizon]
        expected_prediction_keys = {f"h{target_horizon}"}
        if set(raw_predictions) != expected_prediction_keys:
            raise ValueError(
                "Direct single-horizon output horizon keys must equal "
                f"{sorted(expected_prediction_keys)}, observed={sorted(raw_predictions)}"
            )
        for horizon in requested_horizons:
            key = f"h{horizon}"
            raw = raw_predictions.get(key)
            if not isinstance(raw, dict):
                raise ValueError(f"Direct multi-horizon output is missing object field '{key}'")
            required = {
                "predicted_risk",
                "risk_score",
                "confidence",
                "violated_policy_ids",
                "violation_codes",
                "risk_explanation",
                "optimization_guidance",
            }
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"Direct multi-horizon output {key} is missing fields: {missing}")
            if set(raw) != required:
                raise ValueError(
                    f"Direct single-horizon output {key} fields must equal "
                    f"{sorted(required)}, observed={sorted(raw)}"
                )
            risk_score = self._bounded_float(raw.get("risk_score"), field=f"{key}.risk_score")
            confidence = self._bounded_float(raw.get("confidence"), field=f"{key}.confidence")
            policy_ids = self._string_list(
                raw.get("violated_policy_ids"), field=f"{key}.violated_policy_ids", limit=3
            )
            policy_ids = self._normalize_known_policy_ids(policy_ids, known_policy_ids)
            self._validate_policy_ids(
                policy_ids,
                known_policy_ids=known_policy_ids,
                field=f"{key}.violated_policy_ids",
            )
            raw_codes = self._string_list(
                raw.get("violation_codes"), field=f"{key}.violation_codes", limit=2
            )
            invalid_codes = sorted(set(raw_codes) - valid_codes)
            if invalid_codes:
                raise ValueError(f"Direct multi-horizon output {key} has invalid codes: {invalid_codes}")
            codes = raw_codes
            predicted_risk = self._strict_bool(raw.get("predicted_risk"), field=f"{key}.predicted_risk")
            implied_risk = risk_score >= self.risk_threshold or bool(policy_ids or codes)
            if predicted_risk != implied_risk:
                raise ValueError(
                    f"Direct multi-horizon output {key} is inconsistent: "
                    f"predicted_risk={predicted_risk}, risk_score={risk_score}, "
                    f"policy_ids={policy_ids}, codes={codes}"
                )
            explanation = str(raw.get("risk_explanation") or "")[:500]
            guidance = raw.get("optimization_guidance")
            normalized["horizon_predictions"][key] = {
                "valid_prediction": True,
                "predicted_risk": predicted_risk,
                "risk_score": risk_score,
                "confidence": confidence,
                "violated_policy_ids": sorted(dict.fromkeys(policy_ids)),
                "violation_codes": sorted(dict.fromkeys(codes)),
                "risk_explanation": explanation,
                "optimization_guidance": str(guidance)[:500] if guidance not in (None, "") else None,
            }
        return normalized

    def _coerce_iterative_horizon_step(
        self,
        raw: Dict[str, Any],
        *,
        step_index: int,
        max_horizon: int,
        current_action: Any,
        known_policy_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        key = f"h{step_index}"
        required = {
            "predicted_state_delta",
            "predicted_next_action",
            "predicted_risk",
            "risk_score",
            "confidence",
            "violated_policy_ids",
            "violation_codes",
            "risk_explanation",
            "optimization_guidance",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Iterative rollout output {key} is missing fields: {missing}")
        if set(raw) != required:
            raise ValueError(
                f"Iterative rollout output {key} fields must equal "
                f"{sorted(required)}, observed={sorted(raw)}"
            )
        if not isinstance(raw.get("predicted_state_delta"), str):
            raise ValueError(
                f"Iterative rollout output {key}.predicted_state_delta must be a string"
            )
        state_delta = raw["predicted_state_delta"].strip()
        if not state_delta:
            raise ValueError(f"Iterative rollout output {key}.predicted_state_delta is empty")
        next_action = raw.get("predicted_next_action")
        if step_index < max_horizon:
            next_action = self._normalize_harness_action(
                next_action,
                field=f"{key}.predicted_next_action",
            )
        elif next_action is not None:
            raise ValueError(f"{key}.predicted_next_action must be null at the final horizon")
        if step_index == max_horizon:
            next_action = None
        risk_score = self._bounded_float(raw.get("risk_score"), field=f"{key}.risk_score")
        confidence = self._bounded_float(raw.get("confidence"), field=f"{key}.confidence")
        policy_ids = self._string_list(
            raw.get("violated_policy_ids"), field=f"{key}.violated_policy_ids", limit=3
        )
        policy_ids = self._normalize_known_policy_ids(policy_ids, known_policy_ids)
        self._validate_policy_ids(
            policy_ids,
            known_policy_ids=known_policy_ids,
            field=f"{key}.violated_policy_ids",
        )
        codes = self._string_list(raw.get("violation_codes"), field=f"{key}.violation_codes", limit=2)
        valid_codes = {"V-ID", "V-OT", "V-OR", "V-IC"}
        invalid_codes = sorted(set(codes) - valid_codes)
        if invalid_codes:
            raise ValueError(f"Iterative rollout output {key} has invalid codes: {invalid_codes}")
        predicted_risk = self._strict_bool(raw.get("predicted_risk"), field=f"{key}.predicted_risk")
        implied_risk = risk_score >= self.risk_threshold or bool(policy_ids or codes)
        if predicted_risk != implied_risk:
            raise ValueError(f"Iterative rollout output {key} has inconsistent risk fields")
        if not isinstance(raw.get("risk_explanation"), str):
            raise ValueError(
                f"Iterative rollout output {key}.risk_explanation must be a string"
            )
        guidance = raw.get("optimization_guidance")
        if guidance is not None and not isinstance(guidance, str):
            raise ValueError(
                f"Iterative rollout output {key}.optimization_guidance must be a string or null"
            )
        return {
            "valid_prediction": True,
            "predicted_action": copy.deepcopy(current_action),
            "predicted_state_delta": state_delta[:1200],
            "predicted_next_action": copy.deepcopy(next_action),
            "predicted_risk": predicted_risk,
            "risk_score": risk_score,
            "confidence": confidence,
            "violated_policy_ids": sorted(dict.fromkeys(policy_ids)),
            "violation_codes": sorted(dict.fromkeys(codes)),
            "risk_explanation": raw["risk_explanation"][:500],
            "optimization_guidance": (
                guidance[:500]
                if guidance not in (None, "")
                else None
            ),
        }

    @staticmethod
    def _strict_bool(value: Any, *, field: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{field} must be a JSON boolean, got {value!r}")

    @staticmethod
    def _bounded_float(value: Any, *, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a JSON number, got {value!r}")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"{field} must be within [0, 1], got {number}")
        return number

    @staticmethod
    def _string_list(value: Any, *, field: str, limit: int) -> List[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be a JSON array of strings")
        normalized = [item.strip() for item in value if item.strip()]
        if len(normalized) > limit:
            raise ValueError(f"{field} may contain at most {limit} items")
        return normalized

    @staticmethod
    def _known_policy_ids(policies: Optional[List[Dict[str, Any]]]) -> set[str]:
        return {
            str(policy.get("policy_id") or "").strip()
            for policy in policies or []
            if isinstance(policy, dict) and str(policy.get("policy_id") or "").strip()
        }

    @staticmethod
    def _validate_policy_ids(
        policy_ids: List[str],
        *,
        known_policy_ids: set[str] | None,
        field: str,
    ) -> None:
        if known_policy_ids is None:
            return
        unknown = sorted(set(policy_ids) - known_policy_ids)
        if unknown:
            raise ValueError(f"{field} contains unknown policy IDs: {unknown}")

    @staticmethod
    def _normalize_known_policy_ids(
        policy_ids: List[str],
        known_policy_ids: set[str] | None,
    ) -> List[str]:
        """Remove only a display prefix when the resulting ID is an exact known ID."""
        if not known_policy_ids:
            return policy_ids
        normalized: List[str] = []
        for policy_id in policy_ids:
            candidate = policy_id[len("Policy "):].strip() if policy_id.startswith("Policy ") else policy_id
            normalized.append(candidate if candidate in known_policy_ids else policy_id)
        return normalized

    @staticmethod
    def _strict_json_retry_instruction(error: Exception, *, scope: str) -> str:
        message = " ".join(str(error).split())[:600]
        return f"""

## Strict Validation Retry
Your previous JSON was rejected: {message}
Regenerate {scope} from scratch. Return JSON only, include every required field,
use exact policy IDs without a `Policy ` prefix, and obey the HarnessAudit action
schema exactly. Do not explain the correction and do not reuse an invalid value.
"""

    @staticmethod
    def _validate_risk_threshold(value: Any) -> float:
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk_threshold must be numeric, got {value!r}") from exc
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"risk_threshold must be within [0, 1], got {threshold}")
        return threshold

    @staticmethod
    def _normalize_harness_action(value: Any, *, field: str) -> Dict[str, Any]:
        if isinstance(value, str):
            parsed = parse_json_from_text(value)
            if isinstance(parsed, dict):
                value = parsed
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a HarnessAudit action JSON object")
        action_type = value.get("type")
        role = value.get("agent_role")
        if action_type not in {"tool_call", "communication"}:
            raise ValueError(f"{field}.type must be tool_call or communication")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{field}.agent_role must be a non-empty string")
        if action_type == "tool_call":
            tool_name = value.get("tool_name")
            tool_args = value.get("tool_args")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"{field}.tool_name must be a non-empty string")
            if not isinstance(tool_args, dict):
                raise ValueError(f"{field}.tool_args must be a JSON object")
            return {
                "type": "tool_call",
                "agent_role": role.strip(),
                "tool_name": tool_name.strip(),
                "tool_args": copy.deepcopy(tool_args),
            }
        target = value.get("target_role")
        content = value.get("content")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"{field}.target_role must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{field}.content must be a non-empty string")
        return {
            "type": "communication",
            "agent_role": role.strip(),
            "target_role": target.strip(),
            "content": content.strip(),
        }

    @staticmethod
    def _client_usage_snapshot(client: Any) -> Dict[str, Any]:
        snapshot = getattr(client, "token_usage_snapshot", None)
        if not callable(snapshot):
            return {}
        value = snapshot()
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
    
    
    def _to_string(self, obj: Any) -> str:
        """Convert object to string representation."""
        return format_object_to_string(obj)
    
    def _parse_llm_output(self, generated_text: str, original_state: Any, action: Any) -> Any:
        """
        Parse LLM output to extract next state.
        
        Supports:
        - State delta mode (predicts only changes, then synthesizes full state)
        - Full state prediction mode (legacy)
        - JSON format parsing
        - Error handling
        
        Args:
            generated_text: Raw text from LLM
            original_state: Original state before action
            action: Action that was executed
        
        Returns:
            Parsed next state (synthesized from delta or full prediction)
        """
        # Try state delta mode first (more efficient)
        if self.use_state_delta:
            parsed_delta = (
                self._parse_strict_state_delta_json(generated_text)
                if getattr(self, "strict_state_delta_output", False)
                else self._parse_json_output(generated_text)
            )
            # Ensure parsed_delta is a dict before processing - raise error if not
            if not parsed_delta:
                error_msg = f"[WorldModel] Failed to parse JSON output. Raw text: {generated_text[:500]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(parsed_delta, dict):
                error_msg = f"[WorldModel] parsed_delta is not a dict: type={type(parsed_delta)}, value={str(parsed_delta)[:200]}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Add default values for disabled prediction types
            # Note: We set these to None (not empty string) so that subsequent code using
            # `if delta.get('semantic_delta'):` will correctly skip disabled predictions
            if not getattr(self, 'enable_short_term_prediction', True):
                parsed_delta.setdefault("semantic_delta", None)
                logger.debug("[World Model] Short-term prediction disabled, field set to None")
            if not getattr(self, 'enable_long_term_prediction', True):
                parsed_delta.setdefault("long_term_impact", None)
                logger.debug("[World Model] Long-term prediction disabled, field set to None")
            
            if getattr(self, "strict_state_delta_output", False):
                parsed_delta = self._validate_strict_state_delta(parsed_delta)
            elif not self._is_state_delta_format(parsed_delta):
                parsed_delta = self._coerce_state_delta_format(parsed_delta)
                logger.warning(
                    "[WorldModel] Coerced non-standard state_delta response into "
                    "safe default format. Original keys: %s",
                    list(parsed_delta.get("_raw_nonstandard_delta", {}).keys()),
                )
            
            # Calculate risk score based on violated policies (rule-based)
            parsed_delta = self._calculate_risk_score_by_rules(parsed_delta)
            
            # Ensure parsed_delta is still a dict after processing - raise error if not
            if not isinstance(parsed_delta, dict):
                error_msg = f"[WorldModel] parsed_delta is not a dict after _calculate_risk_score_by_rules: type={type(parsed_delta)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Store delta for risk evaluator to access
            if not isinstance(parsed_delta, dict):
                logger.warning(f"[WorldModel] parsed_delta is NOT a dict before storing! value={str(parsed_delta)[:500]}")
            self._last_predicted_delta = parsed_delta
            
            # Apply delta to current state to synthesize next state
            synthesized_state = self._apply_state_delta(original_state, parsed_delta, action)
            return synthesized_state
        
        # Fallback to full state prediction mode
        if self.require_json_output:
            parsed_state = self._parse_json_output(generated_text)
            if parsed_state:
                # Merge with original state structure
                next_state = copy.deepcopy(original_state)
                if isinstance(next_state, dict):
                    next_state.update(parsed_state)
                    next_state["action_history"] = next_state.get("action_history", []) + [action]
                    next_state["step"] = next_state.get("step", 0) + 1
                    next_state["simulated"] = True
                    next_state["llm_prediction"] = generated_text
                    next_state["parsed_from_json"] = True
                return next_state
        
        # If JSON parsing failed, raise error with full debug output
        logger.error("=" * 80)
        logger.error("[World Model] JSON Parsing Failed - Full LLM Output:")
        logger.error("=" * 80)
        logger.error(f"Raw LLM Output (length: {len(generated_text)}):")
        logger.error(generated_text)
        logger.error("=" * 80)
        error_msg = f"Failed to parse JSON from LLM output: {generated_text[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _state_delta_required_fields(self) -> set[str]:
        required = {
            "element_changes",
            "violated_policy_ids",
            "violation_codes",
            "risk_explanation",
            "optimization_guidance",
        }
        if getattr(self, "enable_short_term_prediction", True):
            required.add("semantic_delta")
        if getattr(self, "enable_long_term_prediction", True):
            required.add("long_term_impact")
        return required

    def _parse_strict_state_delta_json(self, text: str) -> Dict[str, Any]:
        """Return the final JSON object without XML/text repair or coercion."""

        decoder = json.JSONDecoder()
        matches: list[Dict[str, Any]] = []
        for match in re.finditer(r"\{", text or ""):
            try:
                candidate, _ = decoder.raw_decode((text or "")[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                matches.append(candidate)
        if not matches:
            raise ValueError("Failed to parse strict state-delta JSON object")
        required = self._state_delta_required_fields()
        exact = [candidate for candidate in matches if set(candidate) == required]
        if exact:
            return exact[-1]
        return max(
            matches,
            key=lambda candidate: len(set(candidate) & required),
        )

    def _validate_strict_state_delta(
        self,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the online state-delta contract without repairing output."""

        if not isinstance(parsed, dict):
            raise ValueError("strict state-delta response must be a JSON object")
        required = self._state_delta_required_fields()
        observed = set(parsed)
        if observed != required:
            missing = sorted(required - observed)
            extra = sorted(observed - required)
            raise ValueError(
                "strict state-delta response must use exact keys; "
                f"missing={missing}, extra={extra}"
            )

        if getattr(self, "enable_short_term_prediction", True) and not isinstance(
            parsed["semantic_delta"], str
        ):
            raise ValueError("semantic_delta must be a JSON string")
        if getattr(self, "enable_long_term_prediction", True) and not isinstance(
            parsed["long_term_impact"], str
        ):
            raise ValueError("long_term_impact must be a JSON string")
        if not isinstance(parsed["risk_explanation"], str):
            raise ValueError("risk_explanation must be a JSON string")
        if parsed["optimization_guidance"] is not None and not isinstance(
            parsed["optimization_guidance"], str
        ):
            raise ValueError("optimization_guidance must be a JSON string or null")

        element_changes = parsed["element_changes"]
        if not isinstance(element_changes, dict) or set(element_changes) != {
            "new_elements",
            "removed_elements",
        }:
            raise ValueError(
                "element_changes must use exact keys new_elements and removed_elements"
            )
        for field in ("new_elements", "removed_elements"):
            values = element_changes[field]
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise ValueError(f"element_changes.{field} must be an array of strings")

        policy_ids = parsed["violated_policy_ids"]
        violation_codes = parsed["violation_codes"]
        for field, values in (
            ("violated_policy_ids", policy_ids),
            ("violation_codes", violation_codes),
        ):
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{field} must be an array of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must not contain duplicates")

        known_policy_ids = set(
            getattr(self, "_active_known_policy_ids", set()) or set()
        )
        unknown_policy_ids = sorted(set(policy_ids) - known_policy_ids)
        if unknown_policy_ids:
            raise ValueError(f"unknown policy IDs: {unknown_policy_ids}")
        known_codes = set(
            getattr(self, "_active_known_violation_codes", set()) or set()
        )
        unknown_codes = sorted(set(violation_codes) - known_codes)
        if unknown_codes:
            raise ValueError(f"unknown violation codes: {unknown_codes}")

        if bool(policy_ids) != bool(violation_codes):
            raise ValueError(
                "violated_policy_ids and violation_codes must be jointly empty or non-empty"
            )
        if policy_ids:
            if not parsed["risk_explanation"].strip():
                raise ValueError("violations require a non-empty risk_explanation")
            guidance = parsed["optimization_guidance"]
            if not isinstance(guidance, str) or not guidance.strip():
                raise ValueError("violations require non-empty optimization_guidance")
        else:
            if parsed["risk_explanation"] != "":
                raise ValueError("a no-violation response requires empty risk_explanation")
            if parsed["optimization_guidance"] is not None:
                raise ValueError("a no-violation response requires null optimization_guidance")

        return parsed
    
    def _is_state_delta_format(self, parsed: Dict[str, Any]) -> bool:
        """
        Check if parsed JSON is in state delta format.
        
        Args:
            parsed: Parsed JSON dictionary
        
        Returns:
            True if it's state delta format, False otherwise
        """
        if not isinstance(parsed, dict):
            return False
        
        # Check required top-level fields
        # Note: risk_score is calculated by _calculate_risk_score_by_rules, not required from LLM output
        required_fields = [
            "element_changes",
            "violated_policy_ids",
            "violation_codes",
            "risk_explanation",
            "optimization_guidance"
        ]
        
        # Add optional fields based on configuration
        if getattr(self, 'enable_short_term_prediction', True):
            required_fields.append("semantic_delta")
        if getattr(self, 'enable_long_term_prediction', True):
            required_fields.append("long_term_impact")
        
        if not all(field in parsed for field in required_fields):
            return False
        
        # Validate nested field types - raise error if wrong type
        if not isinstance(parsed.get("element_changes"), dict):
            error_msg = f"[WorldModel] element_changes is not a dict: type={type(parsed.get('element_changes'))}, value={str(parsed.get('element_changes'))[:200]}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        return True

    def _coerce_state_delta_format(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a non-standard JSON object into a conservative state-delta.

        Some OpenAI-compatible models occasionally answer with only the concrete
        tool/result fields they expect to change, e.g. {"user_lookup": "..."}.
        For long-horizon rollouts this should not invalidate the whole
        prediction point. We preserve the raw object as semantic evidence and
        mark no policy violation unless the model explicitly supplied one.
        """

        raw = parsed if isinstance(parsed, dict) else {"raw_response": str(parsed)}
        violation_codes = raw.get("violation_codes") if isinstance(raw.get("violation_codes"), list) else []
        violated_policy_ids = (
            raw.get("violated_policy_ids") if isinstance(raw.get("violated_policy_ids"), list) else []
        )
        return {
            "semantic_delta": raw.get("semantic_delta") or json.dumps(raw, ensure_ascii=False),
            "element_changes": raw.get("element_changes") if isinstance(raw.get("element_changes"), dict) else {
                "new_elements": [],
                "removed_elements": [],
            },
            "long_term_impact": raw.get("long_term_impact") or "",
            "risk_explanation": raw.get("risk_explanation") or (
                "The world model returned a non-standard state-delta object; "
                "no explicit policy violation was provided in that object."
            ),
            "violated_policy_ids": violated_policy_ids,
            "violation_codes": violation_codes,
            "optimization_guidance": raw.get("optimization_guidance") or "",
            "_raw_nonstandard_delta": raw,
        }
    
    def _calculate_risk_score_by_rules(self, parsed_delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk score based on violated policies using rules.
        
        Note: Goal alignment is now a policy (automatically added to all policy files),
        so goal deviation violations will be included in violated_policy_ids.
        
        Rules:
        - Has violated_policy_ids → risk_score = violation_penalty (default 0.5)
        - No violated_policy_ids → risk_score = 0.0
        
        Args:
            parsed_delta: Parsed delta dictionary from LLM
        
        Returns:
            Updated parsed_delta with calculated risk_score
        """
        violated_policy_ids = parsed_delta.get('violated_policy_ids', [])
        has_violations = violated_policy_ids and len(violated_policy_ids) > 0
        
        # Rule-based risk score calculation
        # Note: Goal alignment violations are now included in violated_policy_ids
        # (goal alignment is automatically added as a policy to all policy files)
        # Risk score is calculated based on policy violations only
        risk_score = 0.0
        if has_violations:
            risk_score = self.risk_score_violation_penalty
        
        # Clamp risk_score to [0.0, 1.0]
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Update parsed_delta with calculated risk score
        # Note: risk_level is NOT calculated here - it comes from policy definitions
        # risk_level in policies is used for reference storage limits
        # risk_explanation is generated by World Model's LLM and should be preserved as-is
        parsed_delta['risk_score'] = risk_score
        
        return parsed_delta
    
    def get_risk_score(self) -> Optional[float]:
        """
        Get the risk score from the last prediction.
        
        Returns:
            Risk score (0.0-1.0) if available, None otherwise
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            return self._last_predicted_delta.get('risk_score')
        return None
    
    def get_risk_explanation(self) -> Optional[str]:
        """
        Get the risk explanation from the last prediction.
        
        Returns:
            Risk explanation string if available, None otherwise
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            return self._last_predicted_delta.get('risk_explanation')
        return None
    
    def get_multi_step_risk_details(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed risk information for multi-step predictions.
        
        Returns:
            Dictionary with multi-step risk details if available, None otherwise.
            Contains:
            - weighted_risk: Overall weighted risk score
            - num_steps: Number of prediction steps
            - step_risks: List of risk scores for each step
            - step_explanations: List of risk explanations for each step
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            delta = self._last_predicted_delta
            if delta.get('multi_step', False):
                return {
                    'weighted_risk': delta.get('risk_score'),
                    'num_steps': delta.get('num_steps'),
                    'step_risks': delta.get('step_risks', []),
                    'step_explanations': delta.get('step_explanations', []),
                }
        return None
    
    def _calculate_weighted_risk(self, step_risks: List[float]) -> float:
        """
        Calculate weighted risk score from multiple step risks.
        
        Uses exponential decay weights: step 0 = 1.0, step 1 = 0.5, step 2 = 0.25, etc.
        
        Args:
            step_risks: List of risk scores for each step
        
        Returns:
            Weighted risk score (0.0-1.0)
        """
        if not step_risks:
            error_msg = "[WorldModel] Cannot calculate weighted risk: step_risks is empty"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if len(step_risks) == 1:
            return step_risks[0]
        
        # Calculate weights: exponential decay
        weights = []
        total_weight = 0.0
        for i in range(len(step_risks)):
            weight = 1.0 / (2 ** i)  # 1.0, 0.5, 0.25, 0.125, ...
            weights.append(weight)
            total_weight += weight
        
        # Calculate weighted average
        weighted_sum = sum(step_risks[i] * weights[i] for i in range(len(step_risks)))
        weighted_risk = weighted_sum / total_weight
        
        return weighted_risk
    
    def _combine_risk_explanations(
        self, 
        step_explanations: List[Optional[str]], 
        step_risks: List[float],
        weighted_risk: float
    ) -> str:
        """
        Combine risk explanations from multiple steps into a single explanation.
        
        Args:
            step_explanations: List of risk explanations for each step
            step_risks: List of risk scores for each step
            weighted_risk: Calculated weighted risk score
        
        Returns:
            Combined risk explanation string
        """
        if not step_explanations:
            return f"Weighted risk score: {weighted_risk:.3f}"
        
        if len(step_explanations) == 1:
            base_explanation = step_explanations[0] or "Risk assessed"
            return f"{base_explanation} (risk: {step_risks[0]:.3f})"
        
        # Build combined explanation
        parts = []
        parts.append(f"Multi-step weighted risk: {weighted_risk:.3f}")
        parts.append("Step risks:")
        
        for i, (explanation, risk) in enumerate(zip(step_explanations, step_risks)):
            step_num = i + 1
            weight = 1.0 / (2 ** i)
            exp_text = explanation or f"Step {step_num}"
            parts.append(f"  Step {step_num} (weight {weight:.3f}): {exp_text} (risk: {risk:.3f})")
        
        return " | ".join(parts)
    
    def _apply_state_delta(self, current_state: Any, delta: Dict[str, Any], action: Any) -> str:
        """
        Apply state delta to current state to synthesize next state.
        
        Strategy:
        1. Keep Goal and Policies unchanged (they don't change due to actions)
        2. Update URL if changed
        3. Update Accessibility Tree based on semantic_delta and element_changes
        4. Update Chat History (add action)
        5. Add metadata
        
        Args:
            current_state: Current state (string representation)
            delta: State delta dictionary from LLM
            action: Action that was executed
        
        Returns:
            Synthesized next state as string
        """
        # Parse current state if it's a string
        if isinstance(current_state, str):
            # Extract components from current state string
            lines = current_state.split('\n')
            goal_line = None
            policies_section = []
            axtree_section = []
            chat_section = []
            
            in_policies = False
            in_axtree = False
            in_chat = False
            
            for line in lines:
                if line.startswith('Goal:'):
                    goal_line = line
                    in_policies = False
                    in_axtree = False
                    in_chat = False
                elif line.startswith('Policies:'):
                    in_policies = True
                    in_axtree = False
                    in_chat = False
                    policies_section.append(line)
                elif line.startswith('Accessibility Tree:'):
                    in_policies = False
                    in_axtree = True
                    in_chat = False
                    axtree_section.append(line)
                elif line.startswith('Chat History:'):
                    in_policies = False
                    in_axtree = False
                    in_chat = True
                    chat_section.append(line)
                else:
                    if in_policies:
                        policies_section.append(line)
                    elif in_axtree:
                        axtree_section.append(line)
                    elif in_chat:
                        chat_section.append(line)
            
            # Build synthesized state
            synthesized_lines = []
            
            # 1. Keep Goal unchanged
            if goal_line:
                synthesized_lines.append(goal_line)
            
            # 2. Keep Policies unchanged
            if policies_section:
                synthesized_lines.extend(policies_section)
            
            # 3. Update Accessibility Tree with changes
            if axtree_section:
                # Keep original axtree header
                synthesized_lines.append(axtree_section[0] if axtree_section else 'Accessibility Tree:')
                # Add semantic delta as annotation
                if delta.get('semantic_delta'):
                    synthesized_lines.append(f"[State Changes] {delta['semantic_delta']}")
                # Add risk-relevant element changes
                risk_aff = delta.get('element_changes', {})
                if risk_aff.get('new_elements'):
                    synthesized_lines.append(f"[New Elements] {', '.join(risk_aff['new_elements'])}")
                # Keep original axtree content (truncated if too long)
                if len(axtree_section) > 1:
                    # Keep first part of original tree, add changes annotation
                    synthesized_lines.extend(axtree_section[1:min(50, len(axtree_section))])
            
            # 4. Update Chat History (add action)
            action_str = str(action) if action else "None"
            if chat_section:
                synthesized_lines.extend(chat_section)
                synthesized_lines.append(f"assistant: {action_str}")
            else:
                synthesized_lines.append(f"Chat History: assistant: {action_str}")
            
            # 5. Add metadata
            synthesized_lines.append("")
            synthesized_lines.append("[World Model Prediction Metadata]")
            
            synthesized_state = '\n'.join(synthesized_lines)
            return synthesized_state
        else:
            # If current_state is not a string, convert delta to string representation
            # This is a fallback for non-string states
            delta_str = json.dumps(delta, indent=2, ensure_ascii=False)
            return f"{current_state}\n\n[State Changes]\n{delta_str}"
    
    def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse output from LLM (supports JSON, XML, and structured text formats).
        
        Parsing priority:
        1. JSON (primary format, with error fixing)
        2. XML (format-insensitive fallback)
        3. Structured Text (format-insensitive fallback)
        
        Args:
            text: Raw text from LLM
        
        Returns:
            Parsed dict or None if parsing fails
        """
        # Try JSON format first (primary format, with robust error fixing)
        parsed = parse_json_from_text(text)
        if isinstance(parsed, dict):
            return parsed
        
        # Try XML format (format-insensitive fallback)
        xml_result = parse_xml_output(text)
        if xml_result:
            # Convert XML format to World Model's expected format
            return convert_xml_to_world_model_format(xml_result)
        
        # Try Structured Text format (format-insensitive fallback)
        structured_result = parse_structured_text(text)
        if structured_result:
            # Convert structured text format to World Model's expected format
            return convert_structured_text_to_world_model_format(structured_result)
        
        return None

    def _parse_json_output_with_required_keys(
        self,
        text: str,
        required_keys: set[str],
    ) -> Optional[Dict[str, Any]]:
        """Select the final complete JSON object matching the requested contract."""
        decoder = json.JSONDecoder()
        matches: List[Dict[str, Any]] = []
        for match in re.finditer(r"\{", text or ""):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and required_keys.issubset(candidate):
                matches.append(candidate)
        if matches:
            return matches[-1]
        return self._parse_json_output(text)
    
    def simulate(self, state: Any, action: Any, policies: Optional[List[Dict[str, Any]]] = None, steps: Optional[int] = None, plan_text: Optional[str] = None) -> Any:
        """
        Simulate next state(s) using LLM prediction.
        
        Args:
            state: Current state (already contains complete chat_history)
            action: Action to simulate
            policies: Optional list of policy dictionaries for risk evaluation (will be formatted with references)
            steps: Number of steps to predict (overrides self.prediction_steps if provided)
            plan_text: Optional execution plan text for progress tracking
        
        Returns:
            Predicted next state (single-step) or list of predicted states (multi-step)
        """
        if not self._model_loaded:
            self._load_model()

        active_policies = policies or []
        self._active_known_policy_ids = self._known_policy_ids(active_policies)
        self._active_known_violation_codes = {
            str(policy.get("violation_code") or "").strip()
            for policy in active_policies
            if isinstance(policy, dict)
            and str(policy.get("violation_code") or "").strip()
        }
        
        # Determine number of steps to predict
        num_steps = steps if steps is not None else getattr(self, 'prediction_steps', 1)
        
        # Single-step prediction (default)
        if num_steps == 1:
            # Create prompt from state, action, and state history
            # Policies are passed as List[Dict[str, Any]] and will be formatted in the prompt
            # Use plan_text parameter if provided, otherwise try to get from state
            # This allows plan_text to be passed explicitly or extracted from state
            # Note: plan_text is only used when violation is detected (to generate optimization_guidance)
            # If plan_text is None, world model will not track plan progress (more efficient)
            effective_plan_text = plan_text
            if effective_plan_text is None and isinstance(state, dict):
                effective_plan_text = state.get('plan_text')
            
            # Only include plan_text in prompt if explicitly provided (for violation-based plan updates)
            # This avoids unnecessary plan tracking overhead for safe actions
            prompt = self._create_prompt(state, action, policies=policies, plan_text=effective_plan_text)
            
            # Check API availability
            if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
                error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Use API-based LLM (via unified LLMClient)
            next_state = self._simulate_api(prompt, state, action)
            return next_state
        
        # Multi-step prediction
        else:
            predicted_states = []
            step_risks = []  # Store risk scores for each step
            step_explanations = []  # Store risk explanations for each step
            step_violated_policy_ids = []  # Collect violated policy IDs from all steps
            step_violation_codes = []  # Collect violation codes from all steps

            current_state = state
            current_action = action

            for step_idx in range(num_steps):
                # Create prompt from current state and action
                prompt = self._create_prompt(current_state, current_action, policies=policies)
                
                # Check API availability
                if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
                    error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Predict next state (this updates _last_predicted_delta)
                next_state = self._simulate_api(prompt, current_state, current_action)
                predicted_states.append(next_state)
                
                # Immediately save risk information for this step before next iteration overwrites it
                # Get risk score and explanation from _last_predicted_delta (set by _simulate_api)
                step_delta = getattr(self, '_last_predicted_delta', None)
                if step_delta is None:
                    error_msg = f"[WorldModel] _last_predicted_delta is None for step {step_idx + 1} in multi-step prediction"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if not isinstance(step_delta, dict):
                    error_msg = f"[WorldModel] step_delta is not a dict: type={type(step_delta)}, value={str(step_delta)[:200]}"
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                
                step_risk = step_delta.get('risk_score')
                step_explanation = step_delta.get('risk_explanation')
                
                # Store risk information for this step
                # World Model must return risk_score for each step, raise error if None
                if step_risk is None:
                    error_msg = f"[WorldModel] Risk score not available for step {step_idx + 1} in multi-step prediction"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                step_risks.append(step_risk)
                step_explanations.append(step_explanation or f"Step {step_idx + 1} risk")

                # Collect violation data from this step
                for pid in step_delta.get('violated_policy_ids', []):
                    if pid not in step_violated_policy_ids:
                        step_violated_policy_ids.append(pid)
                for code in step_delta.get('violation_codes', []):
                    if code not in step_violation_codes:
                        step_violation_codes.append(code)

                # For subsequent steps, we need to generate a plausible next action
                # This is a simplified approach - in practice, you might want to use
                # get_valid_actions() or another method to generate next actions
                if step_idx < num_steps - 1:
                    current_state = self._state_for_next_prediction(
                        previous_state=current_state,
                        predicted_state=next_state,
                        delta=step_delta,
                    )
                    # Generate a plausible next action (simplified - you might want to improve this)
                    # For now, we'll use a placeholder action
                    current_action = {"type": "continue", "step": step_idx + 1}
            
            # Calculate weighted risk score
            # Weight decreases with step depth: step 0 = 1.0, step 1 = 0.5, step 2 = 0.25, etc.
            weighted_risk = self._calculate_weighted_risk(step_risks)
            
            # Log multi-step prediction summary
            logger.debug(
                f"[WorldModel] Multi-step prediction ({num_steps} steps): "
                f"step_risks={step_risks}, weighted_risk={weighted_risk:.3f}"
            )
            
            # Combine risk explanations
            combined_explanation = self._combine_risk_explanations(step_explanations, step_risks, weighted_risk)
            
            # Store multi-step prediction results
            self._last_predicted_delta = {
                'risk_score': weighted_risk,
                'risk_explanation': combined_explanation,
                'multi_step': True,
                'num_steps': num_steps,
                'step_risks': step_risks,
                'step_explanations': step_explanations,
                'violated_policy_ids': step_violated_policy_ids,
                'violation_codes': step_violation_codes,
            }
            
            # Return list of predicted states
            return predicted_states

    @staticmethod
    def _state_for_next_prediction(
        *,
        previous_state: Any,
        predicted_state: Any,
        delta: Optional[Dict[str, Any]],
    ) -> Any:
        """Keep multi-step prediction state in dict form when state-delta returns text."""

        if isinstance(predicted_state, dict):
            return predicted_state
        if not isinstance(previous_state, dict):
            return predicted_state

        next_state = previous_state.copy()
        history = list(next_state.get("prediction_history") or [])
        history.append(
            {
                "previous_predicted_state": str(predicted_state),
                "semantic_delta": (delta or {}).get("semantic_delta"),
                "long_term_impact": (delta or {}).get("long_term_impact"),
                "risk_score": (delta or {}).get("risk_score"),
                "violated_policy_ids": (delta or {}).get("violated_policy_ids", []),
            }
        )
        next_state["previous_predicted_state"] = str(predicted_state)
        next_state["previous_predicted_delta"] = delta or {}
        next_state["prediction_history"] = history[-5:]
        return next_state

    def predict_future_risk(
        self,
        state: Any,
        *,
        policies: Optional[List[Dict[str, Any]]] = None,
        horizon: int = 5,
        prediction_sequence_num: Optional[int] = None,
        predictor: str = "safepred_wm",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict future HarnessAudit risk from a trace prefix.

        This method is separate from ``simulate`` because the HarnessAudit
        integration needs a trace-level future-risk event, not a predicted UI
        state delta for a single action.
        """
        if not self._model_loaded:
            self._load_model()

        if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
            error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        from .prompts import PromptTemplates
        from ..core.future_risk import build_wm_risk_prediction_event
        from ..utils.llm_client import LLMClient

        state_metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
        benchmark = str(state_metadata.get("benchmark") or "").strip().lower()
        if benchmark in {"isbench", "is-bench", "is_bench"} or (
            isinstance(state, dict) and state.get("isbench")
        ):
            prompt = PromptTemplates.isbench_future_risk_prediction(
                state=state,
                policies=policies,
                horizon=horizon,
            )
        else:
            prompt = PromptTemplates.harnessaudit_future_risk_prediction(
                state=state,
                policies=policies,
                horizon=horizon,
            )

        if not hasattr(self, '_llm_client'):
            self._llm_client = LLMClient(
                api_key=self.api_key,
                api_url=self.api_url,
                model_name=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=getattr(self, "transport_max_attempts", 3),
            )

        generated_text = self._llm_client.generate(
            prompt,
            max_tokens=min(int(self.max_tokens or 3000), 3000),
        )
        self._last_generated_text = generated_text
        if hasattr(self._llm_client, '_last_token_usage') and self._llm_client._last_token_usage:
            self._last_token_usage = self._llm_client._last_token_usage.copy()

        parsed = parse_json_from_text(generated_text)
        if not isinstance(parsed, dict):
            parsed = self._parse_first_json_object(generated_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Future-risk world model returned non-JSON content: {generated_text[:500]}")

        return build_wm_risk_prediction_event(
            parsed,
            prediction_sequence_num=prediction_sequence_num,
            horizon=horizon,
            predictor=predictor,
            metadata=metadata,
        )

    @staticmethod
    def _parse_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort parser for fenced or prefixed JSON object outputs."""
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
                    return parsed if isinstance(parsed, dict) else None
        return None
    
    def _simulate_api(self, prompt: str, state: Any, action: Any) -> Any:
        """
        Simulate next state using LLM API with retry on format errors.
        
        Uses unified LLMClient for all providers.
        Automatically retries API calls if JSON parsing fails due to format issues.
        
        Args:
            prompt: Formatted prompt
            state: Current state
            action: Action to simulate
        
        Returns:
            Predicted next state
        """
        # Use unified LLM client
        from ..utils.llm_client import LLMClient
        import time
        
        # Create or reuse LLM client
        if not hasattr(self, '_llm_client'):
            self._llm_client = LLMClient(
                api_key=self.api_key,
                api_url=self.api_url,
                model_name=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=getattr(self, "transport_max_attempts", 3),
            )
        
        # Retry configuration for format errors
        max_format_retries = (
            max(0, int(getattr(self, "max_semantic_attempts", 3)) - 1)
            if getattr(self, "strict_state_delta_output", False)
            else getattr(self, 'max_format_retries', 2)
        )
        format_retry_delay = getattr(self, 'format_retry_delay', 0.5)  # Default: 0.5 seconds
        
        retries = 0
        last_error = None
        
        while retries <= max_format_retries:
            try:
                # Generate text using unified client
                request_prompt = prompt
                strict_output = bool(
                    getattr(self, "strict_state_delta_output", False)
                )
                generate_kwargs: Dict[str, Any] = {
                    "request_ledger_attempt": retries + 1,
                }
                if strict_output:
                    request_prompt += self._strict_state_delta_initial_instruction()
                    generate_kwargs.update(
                        {
                            "response_format": {"type": "json_object"},
                            "enable_thinking": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    )
                    if last_error is not None:
                        request_prompt += self._strict_state_delta_retry_instruction(
                            last_error
                        )
                generated_text = self._llm_client.generate(
                    request_prompt,
                    **generate_kwargs,
                )
                
                # Store generated_text for error handling
                self._last_generated_text = generated_text
                
                # Store token usage from LLM client
                if hasattr(self._llm_client, '_last_token_usage') and self._llm_client._last_token_usage:
                    self._last_token_usage = self._llm_client._last_token_usage.copy()
                
                # Check if generated_text is empty
                if not generated_text or len(generated_text.strip()) == 0:
                    error_msg = "LLM returned empty response"
                    logger.error("=" * 80)
                    logger.error("[World Model] Empty Response Error:")
                    logger.error("=" * 80)
                    logger.error(f"Model: {self.model_name}")
                    logger.error(f"Prompt length: {len(prompt)}")
                    logger.error(f"Prompt preview: {prompt[:500]}...")
                    logger.error(f"Generated text length: {len(generated_text) if generated_text else 0}")
                    logger.error(f"Generated text: {repr(generated_text)}")
                    logger.error("=" * 80)
                    raise ValueError(error_msg)
                
                # Parse generated text to get next state
                next_state = self._parse_llm_output(generated_text, state, action)
                
                # Log World Model prediction result (consolidated log)
                # Note: Use _last_predicted_delta
                if self.use_state_delta:
                    if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
                        parsed_delta = self._last_predicted_delta
                        # Ensure parsed_delta is a dict - raise error if not
                        if not isinstance(parsed_delta, dict):
                            error_msg = f"[WorldModel] _last_predicted_delta is not a dict: type={type(parsed_delta)}, value={str(parsed_delta)[:200]}"
                            logger.error(error_msg)
                            raise TypeError(error_msg)
                        # State delta mode: log complete delta information
                        # Include all fields from parsed_delta (complete world model output)
                        # Extract reasoning part from raw_response (text before JSON code block)
                        reasoning_text = ""
                        if generated_text:
                            # Try to extract reasoning part (text before JSON code block)
                            # Match JSON code blocks: ```json ... ``` or ``` ... ```
                            json_block_patterns = [
                                r'```json\s*\n',
                                r'```json\s*',
                                r'```\s*\n',
                                r'```\s*'
                            ]
                            json_start_idx = len(generated_text)
                            for pattern in json_block_patterns:
                                match = re.search(pattern, generated_text, re.IGNORECASE)
                                if match and match.start() < json_start_idx:
                                    json_start_idx = match.start()
                            
                            if json_start_idx < len(generated_text):
                                reasoning_text = generated_text[:json_start_idx].strip()
                            else:
                                # No JSON block found, use entire text as reasoning
                                reasoning_text = generated_text.strip()
                        
                        # Extract task goal from state if available
                        task_goal = None
                        if isinstance(state, dict):
                            task_goal = state.get("goal", None)
                        
                        world_model_log_json = {
                            "component": "World Model",
                            "mode": "state_delta",
                            "action": str(action) if action else None,
                            "task_goal": task_goal,  # Add task goal to log
                            "reasoning": reasoning_text,  # Only include reasoning part
                            **parsed_delta  # Include all fields from parsed_delta
                        }
                        logger.info(
                            f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                        )
                    else:
                        # Fallback: log raw response if delta parsing failed
                        # Extract task goal from state if available
                        task_goal = None
                        if isinstance(state, dict):
                            task_goal = state.get("goal", None)
                        
                        world_model_log_json = {
                            "component": "World Model",
                            "mode": "state_delta",
                            "action": str(action) if action else None,
                            "task_goal": task_goal,  # Add task goal to log
                            "raw_response": generated_text[:500] + "..." if len(generated_text) > 500 else generated_text,
                            "raw_response_length": len(generated_text),
                        }
                        logger.info(
                            f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                        )
                else:
                    # Full state mode: log parsed state preview
                    # Extract task goal from state if available
                    task_goal = None
                    if isinstance(state, dict):
                        task_goal = state.get("goal", None)
                    
                    parsed_state_str = str(next_state)
                    world_model_log_json = {
                        "component": "World Model",
                        "mode": "full_state",
                        "action": str(action) if action else None,
                        "task_goal": task_goal,  # Add task goal to log
                        "predicted_state_preview": parsed_state_str[:500] + "..." if len(parsed_state_str) > 500 else parsed_state_str,
                        "predicted_state_length": len(parsed_state_str),
                    }
                    logger.info(
                        f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                    )
                
                return next_state
                
            except ValueError as e:
                # Check if this is a JSON parsing error (format issue)
                error_str = str(e)
                is_format_error = (
                    getattr(self, "strict_state_delta_output", False)
                    or "Failed to parse JSON" in error_str
                    or "Failed to parse" in error_str
                )
                
                if is_format_error and retries < max_format_retries:
                    # Format error - retry API call
                    retries += 1
                    last_error = e
                    logger.warning(
                        f"[World Model] JSON format error detected (attempt {retries}/{max_format_retries + 1}), "
                        f"retrying API call after {format_retry_delay}s delay..."
                    )
                    time.sleep(format_retry_delay)
                    continue  # Retry the API call
                else:
                    # Not a format error, or max retries exceeded - raise immediately
                    error_msg = f"LLM simulation failed: {e}"
                    logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
                    # Log the generated text if available (for debugging)
                    if hasattr(self, '_last_generated_text') and self._last_generated_text:
                        logger.error("=" * 80)
                        logger.error("[World Model] Error - Full LLM Output:")
                        logger.error("=" * 80)
                        logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                        logger.error(self._last_generated_text)
                        logger.error("=" * 80)
                    raise RuntimeError(error_msg) from e
                    
            except Exception as e:
                # Other types of errors (not format-related) - raise immediately without retry
                error_msg = f"LLM simulation failed: {e}"
                logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
                # Log the generated text if available (for debugging)
                if hasattr(self, '_last_generated_text') and self._last_generated_text:
                    # Always log the full output for debugging (JSON parsing errors are already logged in _parse_llm_output)
                    # But also log here for other types of errors
                    if not (hasattr(e, '__cause__') and isinstance(e.__cause__, ValueError) and "Failed to parse JSON" in str(e.__cause__)):
                        # Not a JSON parsing error (or not logged yet), log the output
                        logger.error("=" * 80)
                        logger.error("[World Model] Error - Full LLM Output:")
                        logger.error("=" * 80)
                        logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                        logger.error(self._last_generated_text)
                        logger.error("=" * 80)
                raise RuntimeError(error_msg) from e
        
        # If we exhausted all retries, raise the last error
        if last_error:
            error_msg = f"LLM simulation failed after {max_format_retries + 1} attempts (format errors): {last_error}"
            logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
            if hasattr(self, '_last_generated_text') and self._last_generated_text:
                logger.error("=" * 80)
                logger.error("[World Model] Final Error After Retries - Full LLM Output:")
                logger.error("=" * 80)
                logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                logger.error(self._last_generated_text)
                logger.error("=" * 80)
            raise RuntimeError(error_msg) from last_error

    @staticmethod
    def _strict_state_delta_initial_instruction() -> str:
        return (
            "\n\nSTRICT JSON-ONLY OVERRIDE: This instruction supersedes any earlier "
            "request to print a reasoning process or markdown. Keep reasoning "
            "internal. The first non-whitespace character of the response MUST "
            "be `{` and the last MUST be `}`. Return exactly one JSON object "
            "using exactly the fields and types in the requested state-delta "
            "schema. Do not include prose, markdown fences, headings, or extra "
            "keys. Choose exactly one internally consistent policy branch. "
            "NO VIOLATION branch: violated_policy_ids=[], violation_codes=[], "
            "risk_explanation=\"\", and optimization_guidance=null. VIOLATION "
            "branch: use only exact supplied policy IDs and violation codes, "
            "and make both risk_explanation and optimization_guidance non-empty. "
            "Do not hedge, self-debate, or discuss merely hypothetical or "
            "role-confused violations in any field. Keep semantic_delta and "
            "long_term_impact concise descriptions of predicted state effects, "
            "not hidden reasoning transcripts."
        )

    @staticmethod
    def _strict_state_delta_retry_instruction(error: Exception) -> str:
        detail = re.sub(r"\s+", " ", str(error)).strip()[:500]
        return (
            "\n\nYour previous response failed strict state-delta validation: "
            f"{detail}. Return exactly one corrected JSON object matching the "
            "requested schema. Choose one internally consistent branch: either "
            "all violation fields identify a concrete supplied policy violation, "
            "or use empty policy/code arrays, risk_explanation=\"\", and "
            "optimization_guidance=null. Do not hedge or include self-analysis. "
            "Do not include markdown, commentary, or extra keys."
        )
    
    
    def get_valid_actions(self, state: Any) -> List[Any]:
        """
        Get valid actions using LLM to analyze state.
        
        Args:
            state: Current state
        
        Returns:
            List of valid actions
        
        Raises:
            ValueError: If API credentials are not configured
            RuntimeError: If LLM API call fails
        """
        if not self._model_loaded:
            self._load_model()
        
        # Check API availability
        if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
            error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            from .prompts import PromptTemplates
            
            # Create or reuse LLM client
            if not hasattr(self, '_llm_client'):
                from ..utils.llm_client import LLMClient
                self._llm_client = LLMClient(
                    api_key=self.api_key,
                    api_url=self.api_url,
                    model_name=self.model_name,
                    provider=self.provider,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    max_retries=getattr(self, "transport_max_attempts", 3),
                )
            
            # Create prompt for action generation
            state_str = self._to_string(state)
            prompt = PromptTemplates.action_generation(state_str)
            
            # Call LLM
            response = self._llm_client.generate(prompt, temperature=0.7, max_tokens=500)
            
            # Parse actions from response
            actions = self._parse_actions(response)
            if not actions:
                error_msg = "Failed to parse actions from LLM response. No valid actions found."
                logger.error(f"{error_msg} | Response: {response[:200]}")
                raise ValueError(error_msg)
            
            return actions
                    
        except Exception as e:
            error_msg = f"LLM action generation failed: {e}"
            logger.error(f"{error_msg} | State: {state}")
            raise RuntimeError(error_msg) from e
    
    def _parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse actions from LLM response.
        
        Args:
            text: LLM response text
        
        Returns:
            List of action dictionaries
        
        Raises:
            ValueError: If actions cannot be parsed from response
        """
        # Try to parse JSON array
        actions = parse_json_array_from_text(text)
        
        if actions:
            # Validate and normalize actions
            valid_actions = []
            for action in actions:
                if isinstance(action, dict) and "type" in action:
                    valid_actions.append(normalize_action(action))
            
            if valid_actions:
                return valid_actions
            else:
                error_msg = "No valid actions found in parsed JSON array."
                logger.error(f"{error_msg} | Response: {text[:200]}")
                raise ValueError(error_msg)
        
        # If parsing fails, raise error
        error_msg = "Failed to parse actions from LLM response. No valid JSON array found."
        logger.error(f"{error_msg} | Response: {text[:200]}")
        raise ValueError(error_msg)
    
    def simulate_batch(self, states: List[Any], actions: List[Any]) -> List[Any]:
        """
        Simulate state transitions for a batch (optimized for LLM inference).
        
        Args:
            states: List of current states
            actions: List of actions
        
        Returns:
            List of predicted next states
        """
        # Could batch LLM calls for efficiency
        # For now, use sequential simulation
        return super().simulate_batch(states, actions)


# Default world model factory
def WorldModel(
    llm_config: Optional[Dict[str, Any]] = None,
    use_few_shot: bool = False,
    use_chain_of_thought: bool = False,
    require_json_output: bool = True,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> BaseWorldModel:
    """
    Factory function to create world models.
    
    Args:
        llm_config: Optional LLM configuration dictionary (from SafetyConfig.get_llm_config())
        use_few_shot: Whether to use few-shot learning (default: False)
        use_chain_of_thought: Whether to use chain-of-thought reasoning (default: False)
        require_json_output: Whether to require JSON format output (default: True)
        few_shot_examples: List of example dicts with keys: 'state', 'action', 'next_state'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        World model instance (LLMBasedWorldModel)
    
    Example:
        # Using config from SafetyConfig
        config = SafetyConfig.from_yaml("config.yaml")
        llm_config = config.get_llm_config("world_model")
        world_model = WorldModel(llm_config=llm_config)
        
        # With prompt optimizations
        examples = [
            {"state": {...}, "action": {...}, "next_state": {...}},
            {"state": {...}, "action": {...}, "next_state": {...}}
        ]
        world_model = WorldModel(
            llm_config=llm_config,
            use_few_shot=True,
            use_chain_of_thought=True,
            require_json_output=True,
            few_shot_examples=examples
        )
    """
    if llm_config:
        kwargs["llm_config"] = llm_config
    # Add prompt optimization parameters
    kwargs["use_few_shot"] = use_few_shot
    kwargs["use_chain_of_thought"] = use_chain_of_thought
    kwargs["require_json_output"] = require_json_output
    kwargs["few_shot_examples"] = few_shot_examples
    return LLMBasedWorldModel(**kwargs)
