import json
import os
from typing import List, Literal

import imageio

from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.utils.types import StepwisePlan


ACTION_OUTCOME_SCHEMA_VERSION = 1
ACTION_OUTCOME_STATUSES = {
    "completed",
    "failed",
    "terminal_noop",
    "navigation_noop",
    "infrastructure_error",
}
_ERROR_OUTCOME_STATUSES = {"failed", "infrastructure_error"}


def _one_line(value, *, limit):
    return " ".join(str(value or "").split())[:limit]


class OnlineEvalTracker(EvalTracker):

    def __init__(self):
        super().__init__()

        self.plans = []
        self.raw_outputs = []
        self.actor_generation_attempts = []
        self.awareness = None
        self.caption = None
        self.safepred_safeguard = []
        self.safepred_configuration = None
        self.experiment_metadata = None

        self.goal_condition = {}
        self.termination = None

        self.error_stack = []
        self.action_outcomes = []
        self.video_cache = []
    
    def track_plan(self, **kwargs):
        if self.plans and self.plans[-1].get("execution_outcome") is None:
            raise RuntimeError(
                "cannot register a new plan before the previous action outcome"
            )
        row = dict(**kwargs)
        row["execution_outcome"] = None
        self.plans.append(row)

    def complete_plan_execution(
        self,
        *,
        action,
        status,
        attempted,
        state_may_have_changed,
        error_type=None,
        error_message=None,
    ):
        """Atomically bind one yielded plan to its real executor outcome."""

        if status not in ACTION_OUTCOME_STATUSES:
            raise ValueError(f"unsupported action outcome status: {status!r}")
        if not isinstance(attempted, bool):
            raise TypeError("attempted must be boolean")
        if not isinstance(state_may_have_changed, bool):
            raise TypeError("state_may_have_changed must be boolean")
        if status in _ERROR_OUTCOME_STATUSES and (
            attempted is not True or state_may_have_changed is not True
        ):
            raise ValueError(
                f"{status} must be an attempted action with uncertain state"
            )
        if status in {"terminal_noop", "navigation_noop"} and (
            attempted is not False or state_may_have_changed is not False
        ):
            raise ValueError(f"{status} must be a non-attempted no-op")
        if status == "completed" and attempted is not True:
            raise ValueError("completed must be an attempted action")
        sequence_num = len(self.action_outcomes) + 1
        if len(self.plans) != sequence_num:
            raise RuntimeError(
                "action outcomes must be completed once, in yielded plan order"
            )
        plan_row = self.plans[sequence_num - 1]
        if plan_row.get("execution_outcome") is not None:
            raise RuntimeError(f"plan {sequence_num} already has an execution outcome")
        plan_action = str((plan_row.get("plan") or {}).get("action") or "").strip()
        normalized_action = str(action or "").strip()
        if not plan_action or plan_action.lower() != normalized_action.lower():
            raise RuntimeError(
                "executor action does not match the pending yielded plan: "
                f"plan={plan_action!r}, executor={normalized_action!r}"
            )

        normalized_error_type = _one_line(error_type, limit=120) or None
        normalized_error_message = _one_line(error_message, limit=500) or None
        is_error = status in _ERROR_OUTCOME_STATUSES
        if is_error and (not normalized_error_type or not normalized_error_message):
            raise ValueError(f"{status} requires error_type and error_message")
        if not is_error and (normalized_error_type or normalized_error_message):
            raise ValueError(f"{status} must not contain error fields")

        outcome = {
            "status": status,
            "attempted": attempted,
            "state_may_have_changed": state_may_have_changed,
            "error_type": normalized_error_type,
            "error_message": normalized_error_message,
        }
        plan_row["execution_outcome"] = outcome
        ledger_row = {
            "sequence_num": sequence_num,
            "plan_step": plan_row.get("step"),
            "action": plan_action,
            **outcome,
        }
        self.action_outcomes.append(ledger_row)
        if is_error:
            self.error_stack.append(
                {
                    "step": plan_row.get("step"),
                    "action": plan_action,
                    "err_type": normalized_error_type,
                    "msg": normalized_error_message,
                    "execution_status": status,
                }
            )
        return dict(outcome)
    
    def track_raw_output(self, **kwargs):
        self.raw_outputs.append(dict(**kwargs))

    def track_actor_generation_attempt(self, **kwargs):
        self.actor_generation_attempts.append(dict(**kwargs))

    def track_safepred_safeguard(self, **kwargs):
        self.safepred_safeguard.append(dict(**kwargs))

    def set_safepred_configuration(self, **kwargs):
        self.safepred_configuration = dict(**kwargs)

    def set_experiment_metadata(self, metadata):
        self.experiment_metadata = dict(metadata)

    def track_error(self, **kwargs):
        self.error_stack.append(dict(**kwargs))

    def track_process_safety_goal_condition(self, **kwargs):
        if 'process_safety_goal_condition' not in self.goal_condition:
            self.goal_condition['process_safety_goal_condition'] = []
        self.goal_condition['process_safety_goal_condition'].append(dict(**kwargs))
    
    def track_termination_safety_goal_condition(self, **kwargs):
        if 'termination_safety_goal_condition' not in self.goal_condition:
            self.goal_condition['termination_safety_goal_condition'] = []
        self.goal_condition['termination_safety_goal_condition'].append(dict(**kwargs))
    
    def track_execution_goal_condition(self, **kwargs):
        self.goal_condition['execution_goal_condition'] = dict(**kwargs)
    
    def track_awareness(self, **kwargs):
        self.awareness = dict(**kwargs)
    
    def track_caption(self, **kwargs):
        self.caption = dict(**kwargs)
        
    def track_termination(self, **kwargs):
        self.termination = dict(**kwargs)

    def track_video_rgb(self, rgb):
        self.video_cache.append(rgb)

    def save_video(self, save_path: str):
        if not self.video_cache:
            return

        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'video.mp4')
        else:
            assert save_path.endswith('.mp4')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(rgb)
        video_writer.close()

    def save_tracking(self, save_path: str):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        report = {
            'task': self.task,
            'scene': self.scene,
            'model': self.model,
            'awareness': self.awareness,
            'plans': [
                {
                    'step': plan['step'], 
                    'action': plan['plan']['action'], 
                    'caution': plan['plan']['caution']
                }
                for plan in self.plans
            ],
            'termination': self.termination,
            'error_stack': self.error_stack,
            'action_outcome_schema_version': ACTION_OUTCOME_SCHEMA_VERSION,
            'action_outcomes': self.action_outcomes,
        }
        if self.experiment_metadata is not None:
            report['experiment_metadata'] = self.experiment_metadata
        if getattr(self, 'action_recovery', None) is not None:
            report['action_recovery'] = self.action_recovery

        if 'process_safety_goal_condition' in self.goal_condition:
            report['process_safety_goal_condition'] = self.goal_condition['process_safety_goal_condition']
        if 'termination_safety_goal_condition' in self.goal_condition:
            report['termination_safety_goal_condition'] = self.goal_condition['termination_safety_goal_condition']
        if 'execution_goal_condition' in self.goal_condition:
            report['execution_goal_condition'] = self.goal_condition['execution_goal_condition']

        report['raw_outputs'] = self.raw_outputs
        report['actor_generation_attempts'] = self.actor_generation_attempts
        report['safepred_safeguard'] = self.safepred_safeguard
        if self.safepred_configuration is not None or self.safepred_safeguard:
            configuration = dict(self.safepred_configuration or {})
            usage = {
                key: sum(
                    int((item.get('token_usage') or {}).get(key) or 0)
                    for item in self.safepred_safeguard
                )
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            }
            configuration['decision_count'] = len(self.safepred_safeguard)
            configuration['token_usage'] = usage
            report['safepred_configuration'] = configuration
        if self.safepred_safeguard:
            first = self.safepred_safeguard[0]
            for key in (
                    'model',
                    'api_base',
                    'policy_mode',
                    'intervention_mode',
                    'policy_count',
                    'policy_sha256',
                    'uses_test_specific_safety_text',
                    'test_task_id',
                    'training_task_ids',
                    'heldout_task_in_training',
                    'risk_threshold',
                    'experiment_seed',
                    'inference_seed',
                    'semantic_max_attempts',
                    'transport_max_attempts',
                    'feedback_budget',
                    'provider_reasoning_capture',
                ):
                report['safepred_configuration'].setdefault(key, first.get(key))
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        self.save_video(save_dir)
