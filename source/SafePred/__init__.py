"""
SafePred: Safety-TS-LMA (Safety-aware Language Model Agents)

A comprehensive safety-aware planning framework for Web Agents that uses:
- Trajectory graph modeling to capture state-action relationships
- Single-step prediction with integrated risk evaluation
- Risk filtering to identify compliance/safety risks early
- Single-step execution in the real environment

Main Components:
- TrajectoryGraph: Builds and maintains state-action trajectory graphs
- WorldModel: Simulates state transitions and evaluates risks
- SafeAgent: High-level interface for easy integration
- SafePredWrapper: Universal wrapper for easy benchmark integration
"""

from .core.trajectory_graph import TrajectoryGraph, StateNode, ActionEdge
from .models.world_model import WorldModel, BaseWorldModel
from .agent.agent import SafeAgent
from .config.config import SafetyConfig, default_config
from .wrapper import SafePredWrapper
from .adapters.base import BenchmarkAdapter, register_adapter, get_adapter
from .core.future_risk import (
    FUTURE_RISK_OUTPUT_SCHEMA,
    WM_RISK_PREDICTION_EVENT_TYPE,
    build_wm_risk_prediction_event,
)
from .utils.harnessaudit_policy import (
    convert_harnessaudit_task_to_policies,
    load_harnessaudit_task,
    write_harnessaudit_policies,
)

__version__ = "1.0.0"
__all__ = [
    "TrajectoryGraph",
    "StateNode",
    "ActionEdge",
    "WorldModel",
    "BaseWorldModel",
    "SafeAgent",
    "SafetyConfig",
    "default_config",
    "SafePredWrapper",
    "BenchmarkAdapter",
    "register_adapter",
    "get_adapter",
    "FUTURE_RISK_OUTPUT_SCHEMA",
    "WM_RISK_PREDICTION_EVENT_TYPE",
    "build_wm_risk_prediction_event",
    "convert_harnessaudit_task_to_policies",
    "load_harnessaudit_task",
    "write_harnessaudit_policies",
]
