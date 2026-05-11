"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .effects import (
    ChaosEffect,
    CorruptValues,
    RemoveFields,
    ToolCallFailure,
    ToolEffect,
    TruncateFields,
)
from .experiment import ChaosExperiment
from .plugin import ChaosPlugin
from .scenario import ChaosScenario

__all__ = [
    # Core classes
    "ChaosExperiment",
    "ChaosPlugin",
    "ChaosScenario",
    # Effect hierarchy
    "ChaosEffect",
    "ToolEffect",
    # Concrete effects
    "ToolCallFailure",
    "TruncateFields",
    "RemoveFields",
    "CorruptValues",
]
