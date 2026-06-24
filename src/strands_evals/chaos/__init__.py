"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .case import ChaosCase
from .effects import (
    ChaosEffect,
    CorruptValues,
    ExecutionError,
    NetworkError,
    RemoveFields,
    Timeout,
    ToolEffect,
    ToolEffectUnion,
    TruncateFields,
    ValidationError,
)
from .experiment import ChaosExperiment
from .model_effects import (
    FormatCorruptionEffect,
    HallucinationEffect,
    RefusalEffect,
)
from .model_types import (
    ModelOutputCorruptionConfig,
    ModelOutputCorruptionType,
    ModelOutputHallucinationType,
)
from .plugin import ChaosPlugin

__all__ = [
    # Core classes
    "ChaosCase",
    "ChaosExperiment",
    "ChaosPlugin",
    # Effect hierarchy
    "ChaosEffect",
    "ToolEffect",
    "ToolEffectUnion",
    # Pre-hook effects (tool call failures)
    "Timeout",
    "NetworkError",
    "ExecutionError",
    "ValidationError",
    # Post-hook effects (response corruption)
    "TruncateFields",
    "RemoveFields",
    "CorruptValues",
    "ModelOutputCorruptionConfig",
    "ModelOutputCorruptionType",
    "ModelOutputHallucinationType",
    "FormatCorruptionEffect",
    "HallucinationEffect",
    "RefusalEffect",
]
