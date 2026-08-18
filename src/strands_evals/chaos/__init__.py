"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .case import ChaosCase, ChaosEffects
from .effects import (
    ChaosEffect,
    Confabulation,
    CorruptValues,
    EmptyResponse,
    ExecutionError,
    FullRefusal,
    MalformedJson,
    NetworkError,
    RemoveFields,
    SuccessFraming,
    Timeout,
    ToolEffect,
    ToolEffectUnion,
    TruncateFields,
    ValidationError,
)
from .experiment import ChaosExperiment
from .plugin import ChaosPlugin

__all__ = [
    # Core classes
    "ChaosCase",
    "ChaosEffects",
    "ChaosExperiment",
    "ChaosPlugin",
    # Effect hierarchy
    "ChaosEffect",
    "ToolEffect",
    "ToolEffectUnion",
    # Tool effects
    "Timeout",
    "NetworkError",
    "ExecutionError",
    "ValidationError",
    "TruncateFields",
    "RemoveFields",
    "CorruptValues",
    # Model effects
    "MalformedJson",
    "EmptyResponse",
    "Confabulation",
    "FullRefusal",
    "SuccessFraming",
]
