"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .case import ChaosCase, ChaosEffects
from .effects import (
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
