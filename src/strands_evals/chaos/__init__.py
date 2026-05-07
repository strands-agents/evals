"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .aggregation_display import ChaosAggregationDisplay, display_chaos_aggregation
from .aggregator import ChaosScenarioAggregator
from .aggregator_types import (
    ChaosScenarioAggregation,
    CoverageStatus,
    ToolEffectResult,
)
from .effects import (
    TOOL_CORRUPTION_EFFECTS,
    TOOL_ERROR_EFFECTS,
    ChaosEffect,
    CorruptValues,
    RemoveFields,
    ToolCallFailure,
    ToolEffect,
    TruncateFields,
)
from .evaluators import (
    FailureCommunicationEvaluator,
    PartialCompletionEvaluator,
    RecoveryStrategyEvaluator,
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
    # Classification sets
    "TOOL_ERROR_EFFECTS",
    "TOOL_CORRUPTION_EFFECTS",
]
