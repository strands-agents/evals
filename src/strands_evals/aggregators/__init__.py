"""Batch evaluation aggregators for Strands Evals.

Aggregators analyze evaluation results across multiple cases, scenarios,
or trials to produce summary reports and cross-case insights.
"""

from .base import CaseAggregator
from .types import AggregationResult

__all__ = [
    "CaseAggregator",
    "AggregationResult",
]
