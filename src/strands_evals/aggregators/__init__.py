"""Aggregators for evaluation reports.

Public API:

- ``Aggregator``: rolls up per-trial ``EvaluationReport`` objects into an
  ``AggregationReport``. Subclasses can override the grouping, filtering,
  and result-building hooks to add project-specific behavior.
- ``AggregationResult``: per-group aggregated result.
- ``AggregationReport``: top-level report with display and JSON serialization.
"""

from .base import Aggregator, DEFAULT_SUMMARY_PROMPT
from .types import AggregationReport, AggregationResult

__all__ = [
    "Aggregator",
    "AggregationReport",
    "AggregationResult",
    "DEFAULT_SUMMARY_PROMPT",
]
