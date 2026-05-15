"""General-purpose aggregator for per-trial evaluation reports."""

from .base import (
    DEFAULT_SUMMARY_PROMPT,
    Aggregator,
    render_display,
)
from .types import (
    AggregationReport,
    AggregationResult,
    EfficiencyStats,
    PairedComparisonStats,
)

__all__ = [
    "Aggregator",
    "AggregationReport",
    "AggregationResult",
    "EfficiencyStats",
    "PairedComparisonStats",
    "DEFAULT_SUMMARY_PROMPT",
    "render_display",
]
