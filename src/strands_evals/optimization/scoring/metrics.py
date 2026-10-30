"""Metric names and ranges: evaluator scores and usage metrics."""

from __future__ import annotations

from collections.abc import Mapping

from ...types.optimization import Range

USAGE_METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "tokens",
    "seconds",
    "model_calls",
)


class MetricRegistry:
    """Maps metric names as written to evaluator names or usage metrics, and knows each metric's range."""

    def __init__(self, evaluators: Mapping[str, Range]):
        """Register the run's evaluators.

        An evaluator's range is its optional `score_range` attribute, else `(0.0, 1.0)`; usage metrics are
        `(0, None)`.

        Args:
            evaluators: `get_name()` -> its range.

        Raises:
            ValueError: An evaluator is named like a usage metric.
        """
        raise NotImplementedError

    @property
    def evaluators(self) -> tuple[str, ...]:
        """The evaluators' names, in registration order (the default objective maximizes each)."""
        raise NotImplementedError

    def is_usage(self, name: str) -> bool:
        """Whether the resolved metric is a usage metric (no crash scoring, measured on crashed runs too)."""
        raise NotImplementedError

    def range(self, name: str) -> Range:
        """The resolved metric's range."""
        raise NotImplementedError

    def resolve(self, name: str) -> str:
        """An exact evaluator or usage metric name first, then `"Helpfulness"` -> `"HelpfulnessEvaluator"`.

        The shorthand applies only when it matches exactly one evaluator; unknown or ambiguous names
        raise `ValueError`.
        """
        raise NotImplementedError
