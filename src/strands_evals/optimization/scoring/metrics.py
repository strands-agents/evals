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

_USAGE_RANGE: Range = (0.0, None)  # lowest is 0 and unbounded


class MetricRegistry:
    """Maps metric names as written to evaluator names or usage metrics, and knows each metric's range."""

    def __init__(self, evaluators: Mapping[str, Range]):
        """Register the run's evaluators.

        An evaluator's range is its optional `score_range` attribute, else `(0.0, 1.0)`; usage metrics are
        `(0, None)`.

        Args:
            evaluators: `get_name()` -> its range.

        Raises:
            ValueError: An evaluator is named like a usage metric, or its range is empty (low >= high).
        """
        clashes = sorted(name for name in evaluators if name in USAGE_METRICS)
        if clashes:
            raise ValueError(f"evaluators {clashes} are named like usage metrics; give them another name=")
        for name, (low, high) in evaluators.items():
            if low is not None and high is not None and low >= high:
                raise ValueError(f"{name}'s score_range ({low}, {high}) is empty: low must be below high")
        self._ranges: dict[str, Range] = {
            name: (None if low is None else float(low), None if high is None else float(high))
            for name, (low, high) in evaluators.items()
        }

    @property
    def evaluators(self) -> tuple[str, ...]:
        """The evaluators' names, in registration order (the default objective maximizes each)."""
        return tuple(self._ranges)

    def is_usage(self, name: str) -> bool:
        """Whether the resolved metric is a usage metric (no crash scoring, measured on crashed runs too)."""
        return name in USAGE_METRICS

    def range(self, name: str) -> Range:
        """The resolved metric's range."""
        if name in USAGE_METRICS:
            return _USAGE_RANGE
        if name not in self._ranges:
            raise KeyError(f"{name!r} is not a resolved metric; known: {self._known()}")
        return self._ranges[name]

    def resolve(self, name: str) -> str:
        """An exact evaluator or usage metric name first, then `"Helpfulness"` -> `"HelpfulnessEvaluator"`.

        Exact names win, so the shorthand can't be ambiguous: with evaluators `"Helpfulness"` and
        `"HelpfulnessEvaluator"` both registered, `"Helpfulness"` is the first. Unknown names raise
        `ValueError`.
        """
        if name in self._ranges or name in USAGE_METRICS:
            return name
        if f"{name}Evaluator" in self._ranges:
            return f"{name}Evaluator"
        raise ValueError(f"unknown metric {name!r}; known: {self._known()}")

    def _known(self) -> list[str]:
        return [*self._ranges, *USAGE_METRICS]
