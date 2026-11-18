"""Metric names and ranges: evaluator scores, usage metrics, and metrics derived from them."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Derived:
    """A metric computed per run from that run's evaluator scores and usage, then scored like any other.

    Give it to `Maximize`, `Minimize`, `AtMost`, or `AtLeast` in place of a metric name; other goals can then
    name it (`AtMost("cost", 0.05)`). It is fitted, ranked, validated, and tested like an evaluator score.
    For example, the input tokens no cache served (`input_tokens` includes both cache kinds):
    `Derived("uncached_input", lambda m: m["input_tokens"] - m.get("cache_read_tokens", 0) -
    m.get("cache_write_tokens", 0), range=(0, None))`; a cost in dollars weights each token kind by its price
    the same way.

    `fn` reads a run's values by metric name: each evaluator's score (by `get_name()`) and each usage
    metric, not other derived metrics. A value the run doesn't have (an evaluator that found nothing to
    judge or raised, a score taken out as out of range, a cache count the provider never reported) raises
    `KeyError` and is not `in` it, so `m.get(name, default)` gives the default; use that only where a default
    is right, such as a cache count. When `fn` reads a value the run doesn't have, or raises an
    `ArithmeticError` (e.g. a division by zero; a `math` domain error is a `ValueError` and propagates), the
    run has no value of the metric and the reason is kept in `CandidateScores.unavailable`; the case's mean
    comes from its other runs, and a case with no value on any run makes the candidate ineligible. On an
    `agent_failure` run `fn` reads the evaluator scores `agent_failure_score` gives them; if it still can't
    give a value, the run takes the metric's worst end for its goals under `agent_failure_score="worst"`, as
    an evaluator's would (none when that end is unbounded), and no value otherwise. Reading or testing (`in`)
    a name that is not an evaluator or usage metric raises `ValueError`, returning anything but a number
    raises `TypeError`, and any other exception from `fn` propagates; each stops the run. `fn` should be a
    pure function of its input: on resume, it computes the metric again from the saved trials.

    Attributes:
        name: The metric's name, unique among the run's evaluators, usage metrics, and derived metrics.
        fn: Computes the metric from one run's values.
        range: Where its values lie, `(low, high)`; None (or an infinite end) for unbounded. A value outside
            it is taken out of scoring with the reason, never clipped (float noise at an end is set to that
            end). A low end of 0 or more also gives the test report a ratio to the baseline.
    """

    name: str
    fn: Callable[[Mapping[str, float]], float] = field(repr=False)
    range: Range = (None, None)

    def __post_init__(self) -> None:
        """Check the name, the function, and the range; store the range's ends as floats or None.

        Raises:
            TypeError: `name` is not a string, `fn` is not callable, or `range` is not a pair of numbers or
                None.
            ValueError: `name` is empty, or `range` has a NaN end or is empty (low >= high).
        """
        if not isinstance(self.name, str):
            raise TypeError(f"Derived name must be a string, got {self.name!r}")
        if not self.name:
            raise ValueError("Derived name must not be empty")
        if not callable(self.fn):
            raise TypeError(f"Derived {self.name!r}: fn must be callable, got {self.fn!r}")
        ends = tuple(self.range) if isinstance(self.range, (tuple, list)) else ()
        if len(ends) != 2 or any(
            end is not None and (isinstance(end, bool) or not isinstance(end, (int, float))) for end in ends
        ):
            raise TypeError(f"Derived {self.name!r}: range must be (low, high), each a number or None")
        object.__setattr__(self, "range", _checked_range(self.name, "range", ends))


class MetricRegistry:
    """Maps metric names as written to evaluator, usage, or derived metric names, and knows each one's range."""

    def __init__(self, evaluators: Mapping[str, Range], derived: Sequence[Derived] = ()):
        """Register the run's evaluators and derived metrics.

        An evaluator's range is its optional `score_range` attribute, else `(0.0, 1.0)`; usage metrics are
        `(0, None)`; a derived metric's is its `range`. An infinite end means unbounded, the same as None, so
        a combined metric still needs `scale=` (an infinite width would make its term always 0).

        Args:
            evaluators: `get_name()` -> its range.
            derived: Metrics computed from each run's evaluator scores and usage. The same `Derived` given
                twice is registered once.

        Raises:
            ValueError: An evaluator has the same name as a usage metric (e.g. `name="tokens"`), or a derived
                metric the same name as an evaluator, a usage metric, or a different derived metric, so a
                goal naming it would be ambiguous; or a range has a NaN end or is empty (low >= high).
        """
        clashes = sorted(name for name in evaluators if name in USAGE_METRICS)
        if clashes:
            raise ValueError(f"evaluators {clashes} are named like usage metrics; give them another name=")
        self._ranges: dict[str, Range] = {
            name: _checked_range(name, "score_range", range_) for name, range_ in evaluators.items()
        }
        self._derived: dict[str, Derived] = {}
        for metric in derived:
            if metric.name in self._ranges or metric.name in USAGE_METRICS:
                raise ValueError(
                    f"Derived metric {metric.name!r} is named like an evaluator or usage metric; give it another name"
                )
            known = self._derived.setdefault(metric.name, metric)
            if known != metric:
                raise ValueError(
                    f"two different Derived metrics are named {metric.name!r}; define it once and pass that "
                    "object, or name it by its name in the other goals"
                )

    @property
    def evaluators(self) -> tuple[str, ...]:
        """The evaluators' names, in registration order (the default objective maximizes each)."""
        return tuple(self._ranges)

    @property
    def derived(self) -> tuple[Derived, ...]:
        """The derived metrics, in registration order."""
        return tuple(self._derived.values())

    def is_usage(self, name: str) -> bool:
        """Whether the resolved metric is a usage metric (no crash scoring, measured on crashed runs too)."""
        return name in USAGE_METRICS

    def is_derived(self, name: str) -> bool:
        """Whether the resolved metric is a `Derived` one, computed from each run's other metrics."""
        return name in self._derived

    def range(self, name: str) -> Range:
        """The resolved metric's range.

        Raises:
            KeyError: `name` isn't a resolved name. Names the user wrote go through `resolve` first, which
                raises `ValueError`; a `KeyError` here is a caller's bug, not a user's typo.
        """
        if name in USAGE_METRICS:
            return _USAGE_RANGE
        if name in self._derived:
            return self._derived[name].range
        if name not in self._ranges:
            raise KeyError(f"{name!r} is not a resolved metric; known: {self._known()}")
        return self._ranges[name]

    def resolve(self, name: str) -> str:
        """An exact evaluator, usage, or derived metric name first, then `"Helpfulness"` -> `"HelpfulnessEvaluator"`.

        Exact names win, so the shorthand can't be ambiguous: with evaluators `"Helpfulness"` and
        `"HelpfulnessEvaluator"` both registered, `"Helpfulness"` is the first. Unknown names raise
        `ValueError`.
        """
        if name in self._ranges or name in USAGE_METRICS or name in self._derived:
            return name
        if f"{name}Evaluator" in self._ranges:
            return f"{name}Evaluator"
        raise ValueError(f"unknown metric {name!r}; known: {self._known()}")

    def _known(self) -> list[str]:
        return [*self._ranges, *USAGE_METRICS, *self._derived]


def _checked_range(name: str, label: str, range_: Range) -> Range:
    """`range_` with each end a float, or None when unbounded (None or infinite).

    Raises:
        ValueError: An end is NaN, or the range is empty (low >= high).
    """
    low, high = range_
    if (low is not None and math.isnan(low)) or (high is not None and math.isnan(high)):
        raise ValueError(f"{name}'s {label} ({low}, {high}) has a NaN end; use None for unbounded")
    if (low is not None and high is not None and low >= high) or low == math.inf or high == -math.inf:
        raise ValueError(f"{name}'s {label} ({low}, {high}) is empty: low must be below high")
    return (_finite_or_none(low), _finite_or_none(high))


def _finite_or_none(end: float | None) -> float | None:
    """A range end as a float; None (unbounded) when it is None or infinite."""
    return None if end is None or math.isinf(end) else float(end)
