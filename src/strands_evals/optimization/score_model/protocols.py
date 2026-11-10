"""The score model's interfaces; a custom score model subclasses `ScoreModel` and returns a `Fit`.

`ScoreModel` is an ABC because users subclass it (as `Proposer`, `Selection`, `Schedule`); `Fit` and
`MetricModel` are Protocols, met by whatever objects a model returns.

One model per goal metric (each objective term's metric and each constraint metric), never a
model of the combined objective: each metric keeps its own units and range, and the objective is combined
from the per-metric estimates (`ranking.py`). numpy types are annotated `Any`: numpy is in the optional
extra, and every exported signature must resolve without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from ...types.optimization import CandidateScores, Estimate, Range, Values
from ..declaration.space import Space


@dataclass(frozen=True)
class FitData:
    """What a model is fitted on; built by `optimizer.py`. A dataclass: it holds the `Space`.

    Attributes:
        space: The declared space.
        text_blocks: Block name -> its text keys (each block's "rewritten?" factor).
        candidates: Candidate id -> values, in scoring order (baseline first).
        cases: The split's case names (train for the search fit, validation in the end phase).
        scores: Candidate id -> its `CandidateScores` on the split. A model fits the candidates whose
            `is_eligible` is true (constraint violators included), from each one's `metrics_by_run`
            (case -> metric -> per-run values, None where a run has no value).
        ranges: Goal metric -> its range (`MetricRegistry.range`), one per goal metric in
            `Criteria.goal_metrics` order: the metrics a model fits, and the range it scales each by.
    """

    space: Space
    text_blocks: Mapping[str, Sequence[str]]
    candidates: Mapping[str, Values]
    cases: Sequence[str]
    scores: Mapping[str, CandidateScores]
    ranges: Mapping[str, Range]


class MetricModel(Protocol):
    """One metric's posterior.

    `values` (and `reference`) in every method is one configuration: the flat mapping from each declared key
    to its typed value, as a candidate's values are, e.g. `{"model": "large", "cm": "window",
    "cm.window.size": 20, "prompt": "Answer in one line."}`. It holds the keys active under its choices and
    need not be a candidate that was measured: the model predicts any configuration of the space.
    """

    def at(self, values: Values, candidate: str | None = None) -> Estimate:
        """The metric's mean over the cases.

        Args:
            values: The configuration to predict: declared key -> value (see the class docstring).
            candidate: A fitted candidate's id (uses its own effect); None for any values, measured or not
                (then more uncertain: its own effect is unknown).
        """
        ...

    def difference(
        self,
        values: Values,
        reference: Values,
        *,
        candidate: str | None = None,
        reference_candidate: str | None = None,
    ) -> Estimate:
        """The paired gap `values - reference` over the same cases."""
        ...

    def ratio(
        self,
        values: Values,
        reference: Values,
        *,
        candidate: str | None = None,
        reference_candidate: str | None = None,
    ) -> Estimate | None:
        """The ratio `values / reference` over the same cases; None when the reference's mean is 0."""
        ...

    def sample_effects(self, n: int, rng: Any) -> Any:
        """`n` plausible sets of option effects sampled from the posterior.

        Args:
            n: How many sets.
            rng: A numpy `Generator`.

        Returns:
            An `n x p` ndarray.
        """
        ...

    def value_under(self, effects: Any, values: Values) -> Any:
        """The metric's value for `values` under each sampled set of effects."""
        ...

    unit: float
    """The metric's scale in the model, so `Thompson` can weigh different metrics' draws."""


class Fit(Protocol):
    """A fitted score model: one `MetricModel` per goal metric.

    Attributes:
        metrics: Resolved goal metric name -> its model, e.g. "HelpfulnessEvaluator", "tokens".
        candidates: Ids it was fitted on.
        flagged: Metrics whose held-out check failed; any flagged metric makes every consumer use measured
            means for every metric.
        diagnostics: Per metric (z-scores, prior scales, messages); a `BayesBenchModel` fit also gives each
            metric's scaling (noise, unit, offset) and observation count.
    """

    # Read-only, so a fit may hold them as frozen fields or properties, and its metrics as a MetricModel subtype.
    @property
    def metrics(self) -> Mapping[str, MetricModel]: ...

    @property
    def candidates(self) -> frozenset[str]: ...

    @property
    def flagged(self) -> list[str]: ...

    @property
    def diagnostics(self) -> Mapping[str, dict[str, Any]]: ...

    def estimates(self, values: Values, candidate: str | None = None) -> dict[str, Estimate]:
        """Every metric's `at(values, candidate)`; `values` is one configuration, as in `MetricModel`."""
        ...


class ScoreModel(ABC):
    """What `Optimizer(score_model=...)` takes; `BayesBenchModel` is the default.

    `effects.json` and `effects()` are filled only for a `BayesBenchModel` fit; another model's run writes
    `effects.json` with no effects and a reason.
    """

    name: ClassVar[str]

    @abstractmethod
    def fit(self, data: FitData, *, previous: Fit | None = None) -> Fit | None:
        """Fit every goal metric; None when fewer than 2 candidates are eligible.

        `previous` is the last successful fit; its prediction of each candidate new since then is scored
        before the candidate is absorbed (the held-out diagnostics). May raise: the optimizer catches it,
        records it in the `fit` event, and uses measured means until a later refit succeeds.

        Args:
            data: The eligible candidates' observations on one split.
            previous: The last successful fit, or None.
        """

    def config(self) -> dict[str, Any]:
        """JSON-serializable settings, stored in `run.json` and compared on resume; default `{}`."""
        return {}
