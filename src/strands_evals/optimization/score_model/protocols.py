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

from ...types.optimization import Estimate, Values
from ..declaration.space import Space


@dataclass(frozen=True)
class FitData:
    """What a model is fitted on; built by `optimizer.py`. A dataclass: it holds the `Space`.

    Attributes:
        space: The declared space.
        text_blocks: Block name -> its text keys (each block's "rewritten?" factor).
        candidates: Candidate id -> values, in scoring order (baseline first).
        cases: The split's case names (train for the search fit, validation in the end phase).
        observations: Goal metric -> candidate id -> case -> per-run values, from each candidate's
            `CandidateScores.metrics_by_run` with Nones dropped. Eligible candidates only, constraint
            violators included.
    """

    space: Space
    text_blocks: Mapping[str, Sequence[str]]
    candidates: Mapping[str, Values]
    cases: Sequence[str]
    observations: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]


class MetricModel(Protocol):
    """One metric's posterior."""

    def at(self, values: Values, candidate: str | None = None) -> Estimate:
        """The metric's mean over the cases.

        Args:
            values: The configuration.
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
    ) -> Estimate:
        """The ratio `values / reference` over the same cases."""
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

    metrics: Mapping[str, MetricModel]
    candidates: frozenset[str]
    flagged: list[str]
    diagnostics: Mapping[str, dict[str, Any]]

    def estimates(self, values: Values, candidate: str | None = None) -> dict[str, Estimate]:
        """Every metric's `at(values, candidate)`."""
        ...


class ScoreModel(ABC):
    """What `Optimizer(score_model=...)` takes; `BayesBenchModel` is the default.

    `effects.json` and `effects()` are filled only for a `BayesBenchModel` fit in v1; another model's run
    writes `effects.json` with no effects and a reason.
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
