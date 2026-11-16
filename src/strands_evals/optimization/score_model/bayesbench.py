"""`BayesBenchModel`, the default score model; the only importer of `optimization/bayesbench/`."""

from __future__ import annotations

from typing import Any, ClassVar

from .protocols import Fit, FitData, ScoreModel


class BayesBenchModel(ScoreModel):
    """A Bayesian linear model per goal metric over the declared options.

    Terms: an intercept; each choice and binned number (main effects and pairwise); a linear trend per
    number; a "rewritten?" flag per text block; a per-candidate effect; the case. Prior scales by
    empirical Bayes, averaged as a mixture of up to 24 posteriors. Deterministic (closed form), so a
    resumed run refits the same fit.

    One observation per candidate x case: the mean of its per-run values, scaled by the metric's range (an
    unbounded one by twice the baseline's value). The per-run values set only the noise: the mean
    within-case variance / runs, floored, then the first fit's residual mean square if larger.
    Needs the `optimization` extra.
    """

    name: ClassVar[str] = "bayesbench"

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self, data: FitData, *, previous: Fit | None = None) -> Fit | None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError
