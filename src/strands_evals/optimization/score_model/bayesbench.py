"""`BayesBenchModel`, the default score model; the only importer of `optimization/bayesbench/`.

One model per goal metric: a conjugate Gaussian posterior over the columns `_Design` lays out.

- **Observations:** one per candidate and case, the mean of its per-run values, on the model's scale: (value -
  offset) / unit, by the metric's range, or for an unbounded metric by twice the baseline's mean (offset 0).
- **Prior scales:** how far an option or bin moves the metric, an interaction, and a candidate beyond its factors
  are fitted to the data (empirical Bayes over `GRID`), and the posteriors under the likeliest scales are averaged
  by their weights (a mixture of up to `MIXTURE_SIZE`), so a candidate measured lucky is shrunk toward what its
  factors predict by as much as the other candidates' spread says it should be.
- **Noise:** one variance for every observation, in two passes: the mean within-case variance over runs divided by
  the runs (at least `NOISE_FLOOR`), then the first fit's residual mean square, whichever is larger.
- **Diagnostics:** before a candidate new since the previous fit is absorbed, its mean over the cases is scored
  against that fit's prediction (`bayesbench.HeldOutDiagnostics`). The checks live in each fit and are copied
  forward from `previous`, so a fit is a pure function of the chain of fits before it.

numpy and scipy are imported lazily: they are in the optional `optimization` extra.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ...types.optimization import Estimate, Values
from .design import _Design
from .protocols import Fit, FitData, ScoreModel

# Prior sds, in units of the metric's range (or of twice the baseline's mean, for an unbounded metric).
TAU_INTERCEPT = 0.25
TAU_MAIN = 0.15  # a choice option or a number bin; also a number's slope per sd of its values
TAU_PAIR = 0.03  # an interaction
TAU_CANDIDATE = 0.05  # one candidate, around what its factors predict
TAU_REWRITTEN = 0.15  # the average rewrite of a text block, around the default
TAU_CASE = 0.35
NOISE_FLOOR = 1e-6  # on the model's scale: a deterministic metric is estimated as measured

# The prior scales averaged over (every point equally likely before the observations).
GRID = {
    "main": (0.003, 0.01, 0.03, 0.08, 0.15),
    "pair": (0.001, 0.003, 0.01, 0.03),
    "candidate": (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3),
}
MIXTURE_SIZE = 24  # the likeliest grid points kept, at most; those with less than 1e-4 of the weight are dropped

_INSTALL = 'BayesBenchModel needs the optimization extra: pip install "strands-agents-evals[optimization]"'


class BayesBenchModel(ScoreModel):
    """A Bayesian linear model per goal metric over the declared options.

    Terms: an intercept; each choice and binned number (main effects and pairwise); a linear trend per
    number; a "rewritten?" flag per text block; a per-candidate effect; the case. Prior scales by
    empirical Bayes, averaged as a mixture of up to 24 posteriors. Deterministic (closed form), so a
    resumed run refits the same fit.

    One observation per candidate x case: the mean of its per-run values, scaled by the metric's range (an
    unbounded one by twice the baseline's mean). The per-run values set only the noise: the mean
    within-case variance / runs, floored, then the first fit's residual mean square if larger.
    """

    name: ClassVar[str] = "bayesbench"

    def __init__(self) -> None:
        """Check the optional extra is installed.

        Raises:
            ImportError: numpy or scipy is missing, with the install command.
        """
        try:
            import numpy  # noqa: F401  # optional extra: optimization
            import scipy  # noqa: F401  # optional extra: optimization
        except ImportError as err:
            raise ImportError(_INSTALL) from err

    def config(self) -> dict[str, Any]:
        """No settings: `{}`."""
        return {}

    def fit(self, data: FitData, *, previous: Fit | None = None) -> BayesBenchFit | None:
        """Fit every goal metric (the keys of `data.ranges`) to the eligible candidates; None when fewer than 2.

        Eligible = `CandidateScores.is_eligible`; the others are left out of the fit, design columns included.

        Args:
            data: The candidates' scores on one split; the baseline is the first of `data.candidates`.
            previous: The last successful fit, whose held-out checks are carried forward; another model's fit
                counts as none.

        Raises:
            ValueError: A goal metric no eligible candidate has a value of (e.g. an evaluator N/A on every case).
        """
        import numpy as np  # optional extra: optimization

        from ..bayesbench import HeldOutDiagnostics, Mixture, Posterior, prior_scale_weights

        pool = {
            cid: values
            for cid, values in data.candidates.items()
            if cid in data.scores and data.scores[cid].is_eligible
        }
        if len(pool) < 2:
            return None
        baseline = next(iter(data.candidates))
        layout = _Design(data.space, data.text_blocks, pool, data.cases)
        before_fit = previous if isinstance(previous, BayesBenchFit) else None
        checks: dict[str, Any] = copy.deepcopy(dict(before_fit.checks)) if before_fit else {}
        models: dict[str, BayesBenchMetric] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for metric, range_ in data.ranges.items():
            # candidate id -> case -> the runs with a value; the baseline's too (it sets an unbounded metric's unit)
            by_candidate = {
                cid: {
                    case: [v for v in by_metric.get(metric, []) if v is not None]
                    for case, by_metric in data.scores[cid].metrics_by_run.items()
                }
                for cid in (*pool, baseline)
                if cid in data.scores
            }
            offset, unit = _scale(range_, by_candidate.get(baseline))
            rows, ys, spreads = [], [], []
            means: dict[str, float] = {}
            n_cases: dict[str, int] = {}
            for cid in pool:
                per_case = {
                    case: [(v - offset) / unit for v in runs]
                    for case, runs in by_candidate.get(cid, {}).items()
                    if len(runs)
                }
                for case, scaled in per_case.items():
                    rows.append(layout.row(pool[cid], case, cid))
                    ys.append(statistics.fmean(scaled))
                    if len(scaled) >= 2:
                        spreads.append(statistics.variance(scaled) / len(scaled))
                if per_case:
                    means[cid] = statistics.fmean(statistics.fmean(s) for s in per_case.values())
                    n_cases[cid] = len(per_case)
            if not rows:
                raise ValueError(f"BayesBenchModel: no eligible candidate has a value of {metric}")
            phi, y = np.vstack(rows), np.asarray(ys)
            intercept = float(np.mean(y))
            grid = {g: GRID[g] for g in layout.groups}

            def prior(scales: dict[str, float], intercept: float = intercept) -> tuple[Any, Any]:
                return _prior(layout, intercept, scales)

            # Without repetitions (or with identical ones), the residual pass sets the noise.
            noise = max(statistics.fmean(spreads) if spreads else 0.0, NOISE_FLOOR)
            weighted = prior_scale_weights(phi, y, noise, prior, grid)
            mean = Posterior(*prior(weighted[0].scales)).condition(phi, y, noise).mean
            residual = float(np.mean((y - phi @ mean) ** 2))
            if residual > noise:  # how differently candidates do on each case, beyond repetition noise
                noise = residual
                weighted = prior_scale_weights(phi, y, noise, prior, grid)
            kept = [w for w in weighted[:MIXTURE_SIZE] if w.weight >= 1e-4]
            posterior = Mixture([(w.weight, Posterior(*prior(w.scales)).condition(phi, y, noise)) for w in kept])
            total = sum(w.weight for w in kept)
            scales = {g: sum(w.weight * w.scales[g] for w in kept) / total for g in kept[0].scales}
            candidate_sd = (
                math.sqrt(sum(w.weight * w.scales.get("candidate", TAU_CANDIDATE) ** 2 for w in kept) / total)
                if layout.candidate_ids
                else 0.0
            )
            if before_fit is not None and metric in before_fit.metrics:
                check = checks.setdefault(metric, HeldOutDiagnostics())
                earlier = before_fit.metrics[metric]
                for cid in pool:
                    if cid in before_fit.candidates or cid not in means:
                        continue
                    predicted, variance = earlier.posterior.predictive(earlier.design.row(pool[cid]))
                    check.record(
                        # on the earlier fit's scale, as its prediction and noise are: an unbounded metric's unit
                        # moves with the baseline's mean between fits
                        (means[cid] * unit + offset - earlier.offset) / earlier.unit,
                        float(predicted[0]),
                        float(variance[0]) + earlier.own(None),  # a candidate it hadn't fitted: own effect at the prior
                        earlier.noise / n_cases[cid],
                        cid,
                    )
            if metric in checks:
                snapshot = checks[metric].snapshot()
                diagnostics[metric] = {
                    "n_holdout": snapshot.n_holdout,
                    "z_mean": snapshot.z_mean,
                    "z_sq_sum": snapshot.z_sq_sum,
                    "flagged": snapshot.flagged,
                    "messages": list(snapshot.messages),
                }
            else:
                diagnostics[metric] = {"n_holdout": 0, "flagged": False, "messages": ["no held-out candidates yet"]}
            diagnostics[metric].update(
                prior_scales=scales,
                prior_scale_weights=[{"scales": w.scales, "weight": w.weight} for w in kept[:8]],
                scale={"unit": unit, "offset": offset},
                noise=noise,
                observations=int(posterior.n),
            )
            models[metric] = BayesBenchMetric(layout, posterior, offset, unit, noise, candidate_sd)
        return BayesBenchFit(
            metrics=models, candidates=frozenset(pool), diagnostics=diagnostics, design=layout, checks=checks
        )


class BayesBenchMetric:
    """One goal metric's posterior at one fit (a `MetricModel`).

    `values` / `reference` are configurations (declared key -> value, as a candidate's values; measured or
    not), turned into a row of the design by `design.row`.

    Attributes:
        design: The fit's layout.
        posterior: The mixture of posteriors, on the model's scale.
        offset, unit: The model's scale: metric = model value * unit + offset.
        noise: The observation noise variance, on the model's scale.
        candidate_sd: The prior sd of a candidate's own effect (over the mixture, the root of its mean square).
    """

    def __init__(
        self, design: _Design, posterior: Any, offset: float, unit: float, noise: float, candidate_sd: float
    ) -> None:
        self.design, self.posterior = design, posterior
        self.offset, self.unit, self.noise = offset, unit, noise
        self.candidate_sd = candidate_sd

    def own(self, candidate: str | None) -> float:
        """The variance a configuration's own effect adds: none for a fitted candidate, whose effect the posterior
        holds; for any other, the prior's."""
        return 0.0 if candidate is not None and candidate in self.design.fitted else self.candidate_sd**2

    def at(self, values: Values, candidate: str | None = None) -> Estimate:
        """The metric's mean over the cases at `values`, in its own units.

        With a fitted `candidate`, its own effect is the posterior's; otherwise it is unknown and adds the prior's
        variance.
        """
        mean, var = self.posterior.predictive(self.design.row(values, candidate=candidate))
        sd = math.sqrt(max(float(var[0]) + self.own(candidate), 0.0)) * self.unit
        return Estimate(mean=float(mean[0]) * self.unit + self.offset, sd=sd)

    def difference(
        self,
        values: Values,
        reference: Values,
        *,
        candidate: str | None = None,
        reference_candidate: str | None = None,
    ) -> Estimate:
        """`values` minus `reference` over the same cases, in the metric's units; the shared effects cancel."""
        a = self.design.row(values, candidate=candidate) - self.design.row(reference, candidate=reference_candidate)
        mean, var = self.posterior.predictive(a)
        variance = float(var[0]) + self.own(candidate) + self.own(reference_candidate)
        return Estimate(mean=float(mean[0]) * self.unit, sd=math.sqrt(max(variance, 0.0)) * self.unit)

    def ratio(
        self,
        values: Values,
        reference: Values,
        *,
        candidate: str | None = None,
        reference_candidate: str | None = None,
    ) -> Estimate | None:
        """`values` divided by `reference` over the same cases, to first order; None when the reference's mean is 0.

        The sd uses the variance of a ratio of two correlated estimates (the delta method).
        """
        import numpy as np  # optional extra: optimization

        a = self.design.row(values, candidate=candidate)
        b = self.design.row(reference, candidate=reference_candidate)
        mean, var = self.posterior.predictive(np.vstack([a, b]))
        var_a = float(var[0]) + self.own(candidate)
        var_b = float(var[1]) + self.own(reference_candidate)
        covariance = self.posterior.covariance(a, b)
        mean_a = float(mean[0]) * self.unit + self.offset
        mean_b = float(mean[1]) * self.unit + self.offset
        if mean_b == 0:
            return None
        u2 = self.unit**2
        variance = (
            (var_a * u2) / mean_b**2 + (mean_a**2 * var_b * u2) / mean_b**4 - 2 * mean_a * covariance * u2 / mean_b**3
        )
        return Estimate(mean=mean_a / mean_b, sd=math.sqrt(max(variance, 0.0)))

    def sample_effects(self, n: int, rng: Any) -> Any:
        """`n` sets of effects drawn from the posterior (an `n x p` array, on the model's scale)."""
        return self.posterior.sample(n, rng)

    def value_under(self, effects: Any, values: Values) -> Any:
        """The metric at `values`, in its own units, under each set of `sample_effects`."""
        return (effects @ self.design.row(values)) * self.unit + self.offset


@dataclass(frozen=True)
class BayesBenchFit:
    """Every goal metric's model at one fit (a `Fit`).

    Attributes:
        metrics: Goal metric -> its model.
        candidates: The ids fitted.
        diagnostics: Per metric: the held-out checks (`n_holdout`, `z_mean`, `z_sq_sum`, `flagged`, `messages`),
            the prior scales and their weights, the scale (`unit`, `offset`), the noise, and the observation count.
        design: The layout of the columns.
        checks: Per metric, the held-out checks so far; copied, never changed, by the next fit.
    """

    metrics: Mapping[str, BayesBenchMetric]
    candidates: frozenset[str]
    diagnostics: Mapping[str, dict[str, Any]]
    design: _Design = field(repr=False)
    checks: Mapping[str, Any] = field(repr=False)

    @property
    def flagged(self) -> list[str]:
        """The metrics whose held-out check flagged."""
        return [metric for metric, d in self.diagnostics.items() if d.get("flagged")]

    def estimates(self, values: Values, candidate: str | None = None) -> dict[str, Estimate]:
        """Every metric's `at(values, candidate)`."""
        return {metric: model.at(values, candidate) for metric, model in self.metrics.items()}


def _scale(range_: tuple[float | None, float | None], baseline: Mapping[str, Any] | None) -> tuple[float, float]:
    """(offset, unit): the range's low end and width, or for an unbounded metric 0 and twice the baseline's mean
    over the cases (1 when the baseline has none or it is 0)."""
    low, high = range_
    if low is not None and high is not None and high > low:
        return float(low), float(high - low)
    case_means = [statistics.fmean(runs) for runs in (baseline or {}).values() if len(runs)]
    mean = statistics.fmean(case_means) if case_means else 0.0
    return 0.0, (2 * abs(mean) if mean else 1.0)


def _prior(layout: _Design, intercept: float, scales: Mapping[str, float]) -> tuple[Any, Any]:
    """The prior's mean and variances; `scales` overrides the sd of a group (`main`, `pair`, `candidate`)."""
    import numpy as np  # optional extra: optimization

    tau = {"main": TAU_MAIN, "pair": TAU_PAIR, "candidate": TAU_CANDIDATE, **scales}
    mean = np.zeros(layout.p)
    var = np.empty(layout.p)
    mean[0] = intercept
    var[0] = TAU_INTERCEPT**2
    var[layout.main_columns] = tau["main"] ** 2
    var[layout.pair_columns] = tau["pair"] ** 2
    var[layout.n_design : layout.trend_start] = TAU_REWRITTEN**2
    var[layout.trend_start : layout.candidate_start] = tau["main"] ** 2  # a slope per sd of the number's values
    var[layout.candidate_start : layout.case_start] = tau["candidate"] ** 2
    var[layout.case_start :] = TAU_CASE**2
    return mean, var
