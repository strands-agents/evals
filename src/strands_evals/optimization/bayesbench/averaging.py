"""Prior scales weighed by their marginal likelihood, and posteriors averaged by those weights.

How large an effect may be (the prior sd of a group of effects) is rarely known in advance. `prior_scale_weights`
scores every point of a grid of scales by the marginal likelihood it gives the observations (the effects integrated
out, `log_evidence`): its likeliest point is the empirical-Bayes choice. `Mixture` averages the posteriors under several
points by those weights (Bayesian model averaging, every point equally likely beforehand), so scales the observations
can't tell apart all count rather than the likeliest alone.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve

from .posterior import Posterior, _cholesky, _observations, _prior, _row, _rows

_LOG_2PI = math.log(2.0 * math.pi)


class ScaleWeight(NamedTuple):
    """One point of a grid of prior scales and the weight the observations give it."""

    scales: dict[str, float]
    weight: float


def log_evidence(
    x: ArrayLike, y: ArrayLike, noise_var: ArrayLike, prior_mean: ArrayLike, prior_var: ArrayLike
) -> float:
    """The log marginal likelihood of `y`, the effects integrated out.

    The log density of `y` under `N(x prior_mean, x diag(prior_var) x' + diag(noise_var))`; 0 for no observations.

    Args:
        x: The observations' rows, `k x p`.
        y: The observed values, length `k`.
        noise_var: Each observation's noise variance (length `k`), or one variance for all of them.
        prior_mean: The prior's mean of each effect.
        prior_var: The prior's variance of each effect.

    Raises:
        ValueError: As `Posterior` and `Posterior.condition`.
    """
    rows, values, noise = _observations(x, y, noise_var, None, "log_evidence")
    mean, var = _prior(prior_mean, prior_var, rows.shape[1], "log_evidence")
    return _Evidence(rows, values, noise).log(mean, var)


def prior_scale_weights(
    x: ArrayLike,
    y: ArrayLike,
    noise_var: ArrayLike,
    prior: Callable[[dict[str, float]], tuple[ArrayLike, ArrayLike]],
    grid: Mapping[str, Sequence[float]],
) -> list[ScaleWeight]:
    """Every point of a grid of prior scales, weighted by the marginal likelihood it gives the observations.

    The first point returned is the empirical-Bayes choice; the weights are the posterior probability of each point
    under a uniform prior over the grid, i.e. the weights of a `Mixture` of the posteriors under them.

    Args:
        x: The observations' rows, `k x p`.
        y: The observed values, length `k`.
        noise_var: Each observation's noise variance (length `k`), or one variance for all of them.
        prior: A point's scales (group -> sd) -> the prior's mean and variances under them.
        grid: Group -> the sds to try; the points are their product, in `itertools.product` order. An empty grid is
            one point, `{}`.

    Returns:
        Every point with its weight, the weights summing to 1, likeliest first (ties in grid order).

    Raises:
        ValueError: A group with no sd to try, a prior of the wrong length, or as `log_evidence`.
    """
    empty = [group for group, sds in grid.items() if len(sds) == 0]
    if empty:
        raise ValueError(f"prior_scale_weights: groups {empty} have no sd to try")
    rows, values, noise = _observations(x, y, noise_var, None, "prior_scale_weights")
    evidence = _Evidence(rows, values, noise)
    groups = list(grid)
    points = [dict(zip(groups, sds, strict=True)) for sds in itertools.product(*(grid[g] for g in groups))]
    logs = np.array(
        [evidence.log(*_prior(*prior(dict(point)), rows.shape[1], "prior_scale_weights")) for point in points]
    )
    weights = np.exp(logs - logs.max())
    weights /= weights.sum()
    order = sorted(range(len(points)), key=lambda i: -weights[i])
    return [ScaleWeight(points[i], float(weights[i])) for i in order]


class _Evidence:
    """The log marginal likelihood of fixed observations under any diagonal prior."""

    def __init__(self, rows: NDArray[np.float64], values: NDArray[np.float64], noise: NDArray[np.float64]) -> None:
        self.rows, self.values = rows, values
        self.precision = 1.0 / noise
        self.gram = rows.T @ (rows * self.precision[:, None])
        self.cross = rows.T @ (values * self.precision)
        self.constant = len(values) * _LOG_2PI + float(np.sum(np.log(noise)))

    def log(self, prior_mean: NDArray[np.float64], prior_var: NDArray[np.float64]) -> float:
        # Matrix determinant lemma: |x S0 x' + N| = |N| |S0| |S0^-1 + x' N^-1 x|. The quadratic form is the residuals'
        # plus the shrinkage's: two sums of nonnegative terms, so nothing large cancels.
        prior_precision = 1.0 / prior_var
        upper = _cholesky(np.diag(prior_precision) + self.gram)
        mean = cho_solve((upper, False), prior_mean * prior_precision + self.cross)
        residuals = self.values - self.rows @ mean
        shrinkage = mean - prior_mean
        quadratic = float(residuals @ (residuals * self.precision)) + float(shrinkage @ (shrinkage * prior_precision))
        log_det = float(np.sum(np.log(prior_var))) + 2.0 * float(np.sum(np.log(np.diag(upper))))
        return -0.5 * (self.constant + log_det + quadratic)


class Mixture:
    """Posteriors of the same observations under different priors, averaged with weights.

    Answers the same queries as `Posterior`, for the mixture: its mean, and its variance and covariance by the law of
    total variance (the components' average plus the spread of their means). Immutable, like `Posterior`.

    Attributes:
        weights: Each component's weight, normalized to sum to 1.
        components: The posteriors.
        p: The number of effects.
        n: The number of observations conditioned on.
        mean: The mixture's mean, length `p`.
    """

    def __init__(self, components: Sequence[tuple[float, Posterior]]) -> None:
        """Mix the posteriors.

        Args:
            components: (weight, posterior) pairs; the weights need not sum to 1.

        Raises:
            ValueError: No components, a negative or non-finite weight, weights summing to 0, or posteriors with
                different `p` or `n`.
        """
        if not components:
            raise ValueError("Mixture: needs at least one component")
        weights = np.array([float(w) for w, _ in components])
        posteriors = tuple(posterior for _, posterior in components)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(f"Mixture: weights must be finite, >= 0, and not all 0, got {weights.tolist()}")
        first = posteriors[0]
        if any(c.p != first.p or c.n != first.n for c in posteriors):
            raise ValueError("Mixture: components must have the same p and n (the same observations, other priors)")
        self.weights = weights / weights.sum()
        self.weights.flags.writeable = False
        self.components = posteriors
        self.p = first.p
        self.n = first.n
        self.mean = self.weights @ np.vstack([c.mean for c in posteriors])
        self.mean.flags.writeable = False

    def predictive(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The mixture's mean and variance of `x theta` for each row of `x`, without observation noise.

        Args:
            x: Rows, `k x p`; a single row may be 1-D.

        Returns:
            The means and the variances, each of length `k`.

        Raises:
            ValueError: Rows of the wrong length, or a non-finite value.
        """
        rows = _rows(x, self.p, "Mixture.predictive")
        parts = [c.predictive(rows) for c in self.components]
        means = np.vstack([m for m, _ in parts])
        variances = np.vstack([v for _, v in parts])
        mean = self.weights @ means
        return mean, self.weights @ variances + self.weights @ (means - mean) ** 2

    def covariance(self, a: ArrayLike, b: ArrayLike) -> float:
        """The mixture's covariance of `a theta` and `b theta`.

        Args:
            a: One row, length `p`.
            b: Another row, length `p`.

        Raises:
            ValueError: A row of the wrong length, or a non-finite value.
        """
        row_a, row_b = _row(a, self.p, "Mixture.covariance"), _row(b, self.p, "Mixture.covariance")
        within = np.array([c.covariance(row_a, row_b) for c in self.components])
        mean_a = np.array([float(row_a @ c.mean) for c in self.components])
        mean_b = np.array([float(row_b @ c.mean) for c in self.components])
        between = (mean_a - self.weights @ mean_a) * (mean_b - self.weights @ mean_b)
        return float(self.weights @ within + self.weights @ between)

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """`n` draws of `theta`: each draw's component picked by weight, then drawn from it.

        Args:
            n: How many draws.
            rng: The random generator.

        Returns:
            An `n x p` array.

        Raises:
            ValueError: A negative `n`.
        """
        if n < 0:
            raise ValueError(f"Mixture.sample: n must be >= 0, got {n}")
        picks = rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty((n, self.p))
        for j in np.unique(picks):
            out[picks == j] = self.components[j].sample(int(np.sum(picks == j)), rng)
        return out
