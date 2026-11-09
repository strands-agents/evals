"""`Posterior`: the conjugate Gaussian posterior of a linear model with known observation noise.

The `p` effects `theta` have a diagonal Gaussian prior, and each observation is `y = x theta + e` with
`e ~ N(0, noise_var)`, its noise variance known. The posterior is Gaussian, with

    precision = diag(1 / prior_var) + sum over observations of x' x / noise_var
    mean      = precision^-1 (prior_mean / prior_var + sum over observations of x' y / noise_var)

held as the Cholesky factor of the precision: no inverse is ever formed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve, cholesky, solve_triangular


class Posterior:
    """A Gaussian posterior over linear effects, conjugate to observations with known noise.

    Immutable: `condition` returns a new posterior, and the arrays it exposes are read-only. Conditioning on
    observations a batch at a time gives the posterior of conditioning on all of them at once.

    Attributes:
        prior_mean: The prior's mean, length `p`.
        prior_var: The prior's variances (its covariance is diagonal), length `p`.
        p: The number of effects.
        n: The number of observations conditioned on.
        mean: The posterior mean, length `p`.
    """

    def __init__(self, prior_mean: ArrayLike, prior_var: ArrayLike) -> None:
        """The prior, before any observation.

        Args:
            prior_mean: The prior's mean of each effect.
            prior_var: The prior's variance of each effect.

        Raises:
            ValueError: Not two 1-D arrays of one nonzero length, a non-finite value, or a variance <= 0.
        """
        mean, var = _prior(prior_mean, prior_var, None, "Posterior")
        self._set(mean, var, np.diag(1.0 / var), mean / var, 0)

    def _set(
        self,
        prior_mean: NDArray[np.float64],
        prior_var: NDArray[np.float64],
        precision: NDArray[np.float64],
        shift: NDArray[np.float64],
        n: int,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.p = len(prior_mean)
        self.n = n
        self._precision = precision
        self._shift = shift  # prior_mean / prior_var + the sum of x' y / noise_var
        self._upper = _cholesky(precision)  # precision = upper' upper
        self.mean = cho_solve((self._upper, False), shift)
        self.mean.flags.writeable = False

    def condition(self, x: ArrayLike, y: ArrayLike, noise_var: ArrayLike) -> Posterior:
        """The posterior after also observing `y` at the rows `x`.

        Args:
            x: The observations' rows, `k x p`; a single row may be 1-D.
            y: The observed values, length `k`.
            noise_var: Each observation's noise variance (length `k`), or one variance for all of them.

        Raises:
            ValueError: Mismatched shapes, a non-finite value, or a noise variance <= 0.
        """
        rows, values, noise = _observations(x, y, noise_var, self.p, "Posterior.condition")
        out = Posterior.__new__(Posterior)
        out._set(
            self.prior_mean,
            self.prior_var,
            self._precision + rows.T @ (rows / noise[:, None]),
            self._shift + rows.T @ (values / noise),
            self.n + len(values),
        )
        return out

    def predictive(self, x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The posterior mean and variance of `x theta` for each row of `x`, without observation noise.

        Args:
            x: Rows, `k x p`; a single row may be 1-D.

        Returns:
            The means and the variances, each of length `k`.

        Raises:
            ValueError: Rows of the wrong length, or a non-finite value.
        """
        rows = _rows(x, self.p, "Posterior.predictive")
        whitened = solve_triangular(self._upper, rows.T, trans="T")  # x Sigma x' = |upper'^-1 x'|^2
        return rows @ self.mean, np.sum(whitened * whitened, axis=0)

    def covariance(self, a: ArrayLike, b: ArrayLike) -> float:
        """The posterior covariance of `a theta` and `b theta`.

        Args:
            a: One row, length `p`.
            b: Another row, length `p`.

        Raises:
            ValueError: A row of the wrong length, or a non-finite value.
        """
        rows = np.vstack([_row(a, self.p, "Posterior.covariance"), _row(b, self.p, "Posterior.covariance")])
        whitened = solve_triangular(self._upper, rows.T, trans="T")
        return float(whitened[:, 0] @ whitened[:, 1])

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """`n` draws of `theta` from the posterior.

        Args:
            n: How many draws.
            rng: The random generator.

        Returns:
            An `n x p` array.

        Raises:
            ValueError: A negative `n`.
        """
        if n < 0:
            raise ValueError(f"Posterior.sample: n must be >= 0, got {n}")
        z = rng.standard_normal((self.p, n))
        return (self.mean[:, None] + solve_triangular(self._upper, z)).T  # upper^-1 z has covariance precision^-1


def _cholesky(precision: NDArray[np.float64]) -> NDArray[np.float64]:
    """The upper Cholesky factor of `precision`, retried with jitter on the diagonal (1e-9, x10 up to 1e-3).

    Raises:
        numpy.linalg.LinAlgError: Not positive definite even with the largest jitter.
    """
    jitter = 0.0
    while True:
        try:
            return cholesky(precision + jitter * np.eye(len(precision)), lower=False)
        except np.linalg.LinAlgError:
            jitter = 1e-9 if jitter == 0.0 else jitter * 10
            if jitter > 1e-3:
                raise


def _prior(
    prior_mean: ArrayLike, prior_var: ArrayLike, p: int | None, caller: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read-only copies of a diagonal prior's mean and variances, checked (and of length `p` unless None)."""
    mean = np.array(prior_mean, dtype=float)
    var = np.array(prior_var, dtype=float)
    if mean.ndim != 1 or mean.shape != var.shape or len(mean) == 0:
        raise ValueError(
            f"{caller}: prior_mean and prior_var must be 1-D of one nonzero length, got shapes {mean.shape} and "
            f"{var.shape}"
        )
    if p is not None and len(mean) != p:
        raise ValueError(f"{caller}: the prior needs {p} effects, one per column, got {len(mean)}")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var))):
        raise ValueError(f"{caller}: prior_mean and prior_var must be finite")
    if np.any(var <= 0):
        raise ValueError(f"{caller}: prior variances must be > 0, got min {float(var.min())}")
    mean.flags.writeable = False
    var.flags.writeable = False
    return mean, var


def _row(a: ArrayLike, p: int, caller: str) -> NDArray[np.float64]:
    """One finite row of length `p`."""
    row = np.asarray(a, dtype=float)
    if row.shape != (p,) or not np.all(np.isfinite(row)):
        raise ValueError(f"{caller}: a row must be {p} finite values, got shape {row.shape}")
    return row


def _rows(x: ArrayLike, p: int, caller: str) -> NDArray[np.float64]:
    """Finite rows of length `p`, as a 2-D array; a 1-D `x` is one row."""
    rows = np.asarray(x, dtype=float)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != p:
        raise ValueError(f"{caller}: rows must have {p} columns, got shape {rows.shape}")
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{caller}: rows must be finite")
    return rows


def _observations(
    x: ArrayLike, y: ArrayLike, noise_var: ArrayLike, p: int | None, caller: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The rows, values, and per-observation noise variances, checked; `p` None takes it from `x`."""
    values = np.atleast_1d(np.asarray(y, dtype=float))
    if values.ndim != 1:
        raise ValueError(f"{caller}: y must be 1-D, got shape {values.shape}")
    raw = np.asarray(x, dtype=float)
    if p is None:
        if raw.ndim not in (1, 2):
            raise ValueError(f"{caller}: x must be one row or a 2-D array of rows, got shape {raw.shape}")
        p = raw.shape[-1]
    rows = _rows(raw, p, caller)
    if len(rows) != len(values):
        raise ValueError(f"{caller}: x has {len(rows)} rows but y has {len(values)} values")
    try:
        noise = np.broadcast_to(np.asarray(noise_var, dtype=float), values.shape)
    except ValueError:
        raise ValueError(
            f"{caller}: noise_var must be one variance or one per value ({len(values)}), got shape "
            f"{np.shape(noise_var)}"
        ) from None
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(noise))):
        raise ValueError(f"{caller}: y and noise_var must be finite")
    if np.any(noise <= 0):
        raise ValueError(f"{caller}: noise variances must be > 0, got min {float(noise.min())}")
    return rows, values, noise
