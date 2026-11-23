"""`linear_contrast`: one linear combination of the effects, with its sd, 90% interval, and probability of being > 0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from .averaging import Mixture
from .posterior import Posterior

_Z95 = NormalDist().inv_cdf(0.95)


@dataclass(frozen=True)
class Contrast:
    """The posterior of one linear combination of the effects, as a normal with this mean and sd.

    Attributes:
        mean: The posterior mean.
        sd: The posterior sd.
        p_positive: The probability the combination is > 0; with sd 0, 1 if the mean is > 0 else 0.
    """

    mean: float
    sd: float
    p_positive: float

    @property
    def ci90(self) -> tuple[float, float]:
        """The 5% and 95% quantiles: `mean -/+ 1.645 sd`."""
        return (self.mean - _Z95 * self.sd, self.mean + _Z95 * self.sd)


def linear_contrast(posterior: Posterior | Mixture, row: ArrayLike) -> Contrast:
    """The posterior of `row theta`, with the full joint uncertainty of the effects.

    For example, a level's effect is the difference of two cells' rows from `Encoder.row`: effects both rows share
    cancel, and so does their uncertainty. A `Mixture`'s contrast is summarized by its mean and sd, as a normal.

    Args:
        posterior: The posterior over the effects.
        row: The combination's weight on each effect, length `p`.

    Raises:
        ValueError: `row` isn't `p` finite values.
    """
    weights = np.asarray(row, dtype=float)
    if weights.shape != (posterior.p,) or not np.all(np.isfinite(weights)):
        raise ValueError(f"linear_contrast: row must be {posterior.p} finite values, got shape {weights.shape}")
    means, variances = posterior.predictive(weights)
    mean, sd = float(means[0]), math.sqrt(max(float(variances[0]), 0.0))
    return Contrast(mean=mean, sd=sd, p_positive=NormalDist().cdf(mean / sd) if sd > 0 else float(mean > 0))
