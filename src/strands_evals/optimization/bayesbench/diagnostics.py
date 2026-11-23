"""Held-out predictive checks: is the model more wrong than its own uncertainty says?

Each new observation is scored against the posterior from before it was absorbed, `z = (y - mean) / sqrt(var +
noise_var)` with the prediction's mean and variance and the observation's noise variance; a well-specified model gives
`z ~ N(0, 1)`. The z-scores are tested in disjoint windows of `window`, and a complete window flags when either

- `sum(z^2) > chi2.ppf(level, df=window)`: overdispersion, errors larger than the model admits; or
- `|mean(z)| > norm.ppf((1 + level) / 2) / sqrt(window)`: bias, errors to one side.

The flag is sticky: once any window flags, the check stays flagged (a misspecification doesn't un-happen). The
defaults, disjoint windows of 15 at level 0.995, are BayesBench's calibrated rule: rolling windows retest each z-score
about `window` times, so their per-test false-alarm rate compounds.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

WINDOW = 15
LEVEL = 0.995
BOUNDARY = (0.05, 0.95)


@dataclass(frozen=True)
class Diagnostics:
    """A snapshot of a `HeldOutDiagnostics`.

    Attributes:
        n_holdout: The observations scored before being absorbed.
        z_mean: The mean of every z-score so far; 0 with none.
        z_sq_sum: The sum of their squares; 0 with none.
        flagged: Whether any complete window flagged.
        near_boundary: The labels of the observations outside the boundary, in order.
        messages: What a reader should know: the first window incomplete, the flag, observations near the boundary.
    """

    n_holdout: int
    z_mean: float
    z_sq_sum: float
    flagged: bool
    near_boundary: tuple[Hashable, ...]
    messages: tuple[str, ...]


class HeldOutDiagnostics:
    """Held-out z-scores of new observations, tested in disjoint windows for misspecification.

    Record each observation before the posterior absorbs it, with that posterior's prediction of it.

    Attributes:
        window: The z-scores per test.
        level: Each test's level; a well-specified model's window flags with probability at most `2 (1 - level)`.
        boundary: Observed values outside `[low, high]` are noted (a Gaussian approximation of a bounded score
            degrades near its bounds); None notes none.
    """

    def __init__(
        self, window: int = WINDOW, level: float = LEVEL, boundary: tuple[float, float] | None = BOUNDARY
    ) -> None:
        """Start with no observations.

        Args:
            window: The z-scores per test.
            level: Each test's level.
            boundary: The `(low, high)` range outside which an observed value is noted; None notes none.

        Raises:
            ValueError: A `window` below 1, or a `level` not strictly between 0 and 1.
        """
        if window < 1:
            raise ValueError(f"HeldOutDiagnostics: window must be >= 1, got {window}")
        if not 0 < level < 1:
            raise ValueError(f"HeldOutDiagnostics: level must lie strictly between 0 and 1, got {level}")
        self.window = window
        self.level = level
        self.boundary = boundary
        self._chi2_bound = float(chi2.ppf(level, df=window))
        self._bias_bound = float(norm.ppf((1 + level) / 2)) / math.sqrt(window)
        self._z: list[float] = []
        self._near_boundary: list[Hashable] = []
        self._flagged = False

    def record(
        self, y: float, predicted_mean: float, predicted_var: float, noise_var: float, label: Hashable = None
    ) -> float:
        """Score one observation against the posterior from before it; test the window it completes, if any.

        Args:
            y: The observed value.
            predicted_mean: The earlier posterior's mean of it.
            predicted_var: The earlier posterior's variance of it, without noise.
            noise_var: The observation's noise variance.
            label: Names the observation in `Diagnostics.near_boundary`, e.g. its cell's key.

        Returns:
            Its z-score.

        Raises:
            ValueError: A non-finite value, a negative variance, or both variances 0.
        """
        values = (y, predicted_mean, predicted_var, noise_var)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"HeldOutDiagnostics: values must be finite, got {values}")
        if predicted_var < 0 or noise_var < 0 or predicted_var + noise_var <= 0:
            raise ValueError(
                f"HeldOutDiagnostics: variances must be >= 0 and not both 0, got predicted_var={predicted_var}, "
                f"noise_var={noise_var}"
            )
        z = float((y - predicted_mean) / math.sqrt(predicted_var + noise_var))
        self._z.append(z)
        if self.boundary is not None and not (self.boundary[0] <= y <= self.boundary[1]):
            self._near_boundary.append(label)
        if len(self._z) % self.window == 0:
            completed = np.array(self._z[-self.window :])
            if np.sum(completed**2) > self._chi2_bound or abs(np.mean(completed)) > self._bias_bound:
                self._flagged = True
        return z

    @property
    def flagged(self) -> bool:
        """Whether any complete window flagged."""
        return self._flagged

    @property
    def n_holdout(self) -> int:
        """The observations recorded."""
        return len(self._z)

    @property
    def z_scores(self) -> tuple[float, ...]:
        """Every z-score so far, in order."""
        return tuple(self._z)

    def snapshot(self) -> Diagnostics:
        """The z-scores' summary, the flag, and messages."""
        z = np.array(self._z)
        messages = []
        if len(z) < self.window:
            messages.append(f"first window not complete ({len(z)}/{self.window}); no flagging yet")
        if self._flagged:
            messages.append(
                "a held-out z window exceeded its calibrated bound: the model's predictions are more wrong than its "
                "own uncertainty admits (possible unmodeled interaction structure)"
            )
        if self._near_boundary and self.boundary is not None:
            messages.append(
                f"{len(self._near_boundary)} cell(s) with score outside [{self.boundary[0]}, {self.boundary[1]}]: "
                "Gaussian approximation degrades near the boundary"
            )
        return Diagnostics(
            n_holdout=len(z),
            z_mean=float(z.mean()) if len(z) else 0.0,
            z_sq_sum=float(np.sum(z**2)) if len(z) else 0.0,
            flagged=self._flagged,
            near_boundary=tuple(self._near_boundary),
            messages=tuple(messages),
        )
