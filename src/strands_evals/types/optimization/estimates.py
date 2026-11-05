"""The score model's estimate of one metric."""

from __future__ import annotations

from statistics import NormalDist

from pydantic import BaseModel, ConfigDict


class Estimate(BaseModel):
    """One metric's posterior mean and sd, in the metric's own units.

    Saved in `label` events (`CandidateLabel.estimates`) and `picked` events (`ValidationReport.estimates`);
    `effects.json` rows hold mean and ci90 as plain fields instead.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    sd: float

    @property
    def ci90(self) -> tuple[float, float]:
        """The 5% and 95% quantiles: `mean ± 1.645 · sd`."""
        return (self.quantile(0.05), self.quantile(0.95))

    def quantile(self, q: float) -> float:
        """The `q`-quantile of a normal with this mean and sd; the mean when sd is 0.

        Args:
            q: Strictly between 0 and 1.

        Raises:
            ValueError: `q` is not strictly between 0 and 1.
        """
        if not 0 < q < 1:
            raise ValueError(f"q must lie strictly between 0 and 1, got {q!r}")
        return self.mean + NormalDist().inv_cdf(q) * self.sd
