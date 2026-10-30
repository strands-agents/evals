"""The score model's estimate of one metric."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Estimate(BaseModel):
    """One metric's posterior mean and sd, in the metric's own units.

    Saved in `label` events (`CandidateLabel.estimates`) and `selected` events (`ValidationReport.estimates`);
    `effects.json` rows hold mean and ci90 as plain fields instead.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    sd: float

    @property
    def ci90(self) -> tuple[float, float]:
        """`mean ± 1.645 · sd`."""
        raise NotImplementedError

    def quantile(self, q: float) -> float:
        """The `q`-quantile of a normal with this mean and sd.

        Args:
            q: Between 0 and 1.
        """
        raise NotImplementedError
