"""The escalation band a `Cascade` gates on.

The routing logic lives in `strands_evals.evaluators.cascade`.
"""

from typing import Annotated

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from .typed_rubric import _Text

_Probability = Annotated[float, Field(strict=True, ge=0, le=1)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Band:
    """The range of a rubric question's probability that counts as uncertain.

    A `Cascade` sends a case to its fallback evaluator when the gating probability `p_gate` of
    `question` falls inside the band, ends included. `p_gate` is P(true) for a `Noul`, and the
    most likely option's probability for a `Scale` or `Choice`.

    A band belongs to the decision-model checkpoint it was chosen for. Pick it again after
    changing or upgrading the backend.

    Attributes:
        question: The id of the rubric question that gates escalation.
        low: The lowest uncertain probability, in [0, 1].
        high: The highest uncertain probability, in [low, 1].
    """

    question: _Text
    low: _Probability
    high: _Probability

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must not be greater than high ({self.high})")

    def contains(self, probability: float) -> bool:
        """Whether `probability` is inside the band, ends included."""
        return self.low <= probability <= self.high


# Validates a band, or decodes the dict that `_BAND_ADAPTER.dump_python` writes.
_BAND_ADAPTER: TypeAdapter[Band] = TypeAdapter(Band)
