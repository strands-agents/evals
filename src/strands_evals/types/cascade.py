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

    "Uncertain" means near the rubric's pass line, not near 0.5: the band must contain the gating
    question's pass line in `p_gate` terms, or the cases where the verdict flips never escalate.
    A rubric of one `Noul` with `expected=False` and `pass_threshold=0.86` passes a case when
    P(true) <= 0.14, so a band of [0.08, 0.20] contains its line.

    A band and its pass line belong to the decision-model checkpoint they were chosen for. Pick
    them again after changing or upgrading the backend.

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
