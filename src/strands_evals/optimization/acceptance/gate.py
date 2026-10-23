"""The holdout gate: the only thing that changes the champion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...types.optimization import AcceptanceVerdict, CandidateScores
from ..scoring.objectives import Criteria


@dataclass(frozen=True)
class Gate:
    """Acceptance settings; there is no way to turn the gate off.

    Accept only if the challenger's mean holdout improvement exceeds `margin` and the bootstrap CI's
    lower bound is > 0. The false-accept rate per gate test is `(1 - confidence) / 2`
    (0.95 -> 2.5%), not `1 - confidence`.
    """

    runs_per_candidate: int = 3
    holdout_fraction: float = 0.3
    selection_fraction: float = 0.3
    margin: float = 0.0
    confidence: float = 0.95
    min_coverage: float = 0.9


def decide(
    gate: Gate,
    challenger: CandidateScores,
    champion: CandidateScores,
    holdout: Sequence[str],
    criteria: Criteria,
    *,
    seed: int | str,
) -> AcceptanceVerdict:
    """Ordered checks on the holdout: coverage, constraints, tie, margin, CI. Ties reject."""
    raise NotImplementedError


def covers(candidate: CandidateScores, parent: CandidateScores, cases: Sequence[str], min_coverage: float) -> bool:
    """Whether the candidate measured at least `min_coverage` of the cases the parent measured."""
    raise NotImplementedError
