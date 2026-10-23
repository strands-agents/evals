"""Scores, splits, and verdicts for the optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Splits:
    """The run's one train / selection / holdout split; every split holds at least one case."""

    train: tuple[str, ...]
    selection: tuple[str, ...]
    holdout: tuple[str, ...]


@dataclass
class CandidateScores:
    """One candidate's scores on one split.

    `per_case` is the mean of that case's measured per-run scalars; a run's scalar is measured only
    when its status is completed. `per_metric` holds per-metric means (usage over completed and
    agent_failure runs), used by constraints.
    """

    candidate_id: str
    per_case: dict[str, float]
    per_metric: dict[str, dict[str, float]]
    per_run: dict[str, list[float | None]]
    failures: int

    def mean(self, cases: Sequence[str]) -> float | None:
        """Mean per-case score over the given cases that were measured; None if none were."""
        raise NotImplementedError

    def subset(self, cases: Sequence[str]) -> dict[str, float]:
        """Per-case scores restricted to the given cases that were measured."""
        raise NotImplementedError


@dataclass(frozen=True)
class SplitScores:
    """One `CandidateScores` per split, never mixed; a split not scored yet is empty."""

    train: CandidateScores
    selection: CandidateScores
    holdout: CandidateScores


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap result on the cases both candidates measured."""

    mean_diff: float
    ci_low: float
    ci_high: float
    n_cases: int


@dataclass(frozen=True)
class AcceptanceVerdict:
    """The holdout gate's decision for one challenger."""

    accepted: bool
    reason: str
    comparison: PairedComparison | None
    violations: list[str]
    coverage: float


@dataclass(frozen=True)
class Admission:
    """Admission on the selection split.

    `infeasible`: a constraint or coverage failed. `dominated`: the selection strategy said no.
    `cases_led` is None unless admitted.
    """

    outcome: Literal["admitted", "infeasible", "dominated"]
    reason: str
    violations: list[str]
    cases_led: list[str] | None
