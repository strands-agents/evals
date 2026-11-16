"""The `Selection` base class and its context: picks the parent, never what ships."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...types.optimization import Values
from ..score_model.protocols import Fit
from ..score_model.ranking import Assessment


@dataclass(frozen=True)
class SelectionContext:
    """The step's possible parents as selection sees them: train data only.

    A dataclass, not a record: it holds the fit and the RNG, and is never saved.

    Attributes:
        ranking: The step's possible parents (`Step.parents`, else every eligible candidate so far, the
            baseline and infeasible ones included), in ranking order (ties: earlier first); never empty.
            Ineligible candidates and invalid, duplicate, or unscored proposals are not in it.
        assessments: The ranking's `Assessment` for every candidate in `ranking`.
        scores: Candidate id -> case name -> train objective (`CandidateScores.objective_by_case`).
        values: Candidate id -> key -> value.
        parents: Candidate id -> parent id (None for the baseline).
        trial_counts: Candidate id -> train trials that count.
        fit: Read-only; None before 2 candidates are eligible.
        iteration: The current iteration.
        history: Parents chosen so far, oldest first.
        rng: `random.Random(f"{seed}:{iteration}")`, so a resumed run picks the same parents.
    """

    ranking: Sequence[str]
    assessments: Mapping[str, Assessment]
    scores: Mapping[str, Mapping[str, float]]
    values: Mapping[str, Values]
    parents: Mapping[str, str | None]
    trial_counts: Mapping[str, int]
    fit: Fit | None
    iteration: int
    history: Sequence[str]
    rng: random.Random


class Selection(ABC):
    """Picks the parent each iteration; it never decides what ships.

    Not to be confused with `Search`, which decides which proposer changes which variables.
    An ABC because users subclass it (like `Proposer`); internal seams such as `TrialCache` are
    `Protocol`s.

    Attributes:
        name: Stable strategy name, stored in `run.json` with `config()` and compared on resume.
    """

    name: ClassVar[str]

    @abstractmethod
    def choose(self, ctx: SelectionContext) -> str:
        """The parent's id; one outside `ctx.ranking` makes the optimizer raise `ValueError`.

        Take all randomness from `ctx.rng`. Beyond the ranking, a custom strategy can explore: random
        top-k, Thompson over sampled effects, UCB (mean + k·sd), annealing over the iteration, novelty
        via `ctx.history`.
        """

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume."""
        return {}
