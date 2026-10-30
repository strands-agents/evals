"""The `Proposer` base class and its request: proposes values, never decides what ships."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...types.optimization import CandidateScores, CandidateSummary, Evidence, Observation, Patch, TrialRecord
from ..candidate import Candidate
from ..declaration.space import ParamKind, Space
from ..score_model.protocols import Fit
from ..scoring.objectives import Criteria


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a proposer may see when proposing: train data only, never validation or test.

    A dataclass, not a record: it holds the space, the parent, the fit, and a callback, and is never saved.

    Attributes:
        space: To read types, bounds, and defaults.
        keys: The variables this block owns; a patch may only touch these.
        parent: The candidate to change.
        trials: The parent's train trial records (Reflection builds its evidence from them); the parent
            always has them, since every candidate is scored on train.
        scores: The parent's train scores (`SplitScores.train`).
        inputs: Train case name -> `case.input`, for the cases in `trials`. Inputs only: `expected_output`
            is never passed.
        fit: Read-only; None before 2 candidates are eligible.
        candidates: Every scored candidate so far: eligible ones in ranking order, then the ineligible in
            scoring order. Invalid, duplicate, and unscored proposals never appear.
        n: How many patches to return.
        iteration: So a proposer can seed per iteration.
        seed: The run seed.
        redraw: The redraw number within this iteration and block (0 = first ask; `Search.redraws` caps it).
        criteria: Combines per-metric estimates or sampled values into the objective and checks constraints
            (`criteria.feasibility` is the P(met) threshold the ranking uses).
        log: `log(event, **fields)` writes an event; a proposer may write only `screened` and
            `proposer_retry` (any other name raises `ValueError`).
    """

    space: Space
    keys: Sequence[str]
    parent: Candidate
    trials: Sequence[TrialRecord]
    scores: CandidateScores
    inputs: Mapping[str, Any]
    fit: Fit | None
    candidates: Sequence[CandidateSummary]
    n: int
    iteration: int
    seed: int
    redraw: int
    criteria: Criteria
    log: Callable[..., None]


class Proposer(ABC):
    """Proposes patches on a parent and learns from how each one did; never decides what ships.

    An ABC because users subclass it.

    Attributes:
        kinds: The variable types it can propose.
        concurrent: True when it keeps no state between proposals (Reflection): the optimizer makes a
            request's n proposals as n `propose` calls with n=1 at once, Patches kept in call order. False:
            one `propose` call with n; non-concurrent blocks propose one at a time, in block order.
    """

    kinds: ClassVar[frozenset[ParamKind]]
    concurrent: ClassVar[bool] = False

    @abstractmethod
    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Return up to `request.n` patches; `[]` means nothing useful this time (a redraw cause).

        Raising means the proposer itself failed and ends the run with `stop_reason="proposer_error"`.
        """

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        """Called exactly once per patch this proposer returned."""

    def evidence(self, evidence: Evidence) -> None:  # noqa: B027 (optional hook, empty on purpose)
        """Called once per scored candidate another block proposed; does nothing by default.

        Each call is an `evidence` event, replayed on resume like `observe`.
        """

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume.

        Override it so a changed setting is caught on resume; the default `{}` checks nothing.
        """
        return {}
