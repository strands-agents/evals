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
        parent: The candidate to change: the step's parent, or in a step with several blocks the slot's
            values so far (the parent with the earlier blocks' structural patches merged; not yet scored).
        trials: The step's parent's train trial records (Reflection builds its evidence from them); it
            always has them, since every candidate is scored on train. Each trajectory is as the run
            directory reads it back from its transcript, in a fresh run and after a resume alike.
        scores: The step's parent's train scores (`SplitScores.train`).
        inputs: Train case name -> `case.input`, for the cases in `trials`. Inputs only: no expected field
            (`expected_output`, `expected_trajectory`, ...) and no `metadata` is ever passed.
        fit: Read-only; None before 2 candidates are eligible, after a refit that raised, or while any
            metric is flagged.
        candidates: Every scored candidate so far: eligible ones in ranking order, then the ineligible in
            scoring order. Invalid and duplicate proposals never appear.
        n: How many patches to return: the step's `Step.n_candidates` (at most `proposals_per_iteration`),
            or fewer when the budget can't score them all, on a redraw (only the shortfall), for a
            concurrent proposer (1 per call), or in a step with several blocks (1 per slot).
        iteration: So a proposer can seed per iteration.
        seed: The run seed.
        redraw: The redraw number within this iteration and block (0 = first ask; `Search.redraws` caps it).
        criteria: Combines per-metric estimates or sampled values into the objective and checks constraints
            (`criteria.feasibility_threshold` is the P(met) threshold the ranking uses).
        log: `log(event, **fields)` writes an event; a proposer may write only `screened`, `proposer_retry`,
            and `proposer_usage` (any other name raises `ValueError`). `proposer_usage` (`usage=` a `Usage`) is
            a `propose` call's model spend that isn't on any patch it returns (in one event or several): each
            token is reported once, on a returned patch's `usage` or in such an event, never both.
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
            one `propose` call with n in a single-block step, else one n=1 call per slot (structural blocks
            first in each slot); non-concurrent proposers are called one at a time.
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
        """Called once per scored candidate another block proposed, except while this block waits for an open
        `Nested` round's end (then it is told only the round's best); does nothing by default.

        Each call is an `evidence` event, replayed on resume like `observe`.
        """

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume.

        Override it so a changed setting is caught on resume; the default `{}` checks nothing.
        """
        return {}
