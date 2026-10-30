"""What proposers return and are told."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .estimates import Estimate
from .scores import Measured
from .trials import Usage

Value = str | int | float | bool | None
Values = Mapping[str, Value]  # flat typed mapping, dotted keys

ObservationOutcome = Literal["feasible", "infeasible", "ineligible", "invalid", "unscored", "duplicate"]


class Patch(BaseModel):
    """One proposed change to the parent: only the keys to change, not a full candidate.

    `state` is the proposer's private memory, handed back in `observe` and on resume; it must be
    JSON-serializable (checked with `json.dumps` when the patch arrives). `usage` is the proposer's own
    model cost, reported in `search_cost` and never counted against the rollout budget. No proposer
    name: the optimizer knows which block asked, and records it as the candidate's `origin`.
    """

    model_config = ConfigDict(frozen=True)

    updates: dict[str, Value]
    rationale: str
    state: Any = None
    usage: Usage | None = None


class CandidateSummary(BaseModel):
    """One scored candidate as a proposer may see it; train data only.

    Attributes:
        number: Display number, so a prompt can say "version 3".
        measured_objective: The objective from the raw train scores (`CandidateScores.objective`), no model.
        measured: Every metric measured on train, usage included, next to the fit's estimates.
        trial_count: Train trials that count (completed, or `agent_failure` scored by `crash_score`).
        ranking_objective: What the ranking used when the request was built (`Assessment.objective`): the
            fit's resolved estimate, or `measured_objective` without a usable fit.
        rank: Position in that ranking, 1 = best; None for an ineligible candidate.
        estimates: Per goal metric from the fit; empty when assessed by measured means.
        violations: Per constraint, <= 0 when met (`Assessment.violations`); None for an ineligible
            candidate, which has no assessment.
        feasible: Meets every constraint as the ranking assessed it; None for an ineligible candidate.
        changed_keys: Keys changed from its parent.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    parent_id: str | None
    origin: str
    values: dict[str, Value]
    number: int
    measured_objective: float | None
    measured: dict[str, Measured]
    trial_count: int
    ranking_objective: float | None
    rank: int | None
    estimates: dict[str, Estimate]
    violations: list[float] | None
    feasible: bool | None
    eligible: bool
    changed_keys: list[str]


class Observation(BaseModel):
    """Feedback after one of the proposer's own patches was handled; told exactly once per patch.

    Outcomes are labels, not filters: `feasible` = eligible and meeting every constraint as the ranking
    assesses it; `infeasible` = a constraint not met (still a possible parent, ranked lower);
    `ineligible` = some train case unmeasured; `invalid` = the patch failed validation or touched keys
    outside its block (no candidate); `unscored` = the budget couldn't cover it; `duplicate` = its values
    equal an eligible candidate's (`objective` is that candidate's).

    Attributes:
        candidate_id: None for an invalid patch.
        values: The candidate's complete values; None for an invalid patch.
        source_id: The candidate `objective` comes from when it isn't `candidate_id`: a `Nested` round's
            opener is told the round's best. None otherwise.
        objective: The candidate's `Assessment.objective` right after the step's refit (a snapshot); None if
            never measured.
        objectives: Each objective term's raw value (not oriented: the resolved estimate, else the measured
            mean), in objective order, taken with `objective`; what `TPE` is told under `Pareto`. None when
            `objective` is None.
        reason: One sentence from train data, e.g. "tokens <= 4000: mean 4210 violates by 210".
        violations: Per constraint, <= 0 when met, taken with `objective`; None when `objective` is None.
    """

    model_config = ConfigDict(frozen=True)

    patch: Patch
    candidate_id: str | None
    values: dict[str, Value] | None
    source_id: str | None = None
    objective: float | None
    objectives: list[float] | None
    outcome: ObservationOutcome
    reason: str
    violations: list[float] | None


class Evidence(BaseModel):
    """A candidate another block proposed, told to every other block's proposer.

    `objective`, `objectives`, and `violations` are snapshots, as in `Observation`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    values: dict[str, Value]
    origin: str
    objective: float | None
    objectives: list[float] | None
    violations: list[float] | None
    outcome: ObservationOutcome
