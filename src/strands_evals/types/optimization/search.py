"""What proposers return and are told."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer

from .estimates import Estimate
from .scores import LabelOutcome, Measured
from .trials import Usage

Value = str | int | float | bool | None
Values = Mapping[str, Value]  # flat typed mapping, dotted keys

ObservationOutcome = Literal[LabelOutcome, "invalid", "duplicate"]


def _violation_from_json(value: Any) -> Any:
    """JSON null back to infinity: how `_violation_to_json` writes a constraint never measured."""
    return math.inf if value is None else value


def _violation_to_json(value: float) -> float | None:
    """Infinity (a constraint never measured) as null, since JSON has no infinity; any other value as is."""
    return None if value == math.inf else value


# One constraint's violation (`Assessment.violations`): <= 0 when met, infinite when its metric was never measured,
# which JSON (the event log) holds as null.
_Violation = Annotated[
    float, BeforeValidator(_violation_from_json), PlainSerializer(_violation_to_json, when_used="json")
]


class Patch(BaseModel):
    """One proposed change to the parent: only the keys to change, not a full candidate.

    `state` is the proposer's private memory, handed back in `observe` and on resume; it must be
    JSON-serializable (checked with `json.dumps` when the patch arrives). `usage` is the proposer's own
    model cost, reported in `search_cost` and never counted against the rollout budget. No proposer
    name: the optimizer knows which block asked, and records it in the candidate's `origin`.
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
        trial_count: Train trials that count (completed, or `agent_failure` scored by `agent_failure_score`).
        failures: Of `trial_count`, the `agent_failure` trials (those `agent_failure_score` scores; 0 under
            `agent_failure_score=None`, which doesn't count them), so `trial_count - failures` completed: how a
            proposer tells a version that crashed from one that scored low.
        ranking_objective: What the ranking used when the request was built (`Assessment.objective`): the
            objective over the fit's resolved estimates (a flagged metric's adjusted mean instead), or without a
            usable fit the objective over the measured means as the ranking adjusts them (`measured_objective`
            unless the candidate has an N/A case that another assessed candidate was judged on).
        term_values: Each objective term's raw value under that ranking, as `Observation.term_values`;
            None for an ineligible candidate.
        rank: Position in that ranking, 1 = best; None for an ineligible candidate.
        estimates: Per goal metric the fit judged (a flagged metric is left out); empty when assessed by
            measured means.
        adjusted_means: Per goal metric the fit didn't judge (the flagged ones under a fit, every one on
            measured means), the adjusted mean the ranking used (`Assessment.adjusted_means`): the plain mean in
            `measured`, or, for a candidate with an N/A case that another assessed candidate was judged on, that
            mean with such cases filled in; empty when the fit judged every goal metric or the candidate is
            ineligible.
        violations: Per constraint, <= 0 when met, infinite when never measured (`Assessment.violations`;
            null in JSON); None for an ineligible candidate, which has no assessment.
        is_feasible: Meets every constraint as the ranking assessed it; None for an ineligible candidate.
        is_eligible: Every train case measured for every goal metric (`CandidateScores.is_eligible`).
        changed_keys: Keys changed from its parent.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    parent_id: str | None
    origin: list[str]
    values: dict[str, Value]
    number: int
    measured_objective: float | None
    measured: dict[str, Measured]
    trial_count: int
    failures: int
    ranking_objective: float | None
    term_values: list[float] | None
    rank: int | None
    estimates: dict[str, Estimate]
    adjusted_means: dict[str, float]
    violations: list[_Violation] | None
    is_feasible: bool | None
    is_eligible: bool
    changed_keys: list[str]


class Observation(BaseModel):
    """Feedback after one of the proposer's own patches was handled; told exactly once per patch.

    Outcomes are labels, not filters: `feasible` = eligible and meeting every constraint as the ranking
    assesses it; `infeasible` = a constraint not met (still a possible parent, ranked lower);
    `ineligible` = some train case, or the objective, unmeasured; `invalid` = the patch failed validation or
    touched keys outside its block (no candidate); `duplicate` = its values equal an eligible candidate's, an
    ineligible one's that a re-score can't change (no train trial left to run, no failed evaluator to judge
    again), or a candidate's proposed earlier in the same step (`objective` is that candidate's).

    Attributes:
        candidate_id: None for an invalid patch; for a duplicate, the existing candidate's id.
        values: The candidate's complete values; None for an invalid patch.
        source_id: The candidate `objective` comes from when it isn't `candidate_id`: a `Nested` round's
            opener is told the round's best. None otherwise.
        round_candidate_ids: For a `Nested` round's opener, the round's candidates (the opener and the
            candidates its steps saved), whichever of them was the best. None otherwise.
        objective: The candidate's `Assessment.objective` right after the step's refit (a snapshot); for a
            duplicate of an existing candidate, told as soon as it is found (before the redraw), the existing
            candidate's from the step's start (None when it is ineligible); for a same-step twin, told after the
            refit, that candidate's then. None if never measured.
        term_values: Each objective term's raw value (not oriented, unlike `Assessment.terms`: the resolved
            estimate, else the adjusted mean the ranking used, `Assessment.adjusted_means`, metric by metric), in
            objective order,
            taken with `objective`; what `TPE` is told under `Pareto`. None when `objective` is None.
        reason: One sentence from train data, e.g. "tokens <= 4000: mean 4210 violates by 210".
        violations: Per constraint, <= 0 when met, infinite when never measured (null in the event log), taken
            with `objective`; None when `objective` is None.
    """

    model_config = ConfigDict(frozen=True)

    patch: Patch
    candidate_id: str | None
    values: dict[str, Value] | None
    source_id: str | None = None
    round_candidate_ids: list[str] | None = None
    objective: float | None
    term_values: list[float] | None
    outcome: ObservationOutcome
    reason: str
    violations: list[_Violation] | None


class Evidence(BaseModel):
    """A scored candidate told to a proposer that didn't propose it: the baseline, told to every block's proposer
    at setup (`origin` is `["baseline"]`), and each candidate another block proposed, told to every other block's
    proposer (except one waiting for an open `Nested` round's end, which is told only the round's best).

    `objective`, `term_values`, and `violations` are snapshots, as in `Observation`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    values: dict[str, Value]
    origin: list[str]
    objective: float | None
    term_values: list[float] | None
    violations: list[_Violation] | None
    outcome: ObservationOutcome
