"""What search methods receive and return."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .trials import Usage

if TYPE_CHECKING:
    from ...optimization.candidates.candidate import Candidate
    from ...optimization.declaration.space import Space

Value = str | int | float | bool | None

ObservationOutcome = Literal["accepted", "rejected", "admitted", "dominated", "infeasible", "unscored", "duplicate"]


@dataclass
class Patch:
    """One proposed change to the parent: only the keys to change, not a full candidate.

    `state` is the method's private memory, handed back in `observe` and on resume; it must be
    JSON-serializable. `usage` is the proposer's own model cost, reported in `search_cost` and
    never counted against the rollout budget.
    """

    updates: dict[str, Value]
    rationale: str
    method: str
    state: Any = None
    usage: Usage | None = None


@dataclass(frozen=True)
class ReflectionRecord:
    """One of the parent's worst train-split cases, as shown to a proposer."""

    case: str
    score: float
    input: str
    output: str
    feedback: list[str]
    trajectory: str | None


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a method may see when proposing; never holdout data.

    `log(event, **fields)` writes a lineage event. `attempt` is the redraw number within this
    iteration and block (0 = first ask).
    """

    space: Space
    keys: Sequence[str]
    parent: Candidate
    records: Sequence[ReflectionRecord]
    n: int
    iteration: int
    seed: int
    attempt: int
    log: Callable[..., None]


@dataclass(frozen=True)
class Observation:
    """Feedback after a patch was handled; told exactly once per patch.

    `scalar` is the selection-split objective, None if never measured. `reason` comes from
    selection-split data only; gate outcomes say only "rejected at the holdout gate".
    """

    patch: Patch
    scalar: float | None
    outcome: ObservationOutcome
    reason: str
