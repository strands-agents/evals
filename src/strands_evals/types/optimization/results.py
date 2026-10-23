"""Per-iteration result records, read by `on_iteration`, resume, and the results API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .scores import AcceptanceVerdict


@dataclass
class ProposalOutcome:
    """One proposal of one iteration.

    `holdout_score` and `verdict` are None unless the proposal was the iteration's challenger.
    """

    candidate_id: str | None
    parent_ids: tuple[str, ...]
    block: str | None
    method: str | None
    changed_keys: list[str]
    rationale: str
    status: str
    selection_score: float | None
    holdout_score: float | None
    verdict: AcceptanceVerdict | None


@dataclass
class IterationRecord:
    """One iteration; `parent_ids` holds one parent, or two for a merge."""

    iteration: int
    champion_id: str
    parent_ids: tuple[str, ...]
    step: Literal["propose", "merge"]
    proposals: list[ProposalOutcome]
    rollouts_spent: int
