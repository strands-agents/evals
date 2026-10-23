"""Rebuilding a run's state from its directory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...types.optimization import IterationRecord, Observation, Usage
from .run_directory import RunDirectory


@dataclass
class ResumeState:
    """State up to the last `iteration_end` event; a killed iteration is discarded and re-run.

    Attributes:
        pool: Every admitted candidate plus the baseline.
        candidates: Every saved candidate, in the pool or not, so a repeat of one is still a
            duplicate after resume; their scores are rebuilt from their saved trials.
        parents: Candidate id -> parent ids.
        turn: Completed proposing iterations (for `Schedule.blocks_for`).
        exhausted: Blocks that produced zero candidates on their latest turn.
        merges_done: Merges scored so far (capped at 5 per run); `merged_pairs` also holds pairs
            skipped because the merged values already existed.
        observations: Per block name, in lineage order; replayed through each block's
            `method.observe()` before the first new iteration.
        iterations: Rebuilt from the lineage, so the result covers the whole run.
        search_cost: Summed from the `observe` events' `Patch.usage`.
    """

    pool: list[str]
    candidates: list[str]
    parents: dict[str, tuple[str, ...]]
    champion_id: str
    spent: int
    iteration: int
    turn: int
    exhausted: set[str]
    merges_done: int
    merged_pairs: set[frozenset[str]]
    observations: dict[str, list[Observation]]
    iterations: list[IterationRecord]
    search_cost: Usage


def load_state(run: RunDirectory, *, config: Mapping[str, Any]) -> ResumeState:
    """Compare `config` with `run.json` field by field, then rebuild the state from the lineage.

    A mismatch raises `ValueError` naming the field. Budget, `max_workers`, and `on_iteration` may
    differ.
    """
    raise NotImplementedError
