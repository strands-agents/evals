"""What selection strategies receive and return."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .search import Value


@dataclass(frozen=True)
class MergePlan:
    """A variable-wise merge of two frontier members since their latest common ancestor."""

    first: str
    second: str
    ancestor: str
    values: dict[str, Value]
    from_first: tuple[str, ...]
    from_second: tuple[str, ...]


@dataclass(frozen=True)
class SelectionStep:
    """The parent to propose on, or a merge to score instead of calling a method."""

    parent_id: str
    merge: MergePlan | None = None


@dataclass(frozen=True)
class SelectionContext:
    """The pool as selection sees it: selection-split scores only.

    Attributes:
        scores: Candidate id -> case name -> selection-split score.
        values: Candidate id -> key -> value.
        parents: Candidate id -> parent ids.
        champion_id: The current champion.
        selection_cases: Every selection-split case name.
        rng: Seeded per iteration.
        merges_done: Merges so far in this run.
        merged_pairs: Pairs already merged, each tried once.
        validate: `space.validate`.
        name_of: Returns a key's declared name (the merge unit).
    """

    scores: Mapping[str, Mapping[str, float]]
    values: Mapping[str, Mapping[str, Value]]
    parents: Mapping[str, tuple[str, ...]]
    champion_id: str
    selection_cases: Sequence[str]
    rng: random.Random
    merges_done: int
    merged_pairs: frozenset[frozenset[str]]
    validate: Callable[[Mapping[str, Value]], None]
    name_of: Callable[[str], str]
