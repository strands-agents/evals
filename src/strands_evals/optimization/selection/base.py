"""The `Selection` base class: picks the parent, never the champion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from ...types.optimization import SelectionContext, SelectionStep


class Selection(ABC):
    """Decides the parent each iteration (or a merge) and which proposals join the pool.

    Not to be confused with `Search`, which decides which method changes which variables.
    """

    name: ClassVar[str]

    @abstractmethod
    def choose(self, ctx: SelectionContext) -> SelectionStep:
        """The parent, or a merge; an id outside the pool makes the optimizer raise `ValueError`."""

    @abstractmethod
    def admits(self, candidate_id: str, scores: Mapping[str, float], ctx: SelectionContext) -> list[str] | None:
        """The selection cases the candidate leads if admitted, else None."""
