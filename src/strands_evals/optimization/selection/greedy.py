"""Greedy: the parent is always the champion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ...types.optimization import SelectionContext, SelectionStep
from .base import Selection


class Greedy(Selection):
    """Parent = champion, no merges; admits every proposal that passed constraints and coverage."""

    name: ClassVar[str] = "greedy"

    def choose(self, ctx: SelectionContext) -> SelectionStep:
        raise NotImplementedError

    def admits(self, candidate_id: str, scores: Mapping[str, float], ctx: SelectionContext) -> list[str] | None:
        """The cases the candidate leads, possibly `[]`, never None."""
        raise NotImplementedError
