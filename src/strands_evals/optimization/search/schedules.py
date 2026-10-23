"""Schedules: which blocks propose in a given iteration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import Block


class Schedule(ABC):
    """`turn` counts earlier proposing iterations; a merge iteration is not a turn."""

    @abstractmethod
    def blocks_for(self, turn: int, blocks: Sequence[Block]) -> list[Block]: ...


class Joint(Schedule):
    """Every block, every iteration; each patch is its own candidate changing only its block."""

    def blocks_for(self, turn: int, blocks: Sequence[Block]) -> list[Block]:
        raise NotImplementedError


class Alternate(Schedule):
    """One block per iteration, in block order, so an accepted change touched one block."""

    def blocks_for(self, turn: int, blocks: Sequence[Block]) -> list[Block]:
        raise NotImplementedError
