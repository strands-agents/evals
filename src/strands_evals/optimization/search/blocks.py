"""Blocks (a method owning some variables) and the search that holds them."""

from __future__ import annotations

from collections.abc import Sequence

from ..declaration.space import Space
from .method import Method
from .schedules import Schedule


class Block:
    """One method owns some variables; a top-level name expands to every key under it.

    Attributes:
        method: The method proposing values for these variables.
        variables: The declared names or keys as given.
        name: Defaults to `type(method).__name__`. It keys `observe` events and the resume state,
            so it must be stable across resume.
    """

    method: Method
    variables: tuple[str, ...]
    name: str

    def __init__(self, method: Method, *variables: str, name: str | None = None):
        raise NotImplementedError


class Search:
    """Which method changes which variables, and the order blocks take turns in.

    Not to be confused with `Selection`, which picks the parent each iteration.
    Attributes:
        blocks: The user's blocks, in order.
        schedule: Which blocks propose each iteration; `None` means `Alternate()`.
        redraws: Caps re-asks per block per iteration after a duplicate, an invalid patch, a patch
            touching keys outside its block, or no patches at all.
    """

    blocks: tuple[Block, ...]
    schedule: Schedule
    redraws: int

    def __init__(self, blocks: Sequence[Block] = (), schedule: Schedule | None = None, redraws: int = 3):
        """Two blocks with the same name (e.g. two unnamed TPE blocks) raise `ValueError`."""
        raise NotImplementedError

    def resolve(self, space: Space) -> list[tuple[Block, list[str]]]:
        """Each block with its expanded keys; raises `ValueError`.

        Variables in no block go to their type's default block (text/file -> one Reflection block,
        numbers/choices -> one TPE block), appended after the user's blocks, so `Search()` is the
        default search. A variable in two blocks is an error; an automatic TPE block without optuna
        raises `ImportError` naming the variable and the install command.
        """
        raise NotImplementedError
