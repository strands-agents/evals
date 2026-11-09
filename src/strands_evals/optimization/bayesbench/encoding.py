"""`Encoder`: a cell's row of the design matrix, by dummy coding.

Column order is fixed (`Encoder.column_names` names each column):

1. the intercept;
2. main effects: factors in `FactorSpace.names` order, each factor's levels in order, its reference skipped;
3. with `interactions=True`, two-way interactions: factor pairs in `itertools.combinations` order, each pair's
   columns in (level of the first, level of the second) product order, references skipped.

A 5 x 4 x 5 grid has 1 + (4 + 3 + 4) + (12 + 16 + 12) = 52 columns.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .space import FactorSpace


class Encoder:
    """Dummy-codes the cells of a `FactorSpace`: an intercept, main effects, and optionally two-way interactions.

    A cell's row has a 1 in the intercept, in each main-effect column of its non-reference levels, and (with
    `interactions`) in each interaction column of its pairs of non-reference levels; every other entry is 0. The
    reference cell (every factor at its reference) is the intercept alone.

    Attributes:
        space: The factors encoded.
        interactions: Whether two-way interaction columns are included.
        p: The number of columns.
        column_names: Each column's name: `"intercept"`, `"factor=level"`, or `"factor=level:other=level"`.
        main_columns: Factor -> its main-effect columns, one per non-reference level, in level order.
        interaction_columns: (factor, other), in factor order -> their interaction columns; empty without
            `interactions` or with one factor.
    """

    def __init__(self, space: FactorSpace, *, interactions: bool = True) -> None:
        """Lay out the columns.

        Args:
            space: The factors to encode.
            interactions: Include a column per pair of non-reference levels of every two factors.
        """
        self.space = space
        self.interactions = interactions
        self._main: dict[tuple[str, str], int] = {}
        self._pair: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
        self.column_names: list[str] = ["intercept"]
        for factor in space.names:
            for level in space.factors[factor][1:]:
                self._main[(factor, level)] = len(self.column_names)
                self.column_names.append(f"{factor}={level}")
        self._pairs: list[tuple[str, str]] = list(itertools.combinations(space.names, 2)) if interactions else []
        self.interaction_columns: dict[tuple[str, str], list[int]] = {}
        for first, second in self._pairs:
            columns = self.interaction_columns.setdefault((first, second), [])
            for level_a in space.factors[first][1:]:
                for level_b in space.factors[second][1:]:
                    self._pair[((first, level_a), (second, level_b))] = len(self.column_names)
                    columns.append(len(self.column_names))
                    self.column_names.append(f"{first}={level_a}:{second}={level_b}")
        self.main_columns: dict[str, list[int]] = {
            factor: [self._main[(factor, level)] for level in space.factors[factor][1:]] for factor in space.names
        }
        self.p = len(self.column_names)

    def row(self, key: Sequence[str]) -> NDArray[np.float64]:
        """One cell's row of the design matrix.

        Args:
            key: The cell's level of each factor, in factor order (see `FactorSpace.key`).

        Raises:
            ValueError: A key of the wrong length, or with an unknown level.
        """
        if len(key) != len(self.space.names):
            raise ValueError(f"Encoder: a key has {len(self.space.names)} levels, got {len(key)}: {tuple(key)}")
        level_of = dict(zip(self.space.names, key, strict=True))
        for factor, level in level_of.items():
            if level not in self.space.factors[factor]:
                raise ValueError(f"Encoder: unknown level {level!r} of factor {factor!r}")
        x = np.zeros(self.p)
        x[0] = 1.0
        for factor, level in level_of.items():
            column = self._main.get((factor, level))
            if column is not None:
                x[column] = 1.0
        for first, second in self._pairs:
            column = self._pair.get(((first, level_of[first]), (second, level_of[second])))
            if column is not None:
                x[column] = 1.0
        return x

    def matrix(self, keys: Sequence[Sequence[str]]) -> NDArray[np.float64]:
        """The rows of `keys`, stacked: `len(keys) x p` (`0 x p` for no keys).

        Args:
            keys: Cells' keys, as `row` takes them.
        """
        return np.vstack([self.row(key) for key in keys]) if len(keys) else np.zeros((0, self.p))
