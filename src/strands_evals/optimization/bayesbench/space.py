"""`FactorSpace`: named categorical factors, each with its levels in order, the reference first."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence


class FactorSpace:
    """A finite grid of categorical factors, e.g. `FactorSpace({"model": ["small", "large"], "tools": ["off", "on"]})`.

    A cell is one level of every factor; its key is the tuple of those levels in factor order. The first level of each
    factor is its reference: `Encoder` gives it no column, so every effect is measured against it. Factor and level
    order are kept as given.

    Attributes:
        factors: Factor name -> its levels, the reference first.
        names: The factor names, in order.
    """

    def __init__(self, factors: Mapping[str, Sequence[str]]) -> None:
        """Check and keep the factors.

        Args:
            factors: Factor name -> its levels, the reference first.

        Raises:
            ValueError: No factors, a factor with fewer than 2 levels, or a repeated level.
        """
        if not factors:
            raise ValueError("FactorSpace: needs at least one factor")
        for name, levels in factors.items():
            if isinstance(levels, str):
                raise ValueError(f"FactorSpace: factor {name!r} needs a sequence of levels, got the string {levels!r}")
            if len(levels) < 2:
                raise ValueError(f"FactorSpace: factor {name!r} needs at least 2 levels, got {list(levels)}")
            if len(set(levels)) != len(levels):
                raise ValueError(f"FactorSpace: factor {name!r} repeats a level: {list(levels)}")
        self.factors: dict[str, tuple[str, ...]] = {name: tuple(levels) for name, levels in factors.items()}
        self.names: tuple[str, ...] = tuple(self.factors)

    def key(self, levels: Mapping[str, str]) -> tuple[str, ...]:
        """A cell's key: its level of each factor, in factor order.

        Args:
            levels: Factor name -> level, for every factor.

        Raises:
            KeyError: A factor is missing.
            ValueError: An unknown factor or level.
        """
        missing = [name for name in self.names if name not in levels]
        if missing:
            raise KeyError(f"FactorSpace: the levels are missing factors {missing}")
        unknown = [name for name in levels if name not in self.factors]
        if unknown:
            raise ValueError(f"FactorSpace: unknown factors {unknown}")
        for name in self.names:
            if levels[name] not in self.factors[name]:
                raise ValueError(f"FactorSpace: unknown level {levels[name]!r} of factor {name!r}")
        return tuple(levels[name] for name in self.names)

    def cells(self) -> list[tuple[str, ...]]:
        """Every cell's key, in `itertools.product` order over the factors (the last factor varies fastest)."""
        return list(itertools.product(*self.factors.values()))
