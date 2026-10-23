"""The run's train / selection / holdout split."""

from __future__ import annotations

from collections.abc import Sequence

from ...types.optimization import Splits


def split_cases(names: Sequence[str], *, holdout_fraction: float, selection_fraction: float, seed: int) -> Splits:
    """Shuffle with `random.Random(seed)` and carve the holdout first.

    Fewer than one case in any split raises `ValueError` naming the shortfall. Train never doubles
    as selection.
    """
    raise NotImplementedError
