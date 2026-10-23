"""Paired bootstrap comparison, kept separate so it can be swapped later."""

from __future__ import annotations

from collections.abc import Mapping

from ...types.optimization import PairedComparison


def paired_bootstrap(
    challenger: Mapping[str, float],
    champion: Mapping[str, float],
    *,
    level: float,
    seed: int | str,
    resamples: int = 2000,
) -> PairedComparison:
    """Two-sided CI at `level` of the mean per-case difference, over the cases both measured."""
    raise NotImplementedError
