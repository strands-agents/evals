"""The run's train / validation / test split; validation is required."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...case import Case
from ...types.optimization import CaseSplit


@dataclass(frozen=True, kw_only=True)
class Split:
    """Shares or explicit case lists, not a mix; train searches, validation picks, test reports.

    Shares must sum to 1 (`test=None` counts as 0), else `ValueError` naming the sum. Case lists must be
    disjoint and give every case a role. `seed=None` uses the run seed (`Optimizer(seed=)`), as `TPE` and
    `Thompson` do, so two runs with different seeds get different splits; `run.json` stores the seed used.
    """

    train: float | Sequence[Case] = 0.4
    validation: float | Sequence[Case] = 0.3
    test: float | Sequence[Case] | None = 0.3
    seed: int | None = None

    def __post_init__(self) -> None:
        """Check the shares sum to 1 and lists aren't mixed with shares."""
        raise NotImplementedError

    def assign(self, cases: Sequence[Case], *, run_seed: int) -> CaseSplit:
        """Assign every case a role.

        With shares: sorted by name, shuffled with `random.Random(seed)` (`run_seed` when `seed` is None),
        counts by largest remainder.
        0 test cases is allowed (no test report); 1 train case warns (the model's
        case factor drops out).

        Args:
            cases: Every case; names are unique.
            run_seed: The run's seed, used when `seed` is None.

        Returns:
            The three case-name lists; every case is in exactly one.

        Raises:
            ValueError: Fewer than 1 train or 1 validation case, naming the shortfall.
        """
        raise NotImplementedError
