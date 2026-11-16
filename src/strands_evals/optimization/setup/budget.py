"""The rollout budget: one rollout = one case executed once; the end phase's cost is reserved up front."""

from __future__ import annotations

from dataclasses import dataclass

from ...types.optimization import SplitName


@dataclass(frozen=True, kw_only=True)
class Budget:
    """`max_rollouts` always applies; `max_iterations=None` means no iteration cap."""

    max_rollouts: int = 500
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        """`max_rollouts` >= 1 and `max_iterations` None or >= 1, else `ValueError`."""
        if self.max_rollouts < 1:
            raise ValueError(f"Budget: max_rollouts must be at least 1, got {self.max_rollouts}")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(f"Budget: max_iterations must be None or at least 1, got {self.max_iterations}")


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`.

    Two reserves are held back for the end phase: `validation_reserve` = (n_finalists + 1) x |validation| x
    runs and `test_reserve` = 2 x |test| x runs. The search can spend neither, and validation can't spend
    the test share. Under a `Pareto` objective `max(n_finalists, PARETO_MIN_FINALISTS)` (`end_phase.py`)
    replaces `n_finalists`, and `test_reserve` = (that + 1) x |test| x runs (every front member is tested); a
    test report that can't afford every member tests the champion and the baseline first, then the others in
    ranking order while affordable. The optimizer computes both reserves and passes them in, so this module
    imports no constant.
    """

    def __init__(self, budget: Budget, spent: int = 0, *, validation_reserve: int = 0, test_reserve: int = 0):
        """Start counting from `spent`.

        Args:
            budget: The run's limits.
            spent: Rollouts already spent; on resume, rebuilt from the saved trial records.
            validation_reserve: Held back for scoring the finalists and the baseline on validation.
            test_reserve: Held back for the test report.

        Raises:
            ValueError: A negative `spent` or reserve.
        """
        if min(spent, validation_reserve, test_reserve) < 0:
            raise ValueError(
                f"BudgetTracker: spent and reserves must be >= 0, got spent={spent}, "
                f"validation_reserve={validation_reserve}, test_reserve={test_reserve}"
            )
        self.budget = budget
        self.spent = spent
        self.validation_reserve = validation_reserve
        self.test_reserve = test_reserve

    def can_afford(self, rollouts: int, *, phase: SplitName = "train") -> bool:
        """Whether `rollouts <= allowance(phase)`: remaining minus the later phases' reserves.

        `train` (the search) holds back both reserves, `validation` the test reserve, and `test` nothing (so
        there it is `rollouts <= remaining`). Callers pass `TrialRunner.pending(...)`, never `cases x runs`.
        """
        return rollouts <= self.allowance(phase=phase)

    def allowance(self, *, phase: SplitName = "train") -> int:
        """What one `TrialRunner.run` call may spend, first pass and re-runs included.

        Remaining minus the later phases' reserves, >= 0.
        """
        held = {"train": self.validation_reserve + self.test_reserve, "validation": self.test_reserve, "test": 0}
        return max(0, self.remaining - held[phase])

    def spend(self, rollouts: int) -> None:
        """Count `rollouts` as spent (what `TrialRunner.run` returned).

        Raises:
            ValueError: A negative `rollouts`.
        """
        if rollouts < 0:
            raise ValueError(f"BudgetTracker: can't spend a negative number of rollouts, got {rollouts}")
        self.spent += rollouts

    @property
    def remaining(self) -> int:
        """`max_rollouts - spent`, reserves not subtracted."""
        return self.budget.max_rollouts - self.spent
