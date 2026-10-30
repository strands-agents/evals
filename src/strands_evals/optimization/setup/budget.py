"""The rollout budget: one rollout = one case executed once; the end phase's cost is reserved up front."""

from __future__ import annotations

from dataclasses import dataclass

from ...types.optimization import SplitName


@dataclass(frozen=True, kw_only=True)
class Budget:
    """`max_rollouts` always applies; `max_iterations=None` means no iteration cap."""

    max_rollouts: int = 500
    max_iterations: int | None = None


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`.

    Two reserves are held back for the end phase: `validation_reserve` = (finalists + 1) x |validation| x
    runs and `test_reserve` = 2 x |test| x runs. The search can spend neither, and validation can't spend
    the test share. Under a `Pareto` objective `max(finalists, 5)` replaces `finalists`, and
    `test_reserve` = (max(finalists, 5) + 1) x |test| x runs (every front member is tested); a test report
    that can't afford every member tests the champion and the baseline first, then the others in ranking
    order while affordable.
    """

    def __init__(self, budget: Budget, spent: int = 0, *, validation_reserve: int = 0, test_reserve: int = 0):
        """Start counting from `spent`.

        Args:
            budget: The run's limits.
            spent: Rollouts already spent; on resume, rebuilt from the saved trial records.
            validation_reserve: Held back for scoring the finalists and the baseline on validation.
            test_reserve: Held back for the test report.
        """
        raise NotImplementedError

    def can_afford(self, rollouts: int, *, phase: SplitName = "train") -> bool:
        """Whether `rollouts` fit, keeping the later phases' reserves.

        `train` (the search): spent + rollouts + both reserves <= max; `validation`: + the test reserve;
        `test`: spent + rollouts <= max. Callers pass `TrialRunner.pending(...)`, never `cases x runs`.
        """
        raise NotImplementedError

    def allowance(self, *, phase: SplitName = "train") -> int:
        """What `TrialRunner.run` may spend on re-run passes: remaining minus the later phases' reserves, >= 0."""
        raise NotImplementedError

    def spend(self, rollouts: int) -> None:
        """Count `rollouts` as spent (what `TrialRunner.run` returned)."""
        raise NotImplementedError

    @property
    def remaining(self) -> int:
        """`max_rollouts - spent`, reserves not subtracted."""
        raise NotImplementedError
