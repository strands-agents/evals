"""The rollout budget: one rollout = one case executed once; the end phase's cost is reserved up front."""

from __future__ import annotations

from dataclasses import dataclass

from ...types.optimization import SplitName

FEW_ITERATIONS = 3  # the optimizer warns when the budget leaves the search fewer iterations than this


@dataclass(frozen=True, kw_only=True)
class Budget:
    """`max_rollouts` always applies; `max_iterations=None` means no iteration cap.

    `max_iterations=n` is n steps' worth of candidates: `n x proposals_per_iteration` candidate slots, charged as
    a `Nested` round charges its own. A step that saved a candidate uses the `n_candidates` it was given, a round's
    opener 1 (none when it repeats a measured candidate), and a step that saved nothing (every proposal a
    duplicate or invalid) none, so a run can have more than n steps. The last step is cut to the slots left.

    The end phase's reserve comes out of `max_rollouts` first. It has room for `n_finalists` finalists (at least
    `PARETO_MIN_FINALISTS` under `Pareto`), or for fewer when the search couldn't then score that many candidates
    (`affordable_finalists`). The optimizer warns when that leaves the search fewer than `FEW_ITERATIONS`
    iterations or the end phase fewer finalists, each lowered to what `max_iterations` allows.
    """

    max_rollouts: int = 500
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        """`max_rollouts` >= 1 and `max_iterations` None or >= 1, else `ValueError`."""
        if self.max_rollouts < 1:
            raise ValueError(f"Budget: max_rollouts must be at least 1, got {self.max_rollouts}")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(f"Budget: max_iterations must be None or at least 1, got {self.max_iterations}")


def end_phase_reserves(finalists: int, *, validation: int, test: int, test_finalists: bool) -> tuple[int, int]:
    """The end phase's `(validation_reserve, test_reserve)` for `finalists` finalists.

    Validation scores the finalists and the baseline. Test scores the champion and the baseline, or with
    `test_finalists` (as under `Pareto`) every finalist and the baseline.

    Args:
        finalists: The finalists validation scores.
        validation: One candidate's validation rollouts (|validation| x runs).
        test: One candidate's test rollouts (|test| x runs).
        test_finalists: Whether the test report scores every finalist, not only the champion.
    """
    return (finalists + 1) * validation, ((finalists + 1) if test_finalists else 2) * test


def affordable_finalists(
    max_rollouts: int, *, train: int, validation: int, test: int, most: int, test_finalists: bool, spent: int = 0
) -> int:
    """The most finalists, up to `most`, whose reserve still leaves the search room to score as many candidates.

    Every finalist is a search candidate, so each one costs a candidate's train rollouts on top of its share of
    the reserve: the result is the largest `f <= most` with `max_rollouts` >= the baseline's train rollouts +
    `end_phase_reserves(f)` + `f` candidates' train rollouts, and with the reserve fitting in what `spent` leaves.
    Reserving for more would hold back room for finalists the search can't produce. At least 1, even when the
    search can't afford a single candidate.

    Args:
        max_rollouts: The budget's `max_rollouts`.
        train: One candidate's train rollouts (|train| x runs); the baseline costs the same.
        validation: One candidate's validation rollouts (|validation| x runs), at least 1.
        test: One candidate's test rollouts (|test| x runs).
        most: The finalists the end phase takes when the budget allows it.
        test_finalists: As in `end_phase_reserves`.
        spent: Rollouts the search has spent or must still spend (on resume); a search never spends into its own
            reserve, so a run resumed with the budget it started with gets the same result.
    """
    fixed = sum(end_phase_reserves(0, validation=validation, test=test, test_finalists=test_finalists))
    per_finalist = sum(end_phase_reserves(1, validation=validation, test=test, test_finalists=test_finalists)) - fixed
    searchable = (max_rollouts - train - fixed) // (per_finalist + train)
    return max(1, min(most, searchable, (max_rollouts - spent - fixed) // per_finalist))


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`.

    Two reserves are held back for the end phase (`end_phase_reserves`): `validation_reserve` = (finalists + 1)
    x |validation| x runs and `test_reserve` = 2 x |test| x runs. The search can spend neither, and validation
    can't spend the test share. Under a `Pareto` objective `test_reserve` = (finalists + 1) x |test| x runs
    (every front member is tested); a test report that can't afford every member tests the champion and the
    baseline first, then each other member in ranking order that still fits (one that doesn't is skipped). The
    optimizer picks the number of finalists (`affordable_finalists`, up to `max(n_finalists,
    PARETO_MIN_FINALISTS)` under `Pareto`) and passes both reserves in, so this module imports no constant.
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
