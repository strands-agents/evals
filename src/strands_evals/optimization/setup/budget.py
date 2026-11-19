"""The rollout budget: one rollout = one case executed once; the end phase's cost is reserved up front."""

from __future__ import annotations

from dataclasses import dataclass

from ...types.optimization import SplitName

RERUN_PERCENT = 5  # an end-phase arm's re-run headroom per re-run pass, in percent of its first pass (at least 1)


@dataclass(frozen=True, kw_only=True)
class Budget:
    """`max_rollouts` always applies; `max_iterations=None` means no iteration cap.

    `max_iterations=n` is n steps' worth of candidates: `n x proposals_per_iteration` candidate slots, charged as
    a `Nested` round charges its own. A step that saved a candidate uses the `n_candidates` it was given, a round's
    opener 1 (none when it repeats a measured candidate), and a step that saved nothing (every proposal a
    duplicate or invalid) none, so a run can have more than n steps. The last step is cut to the slots left.
    """

    max_rollouts: int = 500
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        """`max_rollouts` >= 1 and `max_iterations` None or >= 1, else `ValueError`."""
        if self.max_rollouts < 1:
            raise ValueError(f"Budget: max_rollouts must be at least 1, got {self.max_rollouts}")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(f"Budget: max_iterations must be None or at least 1, got {self.max_iterations}")


def end_phase_reserve(arms: int, trials: int, rerun_passes: int) -> int:
    """The rollouts an end-phase step holds back: every arm's first pass plus room for its infrastructure re-runs.

    Each arm gets `trials` plus `rerun_passes x ceil(RERUN_PERCENT% x trials)` (at least one re-run per pass
    when it has trials), so a transient failure in validation or test is re-run even when the search spent all
    it could. The step's arms share what is left after their first passes evenly, so each gets its own
    headroom; the re-runs of one arm can use it in any pass.

    Args:
        arms: Candidates the step scores at once (the finalists and the baseline on validation; the champion and
            the baseline, or every front member, on test).
        trials: One arm's first pass: the split's cases x runs per candidate.
        rerun_passes: The optimizer's `rerun_passes`; 0 reserves no headroom.

    Raises:
        ValueError: A negative argument.
    """
    if min(arms, trials, rerun_passes) < 0:
        raise ValueError(
            f"end_phase_reserve: arguments must be >= 0, got arms={arms}, trials={trials}, rerun_passes={rerun_passes}"
        )
    headroom = rerun_passes * ((trials * RERUN_PERCENT + 99) // 100)  # the ceiling, in integers
    return arms * (trials + headroom)


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`.

    Two reserves are held back for the end phase, each `end_phase_reserve(arms, |split| x runs, rerun_passes)`:
    `validation_reserve` for n_finalists + 1 arms and `test_reserve` for 2. The search can spend neither, and
    validation can't spend the test share. Under a `Pareto` objective `max(n_finalists, PARETO_MIN_FINALISTS)`
    (`end_phase.py`) replaces `n_finalists`, and `test_reserve` covers that + 1 arms (every front member is
    tested); a test report that can't afford every member tests the champion and the baseline first, then each
    other member in ranking order that still fits (one that doesn't is skipped). The optimizer computes both
    reserves and passes them in, so this module imports no constant.
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
