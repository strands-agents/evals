"""The run's limits: rollouts (one rollout = one case executed once; the end phase's cost is reserved up front),
candidate slots, and proposer tokens."""

from __future__ import annotations

from dataclasses import dataclass

from ...types.optimization import SplitName


@dataclass(frozen=True, kw_only=True)
class Budget:
    """The run's limits. The search stops at the first limit it reaches, and the end phase runs after any stop.

    `max_rollouts` always applies. It counts agent rollouts, re-runs included, in the search and the end phase.
    The end phase's share is reserved up front (`BudgetTracker`), so the search stops once what's left after
    that reserve can't score another candidate. Judges run inside each trial and add no rollouts of their own;
    proposer calls aren't rollouts.

    `max_iterations=None` means no iteration cap. `max_iterations=n` is n steps' worth of candidates:
    `n x proposals_per_iteration` candidate slots, charged as a `Nested` round charges its own. A step that saved
    a candidate uses the `n_candidates` it was given, a round's opener 1 (none when it repeats a measured
    candidate), and a step that saved nothing (every proposal a duplicate or invalid) none, so a run can have
    more than n steps. The last step is cut to the slots left.

    `max_proposer_tokens=None` means no cap on proposer spend. `max_proposer_tokens=t` stops the search once
    the proposers' reported usage, summed over every block, reaches t tokens (`Usage.tokens`, the total in
    `search_cost`: raw tokens with cache reads included, so it overstates what cached input costs). Usage counts
    when a proposer reports it, on a returned `Patch.usage` or in a `proposer_usage` event; a proposer that
    reports none (TPE, Thompson) never counts. The cap is checked before each step and before each redraw, so
    a step that reaches it stops redrawing and is still scored. The total can pass t by one round of proposer
    calls (up to `proposals_per_iteration` calls per block of the step). That bound counts calls, not tokens;
    one call's spend has no limit here, and a call can retry its agent.

    Nothing caps the agent's or the judges' tokens directly, or wall-clock time.
    """

    max_rollouts: int = 500
    max_iterations: int | None = None
    max_proposer_tokens: int | None = None

    def __post_init__(self) -> None:
        """`max_rollouts` >= 1, and `max_iterations` and `max_proposer_tokens` None or >= 1, else `ValueError`."""
        if self.max_rollouts < 1:
            raise ValueError(f"Budget: max_rollouts must be at least 1, got {self.max_rollouts}")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(f"Budget: max_iterations must be None or at least 1, got {self.max_iterations}")
        if self.max_proposer_tokens is not None and self.max_proposer_tokens < 1:
            raise ValueError(f"Budget: max_proposer_tokens must be None or at least 1, got {self.max_proposer_tokens}")


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`.

    Two reserves are held back for the end phase: `validation_reserve` = (n_finalists + 1) x |validation| x
    runs and `test_reserve` = 2 x |test| x runs. The search can spend neither, and validation can't spend
    the test share. Under a `Pareto` objective `max(n_finalists, PARETO_MIN_FINALISTS)` (`end_phase.py`)
    replaces `n_finalists`, and `test_reserve` = (that + 1) x |test| x runs (every front member is tested); a
    test report that can't afford every member tests the champion and the baseline first, then each other
    member in ranking order that still fits (one that doesn't is skipped). The optimizer computes both reserves
    and passes them in, so this module imports no constant.
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
