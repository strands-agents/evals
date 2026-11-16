import pytest

from strands_evals.optimization import Budget
from strands_evals.optimization.setup.budget import BudgetTracker


def make_tracker(max_rollouts=100, spent=0, validation_reserve=20, test_reserve=10):
    return BudgetTracker(
        Budget(max_rollouts=max_rollouts),
        spent,
        validation_reserve=validation_reserve,
        test_reserve=test_reserve,
    )


def test_the_search_keeps_both_reserves():
    tracker = make_tracker()
    assert tracker.allowance() == 70
    assert tracker.can_afford(70)
    assert not tracker.can_afford(71)


def test_validation_keeps_only_the_test_reserve():
    tracker = make_tracker()
    assert tracker.allowance(phase="validation") == 90
    assert tracker.can_afford(90, phase="validation")
    assert not tracker.can_afford(91, phase="validation")


def test_test_can_spend_everything_left():
    tracker = make_tracker()
    assert tracker.allowance(phase="test") == 100
    assert not tracker.can_afford(101, phase="test")


def test_validation_is_always_affordable_after_the_search_spends_all_it_may():
    tracker = make_tracker()
    tracker.spend(tracker.allowance())
    assert not tracker.can_afford(1)
    assert tracker.can_afford(tracker.validation_reserve, phase="validation")
    tracker.spend(tracker.validation_reserve)
    assert tracker.allowance(phase="validation") == 0  # validation re-runs can't spend the test share
    assert tracker.can_afford(tracker.test_reserve, phase="test")


def test_the_cap_is_never_exceeded_when_spending_what_can_be_afforded():
    tracker = make_tracker(max_rollouts=37, validation_reserve=9, test_reserve=6)
    for phase, ask in [("train", 5), ("train", 13), ("train", 8), ("train", 4), ("validation", 9), ("test", 7)]:
        if tracker.can_afford(ask, phase=phase):
            tracker.spend(ask)
        assert tracker.spent <= 37
    assert tracker.remaining >= 0


def test_allowance_is_never_below_0():
    tracker = make_tracker(spent=95)
    assert tracker.allowance() == 0
    assert tracker.allowance(phase="validation") == 0
    assert tracker.allowance(phase="test") == 5


def test_a_resumed_spend_above_a_smaller_budget_affords_nothing():
    tracker = make_tracker(max_rollouts=50, spent=60, validation_reserve=0, test_reserve=0)
    assert tracker.remaining == -10
    assert tracker.allowance(phase="test") == 0
    assert not tracker.can_afford(1, phase="test")
    assert tracker.can_afford(0, phase="test")  # cache hits cost nothing


def test_spend_counts_from_the_resumed_spend():
    tracker = make_tracker(spent=12)
    tracker.spend(3)
    tracker.spend(0)
    assert tracker.spent == 15
    assert tracker.remaining == 85


def test_negative_spend_or_reserves_raise():
    with pytest.raises(ValueError):
        make_tracker().spend(-1)
    with pytest.raises(ValueError):
        make_tracker(spent=-1)
    with pytest.raises(ValueError):
        make_tracker(test_reserve=-1)


def test_the_default_budget_is_500_rollouts_and_no_iteration_cap():
    assert Budget() == Budget(max_rollouts=500, max_iterations=None)


@pytest.mark.parametrize("kwargs", [{"max_rollouts": 0}, {"max_iterations": 0}])
def test_bad_budgets_raise(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)
