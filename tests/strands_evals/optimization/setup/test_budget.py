import itertools

import pytest

from strands_evals.optimization import Budget
from strands_evals.optimization.setup.budget import BudgetTracker, affordable_finalists, end_phase_reserves


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


def test_end_phase_reserves():
    # The finalists and the baseline on validation; the champion and the baseline on test, or every finalist.
    assert end_phase_reserves(3, validation=6, test=5, test_finalists=False) == (24, 10)
    assert end_phase_reserves(3, validation=6, test=5, test_finalists=True) == (24, 20)
    assert end_phase_reserves(3, validation=6, test=0, test_finalists=True) == (24, 0)


@pytest.mark.parametrize("test_finalists", [False, True])
def test_affordable_finalists_is_the_most_the_search_can_score(test_finalists):
    for max_rollouts, train, validation, test, most in itertools.product(
        range(1, 160), (1, 3, 8), (1, 2, 6), (0, 2, 6), (1, 3, 5)
    ):
        sizes = {"validation": validation, "test": test, "test_finalists": test_finalists}
        # What the search may spend beside the baseline and the reserve for f finalists.
        room = {f: max_rollouts - train - sum(end_phase_reserves(f, **sizes)) for f in range(1, most + 1)}
        found = affordable_finalists(max_rollouts, train=train, most=most, **sizes)
        case = (max_rollouts, train, validation, test, most)
        # The most finalists whose reserve leaves the search that many candidates, else 1.
        assert found == max((f for f in room if room[f] >= f * train), default=1), case
        # Never fewer finalists or search candidates than reserving for all `most` finalists allows.
        candidates = max(0, room[most]) // train
        assert found >= min(most, candidates), case
        assert max(0, room[found]) // train >= candidates, case


@pytest.mark.parametrize("test_finalists", [False, True])
def test_affordable_finalists_fit_in_what_the_search_left(test_finalists):
    sizes = {"train": 8, "validation": 6, "test": 6, "most": 5, "test_finalists": test_finalists}
    for max_rollouts, spent in itertools.product(range(40, 160), range(0, 120)):
        found = affordable_finalists(max_rollouts, **sizes, spent=spent)
        reserve = sum(end_phase_reserves(found, validation=6, test=6, test_finalists=test_finalists))
        # The reserve fits beside what is spent unless even one finalist's doesn't; never more than from scratch.
        assert found == 1 or spent + reserve <= max_rollouts, (max_rollouts, spent)
        assert found <= affordable_finalists(max_rollouts, **sizes)
        # A search spends at most what its own reserve leaves, so that much spent changes nothing.
        before = affordable_finalists(max_rollouts, **sizes)
        held = sum(end_phase_reserves(before, validation=6, test=6, test_finalists=test_finalists))
        if spent <= max_rollouts - held:
            assert found == before, (max_rollouts, spent)
