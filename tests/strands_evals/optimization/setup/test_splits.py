import warnings

import pytest

from strands_evals import Case
from strands_evals.optimization import Split

# small fixtures: the 1-4 test cases warning is tested on its own below
pytestmark = pytest.mark.filterwarnings(r"ignore:Split. \d test case:UserWarning")


def make_cases(n):
    return [Case(name=f"case-{i:02d}", input=str(i)) for i in range(n)]


def test_the_default_split_deals_0_4_0_3_0_3():
    split = Split().assign(make_cases(10), run_seed=0)
    assert (len(split.train), len(split.validation), len(split.test)) == (4, 3, 3)


def test_every_case_gets_exactly_one_role():
    cases = make_cases(17)
    split = Split().assign(cases, run_seed=3)
    assigned = [*split.train, *split.validation, *split.test]
    assert sorted(assigned) == sorted(c.name for c in cases)


def test_counts_follow_the_largest_remainder():
    # 7 x (0.4, 0.3, 0.3) = (2.8, 2.1, 2.1): floors (2, 2, 2), the one left goes to train
    split = Split().assign(make_cases(7), run_seed=0)
    assert (len(split.train), len(split.validation), len(split.test)) == (3, 2, 2)


def test_the_same_seed_gives_the_same_split_whatever_the_case_order():
    cases = make_cases(12)
    first = Split(seed=5).assign(cases, run_seed=0)
    second = Split(seed=5).assign(list(reversed(cases)), run_seed=99)
    assert first == second


def test_seed_none_uses_the_run_seed():
    cases = make_cases(20)
    assert Split().assign(cases, run_seed=1) == Split(seed=1).assign(cases, run_seed=2)
    assert Split().assign(cases, run_seed=1) != Split().assign(cases, run_seed=2)


def test_shares_must_sum_to_1_naming_the_sum():
    with pytest.raises(ValueError, match="sums to 1.3"):
        Split(train=0.5, validation=0.5)


def test_test_none_counts_as_0_and_gives_no_test_cases():
    split = Split(train=0.5, validation=0.5, test=None).assign(make_cases(6), run_seed=0)
    assert split.test == ()
    assert len(split.train) + len(split.validation) == 6


def test_a_float_rounding_sum_passes():
    Split(train=0.1 + 0.2, validation=0.3, test=0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train": 0.0, "validation": 0.7, "test": 0.3},
        {"train": 0.7, "validation": 0.0, "test": 0.3},
        {"train": 1.2, "validation": 0.1, "test": -0.3},
    ],
)
def test_bad_shares_raise(kwargs):
    with pytest.raises(ValueError):
        Split(**kwargs)


def test_shares_and_lists_cant_mix():
    with pytest.raises(ValueError, match="not a mix"):
        Split(train=make_cases(2), validation=0.5, test=None)


def test_too_few_cases_for_validation_names_the_shortfall():
    # 2 x (0.8, 0.1, 0.1) = (1.6, 0.2, 0.2): train 2, validation 0
    with pytest.raises(ValueError, match="0 validation cases of 2"):
        Split(train=0.8, validation=0.1, test=0.1).assign(make_cases(2), run_seed=0)


def test_too_few_cases_for_train_names_the_shortfall():
    with pytest.raises(ValueError, match="0 train cases of 1"):
        Split(train=0.1, validation=0.9, test=None).assign(make_cases(1), run_seed=0)


def test_one_train_case_warns():
    with pytest.warns(UserWarning, match="1 train case"):
        Split(train=0.2, validation=0.8, test=None).assign(make_cases(5), run_seed=0)


@pytest.mark.parametrize("n_test", [1, 4])
def test_one_to_four_test_cases_warn(n_test):
    cases = make_cases(10)
    split = Split(train=cases[:5], validation=cases[5 : 10 - n_test], test=cases[10 - n_test :])
    with pytest.warns(UserWarning, match=f"{n_test} test case"):
        split.assign(cases, run_seed=0)


def test_zero_or_five_test_cases_dont_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Split(train=0.5, validation=0.5, test=None).assign(make_cases(10), run_seed=0)
        Split(train=0.4, validation=0.3, test=0.3).assign(make_cases(17), run_seed=0)  # 7 / 5 / 5


def test_lists_keep_their_cases_and_order():
    cases = make_cases(5)
    split = Split(train=[cases[3], cases[0]], validation=[cases[1]], test=[cases[4], cases[2]])
    assert split.assign(cases, run_seed=0).model_dump() == {
        "train": ("case-03", "case-00"),
        "validation": ("case-01",),
        "test": ("case-04", "case-02"),
    }


def test_lists_with_test_none_have_no_test_cases():
    cases = make_cases(3)
    split = Split(train=cases[:2], validation=cases[2:], test=None).assign(cases, run_seed=0)
    assert split.test == ()


def test_a_case_in_two_lists_raises():
    cases = make_cases(3)
    with pytest.raises(ValueError, match="more than one role"):
        Split(train=cases[:2], validation=cases[1:], test=None)


def test_a_case_with_no_role_raises():
    cases = make_cases(4)
    with pytest.raises(ValueError, match=r"\['case-03'\] have no role"):
        Split(train=cases[:2], validation=cases[2:3], test=None).assign(cases, run_seed=0)


def test_a_listed_case_outside_the_experiment_raises():
    cases = make_cases(3)
    extra = Case(name="other", input="x")
    with pytest.raises(ValueError, match="not in the experiment"):
        Split(train=[*cases[:2], extra], validation=cases[2:], test=None).assign(cases, run_seed=0)


def test_an_empty_validation_list_raises():
    cases = make_cases(3)
    with pytest.raises(ValueError, match="0 validation cases"):
        Split(train=cases, validation=[], test=None).assign(cases, run_seed=0)


def test_duplicate_or_missing_case_names_raise():
    with pytest.raises(ValueError, match="unique, non-None names"):
        Split().assign([*make_cases(5), Case(name="case-00", input="again")], run_seed=0)
    with pytest.raises(ValueError, match="unique, non-None names"):
        Split().assign([*make_cases(5), Case(input="unnamed")], run_seed=0)
