import math

import pytest

from strands_evals.optimization.end_phase import TEST_LEVEL, paired_interval, paired_ratio

CANDIDATE = {"a": 0.9, "b": 0.7, "c": 0.8, "d": 0.6, "only-candidate": 0.0}
REFERENCE = {"a": 0.5, "b": 0.6, "c": 0.4, "d": 0.6, "only-reference": 1.0}


def test_the_interval_is_a_t_interval_over_the_shared_cases():
    # expected from scipy.stats.ttest_rel(a, b).confidence_interval(0.9) on cases a-d
    result = paired_interval(CANDIDATE, REFERENCE, level=TEST_LEVEL)
    assert result.n_cases == 4
    assert result.mean_diff == pytest.approx(0.225)
    assert result.ci_low == pytest.approx(-0.017579150428857)
    assert result.ci_high == pytest.approx(0.467579150428857)


def test_a_wider_level_gives_a_wider_interval():
    narrow = paired_interval(CANDIDATE, REFERENCE, level=0.5)
    wide = paired_interval(CANDIDATE, REFERENCE, level=0.99)
    assert wide.ci_low < narrow.ci_low < narrow.ci_high < wide.ci_high


def test_one_shared_case_gives_no_interval():
    result = paired_interval({"a": 1.0}, {"a": 0.25, "b": 0.0}, level=TEST_LEVEL)
    assert (result.mean_diff, result.ci_low, result.ci_high, result.n_cases) == (0.75, None, None, 1)


def test_no_shared_case_gives_none():
    assert paired_interval({"a": 1.0}, {"b": 1.0}, level=TEST_LEVEL) is None
    assert paired_ratio({"a": 1.0}, {"b": 1.0}, level=TEST_LEVEL) is None


def test_the_baseline_against_itself_is_exactly_0_and_1():
    difference = paired_interval(REFERENCE, REFERENCE, level=TEST_LEVEL)
    ratio = paired_ratio(REFERENCE, REFERENCE, level=TEST_LEVEL)
    assert (difference.mean_diff, difference.ci_low, difference.ci_high) == (0.0, 0.0, 0.0)
    assert ratio.mean_diff == pytest.approx(1.0)
    assert ratio.ci_low == pytest.approx(1.0)
    assert ratio.ci_high == pytest.approx(1.0)


def test_the_ratio_uses_the_delta_method():
    # expected from numpy: r = mean(a) / mean(b), var = (var_a + r^2 var_b - 2 r cov_ab) / (mean(b)^2 n),
    # np.cov with ddof=1, half-width t(0.95, 3) * sqrt(var), on cases a-d
    result = paired_ratio(CANDIDATE, REFERENCE, level=TEST_LEVEL)
    assert result.n_cases == 4
    assert result.mean_diff == pytest.approx(1.428571428571429)
    assert result.ci_low == pytest.approx(0.883315167473689)
    assert result.ci_high == pytest.approx(1.973827689669169)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_case_mean_raises_in_both(bad):
    with pytest.raises(ValueError, match=r"finite; not for \['a'\]"):
        paired_interval({**CANDIDATE, "a": bad}, REFERENCE, level=TEST_LEVEL)
    with pytest.raises(ValueError, match=r"finite; not for \['a'\]"):
        paired_ratio(CANDIDATE, {**REFERENCE, "a": bad}, level=TEST_LEVEL)


def test_a_non_finite_unshared_case_is_ignored():
    assert paired_interval({**CANDIDATE, "only-candidate": math.nan}, REFERENCE, level=TEST_LEVEL).n_cases == 4


def test_a_reference_mean_of_0_gives_no_ratio():
    assert paired_ratio({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0}, level=TEST_LEVEL) is None


def test_one_shared_case_gives_a_ratio_without_an_interval():
    result = paired_ratio({"a": 3.0}, {"a": 2.0}, level=TEST_LEVEL)
    assert (result.mean_diff, result.ci_low, result.ci_high, result.n_cases) == (1.5, None, None, 1)


def test_the_comparison_is_deterministic():
    assert paired_interval(CANDIDATE, REFERENCE, level=TEST_LEVEL) == paired_interval(
        dict(reversed(CANDIDATE.items())), REFERENCE, level=TEST_LEVEL
    )
