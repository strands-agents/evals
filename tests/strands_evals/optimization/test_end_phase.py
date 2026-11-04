import math
import statistics

import pytest
from scipy.stats import t

from strands_evals.optimization.end_phase import TEST_LEVEL, paired_interval, paired_ratio

CANDIDATE = {"a": 0.9, "b": 0.7, "c": 0.8, "d": 0.6, "only-candidate": 0.0}
REFERENCE = {"a": 0.5, "b": 0.6, "c": 0.4, "d": 0.6, "only-reference": 1.0}


def test_the_interval_is_a_t_interval_over_the_shared_cases():
    result = paired_interval(CANDIDATE, REFERENCE, level=TEST_LEVEL)
    differences = [0.4, 0.1, 0.4, 0.0]
    half = t.ppf(0.95, 3) * statistics.stdev(differences) / math.sqrt(4)
    assert result.n_cases == 4
    assert result.mean_diff == pytest.approx(0.225)
    assert result.ci_low == pytest.approx(0.225 - half)
    assert result.ci_high == pytest.approx(0.225 + half)


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
    a = [CANDIDATE[c] for c in "abcd"]
    b = [REFERENCE[c] for c in "abcd"]
    ratio = statistics.fmean(a) / statistics.fmean(b)
    variance = statistics.variance(a) + ratio**2 * statistics.variance(b) - 2 * ratio * statistics.covariance(a, b)
    half = t.ppf(0.95, 3) * math.sqrt(variance / (statistics.fmean(b) ** 2 * 4))
    result = paired_ratio(CANDIDATE, REFERENCE, level=TEST_LEVEL)
    assert result.n_cases == 4
    assert result.mean_diff == pytest.approx(ratio)
    assert result.ci_low == pytest.approx(ratio - half)
    assert result.ci_high == pytest.approx(ratio + half)


def test_a_reference_mean_of_0_gives_no_ratio():
    assert paired_ratio({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0}, level=TEST_LEVEL) is None


def test_one_shared_case_gives_a_ratio_without_an_interval():
    result = paired_ratio({"a": 3.0}, {"a": 2.0}, level=TEST_LEVEL)
    assert (result.mean_diff, result.ci_low, result.ci_high, result.n_cases) == (1.5, None, None, 1)


def test_the_comparison_is_deterministic():
    assert paired_interval(CANDIDATE, REFERENCE, level=TEST_LEVEL) == paired_interval(
        dict(reversed(CANDIDATE.items())), REFERENCE, level=TEST_LEVEL
    )
