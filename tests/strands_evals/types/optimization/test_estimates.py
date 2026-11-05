import math

import pytest

from strands_evals.types.optimization import Estimate


def test_ci90_is_mean_plus_minus_1645_sd():
    low, high = Estimate(mean=0.5, sd=0.1).ci90
    assert low == pytest.approx(0.5 - 0.16448536)
    assert high == pytest.approx(0.5 + 0.16448536)


def test_quantile_is_the_normal_quantile():
    estimate = Estimate(mean=2.0, sd=0.5)
    assert estimate.quantile(0.5) == pytest.approx(2.0)
    assert estimate.quantile(0.95) == pytest.approx(2.0 + 0.5 * 1.6448536)
    assert estimate.quantile(0.1) == pytest.approx(2.0 - 0.5 * 1.2815516)


def test_quantile_with_no_spread_is_the_mean():
    assert Estimate(mean=3.25, sd=0.0).quantile(0.1) == 3.25


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_quantile_outside_0_1_raises(q):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        Estimate(mean=0.0, sd=1.0).quantile(q)


def test_round_trips_through_json():
    estimate = Estimate(mean=0.25, sd=0.01)
    assert Estimate.model_validate_json(estimate.model_dump_json()) == estimate
