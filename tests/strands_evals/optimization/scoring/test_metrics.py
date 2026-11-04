import math

import pytest

from strands_evals.optimization import MetricRegistry
from strands_evals.optimization.scoring.metrics import USAGE_METRICS


def make_registry():
    return MetricRegistry({"HelpfulnessEvaluator": (0.0, 1.0), "Stars": (1, 5), "Helpfulness": (0.0, 1.0)})


def test_evaluators_keep_registration_order():
    assert make_registry().evaluators == ("HelpfulnessEvaluator", "Stars", "Helpfulness")


def test_an_exact_name_wins_over_the_shorthand():
    registry = make_registry()
    assert registry.resolve("Helpfulness") == "Helpfulness"
    assert registry.resolve("HelpfulnessEvaluator") == "HelpfulnessEvaluator"


def test_the_shorthand_drops_the_evaluator_suffix():
    registry = MetricRegistry({"HelpfulnessEvaluator": (0.0, 1.0)})
    assert registry.resolve("Helpfulness") == "HelpfulnessEvaluator"


@pytest.mark.parametrize("name", USAGE_METRICS)
def test_usage_metrics_resolve_to_themselves_and_are_unbounded_above(name):
    registry = make_registry()
    assert registry.resolve(name) == name
    assert registry.is_usage(name)
    assert registry.range(name) == (0.0, None)


def test_an_unknown_metric_fails_naming_the_known_ones():
    with pytest.raises(ValueError, match="unknown metric 'Helpfulnes'.*HelpfulnessEvaluator"):
        make_registry().resolve("Helpfulnes")


def test_ranges_are_stored_as_floats():
    registry = make_registry()
    assert registry.range("Stars") == (1.0, 5.0)
    assert not registry.is_usage("Stars")


def test_an_evaluator_named_like_a_usage_metric_fails():
    with pytest.raises(ValueError, match="tokens"):
        MetricRegistry({"tokens": (0.0, 1.0)})


def test_range_of_an_unresolved_name_raises_key_error():
    with pytest.raises(KeyError):
        make_registry().range("Unknown")


@pytest.mark.parametrize("range_", [(1.0, 1.0), (5.0, 1.0), (math.inf, None), (None, -math.inf)])
def test_an_empty_range_fails(range_):
    with pytest.raises(ValueError, match="is empty"):
        MetricRegistry({"Stars": range_})


@pytest.mark.parametrize(
    ("range_", "stored"),
    [((0.0, math.inf), (0.0, None)), ((-math.inf, 1.0), (None, 1.0)), ((-math.inf, math.inf), (None, None))],
)
def test_an_infinite_end_is_unbounded(range_, stored):
    assert MetricRegistry({"Score": range_}).range("Score") == stored


@pytest.mark.parametrize("range_", [(math.nan, 1.0), (0.0, math.nan)])
def test_a_nan_end_fails(range_):
    with pytest.raises(ValueError, match="NaN"):
        MetricRegistry({"Score": range_})
