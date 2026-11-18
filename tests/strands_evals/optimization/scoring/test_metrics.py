import math

import pytest

from strands_evals.optimization import Derived, MetricRegistry
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


# --- derived metrics ---


def price(m):
    return 3e-6 * m["input_tokens"] + 15e-6 * m["output_tokens"]


COST = Derived("cost", price, range=(0, None))


def test_a_derived_metric_resolves_to_itself_with_its_range():
    registry = MetricRegistry({"Stars": (1, 5)}, [COST])
    assert registry.resolve("cost") == "cost"
    assert registry.range("cost") == (0.0, None)
    assert registry.is_derived("cost") and not registry.is_usage("cost") and not registry.is_derived("Stars")
    assert registry.derived == (COST,) and registry.evaluators == ("Stars",)


def test_an_unknown_metric_lists_the_derived_ones_too():
    with pytest.raises(ValueError, match="unknown metric 'costs'.*'cost'"):
        MetricRegistry({}, [COST]).resolve("costs")


@pytest.mark.parametrize("name", ["Stars", "tokens"])
def test_a_derived_metric_named_like_a_measured_one_fails(name):
    with pytest.raises(ValueError, match="named like an evaluator or usage metric"):
        MetricRegistry({"Stars": (1, 5)}, [Derived(name, price)])


def test_the_same_derived_metric_twice_registers_once_and_two_with_one_name_fail():
    assert MetricRegistry({}, [COST, Derived("cost", price, range=(0, None))]).derived == (COST,)
    with pytest.raises(ValueError, match="two different Derived metrics are named 'cost'"):
        MetricRegistry({}, [COST, Derived("cost", lambda m: m["tokens"], range=(0, None))])
    with pytest.raises(ValueError, match="two different Derived metrics"):
        MetricRegistry({}, [COST, Derived("cost", price)])  # another range


@pytest.mark.parametrize(
    ("range_", "stored"), [((0, None), (0.0, None)), ((-math.inf, 1), (None, 1.0)), ((None, None), (None, None))]
)
def test_a_derived_range_is_stored_as_floats_or_none(range_, stored):
    assert Derived("cost", price, range=range_).range == stored


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"name": "", "fn": price}, ValueError, "must not be empty"),
        ({"name": 3, "fn": price}, TypeError, "must be a string"),
        ({"name": "cost", "fn": "price"}, TypeError, "must be callable"),
        ({"name": "cost", "fn": price, "range": (0, 1, 2)}, TypeError, r"\(low, high\)"),
        ({"name": "cost", "fn": price, "range": (0, "1")}, TypeError, r"\(low, high\)"),
        ({"name": "cost", "fn": price, "range": (True, None)}, TypeError, r"\(low, high\)"),
        ({"name": "cost", "fn": price, "range": (1, 1)}, ValueError, "is empty"),
        ({"name": "cost", "fn": price, "range": (math.nan, None)}, ValueError, "NaN"),
    ],
)
def test_a_bad_derived_metric_fails_at_construction(kwargs, error, match):
    with pytest.raises(error, match=match):
        Derived(**kwargs)
