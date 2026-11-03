import math

import pytest

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Criteria,
    Maximize,
    MetricRegistry,
    Minimize,
    Pareto,
    WeightedSum,
)
from strands_evals.optimization.scoring.objectives import measured_metrics, score
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.optimization import CaseSplit, TrialRecord, TrialStatus, Usage


class HelpfulnessEvaluator(Evaluator):
    pass


REGISTRY = MetricRegistry({"HelpfulnessEvaluator": (0.0, 1.0), "Stars": (1.0, 5.0)})
UNBOUNDED = MetricRegistry({"Stars": (1.0, 5.0), "Open": (None, None)})
SPLITS = CaseSplit(train=("a", "b"), validation=("v",), test=())


def usage(tokens=100, output_tokens=40, cache_read_tokens=None, seconds=2.0):
    return Usage(
        tokens=tokens,
        input_tokens=tokens - output_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=None,
        model_calls=2,
        seconds=seconds,
        models=["m"],
    )


def record(case="a", run=0, split="train", status=TrialStatus.COMPLETED, scores=None, details=None, **usage_fields):
    evaluations = {
        name: EvaluationOutput(score=value, test_pass=True) if not isinstance(value, EvaluationOutput) else value
        for name, value in (scores or {}).items()
    }
    return TrialRecord(
        candidate_id="c1",
        case_name=case,
        run_index=run,
        split=split,
        status=status,
        error_type=None,
        error=None,
        attempts=1,
        previous_attempts=[],
        output="out",
        trajectory=None,
        interactions=None,
        environment_state=None,
        evaluations=evaluations,
        details=details or {},
        evaluator_errors={},
        usage=usage(**usage_fields),
        judge_usage=Usage.model_construct(),
        rollouts=1,
    )


NA = EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)


# --- objectives and constraints ---


def test_targets_resolve_from_names_instances_and_classes():
    assert Maximize("Stars").metric == "Stars"
    assert Maximize(HelpfulnessEvaluator).metric == "HelpfulnessEvaluator"
    assert Maximize(HelpfulnessEvaluator(name="helpful")).metric == "helpful"
    assert AtLeast(HelpfulnessEvaluator, 0.5).metric == "HelpfulnessEvaluator"


def test_a_target_that_is_not_a_metric_fails():
    with pytest.raises(TypeError):
        Maximize(3)  # type: ignore[arg-type]


@pytest.mark.parametrize("settings", [{"weight": 0}, {"weight": -1.0}, {"scale": 0}, {"weight": math.nan}])
def test_weight_and_scale_must_be_positive(settings):
    with pytest.raises(ValueError):
        Maximize("Stars", **settings)


@pytest.mark.parametrize("settings", [{"weight": "2"}, {"scale": True}])
def test_weight_and_scale_of_the_wrong_type_fail_with_type_error(settings):
    with pytest.raises(TypeError, match="must be a number"):
        Maximize("Stars", **settings)


@pytest.mark.parametrize(
    ("objective", "text"),
    [
        (Maximize("Helpfulness"), "Maximize(Helpfulness)"),
        (Minimize("tokens", scale=1000), "Minimize(tokens, scale=1000)"),
        (Minimize("seconds", weight=2, scale=0.5), "Minimize(seconds, weight=2, scale=0.5)"),
        (
            WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000)),
            "WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))",
        ),
        (
            Pareto(Maximize("Helpfulness", weight=2), Minimize("tokens", scale=1000)),
            "Pareto(Maximize(Helpfulness, weight=2), Minimize(tokens, scale=1000))",
        ),
        (AtMost("tokens", 4000), "tokens <= 4000"),
        (AtLeast("Correctness", 0.8), "Correctness >= 0.8"),
    ],
)
def test_str_is_what_run_json_stores(objective, text):
    assert str(objective) == text


@pytest.mark.parametrize("kind", [WeightedSum, Pareto])
def test_combinations_reject_bad_terms(kind):
    with pytest.raises(TypeError):
        kind()
    with pytest.raises(TypeError):
        kind(Maximize("Stars"), AtMost("tokens", 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="more than once"):
        kind(Maximize("Stars"), Minimize("Stars"))
    with pytest.raises(ValueError, match="scale="):
        kind(Maximize("Stars"), Minimize("tokens"))


@pytest.mark.parametrize("kind", [WeightedSum, Pareto])
def test_a_combination_of_one_needs_no_scale(kind):
    assert kind(Minimize("tokens")).objectives == (Minimize("tokens"),)


@pytest.mark.parametrize("bound", ["4000", math.inf, True])
def test_a_bound_must_be_a_finite_number(bound):
    with pytest.raises(TypeError):
        AtMost("tokens", bound)


# --- Criteria ---


def test_the_default_objective_maximizes_every_evaluator():
    criteria = Criteria(None, [], REGISTRY)
    assert criteria.objective == WeightedSum(Maximize("HelpfulnessEvaluator"), Maximize("Stars"))


def test_the_default_objective_asks_for_one_when_an_evaluator_is_unbounded():
    with pytest.raises(ValueError, match="Open has no bounded range, so the default objective .* give objective="):
        Criteria(None, [], UNBOUNDED)


def test_names_resolve_through_the_registry():
    criteria = Criteria(Maximize("Helpfulness"), [AtMost("tokens", 4000)], REGISTRY)
    assert criteria.objective == Maximize("HelpfulnessEvaluator")
    assert criteria.constraints == (AtMost("tokens", 4000),)


def test_an_unknown_metric_fails_at_construction():
    with pytest.raises(ValueError, match="unknown metric"):
        Criteria(Maximize("Helpfulnes"), [], REGISTRY)
    with pytest.raises(ValueError, match="unknown metric"):
        Criteria(None, [AtMost("tokenz", 1)], REGISTRY)


def test_a_lone_objective_with_a_scale_or_weight_fails():
    with pytest.raises(ValueError, match="read only when two or more objectives are combined"):
        Criteria(Maximize("tokens", scale=1000), [], REGISTRY)
    with pytest.raises(ValueError, match="read only when"):
        Criteria(Maximize("Stars", weight=2), [], REGISTRY)


@pytest.mark.parametrize("kind", [WeightedSum, Pareto])
def test_a_combination_of_one_with_a_scale_or_weight_fails_like_a_lone_objective(kind):
    with pytest.raises(ValueError, match="read only when"):
        Criteria(kind(Maximize("Stars", weight=2)), [], REGISTRY)
    with pytest.raises(ValueError, match="read only when"):
        Criteria(kind(Minimize("tokens", scale=1000)), [], REGISTRY)


def test_the_same_metric_twice_after_resolution_fails():
    with pytest.raises(ValueError, match="more than once"):
        Criteria(WeightedSum(Maximize("Helpfulness"), Maximize("HelpfulnessEvaluator")), [], REGISTRY)


def test_an_unbounded_evaluator_in_a_combination_needs_a_scale():
    with pytest.raises(ValueError, match="Open has no bounded range"):
        Criteria(WeightedSum(Maximize("Stars"), Maximize("Open")), [], UNBOUNDED)
    Criteria(WeightedSum(Maximize("Stars"), Maximize("Open", scale=10)), [], UNBOUNDED)


@pytest.mark.parametrize("feasibility_threshold", [0, 1, 1.5])
def test_feasibility_must_lie_strictly_between_0_and_1(feasibility_threshold):
    with pytest.raises(ValueError):
        Criteria(None, [], REGISTRY, feasibility_threshold=feasibility_threshold)


@pytest.mark.parametrize("agent_failure_score", ["best", math.nan, True])
def test_crash_score_must_be_worst_a_number_or_none(agent_failure_score):
    with pytest.raises(ValueError):
        Criteria(None, [], REGISTRY, agent_failure_score=agent_failure_score)


def test_no_objective_and_no_evaluators_fails():
    with pytest.raises(ValueError, match="nothing to optimize"):
        Criteria(None, [], MetricRegistry({}))


@pytest.mark.parametrize("kind", [WeightedSum, Pareto])
def test_a_combination_of_one_is_that_objective(kind):
    criteria = Criteria(kind(Minimize("tokens")), [], REGISTRY)
    assert criteria.objective == Minimize("tokens")
    assert not criteria.is_pareto


def test_goal_metrics_are_the_objectives_then_the_constraints_each_once():
    criteria = Criteria(
        Pareto(Maximize("Helpfulness"), Minimize("tokens", scale=1000)),
        [AtMost("tokens", 4000), AtLeast("Stars", 3)],
        REGISTRY,
    )
    assert criteria.is_pareto
    assert criteria.goal_metrics == ("HelpfulnessEvaluator", "tokens", "Stars")


def test_crash_value_is_the_worst_end_for_the_goals_that_use_the_metric():
    criteria = Criteria(WeightedSum(Maximize("Helpfulness"), Minimize("Stars")), [], REGISTRY)
    assert criteria.agent_failure_value("HelpfulnessEvaluator") == 0.0
    assert criteria.agent_failure_value("Stars") == 5.0
    assert Criteria(Maximize("Helpfulness"), [], REGISTRY).agent_failure_value("Stars") is None  # no goal uses it
    assert criteria.agent_failure_value("tokens") is None  # usage: measured, never scored


def test_crash_value_is_none_when_goals_disagree_or_the_end_is_unbounded():
    disagree = Criteria(Maximize("Stars"), [AtMost("Stars", 4)], REGISTRY)
    assert disagree.agent_failure_value("Stars") is None
    unbounded = Criteria(Maximize("Open"), [], UNBOUNDED)
    assert unbounded.agent_failure_value("Open") is None


def test_crash_value_follows_a_number_or_none():
    stars = MetricRegistry({"Stars": (1.0, 5.0)})
    assert Criteria(None, [], stars, agent_failure_score=1.5).agent_failure_value("Stars") == 1.5
    assert Criteria(None, [], REGISTRY, agent_failure_score=None).agent_failure_value("Stars") is None


@pytest.mark.parametrize("agent_failure_score", ["worst", 0.5])
def test_crash_value_of_an_unknown_metric_raises(agent_failure_score):
    with pytest.raises(KeyError, match="not a resolved metric"):
        Criteria(
            None, [], MetricRegistry({"HelpfulnessEvaluator": (0.0, 1.0)}), agent_failure_score=agent_failure_score
        ).agent_failure_value("typo")


def test_a_crash_score_outside_an_evaluators_range_fails():
    with pytest.raises(ValueError, match=r"agent_failure_score=0.25 is outside Stars's range \[1.0, 5.0\]"):
        Criteria(None, [], REGISTRY, agent_failure_score=0.25)


def test_a_mean_equal_to_the_bound_up_to_float_noise_is_met():
    criteria = Criteria(None, [AtMost("tokens", 0.3)], REGISTRY)
    assert criteria.unmet_constraints({"a": {"tokens": 0.1 + 0.2}}) == []
    assert criteria.unmet_constraints({"a": {"tokens": 0.31}}) != []


def test_a_lone_objective_is_the_metric_oriented_higher_is_better():
    assert Criteria(Maximize("Stars"), [], REGISTRY).objective_value({"Stars": 4.0}) == 4.0
    assert Criteria(Minimize("tokens"), [], REGISTRY).objective_value({"tokens": 300.0}) == -300.0
    assert Criteria(Maximize("Stars"), [], REGISTRY).objective_value({}) is None


def test_a_weighted_sum_is_the_weighted_mean_of_oriented_value_over_scale():
    criteria = Criteria(
        WeightedSum(Maximize("Helpfulness", weight=3), Maximize("Stars"), Minimize("tokens", scale=1000)), [], REGISTRY
    )
    value = criteria.objective_value({"HelpfulnessEvaluator": 0.8, "Stars": 3.0, "tokens": 500.0})
    assert value == pytest.approx((3 * 0.8 + 1 * 3.0 / 4.0 - 1 * 500 / 1000) / 5)


def test_a_missing_evaluator_term_drops_and_its_weight_goes_to_the_other_evaluators():
    criteria = Criteria(
        WeightedSum(Maximize("Helpfulness", weight=3), Maximize("Stars"), Minimize("tokens", scale=1000)), [], REGISTRY
    )
    value = criteria.objective_value({"Stars": 3.0, "tokens": 500.0})
    assert value == pytest.approx((4 * 3.0 / 4.0 - 500 / 1000) / 5)


def test_the_default_objective_is_the_mean_of_the_applicable_scores():
    criteria = Criteria(WeightedSum(Maximize("Helpfulness"), Maximize("Stars", scale=1.0)), [], REGISTRY)
    assert criteria.objective_value({"HelpfulnessEvaluator": 0.4}) == pytest.approx(0.4)
    assert criteria.objective_value({"HelpfulnessEvaluator": 0.4, "Stars": 0.8}) == pytest.approx(0.6)


def test_a_missing_usage_term_or_every_evaluator_term_leaves_it_unmeasured():
    criteria = Criteria(WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000)), [], REGISTRY)
    assert criteria.objective_value({"HelpfulnessEvaluator": 0.8}) is None
    assert criteria.objective_value({"tokens": 100.0}) is None


def test_a_pareto_value_is_the_weighted_gain_with_each_objectives_weight():
    criteria = Criteria(Pareto(Maximize("Helpfulness", weight=3), Minimize("tokens", scale=1000)), [], REGISTRY)
    assert criteria.objective_value({"HelpfulnessEvaluator": 0.8, "tokens": 500.0}) == pytest.approx(
        (3 * 0.8 - 0.5) / 4
    )


def test_terms_are_oriented_raw_values_in_objective_order():
    criteria = Criteria(Pareto(Maximize("Helpfulness"), Minimize("tokens", scale=1000)), [], REGISTRY)
    assert criteria.terms({"HelpfulnessEvaluator": 0.8, "tokens": 500.0}) == (0.8, -500.0)
    assert criteria.terms({"HelpfulnessEvaluator": 0.8}) is None
    assert Criteria(Minimize("tokens"), [], REGISTRY).terms({"tokens": 5.0}) == (-5.0,)


def test_unmet_names_each_constraint_not_met_on_measured_means():
    criteria = Criteria(
        None, [AtMost("tokens", 4000), AtLeast("Stars", 3), AtLeast("HelpfulnessEvaluator", 0.5)], REGISTRY
    )
    by_case = {"a": {"tokens": 4100.0, "Stars": 4.0}, "b": {"tokens": 4320.0, "Stars": 3.0}}
    assert criteria.unmet_constraints(by_case) == [
        "tokens <= 4000: mean 4210 violates by 210",
        "HelpfulnessEvaluator >= 0.5: never measured",
    ]


# --- scoring trials ---


CRITERIA = Criteria(WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000)), [], REGISTRY)


def test_scores_average_per_case_over_runs():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.6}, tokens=100),
        record("a", 1, scores={"HelpfulnessEvaluator": 1.0}, tokens=300),
        record("b", 0, scores={"HelpfulnessEvaluator": 0.5}, tokens=200),
    ]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert train.metrics_by_case["a"]["HelpfulnessEvaluator"] == pytest.approx(0.8)
    assert train.metrics_by_case["a"]["tokens"] == 200.0
    assert train.metrics_by_run["a"]["HelpfulnessEvaluator"] == [0.6, 1.0]
    assert train.objective_by_run["a"] == pytest.approx([(0.6 - 0.1) / 2, (1.0 - 0.3) / 2])
    assert train.objective_by_case["a"] == pytest.approx(0.3)
    # The measured objective combines the per-metric means across cases, not the per-case objectives.
    assert train.objective == pytest.approx(((0.8 + 0.5) / 2 - 0.2) / 2)
    assert train.is_eligible
    assert train.candidate_id == "c1"


def test_an_agent_failure_scores_the_worst_end_and_its_usage_counts():
    records = [
        record("a", 0, status=TrialStatus.AGENT_FAILURE, tokens=50),
        record("b", 0, scores={"HelpfulnessEvaluator": 1.0}, tokens=150),
    ]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert train.metrics_by_case["a"]["HelpfulnessEvaluator"] == 0.0
    assert train.metrics_by_case["a"]["tokens"] == 50.0
    assert train.failures == 1
    assert train.is_eligible


def test_with_crash_score_none_an_agent_failure_is_missing_data():
    criteria = Criteria(Maximize("Helpfulness"), [], REGISTRY, agent_failure_score=None)
    records = [record("a", 0, status=TrialStatus.AGENT_FAILURE), record("b", 0, scores={"HelpfulnessEvaluator": 1.0})]
    train = score("c1", records, criteria, SPLITS).train
    assert "HelpfulnessEvaluator" not in train.metrics_by_case["a"]
    assert not train.is_eligible


@pytest.mark.parametrize("status", [TrialStatus.INFRASTRUCTURE_INVALID, TrialStatus.EVALUATOR_INVALID])
def test_infrastructure_and_verifier_failures_measure_nothing_and_make_it_ineligible(status):
    records = [
        record("a", 0, status=status, scores={"HelpfulnessEvaluator": 0.0}),
        record("b", 0, scores={"HelpfulnessEvaluator": 1.0}),
    ]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert train.metrics_by_case["a"] == {}
    assert train.objective_by_run["a"] == [None]
    assert not train.is_eligible
    assert train.failures == 0


def test_an_evaluator_that_raised_on_a_completed_run_leaves_its_metric_missing():
    errored = record("a", 0, scores={"Stars": 3.0}).model_copy(
        update={"evaluator_errors": {"HelpfulnessEvaluator": "ValueError"}}
    )
    records = [errored, record("b", 0, scores={"HelpfulnessEvaluator": 1.0})]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert "HelpfulnessEvaluator" not in train.metrics_by_case["a"]
    assert train.metrics_by_case["a"]["tokens"] == 100.0
    assert not train.is_eligible


def test_eligibility_is_per_case_so_one_measured_run_is_enough():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.5}),
        record("a", 1, status=TrialStatus.INFRASTRUCTURE_INVALID),
        record("b", 0, scores={"HelpfulnessEvaluator": 1.0}),
        record("b", 1, scores={"HelpfulnessEvaluator": 0.8}),
    ]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert train.objective_by_run["a"][1] is None
    assert train.metrics_by_case["a"]["HelpfulnessEvaluator"] == 0.5
    assert train.is_eligible


def test_a_crash_scores_every_goal_evaluator_even_one_another_run_found_not_applicable():
    criteria = Criteria(None, [], REGISTRY)
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.8, "Stars": NA}),
        record("a", 1, status=TrialStatus.AGENT_FAILURE),
        record("b", 0, scores={"HelpfulnessEvaluator": 1.0, "Stars": 4.0}),
    ]
    train = score("c1", records, criteria, SPLITS).train
    assert train.metrics_by_case["a"]["Stars"] == 1.0  # the crashed run's worst end; run 0's N/A drops


def test_a_score_outside_its_range_is_taken_out_with_a_reason_never_clipped():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 1.4}),
        record("b", 0, scores={"HelpfulnessEvaluator": 0.5}),
    ]
    train = score("c1", records, CRITERIA, SPLITS).train
    assert "HelpfulnessEvaluator" not in train.metrics_by_case["a"]
    reason = train.unavailable["a"][0]["HelpfulnessEvaluator"]
    assert "outside its range [0.0, 1.0]" in reason and "score_range" in reason
    assert not train.is_eligible


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_score_that_is_not_finite_is_taken_out_even_with_no_range(bad):
    registry = MetricRegistry({"Open": (None, None)})
    criteria = Criteria(Maximize("Open"), [], registry)
    records = [record("a", 0, scores={"Open": bad}), record("b", 0, scores={"Open": 3.0})]
    train = score("c1", records, criteria, SPLITS).train
    assert "not a finite number" in train.unavailable["a"][0]["Open"]
    assert train.objective == 3.0
    assert not train.is_eligible


def test_an_evaluator_not_applicable_on_a_case_does_not_count_against_eligibility():
    criteria = Criteria(WeightedSum(Maximize("Helpfulness"), Maximize("Stars")), [], REGISTRY)
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": NA, "Stars": 4.0}),
        record("b", 0, scores={"HelpfulnessEvaluator": 0.5, "Stars": 2.0}),
    ]
    train = score("c1", records, criteria, SPLITS).train
    assert "HelpfulnessEvaluator" not in train.metrics_by_case["a"]
    assert train.objective_by_case["a"] == pytest.approx(1.0)  # Stars only (4 on 1-5, scaled by 4), its weight rescaled
    assert train.is_eligible
    assert train.unavailable == {}


def test_a_case_is_not_applicable_by_its_detail_rows():
    rows = {"HelpfulnessEvaluator": [NA, NA]}
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.0}, details=rows),
        record("b", 0, scores={"HelpfulnessEvaluator": 0.5}),
    ]
    train = score("c1", records, Criteria(Maximize("Helpfulness"), [], REGISTRY), SPLITS).train
    assert "HelpfulnessEvaluator" not in train.metrics_by_case["a"]
    assert train.is_eligible


def test_an_unreported_cache_metric_stays_unmeasured_not_zero():
    criteria = Criteria(Minimize("cache_read_tokens"), [], REGISTRY)
    records = [record("a", 0), record("b", 0, cache_read_tokens=20)]
    train = score("c1", records, criteria, SPLITS).train
    assert "cache_read_tokens" not in train.metrics_by_case["a"]
    assert train.metrics_by_case["b"]["cache_read_tokens"] == 20.0
    assert not train.is_eligible


def test_a_case_without_records_makes_it_ineligible():
    train = score("c1", [record("a", 0, scores={"HelpfulnessEvaluator": 1.0})], CRITERIA, SPLITS).train
    assert not train.is_eligible


def test_splits_are_scored_apart_and_unscored_ones_are_none():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 1.0}),
        record("b", 0, scores={"HelpfulnessEvaluator": 1.0}),
        record("v", 0, split="validation", scores={"HelpfulnessEvaluator": 0.2}),
    ]
    scores = score("c1", records, CRITERIA, SPLITS)
    assert set(scores.train.metrics_by_case) == {"a", "b"}
    assert scores.validation is not None and set(scores.validation.metrics_by_case) == {"v"}
    assert scores.validation.is_eligible
    assert scores.test is None


def test_measured_gives_a_95_percent_t_interval_over_case_means():
    records = [record(case, 0, tokens=tokens) for case, tokens in [("a", 1000), ("b", 1100), ("c", 1200)]]
    tokens = measured_metrics(score("c1", records, CRITERIA, SPLITS).train, REGISTRY)["tokens"]
    assert tokens.mean == pytest.approx(1100.0) and tokens.n == 3
    # t(0.975, df=2) = 4.3027; sd = 100; se = 57.735
    assert tokens.ci95 == pytest.approx((1100 - 4.3027 * 57.735, 1100 + 4.3027 * 57.735), abs=0.1)


def test_measured_clips_the_interval_to_the_metric_range():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.4}, tokens=100),
        record("b", 0, scores={"HelpfulnessEvaluator": 0.8}, tokens=300),
    ]
    result = measured_metrics(score("c1", records, CRITERIA, SPLITS).train, REGISTRY)
    # t(0.975, df=1) = 12.706, so both unclipped intervals reach past the range
    assert result["HelpfulnessEvaluator"].ci95 == (0.0, 1.0)
    assert result["tokens"].ci95 == pytest.approx((0.0, 200 + 12.706 * 100), abs=0.1)
    assert "cache_read_tokens" not in result


def test_measured_uses_the_runs_when_there_is_a_single_case():
    records = [
        record("a", 0, scores={"HelpfulnessEvaluator": 0.6}),
        record("a", 1, scores={"HelpfulnessEvaluator": 0.7}),
        record("a", 2, scores={"HelpfulnessEvaluator": 0.8}),
    ]
    helpfulness = measured_metrics(score("c1", records, CRITERIA, SPLITS).train, REGISTRY)["HelpfulnessEvaluator"]
    assert helpfulness.n == 3
    assert helpfulness.ci95 == pytest.approx((0.7 - 4.3027 * 0.057735, 0.7 + 4.3027 * 0.057735), abs=1e-3)


def test_measured_has_no_interval_for_one_case_and_one_run():
    one = [record("a", 0, scores={"HelpfulnessEvaluator": 0.4})]
    assert measured_metrics(score("c1", one, CRITERIA, SPLITS).train, REGISTRY)["HelpfulnessEvaluator"].ci95 is None
