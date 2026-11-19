import math
import random
from dataclasses import dataclass, field, replace

import pytest

from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Criteria,
    LowerBound,
    Maximize,
    Mean,
    MetricRegistry,
    Minimize,
    Pareto,
    WeightedSum,
    p_met,
)
from strands_evals.optimization.score_model.ranking import (
    Assessment,
    assess,
    label_candidate,
    pareto_front,
    rank,
    round_to_sd,
)
from strands_evals.types.optimization import CandidateScores, Estimate

HELP = "HelpfulnessEvaluator"
STARS = "Stars"
REGISTRY = MetricRegistry({HELP: (0.0, 1.0), STARS: (1.0, 5.0)})


@dataclass
class FakeFit:
    """A fit that returns fixed estimates per candidate."""

    table: dict[str, dict[str, Estimate]]
    flagged: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def metrics(self):
        return {metric: None for row in self.table.values() for metric in row}

    @property
    def candidates(self):
        return frozenset(self.table)

    def estimates(self, values, candidate=None):
        return dict(self.table[candidate])


def scores(candidate_id, cases, objective):
    """Scores whose per-case metric means are `cases`; `objective` is the measured objective."""
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case={},
        metrics_by_case=cases,
        unavailable={},
        objective_by_run={},
        metrics_by_run={},
        objective=objective,
        is_eligible=True,
        failures=0,
    )


def observed(ids, *metrics):
    """Scores with a value of each of `metrics` on one case: the fit branch judges only measured constraints."""
    return {c: scores(c, {"x": dict.fromkeys(metrics, 0.0)}, None) for c in ids}


def e(mean, sd=0.01):
    return Estimate(mean=mean, sd=sd)


def assessment(candidate_id, objective, *, feasible=True, violation=0.0, terms=None):
    return Assessment(
        candidate_id=candidate_id,
        objective=objective,
        terms=terms if terms is not None else (objective,),
        is_feasible=feasible,
        violations=(violation,),
        total_violation=0.0 if feasible else violation,
        p_met={},
        estimates={},
        adjusted_means={},
        is_from_fit=True,
    )


# ---- round_to_sd ------------------------------------------------------------------------------------------------


def test_values_closer_than_a_tenth_of_the_sd_tie():
    assert round_to_sd(0.4999960020499944, 0.000707) == round_to_sd(0.49999999999941763, 0.000707) == 0.5
    assert round_to_sd(0.52, 0.000707) != round_to_sd(0.5, 0.000707)


@pytest.mark.parametrize("sd", [5e-324, 1e-323])
def test_round_to_sd_handles_an_sd_near_the_smallest_float(sd):
    assert round_to_sd(0.5, sd) == 0.5


@pytest.mark.parametrize("value,sd", [(3.25, 0.0), (math.inf, 0.1), (-0.5, -1.0)])
def test_round_to_sd_leaves_value_without_spread_or_finite_value(value, sd):
    assert round_to_sd(value, sd) == value


def test_round_to_sd_rounds_to_the_tenth_of_the_leading_digit():
    assert round_to_sd(1.23456, 0.3) == 1.23
    assert round_to_sd(1234.5, 300.0) == 1230.0


def test_the_same_decimal_is_the_same_float_whatever_the_sd():
    assert round_to_sd(2.5795, 0.0047) == round_to_sd(2.5795, 0.00016) == 2.5795


# ---- p_met ------------------------------------------------------------------------------------------------------


def test_p_met_without_spread_is_0_or_1():
    assert p_met(AtMost(HELP, 0.5), e(0.5, 0.0)) == 1.0
    assert p_met(AtMost(HELP, 0.5), e(0.6, 0.0)) == 0.0
    assert p_met(AtLeast(HELP, 0.5), e(0.5, 0.0)) == 1.0
    assert p_met(AtLeast(HELP, 0.5), e(0.4, 0.0)) == 0.0


def test_a_fitted_value_at_its_bound_up_to_float_noise_is_met():
    noisy = 0.1 + 0.2  # 0.30000000000000004
    assert p_met(AtMost(HELP, 0.3), e(noisy, 0.0)) == 1.0
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 0.3)], REGISTRY)
    out = assess(
        ["c"],
        fit=FakeFit({"c": {HELP: e(0.5, 0.01), STARS: e(noisy, 0.0)}}),
        criteria=criteria,
        values={"c": {}},
        measured=observed(["c"], STARS),
        rule=Mean(),
    )
    assert out["c"].is_feasible and out["c"].violations == (0.0,)


def test_p_met_is_the_normal_probability():
    assert p_met(AtMost(HELP, 0.5), e(0.5, 0.1)) == pytest.approx(0.5)
    assert p_met(AtMost(HELP, 0.5 + 1.6448536 * 0.1), e(0.5, 0.1)) == pytest.approx(0.95)
    assert p_met(AtLeast(HELP, 0.5 - 1.6448536 * 0.1), e(0.5, 0.1)) == pytest.approx(0.95)


# ---- rules ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.5, -0.1, math.nan])
def test_lower_bound_needs_q_below_one_half(q):
    with pytest.raises(ValueError, match="0 < q < 0.5"):
        LowerBound(q)


def test_rule_configs():
    assert Mean().config() == {"kind": "Mean"}
    assert LowerBound(0.2).config() == {"kind": "LowerBound", "q": 0.2}


# ---- assess -----------------------------------------------------------------------------------------------------


def test_assess_from_the_fit_resolves_each_objective_estimate():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.4999960020499944, 0.000707)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=observed(["a"]), rule=Mean())["a"]
    assert a.objective == 0.5
    assert a.terms == (0.5,)
    assert a.is_from_fit
    assert a.estimates == {HELP: e(0.4999960020499944, 0.000707)}
    assert a.is_feasible and a.violations == () and a.total_violation == 0.0


def test_lower_bound_takes_the_worse_side_of_each_term():
    criteria = Criteria(WeightedSum(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.5, 0.1), STARS: e(3.0, 0.5)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=observed(["a"]), rule=LowerBound(0.1))["a"]
    help_low = round_to_sd(e(0.5, 0.1).quantile(0.1), 0.1)
    stars_high = round_to_sd(e(3.0, 0.5).quantile(0.9), 0.5)
    assert help_low < 0.5 and stars_high > 3.0
    assert a.objective == pytest.approx(criteria.objective_value({HELP: help_low, STARS: stars_high}))


def test_a_constraint_is_judged_at_the_feasibility_quantile():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit(
        {
            "safe": {HELP: e(0.5), STARS: e(2.0, 0.5)},  # 90% quantile 2.64 <= 3
            "risky": {HELP: e(0.5), STARS: e(2.8, 0.5)},  # 90% quantile 3.44 > 3, though the mean meets it
        }
    )
    out = assess(
        ["safe", "risky"],
        fit=fit,
        criteria=criteria,
        values={"safe": {}, "risky": {}},
        measured=observed(["safe", "risky"], STARS),
        rule=Mean(),
    )
    assert out["safe"].is_feasible and out["safe"].p_met["Stars <= 3"] >= 0.9
    assert not out["risky"].is_feasible and out["risky"].p_met["Stars <= 3"] < 0.9
    assert out["risky"].violations[0] == pytest.approx(e(2.8, 0.5).quantile(0.9) - 3.0)
    assert out["risky"].total_violation == pytest.approx(out["risky"].violations[0])


def test_at_least_is_judged_at_the_lower_quantile():
    criteria = Criteria(Maximize(HELP), [AtLeast(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit({"a": {HELP: e(0.5), STARS: e(3.2, 0.5)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=observed(["a"], STARS), rule=Mean())["a"]
    assert a.violations[0] == pytest.approx(3.0 - e(3.2, 0.5).quantile(0.1))
    assert not a.is_feasible


def test_feasible_matches_p_met_at_the_threshold():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    rng = random.Random(0)
    table = {f"c{i}": {HELP: e(0.5), STARS: e(rng.uniform(1.5, 3.5), rng.uniform(0.05, 0.8))} for i in range(200)}
    out = assess(
        list(table),
        fit=FakeFit(table),
        criteria=criteria,
        values=dict.fromkeys(table, {}),
        measured=observed(table, STARS),
        rule=Mean(),
    )
    for a in out.values():
        if abs(a.p_met["Stars <= 3"] - 0.9) > 1e-9:
            assert a.is_feasible == (a.p_met["Stars <= 3"] >= 0.9)


def test_a_flagged_fit_or_an_unfitted_candidate_is_assessed_by_measured_means():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "a": scores("a", {"x": {HELP: 0.2}, "y": {HELP: 0.4}}, 0.3),
        "b": scores("b", {"x": {HELP: 0.9}}, 0.9),
    }
    values = {"a": {}, "b": {}}
    fit = FakeFit({"a": {HELP: e(0.7)}})
    out = assess(["a", "b"], fit=fit, criteria=criteria, values=values, measured=measured, rule=LowerBound(0.1))
    assert out["a"].is_from_fit and out["a"].objective == round_to_sd(e(0.7).quantile(0.1), 0.01)
    assert out["a"].adjusted_means == {}  # assessed from the fit
    # rule ignored; b has no value on y, filled from a's measured 0.4 shifted by b's +0.7 on x and clipped to 1
    assert not out["b"].is_from_fit and out["b"].objective == pytest.approx(0.95) and out["b"].estimates == {}
    flagged = FakeFit({"a": {HELP: e(0.7)}}, flagged=[HELP])
    out = assess(["a"], fit=flagged, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert not out["a"].is_from_fit and out["a"].objective == pytest.approx(0.3)
    assert out["a"].terms == (pytest.approx(0.3),)
    unfitted = assess(["a"], fit=None, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert unfitted["a"].objective == pytest.approx(0.3)


def test_a_flagged_metric_is_assessed_by_its_measured_mean_and_the_others_by_the_fit():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(2.0, 0.5)}}, flagged=[STARS])
    measured = {"a": scores("a", {"x": {HELP: 0.2, STARS: 3.4}, "y": {HELP: 0.4, STARS: 3.6}}, 0.3)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.objective == 0.7 and a.terms == (0.7,)  # helpfulness from the fit
    assert a.violations == (pytest.approx(0.5),) and not a.is_feasible  # Stars' measured mean 3.5, not the fit's 2.0
    assert a.is_from_fit and a.estimates == {HELP: e(0.7)} and a.p_met == {}


def test_a_flagged_objective_metric_is_measured_while_an_unflagged_constraint_keeps_the_fit():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(2.8, 0.5)}}, flagged=[HELP])
    measured = {"a": scores("a", {"x": {HELP: 0.2, STARS: 2.0}, "y": {HELP: 0.4, STARS: 2.0}}, 0.3)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=LowerBound(0.1))["a"]
    assert a.objective == pytest.approx(0.3)  # the measured mean; the rule reads only the fit's terms
    assert a.violations == (pytest.approx(e(2.8, 0.5).quantile(0.9) - 3.0),) and not a.is_feasible
    assert a.estimates == {STARS: e(2.8, 0.5)} and a.p_met == {"Stars <= 3": p_met(AtMost(STARS, 3.0), e(2.8, 0.5))}


def test_a_flagged_metric_judges_its_objective_term_and_its_constraint_alike():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, 0.5), AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.9), STARS: e(2.0)}}, flagged=[HELP])
    measured = {"a": scores("a", {"x": {HELP: 0.4, STARS: 4.0}}, 0.4)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.objective == pytest.approx(0.4)
    assert a.violations[0] == pytest.approx(0.1)  # the measured 0.4 misses 0.5, as the objective says
    assert a.violations[1] < 0  # Stars judged by the fit (2.0), not by the measured 4.0
    assert list(a.p_met) == ["Stars <= 3"]


@pytest.mark.parametrize("combination", [WeightedSum, Pareto])
def test_a_combined_objective_mixes_the_fit_and_the_flagged_terms_measured_means(combination):
    criteria = Criteria(combination(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.61234, 0.01), STARS: e(2.0, 0.5)}}, flagged=[STARS])
    measured = {"a": scores("a", {"x": {HELP: 0.2, STARS: 3.0}, "y": {HELP: 0.4, STARS: 4.0}}, 0.0)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    point = {HELP: round_to_sd(0.61234, 0.01), STARS: 3.5}
    assert a.objective == pytest.approx(criteria.objective_value(point))
    assert a.terms == (pytest.approx(0.612), pytest.approx(-3.5))


def test_lower_bound_reads_only_the_fit_terms_of_a_combined_objective():
    criteria = Criteria(WeightedSum(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.9), STARS: e(3.0, 0.5)}}, flagged=[HELP])
    measured = {"a": scores("a", {"x": {HELP: 0.4, STARS: 2.0}, "y": {HELP: 0.6, STARS: 2.0}}, 0.0)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=LowerBound(0.1))["a"]
    stars_high = round_to_sd(e(3.0, 0.5).quantile(0.9), 0.5)
    assert a.terms == (pytest.approx(0.5), -stars_high)  # Helpfulness at its measured mean, Stars at its worse side


def test_a_flagged_metric_never_measured_leaves_its_constraint_unmet():
    criteria = Criteria(Maximize(HELP), [AtLeast(STARS, 2.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(3.0)}}, flagged=[STARS])
    measured = {"a": scores("a", {"x": {HELP: 0.5}}, 0.5)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.objective == 0.7 and a.violations == (math.inf,) and not a.is_feasible


def test_with_one_metric_flagged_the_ranking_follows_the_fit_on_the_others():
    # Measured on few cases, `lucky` looks better; the fit shrinks it below `steady`. Flagging Stars must not
    # hand the helpfulness ranking back to the measured means.
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 4.0)], REGISTRY)
    table = {"steady": {HELP: e(0.6), STARS: e(2.0)}, "lucky": {HELP: e(0.55), STARS: e(2.0)}}
    measured = {
        "steady": scores("steady", {"x": {HELP: 0.6, STARS: 2.0}}, 0.6),
        "lucky": scores("lucky", {"x": {HELP: 0.9, STARS: 2.0}}, 0.9),
    }
    order = ["steady", "lucky"]
    values = dict.fromkeys(order, {})
    flagged = FakeFit(table, flagged=[STARS])
    out = assess(order, fit=flagged, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert rank(out, order) == ["steady", "lucky"]
    every = FakeFit(table, flagged=[HELP, STARS])  # nothing left to judge by the fit: measured means throughout
    out = assess(order, fit=every, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert rank(out, order) == ["lucky", "steady"] and not any(a.is_from_fit for a in out.values())


def test_a_candidate_the_fit_has_not_seen_is_measured_beside_one_with_a_flagged_metric():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"seen": {HELP: e(0.5), STARS: e(2.0)}}, flagged=[STARS])
    measured = {
        "seen": scores("seen", {"x": {HELP: 0.9, STARS: 2.5}}, 0.9),
        "new": scores("new", {"x": {HELP: 0.6, STARS: 2.5}}, 0.6),
    }
    values = {"seen": {}, "new": {}}
    out = assess(["seen", "new"], fit=fit, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert out["seen"].objective == 0.5 and out["seen"].is_from_fit
    assert out["new"].objective == 0.6 and not out["new"].is_from_fit and out["new"].estimates == {}
    assert out["seen"].violations == out["new"].violations == (pytest.approx(-0.5),)  # both by the measured 2.5
    assert rank(out, ["seen", "new"]) == ["new", "seen"]


def test_a_flagged_name_that_is_no_goal_metric_sends_every_metric_to_measured_means():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(2.0)}}, flagged=["the model"])
    measured = {"a": scores("a", {"x": {HELP: 0.3, STARS: 3.5}}, 0.3)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.objective == 0.3 and not a.is_from_fit and a.estimates == {} and not a.is_feasible


def test_measured_constraints_use_the_mean_over_cases():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0), AtLeast(HELP, 0.3)], REGISTRY)
    measured = {"a": scores("a", {"x": {HELP: 0.1, STARS: 3.5}, "y": {HELP: 0.3, STARS: 3.5}}, 0.2)}
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.violations == (pytest.approx(0.5), pytest.approx(0.1))
    assert a.total_violation == pytest.approx(0.6)
    assert not a.is_feasible and a.p_met == {}


def test_a_measured_mean_at_its_bound_up_to_float_noise_is_met():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.3)], REGISTRY)
    measured = {"a": scores("a", {"x": {HELP: 0.1 + 0.2}}, 0.1 + 0.2)}
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.violations == (0.0,) and a.is_feasible
    assert criteria.unmet_constraints(a.adjusted_means) == []


def test_a_constraint_never_measured_is_unmet():
    criteria = Criteria(Maximize(HELP), [AtLeast(STARS, 2.0)], REGISTRY)
    measured = {"a": scores("a", {"x": {HELP: 0.5}}, 0.5)}
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.violations == (math.inf,) and not a.is_feasible


def test_the_fit_judges_a_constraint_never_measured_unmet_whatever_it_predicts():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit({"measured": {HELP: e(0.5), STARS: e(2.0)}, "never": {HELP: e(0.5), STARS: e(2.0)}})
    measured = {
        "measured": scores("measured", {"x": {HELP: 0.5, STARS: 2.0}}, 0.5),
        "never": scores("never", {"x": {HELP: 0.5}, "y": {HELP: 0.5}}, 0.5),  # Stars N/A on every case
    }
    out = assess(
        list(fit.table), fit=fit, criteria=criteria, values=dict.fromkeys(fit.table, {}), measured=measured, rule=Mean()
    )
    assert out["measured"].is_feasible and out["measured"].p_met == {"Stars <= 3": 1.0}
    never = out["never"]
    assert never.is_from_fit and not never.is_feasible
    assert never.violations == (math.inf,) and never.total_violation == math.inf
    assert never.p_met == {} and never.estimates[STARS] == e(2.0)  # the prediction stays, without a P(met)
    label = label_candidate(measured["never"], never, criteria=criteria)
    assert (label.outcome, label.unmet_constraints, label.p_met) == ("infeasible", ["Stars <= 3: never measured"], {})
    assert rank(out, ["never", "measured"]) == ["measured", "never"]


def test_the_fit_still_judges_a_measured_constraint_by_its_estimate():
    # The measured mean violates; the fit's estimate meets the bound and decides.
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.5), STARS: e(2.0)}})
    measured = {"a": scores("a", {"x": {HELP: 0.5, STARS: 4.5}}, 0.5)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.is_feasible and a.violations[0] < 0


def test_an_unmeasured_constraint_ranks_after_any_finite_violation():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, 0.6), AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"over": {HELP: e(0.9), STARS: e(4.0)}, "never": {HELP: e(0.5), STARS: e(2.0)}})
    measured = {
        "over": scores("over", {"x": {HELP: 0.9, STARS: 4.0}}, 0.9),
        "never": scores("never", {"x": {HELP: 0.5}}, 0.5),
    }
    out = assess(
        ["never", "over"], fit=fit, criteria=criteria, values={"never": {}, "over": {}}, measured=measured, rule=Mean()
    )
    never = out["never"]
    assert never.violations[0] > 0 and never.violations[1] == math.inf
    assert set(never.p_met) == {f"{HELP} >= 0.6"}
    label = label_candidate(measured["never"], never, criteria=criteria)
    assert label.unmet_constraints == [f"{HELP} >= 0.6: P(met) 0 < 0.9", "Stars <= 3: never measured"]
    assert rank(out, ["never", "over"]) == ["over", "never"]


def _help(by_case):
    """Per-case means of HELP alone; a case left out is one the evaluator found N/A."""
    return {case: {HELP: value} for case, value in by_case.items()}


def test_measured_means_fill_the_cases_a_candidate_leaves_unjudged():
    # "b" is 0.1 below "a" where both are judged and leaves the evaluator nothing to judge on the hard cases,
    # where "a" scored 0.5; plain means would rank "b" first (0.9 against 0.75).
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "a": scores("a", _help({"e1": 1.0, "e2": 1.0, "h1": 0.5, "h2": 0.5}), 0.75),
        "b": scores("b", _help({"e1": 0.9, "e2": 0.9}), 0.9),
    }
    values = {"a": {}, "b": {}}
    out = assess(["b", "a"], fit=None, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert out["a"].objective == 0.75  # a value on every case: its plain mean
    assert out["b"].objective == pytest.approx(0.65)  # h1 and h2 filled at a's 0.5 shifted by b's -0.1
    assert out["a"].adjusted_means == {HELP: 0.75} and out["b"].adjusted_means == {HELP: pytest.approx(0.65)}
    assert rank(out, ["b", "a"]) == ["a", "b"]
    # Alone, there is nothing to fill from.
    assert (
        assess(["b"], fit=None, criteria=criteria, values=values, measured=measured, rule=Mean())["b"].objective == 0.9
    )
    # Measuring b's hard cases at exactly the filled value leaves its mean where it was.
    measured["b"] = scores("b", _help({"e1": 0.9, "e2": 0.9, "h1": 0.4, "h2": 0.4}), 0.65)
    out = assess(["b", "a"], fit=None, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert out["b"].objective == pytest.approx(0.65)


def test_the_others_mean_leaves_the_candidate_out():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "a": scores("a", _help({"e1": 1.0, "e2": 0.8, "h1": 0.4}), 2.2 / 3),
        "b": scores("b", _help({"e1": 0.6, "e2": 0.6, "h1": 0.2}), 1.4 / 3),
        "c": scores("c", _help({"e1": 0.9, "e2": 0.5}), 0.7),
    }
    out = assess(
        list(measured), fit=None, criteria=criteria, values={c: {} for c in measured}, measured=measured, rule=Mean()
    )
    # The others' means on e1 and e2 are 0.8 and 0.7, so c's shift is (0.1 - 0.2) / 2; h1 is filled at 0.3 - 0.05.
    assert out["c"].objective == pytest.approx((0.9 + 0.5 + 0.25) / 3)
    assert out["a"].objective == pytest.approx(2.2 / 3) and out["b"].objective == pytest.approx(1.4 / 3)


def test_a_filled_case_is_clipped_to_the_metric_range():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "a": scores("a", _help({"e1": 0.9, "e2": 0.9, "h1": 0.1, "h2": 0.1}), 0.5),
        "b": scores("b", _help({"e1": 0.4, "e2": 0.4}), 0.4),
    }
    out = assess(["a", "b"], fit=None, criteria=criteria, values={"a": {}, "b": {}}, measured=measured, rule=Mean())
    assert out["b"].objective == pytest.approx(0.2)  # h1 and h2: 0.1 - 0.5 clipped to 0


def test_with_no_case_in_common_the_others_values_fill_in_unshifted():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {"x": scores("x", _help({"c1": 1.0, "c2": 1.0}), 1.0), "y": scores("y", _help({"c3": 0.0}), 0.0)}
    out = assess(["x", "y"], fit=None, criteria=criteria, values={"x": {}, "y": {}}, measured=measured, rule=Mean())
    assert out["x"].objective == pytest.approx(2 / 3) and out["y"].objective == pytest.approx(2 / 3)


def test_a_candidate_sharing_a_gap_doesn_t_move_the_others_fill():
    # "worse" is 0.05 below the baseline on the one case both were judged on, and "weak" shares its gap on "easy".
    # Each fill is shifted against the candidates it comes from (only the baseline is judged on "easy"), so "weak"
    # can't raise "worse"'s fill: easy is filled at 0.65 and "worse" ranks below the baseline.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "base": scores("base", _help({"easy": 0.7, "hard": 0.5}), 0.6),
        "worse": scores("worse", _help({"hard": 0.45}), 0.45),
        "weak": scores("weak", _help({"hard": 0.1}), 0.1),
    }
    ids = list(measured)
    out = assess(ids, fit=None, criteria=criteria, values={c: {} for c in ids}, measured=measured, rule=Mean())
    assert out["worse"].objective == pytest.approx(0.55)
    assert out["weak"].objective == pytest.approx(0.2)
    assert out["worse"].adjusted_by_case == {"hard": {HELP: 0.45}, "easy": {HELP: pytest.approx(0.65)}}
    assert out["base"].adjusted_by_case == {"easy": {HELP: 0.7}, "hard": {HELP: 0.5}}
    assert rank(out, ids) == ["base", "worse", "weak"]


def test_candidates_equal_where_both_were_judged_tie_whoever_shares_the_gap():
    # "b" equals "a" on "easy" and is N/A on "hard"; "c" and "d" share that gap. Only "a" was judged on "hard", so
    # "b"'s fill there is "a"'s value and the two tie.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "a": scores("a", _help({"easy": 0.8, "hard": 0.4}), 0.6),
        "b": scores("b", _help({"easy": 0.8}), 0.8),
        "c": scores("c", _help({"easy": 0.5}), 0.5),
        "d": scores("d", _help({"easy": 0.5}), 0.5),
    }
    ids = list(measured)
    out = assess(ids, fit=None, criteria=criteria, values={c: {} for c in ids}, measured=measured, rule=Mean())
    assert out["b"].objective == out["a"].objective == pytest.approx(0.6)


def test_a_filled_value_at_a_range_end_up_to_float_noise_isn_t_clipped():
    # The baseline's "hard" value is one float step below the range's low end, which scoring keeps; "f" equals it
    # where judged, so its fill there is that value, and clipping it to 0 would rank "f" above the baseline.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    measured = {
        "base": scores("base", _help({"easy": 0.3, "hard": -(2**-52)}), 0.15),
        "f": scores("f", _help({"easy": 0.3}), 0.3),
    }
    out = assess(
        ["base", "f"], fit=None, criteria=criteria, values={"base": {}, "f": {}}, measured=measured, rule=Mean()
    )
    assert out["f"].objective == out["base"].objective
    assert rank(out, ["base", "f"]) == ["base", "f"]


def test_the_measured_label_is_criteria_unmet_constraints_on_the_adjusted_means():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, 0.9), AtMost(STARS, 3.0)], REGISTRY)
    rng = random.Random(0)
    for _ in range(300):
        measured = {
            c: scores(
                c,
                {
                    f"k{j}": {HELP: round(rng.random(), 4), STARS: round(rng.uniform(1, 5), 4)}
                    for j in range(4)
                    if c == "a" or rng.random() < 0.7
                },
                0.0,
            )
            for c in ("a", "b")
        }
        ids = [c for c in measured if measured[c].metrics_by_case]
        out = assess(ids, fit=None, criteria=criteria, values={c: {} for c in ids}, measured=measured, rule=Mean())
        for c in ids:
            label = label_candidate(measured[c], out[c], criteria=criteria)
            assert label.unmet_constraints == criteria.unmet_constraints(out[c].adjusted_means)
            assert (label.outcome == "feasible") == out[c].is_feasible


def test_a_flagged_metric_is_filled_pairwise_and_its_label_states_the_judged_mean():
    # HELP is flagged, so it is judged by its adjusted mean while STARS comes from the fit; "weak" shares "worse"'s
    # gap on "easy" and must not raise its fill there.
    criteria = Criteria(Maximize(STARS), [AtLeast(HELP, 0.7)], REGISTRY)
    measured = {
        "base": scores("base", {"easy": {HELP: 0.7, STARS: 3.0}, "hard": {HELP: 0.5, STARS: 3.0}}, 3.0),
        "worse": scores("worse", {"easy": {STARS: 3.0}, "hard": {HELP: 0.45, STARS: 3.0}}, 3.0),
        "weak": scores("weak", {"easy": {STARS: 3.0}, "hard": {HELP: 0.1, STARS: 3.0}}, 3.0),
    }
    ids = list(measured)
    fit = FakeFit({c: {HELP: e(0.9), STARS: e(3.0)} for c in ids}, flagged=[HELP])
    out = assess(ids, fit=fit, criteria=criteria, values={c: {} for c in ids}, measured=measured, rule=Mean())
    worse = out["worse"]
    assert worse.is_from_fit and set(worse.estimates) == {STARS}
    assert worse.adjusted_means == {HELP: pytest.approx(0.55)}
    assert worse.adjusted_by_case == {"hard": {HELP: 0.45}, "easy": {HELP: pytest.approx(0.65)}}
    label = label_candidate(measured["worse"], worse, criteria=criteria)
    assert label.unmet_constraints == criteria.unmet_constraints(worse.adjusted_means)


def test_measured_constraints_and_the_label_judge_the_filled_means():
    criteria = Criteria(Maximize(STARS), [AtLeast(HELP, 0.7)], REGISTRY)
    measured = {
        "a": scores("a", {c: {HELP: 1.0 if c[0] == "e" else 0.5, STARS: 3.0} for c in ("e1", "e2", "h1", "h2")}, 3.0),
        # N/A on the hard cases: its plain mean, 0.9, would meet the bound.
        "b": scores(
            "b", {c: {HELP: 0.9, STARS: 3.0} if c[0] == "e" else {STARS: 3.0} for c in ("e1", "e2", "h1", "h2")}, 3.0
        ),
        "c": scores("c", {c: {STARS: 3.0} for c in ("e1", "e2", "h1", "h2")}, 3.0),  # N/A everywhere
    }
    out = assess(
        ["a", "b", "c"], fit=None, criteria=criteria, values={c: {} for c in measured}, measured=measured, rule=Mean()
    )
    assert out["a"].is_feasible
    assert out["b"].violations == (pytest.approx(0.05),) and not out["b"].is_feasible
    assert out["c"].violations == (math.inf,)
    label = label_candidate(measured["b"], out["b"], criteria=criteria)
    assert label.unmet_constraints == [f"{HELP} >= 0.7: mean 0.65 violates by 0.05"]
    assert label.adjusted_means == {HELP: pytest.approx(0.65), STARS: 3.0}
    assert label_candidate(measured["c"], out["c"], criteria=criteria).unmet_constraints == [
        f"{HELP} >= 0.7: never measured"
    ]


def test_the_measured_label_words_constraints_as_criteria_unmet_constraints():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0), AtLeast(HELP, 0.6)], REGISTRY)
    measured = {"a": scores("a", {"x": {HELP: 0.5, STARS: 3.5}, "y": {HELP: 0.4, STARS: 3.6}}, 0.45)}
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    label = label_candidate(measured["a"], a, criteria=criteria)
    assert label.unmet_constraints == criteria.unmet_constraints({HELP: 0.45, STARS: 3.55})
    assert len(label.unmet_constraints) == 2


def test_measured_pareto_terms_use_the_filled_means():
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    measured = {
        "a": scores("a", {c: {HELP: 1.0 if c[0] == "e" else 0.5, STARS: 2.0} for c in ("e1", "e2", "h1", "h2")}, 0.75),
        "b": scores(
            "b", {c: {HELP: 0.9, STARS: 1.0} if c[0] == "e" else {STARS: 1.0} for c in ("e1", "e2", "h1", "h2")}, 0.95
        ),
    }
    out = assess(["a", "b"], fit=None, criteria=criteria, values={"a": {}, "b": {}}, measured=measured, rule=Mean())
    assert out["b"].terms == (pytest.approx(0.65), pytest.approx(-1.0))
    # On plain means, b (0.9, -1.0) would dominate a (0.75, -2.0).
    assert pareto_front(out, ["a", "b"]) == ["a", "b"]


def test_terms_follow_the_objective():
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.6), STARS: e(2.0)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=observed(["a"]), rule=Mean())["a"]
    assert a.terms == (pytest.approx(0.6), pytest.approx(-2.0))


# ---- rank -------------------------------------------------------------------------------------------------------


def test_feasible_first_by_objective_then_infeasible_by_least_violation():
    assessments = {
        "base": assessment("base", 0.5),
        "best": assessment("best", 0.8),
        "close": assessment("close", 0.1, feasible=False, violation=0.1),
        "far": assessment("far", 0.99, feasible=False, violation=2.0),
        "close_better": assessment("close_better", 0.3, feasible=False, violation=0.1),
    }
    order = ["base", "best", "close", "far", "close_better"]
    assert rank(assessments, order) == ["best", "base", "close_better", "close", "far"]


def test_under_pareto_a_candidate_without_terms_loses_an_equal_objective_whatever_the_terms_sign():
    # `no_terms` is scored first; `with_terms`' first term is negative, which an unequal key shape compared
    # against `no_terms`' position.
    no_terms = replace(assessment("no_terms", 0.5), terms=None)
    with_terms = assessment("with_terms", 0.5, terms=(-2.0, 0.8))
    assessments = {"no_terms": no_terms, "with_terms": with_terms}
    assert rank(assessments, ["no_terms", "with_terms"], is_pareto=True) == ["with_terms", "no_terms"]
    # Outside the search's Pareto tie-break, missing terms change nothing: the earlier candidate, or most trials.
    assert rank(assessments, ["no_terms", "with_terms"]) == ["no_terms", "with_terms"]
    counts = {"no_terms": 3, "with_terms": 1}
    assert rank(assessments, ["no_terms", "with_terms"], is_pareto=True, trial_counts=counts) == [
        "no_terms",
        "with_terms",
    ]


def test_rank_skips_ids_without_an_assessment():
    assert rank({"a": assessment("a", 0.5)}, ["b", "a"]) == ["a"]


def test_two_estimates_within_resolution_tie_to_the_earlier_candidate_in_the_search():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    fit = FakeFit({"late": {HELP: e(0.50002, 0.001)}, "early": {HELP: e(0.49999, 0.001)}})
    out = assess(
        ["late", "early"],
        fit=fit,
        criteria=criteria,
        values={"late": {}, "early": {}},
        measured=observed(["late", "early"]),
        rule=Mean(),
    )
    assert out["late"].objective == out["early"].objective
    assert rank(out, ["early", "late"]) == ["early", "late"]


def test_at_the_final_pick_ties_go_to_most_trials_then_earlier():
    assessments = {c: assessment(c, 0.5) for c in ("base", "a", "b")}
    order = ["base", "a", "b"]
    assert rank(assessments, order, trial_counts={"base": 10, "a": 10, "b": 10}) == ["base", "a", "b"]
    assert rank(assessments, order, trial_counts={"base": 10, "a": 10, "b": 20}) == ["b", "base", "a"]


def test_at_the_final_pick_the_baseline_wins_a_pareto_tie_with_another_trade_off():
    assessments = {
        "base": assessment("base", 0.5, terms=(0.6, -0.4)),
        "other": assessment("other", 0.5, terms=(0.8, -0.6)),
    }
    assert rank(assessments, ["base", "other"], is_pareto=True, trial_counts={"base": 5, "other": 5})[0] == "base"


def test_pareto_search_ties_go_to_the_larger_terms():
    assessments = {
        "first": assessment("first", 0.5, terms=(0.6, -0.4)),
        "second": assessment("second", 0.5, terms=(0.8, -0.6)),
    }
    assert rank(assessments, ["first", "second"], is_pareto=True) == ["second", "first"]
    assert rank(assessments, ["first", "second"]) == ["first", "second"]  # not Pareto: the earlier


def test_lower_bound_ranks_a_wide_candidate_below_a_steadier_one():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    fit = FakeFit({"wide": {HELP: e(0.62, 0.2)}, "steady": {HELP: e(0.58, 0.01)}})
    values = {"wide": {}, "steady": {}}

    def top(rule):
        out = assess(
            ["wide", "steady"], fit=fit, criteria=criteria, values=values, measured=observed(values), rule=rule
        )
        return rank(out, ["wide", "steady"], trial_counts={"wide": 3, "steady": 3})[0]

    assert top(Mean()) == "wide"
    assert top(LowerBound(0.1)) == "steady"


def _held_repeatedly(assessments, order, is_pareto):
    """The reference ranking: take the held candidate of what is left, again and again.

    Held: among the feasible (else those with the least violation), the highest objective; under Pareto, the
    front's member with the largest (objective, terms); otherwise the first of the highest.
    """
    rest, out = list(order), []
    while rest:
        feasible = [c for c in rest if assessments[c].is_feasible]
        if feasible:
            pool = feasible
        else:
            least = min(assessments[c].total_violation for c in rest)
            pool = [c for c in rest if assessments[c].total_violation == least]
        if is_pareto:

            def dominates(a, b):
                x, y = assessments[a].terms, assessments[b].terms
                return all(p >= q for p, q in zip(x, y, strict=True)) and x != y

            front = [c for c in pool if not any(dominates(o, c) for o in pool)]
            held = max(front, key=lambda c: (assessments[c].objective, assessments[c].terms))
        else:
            held = max(pool, key=lambda c: assessments[c].objective)
        out.append(held)
        rest.remove(held)
    return out


@pytest.mark.parametrize("is_pareto", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_rank_equals_taking_the_held_candidate_repeatedly(seed, is_pareto):
    criteria = Criteria(Pareto(Maximize(HELP), Maximize(STARS)), [AtMost(STARS, 4.0)], REGISTRY)
    rng = random.Random(seed)
    table = {
        f"c{i}": {HELP: e(rng.choice([0.2, 0.4, 0.6, 0.8]), 0.01), STARS: e(rng.choice([2.0, 3.0, 4.5, 5.0]), 0.01)}
        for i in range(12)
    }
    order = list(table)
    out = assess(
        order,
        fit=FakeFit(table),
        criteria=criteria,
        values=dict.fromkeys(order, {}),
        measured=observed(order, STARS),
        rule=Mean(),
    )
    assert rank(out, order, is_pareto=is_pareto) == _held_repeatedly(out, order, is_pareto)


# ---- pareto_front -----------------------------------------------------------------------------------------------


def test_the_front_holds_exactly_the_feasible_non_dominated_candidates():
    assessments = {
        "a": assessment("a", 0.0, terms=(0.9, -5.0)),
        "b": assessment("b", 0.0, terms=(0.5, -1.0)),
        "dominated": assessment("dominated", 0.0, terms=(0.4, -2.0)),
        "same_as_b": assessment("same_as_b", 0.0, terms=(0.5, -1.0)),
        "cheap_but_infeasible": assessment(
            "cheap_but_infeasible", 0.0, feasible=False, violation=1.0, terms=(1.0, 0.0)
        ),
    }
    order = ["b", "a", "dominated", "same_as_b", "cheap_but_infeasible"]
    assert pareto_front(assessments, order) == ["b", "a", "same_as_b"]


def test_the_front_skips_ids_without_an_assessment():
    assessments = {"a": assessment("a", 0.0, terms=(0.9, -5.0)), "b": assessment("b", 0.0, terms=(0.5, -1.0))}
    assert pareto_front(assessments, ["a", "unassessed", "b"]) == ["a", "b"]


def test_constraints_filter_the_front_and_never_add_a_dimension():
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(STARS)), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit(
        {
            "cheap": {HELP: e(0.5, 0.0), STARS: e(1.0, 0.0)},
            "good": {HELP: e(0.9, 0.0), STARS: e(2.5, 0.0)},
            "over": {HELP: e(0.95, 0.0), STARS: e(3.5, 0.0)},
            "worse": {HELP: e(0.4, 0.0), STARS: e(1.5, 0.0)},
        }
    )
    order = list(fit.table)
    out = assess(
        order, fit=fit, criteria=criteria, values=dict.fromkeys(order, {}), measured=observed(order, STARS), rule=Mean()
    )
    assert pareto_front(out, order) == ["cheap", "good"]


# ---- label_candidate -------------------------------------------------------------------------------------


def _label_scores(candidate_id, help_by_case, *, eligible=True):
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case={c: v for c, v in help_by_case.items() if v is not None},
        metrics_by_case={c: {HELP: v} for c, v in help_by_case.items() if v is not None},
        unavailable={},
        objective_by_run={c: [v] for c, v in help_by_case.items()},
        metrics_by_run={c: {HELP: [v]} for c, v in help_by_case.items()},
        objective=1.0 if eligible else None,
        is_eligible=eligible,
        failures=0,
    )


def _label_assessment(candidate_id, *, objective, feasible, violation=0.0, from_fit=True, p_met=None, n_constraints=0):
    violations = () if n_constraints == 0 else (0.0 if feasible else violation,) * n_constraints
    return Assessment(
        candidate_id=candidate_id,
        objective=objective,
        terms=(objective,),
        is_feasible=feasible,
        violations=violations,
        total_violation=sum(v for v in violations if v > 0),
        p_met=p_met or {},
        estimates={HELP: Estimate(mean=objective, sd=0.01)} if from_fit else {},
        adjusted_means={} if from_fit else {HELP: objective},
        is_from_fit=from_fit,
    )


def test_label_candidate_feasible_from_the_fit():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    label = label_candidate(
        _label_scores("a", {"c1": 0.9}), _label_assessment("a", objective=0.9, feasible=True), criteria=criteria
    )
    assert (label.outcome, label.reason) == ("feasible", "meets every constraint")
    assert label.is_from_fit and label.estimates and label.p_met == {}
    assert label.unmet_constraints == []


def test_label_candidate_infeasible_names_the_unmet_constraint_with_p_met():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.5)], REGISTRY, feasibility_threshold=0.9)
    a = _label_assessment(
        "a", objective=0.9, feasible=False, violation=0.4, n_constraints=1, p_met={f"{HELP} <= 0.5": 0.2}
    )
    label = label_candidate(_label_scores("a", {"c1": 0.9}), a, criteria=criteria)
    assert label.outcome == "infeasible"
    assert label.unmet_constraints == [f"{HELP} <= 0.5: P(met) 0.2 < 0.9"]
    assert label.reason == label.unmet_constraints[0]


def test_label_candidate_infeasible_from_measured_means_lists_the_constraint():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.5)], REGISTRY)
    # measured mean 0.8 over two cases violates "<= 0.5".
    a = _label_assessment("a", objective=0.8, feasible=False, violation=0.3, from_fit=False, n_constraints=1)
    label = label_candidate(_label_scores("a", {"c1": 0.8, "c2": 0.8}), a, criteria=criteria)
    assert (label.outcome, label.is_from_fit, label.estimates, label.p_met) == ("infeasible", False, {}, {})
    assert label.unmet_constraints == [f"{HELP} <= 0.5: mean 0.8 violates by 0.3"]


def test_label_candidate_judges_each_constraint_as_the_ranking_did():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0), AtLeast(HELP, 0.8)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(2.9, 0.5)}}, flagged=[HELP])
    candidate = scores("a", {"x": {HELP: 0.5, STARS: 2.0}, "y": {HELP: 0.7, STARS: 2.0}}, 0.6)
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured={"a": candidate}, rule=Mean())["a"]
    label = label_candidate(candidate, a, criteria=criteria)
    stars_p = p_met(AtMost(STARS, 3.0), e(2.9, 0.5))
    assert label.unmet_constraints == [
        f"Stars <= 3: P(met) {stars_p:.2g} < 0.9",  # the fit's, not the measured 2.0
        f"{HELP} >= 0.8: mean 0.6 violates by 0.2",  # the flagged metric's measured mean
    ]
    assert label.outcome == "infeasible" and label.is_from_fit
    assert label.estimates == {STARS: e(2.9, 0.5)} and list(label.p_met) == ["Stars <= 3"]


@pytest.mark.parametrize(
    "cases",
    [
        {"x": {HELP: 0.5, STARS: 3.5}, "y": {HELP: 0.7, STARS: 3.6}},
        {"x": {HELP: 0.5}},  # Stars never measured
        {"x": {HELP: 0.9, STARS: 2.0}},
    ],
)
def test_label_candidate_by_measured_means_gives_the_criteria_sentences(cases):
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0), AtLeast(HELP, 0.8)], REGISTRY)
    candidate = scores("a", cases, 0.5)
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured={"a": candidate}, rule=Mean())["a"]
    assert label_candidate(candidate, a, criteria=criteria).unmet_constraints == criteria.unmet_constraints(
        a.adjusted_means
    )


def test_label_candidate_ineligible_when_a_train_case_has_no_counted_run():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    candidate = _label_scores("a", {"c1": 0.9, "c2": None}, eligible=False)
    label = label_candidate(candidate, None, criteria=criteria)
    assert label.outcome == "ineligible" and "c2" in label.reason
    assert label.estimates == {} and label.p_met == {} and label.unmet_constraints == []


def test_a_flagged_metric_with_an_na_case_is_judged_by_its_adjusted_mean():
    # "a" has no Stars value on the hard case "y"; its plain mean (2.0) would meet the limit. Filled from "b" (no
    # difference on "x"), its adjusted mean is 3.5, the one the ranking judges and records.
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(2.0)}, "b": {HELP: e(0.6), STARS: e(2.0)}}, flagged=[STARS])
    measured = {
        "a": scores("a", {"x": {HELP: 0.7, STARS: 2.0}, "y": {HELP: 0.7}}, 0.7),
        "b": scores("b", {"x": {HELP: 0.6, STARS: 2.0}, "y": {HELP: 0.6, STARS: 5.0}}, 0.6),
    }
    out = assess(["a", "b"], fit=fit, criteria=criteria, values={"a": {}, "b": {}}, measured=measured, rule=Mean())
    a = out["a"]
    assert a.adjusted_means == {STARS: pytest.approx(3.5)} and a.estimates == {HELP: e(0.7)}
    assert a.violations == (pytest.approx(0.5),) and not a.is_feasible
    label = label_candidate(measured["a"], a, criteria=criteria)
    assert label.unmet_constraints == ["Stars <= 3: mean 3.5 violates by 0.5"]
    assert label.adjusted_means == {STARS: pytest.approx(3.5)} and label.is_from_fit


def test_an_unflagged_constraint_never_measured_stays_unmet_under_a_partly_flagged_fit():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0), AtLeast(HELP, 0.5)], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.9), STARS: e(2.0)}}, flagged=[HELP])
    measured = {"a": scores("a", {"x": {HELP: 0.4}}, 0.4)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.violations == (math.inf, pytest.approx(0.1)) and a.p_met == {}
    assert a.estimates == {STARS: e(2.0)} and a.adjusted_means == {HELP: pytest.approx(0.4)}
    label = label_candidate(measured["a"], a, criteria=criteria)
    assert label.unmet_constraints == [
        "Stars <= 3: never measured",
        "HelpfulnessEvaluator >= 0.5: mean 0.4 violates by 0.1",
    ]


def test_a_flagged_term_never_measured_drops_from_a_weighted_sum():
    criteria = Criteria(WeightedSum(Maximize(HELP), Maximize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.7), STARS: e(4.0)}}, flagged=[STARS])
    measured = {"a": scores("a", {"x": {HELP: 0.5}}, 0.5)}
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.objective == pytest.approx(criteria.objective_value({HELP: 0.7})) and a.terms is None
    assert a.adjusted_means == {} and a.is_from_fit
