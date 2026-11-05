import math
import random
from dataclasses import dataclass, field

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
from strands_evals.optimization.score_model.ranking import Assessment, assess, pareto_front, rank, round_to_sd
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
        is_from_fit=True,
    )


# ---- round_to_sd ------------------------------------------------------------------------------------------------


def test_values_closer_than_a_tenth_of_the_sd_tie():
    assert round_to_sd(0.4999960020499944, 0.000707) == round_to_sd(0.49999999999941763, 0.000707) == 0.5
    assert round_to_sd(0.52, 0.000707) != round_to_sd(0.5, 0.000707)


@pytest.mark.parametrize("value,sd", [(3.25, 0.0), (math.inf, 0.1), (-0.5, -1.0)])
def test_round_to_sd_leaves_value_without_spread_or_finite_value(value, sd):
    assert round_to_sd(value, sd) == value


def test_round_to_sd_rounds_to_the_tenth_of_the_leading_digit():
    assert round_to_sd(1.23456, 0.3) == pytest.approx(1.23)
    assert round_to_sd(1234.5, 300.0) == pytest.approx(1230.0)


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
        measured={},
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
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured={}, rule=Mean())["a"]
    assert a.objective == 0.5
    assert a.terms == (0.5,)
    assert a.is_from_fit
    assert a.estimates == {HELP: e(0.4999960020499944, 0.000707)}
    assert a.is_feasible and a.violations == () and a.total_violation == 0.0


def test_lower_bound_takes_the_worse_side_of_each_term():
    criteria = Criteria(WeightedSum(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.5, 0.1), STARS: e(3.0, 0.5)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured={}, rule=LowerBound(0.1))["a"]
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
        ["safe", "risky"], fit=fit, criteria=criteria, values={"safe": {}, "risky": {}}, measured={}, rule=Mean()
    )
    assert out["safe"].is_feasible and out["safe"].p_met["Stars <= 3"] >= 0.9
    assert not out["risky"].is_feasible and out["risky"].p_met["Stars <= 3"] < 0.9
    assert out["risky"].violations[0] == pytest.approx(e(2.8, 0.5).quantile(0.9) - 3.0)
    assert out["risky"].total_violation == pytest.approx(out["risky"].violations[0])


def test_at_least_is_judged_at_the_lower_quantile():
    criteria = Criteria(Maximize(HELP), [AtLeast(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    fit = FakeFit({"a": {HELP: e(0.5), STARS: e(3.2, 0.5)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured={}, rule=Mean())["a"]
    assert a.violations[0] == pytest.approx(3.0 - e(3.2, 0.5).quantile(0.1))
    assert not a.is_feasible


def test_feasible_matches_p_met_at_the_threshold():
    criteria = Criteria(Maximize(HELP), [AtMost(STARS, 3.0)], REGISTRY, feasibility_threshold=0.9)
    rng = random.Random(0)
    table = {f"c{i}": {HELP: e(0.5), STARS: e(rng.uniform(1.5, 3.5), rng.uniform(0.05, 0.8))} for i in range(200)}
    out = assess(
        list(table), fit=FakeFit(table), criteria=criteria, values=dict.fromkeys(table, {}), measured={}, rule=Mean()
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
    assert not out["b"].is_from_fit and out["b"].objective == 0.9 and out["b"].estimates == {}  # rule ignored
    flagged = FakeFit({"a": {HELP: e(0.7)}}, flagged=[HELP])
    out = assess(["a"], fit=flagged, criteria=criteria, values=values, measured=measured, rule=Mean())
    assert not out["a"].is_from_fit and out["a"].objective == 0.3 and out["a"].terms == (pytest.approx(0.3),)
    assert (
        assess(["a"], fit=None, criteria=criteria, values=values, measured=measured, rule=Mean())["a"].objective == 0.3
    )


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
    assert criteria.unmet_constraints(measured["a"].metrics_by_case) == []


def test_a_constraint_never_measured_is_unmet():
    criteria = Criteria(Maximize(HELP), [AtLeast(STARS, 2.0)], REGISTRY)
    measured = {"a": scores("a", {"x": {HELP: 0.5}}, 0.5)}
    a = assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())["a"]
    assert a.violations == (math.inf,) and not a.is_feasible


def test_terms_follow_the_objective():
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(STARS)), [], REGISTRY)
    fit = FakeFit({"a": {HELP: e(0.6), STARS: e(2.0)}})
    a = assess(["a"], fit=fit, criteria=criteria, values={"a": {}}, measured={}, rule=Mean())["a"]
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


def test_rank_skips_ids_without_an_assessment():
    assert rank({"a": assessment("a", 0.5)}, ["b", "a"]) == ["a"]


def test_two_estimates_within_resolution_tie_to_the_earlier_candidate_in_the_search():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    fit = FakeFit({"late": {HELP: e(0.50002, 0.001)}, "early": {HELP: e(0.49999, 0.001)}})
    out = assess(
        ["late", "early"], fit=fit, criteria=criteria, values={"late": {}, "early": {}}, measured={}, rule=Mean()
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
        out = assess(["wide", "steady"], fit=fit, criteria=criteria, values=values, measured={}, rule=rule)
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
        order, fit=FakeFit(table), criteria=criteria, values=dict.fromkeys(order, {}), measured={}, rule=Mean()
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
    out = assess(order, fit=fit, criteria=criteria, values=dict.fromkeys(order, {}), measured={}, rule=Mean())
    assert pareto_front(out, order) == ["cheap", "good"]
