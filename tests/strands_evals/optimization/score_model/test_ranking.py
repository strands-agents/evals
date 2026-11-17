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
    OneOf,
    Pareto,
    WeightedSum,
    p_met,
)
from strands_evals.optimization.declaration.space import Space
from strands_evals.optimization.score_model.bayesbench import BayesBenchModel
from strands_evals.optimization.score_model.protocols import FitData
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


# ---- relative constraints -----------------------------------------------------------------------------------


def _relative_scores():
    """The baseline and two candidates measured on one case: Helpfulness and tokens."""
    return {
        "base": scores("base", {"x": {HELP: 0.8, "tokens": 1000.0}}, 0.8),
        "a": scores("a", {"x": {HELP: 0.76, "tokens": 850.0}}, 0.76),
        "b": scores("b", {"x": {HELP: 0.7, "tokens": 500.0}}, 0.7),
    }


def test_measured_relative_constraints_compare_with_the_baselines_scores():
    criteria = Criteria(
        Maximize(HELP), [AtLeast(HELP, -0.05, compare="difference"), AtMost("tokens", 0.9, compare="ratio")], REGISTRY
    )
    measured = _relative_scores()
    out = assess(
        ["base", "a", "b"],
        fit=None,
        criteria=criteria,
        values=dict.fromkeys(measured, {}),
        measured=measured,
        rule=Mean(),
        baseline="base",
    )
    # The baseline is 0 below itself and at a ratio of 1, so over "at most 90% of the baseline's tokens".
    assert out["base"].violations == (-0.05, pytest.approx(0.1)) and not out["base"].is_feasible
    assert out["a"].violations == (pytest.approx(-0.01), pytest.approx(-0.05)) and out["a"].is_feasible
    assert out["b"].violations == (pytest.approx(0.05), pytest.approx(-0.4)) and not out["b"].is_feasible
    label = label_candidate(measured["b"], out["b"], criteria=criteria, baseline=measured["base"])
    assert label.unmet_constraints == [f"{HELP} - baseline >= -0.05: difference -0.1 violates by 0.05"]


def test_a_relative_constraint_needs_the_baselines_id():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, -0.05, compare="difference")], REGISTRY)
    measured = _relative_scores()
    with pytest.raises(ValueError, match="compares with the baseline"):
        assess(["a"], fit=None, criteria=criteria, values={"a": {}}, measured=measured, rule=Mean())


def _tokens_fit():
    """A fit of tokens over three options: a (the baseline) 1000, b 700, c 1300, with noise across cases and runs.

    With this seed the model's own ratio of the baseline to itself has an sd of about 1e-10, not 0.
    """
    rng = random.Random(3)
    cases = [f"case{k}" for k in range(8)]
    means = {"a": 1000.0, "b": 700.0, "c": 1300.0}
    measured = {}
    for option, mean in means.items():
        runs = {case: {"tokens": [mean + rng.gauss(0, 30) for _ in range(2)]} for case in cases}
        measured[option] = CandidateScores(
            candidate_id=option,
            objective_by_case={},
            metrics_by_case={case: {"tokens": sum(r["tokens"]) / 2} for case, r in runs.items()},
            unavailable={},
            objective_by_run={},
            metrics_by_run=runs,
            objective=-mean,
            is_eligible=True,
            failures=0,
        )
    values = {option: {"option": option} for option in means}
    space = Space({"option": OneOf("a", "b", "c", default="a")})
    data = FitData(
        space=space, text_blocks={}, candidates=values, cases=cases, scores=measured, ranges={"tokens": (0.0, None)}
    )
    return BayesBenchModel().fit(data), values, measured


def test_a_fit_judges_a_relative_constraint_on_the_paired_estimate_and_the_baseline_exactly():
    fit, values, measured = _tokens_fit()
    criteria = Criteria(
        Minimize("tokens"),
        [
            AtMost("tokens", 1.0, compare="ratio"),
            AtMost("tokens", 0, compare="difference"),
            AtLeast("tokens", 0.5, compare="ratio"),
        ],
        REGISTRY,
    )
    out = assess(
        ["a", "b", "c"], fit=fit, criteria=criteria, values=values, measured=measured, rule=Mean(), baseline="a"
    )
    assert all(a.is_from_fit for a in out.values())
    # "No more tokens than the baseline", with the baseline at the bound: exactly met.
    assert out["a"].violations == (0.0, 0.0, -0.5) and out["a"].is_feasible
    assert out["a"].p_met == {
        "tokens / baseline <= 1": 1.0,
        "tokens - baseline <= 0": 1.0,
        "tokens / baseline >= 0.5": 1.0,
    }
    assert out["b"].is_feasible and out["b"].p_met["tokens / baseline <= 1"] > 0.99
    assert not out["c"].is_feasible and out["c"].p_met["tokens - baseline <= 0"] < 0.01
    ratio = fit.metrics["tokens"].ratio(values["b"], values["a"], candidate="b", reference_candidate="a")
    assert ratio.mean == pytest.approx(0.7, abs=0.05)
    # AtMost at the feasibility quantile, AtLeast at 1 minus it.
    assert out["b"].violations[0] == pytest.approx(ratio.quantile(0.9) - 1.0)
    assert out["b"].violations[2] == pytest.approx(0.5 - ratio.quantile(0.1))
    label = label_candidate(measured["c"], out["c"], criteria=criteria, baseline=measured["a"])
    assert label.unmet_constraints[0].startswith("tokens / baseline <= 1: P(met) ")


def test_with_a_relative_constraint_a_fit_without_the_baseline_leaves_it_to_measured_means():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, -0.05, compare="difference")], REGISTRY)
    measured = _relative_scores()
    fit = FakeFit({"a": {HELP: e(0.76)}, "b": {HELP: e(0.7)}})  # the baseline was left out of it
    out = assess(
        ["a", "b"],
        fit=fit,
        criteria=criteria,
        values=dict.fromkeys(measured, {}),
        measured=measured,
        rule=Mean(),
        baseline="base",
    )
    assert not out["a"].is_from_fit and out["a"].is_feasible
    assert not out["b"].is_from_fit and out["b"].violations == (pytest.approx(0.05),)
    # Without relative constraints the same fit is used.
    absolute = Criteria(Maximize(HELP), [AtLeast(HELP, 0.75)], REGISTRY)
    assert assess(["a"], fit=fit, criteria=absolute, values={"a": {}}, measured=measured, rule=Mean())["a"].is_from_fit


def test_a_label_with_a_relative_constraint_needs_the_baseline():
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, -0.05, compare="difference")], REGISTRY)
    a = _label_assessment("a", objective=0.9, feasible=True, n_constraints=1)
    with pytest.raises(ValueError, match="compares with the baseline"):
        label_candidate(_label_scores("a", {"c1": 0.9}), a, criteria=criteria)


def test_a_fit_that_puts_the_baselines_mean_at_0_leaves_a_ratio_to_measured_means():
    criteria = Criteria(Minimize("tokens"), [AtMost("tokens", 0.8, compare="ratio")], REGISTRY)
    fit = FakeFit({"base": {"tokens": e(0.0)}, "a": {"tokens": e(500.0)}})
    measured = {
        "base": scores("base", {"x": {"tokens": 1000.0}}, -1000.0),
        "a": scores("a", {"x": {"tokens": 700.0}}, -700.0),
    }
    out = assess(
        ["base", "a"],
        fit=fit,
        criteria=criteria,
        values=dict.fromkeys(measured, {}),
        measured=measured,
        rule=Mean(),
        baseline="base",
    )
    assert not out["a"].is_from_fit and out["a"].violations == (pytest.approx(-0.1),) and out["a"].is_feasible
    assert not out["base"].is_from_fit and out["base"].violations == (pytest.approx(0.2),)


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
    out = assess(order, fit=fit, criteria=criteria, values=dict.fromkeys(order, {}), measured={}, rule=Mean())
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
    assert any("<= 0.5" in sentence for sentence in label.unmet_constraints)


def test_label_candidate_ineligible_when_a_train_case_has_no_counted_run():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    candidate = _label_scores("a", {"c1": 0.9, "c2": None}, eligible=False)
    label = label_candidate(candidate, None, criteria=criteria)
    assert label.outcome == "ineligible" and "c2" in label.reason
    assert label.estimates == {} and label.p_met == {} and label.unmet_constraints == []
