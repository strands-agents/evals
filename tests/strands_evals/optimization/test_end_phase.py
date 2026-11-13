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


# ---- FinalPick: construction and from_config -------------------------------------------------------------


from strands_evals.optimization import (  # noqa: E402
    AtMost,
    Criteria,
    FinalPick,
    LowerBound,
    Maximize,
    Mean,
    MetricRegistry,
    Minimize,
    Pareto,
)
from strands_evals.optimization.end_phase import choose_finalists, pick  # noqa: E402
from strands_evals.types.optimization import CandidateScores  # noqa: E402

HELP = "HelpfulnessEvaluator"
TOKENS = "tokens"
REGISTRY = MetricRegistry({HELP: (0.0, 1.0)})


def test_final_pick_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_finalists"):
        FinalPick(n_finalists=0)
    with pytest.raises(ValueError, match="n_finalists"):
        FinalPick(n_finalists=True)  # a bool isn't an int here
    with pytest.raises(TypeError, match="Mean.*LowerBound"):
        FinalPick(rule="mean")


def test_final_pick_round_trips_through_config():
    for rule in (FinalPick(), FinalPick(n_finalists=5, rule=LowerBound(q=0.2))):
        assert FinalPick.from_config(rule.config()) == rule


def test_final_pick_from_config_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown FinalPick rule 'Median'"):
        FinalPick.from_config({"n_finalists": 3, "rule": {"kind": "Median"}})


# ---- choose_finalists ------------------------------------------------------------------------------------


def test_choose_finalists_takes_the_first_n_excluding_the_baseline():
    assert choose_finalists(["0", "a", "b", "c", "d"], "0", 3) == ["a", "b", "c"]


def test_choose_finalists_returns_whatever_the_ranking_holds_when_short():
    assert choose_finalists(["0", "a"], "0", 3) == ["a"]
    assert choose_finalists(["0"], "0", 3) == []  # no finalist, the baseline ships


def test_choose_finalists_under_pareto_uses_the_front():
    # ranking lists ineligible candidates too, but a Pareto run picks finalists from the front.
    ranking = ["0", "a", "b", "c"]
    front = ["c", "a"]  # the feasible non-dominated, in ranking order
    assert choose_finalists(ranking, "0", 3, front=front) == ["c", "a"]
    assert choose_finalists(ranking, "0", 3, front=[]) == []


# ---- pick ------------------------------------------------------------------------------------------------


def _val_scores(candidate_id, value, *, cases=("v1", "v2"), eligible=True):
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case={c: value for c in cases} if eligible else {},
        metrics_by_case={c: {HELP: value} for c in cases} if eligible else {},
        unavailable={},
        objective_by_run={c: [value] for c in cases},
        metrics_by_run={c: {HELP: [value]} for c in cases} if eligible else {},
        objective=value if eligible else None,
        is_eligible=eligible,
        failures=0,
    )


def test_pick_ships_the_feasible_finalist_with_the_highest_objective():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {
        "0": _val_scores("0", 0.6),
        "a": _val_scores("a", 0.9),
        "b": _val_scores("b", 0.7),
    }
    values = {c: {} for c in validation}
    report = pick(
        baseline_id="0",
        finalist_ids=["a", "b"],
        validation=validation,
        fit=None,
        values=values,
        trial_counts={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "a"
    assert report.objectives["a"] == 0.9 and report.objectives["0"] == 0.6
    assert report.problem is None
    assert "a highest on validation" in report.reason
    assert report.front == []  # not Pareto


def test_pick_ships_the_baseline_when_no_finalist_meets_the_constraint():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.5)], REGISTRY, feasibility_threshold=0.9)
    # baseline measured at 0.5 (feasible on validation: mean at the bound counts as met);
    # the finalist measured at 0.9 (infeasible).
    validation = {"0": _val_scores("0", 0.5), "a": _val_scores("a", 0.9)}
    values = {"0": {}, "a": {}}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values=values,
        trial_counts={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0" and "<= 0.5" in report.infeasible["a"]
    assert "baseline" in report.reason


def test_pick_reports_fewer_than_two_eligible_through_problem():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {"0": _val_scores("0", 0.5), "a": _val_scores("a", 0.0, eligible=False)}
    values = {"0": {}, "a": {}}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values=values,
        trial_counts={"0": 2, "a": 0},
        criteria=criteria,
        rule=Mean(),
        problem="fewer than 2 eligible candidates",
    )
    assert report.champion_id == "0" and report.problem.startswith("fewer than 2")
    assert "not eligible on validation" in report.infeasible["a"]
    # no finalist from the search isn't the message here: a finalist was chosen, it just can't be picked.
    assert "a" not in report.objectives or report.objectives["a"] is None


def test_pick_tie_goes_to_the_candidate_with_the_most_trials_before_the_earlier():
    # Baseline and finalist tie on objective; the finalist has more trials, so it wins though the baseline is first.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {"0": _val_scores("0", 0.8), "a": _val_scores("a", 0.8)}
    values = {"0": {}, "a": {}}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values=values,
        trial_counts={"0": 2, "a": 10},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "a"


def test_pick_tie_without_trial_counts_goes_to_the_baseline():
    # Equal objective and equal trial counts: the baseline wins by appearing first in the ranked ids.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {"0": _val_scores("0", 0.8), "a": _val_scores("a", 0.8)}
    values = {"0": {}, "a": {}}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values=values,
        trial_counts={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0"


def test_pick_under_pareto_a_gain_tie_with_another_trade_off_ships_the_baseline():
    # The baseline (help 0.5, 250 tokens) and "a" (help 0.75, 500 tokens) are both on the front with the same
    # weighted gain, (0.5 - 0.25) / 2 == (0.75 - 0.5) / 2: at the final pick the tie goes to the baseline.
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(TOKENS, scale=1000)), [], REGISTRY)

    def scores(candidate_id, help_value, tokens):
        cases = ("v1", "v2")
        return CandidateScores(
            candidate_id=candidate_id,
            objective_by_case={c: 0.25 for c in cases},
            metrics_by_case={c: {HELP: help_value, TOKENS: tokens} for c in cases},
            unavailable={},
            objective_by_run={c: [0.25] for c in cases},
            metrics_by_run={c: {HELP: [help_value], TOKENS: [tokens]} for c in cases},
            objective=0.25,
            is_eligible=True,
            failures=0,
        )

    validation = {"0": scores("0", 0.5, 250.0), "a": scores("a", 0.75, 500.0)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.objectives["0"] == report.objectives["a"]
    assert set(report.front) == {"0", "a"}
    assert report.champion_id == "0"


def test_pick_with_no_finalist_ships_the_baseline():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {"0": _val_scores("0", 0.5)}
    report = pick(
        baseline_id="0",
        finalist_ids=[],
        validation=validation,
        fit=None,
        values={"0": {}},
        trial_counts={"0": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0"
    assert report.reason.startswith("no finalist from the search")
