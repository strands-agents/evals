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
    AtLeast,
    AtMost,
    Criteria,
    FinalPick,
    LowerBound,
    Maximize,
    Mean,
    MetricRegistry,
    Minimize,
    Pareto,
    p_met,
)
from strands_evals.optimization.end_phase import choose_finalists, pick  # noqa: E402
from strands_evals.optimization.score_model.ranking import assess, pareto_front, rank  # noqa: E402
from strands_evals.types.optimization import CandidateScores, Estimate  # noqa: E402

HELP = "HelpfulnessEvaluator"
TOKENS = "tokens"
REGISTRY = MetricRegistry({HELP: (0.0, 1.0)})


def test_final_pick_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_finalists"):
        FinalPick(n_finalists=0)
    with pytest.raises(TypeError, match="n_finalists"):
        FinalPick(n_finalists=True)  # a bool isn't an int here
    with pytest.raises(TypeError, match="n_finalists"):
        FinalPick(n_finalists="3")
    with pytest.raises(TypeError, match="Mean.*LowerBound"):
        FinalPick(rule="mean")


def test_final_pick_round_trips_through_config():
    for rule in (FinalPick(), FinalPick(n_finalists=5, rule=LowerBound(q=0.2))):
        assert FinalPick.from_config(rule.config()) == rule


def test_final_pick_from_config_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown FinalPick rule 'Median'"):
        FinalPick.from_config({"n_finalists": 3, "rule": {"kind": "Median"}})


# ---- choose_finalists ------------------------------------------------------------------------------------


class _FixedFit:
    """A fit with fixed estimates per candidate."""

    flagged: list[str] = []
    diagnostics: dict = {}

    def __init__(self, table):
        self.table = table

    @property
    def candidates(self):
        return frozenset(self.table)

    def estimates(self, values, candidate=None):
        return dict(self.table[candidate])


def _train_scores(candidate_id, metrics_by_case, criteria):
    """Train scores with one run per case; `metrics_by_case` is case -> metric -> value."""
    point = {
        metric: sum(values) / len(values)
        for metric in criteria.goal_metrics
        if (values := [case[metric] for case in metrics_by_case.values() if metric in case])
    }
    objective = criteria.objective_value(point)
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case={c: objective for c in metrics_by_case},
        metrics_by_case=metrics_by_case,
        unavailable={},
        objective_by_run={c: [objective] for c in metrics_by_case},
        metrics_by_run={c: {m: [v] for m, v in case.items()} for c, case in metrics_by_case.items()},
        objective=objective,
        is_eligible=True,
        failures=0,
    )


def _choose(train, criteria, n, *, fit=None, n_validation_cases=6):
    """`choose_finalists` on the search ranking of `train` (scoring order), "0" the baseline."""
    assessments = assess(
        list(train), fit=fit, criteria=criteria, values={c: {} for c in train}, measured=train, rule=Mean()
    )
    ranking = rank(assessments, list(train), is_pareto=criteria.is_pareto)
    front = pareto_front(assessments, ranking) if criteria.is_pareto else None
    finalists = choose_finalists(
        ranking,
        "0",
        n,
        assessments=assessments,
        train=train,
        criteria=criteria,
        n_validation_cases=n_validation_cases,
        front=front,
    )
    return finalists, ranking, front


def _floor_train(criteria, help_by_candidate, tokens_by_candidate):
    """Candidate -> train scores: HELP per case as given, the same tokens on every case."""
    return {
        c: _train_scores(c, {f"t{i}": {HELP: h, TOKENS: tokens_by_candidate[c]} for i, h in enumerate(helps)}, criteria)
        for c, helps in help_by_candidate.items()
    }


FLOOR = Criteria(Minimize(TOKENS), [AtLeast(HELP, 0.5)], REGISTRY, feasibility_threshold=0.9)
# "a" is the cheapest above the floor, but its cases are spread (4 of 6 at 1, 2 at 0), so its mean of 0.67 could
# land below 0.5 on 6 new cases. "b" clears it on every case; "c" sits on it with no spread at all. The baseline
# misses it.
FLOOR_HELP = {
    "0": [0.2] * 6,
    "a": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    "b": [0.9, 0.8, 0.9, 0.8, 0.9, 0.8],
    "c": [0.5] * 6,
}
FLOOR_TOKENS = {"0": 100.0, "a": 200.0, "b": 300.0, "c": 400.0}


def test_choose_finalists_takes_the_first_n_excluding_the_baseline():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    helps = {"0": 0.95, "a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    train = {c: _train_scores(c, {"t0": {HELP: v}, "t1": {HELP: v}}, criteria) for c, v in helps.items()}
    finalists, ranking, _ = _choose(train, criteria, 3)
    assert ranking == ["0", "a", "b", "c", "d"]
    assert finalists == ["a", "b", "c"]  # without constraints, exactly the ranking's first n


def test_choose_finalists_returns_whatever_the_ranking_holds_when_short():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    train = {c: _train_scores(c, {"t0": {HELP: v}, "t1": {HELP: v}}, criteria) for c, v in (("0", 0.5), ("a", 0.9))}
    assert _choose(train, criteria, 3)[0] == ["a"]
    assert _choose({"0": train["0"]}, criteria, 3)[0] == []  # no finalist, the baseline ships


def test_choose_finalists_puts_candidates_meeting_a_floor_with_margin_first():
    train = _floor_train(FLOOR, FLOOR_HELP, FLOOR_TOKENS)
    finalists, ranking, _ = _choose(train, FLOOR, 2)
    assert ranking == ["a", "b", "c", "0"]  # the search ranking is unchanged: feasible first, cheapest first
    # 0.67 - 1.28 * stdev 0.52 * sqrt(1/6 + 1/6) = 0.28 < 0.5: "a" waits behind "b" and "c" (no spread, so
    # judged at its mean, which meets the floor exactly).
    assert finalists == ["b", "c"]
    assert _choose(train, FLOOR, 3)[0] == ["b", "c", "a"]  # then the rest in ranking order


def test_choose_finalists_margin_widens_with_fewer_validation_cases():
    # "b" has a stdev of 0.27 over 6 train cases: 0.75 - 1.28 * 0.27 * sqrt(1/6 + 1/n) >= 0.5 for n = 24, not n = 2.
    help_by = {"0": [0.2] * 6, "a": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0], "b": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]}
    train = _floor_train(FLOOR, help_by, FLOOR_TOKENS)
    assert _choose(train, FLOOR, 1, n_validation_cases=24)[0] == ["b"]
    assert _choose(train, FLOOR, 1, n_validation_cases=2)[0] == ["a"]  # no candidate has margin: ranking order


def test_choose_finalists_judges_the_margin_at_the_criteria_s_feasibility_threshold():
    # At 0.6, z is 0.25: "a" holds, 0.67 - 0.25 * 0.52 * sqrt(1/6 + 1/24) = 0.61 >= 0.5, and ranks first.
    help_by = {"0": [0.2] * 6, "a": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0], "b": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]}
    lenient = Criteria(Minimize(TOKENS), [AtLeast(HELP, 0.5)], REGISTRY, feasibility_threshold=0.6)
    train = _floor_train(lenient, help_by, FLOOR_TOKENS)
    assert _choose(train, lenient, 1, n_validation_cases=24)[0] == ["a"]


def test_choose_finalists_under_a_ceiling_puts_candidates_with_margin_first():
    criteria = Criteria(Maximize(HELP), [AtMost(TOKENS, 1000.0)], REGISTRY)
    tokens = {"0": [1500.0] * 6, "a": [400.0, 1600.0, 400.0, 1600.0, 900.0, 900.0], "b": [500.0] * 6}
    helps = {"0": 0.5, "a": 0.9, "b": 0.8}
    train = {
        c: _train_scores(c, {f"t{i}": {HELP: helps[c], TOKENS: t} for i, t in enumerate(per_case)}, criteria)
        for c, per_case in tokens.items()
    }
    finalists, ranking, _ = _choose(train, criteria, 1)
    assert ranking[0] == "a"  # mean 967 tokens, under the ceiling on train
    assert finalists == ["b"]


@pytest.mark.parametrize(
    "constraints", [[AtLeast(HELP, 0.5), AtMost(TOKENS, 1000.0)], [AtMost(TOKENS, 1000.0), AtLeast(HELP, 0.5)]]
)
def test_choose_finalists_needs_margin_on_every_constraint(constraints):
    # "a" clears the floor on every case, but its tokens spread around the ceiling; "b" is solid on both.
    criteria = Criteria(Maximize(HELP), constraints, REGISTRY)
    per_case = {
        "0": ([0.2] * 6, [1500.0] * 6),
        "a": ([1.0, 0.9] * 3, [400.0, 1600.0, 400.0, 1600.0, 900.0, 900.0]),
        "b": ([0.9, 0.8] * 3, [500.0] * 6),
    }
    train = {
        c: _train_scores(
            c, {f"t{i}": {HELP: h, TOKENS: t} for i, (h, t) in enumerate(zip(*cases, strict=True))}, criteria
        )
        for c, cases in per_case.items()
    }
    finalists, ranking, _ = _choose(train, criteria, 1)
    assert ranking[0] == "a"
    assert finalists == ["b"]


def test_choose_finalists_counts_a_value_at_the_bound_up_to_float_noise_as_met():
    # 0.1 + 0.2 is 0.30000000000000004, above the 0.3 ceiling only by float noise; "b" ranks first but is spread.
    criteria = Criteria(Minimize(TOKENS), [AtMost(HELP, 0.3)], REGISTRY)
    help_by = {"0": [0.9] * 6, "a": [0.1 + 0.2] * 6, "b": [0.0, 0.6, 0.0, 0.6, 0.1, 0.1]}
    train = _floor_train(criteria, help_by, {"0": 50.0, "a": 200.0, "b": 100.0})
    finalists, ranking, _ = _choose(train, criteria, 1)
    assert ranking[:2] == ["b", "a"]
    assert finalists == ["a"]


def test_choose_finalists_judges_the_fit_s_mean_with_the_per_case_spread():
    # Measured, "a" clears the floor by far (0.85); the fit's mean of 0.56 meets it at P(met) >= 0.9 with its own
    # sd, but not once the spread of a new set of cases is allowed for. "b" clears it under both.
    train = _floor_train(
        FLOOR,
        {"0": [0.2] * 6, "a": [1.0, 0.7, 1.0, 0.7, 1.0, 0.7], "b": [0.9, 0.8, 0.9, 0.8, 0.9, 0.8]},
        FLOOR_TOKENS,
    )
    fit = _FixedFit(
        {
            "0": {TOKENS: Estimate(mean=100.0, sd=1.0), HELP: Estimate(mean=0.2, sd=0.01)},
            "a": {TOKENS: Estimate(mean=200.0, sd=1.0), HELP: Estimate(mean=0.56, sd=0.01)},
            "b": {TOKENS: Estimate(mean=300.0, sd=1.0), HELP: Estimate(mean=0.85, sd=0.01)},
        }
    )
    finalists, ranking, _ = _choose(train, FLOOR, 1, fit=fit)
    assert ranking == ["a", "b", "0"]  # both feasible under the fit
    assert finalists == ["b"]
    # On the measured mean "a" would hold: 0.85 - 1.28 * 0.16 * sqrt(1/6 + 1/6) = 0.73 >= 0.5.
    assert _choose(train, FLOOR, 1)[0] == ["a"]


def test_choose_finalists_a_constraint_measured_on_one_case_shows_no_margin():
    # "a" is measured on t0 only (N/A elsewhere): one value has no spread to judge by.
    train = _floor_train(FLOOR, {"0": [0.2] * 6, "b": FLOOR_HELP["b"]}, FLOOR_TOKENS)
    train["a"] = _train_scores(
        "a", {"t0": {HELP: 1.0, TOKENS: 200.0}, **{f"t{i}": {TOKENS: 200.0} for i in range(1, 6)}}, FLOOR
    )
    finalists, ranking, _ = _choose(train, FLOOR, 2)
    assert ranking == ["a", "b", "0"]
    assert finalists == ["b", "a"]


def test_choose_finalists_expects_the_train_measurement_rate_on_validation():
    # "a" has HELP on 3 of 6 train cases (N/A elsewhere), so validation should be expected to measure it on
    # about half its cases too: 3 of 6, not 6 of 6.
    train = _floor_train(FLOOR, {"0": [0.2] * 6, "b": FLOOR_HELP["b"]}, FLOOR_TOKENS)
    helps = [0.5, 0.8, 1.1]
    train["a"] = _train_scores(
        "a",
        {f"t{i}": ({HELP: helps[i]} if i < 3 else {}) | {TOKENS: 200.0} for i in range(6)},
        FLOOR,
    )
    finalists, ranking, _ = _choose(train, FLOOR, 2)
    assert ranking == ["a", "b", "0"]
    # 0.8 - 1.28 * stdev 0.3 * sqrt(1/3 + 1/3) = 0.49 < 0.5: no margin. Dividing the validation term by all
    # 6 cases would give 0.8 - 1.28 * 0.3 * sqrt(1/3 + 1/6) = 0.53 and wrongly show margin.
    assert finalists == ["b", "a"]
    # more validation cases shrink the term at the train rate: 30 cases -> 15 expected measured
    assert _choose(train, FLOOR, 2, n_validation_cases=30)[0] == ["a", "b"]


def test_choose_finalists_judges_the_margin_on_the_adjusted_mean_without_a_fit():
    # "b" leaves the evaluator nothing to judge on the hard cases t3-t5, where the others drop 0.2. On its judged cases
    # alone (mean 0.95) it would clear the floor with margin and go first; with those cases filled in it is judged at
    # 0.85, as "a" is, and neither has margin, so the ranking's order (the cheaper "a" first) stands.
    criteria = Criteria(Minimize(TOKENS), [AtLeast(HELP, 0.82)], REGISTRY, feasibility_threshold=0.9)
    train = {
        "0": _train_scores("0", {f"t{i}": {HELP: 0.4 if i < 3 else 0.2, TOKENS: 50.0} for i in range(6)}, criteria),
        "a": _train_scores("a", {f"t{i}": {HELP: 0.95 if i < 3 else 0.75, TOKENS: 100.0} for i in range(6)}, criteria),
        "b": _train_scores(
            "b",
            {
                f"t{i}": {HELP: h, TOKENS: 200.0} if h is not None else {TOKENS: 200.0}
                for i, h in enumerate([0.9, 1.0, 0.95, None, None, None])
            },
            criteria,
        ),
    }
    finalists, ranking, _ = _choose(train, criteria, 1)
    assert ranking == ["a", "b", "0"]
    assert finalists == ["a"]


def test_choose_finalists_judges_the_margin_spread_over_the_filled_cases_too():
    # "c" equals the others on the easy cases and is N/A on "hard". On its judged cases alone it shows no spread, so
    # it would hold the floor with margin and take the one slot from "b", ranked first and judged everywhere. With
    # "hard" filled in (0.625) it spreads as "b" does, neither has margin, and the ranking's order stands.
    criteria = Criteria(Maximize(HELP), [AtLeast(HELP, 0.72)], REGISTRY, feasibility_threshold=0.9)
    cases = {
        "0": {"e1": 1.0, "e2": 1.0, "hard": 0.6},
        "b": {"e1": 1.0, "e2": 1.0, "hard": 0.65},
        "c": {"e1": 1.0, "e2": 1.0, "hard": None},
    }
    train = {
        c: _train_scores(c, {k: {HELP: v} if v is not None else {} for k, v in by_case.items()}, criteria)
        for c, by_case in cases.items()
    }
    finalists, ranking, _ = _choose(train, criteria, 1)
    assert ranking == ["b", "c", "0"]
    assert finalists == ["b"]


def test_choose_finalists_with_one_train_case_keeps_the_ranking_order():
    train = {
        c: _train_scores(c, {"t0": {HELP: helps[0], TOKENS: FLOOR_TOKENS[c]}}, FLOOR) for c, helps in FLOOR_HELP.items()
    }
    finalists, ranking, _ = _choose(train, FLOOR, 2)
    assert finalists == ranking[:2] == ["a", "b"]


def test_choose_finalists_never_puts_an_infeasible_candidate_first():
    # "b" misses the floor under the fit (0.55 with sd 0.1: its 10% quantile is 0.42), though its train cases have
    # no spread, so judged with their spread alone it would clear the floor.
    train = _floor_train(FLOOR, {"0": [0.2] * 6, "a": FLOOR_HELP["a"], "b": [0.55] * 6}, FLOOR_TOKENS)
    fit = _FixedFit(
        {
            "0": {TOKENS: Estimate(mean=100.0, sd=1.0), HELP: Estimate(mean=0.2, sd=0.01)},
            "a": {TOKENS: Estimate(mean=200.0, sd=1.0), HELP: Estimate(mean=0.67, sd=0.01)},
            "b": {TOKENS: Estimate(mean=300.0, sd=1.0), HELP: Estimate(mean=0.55, sd=0.1)},
        }
    )
    finalists, ranking, _ = _choose(train, FLOOR, 2, fit=fit)
    assert ranking == ["a", "b", "0"]
    assert finalists == ["a", "b"]  # neither has margin ("a" is too spread out, "b" infeasible): ranking order


def test_choose_finalists_under_pareto_puts_front_members_with_margin_first():
    criteria = Criteria(Pareto(Minimize(TOKENS, scale=100), Maximize(HELP)), [AtLeast(HELP, 0.5)], REGISTRY)
    train = _floor_train(criteria, FLOOR_HELP, FLOOR_TOKENS)
    finalists, ranking, front = _choose(train, criteria, 3)
    assert ranking == ["a", "b", "c", "0"] and front == ["a", "b"]  # "c" is dominated by "b"
    assert finalists == ["b", "a"]


def test_choose_finalists_under_pareto_with_an_empty_front_gives_none():
    criteria = Criteria(Pareto(Minimize(TOKENS, scale=1000), Maximize(HELP)), [AtLeast(HELP, 0.9)], REGISTRY)
    train = _floor_train(criteria, FLOOR_HELP, FLOOR_TOKENS)
    finalists, ranking, front = _choose(train, criteria, 3)
    assert ranking and front == [] and finalists == []  # nothing feasible on train, so the baseline ships


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
        completed={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "a"
    assert report.objectives["a"] == 0.9 and report.objectives["0"] == 0.6
    assert report.problem is None
    assert "a highest on validation" in report.reason
    assert report.outcome == "top_finalist" and report.rule == {"kind": "Mean"}
    assert report.p_met == {"0": {}, "a": {}, "b": {}}  # assessed by measured means
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
        completed={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0" and "<= 0.5" in report.infeasible["a"]
    assert "baseline" in report.reason and report.outcome == "no_feasible_finalist"


class _PredictingFit:
    """A validation fit predicting every metric at `estimate` for every candidate."""

    flagged: list[str] = []
    diagnostics: dict = {}

    def __init__(self, ids, metrics, estimate):
        self.candidates = frozenset(ids)
        self.metrics = dict.fromkeys(metrics)
        self._estimate = estimate

    def estimates(self, values, candidate=None):
        return dict.fromkeys(self.metrics, self._estimate)


def test_pick_rejects_a_finalist_whose_constraint_was_never_measured_on_validation():
    grounded = "Grounded"
    criteria = Criteria(
        Maximize(HELP), [AtLeast(grounded, 0.5)], MetricRegistry({HELP: (0.0, 1.0), grounded: (0.0, 1.0)})
    )
    baseline = _val_scores("0", 0.5)
    baseline = baseline.model_copy(
        update={"metrics_by_case": {c: {**m, grounded: 1.0} for c, m in baseline.metrics_by_case.items()}}
    )
    validation = {"0": baseline, "a": _val_scores("a", 0.9)}  # Grounded N/A on every case of "a"
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=_PredictingFit(validation, [HELP, grounded], Estimate(mean=0.9, sd=0.01)),
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        completed={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0"
    assert report.infeasible == {"a": "Grounded >= 0.5: never measured"}


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
        completed={"0": 2, "a": 0},
        criteria=criteria,
        rule=Mean(),
        problem="fewer than 2 eligible candidates",
    )
    assert report.champion_id == "0" and report.problem.startswith("fewer than 2")
    assert "not eligible on validation" in report.infeasible["a"] and report.outcome == "no_feasible_finalist"
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
        completed={"0": 2, "a": 10},
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
        completed={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0" and report.outcome == "baseline_best"


@pytest.mark.parametrize("twins", [0, 1, 2])
def test_pick_by_measured_means_fills_a_finalist_s_n_a_case_and_the_baseline_wins_the_tie(twins):
    # Finalist "a" matches the baseline on v1 and leaves the evaluator nothing to judge on v2, where the baseline
    # (and each twin, a finalist equal to it) scored 0.1; on plain means "a" would ship (0.7 against 0.4). Filled
    # in, it ties the baseline exactly, whatever the number of twins on the case.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)

    def scores(candidate_id, help_by_case):
        judged = {c: v for c, v in help_by_case.items() if v is not None}
        return CandidateScores(
            candidate_id=candidate_id,
            objective_by_case=judged,
            metrics_by_case={c: {HELP: v} for c, v in judged.items()},
            unavailable={},
            objective_by_run={c: [v] for c, v in help_by_case.items()},
            metrics_by_run={c: {HELP: [v]} for c, v in help_by_case.items()},
            objective=sum(judged.values()) / len(judged),
            is_eligible=True,
            failures=0,
        )

    validation = {
        "0": scores("0", {"v1": 0.7, "v2": 0.1}),
        "a": scores("a", {"v1": 0.7, "v2": None}),
        **{f"t{i}": scores(f"t{i}", {"v1": 0.7, "v2": 0.1}) for i in range(twins)},
    }
    report = pick(
        baseline_id="0",
        finalist_ids=[c for c in validation if c != "0"],
        validation=validation,
        fit=None,
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        completed={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.objectives["a"] == report.objectives["0"]
    assert report.champion_id == "0"
    assert report.adjusted_means["a"] == {HELP: report.objectives["0"]} == report.adjusted_means["0"]


def test_pick_by_measured_means_ships_the_baseline_over_a_finalist_below_it_that_shares_a_gap():
    # "worse" is below the baseline on "hard", the only case both were judged on, and "weak" shares its gap on
    # "easy": "weak" must not raise "worse"'s fill there.
    criteria = Criteria(Maximize(HELP), [], REGISTRY)

    def scores(candidate_id, help_by_case):
        return _train_scores(candidate_id, {c: {HELP: v} for c, v in help_by_case.items()}, criteria)

    validation = {
        "0": scores("0", {"easy": 0.7, "hard": 0.5}),
        "worse": scores("worse", {"hard": 0.45}),
        "weak": scores("weak", {"hard": 0.1}),
    }
    report = pick(
        baseline_id="0",
        finalist_ids=["worse", "weak"],
        validation=validation,
        fit=None,
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        completed={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0"
    assert report.objectives["worse"] == pytest.approx(0.55) and report.objectives["0"] == pytest.approx(0.6)


def _pick_measured(criteria, pool, tokens=None):
    """`pick(fit=None)` on validation scores `pool` (candidate -> case -> HELP, N/A left out), "0" the baseline."""
    cases = list(dict.fromkeys(k for by in pool.values() for k in by))
    validation = {
        c: _train_scores(
            c,
            {k: ({HELP: by[k]} if k in by else {}) | ({TOKENS: tokens[c]} if tokens else {}) for k in cases},
            criteria,
        )
        for c, by in pool.items()
    }
    return pick(
        baseline_id="0",
        finalist_ids=[c for c in pool if c != "0"],
        validation=validation,
        fit=None,
        values={c: {} for c in pool},
        trial_counts={c: 1 for c in pool},
        completed={c: 1 for c in pool},
        criteria=criteria,
        rule=Mean(),
    )


def test_pick_weighs_each_fill_by_the_cases_its_difference_rests_on():
    # "w" equals the baseline on a, is below it on c, and is N/A on b. "x", weak and judged on b, shares only a with
    # "w"; counted as much as the baseline, it filled b at 0.625, above both scores measured there, and "w" shipped.
    report = _pick_measured(
        Criteria(Maximize(HELP), [], REGISTRY),
        {"0": {"a": 0.7, "b": 0.5, "c": 0.5}, "w": {"a": 0.7, "c": 0.4}, "x": {"a": 0.1, "b": 0.2}},
    )
    assert report.champion_id == "0"


def test_pick_fills_unshifted_only_when_no_candidate_shares_a_case():
    # "d" is judged only on "w"'s gap (and is over the token cap, so it can't ship). The baseline shares "hard" with
    # "w", so it alone fills "easy"; "d"'s unshifted 1.0 used to go in beside it, and "w" shipped.
    report = _pick_measured(
        Criteria(Maximize(HELP), [AtMost(TOKENS, 1000.0)], REGISTRY),
        {"0": {"easy": 0.7, "hard": 0.5}, "w": {"hard": 0.42}, "d": {"easy": 1.0}},
        tokens={"0": 500.0, "w": 500.0, "d": 2000.0},
    )
    assert report.champion_id == "0"


def test_pick_float_noise_in_a_fill_below_the_floor_does_not_beat_a_tie():
    # Under Minimize at a floor of 0, "w" matches the baseline where judged; its fill on c averages 0.0 with
    # 0.3 + (-0.30000000000000004), a hair below 0. Nothing measured on c is below 0, so it is clipped and the
    # baseline wins the tie.
    report = _pick_measured(
        Criteria(Minimize(HELP), [], REGISTRY),
        {"0": {"a": 0.0, "b": 0.0, "c": 0.0}, "w": {"a": 0.0, "b": 0.0}, "x": {"a": 0.4, "b": 0.2, "c": 0.3}},
    )
    assert report.champion_id == "0"
    assert report.adjusted_means["w"] == {HELP: 0.0}


def test_pick_reports_adjusted_means_for_every_assessed_candidate():
    criteria = Criteria(Maximize(HELP), [], REGISTRY)
    validation = {"0": _val_scores("0", 0.6), "a": _val_scores("a", 0.9), "b": _val_scores("b", 0.7)}
    fit = _FixedFit({"0": {HELP: Estimate(mean=0.6, sd=0.01)}, "a": {HELP: Estimate(mean=0.9, sd=0.01)}})
    report = pick(
        baseline_id="0",
        finalist_ids=["a", "b"],
        validation=validation,
        fit=fit,
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        completed={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    # "b" isn't in the fit, so it is assessed by measured means; the others from the fit.
    assert report.adjusted_means == {"0": {}, "a": {}, "b": {HELP: 0.7}}


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
        completed={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.objectives["0"] == report.objectives["a"]
    assert set(report.front) == {"0", "a"}
    assert report.champion_id == "0" and report.outcome == "baseline_best"


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
        completed={"0": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0" and report.outcome == "no_finalists"
    assert report.reason.startswith("no finalist from the search")


def test_pick_never_ships_a_finalist_with_no_completed_validation_trial():
    # "a" crashed on every validation trial; its failure score (agent_failure_score=1.0) would rank it first.
    criteria = Criteria(Maximize(HELP), [], REGISTRY, agent_failure_score=1.0)
    validation = {"0": _val_scores("0", 0.5), "a": _val_scores("a", 1.0), "b": _val_scores("b", 0.7)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a", "b"],
        validation=validation,
        fit=None,
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        completed={"0": 2, "a": 0, "b": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "b"
    assert report.infeasible == {"a": "every validation trial failed, so nothing was measured"}
    assert report.objectives["a"] is None
    assert "a" in report.measured  # what it did measure (its usage) is still reported


def test_pick_ships_the_baseline_when_the_only_finalist_completed_no_validation_trial():
    criteria = Criteria(Maximize(HELP), [], REGISTRY, agent_failure_score=1.0)
    validation = {"0": _val_scores("0", 0.5), "a": _val_scores("a", 1.0)}
    kwargs = dict(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    report = pick(**kwargs, completed={"0": 2, "a": 0})
    assert report.champion_id == "0" and "a" in report.infeasible
    assert pick(**kwargs, completed={"0": 2}) == report  # a candidate left out counts as 0 completed


def test_pick_falls_back_to_a_baseline_with_no_completed_validation_trial_when_nothing_is_feasible():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.5)], REGISTRY, agent_failure_score=0.0)
    validation = {"0": _val_scores("0", 0.0), "a": _val_scores("a", 0.9)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        completed={"0": 0, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "0" and report.objectives["0"] is None
    assert report.infeasible["0"] == "every validation trial failed, so nothing was measured"
    assert "<= 0.5" in report.infeasible["a"]


def test_pick_never_ships_a_baseline_with_no_completed_validation_trial_over_a_measured_finalist():
    criteria = Criteria(Maximize(HELP), [], REGISTRY, agent_failure_score=1.0)
    validation = {"0": _val_scores("0", 1.0), "a": _val_scores("a", 0.7)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation=validation,
        fit=None,
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        completed={"0": 0, "a": 2},
        criteria=criteria,
        rule=Mean(),
    )
    assert report.champion_id == "a" and "0" in report.infeasible
    assert report.reason.endswith("vs baseline unmeasured")


def test_pick_under_pareto_leaves_a_finalist_with_no_completed_validation_trial_off_the_front():
    # "a" crashed at once on every validation case: the worst score but the fewest tokens, so it would be on the
    # front, and its weighted gain the highest.
    criteria = Criteria(Pareto(Maximize(HELP), Minimize(TOKENS, scale=10)), [], REGISTRY)

    def scores(candidate_id, help_value, tokens):
        cases = ("v1", "v2")
        objective = criteria.objective_value({HELP: help_value, TOKENS: tokens})
        return CandidateScores(
            candidate_id=candidate_id,
            objective_by_case={c: objective for c in cases},
            metrics_by_case={c: {HELP: help_value, TOKENS: tokens} for c in cases},
            unavailable={},
            objective_by_run={c: [objective] for c in cases},
            metrics_by_run={c: {HELP: [help_value], TOKENS: [tokens]} for c in cases},
            objective=objective,
            is_eligible=True,
            failures=2 if help_value == 0.0 else 0,
        )

    validation = {"0": scores("0", 0.5, 6.0), "a": scores("a", 0.0, 0.0), "b": scores("b", 0.85, 9.0)}
    kwargs = dict(
        baseline_id="0",
        finalist_ids=["a", "b"],
        validation=validation,
        fit=None,
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    assert pick(**kwargs, completed={c: 2 for c in validation}).champion_id == "a"  # what the check prevents
    report = pick(**kwargs, completed={"0": 2, "a": 0, "b": 2})
    assert report.champion_id == "b" and "a" not in report.front and "a" in report.infeasible


class _TokensFlagged:
    """A validation fit that predicts 500 tokens for everyone, flagged on tokens."""

    flagged = [TOKENS]
    diagnostics: dict = {}
    metrics: dict = {}

    def __init__(self, helpfulness):
        self.helpfulness = helpfulness
        self.candidates = frozenset(helpfulness)

    def estimates(self, values, candidate=None):
        return {HELP: Estimate(mean=self.helpfulness[candidate], sd=0.01), TOKENS: Estimate(mean=500.0, sd=1.0)}


def test_pick_judges_a_flagged_metric_by_its_measured_mean_and_the_others_by_the_fit():
    criteria = Criteria(Maximize(HELP), [AtMost(TOKENS, 1000)], REGISTRY)

    def scores(candidate_id, help_value, tokens):
        cases = ("v1", "v2")
        return CandidateScores(
            candidate_id=candidate_id,
            objective_by_case={c: help_value for c in cases},
            metrics_by_case={c: {HELP: help_value, TOKENS: tokens} for c in cases},
            unavailable={},
            objective_by_run={c: [help_value] for c in cases},
            metrics_by_run={c: {HELP: [help_value], TOKENS: [tokens]} for c in cases},
            objective=help_value,
            is_eligible=True,
            failures=0,
        )

    validation = {"0": scores("0", 0.9, 800.0), "a": scores("a", 0.5, 1200.0), "b": scores("b", 0.6, 900.0)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a", "b"],
        validation=validation,
        fit=_TokensFlagged({"0": 0.6, "a": 0.8, "b": 0.7}),
        values={c: {} for c in validation},
        trial_counts={c: 2 for c in validation},
        completed={c: 2 for c in validation},
        criteria=criteria,
        rule=Mean(),
    )
    # Helpfulness from the fit ("b" beats the baseline); tokens measured, so "a" misses the limit the fit says it meets.
    assert report.champion_id == "b" and report.objectives == {"0": 0.6, "a": 0.8, "b": 0.7}
    assert report.infeasible == {"a": "tokens <= 1000: mean 1200 violates by 200"}
    assert report.estimates["a"] == {HELP: Estimate(mean=0.8, sd=0.01)}
    # Tokens, judged by its adjusted mean, is reported for every candidate, though each is assessed from the fit.
    assert report.adjusted_means == {"0": {TOKENS: 800.0}, "a": {TOKENS: 1200.0}, "b": {TOKENS: 900.0}}


def test_choose_finalists_judges_a_flagged_constraint_metric_by_its_adjusted_mean():
    # The fit (flagged on tokens) ranks "a" first, but its tokens sit at the limit with a spread, so only "b"
    # (always 600) holds with margin. Judging "a" by the fit's 500 tokens would have let it hold too.
    criteria = Criteria(Maximize(HELP), [AtMost(TOKENS, 1000)], REGISTRY, feasibility_threshold=0.9)
    tokens = {"0": [600.0] * 6, "a": [900.0, 1100.0, 900.0, 1100.0, 1000.0, 1000.0], "b": [600.0] * 6}
    train = {
        c: _train_scores(c, {f"t{i}": {HELP: 0.5, TOKENS: t} for i, t in enumerate(ts)}, criteria)
        for c, ts in tokens.items()
    }
    fit = _TokensFlagged({"0": 0.5, "a": 0.8, "b": 0.7})
    finalists, ranking, _ = _choose(train, criteria, 1, fit=fit)
    assert ranking[:2] == ["a", "b"] and finalists == ["b"]


class _HelpFlagged:
    """A train fit that predicts tokens per candidate, flagged on HELP."""

    flagged = [HELP]
    diagnostics: dict = {}
    metrics: dict = {}

    def __init__(self, tokens):
        self.tokens = tokens
        self.candidates = frozenset(tokens)

    def estimates(self, values, candidate=None):
        return {HELP: Estimate(mean=0.9, sd=0.01), TOKENS: Estimate(mean=self.tokens[candidate], sd=1.0)}


def test_choose_finalists_widens_a_flagged_metric_s_spread_over_its_filled_cases():
    # HELP is flagged, so its floor is judged by the adjusted mean while the fit ranks by tokens. The cheapest, "c",
    # equals the others on the easy cases and is N/A on "hard": on its judged cases alone it shows no spread and
    # would take the one slot. With "hard" filled in it spreads, and "b", steady everywhere, holds with margin.
    criteria = Criteria(Minimize(TOKENS), [AtLeast(HELP, 0.85)], REGISTRY, feasibility_threshold=0.9)
    helps = {
        "0": {"e1": 1.0, "e2": 1.0, "hard": 0.6},
        "b": {"e1": 1.0, "e2": 1.0, "hard": 1.0},
        "c": {"e1": 1.0, "e2": 1.0, "hard": None},
    }
    tokens = {"0": 300.0, "b": 200.0, "c": 100.0}
    train = {
        c: _train_scores(
            c, {k: ({HELP: v} if v is not None else {}) | {TOKENS: tokens[c]} for k, v in by_case.items()}, criteria
        )
        for c, by_case in helps.items()
    }
    finalists, ranking, _ = _choose(train, criteria, 1, fit=_HelpFlagged(tokens))
    assert ranking == ["c", "b", "0"]
    assert finalists == ["b"]


class _FixedFit:
    """A validation fit with fixed estimates per candidate."""

    flagged: list[str] = []

    def __init__(self, table):
        self.table = table

    @property
    def candidates(self):
        return frozenset(self.table)

    def estimates(self, values, candidate=None):
        return dict(self.table[candidate])


def test_a_fitted_pick_records_each_constraints_p_met_and_the_rule():
    criteria = Criteria(Maximize(HELP), [AtMost(HELP, 0.8)], REGISTRY, feasibility_threshold=0.9)
    estimates = {"0": Estimate(mean=0.5, sd=0.05), "a": Estimate(mean=0.7, sd=0.05)}
    report = pick(
        baseline_id="0",
        finalist_ids=["a"],
        validation={c: _val_scores(c, e.mean) for c, e in estimates.items()},
        fit=_FixedFit({c: {HELP: e} for c, e in estimates.items()}),
        values={"0": {}, "a": {}},
        trial_counts={"0": 2, "a": 2},
        completed={"0": 2, "a": 2},
        criteria=criteria,
        rule=LowerBound(q=0.2),
    )
    assert report.champion_id == "a" and report.outcome == "top_finalist"
    assert report.rule == {"kind": "LowerBound", "q": 0.2}
    constraint = "HelpfulnessEvaluator <= 0.8"
    assert report.p_met == {c: {constraint: p_met(AtMost(HELP, 0.8), e)} for c, e in estimates.items()}
    assert report.p_met["a"][constraint] == pytest.approx(0.97725, abs=1e-5)  # 2 sd below the bound
