"""`OptimizationResult.summary()` on hand-built results whose validation reports come from the real `pick`."""

import statistics
from pathlib import Path

import pytest

from strands_evals.optimization import (
    AgentSpend,
    AtMost,
    Criteria,
    LowerBound,
    Maximize,
    Mean,
    MetricRegistry,
    Minimize,
    Pareto,
    RunCost,
    Spend,
)
from strands_evals.optimization.end_phase import paired_interval, pick
from strands_evals.optimization.result import CandidateView, OptimizationResult, _decimals, _run_cost, _value_cell
from strands_evals.optimization.score_model.effects import effects_json
from strands_evals.types.optimization import (
    CandidateScores,
    Estimate,
    IterationRecord,
    Measured,
    PairedComparison,
    TestReport,
    Usage,
)

BASELINE = "58dc753dfc0f"
REGISTRY = MetricRegistry({"WordCount": (0.0, 1.0), "Helpfulness": (0.0, 1.0)})


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


def _scores(criteria, candidate_id, by_case, *, eligible=True):
    """Validation scores from each case's metric values, one run each."""
    objective = {case: criteria.objective_value(metrics) for case, metrics in by_case.items()}
    return CandidateScores(
        candidate_id=candidate_id,
        objective_by_case=objective,
        metrics_by_case=by_case,
        unavailable={},
        objective_by_run={case: [value] for case, value in objective.items()},
        metrics_by_run={case: {m: [v] for m, v in metrics.items()} for case, metrics in by_case.items()},
        objective=statistics.fmean(objective.values()) if eligible else None,
        is_eligible=eligible,
        failures=0,
    )


def _cases(n, **values):
    """`n` cases with these metric values: a number for every case, or a list of one per case."""
    return {f"v{i}": {m: v[i] if isinstance(v, list) else v for m, v in values.items()} for i in range(n)}


def _view(candidate_id, number, parent_id, origin, values, *, changed=None):
    return CandidateView(
        id=candidate_id,
        number=number,
        parent_id=parent_id,
        origin=origin,
        rationale="",
        values=values,
        diff_text="",
        changed=changed or {},
    )


def _usage(tokens):
    return Usage(
        tokens=tokens,
        input_tokens=tokens,
        output_tokens=0,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=0,
        seconds=0.0,
        models=[],
    )


def _measured(mean, n):
    return Measured(mean=mean, ci95=None, n=n)


def _compared(mean, low, high, n):
    return PairedComparison(mean_diff=mean, ci_low=low, ci_high=high, n_cases=n)


def _test(candidate_id, objective, by_metric, n, *, missing=None):
    """A test report: per metric the baseline's mean, the candidate's, and the interval of the difference."""
    return TestReport(
        champion_id=candidate_id,
        baseline_id=BASELINE,
        objective=objective,
        difference={
            m: _compared(new - old, low, high, n).model_copy(update={"candidate_mean": new, "reference_mean": old})
            for m, (old, new, low, high) in by_metric.items()
        },
        ratio={},
        missing=missing or {},
        means={
            BASELINE: {m: _measured(old, n) for m, (old, _, _, _) in by_metric.items()},
            candidate_id: {m: _measured(new, n) for m, (_, new, _, _) in by_metric.items()},
        },
    )


def _result(criteria, candidates, validation, *, test, steps, **fields):
    baseline_values = fields.pop("baseline_values")
    views = {view.id: view for view in [_view(BASELINE, 0, None, ["baseline"], baseline_values), *candidates]}
    agent = fields.pop("agent", (43464, 29629, 9598))
    effects = effects_json(None, criteria)
    effects["reason"] = fields.pop("effects_reason", None)
    proposers = fields.pop("proposers", {"text": "Reflection"})
    # The proposers' tokens go to the first block, the others spent nothing.
    proposer_tokens = dict.fromkeys(proposers, 0) | {next(iter(proposers)): fields.pop("proposer_tokens", 134023)}
    splits = ("train", "validation", "test")
    cost = RunCost(
        agent={
            s: AgentSpend(usage=_usage(n), failed=0, trials=0, attempts=0) for s, n in zip(splits, agent, strict=True)
        },
        judges={
            s: Spend(usage=_usage(n), failed=0) for s, n in zip(splits, fields.pop("judges", (0, 0, 0)), strict=True)
        },
        proposers={block: Spend(usage=_usage(n), failed=0) for block, n in proposer_tokens.items()},
        benchmark={s: _usage(0) for s in splits},
    )
    return OptimizationResult(
        run_dir=Path("runs") / fields["run_id"],
        baseline_id=BASELINE,
        candidates=views,
        validation=validation,
        test=test,
        test_reason=fields.pop("test_reason", None),
        front_test=fields.pop("front_test", {}),
        effects=effects,
        objective=str(criteria.objective),
        constraints=[str(c) for c in criteria.constraints],
        feasibility_threshold=criteria.feasibility_threshold,
        proposers=proposers,
        iterations=[
            IterationRecord(
                iteration=i + 1,
                parent_id=BASELINE,
                round=None,
                role="opener",
                blocks=["text"],
                n_candidates=1,
                proposals=[],
                rollouts_spent=10,
            )
            for i in range(steps)
        ],
        cost=cost,
        reports={},
        rows={},
        **fields,
    )


def _pick(criteria, finalists, estimates, by_case, *, rule=None, problem=None, ineligible=()):
    """`pick` over the baseline and `finalists`, with `estimates` as the validation fit (None: measured means);
    the `ineligible` candidates lack a counted run on some validation case."""
    ids = [BASELINE, *finalists]
    return pick(
        baseline_id=BASELINE,
        finalist_ids=finalists,
        validation={c: _scores(criteria, c, by_case[c], eligible=c not in ineligible) for c in ids},
        fit=None if estimates is None else _FixedFit(estimates),
        values={c: {} for c in ids},
        trial_counts={c: 12 for c in ids},
        completed={c: 12 for c in ids},
        criteria=criteria,
        rule=rule or Mean(),
        problem=problem,
    )


def _poem_run():
    """The poem run: one objective, no constraints; candidate 4, a text change, ships."""
    c2, c3, c4 = "11d0cd495d41", "9ffe82afa9b1", "687be31f0e42"
    criteria = Criteria(Maximize("WordCount"), [], REGISTRY)
    fitted = {BASELINE: (0.156, 0.077), c4: (0.953, 0.083), c3: (0.879, 0.081), c2: (0.655, 0.082)}
    hits = {BASELINE: 0, c4: 12, c3: 11, c2: 8}  # validation cases with the right word count, of 12
    validation = _pick(
        criteria,
        [c4, c3, c2],
        {c: {"WordCount": Estimate(mean=mean, sd=sd)} for c, (mean, sd) in fitted.items()},
        {c: _cases(12, WordCount=[1.0] * k + [0.0] * (12 - k)) for c, k in hits.items()},
    )
    candidates = [
        _view(c2, 2, BASELINE, ["text"], {"system_prompt": "Write a short poem."}),
        _view(c3, 3, c2, ["text"], {"system_prompt": "Write a poem of about ten words."}),
        _view(c4, 4, c3, ["text"], {"system_prompt": "Write a poem of exactly ten words."}),
    ]
    test = _test(c4, _compared(1.0, 1.0, 1.0, 5), {"WordCount": (0.0, 1.0, 1.0, 1.0)}, 5)
    return _result(
        criteria,
        candidates,
        validation,
        test=test,
        steps=4,
        baseline_values={"system_prompt": "Write a poem."},
        run_id="poem-live-20261116-213955-66d9ad",
        rollouts_spent=138,
        max_rollouts=150,
        stop_reason="iterations",
    )


def test_a_single_objective_run():
    assert _poem_run().summary() == "\n".join(
        [
            "Optimization result: poem-live-20261116-213955-66d9ad",
            "=================================================================",
            "Objective:    Maximize(WordCount)",
            "Stop reason:  iterations              Rollouts:   138 / 150",
            "Iterations:   4                       Pick rule:  Mean",
            "Champion:     candidate 4             Outcome:    top_finalist",
            "Parent:       candidate 3",
            "Proposers:    text (Reflection)",
            "Changed:      system_prompt",
            "=================================================================",
            "Validation     id            estimate     sd  measured   n",
            "-----------------------------------------------------------------",
            "baseline       58dc753dfc0f     0.156  0.077     0.000  12",
            "candidate 4 *  687be31f0e42     0.953  0.083     1.000  12",
            "candidate 3    9ffe82afa9b1     0.879  0.081     0.917  12",
            "candidate 2    11d0cd495d41     0.655  0.082     0.667  12",
            "=================================================================",
            "Test vs baseline  baseline  champion    diff            90% CI  n",
            "-----------------------------------------------------------------",
            "WordCount            0.000     1.000  +1.000  [+1.000, +1.000]  5",
            "=================================================================",
            "Tokens          search  validation   test    total  failed",
            "-----------------------------------------------------------------",
            "agent           43,464      29,629  9,598   82,691       0",
            "judges               0           0      0        0       0",
            "proposer text  134,023           0      0  134,023       0",
            "other agents         0           0      0        0       -",
            "total          177,487      29,629  9,598  216,714       -",
            "=================================================================",
            "Notes:",
            "[1] estimate, sd: the validation fit's estimate of the mean and",
            "    its posterior sd; measured: the mean over the cases measured,",
            "    n of them (the runs, when one case has a value); P(met) is -",
            "    for a constraint judged by its measured or adjusted mean; *",
            "    the champion.",
            "[2] Test: baseline, champion, and diff (champion - baseline, with",
            "    its 90% interval) are over the n cases both measured;",
            "    reported only.",
            "[3] Tokens: search is the train split; every attempt counts, the",
            "    ones run again included; failed: the agent's attempts that",
            "    failed, infrastructure failures such as throttling included;",
            "    the judges' attempts with a failed evaluator; a proposer's",
            "    retried failures and requests that failed; other agents are",
            "    the rollouts' agents the factory didn't build (a simulated",
            "    user).",
        ]
    )


def _constrained_run():
    """An objective, a constraint and a cautious pick rule: candidate 5, merging two blocks' patches, ships;
    candidate 2 fails the constraint and candidate 7 has no counted run on some validation case."""
    c2, c5, c7 = "b7a1e09c44d2", "e3f5c8d21a07", "2d4f6a8c0e13"
    criteria = Criteria(Maximize("Helpfulness"), [AtMost("tokens", 4000)], REGISTRY)
    fitted = {
        BASELINE: {"Helpfulness": (0.64, 0.03), "tokens": (3120.0, 210.0)},
        c2: {"Helpfulness": (0.84, 0.03), "tokens": (4410.0, 260.0)},
        c5: {"Helpfulness": (0.77, 0.03), "tokens": (2480.0, 190.0)},
    }
    measured = {BASELINE: (0.63, 3088.4), c2: (0.85, 4377.0), c5: (0.78, 2511.9)}
    validation = _pick(
        criteria,
        [c2, c5, c7],
        {c: {m: Estimate(mean=mean, sd=sd) for m, (mean, sd) in by_metric.items()} for c, by_metric in fitted.items()},
        {
            **{c: _cases(8, Helpfulness=h, tokens=t) for c, (h, t) in measured.items()},
            c7: _cases(6, Helpfulness=0.81, tokens=2390.0),  # 2 of its 8 validation cases have no counted run
        },
        rule=LowerBound(q=0.1),
        ineligible=[c7],
    )
    candidates = [
        _view(c2, 2, BASELINE, ["text"], {"system_prompt": "Answer in detail.", "temperature": 0.7}),
        _view(
            c5,
            5,
            c2,
            ["text", "numbers"],
            {"system_prompt": "Answer briefly.", "temperature": 0.2},
            changed={"temperature": {"old": 0.7, "new": 0.2}},
        ),
        _view(c7, 7, c5, ["numbers"], {"system_prompt": "Answer briefly.", "temperature": 0.0}),
    ]
    test = _test(
        c5,
        _compared(0.14, 0.05, 0.23, 8),
        {"Helpfulness": (0.62, 0.76, 0.05, 0.23), "tokens": (3101.5, 2490.0, -840.2, -382.8)},
        8,
        missing={c5: 1, BASELINE: 0},  # one test case with no counted run of candidate 5
    )
    return _result(
        criteria,
        candidates,
        validation,
        test=test,
        steps=9,
        baseline_values={"system_prompt": "Answer the question.", "temperature": 0.7},
        proposers={"text": "Reflection", "numbers": "TPE"},
        run_id="support-20261117-090412-1f2e3d",
        rollouts_spent=596,
        max_rollouts=600,
        stop_reason="budget",
        agent=(1204331, 301877, 160402),
        proposer_tokens=88210,
        judges=(71004, 16180, 7928),
    )


def test_a_constrained_run_has_a_row_per_candidate_and_goal():
    assert _constrained_run().summary() == "\n".join(
        [
            "Optimization result: support-20261117-090412-1f2e3d",
            "========================================================================================================",
            "Objective:     Maximize(Helpfulness)",
            "Constraints:   tokens <= 4000",
            "P(met) >=:     0.9",
            "Stop reason:   budget                 Rollouts:   596 / 600",
            "Iterations:    9                      Pick rule:  LowerBound(q=0.1)",
            "Champion:      candidate 5            Outcome:    top_finalist",
            "Parent:        candidate 2",
            "Proposers:     text (Reflection), numbers (TPE)",
            "Changed:       system_prompt, temperature",
            "Test missing:  candidate 5: 1 case",
            "========================================================================================================",
            "Validation     id            objective  goal            estimate     sd  measured  n  P(met)  status",
            "--------------------------------------------------------------------------------------------------------",
            "baseline       58dc753dfc0f      0.602  Helpfulness        0.640  0.030     0.630  8          feasible",
            "                                        tokens <= 4000     3,120    210     3,088  8   1.000",
            "candidate 2    b7a1e09c44d2      0.802  Helpfulness        0.840  0.030     0.850  8          infeasible",
            "                                        tokens <= 4000     4,410    260     4,377  8   0.057",
            "candidate 5 *  e3f5c8d21a07      0.732  Helpfulness        0.770  0.030     0.780  8          feasible",
            "                                        tokens <= 4000     2,480    190     2,512  8   1.000",
            "candidate 7    2d4f6a8c0e13          -  Helpfulness            -      -     0.810  6          ineligible",
            "                                        tokens <= 4000         -      -     2,390  6       -",
            "========================================================================================================",
            "Not feasible  why",
            "--------------------------------------------------------------------------------------------------------",
            "candidate 2   tokens <= 4000: P(met) 0.057 < 0.9",
            "candidate 7   not eligible on validation: a case has no counted run for every goal metric",
            "========================================================================================================",
            "Test vs baseline  baseline  champion    diff            90% CI  n",
            "--------------------------------------------------------------------------------------------------------",
            "Helpfulness          0.620     0.760  +0.140  [+0.050, +0.230]  8",
            "tokens               3,102     2,490    -612      [-840, -383]  8",
            "========================================================================================================",
            "Changed      baseline  champion",
            "--------------------------------------------------------------------------------------------------------",
            "temperature  0.7       0.2",
            "========================================================================================================",
            "Tokens               search  validation     test      total  failed",
            "--------------------------------------------------------------------------------------------------------",
            "agent             1,204,331     301,877  160,402  1,666,610       0",
            "judges               71,004      16,180    7,928     95,112       0",
            "proposer text        88,210           0        0     88,210       0",
            "proposer numbers          0           0        0          0       0",
            "other agents              0           0        0          0       -",
            "total             1,363,545     318,057  168,330  1,849,932       -",
            "========================================================================================================",
            "Notes:",
            "[1] estimate, sd: the validation fit's estimate of the mean and its posterior sd; measured: the mean",
            "    over the cases measured, n of them (the runs, when one case has a value); P(met) is - for a",
            "    constraint judged by its measured or adjusted mean; * the champion.",
            "[2] Test: baseline, champion, and diff (champion - baseline, with its 90% interval) are over the n cases",
            "    both measured; reported only.",
            "[3] Tokens: search is the train split; every attempt counts, the ones run again included; failed: the",
            "    agent's attempts that failed, infrastructure failures such as throttling included; the judges'",
            "    attempts with a failed evaluator; a proposer's retried failures and requests that failed; other",
            "    agents are the rollouts' agents the factory didn't build (a simulated user).",
        ]
    )


def _pareto_run():
    """Two objectives under `Pareto`: the baseline has the highest gain on the validated front and ships;
    the test report couldn't afford candidate 6."""
    c3, c6 = "4c9d02be17f3", "0a6e5b3dd981"
    criteria = Criteria(Pareto(Maximize("Helpfulness"), Minimize("tokens", scale=1000)), [], REGISTRY)
    fitted = {BASELINE: (0.81, 950.0), c3: (0.70, 900.0), c6: (0.60, 850.0)}  # each on the front
    validation = _pick(
        criteria,
        [c3, c6],
        {
            c: {"Helpfulness": Estimate(mean=h, sd=0.02), "tokens": Estimate(mean=t, sd=40.0)}
            for c, (h, t) in fitted.items()
        },
        {c: _cases(10, Helpfulness=h, tokens=t) for c, (h, t) in fitted.items()},
    )
    candidates = [
        _view(c3, 3, BASELINE, ["text"], {"system_prompt": "Be brief."}),
        _view(c6, 6, c3, ["text"], {"system_prompt": "Be very brief."}),
    ]
    itself = _test(
        BASELINE,
        _compared(0.0, 0.0, 0.0, 6),
        {"Helpfulness": (0.79, 0.79, 0.0, 0.0), "tokens": (962.0, 962.0, 0.0, 0.0)},
        6,
    )
    c3_test = _test(
        c3,
        _compared(-0.031, -0.058, -0.004, 6),
        {"Helpfulness": (0.79, 0.71, -0.13, -0.03), "tokens": (962.0, 911.5, -84.0, -17.0)},
        6,
    )
    return _result(
        criteria,
        candidates,
        validation,
        test=itself,
        steps=6,
        baseline_values={"system_prompt": "Answer the question."},
        front_test={BASELINE: itself, c3: c3_test},
        test_reason="the budget couldn't test front members 6",
        run_id="pareto-20261117-101500-77aa01",
        rollouts_spent=410,
        max_rollouts=500,
        stop_reason="exhausted",
    )


def test_a_pareto_run_adds_the_front_on_test():
    assert _pareto_run().summary() == "\n".join(
        [
            "Optimization result: pareto-20261117-101500-77aa01",
            "===================================================================================================",
            "Objective:    Pareto(Maximize(Helpfulness), Minimize(tokens, scale=1000))",
            "Stop reason:  exhausted               Rollouts:   410 / 500",
            "Iterations:   6                       Pick rule:  Mean",
            "Champion:     baseline                Outcome:    baseline_best",
            "Parent:       -",
            "Proposers:    text (Reflection)",
            "Changed:      -",
            "Test reason:  the budget couldn't test front members 6",
            "===================================================================================================",
            "Validation   id            objective  goal         estimate     sd  measured   n",
            "---------------------------------------------------------------------------------------------------",
            "baseline *   58dc753dfc0f     -0.070  Helpfulness     0.810  0.020     0.810  10",
            "                                      tokens          950.0   40.0     950.0  10",
            "candidate 3  4c9d02be17f3     -0.100  Helpfulness     0.700  0.020     0.700  10",
            "                                      tokens          900.0   40.0     900.0  10",
            "candidate 6  0a6e5b3dd981     -0.125  Helpfulness     0.600  0.020     0.600  10",
            "                                      tokens          850.0   40.0     850.0  10",
            "===================================================================================================",
            "Test vs baseline  baseline  champion    diff            90% CI  n",
            "---------------------------------------------------------------------------------------------------",
            "objective                -         -  +0.000  [+0.000, +0.000]  6",
            "Helpfulness          0.790     0.790  +0.000  [+0.000, +0.000]  6",
            "tokens               962.0     962.0    +0.0      [+0.0, +0.0]  6",
            "===================================================================================================",
            "Front test vs baseline  id            metric       baseline  member     diff              90% CI  n",
            "---------------------------------------------------------------------------------------------------",
            "baseline *              58dc753dfc0f  objective           -       -  +0.0000  [+0.0000, +0.0000]  6",
            "                                      Helpfulness     0.790   0.790   +0.000    [+0.000, +0.000]  6",
            "                                      tokens          962.0   962.0     +0.0        [+0.0, +0.0]  6",
            "candidate 3             4c9d02be17f3  objective           -       -  -0.0310  [-0.0580, -0.0040]  6",
            "                                      Helpfulness     0.790   0.710   -0.080    [-0.130, -0.030]  6",
            "                                      tokens          962.0   911.5    -50.5      [-84.0, -17.0]  6",
            "candidate 6             0a6e5b3dd981  objective           -       -        -                   -  -",
            "                                      Helpfulness         -       -        -                   -  -",
            "                                      tokens              -       -        -                   -  -",
            "===================================================================================================",
            "Tokens          search  validation   test    total  failed",
            "---------------------------------------------------------------------------------------------------",
            "agent           43,464      29,629  9,598   82,691       0",
            "judges               0           0      0        0       0",
            "proposer text  134,023           0      0  134,023       0",
            "other agents         0           0      0        0       -",
            "total          177,487      29,629  9,598  216,714       -",
            "===================================================================================================",
            "Notes:",
            "[1] estimate, sd: the validation fit's estimate of the mean and its posterior sd; measured: the",
            "    mean over the cases measured, n of them (the runs, when one case has a value); P(met) is - for",
            "    a constraint judged by its measured or adjusted mean; * the champion.",
            "[2] Test: baseline, champion, and diff (champion - baseline, with its 90% interval) are over the n",
            "    cases both measured; reported only.",
            "[3] Tokens: search is the train split; every attempt counts, the ones run again included; failed:",
            "    the agent's attempts that failed, infrastructure failures such as throttling included; the",
            "    judges' attempts with a failed evaluator; a proposer's retried failures and requests that",
            "    failed; other agents are the rollouts' agents the factory didn't build (a simulated user).",
        ]
    )


def _no_finalist_run():
    """The search gave no finalist: the baseline alone on validation, by measured means, and no test cases."""
    criteria = Criteria(Maximize("WordCount"), [], REGISTRY)
    validation = _pick(
        criteria,
        [],
        None,
        {BASELINE: _cases(4, WordCount=[1.0, 0.0, 1.0, 0.0])},
        problem="fewer than 2 eligible candidates",
    )
    return _result(
        criteria,
        [],
        validation,
        test=None,
        steps=1,
        baseline_values={"system_prompt": "Write a poem."},
        test_reason="no test cases",
        effects_reason="no fit: fewer than 2 eligible candidates",
        run_id="poem-20261117-110000-000001",
        rollouts_spent=24,
        max_rollouts=150,
        stop_reason="exhausted",
        agent=(2210, 1105, 0),
        proposer_tokens=0,
    )


def test_a_run_without_finalists_prints_the_recorded_reasons():
    assert _no_finalist_run().summary() == "\n".join(
        [
            "Optimization result: poem-20261117-110000-000001",
            "==============================================================",
            "Objective:       Maximize(WordCount)",
            "Stop reason:     exhausted            Rollouts:   24 / 150",
            "Iterations:      1                    Pick rule:  Mean",
            "Champion:        baseline             Outcome:    no_finalists",
            "Parent:          -",
            "Proposers:       text (Reflection)",
            "Changed:         -",
            "Validation fit:  fewer than 2 eligible candidates",
            "Effects:         no fit: fewer than 2 eligible candidates",
            "Test reason:     no test cases",
            "==============================================================",
            "Validation  id            estimate  sd  measured  n",
            "--------------------------------------------------------------",
            "baseline *  58dc753dfc0f         -   -     0.500  4",
            "==============================================================",
            "Tokens         search  validation  test  total  failed",
            "--------------------------------------------------------------",
            "agent           2,210       1,105     0  3,315       0",
            "judges              0           0     0      0       0",
            "proposer text       0           0     0      0       0",
            "other agents        0           0     0      0       -",
            "total           2,210       1,105     0  3,315       -",
            "==============================================================",
            "Notes:",
            "[1] estimate, sd: the validation fit's estimate of the mean",
            "    and its posterior sd; measured: the mean over the cases",
            "    measured, n of them (the runs, when one case has a value);",
            "    P(met) is - for a constraint judged by its measured or",
            "    adjusted mean; * the champion.",
            "[2] Tokens: search is the train split; every attempt counts,",
            "    the ones run again included; failed: the agent's attempts",
            "    that failed, infrastructure failures such as throttling",
            "    included; the judges' attempts with a failed evaluator; a",
            "    proposer's retried failures and requests that failed;",
            "    other agents are the rollouts' agents the factory didn't",
            "    build (a simulated user).",
        ]
    )


def test_a_lone_objective_gets_its_column_when_the_picks_rounding_shows():
    # With sd 0.15, the pick rounds each estimate to 0.01, so 0.953 ranks as 0.95.
    c1 = "a41f9c03d2e8"
    criteria = Criteria(Maximize("WordCount"), [], REGISTRY)
    fitted = {BASELINE: 0.156, c1: 0.953}
    validation = _pick(
        criteria,
        [c1],
        {c: {"WordCount": Estimate(mean=mean, sd=0.15)} for c, mean in fitted.items()},
        {c: _cases(4, WordCount=mean) for c, mean in fitted.items()},
    )
    result = _result(
        criteria,
        [_view(c1, 1, BASELINE, ["text"], {"system_prompt": "Write a ten-word poem."})],
        validation,
        test=None,
        steps=1,
        baseline_values={"system_prompt": "Write a poem."},
        run_id="poem",
        rollouts_spent=20,
        max_rollouts=150,
        stop_reason="iterations",
    )
    lines = result.summary().splitlines()
    assert lines[lines.index("Validation     id            objective  estimate     sd  measured  n") + 3] == (
        "candidate 1 *  a41f9c03d2e8      0.950     0.953  0.150     0.953  4"
    )


def test_a_lone_minimized_objective_leaves_its_negated_column_out():
    c1 = "a41f9c03d2e8"
    criteria = Criteria(Minimize("tokens"), [], REGISTRY)
    fitted = {BASELINE: 3120.0, c1: 2480.0}
    validation = _pick(
        criteria,
        [c1],
        {c: {"tokens": Estimate(mean=mean, sd=50.0)} for c, mean in fitted.items()},
        {c: _cases(4, tokens=mean) for c, mean in fitted.items()},
    )
    assert validation.objectives == {BASELINE: -3120.0, c1: -2480.0}
    result = _result(
        criteria,
        [_view(c1, 1, BASELINE, ["text"], {"system_prompt": "Be brief."})],
        validation,
        test=None,
        steps=1,
        baseline_values={"system_prompt": "Answer the question."},
        run_id="brief",
        rollouts_spent=20,
        max_rollouts=150,
        stop_reason="iterations",
    )
    assert "Validation     id            estimate  sd  measured  n" in result.summary().splitlines()


def test_the_changed_table_shows_recorded_values_and_inactive_sides_and_no_text():
    c1 = "a41f9c03d2e8"
    criteria = Criteria(Maximize("WordCount"), [], REGISTRY)
    validation = _pick(criteria, [c1], None, {c: _cases(4, WordCount=0.5) for c in (BASELINE, c1)})
    changed = {"manager": {"old": "window", "new": "summary"}, "manager.size": {"old": 8}}
    view = _view(c1, 1, BASELINE, ["structure"], {"manager": "summary", "prompt": "Be brief."}, changed=changed)
    result = _result(
        criteria,
        [view],
        validation.model_copy(update={"champion_id": c1}),
        test=None,
        steps=1,
        baseline_values={"manager": "window", "manager.size": 8, "prompt": "Answer."},
        proposers={"structure": "TPE"},
        run_id="manager",
        rollouts_spent=20,
        max_rollouts=150,
        stop_reason="iterations",
    )
    lines = result.summary().splitlines()
    assert "Changed:      manager, manager.size, prompt" in lines  # the header lists text too
    start = lines.index("Changed       baseline  champion")
    assert lines[start + 2 : start + 5] == [
        "manager       'window'  'summary'",
        "manager.size  8         inactive",
        "=" * len(lines[1]),  # the text value is in diff(), not here
    ]


def test_a_long_note_prints_its_first_line_and_leaves_the_rules_alone():
    # A constraint judged by measured means has no P(met); the model's message runs over two lines.
    problem = "the model raised: ValueError: " + "x" * 150 + "\nsecond line"
    criteria = Criteria(Maximize("Helpfulness"), [AtMost("tokens", 4000)], REGISTRY)
    by_case = {BASELINE: _cases(4, Helpfulness=0.6, tokens=3000.0)}
    validation = _pick(criteria, [], None, by_case, problem=problem)
    result = _result(
        criteria,
        [],
        validation,
        test=None,
        steps=1,
        baseline_values={"system_prompt": "Answer."},
        run_id="note",
        rollouts_spent=20,
        max_rollouts=150,
        stop_reason="exhausted",
    )
    lines = result.summary().splitlines()
    assert f"Validation fit:  {problem.splitlines()[0]}" in lines and "second line" not in result.summary()
    assert len(lines[1]) < 100  # the rule spans the tables, not the note
    assert (
        lines[lines.index(lines[1], 2) + 4]
        == "                          tokens <= 4000         -   -     3,000  4       -"
    )


def test_a_result_written_before_the_summary_fields_loads_and_summarizes():
    data = _constrained_run().to_dict()
    for field in ("objective", "constraints", "feasibility_threshold", "proposers", "max_rollouts"):
        del data[field]
    for field in ("p_met", "rule"):
        del data["validation"][field]
    result = OptimizationResult.from_dict(data)
    assert result.validation.outcome == "top_finalist"  # derived, never stored
    lines = [line.split() for line in result.summary().splitlines()]
    assert lines[2] == ["Objective:", "-"]  # and no constraint lines: none were recorded
    assert lines[3][-2:] == ["Rollouts:", "596"]
    assert lines[4][-3:] == ["Pick", "rule:", "-"]
    assert lines[7] == ["Proposers:", "-"]
    # The constraint rows are labeled by their metric, with no P(met).
    assert ["tokens", "2,480", "190", "2,512", "8", "-"] in lines


def test_a_result_written_before_cost_names_the_resume_that_rewrites_it():
    data = _constrained_run().to_dict()
    del data["cost"]
    data.update(agent_cost={}, search_cost={}, judge_cost={}, retry_cost={})
    with pytest.raises(ValueError, match="resume the run to rewrite it"):
        OptimizationResult.from_dict(data)


def test_a_test_rows_means_cover_the_cases_its_difference_does():
    # The champion lost case e, so the baseline's own test mean (over a-e) and the difference (over a-d) disagree.
    baseline = {"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8, "e": 0.0}
    champion = {"a": 0.3, "b": 0.5, "c": 0.7, "d": 0.9}
    run = _poem_run()
    c4 = run.validation.champion_id
    report = run.test.model_copy(
        update={
            "difference": {"WordCount": paired_interval(champion, baseline, level=0.9)},
            "missing": {c4: 1, BASELINE: 0},
            "means": {BASELINE: {"WordCount": _measured(0.4, 5)}, c4: {"WordCount": _measured(0.6, 4)}},
        }
    )
    lines = run.model_copy(update={"test": report}).summary().splitlines()
    row = next(line for line in lines if line.startswith("WordCount "))
    assert row.split() == ["WordCount", "0.500", "0.600", "+0.100", "[+0.100,", "+0.100]", "4"]
    assert "Test missing:  candidate 4: 1 case" in lines


def test_small_values_keep_three_significant_digits():
    assert [_decimals([0.000105, 0.000305]), _decimals([0.05]), _decimals([0.5]), _decimals([0.0])] == [6, 4, 3, 3]


def test_the_changed_table_cuts_long_values_and_rounds_floats():
    assert _value_cell(0.6818711275851573) == "0.681871"
    assert _value_cell("x" * 60) == repr("x" * 60)[:39] + "…"
    assert _value_cell(0.2) == "0.2" and _value_cell(1.0) == "1.0" and _value_cell(3) == "3"
    assert _value_cell(None) == "None" and _value_cell("long") == "'long'"


def test_a_p_met_short_of_the_threshold_never_prints_as_meeting_it():
    run = _constrained_run()
    assert run._p_met_cell(0.8997) == "0.899" and run._p_met_cell(0.9003) == "0.900"
    assert run._p_met_cell(0.99999) == "1.000" and run._p_met_cell(None) == "-"


def test_a_lone_minimized_objective_at_zero_leaves_its_column_out():
    c1 = "a41f9c03d2e8"
    criteria = Criteria(Minimize("tokens"), [], REGISTRY)
    fitted = {BASELINE: 0.4, c1: 0.0}
    validation = _pick(
        criteria,
        [c1],
        {c: {"tokens": Estimate(mean=mean, sd=0.0001)} for c, mean in fitted.items()},
        {c: _cases(4, tokens=mean) for c, mean in fitted.items()},
    )
    result = _result(
        criteria,
        [_view(c1, 1, BASELINE, ["text"], {"system_prompt": "Be brief."})],
        validation,
        test=None,
        steps=1,
        baseline_values={"system_prompt": "Answer the question."},
        run_id="zero",
        rollouts_spent=20,
        max_rollouts=150,
        stop_reason="iterations",
    )
    assert "objective" not in next(line for line in result.summary().splitlines() if line.startswith("Validation "))


# The run's cost from its saved trials and events.


def _usage_json(tokens, calls=1):
    return Usage(
        tokens=tokens,
        input_tokens=tokens,
        output_tokens=0,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=calls,
        seconds=1.0,
        models=[],
    ).model_dump()


def _attempt(status, *, usage=0, judges=0, rollouts=1):
    return {
        "status": status,
        "error_type": None,
        "error": None,
        "usage": _usage_json(usage, calls=int(usage > 0)),
        "judge_usage": _usage_json(judges, calls=int(judges > 0)),
        "pass_index": 0,
        "rollouts": rollouts,
    }


def _trial(split, status, *, usage, judges=None, errors=None, previous=()):
    return {
        "split": split,
        "status": status,
        "usage": _usage_json(usage),
        "judge_usage": {name: _usage_json(tokens) for name, tokens in (judges or {}).items()},
        "evaluator_errors": dict(errors or {}),
        "previous_attempts": list(previous),
    }


def test_the_agent_and_the_judges_count_every_attempt_each_in_its_own_role():
    trials = [
        # A re-run pass replaced an infrastructure failure: the agent ran twice.
        _trial(
            "train", "completed", usage=100, judges={"J": 10}, previous=[_attempt("infrastructure_invalid", usage=30)]
        ),
        # Throttled once inside the attempt, then completed with one evaluator failed.
        _trial(
            "train",
            "completed",
            usage=60,
            judges={"J": 10, "K": 4},
            errors={"K": "ValueError"},
            previous=[_attempt("infrastructure_invalid", usage=5, rollouts=0)],
        ),
        # A judge-only re-run: the output's usage stays on the new record; the replaced attempt keeps its failed
        # judge's spend only.
        _trial(
            "validation", "completed", usage=50, judges={"J": 10, "K": 5}, previous=[_attempt("completed", judges=7)]
        ),
        # Every evaluator failed, and failed again when judged again.
        _trial(
            "validation",
            "evaluator_invalid",
            usage=40,
            judges={"J": 3, "K": 3},
            errors={"J": "ValueError", "K": "ValueError"},
            previous=[_attempt("evaluator_invalid", judges=6)],
        ),
        _trial("test", "agent_failure", usage=20),
        # Infrastructure failed, and failed again on its re-run pass.
        _trial("test", "infrastructure_invalid", usage=15, previous=[_attempt("infrastructure_invalid", usage=15)]),
    ]
    cost = _run_cost(trials, [], blocks=[])

    assert {split: (s.usage.tokens, s.trials, s.attempts, s.failed) for split, s in cost.agent.items()} == {
        "train": (100 + 30 + 60 + 5, 2, 4, 2),
        "validation": (50 + 40, 2, 2, 0),  # judge-only re-runs don't run the agent
        "test": (20 + 15 + 15, 2, 3, 3),
    }
    assert {split: (s.usage.tokens, s.failed) for split, s in cost.judges.items()} == {
        "train": (10 + 10 + 4, 1),
        "validation": (10 + 5 + 7 + 3 + 3 + 6, 1 + 2),
        "test": (0, 0),
    }
    assert cost.agent["train"].usage.model_calls == 4  # two trials, each with an earlier attempt
    assert cost.proposers == {}


def test_each_block_gets_its_proposers_spend_and_failed_attempts():
    events = [
        {"event": "proposer_usage", "block": "text", "usage": _usage_json(10)},  # the returned patches' usage
        {"event": "proposer_retry", "block": "text", "error_type": "ValueError", "error": "bad"},
        {"event": "proposer_usage", "block": "text", "usage": _usage_json(5)},  # spend on no returned patch
        {"event": "proposer_error", "block": "text", "error_type": "ValueError", "error": "bad"},
        {"event": "proposer_usage", "block": "structure", "usage": _usage_json(3)},
        {"event": "proposer_retry", "block": "other", "error_type": "ValueError", "error": "bad"},
        {"event": "trial", "record": {}},
    ]
    cost = _run_cost([], events, blocks=["numbers", "structure", "text"])

    # Every block, in the search's order, then a block only the events name.
    assert {block: (s.usage.tokens, s.failed) for block, s in cost.proposers.items()} == {
        "numbers": (0, 0),
        "structure": (3, 0),
        "text": (15, 2),
        "other": (0, 1),
    }
    assert list(cost.proposers) == ["numbers", "structure", "text", "other"]
    assert cost.total.tokens == 18
    assert cost.agent["train"] == AgentSpend(usage=Usage.zero(), failed=0, trials=0, attempts=0)


def test_the_rollouts_other_agents_count_per_split_in_every_attempt():
    retried = _attempt("infrastructure_invalid", usage=30) | {"benchmark_usage": _usage_json(4)}
    trials = [
        _trial("train", "completed", usage=100, previous=[retried]) | {"benchmark_usage": _usage_json(6)},
        _trial("test", "completed", usage=50) | {"benchmark_usage": _usage_json(2)},
        _trial("test", "completed", usage=50),  # saved before benchmark usage was recorded
    ]
    cost = _run_cost(trials, [], blocks=[])

    assert {split: usage.tokens for split, usage in cost.benchmark.items()} == {"train": 10, "validation": 0, "test": 2}
    assert cost.agent["train"].usage.tokens == 130  # never in the agent's
    assert cost.total.tokens == 130 + 100 + 12
