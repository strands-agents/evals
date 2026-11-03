import pytest

from strands_evals.cli._optimize.scoring import (
    Criteria,
    MetricRegistry,
    add_usage,
    build_objective,
    measured,
    paired,
    paired_ratio,
    parse_term,
    require_optimization_extra,
    score_split,
    zero_usage,
)
from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.optimization import TrialRecord, TrialStatus

REGISTRY = MetricRegistry({"OutputEvaluator": (0.0, 1.0), "Tone": (0.0, 1.0)})


def record(case, run, status=TrialStatus.COMPLETED, scores=None, seconds=1.0, label=None):
    evaluations = {name: EvaluationOutput(score=score, test_pass=score > 0.5) for name, score in (scores or {}).items()}
    details = {
        name: [EvaluationOutput(score=score, test_pass=True, label=label)] for name, score in (scores or {}).items()
    }
    usage = zero_usage().model_copy(update={"seconds": seconds})
    return TrialRecord(
        candidate_id="c",
        case_name=case,
        run_index=run,
        split="train",
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
        details=details,
        evaluator_errors={},
        usage=usage,
        judge_usage=zero_usage(),
        rollouts=1,
    )


def test_build_objective_defaults_to_maximizing_every_evaluator():
    objective = build_objective([], REGISTRY)

    assert str(objective) == "WeightedSum(Maximize(OutputEvaluator), Maximize(Tone))"
    assert str(build_objective([], MetricRegistry({"Contains": (0.0, 1.0)}))) == "Maximize(Contains)"


def test_build_objective_resolves_shorthand_and_keeps_weights_and_scales():
    objective = build_objective(["maximize:Output:weight=2", "minimize:seconds:scale=10"], REGISTRY)

    assert str(objective) == "WeightedSum(Maximize(OutputEvaluator, weight=2), Minimize(seconds, scale=10))"


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (["maximize:tokens"], "not measured in project mode"),
        (["maximize:Unknown"], "unknown metric"),
        (["maximize:Tone:weight=2"], "two or more terms"),
        (["maximize:Tone", "minimize:seconds"], "give it scale="),
        (["maximize:Tone", "minimize:Tone"], "once in the objective"),
    ],
)
def test_build_objective_rejects_invalid_objectives(specs, message):
    with pytest.raises(ValueError, match=message):
        build_objective(specs, REGISTRY)


@pytest.mark.parametrize("spec", ["best:Tone", "maximize", "maximize:Tone:weight", "maximize:Tone:scale=-1"])
def test_parse_term_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="--objective"):
        parse_term(spec)


def test_registry_rejects_an_evaluator_named_like_a_usage_metric():
    with pytest.raises(ValueError, match="clash"):
        MetricRegistry({"tokens": (0.0, 1.0)})


def test_objective_value_negates_minimize_and_rescales_when_an_evaluator_is_missing():
    criteria = Criteria(build_objective(["maximize:Tone", "maximize:OutputEvaluator:weight=3"], REGISTRY), REGISTRY)
    lone = Criteria(build_objective(["minimize:seconds"], REGISTRY), REGISTRY)

    assert criteria.objective_value({"Tone": 1.0, "OutputEvaluator": 0.0}) == pytest.approx(0.25)
    assert criteria.objective_value({"Tone": 0.8}) == pytest.approx(0.8)
    assert criteria.objective_value({}) is None
    assert lone.objective_value({"seconds": 4.0}) == -4.0


@pytest.mark.parametrize(
    ("specs", "crash_score", "expected"),
    [
        (["maximize:Tone"], "worst", 0.0),
        (["minimize:Tone"], "worst", 1.0),
        (["maximize:Tone"], 0.25, 0.25),
        (["maximize:Tone"], None, None),
    ],
)
def test_crash_value_follows_the_crash_score(specs, crash_score, expected):
    criteria = Criteria(build_objective(specs, REGISTRY), REGISTRY, crash_score)

    assert criteria.crash_value("Tone") == expected
    assert criteria.crash_value("seconds") is None


def test_score_split_scores_crashes_drops_missing_runs_and_marks_eligibility():
    criteria = Criteria(build_objective(["maximize:Tone"], REGISTRY), REGISTRY)
    records = [
        record("a", 0, scores={"Tone": 0.8}),
        record("a", 1, status=TrialStatus.AGENT_FAILURE),
        record("b", 0, status=TrialStatus.INFRASTRUCTURE_INVALID),
    ]

    scores = score_split("c", records, ["a", "b"], criteria)

    assert scores.objective_by_run == {"a": [0.8, 0.0], "b": [None]}
    assert scores.objective_by_case == {"a": pytest.approx(0.4)}
    assert scores.objective == pytest.approx(0.4)
    assert scores.eligible is False
    assert scores.failures == 1


def test_score_split_takes_out_scores_outside_the_range_and_drops_not_applicable_rows():
    criteria = Criteria(build_objective(["maximize:Tone"], REGISTRY), REGISTRY)
    records = [record("a", 0, scores={"Tone": 1.5}), record("b", 0, scores={"Tone": 0.0}, label="not_applicable")]

    scores = score_split("c", records, ["a", "b"], criteria)

    assert list(scores.unavailable) == ["a"]
    assert "score_range" in scores.unavailable["a"][0]["Tone"]
    assert scores.objective_by_case == {}
    assert scores.objective is None


def test_measured_and_paired_report_90_percent_t_intervals():
    criteria = Criteria(build_objective(["maximize:Tone"], REGISTRY), REGISTRY)
    scores = score_split(
        "c", [record("a", 0, scores={"Tone": 0.0}), record("b", 0, scores={"Tone": 1.0})], ["a", "b"], criteria
    )

    tone = measured(scores)["Tone"]
    comparison = paired({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 1.0})

    assert tone.mean == 0.5 and tone.n == 2
    assert tone.ci90 == pytest.approx((0.5 - 3.1569, 0.5 + 3.1569), abs=1e-3)
    assert comparison.mean_diff == 0.5 and comparison.n_cases == 2
    assert comparison.ci_low == pytest.approx(0.5 - 3.1569, abs=1e-3)
    assert paired({"a": 1.0}, {"b": 1.0}).n_cases == 0


def test_paired_ratio_is_one_against_itself_and_none_for_zero_or_signed_metrics():
    values = {"a": 0.5, "b": 1.0, "c": 0.75}

    ratio = paired_ratio(values, values, can_be_negative=False)

    assert ratio is not None and ratio.mean_diff == 1.0 and ratio.ci_low == pytest.approx(1.0)
    assert paired_ratio(values, {"a": 0.0, "b": 0.0, "c": 0.0}, can_be_negative=False) is None
    assert paired_ratio(values, values, can_be_negative=True) is None


def test_add_usage_keeps_unmeasured_cache_tokens_and_joins_models():
    left = zero_usage().model_copy(update={"tokens": 3, "models": ["a"]})
    right = zero_usage().model_copy(update={"tokens": 4, "cache_read_tokens": 2, "models": ["b", "a"]})

    total = add_usage(left, right)

    assert total.tokens == 7 and total.cache_read_tokens == 2 and total.cache_write_tokens is None
    assert total.models == ["a", "b"]


def test_require_optimization_extra_names_the_install_command(monkeypatch):
    monkeypatch.setattr("strands_evals.cli._optimize.scoring.importlib.util.find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match=r"strands-agents-evals\[optimization\]"):
        require_optimization_extra()
