from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from strands_evals import Case, Experiment
from strands_evals.evaluators import Equals, SkillSelectionAccuracyEvaluator
from strands_evals.experimental.comparison.pairing import (
    PairedScore,
    PairedTrialScore,
    aggregate_trial_pairs,
    pair_report_trials,
    pair_reports,
)
from strands_evals.experimental.comparison.summary import summarize_pairs
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import AgentInvocationSpan, Session, SpanInfo, Trace


def _report(rows: list[dict]) -> EvaluationReport:
    return EvaluationReport(
        overall_score=sum(row["score"] for row in rows) / len(rows) if rows else 0.0,
        scores=[row["score"] for row in rows],
        cases=[row["case"] for row in rows],
        test_passes=[True] * len(rows),
        detailed_results=[row.get("details", []) for row in rows],
    )


def _row(
    *,
    name: str | None,
    evaluator: str = "helpfulness",
    score: float,
    input_value: object = "question",
    session_id: str = "session",
    actual_output: str = "output",
    details: list[EvaluationOutput] | None = None,
) -> dict:
    return {
        "score": score,
        "case": {
            "name": name,
            "input": input_value,
            "actual_output": actual_output,
            "session_id": session_id,
            "evaluator": evaluator,
        },
        "details": details or [],
    }


def _session() -> Session:
    now = datetime.now(timezone.utc)
    span_info = SpanInfo(
        trace_id="trace",
        span_id="span",
        session_id="session",
        start_time=now,
        end_time=now,
    )
    span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="question",
        agent_response="answer",
        available_tools=[],
    )
    return Session(
        traces=[Trace(spans=[span], trace_id="trace", session_id="session")],
        session_id="session",
    )


def test_pairs_rows_by_case_and_evaluator_regardless_of_order():
    baseline = _report(
        [
            _row(name="case-b", evaluator="correctness", score=0.4),
            _row(name="case-a", evaluator="helpfulness", score=0.3),
            _row(name="case-a", evaluator="correctness", score=0.6),
        ]
    )
    candidate = _report(
        [
            _row(
                name="case-a",
                evaluator="correctness",
                score=0.9,
                session_id="new-session",
                actual_output="new output",
            ),
            _row(name="case-b", evaluator="correctness", score=0.2),
            _row(name="case-a", evaluator="helpfulness", score=0.5),
        ]
    )

    paired = pair_reports(baseline=baseline, candidate=candidate)

    assert [(pair.evaluator, pair.case_id) for pair in paired] == [
        ("correctness", "case-a"),
        ("correctness", "case-b"),
        ("helpfulness", "case-a"),
    ]
    assert [pair.baseline_index for pair in paired] == [2, 0, 1]
    assert [pair.candidate_index for pair in paired] == [0, 1, 2]
    assert [pair.baseline_score for pair in paired] == pytest.approx([0.6, 0.4, 0.3])
    assert [pair.candidate_score for pair in paired] == pytest.approx([0.9, 0.2, 0.5])
    assert [pair.delta for pair in paired] == pytest.approx([0.3, -0.2, 0.2])


def test_paired_score_is_frozen_and_computes_delta():
    pair = PairedScore(
        case_id="case",
        evaluator="equals",
        baseline_index=0,
        candidate_index=1,
        baseline_score=0.2,
        candidate_score=0.9,
    )

    assert pair.delta == pytest.approx(0.7)
    assert pair.model_dump()["delta"] == pytest.approx(0.7)
    with pytest.raises(ValidationError):
        pair.candidate_score = 0.1


def test_paired_score_round_trip_recomputes_delta():
    pair = PairedScore(
        case_id="case",
        evaluator="equals",
        baseline_index=0,
        candidate_index=1,
        baseline_score=0.2,
        candidate_score=0.9,
    )

    payload = pair.model_dump()
    payload["delta"] = -5.0
    restored = PairedScore.model_validate(payload)

    assert restored.delta == pytest.approx(0.7)
    assert PairedScore.model_validate_json(pair.model_dump_json()) == pair


def test_unnamed_cases_are_paired_by_input_regardless_of_order():
    baseline = _report(
        [
            _row(name=None, score=0.25, input_value={"prompt": "hello"}, session_id="baseline"),
            _row(name=None, score=0.5, input_value={"prompt": "goodbye"}, session_id="baseline"),
        ]
    )
    candidate = _report(
        [
            _row(name=None, score=0.75, input_value={"prompt": "goodbye"}, session_id="candidate"),
            _row(name=None, score=1.0, input_value={"prompt": "hello"}, session_id="candidate"),
        ]
    )

    paired = pair_reports(baseline=baseline, candidate=candidate)

    assert len(paired) == 2
    assert all(pair.case_id.startswith("input:") for pair in paired)
    assert sorted(pair.delta for pair in paired) == pytest.approx([0.25, 0.75])


def test_omits_rows_that_are_not_applicable_in_both_reports():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE, reason="nothing to judge")]
    baseline = _report([_row(name="unused", score=0.0, details=not_applicable)])
    candidate = _report([_row(name="unused", score=0.0, details=not_applicable)])

    assert pair_reports(baseline=baseline, candidate=candidate) == []


def test_rejects_applicability_mismatch():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline = _report([_row(name="case", score=0.0, details=not_applicable)])
    candidate = _report([_row(name="case", score=1.0)])

    with pytest.raises(ValueError, match="applicability differs"):
        pair_reports(baseline=baseline, candidate=candidate)


def test_named_cases_pair_by_name_when_runtime_inputs_differ():
    baseline = _report([_row(name="case", score=0.0, input_value={"when": date(2026, 1, 1)})])
    candidate = _report([_row(name="case", score=1.0, input_value={"payload": b"candidate"})])

    paired = pair_reports(baseline=baseline, candidate=candidate)

    assert len(paired) == 1
    assert paired[0].case_id == "case"
    assert paired[0].delta == 1.0


def test_unnamed_case_pairs_after_report_file_round_trip(tmp_path):
    report = _report(
        [
            _row(
                name=None,
                score=0.5,
                input_value={"docs": {1: "first", 2: "second", 10: "tenth"}},
            )
        ]
    )
    path = tmp_path / "report.json"
    report.to_file(str(path))
    restored = EvaluationReport.from_file(str(path))

    paired = pair_reports(baseline=report, candidate=restored)

    assert len(paired) == 1
    assert paired[0].delta == 0.0


def test_pairs_reports_produced_by_experiment():
    cases = [
        Case(name="case-a", input="question-a", expected_output="answer-a"),
        Case(name="case-b", input="question-b", expected_output="answer-b"),
    ]
    experiment = Experiment(cases=cases, evaluators=[Equals(name="equals")])
    baseline = experiment.run_evaluations(lambda case: case.expected_output)
    candidate = experiment.run_evaluations(lambda case: case.expected_output if case.name == "case-a" else "wrong")

    paired = pair_reports(baseline=baseline, candidate=candidate)

    assert [pair.case_id for pair in paired] == ["case-a", "case-b"]
    assert [pair.baseline_score for pair in paired] == [1.0, 1.0]
    assert [pair.candidate_score for pair in paired] == [1.0, 0.0]
    assert [pair.delta for pair in paired] == [0.0, -1.0]


def test_rejects_missing_case_evaluator_rows():
    baseline = _report([_row(name="case-a", score=0.0)])
    candidate = _report([_row(name="case-b", score=1.0)])

    with pytest.raises(ValueError, match="same case/evaluator rows"):
        pair_reports(baseline=baseline, candidate=candidate)


def test_rejects_duplicate_case_evaluator_rows():
    baseline = _report([_row(name="case", score=0.0), _row(name="case", score=1.0)])

    with pytest.raises(ValueError, match="duplicate case/evaluator row"):
        pair_reports(baseline=baseline, candidate=baseline)


def test_rejects_missing_evaluator_name():
    baseline = _report([_row(name="case", evaluator="", score=0.0)])

    with pytest.raises(ValueError, match="non-empty evaluator name"):
        pair_reports(baseline=baseline, candidate=baseline)


def test_rejects_report_shape_mismatch():
    baseline = _report([_row(name="case", score=0.0)])
    baseline.scores = []

    with pytest.raises(ValueError, match="1 cases but 0 scores"):
        pair_reports(baseline=baseline, candidate=baseline)


def test_pairs_and_aggregates_reports_across_trials():
    baseline_reports = [
        _report(
            [
                _row(name="case-b", evaluator="correctness", score=0.3),
                _row(name="case-a", evaluator="correctness", score=0.1),
            ]
        ),
        _report(
            [
                _row(name="case-a", evaluator="correctness", score=0.5),
                _row(name="case-b", evaluator="correctness", score=0.7),
            ]
        ),
    ]
    candidate_reports = [
        _report(
            [
                _row(name="case-a", evaluator="correctness", score=0.3),
                _row(name="case-b", evaluator="correctness", score=0.4),
            ]
        ),
        _report(
            [
                _row(name="case-b", evaluator="correctness", score=0.9),
                _row(name="case-a", evaluator="correctness", score=0.6),
            ]
        ),
    ]

    trial_pairs = pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)
    aggregated = aggregate_trial_pairs(trial_pairs)
    summary = summarize_pairs(aggregated)[0]

    assert [(pair.case_id, pair.trial_index) for pair in trial_pairs] == [
        ("case-a", 0),
        ("case-a", 1),
        ("case-b", 0),
        ("case-b", 1),
    ]
    assert [pair.delta for pair in trial_pairs] == pytest.approx([0.2, 0.1, 0.1, 0.2])
    assert [pair.case_id for pair in aggregated] == ["case-a", "case-b"]
    assert [pair.baseline_score for pair in aggregated] == pytest.approx([0.3, 0.5])
    assert [pair.candidate_score for pair in aggregated] == pytest.approx([0.45, 0.65])
    assert [pair.delta for pair in aggregated] == pytest.approx([0.15, 0.15])
    assert all(pair.baseline_index is None for pair in aggregated)
    assert all(pair.candidate_index is None for pair in aggregated)
    assert [pair.baseline_trial_count for pair in aggregated] == [2, 2]
    assert [pair.candidate_trial_count for pair in aggregated] == [2, 2]
    assert summary.case_count == 2
    assert summary.mean_delta == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("baseline_reports", "candidate_reports", "message"),
    [
        ([], [], "at least one trial"),
        (
            [_report([_row(name="case", score=0.0)])],
            [
                _report([_row(name="case", score=0.0)]),
                _report([_row(name="case", score=0.0)]),
            ],
            "same number of trials",
        ),
    ],
)
def test_rejects_invalid_trial_report_counts(
    baseline_reports: list[EvaluationReport],
    candidate_reports: list[EvaluationReport],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)


def test_rejects_case_evaluator_row_changes_across_trials():
    baseline_reports = [
        _report([_row(name="case-a", score=0.1)]),
        _report([_row(name="case-b", score=0.2)]),
    ]
    candidate_reports = [
        _report([_row(name="case-a", score=0.3)]),
        _report([_row(name="case-b", score=0.4)]),
    ]

    with pytest.raises(ValueError, match="trial_index=1.*missing=.*case-a.*unexpected=.*case-b"):
        pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)


def test_allows_applicability_changes_across_trials():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline_reports = [
        _report([_row(name="case", score=0.0, details=not_applicable)]),
        _report([_row(name="case", score=0.2)]),
        _report([_row(name="case", score=0.4)]),
    ]
    candidate_reports = [
        _report([_row(name="case", score=0.6)]),
        _report([_row(name="case", score=0.0, details=not_applicable)]),
        _report([_row(name="case", score=0.8)]),
    ]

    trial_pairs = pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)
    aggregated = aggregate_trial_pairs(trial_pairs)
    reordered_pairs = pair_report_trials(
        baseline_reports=baseline_reports,
        candidate_reports=[candidate_reports[1], candidate_reports[0], candidate_reports[2]],
    )
    reordered = aggregate_trial_pairs(reordered_pairs)

    assert [(pair.case_id, pair.trial_index) for pair in trial_pairs] == [
        ("case", 0),
        ("case", 1),
        ("case", 2),
    ]
    assert [pair.delta for pair in trial_pairs] == [None, None, pytest.approx(0.4)]
    assert len(aggregated) == 1
    assert aggregated[0].baseline_score == pytest.approx(0.3)
    assert aggregated[0].candidate_score == pytest.approx(0.7)
    assert aggregated[0].baseline_trial_count == 2
    assert aggregated[0].candidate_trial_count == 2
    assert reordered == aggregated


def test_rejects_when_one_side_is_never_applicable():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline_reports = [
        _report([_row(name="case", evaluator="correctness", score=0.0, details=not_applicable)]),
        _report([_row(name="case", evaluator="correctness", score=0.0, details=not_applicable)]),
    ]
    candidate_reports = [
        _report([_row(name="case", evaluator="correctness", score=0.5)]),
        _report([_row(name="case", evaluator="correctness", score=1.0)]),
    ]
    trial_pairs = pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)

    with pytest.raises(ValueError, match="no applicable trials.*missing='baseline'"):
        aggregate_trial_pairs(trial_pairs)


def test_tracks_full_and_partial_trial_support():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline_reports = [
        _report(
            [
                _row(name="full", score=0.1),
                _row(name="partial", score=0.0, details=not_applicable),
            ]
        ),
        _report([_row(name="full", score=0.2), _row(name="partial", score=0.4)]),
        _report([_row(name="full", score=0.3), _row(name="partial", score=0.6)]),
    ]
    candidate_reports = [
        _report([_row(name="full", score=0.2), _row(name="partial", score=0.5)]),
        _report(
            [
                _row(name="full", score=0.3),
                _row(name="partial", score=0.0, details=not_applicable),
            ]
        ),
        _report([_row(name="full", score=0.4), _row(name="partial", score=0.9)]),
    ]

    trial_pairs = pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)
    aggregated = aggregate_trial_pairs(trial_pairs)

    assert [(pair.case_id, pair.trial_index) for pair in trial_pairs] == [
        ("full", 0),
        ("full", 1),
        ("full", 2),
        ("partial", 0),
        ("partial", 1),
        ("partial", 2),
    ]
    assert [(pair.case_id, pair.baseline_score, pair.candidate_score) for pair in trial_pairs] == [
        ("full", 0.1, 0.2),
        ("full", 0.2, 0.3),
        ("full", 0.3, 0.4),
        ("partial", None, 0.5),
        ("partial", 0.4, None),
        ("partial", 0.6, 0.9),
    ]
    assert [(pair.case_id, pair.baseline_trial_count, pair.candidate_trial_count) for pair in aggregated] == [
        ("full", 3, 3),
        ("partial", 2, 2),
    ]
    assert aggregated[1].baseline_score == pytest.approx(0.5)
    assert aggregated[1].candidate_score == pytest.approx(0.7)


def test_trial_score_is_distinct_from_aggregated_score_and_validates_index():
    trial_pair = PairedTrialScore(
        case_id="case",
        evaluator="correctness",
        baseline_index=0,
        candidate_index=1,
        baseline_score=0.1,
        candidate_score=0.2,
        trial_index=0,
    )

    assert not isinstance(trial_pair, PairedScore)
    assert trial_pair.delta == pytest.approx(0.1)
    with pytest.raises(ValidationError):
        trial_pair.candidate_score = 0.9
    with pytest.raises(ValidationError):
        PairedTrialScore(
            case_id="case",
            evaluator="correctness",
            baseline_index=0,
            candidate_index=1,
            baseline_score=0.1,
            candidate_score=0.2,
            trial_index=-1,
        )


def test_rejects_duplicate_trial_pairs():
    pair = PairedTrialScore(
        case_id="case-a",
        evaluator="correctness",
        baseline_index=0,
        candidate_index=0,
        baseline_score=0.1,
        candidate_score=0.2,
        trial_index=0,
    )

    with pytest.raises(ValueError, match="duplicate trial/case/evaluator row"):
        aggregate_trial_pairs([pair, pair])


def test_aggregate_trial_pairs_returns_empty_list():
    assert aggregate_trial_pairs([]) == []


def test_orders_multiple_evaluators_and_omits_always_not_applicable_rows():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline_reports = [
        _report(
            [
                _row(name="case", evaluator="helpfulness", score=0.2),
                _row(name="case", evaluator="correctness", score=0.4),
                _row(name="case", evaluator="safety", score=0.0, details=not_applicable),
            ]
        ),
        _report(
            [
                _row(name="case", evaluator="safety", score=0.0, details=not_applicable),
                _row(name="case", evaluator="correctness", score=0.6),
                _row(name="case", evaluator="helpfulness", score=0.8),
            ]
        ),
    ]
    candidate_reports = [
        _report(
            [
                _row(name="case", evaluator="correctness", score=0.5),
                _row(name="case", evaluator="safety", score=0.0, details=not_applicable),
                _row(name="case", evaluator="helpfulness", score=0.3),
            ]
        ),
        _report(
            [
                _row(name="case", evaluator="helpfulness", score=0.9),
                _row(name="case", evaluator="correctness", score=0.7),
                _row(name="case", evaluator="safety", score=0.0, details=not_applicable),
            ]
        ),
    ]

    trial_pairs = pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)
    aggregated = aggregate_trial_pairs(trial_pairs)

    assert [(pair.evaluator, pair.case_id) for pair in aggregated] == [
        ("correctness", "case"),
        ("helpfulness", "case"),
    ]


def test_removes_explicitly_corrupted_trials_before_pairing():
    cases = [
        Case(name="case-a", input="input-a", expected_output="answer-a"),
        Case(name="case-b", input="input-b", expected_output="answer-b"),
    ]

    def run(crash_on: str | None = None) -> EvaluationReport:
        def task(case: Case[str, str]) -> dict[str, object]:
            if case.name == crash_on:
                raise ConnectionError("runtime unreachable")
            return {"output": case.expected_output, "trajectory": _session()}

        experiment = Experiment(
            cases=cases,
            evaluators=[
                Equals(name="equals"),
                SkillSelectionAccuracyEvaluator(name="skill"),
            ],
        )
        return experiment.run_evaluations(task)

    baseline_reports = [run(), run(), run()]
    candidate_reports = [run(), run(crash_on="case-a"), run()]
    candidate_reports[1].cases[0].pop("input")

    with pytest.raises(ValueError, match="trial 1"):
        pair_report_trials(baseline_reports=baseline_reports, candidate_reports=candidate_reports)

    pairs = pair_report_trials(
        baseline_reports=baseline_reports,
        candidate_reports=candidate_reports,
        corrupted_trial_indices={1},
    )
    aggregated = aggregate_trial_pairs(pairs)

    assert {pair.trial_index for pair in pairs} == {0, 2}
    assert [(pair.evaluator, pair.case_id) for pair in aggregated] == [
        ("equals", "case-a"),
        ("equals", "case-b"),
    ]


def test_rejects_unknown_or_all_corrupted_trials():
    report = _report([_row(name="case", score=0.1)])

    with pytest.raises(ValueError, match="unknown trials"):
        pair_report_trials(baseline_reports=[report], candidate_reports=[report], corrupted_trial_indices={1})
    with pytest.raises(ValueError, match="no usable trials"):
        pair_report_trials(baseline_reports=[report], candidate_reports=[report], corrupted_trial_indices={0})
