import pytest

from strands_evals.experimental.comparison import pair_reports
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


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

    paired = pair_reports(baseline, candidate)

    assert [(pair.evaluator, pair.case_id) for pair in paired] == [
        ("correctness", "case-a"),
        ("correctness", "case-b"),
        ("helpfulness", "case-a"),
    ]
    assert [pair.delta for pair in paired] == pytest.approx([0.3, -0.2, 0.2])


def test_unnamed_cases_are_paired_by_input():
    baseline = _report([_row(name=None, score=0.25, input_value={"prompt": "hello"}, session_id="baseline")])
    candidate = _report([_row(name=None, score=0.75, input_value={"prompt": "hello"}, session_id="candidate")])

    paired = pair_reports(baseline, candidate)

    assert len(paired) == 1
    assert paired[0].case_id.startswith("input:")
    assert paired[0].delta == 0.5


def test_omits_rows_that_are_not_applicable_in_both_reports():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE, reason="nothing to judge")]
    baseline = _report([_row(name="unused", score=0.0, details=not_applicable)])
    candidate = _report([_row(name="unused", score=0.0, details=not_applicable)])

    assert pair_reports(baseline, candidate) == []


def test_rejects_applicability_mismatch():
    not_applicable = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    baseline = _report([_row(name="case", score=0.0, details=not_applicable)])
    candidate = _report([_row(name="case", score=1.0)])

    with pytest.raises(ValueError, match="applicability differs"):
        pair_reports(baseline, candidate)


def test_rejects_same_named_case_with_different_inputs():
    baseline = _report([_row(name="case", score=0.0, input_value="first")])
    candidate = _report([_row(name="case", score=1.0, input_value="second")])

    with pytest.raises(ValueError, match="case inputs differ"):
        pair_reports(baseline, candidate)


def test_rejects_missing_case_evaluator_rows():
    baseline = _report([_row(name="case-a", score=0.0)])
    candidate = _report([_row(name="case-b", score=1.0)])

    with pytest.raises(ValueError, match="same case/evaluator rows"):
        pair_reports(baseline, candidate)


def test_rejects_duplicate_case_evaluator_rows():
    baseline = _report([_row(name="case", score=0.0), _row(name="case", score=1.0)])

    with pytest.raises(ValueError, match="duplicate case/evaluator row"):
        pair_reports(baseline, baseline)


def test_rejects_missing_evaluator_name():
    baseline = _report([_row(name="case", evaluator="", score=0.0)])

    with pytest.raises(ValueError, match="non-empty evaluator name"):
        pair_reports(baseline, baseline)


def test_rejects_report_shape_mismatch():
    baseline = _report([_row(name="case", score=0.0)])
    baseline.scores = []

    with pytest.raises(ValueError, match="1 cases but 0 scores"):
        pair_reports(baseline, baseline)
