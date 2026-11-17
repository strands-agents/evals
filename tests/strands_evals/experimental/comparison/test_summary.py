import pytest

from strands_evals.experimental.comparison import PairedScore, summarize_pairs


def _pair(*, case_id: str, evaluator: str, baseline: float, candidate: float) -> PairedScore:
    return PairedScore(
        case_id=case_id,
        evaluator=evaluator,
        baseline_index=0,
        candidate_index=0,
        baseline_score=baseline,
        candidate_score=candidate,
    )


def test_summarizes_case_deltas_by_evaluator():
    summaries = summarize_pairs(
        [
            _pair(case_id="case-a", evaluator="helpfulness", baseline=0.8, candidate=0.7),
            _pair(case_id="case-b", evaluator="correctness", baseline=0.2, candidate=0.4),
            _pair(case_id="case-a", evaluator="correctness", baseline=0.1, candidate=0.3),
        ]
    )

    assert [summary.evaluator for summary in summaries] == ["correctness", "helpfulness"]
    assert summaries[0].case_count == 2
    assert summaries[0].mean_delta == pytest.approx(0.2)
    assert summaries[0].confidence_lower == pytest.approx(0.2)
    assert summaries[0].confidence_upper == pytest.approx(0.2)
    assert summaries[1].case_count == 1
    assert summaries[1].mean_delta == pytest.approx(-0.1)
    assert summaries[1].confidence_lower is None
    assert summaries[1].confidence_upper is None


def test_t_interval_reflects_case_variation():
    summary = summarize_pairs(
        [
            _pair(case_id="case-a", evaluator="correctness", baseline=0.0, candidate=1.0),
            _pair(case_id="case-b", evaluator="correctness", baseline=0.0, candidate=2.0),
            _pair(case_id="case-c", evaluator="correctness", baseline=0.0, candidate=3.0),
        ]
    )[0]

    assert summary.mean_delta == 2.0
    assert summary.confidence_lower == pytest.approx(-0.4841377117)
    assert summary.confidence_upper == pytest.approx(4.4841377117)


def test_confidence_level_controls_interval_width():
    pairs = [
        _pair(case_id="case-a", evaluator="correctness", baseline=0.0, candidate=1.0),
        _pair(case_id="case-b", evaluator="correctness", baseline=0.0, candidate=2.0),
        _pair(case_id="case-c", evaluator="correctness", baseline=0.0, candidate=3.0),
    ]

    narrow = summarize_pairs(pairs, confidence_level=0.5)[0]
    wide = summarize_pairs(list(reversed(pairs)), confidence_level=0.95)[0]

    assert narrow.confidence_lower is not None
    assert wide.confidence_lower is not None
    assert narrow.confidence_upper is not None
    assert wide.confidence_upper is not None
    assert narrow.confidence_lower > wide.confidence_lower
    assert narrow.confidence_upper < wide.confidence_upper


def test_returns_empty_list_when_there_are_no_pairs():
    assert summarize_pairs([]) == []


def test_rejects_duplicate_case_evaluator_rows():
    pair = _pair(case_id="case", evaluator="correctness", baseline=0.1, candidate=0.2)

    with pytest.raises(ValueError, match="duplicate case/evaluator row"):
        summarize_pairs([pair, pair])


@pytest.mark.parametrize("confidence_level", [0.0, 1.0])
def test_rejects_invalid_confidence_level(confidence_level: float):
    with pytest.raises(ValueError, match="confidence_level"):
        summarize_pairs([], confidence_level=confidence_level)


def test_rejects_non_finite_deltas():
    pair = _pair(case_id="case", evaluator="correctness", baseline=0.0, candidate=float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        summarize_pairs([pair])
