import pytest

from strands_evals.experimental.comparison import PairedScore, summarize_pairs


def _pair(*, case_id: str, evaluator: str, baseline: float, candidate: float) -> PairedScore:
    return PairedScore(
        case_id=case_id,
        evaluator=evaluator,
        baseline_score=baseline,
        candidate_score=candidate,
        delta=candidate - baseline,
    )


def test_summarizes_case_deltas_by_evaluator():
    summaries = summarize_pairs(
        [
            _pair(case_id="case-b", evaluator="correctness", baseline=0.2, candidate=0.4),
            _pair(case_id="case-a", evaluator="helpfulness", baseline=0.8, candidate=0.7),
            _pair(case_id="case-a", evaluator="correctness", baseline=0.1, candidate=0.3),
        ],
        bootstrap_samples=100,
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


def test_bootstrap_interval_reflects_case_variation():
    summary = summarize_pairs(
        [
            _pair(case_id="case-a", evaluator="correctness", baseline=0.1, candidate=0.2),
            _pair(case_id="case-b", evaluator="correctness", baseline=0.2, candidate=0.5),
            _pair(case_id="case-c", evaluator="correctness", baseline=0.8, candidate=0.4),
        ],
        bootstrap_samples=500,
        seed=7,
    )[0]

    assert summary.confidence_lower is not None
    assert summary.confidence_upper is not None
    assert summary.confidence_lower < summary.mean_delta < summary.confidence_upper


def test_bootstrap_interval_is_deterministic_regardless_of_input_order():
    pairs = [
        _pair(case_id="case-a", evaluator="correctness", baseline=0.1, candidate=0.2),
        _pair(case_id="case-b", evaluator="correctness", baseline=0.2, candidate=0.5),
        _pair(case_id="case-c", evaluator="correctness", baseline=0.8, candidate=0.4),
    ]

    forward = summarize_pairs(pairs, bootstrap_samples=200, seed=7)
    reverse = summarize_pairs(list(reversed(pairs)), bootstrap_samples=200, seed=7)

    assert forward == reverse


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


def test_rejects_invalid_bootstrap_sample_count():
    with pytest.raises(ValueError, match="bootstrap_samples"):
        summarize_pairs([], bootstrap_samples=0)
