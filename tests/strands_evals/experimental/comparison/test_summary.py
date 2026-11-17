from types import SimpleNamespace

import pytest

import strands_evals.experimental.comparison.summary as summary_module
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
    assert summaries[0].wilcoxon_p_value == pytest.approx(0.5)
    assert summaries[0].adjusted_p_value == pytest.approx(0.5)
    assert summaries[1].case_count == 1
    assert summaries[1].mean_delta == pytest.approx(-0.1)
    assert summaries[1].confidence_lower is None
    assert summaries[1].confidence_upper is None
    assert summaries[1].wilcoxon_p_value is None
    assert summaries[1].adjusted_p_value is None


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


def test_adjusts_wilcoxon_p_values_across_evaluators():
    pairs = [
        _pair(case_id=f"case-{index}", evaluator="consistent", baseline=0.1, candidate=candidate)
        for index, candidate in enumerate([0.2, 0.3, 0.4, 0.5])
    ]
    pairs.extend(
        _pair(case_id=f"case-{index}", evaluator="mixed", baseline=0.5, candidate=candidate)
        for index, candidate in enumerate([0.6, 0.7, 0.2, 0.9])
    )

    summaries = summarize_pairs(pairs)

    assert [summary.wilcoxon_p_value for summary in summaries] == pytest.approx([0.125, 0.625])
    assert [summary.adjusted_p_value for summary in summaries] == pytest.approx([0.25, 0.625])


def test_all_zero_deltas_have_p_value_one():
    summary = summarize_pairs(
        [_pair(case_id=f"case-{index}", evaluator="correctness", baseline=0.2, candidate=0.2) for index in range(20)]
    )[0]

    assert summary.wilcoxon_p_value == 1.0
    assert summary.adjusted_p_value == 1.0


@pytest.mark.parametrize(
    ("baseline_scores", "candidate_scores"),
    [
        ([2 / 3] * 6 + [1 / 3] * 2, [1.0] * 6 + [0.0] * 2),
        ([1 / 3] * 6 + [2 / 3] * 2, [2 / 3] * 6 + [1 / 3] * 2),
        ([0.0] * 6 + [1.0] * 2, [1 / 3] * 6 + [2 / 3] * 2),
    ],
)
def test_rounds_equivalent_deltas_before_wilcoxon(
    baseline_scores: list[float],
    candidate_scores: list[float],
):
    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=baseline,
                candidate=candidate,
            )
            for index, (baseline, candidate) in enumerate(zip(baseline_scores, candidate_scores, strict=True))
        ]
    )[0]

    assert summary.mean_delta == pytest.approx(1 / 6)
    assert summary.wilcoxon_p_value == pytest.approx(0.2890625)


@pytest.mark.parametrize("case_count", [13, 14, 50])
def test_unchanged_cases_do_not_change_wilcoxon_method(case_count: int):
    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0 if index < 4 else 1.0,
                candidate=1.0,
            )
            for index in range(case_count)
        ]
    )[0]

    assert summary.wilcoxon_p_value == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("improved_count", "regressed_count", "expected_p_value"),
    [
        (11, 3, 0.057373046875),
        (12, 4, 0.076812744140625),
    ],
)
def test_equal_magnitude_changes_use_exact_sign_test(
    improved_count: int,
    regressed_count: int,
    expected_p_value: float,
):
    deltas = [1.0] * improved_count + [-1.0] * regressed_count
    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0,
                candidate=delta,
            )
            for index, delta in enumerate(deltas)
        ]
    )[0]

    assert summary.wilcoxon_p_value == pytest.approx(expected_p_value)


def test_small_tied_sample_uses_exact_distribution(monkeypatch: pytest.MonkeyPatch):
    def fail_wilcoxon(*args: object, **kwargs: object) -> None:
        pytest.fail("tied samples within the cutoff must not use scipy's approximation")

    monkeypatch.setattr(summary_module, "wilcoxon", fail_wilcoxon)

    summary = summarize_pairs(
        [
            _pair(case_id="case-a", evaluator="correctness", baseline=0.0, candidate=1.0),
            _pair(case_id="case-b", evaluator="correctness", baseline=0.0, candidate=1.0),
            _pair(case_id="case-c", evaluator="correctness", baseline=0.0, candidate=-2.0),
        ]
    )[0]

    assert summary.wilcoxon_p_value == 1.0


def test_small_tied_permutation_test_matches_exact_result():
    deltas = [1.0] * 4 + [2.0] * 3 + [3.0] * 2 + [-1.0] * 2 + [-2.0] * 2 + [-3.0]

    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0,
                candidate=delta,
            )
            for index, delta in enumerate(deltas)
        ]
    )[0]

    assert summary.wilcoxon_p_value == pytest.approx(0.4012451171875)


def test_exact_tied_distribution_fixes_near_binary_false_positive():
    deltas = [1.0] * 5 + [-1.0] * 14 + [0.5]

    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0,
                candidate=delta,
            )
            for index, delta in enumerate(deltas)
        ]
    )[0]

    assert summary.wilcoxon_p_value == pytest.approx(0.063568115234375)


def test_exact_tied_distribution_includes_cutoff(monkeypatch: pytest.MonkeyPatch):
    changed_counts: list[int] = []

    def fake_exact(changed: list[float]) -> float:
        changed_counts.append(len(changed))
        return 0.5

    monkeypatch.setattr(summary_module, "_exact_tied_wilcoxon_p_value", fake_exact)
    deltas = [1.0] * 499 + [-2.0]

    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0,
                candidate=delta,
            )
            for index, delta in enumerate(deltas)
        ]
    )[0]

    assert summary.wilcoxon_p_value == 0.5
    assert changed_counts == [500]


def test_large_tied_sample_uses_deterministic_approximation(monkeypatch: pytest.MonkeyPatch):
    methods: list[str] = []

    def fake_wilcoxon(
        values: list[float],
        *,
        alternative: str,
        method: str,
    ) -> SimpleNamespace:
        assert len(values) == 501
        assert alternative == "two-sided"
        methods.append(method)
        return SimpleNamespace(pvalue=0.5)

    monkeypatch.setattr(summary_module, "wilcoxon", fake_wilcoxon)
    deltas = [1.0] * 500 + [-2.0]

    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.0,
                candidate=delta,
            )
            for index, delta in enumerate(deltas)
        ]
    )[0]

    assert summary.wilcoxon_p_value == 0.5
    assert methods == ["approx"]


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
