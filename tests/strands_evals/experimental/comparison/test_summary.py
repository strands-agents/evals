from types import SimpleNamespace

import pytest

import strands_evals.experimental.comparison.summary as summary_module
from strands_evals.experimental.comparison.pairing import PairedScore
from strands_evals.experimental.comparison.summary import summarize_pairs


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
    assert summaries[0].changed_case_count == 2
    assert summaries[0].higher_is_better is True
    assert summaries[0].minimum_effect == 0.0
    assert summaries[0].mean_delta == pytest.approx(0.2)
    assert summaries[0].confidence_lower == pytest.approx(0.2)
    assert summaries[0].confidence_upper == pytest.approx(0.2)
    assert summaries[0].wilcoxon_p_value == pytest.approx(0.5)
    assert summaries[0].adjusted_p_value == pytest.approx(0.5)
    assert summaries[0].verdict == "insufficient_data"
    assert summaries[1].case_count == 1
    assert summaries[1].mean_delta == pytest.approx(-0.1)
    assert summaries[1].confidence_lower is None
    assert summaries[1].confidence_upper is None
    assert summaries[1].wilcoxon_p_value is None
    assert summaries[1].adjusted_p_value is None
    assert summaries[1].verdict == "insufficient_data"


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
    assert summary.verdict == "no_clear_change"


@pytest.mark.parametrize(
    ("candidate_scores", "expected_verdict"),
    [
        ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "improved"),
        ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "regressed"),
    ],
)
def test_assigns_evidence_based_verdicts(candidate_scores: list[float], expected_verdict: str):
    baseline_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
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

    assert summary.verdict == expected_verdict


@pytest.mark.parametrize(
    ("delta", "lower_is_better", "expected_without_threshold"),
    [
        (0.1, False, "improved"),
        (-0.1, False, "regressed"),
        (0.1, True, "regressed"),
        (-0.1, True, "improved"),
    ],
)
def test_minimum_effect_requires_entire_interval_beyond_threshold(
    delta: float,
    lower_is_better: bool,
    expected_without_threshold: str,
):
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.5,
            candidate=0.5 + delta,
        )
        for index in range(6)
    ]
    lower_names = {"correctness"} if lower_is_better else set()

    without_threshold = summarize_pairs(pairs, minimum_effect=0.05, lower_is_better=lower_names)[0]
    with_threshold = summarize_pairs(pairs, minimum_effect=0.11, lower_is_better=lower_names)[0]

    assert without_threshold.verdict == expected_without_threshold
    assert with_threshold.minimum_effect == 0.11
    assert with_threshold.verdict == "no_clear_change"


@pytest.mark.parametrize(("delta", "lower_is_better"), [(0.25, False), (-0.25, True)])
def test_minimum_effect_boundary_is_not_directional(delta: float, lower_is_better: bool):
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.5,
            candidate=0.5 + delta,
        )
        for index in range(6)
    ]
    lower_names = {"correctness"} if lower_is_better else set()

    summary = summarize_pairs(pairs, minimum_effect=0.25, lower_is_better=lower_names)[0]

    assert summary.confidence_lower == pytest.approx(delta)
    assert summary.confidence_upper == pytest.approx(delta)
    assert summary.verdict == "no_clear_change"


@pytest.mark.parametrize("lower_is_better", [False, True])
def test_interval_must_exclude_zero_for_directional_verdict(lower_is_better: bool):
    deltas = [0.01] * 7 + [5.0]
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=delta,
        )
        for index, delta in enumerate(deltas)
    ]
    lower_names = {"correctness"} if lower_is_better else set()

    summary = summarize_pairs(pairs, lower_is_better=lower_names)[0]

    assert summary.mean_delta > 0.0
    assert summary.confidence_lower is not None
    assert summary.confidence_lower < 0.0
    assert summary.verdict == "no_clear_change"


@pytest.mark.parametrize("lower_is_better", [False, True])
def test_interval_upper_bound_prevents_directional_verdict(lower_is_better: bool):
    deltas = [-0.01] * 7 + [-5.0]
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=delta,
        )
        for index, delta in enumerate(deltas)
    ]
    lower_names = {"correctness"} if lower_is_better else set()

    summary = summarize_pairs(pairs, lower_is_better=lower_names)[0]

    assert summary.mean_delta < 0.0
    assert summary.confidence_upper is not None
    assert summary.confidence_upper > 0.0
    assert summary.verdict == "no_clear_change"


def test_significance_threshold_boundary_is_inclusive():
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=0.1,
        )
        for index in range(6)
    ]

    summary = summarize_pairs(pairs, confidence_level=0.96875)[0]

    assert summary.adjusted_p_value == 0.03125
    assert 1.0 - summary.confidence_level == summary.adjusted_p_value
    assert summary.verdict == "improved"


def test_constant_interval_preserves_rounded_delta_exactly():
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.5,
            candidate=0.6,
        )
        for index in range(6)
    ]

    summary = summarize_pairs(pairs)[0]

    assert summary.confidence_lower == 0.1
    assert summary.confidence_upper == 0.1


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


def test_large_equal_magnitude_changes_use_exact_sign_test():
    deltas = [1.0] * 13 + [-1.0] * 4

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

    assert summary.wilcoxon_p_value == pytest.approx(0.049041748046875)


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


def test_rounds_tiny_deltas_before_assigning_verdict():
    summary = summarize_pairs(
        [
            _pair(
                case_id=f"case-{index}",
                evaluator="correctness",
                baseline=0.8,
                candidate=0.8 + 2.2e-16,
            )
            for index in range(8)
        ]
    )[0]

    assert summary.changed_case_count == 0
    assert summary.mean_delta == 0.0
    assert summary.verdict == "no_clear_change"


def test_supports_lower_is_better_evaluators():
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="attack_success",
            baseline=0.8,
            candidate=0.2,
        )
        for index in range(8)
    ]

    default_summary = summarize_pairs(pairs)[0]
    lower_is_better_summary = summarize_pairs(pairs, lower_is_better={"attack_success"})[0]

    assert default_summary.higher_is_better is True
    assert default_summary.verdict == "regressed"
    assert lower_is_better_summary.higher_is_better is False
    assert lower_is_better_summary.verdict == "improved"


def test_rejects_bare_or_unknown_lower_is_better_names():
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="attack_success",
            baseline=0.8,
            candidate=0.2,
        )
        for index in range(8)
    ]

    with pytest.raises(TypeError, match="not a string"):
        summarize_pairs(pairs, lower_is_better="attack_success")
    with pytest.raises(ValueError, match="unknown=.*attack_sucess.*known=.*attack_success"):
        summarize_pairs(pairs, lower_is_better={"attack_sucess"})


def test_distinguishes_insufficient_data_from_no_clear_change():
    five_changes = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=1.0,
        )
        for index in range(5)
    ]

    default_summary = summarize_pairs(five_changes)[0]
    lower_confidence_summary = summarize_pairs(five_changes, confidence_level=0.9)[0]

    assert default_summary.changed_case_count == 5
    assert default_summary.verdict == "insufficient_data"
    assert lower_confidence_summary.verdict == "improved"


def test_changed_case_count_controls_testability():
    pairs = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=1.0 if index < 4 else 0.0,
        )
        for index in range(20)
    ]

    summary = summarize_pairs(pairs)[0]

    assert summary.case_count == 20
    assert summary.changed_case_count == 4
    assert summary.verdict == "insufficient_data"


@pytest.mark.parametrize(
    ("delta", "standalone_verdict"),
    [(0.1, "improved"), (-0.1, "regressed")],
)
def test_adjusted_p_value_controls_data_sufficiency(
    delta: float,
    standalone_verdict: str,
):
    primary = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=delta,
        )
        for index in range(6)
    ]
    unchanged = [
        _pair(
            case_id=f"case-{index}",
            evaluator="helpfulness",
            baseline=0.5,
            candidate=0.5,
        )
        for index in range(6)
    ]

    alone = summarize_pairs(primary)[0]
    with_family = summarize_pairs(primary + unchanged)[0]

    assert alone.adjusted_p_value == pytest.approx(0.03125)
    assert alone.verdict == standalone_verdict
    assert with_family.adjusted_p_value == pytest.approx(0.0625)
    assert with_family.verdict == "insufficient_data"


def test_adjusted_p_value_controls_directional_verdict():
    primary_deltas = [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    noisy_deltas = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    primary = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=delta,
        )
        for index, delta in enumerate(primary_deltas)
    ]
    noisy = [
        _pair(
            case_id=f"case-{index}",
            evaluator="helpfulness",
            baseline=0.0,
            candidate=delta,
        )
        for index, delta in enumerate(noisy_deltas)
    ]

    alone = summarize_pairs(primary)[0]
    with_family = summarize_pairs(primary + noisy)[0]

    assert alone.wilcoxon_p_value == pytest.approx(0.0390625)
    assert alone.adjusted_p_value == pytest.approx(0.0390625)
    assert alone.verdict == "improved"
    assert with_family.adjusted_p_value == pytest.approx(0.078125)
    assert with_family.verdict == "no_clear_change"


def test_data_sufficiency_uses_each_evaluators_minimum_adjusted_p_value():
    strong = [
        _pair(
            case_id=f"case-{index}",
            evaluator="correctness",
            baseline=0.0,
            candidate=1.0,
        )
        for index in range(10)
    ]
    sparse = [
        _pair(
            case_id=f"case-{index}",
            evaluator="helpfulness",
            baseline=0.0,
            candidate=1.0 if index < 3 else 0.0,
        )
        for index in range(10)
    ]

    summaries = summarize_pairs(strong + sparse)

    assert summaries[0].verdict == "improved"
    assert summaries[1].changed_case_count == 3
    assert summaries[1].verdict == "insufficient_data"


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


@pytest.mark.parametrize("minimum_effect", [-0.1, float("nan"), float("inf")])
def test_rejects_invalid_minimum_effect(minimum_effect: float):
    with pytest.raises(ValueError, match="minimum_effect"):
        summarize_pairs([], minimum_effect=minimum_effect)
