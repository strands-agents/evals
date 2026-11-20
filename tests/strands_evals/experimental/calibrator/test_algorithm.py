from dataclasses import replace
from types import MappingProxyType

import pytest

from strands_evals.experimental.calibrator import (
    CalibratedEqualAlgorithm,
    CalibrationData,
    CalibrationExample,
    CalibrationObservation,
    CalibrationTarget,
    CalibrationTargetDefinition,
    EqualWeightingAlgorithm,
)


def _data(
    *,
    target: float | None = None,
    target_definition: CalibrationTargetDefinition | None = None,
) -> CalibrationData:
    return CalibrationData(
        observations=(
            CalibrationObservation("first", "case-a", 0, 0, 0.2),
            CalibrationObservation("second", "case-a", 0, 0, 0.8),
        ),
        examples=(
            CalibrationExample(
                case_name="case-a",
                agent_run=0,
                evaluator_scores=MappingProxyType({"first": 0.2, "second": 0.8}),
                target=target,
            ),
        ),
        evaluator_names=("first", "second"),
        case_names=("case-a",),
        agent_runs=1,
        judge_replays=1,
        step_sizes=MappingProxyType({"first": 0.5, "second": 0.5}),
        variance_components=MappingProxyType(
            {
                "first": MappingProxyType({"case": 0.0, "agent": 0.0, "judge": 0.0}),
                "second": MappingProxyType({"case": 0.0, "agent": 0.0, "judge": 0.0}),
            }
        ),
        repeatabilities=MappingProxyType({"first": 1.0, "second": 1.0}),
        target_definition=target_definition,
    )


def _target_data(score_targets: list[tuple[float, float]]) -> CalibrationData:
    examples = tuple(
        CalibrationExample(
            case_name=f"case-{index}",
            agent_run=0,
            evaluator_scores=MappingProxyType(
                {
                    "first": score,
                    "second": score,
                }
            ),
            target=target,
        )
        for index, (score, target) in enumerate(score_targets)
    )
    return replace(
        _data(),
        observations=(),
        examples=examples,
        case_names=tuple(example.case_name for example in examples),
        target_definition=CalibrationTargetDefinition(name="task_success", version="1"),
    )


def test_calibration_data_exposes_evaluator_means() -> None:
    assert _data().evaluator_means() == {
        "first": pytest.approx(0.2),
        "second": pytest.approx(0.8),
    }


def test_calibration_data_exposes_complete_targets() -> None:
    definition = CalibrationTargetDefinition(name="task_success", version="1")
    data = _data(target=1.0, target_definition=definition)

    assert data.has_targets is True
    assert data.target_definition == definition
    assert data.examples[0].target == pytest.approx(1.0)


def test_calibration_data_exposes_absent_targets() -> None:
    assert _data().has_targets is False


def test_calibration_data_rejects_target_without_definition() -> None:
    with pytest.raises(ValueError, match="target_definition is required"):
        _data(target=1.0)


def test_calibration_data_rejects_definition_without_complete_targets() -> None:
    definition = CalibrationTargetDefinition(name="task_success", version="1")

    with pytest.raises(ValueError, match="every calibration example must contain a target"):
        _data(target_definition=definition)


@pytest.mark.parametrize(
    ("name", "version", "message"),
    [
        ("", "1", "name"),
        ("task_success", "", "version"),
    ],
)
def test_target_definition_rejects_empty_identity(
    name: str,
    version: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationTargetDefinition(name=name, version=version)


@pytest.mark.parametrize(
    ("case_name", "agent_run", "value", "message"),
    [
        ("", 0, 1.0, "case_name"),
        ("case-a", -1, 1.0, "agent_run"),
        ("case-a", 0, float("nan"), "finite"),
        ("case-a", 0, True, "finite"),
    ],
)
def test_calibration_target_rejects_invalid_values(
    case_name: str,
    agent_run: int,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationTarget(
            case_name=case_name,
            agent_run=agent_run,
            value=value,
        )


def test_calibration_target_normalizes_integer_values() -> None:
    target = CalibrationTarget(case_name="case-a", agent_run=0, value=1)

    assert target.value == 1.0
    assert isinstance(target.value, float)


def test_equal_weighting_algorithm_fits_common_model_interface() -> None:
    model = EqualWeightingAlgorithm().fit(_data())

    assert model.algorithm_id == "equal_weighting"
    assert model.algorithm_version == 1
    assert dict(model.weights) == {"first": 0.5, "second": 0.5}
    assert model.predict({"first": 0.2, "second": 0.8}) == pytest.approx(0.5)


def test_equal_weighting_model_rejects_changed_evaluator_set() -> None:
    model = EqualWeightingAlgorithm().fit(_data())

    with pytest.raises(ValueError, match="evaluator scores do not match"):
        model.predict({"first": 0.2})


def test_calibrated_equal_algorithm_learns_probability_mapping() -> None:
    data = _target_data(
        [
            (0.2, 0.0),
            (0.2, 0.0),
            (0.2, 0.0),
            (0.2, 1.0),
            (0.8, 0.0),
            (0.8, 1.0),
            (0.8, 1.0),
            (0.8, 1.0),
        ]
    )

    model = CalibratedEqualAlgorithm().fit(data)

    assert dict(model.weights) == {"first": 0.5, "second": 0.5}
    assert model.intercept == pytest.approx(-1.83102, abs=1e-4)
    assert model.scale == pytest.approx(3.66204, abs=1e-4)
    assert model.predict({"first": 0.2, "second": 0.2}) == pytest.approx(0.25, abs=1e-6)
    assert model.predict({"first": 0.8, "second": 0.8}) == pytest.approx(0.75, abs=1e-6)


def test_calibrated_equal_algorithm_is_deterministic_and_restorable() -> None:
    data = _target_data(
        [
            (0.2, 0.0),
            (0.2, 1.0),
            (0.8, 0.0),
            (0.8, 1.0),
        ]
    )
    algorithm = CalibratedEqualAlgorithm()

    first = algorithm.fit(data)
    second = algorithm.fit(data)
    restored = algorithm.restore(first.parameters)

    assert second.parameters == first.parameters
    assert restored.parameters == first.parameters
    assert restored.predict({"first": 0.4, "second": 0.6}) == pytest.approx(
        first.predict({"first": 0.4, "second": 0.6})
    )


def test_calibrated_equal_algorithm_uses_base_rate_when_scores_do_not_vary() -> None:
    model = CalibratedEqualAlgorithm().fit(
        _target_data(
            [
                (0.5, 0.0),
                (0.5, 0.0),
                (0.5, 0.0),
                (0.5, 1.0),
            ]
        )
    )

    assert model.scale == 0.0
    assert model.predict({"first": 0.0, "second": 1.0}) == pytest.approx(0.25)


def test_calibrated_equal_algorithm_requires_targets() -> None:
    with pytest.raises(ValueError, match="requires complete external targets"):
        CalibratedEqualAlgorithm().fit(_data())


def test_calibrated_equal_algorithm_requires_binary_targets() -> None:
    with pytest.raises(ValueError, match="binary targets"):
        CalibratedEqualAlgorithm().fit(
            _target_data(
                [
                    (0.2, 0.0),
                    (0.8, 0.5),
                ]
            )
        )


def test_calibrated_equal_algorithm_requires_both_target_classes() -> None:
    with pytest.raises(ValueError, match="both binary target classes"):
        CalibratedEqualAlgorithm().fit(
            _target_data(
                [
                    (0.2, 1.0),
                    (0.8, 1.0),
                ]
            )
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            {
                "weights": {"first": 0.25, "second": 0.75},
                "intercept": 0.0,
                "scale": 1.0,
            },
            "weights must be equal",
        ),
        (
            {
                "weights": {"first": 0.5, "second": 0.5},
                "intercept": float("nan"),
                "scale": 1.0,
            },
            "intercept must be a finite number",
        ),
        (
            {
                "weights": {"first": 0.5, "second": 0.5},
                "intercept": 0.0,
                "scale": "invalid",
            },
            "scale must be a finite number",
        ),
    ],
)
def test_calibrated_equal_restore_rejects_invalid_parameters(
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CalibratedEqualAlgorithm().restore(parameters)
