from types import MappingProxyType

import pytest

from strands_evals.experimental.calibrator import (
    CalibrationData,
    CalibrationExample,
    CalibrationObservation,
    EqualWeightingAlgorithm,
)


def _data() -> CalibrationData:
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
    )


def test_calibration_data_exposes_evaluator_means() -> None:
    assert _data().evaluator_means() == {
        "first": pytest.approx(0.2),
        "second": pytest.approx(0.8),
    }


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
