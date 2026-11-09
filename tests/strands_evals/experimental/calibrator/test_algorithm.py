import pytest

from strands_evals.experimental.calibrator import (
    CalibrationData,
    CalibrationExample,
    CalibrationObservation,
    WeightingAlgorithm,
)


def _data() -> CalibrationData:
    observations = (
        CalibrationObservation("fine", "case-a", 0, 0, 0.0),
        CalibrationObservation("coarse", "case-a", 0, 0, 0.0),
        CalibrationObservation("fine", "case-b", 0, 0, 1.0),
        CalibrationObservation("coarse", "case-b", 0, 0, 1.0),
    )
    return CalibrationData(
        observations=observations,
        examples=(
            CalibrationExample(
                case_name="case-a",
                agent_run=0,
                evaluator_scores={"coarse": 0.0, "fine": 0.0},
            ),
            CalibrationExample(
                case_name="case-b",
                agent_run=0,
                evaluator_scores={"coarse": 1.0, "fine": 1.0},
            ),
        ),
        evaluator_names=("coarse", "fine"),
        case_names=("case-a", "case-b"),
        agent_runs=1,
        judge_replays=1,
        step_sizes={"coarse": 1.0, "fine": 0.25},
        variance_components={
            "coarse": {"agent": 0.0, "case": 0.5, "judge": 0.0},
            "fine": {"agent": 0.0, "case": 0.5, "judge": 0.0},
        },
        repeatabilities={"coarse": 1.0, "fine": 1.0},
    )


def test_calibration_data_exposes_evaluator_means() -> None:
    assert _data().evaluator_means() == {
        "coarse": 0.5,
        "fine": 0.5,
    }


def test_equal_weighting_algorithm_fits_common_model_interface() -> None:
    model = WeightingAlgorithm("equal").fit(_data())

    assert model.algorithm_id == "equal_weighting"
    assert model.algorithm_version == 1
    assert model.weights == {"coarse": 0.5, "fine": 0.5}
    assert model.predict({"coarse": 0.0, "fine": 1.0}) == pytest.approx(0.5)


def test_inverse_step_algorithm_uses_measurement_diagnostics() -> None:
    model = WeightingAlgorithm("inverse_step").fit(_data())

    assert model.weights == pytest.approx({"coarse": 0.2, "fine": 0.8})
    assert model.predict({"coarse": 0.0, "fine": 1.0}) == pytest.approx(0.8)


def test_weighted_model_rejects_changed_evaluator_set() -> None:
    model = WeightingAlgorithm("equal").fit(_data())

    with pytest.raises(ValueError, match="evaluator scores do not match the fitted model"):
        model.predict({"coarse": 0.5})
