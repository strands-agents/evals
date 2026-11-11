import pytest

from strands_evals.experimental.calibrator import (
    CalibrationAlgorithm,
    CalibrationData,
    CalibrationModel,
    CalibrationObservation,
    EqualWeightingAlgorithm,
    EqualWeightingModel,
)


def _data() -> CalibrationData:
    return CalibrationData(
        observations=(
            CalibrationObservation(
                evaluator_name="first",
                case_name="case-a",
                agent_run=0,
                judge_replay=0,
                score=0.2,
            ),
            CalibrationObservation(
                evaluator_name="second",
                case_name="case-a",
                agent_run=0,
                judge_replay=0,
                score=0.8,
            ),
        ),
        evaluator_names=("first", "second"),
        case_names=("case-a",),
        agent_runs=1,
        judge_replays=1,
        step_sizes={"first": 0.5, "second": 0.5},
        variance_components={
            "first": {"case": 0.0, "agent": 0.0, "judge": 0.0},
            "second": {"case": 0.0, "agent": 0.0, "judge": 0.0},
        },
        repeatabilities={"first": 1.0, "second": 1.0},
    )


def test_equal_weighting_algorithm_fits_common_model_interface() -> None:
    algorithm = EqualWeightingAlgorithm()
    model = algorithm.fit(_data())

    assert isinstance(algorithm, CalibrationAlgorithm)
    assert isinstance(model, CalibrationModel)
    assert algorithm.identity == "equal_weighting:v1"
    assert model.weights == {"first": 0.5, "second": 0.5}
    assert model.predict({"first": 0.2, "second": 0.8}) == pytest.approx(0.5)


def test_equal_weighting_model_rejects_changed_evaluator_set() -> None:
    model = EqualWeightingAlgorithm().fit(_data())

    with pytest.raises(ValueError, match="evaluator scores do not match"):
        model.predict({"first": 0.2})


def test_equal_weighting_model_copies_and_protects_weights() -> None:
    evaluator_weights = {"first": 0.5, "second": 0.5}
    model = EqualWeightingModel(evaluator_weights=evaluator_weights)

    evaluator_weights["first"] = 1.0

    assert model.weights == {"first": 0.5, "second": 0.5}
    with pytest.raises(TypeError):
        model.evaluator_weights["first"] = 1.0  # type: ignore[index]
