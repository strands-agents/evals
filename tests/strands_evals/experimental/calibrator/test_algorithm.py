from collections.abc import Mapping
from typing import Any

import pytest

from strands_evals.experimental.calibrator import (
    CalibrationData,
    CalibrationModel,
    CalibrationObservation,
)


class _NonLinearModel(CalibrationModel):
    @property
    def parameters(self) -> Mapping[str, Any]:
        return {}

    def predict(self, evaluator_scores: Mapping[str, float]) -> float:
        return max(evaluator_scores.values())


def _observation(
    evaluator_name: str,
    case_name: str,
    agent_run: int,
    score: float,
    judge_replay: int = 0,
) -> CalibrationObservation:
    return CalibrationObservation(
        evaluator_name=evaluator_name,
        case_name=case_name,
        agent_run=agent_run,
        judge_replay=judge_replay,
        score=score,
    )


def _data() -> CalibrationData:
    step_sizes = {"first": 0.5, "second": 0.5}
    variance_components = {
        "first": {"case": 0.0, "agent": 0.0, "judge": 0.0},
        "second": {"case": 0.0, "agent": 0.0, "judge": 0.0},
    }
    repeatabilities = {"first": 1.0, "second": 1.0}
    return CalibrationData(
        observations=(
            _observation("second", "case-b", 1, 0.9, judge_replay=1),
            _observation("first", "case-a", 0, 0.4, judge_replay=1),
            _observation("second", "case-a", 1, 0.8),
            _observation("first", "case-b", 0, 0.1),
            _observation("second", "case-b", 1, 0.7),
            _observation("first", "case-a", 0, 0.2),
            _observation("first", "case-a", 1, 0.6),
            _observation("second", "case-a", 0, 0.5),
            _observation("first", "case-b", 1, 0.9),
            _observation("second", "case-b", 0, 0.3),
        ),
        evaluator_names=("first", "second"),
        case_names=("case-a", "case-b"),
        agent_runs=2,
        judge_replays=2,
        step_sizes=step_sizes,
        variance_components=variance_components,
        repeatabilities=repeatabilities,
    )


def test_calibration_data_derives_replay_aggregated_samples() -> None:
    samples = _data().samples

    assert [(sample.case_name, sample.agent_run) for sample in samples] == [
        ("case-a", 0),
        ("case-b", 0),
        ("case-a", 1),
        ("case-b", 1),
    ]
    assert [sample.evaluator_scores for sample in samples] == [
        {"first": pytest.approx(0.3), "second": pytest.approx(0.5)},
        {"first": pytest.approx(0.1), "second": pytest.approx(0.3)},
        {"first": pytest.approx(0.6), "second": pytest.approx(0.8)},
        {"first": pytest.approx(0.9), "second": pytest.approx(0.8)},
    ]


def test_calibration_data_copies_mapping_inputs() -> None:
    step_sizes = {"first": 0.5}
    variance_components = {"first": {"case": 0.0, "agent": 0.0, "judge": 0.0}}
    repeatabilities = {"first": 1.0}
    data = CalibrationData(
        observations=(),
        evaluator_names=("first",),
        case_names=(),
        agent_runs=1,
        judge_replays=1,
        step_sizes=step_sizes,
        variance_components=variance_components,
        repeatabilities=repeatabilities,
    )

    step_sizes["first"] = 1.0
    variance_components["first"]["case"] = 1.0
    repeatabilities["first"] = 0.0

    assert data.step_sizes == {"first": 0.5}
    assert data.variance_components["first"]["case"] == 0.0
    assert data.repeatabilities == {"first": 1.0}
    with pytest.raises(TypeError):
        data.step_sizes["first"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("evaluator_names", [(), ("first", "first")])
def test_calibration_data_rejects_invalid_evaluator_names(
    evaluator_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        CalibrationData(
            observations=(),
            evaluator_names=evaluator_names,
            case_names=(),
            agent_runs=1,
            judge_replays=1,
            step_sizes={},
            variance_components={},
            repeatabilities={},
        )


def test_non_linear_model_does_not_need_weights() -> None:
    model = _NonLinearModel()

    assert model.weights is None
    assert model.predict({"first": 0.2, "second": 0.8}) == pytest.approx(0.8)
