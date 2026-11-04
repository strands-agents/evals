import pytest

from strands_evals.experimental.calibrator import (
    AlgorithmIdentity,
    CalibrationData,
    CalibrationObservation,
)


@pytest.mark.parametrize(
    ("name", "version", "message"),
    [
        ("", 1, "name must be a non-empty string"),
        ("equal_weighting", 0, "version must be an integer"),
        ("equal_weighting", True, "version must be an integer"),
    ],
)
def test_algorithm_identity_rejects_invalid_values(name: str, version: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AlgorithmIdentity(name=name, version=version)


def _data() -> CalibrationData:
    return CalibrationData(
        observations=(
            CalibrationObservation("first", "case-a", 0, 0, 0.2),
            CalibrationObservation("first", "case-a", 0, 1, 0.4),
            CalibrationObservation("second", "case-a", 0, 0, 0.8),
            CalibrationObservation("second", "case-a", 0, 1, 0.6),
        ),
        evaluator_names=("first", "second"),
        case_names=("case-a",),
        agent_runs=1,
        judge_replays=2,
        step_sizes={"first": 0.5, "second": 0.5},
        variance_components={
            "first": {"case": 0.0, "agent": 0.0, "judge": 0.0},
            "second": {"case": 0.0, "agent": 0.0, "judge": 0.0},
        },
        repeatabilities={"first": 1.0, "second": 1.0},
    )


def test_calibration_data_derives_replay_aggregated_samples() -> None:
    assert len(_data().samples) == 1
    assert _data().samples[0].case_name == "case-a"
    assert _data().samples[0].agent_run == 0
    assert _data().samples[0].evaluator_scores == {
        "first": pytest.approx(0.3),
        "second": pytest.approx(0.7),
    }
