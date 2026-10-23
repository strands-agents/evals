import json

import pytest
from pydantic import ValidationError

from strands_evals import Case
from strands_evals.evaluators import ConcisenessEvaluator
from strands_evals.experimental import CalibrationProfile, ProfileCompatibility
from strands_evals.experimental.calibrator._calibration import _build_calibration_context
from strands_evals.experimental.calibrator.algorithm import WeightingMethod


def _context(
    dataset: str = "dataset-a",
    evaluators: str = "evaluators-a",
    step_sizes: str = "steps-a",
) -> dict:
    return {
        "version": "1",
        "dataset": dataset,
        "evaluators": evaluators,
        "step_sizes": step_sizes,
        "overall": "overall-a",
    }


def _bootstrap_summary(mean: float) -> dict[str, float | int | None]:
    return {
        "mean": mean,
        "lower_2_5": max(0.0, mean - 0.1),
        "median": mean,
        "upper_97_5": min(1.0, mean + 0.1),
        "defined_replicates": 100,
        "undefined_replicates": 0,
    }


def _profile(
    *,
    algorithm_id: str = "inverse_sqrt_step_weighting",
    algorithm_version: int = 1,
    weighting_method: WeightingMethod | None = "inverse_sqrt_step",
    weights: dict[str, float] | None = None,
    calibration_context: dict | None = None,
    context_free: bool = False,
    fit_structure_fingerprint: str | None = None,
) -> CalibrationProfile:
    context = None if context_free else (_context() if calibration_context is None else calibration_context)
    profile_weights = {"stable": 0.8, "noisy": 0.2} if weights is None else weights
    return CalibrationProfile.create(
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        model_parameters={
            "weighting_method": weighting_method,
            "weights": profile_weights,
            "used_equal_fallback": False,
        },
        weighting_method=weighting_method,
        weights=profile_weights,
        status="uncalibrated",
        step_sizes={"stable": 0.25, "noisy": 0.5},
        variance_components={
            "stable": {"case": 0.4, "agent": 0.01, "judge": 0.02},
            "noisy": {"case": 0.1, "agent": 0.03, "judge": 0.2},
        },
        repeatabilities={"stable": 0.9, "noisy": 0.3},
        bootstrap={
            "replicates": 100,
            "seed": 42,
            "repeatability_intervals": {
                "stable": _bootstrap_summary(0.9),
                "noisy": _bootstrap_summary(0.3),
            },
            "candidate_weight_intervals": {
                "inverse_sqrt_step": {
                    "stable": _bootstrap_summary(0.8),
                    "noisy": _bootstrap_summary(0.2),
                }
            },
            "composite_score_intervals": {
                "inverse_sqrt_step": _bootstrap_summary(0.6),
            },
        },
        case_count=4,
        agent_runs=2,
        judge_replays=3,
        calibration_context=context,
        fit_structure_fingerprint=fit_structure_fingerprint,
    )


def test_profile_is_deterministic_and_round_trips_through_json(tmp_path) -> None:
    profile = _profile()
    assert profile == _profile()
    path = tmp_path / "profiles" / "support-agent"

    profile.to_file(str(path))
    loaded = CalibrationProfile.from_file(str(path.with_suffix(".json")))

    assert loaded == profile
    serialized = json.loads(path.with_suffix(".json").read_text())
    assert serialized["profile_id"] == profile.profile_id
    assert serialized["schema_version"] == 1
    assert serialized["algorithm_id"] == "inverse_sqrt_step_weighting"
    assert serialized["bootstrap"]["replicates"] == 100


def test_profile_rejects_content_that_does_not_match_id() -> None:
    data = _profile().to_dict()
    data["case_count"] = 5

    with pytest.raises(ValidationError, match="profile ID does not match"):
        CalibrationProfile.from_dict(data)


@pytest.mark.parametrize("repeatability", [-0.1, 1.1])
def test_profile_rejects_out_of_range_repeatability(repeatability: float) -> None:
    data = _profile().to_dict()
    data.pop("profile_id")
    data["repeatabilities"]["stable"] = repeatability

    with pytest.raises(ValidationError, match="repeatabilities must be finite and between 0 and 1"):
        CalibrationProfile.create(**data)


def test_profile_rejects_missing_variance_component() -> None:
    data = _profile().to_dict()
    data.pop("profile_id")
    del data["variance_components"]["stable"]["judge"]

    with pytest.raises(ValidationError, match="variance components for 'stable' must contain exactly"):
        CalibrationProfile.create(**data)


def test_profile_rejects_invalid_weights() -> None:
    data = _profile().to_dict()
    data.pop("profile_id")
    data["weights"] = {"stable": 0.9, "noisy": 0.2}

    with pytest.raises(ValidationError, match="profile weights must sum to 1"):
        CalibrationProfile.create(**data)


def test_profile_rejects_weighting_method_that_does_not_match_algorithm() -> None:
    data = _profile().to_dict()
    data.pop("profile_id")
    data["algorithm_id"] = "equal_weighting"

    with pytest.raises(ValidationError, match="weighting_method does not match algorithm_id"):
        CalibrationProfile.create(**data)


def test_custom_non_weighting_profile_can_omit_weights() -> None:
    profile = CalibrationProfile.create(
        algorithm_id="maximum_score",
        algorithm_version=1,
        model_parameters={"reducer": "maximum"},
        weighting_method=None,
        weights=None,
        status="uncalibrated",
        step_sizes={"first": 0.5, "second": 0.5},
        variance_components={
            "first": {"case": 0.1, "agent": 0.0, "judge": 0.0},
            "second": {"case": 0.2, "agent": 0.0, "judge": 0.0},
        },
        repeatabilities={"first": 1.0, "second": 1.0},
        case_count=2,
        agent_runs=2,
        judge_replays=2,
        calibration_context=_context(),
    )

    assert profile.weights is None
    assert profile.weighting_method is None
    assert profile.runtime_compatibility == ProfileCompatibility.COMPATIBLE


def test_profile_create_rejects_unknown_fields() -> None:
    data = _profile().to_dict()
    data.pop("profile_id")
    data["weighting_methd"] = "equal"

    with pytest.raises(ValueError, match="unexpected calibration profile fields.*weighting_methd"):
        CalibrationProfile.create(**data)


def test_profile_rejects_unsupported_schema_version() -> None:
    data = _profile().to_dict()
    data["schema_version"] = 2

    with pytest.raises(ValidationError, match="unsupported calibration profile schema version"):
        CalibrationProfile.from_dict(data)


def test_profile_scoring_compatibility_allows_new_dataset() -> None:
    profile = _profile()
    metadata_only_change = _context()
    metadata_only_change["metadata"] = {"new_diagnostic": True}

    assert profile.compatibility_with(_context()) == ProfileCompatibility.COMPATIBLE
    assert profile.compatibility_with(metadata_only_change) == ProfileCompatibility.COMPATIBLE
    assert profile.compatibility_with(_context("dataset-b")) == ProfileCompatibility.COMPATIBLE
    assert profile.compatibility_with(_context("dataset-b"), for_refit=True) == ProfileCompatibility.INCOMPATIBLE
    assert profile.compatibility_with(_context(evaluators="evaluators-b")) == ProfileCompatibility.INCOMPATIBLE
    assert profile.compatibility_with(_context(step_sizes="steps-b")) == ProfileCompatibility.INCOMPATIBLE
    assert profile.compatibility_with(None) == ProfileCompatibility.INCOMPATIBLE


def test_profile_classifies_algorithm_versions() -> None:
    profile = _profile()

    assert profile.algorithm_compatibility("inverse_sqrt_step_weighting", 1) == ProfileCompatibility.COMPATIBLE
    assert profile.algorithm_compatibility("inverse_sqrt_step_weighting", 2) == ProfileCompatibility.STALE
    assert profile.algorithm_compatibility("inverse_sqrt_step_weighting", 0) == ProfileCompatibility.INCOMPATIBLE
    assert profile.algorithm_compatibility("other", 1) == ProfileCompatibility.INCOMPATIBLE


def test_profile_rejects_unsupported_calibration_context_version() -> None:
    context = _context()
    context["version"] = "2"

    with pytest.raises(ValidationError, match="calibration_context has an unsupported version"):
        _profile(calibration_context=context)


def test_context_free_profile_requires_fit_fingerprint() -> None:
    with pytest.raises(ValidationError, match="context-free profiles require a fit_structure_fingerprint"):
        _profile(context_free=True)


def test_context_free_profile_is_compatible_with_context_free_scoring() -> None:
    profile = _profile(
        context_free=True,
        fit_structure_fingerprint="a" * 64,
    )

    assert profile.runtime_compatibility == ProfileCompatibility.COMPATIBLE
    assert profile.compatibility_with(None) == ProfileCompatibility.COMPATIBLE
    assert profile.compatibility_with(_context()) == ProfileCompatibility.INCOMPATIBLE


def test_mutated_profile_is_rejected_before_serialization() -> None:
    profile = _profile()
    assert profile.weights is not None
    profile.weights["stable"] = 0.7
    profile.weights["noisy"] = 0.3

    with pytest.raises(ValidationError, match="profile ID does not match"):
        profile.to_dict()


def test_calibration_context_keeps_audit_metadata_without_raw_inputs_or_rubrics() -> None:
    context = _build_calibration_context(
        cases=[Case(name="private customer case name", input="private customer input")],
        evaluators=[
            ConcisenessEvaluator(
                model="judge-model",
                system_prompt="private evaluator rubric",
                name="concise",
            )
        ],
        step_sizes={"concise": 0.5},
    )

    metadata = context["metadata"]
    assert metadata["dataset"] == {
        "fingerprint": context["dataset"],
        "case_count": 1,
        "named_case_count": 1,
    }
    evaluator_metadata = metadata["evaluators"][0]
    assert evaluator_metadata["name"] == "concise"
    assert evaluator_metadata["judge_model"]["model_id"] == "judge-model"
    assert evaluator_metadata["system_prompt"]["character_count"] == len("private evaluator rubric")

    serialized = json.dumps(context)
    assert "private customer case name" not in serialized
    assert "private customer input" not in serialized
    assert "private evaluator rubric" not in serialized
