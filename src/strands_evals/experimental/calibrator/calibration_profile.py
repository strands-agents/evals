"""Versioned, serializable evaluator calibration profiles."""

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ._calibration import CalibrationStatus
from .algorithm import WeightingMethod

PROFILE_SCHEMA_VERSION = 1
_CALIBRATION_CONTEXT_VERSION = "1"
_SCORING_COMPATIBILITY_COMPONENTS = ("version", "evaluators", "step_sizes")
_REFIT_COMPATIBILITY_COMPONENTS = ("version", "dataset", "evaluators", "step_sizes")
_VARIANCE_COMPONENTS = {"agent", "case", "judge"}

BootstrapSummary = dict[str, float | int | None]


class ProfileCompatibility(str, Enum):
    """Compatibility between a profile and a runtime configuration."""

    COMPATIBLE = "compatible"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"


class BootstrapDiagnostics(BaseModel):
    """Bootstrap settings and uncertainty summaries saved with a profile."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    replicates: int = Field(ge=2)
    seed: int
    repeatability_intervals: dict[str, BootstrapSummary]
    candidate_weight_intervals: dict[WeightingMethod, dict[str, BootstrapSummary]]
    composite_score_intervals: dict[WeightingMethod, BootstrapSummary]


class CalibrationProfile(BaseModel):
    """Durable record of one fitted calibration model and its diagnostics."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
    )

    schema_version: int = PROFILE_SCHEMA_VERSION
    profile_id: str
    algorithm_id: str = Field(min_length=1)
    algorithm_version: int = Field(ge=1)
    model_parameters: dict[str, Any]
    weights: dict[str, float] | None = None
    status: CalibrationStatus
    step_sizes: dict[str, float]
    variance_components: dict[str, dict[str, float]]
    repeatabilities: dict[str, float]
    case_count: int = Field(ge=2)
    agent_runs: int = Field(ge=2)
    judge_replays: int = Field(ge=2)
    calibration_context: dict[str, Any] | None = None
    fit_structure_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    bootstrap: BootstrapDiagnostics | None = None

    @model_validator(mode="after")
    def _validate_profile(self) -> Self:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported calibration profile schema version: {self.schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )

        evaluator_names = set(self.step_sizes)
        if not evaluator_names:
            raise ValueError("calibration profile must contain at least one evaluator")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.step_sizes.values()):
            raise ValueError("profile step sizes must be finite and greater than zero")
        if set(self.variance_components) != evaluator_names:
            raise ValueError("variance_components evaluator names must match step_sizes")
        if set(self.repeatabilities) != evaluator_names:
            raise ValueError("repeatabilities evaluator names must match step_sizes")

        if self.weights is not None:
            if set(self.weights) != evaluator_names:
                raise ValueError("weights evaluator names must match step_sizes")
            if any(not math.isfinite(value) or value < 0.0 for value in self.weights.values()):
                raise ValueError("profile weights must be finite and non-negative")
            if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12):
                raise ValueError("profile weights must sum to 1")

        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.repeatabilities.values()):
            raise ValueError("profile repeatabilities must be finite and between 0 and 1")
        for evaluator_name, components in self.variance_components.items():
            if set(components) != _VARIANCE_COMPONENTS:
                raise ValueError(
                    f"profile variance components for {evaluator_name!r} must contain exactly "
                    f"{sorted(_VARIANCE_COMPONENTS)}"
                )
            if any(not math.isfinite(value) or value < 0.0 for value in components.values()):
                raise ValueError(f"profile variance components for {evaluator_name!r} must be finite and non-negative")

        try:
            _canonical_json(self.model_parameters)
        except (TypeError, ValueError) as error:
            raise ValueError("model_parameters must be deterministic and JSON-compatible") from error

        if self.calibration_context is None:
            if self.fit_structure_fingerprint is None:
                raise ValueError("context-free profiles require a fit_structure_fingerprint")
        elif not _has_compatibility_components(self.calibration_context):
            raise ValueError("calibration_context is missing required compatibility components")
        elif self.calibration_context["version"] != _CALIBRATION_CONTEXT_VERSION:
            raise ValueError("calibration_context has an unsupported version")

        expected_id = self.calculate_profile_id(
            self.model_dump(mode="json", exclude={"profile_id"}),
        )
        if self.profile_id != expected_id:
            raise ValueError("calibration profile ID does not match its contents")
        return self

    @classmethod
    def create(cls, **data: Any) -> Self:
        """Create a profile and derive its deterministic content ID."""
        profile_data = dict(data)
        accepted_fields = set(cls.model_fields) - {"profile_id"}
        unexpected_fields = sorted(set(profile_data) - accepted_fields)
        if unexpected_fields:
            raise ValueError(f"unexpected calibration profile fields: {unexpected_fields}")

        bootstrap = profile_data.get("bootstrap")
        if isinstance(bootstrap, dict):
            profile_data["bootstrap"] = BootstrapDiagnostics.model_validate(bootstrap)

        draft = cls.model_construct(profile_id="", **profile_data)
        payload = draft.model_dump(mode="json", exclude={"profile_id"})
        return cls(profile_id=cls.calculate_profile_id(payload), **payload)

    @staticmethod
    def calculate_profile_id(payload: dict[str, Any]) -> str:
        """Calculate a deterministic SHA-256 identifier for profile contents."""
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def algorithm_compatibility(
        self,
        algorithm_id: str,
        algorithm_version: int,
    ) -> ProfileCompatibility:
        """Classify this profile against one algorithm implementation."""
        if algorithm_id != self.algorithm_id:
            return ProfileCompatibility.INCOMPATIBLE
        if self.algorithm_version < algorithm_version:
            return ProfileCompatibility.STALE
        if self.algorithm_version > algorithm_version:
            return ProfileCompatibility.INCOMPATIBLE
        return ProfileCompatibility.COMPATIBLE

    def compatibility_with(
        self,
        calibration_context: dict[str, Any] | None,
        *,
        for_refit: bool = False,
    ) -> ProfileCompatibility:
        """Classify this profile against a scoring or refitting configuration."""
        if self.calibration_context is None and calibration_context is None:
            return ProfileCompatibility.COMPATIBLE
        if not _has_compatibility_components(self.calibration_context) or not _has_compatibility_components(
            calibration_context
        ):
            return ProfileCompatibility.INCOMPATIBLE

        components = _REFIT_COMPATIBILITY_COMPONENTS if for_refit else _SCORING_COMPATIBILITY_COMPONENTS
        if any(calibration_context[component] != self.calibration_context[component] for component in components):
            return ProfileCompatibility.INCOMPATIBLE
        return ProfileCompatibility.COMPATIBLE

    @property
    def runtime_compatibility(self) -> ProfileCompatibility:
        """Return schema compatibility with this runtime."""
        return ProfileCompatibility.COMPATIBLE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible profile dictionary."""
        data = self.model_dump(mode="json")
        self.__class__.model_validate(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Load and validate a profile dictionary."""
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Write the profile to a deterministic JSON file."""
        file_path = _json_path(path, require_extension=False)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load and validate a profile from a JSON file."""
        file_path = _json_path(path, require_extension=True)
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("calibration profile JSON must contain an object")
        return cls.from_dict(data)


def _canonical_json(value: Any) -> str:
    """Serialize a profile payload deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_path(path: str, *, require_extension: bool) -> Path:
    """Normalize and validate a profile JSON path."""
    file_path = Path(path)
    if file_path.suffix and file_path.suffix != ".json":
        raise ValueError(f"only .json calibration profiles are supported, got: {path}")
    if require_extension and file_path.suffix != ".json":
        raise ValueError(f"calibration profile path must use a .json extension, got: {path}")
    return file_path if file_path.suffix else file_path.with_suffix(".json")


def _has_compatibility_components(
    calibration_context: dict[str, Any] | None,
) -> TypeGuard[dict[str, Any]]:
    """Return whether a context contains every required fingerprint."""
    return calibration_context is not None and all(
        isinstance(calibration_context.get(component), str) and bool(calibration_context[component])
        for component in _REFIT_COMPATIBILITY_COMPONENTS
    )
