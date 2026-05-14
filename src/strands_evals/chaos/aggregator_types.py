"""Data models for chaos scenario aggregation results."""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..aggregators.types import AggregationResult


class CoverageStatus(str, Enum):
    """Status of a tool×effect combination in the coverage matrix."""

    PASSED = "passed"  # Agent handled the failure correctly
    FAILED = "failed"  # Agent did not handle the failure
    NOT_TESTED = "not_tested"  # Combination was not enumerated (capped by max_scenarios)


class ToolEffectResult(AggregationResult):
    """Result for a single tool×effect scenario evaluation.

    Inherits numeric stats from AggregationResult and adds chaos-specific fields.
    """

    tool_name: str
    effect_type: str
    scenario_label: str

    # Convenience fields for single-scenario results
    score: float = 0.0
    passed: bool = False
    reason: str = ""


class ChaosScenarioAggregation(AggregationResult):
    """Aggregated results for one original case across all chaos scenarios.

    Extends the base AggregationResult with chaos-specific coverage analysis
    and baseline comparison.
    """

    # --- Coverage matrix (chaos-specific) ---
    coverage_matrix: dict[str, dict[str, CoverageStatus]] = Field(
        default_factory=dict,
        description=(
            "Outer key: tool_name, Inner key: effect_type → status. "
            'e.g. {"check_inventory": {"timeout": "passed", "truncate_fields": "failed"}}'
        ),
    )

    # --- Baseline comparison (chaos-specific) ---
    baseline_score: Optional[float] = Field(
        default=None, description="Score from the baseline (no-chaos) scenario"
    )
    baseline_passed: Optional[bool] = Field(
        default=None, description="Whether the baseline scenario passed"
    )
    degradation_from_baseline: Optional[float] = Field(
        default=None, description="baseline_score - mean_score (positive = degradation)"
    )

    # --- Per-scenario detail ---
    scenario_results: list[ToolEffectResult] = Field(default_factory=list)


class ChaosAggregationReport(BaseModel):
    """Report containing all chaos scenario aggregation results.

    Provides .run_display() and .to_file() matching the EvaluationReport interface.

    Example::

        aggregation_report = experiment.aggregate_evaluations()
        aggregation_report.run_display()
        aggregation_report.to_file("chaos_aggregation_report.json")
    """

    aggregations: list[ChaosScenarioAggregation] = Field(default_factory=list)

    # Internal: raw reports for display (not serialized)
    _reports: list = []

    class Config:
        arbitrary_types_allowed = True

    def run_display(self):
        """Render the aggregation report interactively.

        Collapsed view shows a summary row per case. Expanding a case reveals
        Stats + Summary panels and a full Coverage Matrix.
        """
        from .aggregation_display import display_chaos_aggregation

        display_chaos_aggregation(self.aggregations, reports=self._reports)

    def display(self):
        """Render the report statically (non-interactive)."""
        from .aggregation_display import display_chaos_aggregation

        display_chaos_aggregation(self.aggregations, reports=self._reports, static=True)

    def to_file(self, path: str):
        """Write the aggregation report to a JSON file.

        Args:
            path: The file path where the report will be saved.
                  If no extension is provided, ".json" will be added automatically.

        Raises:
            ValueError: If the path has a non-JSON extension.
        """
        file_path = Path(path)

        if file_path.suffix:
            if file_path.suffix != ".json":
                raise ValueError(
                    f"Only .json format is supported. Got path with extension: {path}. "
                    f"Please use a .json extension or provide a path without an extension."
                )
        else:
            file_path = file_path.with_suffix(".json")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "ChaosAggregationReport":
        """Load an aggregation report from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A ChaosAggregationReport instance.

        Raises:
            ValueError: If the file does not have a .json extension.
        """
        file_path = Path(path)

        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}. "
                f"Please provide a path with .json extension."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.model_validate(data)
