"""Data models for chaos scenario aggregation results."""

from enum import Enum
from typing import Optional

from pydantic import Field

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
