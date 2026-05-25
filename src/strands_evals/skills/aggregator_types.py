"""Data models for skills evaluation aggregation results."""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..aggregators.types import AggregationResult


class PairedComparisonStats(BaseModel):
    """One (case × metric) paired comparison: variant vs baseline.

    Attributes:
        metric_name: Identifier for the metric (e.g. "pass_rate", "tokens",
            "latency_s", "cost_usd").
        baseline_mean: Mean value of the metric under the baseline condition.
        variant_mean: Mean value of the metric under the variant condition.
        delta: Signed difference, variant_mean - baseline_mean. Positive means
            the variant scored higher; whether that is "better" depends on the
            metric (higher pass_rate is good, higher latency is bad).
        delta_pct: delta expressed as a percentage of baseline_mean. None when
            baseline_mean is zero or the percentage is not meaningful.
        test_used: Statistical test identifier — one of "wilcoxon",
            "paired_t", or "mcnemar".
        p_value: Two-sided p-value from the paired test.
        ci_low: Lower bound of the 95% confidence interval on the delta
            (bootstrap percentile method).
        ci_high: Upper bound of the 95% confidence interval on the delta.
        n_used: Number of paired trials that contributed after corruption
            filtering.
        n_corrupted: Number of paired trials that were dropped because at
            least one side was marked corrupted.
    """

    metric_name: str
    baseline_mean: float
    variant_mean: float
    delta: float
    delta_pct: Optional[float] = None
    test_used: str
    p_value: float
    ci_low: float
    ci_high: float
    n_used: int
    n_corrupted: int = 0


class SkillEvalAggregation(AggregationResult):
    """Aggregated results for one (case, evaluator) under skills evaluation.

    Extends AggregationResult with paired-comparison statistics, raw
    per-condition values for downstream diagnostics, corruption accounting,
    and trajectory pointers for drill-down.
    """

    # --- Paired statistics across metrics ---
    paired_stats: list[PairedComparisonStats] = Field(default_factory=list)

    # --- Raw per-condition values for downstream diagnostics ---
    # Key = metric_name, Value = list of per-trial values in trial_idx order.
    raw_baseline_values: dict[str, list[float]] = Field(default_factory=dict)
    raw_variant_values: dict[str, list[float]] = Field(default_factory=dict)

    # --- Corruption accounting (over paired trials) ---
    n_total: int = 0
    n_corrupted: int = 0
    n_used: int = 0

    # --- Trajectory pointers (session_ids) for drill-down ---
    trajectory_pointers_baseline: list[str] = Field(default_factory=list)
    trajectory_pointers_variant: list[str] = Field(default_factory=list)


class SkillEvalAggregationReport(BaseModel):
    """Report containing all skills aggregation results.

    Provides .run_display(), .display(), .to_file() and .from_file()
    matching the EvaluationReport / ChaosAggregationReport interfaces.

    Example::

        aggregation_report = experiment.aggregate_evaluations()
        aggregation_report.run_display()
        aggregation_report.to_file("skill_eval_report.json")
    """

    aggregations: list[SkillEvalAggregation] = Field(default_factory=list)

    def run_display(self):
        """Render the aggregation report interactively.

        Collapsed view shows one row per (case, evaluator) with Δ-metrics.
        Expanding reveals full paired-statistics panels per metric.
        """
        from .aggregation_display import display_skill_aggregation

        display_skill_aggregation(self.aggregations, static=False)

    def display(self):
        """Render the report statically (non-interactive)."""
        from .aggregation_display import display_skill_aggregation

        display_skill_aggregation(self.aggregations, static=True)

    def to_file(self, path: str):
        """Write the aggregation report to a JSON file.

        Args:
            path: File path. If no extension is provided, ".json" is added.

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
    def from_file(cls, path: str) -> "SkillEvalAggregationReport":
        """Load an aggregation report from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A SkillEvalAggregationReport instance.

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
