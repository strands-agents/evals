"""Data models for aggregated evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EfficiencyStats(BaseModel):
    """Per-group rollup of efficiency metrics across N trials.

    All means are computed over trials that reported the metric. Trials
    missing a particular metric are excluded from that metric's mean only.
    """

    mean_tokens_in: Optional[float] = None
    mean_tokens_out: Optional[float] = None
    mean_wall_clock_s: Optional[float] = None
    mean_cost_usd: Optional[float] = None
    mean_tool_calls: Optional[float] = None
    n_samples: int = 0


class PairedComparisonStats(BaseModel):
    """Paired comparison of one metric between two conditions.

    Populated only when the aggregator detects exactly two conditions with
    a shared pairing key (``trial_idx``).
    """

    metric_name: str
    baseline_label: str
    variant_label: str
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


class AggregationResult(BaseModel):
    """Aggregated result for one logical group.

    A "group" is what the aggregator chose to roll up over — typically
    ``(case_key, evaluator_name)`` for the default aggregator. Subclasses
    may group differently via overriding ``_group_key``.
    """

    group_key: str
    evaluator_name: str

    # Descriptive stats over the trials in this group.
    mean_score: float
    min_score: float
    max_score: float
    pass_rate: float
    num_results: int
    num_passed: int
    num_failed: int

    # Sample counts after corruption filtering. n_total = n_used + n_corrupted.
    n_total: int = 0
    n_corrupted: int = 0
    n_used: int = 0

    # Efficiency rollup. None when no trial reported efficiency metrics.
    efficiency: Optional[EfficiencyStats] = None

    # LLM-generated summary string. Empty when no summarizer was provided.
    summary: str = ""

    # Populated when exactly two conditions are detected for this group.
    paired_stats: list[PairedComparisonStats] = Field(default_factory=list)

    # Raw per-trial values keyed by metric name. Preserved for downstream
    # diagnostic layers that do not want to re-parse trajectories.
    raw_values: dict[str, list[float]] = Field(default_factory=dict)

    # Per-trial trajectory pointers (e.g. session_ids), in trial order.
    trajectory_pointers: list[str] = Field(default_factory=list)

    # Free-form extension point for subclasses.
    metadata: dict = Field(default_factory=dict)


class AggregationReport(BaseModel):
    """Top-level report with display and JSON serialization."""

    aggregations: list[AggregationResult] = Field(default_factory=list)

    def run_display(self) -> None:
        """Render the report to the terminal."""
        from .base import render_display

        render_display(self)

    def to_file(self, path: str) -> None:
        """Write the report to a JSON file.

        Args:
            path: Output path. If no extension is given, ``.json`` is added.

        Raises:
            ValueError: If the path has a non-JSON extension.
        """
        file_path = Path(path)
        if file_path.suffix:
            if file_path.suffix != ".json":
                raise ValueError(
                    f"Only .json format is supported. Got path with extension: {path}."
                )
        else:
            file_path = file_path.with_suffix(".json")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "AggregationReport":
        """Load a report from a JSON file.

        Raises:
            ValueError: If the file does not have a ``.json`` extension.
        """
        file_path = Path(path)
        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}."
            )
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
