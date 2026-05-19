"""Data models for aggregated evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class AggregationResult(BaseModel):
    """Aggregated result for one logical group.

    A "group" is what the aggregator chose to roll up over — typically
    ``(case_key, evaluator_name)`` for the default ``Aggregator``. Subclasses
    can group differently by overriding ``Aggregator._group_key``.
    """

    group_key: str
    evaluator_name: str

    # Descriptive stats over the entries in this group.
    mean_score: float
    min_score: float
    max_score: float
    pass_rate: float
    num_results: int
    num_passed: int
    num_failed: int

    # Aggregated free-text reasons from per-trial evaluations.
    reasons: list[str] = Field(default_factory=list)

    # LLM-generated summary string. Empty when no summarizer was provided.
    summary: str = ""

    # Free-form extension point for subclasses that need to attach
    # project-specific stats (e.g. paired comparisons, win-rate CIs).
    extra: dict = Field(default_factory=dict)


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
