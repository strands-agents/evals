"""Data models for evaluation aggregation results."""

from pydantic import BaseModel, Field


class AggregationResult(BaseModel):
    """Base aggregation result for a group of evaluation results.

    Provides quantitative statistics that any aggregator can produce
    regardless of the grouping dimension.
    """

    group_key: str = Field(..., description="Identifier for this group (e.g., case name)")
    evaluator_name: str

    # --- Quantitative stats ---
    mean_score: float
    min_score: float
    max_score: float
    pass_rate: float  # Fraction of results that passed (0.0 to 1.0)
    num_results: int
    num_passed: int
    num_failed: int

    # --- Narrative summary ---
    summary: str = Field(default="", description="Aggregated summary of all reason fields")
