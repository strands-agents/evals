"""Metric names: evaluator scores and usage metrics."""

from __future__ import annotations

from collections.abc import Sequence

from ...types.optimization import TrialRecord

USAGE_METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "tokens",
    "seconds",
    "model_calls",
)


class MetricRegistry:
    """Maps metric names as written to evaluator names or usage metrics."""

    def __init__(self, evaluator_names: Sequence[str]):
        raise NotImplementedError

    def resolve(self, name: str) -> str:
        """`"Helpfulness"` -> `"HelpfulnessEvaluator"`; an unknown name raises `ValueError`."""
        raise NotImplementedError


def _case_metrics(records: Sequence[TrialRecord]) -> dict[str, dict[str, float]]:
    """Per-case metric means.

    Evaluator metrics average completed runs with N/A rows dropped; usage metrics average completed
    and agent_failure runs.
    """
    raise NotImplementedError
