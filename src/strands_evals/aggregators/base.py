"""Base CaseAggregator class.

Provides a default implementation that groups results by evaluator and
computes numeric statistics (mean/min/max score, pass rate). Derived
classes override `summarize_reasons()` to add LLM-based or domain-specific
narrative summaries.
"""

import logging
from collections import defaultdict
from typing import Any

from ..types.evaluation_report import EvaluationReport
from .types import AggregationResult

logger = logging.getLogger(__name__)


class CaseAggregator:
    """Base class for evaluation aggregators.

    An aggregator takes a flat list of EvaluationReports (produced by an
    Experiment) and re-groups/analyzes them along a specific dimension
    (e.g., chaos scenarios, trials, case categories).

    The default implementation groups by evaluator name and computes numeric
    stats across all cases. Subclasses can override:
        - `aggregate()` for custom grouping logic
        - `summarize_reasons()` for LLM-based or domain-specific narrative generation

    Example::

        from strands_evals.aggregators import CaseAggregator

        aggregator = CaseAggregator()
        reports = experiment.run_evaluations(task=my_task)
        results = aggregator.aggregate(reports)

        for r in results:
            print(f"{r.group_key}: mean={r.mean_score:.2f}, pass_rate={r.pass_rate:.0%}")
    """

    def __init__(self, name: str | None = None):
        """Initialize the aggregator.

        Args:
            name: Optional human-readable name for this aggregator.
        """
        self.name = name or self.__class__.__name__

    def aggregate(self, reports: list[EvaluationReport]) -> list[AggregationResult]:
        """Aggregate evaluation reports into summary results.

        Default implementation groups all case results by evaluator name and
        computes numeric statistics. The `summary` field is populated by
        calling `summarize_reasons()`.

        Args:
            reports: Flat list of EvaluationReport objects from an Experiment run.

        Returns:
            List of AggregationResult objects, one per evaluator.
        """
        if not reports:
            return []

        results = []
        for report in reports:
            stats = self._compute_stats(report.scores, report.test_passes)
            summary = self.summarize_reasons(report.reasons)

            results.append(
                AggregationResult(
                    group_key=report.evaluator_name or "Unknown",
                    evaluator_name=report.evaluator_name or "Unknown",
                    summary=summary,
                    **stats,
                )
            )

        return results

    def summarize_reasons(self, reasons: list[str]) -> str:
        """Produce a narrative summary from a list of per-case reason strings.

        The base implementation concatenates unique non-empty reasons.
        Override in subclasses to use LLM-as-a-Judge or domain-specific logic.

        Args:
            reasons: List of reason strings from individual evaluations.

        Returns:
            A summary string.
        """
        return self._concatenate_reasons(reasons)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(scores: list[float], passes: list[bool]) -> dict[str, Any]:
        """Compute basic statistics from a list of scores and pass/fail flags.

        Args:
            scores: List of numeric scores.
            passes: List of boolean pass/fail indicators.

        Returns:
            Dict with mean_score, min_score, max_score, pass_rate,
            num_results, num_passed, num_failed.
        """
        if not scores:
            return {
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "pass_rate": 0.0,
                "num_results": 0,
                "num_passed": 0,
                "num_failed": 0,
            }

        num_passed = sum(1 for p in passes if p)
        num_failed = len(passes) - num_passed

        return {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": num_passed / len(passes) if passes else 0.0,
            "num_results": len(scores),
            "num_passed": num_passed,
            "num_failed": num_failed,
        }

    @staticmethod
    def _concatenate_reasons(reasons: list[str], max_reasons: int = 10) -> str:
        """Combine multiple reason strings by deduplication and concatenation.

        Args:
            reasons: List of reason strings from individual evaluations.
            max_reasons: Maximum number of unique reasons to include.

        Returns:
            Combined summary string.
        """
        unique_reasons = []
        seen: set[str] = set()
        for reason in reasons:
            if reason and reason not in seen:
                seen.add(reason)
                unique_reasons.append(reason)
                if len(unique_reasons) >= max_reasons:
                    break

        if not unique_reasons:
            return ""

        if len(unique_reasons) == 1:
            return unique_reasons[0]

        summary_parts = [f"({i + 1}) {r}" for i, r in enumerate(unique_reasons)]
        return " | ".join(summary_parts)
