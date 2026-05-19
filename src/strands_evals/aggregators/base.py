"""General-purpose aggregator over evaluation reports.

The default ``Aggregator`` groups per-trial results by
``(case_key, evaluator_name)`` and computes descriptive statistics
(mean / min / max / pass_rate) plus aggregated reasons. An LLM summary is
generated per group when a model is configured.

Project-specific behavior (paired comparisons, win-rate CIs, corruption
filtering, efficiency rollups, etc.) is intentionally out of scope here.
Subclasses can layer those on top by overriding ``_group_key``,
``_filter_entry``, or ``_build_result``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional, cast

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..types.evaluation_report import EvaluationReport
from .types import AggregationReport, AggregationResult

logger = logging.getLogger(__name__)


# Default LLM system prompt for the summary field. Override via constructor.
DEFAULT_SUMMARY_PROMPT = """\
You are an evaluation analyst.

You will receive aggregated results for a group of agent evaluation trials.
Each trial reports a score and a pass/fail flag, with an optional free-text
reason.

Produce a single paragraph (100 words max) that states the overall
performance pattern (high pass rate, common failure modes, score
distribution shape, etc.). Interpret the numbers; do not repeat them
verbatim. Plain prose, no bullets, no multiple paragraphs.
"""


class _Summary(BaseModel):
    """Structured output type for the LLM summary judge."""

    reasoning: str = Field(description="Brief analysis of the aggregated results")
    summary: str = Field(description="Single paragraph summary")


class Aggregator:
    """Rolls up per-trial ``EvaluationReport`` objects into an ``AggregationReport``.

    Args:
        name: Human-readable name for this aggregator.
        model: Optional model ID or ``Model`` instance for the LLM summary.
            When ``None``, the ``summary`` field on each result is empty.
        system_prompt: Optional system prompt for the summary judge. Defaults
            to ``DEFAULT_SUMMARY_PROMPT``.

    Extension points for subclasses:

    - ``_group_key(entry)``: choose how entries are grouped. Default groups by
      ``(case_key, evaluator_name)``.
    - ``_filter_entry(entry)``: drop entries before aggregation. Default keeps
      all entries.
    - ``_build_result(group_key, entries)``: build the per-group result.
      Default produces an ``AggregationResult`` with descriptive stats and an
      optional LLM summary. Override to attach project-specific fields via
      ``AggregationResult.extra``.
    """

    def __init__(
        self,
        name: str = "Aggregator",
        model: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SUMMARY_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, reports: list[EvaluationReport]) -> AggregationReport:
        """Aggregate a list of evaluation reports into one report.

        Args:
            reports: Per-trial evaluation outputs. Empty list returns an
                empty report.

        Returns:
            An ``AggregationReport`` with one ``AggregationResult`` per
            group, sorted by ``(group_key, evaluator_name)``.
        """
        if not reports:
            return AggregationReport(aggregations=[])

        grouped = self._group_results(reports)

        aggregations: list[AggregationResult] = []
        for group_key, entries in grouped.items():
            kept = [e for e in entries if self._filter_entry(e)]
            if not kept:
                continue
            aggregations.append(self._build_result(group_key, kept))

        aggregations.sort(key=lambda a: (a.group_key, a.evaluator_name))
        return AggregationReport(aggregations=aggregations)

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_results(
        self, reports: list[EvaluationReport]
    ) -> dict[tuple, list[dict]]:
        """Flatten reports into per-trial entries, grouped by ``_group_key``."""
        grouped: dict[tuple, list[dict]] = defaultdict(list)

        for report in reports:
            evaluator_name = report.evaluator_name or "Unknown"
            for i, case_data in enumerate(report.cases):
                entry = self._make_entry(case_data, i, evaluator_name, report)
                key = self._group_key(entry)
                if key is None:
                    continue
                grouped[key].append(entry)

        return grouped

    def _make_entry(
        self,
        case_data: dict,
        index: int,
        evaluator_name: str,
        report: EvaluationReport,
    ) -> dict:
        """Build a per-trial entry from a raw case row.

        Subclasses can override to attach extra fields. The base entry
        carries the case name, evaluator name, score, pass flag, reason,
        and the raw metadata dict so subclasses can inspect arbitrary
        keys without a second pass over the report.
        """
        metadata = case_data.get("metadata") or {}
        case_key = metadata.get("case_key") or case_data.get("name", "") or ""
        return {
            "case_key": case_key,
            "evaluator_name": evaluator_name,
            "score": (
                report.scores[index] if index < len(report.scores) else 0.0
            ),
            "passed": (
                report.test_passes[index]
                if index < len(report.test_passes)
                else False
            ),
            "reason": (
                report.reasons[index] if index < len(report.reasons) else ""
            ),
            "metadata": metadata,
        }

    def _group_key(self, entry: dict) -> Optional[tuple]:
        """Return the group key for an entry, or ``None`` to skip the entry.

        Default: ``(case_key, evaluator_name)``. Entries without a case_key
        are dropped.
        """
        if not entry["case_key"]:
            logger.debug("Skipping entry with no case_key or name")
            return None
        return (entry["case_key"], entry["evaluator_name"])

    def _filter_entry(self, entry: dict) -> bool:
        """Return ``True`` to keep the entry, ``False`` to drop it.

        Default keeps all entries. Subclasses can override to drop entries
        based on metadata (e.g. ``entry["metadata"].get("corrupted")``).
        """
        return True

    # ------------------------------------------------------------------
    # Per-group aggregation
    # ------------------------------------------------------------------

    def _build_result(
        self, group_key: tuple, entries: list[dict]
    ) -> AggregationResult:
        """Build one ``AggregationResult`` from a group of entries.

        Override to attach project-specific fields (paired stats, win-rate
        CIs, etc.) by writing into ``AggregationResult.extra``.
        """
        evaluator_name = entries[0]["evaluator_name"]
        group_key_str = self._format_group_key(group_key)

        stats = self._compute_stats(
            scores=[e["score"] for e in entries],
            passes=[e["passed"] for e in entries],
        )

        reasons = [e["reason"] for e in entries if e["reason"]]

        summary = self._summarize(
            group_key=group_key_str,
            evaluator_name=evaluator_name,
            stats=stats,
            reasons=reasons,
        )

        return AggregationResult(
            group_key=group_key_str,
            evaluator_name=evaluator_name,
            mean_score=stats["mean_score"],
            min_score=stats["min_score"],
            max_score=stats["max_score"],
            pass_rate=stats["pass_rate"],
            num_results=stats["num_results"],
            num_passed=stats["num_passed"],
            num_failed=stats["num_failed"],
            reasons=reasons,
            summary=summary,
        )

    @staticmethod
    def _format_group_key(group_key: tuple) -> str:
        """Render a group key tuple as the ``group_key`` string field.

        For the default ``(case_key, evaluator_name)`` shape, returns the
        case_key. Subclasses with richer keys can override or rely on the
        default which joins all non-evaluator parts with ``" / "``.
        """
        if len(group_key) == 2:
            return str(group_key[0])
        return " / ".join(str(p) for p in group_key[:-1])

    @staticmethod
    def _compute_stats(scores: list[float], passes: list[bool]) -> dict[str, Any]:
        """Compute mean / min / max / pass_rate and counts over a group."""
        n = len(scores)
        if n == 0:
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
        return {
            "mean_score": float(sum(scores) / n),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "pass_rate": float(num_passed / n),
            "num_results": n,
            "num_passed": num_passed,
            "num_failed": n - num_passed,
        }

    # ------------------------------------------------------------------
    # Summarization (LLM, optional)
    # ------------------------------------------------------------------

    def _summarize(
        self,
        group_key: str,
        evaluator_name: str,
        stats: dict[str, Any],
        reasons: list[str],
    ) -> str:
        """Generate an LLM summary. Returns empty string when no model is set."""
        if self.model is None:
            return ""

        lines = [
            f"Group: {group_key}",
            f"Evaluator: {evaluator_name}",
            (
                f"Trials: n={stats['num_results']}, "
                f"pass_rate={stats['pass_rate']:.3f}, "
                f"mean_score={stats['mean_score']:.3f}, "
                f"min_score={stats['min_score']:.3f}, "
                f"max_score={stats['max_score']:.3f}"
            ),
        ]

        if reasons:
            lines.append("")
            lines.append("Sample reasons:")
            for r in reasons[:5]:
                lines.append(f"  - {r}")

        prompt = "\n".join(lines)
        return self._invoke_summary_model(prompt)

    def _invoke_summary_model(self, prompt: str) -> str:
        """Invoke the configured model. Isolated for easy override in tests."""
        try:
            # Imported lazily so the package does not require strands at import
            # time when no summarizer is configured.
            from strands import Agent

            agent = Agent(
                model=self.model,
                system_prompt=self.system_prompt,
                callback_handler=None,
            )
            result = agent(prompt, structured_output_model=_Summary)
            return cast(_Summary, result.structured_output).summary
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            return ""


# ----------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------


def render_display(report: AggregationReport) -> None:
    """Render an ``AggregationReport`` to the terminal.

    Layout:
      1. A flat results table: one row per group with mean / pass-rate / n.
      2. Per-group summary panels when summaries are present.
    """
    console = Console()
    if not report.aggregations:
        console.print("[dim]No aggregations to display.[/dim]")
        return

    _render_results_table(console, report)

    for a in report.aggregations:
        if not a.summary:
            continue
        title = f"{a.group_key} — {a.evaluator_name}"
        console.print(Panel(a.summary, title=title, expand=False))


def _render_results_table(console: Console, report: AggregationReport) -> None:
    table = Table(title="Aggregation Results", show_lines=False)
    table.add_column("Group", style="cyan", no_wrap=False)
    table.add_column("Evaluator", style="magenta")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("n", justify="right")

    for a in report.aggregations:
        table.add_row(
            a.group_key,
            a.evaluator_name,
            f"{a.mean_score:.3f}",
            f"{a.min_score:.3f}",
            f"{a.max_score:.3f}",
            f"{a.pass_rate:.2%}",
            str(a.num_results),
        )
    console.print(table)
