"""Rich console display for ChaosScenarioAggregation results.

Interactive table with expand/collapse. Collapsed shows a summary row per case.
Expanded shows the pretty view: Stats + Summary panels on top, Coverage Matrix below.
"""

import re

from rich.panel import Panel
from rich.table import Table

from ..display.display_console import CollapsibleTableReportDisplay, console
from .aggregator_types import CoverageStatus

# All effects in display order: error effects first, then corruption effects
_ALL_EFFECTS = [
    "timeout",
    "network_error",
    "execution_error",
    "validation_error",
    "truncate_fields",
    "remove_fields",
    "corrupt_values",
]

_EFFECT_SHORT_NAMES = {
    "timeout": "TIMEOUT",
    "network_error": "NET_ERR",
    "execution_error": "EXEC_ERR",
    "validation_error": "VALID_ERR",
    "truncate_fields": "TRUNCATE",
    "remove_fields": "REMOVE",
    "corrupt_values": "CORRUPT",
}


class ChaosAggregationDisplay(CollapsibleTableReportDisplay):
    """Interactive console display for chaos scenario aggregation results.

    Collapsed: single summary row per case (name, avg score, pass rate).
    Expanded: Stats + Summary panels on top, full Coverage Matrix below.
    """

    def __init__(self, aggregations: list, reports: list | None = None):
        """Initialize the display from aggregation results.

        Args:
            aggregations: List of ChaosScenarioAggregation objects.
            reports: Optional flat list of EvaluationReport objects (for input display).
        """
        self._aggregations = aggregations
        self._reports = reports

        # Build items dict for the base class interaction loop
        items = {}
        overall_score = 0.0
        if aggregations:
            overall_score = sum(a.mean_score for a in aggregations) / len(aggregations)
            for i, agg in enumerate(aggregations):
                items[str(i)] = {
                    "details": {
                        "name": agg.group_key,
                        "score": f"{agg.mean_score:.2f}",
                        "test_pass": agg.pass_rate >= 0.5,
                    },
                    "detailed_results": [],
                    "expanded": False,
                }

        super().__init__(items=items, overall_score=overall_score)

    # Evaluators where 0.5 is the expected neutral baseline score
    _NEUTRAL_BASELINE_EVALUATORS = frozenset({
        "RecoveryStrategyEvaluator",
        "FailureCommunicationEvaluator",
    })

    def display_items(self):
        """Render the aggregation report."""
        if not self._aggregations:
            console.print(
                Panel("[bold blue]No aggregation results[/bold blue]", title="📊 Chaos Aggregation Report")
            )
            return

        # Compute per-evaluator stats for header
        from collections import defaultdict
        eval_stats: dict[str, dict] = defaultdict(lambda: {"scores": [], "passes": []})
        for agg in self._aggregations:
            eval_stats[agg.evaluator_name]["scores"].append(agg.mean_score)
            eval_stats[agg.evaluator_name]["passes"].append(agg.pass_rate)

        # Scenarios count and case count
        num_scenarios = self._aggregations[0].num_results if self._aggregations else 0
        case_names = set(agg.group_key for agg in self._aggregations)
        num_cases = len(case_names)
        num_evaluators = len(eval_stats)

        # Build header with mini-table inside panel
        header_table = Table(show_header=True, show_edge=False, box=None, padding=(0, 2))
        header_table.add_column("Evaluator", style="bold")
        header_table.add_column("Avg Score", justify="center", style="green")
        header_table.add_column("Pass Rate", justify="center", style="green")
        header_table.add_column("", style="dim")

        for eval_name in sorted(eval_stats.keys()):
            stats = eval_stats[eval_name]
            avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            pr = sum(stats["passes"]) / len(stats["passes"]) if stats["passes"] else 0.0
            note = "(0.5 = neutral baseline)" if eval_name in self._NEUTRAL_BASELINE_EVALUATORS else ""
            header_table.add_row(eval_name, f"{avg:.2f}", f"{pr:.0%}", note)

        from rich.console import Group
        from rich.text import Text

        dimensions = Text(f"Cases: {num_cases}    Scenarios: {num_scenarios}    Evaluators: {num_evaluators}")
        dimensions.stylize("bold blue")

        console.print(Panel(
            Group(dimensions, Text(""), header_table),
            title="📊 Chaos Aggregation Report",
        ))

        # Summary table — one row per (case, evaluator)
        table = Table(title="Test Case Results", show_lines=True)
        table.add_column("index", style="cyan")
        table.add_column("name", style="magenta")
        table.add_column("evaluator", style="yellow")
        table.add_column("avg_score", style="green")
        table.add_column("baseline_score", style="green")
        table.add_column("pass_rate", style="green")

        for i, agg in enumerate(self._aggregations):
            key = str(i)
            expanded = self.items[key]["expanded"]
            symbol = "▼" if expanded else "▶"
            baseline = f"{agg.baseline_score:.2f}" if agg.baseline_score is not None else "—"

            # Evaluator name with neutral baseline note
            evaluator_cell = agg.evaluator_name
            if agg.evaluator_name in self._NEUTRAL_BASELINE_EVALUATORS:
                evaluator_cell = f"{agg.evaluator_name}\n[dim](0.5 = neutral baseline)[/dim]"

            table.add_row(
                f"{symbol} {i}",
                agg.group_key,
                evaluator_cell,
                f"{agg.mean_score:.2f}",
                baseline,
                f"{agg.pass_rate:.0%}",
            )

        console.print(table)

        # Expanded detail panels for each expanded case
        for i, agg in enumerate(self._aggregations):
            key = str(i)
            if not self.items[key]["expanded"]:
                continue

            console.print()
            console.print(f"[bold magenta]Case: {agg.group_key}[/bold magenta]")
            console.print(f"[dim]Evaluator: {agg.evaluator_name}[/dim]")

            # Top row: Stats (left) + Summary (right)
            stats_panel = self._build_stats_panel(agg)
            summary_panel = self._build_summary_panel(agg)

            top_row = Table(show_header=False, show_edge=False, box=None, expand=True, padding=0)
            top_row.add_column(ratio=1)
            top_row.add_column(ratio=2)
            top_row.add_row(stats_panel, summary_panel)
            console.print(top_row)

            # Bottom: Coverage Matrix
            matrix_panel = self._build_coverage_matrix_panel(agg)
            console.print(matrix_panel)

    @staticmethod
    def _build_stats_panel(agg) -> Panel:
        """Build the stats panel."""
        stats_lines = [
            f"[bold]avg_score:[/bold]    {agg.mean_score:.2f}",
            f"[bold]min_score:[/bold]    {agg.min_score:.2f}",
            f"[bold]max_score:[/bold]    {agg.max_score:.2f}",
            f"[bold]pass_rate:[/bold]    {agg.pass_rate:.0%} ({agg.num_passed}/{agg.num_results})",
        ]
        if agg.baseline_score is not None:
            bl_status = "✅" if agg.baseline_passed else "❌"
            stats_lines.append(f"[bold]baseline:[/bold]     {agg.baseline_score:.2f} {bl_status}")
        if agg.degradation_from_baseline is not None:
            stats_lines.append(f"[bold]degradation:[/bold]  {agg.degradation_from_baseline:.2f}")

        return Panel(
            "\n".join(stats_lines),
            title="[bold blue]Stats[/bold blue]",
            border_style="blue",
        )

    @staticmethod
    def _build_summary_panel(agg) -> Panel:
        """Build the summary/reason panel."""
        summary_text = agg.summary if agg.summary else "[dim]No summary available[/dim]"
        return Panel(
            summary_text,
            title="[bold cyan]Summary[/bold cyan]",
            border_style="cyan",
        )

    @staticmethod
    def _build_coverage_matrix_panel(agg) -> Panel:
        """Build the full-width coverage matrix with all effects as columns."""
        matrix_table = Table(
            show_header=True,
            show_lines=True,
            expand=True,
        )

        matrix_table.add_column("tool", style="bold", no_wrap=True)
        for effect in _ALL_EFFECTS:
            short_name = _EFFECT_SHORT_NAMES.get(effect, effect.upper())
            matrix_table.add_column(short_name, justify="center")

        for tool_name in sorted(agg.coverage_matrix.keys()):
            tool_effects = agg.coverage_matrix[tool_name]
            cells = [tool_name]
            for effect in _ALL_EFFECTS:
                status = tool_effects.get(effect, CoverageStatus.NOT_TESTED)
                if status == CoverageStatus.PASSED:
                    cells.append("[green bold]PASS[/green bold]")
                elif status == CoverageStatus.FAILED:
                    cells.append("[red bold]FAIL[/red bold]")
                else:
                    cells.append("[dim]—[/dim]")
            matrix_table.add_row(*cells)

        return Panel(
            matrix_table,
            title="[bold yellow]Coverage Matrix[/bold yellow]",
            subtitle="[dim]Error Effects (pre-hook) │ Corruption Effects (post-hook)[/dim]",
            border_style="yellow",
        )


def display_chaos_aggregation(
    aggregations: list,
    reports: list | None = None,
    static: bool = False,
):
    """Display chaos aggregation results.

    Shows an interactive table with one row per case. Expanding a case
    reveals Stats + Summary panels and a full Coverage Matrix.

    Args:
        aggregations: List of ChaosScenarioAggregation objects.
        reports: Optional flat list of EvaluationReport objects.
        static: If True, display once without interaction.
    """
    display = ChaosAggregationDisplay(aggregations, reports=reports)
    display.run(static=static)
