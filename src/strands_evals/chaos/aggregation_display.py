"""Rich console display for ChaosScenarioAggregation results.

Interactive table with expand/collapse. Collapsed shows a summary row per case.
Expanded shows Stats + Summary panels on top, scenario-centric Coverage Matrix below.

The coverage matrix is scenario-centric:
- Rows = scenarios
- Columns = tools
- Cells = effects applied to that tool in that scenario + pass/fail
"""

from rich.panel import Panel
from rich.table import Table

from ..display.display_console import CollapsibleTableReportDisplay, console
from .aggregator_types import CoverageStatus


class ChaosAggregationDisplay(CollapsibleTableReportDisplay):
    """Interactive console display for chaos scenario aggregation results.

    Collapsed: single summary row per case (name, avg score, pass rate).
    Expanded: Stats + Summary panels on top, scenario-centric Coverage Matrix below.
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

            # Bottom: Scenario-centric Coverage Matrix
            matrix_panel = self._build_scenario_matrix_panel(agg)
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
    def _build_scenario_matrix_panel(agg) -> Panel:
        """Build a scenario-centric coverage matrix.

        Rows = scenarios (from scenario_results)
        Columns = tools
        Cells = effects applied + pass/fail
        """
        # Collect all tools from the coverage matrix
        all_tools = sorted(agg.coverage_matrix.keys())

        if not all_tools and not agg.scenario_results:
            return Panel("[dim]No scenario data[/dim]", title="[bold yellow]Coverage Matrix[/bold yellow]")

        # Group scenario_results by scenario_label
        from collections import defaultdict
        scenarios: dict[str, list] = defaultdict(list)
        for sr in agg.scenario_results:
            scenarios[sr.scenario_label].append(sr)

        # Build table: rows = scenarios, columns = tools
        matrix_table = Table(show_header=True, show_lines=True, expand=True)
        matrix_table.add_column("scenario", style="bold", no_wrap=True)

        for tool in all_tools:
            matrix_table.add_column(tool, justify="center")

        matrix_table.add_column("score", justify="center", style="green")
        matrix_table.add_column("result", justify="center")

        for scenario_name, results in scenarios.items():
            # Build a lookup: tool_name -> effect_type for this scenario
            tool_to_effect: dict[str, str] = {}
            scenario_passed = True
            scenario_score = 0.0

            for r in results:
                tool_to_effect[r.tool_name] = r.effect_type
                scenario_score = r.score  # All results in same scenario share the score
                if not r.passed:
                    scenario_passed = False

            cells = [scenario_name]
            for tool in all_tools:
                effect = tool_to_effect.get(tool)
                if effect is None:
                    cells.append("[dim]—[/dim]")
                else:
                    # Show effect name with color based on pass/fail
                    status = agg.coverage_matrix.get(tool, {}).get(effect, CoverageStatus.NOT_TESTED)
                    if status == CoverageStatus.PASSED:
                        cells.append(f"[green]{effect}[/green]")
                    elif status == CoverageStatus.FAILED:
                        cells.append(f"[red]{effect}[/red]")
                    else:
                        cells.append(f"[dim]{effect}[/dim]")

            cells.append(f"{scenario_score:.2f}")
            cells.append("[green bold]PASS[/green bold]" if scenario_passed else "[red bold]FAIL[/red bold]")
            matrix_table.add_row(*cells)

        return Panel(
            matrix_table,
            title="[bold yellow]Coverage Matrix[/bold yellow]",
            subtitle="[dim]Rows = scenarios │ Columns = tools │ Cells = effect applied[/dim]",
            border_style="yellow",
        )


def display_chaos_aggregation(
    aggregations: list,
    reports: list | None = None,
    static: bool = False,
):
    """Display chaos aggregation results.

    Shows an interactive table with one row per case. Expanding a case
    reveals Stats + Summary panels and a scenario-centric Coverage Matrix.

    Args:
        aggregations: List of ChaosScenarioAggregation objects.
        reports: Optional flat list of EvaluationReport objects.
        static: If True, display once without interaction.
    """
    display = ChaosAggregationDisplay(aggregations, reports=reports)
    display.run(static=static)
