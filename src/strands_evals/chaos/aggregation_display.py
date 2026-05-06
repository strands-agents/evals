"""Rich console display for ChaosScenarioAggregation results.

Supports two display modes:
- "traditional": Interactive table with expand/collapse (inherits from base)
- "pretty": Side-by-side panels (Stats | Coverage Matrix | Reason) per case
"""

import re
from collections import OrderedDict

from rich.panel import Panel
from rich.table import Table

from ..display.display_console import CollapsibleTableReportDisplay, console
from .aggregator_types import CoverageStatus


class ChaosAggregationDisplay(CollapsibleTableReportDisplay):
    """Interactive console display for chaos scenario aggregation results.

    Inherits the expand/collapse interaction loop from CollapsibleTableReportDisplay.
    Overrides display_items() to render a table where each test case is a group
    of rows (one per scenario), separated by horizontal lines between cases.

    Columns: index | name | scenario | score | test_pass | reason | input
    """

    def __init__(self, aggregations: list, reports: list | None = None):
        """Initialize the display from aggregation results.

        Args:
            aggregations: List of ChaosScenarioAggregation objects.
            reports: Optional flat list of EvaluationReport objects from the
                ChaosExperiment run. Used to extract per-scenario input values.
        """
        self._aggregations = aggregations
        self._reports = reports

        # Build input lookup: (original_case_name, scenario_name) -> input
        self._input_lookup: dict[tuple[str, str], str] = {}
        if reports:
            suffix_re = re.compile(r"\s*\[([^\]]+)\]\s*$")
            for report in reports:
                for case_data in report.cases:
                    raw_name = case_data.get("name", "") or ""
                    metadata = case_data.get("metadata") or {}
                    scenario = metadata.get("chaos_scenario", "")

                    match = suffix_re.search(raw_name)
                    original_name = raw_name[: match.start()].strip() if match else raw_name

                    input_val = case_data.get("input", "")
                    self._input_lookup[(original_name, scenario)] = (
                        input_val if isinstance(input_val, str) else str(input_val)
                    )

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

    def display_items(self):
        """Render the aggregation report in traditional table mode."""
        # Header panel
        if not self._aggregations:
            console.print(
                Panel("[bold blue]No aggregation results[/bold blue]", title="📊 Chaos Aggregation Report")
            )
            return

        total_scenarios = sum(a.num_results for a in self._aggregations)
        total_passed = sum(a.num_passed for a in self._aggregations)
        overall_pass_rate = total_passed / total_scenarios if total_scenarios else 0.0
        mean_scores = [a.mean_score for a in self._aggregations]
        overall_mean = sum(mean_scores) / len(mean_scores) if mean_scores else 0.0

        score_str = f"[bold blue]Overall Score: {overall_mean:.2f}[/bold blue]"
        pass_str = f"[bold blue]Pass Rate: {overall_pass_rate:.0%}[/bold blue]"
        spacing = "           "
        console.print(Panel(f"{score_str}{spacing}{pass_str}", title="📊 Chaos Aggregation Report"))

        # Table
        table = Table(title="Test Case Results", show_lines=True)
        table.add_column("index", style="cyan")
        table.add_column("name", style="magenta")
        table.add_column("scenario", style="yellow")
        table.add_column("score", style="green")
        table.add_column("test_pass", style="green")
        table.add_column("reason", style="dim")
        table.add_column("input", style="dim")

        for i, agg in enumerate(self._aggregations):
            key = str(i)
            expanded = self.items[key]["expanded"]
            symbol = "▼" if expanded else "▶"

            if not expanded:
                table.add_row(
                    f"{symbol} {i}",
                    agg.group_key,
                    f"{agg.num_results} scenarios",
                    f"{agg.mean_score:.2f}",
                    f"{agg.pass_rate:.0%}",
                    "...",
                    "...",
                )
            else:
                # Baseline row
                if agg.baseline_score is not None:
                    bl_icon = "✅" if agg.baseline_passed else "❌"
                    bl_input = self._input_lookup.get((agg.group_key, "baseline"), "")
                    table.add_row(
                        f"{symbol} {i}",
                        agg.group_key,
                        "baseline\n[dim](no chaos)[/dim]",
                        f"{agg.baseline_score:.2f}",
                        bl_icon,
                        "",
                        bl_input,
                    )

                # Group scenario_results by scenario_label
                scenarios_grouped: OrderedDict[str, list] = OrderedDict()
                for sr in agg.scenario_results:
                    scenarios_grouped.setdefault(sr.scenario_label, []).append(sr)

                for scenario_label, srs in scenarios_grouped.items():
                    effects_parts = [f"{sr.tool_name}:{sr.effect_type}" for sr in srs]
                    effects_str = ", ".join(effects_parts)
                    first_sr = srs[0]
                    icon = "✅" if first_sr.passed else "❌"
                    sr_input = self._input_lookup.get((agg.group_key, scenario_label), "")

                    scenario_cell = f"{scenario_label}\n[dim]({effects_str})[/dim]"
                    table.add_row(
                        "",
                        "",
                        scenario_cell,
                        f"{first_sr.score:.2f}",
                        icon,
                        first_sr.reason,
                        sr_input,
                    )

                # Summary row
                table.add_row(
                    "",
                    "",
                    "[bold]SUMMARY[/bold]",
                    f"[bold]avg={agg.mean_score:.2f}[/bold]",
                    f"[bold]{agg.pass_rate:.0%}[/bold]",
                    agg.summary,
                    "",
                )

        console.print(table)


class ChaosAggregationPrettyDisplay:
    """Pretty display mode: Stats + Summary on top, Coverage Matrix below.

    For each test case, renders:
    - Top row: Stats panel (left) | Summary panel (right)
    - Bottom: Full-width Coverage Matrix with all tool failure types as columns
    """

    # All effects in display order: error effects first, then corruption effects
    _ALL_EFFECTS = [
        # Pre-hook (error effects)
        "timeout",
        "network_error",
        "execution_error",
        "validation_error",
        # Post-hook (corruption effects)
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

    def __init__(self, aggregations: list):
        """Initialize the pretty display.

        Args:
            aggregations: List of ChaosScenarioAggregation objects.
        """
        self._aggregations = aggregations

    def display(self):
        """Render the pretty display."""
        if not self._aggregations:
            console.print(
                Panel("[bold blue]No aggregation results[/bold blue]", title="📊 Chaos Aggregation Report")
            )
            return

        # Overall header
        total_scenarios = sum(a.num_results for a in self._aggregations)
        total_passed = sum(a.num_passed for a in self._aggregations)
        overall_pass_rate = total_passed / total_scenarios if total_scenarios else 0.0
        mean_scores = [a.mean_score for a in self._aggregations]
        overall_mean = sum(mean_scores) / len(mean_scores) if mean_scores else 0.0

        score_str = f"[bold blue]Overall Score: {overall_mean:.2f}[/bold blue]"
        pass_str = f"[bold blue]Pass Rate: {overall_pass_rate:.0%}[/bold blue]"
        spacing = "           "
        console.print(Panel(f"{score_str}{spacing}{pass_str}", title="📊 Chaos Aggregation Report"))
        console.print()

        # Per-case display
        for agg in self._aggregations:
            console.print(f"[bold magenta]Case: {agg.group_key}[/bold magenta]")
            console.print(f"[dim]Evaluator: {agg.evaluator_name}[/dim]")

            # --- Top row: Stats (left) + Summary (right) ---
            stats_panel = self._build_stats_panel(agg)
            summary_panel = self._build_summary_panel(agg)

            # Use a 2-column table for reliable side-by-side layout
            top_row = Table(show_header=False, show_edge=False, box=None, expand=True, padding=0)
            top_row.add_column(ratio=1)
            top_row.add_column(ratio=2)
            top_row.add_row(stats_panel, summary_panel)

            # --- Bottom: Full-width Coverage Matrix ---
            matrix_panel = self._build_coverage_matrix_panel(agg)

            console.print(top_row)
            console.print(matrix_panel)
            console.print()

    def _build_stats_panel(self, agg) -> Panel:
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

    def _build_summary_panel(self, agg) -> Panel:
        """Build the summary/reason panel."""
        summary_text = agg.summary if agg.summary else "[dim]No summary available[/dim]"
        return Panel(
            summary_text,
            title="[bold cyan]Summary[/bold cyan]",
            border_style="cyan",
        )

    def _build_coverage_matrix_panel(self, agg) -> Panel:
        """Build the full-width coverage matrix with all effects as columns."""
        matrix_table = Table(
            show_header=True,
            show_lines=True,
            expand=True,
        )

        # First column: tool name
        matrix_table.add_column("tool", style="bold", no_wrap=True)

        # Add a column for each effect type (all 7)
        for effect in self._ALL_EFFECTS:
            short_name = self._EFFECT_SHORT_NAMES.get(effect, effect.upper())
            matrix_table.add_column(short_name, justify="center")

        # Add rows for each tool
        for tool_name in sorted(agg.coverage_matrix.keys()):
            tool_effects = agg.coverage_matrix[tool_name]
            cells = [tool_name]
            for effect in self._ALL_EFFECTS:
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
    mode: str = "traditional",
):
    """Display chaos aggregation results.

    Args:
        aggregations: List of ChaosScenarioAggregation objects.
        reports: Optional flat list of EvaluationReport objects (for input display).
        static: If True, display once without interaction (traditional mode only).
        mode: Display mode — "traditional" for interactive table,
            "pretty" for side-by-side panels (Stats | Coverage Matrix | Reason).
    """
    if mode == "pretty":
        display = ChaosAggregationPrettyDisplay(aggregations)
        display.display()
    else:
        display = ChaosAggregationDisplay(aggregations, reports=reports)
        display.run(static=static)
