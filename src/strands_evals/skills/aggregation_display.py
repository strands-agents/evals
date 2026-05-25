"""Rich console display for SkillEvalAggregation results.

Interactive table with expand/collapse. Collapsed shows one row per
(case, evaluator) with Δ-metrics across the configured metrics. Expanded
shows full paired-statistics panels per metric with p-value, CI, n_used,
and the test that was selected.
"""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..display.display_console import CollapsibleTableReportDisplay, console
from .aggregator_types import SkillEvalAggregation


# Significance thresholds for visual highlighting only.
_SIG_ALPHA = 0.05
_SIG_ALPHA_STRONG = 0.01


class SkillEvalAggregationDisplay(CollapsibleTableReportDisplay):
    """Interactive console display for skill aggregation results.

    Collapsed: one summary row per (case, evaluator) with Δ-metrics and a
    significance indicator.
    Expanded: per-metric panels with full paired stats + corruption counts.
    """

    def __init__(self, aggregations: list[SkillEvalAggregation]):
        self._aggregations = aggregations

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
        """Render the skill aggregation report."""
        if not self._aggregations:
            console.print(
                Panel(
                    "[bold blue]No aggregation results[/bold blue]",
                    title="📊 Skill Evaluation Aggregation Report",
                )
            )
            return

        # Header summary.
        case_names = sorted({a.group_key for a in self._aggregations})
        evaluator_names = sorted({a.evaluator_name for a in self._aggregations})
        metric_names = sorted({
            ps.metric_name for a in self._aggregations for ps in a.paired_stats
        })

        dimensions = Text(
            f"Cases: {len(case_names)}    Evaluators: {len(evaluator_names)}    "
            f"Metrics: {len(metric_names)}"
        )
        dimensions.stylize("bold blue")

        console.print(
            Panel(
                Group(dimensions),
                title="📊 Skill Evaluation Aggregation Report",
            )
        )

        # Summary table — one row per (case, evaluator).
        # Columns: index, task, evaluator, n_used, then one Δ column per metric, then p.
        table = Table(title="Paired Comparison Summary", show_lines=True)
        table.add_column("index", style="cyan")
        table.add_column("task", style="magenta")
        table.add_column("evaluator", style="yellow")
        table.add_column("n_used", justify="center")
        table.add_column("n_corrupt", justify="center", style="dim")

        # One Δ column per metric, sorted for stable layout.
        for metric in metric_names:
            table.add_column(f"Δ{metric}", justify="right")

        table.add_column("min p", justify="center")

        for i, agg in enumerate(self._aggregations):
            key = str(i)
            expanded = self.items[key]["expanded"]
            symbol = "▼" if expanded else "▶"

            # Build a lookup: metric_name -> PairedComparisonStats.
            by_metric = {ps.metric_name: ps for ps in agg.paired_stats}

            row_cells = [
                f"{symbol} {i}",
                agg.group_key,
                agg.evaluator_name,
                str(agg.n_used),
                str(agg.n_corrupted) if agg.n_corrupted > 0 else "[dim]0[/dim]",
            ]

            for metric in metric_names:
                ps = by_metric.get(metric)
                if ps is None:
                    row_cells.append("[dim]—[/dim]")
                else:
                    row_cells.append(self._format_delta_cell(ps))

            min_p = min(
                (ps.p_value for ps in agg.paired_stats),
                default=float("nan"),
            )
            row_cells.append(self._format_p_cell(min_p))

            table.add_row(*row_cells)

        console.print(table)

        # Expanded panels.
        for i, agg in enumerate(self._aggregations):
            key = str(i)
            if not self.items[key]["expanded"]:
                continue

            console.print()
            console.print(f"[bold magenta]Task: {agg.group_key}[/bold magenta]")
            console.print(f"[dim]Evaluator: {agg.evaluator_name}[/dim]")
            console.print(
                f"[dim]Pairs: {agg.n_used} used / {agg.n_total} total "
                f"({agg.n_corrupted} dropped to corruption)[/dim]"
            )

            stats_panel = self._build_stats_panel(agg)
            summary_panel = self._build_summary_panel(agg)

            top_row = Table(show_header=False, show_edge=False, box=None, expand=True, padding=0)
            top_row.add_column(ratio=1)
            top_row.add_column(ratio=2)
            top_row.add_row(stats_panel, summary_panel)
            console.print(top_row)

            paired_panel = self._build_paired_stats_panel(agg)
            console.print(paired_panel)

    # ------------------------------------------------------------------
    # Cell formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_delta_cell(ps) -> str:
        """Format a Δmetric cell with color based on sign and significance."""
        delta = ps.delta
        p = ps.p_value

        # Sign color: positive Δ = green (often "better"), negative = red.
        # This is metric-agnostic; users should know that lower latency is good.
        if delta > 0:
            color = "green"
        elif delta < 0:
            color = "red"
        else:
            color = "white"

        # Significance weight.
        weight = ""
        if p < _SIG_ALPHA_STRONG:
            weight = "bold "
        elif p < _SIG_ALPHA:
            weight = ""
        else:
            color = "dim"

        return f"[{weight}{color}]{delta:+.3f}[/{weight}{color}]"

    @staticmethod
    def _format_p_cell(p: float) -> str:
        """Format a p-value cell."""
        if p != p:  # NaN
            return "[dim]—[/dim]"
        if p < _SIG_ALPHA_STRONG:
            return f"[bold green]{p:.3f}[/bold green]"
        if p < _SIG_ALPHA:
            return f"[green]{p:.3f}[/green]"
        return f"[dim]{p:.3f}[/dim]"

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stats_panel(agg: SkillEvalAggregation) -> Panel:
        """Variant-side aggregate stats (mirrors the chaos stats panel)."""
        lines = [
            f"[bold]variant mean_score:[/bold]  {agg.mean_score:.2f}",
            f"[bold]variant min_score:[/bold]   {agg.min_score:.2f}",
            f"[bold]variant max_score:[/bold]   {agg.max_score:.2f}",
            f"[bold]variant pass_rate:[/bold]   {agg.pass_rate:.0%} "
            f"({agg.num_passed}/{agg.num_results})",
            f"[bold]pairs used:[/bold]          {agg.n_used}/{agg.n_total}",
        ]
        if agg.n_corrupted > 0:
            lines.append(
                f"[bold]pairs dropped:[/bold]       {agg.n_corrupted} (corruption)"
            )
        return Panel(
            "\n".join(lines),
            title="[bold blue]Stats[/bold blue]",
            border_style="blue",
        )

    @staticmethod
    def _build_summary_panel(agg: SkillEvalAggregation) -> Panel:
        """LLM narrative summary (empty when no model configured)."""
        summary_text = agg.summary if agg.summary else "[dim]No summary available[/dim]"
        return Panel(
            summary_text,
            title="[bold cyan]Summary[/bold cyan]",
            border_style="cyan",
        )

    @staticmethod
    def _build_paired_stats_panel(agg: SkillEvalAggregation) -> Panel:
        """Per-metric paired statistics table."""
        if not agg.paired_stats:
            return Panel(
                "[dim]No paired statistics available[/dim]",
                title="[bold yellow]Paired Statistics[/bold yellow]",
                border_style="yellow",
            )

        t = Table(show_header=True, show_lines=True, expand=True)
        t.add_column("metric", style="bold", no_wrap=True)
        t.add_column("baseline", justify="right")
        t.add_column("variant", justify="right")
        t.add_column("Δ", justify="right")
        t.add_column("95% CI", justify="center")
        t.add_column("p", justify="center")
        t.add_column("test", justify="center", style="dim")
        t.add_column("n", justify="center")

        for ps in agg.paired_stats:
            t.add_row(
                ps.metric_name,
                f"{ps.baseline_mean:.3f}",
                f"{ps.variant_mean:.3f}",
                f"{ps.delta:+.3f}",
                f"[{ps.ci_low:+.3f}, {ps.ci_high:+.3f}]",
                SkillEvalAggregationDisplay._format_p_cell(ps.p_value),
                ps.test_used,
                str(ps.n_used),
            )

        return Panel(
            t,
            title="[bold yellow]Paired Statistics[/bold yellow]",
            subtitle=(
                "[dim]Δ = variant - baseline │ CI via 1000-resample bootstrap │ "
                "bold p < 0.01, green p < 0.05[/dim]"
            ),
            border_style="yellow",
        )


def display_skill_aggregation(
    aggregations: list[SkillEvalAggregation], static: bool = False
):
    """Display skill aggregation results.

    Args:
        aggregations: List of SkillEvalAggregation objects.
        static: If True, display once without interaction.
    """
    display = SkillEvalAggregationDisplay(aggregations)
    display.run(static=static)
