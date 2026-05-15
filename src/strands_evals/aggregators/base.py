"""General-purpose aggregator over per-trial evaluation reports.

Groups reports by ``(case_key, evaluator_name)``, computes descriptive
statistics, filters corrupted trials, rolls up efficiency metrics, and
optionally computes paired statistics when exactly two conditions are
present. An LLM summary is generated per group when a model is configured.

Inputs are ``EvaluationReport`` objects whose per-case metadata may carry:

    metadata = {
        "case_key": str,            # logical case identifier (optional;
                                    # falls back to case_data["name"])
        "condition_label": str,     # condition this trial belongs to;
                                    # required for paired comparisons
        "trial_idx": int,           # pairing key across conditions
        "corrupted": bool,          # excluded from stats when True
        "efficiency": {             # all keys optional
            "tokens_in": int,
            "tokens_out": int,
            "wall_clock_s": float,
            "cost_usd": float,
            "tool_calls": int,
        },
    }
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional, cast

import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats as sp_stats

from ..types.evaluation_report import EvaluationReport
from .types import (
    AggregationReport,
    AggregationResult,
    EfficiencyStats,
    PairedComparisonStats,
)

logger = logging.getLogger(__name__)


# Metrics aggregated for paired comparisons. Override via constructor.
_DEFAULT_METRICS = ("pass_rate", "tokens_in", "tokens_out", "wall_clock_s", "cost_usd")

# Test selection: "auto" picks paired_t / wilcoxon based on normality.
_VALID_TESTS = {"auto", "wilcoxon", "paired_t", "mcnemar"}

# Shapiro-Wilk alpha for "auto" — use paired-t when it fails to reject normality.
_NORMALITY_ALPHA = 0.05

# Bootstrap CI configuration.
_BOOTSTRAP_RESAMPLES = 1000
_CI_LEVEL = 0.95


# Default LLM system prompt for the summary field. Override via constructor.
DEFAULT_SUMMARY_PROMPT = """\
You are an evaluation analyst.

You will receive aggregated results for N agent evaluation trials. The trials
may span multiple conditions (e.g. baseline vs variant). Each trial reports a
score, a pass/fail flag, and optionally efficiency metrics (tokens, latency,
cost, tool calls).

Produce a single paragraph (100 words max) that:

1. States the overall performance pattern (high pass rate, degradation, etc.).
2. Calls out meaningful differences across conditions, if any.
3. Notes efficiency tradeoffs if relevant.

Be specific. Interpret the numbers; do not repeat them verbatim. Plain prose,
no bullets, no multiple paragraphs.
"""


class _Summary(BaseModel):
    """Structured output type for the LLM summary judge."""

    reasoning: str = Field(description="Brief analysis of the aggregated results")
    summary: str = Field(description="Single paragraph summary")


class Aggregator:
    """Rolls up per-trial ``EvaluationReport`` objects into an ``AggregationReport``.

    Args:
        name: Human-readable name for this aggregator.
        metrics: Metric names to aggregate for paired comparisons. ``pass_rate``
            is read from the boolean ``test_passes``; all others are read from
            ``case.metadata["efficiency"][metric_name]``.
        stats_test: Test for paired metric comparisons. One of ``"auto"``,
            ``"wilcoxon"``, ``"paired_t"``, ``"mcnemar"``. ``"auto"`` picks
            paired-t or wilcoxon per metric via Shapiro-Wilk; ``pass_rate``
            always uses ``mcnemar``.
        model: Optional model ID or ``Model`` instance for the LLM summary.
            When ``None``, the ``summary`` field is left empty.
        system_prompt: Optional system prompt for the summary judge. Defaults
            to ``DEFAULT_SUMMARY_PROMPT``.
    """

    def __init__(
        self,
        name: str = "Aggregator",
        metrics: tuple[str, ...] | list[str] = _DEFAULT_METRICS,
        stats_test: str = "auto",
        model: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        if stats_test not in _VALID_TESTS:
            raise ValueError(
                f"stats_test must be one of {sorted(_VALID_TESTS)}, got {stats_test!r}"
            )
        self.name = name
        self.metrics = list(metrics)
        self.stats_test = stats_test
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SUMMARY_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, reports: list[EvaluationReport]) -> AggregationReport:
        """Aggregate a flat list of evaluation reports into one report.

        Args:
            reports: Per-trial evaluation outputs. Empty list returns an
                empty report.

        Returns:
            An ``AggregationReport`` with one ``AggregationResult`` per
            ``(case_key, evaluator_name)`` group.
        """
        if not reports:
            return AggregationReport(aggregations=[])

        grouped = self._group_results(reports)

        aggregations: list[AggregationResult] = []
        for (case_key, evaluator_name), entries in grouped.items():
            aggregation = self._build_aggregation(case_key, evaluator_name, entries)
            aggregations.append(aggregation)

        aggregations.sort(key=lambda a: (a.group_key, a.evaluator_name))
        return AggregationReport(aggregations=aggregations)

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _case_key(self, case_data: dict, metadata: dict) -> str:
        """Logical case identifier. Falls back to ``case_data["name"]``."""
        return metadata.get("case_key") or case_data.get("name", "") or ""

    def _group_results(
        self, reports: list[EvaluationReport]
    ) -> dict[tuple[str, str], list[dict]]:
        """Flatten reports into per-trial entries grouped by (case_key, evaluator)."""
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for report in reports:
            evaluator_name = report.evaluator_name or "Unknown"
            for i, case_data in enumerate(report.cases):
                metadata = case_data.get("metadata") or {}
                case_key = self._case_key(case_data, metadata)
                if not case_key:
                    logger.debug("Skipping case with no case_key or name")
                    continue
                grouped[(case_key, evaluator_name)].append({
                    "condition_label": metadata.get("condition_label"),
                    "trial_idx": metadata.get("trial_idx"),
                    "score": report.scores[i] if i < len(report.scores) else 0.0,
                    "passed": report.test_passes[i] if i < len(report.test_passes) else False,
                    "reason": report.reasons[i] if i < len(report.reasons) else "",
                    "corrupted": bool(metadata.get("corrupted", False)),
                    "efficiency": metadata.get("efficiency") or {},
                    "session_id": case_data.get("session_id", ""),
                    "metadata": metadata,
                })

        return grouped

    # ------------------------------------------------------------------
    # Per-group aggregation
    # ------------------------------------------------------------------

    def _build_aggregation(
        self, case_key: str, evaluator_name: str, entries: list[dict]
    ) -> AggregationResult:
        """Build one ``AggregationResult`` from grouped trial entries."""
        n_total = len(entries)
        used_entries = [e for e in entries if not e["corrupted"]]
        n_corrupted = n_total - len(used_entries)
        n_used = len(used_entries)

        # Descriptive stats over used (non-corrupted) entries.
        base_stats = self._compute_stats(
            scores=[e["score"] for e in used_entries],
            passes=[e["passed"] for e in used_entries],
        )

        # Efficiency rollup over used entries.
        efficiency = self._aggregate_efficiency(used_entries)

        # Raw per-trial values and trajectory pointers.
        raw_values = self._collect_raw_values(used_entries)
        trajectory_pointers = [e["session_id"] for e in used_entries if e["session_id"]]

        # Paired stats when exactly 2 conditions exist with trial_idx pairing.
        paired_stats = self._compute_paired_stats_if_applicable(used_entries)

        # LLM summary (optional).
        summary = self._summarize(
            case_key=case_key,
            evaluator_name=evaluator_name,
            base_stats=base_stats,
            efficiency=efficiency,
            paired_stats=paired_stats,
            reasons=[e["reason"] for e in used_entries if e["reason"]],
        )

        return AggregationResult(
            group_key=case_key,
            evaluator_name=evaluator_name,
            mean_score=base_stats["mean_score"],
            min_score=base_stats["min_score"],
            max_score=base_stats["max_score"],
            pass_rate=base_stats["pass_rate"],
            num_results=base_stats["num_results"],
            num_passed=base_stats["num_passed"],
            num_failed=base_stats["num_failed"],
            n_total=n_total,
            n_corrupted=n_corrupted,
            n_used=n_used,
            efficiency=efficiency,
            summary=summary,
            paired_stats=paired_stats,
            raw_values=raw_values,
            trajectory_pointers=trajectory_pointers,
        )

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

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
    # Efficiency rollup
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_efficiency(entries: list[dict]) -> Optional[EfficiencyStats]:
        """Roll up efficiency metrics across used entries.

        Returns ``None`` when no entry carries any efficiency metric.
        """
        keys = ("tokens_in", "tokens_out", "wall_clock_s", "cost_usd", "tool_calls")
        sums: dict[str, float] = {k: 0.0 for k in keys}
        counts: dict[str, int] = {k: 0 for k in keys}

        for e in entries:
            eff = e.get("efficiency") or {}
            for k in keys:
                v = eff.get(k)
                if v is None:
                    continue
                try:
                    sums[k] += float(v)
                    counts[k] += 1
                except (TypeError, ValueError):
                    logger.debug("Non-numeric efficiency value for %r: %r", k, v)

        if all(c == 0 for c in counts.values()):
            return None

        def _mean(k: str) -> Optional[float]:
            return sums[k] / counts[k] if counts[k] else None

        return EfficiencyStats(
            mean_tokens_in=_mean("tokens_in"),
            mean_tokens_out=_mean("tokens_out"),
            mean_wall_clock_s=_mean("wall_clock_s"),
            mean_cost_usd=_mean("cost_usd"),
            mean_tool_calls=_mean("tool_calls"),
            n_samples=max(counts.values()),
        )

    # ------------------------------------------------------------------
    # Raw per-trial values
    # ------------------------------------------------------------------

    def _collect_raw_values(self, entries: list[dict]) -> dict[str, list[float]]:
        """Collect per-trial values for each tracked metric, in trial order."""
        out: dict[str, list[float]] = {}
        for metric in self.metrics:
            vals: list[float] = []
            for e in entries:
                if metric == "pass_rate":
                    vals.append(1.0 if e["passed"] else 0.0)
                    continue
                v = e.get("efficiency", {}).get(metric)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            if vals:
                out[metric] = vals
        return out

    # ------------------------------------------------------------------
    # Paired comparison (when exactly 2 conditions exist)
    # ------------------------------------------------------------------

    def _compute_paired_stats_if_applicable(
        self, entries: list[dict]
    ) -> list[PairedComparisonStats]:
        """Detect 2 conditions with trial_idx pairing; compute paired stats.

        Returns an empty list when fewer than two conditions are present,
        when more than two are present, or when no pairs can be formed.
        """
        labeled = [
            e for e in entries
            if e["condition_label"] is not None and e["trial_idx"] is not None
        ]
        if not labeled:
            return []

        # Preserve first-seen order so the user has a stable baseline_label.
        seen_order: list[str] = []
        for e in labeled:
            if e["condition_label"] not in seen_order:
                seen_order.append(e["condition_label"])
        if len(seen_order) != 2:
            return []

        baseline_label, variant_label = seen_order
        by_pair: dict[int, dict[str, dict]] = defaultdict(dict)
        for e in labeled:
            by_pair[e["trial_idx"]][e["condition_label"]] = e

        pairs: list[tuple[dict, dict]] = []
        for tidx, both in by_pair.items():
            if baseline_label in both and variant_label in both:
                pairs.append((both[baseline_label], both[variant_label]))

        if len(pairs) < 2:
            return []

        out: list[PairedComparisonStats] = []
        for metric in self.metrics:
            b_vals, v_vals = self._collect_metric_pairs(pairs, metric)
            stat = self._compute_paired_stat(
                metric=metric,
                baseline_label=baseline_label,
                variant_label=variant_label,
                b_vals=b_vals,
                v_vals=v_vals,
            )
            if stat is not None:
                out.append(stat)
        return out

    @staticmethod
    def _collect_metric_pairs(
        pairs: list[tuple[dict, dict]], metric: str
    ) -> tuple[list[float], list[float]]:
        """Pull baseline and variant values for one metric across pairs."""
        b_vals: list[float] = []
        v_vals: list[float] = []
        for b, v in pairs:
            if metric == "pass_rate":
                b_vals.append(1.0 if b["passed"] else 0.0)
                v_vals.append(1.0 if v["passed"] else 0.0)
                continue
            b_val = b["efficiency"].get(metric)
            v_val = v["efficiency"].get(metric)
            if b_val is None or v_val is None:
                continue
            try:
                b_vals.append(float(b_val))
                v_vals.append(float(v_val))
            except (TypeError, ValueError):
                continue
        return b_vals, v_vals

    def _compute_paired_stat(
        self,
        metric: str,
        baseline_label: str,
        variant_label: str,
        b_vals: list[float],
        v_vals: list[float],
    ) -> Optional[PairedComparisonStats]:
        """Compute one paired comparison. Returns None when too few pairs."""
        if len(b_vals) < 2 or len(v_vals) < 2 or len(b_vals) != len(v_vals):
            return None

        b_arr = np.asarray(b_vals, dtype=float)
        v_arr = np.asarray(v_vals, dtype=float)
        deltas = v_arr - b_arr

        b_mean = float(b_arr.mean())
        v_mean = float(v_arr.mean())
        delta = v_mean - b_mean
        delta_pct = (delta / b_mean) if abs(b_mean) > 1e-12 else None

        if metric == "pass_rate" or self.stats_test == "mcnemar":
            test_used, p_value = self._mcnemar(b_arr, v_arr)
        elif self.stats_test == "wilcoxon":
            test_used, p_value = self._wilcoxon(deltas)
        elif self.stats_test == "paired_t":
            test_used, p_value = self._paired_t(b_arr, v_arr)
        else:
            test_used, p_value = self._auto_test(b_arr, v_arr, deltas)

        ci_low, ci_high = self._bootstrap_ci(deltas)

        return PairedComparisonStats(
            metric_name=metric,
            baseline_label=baseline_label,
            variant_label=variant_label,
            baseline_mean=b_mean,
            variant_mean=v_mean,
            delta=delta,
            delta_pct=delta_pct,
            test_used=test_used,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            n_used=len(b_arr),
            n_corrupted=0,
        )

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def _wilcoxon(deltas: np.ndarray) -> tuple[str, float]:
        """Two-sided Wilcoxon signed-rank test."""
        if np.all(deltas == 0):
            return "wilcoxon", 1.0
        try:
            result = sp_stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            return "wilcoxon", float(result.pvalue)
        except ValueError as e:
            logger.warning("wilcoxon failed: %s", e)
            return "wilcoxon", float("nan")

    @staticmethod
    def _paired_t(b_arr: np.ndarray, v_arr: np.ndarray) -> tuple[str, float]:
        """Two-sided paired t-test."""
        try:
            result = sp_stats.ttest_rel(v_arr, b_arr)
            return "paired_t", float(result.pvalue)
        except ValueError as e:
            logger.warning("paired_t failed: %s", e)
            return "paired_t", float("nan")

    @classmethod
    def _auto_test(
        cls, b_arr: np.ndarray, v_arr: np.ndarray, deltas: np.ndarray
    ) -> tuple[str, float]:
        """Pick paired_t when deltas pass Shapiro-Wilk, else wilcoxon."""
        n = len(deltas)
        if n < 3 or np.all(deltas == 0):
            return cls._wilcoxon(deltas)
        try:
            sw = sp_stats.shapiro(deltas)
            if sw.pvalue > _NORMALITY_ALPHA:
                return cls._paired_t(b_arr, v_arr)
        except ValueError as e:
            logger.debug("shapiro failed (n=%d): %s; falling back to wilcoxon", n, e)
        return cls._wilcoxon(deltas)

    @staticmethod
    def _mcnemar(b_arr: np.ndarray, v_arr: np.ndarray) -> tuple[str, float]:
        """McNemar test for paired binary outcomes.

        Uses an exact two-sided binomial test on discordant pairs when there
        are fewer than 25 discordants, otherwise the chi-square asymptotic
        with continuity correction.
        """
        b_bin = (b_arr > 0.5).astype(int)
        v_bin = (v_arr > 0.5).astype(int)
        b = int(np.sum((b_bin == 1) & (v_bin == 0)))
        c = int(np.sum((b_bin == 0) & (v_bin == 1)))
        n_disc = b + c

        if n_disc == 0:
            return "mcnemar", 1.0
        if n_disc < 25:
            k = min(b, c)
            try:
                result = sp_stats.binomtest(k, n_disc, 0.5, alternative="two-sided")
                return "mcnemar", float(result.pvalue)
            except AttributeError:
                # scipy < 1.7 fallback.
                p = float(sp_stats.binom_test(k, n_disc, 0.5, alternative="two-sided"))
                return "mcnemar", p
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        return "mcnemar", float(1.0 - sp_stats.chi2.cdf(statistic, df=1))

    @staticmethod
    def _bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
        """Bootstrap percentile CI on the mean delta."""
        if len(deltas) < 2:
            m = float(deltas.mean()) if len(deltas) else 0.0
            return m, m
        if np.all(deltas == deltas[0]):
            v = float(deltas[0])
            return v, v
        try:
            res = sp_stats.bootstrap(
                (deltas,),
                np.mean,
                confidence_level=_CI_LEVEL,
                n_resamples=_BOOTSTRAP_RESAMPLES,
                method="percentile",
                vectorized=True,
            )
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception as e:
            logger.warning("bootstrap CI failed: %s", e)
            m = float(deltas.mean())
            return m, m

    # ------------------------------------------------------------------
    # Summarization (LLM, optional)
    # ------------------------------------------------------------------

    def _summarize(
        self,
        case_key: str,
        evaluator_name: str,
        base_stats: dict[str, Any],
        efficiency: Optional[EfficiencyStats],
        paired_stats: list[PairedComparisonStats],
        reasons: list[str],
    ) -> str:
        """Generate an LLM summary. Returns empty string when no model is set."""
        if self.model is None:
            return ""

        # Build a compact context block for the judge.
        lines = [
            f"Task: {case_key}",
            f"Evaluator: {evaluator_name}",
            f"Trials: n={base_stats['num_results']}, "
            f"pass_rate={base_stats['pass_rate']:.3f}, "
            f"mean_score={base_stats['mean_score']:.3f}",
        ]
        if efficiency is not None:
            eff_parts = []
            if efficiency.mean_tokens_in is not None:
                eff_parts.append(f"tokens_in={efficiency.mean_tokens_in:.0f}")
            if efficiency.mean_tokens_out is not None:
                eff_parts.append(f"tokens_out={efficiency.mean_tokens_out:.0f}")
            if efficiency.mean_wall_clock_s is not None:
                eff_parts.append(f"latency_s={efficiency.mean_wall_clock_s:.2f}")
            if efficiency.mean_cost_usd is not None:
                eff_parts.append(f"cost_usd={efficiency.mean_cost_usd:.4f}")
            if eff_parts:
                lines.append("Efficiency: " + ", ".join(eff_parts))

        if paired_stats:
            lines.append("Paired comparisons:")
            for ps in paired_stats:
                sig = "significant" if ps.p_value < 0.05 else "not significant"
                lines.append(
                    f"  {ps.metric_name}: {ps.baseline_label}={ps.baseline_mean:.3f}, "
                    f"{ps.variant_label}={ps.variant_mean:.3f}, Δ={ps.delta:+.3f} "
                    f"(p={ps.p_value:.3f} via {ps.test_used}, "
                    f"95% CI=[{ps.ci_low:+.3f}, {ps.ci_high:+.3f}], {sig})"
                )

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
                model=self.model, system_prompt=self.system_prompt, callback_handler=None
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
      1. A flat results table: one row per (group_key, evaluator_name).
      2. If any group has paired stats, a delta-comparison table grouped
         by case with one row per metric.
      3. Per-group summary panels when summaries are present.
    """
    console = Console()
    if not report.aggregations:
        console.print("[dim]No aggregations to display.[/dim]")
        return

    _render_results_table(console, report)

    paired_groups = [a for a in report.aggregations if a.paired_stats]
    if paired_groups:
        _render_paired_table(console, paired_groups)

    summarized = [a for a in report.aggregations if a.summary]
    for a in summarized:
        title = f"{a.group_key} — {a.evaluator_name}"
        console.print(Panel(a.summary, title=title, expand=False))


def _render_results_table(console: Console, report: AggregationReport) -> None:
    table = Table(title="Aggregation Results", show_lines=False)
    table.add_column("Case", style="cyan", no_wrap=False)
    table.add_column("Evaluator", style="magenta")
    table.add_column("Mean", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("n", justify="right")
    table.add_column("n_corrupt", justify="right")

    for a in report.aggregations:
        table.add_row(
            a.group_key,
            a.evaluator_name,
            f"{a.mean_score:.3f}",
            f"{a.pass_rate:.2%}",
            str(a.n_used),
            str(a.n_corrupted),
        )
    console.print(table)


def _render_paired_table(
    console: Console, paired_groups: list[AggregationResult]
) -> None:
    table = Table(title="Paired Comparison (variant − baseline)", show_lines=False)
    table.add_column("Case", style="cyan")
    table.add_column("Evaluator", style="magenta")
    table.add_column("Metric")
    table.add_column("Δ", justify="right")
    table.add_column("p", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("n", justify="right")

    for a in paired_groups:
        for ps in a.paired_stats:
            sig = "[bold green]" if ps.p_value < 0.05 else ""
            sig_close = "[/]" if sig else ""
            table.add_row(
                a.group_key,
                a.evaluator_name,
                ps.metric_name,
                f"{sig}{ps.delta:+.3f}{sig_close}",
                f"{ps.p_value:.3f}",
                f"[{ps.ci_low:+.3f}, {ps.ci_high:+.3f}]",
                ps.test_used,
                str(ps.n_used),
            )
    console.print(table)
