"""SkillEvalAggregator — paired-comparison aggregator for skills evaluation.

Given a flat list of EvaluationReports from a SkillEvalExperiment, this
aggregator:

1. Re-groups results by (original_case_name, evaluator_name).
2. Within each group, pairs trials by trial_idx using metadata flags:
   - case.metadata["variant_label"] identifies baseline vs variant
   - case.metadata["trial_idx"] identifies the pair
3. For each (case, metric), computes a paired statistical test
   (Wilcoxon / paired-t / McNemar) plus a bootstrap CI on the delta.
4. Filters out pairs where either side is marked corrupted before stats.

Efficiency metrics (tokens, latency, cost) are read from
``case.metadata["efficiency"]`` — a dict like
``{"tokens_in": int, "tokens_out": int, "latency_s": float, "cost_usd": float,
"tool_calls": int}``. Missing keys are treated as missing data for that
trial / metric.
"""

import logging
from collections import defaultdict
from typing import Any, Optional, cast

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats
from strands import Agent
from strands.models.model import Model

from ..aggregators.base import CaseAggregator
from ..types.evaluation_report import EvaluationReport
from .aggregator_types import (
    PairedComparisonStats,
    SkillEvalAggregation,
    SkillEvalAggregationReport,
)

logger = logging.getLogger(__name__)


# Default metrics aggregated. Override via constructor `metrics=`.
_DEFAULT_METRICS = ("pass_rate", "tokens", "latency_s", "cost_usd")

# Test selection: "auto" picks between wilcoxon and paired_t based on normality.
_VALID_TESTS = {"auto", "wilcoxon", "paired_t", "mcnemar"}

# Normality threshold for "auto" — paired-t when Shapiro-Wilk fails to reject.
_NORMALITY_ALPHA = 0.05

# Bootstrap CI configuration.
_BOOTSTRAP_RESAMPLES = 1000
_CI_LEVEL = 0.95


# Default LLM prompt for narrative summarization.
_SUMMARIZE_SYSTEM_PROMPT = """\
You are an evaluation analyst for AI agent skill comparisons.

You will receive paired comparison results for one task across multiple
metrics (pass_rate, tokens, latency_s, cost_usd). The agent was run twice
on each trial — once with a baseline configuration, once with a variant.

Produce a single paragraph (100 words max) that:
1. States whether the variant improved or degraded performance on this task.
2. Identifies which metrics moved meaningfully (effect size + statistical
   significance) and which did not.
3. Notes any tradeoffs (e.g., "improves pass rate but at 30% higher cost").

Be specific. Do not repeat raw numbers verbatim — interpret them. Do not
use bullet points or multiple paragraphs.
"""


class SkillSummary(BaseModel):
    """Structured output for LLM-based skill comparison summarization."""

    reasoning: str = Field(description="Brief analysis of the variant vs baseline comparison")
    summary: str = Field(description="Single paragraph summary")


class SkillEvalAggregator(CaseAggregator):
    """Aggregates evaluation results across baseline / variant pairs.

    Designed to work with the output of SkillEvalExperiment, which tags each
    case with metadata["variant_label"] and metadata["trial_idx"]. Produces
    one SkillEvalAggregation per (original_case, evaluator) pair.

    Args:
        baseline_label: Variant label that identifies the baseline condition.
            Defaults to "baseline".
        variant_label: Variant label that identifies the variant condition.
            Defaults to "variant".
        metrics: Metric names to aggregate. Defaults to ("pass_rate", "tokens",
            "latency_s", "cost_usd"). "pass_rate" is special-cased to read from
            test_passes; all others are read from
            case.metadata["efficiency"][metric_name].
        stats_test: Statistical test to use. One of "auto", "wilcoxon",
            "paired_t", "mcnemar". "auto" picks wilcoxon vs paired_t per
            metric based on Shapiro-Wilk normality at alpha=0.05;
            pass_rate always uses mcnemar regardless of this setting.
        model: Model for LLM-as-a-Judge summarization. Accepts a model ID
            string or a Model instance. If None (default), summarization is
            skipped and the summary field is empty.
        system_prompt: Optional custom system prompt for the summarization
            judge.
        name: Optional human-readable name for this aggregator.

    Example::

        aggregator = SkillEvalAggregator(
            metrics=["pass_rate", "tokens", "latency_s"],
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        )

        reports = experiment.run_evaluations(task=my_task)
        report = aggregator.aggregate(reports)
        report.run_display()
    """

    def __init__(
        self,
        baseline_label: str = "baseline",
        variant_label: str = "variant",
        metrics: tuple[str, ...] | list[str] = _DEFAULT_METRICS,
        stats_test: str = "auto",
        model: Optional[Model | str] = None,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "SkillEvalAggregator")
        if stats_test not in _VALID_TESTS:
            raise ValueError(
                f"stats_test must be one of {_VALID_TESTS}, got {stats_test!r}"
            )
        self.baseline_label = baseline_label
        self.variant_label = variant_label
        self.metrics = list(metrics)
        self.stats_test = stats_test
        self.model = model
        self.system_prompt = system_prompt or _SUMMARIZE_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, reports: list[EvaluationReport]) -> SkillEvalAggregationReport:
        """Aggregate skill experiment reports into per-(case, evaluator) results.

        Args:
            reports: Flat list of EvaluationReport objects from SkillEvalExperiment.

        Returns:
            SkillEvalAggregationReport with .run_display() and .to_file() methods.
        """
        if not reports:
            return SkillEvalAggregationReport(aggregations=[])

        grouped = self._group_results(reports)

        aggregations = []
        for (case_name, evaluator_name), entries in grouped.items():
            aggregation = self._build_aggregation(case_name, evaluator_name, entries)
            aggregations.append(aggregation)

        aggregations.sort(key=lambda a: (a.group_key, a.evaluator_name))
        return SkillEvalAggregationReport(aggregations=aggregations)

    # ------------------------------------------------------------------
    # Internal grouping
    # ------------------------------------------------------------------

    def _group_results(
        self, reports: list[EvaluationReport]
    ) -> dict[tuple[str, str], list[dict]]:
        """Group report entries by (original_case_name, evaluator_name).

        Each entry is a dict with: variant_label, trial_idx, score, passed,
        reason, corrupted, efficiency, session_id, metadata.
        """
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for report in reports:
            evaluator_name = report.evaluator_name or "Unknown"

            for i, case_data in enumerate(report.cases):
                metadata = case_data.get("metadata") or {}
                variant_label = metadata.get("variant_label")
                trial_idx = metadata.get("trial_idx")
                if variant_label is None or trial_idx is None:
                    logger.debug(
                        "Skipping case without variant_label/trial_idx metadata: %s",
                        case_data.get("name"),
                    )
                    continue

                original_name = metadata.get("original_case_name") or case_data.get("name", "") or ""

                grouped[(original_name, evaluator_name)].append({
                    "variant_label": variant_label,
                    "trial_idx": trial_idx,
                    "score": report.scores[i] if i < len(report.scores) else 0.0,
                    "passed": report.test_passes[i] if i < len(report.test_passes) else False,
                    "reason": report.reasons[i] if i < len(report.reasons) else "",
                    "corrupted": bool(metadata.get("corrupted", False)),
                    "efficiency": metadata.get("efficiency") or {},
                    "session_id": case_data.get("session_id", ""),
                    "metadata": metadata,
                })

        return grouped

    def _build_aggregation(
        self, case_name: str, evaluator_name: str, entries: list[dict]
    ) -> SkillEvalAggregation:
        """Build a SkillEvalAggregation from grouped entries."""
        # Index entries by (variant_label, trial_idx) so we can pair them.
        by_key: dict[tuple[str, int], dict] = {
            (e["variant_label"], e["trial_idx"]): e for e in entries
        }

        # Find paired trial indices — must have both baseline and variant.
        baseline_indices = {
            tidx for (vl, tidx) in by_key if vl == self.baseline_label
        }
        variant_indices = {
            tidx for (vl, tidx) in by_key if vl == self.variant_label
        }
        paired_indices = sorted(baseline_indices & variant_indices)

        # Build pair rows, applying corruption filter.
        rows: list[dict] = []
        n_corrupt_pairs = 0
        for tidx in paired_indices:
            b = by_key[(self.baseline_label, tidx)]
            v = by_key[(self.variant_label, tidx)]
            if b["corrupted"] or v["corrupted"]:
                n_corrupt_pairs += 1
                continue
            rows.append({"trial_idx": tidx, "baseline": b, "variant": v})

        n_total = len(paired_indices)
        n_used = len(rows)

        # Compute paired stats per metric.
        paired_stats: list[PairedComparisonStats] = []
        raw_baseline_values: dict[str, list[float]] = {}
        raw_variant_values: dict[str, list[float]] = {}

        for metric in self.metrics:
            b_vals, v_vals = self._collect_metric_pairs(rows, metric)
            raw_baseline_values[metric] = b_vals
            raw_variant_values[metric] = v_vals
            stat = self._compute_paired_stat(metric, b_vals, v_vals, n_corrupt_pairs)
            if stat is not None:
                paired_stats.append(stat)

        # Compute the AggregationResult base stats from the variant-side scores.
        variant_scores = [r["variant"]["score"] for r in rows]
        variant_passes = [r["variant"]["passed"] for r in rows]
        base_stats = self._compute_stats(variant_scores, variant_passes)

        # Summarize (LLM if configured, else empty).
        if self.model is not None and paired_stats:
            summary = self._summarize_for_aggregation(
                case_name, evaluator_name, paired_stats
            )
        else:
            summary = ""

        return SkillEvalAggregation(
            group_key=case_name,
            evaluator_name=evaluator_name,
            mean_score=base_stats["mean_score"],
            min_score=base_stats["min_score"],
            max_score=base_stats["max_score"],
            pass_rate=base_stats["pass_rate"],
            num_results=base_stats["num_results"],
            num_passed=base_stats["num_passed"],
            num_failed=base_stats["num_failed"],
            summary=summary,
            paired_stats=paired_stats,
            raw_baseline_values=raw_baseline_values,
            raw_variant_values=raw_variant_values,
            n_total=n_total,
            n_corrupted=n_corrupt_pairs,
            n_used=n_used,
            trajectory_pointers_baseline=[r["baseline"]["session_id"] for r in rows],
            trajectory_pointers_variant=[r["variant"]["session_id"] for r in rows],
        )

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_metric_pairs(
        rows: list[dict], metric: str
    ) -> tuple[list[float], list[float]]:
        """Pull baseline and variant values for one metric.

        pass_rate is special-cased to use the boolean `passed` field.
        Other metrics are read from entry["efficiency"][metric]. Pairs where
        either side is missing the metric are dropped from the returned lists.
        """
        b_vals: list[float] = []
        v_vals: list[float] = []
        for r in rows:
            b_entry = r["baseline"]
            v_entry = r["variant"]

            if metric == "pass_rate":
                b_vals.append(1.0 if b_entry["passed"] else 0.0)
                v_vals.append(1.0 if v_entry["passed"] else 0.0)
                continue

            b_val = b_entry["efficiency"].get(metric)
            v_val = v_entry["efficiency"].get(metric)
            if b_val is None or v_val is None:
                continue
            try:
                b_vals.append(float(b_val))
                v_vals.append(float(v_val))
            except (TypeError, ValueError):
                logger.warning(
                    "Non-numeric value for metric %r on trial %r; skipping.",
                    metric,
                    r["trial_idx"],
                )

        return b_vals, v_vals

    # ------------------------------------------------------------------
    # Paired statistics
    # ------------------------------------------------------------------

    def _compute_paired_stat(
        self,
        metric: str,
        b_vals: list[float],
        v_vals: list[float],
        n_corrupted: int,
    ) -> Optional[PairedComparisonStats]:
        """Compute a paired comparison for one metric.

        Returns None if there is not enough data to compute the stat
        (fewer than 2 paired observations).
        """
        if len(b_vals) < 2 or len(v_vals) < 2 or len(b_vals) != len(v_vals):
            return None

        b_arr = np.asarray(b_vals, dtype=float)
        v_arr = np.asarray(v_vals, dtype=float)
        deltas = v_arr - b_arr

        b_mean = float(b_arr.mean())
        v_mean = float(v_arr.mean())
        delta = v_mean - b_mean
        delta_pct = (delta / b_mean) if abs(b_mean) > 1e-12 else None

        # Pick the test.
        if metric == "pass_rate" or self.stats_test == "mcnemar":
            test_used, p_value = self._mcnemar(b_arr, v_arr)
        elif self.stats_test == "wilcoxon":
            test_used, p_value = self._wilcoxon(deltas)
        elif self.stats_test == "paired_t":
            test_used, p_value = self._paired_t(b_arr, v_arr)
        else:  # auto
            test_used, p_value = self._auto_test(b_arr, v_arr, deltas)

        ci_low, ci_high = self._bootstrap_ci(deltas)

        return PairedComparisonStats(
            metric_name=metric,
            baseline_mean=b_mean,
            variant_mean=v_mean,
            delta=delta,
            delta_pct=delta_pct,
            test_used=test_used,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            n_used=len(b_arr),
            n_corrupted=n_corrupted,
        )

    @staticmethod
    def _wilcoxon(deltas: np.ndarray) -> tuple[str, float]:
        """Two-sided Wilcoxon signed-rank test on paired differences."""
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
        """Paired t-test."""
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
        """Pick paired_t when deltas pass Shapiro-Wilk, else wilcoxon.

        For n < 3 or all-zero deltas, falls back to wilcoxon.
        """
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
        """McNemar test for paired binary outcomes (0/1).

        Uses exact binomial when the number of discordant pairs is small
        (< 25), and the asymptotic chi-square with continuity correction
        otherwise.
        """
        b_bin = (b_arr > 0.5).astype(int)
        v_bin = (v_arr > 0.5).astype(int)
        # b: baseline=1, variant=0
        # c: baseline=0, variant=1
        b = int(np.sum((b_bin == 1) & (v_bin == 0)))
        c = int(np.sum((b_bin == 0) & (v_bin == 1)))

        n_disc = b + c
        if n_disc == 0:
            return "mcnemar", 1.0

        if n_disc < 25:
            # Exact two-sided binomial test on min(b, c) under H0: p = 0.5.
            k = min(b, c)
            try:
                result = sp_stats.binomtest(k, n_disc, 0.5, alternative="two-sided")
                return "mcnemar", float(result.pvalue)
            except AttributeError:
                # scipy < 1.7 — fall back to legacy binom_test
                p = float(sp_stats.binom_test(k, n_disc, 0.5, alternative="two-sided"))
                return "mcnemar", p

        # Asymptotic with continuity correction.
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = float(1.0 - sp_stats.chi2.cdf(statistic, df=1))
        return "mcnemar", p_value

    @staticmethod
    def _bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
        """Bootstrap percentile CI on the mean delta.

        Returns (ci_low, ci_high) at the configured level (default 95%).
        For n < 2 or all-equal deltas, returns a degenerate (mean, mean) CI.
        """
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
    # Summarization
    # ------------------------------------------------------------------

    def summarize_reasons(self, reasons: list[str]) -> str:
        """Default-style summarization across raw reason strings.

        Only used when callers explicitly invoke summarize_reasons() with a
        list of strings. The aggregator's own summarization is driven by
        `_summarize_for_aggregation()` which has richer context.
        """
        non_empty = [r for r in reasons if r]
        if not non_empty or self.model is None:
            return ""
        prompt = (
            "Summarize the following per-trial evaluation reasons into a "
            "concise 2-3 sentence summary:\n\n"
            + "\n".join(f"- {r}" for r in non_empty[:20])
        )
        try:
            agent = Agent(
                model=self.model, system_prompt=self.system_prompt, callback_handler=None
            )
            result = agent(prompt, structured_output_model=SkillSummary)
            return cast(SkillSummary, result.structured_output).summary
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return ""

    def _summarize_for_aggregation(
        self, case_name: str, evaluator_name: str, paired_stats: list[PairedComparisonStats]
    ) -> str:
        """Produce an LLM summary of the paired stats for one (case, evaluator)."""
        if self.model is None or not paired_stats:
            return ""

        metric_lines = []
        for ps in paired_stats:
            sig = "significant" if ps.p_value < 0.05 else "not significant"
            metric_lines.append(
                f"  - {ps.metric_name}: baseline={ps.baseline_mean:.3f}, "
                f"variant={ps.variant_mean:.3f}, Δ={ps.delta:+.3f} "
                f"(p={ps.p_value:.3f} via {ps.test_used}, 95% CI=[{ps.ci_low:+.3f}, "
                f"{ps.ci_high:+.3f}], {sig}, n={ps.n_used})"
            )

        prompt = (
            f"Task: {case_name}\n"
            f"Evaluator: {evaluator_name}\n\n"
            f"Paired metric comparisons (variant vs baseline):\n"
            f"{chr(10).join(metric_lines)}\n\n"
            "Summarize the variant's effect on this task."
        )

        try:
            agent = Agent(
                model=self.model, system_prompt=self.system_prompt, callback_handler=None
            )
            result = agent(prompt, structured_output_model=SkillSummary)
            return cast(SkillSummary, result.structured_output).summary
        except Exception as e:
            logger.warning(
                f"LLM summarization failed for case '{case_name}', evaluator "
                f"'{evaluator_name}': {e}"
            )
            return ""
