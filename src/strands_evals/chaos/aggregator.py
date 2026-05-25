"""ChaosScenarioAggregator — aggregates evaluation results across chaos scenarios.

Given a flat list of EvaluationReports from a ChaosExperiment, this aggregator:
1. Re-groups results by the original case name (stripping the [scenario] suffix).
2. Within each group, organizes results by (tool_name, effect_type) pairs
   extracted from case metadata["chaos_scenario"].
3. Produces a ChaosScenarioAggregation per (original_case, evaluator) pair
   containing quantitative stats, a coverage matrix, and baseline comparison.
4. Uses LLM-as-a-Judge to produce a narrative summary of the agent's
   resilience across scenarios (when model is provided).
"""

import logging
from collections import defaultdict
from typing import Optional, cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..aggregators.base import CaseAggregator
from ..types.evaluation_report import EvaluationReport
from .aggregator_types import (
    ChaosAggregationReport,
    ChaosScenarioAggregation,
    CoverageStatus,
    ToolEffectResult,
)

logger = logging.getLogger(__name__)

# All known effect types for coverage matrix population
_ALL_EFFECT_TYPES = [
    "timeout", "network_error", "execution_error", "validation_error",
    "truncate_fields", "remove_fields", "corrupt_values",
]

# Regex to strip the scenario suffix from case names: "case_name|scenario_name"
_SCENARIO_SEPARATOR = "|"

# The baseline scenario name used by ChaosExperiment
_BASELINE_SCENARIO_NAME = "baseline"

# Default system prompt for LLM-based reason summarization
_SUMMARIZE_SYSTEM_PROMPT = """\
You are an evaluation analyst for AI agent resilience testing.

You will receive per-scenario evaluation results from chaos testing, where each
scenario injected one or more failures into the agent's environment.

Produce a single paragraph (100 words max) that:
1. States which failure modes the agent handled well vs. poorly.
2. Notes any pattern (e.g., "handles timeouts but not data corruption").
3. Highlights the most critical gap if one stands out.

Be specific and actionable. Do not repeat raw reasons. Do not use bullet points nor multiple paragraphs.
"""

_SUMMARIZE_USER_TEMPLATE = """\
Case: {case_name}
Evaluator: {evaluator_name}
Baseline passed: {baseline_passed}
Pass rate under chaos: {pass_rate:.0%} ({num_passed}/{num_results} scenarios passed)

Per-scenario results:
{scenario_details}

Summarize the agent's resilience pattern for this case.
"""


class ResilienceSummary(BaseModel):
    """Structured output for LLM-based resilience summarization."""

    reasoning: str = Field(description="Brief analysis of the agent's resilience patterns")
    summary: str = Field(description="Single paragraph summary")


class ChaosScenarioAggregator(CaseAggregator):
    """Aggregates evaluation results across chaos scenarios for each original case.

    Designed to work with the output of ChaosExperiment, which tags each case
    with metadata["chaos_scenario"] and appends "[scenario_name]" to case names.

    Produces one ChaosScenarioAggregation per (original_case, evaluator) pair.

    Args:
        known_tools: Optional list of tool names that could be tested. Used to
            populate NOT_TESTED entries in the coverage matrix for tools that
            weren't covered by any scenario.
        known_effects: Optional list of effect types to track. Defaults to all
            known effect types.
        scenarios: Optional list of ChaosScenario objects. When provided, the
            aggregator uses typed scenario lookup instead of metadata/name parsing.
            Automatically populated by ChaosExperiment if not set.
        model: Model for LLM-as-a-Judge reason summarization. Accepts a model ID
            string or a Model instance. If None (default), summarization uses
            simple concatenation — no model calls are made.
        system_prompt: Optional custom system prompt for the summarization judge.
        name: Optional human-readable name for this aggregator.

    Example::

        from strands_evals.chaos import ChaosScenarioAggregator, display_chaos_aggregation

        aggregator = ChaosScenarioAggregator(
            known_tools=["search_tool", "database_tool"],
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        )

        reports = experiment.run_evaluations(task=my_task)
        aggregations = aggregator.aggregate(reports)
        display_chaos_aggregation(aggregations, reports=reports)
    """

    def __init__(
        self,
        known_tools: Optional[list[str]] = None,
        known_effects: Optional[list[str]] = None,
        scenarios: Optional[list] = None,
        model: Optional[Model | str] = None,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ChaosScenarioAggregator")
        self.known_tools = known_tools or []
        self.known_effects = known_effects or _ALL_EFFECT_TYPES
        self.model = model
        self.system_prompt = system_prompt or _SUMMARIZE_SYSTEM_PROMPT

        # Build scenario lookup: scenario_name -> {tool_name: [effect_types]}
        self._scenario_effects: dict[str, dict[str, list[str]]] = {}
        if scenarios:
            for scenario in scenarios:
                tool_effects: dict[str, list[str]] = {}
                for tool_name, effects in scenario.effects.items():
                    effect_names = []
                    for e in effects:
                        if hasattr(e, "error_type"):
                            effect_names.append(e.error_type)
                        elif hasattr(e, "max_length"):
                            effect_names.append("truncate_fields")
                        elif hasattr(e, "remove_ratio"):
                            effect_names.append("remove_fields")
                        elif hasattr(e, "corrupt_ratio"):
                            effect_names.append("corrupt_values")
                        else:
                            effect_names.append(type(e).__name__.lower())
                    tool_effects[tool_name] = effect_names
                self._scenario_effects[scenario.name] = tool_effects

    def aggregate(self, reports: list[EvaluationReport]) -> ChaosAggregationReport:
        """Aggregate chaos experiment reports into per-case scenario aggregations.

        Args:
            reports: Flat list of EvaluationReport objects from ChaosExperiment.

        Returns:
            ChaosAggregationReport with .run_display() and .to_file() methods.
        """
        if not reports:
            return ChaosAggregationReport(aggregations=[])

        grouped = self._group_results(reports)

        aggregations = []
        for (case_name, evaluator_name), entries in grouped.items():
            aggregation = self._build_aggregation(case_name, evaluator_name, entries)
            aggregations.append(aggregation)

        aggregations.sort(key=lambda a: (a.group_key, a.evaluator_name))
        return ChaosAggregationReport(aggregations=aggregations)

    def summarize_reasons(self, reasons: list[str]) -> str:
        """Produce a narrative summary from per-scenario reason strings.

        Only produces a summary when self.model is set. Returns empty string otherwise.

        Args:
            reasons: List of reason strings from individual evaluations.

        Returns:
            A summary string, or empty if no model configured.
        """
        non_empty = [r for r in reasons if r]
        if not non_empty or self.model is None:
            return ""

        prompt = (
            "Summarize the following evaluation reasons into a concise 2-3 sentence summary:\n\n"
            + "\n".join(f"- {r}" for r in non_empty[:20])
        )

        try:
            agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = agent(prompt, structured_output_model=ResilienceSummary)
            rating = cast(ResilienceSummary, result.structured_output)
            return rating.summary
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return ""

    def _summarize_for_aggregation(
        self,
        case_name: str,
        evaluator_name: str,
        entries: list[dict],
        stats: dict,
        baseline_passed: Optional[bool],
    ) -> str:
        """Produce a summary for a specific aggregation group using LLM-as-a-Judge.

        Creates an Agent with the configured model and system prompt, then invokes
        it with structured output to get a ResilienceSummary.

        Args:
            case_name: The original case name.
            evaluator_name: The evaluator name.
            entries: The chaos scenario entries (excluding baseline).
            stats: Computed stats dict.
            baseline_passed: Whether baseline passed (None if no baseline).

        Returns:
            Summary string.
        """
        reasons = [e["reason"] for e in entries]

        # Build detailed prompt for LLM
        scenario_lines = []
        for entry in entries:
            status = "PASSED" if entry["passed"] else "FAILED"
            scenario_lines.append(
                f"  - [{status}] {entry['scenario_name']} (score={entry['score']:.2f}): {entry['reason']}"
            )

        prompt = _SUMMARIZE_USER_TEMPLATE.format(
            case_name=case_name,
            evaluator_name=evaluator_name,
            baseline_passed=baseline_passed if baseline_passed is not None else "N/A",
            pass_rate=stats["pass_rate"],
            num_passed=stats["num_passed"],
            num_results=stats["num_results"],
            scenario_details="\n".join(scenario_lines),
        )

        try:
            agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = agent(prompt, structured_output_model=ResilienceSummary)
            rating = cast(ResilienceSummary, result.structured_output)
            return rating.summary
        except Exception as e:
            logger.warning(f"LLM summarization failed for case '{case_name}': {e}")
            return ""

    # ------------------------------------------------------------------
    # Internal grouping and aggregation logic
    # ------------------------------------------------------------------

    def _group_results(
        self, reports: list[EvaluationReport]
    ) -> dict[tuple[str, str], list[dict]]:
        """Group report entries by (original_case_name, evaluator_name).

        Each entry in the returned lists is a dict with:
            - scenario_name: str
            - score: float
            - passed: bool
            - reason: str
            - metadata: dict (from the case)
        """
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for report in reports:
            evaluator_name = report.evaluator_name

            for i, case_data in enumerate(report.cases):
                raw_name = case_data.get("name", "") or ""
                original_name, scenario_name = self._parse_case_name(raw_name)

                metadata = case_data.get("metadata") or {}
                if "chaos_scenario" in metadata:
                    scenario_name = metadata["chaos_scenario"]

                score = report.scores[i] if i < len(report.scores) else 0.0
                passed = report.test_passes[i] if i < len(report.test_passes) else False
                reason = report.reasons[i] if i < len(report.reasons) else ""

                grouped[(original_name, evaluator_name)].append(
                    {
                        "scenario_name": scenario_name,
                        "score": score,
                        "passed": passed,
                        "reason": reason,
                        "metadata": metadata,
                    }
                )

        return grouped

    def _build_aggregation(
        self,
        case_name: str,
        evaluator_name: str,
        entries: list[dict],
    ) -> ChaosScenarioAggregation:
        """Build a ChaosScenarioAggregation from grouped entries."""
        # Separate baseline from chaos scenarios
        baseline_entries = [e for e in entries if e["scenario_name"] == _BASELINE_SCENARIO_NAME]
        chaos_entries = [e for e in entries if e["scenario_name"] != _BASELINE_SCENARIO_NAME]

        # Compute stats over chaos scenarios only (baseline is reference)
        chaos_scores = [e["score"] for e in chaos_entries]
        chaos_passes = [e["passed"] for e in chaos_entries]
        stats = self._compute_stats(chaos_scores, chaos_passes)

        # Baseline comparison
        baseline_score: Optional[float] = None
        baseline_passed: Optional[bool] = None
        degradation: Optional[float] = None

        if baseline_entries:
            baseline_score = baseline_entries[0]["score"]
            baseline_passed = baseline_entries[0]["passed"]
            if chaos_scores:
                degradation = baseline_score - stats["mean_score"]

        # Build per-scenario ToolEffectResults and coverage matrix
        scenario_results = []
        coverage_matrix: dict[str, dict[str, CoverageStatus]] = {}

        for entry in chaos_entries:
            scenario_name = entry["scenario_name"]
            metadata = entry["metadata"]
            tool_effects = self._extract_tool_effects_from_metadata(metadata, scenario_name)

            for tool_name, effect_type in tool_effects:
                result = ToolEffectResult(
                    group_key=f"{case_name}/{tool_name}/{effect_type}",
                    evaluator_name=evaluator_name,
                    mean_score=entry["score"],
                    min_score=entry["score"],
                    max_score=entry["score"],
                    pass_rate=1.0 if entry["passed"] else 0.0,
                    num_results=1,
                    num_passed=1 if entry["passed"] else 0,
                    num_failed=0 if entry["passed"] else 1,
                    tool_name=tool_name,
                    effect_type=effect_type,
                    scenario_label=scenario_name,
                    score=entry["score"],
                    passed=entry["passed"],
                    reason=entry["reason"],
                )
                scenario_results.append(result)

                if tool_name not in coverage_matrix:
                    coverage_matrix[tool_name] = {}
                coverage_matrix[tool_name][effect_type] = (
                    CoverageStatus.PASSED if entry["passed"] else CoverageStatus.FAILED
                )

        self._fill_not_tested(coverage_matrix)

        # Summarize reasons (LLM-as-a-Judge only if model was explicitly provided)
        if self.model is not None:
            summary = self._summarize_for_aggregation(
                case_name, evaluator_name, chaos_entries, stats, baseline_passed
            )
        else:
            summary = ""

        return ChaosScenarioAggregation(
            group_key=case_name,
            evaluator_name=evaluator_name,
            mean_score=stats["mean_score"],
            min_score=stats["min_score"],
            max_score=stats["max_score"],
            pass_rate=stats["pass_rate"],
            num_results=stats["num_results"],
            num_passed=stats["num_passed"],
            num_failed=stats["num_failed"],
            coverage_matrix=coverage_matrix,
            baseline_score=baseline_score,
            baseline_passed=baseline_passed,
            degradation_from_baseline=degradation,
            scenario_results=scenario_results,
            summary=summary,
        )

    def _extract_tool_effects_from_metadata(
        self, metadata: dict, scenario_name: str
    ) -> list[tuple[str, str]]:
        """Extract (tool_name, effect_type) pairs for a scenario.

        Uses the scenario lookup populated from ChaosScenario objects.
        """
        if scenario_name in self._scenario_effects:
            pairs = []
            for tool_name, effect_types in self._scenario_effects[scenario_name].items():
                for effect_type in effect_types:
                    pairs.append((tool_name, effect_type))
            if pairs:
                return pairs

        if scenario_name and scenario_name != _BASELINE_SCENARIO_NAME:
            return [(scenario_name, "unknown")]

        return []

    def _fill_not_tested(self, coverage_matrix: dict[str, dict[str, CoverageStatus]]) -> None:
        """Fill NOT_TESTED entries for known tool×effect combinations not covered."""
        all_tools = set(coverage_matrix.keys()) | set(self.known_tools)

        for tool_name in all_tools:
            if tool_name not in coverage_matrix:
                coverage_matrix[tool_name] = {}
            for effect_type in self.known_effects:
                if effect_type not in coverage_matrix[tool_name]:
                    coverage_matrix[tool_name][effect_type] = CoverageStatus.NOT_TESTED

    @staticmethod
    def _parse_case_name(raw_name: str) -> tuple[str, str]:
        """Parse a tagged case name into (original_name, scenario_name).

        ChaosExperiment tags cases as "original_name|scenario_name".
        """
        if _SCENARIO_SEPARATOR in raw_name:
            parts = raw_name.rsplit(_SCENARIO_SEPARATOR, 1)
            return parts[0].strip(), parts[1].strip()
        return raw_name, "unknown"
