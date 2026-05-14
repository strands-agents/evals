"""Skill evaluation aggregator for Strands Evals.

Provides paired-comparison aggregation for evaluating agent skills against
a baseline. Designed for the canonical A/B test: run N trials of an agent
with a skill vs. without, pair by trial_idx, report per-task Δ-metrics
(pass rate, tokens, latency, cost) with paired statistical tests
(Wilcoxon / paired-t / McNemar) and bootstrap confidence intervals.

Closes strands-agents/evals#228.

Example::

    from strands_evals.skills import (
        SkillEvalAggregator,
        SkillEvalExperiment,
    )

    experiment = SkillEvalExperiment(
        cases=test_cases,
        variant_labels=["baseline", "variant"],
        evaluators=[my_evaluator],
        num_trials=30,
        aggregator=SkillEvalAggregator(model="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    )

    reports = experiment.run_evaluations(task=my_task)
    agg_report = experiment.aggregate_evaluations()
    agg_report.run_display()
    agg_report.to_file("skill_eval_report.json")
"""

from .aggregation_display import (
    SkillEvalAggregationDisplay,
    display_skill_aggregation,
)
from .aggregator import SkillEvalAggregator
from .aggregator_types import (
    PairedComparisonStats,
    SkillEvalAggregation,
    SkillEvalAggregationReport,
)
from .experiment import SkillEvalExperiment

__all__ = [
    "PairedComparisonStats",
    "SkillEvalAggregation",
    "SkillEvalAggregationDisplay",
    "SkillEvalAggregationReport",
    "SkillEvalAggregator",
    "SkillEvalExperiment",
    "display_skill_aggregation",
]
