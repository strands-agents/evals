"""Skill Evaluation Experiment.

Composes the base Experiment to run test cases across baseline / variant
conditions paired by trial_idx. Each (case × variant × trial_idx) becomes
one expanded case in the underlying Experiment.

The task function receives the expanded Case. It can read
``case.metadata["variant_label"]`` to know which condition it is running
under, and ``case.metadata["trial_idx"]`` for the pair index. The aggregator
re-groups results back into (case, evaluator) pairs after the fact.

Efficiency metrics (tokens, latency, cost) should be written by the task
function into ``case.metadata["efficiency"]``. The aggregator reads them
from there. Setting ``case.metadata["corrupted"] = True`` marks a trial as
crashed; the aggregator drops corrupted pairs before computing stats.
"""

import logging
import uuid
from collections.abc import Callable
from typing import Any, Optional

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport
from .aggregator import SkillEvalAggregator
from .aggregator_types import SkillEvalAggregationReport

logger = logging.getLogger(__name__)


_DEFAULT_BASELINE_LABEL = "baseline"
_DEFAULT_VARIANT_LABEL = "variant"


class SkillEvalExperiment:
    """Runs cases × variants × trials by composing the base Experiment.

    For each (case, variant_label, trial_idx) combination, creates an
    expanded case tagged with metadata so the aggregator can recover pairs.

    Example::

        from strands_evals.skills import (
            SkillEvalExperiment,
            SkillEvalAggregator,
        )

        experiment = SkillEvalExperiment(
            cases=test_cases,
            variant_labels=["baseline", "variant"],
            evaluators=[my_evaluator],
            num_trials=30,
            aggregator=SkillEvalAggregator(),
        )

        reports = experiment.run_evaluations(task=my_task)
        agg_report = experiment.aggregate_evaluations()
        agg_report.run_display()

    Args:
        cases: Test cases to evaluate. Each case is expanded into
            ``num_trials × len(variant_labels)`` runs.
        variant_labels: Names of the conditions being compared. Defaults to
            ``["baseline", "variant"]``. The aggregator uses these to find
            pairs — the strings here must match ``baseline_label`` and
            ``variant_label`` on the aggregator.
        evaluators: Evaluators to assess results.
        num_trials: Number of trial repetitions per (case × variant).
            Defaults to 30.
        aggregator: Optional SkillEvalAggregator. If provided,
            aggregate_evaluations() can be called after run_evaluations().
    """

    def __init__(
        self,
        cases: list[Case],
        variant_labels: Optional[list[str]] = None,
        evaluators: Optional[list[Evaluator]] = None,
        num_trials: int = 30,
        aggregator: Optional[SkillEvalAggregator] = None,
    ):
        if num_trials < 1:
            raise ValueError(f"num_trials must be >= 1, got {num_trials}")

        self._original_cases = cases
        self._variant_labels = variant_labels or [_DEFAULT_BASELINE_LABEL, _DEFAULT_VARIANT_LABEL]
        if len(self._variant_labels) < 2:
            raise ValueError(
                f"variant_labels must contain at least two labels (baseline + variant), "
                f"got {self._variant_labels!r}"
            )
        self._evaluators = evaluators
        self._num_trials = num_trials
        self._aggregator = aggregator
        self._last_reports: list[EvaluationReport] = []

        # Build the expanded case list.
        self._expanded_cases: list[Case] = []
        for case in cases:
            for variant_label in self._variant_labels:
                for trial_idx in range(num_trials):
                    session_id = str(uuid.uuid4())
                    expanded_metadata: dict[str, Any] = dict(case.metadata or {})
                    expanded_metadata.update({
                        "original_case_name": case.name,
                        "variant_label": variant_label,
                        "trial_idx": trial_idx,
                    })
                    expanded_name = (
                        f"{case.name}|{variant_label}|{trial_idx}"
                        if case.name
                        else f"{variant_label}|{trial_idx}"
                    )
                    expanded_case = case.model_copy(
                        update={
                            "name": expanded_name,
                            "session_id": session_id,
                            "metadata": expanded_metadata,
                        }
                    )
                    self._expanded_cases.append(expanded_case)

        # Internal Experiment with expanded cases.
        self._experiment = Experiment(
            cases=self._expanded_cases,
            evaluators=evaluators,
        )

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def cases(self) -> list[Case]:
        """The original (unexpanded) test cases."""
        return self._original_cases

    @property
    def variant_labels(self) -> list[str]:
        """Variant labels configured for this experiment."""
        return list(self._variant_labels)

    @property
    def num_trials(self) -> int:
        """Trial repetitions per (case × variant)."""
        return self._num_trials

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_evaluations(
        self, task: Callable[[Case], Any], **kwargs
    ) -> list[EvaluationReport]:
        """Run evaluations across all (case × variant × trial) combinations.

        Args:
            task: The task function to evaluate. Takes a Case and returns
                output. Should set ``case.metadata["efficiency"]`` and may
                set ``case.metadata["corrupted"] = True`` on crash.
            **kwargs: Additional kwargs passed to base
                Experiment.run_evaluations.

        Returns:
            List of EvaluationReport objects covering all conditions.
        """
        reports = self._experiment.run_evaluations(task, **kwargs)
        self._last_reports = reports

        n_runs = len(self._original_cases) * len(self._variant_labels) * self._num_trials
        logger.info(
            f"Skill experiment complete: {len(reports)} reports "
            f"({len(self._original_cases)} cases × {len(self._variant_labels)} variants "
            f"× {self._num_trials} trials = {n_runs} runs)"
        )
        return reports

    async def run_evaluations_async(
        self, task: Callable[[Case], Any], max_workers: int = 10, **kwargs
    ) -> list[EvaluationReport]:
        """Run evaluations asynchronously."""
        reports = await self._experiment.run_evaluations_async(
            task, max_workers=max_workers, **kwargs
        )
        self._last_reports = reports
        return reports

    def aggregate_evaluations(self) -> SkillEvalAggregationReport:
        """Aggregate the last run's reports into a SkillEvalAggregationReport.

        Must be called after run_evaluations().

        Returns:
            SkillEvalAggregationReport with .run_display() and .to_file()
            methods.

        Raises:
            RuntimeError: If no aggregator was configured or
                run_evaluations() hasn't been called.
        """
        if self._aggregator is None:
            raise RuntimeError(
                "No aggregator configured. Pass aggregator=SkillEvalAggregator() "
                "to SkillEvalExperiment.__init__()."
            )
        if not self._last_reports:
            raise RuntimeError(
                "No evaluation reports available. Call run_evaluations() first."
            )
        return self._aggregator.aggregate(self._last_reports)
