"""Private repeated-evaluation collection helpers."""

import copy
from collections.abc import Callable
from typing import Any

from ...experiment import Experiment
from ...types.evaluation import EvaluationData
from ...types.evaluation_report import EvaluationReport
from ._calibration import _extract_report_scores


class _InMemoryEvaluationDataStore:
    """Task-output cache used to replay evaluators on fixed outputs."""

    def __init__(self) -> None:
        self._results: dict[str, EvaluationData] = {}

    def load(self, case_name: str) -> EvaluationData | None:
        """Return an isolated copy of a cached task result."""
        result = self._results.get(case_name)
        return result.model_copy(deep=True) if result is not None else None

    def save(self, case_name: str, result: EvaluationData) -> None:
        """Cache an isolated copy of a task result."""
        self._results[case_name] = result.model_copy(deep=True)


async def _collect_repeated_reports(
    experiment: Experiment[Any, Any],
    task: Callable[..., Any],
    *,
    agent_runs: int,
    judge_replays: int,
    max_workers: int,
    calibration_context: dict[str, Any],
) -> list[list[EvaluationReport]]:
    """Run agent attempts and replay evaluators against each fixed output."""
    reports_by_run: list[list[EvaluationReport]] = []
    for _ in range(agent_runs):
        task_store = _InMemoryEvaluationDataStore()
        replay_reports: list[EvaluationReport] = []
        for _ in range(judge_replays):
            report = await experiment.run_evaluations_async(
                task,
                max_workers=max_workers,
                evaluation_data_store=task_store,
            )
            report.calibration_context = copy.deepcopy(calibration_context)
            _extract_report_scores(report)
            replay_reports.append(report)
        reports_by_run.append(replay_reports)
    return reports_by_run
