"""Private repeated-evaluation collection helpers."""

from collections.abc import Callable
from typing import Any

from ...experiment import Experiment
from ...types.evaluation import EvaluationData
from ...types.evaluation_report import EvaluationReport
from ._calibration import _extract_case_signatures, _extract_report_scores


class CalibrationRunError(ValueError):
    """Error from repeated collection or fitting with raw reports attached."""

    def __init__(
        self,
        message: str,
        reports_by_run: list[list[EvaluationReport]],
    ) -> None:
        """Initialize the error with every report collected before failure."""
        super().__init__(message)
        self.reports_by_run = reports_by_run


class _InMemoryEvaluationDataStore:
    """Task-output cache used to replay evaluators on fixed outputs."""

    def __init__(self) -> None:
        self._results: dict[str, EvaluationData] = {}

    def load(self, case_name: str) -> EvaluationData | None:
        """Return an isolated copy of a cached task result."""
        result = self._results.get(case_name)
        return _copy_result(case_name, result) if result is not None else None

    def save(self, case_name: str, result: EvaluationData) -> None:
        """Cache an isolated copy of a task result."""
        self._results[case_name] = _copy_result(case_name, result)


def _copy_result(case_name: str, result: EvaluationData) -> EvaluationData:
    """Copy a task result while naming unsupported replay values clearly."""
    try:
        return result.model_copy(deep=True)
    except Exception as error:
        raise TypeError(f"calibration requires deep-copyable task output for case {case_name!r}: {error}") from error


def _partial_reports(
    reports_by_run: list[list[EvaluationReport]],
    replay_reports: list[EvaluationReport],
) -> list[list[EvaluationReport]]:
    """Return completed runs plus any reports from the active run."""
    partial = [*reports_by_run]
    if replay_reports:
        partial.append(replay_reports)
    return partial


async def _collect_repeated_reports(
    experiment: Experiment[Any, Any],
    task: Callable[..., Any],
    *,
    agent_runs: int,
    judge_replays: int,
    max_workers: int,
) -> list[list[EvaluationReport]]:
    """Run agent attempts and replay evaluators against each fixed output."""
    reports_by_run: list[list[EvaluationReport]] = []
    for _ in range(agent_runs):
        task_store = _InMemoryEvaluationDataStore()
        replay_reports: list[EvaluationReport] = []
        for _ in range(judge_replays):
            try:
                report = await experiment.run_evaluations_async(
                    task,
                    max_workers=max_workers,
                    evaluation_data_store=task_store,
                )
            except Exception as error:
                raise CalibrationRunError(
                    f"calibration collection failed: {error}",
                    _partial_reports(reports_by_run, replay_reports),
                ) from error

            replay_reports.append(report)
            try:
                _extract_report_scores(report)
                _extract_case_signatures(report)
            except ValueError as error:
                raise CalibrationRunError(
                    str(error),
                    _partial_reports(reports_by_run, replay_reports),
                ) from error
        reports_by_run.append(replay_reports)
    return reports_by_run
