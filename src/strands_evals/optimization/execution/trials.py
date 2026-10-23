"""Scoring one candidate on cases through `Experiment`, with caching and the re-run pass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...types.evaluation_report import EvaluationReport
from ...types.optimization import SplitName, TrialRecord, TrialStatus
from .rollout import TaskFn
from .traces import TraceCapture


@dataclass
class TrialLogEntry:
    """What the task wrapper recorded for one trial."""

    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    trace_id: int
    attempt_span_id: int


class TrialLog:
    """One run's trial status, written by `instrument`, keyed by case name.

    `setup_error` holds a resource named like a declared marker; `TrialRunner.run` raises it.
    """

    setup_error: TypeError | None

    def record(self, case: str, entry: TrialLogEntry) -> None:
        raise NotImplementedError

    def records(self) -> dict[str, TrialLogEntry]:
        raise NotImplementedError


class TrialCache(Protocol):
    """Saved trials and reports; implemented by `record.RunDirectory`."""

    def load(self, candidate_id: str, run_index: int, case: str) -> TrialRecord | None: ...

    def save(self, record: TrialRecord) -> None: ...

    def save_report(
        self, candidate_id: str, split: str, run_index: int, pass_index: int, report: EvaluationReport
    ) -> None:
        """`pass_index` 0 = first pass, k = k-th re-run."""
        ...


class TrialRunner:
    """Runs a candidate's uncached trials through a fresh `Experiment` per call."""

    def __init__(
        self,
        cases: Sequence[Case],
        evaluators: Sequence[Evaluator],
        traces: TraceCapture,
        cache: TrialCache,
        *,
        split_of: Mapping[str, SplitName],
        max_workers: int,
        retries: int,
    ):
        raise NotImplementedError

    def pending(self, candidate_id: str, cases: Sequence[str], *, runs: int) -> int:
        """Number of (case, run) trials with no saved record: the real cost of the first passes."""
        raise NotImplementedError

    async def run(
        self,
        candidate_id: str,
        task: TaskFn,
        cases: Sequence[str],
        *,
        runs: int,
        allowance: int,
    ) -> tuple[list[TrialRecord], int]:
        """Score every run, then one re-run pass over infrastructure failures; returns (records, rollouts spent).

        Returns one record per requested (case, run), cached ones included, so `score()` sees the whole
        candidate.

        `task` is not yet instrumented (`make_task(...)` or the `build_task` path's task); each run wraps
        it with `instrument(task, TrialLog(), traces)`. `allowance` caps only the re-run pass. A judge-only
        infrastructure failure re-runs just the judges on the saved output, for 0 rollouts. Each trial is
        saved as soon as it finishes.
        """
        raise NotImplementedError
