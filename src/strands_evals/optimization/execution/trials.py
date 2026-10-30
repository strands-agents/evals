"""Scoring one candidate on cases through `Experiment`, with caching and the re-run passes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...types.optimization import SplitName, TrialRecord
from .rollout import TaskFn
from .traces import TraceCapture


class TrialCache(Protocol):
    """Saved trials; implemented by `record.RunDirectory`.

    No reports: the trial records hold every evaluation, each evaluator's per-row outputs included
    (`TrialRecord.details`), so reports are rebuilt from them (`RunDirectory.reports`).
    """

    def load_trial(self, candidate_id: str, run_index: int, case_name: str) -> TrialRecord | None:
        """Return a saved trial.

        A saved `infrastructure_invalid` trial is returned too; `TrialRunner` treats it as not yet
        run, so it is retried and replaced.

        Args:
            candidate_id: The candidate.
            run_index: Which of `runs_per_candidate` runs.
            case_name: The case name.

        Returns:
            The saved trial, or None if it was never saved.
        """

    def save_trial(self, record: TrialRecord) -> None:
        """Save a trial; a later save for the same (candidate, run, case) replaces it."""


class TrialRunner:
    """Runs a candidate's uncached trials through a fresh `Experiment` per call.

    `rerun_passes` caps the infrastructure re-run passes per call; this runner never retries inside a
    trial. `Experiment` keeps its own throttling retries inside one trial's task span; each retry is an
    attempt (`TrialRecord.earlier`).
    """

    def __init__(
        self,
        cases: Sequence[Case],
        evaluators: Sequence[Evaluator],
        traces: TraceCapture,
        cache: TrialCache,
        *,
        split_of: Mapping[str, SplitName],
        max_workers: int,
        rerun_passes: int,
    ):
        """Build the runner once per run.

        Each call builds a fresh `Experiment(uncached cases, evaluators)` with `diagnosis_config` off, after
        `TraceCapture` is installed, and runs it in an empty OTel context so an outer span can't merge
        trials into one trace. The user's `Experiment` instance is never run (its tracer is a no-op).

        With no evaluators, `Experiment` substitutes a placeholder `Evaluator()` that always raises. The
        placeholder (`type(e) is Evaluator`) is dropped from `evaluations`, `evaluator_errors`, and the
        every-evaluator-failed check, so a trial with zero evaluators (a usage-only objective) is
        `completed`, not `verifier_invalid`.

        Each evaluator is passed to `Experiment` wrapped in a thin private proxy that forwards everything
        `Experiment` reads (`get_name`, `get_type_name`, `evaluation_level`, the label inputs) and, when
        the evaluator raises, runs `classify` on the real exception, records the result under the
        current trace id, and re-raises. So a judge failing on infrastructure (throttling, transport,
        model-service errors) is told apart by the same rule as an agent's, never by the exception's
        type name or message (a throttled botocore call is a `ClientError` either way). The last
        attempt's classification counts: `Experiment` retries throttled judges itself.

        Args:
            cases: Every case of the run.
            evaluators: The experiment's evaluators.
            traces: The run's trace capture, installed before any `Experiment` is built.
            cache: Saved trials (the run directory).
            split_of: Case name -> its split, given once so a record's split can't be wrong.
            max_workers: Parallel trials within one call.
            rerun_passes: Max infrastructure re-run passes per call.
        """
        raise NotImplementedError

    def pending(self, candidate_id: str, case_names: Sequence[str], *, runs: int) -> int:
        """Number of (case, run) trials still to run: the real cost of the first passes.

        Counts trials with no saved record and saved `infrastructure_invalid` ones, so a resumed run
        with a larger budget retries what an exhausted allowance left unmeasured. Every affordability
        check uses it, so cached trials cost 0.

        Args:
            candidate_id: The candidate.
            case_names: The case names to score.
            runs: Runs per case.
        """
        raise NotImplementedError

    async def run(
        self,
        candidate_id: str,
        task: TaskFn,
        case_names: Sequence[str],
        *,
        runs: int,
        allowance: int,
    ) -> tuple[list[TrialRecord], int]:
        """Score every run, then re-run infrastructure failures; returns (records, rollouts spent).

        Returns one record per requested (case, run), cached ones included, so `score()` sees the whole
        candidate. Saved `infrastructure_invalid` trials are not cache hits: they run again in the first
        pass and count as rollouts. A saved `verifier_invalid` trial is re-judged (0 rollouts) when its
        candidate is scored again.

        `task` is not yet instrumented (`make_task(...)` or the `build_task` path's task); each run wraps
        it with `instrument(task, TrialLog(), traces)`. After every run's first pass, up to `rerun_passes`
        re-run passes each retry the trials still `infrastructure_invalid`. `allowance` is
        `BudgetTracker.allowance(phase=...)` and caps only the re-run passes; trials left when it runs out
        stay `infrastructure_invalid`. A judge-only infrastructure failure re-runs just the judges on the
        saved output, for 0 rollouts. A record a re-run replaces moves into the new record's `earlier`.
        Each record's and attempt's `rollouts` is set to what it cost (1 when the agent ran, else 0).
        Each trial is saved as soon as it finishes.

        Args:
            candidate_id: The candidate.
            task: The candidate's task, not yet instrumented.
            case_names: The case names to score.
            runs: Runs per case (`runs_per_candidate`).
            allowance: Rollouts the re-run passes may spend.

        Returns:
            One record per requested (case, run), cached ones included, and the rollouts spent.

        Raises:
            TypeError: The bound factory raised `_FactoryArgumentError` (a resource or kwarg named like a
                declared marker, or a kwarg the factory doesn't accept).
        """
        raise NotImplementedError
