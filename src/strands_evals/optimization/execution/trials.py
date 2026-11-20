"""Scoring one candidate on cases through `Experiment`, with caching and the re-run passes."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import opentelemetry.context as otel_context
import opentelemetry.trace as trace_api
from pydantic_core import to_json

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...types.evaluation import EvaluationData, EvaluationOutput
from ...types.optimization import AttemptRecord, SplitName, TrialRecord, TrialStatus, Usage, near
from ...types.trace import Session
from .rollout import _THREAD_POOL, TaskFn, TrialLog, TrialLogEntry, instrument
from .status import classify
from .traces import EvaluatorView, TraceCapture

logger = logging.getLogger(__name__)

_Key = tuple[str, int]  # (case name, run index)

# The wait before re-running a trial that failed on infrastructure, in seconds, counted from when it failed: drawn
# from [2.5, 5] for the first re-run pass, and each later pass's range doubles, up to [30, 60].
_RERUN_WAIT = 5.0
_MAX_RERUN_WAIT = 60.0
_JITTER = random.SystemRandom()  # OS entropy: neither `random.seed()` nor a fork puts separate runs in step
_sleep = asyncio.sleep  # tests replace it and `_clock` to record the waits
_clock = time.monotonic


def _rerun_wait(pass_index: int) -> float:
    """Seconds to wait before re-run pass `pass_index` (1-based): uniform in [cap / 2, cap], where cap is
    `_RERUN_WAIT * 2 ** (pass_index - 1)`, at most `_MAX_RERUN_WAIT`."""
    cap = min(_MAX_RERUN_WAIT, _RERUN_WAIT * 2.0 ** min(pass_index - 1, 32))  # a large exponent would overflow
    return _JITTER.uniform(cap / 2, cap)


@dataclass
class _TrialState:
    """What one trial's evaluator proxies saw: its trace, every trace they registered, which evaluators raised."""

    trace_id: str | None = None
    trace_ids: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)  # evaluator name -> exception type
    infrastructure: set[str] = field(default_factory=set)  # evaluators whose error is `infrastructure_invalid`
    released: bool = False  # the trial's traces were released: a judge still running must not register again


class _EvaluatorProxy:
    """An evaluator as `Experiment` sees it: every attribute is the evaluator's own but two methods.

    `evaluate_async` registers the current trace, so a judge-only re-run (no attempt span) keeps its
    judge spans, and records the type of any exception before re-raising it. A non-finite score (NaN or
    infinity), in a row or from the aggregator, raises `ValueError`, so it is a failed evaluator like any
    other (re-judged by the re-run passes) and the record never holds a score the save can't write. So does a
    row's score outside the evaluator's range (its `score_range`, else (0, 1); N/A rows aside), which scoring
    would otherwise take out with no repair. Not a
    subclass: `get_type_name` is a classmethod and would name the proxy.

    It also records whether `classify` classifies the exception as `infrastructure_invalid`, so the re-run
    pass waits before re-judging that evaluator.
    """

    def __init__(self, evaluator: Evaluator, traces: TraceCapture, state: _TrialState) -> None:
        self._evaluator = evaluator
        self._traces = traces
        self._state = state

    def __getattr__(self, name: str) -> Any:
        evaluator = self.__dict__.get("_evaluator")  # not self._evaluator: copy and pickle call this before __init__
        if evaluator is None:
            raise AttributeError(name)
        return getattr(evaluator, name)

    async def evaluate_async(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        context = trace_api.get_current_span().get_span_context()
        if context.is_valid and not self._state.released:
            self._state.trace_id = format(context.trace_id, "032x")
            self._state.trace_ids.add(self._state.trace_id)
            self._traces.register(self._state.trace_id)
        name = self._evaluator.get_name()
        try:
            outputs = await self._evaluator.evaluate_async(evaluation_case)
            # The range scoring uses: `score_range`, else (0, 1); a None end is unbounded.
            low, high = getattr(self._evaluator, "score_range", None) or (0.0, 1.0)
            for output in outputs:
                score = output.score
                if not math.isfinite(score):
                    raise ValueError(f"evaluator {name} returned a score that isn't finite: {score!r}")
                if output.not_applicable:
                    continue  # nothing to judge, so the score is not a verdict and no range applies
                # Float noise at an end is in range, as scoring's own range check treats it.
                below = low is not None and score < low and not near(score, low)
                above = high is not None and score > high and not near(score, high)
                if below or above:
                    logger.warning(
                        "evaluator=<%s>, score=<%r>, range=<[%s, %s]> | score outside the evaluator's range, judged "
                        "again by the re-run passes; give the evaluator a score_range = (low, high) (None for "
                        "unbounded)",
                        name,
                        score,
                        low,
                        high,
                    )
                    raise ValueError(f"evaluator {name} returned {score!r}, outside its range [{low}, {high}]")
        except Exception as error:
            self._state.errors[name] = type(error).__name__
            if classify(error) is TrialStatus.INFRASTRUCTURE_INVALID:
                self._state.infrastructure.add(name)
            else:
                self._state.infrastructure.discard(name)
            raise
        self._state.errors.pop(name, None)  # Experiment retried a throttled call and it succeeded
        self._state.infrastructure.discard(name)
        return outputs

    def aggregator(self, outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        """The evaluator's own aggregator; a non-finite result raises `ValueError` (the span records it)."""
        score, passed, reason = self._evaluator.aggregator(outputs)
        if not math.isfinite(score):
            raise ValueError(f"evaluator {self._evaluator.get_name()} aggregated a score that isn't finite: {score!r}")
        return score, passed, reason


class _TrialStore:
    """One trial's in-memory `EvaluationDataStore`: captures the task's live output, replays it to re-judge."""

    def __init__(self, data: EvaluationData | None = None) -> None:
        self.data = data

    def load(self, case_name: str) -> EvaluationData | None:
        return self.data

    def save(self, case_name: str, result: EvaluationData) -> None:
        self.data = result


class TrialCache(Protocol):
    """Saved trials; implemented by `record.RunDirectory`.

    No reports: the trial records hold every evaluation, each evaluator's per-row outputs included
    (`TrialRecord.details`), so reports are rebuilt from them (`OptimizationResult.reports`).
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

    def trial_status(self, candidate_id: str, run_index: int, case_name: str) -> TrialStatus | None:
        """The saved trial's status without loading the record (`pending` calls it per case and run).

        Returns:
            The status of `load_trial`'s record, or None if it was never saved.
        """

    def save_trial(self, record: TrialRecord) -> None:
        """Save a trial; a later save for the same (candidate, run, case) replaces it."""


class TrialRunner:
    """Runs a candidate's uncached trials, each through its own one-case `Experiment`.

    `rerun_passes` caps the re-run passes per call (infrastructure re-runs and failed-evaluator re-judges);
    this runner never retries inside a trial. `Experiment` keeps its own throttling retries inside one trial's
    task span; each retry is an attempt (`TrialRecord.previous_attempts`).

    A transient model error is best retried inside the call by the model client, which the agent factory
    configures (`Optimizer`'s `rerun_passes` says how for Bedrock). The re-run passes are for what still
    fails, and a trial whose agent or evaluator failed on infrastructure is re-run only after a wait from
    when it failed (`run`).
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

        Each trial builds a fresh `Experiment([case], evaluators)` with `diagnosis_config` off, after
        `TraceCapture` is installed, and runs it in an empty OTel context so an outer span can't merge
        trials into one trace. The user's `Experiment` instance is never run (its tracer may predate the
        capture and be a no-op).

        With no evaluators, `Experiment` substitutes a placeholder `Evaluator()` that always raises. The
        caller passes the evaluators without that placeholder (`Optimizer` drops it), and only
        `evaluators` are read, so a trial with zero evaluators (a usage-only objective) is `completed`,
        not `evaluator_invalid`.

        Each evaluator is passed to `Experiment` wrapped in a thin private proxy that delegates every
        attribute to the evaluator (so `get_name`, `get_type_name`, and the label mapping are its own)
        and wraps `evaluate_async` and `aggregator`. The proxy registers the current trace with
        `TraceCapture`, since a judge-only re-run has no attempt span to do it and its judge spans would
        be dropped. When the evaluator raises, it records the exception's type name and re-raises; a
        non-finite score, or a row's score outside the evaluator's range, raises `ValueError` the same
        way. A failed judge, on infrastructure or not, never changes the trial's status: the trial stays
        `completed` with that evaluator in `evaluator_errors` (every evaluator failed is
        `evaluator_invalid`), and only the agent side makes a trial `infrastructure_invalid`.

        When `Experiment` fails on the case after the task returned (it can't serialize the output for
        its span, say), it writes a case-level placeholder row (score 0) for every evaluator, and none is
        taken as a score. With at least one evaluator the trial is `evaluator_invalid`, else `completed`;
        its `error_type` and `error` are the exception `Experiment` caught, and no evaluator is in
        `evaluator_errors`, so it is never re-judged: judging again is expected to fail the same way, or,
        when `Experiment` failed before saving the output, would judge an output of None.

        Args:
            cases: Every case of the run.
            evaluators: The experiment's evaluators.
            traces: The run's trace capture, installed before any `Experiment` is built.
            cache: Saved trials (the run directory).
            split_of: Case name -> its split, given once so a record's split can't be wrong.
            max_workers: Parallel trials within one call.
            rerun_passes: Max re-run passes per call (infrastructure re-runs and failed-evaluator re-judges).

        Raises:
            ValueError: A case without a name, two cases with one name, a case missing from `split_of`,
                two evaluators with one `get_name()`, `max_workers` < 1, or `rerun_passes` < 0.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if rerun_passes < 0:
            raise ValueError(f"rerun_passes must be at least 0, got {rerun_passes}")
        self._cases: dict[str, Case] = {}
        for case in cases:
            if case.name is None:
                raise ValueError("every case needs a name, so its trials can be cached and paired")
            if case.name in self._cases:
                raise ValueError(f"two cases are named {case.name!r}; case names must be unique")
            if case.name not in split_of:
                raise ValueError(f"case {case.name!r} has no split in split_of")
            self._cases[case.name] = case
        # results, errors, and judge spend are keyed by get_name(), so two evaluators sharing one would collide
        evaluator_names = Counter(e.get_name() for e in evaluators)
        duplicates = sorted(name for name, count in evaluator_names.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"evaluator names must be unique, got {duplicates} more than once; "
                "pass name=... to tell instances of one evaluator class apart"
            )
        self._evaluators = list(evaluators)
        self._traces = traces
        self._cache = cache
        self._split_of = dict(split_of)
        self._max_workers = max_workers
        self._rerun_passes = rerun_passes

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
        return sum(
            1
            for name in case_names
            for run_index in range(runs)
            if self._cache.trial_status(candidate_id, run_index, name) in (None, TrialStatus.INFRASTRUCTURE_INVALID)
        )

    async def run(
        self,
        candidate_id: str,
        task: TaskFn,
        case_names: Sequence[str],
        *,
        runs: int,
        allowance: int,
        rejudge: bool = False,
        limiter: asyncio.Semaphore | None = None,
    ) -> tuple[list[TrialRecord], int]:
        """Score every run, then re-run failures; returns (records, rollouts spent).

        Returns one record per requested (case, run), cached ones included, so `score()` sees the whole
        candidate. Saved `infrastructure_invalid` trials are not cache hits: they run again in the first
        pass and count as rollouts.

        Each trial is its own `Experiment` call over one case, so it is saved as soon as that call
        returns. Every trial of the call, all runs included, runs at once under one `max_workers` cap,
        case-major (each case's runs, then the next case's); sync tasks run in the call's own pool of
        `max_workers` threads, so the cap holds for them too. With `limiter`, that semaphore caps the
        trials instead, so candidates scored at once share one cap rather than one each. Each trial gets a
        private in-memory store: it captures the task's live output (`Session` included), which the call
        keeps and replays on a judge-only re-run, which runs only the evaluators that failed.

        `task` is not yet instrumented (`make_task(...)` or the `build_task` path's task); each trial (one
        case, one run) wraps it with `instrument(task, TrialLog(), traces)`. After every run's first pass, up
        to `rerun_passes` re-run passes each retry the trials still `infrastructure_invalid` or with a failed evaluator.
        A trial whose agent failed on infrastructure, or whose evaluator failed with an error `classify` classifies
        as `infrastructure_invalid`, is re-run only once a jittered 2.5-5 s has passed since it failed; each later
        pass's range doubles, up to 30-60 s. The wait holds no slot of the cap, and the pass's other trials (a
        re-judge of a bad verdict, say) start at once.
        `allowance` caps the rollouts of the whole call, first pass included: the caller checks
        `can_afford(pending(...))` first, so the first pass always fits, and the re-runs get what is left of
        it (`BudgetTracker.allowance(phase=...)` read before the call, so `max_rollouts` and the later
        phases' reserves hold). Trials left when it runs out stay `infrastructure_invalid`. A judge-only
        re-run costs 0 rollouts. A record a
        re-run replaces moves into the new record's `previous_attempts`. Each record's and attempt's
        `rollouts` is set to what it cost (1 when a pass ran the agent; 0 for a judge-only re-run and for an
        attempt `Experiment` retried on throttling).

        Each cost stays with what it bought. After a judge-only re-run, the new record keeps the reused
        output's usage and the judge spend of every score it holds; the replaced attempt keeps its
        rollout, no usage, and only its failed judges' spend. So the agent's tokens are counted once and
        the retry cost holds only failed work.

        With `rejudge`, a saved trial with failed evaluators (`evaluator_invalid`, or `completed` with
        some) has just those re-judged on its saved output, for 0 rollouts. Without it, cached trials are
        final, so a resumed run calls no judge the uninterrupted run didn't.

        Every trace a trial registered (`TrialLog.trace_ids`, and the evaluator proxies') is released in a
        `finally` once its record is built, so a setup error, a cancelled attempt, or an exception here
        leaks no spans; never earlier, since usage and judge spans are read after `Experiment` returns.

        Args:
            candidate_id: The candidate.
            task: The candidate's task, not yet instrumented.
            case_names: The case names to score.
            runs: Runs per case (`runs_per_candidate`).
            allowance: Rollouts the whole call may spend, first pass included.
            rejudge: Re-judge saved trials' failed evaluators; set only when an ineligible candidate is
                scored again.
            limiter: A semaphore the trials of several concurrent calls share, so `max_workers` caps them
                together; None gives this call its own cap.

        Returns:
            One record per requested (case, run), cached ones included, and the rollouts spent.

        Raises:
            TypeError: `_FactoryArgumentError` (a resource or kwarg named like a declared marker, a kwarg
                the factory doesn't accept, or an environment that gave no mapping).
            RuntimeError: Trial spans aren't recorded (an OTel SDK disabled or sampling out).
            ValueError: A case name the runner wasn't built with, a case name listed twice, or `runs` < 1.
        """
        if runs < 1:
            raise ValueError(f"runs must be at least 1, got {runs}")
        unknown = [name for name in case_names if name not in self._cases]
        if unknown:
            raise ValueError(f"unknown case names: {unknown}")
        duplicates = sorted(name for name, count in Counter(case_names).items() if count > 1)
        if duplicates:
            raise ValueError(f"case names listed more than once: {duplicates}")
        keys = [(name, run_index) for name in case_names for run_index in range(runs)]  # case-major
        call = _Call(
            candidate_id=candidate_id,
            task=task,
            semaphore=limiter if limiter is not None else asyncio.Semaphore(self._max_workers),
        )
        jobs: list[tuple[_Key, bool]] = []  # (key, judge only), case-major
        for key in keys:
            saved = self._cache.load_trial(candidate_id, key[1], key[0])
            if saved is not None:
                call.records[key] = saved
            if _needs_agent(saved):
                jobs.append((key, False))
            elif rejudge and _has_failed_evaluators(saved):
                jobs.append((key, True))
        touched = {key for key, _ in jobs}  # the re-run passes never touch a trial this call didn't run
        pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="trial")
        token = _THREAD_POOL.set(pool)
        try:
            spent = await self._pass(call, jobs, pass_index=0)
            for pass_index in range(1, self._rerun_passes + 1):
                agent_runs = max(allowance - spent, 0)  # the allowance covers the whole call, first pass included
                jobs = []
                for key in keys:
                    if key not in touched:
                        continue
                    record = call.records.get(key)
                    if _needs_agent(record) and agent_runs > 0:
                        jobs.append((key, False))
                        agent_runs -= 1
                    elif _has_failed_evaluators(record):
                        jobs.append((key, True))
                if not jobs:
                    break
                failed = [call.infrastructure[key] for key, _ in jobs if key in call.infrastructure]
                wait = _rerun_wait(pass_index) if failed else 0.0
                if failed:
                    logger.warning(
                        "candidate_id=<%s>, pass=<%d>, trials=<%d>, error_type=<%s>, wait_seconds=<%.1f> | "
                        "waiting before re-running trials that failed on infrastructure",
                        candidate_id,
                        pass_index,
                        len(failed),
                        failed[0][0],
                        wait,
                    )
                spent += await self._pass(call, jobs, pass_index=pass_index, wait=wait)
        except Exception:
            pool.shutdown(wait=False)  # `_pass` raises only after every trial it started has returned
            raise
        else:
            pool.shutdown()
        finally:
            # not shut down on a cancel: a trial left running may still need the pool to exit its environment
            _THREAD_POOL.reset(token)
        return [call.records[key] for key in keys], spent

    async def _pass(self, call: _Call, jobs: list[tuple[_Key, bool]], *, pass_index: int, wait: float = 0.0) -> int:
        """Run one pass's trials at once under the call's cap; returns the rollouts it spent.

        A trial that last failed on infrastructure (`call.infrastructure`) starts only once `wait` seconds
        have passed since it failed, waiting outside the cap.

        On a setup error or a failed save, trials not yet started are skipped, the running ones finish
        and are saved, then the first error is raised.
        """
        failures: list[BaseException] = []
        ran = 0

        async def one(key: _Key, judge_only: bool) -> None:
            nonlocal ran
            if key in call.infrastructure:
                remaining = wait - (_clock() - call.infrastructure[key][1])
                if remaining > 0:
                    await _sleep(remaining)
            async with call.semaphore:
                if failures:
                    return
                try:
                    record = await self._trial(call, key, judge_only=judge_only, pass_index=pass_index)
                    self._cache.save_trial(record)
                except Exception as error:
                    failures.append(error)
                    return
                call.records[key] = record
                call.passes[key] = pass_index
                ran += 0 if judge_only else 1

        await asyncio.gather(*(one(key, judge_only) for key, judge_only in jobs))
        if failures:
            raise failures[0]
        return ran

    async def _trial(self, call: _Call, key: _Key, *, judge_only: bool, pass_index: int) -> TrialRecord:
        """One `Experiment` call over one case: the whole trial, or, when `judge_only`, its failed evaluators."""
        case_name, run_index = key
        case = self._cases[case_name]
        previous = call.records.get(key)
        if judge_only:
            assert previous is not None
            evaluators = [e for e in self._evaluators if e.get_name() in previous.evaluator_errors]
            data = call.live.get(key) or _evaluation_data(case, previous)
        else:
            evaluators, data = self._evaluators, None
        state = _TrialState()
        store = _TrialStore(data)
        log = TrialLog()
        experiment: Experiment = Experiment(
            cases=[case],
            evaluators=cast(list[Evaluator], [_EvaluatorProxy(e, self._traces, state) for e in evaluators]),
        )
        try:
            token = otel_context.attach(otel_context.Context())  # an outer span can't merge trials into one trace
            try:
                report = await experiment.run_evaluations_async(
                    instrument(call.task, log, self._traces), max_workers=1, evaluation_data_store=store
                )
            finally:
                otel_context.detach(token)
            if log.setup_error is not None:
                raise log.setup_error
            entry = log.records().get(case_name)
            trace_id = entry.trace_id if entry is not None else state.trace_id
            if trace_id is None:
                raise RuntimeError(
                    f"case {case_name!r} run {run_index}: Experiment ran neither the task nor an evaluator"
                )
            view = self._traces.evaluator_view(trace_id)
            if judge_only:
                assert previous is not None
                record = self._rejudged(
                    call, previous, trace_id, report, evaluators, state, view, pass_index=call.passes.get(key, 0)
                )
            else:
                assert entry is not None
                record = self._ran(call, key, entry, store.data, report, state, view, pass_index=pass_index)
        finally:
            # every trace this trial registered, read or not: a setup error, a cancelled attempt, or an
            # exception above would otherwise leave its spans in memory for the rest of the run
            state.released = True
            self._traces.release(log.trace_ids | state.trace_ids)
        if store.data is not None and record.status is not TrialStatus.INFRASTRUCTURE_INVALID:
            call.live[key] = store.data
        judges = [name for name in record.evaluator_errors if name in state.infrastructure]
        if record.status is TrialStatus.INFRASTRUCTURE_INVALID:
            call.infrastructure[key] = (record.error_type or "", _clock())
        elif judges:
            call.infrastructure[key] = (record.evaluator_errors[judges[0]], _clock())
        else:
            call.infrastructure.pop(key, None)
        return record

    def _ran(
        self,
        call: _Call,
        key: _Key,
        entry: TrialLogEntry,
        data: EvaluationData | None,
        report: Any,
        state: _TrialState,
        view: EvaluatorView,
        *,
        pass_index: int,
    ) -> TrialRecord:
        """The record of a trial whose agent ran in this attempt."""
        case_name, run_index = key
        previous = call.records.get(key)
        earlier = (
            [*previous.previous_attempts, _replaced(previous, pass_index=call.passes.get(key, 0))] if previous else []
        )
        earlier += [
            AttemptRecord(
                status=status,
                error_type=error_type,
                error=error,
                usage=self._traces.usage(entry.trace_id, span_id),
                judge_usage=Usage.zero(),
                pass_index=pass_index,
                rollouts=0,  # Experiment's throttling retries aren't counted against the budget
                benchmark_usage=self._traces.benchmark_usage(entry.trace_id, span_id),
            )
            for span_id, status, error_type, error in entry.previous_attempts
        ]
        completed = entry.status is TrialStatus.COMPLETED
        evaluations, details, errors, judge_usage = (
            self._judged({}, {}, {}, self._evaluators, report, state, view) if completed else ({}, {}, {}, {})
        )
        error_type, error = entry.error_type, entry.error
        if completed:
            case_error_type, case_error = self._traces.case_error(entry.trace_id)
            if case_error_type is not None:
                # `Experiment` failed on the case after the task returned, so every row is a case-level
                # placeholder. No evaluator is left in `errors`: a re-judge would replay the same output
                # through the same failure and only spend judge calls.
                error_type, error = case_error_type, case_error
                evaluations, details, errors = {}, {}, {}
                logger.warning(
                    "candidate_id=<%s>, case_name=<%s>, run_index=<%s>, error_type=<%s>, error=<%s> | case failed "
                    "after the task returned | no evaluator score recorded",
                    call.candidate_id,
                    case_name,
                    run_index,
                    error_type,
                    error,
                )
        status = entry.status
        if completed and self._evaluators and not evaluations:
            status = TrialStatus.EVALUATOR_INVALID
        kept = data if completed else None  # an attempt that raised keeps no output
        environment_state = (
            [
                s.model_copy(update={"state": _storable(s.state, call, "environment_state")})
                for s in kept.actual_environment_state
            ]
            if kept is not None and kept.actual_environment_state is not None
            else None
        )
        return TrialRecord(
            candidate_id=call.candidate_id,
            case_name=case_name,
            run_index=run_index,
            split=self._split_of[case_name],
            status=status,
            error_type=error_type,
            error=error,
            attempts=len(earlier) + 1,
            previous_attempts=earlier,
            output=_storable(kept.actual_output, call, "output") if kept is not None else None,
            trajectory=_storable(kept.actual_trajectory, call, "trajectory") if kept is not None else None,
            interactions=_storable(kept.actual_interactions, call, "interactions") if kept is not None else None,
            environment_state=environment_state,
            evaluations=evaluations,
            details=details,
            evaluator_errors=errors,
            usage=self._traces.usage(entry.trace_id, entry.attempt_span_id),
            judge_usage=judge_usage,
            rollouts=1,
            benchmark_usage=self._traces.benchmark_usage(entry.trace_id, entry.attempt_span_id),
        )

    def _rejudged(
        self,
        call: _Call,
        previous: TrialRecord,
        trace_id: str,
        report: Any,
        evaluators: Sequence[Evaluator],
        state: _TrialState,
        view: EvaluatorView,
        *,
        pass_index: int,
    ) -> TrialRecord:
        """The record after a judge-only re-run: the agent's fields kept, the failed evaluators judged again.

        `pass_index` is the pass `previous` ran in (0 for a record from an earlier call).

        Each cost stays with what it bought: the new record keeps the output's usage (its benchmark
        usage too) and each kept score's judge spend; the replaced attempt keeps its rollout, no usage,
        and only its failed judges' spend.

        When `Experiment` failed on the case after the evaluators ran, its rows are case-level placeholders,
        as in `_ran`: none is taken as a score, no re-judged evaluator is left in `evaluator_errors` (judging
        again is expected to fail the same way), and `error_type` and `error` name the exception. The first
        pass's scores are kept.
        """
        kept = {name: previous.judge_usage[name] for name in previous.evaluations if name in previous.judge_usage}
        evaluations, details, errors, judge_usage = self._judged(
            dict(previous.evaluations), dict(previous.details), {}, evaluators, report, state, view
        )
        error_type, error = previous.error_type, previous.error
        case_error_type, case_error = self._traces.case_error(trace_id)
        if case_error_type is not None:
            error_type, error = case_error_type, case_error
            for evaluator in evaluators:
                name = evaluator.get_name()
                evaluations.pop(name, None)
                details.pop(name, None)
                errors.pop(name, None)
            logger.warning(
                "candidate_id=<%s>, case_name=<%s>, run_index=<%s>, error_type=<%s>, error=<%s> | case failed "
                "after the evaluators were judged again | no evaluator score recorded",
                call.candidate_id,
                previous.case_name,
                previous.run_index,
                case_error_type,
                case_error,
            )
        replaced = _replaced(previous, pass_index=pass_index).model_copy(
            update={
                "usage": Usage.zero(),
                "benchmark_usage": Usage.zero(),
                "judge_usage": sum(
                    (u for name, u in previous.judge_usage.items() if name in previous.evaluator_errors),
                    Usage.zero(),
                ),
            }
        )
        earlier = [*previous.previous_attempts, replaced]
        return previous.model_copy(
            update={
                "status": TrialStatus.EVALUATOR_INVALID if not evaluations else TrialStatus.COMPLETED,
                "error_type": error_type,
                "error": error,
                "attempts": len(earlier) + 1,
                "previous_attempts": earlier,
                "evaluations": evaluations,
                "details": details,
                "evaluator_errors": errors,
                "judge_usage": {**kept, **judge_usage},
                "rollouts": 0,
            }
        )

    def _judged(
        self,
        evaluations: dict[str, EvaluationOutput],
        details: dict[str, list[EvaluationOutput]],
        errors: dict[str, str],
        evaluators: Sequence[Evaluator],
        report: Any,
        state: _TrialState,
        view: EvaluatorView,
    ) -> tuple[dict[str, EvaluationOutput], dict[str, list[EvaluationOutput]], dict[str, str], dict[str, Usage]]:
        """Fold this attempt's evaluator results into `evaluations` / `details` / `errors`.

        `Experiment`'s report holds one row per evaluator; a failed evaluator's row is a score-0
        placeholder, dropped here. Case-level placeholder rows (`Experiment` failed on the case itself) are
        dropped by `_ran` and `_rejudged`. Experiment's own no-evaluator placeholder is never in `evaluators`.
        """
        rows = {row["evaluator"]: i for i, row in enumerate(report.cases)} if evaluators else {}
        judge_usage: dict[str, Usage] = {}
        for evaluator in evaluators:
            name = evaluator.get_name()
            judge_usage[name] = view.judge_usage.get(name, Usage.zero())
            failed = state.errors.get(name) or view.errors.get(name)  # the span covers an aggregator error too
            if failed is not None or name not in rows:
                errors[name] = failed or "Exception"
                evaluations.pop(name, None)
                details.pop(name, None)
                continue
            i = rows[name]
            evaluations[name] = EvaluationOutput(
                score=float(report.scores[i]),
                test_pass=bool(report.test_passes[i]),
                reason=report.reasons[i],
                label=view.labels.get(name),
            )
            details[name] = list(report.detailed_results[i])
        return evaluations, details, errors, judge_usage


@dataclass
class _Call:
    """One `TrialRunner.run` call's state, shared by its passes."""

    candidate_id: str
    task: TaskFn
    semaphore: asyncio.Semaphore
    records: dict[_Key, TrialRecord] = field(default_factory=dict)
    live: dict[_Key, EvaluationData] = field(default_factory=dict)  # each trial's output as the task returned it
    passes: dict[_Key, int] = field(default_factory=dict)  # the pass each record of this call ran in
    # trials whose last attempt failed on infrastructure (the agent, else an evaluator) -> (error type, when)
    infrastructure: dict[_Key, tuple[str, float]] = field(default_factory=dict)
    warned: set[str] = field(default_factory=set)  # the fields `_storable` has warned about


def _needs_agent(record: TrialRecord | None) -> bool:
    """No record yet, or one whose agent side failed on infrastructure: the agent must run (1 rollout)."""
    return record is None or record.status is TrialStatus.INFRASTRUCTURE_INVALID


def _has_failed_evaluators(record: TrialRecord | None) -> bool:
    return (
        record is not None
        and record.status in (TrialStatus.COMPLETED, TrialStatus.EVALUATOR_INVALID)
        and bool(record.evaluator_errors)
    )


def _replaced(record: TrialRecord, *, pass_index: int) -> AttemptRecord:
    """A record a re-run replaces, as one earlier attempt."""
    return AttemptRecord(
        status=record.status,
        error_type=record.error_type,
        error=record.error,
        usage=record.usage,
        judge_usage=sum(record.judge_usage.values(), Usage.zero()),
        pass_index=pass_index,
        rollouts=record.rollouts,
        benchmark_usage=record.benchmark_usage,
    )


def _evaluation_data(case: Case, record: TrialRecord) -> EvaluationData:
    """A saved trial's output as `Experiment` hands it to evaluators, to re-judge without running the agent.

    The input is the case's own: an `input` the rollout returned to replace it isn't saved in the record.
    """
    return EvaluationData(
        name=case.name,
        input=case.input,
        expected_output=case.expected_output,
        expected_assertion=case.expected_assertion,
        expected_trajectory=case.expected_trajectory,
        expected_interactions=case.expected_interactions,
        expected_environment_state=case.expected_environment_state,
        metadata=case.metadata,
        actual_output=record.output,
        actual_trajectory=record.trajectory,
        actual_interactions=record.interactions,
        actual_environment_state=record.environment_state,
    )


def _storable(value: Any, call: _Call, field_name: str) -> Any:
    """`value` if it can be saved as JSON, else a copy that can, warning once per call and field.

    So a non-JSON value the rollout returned doesn't make the save fail (a container that holds itself
    still does). NaN and infinity count as non-JSON, as the save writes strict JSON. `output` becomes
    `str(output)`. Any other field keeps its shape and is converted as pydantic saves it: models,
    datetimes, and non-string keys become JSON, bytes base64, NaN and infinity null, and only what
    pydantic can't encode becomes `str()`. A `Session` is returned as is.
    """
    if isinstance(value, Session):
        return value
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        if field_name not in call.warned:
            call.warned.add(field_name)
            logger.warning(
                "candidate_id=<%s>, field=<%s>, type=<%s> | trial value isn't JSON-serializable, converted to JSON",
                call.candidate_id,
                field_name,
                type(value).__name__,
            )
        return (
            str(value)
            if field_name == "output"
            # Through JSON text, so NaN in a model's float field is null here too, as it is after a reload.
            else json.loads(to_json(value, fallback=str, bytes_mode="base64", inf_nan_mode="null"))
        )
    return value
