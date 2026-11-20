import asyncio
import copy
import datetime
import enum
import errno
import gc
import json
import logging
import math
import os
import threading

import opentelemetry.trace as trace_api
import pytest
from botocore.exceptions import CredentialRetrievalError, EndpointConnectionError, ProfileNotFound
from pydantic import BaseModel
from strands import Agent
from strands.types.exceptions import ModelThrottledException

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import Budget, InfrastructureError, TextValue, tunable
from strands_evals.optimization.declaration.tunable import _FactoryArgumentError
from strands_evals.optimization.execution.rollout import default_rollout, make_task
from strands_evals.optimization.execution.traces import TraceCapture
from strands_evals.optimization.execution.trials import TrialRunner, _EvaluatorProxy, _rerun_wait, _TrialState
from strands_evals.optimization.record.run_directory import RunDirectory
from strands_evals.optimization.setup.budget import BudgetTracker
from strands_evals.types import EvaluationOutput
from strands_evals.types.evaluation import NOT_APPLICABLE, EnvironmentState, EvaluationData
from strands_evals.types.optimization import TrialRecord, TrialStatus
from strands_evals.types.trace import Session
from tests.strands_evals.optimization.scripted_model import ScriptedModel


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("strands_evals.experiment._INITIAL_RETRY_DELAY", 0)


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """The waits before re-run passes, recorded instead of slept."""
    seconds = []

    async def sleep(delay):
        seconds.append(delay)

    monkeypatch.setattr("strands_evals.optimization.execution.trials._sleep", sleep)
    return seconds


@pytest.fixture
def traces():
    return TraceCapture()


class MemoryCache:
    """An in-memory `TrialCache`; `saved` lists every save in order."""

    def __init__(self):
        self.trials: dict[tuple[str, int, str], TrialRecord] = {}
        self.saved: list[TrialRecord] = []

    def load_trial(self, candidate_id, run_index, case_name):
        return self.trials.get((candidate_id, run_index, case_name))

    def trial_status(self, candidate_id, run_index, case_name):
        record = self.trials.get((candidate_id, run_index, case_name))
        return None if record is None else record.status

    def save_trial(self, record):
        self.trials[(record.candidate_id, record.run_index, record.case_name)] = record
        self.saved.append(record)


class Fixed(Evaluator):
    """Scores every trial `score`; raises `error` on its first `failures` calls."""

    def __init__(self, name="Fixed", score=1.0, failures=0, error=None):
        super().__init__(name=name)
        self.score, self.failures, self.calls = score, failures, 0
        self.error = error or ValueError("bad judge output")

    async def evaluate_async(self, evaluation_case):
        self.calls += 1
        if self.calls <= self.failures:
            raise copy.copy(self.error)
        return [EvaluationOutput(score=self.score, test_pass=True, reason=f"saw {evaluation_case.actual_output}")]


class Judge(Evaluator):
    """An LLM judge: a real agent on a scripted model, so its `chat` spans land under the evaluator span."""

    def __init__(self, name="Judge", tokens=30, failures=0):
        super().__init__(name=name)
        self.tokens, self.failures, self.calls = tokens, failures, 0

    async def evaluate_async(self, evaluation_case):
        self.calls += 1
        usage = {"inputTokens": self.tokens - 3, "outputTokens": 3, "totalTokens": self.tokens}
        await Agent(model=ScriptedModel(["0.5"], usage=usage), callback_handler=None).invoke_async("judge")
        if self.calls <= self.failures:
            raise ValueError("unparseable verdict")
        return [EvaluationOutput(score=0.5, test_pass=True, reason="half")]


def _cases(*names):
    return [Case(name=name, input=f"question {name}") for name in names]


def _runner(traces, cache, cases, evaluators, *, max_workers=4, rerun_passes=2):
    return TrialRunner(
        cases,
        evaluators,
        traces,
        cache,
        split_of={case.name: "train" for case in cases},
        max_workers=max_workers,
        rerun_passes=rerun_passes,
    )


def _agent_task(reply="answer", calls=None):
    """A task running a real agent on a scripted model (15 tokens per call)."""

    async def task(case):
        if calls is not None:
            calls.append(case.name)
        agent = Agent(model=ScriptedModel([reply]), callback_handler=None)
        return {"output": str(await agent.invoke_async(case.input))}

    return task


async def _plain(case):
    return f"answer to {case.name}"


# scoring end to end


async def test_a_candidate_scores_every_case_and_run(traces):
    cache = MemoryCache()
    runner = _runner(traces, cache, _cases("a", "b"), [Fixed(score=0.8)])

    records, spent = await runner.run("cand", _agent_task(), ["a", "b"], runs=2, allowance=0)

    assert spent == 4
    assert [(r.case_name, r.run_index) for r in records] == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    record = records[0]
    assert record.status is TrialStatus.COMPLETED
    assert record.split == "train"
    assert record.output.strip() == "answer"
    assert isinstance(record.trajectory, Session)
    assert record.usage.tokens == 15
    assert record.usage.model_calls == 1
    assert record.evaluations["Fixed"].score == 0.8
    assert record.details["Fixed"][0].reason.strip() == "saw answer"
    assert record.evaluator_errors == {}
    assert record.rollouts == 1
    assert record.attempts == 1
    assert len(cache.saved) == 4


async def test_a_sync_task_works(traces):
    def task(case):
        return f"sync {case.name}"

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", task, ["a"], runs=1, allowance=0
    )

    assert records[0].output == "sync a"
    assert records[0].status is TrialStatus.COMPLETED


async def test_records_come_back_in_the_order_asked(traces):
    runner = _runner(traces, MemoryCache(), _cases("a", "b", "c"), [Fixed()])

    records, _ = await runner.run("cand", _plain, ["c", "a"], runs=1, allowance=0)

    assert [r.case_name for r in records] == ["c", "a"]


async def test_an_outer_span_doesnt_merge_trials_into_one_trace(traces):
    runner = _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()])

    with trace_api.get_tracer("optimization-tests").start_as_current_span("user span"):
        records, _ = await runner.run("cand", _agent_task(), ["a", "b"], runs=1, allowance=0)

    assert [r.usage.tokens for r in records] == [15, 15]
    assert all(isinstance(r.trajectory, Session) for r in records)


async def test_output_that_isnt_json_is_saved_as_text(traces, caplog):
    class Thing:
        def __str__(self):
            return "a thing"

    async def task(case):
        return {"output": Thing()}

    with caplog.at_level(logging.WARNING):
        records, _ = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()]).run(
            "cand", task, ["a", "b"], runs=1, allowance=0
        )

    assert [r.output for r in records] == ["a thing", "a thing"]
    assert sum("isn't JSON-serializable" in message for message in caplog.messages) == 1


async def test_trajectory_interactions_and_state_that_arent_json_keep_their_shape(traces, caplog):
    class Thing:
        def __str__(self):
            return "a thing"

    class Color(enum.Enum):
        RED = "red"

    class ToolCall(BaseModel):
        at: datetime.datetime

    at = datetime.datetime(2026, 1, 2, 3, 4, 5)

    async def task(case):
        return {
            "output": "ok",
            "trajectory": [{"tool": "lookup", "result": Thing(), "call": ToolCall(at=at)}, {Color.RED: 1}],
            "interactions": [{"node_name": "agent", "messages": [Thing()]}],
            "environment_state": [EnvironmentState(name="grid", state={(0, 1): Thing()})],
        }

    with caplog.at_level(logging.WARNING):
        records, _ = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()]).run(
            "cand", task, ["a", "b"], runs=1, allowance=0
        )

    record = records[0]
    assert record.status is TrialStatus.COMPLETED
    assert record.trajectory == [
        {"tool": "lookup", "result": "a thing", "call": {"at": "2026-01-02T03:04:05"}},  # as pydantic saves it
        {"red": 1},  # a key that isn't a str is kept, as pydantic encodes it
    ]
    assert record.interactions == [{"node_name": "agent", "messages": ["a thing"]}]
    assert record.environment_state == [EnvironmentState(name="grid", state={"0,1": "a thing"})]
    assert TrialRecord.model_validate_json(record.model_dump_json()) == record  # the save can't fail
    assert sum("isn't JSON-serializable" in message for message in caplog.messages) == 3  # once per field


async def test_nan_and_infinity_are_converted_since_the_save_writes_strict_json(traces):
    async def task(case):
        return {"output": {"score": math.nan}, "trajectory": [{"x": math.nan, "y": [math.inf]}]}

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", task, ["a"], runs=1, allowance=0
    )

    record = records[0]
    assert record.output == "{'score': nan}"
    assert record.trajectory == [{"x": None, "y": [None]}]
    json.dumps(record.model_dump(mode="json"), allow_nan=False)  # what the run directory writes


async def test_nan_in_a_models_float_field_is_null_in_memory_as_after_a_reload(traces):
    class Step(BaseModel):
        confidence: float

    async def task(case):
        return {"output": "ok", "trajectory": [Step(confidence=math.nan)]}

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", task, ["a"], runs=1, allowance=0
    )

    record = records[0]
    assert record.trajectory == [{"confidence": None}]
    assert TrialRecord.model_validate_json(record.model_dump_json()) == record


# cache


async def test_a_second_call_is_all_cache_hits(traces):
    cache = MemoryCache()
    evaluator = Fixed()
    calls = []
    runner = _runner(traces, cache, _cases("a", "b"), [evaluator])
    first, _ = await runner.run("cand", _agent_task(calls=calls), ["a", "b"], runs=1, allowance=0)

    second, spent = await runner.run("cand", _agent_task(calls=calls), ["a", "b"], runs=1, allowance=0)

    assert spent == 0
    assert calls == ["a", "b"]
    assert evaluator.calls == 2
    assert second == first


async def test_pending_counts_trials_without_a_measured_record(traces):
    cache = MemoryCache()
    runner = _runner(traces, cache, _cases("a", "b"), [Fixed()], rerun_passes=0)
    assert runner.pending("cand", ["a", "b"], runs=2) == 4

    async def task(case):
        if case.name == "b":
            raise InfrastructureError("sandbox down")
        return "ok"

    await runner.run("cand", task, ["a", "b"], runs=2, allowance=0)

    assert runner.pending("cand", ["a", "b"], runs=2) == 2  # b's two infrastructure failures

    # Statuses only: pending never loads a record (a transcript read per trial).
    cache.load_trial = None
    assert runner.pending("cand", ["a", "b"], runs=2) == 2


# concurrency


async def test_every_run_shares_one_max_workers_cap(traces):
    running, peak, started = set(), [0], []

    async def task(case):
        started.append(case.name)
        running.add(id(asyncio.current_task()))
        peak[0] = max(peak[0], len(running))
        await asyncio.sleep(0.01)
        running.discard(id(asyncio.current_task()))
        return "ok"

    runner = _runner(traces, MemoryCache(), _cases("a", "b", "c"), [Fixed()], max_workers=2)

    records, spent = await runner.run("cand", task, ["a", "b", "c"], runs=2, allowance=0)

    assert spent == 6
    assert peak[0] == 2
    assert started[:2] == ["a", "a"]  # case-major: both of a's runs go first, at once
    assert {r.run_index for r in records} == {0, 1}


async def test_sync_tasks_run_max_workers_at_once_past_asyncios_shared_pool(traces):
    # asyncio's shared pool holds min(32, cpus + 4) threads; one more trial than that must still run at once
    n = min(32, (os.cpu_count() or 1) + 4) + 1
    barrier = threading.Barrier(n, timeout=10)
    seen = []

    def task(case):
        seen.append((threading.current_thread().name, trace_api.get_current_span().name))
        barrier.wait()  # breaks, failing the trial, unless all n trials are running at once
        return "ok"

    names = [f"c{i}" for i in range(n)]
    records, _ = await _runner(traces, MemoryCache(), _cases(*names), [Fixed()], max_workers=n).run(
        "cand", task, names, runs=1, allowance=0
    )

    assert [r.status for r in records] == [TrialStatus.COMPLETED] * n
    assert all(thread.startswith("trial") for thread, _ in seen)
    assert sorted(span for _, span in seen) == sorted(f"attempt {name}" for name in names)  # keeps the span


async def test_a_kill_mid_call_keeps_every_finished_trial(traces):
    cache = MemoryCache()
    gate = asyncio.Event()

    async def task(case):
        if case.name == "c":
            await gate.wait()  # never set: the process is killed while c runs
        return "ok"

    runner = _runner(traces, cache, _cases("a", "b", "c"), [Fixed()], max_workers=1)
    call = asyncio.create_task(runner.run("cand", task, ["a", "b", "c"], runs=1, allowance=0))
    while len(cache.saved) < 2:
        await asyncio.sleep(0.001)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert [r.case_name for r in cache.saved] == ["a", "b"]
    assert _runner(traces, cache, _cases("a", "b", "c"), [Fixed()]).pending("cand", ["a", "b", "c"], runs=1) == 1


async def test_a_setup_error_raises_after_the_running_trials_are_saved(traces):
    cache = MemoryCache()

    async def task(case):
        if case.name == "b":
            raise _FactoryArgumentError("resource 'prompt' is named like a declared marker")
        return "ok"

    runner = _runner(traces, cache, _cases("a", "b", "c"), [Fixed()], max_workers=1)

    with pytest.raises(TypeError, match="named like a declared marker"):
        await runner.run("cand", task, ["a", "b", "c"], runs=1, allowance=0)

    assert [r.case_name for r in cache.saved] == ["a"]


async def test_a_failed_save_raises_without_starting_more_trials(traces):
    class FullDisk(MemoryCache):
        def save_trial(self, record):
            raise OSError("disk full")

    calls = []

    async def task(case):
        calls.append(case.name)
        return "ok"

    runner = _runner(traces, FullDisk(), _cases("a", "b", "c"), [Fixed()], max_workers=1)

    with pytest.raises(OSError, match="disk full"):
        await runner.run("cand", task, ["a", "b", "c"], runs=1, allowance=0)
    await asyncio.sleep(0.05)  # a trial left running after `run` raised would start by now

    assert calls == ["a"]


async def test_a_failed_call_shuts_its_thread_pool_down(traces):
    class FullDisk(MemoryCache):
        def save_trial(self, record):
            raise OSError("disk full")

    def task(case):  # sync, so it runs on the call's pool
        return "ok"

    before = {t for t in threading.enumerate() if t.name.startswith("trial")}
    gc.disable()  # the raised error's traceback holds the pool; only an explicit shutdown frees its threads
    try:
        with pytest.raises(OSError, match="disk full"):
            await _runner(traces, FullDisk(), _cases("a", "b"), [Fixed()], max_workers=2).run(
                "cand", task, ["a", "b"], runs=1, allowance=0
            )
        new = {t for t in threading.enumerate() if t.name.startswith("trial")} - before
        for thread in new:
            thread.join(timeout=5)  # shutdown(wait=False): the idle threads exit on their own
    finally:
        gc.enable()

    assert [t for t in new if t.is_alive()] == []


async def test_a_cancelled_call_still_exits_a_sync_environment(traces):
    events, release = [], threading.Event()

    class Sandbox:
        def __enter__(self):
            events.append("enter")
            return {}

        def __exit__(self, *exc_info):
            events.append("exit")

    def rollout(agent_factory, case):
        release.wait(5)
        return {"output": "ok"}

    @tunable(prompt=TextValue(default="p"))
    def build(prompt):
        return prompt

    task = make_task(build.bind({}), rollout, environment=lambda case: Sandbox())
    call = asyncio.create_task(
        _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run("cand", task, ["a"], runs=1, allowance=0)
    )
    while events != ["enter"]:
        await asyncio.sleep(0.001)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    release.set()  # the rollout's thread can't be stopped; once it returns, the environment must still exit
    for _ in range(500):
        if "exit" in events:
            break
        await asyncio.sleep(0.01)

    assert events == ["enter", "exit"]


# statuses


async def test_every_evaluator_failing_is_evaluator_invalid(traces):
    runner = _runner(traces, MemoryCache(), _cases("a"), [Fixed("A", failures=9), Fixed("B", failures=9)])

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].status is TrialStatus.EVALUATOR_INVALID
    assert records[0].evaluations == {}
    assert records[0].evaluator_errors == {"A": "ValueError", "B": "ValueError"}


class Scores(Evaluator):
    """Returns the next of `scores` on each call (the last one repeated)."""

    def __init__(self, *scores, name="Scores"):
        super().__init__(name=name)
        self.scores, self.calls = scores, 0

    async def evaluate_async(self, evaluation_case):
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [EvaluationOutput(score=score, test_pass=True)]


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
async def test_a_score_that_isnt_finite_is_a_failed_evaluator(traces, score):
    runner = _runner(traces, MemoryCache(), _cases("a"), [Fixed("A"), Scores(score)], rerun_passes=0)

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].status is TrialStatus.COMPLETED
    assert "Scores" not in records[0].evaluations
    assert records[0].evaluator_errors == {"Scores": "ValueError"}


async def test_an_aggregate_that_isnt_finite_is_a_failed_evaluator(traces):
    evaluator = Fixed("A")
    evaluator.aggregator = lambda outputs: (math.nan, True, "")
    runner = _runner(traces, MemoryCache(), _cases("a"), [evaluator], rerun_passes=0)

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].status is TrialStatus.EVALUATOR_INVALID
    assert records[0].evaluator_errors == {"A": "ValueError"}


async def test_a_score_that_isnt_finite_is_re_judged(traces):
    runner = _runner(traces, MemoryCache(), _cases("a"), [Scores(math.nan, 0.5)], rerun_passes=1)

    records, spent = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert spent == 1
    assert records[0].status is TrialStatus.COMPLETED
    assert records[0].evaluations["Scores"].score == 0.5
    assert [a.status for a in records[0].previous_attempts] == [TrialStatus.EVALUATOR_INVALID]


async def test_a_score_outside_the_range_is_a_failed_evaluator_and_re_judged(traces):
    runner = _runner(traces, MemoryCache(), _cases("a"), [Scores(4.0, 0.5)], rerun_passes=1)

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].evaluations["Scores"].score == 0.5
    assert [a.status for a in records[0].previous_attempts] == [TrialStatus.EVALUATOR_INVALID]


async def test_the_range_check_uses_score_range_and_float_noise_and_skips_not_applicable(traces):
    stars = Scores(5.0, name="Stars")
    stars.score_range = (1.0, 5.0)
    edge = Scores(1 + 1e-15, name="Edge")
    unbounded = Scores(1e6, name="Unbounded")
    unbounded.score_range = (0.0, None)

    class NotApplicable(Evaluator):
        async def evaluate_async(self, evaluation_case):
            return [EvaluationOutput(score=-1.0, test_pass=True, label=NOT_APPLICABLE)]

    evaluators = [stars, edge, unbounded, NotApplicable(name="NA")]
    runner = _runner(traces, MemoryCache(), _cases("a"), evaluators, rerun_passes=0)

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].evaluator_errors == {}
    assert set(records[0].evaluations) == {"Stars", "Edge", "Unbounded", "NA"}


async def test_a_score_that_isnt_finite_doesnt_fail_the_save(traces, tmp_path):
    run = RunDirectory.create(tmp_path, {"cases": []})
    runner = _runner(traces, run, _cases("a", "b"), [Scores(math.nan)], rerun_passes=0)

    records, _ = await runner.run("cand", _plain, ["a", "b"], runs=1, allowance=0)

    assert [r.status for r in records] == [TrialStatus.EVALUATOR_INVALID] * 2
    assert RunDirectory.open(run.root).load_trial("cand", 0, "b") == records[1]


async def test_one_evaluator_failing_leaves_the_trial_completed(traces):
    runner = _runner(traces, MemoryCache(), _cases("a"), [Fixed("A"), Fixed("B", failures=9)], rerun_passes=0)

    records, _ = await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].status is TrialStatus.COMPLETED
    assert set(records[0].evaluations) == {"A"}
    assert records[0].evaluator_errors == {"B": "ValueError"}


async def test_a_judge_throttled_on_every_try_leaves_the_trial_completed(traces, waits):
    throttled = Fixed("Throttled", failures=99, error=ModelThrottledException("slow down"))
    runner = _runner(traces, MemoryCache(), _cases("a"), [Fixed("A"), throttled], rerun_passes=1)

    records, spent = await runner.run("cand", _plain, ["a"], runs=1, allowance=5)

    assert records[0].status is TrialStatus.COMPLETED
    assert records[0].evaluator_errors == {"Throttled": "ModelThrottledException"}
    assert spent == 1  # the re-run pass re-judged it, for no rollout
    assert len(waits) == 1  # after waiting, as throttling is infrastructure
    once = Fixed("Throttled", failures=99, error=ModelThrottledException("slow down"))
    await _runner(traces, MemoryCache(), _cases("a"), [Fixed("A"), once], rerun_passes=0).run(
        "cand", _plain, ["a"], runs=1, allowance=5
    )
    assert throttled.calls == 2 * once.calls  # judged in the first pass, then again in the re-run pass


async def test_no_evaluators_means_completed(traces):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), []).run("cand", _plain, ["a"], runs=1, allowance=0)

    assert records[0].status is TrialStatus.COMPLETED
    assert records[0].evaluations == {}
    assert records[0].evaluator_errors == {}
    assert records[0].judge_usage == {}


# placeholder rows


async def _unserializable(case):
    """Returns an output `Experiment` can't serialize for its span, so it fails after the task returned."""
    return {"output": {(0, 1): "x"}}


def _unserializable_error():
    with pytest.raises(TypeError) as error:
        json.dumps({(0, 1): "x"})
    return str(error.value)


async def test_a_case_experiment_fails_after_the_task_returned_is_unmeasured_and_never_judged(traces, caplog):
    judge = Fixed("A")

    with caplog.at_level(logging.WARNING):
        records, spent = await _runner(traces, MemoryCache(), _cases("a"), [judge]).run(
            "cand", _unserializable, ["a"], runs=1, allowance=5
        )

    record = records[0]
    assert record.status is TrialStatus.EVALUATOR_INVALID
    assert record.evaluations == {}  # not the placeholder's 0
    assert record.evaluator_errors == {}  # nothing to re-judge: no output was saved
    assert (record.error_type, record.error) == ("TypeError", _unserializable_error())
    assert record.output is None
    assert judge.calls == 0  # neither in the first pass nor in a re-run pass
    assert (spent, record.attempts) == (1, 1)
    assert "case failed after the task returned" in caplog.text


async def test_a_case_experiment_failed_stays_unjudged_after_a_resume_with_rejudge(traces, tmp_path):
    run = RunDirectory.create(tmp_path, {"cases": []})
    judge = Fixed("A")
    first, _ = await _runner(traces, run, _cases("a"), [judge], rerun_passes=0).run(
        "cand", _unserializable, ["a"], runs=1, allowance=1
    )
    runner = _runner(traces, RunDirectory.open(run.root), _cases("a"), [judge])

    assert runner.pending("cand", ["a"], runs=1) == 0
    records, spent = await runner.run("cand", _unserializable, ["a"], runs=1, allowance=1, rejudge=True)

    assert (spent, judge.calls) == (0, 0)
    assert records[0].status is TrialStatus.EVALUATOR_INVALID
    assert records == first


async def test_with_no_evaluators_a_case_experiment_failed_is_completed_with_its_error(traces):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), []).run(
        "cand", _unserializable, ["a"], runs=1, allowance=0
    )

    assert records[0].status is TrialStatus.COMPLETED  # a usage-only objective still counts the trial
    assert records[0].error_type == "TypeError"


async def test_rows_experiment_replaced_after_its_evaluators_ran_are_not_scores(traces, monkeypatch):
    dumps = []

    def broken(self, *args, **kwargs):
        dumps.append(self.name)
        raise RuntimeError("can't dump")

    monkeypatch.setattr(EvaluationData, "model_dump", broken)  # Experiment dumps the case after judging it
    flaky, judge = Fixed("Flaky", failures=1), Judge("J", tokens=30)

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [flaky, judge]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    record = records[0]
    assert dumps == ["a"]  # hit once, in the first pass: the evaluator that raised isn't re-judged either
    assert (flaky.calls, judge.calls) == (1, 1)
    assert record.status is TrialStatus.EVALUATOR_INVALID
    assert (record.evaluations, record.evaluator_errors) == ({}, {})
    assert (record.error_type, record.error) == ("RuntimeError", "can't dump")
    assert record.output == "answer to a"  # saved before the evaluators ran
    assert record.judge_usage["J"].tokens == 30  # the judge's spend is still counted


@pytest.mark.parametrize("failures", [1, 2], ids=["judged-ok", "raised-again"])
async def test_rows_experiment_replaced_in_a_re_judge_are_not_scores(traces, monkeypatch, caplog, failures):
    cache = MemoryCache()
    ok, broken = Fixed("Ok"), Fixed("Broken", score=0.3, failures=failures)  # 2: raises in the re-judge too
    await _runner(traces, cache, _cases("a"), [ok, broken], rerun_passes=0).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    def failing_dump(self, *args, **kwargs):
        raise RuntimeError("can't dump")

    monkeypatch.setattr(EvaluationData, "model_dump", failing_dump)  # Experiment dumps the case after judging it
    with caplog.at_level(logging.WARNING):
        records, spent = await _runner(traces, cache, _cases("a"), [ok, broken]).run(
            "cand", _plain, ["a"], runs=1, allowance=0, rejudge=True
        )

    record = records[0]
    assert (spent, broken.calls) == (0, 2)  # judged again once, in the first re-judge; not in the re-run passes
    assert "Broken" not in record.evaluations  # not the placeholder's 0
    assert record.evaluator_errors == {}  # nothing left to judge again, even when it raised again
    assert (record.error_type, record.error) == ("RuntimeError", "can't dump")
    assert record.evaluations["Ok"].score == 1.0  # the first pass's score is kept
    assert record.status is TrialStatus.COMPLETED
    assert "case failed after the evaluators were judged again" in caplog.text

    monkeypatch.undo()
    await _runner(traces, cache, _cases("a"), [ok, broken]).run(
        "cand", _plain, ["a"], runs=1, allowance=0, rejudge=True
    )
    assert broken.calls == 2  # a later re-judge leaves it alone too


async def test_a_task_that_raised_isnt_read_as_a_case_failure(traces, caplog):
    async def crashes(case):
        raise ValueError("agent bug")  # recorded on the case span too, as every task exception is

    with caplog.at_level(logging.WARNING):
        records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed("A")]).run(
            "cand", crashes, ["a"], runs=1, allowance=0
        )

    assert records[0].status is TrialStatus.AGENT_FAILURE
    assert (records[0].error_type, records[0].error) == ("ValueError", "agent bug")
    assert "case failed after the task returned" not in caplog.text


async def test_a_session_that_cant_be_built_after_the_rollout_returned_is_unmeasured(traces, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("can't map spans")

    monkeypatch.setattr(traces, "session", broken)
    judge = Fixed("A")

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [judge]).run(
        "cand", _plain, ["a"], runs=1, allowance=5
    )

    assert records[0].status is TrialStatus.EVALUATOR_INVALID
    assert (records[0].error_type, records[0].error) == ("RuntimeError", "can't map spans")
    assert (spent, judge.calls) == (1, 0)


async def test_an_agent_failure_is_recorded_without_evaluations(traces):
    async def task(case):
        raise ValueError("the agent broke")

    evaluator = Fixed()
    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [evaluator]).run(
        "cand", task, ["a"], runs=1, allowance=5
    )

    assert records[0].status is TrialStatus.AGENT_FAILURE
    assert records[0].error_type == "ValueError"
    assert records[0].evaluations == {}
    assert evaluator.calls == 0
    assert spent == 1  # agent failures aren't re-run


# attempts and re-run passes


async def test_a_trial_throttled_once_then_completed_has_two_attempts(traces):
    tries = []

    async def task(case):  # raised from the task: a Strands agent retries a throttled model call itself
        tries.append(case.name)
        if len(tries) == 1:
            raise ModelThrottledException("slow down")
        return "answer"

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", task, ["a"], runs=1, allowance=0
    )

    record = records[0]
    assert record.status is TrialStatus.COMPLETED
    assert record.attempts == 2
    assert record.previous_attempts[0].status is TrialStatus.INFRASTRUCTURE_INVALID
    assert record.previous_attempts[0].error_type == "ModelThrottledException"
    assert record.previous_attempts[0].rollouts == 0
    assert record.rollouts == 1
    assert spent == 1


async def test_an_infrastructure_failure_is_re_run_within_the_allowance(traces):
    tries = []

    async def task(case):
        tries.append(case.name)
        if len(tries) == 1:
            raise InfrastructureError("sandbox down")
        return "ok"

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand",
        task,
        ["a"],
        runs=1,
        allowance=2,  # the first pass's rollout, plus one re-run
    )

    record = records[0]
    assert record.status is TrialStatus.COMPLETED
    assert spent == 2
    assert record.attempts == 2
    assert record.previous_attempts[0].status is TrialStatus.INFRASTRUCTURE_INVALID
    assert record.previous_attempts[0].rollouts == 1
    assert record.previous_attempts[0].pass_index == 0
    assert record.rollouts + sum(a.rollouts for a in record.previous_attempts) == spent


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EMFILE, "Too many open files"),
        ProfileNotFound(profile="bedrock"),
        CredentialRetrievalError(provider="assume-role", error_msg="metadata service timed out"),
    ],
    ids=["OSError-EMFILE", "ProfileNotFound", "CredentialRetrievalError"],
)
async def test_a_factory_that_runs_out_of_files_or_credentials_is_re_run(traces, error):
    built = []

    def factory():
        built.append(error)
        if len(built) == 1:
            raise error
        return Agent(model=ScriptedModel(["answer"]), callback_handler=None)

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", make_task(factory, default_rollout), ["a"], runs=1, allowance=2
    )

    record = records[0]
    assert record.status is TrialStatus.COMPLETED
    assert spent == 2
    assert [(a.status, a.error_type) for a in record.previous_attempts] == [
        (TrialStatus.INFRASTRUCTURE_INVALID, type(error).__name__)
    ]


async def test_an_exhausted_allowance_leaves_the_trial_infrastructure_invalid(traces):
    async def task(case):
        raise InfrastructureError("sandbox down")

    records, spent = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()]).run(
        "cand", task, ["a", "b"], runs=1, allowance=3
    )

    assert spent == 3  # the allowance covers the whole call: two first passes, one re-run
    assert [r.status for r in records] == [TrialStatus.INFRASTRUCTURE_INVALID] * 2


async def test_a_call_never_spends_more_than_the_budget_allows(traces, waits):
    budget = BudgetTracker(Budget(max_rollouts=2))

    async def task(case):
        raise InfrastructureError("sandbox down")

    runner = _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()])
    assert budget.can_afford(runner.pending("cand", ["a", "b"], runs=1))

    _, spent = await runner.run("cand", task, ["a", "b"], runs=1, allowance=budget.allowance())
    budget.spend(spent)

    assert spent == 2  # the first pass used the whole allowance: no re-run
    assert budget.remaining == 0
    assert waits == []  # and no wait for a pass that never runs


async def test_a_re_run_after_resume_keeps_the_earlier_attempt(traces, waits):
    cache = MemoryCache()

    async def down(case):
        raise InfrastructureError("sandbox down")

    await _runner(traces, cache, _cases("a"), [Fixed()], rerun_passes=0).run("cand", down, ["a"], runs=1, allowance=0)

    records, spent = await _runner(traces, cache, _cases("a"), [Fixed()]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert spent == 1
    assert records[0].status is TrialStatus.COMPLETED
    assert [a.status for a in records[0].previous_attempts] == [TrialStatus.INFRASTRUCTURE_INVALID]
    assert waits == []  # the first pass of a call never waits


# the wait before a re-run pass


@pytest.fixture
def clock(monkeypatch):
    """A fake clock that only the waits move; returns [now]."""
    now = [0.0]

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr("strands_evals.optimization.execution.trials._sleep", sleep)
    monkeypatch.setattr("strands_evals.optimization.execution.trials._clock", lambda: now[0])
    return now


async def test_a_short_outage_is_waited_out_before_the_re_run(traces, clock):
    async def task(case):  # the endpoint is unreachable for the first second
        if clock[0] < 1.0:
            raise EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        return "ok"

    records, spent = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()]).run(
        "cand", task, ["a", "b"], runs=1, allowance=4
    )

    assert [r.status for r in records] == [TrialStatus.COMPLETED] * 2
    assert spent == 4  # the first pass, then one re-run pass
    assert 2.5 <= clock[0] <= 5.0


async def test_each_re_run_pass_waits_about_twice_as_long_as_the_one_before(traces, waits, caplog):
    async def down(case):
        raise InfrastructureError("sandbox down")

    with caplog.at_level(logging.WARNING):
        records, spent = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()], rerun_passes=3).run(
            "cand", down, ["a"], runs=1, allowance=10
        )

    assert records[0].status is TrialStatus.INFRASTRUCTURE_INVALID
    assert spent == 4
    assert len(waits) == 3
    for wait, (low, high) in zip(waits, [(2.5, 5.0), (5.0, 10.0), (10.0, 20.0)], strict=True):
        assert low <= wait <= high
    logged = [r.getMessage() for r in caplog.records if "waiting before re-running" in r.getMessage()]
    assert len(logged) == 3
    assert "pass=<1>, trials=<1>, error_type=<InfrastructureError>" in logged[0]


@pytest.mark.parametrize("pass_index", [5, 6, 10_000])
def test_the_wait_before_a_re_run_pass_is_capped(pass_index):
    assert all(30.0 <= _rerun_wait(pass_index) <= 60.0 for _ in range(100))


def test_the_wait_before_a_re_run_pass_is_jittered():
    assert len({_rerun_wait(1) for _ in range(20)}) > 1


async def test_a_judge_that_failed_on_infrastructure_waits_before_its_re_judge(traces, waits):
    down = Fixed("Down", failures=1, error=EndpointConnectionError(endpoint_url="https://x"))

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [down]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert records[0].status is TrialStatus.COMPLETED
    assert records[0].evaluator_errors == {}
    assert spent == 1  # a re-judge costs no rollout
    assert len(waits) == 1 and 2.5 <= waits[0] <= 5.0


async def test_a_re_judge_of_a_bad_verdict_starts_immediately(traces, waits):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed("Flaky", failures=1)]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert records[0].evaluator_errors == {}
    assert waits == []


class Raises(Evaluator):
    """Raises the next of `errors` on each call; scores 1.0 once they run out."""

    def __init__(self, *errors, name="Raises"):
        super().__init__(name=name)
        self.errors = list(errors)

    async def evaluate_async(self, evaluation_case):
        if self.errors:
            raise self.errors.pop(0)
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]


@pytest.mark.parametrize(
    ("errors", "expected_waits"),
    [
        # an outage, then a bad verdict: only the re-judge after the outage waits
        ((EndpointConnectionError(endpoint_url="https://x"), ValueError("bad verdict")), 1),
        # throttled, then (in Experiment's retry) a bad verdict: the last error decides, so no wait
        ((ModelThrottledException("slow down"), ValueError("bad verdict")), 0),
    ],
)
async def test_only_a_judges_last_error_decides_the_wait(traces, waits, errors, expected_waits):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Raises(*errors)]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert records[0].evaluator_errors == {}
    assert len(waits) == expected_waits


async def test_a_judge_that_recovered_from_throttling_but_failed_to_aggregate_doesnt_wait(traces, waits):
    evaluator = Raises(ModelThrottledException("slow down"))
    evaluator.aggregator = lambda outputs: (math.nan, True, "")

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [evaluator], rerun_passes=1).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert records[0].evaluator_errors == {"Raises": "ValueError"}  # re-judged, and failed to aggregate again
    assert waits == []


async def test_no_wait_when_the_allowance_leaves_only_bad_verdicts_to_re_judge(traces, waits):
    async def task(case):
        if case.name == "a":
            raise InfrastructureError("sandbox down")
        return "ok"

    records, spent = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed(failures=1)]).run(
        "cand", task, ["a", "b"], runs=1, allowance=2
    )

    assert spent == 2  # no rollout left to re-run "a"
    assert [r.status for r in records] == [TrialStatus.INFRASTRUCTURE_INVALID, TrialStatus.COMPLETED]
    assert records[1].evaluator_errors == {}
    assert records[1].rollouts == 0  # "b" was re-judged
    assert waits == []


async def test_the_wait_counts_from_when_the_trial_failed(traces, clock):
    started = []

    async def task(case):
        started.append((case.name, clock[0]))
        if case.name == "a" and len(started) == 1:
            raise InfrastructureError("sandbox down")
        clock[0] += 10.0  # "b" takes longer than any first wait
        return "ok"

    records, _ = await _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed()], max_workers=1).run(
        "cand", task, ["a", "b"], runs=1, allowance=4
    )

    assert [r.status for r in records] == [TrialStatus.COMPLETED] * 2
    assert started == [("a", 0.0), ("b", 0.0), ("a", 10.0)]  # "a" had already waited 10 s: no more wait


async def test_a_wait_holds_no_slot_and_delays_no_other_trial(traces, monkeypatch):
    rejudged = asyncio.Event()

    async def sleep(delay):  # ends only once "b" has been re-judged
        await asyncio.wait_for(rejudged.wait(), timeout=5)

    monkeypatch.setattr("strands_evals.optimization.execution.trials._sleep", sleep)

    class Flaky(Fixed):
        async def evaluate_async(self, evaluation_case):
            if self.calls >= 1:  # the first call, "b"'s in the first pass, fails
                rejudged.set()
            return await super().evaluate_async(evaluation_case)

    async def task(case):
        if case.name == "a" and not rejudged.is_set():
            raise InfrastructureError("sandbox down")
        return "ok"

    records, spent = await _runner(traces, MemoryCache(), _cases("a", "b"), [Flaky(failures=1)], max_workers=1).run(
        "cand", task, ["a", "b"], runs=1, allowance=4
    )

    assert [r.status for r in records] == [TrialStatus.COMPLETED] * 2
    assert [r.evaluator_errors for r in records] == [{}, {}]
    assert spent == 3  # "a" twice, "b" once and re-judged for no rollout


async def test_a_failed_judge_is_re_judged_for_no_rollout(traces):
    flaky = Fixed("Flaky", score=0.6, failures=1)
    calls = []

    records, spent = await _runner(traces, MemoryCache(), _cases("a"), [Fixed("A"), flaky]).run(
        "cand", _agent_task(calls=calls), ["a"], runs=1, allowance=0
    )

    record = records[0]
    assert calls == ["a"]  # the agent ran once
    assert spent == 1
    assert record.status is TrialStatus.COMPLETED
    assert record.evaluations["Flaky"].score == 0.6
    assert record.evaluations["A"].score == 1.0  # kept from the first attempt
    assert record.evaluator_errors == {}
    assert record.rollouts == 0
    assert record.usage.tokens == 15  # the agent's usage stays with its output
    replaced = record.previous_attempts[-1]
    assert replaced.rollouts == 1
    assert replaced.usage.tokens == 0
    assert record.rollouts + sum(a.rollouts for a in record.previous_attempts) == spent


async def test_a_re_judge_keeps_each_cost_with_what_it_bought(traces):
    records, _ = await _runner(
        traces, MemoryCache(), _cases("a"), [Judge("Kept", tokens=500), Judge("Flaky", tokens=400, failures=1)]
    ).run("cand", _agent_task(), ["a"], runs=1, allowance=0)

    record = records[0]
    assert record.judge_usage["Kept"].tokens == 500  # carried over with its score
    assert record.judge_usage["Flaky"].tokens == 400  # the re-judge's own spend
    assert record.previous_attempts[-1].judge_usage.tokens == 400  # only the failed judge's spend: retry cost
    assert record.usage.tokens == 15


def _conversation_task(fail_first=None):
    """`make_task` on a factory agent answering a simulated user built in the rollout: on attempt n, the agent
    spends 15·n tokens and the user 1,000·n; the first attempt raises `fail_first` after both have spoken."""
    tries = []

    def factory():
        n = len(tries)
        agent_usage = {"inputTokens": 10 * n, "outputTokens": 5 * n, "totalTokens": 15 * n}
        return Agent(model=ScriptedModel(["answer"], usage=agent_usage), callback_handler=None)

    async def rollout(agent_factory, case):
        tries.append(case.name)
        n = len(tries)
        usage = {"inputTokens": 980 * n, "outputTokens": 20 * n, "totalTokens": 1000 * n}
        user = Agent(model=ScriptedModel(["thanks"], usage=usage), callback_handler=None)
        reply = await user.invoke_async(str(await agent_factory().invoke_async(case.input)))
        if fail_first is not None and len(tries) == 1:
            raise fail_first
        return {"output": str(reply)}

    return make_task(factory, rollout)


async def test_the_benchmarks_agents_are_recorded_apart_from_the_candidates(traces):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", _conversation_task(), ["a"], runs=1, allowance=0
    )

    assert (records[0].usage.tokens, records[0].benchmark_usage.tokens) == (15, 1000)


@pytest.mark.parametrize(
    "error, allowance",
    [(ModelThrottledException("slow down"), 0), (InfrastructureError("sandbox down"), 2)],
    ids=["retried-by-experiment", "re-run-pass"],
)
async def test_an_earlier_attempt_keeps_its_benchmark_usage(traces, error, allowance):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run(
        "cand", _conversation_task(fail_first=error), ["a"], runs=1, allowance=allowance
    )

    (earlier,) = records[0].previous_attempts
    assert (earlier.usage.tokens, earlier.benchmark_usage.tokens) == (15, 1000)
    assert (records[0].usage.tokens, records[0].benchmark_usage.tokens) == (30, 2000)


async def test_a_re_judge_keeps_the_benchmark_usage_with_the_output(traces):
    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [Fixed("Flaky", failures=1)]).run(
        "cand", _conversation_task(), ["a"], runs=1, allowance=0
    )

    assert records[0].benchmark_usage.tokens == 1000
    assert records[0].previous_attempts[-1].benchmark_usage.tokens == 0


# rejudge


async def test_a_cached_trial_with_a_failed_evaluator_is_final_without_rejudge(traces):
    cache = MemoryCache()
    broken = Fixed("Broken", failures=1)
    await _runner(traces, cache, _cases("a"), [broken], rerun_passes=0).run("cand", _plain, ["a"], runs=1, allowance=0)

    records, spent = await _runner(traces, cache, _cases("a"), [broken]).run("cand", _plain, ["a"], runs=1, allowance=0)

    assert spent == 0
    assert broken.calls == 1
    assert records[0].status is TrialStatus.EVALUATOR_INVALID


async def test_rejudge_re_judges_a_cached_trials_failed_evaluators(traces):
    cache = MemoryCache()
    ok, broken = Fixed("Ok"), Fixed("Broken", score=0.3, failures=1)
    await _runner(traces, cache, _cases("a"), [ok, broken], rerun_passes=0).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    records, spent = await _runner(traces, cache, _cases("a"), [ok, broken]).run(
        "cand", _plain, ["a"], runs=1, allowance=0, rejudge=True
    )

    assert spent == 0
    assert ok.calls == 1  # an evaluator that succeeded isn't re-run
    assert broken.calls == 2
    assert records[0].evaluations["Broken"].score == 0.3
    assert records[0].evaluations["Broken"].reason == "saw answer to a"  # judged on the saved output


# releasing spans


def _kept(traces):
    """Trace ids whose spans the capture still holds."""
    return set(traces._filter._spans)


async def test_every_trace_is_released_after_scoring_and_re_judging(traces):
    before = _kept(traces)
    runner = _runner(traces, MemoryCache(), _cases("a", "b"), [Fixed("A"), Judge("Flaky", failures=1)])

    await runner.run("cand", _agent_task(), ["a", "b"], runs=2, allowance=0)

    assert _kept(traces) == before


async def test_a_setup_error_releases_its_trace(traces):
    before = _kept(traces)

    async def task(case):
        raise _FactoryArgumentError("resource 'prompt' is named like a declared marker")

    with pytest.raises(TypeError):
        await _runner(traces, MemoryCache(), _cases("a"), [Fixed()]).run("cand", task, ["a"], runs=1, allowance=0)

    assert _kept(traces) == before


async def test_a_cancelled_trial_saves_nothing_and_keeps_no_spans(traces):
    before = _kept(traces)
    started = asyncio.Event()
    cache = MemoryCache()

    async def task(case):
        started.set()
        await asyncio.sleep(0.2)
        return "ok"

    call = asyncio.create_task(
        _runner(traces, cache, _cases("a"), [Fixed()]).run("cand", task, ["a"], runs=1, allowance=0)
    )
    await started.wait()
    assert _kept(traces) != before  # the attempt registered its trace
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    # Long enough for a trial left running to finish and be judged: until Experiment cancels its own
    # workers, it does, and its judge must not register the released trace again.
    await asyncio.sleep(0.3)

    assert cache.saved == []
    assert _kept(traces) == before


async def test_a_trial_that_fails_while_its_record_is_built_releases_its_trace(traces, monkeypatch):
    before = _kept(traces)
    runner = _runner(traces, MemoryCache(), _cases("a"), [Fixed()])

    def broken(*args, **kwargs):
        raise RuntimeError("record building failed")

    monkeypatch.setattr(runner, "_ran", broken)

    with pytest.raises(RuntimeError, match="record building failed"):
        await runner.run("cand", _plain, ["a"], runs=1, allowance=0)

    assert _kept(traces) == before


# the evaluator proxy


def test_the_proxy_is_the_evaluator_to_experiment(traces):
    evaluator = Fixed("Mine")
    evaluator._score_mapping = {"good": 1.0}
    proxy = _EvaluatorProxy(evaluator, traces, _TrialState())

    assert proxy.get_name() == "Mine"
    assert proxy.get_type_name() == "Fixed"
    outputs = [EvaluationOutput(score=0.5, test_pass=True, reason="ok")]
    assert proxy.aggregator(outputs) == evaluator.aggregator(outputs)  # wrapped, only to check the score
    assert proxy._score_mapping == {"good": 1.0}
    assert proxy.evaluation_level is None
    assert copy.copy(proxy).get_name() == "Mine"


async def test_labels_come_from_the_evaluators_own_mapping(traces):
    evaluator = Fixed("Mapped", score=1.0)
    evaluator._score_mapping = {"Very helpful": 1.0}

    records, _ = await _runner(traces, MemoryCache(), _cases("a"), [evaluator]).run(
        "cand", _plain, ["a"], runs=1, allowance=0
    )

    assert records[0].evaluations["Mapped"].label == "Very helpful"


# construction


def test_cases_need_unique_names_and_a_split(traces):
    with pytest.raises(ValueError, match="two cases"):
        _runner(traces, MemoryCache(), _cases("a", "a"), [])
    with pytest.raises(ValueError, match="no split"):
        TrialRunner(_cases("a"), [], traces, MemoryCache(), split_of={}, max_workers=1, rerun_passes=0)
    with pytest.raises(ValueError, match="max_workers"):
        _runner(traces, MemoryCache(), _cases("a"), [], max_workers=0)


def test_evaluators_need_unique_names(traces):
    with pytest.raises(ValueError, match=r"evaluator names must be unique, got \['Fixed'\]"):
        _runner(traces, MemoryCache(), _cases("a"), [Fixed(), Fixed(score=0.2)])
    _runner(traces, MemoryCache(), _cases("a"), [Fixed(), Fixed(name="Other")])  # name= tells them apart


async def test_an_unknown_case_name_is_an_error(traces):
    with pytest.raises(ValueError, match="unknown case names"):
        await _runner(traces, MemoryCache(), _cases("a"), []).run("cand", _plain, ["z"], runs=1, allowance=0)


async def test_a_case_name_listed_twice_is_an_error(traces):
    cache = MemoryCache()
    with pytest.raises(ValueError, match=r"listed more than once: \['a'\]"):
        await _runner(traces, cache, _cases("a", "b"), []).run("cand", _plain, ["a", "b", "a"], runs=1, allowance=0)
    assert cache.saved == []
