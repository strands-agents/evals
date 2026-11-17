import asyncio
import json
from collections import Counter

import pytest

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.experiment import Experiment
from strands_evals.optimization import (
    Block,
    Budget,
    Joint,
    OneOf,
    OptimizationResult,
    Optimizer,
    Patch,
    Proposer,
    RotationResult,
    Search,
    Split,
    pool_tests,
    rotate,
    rotate_async,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

SPLIT = Split(train=0.5, validation=0.25, test=0.25)
HALVES = Split(train=0.25, validation=0.25, test=0.5)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("strands_evals.experiment._INITIAL_RETRY_DELAY", 0)


class Temperature(Evaluator):
    """Scores the rollout's output, a number in 0 to 1."""

    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=float(evaluation_case.actual_output), test_pass=True, reason="t")]


@tunable(temperature=OneOf(0.0, 0.5, 1.0))
def create_agent(temperature):
    return temperature


class Scripted(Proposer):
    """Proposes the given updates in order, one per slot."""

    kinds = frozenset({"int", "float", "one_of", "choice", "text"})

    def __init__(self, *updates):
        self.updates = list(updates)

    async def propose(self, request):
        return [Patch(updates=self.updates.pop(0), rationale="next") for _ in range(min(request.n, len(self.updates)))]

    def observe(self, observation):
        pass


def _experiment(n=20):
    return Experiment(cases=[Case(name=f"case{i:02d}", input=f"q{i}") for i in range(n)], evaluators=[Temperature()])


def _rollout(fail_on=(), active=None, fold=None, blocked=None):
    """The agent's temperature times a per-case weight, so differences vary by case; raises on `fail_on` cases.

    With `active`, counts the rollouts in flight (`now`), the most at once (`peak`), and whether two folds
    (each rollout tagged `fold`) had rollouts in flight at once (`together`). With `blocked`, waits on that
    event first."""

    async def rollout(agent_factory, case):
        temperature = agent_factory()
        if case.name in fail_on:
            raise ValueError("crashes on this case")
        if blocked is not None:
            await blocked.wait()
        if active is not None:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            active["folds"][fold] += 1
            active["together"] = active["together"] or sum(n > 0 for n in active["folds"].values()) > 1
            await asyncio.sleep(0.02)
            active["folds"][fold] -= 1
            active["now"] -= 1
        return str(temperature * (0.5 + 0.25 * (int(case.name[-2:]) % 3)))

    return rollout


def _fold(experiment, fold, base_dir, *, search=None, rollout=None, **kwargs):
    """An optimizer that tries temperature 0.5, then 1.0, on one fold."""
    if search is None:
        search = Search([Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature")], schedule=Joint())
    kwargs.setdefault("max_workers", 64)
    return Optimizer(
        experiment,
        create_agent,
        rollout_fn=rollout if rollout is not None else _rollout(),
        search=search,
        split=fold,
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        base_dir=base_dir,
        **kwargs,
    )


def _resume(run_dir, experiment, proposer=None):
    """`Optimizer.resume` of a `_fold` run."""
    search = Search([Block(proposer if proposer is not None else Scripted(), "temperature")], schedule=Joint())
    return Optimizer.resume(run_dir, experiment=experiment, factory=create_agent, rollout_fn=_rollout(), search=search)


async def test_rotate_tests_every_case_once_and_pools_the_folds_reports(tmp_path):
    experiment = _experiment()
    folds = SPLIT.folds(experiment.cases)
    optimizers = [_fold(experiment, fold, tmp_path) for fold in folds]
    rotation = await rotate_async(optimizers)

    assert isinstance(rotation, RotationResult) and rotation.test_reason is None
    assert len(rotation.folds) == 4 and all(isinstance(r, OptimizationResult) for r in rotation.folds)
    assert len({r.run_dir for r in rotation.folds}) == 4
    tested = [case for r in rotation.folds for case in r.test.difference_by_case["Temperature"]]
    assert sorted(tested) == [case.name for case in experiment.cases]
    for result, fold in zip(rotation.folds, folds, strict=True):
        assert set(result.test.objective_difference_by_case) == {case.name for case in fold.test}
        assert result.best.values["temperature"] == 1.0

    sizes = [len(fold.train) + len(fold.validation) for fold in folds]
    assert rotation.test == pool_tests([r.test for r in rotation.folds], train_sizes=sizes)
    pooled = rotation.test.difference["Temperature"]
    assert pooled.n_cases == 20 and pooled.mean_diff > 0
    assert rotation.test.procedure["Temperature"].n_cases == 20
    assert all(o._limiter is None for o in optimizers)


async def test_resumed_folds_rotate_again_to_the_same_pool(tmp_path):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    first = await rotate_async([_fold(experiment, fold, tmp_path) for fold in folds])
    again = await rotate_async([_resume(result.run_dir, experiment) for result in first.folds])
    assert again.test == first.test
    assert [r.run_dir for r in again.folds] == [r.run_dir for r in first.folds]


def test_rotate_runs_from_sync_code(tmp_path):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    rotation = rotate([_fold(experiment, fold, tmp_path) for fold in folds])
    assert rotation.test.n_folds == 2 and rotation.test.difference["Temperature"].n_cases == 10


async def test_the_folds_share_one_max_workers_cap(tmp_path):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    active = {"now": 0, "peak": 0, "folds": Counter(), "together": False}
    optimizers = [
        _fold(experiment, fold, tmp_path, rollout=_rollout(active=active, fold=i), max_workers=2)
        for i, fold in enumerate(folds)
    ]
    await rotate_async(optimizers)
    # Each fold alone may run 2 trials at once; running at the same time, together they still run at most 2.
    assert active["peak"] == 2
    assert active["together"]


async def test_a_fold_without_a_test_report_leaves_the_pool_out(tmp_path):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    failing = {case.name for case in folds[1].test}
    optimizers = [
        _fold(experiment, folds[0], tmp_path),
        _fold(experiment, folds[1], tmp_path, rollout=_rollout(fail_on=failing)),
    ]
    rotation = await rotate_async(optimizers)
    assert rotation.test is None
    assert rotation.test_reason == (
        "fold 2 has no test report: every test trial of the baseline failed, so there is nothing to compare"
    )


async def test_a_saved_report_without_per_case_differences_leaves_the_pool_out(tmp_path):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    first = await rotate_async([_fold(experiment, fold, tmp_path) for fold in folds])
    # Fold 1's test event as a run saved it before the per-case differences were recorded.
    events = first.folds[0].run_dir / "events.jsonl"
    lines = [json.loads(line) for line in events.read_text().splitlines()]
    for event in lines:
        if event["event"] == "test":
            del event["report"]["objective_difference_by_case"], event["report"]["difference_by_case"]
    events.write_text("".join(json.dumps(event) + "\n" for event in lines))
    again = await rotate_async([_resume(result.run_dir, experiment) for result in first.folds])
    assert again.test is None and len(again.folds) == 2
    assert again.test_reason == "pool_tests: report 1 has no per-case differences for the objective, Temperature"


async def test_a_fold_that_raises_cancels_the_others_which_resume_and_rotate_again(tmp_path, caplog):
    experiment = _experiment(n=10)
    folds = HALVES.folds(experiment.cases)
    every_case = {case.name for case in experiment.cases}
    optimizers = [
        _fold(experiment, folds[0], tmp_path, rollout=_rollout(blocked=asyncio.Event())),
        _fold(experiment, folds[1], tmp_path, rollout=_rollout(fail_on=every_case)),
    ]
    cancelled = []
    run_async = optimizers[0].run_async

    async def run_and_record():
        try:
            return await run_async()
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    optimizers[0].run_async = run_and_record
    with pytest.raises(RuntimeError, match="every train trial of the baseline failed"):
        await rotate_async(optimizers)
    assert cancelled == [1]  # fold 1, still waiting in its baseline's trials, was cancelled before the raise
    assert "fold=<2>" in caplog.text
    run_dir = optimizers[0]._run.root
    assert str(run_dir) in caplog.text and str(optimizers[1]._run.root) in caplog.text
    assert not (run_dir / "result.json").exists()
    assert all(o._limiter is None for o in optimizers)

    # The cancelled fold resumes; the one whose setup failed starts again on its fold.
    resumed = _resume(run_dir, experiment, Scripted({"temperature": 0.5}, {"temperature": 1.0}))
    rotation = await rotate_async([resumed, _fold(experiment, folds[1], tmp_path)])
    assert rotation.folds[0].run_dir == run_dir
    assert rotation.test.difference["Temperature"].n_cases == 10


def test_rotate_rejects_optimizers_that_arent_folds_of_one_rotation(tmp_path):
    experiment = _experiment()
    folds = SPLIT.folds(experiment.cases)
    with pytest.raises(ValueError, match="one optimizer per fold, 2 or more; got 1"):
        rotate([_fold(experiment, folds[0], tmp_path)])
    with pytest.raises(ValueError, match=r"same; got \[4, 64\]"):
        rotate(
            [_fold(experiment, folds[0], tmp_path, max_workers=4), *(_fold(experiment, f, tmp_path) for f in folds[1:])]
        )
    with pytest.raises(ValueError, match=r"tested by more than one: \[.+\], by none: \[.+\]"):
        rotate([_fold(experiment, folds[0], tmp_path) for _ in folds])
    other = _experiment(n=24)
    with pytest.raises(ValueError, match="split different cases"):
        rotate([_fold(experiment, folds[0], tmp_path), _fold(other, SPLIT.folds(other.cases)[1], tmp_path)])
    search = Search([Block(Scripted({"temperature": 1.0}), "temperature")])
    with pytest.raises(ValueError, match="folds 1 and 2 share one Scripted; .* give each fold its own Search"):
        rotate([_fold(experiment, fold, tmp_path, search=search) for fold in folds])

    @tunable(temperature=OneOf(0.5, 0.0, 1.0))
    def create_warmer_agent(temperature):
        return temperature

    warmer = Optimizer(experiment, create_warmer_agent, split=folds[1], base_dir=tmp_path, max_workers=64)
    with pytest.raises(ValueError, match="agents have different defaults"):
        rotate([_fold(experiment, folds[0], tmp_path), warmer, *(_fold(experiment, f, tmp_path) for f in folds[2:])])
