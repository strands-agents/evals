import asyncio
import dataclasses
import hashlib
import itertools
import json
import math
import pickle
import shutil
import threading
import warnings
from copy import deepcopy
from pathlib import Path

import botocore.exceptions
import pytest
from strands import Agent

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.experiment import Experiment
from strands_evals.optimization import (
    TPE,
    Alternate,
    AtLeast,
    Block,
    Budget,
    Evidence,
    FileText,
    FinalPick,
    Float,
    InfrastructureError,
    Int,
    Joint,
    JsonField,
    Maximize,
    Minimize,
    Nested,
    OneOf,
    OptimizationResult,
    Optimizer,
    Pareto,
    Patch,
    Proposer,
    ProposerError,
    Schedule,
    ScoreModel,
    Search,
    Split,
    Step,
    TextValue,
    Usage,
    WeightedSum,
    choice,
    component,
    tunable,
)
from strands_evals.optimization.declaration.space import Space
from strands_evals.optimization.optimizer import _same_name, _slots_used
from strands_evals.optimization.record.resume import load_state
from strands_evals.optimization.record.run_directory import RunDirectory
from strands_evals.optimization.result import _reports, _run_cost
from strands_evals.optimization.search.blocks import _saved
from strands_evals.optimization.selection.top_ranked import TopRanked
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.optimization import CaseSplit, Estimate, TrialRecord, TrialStatus
from strands_evals.types.trace import InferenceSpan
from tests.strands_evals.optimization.scripted_model import ScriptedModel


class Score(Evaluator):
    def __init__(self, name="Score", score_range=None):
        super().__init__(name=name)
        if score_range is not None:
            self.score_range = score_range

    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]


class Other(TopRanked):
    name = "other"


class Window:
    def __init__(self, size=10):
        self.size = size


class Summary:
    def __init__(self, ratio=0.5):
        self.ratio = ratio


@tunable(
    manager=choice(component(Window, size=Int(1, 100)), component(Summary)),
    temperature=OneOf(0.0, 0.5, 1.0),
)
def create_agent(manager, temperature):
    return (manager, temperature)


def _experiment(n=20, evaluators=None):
    cases = [Case(name=f"case{i:02d}", input=f"q{i}") for i in range(n)]
    return Experiment(cases=cases, evaluators=[Score()] if evaluators is None else evaluators)


def _optimizer(experiment=None, **kwargs):
    return Optimizer(_experiment() if experiment is None else experiment, create_agent, **kwargs)


def test_defaults_build():
    optimizer = _optimizer()
    config = optimizer._config()
    assert config["objective"] == "Maximize(Score)"
    assert config["selection"] == {"name": "top_ranked", "config": {}}
    assert config["score_model"]["name"] == "bayesbench"
    assert config["final_pick"] == {"n_finalists": 3, "rule": {"kind": "Mean"}}
    assert config["budget"] == {"max_rollouts": 500, "max_iterations": None}
    assert config["split"]["seed"] == 0
    assert {role: len(names) for role, names in config["split"]["cases"].items()} == {
        "train": 8,
        "validation": 6,
        "test": 6,
    }
    assert config["cases"] == sorted(config["cases"]) and len(config["cases"]) == 20


def test_argument_combinations_are_type_errors():
    experiment = _experiment()
    with pytest.raises(TypeError, match="not both or neither"):
        Optimizer(experiment)
    with pytest.raises(TypeError, match="not both or neither"):
        Optimizer(experiment, create_agent, build_task=lambda ctx: None, space={"x": Int(1, 3)})
    with pytest.raises(TypeError, match="needs space="):
        Optimizer(experiment, build_task=lambda ctx: None)
    with pytest.raises(TypeError, match="goes with build_task"):
        Optimizer(experiment, create_agent, space={"x": Int(1, 3)})
    for argument in ("rollout_fn", "environment"):
        with pytest.raises(TypeError, match=f"{argument}= works only with factory="):
            Optimizer(experiment, build_task=lambda ctx: None, space={"x": Int(1, 3)}, **{argument: lambda *a: None})
    with pytest.raises(TypeError, match="TunableFactory"):
        Optimizer(experiment, lambda: None)
    with pytest.raises(TypeError, match="selection must be a Selection"):
        Optimizer(experiment, create_agent, selection="top_ranked")


def test_build_task_builds_its_space_at_construction():
    with pytest.raises(TypeError):
        Optimizer(_experiment(), build_task=lambda ctx: None, space={"x": Int(5, 1)})
    optimizer = Optimizer(_experiment(), build_task=lambda ctx: None, space={"x": Int(1, 3, default=2)})
    assert [p["key"] for p in optimizer._config()["space"]] == ["x"]
    assert optimizer._config()["agent"]["factory"] is None
    assert optimizer._config()["agent"]["build_task"].endswith(
        ":test_build_task_builds_its_space_at_construction.<locals>.<lambda>"
    )


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("runs_per_candidate", 0, ValueError),
        ("proposals_per_iteration", 0, ValueError),
        ("max_workers", 0, ValueError),
        ("rerun_passes", -1, ValueError),
        ("runs_per_candidate", 1.0, TypeError),
        ("seed", True, TypeError),
    ],
)
def test_settings_are_checked(argument, value, error):
    with pytest.raises(error, match=argument):
        _optimizer(**{argument: value})


def test_cases_need_unique_names():
    unnamed = Experiment(cases=[Case(name="a", input="q"), Case(input="q")], evaluators=[Score()])
    with pytest.raises(ValueError, match=r"cases \[1\] have none"):
        _optimizer(unnamed)


def test_evaluators_need_unique_names(tmp_path):
    with pytest.raises(ValueError, match=r"evaluator names must be unique; repeated: \['Score'\]"):
        _optimizer(_experiment(evaluators=[Score(), Score()]), base_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []  # rejected before any run folder exists
    _optimizer(_experiment(evaluators=[Score(), Score(name="Score2")]))


def test_no_evaluators_and_no_objective():
    with pytest.raises(ValueError, match="nothing to optimize"):
        _optimizer(_experiment(evaluators=[]))
    # A usage-only objective needs no evaluator; it warns, since "worst" scores only evaluators.
    with pytest.warns(UserWarning, match="every goal is a usage metric"):
        _optimizer(_experiment(evaluators=[]), objective=Minimize("tokens"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _optimizer(objective=Minimize("tokens"), constraints=[AtLeast("Score", 0.5)])
        _optimizer(_experiment(evaluators=[]), objective=Minimize("tokens"), agent_failure_score=None)


def test_bad_name():
    with pytest.raises(ValueError, match="lowercase"):
        _optimizer(name="My Run")


def test_split_shortfall():
    with pytest.raises(ValueError, match="0 validation cases"):
        _optimizer(_experiment(n=1))


def test_budget_must_cover_the_baseline_and_the_reserves():
    # 20 cases: 8 train, 6 validation, 6 test. Each end-phase arm reserves its first pass plus 2 re-run passes
    # x 1 re-run: baseline 8 + validation (3 + 1) x (6 + 2) + test 2 x (6 + 2) = 56.
    with pytest.raises(ValueError, match="re-run headroom included\\): give max_rollouts >= 56"):
        _optimizer(budget=Budget(max_rollouts=55))
    with pytest.warns(UserWarning, match=r"less than one iteration \(32 rollouts\)"):
        _optimizer(budget=Budget(max_rollouts=56))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _optimizer(budget=Budget(max_rollouts=88))
    # Twice the runs: 16 + 4 x (12 + 2) + 2 x (12 + 2) = 100.
    with pytest.raises(ValueError, match="max_rollouts >= 100"):
        _optimizer(budget=Budget(max_rollouts=99), runs_per_candidate=2)
    # No re-run passes: no headroom, the first passes only.
    with pytest.raises(ValueError, match="max_rollouts >= 44"):
        _optimizer(budget=Budget(max_rollouts=43), rerun_passes=0)
    # No test cases: no test reserve.
    optimizer = _optimizer(split=Split(train=0.5, validation=0.5, test=None))
    assert (optimizer._budget.validation_reserve, optimizer._budget.test_reserve) == (4 * (10 + 2), 0)


def test_pareto_reserves_every_front_member():
    objective = Pareto(Maximize("Score"), Minimize("tokens", scale=1000))
    optimizer = _optimizer(objective=objective)
    # max(3, 5) finalists + the baseline, on 6 validation and 6 test cases, each with 2 re-runs of headroom.
    assert (optimizer._budget.validation_reserve, optimizer._budget.test_reserve) == (48, 48)


def test_metrics_record_roles_ranges_and_effective_scale():
    experiment = _experiment(evaluators=[Score(), Score(name="Stars", score_range=(1, 5))])
    objective = WeightedSum(Maximize("Score", weight=2), Maximize("Stars"), Minimize("tokens", scale=1000))
    config = _optimizer(experiment, objective=objective, constraints=[AtLeast("Stars", 3)])._config()
    assert config["metrics"] == {
        "Score": {"kind": "evaluator", "roles": ["objective"], "range": [0.0, 1.0], "weight": 2.0, "scale": 1.0},
        "Stars": {
            "kind": "evaluator",
            "roles": ["objective", "constraint"],
            "range": [1, 5],
            "weight": 1.0,
            "scale": 4,
        },
        "tokens": {"kind": "usage", "roles": ["objective"], "range": [0.0, None], "weight": 1.0, "scale": 1000.0},
    }
    assert config["constraints"] == ["Stars >= 3"]


def test_space_stores_text_as_a_hash():
    @tunable(prompt=TextValue(default="be helpful", edit_instructions="keep it short"), k=Int(1, 9, default=3))
    def agent(prompt, k):
        return prompt

    config = Optimizer(_experiment(), agent)._config()  # the default text block is Reflection
    space = {p["key"]: p for p in config["space"]}
    assert space["prompt"]["default"] == {"sha256": hashlib.sha256(b"be helpful").hexdigest(), "length": 10}
    assert "be helpful" not in json.dumps(space)
    assert space["prompt"]["edit_instructions"]["length"] == len("keep it short")
    assert space["k"]["default"] == 3 and space["k"]["min_value"] == 1


def test_config_round_trips_through_resume(tmp_path):
    run = RunDirectory.create(tmp_path, _optimizer()._config())
    reopened = RunDirectory.open(run.root)
    assert set(run.config["code"]) == {"version", "commit", "dirty", "numpy", "optuna"}
    load_state(reopened, config=_optimizer()._config())  # same settings: nothing differs
    with pytest.raises(ValueError, match="^objective changed"):
        load_state(reopened, config=_optimizer(objective=Minimize("Score"))._config())
    with pytest.raises(ValueError, match=r"^selection\.name changed: 'top_ranked' -> 'other'"):
        load_state(reopened, config=_optimizer(selection=Other())._config())


def test_reordered_cases_still_resume(tmp_path):
    experiment = _experiment()
    run = RunDirectory.create(tmp_path, _optimizer(experiment)._config())
    reordered = Experiment(cases=list(reversed(experiment.cases)), evaluators=[Score()])
    load_state(RunDirectory.open(run.root), config=_optimizer(reordered)._config())


def test_final_pick_and_split_settings_are_stored():
    config = _optimizer(final_pick=FinalPick(n_finalists=2), split=Split(seed=7), seed=3)._config()
    assert config["final_pick"]["n_finalists"] == 2
    assert config["split"]["seed"] == 7 and config["seed"] == 3
    assert (config["split"]["train"], config["split"]["validation"], config["split"]["test"]) == (0.4, 0.3, 0.3)
    assert config["agent"]["factory"].endswith(":create_agent")
    assert config["agent"]["rollout_fn"] is None


# the run's setup and the baseline


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("strands_evals.experiment._INITIAL_RETRY_DELAY", 0)


class NotApplicable(Evaluator):
    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=0.0, test_pass=True, reason="nothing to judge", label=NOT_APPLICABLE)]


def _rollout(fail=None):
    async def rollout(agent_factory, case):
        agent_factory()
        if fail is not None:
            raise fail
        return f"answer to {case.name}"

    return rollout


def _quiet(**kwargs):
    """An optimizer whose step fits in max_workers, so setup doesn't warn about batching."""
    return _optimizer(max_workers=64, **kwargs)


def _events(optimizer):
    return [json.loads(line) for line in (optimizer._run.root / "events.jsonl").read_text().splitlines()]


async def test_run_dir_is_set_before_the_baseline_is_scored(tmp_path):
    seen = []
    optimizer = _quiet(base_dir=tmp_path, rollout_fn=_rollout(), budget=Budget(max_iterations=1))
    optimizer._on_iteration = lambda record: seen.append(optimizer.run_dir)
    assert optimizer.run_dir is None
    task = asyncio.ensure_future(optimizer.run_async())
    await asyncio.sleep(0)  # run_async creates the folder before its first await
    assert optimizer.run_dir is not None and optimizer.run_dir.parent == tmp_path
    assert not any(e["event"] == "trial" for e in _events(optimizer))  # nothing spent yet
    result = await task
    assert seen and set(seen) == {optimizer.run_dir} and result.run_dir == optimizer.run_dir
    resumed = Optimizer.resume(optimizer.run_dir, experiment=_experiment(), factory=create_agent, rollout_fn=_rollout())
    assert resumed.run_dir == optimizer.run_dir


def test_runs_go_under_optimization_runs_in_the_working_directory_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    optimizer = _quiet(rollout_fn=_rollout())
    optimizer._setup()
    assert optimizer.run_dir.parent == (tmp_path / "optimization" / "runs").resolve()


async def test_the_baseline_is_saved_and_scored_on_train(tmp_path):
    optimizer = _quiet(base_dir=tmp_path, rollout_fn=_rollout(), name="quick")
    optimizer._setup()
    scores = await optimizer._score_baseline()
    assert scores.is_eligible and scores.objective == 1.0
    assert sorted(scores.objective_by_case) == sorted(optimizer._case_split.train)
    assert optimizer._run.run_id.startswith("quick-")
    assert json.loads((optimizer._run.root / "run.json").read_text())["objective"] == "Maximize(Score)"
    events = _events(optimizer)
    assert [e["event"] for e in events] == ["config", "propose"] + ["trial"] * 8 + ["score"]
    assert events[1]["number"] == 0 and events[1]["origin"] == ["baseline"] and events[1]["parent_id"] is None
    assert events[-1]["rollouts"] == 8 and events[-1]["statuses"] == {"completed": 8}
    assert {e["record"]["split"] for e in events if e["event"] == "trial"} == {"train"}
    assert optimizer._budget.spent == 8


async def test_a_baseline_with_unmeasured_cases_stops_at_setup(tmp_path):
    optimizer = _quiet(base_dir=tmp_path, rollout_fn=_rollout(InfrastructureError("sandbox down")), rerun_passes=0)
    optimizer._setup()
    with pytest.raises(
        RuntimeError, match=r"^8/8 train cases unmeasured .*: trials 8 infrastructure_invalid; first error"
    ):
        await optimizer._score_baseline()
    stop = _events(optimizer)[-1]
    assert stop["event"] == "stop" and stop["reason"] == "setup" and "sandbox down" in stop["message"]


async def test_a_baseline_experiment_fails_after_its_task_returned_stops_at_setup(tmp_path):
    async def rollout(agent_factory, case):
        return {"output": {(0, 1): "x"}}  # Experiment can't serialize it for its span

    optimizer = _quiet(base_dir=tmp_path, rollout_fn=rollout)
    optimizer._setup()
    with pytest.raises(
        RuntimeError, match=r"^8/8 train cases unmeasured .*: trials 8 evaluator_invalid; first error .*: TypeError: "
    ):
        await optimizer._score_baseline()


async def test_a_crashing_baseline_scores_worst_and_is_eligible(tmp_path):
    async def rollout(agent_factory, case):
        if case.name != optimizer._case_split.train[0]:
            raise ValueError("agent bug")
        return "answer"

    optimizer = _quiet(base_dir=tmp_path, rollout_fn=rollout)
    optimizer._setup()
    scores = await optimizer._score_baseline()
    assert scores.is_eligible and scores.objective == 1 / 8 and scores.failures == 7


async def test_a_baseline_with_no_completed_trial_stops_at_setup(tmp_path):
    optimizer = _quiet(base_dir=tmp_path, rollout_fn=_rollout(ValueError("agent bug")))
    optimizer._setup()
    with pytest.raises(
        RuntimeError, match=r"^every train trial of the baseline failed.*: trials 8 agent_failure; first error"
    ):
        await optimizer._score_baseline()
    stop = _events(optimizer)[-1]
    assert stop["event"] == "stop" and stop["reason"] == "setup" and "agent bug" in stop["message"]


async def test_a_goal_metric_never_measured_stops_at_setup(tmp_path):
    experiment = _experiment(evaluators=[Score(), NotApplicable(name="Grounded")])
    optimizer = _quiet(
        experiment=experiment,
        base_dir=tmp_path,
        rollout_fn=_rollout(),
        objective=Maximize("Score"),
        constraints=[AtLeast("Grounded", 0.5)],
    )
    optimizer._setup()
    with pytest.raises(RuntimeError, match=r"goal metric\(s\) \['Grounded'\] have no value on any train case"):
        await optimizer._score_baseline()
    assert _events(optimizer)[-1]["reason"] == "setup"


async def test_build_task_gets_the_candidates_values(tmp_path):
    built = []

    def build_task(ctx):
        built.append(ctx)

        async def task(case):
            return f"k={ctx.resolve('k')}"

        return task

    optimizer = Optimizer(
        _experiment(), build_task=build_task, space={"k": Int(1, 9, default=3)}, base_dir=tmp_path, max_workers=64
    )
    optimizer._setup()
    await optimizer._score_baseline()
    assert len(built) == 1 and built[0].values == {"k": 3} and built[0].workspace is None
    trial = next(e for e in _events(optimizer) if e["event"] == "trial")
    assert trial["record"]["output"] == "k=3"


async def test_build_task_gets_a_workspace_for_any_file_backed_value(tmp_path):
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"prompt": "Be brief."}) + "\n")
    built = []

    def build_task(ctx):
        built.append(ctx)

        async def task(case):
            return json.loads(next(ctx.workspace.rglob("agent.json")).read_text())["prompt"]

        return task

    optimizer = Optimizer(
        _experiment(),
        build_task=build_task,
        space={"prompt": JsonField(config, "/prompt")},
        base_dir=tmp_path / "runs",
        max_workers=64,
    )
    optimizer._setup()
    await optimizer._score_baseline()
    assert built[0].workspace is not None
    trial = next(e for e in _events(optimizer) if e["event"] == "trial")
    assert trial["record"]["output"] == "Be brief."


async def test_build_task_copies_nothing_until_it_reads_the_workspace(tmp_path):
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"prompt": "Be brief."}) + "\n")

    def build_task(ctx):
        async def task(case):
            return ctx.resolve("prompt")

        return task

    optimizer = Optimizer(
        _experiment(),
        build_task=build_task,
        space={"prompt": JsonField(config, "/prompt")},
        base_dir=tmp_path / "runs",
        max_workers=64,
    )
    optimizer._setup()
    await optimizer._score_baseline()
    assert not (optimizer._run.root / "candidates").exists()
    trial = next(e for e in _events(optimizer) if e["event"] == "trial")
    assert trial["record"]["output"] == "Be brief."


def _spy_copies(monkeypatch, *, every_call=False):
    """Record the thread of every `create_candidate_copy` call that makes a copy (or of every call)."""
    threads = []
    real = Space.create_candidate_copy

    def spy(self, values, directory):
        if every_call or not self._has_copy(directory, self._copy_key(values)):
            threads.append(threading.get_ident())
        return real(self, values, directory)

    monkeypatch.setattr(Space, "create_candidate_copy", spy)
    return threads


async def test_a_path_backed_value_is_copied_off_the_event_loop(tmp_path, monkeypatch):
    prompt = tmp_path / "prompts" / "system.md"
    prompt.parent.mkdir()
    prompt.write_text("Be brief.\n")

    @tunable(prompt=FileText(prompt, as_path=True))
    def make(prompt):
        return prompt

    async def rollout(agent_factory, case):
        return Path(agent_factory()).read_text()

    threads = _spy_copies(monkeypatch)
    optimizer = Optimizer(_experiment(n=4), make, rollout_fn=rollout, base_dir=tmp_path / "runs", max_workers=8)
    optimizer._setup()
    await optimizer._score_baseline()
    # One copy, made in a worker thread: concurrent trials kept running on the loop.
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    outputs = {e["record"]["output"] for e in _events(optimizer) if e["event"] == "trial"}
    assert outputs == {"Be brief.\n"}


async def test_trials_reading_the_workspace_at_once_make_one_copy(tmp_path, monkeypatch):
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"prompt": "Be brief."}) + "\n")

    def build_task(ctx):
        def task(case):  # sync: the trials read the workspace at once, from worker threads
            return json.loads(next(ctx.workspace.rglob("agent.json")).read_text())["prompt"]

        return task

    threads = _spy_copies(monkeypatch, every_call=True)
    optimizer = Optimizer(
        _experiment(n=8),
        build_task=build_task,
        space={"prompt": JsonField(config, "/prompt")},
        base_dir=tmp_path / "runs",
        max_workers=8,
    )
    optimizer._setup()
    await optimizer._score_baseline()
    assert len(threads) == 1  # later reads don't even check for the copy
    outputs = {e["record"]["output"] for e in _events(optimizer) if e["event"] == "trial"}
    assert outputs == {"Be brief."}


def test_setup_warns_when_a_step_runs_in_batches(tmp_path):
    optimizer = _optimizer(base_dir=tmp_path)
    with pytest.warns(UserWarning, match=r"4 candidates x 8 train cases x 1 runs = 32 trials, but max_workers is 4"):
        optimizer._setup()


# the search loop


class Temperature(Evaluator):
    """Scores the agent's temperature (the rollout's output), so higher is better."""

    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=float(evaluation_case.actual_output), test_pass=True, reason="t")]


class Scripted(Proposer):
    """Proposes the given updates in order (an exception is raised when reached); records what it is told."""

    kinds = frozenset({"int", "float", "one_of", "choice", "text"})

    def __init__(self, *updates):
        self.updates = list(updates)
        self.requests, self.observed, self.evidenced = [], [], []

    async def propose(self, request):
        self.requests.append(request)
        patches = []
        for _ in range(request.n):
            if not self.updates:
                break
            update = self.updates.pop(0)
            if isinstance(update, Exception):
                raise update
            patches.append(Patch(updates=update, rationale=f"try {update}"))
        return patches

    def observe(self, observation):
        self.observed.append(observation)

    def evidence(self, evidence):
        self.evidenced.append(evidence)


async def _temperature_rollout(agent_factory, case):
    _, temperature = agent_factory()
    return str(temperature)


def _searching(*blocks, schedule=None, **kwargs):
    """Every key in a scripted block (the default TPE block can't propose yet): pass `manager` for the rest."""
    kwargs.setdefault("budget", Budget(max_iterations=3))
    named = {v for b in blocks for v in b.variables}
    blocks = [*blocks, *([Block(Scripted(), "manager")] if not named & {"manager"} else [])]
    if "temperature" not in named:
        blocks.append(Block(Scripted(), "temperature"))
    kwargs.setdefault("max_workers", 64)
    return _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=_temperature_rollout,
        search=Search(list(blocks), schedule=schedule),
        **kwargs,
    )


async def _search(optimizer):
    await optimizer._start()
    reason = await optimizer._run_search()
    optimizer._stop(reason)
    return reason, _events(optimizer)


def _names(events, start=0):
    return [e["event"] for e in events[start:] if e["event"] != "trial"]


async def test_setup_tells_every_block_the_baseline_as_evidence(tmp_path):
    structure, numbers = Scripted(), Scripted()
    optimizer = _searching(
        Block(structure, "manager"), Block(numbers, "temperature", "manager.Window.size"), base_dir=tmp_path
    )
    await optimizer._start()
    events = _events(optimizer)
    (label,) = [e["label"] for e in events if e["event"] == "label"]
    shared = [e for e in events if e["event"] == "evidence"]
    # After the baseline's fit and label, one `evidence` event per block, which resume replays.
    assert _names(events)[-4:] == ["fit", "label", "evidence", "evidence"]
    assert [(e["iteration"], e["block"]) for e in shared] == [(0, "structure"), (0, "numbers")]
    for proposer, event in zip((structure, numbers), shared, strict=True):
        (evidence,) = proposer.evidenced
        assert event["evidence"] == evidence.model_dump(mode="json")
        assert evidence.candidate_id == optimizer._baseline.id and evidence.origin == ["baseline"]
        assert evidence.outcome == label["outcome"] == "feasible"
        assert (evidence.objective, evidence.term_values, evidence.violations) == (0.0, [0.0], [])


async def test_tpe_counts_the_baseline_toward_its_startup(tmp_path):
    optimizer = _searching(
        Block(TPE(n_startup_trials=1), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    _, events = await _search(optimizer)
    # The first ask's study already holds the baseline. Only the first patch is checked: one repeating the
    # baseline is redrawn.
    first = next(e for e in events if e["event"] == "observe" and e["block"] == "numbers")
    assert first["observation"]["patch"]["rationale"] == "TPE: modeled on 1 told trials"


async def test_one_iteration_writes_its_events_in_order(tmp_path):
    proposer = Scripted({"temperature": 0.5})
    records = []
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        on_iteration=records.append,
        proposals_per_iteration=1,
    )
    reason, events = await _search(optimizer)
    assert reason == "iterations"
    first = next(i for i, e in enumerate(events) if e["event"] == "select")
    assert _names(events, first) == [
        "select",
        "draw",
        "propose",
        "score",
        "fit",
        "label",
        "observe",
        "evidence",
        "iteration_end",
        "stop",
    ]
    (observation,) = proposer.observed
    assert observation.outcome == "feasible" and observation.objective == 0.5 and observation.violations == []
    assert records[0].proposals[0].origin == ["numbers"] and records[0].rollouts_spent == 8
    assert optimizer._budget.spent == 16
    # The parent's train trials, train scores, and train inputs only (invariant 2).
    (request,) = proposer.requests
    assert {t.split for t in request.trials} == {"train"} and set(request.inputs) == set(optimizer._case_split.train)
    assert [c.number for c in request.candidates] == [0]
    assert request.candidates[0].term_values == [0.0]  # the baseline's temperature, by measured means


async def test_two_slots_of_one_step_with_equal_values_are_scored_once(tmp_path):
    proposer = Scripted({"temperature": 0.5}, {"temperature": 0.5})
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Joint(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=2,
    )
    await _search(optimizer)
    # Both slots are accepted (neither repeats a measured candidate), but the twin is scored once.
    assert [o.outcome for o in proposer.observed] == ["feasible", "duplicate"]
    assert proposer.observed[1].reason == "repeats a candidate of the same step"
    assert optimizer._budget.spent == 16  # the baseline and one candidate, not two
    proposals = optimizer._iterations[0].proposals
    assert len({p.candidate_id for p in proposals}) == 1 and len(proposals) == 2


async def test_a_step_is_cut_to_what_the_budget_can_score(tmp_path):
    # 56 needed (baseline 8 + reserves 48), so 64 leaves room for one candidate of 8 rollouts, not two.
    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Joint(),
        base_dir=tmp_path,
        budget=Budget(max_rollouts=64, max_iterations=1),
        proposals_per_iteration=2,
    )
    await _search(optimizer)
    # One slot, so the step never proposes a candidate it couldn't score.
    assert optimizer._iterations[0].n_candidates == 1
    assert [o.outcome for o in proposer.observed] == ["feasible"]
    assert proposer.updates == [{"temperature": 1.0}]  # never asked for
    assert optimizer._budget.spent == 16


async def test_a_duplicate_or_invalid_patch_is_told_and_redrawn(tmp_path):
    proposer = Scripted({"temperature": 0.0}, {"temperature": 0.7}, {"manager": "Summary"}, {"temperature": 1.0})
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Joint(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    assert [o.outcome for o in proposer.observed] == ["duplicate", "invalid", "invalid", "feasible"]
    assert "outside" in proposer.observed[2].reason
    assert [r.redraw for r in proposer.requests] == [0, 1, 2, 3]
    assert [p.outcome for p in optimizer._iterations[0].proposals] == ["duplicate", "invalid", "invalid", "feasible"]


async def test_a_candidate_whose_agent_cant_be_built_is_invalid_and_costs_nothing(tmp_path):
    @tunable(
        manager=choice(component(Window, size=Int(1, 100)), component(Summary)),
        temperature=OneOf(0.0, 0.5, 1.0),
    )
    def picky(manager, temperature):
        if temperature == 1.0:
            raise ValueError("temperature 1.0 is not supported")
        return (manager, temperature)

    proposer = Scripted({"temperature": 1.0}, {"temperature": 0.5})
    optimizer = Optimizer(
        _experiment(evaluators=[Temperature()]),
        picky,
        rollout_fn=_temperature_rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    await _search(optimizer)
    assert [o.outcome for o in proposer.observed] == ["invalid", "feasible"]
    assert proposer.observed[0].reason == "the agent can't be built: ValueError: temperature 1.0 is not supported"
    assert optimizer._budget.spent == 16  # the baseline and the 0.5 candidate, nothing for the one that can't build


async def test_a_factory_failing_on_infrastructure_is_not_invalid(tmp_path):
    calls = {"0.5": 0}

    @tunable(temperature=OneOf(0.0, 0.5, 1.0))
    def expiring(temperature):
        if temperature == 0.5:
            calls["0.5"] += 1
            if calls["0.5"] == 1:  # the build check: credentials expired, refreshed before the trials
                raise botocore.exceptions.CredentialRetrievalError(provider="custom-process", error_msg="expired")
        return (None, temperature)

    proposer = Scripted({"temperature": 0.5})
    optimizer = Optimizer(
        _experiment(evaluators=[Temperature()]),
        expiring,
        rollout_fn=_temperature_rollout,
        search=Search([Block(proposer, "temperature")], schedule=Joint()),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    with pytest.warns(UserWarning, match=r"the agent couldn't be built \(CredentialRetrievalError"):
        await _search(optimizer)
    assert [o.outcome for o in proposer.observed] == ["feasible"]


async def test_a_factory_needing_a_resource_is_not_build_checked(tmp_path):
    @tunable(temperature=OneOf(0.0, 0.5, 1.0))
    def sandboxed(temperature, sandbox):
        return (sandbox, temperature)

    proposer = Scripted({"temperature": 0.5})
    optimizer = Optimizer(
        _experiment(evaluators=[Temperature()]),
        sandboxed,
        rollout_fn=_temperature_rollout,
        environment=lambda case: {"sandbox": "box"},
        search=Search([Block(proposer, "temperature")], schedule=Joint()),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    await _search(optimizer)
    assert [o.outcome for o in proposer.observed] == ["feasible"]


async def test_a_proposer_that_raises_stops_the_search_before_the_end_phase(tmp_path):
    optimizer = _searching(
        Block(Scripted(RuntimeError("model down")), "temperature"), schedule=Alternate(), base_dir=tmp_path
    )
    with pytest.raises(ProposerError, match="block 'numbers' raised RuntimeError: model down") as raised:
        await optimizer.run_async()
    assert (raised.value.block, raised.value.run_dir) == ("numbers", optimizer._run.root)
    assert "Optimizer.resume" in str(raised.value) and str(raised.value.__cause__) == "model down"
    assert "to end it with what it found" not in str(raised.value)  # no closed iteration to end with
    copied = pickle.loads(pickle.dumps(raised.value))
    assert (str(copied), copied.block, copied.run_dir) == (str(raised.value), "numbers", optimizer._run.root)
    events = _events(optimizer)
    error = next(e for e in events if e["event"] == "proposer_error")
    assert (error["error_type"], error["error"]) == ("RuntimeError", "model down")
    assert events[-1] == error  # no `stop`: the run is left as a kill leaves it
    assert {e["record"]["split"] for e in events if e["event"] == "trial"} == {"train"}


async def test_a_proposer_returning_more_patches_than_asked_stops_the_search(tmp_path):
    class Greedy(Scripted):
        async def propose(self, request):
            return [
                Patch(updates={"temperature": 0.5}, rationale=""),
                Patch(updates={"temperature": 1.0}, rationale=""),
            ]

    optimizer = _searching(
        Block(Greedy(), "temperature"), schedule=Alternate(), base_dir=tmp_path, proposals_per_iteration=1
    )
    with pytest.raises(ProposerError, match="returned 2 patches for a request of 1"):
        await _search(optimizer)
    events = _events(optimizer)
    assert "returned 2 patches for a request of 1" in next(e for e in events if e["event"] == "proposer_error")["error"]


async def test_joint_merges_one_patch_of_every_block_per_candidate(tmp_path):
    structure = Scripted({"manager": "Summary"}, {"manager": "Window"})
    numbers = Scripted({"temperature": 1.0}, {"temperature": 0.5})
    optimizer = _searching(
        Block(structure, "manager"),
        Block(numbers, "temperature", "manager.Window.size"),
        schedule=Joint(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=2,
    )
    await _search(optimizer)
    proposals = optimizer._iterations[0].proposals
    assert [p.origin for p in proposals] == [["structure", "numbers"]] * 2
    # The numbers block is asked once per slot, on the slot's structure.
    assert [r.n for r in numbers.requests] == [1, 1]
    assert [r.parent.values["manager"] for r in numbers.requests] == ["Summary", "Window"]
    assert [o.outcome for o in structure.observed] == ["feasible", "feasible"]
    # Each candidate merges both blocks' patches, so the only evidence is the baseline's.
    assert [e.candidate_id for e in numbers.evidenced + structure.evidenced] == [optimizer._baseline.id] * 2


async def test_a_round_opener_is_told_the_rounds_best_at_its_end(tmp_path):
    outer = Scripted({"manager": "Summary"}, {"manager": "Window"})
    inner = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _searching(
        Block(outer, "manager", name="structure"),
        Block(inner, "temperature"),
        schedule=Nested("structure", inner_iterations=2),
        base_dir=tmp_path,
        budget=Budget(max_iterations=4),
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    roles = [(r.round, r.role) for r in optimizer._iterations]
    assert roles == [((0,), "opener"), ((0,), "inner"), ((0,), "inner"), ((1,), "opener")]
    # Round 0's opener (manager=Summary) is told at the round's end, with the round's best (temperature 1.0).
    first = outer.observed[0]
    assert first.values["manager"] == "Summary" and first.objective == 1.0 and first.source_id is not None
    # The round's candidates, opener first, then proposal order: TPE re-values the opener's trial by them.
    round_zero = [p.candidate_id for r in optimizer._iterations[:3] for p in r.proposals]
    assert first.round_candidate_ids == round_zero and first.candidate_id == round_zero[0]
    # The outer block is told the baseline as evidence, never the inner candidates.
    assert [e.candidate_id for e in outer.evidenced] == [optimizer._baseline.id]


def _proposer_tokens(root):
    """Each block's proposer tokens, as the result of the run, or of its resume, counts them from the record."""
    cost = _run_cost([], RunDirectory.open(root).events.read(), blocks=[])
    return {block: spend.usage.tokens for block, spend in cost.proposers.items()}


async def test_every_proposer_calls_spend_counts_on_resume(tmp_path):
    spend = Usage(
        input_tokens=9,
        output_tokens=1,
        tokens=10,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=1,
        seconds=1.0,
        models=["m"],
    )

    class Spending(Scripted):
        async def propose(self, request):
            patches = await super().propose(request)
            return [p.model_copy(update={"usage": spend}) for p in patches]

    optimizer = _searching(
        Block(Spending({"manager": "Summary"}, {"manager": "Window"}), "manager", name="structure"),
        Block(Spending({"temperature": 0.5}, {"temperature": 1.0}), "temperature"),
        schedule=Nested("structure", inner_iterations=2),
        base_dir=tmp_path,
        budget=Budget(max_iterations=4),
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    assert _proposer_tokens(optimizer._run.root) == {"structure": 20, "numbers": 20}  # the openers; the inners

    def killed_at(where, keep):
        return _proposer_tokens(_killed(optimizer._run.root, tmp_path / where, keep))

    # Round 0 open: its opener is observed only at the round's end, but was paid for.
    assert killed_at("open round", lambda events: _index(events, "iteration_end")) == {"structure": 10}
    # Killed in iteration 2 after its proposal was scored: the iteration runs again, its spend still counts.
    killed = killed_at("killed", lambda events: _index(events, "score", after=_index(events, "iteration_end")))
    assert killed == {"structure": 10, "numbers": 10}


async def test_a_run_killed_in_its_first_iteration_keeps_that_proposer_spend(tmp_path):
    spend = Usage(
        input_tokens=9,
        output_tokens=1,
        tokens=10,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=1,
        seconds=1.0,
        models=["m"],
    )

    class Spending(Scripted):
        async def propose(self, request):
            patches = await super().propose(request)
            return [p.model_copy(update={"usage": spend}) for p in patches]

    original = _searching(
        Block(Spending({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path / "runs",
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    await original.run_async()
    # Killed just after iteration 1's proposal was paid for; the resumed proposer proposes nothing (no spend).
    copy = _killed(original._run.root, tmp_path, lambda events: _index(events, "proposer_usage"))
    search = Search([Block(Spending(), "temperature"), Block(Scripted(), "manager")], schedule=Alternate())
    result = await _resume(copy, search=search, budget=Budget(max_iterations=1)).run_async()
    assert result.cost.proposers["numbers"].usage.tokens == 10
    assert OptimizationResult.from_file(copy / "result.json").cost == result.cost


async def test_on_iteration_may_be_async_and_its_error_stops_the_run(tmp_path):
    records = []

    async def collect(record):
        await asyncio.sleep(0)
        records.append(record)

    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path / "async",
        budget=Budget(max_iterations=1),
        on_iteration=collect,
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    assert [r.iteration for r in records] == [1]

    def broken(record):
        raise RuntimeError("callback bug")

    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path / "raising",
        budget=Budget(max_iterations=1),
        on_iteration=broken,
        proposals_per_iteration=1,
    )
    with pytest.raises(RuntimeError, match="callback bug"):
        await optimizer.run_async()
    # The iteration was closed first, so the run resumes past it.
    assert [e["event"] for e in _events(optimizer)][-1] == "iteration_end"


async def test_a_refit_that_raises_is_recorded_and_the_run_goes_on(tmp_path):
    from strands_evals.optimization.score_model.protocols import ScoreModel

    class Broken(ScoreModel):
        name = "broken"

        def fit(self, data, *, previous=None):
            raise ArithmeticError("singular")

    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        score_model=Broken(),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    with pytest.warns(UserWarning, match="refit raised") as warned:
        reason, events = await _search(optimizer)
    assert reason == "iterations"
    assert len([w for w in warned if "refit raised" in str(w.message)]) == 1  # once per distinct error
    fits = [e for e in events if e["event"] == "fit"]
    # The setup's fit and iteration 1's.
    assert [f["error"] for f in fits] == ["ArithmeticError: singular"] * 2 and not any(f["is_fitted"] for f in fits)


async def test_a_validation_fit_of_none_says_the_model_returned_no_fit(tmp_path):
    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        score_model=_Sequence(),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    result = await optimizer.run_async()
    assert result.validation.problem == "the model returned no fit"


class _FlaggedFit:
    """A fit whose held-out check flagged its only goal metric; nothing may decide with it."""

    flagged = ["Temperature"]
    diagnostics: dict = {}


class _Sequence(ScoreModel):
    """A score model returning the given fits in turn (None once they run out)."""

    name = "sequence"

    def __init__(self, *fits):
        self.fits = list(fits)

    def fit(self, data, *, previous=None):
        return self.fits.pop(0) if self.fits else None


async def test_a_flagged_validation_fit_is_the_picks_problem(tmp_path):
    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        score_model=_Sequence(*(_FlaggedFit() for _ in range(5))),  # every fit, the validation fit's included
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    result = await optimizer.run_async()
    assert result.validation.problem == "flagged: Temperature"
    assert not any(result.validation.estimates.values())  # picked by measured means
    assert "Validation fit:  flagged: Temperature" in result.summary()


class _ScoreFlagged:
    """A fit that flagged `Score` only and estimates Temperature at the candidate's temperature."""

    flagged = ["Score"]
    diagnostics: dict = {}
    candidates = type("_Every", (), {"__contains__": lambda self, _: True})()

    def estimates(self, values, candidate=None):
        return {"Temperature": Estimate(mean=float(values["temperature"]), sd=0.01)}


async def test_a_partly_flagged_validation_fit_says_the_other_metrics_kept_the_fit(tmp_path):
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Score()]),
        rollout_fn=_temperature_rollout,
        search=Search(
            [Block(Scripted({"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")], schedule=Alternate()
        ),
        objective=Maximize("Temperature"),
        constraints=[AtLeast("Score", 0.5)],
        base_dir=tmp_path,
        score_model=_Sequence(*(_ScoreFlagged() for _ in range(5))),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
    )
    result = await optimizer.run_async()
    v = result.validation
    assert v.problem == "flagged: Score (by adjusted means; the other metrics by the fit)"
    assert all(set(e) == {"Temperature"} for e in v.estimates.values())  # the fit judged the unflagged metric
    assert all(set(a) == {"Score"} for a in v.adjusted_means.values())
    assert f"Validation fit:  {v.problem}" in result.summary().splitlines()


async def test_a_flagged_fit_reaches_the_proposers_and_the_selection(tmp_path):
    seen = []

    class Watching(TopRanked):
        name = "watching"

        def choose(self, ctx):
            seen.append(ctx.fit)
            return super().choose(ctx)

    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        score_model=_Sequence(_FlaggedFit(), _FlaggedFit(), _FlaggedFit()),
        selection=Watching(),
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    # Passed on with its flags: each consumer leaves out the flagged metrics' models, as the ranking does.
    assert optimizer._fit is not None and optimizer._fit.flagged
    assert proposer.requests and all(r.fit is not None and r.fit.flagged for r in proposer.requests)
    assert seen and all(fit is not None and fit.flagged for fit in seen)


class _GroundedFlagged:
    """A fit of `Temperature` (the temperature plus 0.25) and `Grounded` (0.0, though every run scores 1.0) whose
    held-out check flagged `Grounded`."""

    flagged = ["Grounded"]
    diagnostics: dict = {}
    metrics: dict = {}

    def __init__(self, candidates):
        self.candidates = frozenset(candidates)

    def estimates(self, values, candidate=None):
        return {
            "Temperature": Estimate(mean=values["temperature"] + 0.25, sd=0.01),
            "Grounded": Estimate(mean=0.0, sd=0.01),
        }


class _FlagsGrounded(ScoreModel):
    name = "flags_grounded"

    def fit(self, data, *, previous=None):
        return _GroundedFlagged(c for c in data.candidates if data.scores[c].is_eligible)


async def test_a_flagged_metric_is_told_by_its_measured_mean_and_the_others_by_the_fit(tmp_path):
    class Grounded(Evaluator):
        async def evaluate_async(self, evaluation_case):
            return [EvaluationOutput(score=1.0, test_pass=True, reason="g")]

    proposer = Scripted({"temperature": 0.5})
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Grounded(name="Grounded")]),
        rollout_fn=_temperature_rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Alternate()),
        max_workers=64,
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        objective=WeightedSum(Minimize("Temperature"), Maximize("Grounded")),
        constraints=[AtLeast("Grounded", 0.5)],
        score_model=_FlagsGrounded(),
    )
    await _search(optimizer)
    (request,) = proposer.requests
    (baseline,) = request.candidates
    assert baseline.estimates == {"Temperature": Estimate(mean=0.25, sd=0.01)}
    assert baseline.term_values == [0.25, 1.0]  # Temperature from the fit (raw, though minimized), Grounded measured
    assert baseline.ranking_objective == pytest.approx(0.875) and baseline.is_feasible
    (observation,) = proposer.observed
    assert observation.term_values == [0.75, 1.0] and observation.objective == pytest.approx(0.625)
    assert observation.outcome == "feasible"  # Grounded >= 0.5 judged on the measured 1.0, not the fit's 0.0


async def test_a_model_that_returns_none_keeps_the_fit_in_use(tmp_path):
    flagged = _FlaggedFit()
    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        score_model=_Sequence(flagged),  # the setup's refit; iteration 1's returns None
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
    )
    _, events = await _search(optimizer)
    assert optimizer._fit is flagged
    assert [f["is_fitted"] for f in events if f["event"] == "fit"] == [True, False]


async def test_the_budget_stops_the_search_before_the_end_phases_reserve(tmp_path):
    # 56 needed (baseline 8 + reserves 48), so 72 leaves room for two candidates of 8 rollouts.
    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}, {"temperature": 0.5}), "temperature"),
        schedule=Alternate(),
        base_dir=tmp_path,
        budget=Budget(max_rollouts=72, max_iterations=None),
        proposals_per_iteration=1,
    )
    reason, _ = await _search(optimizer)
    # Iterations: temperature 0.5, the empty manager block, temperature 1.0; then the reserve is all that's left.
    assert reason == "budget" and optimizer._budget.spent == 24 and len(optimizer._iterations) == 3


async def test_a_selection_outside_the_pool_raises(tmp_path):
    class Rogue(TopRanked):
        name = "rogue"

        def choose(self, ctx):
            return "nope"

    optimizer = _searching(Block(Scripted({"temperature": 0.5}), "temperature"), base_dir=tmp_path, selection=Rogue())
    await optimizer._start()
    with pytest.raises(ValueError, match="Rogue.choose returned 'nope'"):
        await optimizer._run_search()


# the end phase


def _two_steps(*updates, **kwargs):
    """A search over `temperature` that proposes `updates` one per step, then stops after two steps."""
    kwargs.setdefault("budget", Budget(max_iterations=2))
    return _searching(
        Block(Scripted(*updates), "temperature"),
        schedule=Joint(),
        proposals_per_iteration=1,
        **kwargs,
    )


async def test_the_end_phase_ships_the_best_finalist_and_writes_the_run_files(tmp_path):
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path)
    result = await optimizer.run_async()
    events = _events(optimizer)
    names = _names(events, next(i for i, e in enumerate(events) if e["event"] == "stop"))
    # Two finalists and the baseline on validation, then the champion and the baseline on test.
    assert names == ["stop", "finalists", "score", "score", "score", "picked", "score", "score", "test"]
    assert result.improved and result.best.values["temperature"] == 1.0 and result.stop_reason == "iterations"
    assert result.best.changed_values() == {"temperature": {"old": 0.0, "new": 1.0}}
    assert result.validation.problem is None and result.validation.estimates  # picked on the validation fit
    assert {e["split"] for e in events if e["event"] == "score"} == {"train", "validation", "test"}

    test = result.test
    assert (test.champion_id, test.baseline_id) == (result.best.id, result.baseline_id)
    assert test.objective.mean_diff == pytest.approx(1.0) and test.objective.n_cases == 6
    assert test.ratio == {"Temperature": None}  # the baseline's mean is 0
    assert test.missing == {result.best.id: 0, result.baseline_id: 0} and result.test_failed_ids == []
    assert result.rollouts_spent == optimizer._budget.spent == 8 * 3 + 6 * 3 + 6 * 2
    assert set(result.cost.agent) == set(result.cost.judges) == {"train", "validation", "test"}

    root = optimizer._run.root
    assert json.loads((root / "best.values.json").read_text()) == {"temperature": {"old": 0.0, "new": 1.0}}
    assert (root / "best.diff").read_text() == ""  # no text changed
    assert json.loads((root / "effects.json").read_text()) == result.effects
    assert OptimizationResult.from_file(root / "result.json") == result


async def test_the_baseline_ships_when_nothing_beats_it_and_is_tested_against_itself(tmp_path):
    optimizer = _two_steps(
        {"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path, objective=Minimize("Temperature")
    )
    result = await optimizer.run_async()
    assert not result.improved and result.best.id == result.baseline_id
    assert result.test.champion_id == result.test.baseline_id == result.baseline_id
    assert result.test.objective.mean_diff == 0.0 and set(result.test.means) == {result.baseline_id}
    assert (optimizer._run.root / "best.diff").read_text() == ""
    assert json.loads((optimizer._run.root / "best.values.json").read_text()) == {}
    # The baseline is measured once on test.
    assert optimizer._budget.spent == 8 * 3 + 6 * 3 + 6


async def test_a_test_arm_with_no_completed_trial_skips_the_report(tmp_path):
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path)
    test_cases = set(optimizer._case_split.test)

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 1.0 and case.name in test_cases:
            raise ValueError("crashes on test")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    assert result.best.values["temperature"] == 1.0  # picked on validation, where it ran
    assert result.test is None
    assert result.test_reason == "every test trial of the champion failed, so there is nothing to compare"
    assert _names(_events(optimizer))[-1] == "test"  # agent failures are final: a resume reuses this
    # The result says which tested candidate measured nothing on test, in result.json and on resume too.
    assert result.test_failed_ids == [result.best.id]
    root = optimizer._run.root
    assert OptimizationResult.from_file(root / "result.json").test_failed_ids == [result.best.id]
    assert load_state(RunDirectory.open(root), config=optimizer._config()).test_failed_ids == [result.best.id]


async def test_a_finalist_with_no_completed_validation_trial_cannot_ship(tmp_path):
    # A crash counts as 0.5, above the baseline's 0.0, so without the check the crashing finalist would ship.
    optimizer = _two_steps({"temperature": 1.0}, base_dir=tmp_path, agent_failure_score=0.5)
    validation_cases = set(optimizer._case_split.validation)

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 1.0 and case.name in validation_cases:
            raise ValueError("crashes on validation")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    crashed = next(c.id for c in result.candidates.values() if c.values["temperature"] == 1.0)
    assert result.best.id == result.baseline_id and not result.improved
    assert result.validation.infeasible[crashed] == "every validation trial failed, so nothing was measured"
    # Left out of the validation fit as well, so only the baseline is left to fit.
    assert result.validation.problem == "fewer than 2 eligible candidates with a completed validation trial"


@pytest.mark.parametrize("crashes", [1.0, 0.0])
async def test_the_validation_fit_leaves_out_a_candidate_with_no_completed_validation_trial(tmp_path, crashes):
    fitted = []

    class Spy(_Sequence):
        def fit(self, data, *, previous=None):
            fitted.append((tuple(data.cases), list(data.candidates)))
            return None

    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path, score_model=Spy())
    validation_cases = optimizer._case_split.validation

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == crashes and case.name in validation_cases:
            raise ValueError("crashes on validation")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    by_temperature = {c.values["temperature"]: c.id for c in result.candidates.values()}
    # The baseline, then the finalists in ranking order, without the one that crashed (the baseline included).
    expected = [by_temperature[t] for t in (0.0, 1.0, 0.5) if t != crashes]
    assert [ids for cases, ids in fitted if cases == validation_cases] == [expected]


@pytest.mark.parametrize("split", ["validation", "test"])
async def test_the_end_phase_re_runs_a_transient_infrastructure_failure_after_the_search_spent_all_it_may(
    tmp_path, split
):
    # The baseline and two candidates on train, then only the end phase's reserves are left; two finalists, so
    # validation's reserve has no room for more arms. (On test, validation's unused headroom also reaches the
    # test arms; the test reserve's own size is pinned in test_budget_must_cover_the_baseline_and_the_reserves.)
    reserves = _two_steps(final_pick=FinalPick(n_finalists=2))._budget
    budget = Budget(max_rollouts=8 * 3 + reserves.validation_reserve + reserves.test_reserve, max_iterations=None)
    optimizer = _two_steps(
        {"temperature": 0.5},
        {"temperature": 1.0},
        base_dir=tmp_path,
        budget=budget,
        final_pick=FinalPick(n_finalists=2),
    )
    flaky = getattr(optimizer._case_split, split)[0]
    failed = []

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 1.0 and case.name == flaky and not failed:
            failed.append(case.name)  # fails once, then succeeds when re-run
            raise InfrastructureError("sandbox failed to start")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    assert result.stop_reason == "budget" and failed == [flaky]
    assert result.rollouts_spent <= budget.max_rollouts
    hottest = next(c.id for c in result.candidates.values() if c.values["temperature"] == 1.0)
    statuses = [
        e["record"]["status"]
        for e in _events(optimizer)
        if e["event"] == "trial" and e["record"]["candidate_id"] == hottest and e["record"]["case_name"] == flaky
    ]
    assert statuses == ["infrastructure_invalid", "completed"]
    # The best candidate is still eligible on validation and ships, and its test report misses no case.
    assert result.best.values["temperature"] == 1.0 and result.validation.infeasible == {}
    assert result.test.missing == {result.best.id: 0, result.baseline_id: 0}


async def test_a_finalist_failing_a_constraint_on_validation_cannot_ship(tmp_path):
    experiment = _experiment(n=14, evaluators=[Temperature()])
    split = Split(train=experiment.cases[:8], validation=experiment.cases[8:14])
    validation = {case.name for case in split.validation}

    class Grounded(Evaluator):
        """Met everywhere, except by temperature 1.0 on validation cases."""

        async def evaluate_async(self, evaluation_case):
            bad = evaluation_case.actual_output == "1.0" and evaluation_case.name in validation
            return [EvaluationOutput(score=0.0 if bad else 1.0, test_pass=not bad, reason="g")]

    blocks = [Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"), Block(Scripted(), "manager")]
    optimizer = _optimizer(
        Experiment(cases=experiment.cases, evaluators=[Temperature(), Grounded(name="Grounded")]),
        rollout_fn=_temperature_rollout,
        search=Search(blocks, schedule=Joint()),
        max_workers=64,
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=2),
        objective=Maximize("Temperature"),
        constraints=[AtLeast("Grounded", 0.5)],
        split=split,
    )
    result = await optimizer.run_async()
    # temperature 1.0 is the best and feasible on train, so a finalist; it fails the constraint on validation.
    by_temperature = {v.values["temperature"]: v.id for v in result.candidates.values()}
    assert by_temperature[1.0] in result.validation.infeasible
    assert result.best.values["temperature"] == 0.5
    assert result.test is None and result.test_reason == "no test cases"


async def test_the_finalists_are_the_candidates_meeting_a_constraint_with_margin(tmp_path):
    experiment = _experiment(n=33, evaluators=[Temperature()])
    split = Split(train=experiment.cases[:30], validation=experiment.cases[30:33])
    # Temperature 0.5 is grounded on 22 of the 30 train cases (0.73, above the floor) and 1 of the 3 validation
    # cases; 1.0 everywhere; the baseline (0.0) nowhere.
    met_at_half = {case.name for case in [*split.train[:22], *split.validation[:1]]}

    class Grounded(Evaluator):
        async def evaluate_async(self, evaluation_case):
            output = evaluation_case.actual_output
            met = output == "1.0" or (output == "0.5" and evaluation_case.name in met_at_half)
            return [EvaluationOutput(score=float(met), test_pass=met, reason="g")]

    blocks = [Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"), Block(Scripted(), "manager")]
    optimizer = _optimizer(
        Experiment(cases=experiment.cases, evaluators=[Temperature(), Grounded(name="Grounded")]),
        rollout_fn=_temperature_rollout,
        search=Search(blocks, schedule=Joint()),
        max_workers=64,
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=2),
        objective=Minimize("Temperature"),
        constraints=[AtLeast("Grounded", 0.5)],
        final_pick=FinalPick(n_finalists=1),
        score_model=_Sequence(),  # no fit: the search ranks and the pick decides by measured means
        split=split,
    )
    result = await optimizer.run_async()
    by_temperature = {v.values["temperature"]: v.id for v in result.candidates.values()}
    # 0.5 ranks first on train (feasible and cheaper), but with 3 validation cases 0.73 - 1.28 * 0.45 *
    # sqrt(1/30 + 1/3) = 0.38 < 0.5 (with 30 it would be 0.58), so the one finalist is 1.0, which meets the floor
    # on every case.
    finalists = next(e for e in _events(optimizer) if e["event"] == "finalists")
    assert finalists["candidate_ids"] == [by_temperature[1.0]]
    assert result.best.values["temperature"] == 1.0


@pytest.mark.parametrize("measured_means", [False, True], ids=["fit", "measured-means"])
async def test_a_candidate_whose_constraint_is_never_measured_cannot_ship(tmp_path, measured_means):
    experiment = _experiment(n=14, evaluators=[Temperature()])
    split = Split(train=experiment.cases[:8], validation=experiment.cases[8:14])

    class Grounded(Evaluator):
        """Met by every temperature but 1.0, which it finds nothing to judge in, on every case."""

        async def evaluate_async(self, evaluation_case):
            if evaluation_case.actual_output == "1.0":
                return [EvaluationOutput(score=0.0, test_pass=True, reason="n/a", label=NOT_APPLICABLE)]
            return [EvaluationOutput(score=1.0, test_pass=True, reason="g")]

    blocks = [Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"), Block(Scripted(), "manager")]
    optimizer = _optimizer(
        Experiment(cases=experiment.cases, evaluators=[Temperature(), Grounded(name="Grounded")]),
        rollout_fn=_temperature_rollout,
        search=Search(blocks, schedule=Joint()),
        max_workers=64,
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=2),
        objective=Maximize("Temperature"),
        constraints=[AtLeast("Grounded", 0.5)],
        split=split,
        **({"score_model": _Sequence()} if measured_means else {}),  # a model that never fits: measured means
    )
    result = await optimizer.run_async()
    never = next(c.id for c in result.candidates.values() if c.values["temperature"] == 1.0)
    assert result.best.values["temperature"] == 0.5
    assert result.validation.infeasible[never] == "Grounded >= 0.5: never measured"
    (row,) = [r for r in result.to_rows("proposal") if r["candidate_id"] == never]
    assert row["is_from_fit"] == (not measured_means)
    assert row["unmet_constraints"] == ["Grounded >= 0.5: never measured"] and row["p_met.Grounded >= 0.5"] is None
    events = _events(optimizer)
    (observed,) = [e for e in events if e["event"] == "observe" and e["candidate_id"] == never]
    assert observed["observation"]["reason"] == "Grounded >= 0.5: never measured"
    (shared,) = [e for e in events if e["event"] == "evidence" and e["evidence"]["candidate_id"] == never]
    assert observed["observation"]["violations"] == shared["evidence"]["violations"] == [None]  # infinite, in JSON
    state = load_state(RunDirectory.open(optimizer._run.root), config=optimizer._config())
    for event in (observed, shared):
        (replayed,) = [f for f in state.feedback[event["block"]] if f.candidate_id == never]
        assert replayed.violations == [math.inf]


async def test_validation_and_test_runs_never_reach_the_search(tmp_path):
    seen = []

    class Spy(_Sequence):
        def fit(self, data, *, previous=None):
            seen.append((tuple(data.cases), previous))
            return None

    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=Joint(),
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=2),
        score_model=Spy(),
    )
    await optimizer.run_async()
    train, validation = tuple(optimizer._case_split.train), tuple(optimizer._case_split.validation)
    # Every search refit on train; one fresh fit on validation (no previous fit), never on test.
    assert [cases for cases, _ in seen[:-1]] == [train] * (len(seen) - 1)
    assert seen[-1] == (validation, None)
    assert all({t.split for t in r.trials} == {"train"} for r in proposer.requests)


async def test_the_default_search_runs_end_to_end_and_observes_every_patch_once(tmp_path):
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=_temperature_rollout,
        max_workers=64,
        base_dir=tmp_path,
        budget=Budget(max_iterations=6),
        proposals_per_iteration=2,
    )
    result = await optimizer.run_async()
    assert [b.name for b, _ in optimizer._blocks] == ["structure", "numbers"]
    # max_iterations=6 is 6 x 2 candidate slots; a step that saved nothing (all duplicates) costs none.
    assert result.stop_reason == "iterations" and _slots_used(result.iterations) == 12
    assert len(result.iterations) > 6 and any(not _saved(r) for r in result.iterations)
    observed = [e for e in _events(optimizer) if e["event"] == "observe"]
    # One observe per patch, invalid and duplicate ones included, each round's opener at its round's end.
    assert len(observed) == sum(len(p.origin) for r in result.iterations for p in r.proposals) > 0
    assert all(not block.proposer._pending for block, _ in optimizer._blocks)
    assert result.best.values["temperature"] >= result.baseline.values["temperature"]


async def test_the_default_structure_block_models_from_its_second_round(tmp_path):
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=_temperature_rollout,
        max_workers=64,
        base_dir=tmp_path,
        # A round uses at most 5 slots (its opener and 4 inner steps of 1), so at least 3 rounds open in 11.
        budget=Budget(max_iterations=11),
        proposals_per_iteration=1,
    )
    await _search(optimizer)
    structure = next(block.proposer for block, _ in optimizer._blocks if block.name == "structure")
    assert isinstance(structure, TPE) and structure.n_startup_trials == 2
    events = _events(optimizer)
    (first,) = [e for e in events if e["event"] == "evidence" and e["block"] == "structure"]
    assert (first["iteration"], first["evidence"]["candidate_id"]) == (0, optimizer._baseline.id)
    # Each round's opener, observed at its round's end. Its study holds the baseline, then one trial per
    # closed round, so the startup of 2 ends after the first round (without the baseline, after the second).
    openers = [
        e["observation"]["patch"]["rationale"] for e in events if e["event"] == "observe" and e["block"] == "structure"
    ]
    assert len(openers) >= 3
    assert openers == ["TPE: random startup"] + [f"TPE: modeled on {k} told trials" for k in range(2, len(openers) + 1)]


# resume


def _default_search(base_dir, **kwargs):
    kwargs.setdefault("budget", Budget(max_iterations=6))
    kwargs.setdefault("rollout_fn", _temperature_rollout)
    return _optimizer(
        _experiment(evaluators=[Temperature()]),
        max_workers=64,
        base_dir=base_dir,
        proposals_per_iteration=2,
        **kwargs,
    )


def _resume(run_dir, **kwargs):
    return Optimizer.resume(
        run_dir,
        experiment=_experiment(evaluators=[Temperature()]),
        factory=create_agent,
        rollout_fn=_temperature_rollout,
        **kwargs,
    )


def _killed(root, tmp_path, keep):
    """A copy of the run `root` with its event log cut after the first `keep(events)` events, as a kill leaves it."""
    copy = tmp_path / "killed" / root.name
    shutil.copytree(root, copy)
    lines = (copy / "events.jsonl").read_text().splitlines()
    cut = keep([json.loads(line) for line in lines])
    (copy / "events.jsonl").write_text("\n".join(lines[:cut]) + "\n")
    return copy


def _index(events, name, nth=1, after=0):
    """Just past the `nth` `name` event at or after `after`."""
    hits = [i for i, e in enumerate(events) if e["event"] == name and i >= after]
    return hits[nth - 1] + 1


def _close(value):
    """`value` with every float replaced by `pytest.approx` at 1e-12, for comparing reported diagnostics: the
    fit's linear algebra (numpy's BLAS) may round its last bit differently from one run to the next."""
    if isinstance(value, dict):
        return {key: _close(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_close(item) for item in value]
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-12, abs=1e-12)
    return value


def _assert_same(resumed, original, *, rerun=None):
    """Everything a run decides is the same; timings (seconds) and paths aren't compared. The iteration a kill
    interrupted (`rerun`) is run again on its saved trials, so it spends 0 rollouts the second time."""
    assert resumed.candidates == original.candidates
    for again, first in zip(resumed.iterations, original.iterations, strict=True):
        if again.iteration == rerun:
            assert again.rollouts_spent == 0 and again.model_copy(update={"rollouts_spent": 0}) == first.model_copy(
                update={"rollouts_spent": 0}
            )
        else:
            assert again == first
    assert resumed.stop_reason == original.stop_reason
    assert resumed.rollouts_spent == original.rollouts_spent
    assert resumed.effects == _close(original.effects)  # what the fit reports, to the last bit or two
    validation_fields = ("champion_id", "front", "objectives", "estimates", "adjusted_means", "infeasible")
    for field in (*validation_fields, "p_met", "problem", "rule", "outcome", "reason"):
        assert getattr(resumed.validation, field) == getattr(original.validation, field)
    for field in ("objective", "constraints", "feasibility_threshold", "proposers"):
        assert getattr(resumed, field) == getattr(original, field)
    for field in ("champion_id", "objective", "difference", "ratio", "missing"):
        assert getattr(resumed.test, field) == getattr(original.test, field)
    assert resumed.test_failed_ids == original.test_failed_ids


@pytest.mark.parametrize(
    "where",
    [
        "during setup",  # some baseline trials saved, no iteration closed
        "mid-iteration",  # iteration 3 proposed and scored, not closed
        "after the search",  # stopped, finalists chosen, nothing validated
        "after the pick",  # picked on validation, the test report not run
    ],
)
async def test_a_killed_run_resumes_to_the_same_result_spending_nothing_twice(tmp_path, where):
    original_optimizer = _default_search(tmp_path / "runs")
    original = await original_optimizer.run_async()
    cuts = {
        "during setup": lambda events: _index(events, "trial", nth=3),
        "mid-iteration": lambda events: _index(events, "score", after=_index(events, "iteration_end", nth=2)),
        "after the search": lambda events: _index(events, "finalists"),
        "after the pick": lambda events: _index(events, "picked"),
    }
    copy = _killed(original_optimizer._run.root, tmp_path, cuts[where])
    trials_before = sum(1 for e in _events_at(copy) if e["event"] == "trial")
    trials_cut = len(_events_at(copy))  # the killed log's length, before resume appends to it

    resumed_optimizer = _resume(copy)
    resumed = await resumed_optimizer.run_async()

    killed = 1 + sum(1 for e in _events_at(copy)[:trials_cut] if e["event"] == "iteration_end")
    _assert_same(resumed, original, rerun=killed if where == "mid-iteration" else None)
    # Every saved trial was reused: the total spent is the uninterrupted run's.
    assert resumed_optimizer._budget.spent == original_optimizer._budget.spent
    new_trials = sum(1 for e in _events_at(copy) if e["event"] == "trial") - trials_before
    assert new_trials == sum(1 for e in _events(original_optimizer) if e["event"] == "trial") - trials_before


async def test_a_run_killed_after_setup_tells_the_baseline_once_on_resume(tmp_path):
    original_optimizer = _default_search(tmp_path / "runs")
    original = await original_optimizer.run_async()
    # Setup's events (the baseline's evidence among them) are written, but no iteration closed.
    copy = _killed(original_optimizer._run.root, tmp_path, lambda events: _index(events, "select") - 1)
    resumed_optimizer = _resume(copy)
    resumed = await resumed_optimizer.run_async()
    _assert_same(resumed, original)
    baseline = resumed_optimizer._baseline.id
    blocks = [block.name for block, _ in resumed_optimizer._blocks]
    logged = [
        e["block"] for e in _events_at(copy) if e["event"] == "evidence" and e["evidence"]["candidate_id"] == baseline
    ]
    assert logged == blocks * 2  # setup ran again after the resume event
    state = load_state(RunDirectory.open(copy), config=resumed_optimizer._config())
    for block in blocks:
        told = [t for t in state.feedback[block] if isinstance(t, Evidence) and t.candidate_id == baseline]
        assert len(told) == 1


async def test_a_refit_that_raises_on_replay_falls_back_to_measured_means_and_the_resume_goes_on(tmp_path):
    original_optimizer = _default_search(tmp_path / "runs")
    await original_optimizer.run_async()
    copy = _killed(
        original_optimizer._run.root,
        tmp_path,
        lambda events: _index(events, "score", after=_index(events, "iteration_end", nth=2)),
    )
    resumed_optimizer = _resume(copy)
    model = resumed_optimizer._score_model
    real_fit, calls = model.fit, []

    def fit(data, *, previous=None):  # the first replayed refit raises (e.g. another numpy), then fits again
        calls.append(previous)
        if len(calls) == 1:
            raise ArithmeticError("singular")
        return real_fit(data, previous=previous)

    model.fit = fit
    with pytest.warns(UserWarning, match="replaying a refit on resume raised"):
        resumed = await resumed_optimizer.run_async()

    # The next refit gets the last successful one as `previous`: none yet, since the first raised.
    assert len(calls) > 2 and calls[1] is None
    assert resumed.best is not None and resumed.stop_reason == "iterations"


def _events_at(root):
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("failing", [1, 2])
async def test_a_search_a_proposer_stopped_resumes_and_ends_as_an_uninterrupted_run(tmp_path, failing):
    updates = [{"temperature": 0.5}, {"temperature": 1.0}]
    expected = await _two_steps(*updates, base_dir=tmp_path / "runs").run_async()
    stopped = _two_steps(*updates[: failing - 1], RuntimeError("throttled"), base_dir=tmp_path / "stopped")
    with pytest.raises(ProposerError, match="throttled"):
        await stopped.run_async()
    assert not [e for e in _events(stopped) if e["event"] == "stop"]
    search = Search(
        [Block(Scripted(*updates[failing - 1 :]), "temperature"), Block(Scripted(), "manager")], schedule=Joint()
    )
    resumed = await _resume(stopped._run.root, search=search).run_async()
    _assert_same(resumed, expected)
    assert len(resumed.iterations) == 2 and resumed.best.values["temperature"] == 1.0


async def test_a_search_a_proposer_stopped_can_end_with_what_it_found(tmp_path):
    """A proposer that can't be fixed: resumed with its closed iterations as the cap, the run goes to the end phase."""
    stopped = _two_steps({"temperature": 0.5}, RuntimeError("no access"), base_dir=tmp_path)
    with pytest.raises(ProposerError, match=r"also pass budget=Budget\(max_rollouts=500, max_iterations=1\)"):
        await stopped.run_async()
    search = Search([Block(Scripted(), "temperature"), Block(Scripted(), "manager")], schedule=Joint())
    budget = Budget(max_rollouts=500, max_iterations=1)
    result = await _resume(stopped._run.root, search=search, budget=budget).run_async()
    assert result.stop_reason == "iterations" and len(result.iterations) == 1
    assert result.best.values["temperature"] == 0.5 and result.test is not None


async def test_costs_after_resume_count_a_killed_iterations_trials(tmp_path):
    original = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path / "runs")
    await original.run_async()
    # Killed once iteration 2's candidate (temperature 1.0) is scored; the resumed budget stops the search
    # before iteration 2 runs again, so that candidate is in no result.
    copy = _killed(
        original._run.root, tmp_path, lambda events: _index(events, "score", after=_index(events, "iteration_end"))
    )
    search = Search([Block(Scripted(), "temperature"), Block(Scripted(), "manager")], schedule=Joint())
    resumed = _resume(copy, search=search, budget=Budget(max_iterations=1))
    result = await resumed.run_async()
    assert all(c.values["temperature"] != 1.0 for c in result.candidates.values())
    trials = [e["record"] for e in _events_at(copy) if e["event"] == "trial"]
    for split in ("train", "validation", "test"):
        seconds = sum(r["usage"]["seconds"] for r in trials if r["split"] == split)
        assert result.cost.agent[split].usage.seconds == pytest.approx(seconds)
    assert result.rollouts_spent == len(trials)


_SPEND = Usage(
    input_tokens=9,
    output_tokens=1,
    tokens=10,
    cache_read_tokens=None,
    cache_write_tokens=None,
    model_calls=1,
    seconds=1.0,
    models=["m"],
)


class Paid(Scripted):
    """`Scripted`, each patch carrying 10 tokens of proposer spend."""

    async def propose(self, request):
        return [p.model_copy(update={"usage": _SPEND}) for p in await super().propose(request)]


# Iteration 1 proposes 0.5; iteration 2 proposes 0.25 (not an option, so invalid), then 1.0 on its redraw.
_STEPS = ({"temperature": 0.5}, {"temperature": 0.25}, {"temperature": 1.0})


def _replay_search(proposer):
    return Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint())


async def _two_paid_steps(base_dir):
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=_temperature_rollout,
        search=_replay_search(Paid(*_STEPS)),
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=base_dir,
    )
    result = await optimizer.run_async()
    assert [p.outcome for p in result.iterations[1].proposals] == ["invalid", "feasible"]
    return optimizer, result


def _scored_in(iteration):
    """Just past the first `score` after the iteration's `select`: its candidate scored, the iteration not closed."""
    return lambda events: _index(events, "score", after=_index(events, "select", nth=iteration))


@pytest.mark.parametrize(
    ("killed", "again"),
    [
        (1, None),
        (2, None),
        (2, "after reopening"),  # the resumed process killed again before it drew: the first draw is replayed
        (2, "mid-iteration"),  # killed again after it replayed and scored: its own draw is replayed
    ],
)
async def test_a_killed_iteration_runs_again_on_the_proposals_it_drew(tmp_path, killed, again):
    original_optimizer, original = await _two_paid_steps(tmp_path / "runs")
    copy = _killed(original_optimizer._run.root, tmp_path, _scored_in(killed))
    # Asked again, the proposer would answer differently: it holds only what the run proposes after the kill.
    later = _STEPS[1:] if killed == 1 else ()
    if again is not None:
        await _resume(copy, search=_replay_search(Paid(*later))).run_async()
        reopened = _index(_events_at(copy), "resume")
        keep = reopened if again == "after reopening" else _index(_events_at(copy), "score", after=reopened)
        copy = _killed(copy, tmp_path / "again", lambda events: keep)

    proposer = Paid(*later)
    resumed = await _resume(copy, search=_replay_search(proposer)).run_async()

    _assert_same(resumed, original, rerun=killed)
    assert killed not in {r.iteration for r in proposer.requests}
    # Told every patch as the uninterrupted run's proposer was, the replayed invalid one included.
    told = [(o.patch, o.outcome) for o in original_optimizer._proposer("numbers").observed]
    assert [(o.patch, o.outcome) for o in proposer.observed] == told
    assert resumed.cost.proposers == original.cost.proposers  # every patch paid for once
    trials = {e["record"]["candidate_id"] for e in _events_at(copy) if e["event"] == "trial"}
    assert trials <= set(resumed.candidates)


async def test_a_step_killed_before_its_proposals_are_all_in_asks_the_proposers_again(tmp_path):
    original_optimizer, original = await _two_paid_steps(tmp_path / "runs")
    # Killed once iteration 2's first proposal is found invalid, before its redraw: nothing of it was scored.
    copy = _killed(original_optimizer._run.root, tmp_path, lambda events: _index(events, "invalid"))
    proposer = Paid(*_STEPS[1:])

    resumed = await _resume(copy, search=_replay_search(proposer)).run_async()

    _assert_same(resumed, original)
    assert [(r.iteration, r.redraw) for r in proposer.requests] == [(2, 0), (2, 1)]


@pytest.mark.parametrize(("field", "value"), [("parent_id", "another"), ("blocks", ["numbers"])])
async def test_a_draw_recorded_for_another_parent_or_blocks_is_not_replayed(tmp_path, field, value):
    original_optimizer, original = await _two_paid_steps(tmp_path / "runs")
    copy = _killed(original_optimizer._run.root, tmp_path, _scored_in(2))
    events = _events_at(copy)
    for event in events:
        if event["event"] == "draw" and event["iteration"] == 2:
            event[field] = value
    (copy / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    proposer = Paid(*_STEPS[1:])

    resumed = await _resume(copy, search=_replay_search(proposer)).run_async()

    assert [(r.iteration, r.redraw) for r in proposer.requests] == [(2, 0), (2, 1)]
    _assert_same(resumed, original, rerun=2)


async def test_a_replayed_step_cut_narrower_on_resume_takes_its_first_proposals(tmp_path):
    original_optimizer = _searching(
        Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"),
        schedule=Joint(),
        base_dir=tmp_path / "runs",
        budget=Budget(max_iterations=1),
        proposals_per_iteration=2,
    )
    original = await original_optimizer.run_async()
    assert len(original.iterations[0].proposals) == 2
    # Killed once both proposals were saved, before either was scored.
    copy = _killed(original_optimizer._run.root, tmp_path, lambda events: _index(events, "propose", nth=3))
    proposer = Scripted({"temperature": 1.0})

    # 8 baseline + 48 reserve + 8: room for one candidate of the two the step drew.
    resumed = await _resume(
        copy, search=_replay_search(proposer), budget=Budget(max_rollouts=64, max_iterations=1)
    ).run_async()

    (step,) = resumed.iterations
    assert step.n_candidates == 1
    assert [resumed.candidates[p.candidate_id].values["temperature"] for p in step.proposals] == [0.5]
    assert proposer.requests == [] and resumed.rollouts_spent <= 52


async def test_a_step_wider_on_resume_asks_the_proposers_again(tmp_path):
    # 8 baseline + 48 reserve + 8: the step is cut to one candidate.
    with pytest.warns(UserWarning, match="leaves less than one iteration"):
        original_optimizer = _searching(
            Block(Scripted({"temperature": 0.5}), "temperature"),
            schedule=Joint(),
            base_dir=tmp_path / "runs",
            budget=Budget(max_rollouts=64, max_iterations=1),
            proposals_per_iteration=2,
        )
    await original_optimizer.run_async()
    copy = _killed(original_optimizer._run.root, tmp_path, _scored_in(1))
    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})

    # A larger budget makes room for both: asked again, as an uninterrupted run with that budget would be.
    with pytest.warns(UserWarning, match="leaves less than one iteration"):  # the saved budget, then replaced
        resumed_optimizer = _resume(copy, search=_replay_search(proposer), budget=Budget(max_iterations=1))
    resumed = await resumed_optimizer.run_async()

    (step,) = resumed.iterations
    assert [r.iteration for r in proposer.requests] == [1, 1]
    assert [resumed.candidates[p.candidate_id].values["temperature"] for p in step.proposals] == [0.5, 1.0]


async def test_a_replayed_slot_merges_each_blocks_patch_as_drawn(tmp_path):
    def search(structure, numbers):
        return Search([Block(numbers, "temperature"), Block(structure, "manager")], schedule=Joint())

    original_optimizer = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=_temperature_rollout,
        search=search(Scripted({"manager": "Summary"}), Scripted({"temperature": 0.5})),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path / "runs",
    )
    original = await original_optimizer.run_async()
    (proposal,) = original.iterations[0].proposals
    assert len(proposal.origin) == 2  # the structural block first, then the numbers block on its structure
    copy = _killed(original_optimizer._run.root, tmp_path, _scored_in(1))

    resumed = await _resume(copy, search=search(Scripted(), Scripted())).run_async()

    _assert_same(resumed, original, rerun=1)
    assert resumed.candidates[proposal.candidate_id].values["manager"] == "Summary"


async def test_resuming_a_finished_run_returns_its_result_and_spends_nothing(tmp_path):
    original_optimizer = _default_search(tmp_path)
    original = await original_optimizer.run_async()
    root = original_optimizer._run.root
    trials = sum(1 for e in _events_at(root) if e["event"] == "trial")

    with pytest.warns(UserWarning, match=r"search already stopped \(iterations\).*returns the finished result"):
        resumed = _resume(root, budget=Budget(max_rollouts=2000, max_iterations=50))
    assert [e["event"] for e in _events_at(root)][-1] != "resume"  # resume() itself writes nothing
    again = await resumed.run_async()

    _assert_same(again, original)
    # The budget the search stopped under, not the larger one it was resumed with.
    assert again.max_rollouts == original.max_rollouts != 2000
    assert f"Rollouts:   {original.rollouts_spent:,} / {original.max_rollouts:,}" in again.summary()
    assert sum(1 for e in _events_at(root) if e["event"] == "trial") == trials
    assert [e["event"] for e in _events_at(root)][-1] == "resume"  # nothing else appended
    assert OptimizationResult.from_file(root / "result.json") == again


async def test_a_smaller_budget_after_the_pick_skips_the_test_report(tmp_path):
    original_optimizer = _default_search(tmp_path / "runs")
    original = await original_optimizer.run_async()
    copy = _killed(original_optimizer._run.root, tmp_path, lambda events: _index(events, "picked"))
    spent = sum(1 for e in _events_at(copy) if e["event"] == "trial")  # one rollout each, no re-runs here
    with pytest.raises(ValueError, match="already spent"):
        _resume(copy, budget=Budget(max_rollouts=spent - 1))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="budget= would let")  # a cap that allows no more search
        resumed = _resume(copy, budget=Budget(max_rollouts=spent))
    result = await resumed.run_async()
    assert result.test is None
    assert result.test_reason == (
        f"the test report needs 12 rollouts, 0 are left; resume with max_rollouts >= {spent + 12} to run it"
    )
    # Not saved as done: resuming with the budget the reason names runs the test report.
    assert "test" not in [e["event"] for e in _events_at(copy)]
    again = await _resume(copy, budget=Budget(max_rollouts=spent + 12)).run_async()
    assert again.test is not None
    for field in ("champion_id", "objective", "difference", "ratio", "missing"):  # timings differ
        assert getattr(again.test, field) == getattr(original.test, field)


def _two_step_search():
    """`_two_steps`'s search, for resuming its runs (`Scripted` proposers compare equal whatever they propose)."""
    return Search([Block(Scripted(), "temperature"), Block(Scripted(), "manager")], schedule=Joint())


@pytest.mark.parametrize("pattern", ["infrastructure", "mixed"])
async def test_a_test_report_lost_to_infrastructure_is_not_saved_and_resume_runs_it(tmp_path, pattern):
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path)
    test_cases = list(optimizer._case_split.test)
    down = {"sandbox": True}

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 1.0 and case.name in test_cases and down["sandbox"]:
            if pattern == "infrastructure" or case.name == test_cases[0]:
                raise InfrastructureError("sandbox down")
            raise ValueError("crashes on test")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    assert result.best.values["temperature"] == 1.0 and result.test is None
    n = 6 if pattern == "infrastructure" else 1
    assert result.test_reason == (
        f"every test trial of the champion failed, {n} on infrastructure; resume with max_rollouts >= "
        f"{result.rollouts_spent + n} to run them again"
    )
    root = optimizer._run.root
    assert "test" not in [e["event"] for e in _events_at(root)]

    # The sandbox is back: a resume re-runs the trials left infrastructure_invalid and saves the report.
    down["sandbox"] = False
    resumed = Optimizer.resume(
        root,
        experiment=_experiment(evaluators=[Temperature()]),
        factory=create_agent,
        rollout_fn=rollout,
        search=_two_step_search(),
    )
    again = await resumed.run_async()
    assert again.best.id == result.best.id and again.test is not None
    assert again.test.missing == {result.best.id: 0, result.baseline_id: 0}  # an agent failure counts, at worst
    # Only the trials left infrastructure_invalid ran again; the agent failures stayed final.
    statuses = {
        e["record"]["case_name"]: e["record"]["status"]
        for e in _events_at(root)
        if e["event"] == "trial" and e["record"]["candidate_id"] == result.best.id and e["record"]["split"] == "test"
    }
    assert sorted(statuses.values()) == ["agent_failure"] * (6 - n) + ["completed"] * n
    assert again.rollouts_spent == result.rollouts_spent + n
    assert [e["event"] for e in _events_at(root)][-1] == "test"


async def test_a_test_report_lost_to_infrastructure_at_a_tight_budget_names_the_budget_a_resume_needs(tmp_path):
    # The search spends down to the reserves, and the champion's test re-runs use up most of what is left.
    reserves = _two_steps(final_pick=FinalPick(n_finalists=2))._budget
    budget = Budget(max_rollouts=8 * 3 + reserves.validation_reserve + reserves.test_reserve, max_iterations=None)
    optimizer = _two_steps(
        {"temperature": 0.5},
        {"temperature": 1.0},
        base_dir=tmp_path,
        budget=budget,
        final_pick=FinalPick(n_finalists=2),
    )
    test_cases = set(optimizer._case_split.test)
    down = {"sandbox": True}

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 1.0 and case.name in test_cases and down["sandbox"]:
            raise InfrastructureError("sandbox down")
        return str(temperature)

    def resume(**kwargs):
        return Optimizer.resume(
            optimizer._run.root,
            experiment=_experiment(evaluators=[Temperature()]),
            factory=create_agent,
            rollout_fn=rollout,
            search=_two_step_search(),
            **kwargs,
        )

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    spent = result.rollouts_spent
    assert result.test is None and spent < budget.max_rollouts < spent + 6
    assert result.test_reason == (
        "every test trial of the champion failed, 6 on infrastructure; "
        f"resume with max_rollouts >= {spent + 6} to run them again"
    )

    # The sandbox is back, but the saved budget can't afford the 6 trials: still not saved as done.
    down["sandbox"] = False
    again = await resume().run_async()
    assert again.test is None and again.rollouts_spent == spent
    assert again.test_reason == (
        f"the test report needs 6 rollouts, {budget.max_rollouts - spent} are left; resume with max_rollouts >= "
        f"{spent + 6} to run it"
    )
    assert "test" not in [e["event"] for e in _events_at(optimizer._run.root)]

    # With the budget the reason names, the resume runs them and saves the report.
    final = await resume(budget=Budget(max_rollouts=spent + 6)).run_async()
    assert final.test is not None and final.test.missing == {final.best.id: 0, final.baseline_id: 0}
    assert final.rollouts_spent == spent + 6
    assert [e["event"] for e in _events_at(optimizer._run.root)][-1] == "test"


async def test_a_test_report_is_final_when_a_failed_arm_has_only_agent_failures(tmp_path):
    # The champion crashes on every test case and the baseline's test trials hit infrastructure: re-running
    # the baseline can't give the champion a completed trial, so the outcome is final.
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path)
    test_cases = set(optimizer._case_split.test)

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if case.name in test_cases and temperature == 1.0:
            raise ValueError("crashes on test")
        if case.name in test_cases and temperature == 0.0:
            raise InfrastructureError("sandbox down")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    assert result.test is None
    assert result.test_reason == "every test trial of the baseline failed, so there is nothing to compare"
    assert _names(_events(optimizer))[-1] == "test"


def test_a_ctrl_c_during_the_search_stops_run_and_resume_finishes_it(tmp_path, sigint):
    async def on_iteration(record):
        if record.iteration == 2:
            sigint.send()  # a Ctrl-C once iteration 2 is closed
            await asyncio.sleep(10)  # cancelled; otherwise the search goes on after it

    original_optimizer = _default_search(tmp_path / "original")
    original = original_optimizer.run()
    optimizer = _default_search(tmp_path / "runs", on_iteration=on_iteration)
    with pytest.raises(sigint.error):
        optimizer.run()

    root = optimizer._run.root
    events = _events_at(root)
    names = [e["event"] for e in events]
    assert names.count("iteration_end") == 2 and "stop" not in names and not (root / "result.json").exists()
    assert {e["record"]["split"] for e in events if e["event"] == "trial"} == {"train"}
    resumed_optimizer = _resume(root)
    _assert_same(resumed_optimizer.run(), original)
    assert resumed_optimizer._budget.spent == original_optimizer._budget.spent


@pytest.mark.parametrize("sync", [False, True], ids=["async rollout", "sync rollout"])
def test_a_ctrl_c_during_validation_cancels_the_trial_in_flight_and_resume_runs_it(tmp_path, sigint, sync):
    original_optimizer = _default_search(tmp_path / "original")
    original = original_optimizer.run()
    validation = set(original_optimizer._case_split.validation)
    first, release, returned = threading.Lock(), threading.Event(), threading.Event()

    def interrupts(case):
        """Send a Ctrl-C from the first validation trial only."""
        if case.name in validation and first.acquire(blocking=False):
            sigint.send()
            return True
        return False

    async def async_rollout(agent_factory, case):
        if interrupts(case):
            await asyncio.sleep(10)  # in flight when the run is cancelled
        return await _temperature_rollout(agent_factory, case)

    def sync_rollout(agent_factory, case):
        if interrupts(case):
            release.wait(10)  # in flight when the run is cancelled; its thread can't be stopped
            returned.set()
        _, temperature = agent_factory()
        return str(temperature)

    optimizer = _default_search(tmp_path / "runs", rollout_fn=sync_rollout if sync else async_rollout)
    with pytest.raises(sigint.error):
        optimizer.run()
    if sync:
        assert not returned.is_set()  # `run()` didn't wait for the sync rollout
        release.set()
        assert returned.wait(10)

    root = optimizer._run.root
    events = _events_at(root)
    names = [e["event"] for e in events]
    assert "finalists" in names and "picked" not in names and not (root / "result.json").exists()

    def validation_trials(events):
        return sum(1 for e in events if e["event"] == "trial" and e["record"]["split"] == "validation")

    assert validation_trials(events) < validation_trials(_events(original_optimizer))  # the cancelled one isn't saved
    with pytest.warns(UserWarning, match="differently named agent"):  # resumed with the plain rollout on purpose
        resumed_optimizer = _resume(root)
    _assert_same(resumed_optimizer.run(), original)
    assert resumed_optimizer._budget.spent == original_optimizer._budget.spent


async def test_a_budget_that_would_search_further_after_the_search_stopped_warns(tmp_path):
    original_optimizer = _default_search(tmp_path / "runs")
    original = await original_optimizer.run_async()
    root = original_optimizer._run.root
    copy = _killed(root, tmp_path, lambda events: _index(events, "picked"))
    with pytest.warns(UserWarning, match=r"search already stopped \(iterations\).*only finishes the end phase"):
        _resume(copy, budget=Budget(max_iterations=7))

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="budget= would let")
        _resume(root)  # no budget given
        _resume(root, budget=Budget(max_iterations=6))  # the cap it stopped under: it would stop where it did
        tiny = _resume(root, budget=Budget(max_rollouts=1))  # finished: no cover check, nothing left to spend
    _assert_same(await tiny.run_async(), original)


async def test_the_budget_warning_counts_only_the_search_spend_against_a_budget_stop(tmp_path):
    original_optimizer = _default_search(tmp_path / "runs", budget=Budget(max_rollouts=92))
    original = await original_optimizer.run_async()
    assert original.stop_reason == "budget"
    tracker = original_optimizer._budget
    searched = sum(1 for e in _events(original_optimizer) if e["event"] == "trial" and e["record"]["split"] == "train")
    enough = searched + tracker.validation_reserve + tracker.test_reserve + 8  # one more candidate's 8 train cases
    root = original_optimizer._run.root

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="budget= would let")
        _resume(root, budget=Budget(max_rollouts=enough - 1))
    with pytest.warns(UserWarning, match=r"search already stopped \(budget\)"):
        _resume(root, budget=Budget(max_rollouts=enough))


async def test_no_budget_warning_when_the_search_stopped_for_another_reason(tmp_path):
    optimizer = _default_search(tmp_path)
    await optimizer.run_async()
    root = optimizer._run.root
    events = _events_at(root)
    for event in events:
        if event["event"] == "stop":
            event["reason"] = "exhausted"  # a bigger budget wouldn't have searched further either
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="budget= would let")
        _resume(root, budget=Budget(max_rollouts=5000, max_iterations=50))


async def test_a_run_another_optimizer_is_writing_cannot_be_run_again(tmp_path):
    writer = _default_search(tmp_path)
    await writer._start()  # scored the baseline; still writing
    root = writer._run.root
    resumed = _resume(root)  # resume only reads
    before = _events_at(root)

    with pytest.raises(RuntimeError, match="is open for writing in another process"):
        await resumed.run_async()
    assert _events_at(root) == before  # refused before writing anything

    writer._run.events.append("label", candidate_id="x", label={})  # the writer goes on, then stops
    writer._run.unlock()
    with pytest.raises(RuntimeError, match="events.jsonl changed after it was read"):
        await resumed.run_async()  # what resume read is stale


async def test_a_run_whose_on_iteration_raised_resumes_in_the_same_process(tmp_path):
    def broken(record):
        raise RuntimeError("callback bug")

    optimizer = _default_search(tmp_path, on_iteration=broken)
    with pytest.raises(RuntimeError, match="callback bug"):
        await optimizer.run_async()

    result = await _resume(optimizer._run.root).run_async()  # the failed run released its lock
    assert result.stop_reason == "iterations"


def test_resume_checks_the_cases_and_the_config(tmp_path):
    optimizer = _default_search(tmp_path)
    optimizer._setup()
    root = optimizer._run.root
    with pytest.raises(ValueError, match="^cases changed"):
        Optimizer.resume(
            root,
            experiment=_experiment(n=19, evaluators=[Temperature()]),
            factory=create_agent,
            rollout_fn=_temperature_rollout,
        )
    with pytest.raises(ValueError, match="^objective changed"):
        _resume(root, objective=Minimize("Temperature"))


async def test_a_run_stopped_at_setup_cannot_resume(tmp_path):
    optimizer = _quiet(base_dir=tmp_path, rollout_fn=_rollout(InfrastructureError("sandbox down")), rerun_passes=0)
    with pytest.raises(RuntimeError, match="unmeasured"):
        await optimizer.run_async()
    with pytest.raises(RuntimeError, match="stopped at setup"):
        Optimizer.resume(optimizer._run.root, experiment=_experiment(), factory=create_agent, rollout_fn=_rollout())


@tunable(
    manager=choice(component(Window, size=Int(1, 100)), component(Summary)),
    temperature=OneOf(0.0, 0.5, 1.0),
)
def renamed_agent(manager, temperature):
    return (manager, temperature)


async def _renamed_rollout(agent_factory, case):
    return await _temperature_rollout(agent_factory, case)


def test_resume_warns_when_the_agent_is_named_differently(tmp_path):
    optimizer = _default_search(tmp_path)
    optimizer._setup()
    root = optimizer._run.root
    experiment = _experiment(evaluators=[Temperature()])

    with pytest.warns(UserWarning, match=r"factory '[\w.]+:create_agent' -> '[\w.]+:renamed_agent'"):
        Optimizer.resume(root, experiment=experiment, factory=renamed_agent, rollout_fn=_temperature_rollout)
    with pytest.warns(UserWarning, match=r"\(rollout_fn '[\w.]+:_temperature_rollout' -> '[\w.]+:_renamed_rollout'\)"):
        Optimizer.resume(root, experiment=experiment, factory=create_agent, rollout_fn=_renamed_rollout)

    # The script that started the run, run as __main__, may be imported under its name to resume it.
    log = root / "events.jsonl"
    events = _events_at(root)
    agent = events[0]["config"]["agent"]
    agent.update({k: "__main__:" + v.partition(":")[2] for k, v in agent.items() if v is not None})
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="resuming run .* differently named agent")
        _resume(root)


def test_same_name_compares_module_and_qualname_but_lets_main_match_any_module():
    assert _same_name("agents:create_agent", "agents:create_agent")
    assert not _same_name("agents:create_agent", "agents:create_agent_v2")
    assert not _same_name("agents:create_agent", "other:create_agent")
    assert _same_name("__main__:create_agent", "optimize:create_agent")
    assert _same_name("optimize:create_agent", "__main__:create_agent")
    assert not _same_name("__main__:create_agent", "optimize:other_agent")
    assert _same_name(None, None)
    assert not _same_name(None, "agents:rollout") and not _same_name("agents:rollout", None)


def test_resume_warns_when_numpy_or_optuna_changed(tmp_path):
    optimizer = _default_search(tmp_path)
    optimizer._setup()
    log = optimizer._run.root / "events.jsonl"
    events = _events_at(optimizer._run.root)
    events[0]["config"]["code"]["optuna"] = "0.0.1"
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.warns(UserWarning, match=r"optuna 0\.0\.1 -> "):
        _resume(optimizer._run.root)


# a proposer serves one optimizer


def _user_tpe_search():
    """A search with a user-made proposer (default blocks get fresh proposers on every resolve)."""
    return Search([Block(TPE(), "temperature")])


async def test_resuming_with_the_stopped_runs_proposers_raises_and_a_new_search_resumes_exactly(tmp_path):
    uninterrupted = await _default_search(tmp_path / "full", search=_user_tpe_search()).run_async()

    def stop_after_three(record):
        if record.iteration == 3:
            raise RuntimeError("callback bug")

    search = _user_tpe_search()
    stopped = _default_search(tmp_path / "stopped", search=search, on_iteration=stop_after_three)
    with pytest.raises(RuntimeError, match="callback bug"):
        await stopped.run_async()
    root = stopped._run.root
    log = (root / "events.jsonl").read_text()
    # Its TPE holds the three iterations' history; resume would replay it on top.
    with pytest.raises(ValueError, match=r"^TPE \(block 'numbers'\) is bound to another Optimizer"):
        _resume(root, search=search)
    assert (root / "events.jsonl").read_text() == log
    fresh = _user_tpe_search()
    resumed = await _resume(root, search=fresh).run_async()
    _assert_same(resumed, uninterrupted)
    # The resumed optimizer bound its proposers before replaying the history into them.
    with pytest.raises(ValueError, match="is bound to another Optimizer"):
        _resume(root, search=fresh)


async def test_a_proposer_another_optimizer_ran_with_is_rejected_before_anything_is_written(tmp_path):
    search = _user_tpe_search()
    first = _default_search(tmp_path / "first", search=search, budget=Budget(max_iterations=1))
    second = _default_search(tmp_path / "second", search=search)  # constructing both is fine: neither ran
    await first.run_async()
    with pytest.raises(ValueError, match="is bound to another Optimizer"):
        await second.run_async()
    assert not (tmp_path / "second").exists()
    # Another Search holding the same instance is caught at construction.
    with pytest.raises(ValueError, match="is bound to another Optimizer"):
        _default_search(tmp_path / "third", search=Search([Block(search.blocks[0].proposer, "temperature")]))


async def test_two_optimizers_started_at_once_on_one_search_bind_it_to_one(tmp_path):
    search = _user_tpe_search()
    optimizers = [_default_search(tmp_path / str(i), search=search, budget=Budget(max_iterations=1)) for i in range(2)]
    outcomes = await asyncio.gather(*(o.run_async() for o in optimizers), return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    results = [o for o in outcomes if not isinstance(o, BaseException)]
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert "is bound to another Optimizer" in str(errors[0])
    assert len(results) == 1 and results[0].stop_reason == "iterations"
    # The proposers are claimed before anything is written, so the second spent nothing and left no folder.
    assert sum(o._run is not None for o in optimizers) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_two_threads_running_optimizers_on_one_search_bind_it_to_one(tmp_path):
    search = _user_tpe_search()
    optimizers = [_default_search(tmp_path / str(i), search=search, budget=Budget(max_iterations=1)) for i in range(2)]
    barrier = threading.Barrier(2)
    outcomes = []

    def run(optimizer):
        barrier.wait()
        try:
            outcomes.append(optimizer.run())
        except ValueError as err:
            outcomes.append(err)

    threads = [threading.Thread(target=run, args=(o,)) for o in optimizers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(isinstance(o, ValueError) for o in outcomes) == 1
    assert sum(not isinstance(o, ValueError) for o in outcomes) == 1


async def test_running_one_optimizer_twice_raises_and_points_to_resume(tmp_path):
    def stop_after_two(record):
        if record.iteration == 2:
            raise RuntimeError("callback bug")

    optimizer = _default_search(tmp_path, search=_user_tpe_search(), on_iteration=stop_after_two)
    with pytest.raises(RuntimeError, match="callback bug"):
        await optimizer.run_async()
    log = (optimizer._run.root / "events.jsonl").read_text()
    with pytest.raises(RuntimeError, match=r"already started a run .*Optimizer\.resume"):
        await optimizer.run_async()
    with pytest.raises(RuntimeError, match="already started a run"):
        optimizer.run()
    assert (optimizer._run.root / "events.jsonl").read_text() == log


async def test_an_optimizer_that_stops_at_setup_leaves_its_proposers_free(tmp_path):
    search = _user_tpe_search()
    broken = _quiet(
        base_dir=tmp_path, search=search, rollout_fn=_rollout(InfrastructureError("sandbox down")), rerun_passes=0
    )
    with pytest.raises(RuntimeError, match="unmeasured"):
        await broken.run_async()
    fixed = _quiet(base_dir=tmp_path, search=search, rollout_fn=_rollout(), budget=Budget(max_iterations=1))
    assert (await fixed.run_async()).stop_reason == "iterations"


async def test_a_run_cancelled_during_its_baseline_leaves_its_proposers_free(tmp_path):
    search = _user_tpe_search()
    started, never = asyncio.Event(), asyncio.Event()

    async def blocked(agent_factory, case):
        started.set()
        await never.wait()

    run = asyncio.create_task(_quiet(base_dir=tmp_path / "first", search=search, rollout_fn=blocked).run_async())
    await started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert "_bound_to" not in vars(search.blocks[0].proposer)  # as it was before the run
    fixed = _quiet(base_dir=tmp_path / "second", search=search, rollout_fn=_rollout(), budget=Budget(max_iterations=1))
    assert (await fixed.run_async()).stop_reason == "iterations"


async def test_a_run_whose_directory_could_not_be_created_can_be_run_again(tmp_path, monkeypatch):
    create, calls = RunDirectory.create, []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("base_dir is read-only")
        return create(*args, **kwargs)

    monkeypatch.setattr(RunDirectory, "create", failing_once)
    search = _user_tpe_search()
    optimizer = _quiet(base_dir=tmp_path, search=search, rollout_fn=_rollout(), budget=Budget(max_iterations=1))
    with pytest.raises(PermissionError):
        await optimizer.run_async()
    assert optimizer.run_dir is None and "_bound_to" not in vars(search.blocks[0].proposer)
    assert (await optimizer.run_async()).stop_reason == "iterations"


async def test_a_proposer_whose_config_reads_its_own_attributes_can_run(tmp_path):
    class Fixed(Proposer):
        kinds = frozenset({"one_of"})

        def __init__(self, value=0.5):
            self.value = value

        async def propose(self, request):
            return [Patch(updates={"temperature": self.value}, rationale="fixed")][: request.n]

        def observe(self, observation):
            pass

        def config(self):
            return dict(vars(self))

    result = await _searching(Block(Fixed(), "temperature"), base_dir=tmp_path).run_async()
    assert json.loads((result.run_dir / "run.json").read_text())["search"]["blocks"][0]["config"] == {"value": 0.5}


async def test_a_frozen_proposer_is_bound_and_a_copy_of_a_bound_one_is_rejected(tmp_path):
    @dataclasses.dataclass(frozen=True)
    class Silent(Proposer):
        kinds = frozenset({"one_of"})

        async def propose(self, request):
            return []

        def observe(self, observation):
            pass

    silent = Silent()
    unused_copy = deepcopy(silent)
    await _searching(Block(silent, "temperature"), base_dir=tmp_path / "first").run_async()
    with pytest.raises(ValueError, match=r"^Silent \(block 'numbers'\) is bound to another Optimizer"):
        _searching(Block(silent, "temperature"), base_dir=tmp_path / "second")
    with pytest.raises(ValueError, match="is bound to another Optimizer"):
        _searching(Block(deepcopy(silent), "temperature"), base_dir=tmp_path / "third")
    _searching(Block(unused_copy, "temperature"), base_dir=tmp_path / "fourth")


# fixes from the gap review and #532


class _OneStep(Schedule):
    """A custom schedule: one step of the first block, as given, then None."""

    name = "one_step"

    def __init__(self, **step):
        self.step = step

    def next(self, ctx):
        if ctx.past:
            return None
        block = ctx.blocks[0]
        fields = {"n_candidates": 1, "parents": None, "round": None, "role": "inner", "observe_at_round_end": ()}
        return Step(blocks=(block,), **{**fields, **self.step})


async def test_a_proposers_spend_on_no_patch_is_counted_under_its_block(tmp_path):
    spend = Usage(
        input_tokens=90,
        output_tokens=10,
        tokens=100,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=1,
        seconds=1.0,
        models=["m"],
    )

    class Spender(Scripted):
        async def propose(self, request):
            request.log("proposer_usage", usage=spend)  # the agent ran, nothing usable came out
            return []

    optimizer = _searching(Block(Spender(), "temperature"), schedule=Alternate(), base_dir=tmp_path)
    await _search(optimizer)
    logged = [e for e in _events(optimizer) if e["event"] == "proposer_usage"]
    assert logged and _proposer_tokens(optimizer._run.root) == {"numbers": 100 * len(logged)}


async def test_a_proposer_usage_event_without_a_usage_is_refused_before_it_is_written(tmp_path):
    class Miscounting(Scripted):
        async def propose(self, request):
            request.log("proposer_usage", usage={"tokens": 1})
            return []

    optimizer = _searching(Block(Miscounting(), "temperature"), schedule=Alternate(), base_dir=tmp_path)
    with pytest.raises(ProposerError, match="needs usage= a Usage"):
        await _search(optimizer)
    events = _events(optimizer)
    error = next(e for e in events if e["event"] == "proposer_error")
    assert error["error_type"] == "TypeError" and "needs usage= a Usage" in error["error"]
    assert "proposer_usage" not in _names(events)


async def test_a_concurrent_proposer_that_raises_cancels_its_sibling_calls(tmp_path):
    started, finished = [], []

    class Racing(Scripted):
        concurrent = True

        async def propose(self, request):
            started.append(request.n)
            if len(started) == 1:
                raise RuntimeError("model gone")
            await asyncio.sleep(0.5)
            request.log("screened", key="temperature", reason="too late")  # must never happen
            finished.append(request.n)
            return []

    optimizer = _searching(
        Block(Racing(), "temperature"), schedule=Alternate(), base_dir=tmp_path, proposals_per_iteration=3
    )
    with pytest.raises(ProposerError, match="model gone"):
        await _search(optimizer)
    events = _events(optimizer)
    assert len(started) == 3 and finished == []
    assert "screened" not in _names(events) and events[-1]["event"] == "proposer_error"


async def test_an_openers_duplicate_doesnt_drop_the_rest_of_its_batch(tmp_path):
    proposer = Scripted({"temperature": 0.0}, {"temperature": 0.5})  # the first repeats the baseline
    optimizer = _searching(
        Block(proposer, "temperature"),
        schedule=_OneStep(n_candidates=2, round=(0,), role="opener", observe_at_round_end=("numbers",)),
        base_dir=tmp_path,
    )
    await _search(optimizer)
    # Both patches recorded and each observed exactly once (an opener step's patches are told at the round's end).
    assert [p.outcome for p in optimizer._iterations[0].proposals] == ["duplicate", "feasible"]
    assert sorted(o.values["temperature"] for o in proposer.observed) == [0.0, 0.5]


async def test_a_step_with_no_possible_parents_raises(tmp_path):
    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}), "temperature"), schedule=_OneStep(parents=()), base_dir=tmp_path
    )
    await optimizer._start()
    with pytest.raises(ValueError, match="_OneStep.next returned a step with no possible parents"):
        await optimizer._run_search()


async def test_a_run_killed_during_validation_resumes_with_its_own_budget(tmp_path):
    # Stops on the budget, so the search leaves only the reserve: 8 baseline + 48 reserve + 3 steps of 16.
    original_optimizer = _default_search(tmp_path / "runs", budget=Budget(max_rollouts=104))
    original = await original_optimizer.run_async()
    assert original.stop_reason == "budget"
    copy = _killed(
        original_optimizer._run.root,
        tmp_path,
        lambda events: _index(events, "trial", nth=3, after=_index(events, "finalists")),
    )
    resumed_optimizer = _resume(copy)  # no budget given: the saved one still covers what is left
    resumed = await resumed_optimizer.run_async()
    _assert_same(resumed, original)
    assert resumed_optimizer._budget.spent == original_optimizer._budget.spent <= 104


async def test_resume_rebuilds_scores_as_the_closed_iterations_left_them(tmp_path):
    # temperature 0.5's trials hit infrastructure the first time, so it is ineligible until step 2 re-scores it.
    failing = {"left": 8}

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 0.5 and failing["left"] > 0:
            failing["left"] -= 1
            raise InfrastructureError("sandbox down")
        return str(temperature)

    def build(base_dir, proposer):
        return _optimizer(
            _experiment(evaluators=[Temperature()]),
            rollout_fn=rollout,
            search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
            max_workers=64,
            rerun_passes=0,
            proposals_per_iteration=1,
            base_dir=base_dir,
            budget=Budget(max_iterations=3),
        )

    original_optimizer = build(
        tmp_path / "runs", Scripted({"temperature": 0.5}, {"temperature": 0.5}, {"temperature": 1.0})
    )
    original = await original_optimizer.run_async()
    rescored = original.iterations[1].proposals[0]
    assert rescored.outcome == "feasible" and original.iterations[0].proposals[0].outcome == "ineligible"
    # Killed right after step 2 re-scored it, before the step closed.
    copy = _killed(
        original_optimizer._run.root,
        tmp_path,
        lambda events: _index(events, "score", after=_index(events, "iteration_end", nth=1)),
    )
    resumed_optimizer = Optimizer.resume(
        copy,
        experiment=_experiment(evaluators=[Temperature()]),
        factory=create_agent,
        rollout_fn=rollout,
        search=Search(
            [Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"), Block(Scripted(), "manager")],
            schedule=Joint(),
        ),
    )
    await resumed_optimizer._start()
    assert not resumed_optimizer._train[rescored.candidate_id].is_eligible  # as step 1 left it


def test_reports_keep_a_re_judges_earlier_scores():
    case = Case(name="a", input="question a")
    split = CaseSplit(train=("a",), validation=(), test=())
    record = TrialRecord(
        candidate_id="c",
        case_name="a",
        run_index=0,
        split="train",
        status=TrialStatus.COMPLETED,
        error_type="RuntimeError",
        error="can't dump",  # Experiment failed on the case when "Second" was judged again
        attempts=2,
        previous_attempts=[],
        output="x",
        trajectory=None,
        interactions=None,
        environment_state=None,
        evaluations={"Score": EvaluationOutput(score=1.0, test_pass=True, reason="right")},
        details={"Score": []},
        evaluator_errors={},
        usage=Usage.zero(),
        judge_usage={},
        rollouts=0,
    )
    evaluators = [Score(), Score(name="Second")]

    report = _reports(
        {"c": {("a", 0): record}}, {"c": {("a", 0): None}}, cases=[case], evaluators=evaluators, split=split
    )["c"]["train"]

    assert report.scores == [1.0, 0.0]
    assert report.reasons == ["right", "An error occurred: can't dump"]


async def test_an_ineligible_candidate_a_re_score_cant_change_is_a_duplicate(tmp_path):
    # temperature 0.5's output can't be serialized for its span, so its trials are evaluator_invalid for good.
    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        return {"output": {(0, 1): "x"}} if temperature == 0.5 else str(temperature)

    proposer = Scripted({"temperature": 0.5}, {"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
        max_workers=64,
        proposals_per_iteration=1,
        base_dir=tmp_path,
        budget=Budget(max_iterations=2),
    )
    result = await optimizer.run_async()

    first, second = result.iterations[0].proposals, result.iterations[1].proposals
    assert [p.outcome for p in first] == ["ineligible"]
    assert [p.outcome for p in second] == ["duplicate", "feasible"]  # 0.5 again isn't scored, the redraw is
    assert second[0].candidate_id == first[0].candidate_id
    assert not any(e["event"] == "propose" and e["rescore"] for e in _events(optimizer))
    assert [o.outcome for o in proposer.observed] == ["ineligible", "duplicate", "feasible"]


class _FailsOnHalf(Evaluator):
    """Scores the temperature, but raises on temperature 0.5's output the first `failures` times."""

    def __init__(self, failures):
        super().__init__(name="Temperature")
        self.failures, self.raised = failures, 0

    async def evaluate_async(self, evaluation_case):
        if evaluation_case.actual_output == "0.5" and self.raised < self.failures:
            self.raised += 1
            raise ValueError("bad judge output")
        return [EvaluationOutput(score=float(evaluation_case.actual_output), test_pass=True, reason="t")]


def _repeating_half(tmp_path, evaluator, max_iterations):
    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        return str(temperature)

    proposer = Scripted({"temperature": 0.5}, {"temperature": 0.5})
    optimizer = _optimizer(
        _experiment(evaluators=[evaluator]),
        rollout_fn=rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
        max_workers=64,
        proposals_per_iteration=1,
        rerun_passes=0,
        base_dir=tmp_path,
        budget=Budget(max_iterations=max_iterations),
    )
    return optimizer, proposer


async def test_an_ineligible_repeat_with_a_failed_evaluator_is_judged_again(tmp_path):
    optimizer, _ = _repeating_half(tmp_path, _FailsOnHalf(failures=8), max_iterations=2)  # every 0.5 trial once
    result = await optimizer.run_async()

    first, second = result.iterations[0].proposals, result.iterations[1].proposals
    assert [p.outcome for p in first] == ["ineligible"]
    assert [p.outcome for p in second] == ["feasible"]  # re-judged on the saved outputs, now measured
    assert second[0].candidate_id == first[0].candidate_id
    assert any(e["event"] == "propose" and e["rescore"] for e in _events(optimizer))
    assert result.iterations[1].rollouts_spent == 0  # judges only


async def test_a_re_judge_that_leaves_the_candidate_unmeasured_saves_nothing(tmp_path):
    evaluator = _FailsOnHalf(failures=10**6)  # fails on 0.5's output every time
    optimizer, proposer = _repeating_half(tmp_path, evaluator, max_iterations=5)
    result = await optimizer.run_async()

    first, second = result.iterations[0].proposals, result.iterations[1].proposals
    assert [p.outcome for p in first] == ["ineligible"]
    assert [p.outcome for p in second] == ["duplicate"]  # judged again, still unmeasured: nothing saved
    assert "still unmeasured after its re-judge" in proposer.observed[-1].reason
    assert any(e["event"] == "propose" and e["rescore"] for e in _events(optimizer))
    assert len(result.iterations) == 2 and result.stop_reason == "exhausted"  # the step saved nothing
    assert evaluator.raised == 16  # 8 train trials, judged twice


async def test_a_failed_record_kept_on_resume_names_no_transcript_of_a_killed_rerun(tmp_path):
    # temperature 0.5's trials hit infrastructure in step 1; step 2 re-runs them, saving transcripts, and is killed.
    failing = {"left": 8}

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 0.5 and failing["left"] > 0:
            failing["left"] -= 1
            raise InfrastructureError("sandbox down")
        return {"output": str(temperature), "trajectory": [case.name, temperature]}

    def search(*updates):
        return Search([Block(Scripted(*updates), "temperature"), Block(Scripted(), "manager")], schedule=Joint())

    original = _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=rollout,
        search=search({"temperature": 0.5}, {"temperature": 0.5}, {"temperature": 1.0}),
        max_workers=64,
        rerun_passes=0,
        proposals_per_iteration=1,
        base_dir=tmp_path / "runs",
        budget=Budget(max_iterations=3),
    )
    await original.run_async()
    copy = _killed(
        original._run.root, tmp_path, lambda events: _index(events, "score", after=_index(events, "iteration_end"))
    )
    # Without the killed step's `draw` event, the resumed step 2 asks its proposers again (as a resumed step that
    # is wider than the killed one does) and gets 1.0 instead, so 0.5 keeps step 1's failed records.
    lines = (copy / "events.jsonl").read_text().splitlines()
    draws = [i for i, line in enumerate(lines) if json.loads(line)["event"] == "draw"]
    del lines[draws[-1]]
    (copy / "events.jsonl").write_text("\n".join(lines) + "\n")
    result = await Optimizer.resume(
        copy,
        experiment=_experiment(evaluators=[Temperature()]),
        factory=create_agent,
        rollout_fn=rollout,
        search=search({"temperature": 1.0}),
    ).run_async()

    half = next(c for c, view in result.candidates.items() if view.values["temperature"] == 0.5)
    last = {}
    for event in _events_at(copy):
        if event["event"] == "trial" and event["record"]["candidate_id"] == half:
            last[(event["record"]["case_name"], event["record"]["run_index"])] = event["record"]
    # The last trial events are the killed step's re-runs, which saved transcripts.
    assert last and all(saved["transcript"] is not None for saved in last.values())
    cases = [row for row in result.to_rows("case") if row["candidate_id"] == half]
    assert len(cases) == len(last) and all(row["status"] == "infrastructure_invalid" for row in cases)
    assert all(row["transcript"] is None for row in cases)
    rows = result.reports[half]["train"].cases
    assert len(rows) == len(cases) and all(row["transcript"] is None for row in rows)


async def test_max_iterations_counts_candidate_slots_not_steps(tmp_path):
    # temperature has 3 options: once each is measured, every numbers step is all duplicates and costs no slot.
    proposer = Scripted({"temperature": 0.5}, {"temperature": 0.0}, {"temperature": 1.0}, *[{"temperature": 0.5}] * 30)
    optimizer = _searching(
        Block(proposer, "temperature"), schedule=Alternate(), base_dir=tmp_path, budget=Budget(max_iterations=2)
    )
    reason, _ = await _search(optimizer)
    # The first step saves 0.5 and 1.0 (4 slots of the 8); every later step repeats a measured value and costs no
    # slot, so the cap doesn't stop the run after 2 steps: Alternate does, after a full cycle that saved nothing.
    assert _slots_used(optimizer._iterations) == 4 and len(optimizer._iterations) > 2
    assert reason == "exhausted"


# the results


class Flaky(Evaluator):
    """Raises on the case named `fails_on`, so it is in that trial's evaluator_errors (not a goal metric)."""

    fails_on = ""

    async def evaluate_async(self, evaluation_case):
        if evaluation_case.name == self.fails_on:
            raise ValueError("flaky judge")
        return [EvaluationOutput(score=1.0, test_pass=True, reason="fine")]


def _crashing_temperature_rollout(crashes_on):
    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if case.name == crashes_on:
            raise ValueError("boom")
        return str(temperature)

    return rollout


def _uniform(rows):
    return all(list(row) == list(rows[0]) for row in rows)


async def test_rows_cover_every_proposal_iteration_and_trial(tmp_path):
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 7.0}, {"temperature": 1.0}, base_dir=tmp_path)
    result = await optimizer.run_async()

    proposals = result.to_rows("proposal")
    assert _uniform(proposals)
    assert [(r["iteration"], r["outcome"], r["number"]) for r in proposals] == [
        (0, "feasible", 0),
        (1, "feasible", 1),
        (2, "invalid", None),
        (2, "feasible", 2),
    ]
    baseline, first, invalid, second = proposals
    assert baseline["block"] is None and baseline["parent_id"] is None and baseline["rollouts"] == 8
    assert baseline["reason"] is not None and baseline["train.Temperature"] == 0.0
    assert invalid["reason"] and invalid["candidate_id"] is None and invalid["values"] is None
    assert invalid["is_from_fit"] is None and invalid["rollouts"] == 0
    assert second["champion"] and second["finalist"] and second["rank"] == 1
    assert second["changed_keys"] == ["temperature"] and second["values"]["temperature"] == 1.0
    assert first["finalist"] and not first["champion"] and first["validation_objective"] == pytest.approx(0.5)
    assert first["trials"] == 8 and first["failures"] == 0 and first["on_front"] is None
    assert "estimate.Temperature" in first and "train_ci95.tokens" in first

    iterations = result.to_rows("iteration")
    assert _uniform(iterations) and [r["iteration"] for r in iterations] == [1, 2]
    assert [r["rollouts_total"] for r in iterations] == [16, 24]
    assert iterations[1]["invalid"] == 1 and iterations[1]["feasible"] == 1
    assert iterations[1]["best_after"][0] == result.best.id and iterations[1]["best_objective"] is not None

    cases = result.to_rows("case")
    assert _uniform(cases)
    trials = sum(len(records) for records in optimizer._records.values())
    assert len(cases) == trials == 8 * 3 + 6 * 3 + 6 * 2
    assert [r["number"] for r in cases] == sorted(r["number"] for r in cases)
    champion = [r for r in cases if r["candidate_id"] == result.best.id]
    assert [r["split"] for r in champion] == ["train"] * 8 + ["validation"] * 6 + ["test"] * 6
    assert all(
        r["objective"] == 1.0 and r["score.Temperature"] == 1.0 and r["applicable.Temperature"] for r in champion
    )


async def test_reports_match_what_experiment_returns_for_the_same_trials(tmp_path):
    train = list(_optimizer()._case_split.train)  # the same seed and cases give the same split
    crashes_on, Flaky.fails_on = train[1], train[3]  # one crashed task, one failed evaluator
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Flaky(name="Flaky")]),
        rollout_fn=_crashing_temperature_rollout(crashes_on),
        objective=Maximize("Temperature"),
        search=Search(
            [Block(Scripted({"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")], schedule=Joint()
        ),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    result = await optimizer.run_async()
    assert list(optimizer._case_split.train) == train

    def task(case):
        if case.name == crashes_on:
            raise ValueError("boom")
        return "0.0"

    cases = {case.name: case for case in optimizer._cases}
    expected = await Experiment(
        cases=[cases[name] for name in train], evaluators=[Temperature(), Flaky(name="Flaky")]
    ).run_evaluations_async(task)
    expected = json.loads(expected.model_dump_json())
    report = result.reports[result.baseline_id]["train"]
    assert report.overall_score == pytest.approx(expected["overall_score"])
    assert report.scores == expected["scores"] and report.test_passes == expected["test_passes"]
    assert all(row["transcript"] is None for row in report.cases)  # this rollout saves no trajectory
    tags = ("run_index", "transcript")
    assert [{k: v for k, v in row.items() if k not in tags} for row in report.cases] == expected["cases"]
    # Experiment keeps the evaluator's message; the trial record keeps only its type.
    reasons = [r.replace("flaky judge", "ValueError") for r in expected["reasons"]]
    assert report.reasons == reasons
    assert set(result.reports[result.best.id]) == {"train", "validation", "test"}


async def test_reports_match_experiment_for_a_case_it_failed_on_after_the_task_returned():
    case = Case(name="a", input="question a")

    async def task(case):
        return {"output": {(0, 1): "x"}}  # Experiment can't serialize it for its span

    expected = await Experiment(cases=[case], evaluators=[Score()]).run_evaluations_async(task)
    record = TrialRecord(
        candidate_id="c",
        case_name="a",
        run_index=0,
        split="train",
        status=TrialStatus.EVALUATOR_INVALID,
        error_type="TypeError",
        error=expected.reasons[0].removeprefix("An error occurred: "),
        attempts=1,
        previous_attempts=[],
        output=None,
        trajectory=None,
        interactions=None,
        environment_state=None,
        evaluations={},
        details={},
        evaluator_errors={},
        usage=Usage.zero(),
        judge_usage={},
        rollouts=1,
    )

    split = CaseSplit(train=("a",), validation=(), test=())
    report = _reports(
        {"c": {("a", 0): record}}, {"c": {("a", 0): None}}, cases=[case], evaluators=[Score()], split=split
    )["c"]["train"]

    assert expected.reasons[0].startswith("An error occurred: keys must be str")
    assert (report.scores, report.test_passes, report.reasons) == (
        expected.scores,
        expected.test_passes,
        expected.reasons,
    )
    assert [{k: v for k, v in row.items() if k not in ("run_index", "transcript")} for row in report.cases] == (
        expected.cases
    )


async def _listing_rollout(agent_factory, case):
    """Returns its own trajectory, the case and the temperature, so where it is read back can be checked."""
    _, temperature = agent_factory()
    return {"output": str(temperature), "trajectory": [case.name, str(temperature)]}


async def _agent_rollout(agent_factory, case):
    """A real agent on a scripted model that answers with the temperature, so each trial has a `Session`."""
    _, temperature = agent_factory()
    agent = Agent(model=ScriptedModel([str(temperature)]), callback_handler=None)
    return str(await agent.invoke_async(case.input)).strip()


def _texts(session):
    """The text of each message the session's last model call was sent or answered."""
    last = [s for t in session.traces for s in t.spans if isinstance(s, InferenceSpan)][-1]
    return [block.text for message in last.messages for block in message.content]


def _two_step_optimizer(proposer, base_dir, rollout_fn=_listing_rollout):
    return _optimizer(
        _experiment(evaluators=[Temperature()]),
        rollout_fn=rollout_fn,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=base_dir,
    )


def _assert_parents_trajectories(requests, train):
    assert requests
    for request in requests:
        temperature = str(request.parent.values["temperature"])
        assert sorted(t.case_name for t in request.trials) == sorted(train)
        assert all(t.trajectory == [t.case_name, temperature] for t in request.trials)


async def test_trial_records_drop_their_trajectories_and_proposers_read_the_parents_back(tmp_path):
    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _two_step_optimizer(proposer, tmp_path, rollout_fn=_agent_rollout)
    result = await optimizer.run_async()

    assert all(r.trajectory is None for records in optimizer._records.values() for r in records.values())
    inputs = {case.name: case.input for case in optimizer._cases}
    assert proposer.requests
    kept = dict(optimizer._transcripts[result.best.id])
    # Read only now, after the run: from the records as each step started, not as the optimizer holds them later.
    optimizer._records.clear()
    optimizer._transcripts.clear()
    for request in proposer.requests:
        temperature = str(request.parent.values["temperature"])
        assert sorted(t.case_name for t in request.trials) == sorted(optimizer._case_split.train)
        assert all(_texts(t.trajectory) == [inputs[t.case_name], temperature] for t in request.trials)
    # The reports name each trial's session by its transcript.
    rows = result.reports[result.best.id]["test"].cases
    assert rows and all(r["actual_trajectory"] is None for r in rows)
    assert all(r["transcript"] == kept[(r["name"], r["run_index"])]["hash"] for r in rows)
    sessions = [result.load_transcript(r["transcript"]) for r in rows]
    assert all(_texts(session) == [r["input"], "1.0"] for session, r in zip(sessions, rows, strict=True))
    # A whole `Session` dump, as earlier versions saved it, reads back too.
    (result.run_dir / "transcripts" / "0123456789abcdef.json").write_text(sessions[0].model_dump_json())
    assert result.load_transcript("0123456789abcdef") == sessions[0]


async def test_resume_reads_no_transcript_and_proposers_still_get_the_parents(tmp_path, monkeypatch):
    original = _two_step_optimizer(Scripted({"temperature": 0.5}, {"temperature": 1.0}), tmp_path / "runs")
    await original.run_async()
    # Killed before the second iteration's `draw` event, so the resumed run asks its proposers again instead of
    # replaying what the killed attempt drew.
    copy = _killed(
        original._run.root, tmp_path, lambda events: _index(events, "draw", after=_index(events, "iteration_end")) - 1
    )
    # The killed iteration also replaced one of its parent's train trials: the resumed run's proposers must still
    # get the trial the closed iterations left, not the replacement the run directory's index now holds.
    events = _events_at(copy)
    parent = original._iterations[1].parent_id
    first = next(e for e in events if e["event"] == "trial" and e["record"]["candidate_id"] == parent)
    (copy / "transcripts" / "0123456789abcdef.json").write_text('["replaced"]')
    replaced = {**first["record"], "transcript": {"hash": "0123456789abcdef", "format": "list"}}
    with (copy / "events.jsonl").open("a") as log:
        log.write(json.dumps({**first, "iteration": 2, "record": replaced}) + "\n")
    proposer = Scripted({"temperature": 1.0})
    resumed = Optimizer.resume(
        copy,
        experiment=_experiment(evaluators=[Temperature()]),
        factory=create_agent,
        rollout_fn=_listing_rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
    )
    read = []
    load_transcript = resumed._run.load_transcript
    monkeypatch.setattr(resumed._run, "load_transcript", lambda t: read.append(t) or load_transcript(t))

    await resumed._start()
    assert read == [] and len(resumed._records) == 2
    assert all(r.trajectory is None for records in resumed._records.values() for r in records.values())
    await resumed._run_search()
    assert read == []  # nothing has read the requests' trials yet
    _assert_parents_trajectories(proposer.requests, resumed._case_split.train)
    _assert_parents_trajectories(proposer.requests, resumed._case_split.train)
    # Read when first used, once for the step: exactly the parent's train trials, as the closed iterations left them.
    assert sorted(map(str, read)) == sorted(
        str(resumed._transcripts[parent][(c, 0)]) for c in resumed._case_split.train
    )


async def test_a_search_whose_proposers_never_read_trials_reads_no_transcript(tmp_path, monkeypatch):
    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _two_step_optimizer(proposer, tmp_path, rollout_fn=_agent_rollout)
    read = []
    await optimizer._start()
    load_transcript = optimizer._run.load_transcript
    monkeypatch.setattr(optimizer._run, "load_transcript", lambda t: read.append(t) or load_transcript(t))

    await optimizer._run_search()
    assert len(proposer.requests) >= 2 and read == []


async def test_the_result_reloads_whole_without_the_run_folder_and_summarizes(tmp_path):
    optimizer = _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path / "runs")
    result = await optimizer.run_async()
    root = optimizer._run.root
    assert json.loads((root / "result.json").read_text()) == result.to_dict()
    path = result.to_file(tmp_path / "copy.json")
    shutil.rmtree(root)
    reloaded = OptimizationResult.from_file(path)
    assert reloaded == result and reloaded.to_rows("case") == result.to_rows("case")
    assert reloaded.reports[result.best.id]["test"] == result.reports[result.best.id]["test"]

    assert (result.objective, result.constraints, result.max_rollouts) == ("Maximize(Temperature)", [], 500)
    assert result.feasibility_threshold == 0.9 and result.proposers == {"numbers": "Scripted", "block": "Scripted"}
    assert (result.validation.outcome, result.validation.rule) == ("top_finalist", {"kind": "Mean"})
    text = result.summary()
    assert (
        f"Champion:     candidate {result.best.number} " in text
        and "Proposers:    numbers (Scripted), block (Scripted)" in text
    )
    assert f"candidate {result.best.number} *  {result.best.id}" in text and "Test vs baseline" in text
    assert "temperature  0.0       1.0" in text
    with pytest.raises(ValueError, match="unknown grain"):
        result.to_rows("trial")


async def test_the_summary_names_the_flagged_metrics_or_why_there_are_no_effects(tmp_path):
    result = await _two_steps({"temperature": 0.5}, {"temperature": 1.0}, base_dir=tmp_path).run_async()
    metrics = {
        "Temperature": {"diagnostics": {"flagged": False}},
        "tokens": {"diagnostics": {"flagged": True}},
        "seconds": {"diagnostics": {"flagged": True}},
    }

    def effects_lines(effects):
        text = result.model_copy(update={"effects": {**result.effects, **effects}}).summary()
        return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith(("Effects:", "Held-"))]

    assert effects_lines({"metrics": metrics, "reason": None}) == [
        "flagged tokens, seconds; from then on the search ranked them by measured means, the other metrics by the fit"
    ]
    one = {**metrics, "seconds": {"diagnostics": {"flagged": False}}}
    assert effects_lines({"metrics": one, "reason": None}) == [
        "flagged tokens; from then on the search ranked it by measured means, the other metrics by the fit"
    ]
    every = {m: {"diagnostics": {"flagged": True}} for m in metrics}
    assert effects_lines({"metrics": every, "reason": None}) == [
        "flagged Temperature, tokens, seconds; from then on the search ranked by measured means, not the fit"
    ]
    # Empty effects with a reason (as a flagged fit gave before) print only the reason.
    empty = {"effects": [], "factors": [], "omitted": [], "metrics": metrics, "reason": "flagged: tokens, seconds"}
    assert effects_lines(empty) == ["flagged: tokens, seconds"]
    assert effects_lines({"metrics": {"tokens": {"diagnostics": {"flagged": False}}}, "reason": None}) == []


def _with_transcripts(experiment=None):
    """What a run passes, and resume passes again: two evaluators; each trial a distinct trajectory, except on a
    crashed case and on a case whose rollout returns none (those two case names are returned too)."""
    train = list(_optimizer()._case_split.train)  # the same seed and cases give the same split
    crashes_on, bare = train[1], train[2]
    calls = itertools.count()

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if case.name == crashes_on:
            raise ValueError("boom")
        if case.name == bare:
            return str(temperature)
        return {"output": str(temperature), "trajectory": [f"only in the transcript {next(calls)}", temperature]}

    passed = {
        "experiment": _experiment(evaluators=[Temperature(), Score()]) if experiment is None else experiment,
        "factory": create_agent,
        "rollout_fn": rollout,
        "objective": Maximize("Temperature"),
        "search": Search(
            [Block(Scripted({"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")], schedule=Joint()
        ),
    }
    return passed, {crashes_on, bare}


def _run_with_transcripts(base_dir):
    passed, without = _with_transcripts()
    optimizer = Optimizer(
        **passed,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        runs_per_candidate=2,
        max_workers=64,
        base_dir=base_dir,
    )
    return optimizer, without


async def test_the_result_names_each_trajectory_by_its_transcript_hash_and_is_written_compact(tmp_path):
    optimizer, without = _run_with_transcripts(tmp_path / "runs")
    result = await optimizer.run_async()
    root = result.run_dir
    text = (root / "result.json").read_text(encoding="utf-8")
    assert "only in the transcript" not in text
    assert text.count("\n") == 1  # compact
    assert result.to_file(tmp_path / "copy.json").read_text(encoding="utf-8") == text

    hashes = {(r["candidate_id"], r["case_name"], r["run_index"]): r["transcript"] for r in result.to_rows("case")}
    saved = [digest for digest in hashes.values() if digest is not None]
    assert len(set(saved)) == len(saved)  # one distinct trajectory per trial
    rows = [(c, row) for c, splits in result.reports.items() for report in splits.values() for row in report.cases]
    assert len(rows) == 2 * len(hashes)  # one row per evaluator per trial
    assert {row["name"] for _, row in rows if row["transcript"] is None} == without
    for candidate_id, row in rows:
        digest = hashes[(candidate_id, row["name"], row["run_index"])]
        assert row.get("actual_trajectory") is None and row["transcript"] == digest  # a failed row has no key
        if digest is not None:
            kept = optimizer._transcripts[candidate_id][(row["name"], row["run_index"])]
            assert kept == {"hash": digest, "format": "list"}  # the kept record's own transcript
            trajectory = result.load_transcript(digest)
            temperature = result.candidates[candidate_id].values["temperature"]
            assert trajectory[0].startswith("only in the transcript") and trajectory[1] == temperature
    assert OptimizationResult.from_file(root / "result.json") == result
    assert result.load_transcript(None) is None
    with pytest.raises(ValueError, match="not a transcript hash"):  # a result.json from elsewhere
        result.load_transcript("../../run")

    # Moved, the run folder is passed in.
    moved = shutil.copytree(root, tmp_path / "moved")
    shutil.rmtree(root)
    reloaded = OptimizationResult.from_file(moved / "result.json")
    with pytest.raises(FileNotFoundError):
        reloaded.load_transcript(saved[0])
    assert reloaded.load_transcript(saved[0], run_dir=moved)[0].startswith("only in the transcript")

    # A result written with the trajectories inline still loads.
    old = result.to_dict()
    for splits in old["reports"].values():
        for report in splits.values():
            for row in report["cases"]:
                row["actual_trajectory"] = ["inline"]
                del row["transcript"]
    (tmp_path / "old.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
    reloaded = OptimizationResult.from_file(tmp_path / "old.json")
    assert reloaded.reports[result.baseline_id]["train"].cases[0]["actual_trajectory"] == ["inline"]


async def test_a_finished_run_resumed_rewrites_the_same_transcript_hashes(tmp_path):
    optimizer, _ = _run_with_transcripts(tmp_path)
    original = await optimizer.run_async()

    # Fresh proposers, as a new process would pass them; the same cases, since a failed trial's row holds the
    # case's session id.
    passed, _ = _with_transcripts(optimizer._experiment)
    again = await Optimizer.resume(original.run_dir, **passed).run_async()
    assert any(row["transcript"] for splits in again.reports.values() for r in splits.values() for row in r.cases)
    assert again.reports == original.reports
    assert OptimizationResult.from_file(original.run_dir / "result.json") == again


# ---- concurrent vs. one-at-a-time scoring ---------------------------------------------------------------


async def test_max_workers_caps_the_trials_of_a_whole_step(tmp_path):
    # More candidates than workers: the step's calls share one cap, so one trial runs at a time.
    peak = 0
    in_flight = 0

    async def rollout(agent_factory, case):
        nonlocal peak, in_flight
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.02)  # long enough that any trial allowed to start would overlap
            _, temperature = agent_factory()
            return str(temperature)
        finally:
            in_flight -= 1

    optimizer = _searching(
        Block(Scripted({"temperature": 0.5}, {"temperature": 1.0}), "temperature"),
        schedule=Joint(),
        base_dir=tmp_path,
        budget=Budget(max_iterations=1),
        proposals_per_iteration=2,
        max_workers=1,
    )
    optimizer._rollout_fn = rollout
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the batching warning when max_workers < step size
        await _search(optimizer)

    assert len(optimizer._iterations[0].proposals) == 2  # two candidates, scored at once
    assert peak == 1


async def test_scoring_a_step_at_once_gives_the_same_events_and_pick_as_one_at_a_time(tmp_path):
    # Same seed, same scripted proposals; max_workers alone changes whether the step's candidates run concurrently.
    updates = [{"temperature": 0.0}, {"temperature": 0.5}, {"temperature": 1.0}, {"temperature": 0.5}]

    def run_at(max_workers, base_dir):
        return _searching(
            Block(Scripted(*updates), "temperature"),
            schedule=Alternate(),
            base_dir=base_dir,
            budget=Budget(max_iterations=2),
            proposals_per_iteration=2,
            max_workers=max_workers,
        )

    together = run_at(64, tmp_path / "together")
    at_once = await together.run_async()
    sequential = run_at(1, tmp_path / "sequential")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the batching warning when max_workers < step size
        one_at_a_time = await sequential.run_async()

    # Same candidates in the same order and the same validation pick.
    assert list(at_once.candidates) == list(one_at_a_time.candidates)
    assert at_once.best.id == one_at_a_time.best.id

    # Same events in the same order (iteration_end to iteration_end), ignoring trial ordering and timings.
    def events(opt):
        out = []
        for raw in _events(opt):
            event = raw["event"]
            if event == "trial":
                continue  # concurrent trials interleave; the loop is asserted everywhere else
            out.append((event, raw.get("iteration"), raw.get("candidate_id") or raw.get("parent_id")))
        return out

    assert events(together) == events(sequential)


# ---- Pareto end phase -----------------------------------------------------------------------------------


class Cool(Evaluator):
    """Scores one minus the agent's temperature, so it trades off against Temperature."""

    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=1 - float(evaluation_case.actual_output), test_pass=True, reason="c")]


async def test_pareto_ships_the_front_s_max_gain_member_and_tests_every_member(tmp_path):
    # Every temperature trades Temperature for Cool, so all three candidates are on the front; Temperature's
    # weight 2 makes the hottest the max-gain member.
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Cool()]),
        rollout_fn=_temperature_rollout,
        search=Search(
            [Block(Scripted({"temperature": 1.0}, {"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")],
            schedule=Joint(),
        ),
        objective=Pareto(Maximize("Temperature", weight=2), Maximize("Cool")),
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    result = await optimizer.run_async()
    temperature = {c.id: c.values["temperature"] for c in result.candidates.values()}
    assert [temperature[c] for c in result.validation.front] == [1.0, 0.5, 0.0]
    assert result.validation.champion_id == result.best.id and temperature[result.best.id] == 1.0
    assert [c.id for c in result.front] == result.validation.front
    # Every validated front member is tested against the baseline.
    assert list(result.front_test) == result.validation.front
    assert all(
        (report.champion_id, report.baseline_id) == (member, result.baseline_id)
        for member, report in result.front_test.items()
    )


async def test_pareto_a_test_report_the_budget_skipped_names_the_budget_for_every_front_member(tmp_path):
    def shared():
        """What the run and its resumes are given alike (fresh proposers each time)."""
        return {
            "rollout_fn": _temperature_rollout,
            "search": Search(
                [
                    Block(Scripted({"temperature": 1.0}, {"temperature": 0.5}), "temperature"),
                    Block(Scripted(), "manager"),
                ],
                schedule=Joint(),
            ),
            "objective": Pareto(Maximize("Temperature", weight=2), Maximize("Cool")),
        }

    experiment = _experiment(evaluators=[Temperature(), Cool()])
    original = await Optimizer(
        experiment,
        create_agent,
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path / "runs",
        **shared(),
    ).run_async()
    assert len(original.validation.front) == 3  # the champion, the baseline, and one more member
    copy = _killed(original.run_dir, tmp_path, lambda events: _index(events, "picked"))
    spent = sum(1 for e in _events_at(copy) if e["event"] == "trial")

    skipped = await Optimizer.resume(
        copy, experiment=experiment, factory=create_agent, budget=Budget(max_rollouts=spent), **shared()
    ).run_async()
    assert skipped.test is None and skipped.test_reason == (
        f"the test report needs 12 rollouts, 0 are left; resume with max_rollouts >= {spent + 18} to run it with "
        "every front member"
    )
    tested = await Optimizer.resume(
        copy, experiment=experiment, factory=create_agent, budget=Budget(max_rollouts=spent + 18), **shared()
    ).run_async()
    assert list(tested.front_test) == original.validation.front and tested.test_reason is None


async def test_a_front_member_with_no_completed_test_trial_is_listed_as_failed(tmp_path):
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Cool()]),
        rollout_fn=_temperature_rollout,
        search=Search(
            [Block(Scripted({"temperature": 1.0}, {"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")],
            schedule=Joint(),
        ),
        objective=Pareto(Maximize("Temperature", weight=2), Maximize("Cool")),
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    test_cases = set(optimizer._case_split.test)

    async def rollout(agent_factory, case):
        _, temperature = agent_factory()
        if temperature == 0.5 and case.name in test_cases:
            raise ValueError("crashes on test")
        return str(temperature)

    optimizer._rollout_fn = rollout
    result = await optimizer.run_async()
    member = next(c.id for c in result.candidates.values() if c.values["temperature"] == 0.5)
    assert member in result.validation.front and result.best.id != member and result.test is not None
    assert result.test_failed_ids == [member] and member not in result.front_test


async def test_measured_term_values_are_the_metric_values_under_minimize(tmp_path):
    proposer = Scripted({"temperature": 0.5})
    optimizer = _optimizer(
        _experiment(evaluators=[Temperature(), Cool()]),
        rollout_fn=_temperature_rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Alternate()),
        objective=Minimize("Cool"),
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    await _search(optimizer)
    (request,) = proposer.requests
    assert not request.candidates[0].estimates  # no fit yet: measured means
    assert request.candidates[0].term_values == [1.0]  # the baseline's Cool (temperature 0), not negated


class EvenOnly(Evaluator):
    """At temperature 0, 1.0 on even-numbered cases and 0.5 on odd ones; at any other, 0.9 on even ones and N/A on
    odd ones."""

    async def evaluate_async(self, evaluation_case):
        even = int(evaluation_case.input[1:]) % 2 == 0
        if float(evaluation_case.actual_output) == 0.0:
            return [EvaluationOutput(score=1.0 if even else 0.5, test_pass=True, reason="e")]
        if even:
            return [EvaluationOutput(score=0.9, test_pass=True, reason="e")]
        return [EvaluationOutput(score=0.0, test_pass=True, reason="nothing to judge", label=NOT_APPLICABLE)]


async def test_without_a_fit_n_a_cases_are_filled_for_the_ranking_and_the_term_values(tmp_path):
    proposer = Scripted({"temperature": 0.5}, {"temperature": 1.0})
    optimizer = _optimizer(
        _experiment(evaluators=[EvenOnly()]),
        rollout_fn=_temperature_rollout,
        search=Search([Block(proposer, "temperature"), Block(Scripted(), "manager")], schedule=Joint()),
        objective=Maximize("EvenOnly"),
        score_model=_Sequence(),  # never fits, so the ranking uses measured means
        budget=Budget(max_iterations=2),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    await _search(optimizer)
    train = optimizer._case_split.train
    assert 0 < sum(int(case[4:]) % 2 for case in train) < len(train)  # both judged and N/A cases in train
    summaries = {c.values["temperature"]: c for c in proposer.requests[1].candidates}
    baseline, partial = summaries[0.0], summaries[0.5]
    assert partial.measured_objective == pytest.approx(0.9)  # its even cases only
    # Its odd cases are filled at the baseline's 0.5 shifted by -0.1, so it ranks below the baseline.
    assert baseline.ranking_objective == baseline.measured_objective
    assert partial.ranking_objective == pytest.approx(baseline.ranking_objective - 0.1)
    assert partial.term_values == [partial.ranking_objective]
    assert partial.adjusted_means == {"EvenOnly": pytest.approx(partial.ranking_objective)}
    assert baseline.adjusted_means == {"EvenOnly": baseline.measured_objective}
    assert (baseline.rank, partial.rank) == (1, 2)


async def test_without_a_fit_the_label_rows_and_summary_report_the_adjusted_means(tmp_path):
    optimizer = _optimizer(
        _experiment(evaluators=[EvenOnly()]),
        rollout_fn=_temperature_rollout,
        search=Search(
            [Block(Scripted({"temperature": 0.5}), "temperature"), Block(Scripted(), "manager")], schedule=Joint()
        ),
        objective=Maximize("EvenOnly"),
        score_model=_Sequence(),  # never fits: the search and the final pick use measured means
        budget=Budget(max_iterations=1),
        proposals_per_iteration=1,
        max_workers=64,
        base_dir=tmp_path,
    )
    result = await optimizer.run_async()
    validation = optimizer._case_split.validation
    assert 0 < sum(int(case[4:]) % 2 for case in validation) < len(validation)  # judged and N/A cases
    partial = next(c.id for c in result.candidates.values() if c.values["temperature"] == 0.5)
    (row,) = [r for r in result.to_rows("proposal") if r["candidate_id"] == partial]
    assert row["train.EvenOnly"] == pytest.approx(0.9)  # its judged cases only
    assert row["adjusted.EvenOnly"] == pytest.approx(row["objective_at_step"])  # what the ranking used
    assert row["adjusted.EvenOnly"] < row["train.EvenOnly"]
    v = result.validation
    assert v.problem is not None and not any(v.estimates.values())
    adjusted = v.adjusted_means[partial]["EvenOnly"]
    assert adjusted == pytest.approx(v.objectives[partial]) and adjusted < v.measured[partial]["EvenOnly"].mean
    assert v.adjusted_means[result.baseline_id] == {"EvenOnly": v.measured[result.baseline_id]["EvenOnly"].mean}
    lines = result.summary().splitlines()
    head = next(line for line in lines if line.startswith("Validation  "))
    assert head.split() == [
        "Validation",
        "id",
        "estimate",
        "sd",
        "adjusted",
        "measured",
        "n",
    ]  # the objective = adjusted
    row = next(line for line in lines if partial in line).split()
    assert row[-5:] == ["-", "-", f"{adjusted:.3f}", "0.900", "1"]


# ---- resume inside a two-level Nested -------------------------------------------------------------------


@tunable(
    outer=choice(a=component(Window, size=Int(1, 100)), b=component(Summary, ratio=Float(0.1, 0.9))),
    inner=choice(x=component(Window, size=Int(1, 100)), y=component(Summary, ratio=Float(0.1, 0.9))),
    temperature=Float(0.0, 1.0, default=0.5),
)
def _two_structural(outer, inner, temperature):
    return (outer, inner, temperature)


async def _two_structural_rollout(agent_factory, case):
    *_, temperature = agent_factory()
    return str(temperature)


def _two_structural_search():
    # Two structural blocks: the default schedule nests one Nested in the other.
    return Search([Block(TPE(n_startup_trials=2), "outer"), Block(TPE(n_startup_trials=2), "inner")])


@pytest.mark.parametrize("killed_step", ["inner", "opener"])
async def test_resume_inside_a_two_level_nested_matches_the_uninterrupted_run(tmp_path, killed_step):
    experiment = Experiment(cases=_experiment().cases, evaluators=[Temperature()])
    original_optimizer = Optimizer(
        experiment,
        _two_structural,
        rollout_fn=_two_structural_rollout,
        search=_two_structural_search(),
        max_workers=64,
        base_dir=tmp_path / "runs",
        budget=Budget(max_iterations=10),
        proposals_per_iteration=2,
    )
    original = await original_optimizer.run_async()
    assert any(len(r.round or ()) == 2 for r in original.iterations)

    def inner_round_scored(events):
        """Just past the first `score` after an inner round's first closed step: the next step, killed."""
        closed = next(
            i for i, e in enumerate(events) if e["event"] == "iteration_end" and len(e["record"]["round"] or ()) == 2
        )
        return _index(events, "score", after=closed)

    # The second inner round's opener: its patch deferred to the round's end, the outer round open.
    opener = [r.iteration for r in original.iterations if r.role == "opener" and len(r.round or ()) == 2][1]

    def inner_opener_scored(events):
        """Just past the first `score` after the opener's `select`."""
        return _index(events, "score", after=_index(events, "select", nth=opener))

    cut = inner_round_scored if killed_step == "inner" else inner_opener_scored
    copy = _killed(original_optimizer._run.root, tmp_path, cut)
    killed = 1 + sum(1 for e in _events_at(copy) if e["event"] == "iteration_end")
    assert killed_step == "inner" or killed == opener
    resumed = await Optimizer.resume(
        copy,
        experiment=experiment,
        factory=_two_structural,
        rollout_fn=_two_structural_rollout,
        search=_two_structural_search(),
    ).run_async()
    _assert_same(resumed, original, rerun=killed)
