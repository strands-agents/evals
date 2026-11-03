import pytest

from strands_evals.case import Case
from strands_evals.cli._optimize.engine import (
    EvaluatorRow,
    InterimOptimizer,
    RolloutOutput,
    Settings,
    SetupError,
    classify,
)
from strands_evals.cli._optimize.proposers import FileProposer
from strands_evals.cli._optimize.record import RunDirectory
from strands_evals.cli._optimize.scoring import Criteria, MetricRegistry, build_objective
from strands_evals.cli._optimize.space import CliSpace
from strands_evals.cli._optimize.splits import assign_split
from strands_evals.optimization.declaration.space import ParamInfo
from strands_evals.optimization.execution.status import InfrastructureError
from strands_evals.optimization.setup.budget import Budget
from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.optimization import Patch, TrialStatus

CASES = [Case(name=f"case-{index:02d}", input=f"question {index}", expected_output="ok") for index in range(10)]
SPLIT = assign_split([case.name for case in CASES], (0.4, 0.3, 0.3), 0)
REGISTRY = MetricRegistry({"Score": (0.0, 1.0)})


def space(tmp_path):
    return CliSpace(
        params=(
            ParamInfo(key="prompt", name="prompt", kind="text", default="base", size_cap=100),
            ParamInfo(key="turns", name="turns", kind="int", default=3, min_value=1, max_value=9),
        ),
        root=tmp_path,
    )


def patches(*updates):
    return FileProposer(
        patches=tuple(Patch(updates=update, rationale=f"patch {index}") for index, update in enumerate(updates)),
        keys=("prompt", "turns"),
        source="patches.json",
        name="file",
    )


def evaluate(case, result):
    score = float(result.output)
    output = EvaluationOutput(score=score, test_pass=score >= 0.5, reason=f"scored {score}")
    return [EvaluatorRow("Score", output, [output], None)]


def scorer(table):
    def rollout(values, case):
        role = next(role for role in ("train", "validation", "test") if case.name in getattr(SPLIT, role))
        return RolloutOutput(output=table[values["prompt"]][role], trajectory=[{"content": values["prompt"]}])

    return rollout


def optimizer(tmp_path, proposers, rollout, *, budget=500, iterations=None, per_iteration=4):
    return InterimOptimizer(
        space=space(tmp_path),
        cases=CASES,
        criteria=Criteria(build_objective([], REGISTRY), REGISTRY),
        settings=Settings(
            split=SPLIT,
            budget=Budget(max_rollouts=budget, max_iterations=iterations),
            finalists=3,
            runs_per_candidate=1,
            proposals_per_iteration=per_iteration,
            rerun_passes=2,
        ),
        proposers=proposers,
        rollout=rollout,
        evaluate=evaluate,
        run=_run(tmp_path),
        progress=lambda message: None,
    )


def _run(tmp_path):
    existing = sorted((tmp_path / "runs").glob("*")) if (tmp_path / "runs").exists() else []
    return (
        RunDirectory.open(existing[0])
        if existing
        else RunDirectory.create(tmp_path / "runs", {"schema_version": 1}, name=None)
    )


def events(tmp_path):
    return _run(tmp_path).events.read()


def test_ships_the_best_finalist_and_reports_it_against_the_baseline_on_test(tmp_path):
    table = {
        "base": dict.fromkeys(("train", "validation", "test"), 0.2),
        "better": dict.fromkeys(("train", "validation", "test"), 0.9),
    }

    result = optimizer(tmp_path, [patches({"prompt": "better"})], scorer(table)).run()

    assert result["improved"] is True
    assert result["best"]["values"] == {"prompt": "better", "turns": 3}
    assert result["stop_reason"] == "exhausted"
    assert result["test"]["objective"]["mean_diff"] == pytest.approx(0.7)
    assert result["validation"]["problem"].startswith("no score model")
    assert "+better" in result["diff"]
    assert result["rollouts_spent"] == 4 + 4 + 2 * 3 + 2 * 3
    names = [event["event"] for event in events(tmp_path)]
    assert names[:4] == ["config", "propose", "trial", "trial"]
    assert names.index("stop") < names.index("finalists") < names.index("selected") < names.index("test")
    assert names[-1] == "test"
    assert (_run(tmp_path).root / "best.diff").read_text() == result["diff"]


def test_keeps_the_baseline_when_a_train_winner_loses_on_validation(tmp_path):
    table = {
        "base": dict.fromkeys(("train", "validation", "test"), 0.5),
        "overfit": {"train": 1.0, "validation": 0.0, "test": 0.0},
    }

    result = optimizer(tmp_path, [patches({"prompt": "overfit"})], scorer(table)).run()

    assert result["improved"] is False
    assert result["best"]["id"] == result["baseline"]["id"]
    assert result["validation"]["reason"].startswith("baseline ships")
    assert result["test"]["objective"] == {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_cases": 3}
    assert result["diff"] == ""


def test_ships_the_validation_winner_even_when_test_says_it_is_worse(tmp_path):
    table = {
        "base": dict.fromkeys(("train", "validation", "test"), 0.5),
        "lucky": {"train": 0.9, "validation": 0.9, "test": 0.1},
    }

    result = optimizer(tmp_path, [patches({"prompt": "lucky"})], scorer(table)).run()

    assert result["improved"] is True
    assert result["test"]["objective"]["mean_diff"] == pytest.approx(-0.4)


def test_a_tie_on_validation_keeps_the_baseline(tmp_path):
    table = {
        "base": dict.fromkeys(("train", "validation", "test"), 0.5),
        "same": {"train": 0.6, "validation": 0.5, "test": 0.5},
    }

    result = optimizer(tmp_path, [patches({"prompt": "same"})], scorer(table)).run()

    assert result["improved"] is False


def test_stops_on_budget_and_never_spends_the_end_phase_reserve(tmp_path):
    table = {
        name: dict.fromkeys(("train", "validation", "test"), 0.1 * index)
        for index, name in enumerate(["base", "a", "b", "c"])
    }
    budget = 4 + 4 + (3 + 1) * 3 + 2 * 3

    result = optimizer(
        tmp_path, [patches({"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"})], scorer(table), budget=budget
    ).run()

    assert result["stop_reason"] == "budget"
    assert result["rollouts_spent"] <= budget
    outcomes = [proposal["outcome"] for proposal in result["iterations"][0]["proposals"]]
    assert outcomes == ["feasible", "unscored", "unscored"]
    assert result["test"] is not None


def test_stops_after_the_iteration_cap(tmp_path):
    table = {name: dict.fromkeys(("train", "validation", "test"), 0.5) for name in ("base", "a", "b")}
    proposer = FileProposer(
        patches=(Patch(updates={"prompt": "a"}, rationale=""), Patch(updates={"prompt": "b"}, rationale="")),
        keys=("prompt", "turns"),
        source="patches.json",
        name="file",
    )
    result = optimizer(tmp_path, [proposer], scorer(table), iterations=1, per_iteration=1).run()

    assert result["stop_reason"] == "iterations"
    assert len(result["iterations"]) == 1


def test_records_invalid_and_duplicate_patches_without_spending(tmp_path):
    table = {"base": dict.fromkeys(("train", "validation", "test"), 0.5)}

    result = optimizer(tmp_path, [patches({"turns": 99}, {"prompt": "base"}, {"unknown": "x"})], scorer(table)).run()

    outcomes = [proposal["outcome"] for proposal in result["iterations"][0]["proposals"]]
    assert outcomes == ["invalid", "duplicate", "invalid"]
    assert result["rollouts_spent"] == 4 + 3 + 3


class Failing:
    name, kind, keys = "text", "Reflection", ("prompt",)

    def config(self):
        return {}

    def propose(self, context):
        raise RuntimeError("model unavailable")


def test_a_proposer_error_stops_the_search_and_still_runs_the_end_phase(tmp_path):
    table = {"base": dict.fromkeys(("train", "validation", "test"), 0.5)}

    result = optimizer(tmp_path, [Failing()], scorer(table)).run()

    assert result["stop_reason"] == "proposer_error"
    assert result["improved"] is False
    names = [event["event"] for event in events(tmp_path)]
    assert "proposer_retry" in names and "proposer_error" in names


def test_raises_a_setup_error_when_every_baseline_trial_fails(tmp_path):
    def rollout(values, case):
        raise ValueError("agent crashed")

    with pytest.raises(SetupError, match="agent_failure"):
        optimizer(tmp_path, [patches({"prompt": "x"})], rollout).run()

    last = events(tmp_path)[-1]
    assert last["event"] == "stop" and last["reason"] == "setup"
    with pytest.raises(SetupError, match="start a new run"):
        optimizer(tmp_path, [patches({"prompt": "x"})], rollout).restore(events(tmp_path))


def test_reruns_an_infrastructure_failure_and_counts_every_attempt(tmp_path):
    table = {"base": dict.fromkeys(("train", "validation", "test"), 0.5)}
    seen = set()

    def flaky(values, case):
        if case.name not in seen:
            seen.add(case.name)
            raise InfrastructureError("sandbox down")
        return scorer(table)(values, case)

    result = optimizer(tmp_path, [patches()], flaky).run()

    trials = [event["record"] for event in events(tmp_path) if event["event"] == "trial"]
    assert all(trial["status"] == "completed" for trial in trials)
    first = next(trial for trial in trials if trial["previous_attempts"])
    assert first["attempts"] == 2 and first["previous_attempts"][0]["status"] == "infrastructure_invalid"
    assert result["rollouts_spent"] == len(trials) + len(seen)


def test_resume_reuses_saved_trials_and_reruns_only_the_killed_iteration(tmp_path):
    table = {
        name: dict.fromkeys(("train", "validation", "test"), score)
        for name, score in (("base", 0.2), ("a", 0.6), ("b", 0.9))
    }
    calls = {"count": 0}

    def killed(values, case):
        calls["count"] += 1
        if calls["count"] == 7:
            raise KeyboardInterrupt
        return scorer(table)(values, case)

    first = optimizer(tmp_path, [patches({"prompt": "a"}, {"prompt": "b"})], killed)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    calls["count"] = 100

    resumed = optimizer(tmp_path, [patches({"prompt": "a"}, {"prompt": "b"})], killed)
    resumed.restore(events(tmp_path))
    result = resumed.run()

    assert result["best"]["values"]["prompt"] == "b"
    assert calls["count"] - 100 == result["rollouts_spent"] - 6
    assert [event["event"] for event in events(tmp_path)].count("config") == 1


def test_a_finished_run_replays_without_rollouts_or_new_events(tmp_path):
    table = {
        "base": dict.fromkeys(("train", "validation", "test"), 0.2),
        "better": dict.fromkeys(("train", "validation", "test"), 0.9),
    }
    original = optimizer(tmp_path, [patches({"prompt": "better"})], scorer(table)).run()
    count = len(events(tmp_path))

    def never(values, case):
        raise AssertionError("a finished run must not roll out")

    replay = optimizer(tmp_path, [patches({"prompt": "better"})], never, budget=10_000)
    replay.restore(events(tmp_path))

    assert replay.run() == original
    assert len(events(tmp_path)) == count


def test_rejects_cases_without_unique_names(tmp_path):
    with pytest.raises(ValueError, match="unique name"):
        InterimOptimizer(
            space=space(tmp_path),
            cases=[CASES[0], CASES[0]],
            criteria=Criteria(build_objective([], REGISTRY), REGISTRY),
            settings=Settings(SPLIT, Budget(), 3, 1, 4, 2),
            proposers=[patches()],
            rollout=scorer({}),
            evaluate=evaluate,
            run=_run(tmp_path),
            progress=lambda message: None,
        )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (InfrastructureError("x"), TrialStatus.INFRASTRUCTURE_INVALID),
        (RuntimeError("Runner failed (1): ThrottlingException: slow down"), TrialStatus.INFRASTRUCTURE_INVALID),
        (RuntimeError("connection reset by peer"), TrialStatus.INFRASTRUCTURE_INVALID),
        (ValueError("bad answer"), TrialStatus.AGENT_FAILURE),
        (TimeoutError("deadline"), TrialStatus.AGENT_FAILURE),
    ],
)
def test_classify_separates_infrastructure_from_agent_failures(error, status):
    assert classify(error) is status


def test_settings_reject_a_budget_below_the_baseline_and_reserves():
    settings = Settings(SPLIT, Budget(max_rollouts=10), 3, 1, 4, 2)

    with pytest.raises(ValueError, match="validation reserve 12, test reserve 6"):
        settings.check_budget()
