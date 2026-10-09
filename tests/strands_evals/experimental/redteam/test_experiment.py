"""Tests for RedTeamExperiment."""

import warnings

import pytest
from strands.models.model import Model

from strands_evals import LocalFileTaskResultStore
from strands_evals.case import Case
from strands_evals.evaluators import Evaluator
from strands_evals.evaluators.prompt_templates.case_prompt_template import compose_test_prompt
from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.experiment import RedTeamExperiment
from strands_evals.experimental.redteam.report import RedTeamReport
from strands_evals.experimental.redteam.strategies import (
    BadLikertJudgeStrategy,
    CrescendoStrategy,
    GoatStrategy,
    PairStrategy,
    PromptStrategy,
    SequentialBreakStrategy,
)
from strands_evals.experimental.redteam.strategies.base import RUN_RESULTS, AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig
from strands_evals.types import EvaluationOutput


class _StubModel(Model):
    """Minimal Model subclass used to exercise the runtime-object branch of `_serialize_model`."""

    def __init__(self, config: dict | None) -> None:
        # `config` may be a non-dict (e.g. None) on purpose for the non-coercible branch.
        self.config = config  # type: ignore[assignment]

    def get_config(self):
        return self.config

    def update_config(self, **kwargs):
        pass

    def structured_output(self, *args, **kwargs):
        raise NotImplementedError

    async def stream(self, *args, **kwargs):
        raise NotImplementedError


class _StubStrategy(AttackStrategy):
    def __init__(self, name="stub", *, label=None):
        super().__init__(label=label)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        return AttackRunResult(conversation=[], metadata={})

    def reset(self) -> None:
        pass


class _FakeSession:
    """Minimal TargetSession standing in for a real target in wiring tests."""

    def __init__(self, reply="ok"):
        self._reply = reply
        self.trace: list[dict] = []

    def invoke(self, _message):
        return self._reply

    def reset(self):
        self.trace.clear()

    def snapshot(self):
        return None

    def restore(self, checkpoint):
        pass


def _case(name: str = "c0") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="goal")),
    )


def test_default_evaluator_is_attack_success():
    exp = RedTeamExperiment(cases=[_case()])
    assert len(exp.evaluators) == 1
    assert isinstance(exp.evaluators[0], AttackSuccessEvaluator)


def test_custom_evaluators_respected():
    custom = AttackSuccessEvaluator()
    exp = RedTeamExperiment(cases=[_case()], evaluators=[custom])
    assert exp.evaluators == [custom]


def test_run_evaluations_returns_red_team_report():
    """Even with no cases, the override returns a RedTeamReport (not list)."""
    exp = RedTeamExperiment(cases=[])
    report = exp.run_evaluations(task=lambda case: {"output": []})
    assert isinstance(report, RedTeamReport)


def test_run_evaluations_uses_default_task_with_agent_factory():
    """A run-time `agent_factory` enables run_evaluations() with no explicit task."""
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    report = exp.run_evaluations(agent_factory=_FakeSession)
    assert isinstance(report, RedTeamReport)


def test_run_evaluations_raises_when_neither_task_nor_agent_factory():
    exp = RedTeamExperiment(cases=[_case()])
    with pytest.raises(ValueError, match="task.*agent_factory"):
        exp.run_evaluations()


def test_run_evaluations_rejects_task_and_agent_factory():
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    with pytest.raises(ValueError, match="either `task` or `agent_factory`"):
        exp.run_evaluations(task=lambda case: {"output": []}, agent_factory=_FakeSession)


def test_run_time_path_emits_no_deprecation_warning():
    """The supported shortcut (RedTeamCase inputs, run-time factory) warns about nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
        exp.run_evaluations(agent_factory=_FakeSession)


def test_constructor_agent_is_deprecated_but_still_runs():
    with pytest.warns(DeprecationWarning, match="`agent`"):
        exp = RedTeamExperiment(cases=[_case()], agent=_FakeSession(), attack_strategies=[_StubStrategy()])
    assert isinstance(exp.run_evaluations(), RedTeamReport)


def test_constructor_agent_factory_is_deprecated_fallback():
    calls: list[_FakeSession] = []

    def factory():
        calls.append(_FakeSession())
        return calls[-1]

    with pytest.warns(DeprecationWarning, match="`agent_factory`"):
        exp = RedTeamExperiment(cases=[_case()], agent_factory=factory, attack_strategies=[_StubStrategy()])
    exp.run_evaluations()
    assert len(calls) == 1


def test_run_time_agent_factory_wins_over_constructor_values():
    """The run-time factory takes precedence over the deprecated constructor `agent` / `agent_factory`."""
    run_time_calls: list[_FakeSession] = []
    constructor_calls: list[_FakeSession] = []

    def run_time_factory():
        run_time_calls.append(_FakeSession())
        return run_time_calls[-1]

    def constructor_factory():
        constructor_calls.append(_FakeSession())
        return constructor_calls[-1]

    with pytest.warns(DeprecationWarning):
        exp = RedTeamExperiment(
            cases=[_case("c0"), _case("c1")],
            agent=_FakeSession(),
            agent_factory=constructor_factory,
            attack_strategies=[_StubStrategy()],
        )
    exp.run_evaluations(agent_factory=run_time_factory)
    assert len(run_time_calls) == 2
    assert constructor_calls == []


def test_plain_case_is_deprecated():
    with pytest.warns(DeprecationWarning, match="RedTeamCase"):
        RedTeamExperiment(cases=[Case(name="c0", input="hello")])


def test_duplicate_strategy_label_raises():
    with pytest.raises(ValueError, match="Duplicate strategy label"):
        RedTeamExperiment(
            cases=[_case()],
            attack_strategies=[_StubStrategy(label="dup"), _StubStrategy(label="dup")],
        )


def test_cross_product_expands_cases():
    """N cases × M strategies -> N*M work items, each tagged with its strategy label."""
    captured: list[str] = []

    def task(case):
        captured.append(case.name)
        assert case.metadata["strategy"] in {"cre-10", "cre-30"}
        return {"output": []}

    exp = RedTeamExperiment(
        cases=[_case("c0"), _case("c1")],
        attack_strategies=[_StubStrategy(label="cre-10"), _StubStrategy(label="cre-30")],
    )
    exp.run_evaluations(task=task)

    assert sorted(captured) == ["c0__cre-10", "c0__cre-30", "c1__cre-10", "c1__cre-30"]


async def test_run_evaluations_async_returns_report():
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    report = await exp.run_evaluations_async(agent_factory=_FakeSession)
    assert isinstance(report, RedTeamReport)


def test_report_cls_is_red_team_report():
    assert RedTeamExperiment.report_cls is RedTeamReport


async def test_max_workers_must_be_positive():
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    with pytest.raises(ValueError, match="max_workers"):
        await exp.run_evaluations_async(max_workers=0, agent_factory=_FakeSession)


async def test_parallel_rejects_deprecated_shared_agent():
    """Parallel runs always require `agent_factory` -- the runner does not deepcopy a shared target."""
    with pytest.warns(DeprecationWarning):
        exp = RedTeamExperiment(cases=[_case()], agent=_FakeSession(), attack_strategies=[_StubStrategy()])
    with pytest.raises(TypeError, match="agent_factory"):
        await exp.run_evaluations_async(max_workers=2)


async def test_parallel_uses_agent_factory_per_case():
    """When parallel and `agent_factory` is set, each case gets a fresh target from the factory."""
    factory_calls: list[_FakeSession] = []

    def factory():
        sess = _FakeSession()
        factory_calls.append(sess)
        return sess

    exp = RedTeamExperiment(
        cases=[_case("c0"), _case("c1"), _case("c2")],
        attack_strategies=[_StubStrategy()],
    )
    report = await exp.run_evaluations_async(max_workers=3, agent_factory=factory)
    assert isinstance(report, RedTeamReport)
    assert len(factory_calls) == 3


async def test_parallel_report_per_case_isolation():
    """End-to-end: under `max_workers > 1`, each case's output lands on its own report row.

    Pins the property the parallel path is *for* -- that concurrent cases don't bleed each
    other's outputs through the shared strategy.
    """

    class _EchoStrategy(AttackStrategy):
        @property
        def name(self) -> str:
            return "echo"

        def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
            target_session.invoke(case.input)
            return AttackRunResult(
                conversation=[{"role": "attacker", "content": case.input}],
                metadata={"echoed": case.input},
            )

    def factory():
        return _FakeSession()

    cases = [_case(f"c{i}") for i in range(5)]
    for i, case in enumerate(cases):
        case.input = f"msg-{i}"

    exp = RedTeamExperiment(cases=cases, attack_strategies=[_EchoStrategy()])
    report = await exp.run_evaluations_async(max_workers=4, agent_factory=factory)

    by_name = {r.case_name: r for r in report.attack_results()}
    assert set(by_name) == {"c0__echo", "c1__echo", "c2__echo", "c3__echo", "c4__echo"}
    for i in range(5):
        attack = by_name[f"c{i}__echo"]
        assert attack.conversation == [{"role": "attacker", "content": f"msg-{i}"}]


def test_async_task_rejected():
    async def _async_task(case):
        return {"output": []}

    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    with pytest.raises(ValueError, match="Async task is not supported"):
        exp.run_evaluations(task=_async_task)


def test_agent_setters_are_deprecated():
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    assert exp.agent is None
    assert exp.agent_factory is None
    sess = _FakeSession()
    with pytest.warns(DeprecationWarning, match="`agent`"):
        exp.agent = sess
    with pytest.warns(DeprecationWarning, match="`agent_factory`"):
        exp.agent_factory = _FakeSession
    assert exp.agent is sess
    assert exp.agent_factory is _FakeSession


def test_to_dict_persists_strategies_and_model():
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            CrescendoStrategy(max_turns=3, max_backtracks=2, success_threshold=0.6, label="cre-fast"),
            PromptStrategy("gradual_escalation", "TPL", max_turns=4),
        ],
        model="claude-3-5",
    )
    out = exp.to_dict()
    assert out["model"] == "claude-3-5"
    assert "agent" not in out  # live target is never persisted
    strategies = out["attack_strategies"]
    assert strategies[0] == {
        "strategy_type": "CrescendoStrategy",
        "label": "cre-fast",
        "max_turns": 3,
        "max_backtracks": 2,
        "success_threshold": 0.6,
    }
    assert strategies[1] == {
        "strategy_type": "PromptStrategy",
        "strategy_name": "gradual_escalation",
        "system_prompt_template": "TPL",
        "max_turns": 4,
    }


def test_to_dict_persists_bad_likert_judge_strategy():
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            BadLikertJudgeStrategy(refine_rounds=1, success_threshold=0.6, model="claude-stub", label="blj-fast"),
        ],
    )
    out = exp.to_dict()
    assert out["attack_strategies"][0] == {
        "strategy_type": "BadLikertJudgeStrategy",
        "label": "blj-fast",
        "refine_rounds": 1,
        "success_threshold": 0.6,
        "model": "claude-stub",
    }


def test_to_dict_persists_goat_strategy():
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            GoatStrategy(max_turns=4, success_threshold=0.6, model="claude-stub", store_reasoning=True, label="g"),
        ],
    )
    out = exp.to_dict()
    assert out["attack_strategies"][0] == {
        "strategy_type": "GoatStrategy",
        "label": "g",
        "max_turns": 4,
        "success_threshold": 0.6,
        "store_reasoning": True,
        "model": "claude-stub",
    }


def test_to_dict_persists_pair_strategy():
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            PairStrategy(max_turns=3, success_threshold=0.9, model="claude-stub", label="p"),
        ],
    )
    out = exp.to_dict()
    assert out["attack_strategies"][0] == {
        "strategy_type": "PairStrategy",
        "label": "p",
        "max_turns": 3,
        "success_threshold": 0.9,
        "model": "claude-stub",
    }


def test_to_dict_persists_sequentialbreak_strategy():
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            SequentialBreakStrategy(
                variants=["dc_t1"], max_turns=1, success_threshold=0.6, model="claude-stub", label="sb"
            ),
        ],
    )
    out = exp.to_dict()
    assert out["attack_strategies"][0] == {
        "strategy_type": "SequentialBreakStrategy",
        "label": "sb",
        "variants": ["dc_t1"],
        "max_turns": 1,
        "success_threshold": 0.6,
        "model": "claude-stub",
    }


def test_round_trip_preserves_all_builtin_strategies(tmp_path):
    """All built-ins resolve through the registry and survive a to_file/from_file cycle."""
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[
            CrescendoStrategy(label="cre"),
            PromptStrategy("gradual_escalation", "TPL", label="prompt"),
            BadLikertJudgeStrategy(label="blj"),
            GoatStrategy(label="goat"),
            PairStrategy(label="pair"),
            SequentialBreakStrategy(label="sb"),
        ],
    )
    p = tmp_path / "rt.json"
    exp.to_file(str(p))
    loaded = RedTeamExperiment.from_file(str(p))

    assert [type(s).__name__ for s in loaded.attack_strategies] == [
        "CrescendoStrategy",
        "PromptStrategy",
        "BadLikertJudgeStrategy",
        "GoatStrategy",
        "PairStrategy",
        "SequentialBreakStrategy",
    ]
    assert [s.label for s in loaded.attack_strategies] == ["cre", "prompt", "blj", "goat", "pair", "sb"]


def test_from_dict_round_trip_runs(tmp_path):
    """Reload via to_file/from_file, then run with a task; no target needs reattaching."""
    exp = RedTeamExperiment(
        cases=[_case("c0"), _case("c1")],
        attack_strategies=[CrescendoStrategy(max_turns=3, label="cre")],
        model="claude-3-5",
    )
    p = tmp_path / "rt.json"
    exp.to_file(str(p))
    loaded = RedTeamExperiment.from_file(str(p))

    assert loaded.agent is None
    assert loaded._model == "claude-3-5"
    assert [type(c).__name__ for c in loaded.cases] == ["RedTeamCase", "RedTeamCase"]
    assert loaded.cases[0].config.attack_goal.actor_goal == "goal"
    assert [s.label for s in loaded.attack_strategies] == ["cre"]
    assert isinstance(loaded.evaluators[0], AttackSuccessEvaluator)
    # Symmetric: any field-level drift (added/dropped/reordered key, lossy coercion)
    # would be caught here in one assertion, complementing the per-field checks above.
    assert RedTeamExperiment.from_dict(exp.to_dict()).to_dict() == exp.to_dict()

    # Without a task or agent_factory there is nothing to run.
    with pytest.raises(ValueError, match="task.*agent_factory"):
        loaded.run_evaluations()

    # A stub task skips real LLM calls.
    report = loaded.run_evaluations(task=lambda case: {"output": []})
    assert isinstance(report, RedTeamReport)


def test_from_dict_accepts_custom_strategies(tmp_path):
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy(label="stub-a")])
    p = tmp_path / "rt.json"
    exp.to_file(str(p))
    loaded = RedTeamExperiment.from_file(str(p), custom_strategies=[_StubStrategy])
    assert [type(s).__name__ for s in loaded.attack_strategies] == ["_StubStrategy"]
    assert loaded.attack_strategies[0].label == "stub-a"


def test_to_dict_serializes_model_instance():
    """A Model instance with `config['model_id']` round-trips into out['model']."""
    model = _StubModel(config={"model_id": "claude-stub"})
    exp = RedTeamExperiment(
        cases=[_case()],
        attack_strategies=[CrescendoStrategy(model=model)],
        model=model,
    )
    out = exp.to_dict()
    assert out["model"] == "claude-stub"
    assert out["attack_strategies"][0]["model"] == "claude-stub"


def test_to_dict_drops_non_coercible_model_with_warning(caplog):
    """A Model that doesn't expose dict config + model_id logs a warning and is dropped."""
    model = _StubModel(config=None)
    exp = RedTeamExperiment(cases=[_case()], model=model)
    with caplog.at_level("WARNING"):
        out = exp.to_dict()
    assert "model" not in out
    assert any("non-coercible Model" in record.message for record in caplog.records)


def test_from_dict_unknown_strategy_raises(tmp_path):
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    p = tmp_path / "rt.json"
    exp.to_file(str(p))
    with pytest.raises(ValueError, match="_StubStrategy"):
        RedTeamExperiment.from_file(str(p))  # no custom_strategies


def test_run_evaluations_twice_is_idempotent():
    """Re-running must not re-expand an already-expanded case list (no c0__cre__cre)."""
    runs: list[list[str]] = []

    def task(case):
        return {"output": []}

    exp = RedTeamExperiment(
        cases=[_case("c0")],
        attack_strategies=[_StubStrategy(label="a"), _StubStrategy(label="b")],
    )
    for _ in range(2):
        captured: list[str] = []

        def _task(case, _cap=captured):
            _cap.append(case.name)
            return {"output": []}

        exp.run_evaluations(task=_task)
        runs.append(sorted(captured))

    assert runs[0] == ["c0__a", "c0__b"]
    assert runs[0] == runs[1]  # second run identical, not squared
    # held cases were never mutated
    assert [c.name for c in exp.cases] == ["c0"]


class _PassEvaluator(Evaluator):
    """Deterministic evaluator so report tests don't call a judge model."""

    def evaluate(self, evaluation_case):
        return [EvaluationOutput(score=0.0, test_pass=True, reason="defended")]


_PRUNED = [{"role": "attacker", "content": "direct ask"}, {"role": "target", "content": "no"}]


class _RunStatsStrategy(AttackStrategy):
    """Returns fixed run stats so tests can check they reach the report."""

    @property
    def name(self) -> str:
        return "stats"

    def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        return AttackRunResult(conversation=[], metadata={"turns_used": 3, "backtracks": 1}, pruned_branches=_PRUNED)


def test_custom_task_run_results_reach_report():
    """A user task that returns `to_environment_state()` fills the report's run stats; no side channel needed."""
    strategy = _RunStatsStrategy()

    def task(case):
        session = _FakeSession()
        result = strategy.run_attack(case, session, max_turns=5)
        return {
            "output": result.conversation,
            "trajectory": list(session.trace),
            "environment_state": [result.to_environment_state()],
        }

    exp = RedTeamExperiment(cases=[_case("c0")], attack_strategies=[strategy], evaluators=[_PassEvaluator()])
    (result,) = exp.run_evaluations(task=task).attack_results()

    assert result.turns_used == 3
    assert result.backtracks == 1
    assert result.pruned_branches == _PRUNED


class _EnvStatePromptEvaluator(Evaluator):
    """Records the judge prompt an `uses_environment_state=True` evaluator would send."""

    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def evaluate(self, evaluation_case):
        self.prompts.append(compose_test_prompt(evaluation_case, "rubric", False, uses_environment_state=True))
        return [EvaluationOutput(score=0.0, test_pass=True, reason="defended")]


def test_environment_state_evaluator_sees_run_results():
    """Pins a known trade-off: evaluators that read environment state now see the run stats.

    Before run stats moved into `environment_state`, such an evaluator raised for a red team run; now it
    receives the `RUN_RESULTS` entry, including the full `pruned_branches` payload.
    """
    evaluator = _EnvStatePromptEvaluator()
    exp = RedTeamExperiment(cases=[_case("c0")], attack_strategies=[_RunStatsStrategy()], evaluators=[evaluator])
    exp.run_evaluations(agent_factory=_FakeSession)

    (prompt,) = evaluator.prompts
    assert f"<ActualEnvironmentState>[EnvironmentState(name='{RUN_RESULTS}'" in prompt
    assert "'turns_used': 3" in prompt
    assert "'pruned_branches': " + repr(_PRUNED) in prompt


def test_cached_rerun_keeps_run_stats(tmp_path):
    """A rerun served from the evaluation data store still shows turns and blocked attempts."""
    runs = 0

    def factory():
        nonlocal runs
        runs += 1
        return _FakeSession()

    store = LocalFileTaskResultStore(tmp_path)
    exp = RedTeamExperiment(cases=[_case("c0")], attack_strategies=[_RunStatsStrategy()], evaluators=[_PassEvaluator()])
    first = exp.run_evaluations(evaluation_data_store=store, agent_factory=factory).attack_results()
    second = exp.run_evaluations(evaluation_data_store=store, agent_factory=factory).attack_results()

    assert runs == 1  # the second run came from the cache
    for (result,) in (first, second):
        assert result.turns_used == 3
        assert result.backtracks == 1
        assert result.pruned_branches == _PRUNED


class _BreachEvaluator(Evaluator):
    """Scores every attack as a full breach."""

    def evaluate(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=False, reason="breached")]


def test_overall_score_excludes_errored_attacks():
    """`overall_score` keeps the red team rule: an errored attack is not a 0.0 judgment."""

    def task(case):
        if case.name.startswith("c1"):
            raise RuntimeError("target down")
        return {"output": []}

    exp = RedTeamExperiment(
        cases=[_case("c0"), _case("c1")], attack_strategies=[_StubStrategy()], evaluators=[_BreachEvaluator()]
    )
    report = exp.run_evaluations(task=task)

    assert [r.state for r in report.attack_results()] == ["breached", "errored"]
    assert report.overall_score == 1.0
