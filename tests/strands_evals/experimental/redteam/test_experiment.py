"""Tests for RedTeamExperiment."""

import pytest
from strands.models.model import Model

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
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


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


def test_run_evaluations_uses_default_task_when_agent_provided():
    """Agent on construction enables run_evaluations() with no explicit task."""
    exp = RedTeamExperiment(
        cases=[_case()],
        agent=_FakeSession(),
        attack_strategies=[_StubStrategy()],
    )
    report = exp.run_evaluations()
    assert isinstance(report, RedTeamReport)


def test_run_evaluations_raises_when_neither_agent_nor_task():
    exp = RedTeamExperiment(cases=[_case()])
    with pytest.raises(ValueError, match="agent.*task"):
        exp.run_evaluations()


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
        agent=_FakeSession(),
        attack_strategies=[_StubStrategy(label="cre-10"), _StubStrategy(label="cre-30")],
    )
    exp.run_evaluations(task=task)

    assert sorted(captured) == ["c0__cre-10", "c0__cre-30", "c1__cre-10", "c1__cre-30"]


async def test_run_evaluations_async_returns_report():
    exp = RedTeamExperiment(
        cases=[_case()],
        agent=_FakeSession(),
        attack_strategies=[_StubStrategy()],
    )
    report = await exp.run_evaluations_async()
    assert isinstance(report, RedTeamReport)


async def test_max_workers_must_be_one():
    exp = RedTeamExperiment(cases=[_case()], agent=_FakeSession(), attack_strategies=[_StubStrategy()])
    with pytest.raises(ValueError, match="max_workers=1"):
        await exp.run_evaluations_async(max_workers=4)


def test_async_task_rejected():
    async def _async_task(case):
        return {"output": []}

    exp = RedTeamExperiment(cases=[_case()], agent=_FakeSession(), attack_strategies=[_StubStrategy()])
    with pytest.raises(ValueError, match="Async task is not supported"):
        exp.run_evaluations(task=_async_task)


def test_agent_setter_round_trip():
    """`exp.agent = ...` is the canonical way to attach a target after `from_file`."""
    exp = RedTeamExperiment(cases=[_case()], attack_strategies=[_StubStrategy()])
    assert exp.agent is None
    sess = _FakeSession()
    exp.agent = sess
    assert exp.agent is sess


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


def test_from_dict_round_trip_runs_after_setting_agent(tmp_path):
    """Reload via to_file/from_file, then attach an agent and run."""
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

    # Without agent, run_evaluations raises the existing message.
    with pytest.raises(ValueError, match="agent.*task"):
        loaded.run_evaluations()

    # Attach an agent and supply a stub task to skip real LLM calls.
    loaded.agent = _FakeSession()
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
        agent=_FakeSession(),
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
