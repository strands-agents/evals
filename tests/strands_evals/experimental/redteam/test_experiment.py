"""Tests for RedTeamExperiment."""

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.experiment import RedTeamExperiment
from strands_evals.experimental.redteam.report import RedTeamReport
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _StubStrategy(AttackStrategy):
    def __init__(self, name="stub", *, label=None):
        super().__init__(label=label)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run_attack(self, case, call_target, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        return AttackRunResult(conversation=[], metadata={})

    def reset(self) -> None:
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
        agent=lambda msg: "ok",
        attack_strategies=[_StubStrategy()],
        max_turns=2,
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
        agent=lambda msg: "ok",
        attack_strategies=[_StubStrategy(label="cre-10"), _StubStrategy(label="cre-30")],
    )
    exp.run_evaluations(task=task)

    assert sorted(captured) == ["c0__cre-10", "c0__cre-30", "c1__cre-10", "c1__cre-30"]
