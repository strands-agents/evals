"""Tests for RedTeamExperiment."""

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.experiment import RedTeamExperiment
from strands_evals.experimental.redteam.report import RedTeamReport
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


def _case(name: str = "c0") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(
                risk_category="guideline_bypass",
                actor_goal="goal",
            ),
            system_prompt_template="prompt {actor_profile} {max_turns}",
            strategy="gradual_escalation",
        ),
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


def test_run_evaluations_uses_default_task_when_target_provided():
    """Target on construction enables run_evaluations() with no explicit task."""
    exp = RedTeamExperiment(cases=[], target=lambda msg: "ok", max_turns=2)
    report = exp.run_evaluations()
    assert isinstance(report, RedTeamReport)


def test_run_evaluations_raises_when_neither_target_nor_task():
    exp = RedTeamExperiment(cases=[_case()])
    with pytest.raises(ValueError, match="target.*task"):
        exp.run_evaluations()
