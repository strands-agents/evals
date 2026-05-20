"""Tests for AttackStrategy implementations."""

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import BUILTIN_STRATEGIES, DEFAULT_STRATEGY
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


def test_redteam_config_resolves_default_template_for_custom_case():
    """Custom case with bare AttackGoal auto-fills template + strategy from default."""
    config = RedTeamConfig(attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="g"))
    assert config.strategy == DEFAULT_STRATEGY
    assert config.system_prompt_template == BUILTIN_STRATEGIES[DEFAULT_STRATEGY].system_prompt_template


def test_redteam_config_resolves_template_for_named_strategy():
    """Naming an existing strategy fills its template even if user omits it."""
    config = RedTeamConfig(
        attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="g"),
        strategy=DEFAULT_STRATEGY,
    )
    assert config.system_prompt_template == BUILTIN_STRATEGIES[DEFAULT_STRATEGY].system_prompt_template


def test_redteam_case_with_minimal_config():
    """Custom RedTeamCase construction works with just AttackGoal in config."""
    case = RedTeamCase(
        name="custom_0",
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="guideline_bypass", actor_goal="g")),
    )
    assert case.config.system_prompt_template is not None
    assert case.config.strategy == DEFAULT_STRATEGY
