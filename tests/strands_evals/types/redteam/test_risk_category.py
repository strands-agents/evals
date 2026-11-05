"""Tests for risk category aliasing of deprecated slugs."""

import warnings

import pytest

from strands_evals.redteam.case import RedTeamCase
from strands_evals.types.redteam import RISK_CATEGORIES, AttackGoal, RedTeamConfig
from strands_evals.types.redteam.risk_category import DEPRECATED_RISK_CATEGORIES


def test_deprecated_aliases_point_at_current_categories():
    assert set(DEPRECATED_RISK_CATEGORIES).isdisjoint(RISK_CATEGORIES)
    assert set(DEPRECATED_RISK_CATEGORIES.values()) <= set(RISK_CATEGORIES)


@pytest.mark.parametrize(("old", "new"), list(DEPRECATED_RISK_CATEGORIES.items()))
def test_attack_goal_resolves_deprecated_category(old, new):
    with pytest.warns(DeprecationWarning, match=f"'{old}' is deprecated"):
        goal = AttackGoal(risk_category=old, actor_goal="g")
    assert goal.risk_category == new


def test_case_metadata_carries_resolved_category():
    with pytest.warns(DeprecationWarning):
        case = RedTeamCase(
            name="c",
            input="hi",
            config=RedTeamConfig(attack_goal=AttackGoal(risk_category="system_prompt_leak", actor_goal="g")),
        )
    assert case.metadata["risk_category"] == "sensitive_data"


def test_reloaded_case_resolves_deprecated_category_in_metadata():
    saved = {
        "name": "c",
        "input": "hi",
        "metadata": {"risk_category": "system_prompt_leak", "actor_goal": "g"},
        "config": {"attack_goal": {"risk_category": "system_prompt_leak", "actor_goal": "g"}},
    }
    with pytest.warns(DeprecationWarning):
        case = RedTeamCase.model_validate(saved)
    assert case.config.attack_goal.risk_category == "sensitive_data"
    assert case.metadata["risk_category"] == "sensitive_data"


@pytest.mark.parametrize("category", ["prompt_injection", "my_custom_category"])
def test_attack_goal_keeps_other_categories_without_warning(category):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        goal = AttackGoal(risk_category=category, actor_goal="g")
    assert goal.risk_category == category
