"""Tests for RedTeamCase metadata sync behavior."""

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


def _config(**attack_goal_kwargs) -> RedTeamConfig:
    kwargs = {"risk_category": "guideline_bypass", "actor_goal": "goal"}
    kwargs.update(attack_goal_kwargs)
    return RedTeamConfig(attack_goal=AttackGoal(**kwargs))


def test_metadata_none_is_populated_from_attack_goal():
    """A case with no metadata gets the full attack_goal dump as its metadata."""
    case = RedTeamCase(name="c0", input="hello", config=_config())

    assert case.metadata == case.config.attack_goal.model_dump()
    assert case.metadata["risk_category"] == "guideline_bypass"
    assert case.metadata["actor_goal"] == "goal"
    assert case.metadata["severity"] == "medium"


def test_existing_metadata_keys_are_preserved_not_overwritten():
    """Caller-provided metadata wins over the attack_goal dump for overlapping keys."""
    case = RedTeamCase(
        name="c0",
        input="hello",
        config=_config(severity="critical"),
        metadata={"severity": "low", "custom": "value"},
    )

    # setdefault semantics: caller's "severity" survives, config's does not override it.
    assert case.metadata["severity"] == "low"
    # Caller-only keys are untouched.
    assert case.metadata["custom"] == "value"
    # Keys absent from caller metadata are still filled in from attack_goal.
    assert case.metadata["risk_category"] == "guideline_bypass"
    assert case.metadata["actor_goal"] == "goal"


def test_partial_metadata_is_filled_in_from_attack_goal():
    """Keys the caller didn't set are backfilled from attack_goal without touching the rest."""
    case = RedTeamCase(
        name="c0",
        input="hello",
        config=_config(context="ctx", success_criteria="criteria"),
        metadata={"unrelated": "kept"},
    )

    assert case.metadata["unrelated"] == "kept"
    assert case.metadata["context"] == "ctx"
    assert case.metadata["success_criteria"] == "criteria"


def test_empty_dict_metadata_is_filled_in_from_attack_goal():
    """An explicit empty dict (falsy but not None) still gets backfilled, exercising the else branch."""
    case = RedTeamCase(name="c0", input="hello", config=_config(), metadata={})

    assert case.metadata == case.config.attack_goal.model_dump()
