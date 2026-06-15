"""Shared types for red team evaluation."""

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]


class AttackGoal(BaseModel):
    """User-facing attack specification for a red team case."""

    risk_category: str
    actor_goal: str
    context: str = ""
    severity: Severity = "medium"
    success_criteria: str | None = None


class RedTeamConfig(BaseModel):
    """Runtime configuration for a red team case.

    Cases are strategy-agnostic: `RedTeamExperiment` holds the strategy and applies
    it via cross-product at run time, so this config carries only the attack goal
    and actor traits.
    """

    attack_goal: AttackGoal
    traits: dict = Field(default_factory=dict)
