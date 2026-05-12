"""Shared types for red team module."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["low", "medium", "high", "critical"]


class RedTeamCaseMetadata(BaseModel):
    """Typed metadata for a red team Case. Extra fields are allowed."""

    model_config = ConfigDict(extra="allow")

    attack_type: str
    actor_goal: str
    context: str = ""
    traits: dict = Field(default_factory=dict)
    severity: Severity = "medium"
    evaluation_metrics: list[str] = Field(default_factory=list)
    system_prompt_template: str | None = None
    strategy: str | None = None
