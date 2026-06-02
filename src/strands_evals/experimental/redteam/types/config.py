"""Shared types for red team evaluation."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

Severity = Literal["low", "medium", "high", "critical"]


class AttackGoal(BaseModel):
    """User-facing attack specification for a red team case."""

    risk_category: str
    actor_goal: str
    context: str = ""
    severity: Severity = "medium"
    success_criteria: str | None = None


class RedTeamConfig(BaseModel):
    """Full runtime configuration for a red team case.

    Combines the user-facing AttackGoal with strategy and simulator details.
    The generator fills these automatically; for custom cases, ``strategy``
    defaults to the built-in default and ``system_prompt_template`` is
    resolved from the strategy registry when omitted.
    """

    attack_goal: AttackGoal
    traits: dict = Field(default_factory=dict)
    system_prompt_template: str | None = None
    strategy: str | None = None

    @model_validator(mode="after")
    def _resolve_template_from_strategy(self) -> Self:
        if self.system_prompt_template is not None:
            return self
        # Lazy import to avoid circular dep with strategies/.
        from ..strategies import BUILTIN_STRATEGIES, DEFAULT_STRATEGY

        strategy_name = self.strategy or DEFAULT_STRATEGY
        strategy = BUILTIN_STRATEGIES.get(strategy_name)
        if strategy is not None and strategy.system_prompt_template is not None:
            self.system_prompt_template = strategy.system_prompt_template
            if self.strategy is None:
                self.strategy = strategy_name
        return self
