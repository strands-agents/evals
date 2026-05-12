"""Prompt-based attack strategy using ActorSimulator prompt templates."""

from __future__ import annotations

from strands_evals.redteam.strategies.base import AttackStrategy


class PromptStrategy(AttackStrategy):
    """AttackStrategy that delegates to ActorSimulator via system prompt templates."""

    def __init__(self, strategy_name: str, system_prompt_template: str):
        self._name = strategy_name
        self._system_prompt_template = system_prompt_template

    @property
    def name(self) -> str:
        return self._name

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template
