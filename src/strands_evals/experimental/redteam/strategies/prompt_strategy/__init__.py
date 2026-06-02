"""Prompt-based attack strategies."""

from __future__ import annotations

from ..base import AttackStrategy


class PromptStrategy(AttackStrategy):
    """AttackStrategy that drives the attack via a system prompt template."""

    def __init__(self, strategy_name: str, system_prompt_template: str):
        self._name = strategy_name
        self._system_prompt_template = system_prompt_template

    @property
    def name(self) -> str:
        return self._name

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template


__all__ = ["PromptStrategy"]
