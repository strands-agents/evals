"""Base class for attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AttackStrategy(ABC):
    """Base class for red team attack strategies.

    Prompt-based strategies expose ``system_prompt_template``; turn-level
    strategies override ``enhance()`` to adapt per turn.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def system_prompt_template(self) -> str | None:
        """System prompt template if prompt-driven, else None."""
        return None

    def enhance(self, prompt: str, **kwargs: Any) -> str:
        """Return the next adversarial message. Default: passthrough.

        Called once per turn by the task runner with ``target_response``,
        ``conversation``, and ``attack_goal`` in ``kwargs``. Algorithmic
        strategies (e.g., PAIR, TAP) override this and may hold an
        ``Evaluator`` on the instance to score turn-level candidates.
        """
        return prompt

    def reset(self) -> None:  # noqa: B027
        """Clear runtime state between cases.

        Strategy instances are shared across cases via the registry, so
        overrides must clear any per-case mutable state here. Called by
        the task runner before each case.
        """
