"""Base class for attack strategies."""

from abc import ABC, abstractmethod


class AttackStrategy(ABC):
    """Base class for multi-turn attack strategies.

    Two extension paths are supported:

    - Prompt-based strategies expose ``system_prompt_template`` and delegate
      turn-by-turn adaptation to ``ActorSimulator``. ``enhance()`` is not
      invoked by the current runner for these. ``PromptStrategy`` is the
      built-in implementation.
    - Turn-level strategies implement ``enhance()`` to produce the next
      adversarial message given the growing conversation. A runner path
      for these lands alongside the strategies that need it.

    ``enhance()`` receives additional context via ``**kwargs`` so the
    interface can grow without breaking existing subclasses. ``reset()``
    lets strategies clear per-case state when a runner reuses instances.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for reporting and case naming."""
        ...

    def enhance(self, prompt: str, **kwargs) -> str:
        """Generate the next adversarial message for one turn.

        Default implementation is a no-op passthrough; prompt-based strategies
        rely on ``system_prompt_template`` and do not need to override this.
        Turn-level strategies should override to produce the next adversarial
        message from ``conversation_history``, ``turn_number``, and friends
        passed via ``kwargs``.

        Args:
            prompt: The initial attack prompt (from the case input).
            **kwargs: Optional contextual data supplied by turn-level runners.

        Returns:
            The next message to send to the target agent.
        """
        return prompt

    @property
    def system_prompt_template(self) -> str | None:
        """Optional ActorSimulator system-prompt template. None for turn-level strategies."""
        return None

    def reset(self) -> None:  # noqa: B027 - intentionally not abstract; override is optional
        """Reset internal state between cases."""
