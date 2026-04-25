"""Base class for attack strategies.

Defines the interface that all attack strategies implement. The interface
is designed so that:
- enhance() receives kwargs that grow over time without breaking
  existing strategies (forward-compatible via **kwargs)
- reset() lets the experiment runner reuse strategy instances across
  cases without leaking state
- Strategies are stateful per-case but stateless across cases
"""

from abc import ABC, abstractmethod


class AttackStrategy(ABC):
    """Base class for multi-turn attack strategies.

    Subclasses implement enhance() to produce the next adversarial
    message given a prompt and conversation context.

    The runner calls enhance() once per turn, passing the growing
    conversation_history so the strategy can adapt. Additional
    context (attack_goal, tool_definitions, etc.) is passed via
    kwargs — strategies take what they need and ignore the rest.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for reporting and case naming."""
        ...

    @abstractmethod
    def enhance(self, prompt: str, **kwargs) -> str:
        """Generate the next adversarial message for one turn.

        Args:
            prompt: The initial attack prompt (from the case input).
            **kwargs: Contextual data passed by the runner:
                - conversation_history: list[dict] — turns so far
                - attack_goal: AttackGoal — structured objective
                - turn_number: int — current turn index
                - max_turns: int — turn budget
                - tool_definitions: list[dict] — target agent's tools
                - trajectory: list[dict] — tool calls observed so far

        Returns:
            The next message to send to the target agent.
        """
        ...

    def reset(self) -> None:  # noqa: B027 - intentionally not abstract; override is optional
        """Reset internal state between cases."""
