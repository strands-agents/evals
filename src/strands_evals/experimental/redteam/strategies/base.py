"""Base class for attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from strands.models.model import Model

if TYPE_CHECKING:
    from ..case import RedTeamCase


@dataclass
class AttackRunResult:
    """Result of a single ``AttackStrategy.run_attack`` execution.

    The strategy owns the multi-turn loop and returns this object; the task
    runner assembles the ``{"output", "trajectory", ...}`` dict the base
    ``Experiment`` expects, pairing this conversation with the tool trace it
    captured via ``call_target``. The authoritative success verdict is
    re-computed independently by ``AttackSuccessEvaluator`` over the trace, so
    ``strategy_succeeded``/``strategy_score`` are observability only.

    The tool trace is deliberately NOT a field here: it is captured by the
    injected ``call_target`` closure that the task runner owns, so a strategy
    has no role in producing it.

    Attributes:
        conversation: Ordered attacker/target turns; mapped to ``output``.
        strategy_succeeded: The strategy's own "early-stop fired" signal, NOT the
            verdict (the evaluator decides pass/fail).
        strategy_score: The strategy's internal stop-decision score, if any.
        metadata: Free-form run details (e.g. ``turns_used``, ``backtracks``).
    """

    conversation: list[dict[str, Any]]
    strategy_succeeded: bool = False
    strategy_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttackStrategy(ABC):
    """Base class for red team attack strategies.

    Every strategy owns its own multi-turn loop via ``run_attack``. The task
    runner injects a ``call_target`` callable (which already handles target
    invocation, tool-trace capture, and per-case isolation), so strategies
    never build target-calling themselves.
    """

    def __init__(self, *, label: str | None = None) -> None:
        """Initialize the strategy.

        Args:
            label: Instance identifier used for cross-product case naming, cache
                keys, and report grouping. Defaults to ``name`` (the type id).
                Pass distinct labels to compare instances of the same strategy
                (e.g. different ``max_turns``) in one experiment.
        """
        self._label = label

    @property
    @abstractmethod
    def name(self) -> str:
        """Type identifier for the strategy (e.g. ``"crescendo"``)."""
        ...

    @property
    def label(self) -> str:
        """Instance identifier; defaults to ``name`` when none was provided."""
        return self._label or self.name

    @property
    def system_prompt_template(self) -> str | None:
        """System prompt template if prompt-driven, else None."""
        return None

    @abstractmethod
    def run_attack(
        self,
        case: RedTeamCase,
        call_target: Callable[[str], str],
        *,
        max_turns: int,
        model: Model | str | None = None,
        **kwargs: Any,
    ) -> AttackRunResult:
        """Run the full multi-turn attack and return its result.

        Args:
            case: The red team case carrying the attack goal.
            call_target: Injected callable mapping an attacker message to the
                target's response. Already captures the tool trace and applies
                per-case isolation; strategies must not build their own.
            max_turns: Experiment-level ceiling on attacker/target turns. A
                strategy with its own ``max_turns`` should run the smaller of the
                two (the ceiling never lets a strategy exceed the experiment
                budget, but a strategy may choose to stop sooner).
            model: Model for any strategy-internal LLM calls (e.g. the attacker
                agent). Strategies resolve this against their own ``__init__``
                model, ctor value taking precedence.
            **kwargs: Reserved for forward compatibility.

        Returns:
            An :class:`AttackRunResult` capturing the conversation and metadata.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Clear runtime state between cases.

        Strategy instances are shared across cases via the registry, so
        overrides must clear any per-case mutable state here. Called by
        the task runner before each case.
        """
