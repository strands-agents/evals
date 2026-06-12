"""Base class for attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from strands.models.model import Model

if TYPE_CHECKING:
    from ..case import RedTeamCase
    from .target_session import TargetSession


@dataclass
class AttackRunResult:
    """Result of a single ``AttackStrategy.run_attack`` execution.

    The strategy owns the multi-turn loop and returns this object; the task
    runner assembles the ``{"output", "trajectory", ...}`` dict the base
    ``Experiment`` expects, pairing this conversation with the tool trace the
    injected ``TargetSession`` captured. The authoritative success verdict is
    re-computed independently by ``AttackSuccessEvaluator`` over the trace, so
    ``strategy_succeeded``/``strategy_score`` are observability only.

    The tool trace is deliberately NOT a field here: it lives on the
    ``TargetSession`` the task runner owns, so a strategy has no role in
    producing it.

    Attributes:
        conversation: Ordered attacker/target turns; mapped to ``output``.
        strategy_succeeded: The strategy's own "early-stop fired" signal, NOT the
            verdict (the evaluator decides pass/fail).
        strategy_score: The strategy's internal stop-decision score, if any.
        metadata: Free-form run details (e.g. ``turns_used``, ``backtracks``).
        pruned_branches: Refused turns dropped from ``conversation`` during
            backtracking, kept as evidence ("defended N framings, complied on
            N+1"). Each entry is an ordered attacker/target pair. Empty for
            strategies that do not backtrack.
    """

    conversation: list[dict[str, Any]]
    strategy_succeeded: bool = False
    strategy_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pruned_branches: list[dict[str, Any]] = field(default_factory=list)


class AttackStrategy(ABC):
    """Base class for red team attack strategies.

    Every strategy owns its own multi-turn loop via ``run_attack``. The task
    runner injects a ``TargetSession`` (which already handles target invocation,
    tool-trace capture, and per-case isolation), so strategies never build
    target-calling themselves. Strategies that backtrack use the session's
    snapshot/restore; strategies that only converse forward use
    ``session.invoke`` alone.
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

    @abstractmethod
    def run_attack(
        self,
        case: RedTeamCase,
        target_session: TargetSession,
        *,
        max_turns: int,
        model: Model | str | None = None,
        **kwargs: Any,
    ) -> AttackRunResult:
        """Run the full multi-turn attack and return its result.

        Args:
            case: The red team case carrying the attack goal.
            target_session: Injected session for talking to the target. Use
                ``target_session.invoke(message)`` to send a turn; it captures
                the tool trace and applies per-case isolation, so strategies must
                not build their own target-calling. Backtracking strategies use
                ``target_session.snapshot`` / ``restore``.
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
        """Optional per-case reset hook; a no-op by default.

        The built-in strategies are STATELESS across cases: they hold only static
        config on ``self`` and build their attacker/judge agents as run_attack-local
        objects (the judge fresh per scoring call), so no per-case state can carry and
        they do not override this. Statelessness is the primary isolation mechanism,
        NOT this hook. The task runner still calls ``reset()`` before each case, so the
        hook remains for the rare custom strategy that genuinely must hold cross-case
        mutable state on ``self`` and scrub it here.
        """
