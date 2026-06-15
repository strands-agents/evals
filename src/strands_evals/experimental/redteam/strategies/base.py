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
    """Result of one `AttackStrategy.run_attack` execution.

    `strategy_succeeded` / `strategy_score` are observability only; the authoritative verdict comes from
    `AttackSuccessEvaluator` over the trace.

    Attributes:
        conversation: Ordered attacker/target turns; mapped to `output`.
        strategy_succeeded: The strategy's own early-stop signal, not the verdict.
        strategy_score: The strategy's internal stop-decision score, if any.
        metadata: Free-form run details (e.g. `turns_used`, `backtracks`).
        pruned_branches: Refused (attacker, target) pairs dropped during backtracking; empty for append-only
            strategies.
    """

    conversation: list[dict[str, Any]]
    strategy_succeeded: bool = False
    strategy_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pruned_branches: list[dict[str, Any]] = field(default_factory=list)


class AttackStrategy(ABC):
    """Base class for red team attack strategies.

    The task runner injects a `TargetSession` into `run_attack`; strategies drive it via `invoke` (and
    `snapshot`/`restore` if they backtrack).
    """

    def __init__(self, *, label: str | None = None) -> None:
        """Initialize the strategy.

        Args:
            label: Instance identifier for cross-product case naming and report grouping. Defaults to `name`.
                Pass distinct labels to compare two instances of the same strategy in one experiment.
        """
        self._label = label

    @property
    @abstractmethod
    def name(self) -> str:
        """Type identifier for the strategy (e.g. `"crescendo"`)."""
        ...

    @property
    def label(self) -> str:
        """Instance identifier; defaults to `name`."""
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
        """Run the multi-turn attack and return its result.

        Args:
            case: The red team case carrying the attack goal.
            target_session: Session for talking to the target; use `target_session.invoke(message)`.
            max_turns: Experiment-level ceiling. A strategy with its own `max_turns` should run
                `min(self._max_turns, max_turns)`.
            model: Model for any strategy-internal LLM calls; ctor model takes precedence.
            **kwargs: Reserved for forward compatibility.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Per-case reset hook; no-op by default. Override only if `self` holds mutable state."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy's static config.

        The base persists `strategy_type` and `label`; subclasses override to add their own ctor fields.
        """
        out: dict[str, Any] = {"strategy_type": type(self).__name__}
        if self._label is not None:
            out["label"] = self._label
        return out

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        custom_strategies: list[type[AttackStrategy]] | None = None,
    ) -> AttackStrategy:
        """Reconstruct a strategy from its `to_dict` payload.

        Resolves `strategy_type` against the built-in subclasses plus any `custom_strategies` the caller passes.
        """
        # Lazy import: subclasses inherit from AttackStrategy, so importing them at module top
        # would cycle (base -> strategies/__init__ -> crescendo -> base).
        from . import (
            BadLikertJudgeStrategy,
            CrescendoStrategy,
            GoatStrategy,
            PairStrategy,
            PromptStrategy,
            SequentialBreakStrategy,
        )

        registry: dict[str, type[AttackStrategy]] = {
            BadLikertJudgeStrategy.__name__: BadLikertJudgeStrategy,
            CrescendoStrategy.__name__: CrescendoStrategy,
            GoatStrategy.__name__: GoatStrategy,
            PairStrategy.__name__: PairStrategy,
            PromptStrategy.__name__: PromptStrategy,
            SequentialBreakStrategy.__name__: SequentialBreakStrategy,
        }
        for custom in custom_strategies or []:
            registry[custom.__name__] = custom

        payload = dict(data)
        type_name = payload.pop("strategy_type", None)
        if type_name is None:
            raise ValueError("Strategy dict is missing required key 'strategy_type'.")
        if type_name not in registry:
            raise ValueError(f"Cannot find strategy {type_name!r}. Pass `custom_strategies=[...]` to register it.")
        target_cls = registry[type_name]
        return target_cls(**payload)
