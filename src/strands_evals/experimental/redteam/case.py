"""Red team case type."""

from typing import Any

from pydantic import PrivateAttr, model_validator
from typing_extensions import Self

from ...case import Case
from ...types import InputT, OutputT
from .strategies.base import AttackStrategy
from .types import RedTeamConfig


class RedTeamCase(Case[InputT, OutputT]):
    """Case carrying a typed RedTeamConfig. AttackGoal fields are mirrored into metadata."""

    config: RedTeamConfig
    # Runtime only: a private attribute isn't serialized, and `RedTeamExperiment` sets it on every run.
    _strategy: AttackStrategy | None = PrivateAttr(default=None)

    @property
    def strategy(self) -> AttackStrategy:
        """The attack strategy for this case x strategy variant.

        `RedTeamExperiment` attaches its own instance to each expanded variant, replacing any strategy set on
        the base case; base cases have none. Set it by assignment (`case.strategy = ...`); it isn't a
        constructor argument.

        Raises:
            AttributeError: If no strategy is attached, so `hasattr` and `getattr(case, "strategy", None)` work.
        """
        if self._strategy is None:
            raise AttributeError(
                f"RedTeamCase {self.name!r} has no strategy; run it through RedTeamExperiment "
                "with attack_strategies set."
            )
        return self._strategy

    @strategy.setter
    def strategy(self, value: AttackStrategy) -> None:
        self._strategy = value

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """Deep-copy the case but share its strategy.

        One strategy instance is shared by all of an experiment's variants, and it can hold a `Model` that
        can't be deep-copied (the default `BedrockModel` holds thread locks).
        """
        memo = {} if memo is None else memo
        if self._strategy is not None:
            memo[id(self._strategy)] = self._strategy
        return super().__deepcopy__(memo)

    @model_validator(mode="after")
    def _sync_metadata_from_config(self) -> Self:
        dump = dict(self.config.attack_goal.model_dump())
        if self.metadata is None:
            self.metadata = dump
        else:
            for key, value in dump.items():
                self.metadata.setdefault(key, value)
        return self
