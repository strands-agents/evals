"""Red team case type."""

from pydantic import model_validator
from typing_extensions import Self

from ...case import Case
from ...types import InputT, OutputT
from .types import RedTeamConfig


class RedTeamCase(Case[InputT, OutputT]):
    """Case carrying a typed RedTeamConfig. AttackGoal fields are mirrored into metadata."""

    config: RedTeamConfig

    @model_validator(mode="after")
    def _sync_metadata_from_config(self) -> Self:
        dump = {
            **self.config.attack_goal.model_dump(),
            "strategy": self.config.strategy,
        }
        if self.metadata is None:
            self.metadata = dump
        else:
            for key, value in dump.items():
                self.metadata.setdefault(key, value)
        return self
