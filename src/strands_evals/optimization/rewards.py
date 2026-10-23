"""Reward functions shared with RL trainers, and their evaluator adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import NotRequired, TypedDict

from ..evaluators.evaluator import Evaluator


class RewardResult(TypedDict):
    reward_value: float
    reason: NotRequired[str]


class RewardFunction(ABC):
    """A reward over rollout fields (mirroring `EvaluationData`); absorb extra fields with `**_`."""

    @abstractmethod
    def __call__(self, **rollout_fields: Any) -> RewardResult: ...


class RewardFunctionEvaluator(Evaluator):
    """Adapts a `RewardFunction` into an `Evaluator`; the reason falls back to `f"reward_value={score}"`."""

    def __init__(self, reward: RewardFunction, *, pass_threshold: float = 1.0, name: str | None = None):
        raise NotImplementedError
