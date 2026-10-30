"""Reward functions shared with RL trainers, and their evaluator adapter. Imports no other optimizer layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import NotRequired, TypedDict

from ...evaluators.evaluator import Evaluator
from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.optimization import Range


class RewardResult(TypedDict):
    """What a `RewardFunction` returns: the number, plus an optional reason (an RL trainer reads only the number)."""

    reward_value: float
    reason: NotRequired[str]


class RewardFunction(ABC):
    """A reward over rollout fields (mirroring `EvaluationData`); absorb extra fields with `**_`."""

    @abstractmethod
    def __call__(self, **rollout_fields: Any) -> RewardResult: ...


class RewardFunctionEvaluator(Evaluator[InputT, OutputT]):
    """Adapts a `RewardFunction` into an `Evaluator`; the reason falls back to `f"reward_value={score}"`.

    Named after the reward's class unless `name=` is given, so two different rewards need no `name=`;
    two instances of the same reward class do. Refer to it in an objective by that name or by the
    instance (`Maximize(instance)`); `Maximize(RewardFunctionEvaluator)` names no particular reward.
    """

    def __init__(
        self,
        reward: RewardFunction,
        *,
        pass_threshold: float = 1.0,
        name: str | None = None,
        score_range: Range = (0.0, 1.0),
    ):
        """Wrap `reward` as an evaluator.

        Args:
            reward: The reward function to adapt.
            pass_threshold: `test_pass` is `reward_value >= pass_threshold`.
            name: The evaluator name; defaults to the reward's class name, e.g. `"ExactMatchReward"`.
            score_range: Where `reward_value` lies, read as the evaluator's range (crash "worst", scaling,
                `WeightedSum`'s default scale); `(low, None)` for an unbounded reward.
        """
        raise NotImplementedError

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Call the reward with the case's fields; `test_pass` = `reward_value >= pass_threshold`."""
        raise NotImplementedError

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Same as `evaluate`; rewards are plain functions, so nothing is awaited."""
        raise NotImplementedError
