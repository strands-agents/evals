"""Recovery Strategy Evaluator.

Evaluates appropriateness of agent's recovery strategy when handling failures.
Focuses on what the agent *did* (actions), not what it *said* (communication).
"""

from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.trace import EvaluationLevel
from ...evaluators.evaluator import Evaluator
from .prompt_templates.recovery_strategy_v0 import SYSTEM_PROMPT as SYSTEM_PROMPT_V0

_PROMPT_VERSIONS = {"v0": SYSTEM_PROMPT_V0}


class RecoveryStrategyScore(str, Enum):
    """Categorical recovery strategy ratings."""

    FAILURE = "Failure"
    POOR = "Poor"
    ACCEPTABLE = "Acceptable"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class RecoveryStrategyRating(BaseModel):
    """Structured output for recovery strategy evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: RecoveryStrategyScore = Field(description="Categorical recovery strategy rating")


class RecoveryStrategyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates appropriateness of agent's recovery strategy when handling failures.

    Scores the agent's actions and decisions (not communication) on a 5-point scale:
    - Excellent (1.0): Optimal recovery actions, justified retries, broad exploration
    - Good (0.75): Reasonable strategy with minor inefficiencies
    - Acceptable (0.5): No failures occurred, or basic recovery attempted
    - Poor (0.25): Wasteful retries, ignored working alternatives
    - Failure (0.0): No adaptation, fixated on broken tools

    Example::

        from strands_evals.chaos.evaluators import RecoveryStrategyEvaluator

        evaluator = RecoveryStrategyEvaluator()
        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=cases,
            evaluators=[evaluator],
        )
    """

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        RecoveryStrategyScore.FAILURE: 0.0,
        RecoveryStrategyScore.POOR: 0.25,
        RecoveryStrategyScore.ACCEPTABLE: 0.5,
        RecoveryStrategyScore.GOOD: 0.75,
        RecoveryStrategyScore.EXCELLENT: 1.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.version = version
        default_prompt = _PROMPT_VERSIONS.get(version, SYSTEM_PROMPT_V0)
        self.system_prompt = system_prompt if system_prompt is not None else default_prompt
        self.model = model

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=RecoveryStrategyRating)
        rating = cast(RecoveryStrategyRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=RecoveryStrategyRating)
        rating = cast(RecoveryStrategyRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
