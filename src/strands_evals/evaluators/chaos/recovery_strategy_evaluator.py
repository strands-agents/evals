from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.trace import EvaluationLevel
from ..evaluator import Evaluator
from .prompt_templates.recovery_strategy import get_template


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
    """Evaluates appropriateness of agent's recovery strategy when handling failures."""

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
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.version = version
        default_prompt = get_template(version).SYSTEM_PROMPT
        self.system_prompt = system_prompt if system_prompt is not None else default_prompt
        self.model = model

    def _build_output(self, rating: RecoveryStrategyRating) -> list[EvaluationOutput]:
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=RecoveryStrategyRating)
        rating = cast(RecoveryStrategyRating, result.structured_output)
        return self._build_output(rating)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=RecoveryStrategyRating)
        rating = cast(RecoveryStrategyRating, result.structured_output)
        return self._build_output(rating)
