from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.helpfulness import get_template


class HelpfulnessScore(str, Enum):
    """Categorical helpfulness ratings."""

    NOT_HELPFUL = "Not helpful at all"
    VERY_UNHELPFUL = "Very unhelpful"
    SOMEWHAT_UNHELPFUL = "Somewhat unhelpful"
    NEUTRAL = "Neutral/Mixed"
    SOMEWHAT_HELPFUL = "Somewhat helpful"
    VERY_HELPFUL = "Very helpful"
    ABOVE_AND_BEYOND = "Above and beyond"


class HelpfulnessRating(BaseModel):
    """Structured output for helpfulness evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: HelpfulnessScore = Field(description="Categorical helpfulness rating")


class HelpfulnessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates helpfulness of agent responses from the user's perspective."""

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        HelpfulnessScore.NOT_HELPFUL: 0.0,
        HelpfulnessScore.VERY_UNHELPFUL: 0.167,
        HelpfulnessScore.SOMEWHAT_UNHELPFUL: 0.333,
        HelpfulnessScore.NEUTRAL: 0.5,
        HelpfulnessScore.SOMEWHAT_HELPFUL: 0.667,
        HelpfulnessScore.VERY_HELPFUL: 0.833,
        HelpfulnessScore.ABOVE_AND_BEYOND: 1.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
        include_inputs: bool = True,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        self.include_inputs = include_inputs

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=HelpfulnessRating)
        rating = cast(HelpfulnessRating, result.structured_output)
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
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=HelpfulnessRating)
        rating = cast(HelpfulnessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
