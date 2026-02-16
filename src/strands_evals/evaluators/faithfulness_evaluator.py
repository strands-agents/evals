from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.faithfulness import get_template


class FaithfulnessScore(str, Enum):
    """Categorical faithfulness ratings."""

    NOT_AT_ALL = "Not At All"
    NOT_GENERALLY = "Not Generally"
    NEUTRAL = "Neutral/Mixed"
    GENERALLY_YES = "Generally Yes"
    COMPLETELY_YES = "Completely Yes"


class FaithfulnessRating(BaseModel):
    """Structured output for faithfulness evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: FaithfulnessScore = Field(description="Categorical faithfulness rating")


class FaithfulnessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates faithfulness of agent responses against conversation history."""

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        FaithfulnessScore.NOT_AT_ALL: 0.0,
        FaithfulnessScore.NOT_GENERALLY: 0.25,
        FaithfulnessScore.NEUTRAL: 0.5,
        FaithfulnessScore.GENERALLY_YES: 0.75,
        FaithfulnessScore.COMPLETELY_YES: 1.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=FaithfulnessRating)
        rating = cast(FaithfulnessRating, result.structured_output)
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
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=FaithfulnessRating)
        rating = cast(FaithfulnessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
