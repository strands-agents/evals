from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.tool_parameter_accuracy import get_template


class ToolParameterAccuracyScore(str, Enum):
    """Binary tool parameter accuracy ratings."""

    YES = "Yes"
    NO = "No"


class ToolParameterAccuracyRating(BaseModel):
    """Structured output for tool parameter accuracy evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: ToolParameterAccuracyScore = Field(description="Score should be one of 'Yes' or 'No'")


class ToolParameterAccuracyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether tool call parameters faithfully use information from the preceding context."""

    evaluation_level = EvaluationLevel.TOOL_LEVEL

    _score_mapping = {
        ToolParameterAccuracyScore.YES: 1.0,
        ToolParameterAccuracyScore.NO: 0.0,
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
        tool_inputs = self._parse_trajectory(evaluation_case)
        results = []

        for tool_input in tool_inputs:
            prompt = self._format_tool_level_prompt(tool_input)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = evaluator_agent(prompt, structured_output_model=ToolParameterAccuracyRating)
            rating = cast(ToolParameterAccuracyRating, result.structured_output)
            normalized_score = self._score_mapping[rating.score]
            results.append(
                EvaluationOutput(
                    score=normalized_score,
                    test_pass=normalized_score == 1.0,
                    reason=rating.reasoning,
                    label=rating.score,
                )
            )

        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        tool_inputs = self._parse_trajectory(evaluation_case)
        results = []

        for tool_input in tool_inputs:
            prompt = self._format_tool_level_prompt(tool_input)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = await evaluator_agent.invoke_async(prompt, structured_output_model=ToolParameterAccuracyRating)
            rating = cast(ToolParameterAccuracyRating, result.structured_output)
            normalized_score = self._score_mapping[rating.score]
            results.append(
                EvaluationOutput(
                    score=normalized_score,
                    test_pass=normalized_score == 1.0,
                    reason=rating.reasoning,
                    label=rating.score,
                )
            )

        return results
