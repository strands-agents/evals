from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import TypeVar, Union

from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.goal_success_rate import get_template

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class GoalSuccessScore(str, Enum):
    """Binary goal success ratings."""

    YES = "Yes"
    NO = "No"


class GoalSuccessRating(BaseModel):
    """Structured output for goal success evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: GoalSuccessScore = Field(description="Score should be one of 'Yes' or 'No'")


class GoalSuccessRateEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether all user goals were successfully achieved in a conversation."""

    evaluation_level = EvaluationLevel.SESSION_LEVEL

    _score_mapping = {
        GoalSuccessScore.YES: 1.0,
        GoalSuccessScore.NO: 0.0,
    }

    def __init__(
        self,
        version: str = "v1",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        prompt = self._format_prompt(evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=GoalSuccessRating)
        rating = cast(GoalSuccessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        prompt = self._format_prompt(evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=GoalSuccessRating)
        rating = cast(GoalSuccessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    def _format_prompt(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        """Format evaluation prompt from evaluation case."""

        goal = evaluation_case.metadata.get("goal")
        session_input = self._parse_trajectory(evaluation_case)

        parts = []

        if goal:
            parts.append(f"# User Goal\n{goal}")

        if session_input.available_tools:
            parts.append(f"# Available tools\n{self._format_tools(session_input.available_tools)}")

        if session_input.session_history:
            parts.append(f"# Conversation record\n{self._format_session_history(session_input.session_history)}")

        return "\n\n".join(parts)
