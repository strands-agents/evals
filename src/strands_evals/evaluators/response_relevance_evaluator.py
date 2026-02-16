from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.response_relevance import get_template


class ResponseRelevanceScore(str, Enum):
    """Categorical response relevance ratings."""

    NOT_AT_ALL = "Not At All"
    NOT_GENERALLY = "Not Generally"
    NEUTRAL_MIXED = "Neutral/Mixed"
    GENERALLY_YES = "Generally Yes"
    COMPLETELY_YES = "Completely Yes"


class ResponseRelevanceRating(BaseModel):
    """Structured output for response relevance evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: ResponseRelevanceScore = Field(description="Categorical response relevance rating")


class ResponseRelevanceEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates the relevance of agent responses to user questions."""

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        ResponseRelevanceScore.NOT_AT_ALL: 0.0,
        ResponseRelevanceScore.NOT_GENERALLY: 0.25,
        ResponseRelevanceScore.NEUTRAL_MIXED: 0.5,
        ResponseRelevanceScore.GENERALLY_YES: 0.75,
        ResponseRelevanceScore.COMPLETELY_YES: 1.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
        include_inputs: bool = True,
    ):
        super().__init__()
        self.system_prompt = system_prompt or get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        self.include_inputs = include_inputs

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=ResponseRelevanceRating)
        return self._create_evaluation_output(result)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=ResponseRelevanceRating)
        return self._create_evaluation_output(result)

    def _create_evaluation_output(self, result: AgentResult) -> list[EvaluationOutput]:
        rating = cast(ResponseRelevanceRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
