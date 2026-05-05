from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator
from .prompt_templates.refusal import get_template


class RefusalScore(str, Enum):
    """Binary refusal ratings."""

    YES = "Yes"
    NO = "No"


class RefusalRating(BaseModel):
    """Structured output for refusal evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score, using no more than 250 words")
    score: RefusalScore = Field(description="Score should be one of 'Yes' or 'No'")


class RefusalEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether agent responses refuse to address the prompt request."""

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        RefusalScore.YES: 1.0,
        RefusalScore.NO: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
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
        result = evaluator_agent(prompt, structured_output_model=RefusalRating)
        rating = cast(RefusalRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score == 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
