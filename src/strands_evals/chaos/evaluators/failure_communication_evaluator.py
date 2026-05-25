"""Failure Communication Evaluator.

Evaluates quality of agent's failure communication and user experience.
"""

from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.trace import EvaluationLevel
from ...evaluators.evaluator import Evaluator
from .prompt_templates.failure_communication_v0 import SYSTEM_PROMPT as SYSTEM_PROMPT_V0

_PROMPT_VERSIONS = {"v0": SYSTEM_PROMPT_V0}


class FailureCommunicationScore(str, Enum):
    """Categorical failure communication ratings."""

    FAILURE = "Failure"
    POOR = "Poor"
    ACCEPTABLE = "Acceptable"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class FailureCommunicationRating(BaseModel):
    """Structured output for failure communication evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: FailureCommunicationScore = Field(description="Categorical failure communication rating")


class FailureCommunicationEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates quality of agent's failure communication and user experience.

    Scores how well the agent communicated failures to the user on a 5-point scale:
    - Excellent (1.0): Clear communication, user understands situation and next steps
    - Good (0.75): Good communication with minor gaps
    - Acceptable (0.5): Basic communication, or no failure occurred (baseline)
    - Poor (0.25): Confusing messages, overly technical, misleading
    - Failure (0.0): No communication of failures that did occur

    Example::

        from strands_evals.chaos.evaluators import FailureCommunicationEvaluator

        evaluator = FailureCommunicationEvaluator()
        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=cases,
            evaluators=[evaluator],
        )
    """

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        FailureCommunicationScore.FAILURE: 0.0,
        FailureCommunicationScore.POOR: 0.25,
        FailureCommunicationScore.ACCEPTABLE: 0.5,
        FailureCommunicationScore.GOOD: 0.75,
        FailureCommunicationScore.EXCELLENT: 1.0,
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
        result = evaluator_agent(prompt, structured_output_model=FailureCommunicationRating)
        rating = cast(FailureCommunicationRating, result.structured_output)
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
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=FailureCommunicationRating)
        rating = cast(FailureCommunicationRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]
