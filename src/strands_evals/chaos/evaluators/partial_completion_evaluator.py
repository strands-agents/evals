"""Partial Completion Evaluator.

Evaluates what percentage of task objectives were achieved despite failures.
"""

from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.trace import EvaluationLevel
from ...evaluators.evaluator import Evaluator
from .prompt_templates.partial_completion_v0 import SYSTEM_PROMPT as SYSTEM_PROMPT_V0

_PROMPT_VERSIONS = {"v0": SYSTEM_PROMPT_V0}


class PartialCompletionRating(BaseModel):
    """Structured output for partial completion evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    completion_percentage: float = Field(
        description="Completion percentage from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )


class PartialCompletionEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates what percentage of task objectives were achieved despite failures.

    Returns a continuous score from 0.0 to 1.0 representing the fraction of the
    user's goal that was successfully completed. Passes if >= 0.5.

    Key principles:
    - Subtasks are derived from the user's goal, not mapped 1:1 to tools
    - LLM knowledge-based responses don't count as tool-dependent task completion
    - Legitimate fallback strategies (alternative tools) do count

    Example::

        from strands_evals.chaos.evaluators import PartialCompletionEvaluator

        evaluator = PartialCompletionEvaluator()
        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=cases,
            evaluators=[evaluator],
        )
    """

    evaluation_level = EvaluationLevel.TRACE_LEVEL

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
        result = evaluator_agent(prompt, structured_output_model=PartialCompletionRating)
        rating = cast(PartialCompletionRating, result.structured_output)

        return [
            EvaluationOutput(
                score=rating.completion_percentage,
                test_pass=rating.completion_percentage >= 0.5,
                reason=rating.reasoning,
                label=f"{rating.completion_percentage:.2f}",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_trace_level_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=PartialCompletionRating)
        rating = cast(PartialCompletionRating, result.structured_output)

        return [
            EvaluationOutput(
                score=rating.completion_percentage,
                test_pass=rating.completion_percentage >= 0.5,
                reason=rating.reasoning,
                label=f"{rating.completion_percentage:.2f}",
            )
        ]
