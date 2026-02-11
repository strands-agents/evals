from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, ToolExecution, TraceLevelInput
from .evaluator import Evaluator
from .prompt_templates.conciseness import get_template


class ConcisenessScore(str, Enum):
    """Categorical conciseness ratings."""

    NOT_CONCISE = "Not Concise"
    PARTIALLY_CONCISE = "Partially Concise"
    PERFECTLY_CONCISE = "Perfectly Concise"


class ConcisenessRating(BaseModel):
    """Structured output for conciseness evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: ConcisenessScore = Field(description="Categorical conciseness rating")


class ConcisenessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates how concise the assistant's response is."""

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        ConcisenessScore.NOT_CONCISE: 0.0,
        ConcisenessScore.PARTIALLY_CONCISE: 0.5,
        ConcisenessScore.PERFECTLY_CONCISE: 1.0,
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
        prompt = self._format_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=ConcisenessRating)
        return self._create_evaluation_output(result)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=ConcisenessRating)
        return self._create_evaluation_output(result)

    def _create_evaluation_output(self, result) -> list[EvaluationOutput]:
        rating = cast(ConcisenessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 0.5,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    def _get_last_turn(self, evaluation_case: EvaluationData[InputT, OutputT]) -> TraceLevelInput:
        """Extract the most recent turn from the conversation for evaluation."""
        parsed_inputs = self._parse_trajectory(evaluation_case)
        if not parsed_inputs:
            raise ValueError(
                "No turn-level inputs could be parsed from the trajectory. "
                "Ensure actual_trajectory is a Session with at least one AgentInvocationSpan."
            )
        return parsed_inputs[-1]

    def _extract_user_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Extract user prompt from last message in session history.

        Args:
            parsed_input: Trace-level input containing session history

        Returns:
            User prompt text, or empty string if not available
        """
        if not parsed_input.session_history:
            return ""

        last_msg = parsed_input.session_history[-1]
        if not isinstance(last_msg, list) and self._has_text_content(last_msg):
            return self._extract_text_content(last_msg)

        return ""

    def _format_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Format evaluation prompt from parsed trace data.

        Args:
            parsed_input: Trace-level input containing agent response and session history

        Returns:
            Formatted prompt string with conversation history and target turn
        """
        parts = []

        if parsed_input.session_history:
            history_lines = []
            for msg in parsed_input.session_history:
                if isinstance(msg, list) and msg and isinstance(msg[0], ToolExecution):
                    continue  # Skip tool execution lists
                if not isinstance(msg, list) and self._has_text_content(msg):
                    text = self._extract_text_content(msg)
                    history_lines.append(f"{msg.role.value.capitalize()}: {text}")
            history_str = "\n".join(history_lines)
            parts.append(f"# Previous turns:\n{history_str}")

        user_prompt = self._extract_user_prompt(parsed_input)
        parts.append(f"# Target turn to evaluate:\nUser: {user_prompt}\nAssistant: {parsed_input.agent_response.text}")

        return "\n\n".join(parts)
