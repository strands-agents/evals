from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, TextContent, ToolExecution, TraceLevelInput
from .evaluator import Evaluator
from .prompt_templates.coherence import get_template


class CoherenceScore(str, Enum):
    """Categorical coherence ratings."""

    NOT_AT_ALL = "Not At All"
    NOT_GENERALLY = "Not Generally"
    NEUTRAL_MIXED = "Neutral/Mixed"
    GENERALLY_YES = "Generally Yes"
    COMPLETELY_YES = "Completely Yes"


class CoherenceRating(BaseModel):
    """Structured output for coherence evaluation."""

    reasoning: str = Field(description="step by step reasoning to derive the final score, using no more than 250 words")
    score: CoherenceScore = Field(description="Categorical coherence rating")


class CoherenceEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates the logical cohesion of the assistant's response.

    This evaluator assesses whether the assistant's response maintains logical consistency,
    flows naturally, and presents ideas in a well-organized manner. It uses an LLM-as-judge
    approach to provide categorical ratings that are then normalized to numeric scores.

    Scores:
    - NOT_AT_ALL (0.0): Response is completely incoherent or contradictory
    - NOT_GENERALLY (0.25): Response has significant logical gaps or inconsistencies
    - NEUTRAL_MIXED (0.5): Response has both coherent and incoherent elements
    - GENERALLY_YES (0.75): Response is mostly coherent with minor reasoning issues
    - COMPLETELY_YES (1.0): Response is fully coherent and logically consistent
    """

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        CoherenceScore.NOT_AT_ALL: 0.0,
        CoherenceScore.NOT_GENERALLY: 0.25,
        CoherenceScore.NEUTRAL_MIXED: 0.5,
        CoherenceScore.GENERALLY_YES: 0.75,
        CoherenceScore.COMPLETELY_YES: 1.0,
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
        result = evaluator_agent(prompt, structured_output_model=CoherenceRating)
        return self._create_evaluation_output(result)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)
        prompt = self._format_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=CoherenceRating)
        return self._create_evaluation_output(result)

    def _create_evaluation_output(self, result) -> list[EvaluationOutput]:
        rating = cast(CoherenceRating, result.structured_output)
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
            first_content = last_msg.content[0]
            if isinstance(first_content, TextContent):
                return first_content.text

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
                    first_content = msg.content[0]
                    if isinstance(first_content, TextContent):
                        history_lines.append(f"{msg.role.value.capitalize()}: {first_content.text}")
            history_str = "\n".join(history_lines)
            parts.append(f"# Previous turns:\n{history_str}")

        user_prompt = self._extract_user_prompt(parsed_input)
        parts.append(f"# Target turn to evaluate:\nUser: {user_prompt}\nAssistant: {parsed_input.agent_response.text}")

        return "\n\n".join(parts)
