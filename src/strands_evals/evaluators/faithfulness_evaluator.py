from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, TraceLevelInput
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
        prompt = self._format_prompt(parsed_input)
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
        prompt = self._format_prompt(parsed_input)
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

    def _get_last_turn(self, evaluation_case: EvaluationData[InputT, OutputT]) -> TraceLevelInput:
        """Extract the most recent turn from the conversation for evaluation."""
        parsed_inputs = self._parse_trajectory(evaluation_case)
        if not parsed_inputs:
            raise ValueError(
                "No turn-level inputs could be parsed from the trajectory. "
                "Ensure actual_trajectory is a Session with at least one AgentInvocationSpan."
            )
        return parsed_inputs[-1]

    def _format_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Format evaluation prompt from parsed turn data."""
        parts = []

        if parsed_input.session_history:
            history_lines = []
            for msg in parsed_input.session_history:
                if isinstance(msg, list):
                    # Handle tool execution lists
                    for tool_exec in msg:
                        history_lines.append(f"Action: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                        history_lines.append(f"Tool: {tool_exec.tool_result.content}")
                else:
                    text = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
                    history_lines.append(f"{msg.role.value.capitalize()}: {text}")
            history_str = "\n".join(history_lines)
            parts.append(f"# Conversation History:\n{history_str}")

        parts.append(f"# Assistant's Response:\n{parsed_input.agent_response.text}")

        return "\n\n".join(parts)
