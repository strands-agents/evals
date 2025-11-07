import logging
from enum import Enum

from pydantic import BaseModel, Field
from strands import Agent
from typing_extensions import TypeVar

from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.trace import EvaluationLevel, ToolLevelInput
from .evaluator import Evaluator
from .prompt_templates.tool_selection_accuracy import get_template

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ToolSelectionScore(str, Enum):
    """Binary tool selection accuracy ratings."""

    YES = "Yes"
    NO = "No"


class ToolSelectionRating(BaseModel):
    """Structured output for tool selection accuracy evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: ToolSelectionScore = Field(description="Score should be one of 'Yes' or 'No'")


class ToolSelectionAccuracyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether tool calls are justified at specific points in the conversation."""

    evaluation_level = EvaluationLevel.TOOL_LEVEL

    _score_mapping = {
        ToolSelectionScore.YES: 1.0,
        ToolSelectionScore.NO: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: str | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        tool_inputs = self._parse_trajectory(evaluation_case)
        logging.info(f"Tool inputs: {tool_inputs}")
        results = []

        for tool_input in tool_inputs:
            prompt = self._format_prompt(tool_input)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            rating = evaluator_agent.structured_output(ToolSelectionRating, prompt)
            normalized_score = self._score_mapping[rating.score]
            result = EvaluationOutput(
                score=normalized_score, test_pass=normalized_score == 1.0, reason=rating.reasoning
            )
            results.append(result)

        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        tool_inputs = self._parse_trajectory(evaluation_case)
        results = []

        for tool_input in tool_inputs:
            prompt = self._format_prompt(tool_input)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            rating = await evaluator_agent.structured_output_async(ToolSelectionRating, prompt)
            normalized_score = self._score_mapping[rating.score]
            result = EvaluationOutput(
                score=normalized_score, test_pass=normalized_score == 1.0, reason=rating.reasoning
            )
            results.append(result)

        return results

    def _format_prompt(self, tool_input: ToolLevelInput) -> str:
        """Format evaluation prompt from tool-level input."""
        parts = []

        # Format available tools
        if tool_input.available_tools:
            parts.append(f"## Available tool-calls\n{self._format_tools(tool_input.available_tools)}")

        # Format previous conversation history
        if tool_input.session_history:
            history_lines = []
            for msg in tool_input.session_history:
                if isinstance(msg, list):
                    # Handle tool execution lists
                    for tool_exec in msg:
                        history_lines.append(f"Action: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                        history_lines.append(f"Tool: {tool_exec.tool_result.content}")
                else:
                    text = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
                    history_lines.append(f"{msg.role.value.capitalize()}: {text}")
            history_str = "\n".join(history_lines)
            parts.append(f"## Previous conversation history\n{history_str}")

        # Format target tool call to evaluate
        tool_details = tool_input.tool_execution_details
        tool_call_str = f"Action: {tool_details.tool_call.name}({tool_details.tool_call.arguments})"
        parts.append(f"## Target tool-call to evaluate\n{tool_call_str}")

        return "\n\n".join(parts)
