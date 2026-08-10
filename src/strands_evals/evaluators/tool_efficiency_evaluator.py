from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, SessionLevelInput
from .evaluator import Evaluator
from .prompt_templates.tool_efficiency import get_template


class ToolCallCategory(str, Enum):
    """Classification categories for individual tool calls."""

    NECESSARY = "necessary"
    REDUNDANT = "redundant"
    ERRORED = "errored"
    UNNECESSARY = "unnecessary"


class ToolCallClassification(BaseModel):
    """Classification result for a single tool call."""

    tool_name: str
    call_index: int = Field(description="0-based position in the trajectory")
    category: ToolCallCategory
    reasoning: str = Field(description="One sentence explaining the classification")


class ToolEfficiencyRating(BaseModel):
    """Structured output for the tool efficiency evaluation."""

    classifications: list[ToolCallClassification]
    necessary_count: int
    total_count: int
    reasoning: str = Field(description="Overall assessment of tool usage efficiency")


class ToolEfficiencyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether all tool calls in a trajectory were necessary.

    Operates at SESSION_LEVEL. Reads the full trajectory and asks an LLM judge
    to classify each tool call as NECESSARY, REDUNDANT, ERRORED, or UNNECESSARY.

    Score is calculated as necessary_count / total_count (1.0 if no tool calls).
    The per-call breakdown is stored in EvaluationOutput.label as a JSON string.
    """

    evaluation_level = EvaluationLevel.SESSION_LEVEL

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
        system_prompt: str | None = None,
        max_tool_result_length: int = 2000,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        self.max_tool_result_length = max_tool_result_length

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        session_input: SessionLevelInput = self._parse_trajectory(evaluation_case)
        prompt = self._format_prompt(session_input)

        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=ToolEfficiencyRating)
        rating = cast(ToolEfficiencyRating, result.structured_output)

        score = rating.necessary_count / rating.total_count if rating.total_count > 0 else 1.0

        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= 0.5,
                reason=rating.reasoning,
                label=rating.model_dump_json(),
            )
        ]

    def _format_prompt(self, session_input: SessionLevelInput) -> str:
        """Format evaluation prompt from session-level input."""
        parts = []

        if session_input.available_tools:
            parts.append(f"# Available tools\n{self._format_tools(session_input.available_tools)}")

        if session_input.session_history:
            parts.append(f"# Conversation record\n{self._format_session_history_with_truncation(session_input)}")

        return "\n\n".join(parts)

    def _format_session_history_with_truncation(self, session_input: SessionLevelInput) -> str:
        """Format session history, truncating long tool results."""
        lines = []
        for ctx in session_input.session_history:
            lines.append(f"User: {ctx.user_prompt.text}")
            if ctx.tool_execution_history:
                for tool_exec in ctx.tool_execution_history:
                    lines.append(f"Action: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                    result_content = tool_exec.tool_result.content
                    if len(result_content) > self.max_tool_result_length:
                        result_content = result_content[: self.max_tool_result_length] + "... [truncated]"
                    if tool_exec.tool_result.error:
                        lines.append(f"Tool Error: {tool_exec.tool_result.error}")
                    else:
                        lines.append(f"Tool: {result_content}")
            lines.append(f"Assistant: {ctx.agent_response.text}")
        return "\n".join(lines)
