from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, TraceLevelInput
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
