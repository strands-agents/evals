import inspect
import logging

from typing_extensions import Any, Generic, TypeVar, Union

from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    Conversation,
    EvaluationLevel,
    Session,
    TextContent,
    ToolConfig,
    ToolExecutionSpan,
    ToolLevelInput,
    TurnLevelInput,
    UserMessage,
)

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """

    # Optional: subclasses can set this to enable trace parsing
    evaluation_level: EvaluationLevel | None = None

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses, especially if you want to run evaluations asynchronously."
        )

    def _parse_turn_level(self, session: Session) -> list[TurnLevelInput]:
        """
        Parse trace for turn-level evaluation.

        Returns one TurnLevelInput per agent turn, with conversation history
        up to that point.
        """
        evaluation_inputs: list[TurnLevelInput] = []
        previous_turns: list[Union[UserMessage, AssistantMessage]] = []

        for trace in session.traces:
            for span in trace.spans:
                if not isinstance(span, AgentInvocationSpan):
                    continue

                # Add user message to conversation history
                try:
                    text_content = TextContent(text=span.user_prompt)
                    previous_turns.append(UserMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create user message: {e}")
                    continue

                # Create turn input with history up to this point
                turn_input = TurnLevelInput(
                    span_info=span.span_info,
                    agent_response=TextContent(text=span.agent_response),
                    conversation_history=list(previous_turns),
                )
                evaluation_inputs.append(turn_input)

                # Add agent response to conversation history for next turn
                try:
                    text_content = TextContent(text=span.agent_response)
                    previous_turns.append(AssistantMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create assistant message: {e}")

        return evaluation_inputs

    def _parse_tool_level(self, session: Session) -> list[ToolLevelInput]:
        """
        Parse trace for tool-level evaluation.

        Returns one ToolLevelInput per tool execution, with conversation
        history and available tools context.
        """
        

        evaluator_inputs: list[ToolLevelInput] = []
        conversation_history: list[Conversation] = []
        available_tools: list[ToolConfig] = []

        for trace in session.traces:
            user_prompt = None
            agent_response = None
            tool_calls: list[ToolExecutionSpan] = []

            # First pass: collect agent invocation info
            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    if span.available_tools:
                        available_tools = span.available_tools
                    if hasattr(span, "user_prompt") and span.user_prompt:
                        user_prompt = span.user_prompt
                    if hasattr(span, "agent_response") and span.agent_response:
                        agent_response = span.agent_response
                elif isinstance(span, ToolExecutionSpan):
                    tool_calls.append(span)

            # Second pass: create tool-level inputs
            for span in trace.spans:
                if isinstance(span, ToolExecutionSpan):
                    evaluator_inputs.append(
                        ToolLevelInput(
                            span_info=span.span_info,
                            available_tools=available_tools or [],
                            tool_execution_details=span,
                            conversation_history=list(conversation_history),
                        )
                    )

            # Update conversation history after processing all spans in trace
            if user_prompt and agent_response:
                conversation_history.append(
                    Conversation(
                        user_prompt=TextContent(text=user_prompt),
                        agent_response=TextContent(text=agent_response),
                        tool_execution_history=tool_calls if tool_calls else None,
                    )
                )

        return evaluator_inputs

    def _parse_trajectory(self, evaluation_case: EvaluationData[InputT, OutputT]) -> Any:
        """Parse Session trajectory based on evaluation level."""
        trajectory = evaluation_case.actual_trajectory

        if not isinstance(trajectory, Session):
            raise TypeError(
                f"Trace parsing requires actual_trajectory to be a Session object, " f"got {type(trajectory).__name__}."
            )

        if self.evaluation_level == EvaluationLevel.TURN_LEVEL:
            return self._parse_turn_level(trajectory)
        elif self.evaluation_level == EvaluationLevel.TOOL_LEVEL:
            return self._parse_tool_level(trajectory)
        else:
            raise ValueError(f"Unsupported evaluation level: {self.evaluation_level}")

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the name of the evaluator type.

        Returns:
            str: The name of the evaluator type.
        """
        return cls.__name__

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.

        Returns:
            dict: A dictionary containing the evaluator's information. Omit private attributes
            (attributes starting with '_') and attributes with default values.
        """
        _dict = {"evaluator_type": self.get_type_name()}

        # Get default values from __init__ signature
        sig = inspect.signature(self.__class__.__init__)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
        for k, v in self.__dict__.items():
            if not k.startswith("_") and (k not in defaults or v != defaults[k]):
                _dict[k] = v
        return _dict
