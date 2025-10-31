import logging

from typing_extensions import Union

from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    Conversation,
    ConversationLevelInput,
    EvaluationLevel,
    Session,
    SpanInfo,
    TextContent,
    ToolConfig,
    ToolExecutionSpan,
    ToolLevelInput,
    TurnLevelInput,
    UserMessage,
)

logger = logging.getLogger(__name__)


class TraceExtractor:
    """Extracts structured evaluation inputs from Session traces."""

    def __init__(self, evaluation_level: EvaluationLevel):
        self.evaluation_level = evaluation_level

    def extract(self, session: Session) -> Union[list[TurnLevelInput], list[ToolLevelInput], ConversationLevelInput]:
        """Extract evaluation inputs based on configured level."""
        if not isinstance(session, Session):
            raise TypeError(f"Expected Session object, got {type(session).__name__}")

        if self.evaluation_level == EvaluationLevel.TURN_LEVEL:
            return self._extract_turn_level(session)
        elif self.evaluation_level == EvaluationLevel.TOOL_LEVEL:
            return self._extract_tool_level(session)
        elif self.evaluation_level == EvaluationLevel.CONVERSATION_LEVEL:
            return self._extract_conversation_level(session)
        else:
            raise ValueError(f"Unsupported evaluation level: {self.evaluation_level}")

    def _extract_turn_level(self, session: Session) -> list[TurnLevelInput]:
        """Extract turn-level inputs with conversation history up to each turn."""
        evaluation_inputs: list[TurnLevelInput] = []
        previous_turns: list[Union[UserMessage, AssistantMessage]] = []

        for trace in session.traces:
            for span in trace.spans:
                if not isinstance(span, AgentInvocationSpan):
                    continue

                try:
                    text_content = TextContent(text=span.user_prompt)
                    previous_turns.append(UserMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create user message: {e}")
                    continue

                turn_input = TurnLevelInput(
                    span_info=span.span_info,
                    agent_response=TextContent(text=span.agent_response),
                    conversation_history=list(previous_turns),
                )
                evaluation_inputs.append(turn_input)

                try:
                    text_content = TextContent(text=span.agent_response)
                    previous_turns.append(AssistantMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create assistant message: {e}")

        return evaluation_inputs

    def _extract_tool_level(self, session: Session) -> list[ToolLevelInput]:
        """Extract tool-level inputs with conversation and tool context."""
        evaluator_inputs: list[ToolLevelInput] = []
        conversation_history: list[Conversation] = []
        available_tools: list[ToolConfig] = []

        for trace in session.traces:
            user_prompt = None
            agent_response = None
            tool_calls: list[ToolExecutionSpan] = []

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

            if user_prompt and agent_response:
                conversation_history.append(
                    Conversation(
                        user_prompt=TextContent(text=user_prompt),
                        agent_response=TextContent(text=agent_response),
                        tool_execution_history=tool_calls if tool_calls else None,
                    )
                )

        return evaluator_inputs

    def _extract_conversation_level(self, session: Session) -> ConversationLevelInput:
        """Extract conversation-level input with full history."""
        conversation_history: list[Conversation] = []
        available_tools: list[ToolConfig] = []
        span_info: SpanInfo | None = None

        for trace in session.traces:
            tool_calls: list[ToolExecutionSpan] = []

            for span in trace.spans:
                if isinstance(span, ToolExecutionSpan):
                    tool_calls.append(span)

            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    if not span_info:
                        span_info = span.span_info
                    if span.available_tools and not available_tools:
                        available_tools = span.available_tools

                    conversation_history.append(
                        Conversation(
                            user_prompt=TextContent(text=span.user_prompt),
                            agent_response=TextContent(text=span.agent_response),
                            tool_execution_history=tool_calls if tool_calls else None,
                        )
                    )

        if not span_info:
            raise ValueError("No AgentInvocationSpan found in session")

        return ConversationLevelInput(
            span_info=span_info,
            conversation_history=conversation_history,
            available_tools=available_tools if available_tools else None,
        )
