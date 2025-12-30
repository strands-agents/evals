import logging

from typing_extensions import Union

from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    Context,
    EvaluationLevel,
    Session,
    SessionLevelInput,
    SpanInfo,
    TextContent,
    ToolConfig,
    ToolExecution,
    ToolExecutionSpan,
    ToolLevelInput,
    TraceLevelInput,
    UserMessage,
)

logger = logging.getLogger(__name__)


class TraceExtractor:
    """Extracts structured evaluation inputs from Session traces."""

    def __init__(self, evaluation_level: EvaluationLevel):
        self.evaluation_level = evaluation_level

    def extract(self, session: Session) -> Union[list[TraceLevelInput], list[ToolLevelInput], SessionLevelInput]:
        """Extract evaluation inputs based on configured level."""
        if not isinstance(session, Session):
            raise TypeError(f"Expected Session object, got {type(session).__name__}")

        if self.evaluation_level == EvaluationLevel.TRACE_LEVEL:
            return self._extract_trace_level(session)
        elif self.evaluation_level == EvaluationLevel.TOOL_LEVEL:
            return self._extract_tool_level(session)
        elif self.evaluation_level == EvaluationLevel.SESSION_LEVEL:
            return self._extract_session_level(session)
        else:
            raise ValueError(f"Unsupported evaluation level: {self.evaluation_level}")

    def _extract_trace_level(self, session: Session) -> list[TraceLevelInput]:
        """Extract trace-level inputs with session history up to each turn."""
        evaluation_inputs: list[TraceLevelInput] = []
        previous_turns: list[Union[UserMessage, list[ToolExecution], AssistantMessage]] = []

        for trace in session.traces:
            tool_spans = self._find_tool_execution_spans(trace)

            for span in trace.spans:
                if not isinstance(span, AgentInvocationSpan):
                    continue

                try:
                    text_content = TextContent(text=span.user_prompt)
                    previous_turns.append(UserMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create user message: {e}")
                    continue

                # Include tool executions in session history
                if tool_spans:
                    try:
                        tool_executions = [
                            ToolExecution(tool_call=ts.tool_call, tool_result=ts.tool_result)
                            for ts in tool_spans
                        ]
                        previous_turns.append(tool_executions)
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to create tool executions: {e}")

                trace_input = TraceLevelInput(
                    span_info=span.span_info,
                    agent_response=TextContent(text=span.agent_response),
                    session_history=list(previous_turns),
                )
                evaluation_inputs.append(trace_input)

                try:
                    text_content = TextContent(text=span.agent_response)
                    previous_turns.append(AssistantMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create assistant message: {e}")

        return evaluation_inputs

    def _extract_tool_level(self, session: Session) -> list[ToolLevelInput]:
        """Extract tool-level inputs with session and tool context."""
        evaluator_inputs: list[ToolLevelInput] = []
        session_history: list[Union[UserMessage, list[ToolExecution], AssistantMessage]] = []
        available_tools: list[ToolConfig] = []

        for trace in session.traces:
            agent_span = self._find_agent_invocation_span(trace)
            tool_spans = self._find_tool_execution_spans(trace)

            if agent_span and agent_span.available_tools:
                available_tools = agent_span.available_tools

            if agent_span and agent_span.user_prompt:
                session_history.append(UserMessage(content=[TextContent(text=agent_span.user_prompt)]))

            for tool_span in tool_spans:
                evaluator_inputs.append(
                    ToolLevelInput(
                        span_info=tool_span.span_info,
                        available_tools=available_tools,
                        tool_execution_details=tool_span,
                        session_history=list(session_history),
                    )
                )

            if tool_spans:
                tool_executions = [
                    ToolExecution(tool_call=span.tool_call, tool_result=span.tool_result) for span in tool_spans
                ]
                session_history.append(tool_executions)

            if agent_span and agent_span.agent_response:
                session_history.append(AssistantMessage(content=[TextContent(text=agent_span.agent_response)]))

        return evaluator_inputs

    def _find_agent_invocation_span(self, trace) -> AgentInvocationSpan | None:
        """Find the AgentInvocationSpan in a trace."""
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan):
                return span
        return None

    def _find_tool_execution_spans(self, trace) -> list[ToolExecutionSpan]:
        """Find all ToolExecutionSpans in a trace."""
        return [span for span in trace.spans if isinstance(span, ToolExecutionSpan)]

    def _extract_session_level(self, session: Session) -> SessionLevelInput:
        """Extract session-level input with full history."""
        session_history: list[Context] = []
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

                    tool_executions = (
                        [ToolExecution(tool_call=tc.tool_call, tool_result=tc.tool_result) for tc in tool_calls]
                        if tool_calls
                        else None
                    )

                    session_history.append(
                        Context(
                            user_prompt=TextContent(text=span.user_prompt),
                            agent_response=TextContent(text=span.agent_response),
                            tool_execution_history=tool_executions,
                        )
                    )

        if not span_info:
            raise ValueError("No AgentInvocationSpan found in session")

        return SessionLevelInput(
            span_info=span_info,
            session_history=session_history,
            available_tools=available_tools if available_tools else None,
        )
