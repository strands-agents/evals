import logging
from datetime import datetime, timezone

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


def _to_aware_utc(dt: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class TraceExtractor:
    """Extracts structured evaluation inputs from Session traces."""

    def __init__(self, evaluation_level: EvaluationLevel):
        self.evaluation_level = evaluation_level

    def extract(self, session: Session) -> list[TraceLevelInput] | list[ToolLevelInput] | SessionLevelInput:
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
        previous_turns: list[UserMessage | list[ToolExecution] | AssistantMessage] = []

        for trace in session.traces:
            agent_spans = self._find_agent_invocation_spans(trace)
            tool_spans = self._find_tool_execution_spans(trace)

            # Resolve which tools belong to which agent via ancestry
            tool_to_agent = self._resolve_tool_ownership(trace, agent_spans) if len(agent_spans) > 1 else None

            for span in agent_spans:
                # Skip spans with no evaluable content
                if not span.user_prompt and not span.agent_response:
                    continue

                try:
                    text_content = TextContent(text=span.user_prompt)
                    previous_turns.append(UserMessage(content=[text_content]))
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to create user message: {e}")
                    continue

                # Include only tool executions owned by this agent
                if tool_to_agent:
                    owned_tools = [ts for ts in tool_spans if tool_to_agent.get(id(ts)) is span]
                else:
                    owned_tools = tool_spans

                if owned_tools:
                    try:
                        tool_executions = [
                            ToolExecution(tool_call=ts.tool_call, tool_result=ts.tool_result) for ts in owned_tools
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
        """Extract tool-level inputs with session and tool context.

        Note: spans are not scoped by parent_span_id, so nested agent-as-tool traces
        may leak child-agent internals into the parent's history.
        """
        evaluator_inputs: list[ToolLevelInput] = []
        session_history: list[UserMessage | list[ToolExecution] | AssistantMessage] = []
        available_tools: list[ToolConfig] = []

        for trace in session.traces:
            agent_spans = self._find_agent_invocation_spans(trace)
            tool_spans = self._find_tool_execution_spans(trace)

            # Determine root agent for session history
            agent_span: AgentInvocationSpan | None = None
            if agent_spans:
                agent_span = self._find_root_agent(agent_spans)

                if agent_span.available_tools:
                    available_tools = agent_span.available_tools
                if agent_span.user_prompt:
                    session_history.append(UserMessage(content=[TextContent(text=agent_span.user_prompt)]))

            # Resolve per-agent tool scoping for multi-agent traces
            tool_to_agent = self._resolve_tool_ownership(trace, agent_spans) if len(agent_spans) > 1 else None

            tool_executions = [
                ToolExecution(tool_call=span.tool_call, tool_result=span.tool_result) for span in tool_spans
            ]
            tool_end_times = [
                max(_to_aware_utc(span.span_info.end_time), _to_aware_utc(span.span_info.start_time))
                for span in tool_spans
            ]

            for index, tool_span in enumerate(tool_spans):
                target_start = _to_aware_utc(tool_span.span_info.start_time)
                prior_executions = [
                    tool_executions[position]
                    for position in range(len(tool_spans))
                    if position != index
                    and tool_end_times[position] <= target_start
                    and (tool_end_times[position] < target_start or position < index)
                ]

                if tool_to_agent:
                    owning_agent = tool_to_agent.get(id(tool_span))
                    scoped_tools = owning_agent.available_tools if owning_agent and owning_agent.available_tools else []
                else:
                    scoped_tools = available_tools

                evaluator_inputs.append(
                    ToolLevelInput(
                        span_info=tool_span.span_info,
                        available_tools=scoped_tools,
                        tool_execution_details=tool_span,
                        session_history=list(session_history) + ([prior_executions] if prior_executions else []),
                    )
                )

            if tool_spans:
                session_history.append(tool_executions)

            if agent_span and agent_span.agent_response:
                session_history.append(AssistantMessage(content=[TextContent(text=agent_span.agent_response)]))

        return evaluator_inputs

    def _find_agent_invocation_spans(self, trace) -> list[AgentInvocationSpan]:
        """Find all AgentInvocationSpans in a trace."""
        return [span for span in trace.spans if isinstance(span, AgentInvocationSpan)]

    def _find_root_agent(self, agent_spans: list[AgentInvocationSpan]) -> AgentInvocationSpan:
        """Find the root agent span from a list of agent spans.

        Prefers the agent with no parent that has content (true root of the trace),
        then falls back to one with a user_prompt, then the last in the list.
        """
        return next(
            (s for s in agent_spans if s.span_info.parent_span_id is None and (s.user_prompt or s.agent_response)),
            next(
                (s for s in agent_spans if s.user_prompt),
                agent_spans[-1],
            ),
        )

    def _find_tool_execution_spans(self, trace) -> list[ToolExecutionSpan]:
        """Find all ToolExecutionSpans in a trace."""
        return [span for span in trace.spans if isinstance(span, ToolExecutionSpan)]

    def _resolve_tool_ownership(self, trace, agent_spans: list[AgentInvocationSpan]) -> dict[int, AgentInvocationSpan]:
        """Map each tool span to its owning AgentInvocationSpan via parent_span_id ancestry.

        Walks the parent chain from each ToolExecutionSpan until it finds an
        AgentInvocationSpan. Falls back to the root (outermost) agent when the
        parent chain is incomplete.
        """
        span_by_id: dict[str, object] = {}
        for span in trace.spans:
            if span.span_info.span_id:
                span_by_id[span.span_info.span_id] = span

        agent_by_id = {s.span_info.span_id: s for s in agent_spans if s.span_info.span_id}
        root_agent = self._find_root_agent(agent_spans)

        # Walk each tool span's parent chain to find its nearest owning AgentInvocationSpan, falling back to root.
        result: dict[int, AgentInvocationSpan] = {}
        for span in trace.spans:
            if not isinstance(span, ToolExecutionSpan):
                continue
            current_id = span.span_info.parent_span_id
            found: AgentInvocationSpan | None = None
            visited: set[str] = set()
            while current_id and current_id not in visited:
                visited.add(current_id)
                if current_id in agent_by_id:
                    found = agent_by_id[current_id]
                    break
                parent = span_by_id.get(current_id)
                if parent:
                    current_id = parent.span_info.parent_span_id
                else:
                    break
            result[id(span)] = found or root_agent

        return result

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
