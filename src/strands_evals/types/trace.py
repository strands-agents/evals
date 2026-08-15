"""
Generic trace types for agent observability.

These types represent standard observability primitives for agents.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, field_serializer
from typing_extensions import Any, Mapping, Sequence, TypeAlias


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(str, Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class SpanType(str, Enum):
    INFERENCE = "inference"
    TOOL_EXECUTION = "execute_tool"
    AGENT_INVOCATION = "invoke_agent"


class EvaluationLevel(str, Enum):
    """Type of evaluation based on trace granularity."""

    SESSION_LEVEL = "Session"
    TRACE_LEVEL = "Trace"
    TOOL_LEVEL = "ToolCall"


class ToolCall(BaseModel):
    name: str
    arguments: dict
    tool_call_id: str | None = None


class ToolResult(BaseModel):
    content: str
    error: str | None = None
    tool_call_id: str | None = None


class ToolConfig(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class TextContent(BaseModel):
    content_type: ContentType = ContentType.TEXT
    text: str


class ToolCallContent(ToolCall):
    content_type: ContentType = ContentType.TOOL_USE


class ToolResultContent(ToolResult):
    content_type: ContentType = ContentType.TOOL_RESULT


class UserMessage(BaseModel):
    role: Role = Role.USER
    content: list[TextContent | ToolResultContent]


class AssistantMessage(BaseModel):
    role: Role = Role.ASSISTANT
    content: list[TextContent | ToolCallContent]


class SpanInfo(BaseModel):
    trace_id: str | None = None
    span_id: str | None = None
    session_id: str
    parent_span_id: str | None = None
    start_time: datetime
    end_time: datetime

    @field_serializer("start_time", "end_time")
    def serialize_datetime_utc(self, dt: datetime) -> str:
        """Serialize datetime fields in UTC timezone with ISO format."""
        # Convert to UTC if timezone-aware, otherwise assume it's already UTC
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(timezone.utc)
        else:
            utc_dt = dt.replace(tzinfo=timezone.utc)
        # Return ISO format string with 'Z' suffix for UTC
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class BaseSpan(BaseModel):
    span_info: SpanInfo
    metadata: dict | None = {}


class InferenceSpan(BaseSpan):
    span_type: SpanType = SpanType.INFERENCE
    messages: list[UserMessage | AssistantMessage]


class ToolExecutionSpan(BaseSpan):
    span_type: SpanType = SpanType.TOOL_EXECUTION
    tool_call: ToolCall
    tool_result: ToolResult
    agent_span_id: str | None = None


class AgentInvocationSpan(BaseSpan):
    span_type: SpanType = SpanType.AGENT_INVOCATION
    user_prompt: str
    agent_response: str
    available_tools: list[ToolConfig]
    system_prompt: str | None = None


SpanUnion: TypeAlias = InferenceSpan | ToolExecutionSpan | AgentInvocationSpan


def _to_aware_utc(dt: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _find_root_agent_span(agent_spans: Sequence[AgentInvocationSpan]) -> AgentInvocationSpan | None:
    """Search for a trace's root agent span.

    Prefers a parentless agent with content, then any parentless agent, then all
    agents; within the chosen tier the earliest `start_time` wins.
    """
    if not agent_spans:
        return None

    parentless = [s for s in agent_spans if s.span_info.parent_span_id is None]
    with_content = [s for s in parentless if s.user_prompt or s.agent_response]
    candidates = with_content or parentless or list(agent_spans)
    return min(candidates, key=lambda s: _to_aware_utc(s.span_info.start_time))


class Trace(BaseModel):
    """A single trace within a session.

    A Trace may contain multiple ``AgentInvocationSpan`` instances in multi-agent systems.
    """

    spans: list[SpanUnion]
    trace_id: str
    session_id: str

    def model_post_init(self, __context: Any) -> None:
        """Assign agent_span_id on tool spans if not already set.

        Note: mutates span objects in place; reusing a span across multiple
        Trace constructions will keep the first assignment.
        """
        tool_spans = [s for s in self.spans if isinstance(s, ToolExecutionSpan) and not s.agent_span_id]
        if not tool_spans:
            return

        agent_spans = [s for s in self.spans if isinstance(s, AgentInvocationSpan) and s.span_info.span_id]
        if not agent_spans:
            return

        spans_by_parent_id: dict[str, list[SpanUnion]] = {}
        for span in self.spans:
            parent_id = span.span_info.parent_span_id
            if parent_id:
                spans_by_parent_id.setdefault(parent_id, []).append(span)

        for agent_span in agent_spans:
            agent_id = agent_span.span_info.span_id
            if not agent_id:
                continue
            stack: list[SpanUnion] = list(spans_by_parent_id.get(agent_id, []))
            seen: set[int] = {id(agent_span)}
            while stack:
                span = stack.pop()
                if id(span) in seen:
                    continue
                seen.add(id(span))
                span_id = span.span_info.span_id
                # Nested agent span owns its subtree
                if isinstance(span, AgentInvocationSpan):
                    continue
                if isinstance(span, ToolExecutionSpan) and not span.agent_span_id:
                    span.agent_span_id = agent_id
                if span_id:
                    stack.extend(spans_by_parent_id.get(span_id, []))

        root = _find_root_agent_span(agent_spans)
        root_id = root.span_info.span_id if root else None
        for span in tool_spans:
            if not span.agent_span_id:
                span.agent_span_id = root_id


class Session(BaseModel):
    traces: list[Trace]
    session_id: str


class BaseEvaluationInput(BaseModel):
    """Base class for all evaluation inputs"""

    span_info: SpanInfo


class ToolExecution(BaseModel):
    tool_call: ToolCall
    tool_result: ToolResult


class Context(BaseModel):
    user_prompt: TextContent
    agent_response: TextContent
    tool_execution_history: list[ToolExecution] | None = None


class SessionLevelInput(BaseEvaluationInput):
    """Input for session-level evaluators"""

    session_history: list[Context]
    available_tools: list[ToolConfig] | None = None


class TraceLevelInput(BaseEvaluationInput):
    """Input for trace-level evaluators"""

    agent_response: TextContent
    session_history: list[UserMessage | list[ToolExecution] | AssistantMessage]


class ToolLevelInput(BaseEvaluationInput):
    """Input for tool-level evaluators.

    `session_history` never includes the call under evaluation. Within the same trace, only tool
    executions that completed before this tool started (`end_time <= start_time`) are included,
    with list position as tiebreaker for equal timestamps. Cross-trace history is unfiltered.
    """

    available_tools: list[ToolConfig]
    tool_execution_details: ToolExecutionSpan
    session_history: list[UserMessage | list[ToolExecution] | AssistantMessage]


class EvaluatorScore(BaseModel):
    explanation: str
    value: int | float | None = None
    error: str | None = None


class TokenUsage(BaseModel):
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class EvaluatorResult(BaseModel):
    span_info: SpanInfo
    evaluator_name: str
    score: EvaluatorScore
    token_usage: TokenUsage | None = None


class EvaluationResponse(BaseModel):
    evaluator_results: list[EvaluatorResult]


AttributeValue = Mapping[
    str, str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
]

Attributes = Mapping[str, AttributeValue] | None
