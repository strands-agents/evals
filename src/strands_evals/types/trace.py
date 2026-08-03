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
    owning_agent_span_id: str | None = None


class AgentInvocationSpan(BaseSpan):
    span_type: SpanType = SpanType.AGENT_INVOCATION
    user_prompt: str
    agent_response: str
    available_tools: list[ToolConfig]
    system_prompt: str | None = None


SpanUnion: TypeAlias = InferenceSpan | ToolExecutionSpan | AgentInvocationSpan


class Trace(BaseModel):
    """A single trace within a session.

    A Trace may contain multiple ``AgentInvocationSpan`` instances in multi-agent systems.
    """

    spans: list[SpanUnion]
    trace_id: str
    session_id: str

    def model_post_init(self, __context: Any) -> None:
        """Assign owning_agent_span_id on tool spans if not already set.

        Note: mutates span objects in place; reusing a span across multiple
        Trace constructions will keep the first assignment.
        """
        tool_spans = [s for s in self.spans if isinstance(s, ToolExecutionSpan) and not s.owning_agent_span_id]
        if not tool_spans:
            return

        agent_by_id: dict[str, AgentInvocationSpan] = {}
        span_by_id: dict[str, BaseSpan] = {}
        for span in self.spans:
            sid = span.span_info.span_id
            if sid:
                span_by_id[sid] = span
                if isinstance(span, AgentInvocationSpan):
                    agent_by_id[sid] = span

        if not agent_by_id:
            return

        # Root agent: prefer parentless with content, then any parentless, then first
        root_agent_id: str | None = None
        for sid, agent in agent_by_id.items():
            if agent.span_info.parent_span_id is None and (agent.user_prompt or agent.agent_response):
                root_agent_id = sid
                break
        if root_agent_id is None:
            for sid, agent in agent_by_id.items():
                if agent.span_info.parent_span_id is None:
                    root_agent_id = sid
                    break
        if root_agent_id is None:
            root_agent_id = next(iter(agent_by_id))

        owner_cache: dict[str, str | None] = {}
        for span in tool_spans:
            current_id = span.span_info.parent_span_id
            visited: list[str] = []
            seen: set[str] = set()
            found: str | None = None
            while current_id and current_id not in owner_cache and current_id not in seen:
                if current_id in agent_by_id:
                    found = current_id
                    break
                seen.add(current_id)
                visited.append(current_id)
                parent = span_by_id.get(current_id)
                current_id = parent.span_info.parent_span_id if parent else None
            else:
                if current_id and current_id in owner_cache:
                    found = owner_cache[current_id]
            for vid in visited:
                owner_cache[vid] = found
            span.owning_agent_span_id = found or root_agent_id


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
