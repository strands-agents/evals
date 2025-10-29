"""
Generic trace types for agent observability.

These types represent standard observability primitives for agents.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, field_serializer
from typing_extensions import Mapping, Sequence, TypeAlias, Union


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
    content: list[Union[TextContent, ToolResultContent]]


class AssistantMessage(BaseModel):
    role: Role = Role.ASSISTANT
    content: list[Union[TextContent, ToolCallContent]]


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
    messages: list[Union[UserMessage, AssistantMessage]]


class ToolExecutionSpan(BaseSpan):
    span_type: SpanType = SpanType.TOOL_EXECUTION
    tool_call: ToolCall
    tool_result: ToolResult


class AgentInvocationSpan(BaseSpan):
    span_type: SpanType = SpanType.AGENT_INVOCATION
    user_prompt: str
    agent_response: str
    available_tools: list[ToolConfig]


SpanUnion: TypeAlias = Union[InferenceSpan, ToolExecutionSpan, AgentInvocationSpan]


class Trace(BaseModel):
    spans: list[SpanUnion]
    trace_id: str
    session_id: str


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
    session_history: list[Union[UserMessage, list[ToolExecution], AssistantMessage]]


class ToolLevelInput(BaseEvaluationInput):
    """Input for tool-level evaluators"""

    available_tools: list[ToolConfig]
    tool_execution_details: ToolExecutionSpan
    session_history: list[Union[UserMessage, list[ToolExecution], AssistantMessage]]


class EvaluatorScore(BaseModel):
    explanation: str
    value: Union[int, float] | None = None
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
