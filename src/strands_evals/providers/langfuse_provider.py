"""Langfuse trace provider for retrieving agent traces from Langfuse."""

import json
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

from ..types.evaluation import TaskOutput
from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)
from .exceptions import (
    ProviderError,
    SessionNotFoundError,
    TraceNotFoundError,
    TraceProviderError,
)
from .trace_provider import SessionFilter, TraceProvider

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100


class LangfuseProvider(TraceProvider):
    """Retrieves agent trace data from Langfuse for evaluation.

    Fetches traces and observations via the Langfuse Python SDK,
    converts Langfuse observations to typed evals spans, and returns
    Session objects ready for the evaluation pipeline.
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        if Langfuse is None:
            raise ProviderError(
                "Langfuse SDK is not installed. Install it with: pip install 'strands-evals[langfuse]'"
            )

        resolved_public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        resolved_secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        resolved_host = host or os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

        if not resolved_public_key or not resolved_secret_key:
            raise ProviderError(
                "Langfuse credentials required. Provide public_key/secret_key or set "
                "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY environment variables."
            )

        self._client = Langfuse(
            public_key=resolved_public_key,
            secret_key=resolved_secret_key,
            host=resolved_host,
        )

    def get_evaluation_data(self, session_id: str) -> TaskOutput:
        """Fetch all traces for a session and return evaluation data."""
        try:
            all_traces = self._fetch_traces_for_session(session_id)
        except TraceProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Langfuse: failed to fetch traces for session '{session_id}': {e}") from e

        if not all_traces:
            raise SessionNotFoundError(f"Langfuse: no traces found for session_id='{session_id}'")

        session = self._build_session(session_id, all_traces)

        if not session.traces:
            raise SessionNotFoundError(
                f"Langfuse: traces found for session_id='{session_id}' but none contained convertible observations"
            )

        output = self._extract_output(session)

        return TaskOutput(output=output, trajectory=session)

    def list_sessions(self, session_filter: SessionFilter | None = None) -> Iterator[str]:
        """Yield session IDs from Langfuse, with optional time filtering."""
        try:
            page = 1
            count = 0
            limit = session_filter.limit if session_filter and session_filter.limit else None
            while True:
                kwargs: dict[str, Any] = {"page": page, "limit": _PAGE_SIZE}
                if session_filter:
                    if session_filter.start_time:
                        kwargs["from_timestamp"] = session_filter.start_time
                    if session_filter.end_time:
                        kwargs["to_timestamp"] = session_filter.end_time

                response = self._client.api.sessions.list(**kwargs)

                for s in response.data:
                    yield s.id
                    count += 1
                    if limit is not None and count >= limit:
                        return

                if page >= response.meta.total_pages:
                    break
                page += 1
        except TraceProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Langfuse: failed to list sessions: {e}") from e

    def get_evaluation_data_by_trace_id(self, trace_id: str) -> TaskOutput:
        """Fetch a single trace by ID and return evaluation data."""
        try:
            trace_detail = self._client.api.trace.get(trace_id)
        except Exception as e:
            raise TraceNotFoundError(f"Langfuse: trace not found for trace_id='{trace_id}': {e}") from e

        session_id = trace_detail.session_id or trace_id
        observations = trace_detail.observations or []

        spans = self._convert_observations(observations, session_id)
        trace = Trace(trace_id=trace_id, session_id=session_id, spans=spans)
        session = Session(session_id=session_id, traces=[trace] if trace.spans else [])
        output = self._extract_output(session)

        return TaskOutput(output=output, trajectory=session)

    # --- Internal: fetching ---

    def _fetch_all_pages(self, fetch_fn: Callable[..., Any], **kwargs: Any) -> list[Any]:
        """Fetch all pages from a paginated Langfuse API endpoint."""
        all_items: list = []
        page = 1
        while True:
            response = fetch_fn(page=page, limit=_PAGE_SIZE, **kwargs)
            all_items.extend(response.data)
            if page >= response.meta.total_pages:
                break
            page += 1
        return all_items

    def _fetch_traces_for_session(self, session_id: str) -> list:
        """Fetch all trace metadata for a session, handling pagination."""
        return self._fetch_all_pages(self._client.api.trace.list, session_id=session_id)

    def _fetch_observations(self, trace_id: str) -> list[Any]:
        """Fetch all observations for a trace, handling pagination."""
        return self._fetch_all_pages(self._client.api.observations.get_many, trace_id=trace_id)

    # --- Internal: building Session ---

    def _build_session(self, session_id: str, langfuse_traces: list[Any]) -> Session:
        """Convert Langfuse traces + observations into an evals Session."""
        traces = []
        for lf_trace in langfuse_traces:
            observations = self._fetch_observations(lf_trace.id)
            spans = self._convert_observations(observations, session_id)
            if spans:
                traces.append(Trace(trace_id=lf_trace.id, session_id=session_id, spans=spans))
        return Session(session_id=session_id, traces=traces)

    def _convert_observations(self, observations: list[Any], session_id: str) -> list[Any]:
        """Convert a list of Langfuse observations to typed evals spans."""
        spans = []
        for obs in observations:
            try:
                span = self._convert_observation(obs, session_id)
                if span is not None:
                    spans.append(span)
            except Exception as e:
                logger.warning("observation_id=<%s>, error=<%s> | failed to convert observation", obs.id, e)
        return spans

    def _convert_observation(self, obs: Any, session_id: str) -> Any:
        """Convert a single Langfuse observation to a typed span, or None to skip."""
        obs_type = obs.type

        if obs_type == "GENERATION":
            return self._convert_generation(obs, session_id)

        if obs_type != "SPAN":
            logger.debug("Skipping observation with type: %s", obs_type)
            return None

        obs_name = obs.name or ""
        if obs_name.startswith("execute_tool"):
            return self._convert_tool_execution(obs, session_id)
        if obs_name.startswith("invoke_agent"):
            return self._convert_agent_invocation(obs, session_id)

        logger.debug("Skipping SPAN with unrecognized name: %s", obs_name)
        return None

    def _create_span_info(self, obs: Any, session_id: str) -> SpanInfo:
        return SpanInfo(
            trace_id=obs.trace_id,
            span_id=obs.id,
            session_id=session_id,
            parent_span_id=obs.parent_observation_id,
            start_time=obs.start_time,
            end_time=obs.end_time,
        )

    # --- Internal: conversion methods ---

    def _convert_generation(self, obs: Any, session_id: str) -> InferenceSpan:
        """Convert a GENERATION observation to an InferenceSpan."""
        span_info = self._create_span_info(obs, session_id)
        messages = self._extract_messages_from_generation(obs)
        return InferenceSpan(span_info=span_info, messages=messages, metadata=obs.metadata or {})

    def _extract_messages_from_generation(self, obs: Any) -> list[UserMessage | AssistantMessage]:
        """Extract messages from a GENERATION observation's input/output."""
        messages: list[UserMessage | AssistantMessage] = []

        # Process input messages
        obs_input = obs.input
        if isinstance(obs_input, list):
            for msg in obs_input:
                if isinstance(msg, dict):
                    converted = self._convert_message(msg)
                    if converted:
                        messages.append(converted)

        # Process output message
        obs_output = obs.output
        if isinstance(obs_output, dict):
            converted = self._convert_message(obs_output)
            if converted:
                messages.append(converted)

        return messages

    def _convert_message(self, msg: dict) -> UserMessage | AssistantMessage | None:
        """Convert a Langfuse message dict to a UserMessage or AssistantMessage."""
        role = msg.get("role", "")
        content_data = msg.get("content", [])

        if role == "assistant":
            content = self._parse_assistant_content(content_data)
            return AssistantMessage(content=content) if content else None
        elif role == "user":
            content = self._parse_user_content(content_data)
            return UserMessage(content=content) if content else None
        else:
            # Tool messages come back as user messages with tool results
            if isinstance(content_data, list):
                tool_results = self._parse_tool_result_content(content_data)
                if tool_results:
                    return UserMessage(content=tool_results)
            return None

    def _parse_user_content(self, content_data: Any) -> list[TextContent | ToolResultContent]:
        """Parse user message content."""
        result: list[TextContent | ToolResultContent] = []
        if isinstance(content_data, list):
            for item in content_data:
                if isinstance(item, dict) and "text" in item:
                    result.append(TextContent(text=item["text"]))
        elif isinstance(content_data, str):
            result.append(TextContent(text=content_data))
        return result

    def _parse_assistant_content(self, content_data: Any) -> list[TextContent | ToolCallContent]:
        """Parse assistant message content."""
        result: list[TextContent | ToolCallContent] = []
        if isinstance(content_data, list):
            for item in content_data:
                if isinstance(item, dict):
                    if "text" in item:
                        result.append(TextContent(text=item["text"]))
                    elif "toolUse" in item:
                        tu = item["toolUse"]
                        result.append(ToolCallContent(
                            name=tu["name"],
                            arguments=tu.get("input", {}),
                            tool_call_id=tu.get("toolUseId"),
                        ))
        elif isinstance(content_data, str):
            result.append(TextContent(text=content_data))
        return result

    def _parse_tool_result_content(self, content_data: list) -> list[TextContent | ToolResultContent]:
        """Parse tool result content from a message."""
        result: list[TextContent | ToolResultContent] = []
        for item in content_data:
            if isinstance(item, dict) and "toolResult" in item:
                tr = item["toolResult"]
                text = ""
                if "content" in tr and tr["content"]:
                    c = tr["content"]
                    text = c[0].get("text", "") if isinstance(c, list) else str(c)
                result.append(ToolResultContent(
                    content=text,
                    error=tr.get("error"),
                    tool_call_id=tr.get("toolUseId"),
                ))
        return result

    def _convert_tool_execution(self, obs: Any, session_id: str) -> ToolExecutionSpan:
        """Convert an execute_tool SPAN observation to a ToolExecutionSpan."""
        span_info = self._create_span_info(obs, session_id)
        obs_input = obs.input or {}

        if isinstance(obs_input, dict):
            tool_name = obs_input.get("name", "")
            tool_arguments = obs_input.get("arguments", {})
            tool_call_id = obs_input.get("toolUseId")
        else:
            tool_name = ""
            tool_arguments = {}
            tool_call_id = None

        result_content, result_error = self._parse_tool_result(obs.output)
        tool_call = ToolCall(name=tool_name, arguments=tool_arguments, tool_call_id=tool_call_id)
        tool_result = ToolResult(content=result_content, error=result_error, tool_call_id=tool_call_id)

        return ToolExecutionSpan(
            span_info=span_info, tool_call=tool_call, tool_result=tool_result, metadata=obs.metadata or {}
        )

    def _parse_tool_result(self, obs_output: Any) -> tuple[str, str | None]:
        """Parse tool execution output into (content, error)."""
        if isinstance(obs_output, str):
            return obs_output, None

        if isinstance(obs_output, dict):
            content = obs_output.get("result", str(obs_output))
            status = obs_output.get("status", "")
            error = None if status == "success" else (str(status) if status else None)
            return content, error

        content = str(obs_output) if obs_output is not None else ""
        return content, None

    def _convert_agent_invocation(self, obs: Any, session_id: str) -> AgentInvocationSpan:
        """Convert an invoke_agent SPAN observation to an AgentInvocationSpan."""
        span_info = self._create_span_info(obs, session_id)
        obs_input = obs.input
        obs_output = obs.output

        # Extract user prompt from input
        user_prompt = self._extract_user_prompt(obs_input)

        # Extract agent response from output
        agent_response = self._extract_agent_response(obs_output)

        # Extract available tools from metadata
        available_tools = self._extract_available_tools(obs.metadata)

        return AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=available_tools,
            metadata=obs.metadata or {},
        )

    def _extract_user_prompt(self, obs_input: Any) -> str:
        """Extract user prompt from observation input (handles string or list formats)."""
        if isinstance(obs_input, str):
            return obs_input
        if isinstance(obs_input, list):
            for item in obs_input:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        if isinstance(obs_input, dict) and "text" in obs_input:
            return obs_input["text"]
        return str(obs_input) if obs_input else ""

    def _extract_agent_response(self, obs_output: Any) -> str:
        """Extract agent response from observation output (handles string or dict formats)."""
        if isinstance(obs_output, str):
            return obs_output
        if isinstance(obs_output, dict):
            if "text" in obs_output:
                return obs_output["text"]
            if "content" in obs_output:
                content = obs_output["content"]
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            return item["text"]
                elif isinstance(content, str):
                    return content
        return str(obs_output) if obs_output else ""

    def _extract_available_tools(self, metadata: Any) -> list[ToolConfig]:
        """Extract available tools from observation metadata."""
        if not metadata or not isinstance(metadata, dict):
            return []
        tools_data = metadata.get("tools")
        if not tools_data:
            return []
        try:
            if isinstance(tools_data, str):
                tools_list = json.loads(tools_data)
            else:
                tools_list = tools_data
            return [ToolConfig(name=name) for name in tools_list if isinstance(name, str)]
        except (json.JSONDecodeError, TypeError):
            return []

    def _extract_output(self, session: Session) -> str:
        """Extract the final agent response from the session for TaskOutput.output."""
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan):
                    return span.agent_response
        return ""
