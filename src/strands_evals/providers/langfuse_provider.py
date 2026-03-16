"""Langfuse trace provider for retrieving agent traces from Langfuse."""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

from httpx import ReadTimeout
from langfuse import Langfuse
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential

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
    TraceProviderError,
)
from .trace_provider import TraceProvider

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_DEFAULT_TIMEOUT = 120
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2


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
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        """Initialize the Langfuse provider.

        Credentials can be passed directly or read from environment variables
        (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).

        Example::

            from strands_evals.providers import LangfuseProvider

            # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env
            provider = LangfuseProvider()

            # Or pass credentials explicitly
            provider = LangfuseProvider(
                public_key="pk-...",
                secret_key="sk-...",
            )

        Args:
            public_key: Langfuse public API key. Falls back to LANGFUSE_PUBLIC_KEY env var.
            secret_key: Langfuse secret API key. Falls back to LANGFUSE_SECRET_KEY env var.
            host: Langfuse API host URL. Falls back to LANGFUSE_HOST env var,
                defaulting to https://us.cloud.langfuse.com.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If no public key or secret key can be resolved.
        """
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
        self._request_options = {"timeout_in_seconds": timeout}

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

    # --- Internal: fetching ---

    def _fetch_all_pages(self, fetch_fn: Callable[..., Any], **kwargs: Any) -> list[Any]:
        """Fetch all pages from a paginated Langfuse API endpoint."""
        all_items: list = []
        page = 1
        while True:
            response = self._call_with_retry(fetch_fn, page=page, limit=_PAGE_SIZE, **kwargs)
            all_items.extend(response.data)
            if page >= response.meta.total_pages:
                break
            page += 1
        return all_items

    def _call_with_retry(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Call a Langfuse API function with timeout and retry on ReadTimeout."""
        kwargs.setdefault("request_options", self._request_options)
        retrier = Retrying(
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=_RETRY_BACKOFF_BASE),
            retry=retry_if_exception_type(ReadTimeout),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        return retrier(fn, **kwargs)

    def _fetch_traces_for_session(self, session_id: str) -> list:
        """Fetch all trace metadata for a session, handling pagination."""
        return self._fetch_all_pages(self._client.api.trace.list, session_id=session_id)

    def _fetch_observations(self, trace_id: str) -> list[Any]:
        """Fetch all observations for a trace, handling pagination."""
        return self._fetch_all_pages(self._client.api.observations.get_many, trace_id=trace_id)

    # --- Internal: building Session ---

    def _build_session(self, session_id: str, langfuse_traces: list[Any]) -> Session:
        """Fetch observations for each Langfuse trace and assemble a Session.

        For each trace, fetches all observations via the Langfuse API,
        converts them to typed spans, and groups them into Trace objects.
        Traces with no convertible observations are excluded.
        """
        traces = []
        for lf_trace in langfuse_traces:
            observations = self._fetch_observations(lf_trace.id)
            spans = self._convert_observations(observations, session_id)
            if spans:
                traces.append(Trace(trace_id=lf_trace.id, session_id=session_id, spans=spans))
        return Session(session_id=session_id, traces=traces)

    def _convert_observations(self, observations: list[Any], session_id: str) -> list[Any]:
        """Convert Langfuse observations to typed spans, skipping unconvertible ones.

        Each observation is routed through _convert_observation based on its type
        and name. Observations that fail conversion are logged and skipped.
        """
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
        """Route a single Langfuse observation to the appropriate span converter.

        Langfuse normalizes traces from ALL frameworks into its own Observation
        format. The `obs.type` field is universal across frameworks:

        - GENERATION — LLM call (LangChain, Strands, LlamaIndex, etc.)
        - TOOL — Tool invocation (LangChain sends these)
        - CHAIN — Orchestration/agent (root chain = agent invocation)
        - SPAN — Strands-specific spans (fallback by ``obs.name``)

        Routing:
            obs.type == "GENERATION"                        → InferenceSpan
            obs.type == "TOOL"                              → ToolExecutionSpan
            obs.type == "CHAIN" and no parent               → AgentInvocationSpan
            obs.type == "AGENT" and no parent               → AgentInvocationSpan (Langfuse v4+)
            obs.type == "SPAN", name starts "execute_tool"  → ToolExecutionSpan
            obs.type == "SPAN", name starts "invoke_agent"  → AgentInvocationSpan
            Otherwise                                       → None (skipped)
        """
        obs_type = obs.type

        if obs_type == "GENERATION":
            return self._convert_generation(obs, session_id)

        if obs_type == "TOOL":
            return self._convert_tool_execution(obs, session_id)

        if obs_type in ("CHAIN", "AGENT") and obs.parent_observation_id is None:
            return self._convert_agent_invocation(obs, session_id)

        # Strands-specific fallback for SPAN type
        if obs_type == "SPAN":
            obs_name = obs.name or ""
            if obs_name.startswith("execute_tool"):
                return self._convert_tool_execution(obs, session_id)
            if obs_name.startswith("invoke_agent"):
                return self._convert_agent_invocation(obs, session_id)

        logger.debug("Skipping observation: type=%s, name=%s", obs_type, obs.name)
        return None

    def _create_span_info(self, obs: Any, session_id: str) -> SpanInfo:
        """Map Langfuse observation metadata to a SpanInfo.

        Langfuse observation fields used:
            obs.trace_id: str                — Langfuse trace ID
            obs.id: str                      — Langfuse observation ID (becomes span_id)
            obs.parent_observation_id: str   — parent observation ID (or None)
            obs.start_time: datetime
            obs.end_time: datetime
        """
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
        """Convert a Langfuse GENERATION observation to an InferenceSpan.

        Langfuse observation (obs.type == "GENERATION"):
            obs.input: list[dict] — conversation messages, each with "role" and "content" keys
                [{"role": "user", "content": [{"text": "..."}]},
                 {"role": "assistant", "content": [{"text": "..."}, {"toolUse": {...}}]},
                 {"role": "tool", "content": [{"toolResult": {...}}]}]
            obs.output: dict — single assistant response message
                {"role": "assistant", "content": [{"text": "..."}, {"toolUse": {...}}]}
            obs.metadata: dict | None

        Returns:
            InferenceSpan with messages list containing UserMessage and AssistantMessage objects.
        """
        span_info = self._create_span_info(obs, session_id)
        messages = self._extract_messages_from_generation(obs)
        return InferenceSpan(span_info=span_info, messages=messages, metadata=obs.metadata or {})

    def _extract_messages_from_generation(self, obs: Any) -> list[UserMessage | AssistantMessage]:
        """Extract typed messages from a GENERATION observation's input and output.

        Reads obs.input (list of message dicts) and obs.output (single message dict),
        converting each to UserMessage or AssistantMessage via _convert_message.

        Langfuse format:
            obs.input:  [{"role": "user"|"assistant"|"tool", "content": [...]}, ...]
            obs.output: {"role": "assistant", "content": [...]}

        Messages with unrecognized roles or empty content are dropped.
        """
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
        """Convert a single Langfuse message dict to a typed message.

        Input format (msg):
            {"role": "user",      "content": [{"text": "..."}]}
            {"role": "assistant", "content": [{"text": "..."}, {"toolUse": {...}}]}
            {"role": "tool",      "content": [{"toolResult": {"toolUseId": "...", "status": "...", "content": [...]}}]}

        Mapping:
            role == "assistant" → AssistantMessage (text + tool calls)
            role == "user"      → UserMessage (text content)
            role == "tool"      → UserMessage (tool result content)
            other/empty         → None
        """
        role = msg.get("role", "")
        content_data = msg.get("content", [])

        if role == "assistant":
            assistant_content = self._parse_assistant_content(content_data)
            # LangChain format: tool_calls as a separate field
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        assistant_content.append(
                            ToolCallContent(
                                name=tc["name"],
                                arguments=tc.get("args") or tc.get("input") or {},
                                tool_call_id=tc.get("id"),
                            )
                        )
            return AssistantMessage(content=assistant_content) if assistant_content else None
        elif role == "user":
            user_content = self._parse_user_content(content_data)
            return UserMessage(content=user_content) if user_content else None
        else:
            # Tool messages come back as user messages with tool results
            if isinstance(content_data, list):
                tool_content = self._parse_tool_result_content(content_data)
                if tool_content:
                    return UserMessage(content=tool_content)
            return None

    def _parse_user_content(self, content_data: Any) -> list[TextContent | ToolResultContent]:
        """Parse user message content into typed content blocks.

        Input formats:
            list[dict]: [{"text": "user question"}]     → [TextContent(text="user question")]
            str:        "user question"                  → [TextContent(text="user question")]
        """
        result: list[TextContent | ToolResultContent] = []
        if isinstance(content_data, list):
            for item in content_data:
                if isinstance(item, dict) and "text" in item:
                    result.append(TextContent(text=item["text"]))
        elif isinstance(content_data, str):
            result.append(TextContent(text=content_data))
        return result

    def _parse_assistant_content(self, content_data: Any) -> list[TextContent | ToolCallContent]:
        """Parse assistant message content into typed content blocks.

        Input format (list[dict]):
            [
                {"text": "I'll help with that."},
                {"toolUse": {"name": "shell", "input": {"command": "ls"}, "toolUseId": "tooluse_abc123"}},
                {"reasoningContent": {...}}
            ]

        Output:
            [TextContent(text="I'll help with that."),
             ToolCallContent(name="shell", arguments={"command": "ls"}, tool_call_id="tooluse_abc123")]

        Also accepts a plain string, converted to [TextContent(text=...)].
        Blocks with unrecognized keys (e.g. "reasoningContent") are skipped.
        """
        result: list[TextContent | ToolCallContent] = []
        if isinstance(content_data, list):
            for item in content_data:
                if isinstance(item, dict):
                    if "text" in item:
                        result.append(TextContent(text=item["text"]))
                    elif "toolUse" in item:
                        tu = item["toolUse"]
                        result.append(
                            ToolCallContent(
                                name=tu["name"],
                                arguments=tu.get("input", {}),
                                tool_call_id=tu.get("toolUseId"),
                            )
                        )
        elif isinstance(content_data, str):
            result.append(TextContent(text=content_data))
        return result

    def _parse_tool_result_content(self, content_data: list) -> list[TextContent | ToolResultContent]:
        """Parse tool result content blocks from a Langfuse "tool" role message.

        Input format (list[dict]):
            [{"toolResult": {
                "toolUseId": "tooluse_abc123",
                "status": "success" | "error",
                "content": [{"text": "result text"}]
            }}]

        Output:
            [ToolResultContent(content="result text", error=None, tool_call_id="tooluse_abc123")]

        Only the first text item in "content" is extracted. Items without a
        "toolResult" key are skipped.
        """
        result: list[TextContent | ToolResultContent] = []
        for item in content_data:
            if isinstance(item, dict) and "toolResult" in item:
                tr = item["toolResult"]
                text = ""
                if "content" in tr and tr["content"]:
                    c = tr["content"]
                    text = c[0].get("text", "") if isinstance(c, list) else str(c)
                result.append(
                    ToolResultContent(
                        content=text,
                        error=tr.get("error"),
                        tool_call_id=tr.get("toolUseId"),
                    )
                )
        return result

    def _convert_tool_execution(self, obs: Any, session_id: str) -> ToolExecutionSpan:
        """Convert a tool observation to a ToolExecutionSpan.

        Handles two formats:

        **Strands** (obs.type == "SPAN", name starts with "execute_tool"):
            obs.input: ``{"name": "calc", "arguments": {"x": "2+2"}, "toolUseId": "..."}``
            obs.output: ``"42"`` or ``{"result": "4", "status": "success"}``

        **LangChain / universal** (obs.type == "TOOL"):
            obs.name: tool name (e.g. ``"add_numbers"``)
            obs.input: tool arguments (dict or other)
            obs.output: tool result
        """
        span_info = self._create_span_info(obs, session_id)
        obs_input = obs.input or {}

        if isinstance(obs_input, dict) and "name" in obs_input:
            # Strands format: input carries name/arguments/toolUseId
            tool_name = obs_input.get("name", "")
            tool_arguments = obs_input.get("arguments", {})
            tool_call_id = obs_input.get("toolUseId")
        elif obs.type == "TOOL":
            # LangChain/universal: obs.name is the tool, obs.input is arguments
            tool_name = obs.name or ""
            if isinstance(obs_input, dict):
                tool_arguments = obs_input
            elif isinstance(obs_input, str):
                # Try parsing as JSON; LangChain may send stringified dicts
                try:
                    parsed = json.loads(obs_input)
                    tool_arguments = parsed if isinstance(parsed, dict) else {"input": obs_input}
                except (json.JSONDecodeError, ValueError):
                    tool_arguments = {"input": obs_input}
            elif obs_input:
                tool_arguments = {"input": str(obs_input)}
            else:
                tool_arguments = {}
            tool_call_id = None
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
        """Parse tool execution output into (content, error).

        Input formats:
            str:  ``"42"``                                            → ``("42", None)``
            dict: ``{"result": "4", "status": "success"}``            → ``("4", None)``
            dict: ``{"result": "...", "status": "error"}``             → ``("...", "error")``
            dict: ``{"content": "Weather...", "type": "tool", ...}``   → ``("Weather...", None)``
                  (LangChain ToolMessage format via Langfuse)
            None:                                                      → ``("", None)``
        """
        if isinstance(obs_output, str):
            return obs_output, None

        if isinstance(obs_output, dict):
            # Strands format: {"result": "...", "status": "success"|"error"}
            if "result" in obs_output:
                content = obs_output["result"]
                status = obs_output.get("status", "")
                error = None if status == "success" else (str(status) if status else None)
                return content, error
            # LangChain ToolMessage format: {"content": "...", "type": "tool", ...}
            if "content" in obs_output:
                content = obs_output["content"]
                if isinstance(content, str):
                    return content, None
                return str(content), None
            return str(obs_output), None

        content = str(obs_output) if obs_output is not None else ""
        return content, None

    def _convert_agent_invocation(self, obs: Any, session_id: str) -> AgentInvocationSpan:
        """Convert an agent observation to an AgentInvocationSpan.

        Handles two formats:

        **Strands** (obs.type == "SPAN", name starts with "invoke_agent"):
            obs.input: ``"Hello"`` | ``[{"text": "Hello"}]`` | ``{"text": "Hello"}``
            obs.output: ``"Hi!"`` | ``{"message": "Hi!", "finish_reason": "end_turn"}``

        **LangChain / universal** (obs.type == "CHAIN", root observation):
            obs.input: ``{"input": "question"}`` or ``{"messages": [...]}``
            obs.output: ``{"output": "answer"}`` or ``{"content": "answer"}``
        """
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

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from a message content field.

        Handles formats seen in Strands observations via OTEL→Langfuse:
            str:        ``"Hello"``                                  → ``"Hello"``
            str (JSON): ``'[{"text": "Hello"}]'``                    → ``"Hello"``
            list[dict]: ``[{"text": "Hello"}]``                      → ``"Hello"``
            None:                                                    → ``""``
        """
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                return content
            return self._first_text_from_list(parsed) or content
        if isinstance(content, list):
            return self._first_text_from_list(content) or ""
        return str(content) if content else ""

    def _first_text_from_list(self, items: Any) -> str | None:
        """Return the first "text" value from a list of dicts, or None."""
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        return None

    def _find_message_content_by_role(
        self, messages: list, roles: tuple[str, ...], *, reverse: bool = False
    ) -> str | None:
        """Find content of the first message matching any of the given roles.

        Searches ``messages`` (a list of dicts with ``type`` and ``content`` keys)
        for the first entry whose ``type`` is in *roles*.  Returns ``None`` when no
        match is found so callers can fall through to other extraction strategies.
        """
        if not isinstance(messages, list):
            return None
        items = reversed(messages) if reverse else iter(messages)
        for msg in items:
            if isinstance(msg, dict) and msg.get("type") in roles:
                return str(msg.get("content", ""))
        # Fallback: last message
        if messages and isinstance(messages[-1], dict):
            return str(messages[-1].get("content", ""))
        return None

    def _extract_user_prompt(self, obs_input: Any) -> str:
        """Extract user prompt string from observation input.

        Input formats:
            str:        ``"Hello"``                          → ``"Hello"``
            list[dict]: ``[{"text": "Hello"}]``              → ``"Hello"``
            list[dict]: ``[{"role": "user", "content": ...}]``
                                                             → ``"Hello"`` (Strands via OTEL)
            dict:       ``{"text": "Hello"}``                → ``"Hello"``
            dict:       ``{"input": "Hello"}``               → ``"Hello"`` (LangChain CHAIN)
            dict:       ``{"messages": [{"type": "human", "content": "Hello"}]}``
                                                             → ``"Hello"`` (LangChain messages)
            None:                                            → ``""``
        """
        if isinstance(obs_input, str):
            return obs_input
        if isinstance(obs_input, list):
            for item in obs_input:
                if not isinstance(item, dict):
                    continue
                if "text" in item:
                    return item["text"]
                # Strands message-list format via OTEL→Langfuse
                if item.get("role") in ("user", "human"):
                    return self._extract_text_from_content(item.get("content"))
        if isinstance(obs_input, dict):
            if "text" in obs_input:
                return obs_input["text"]
            if "input" in obs_input:
                return str(obs_input["input"])
            if "messages" in obs_input:
                result = self._find_message_content_by_role(obs_input["messages"], ("human", "user"))
                if result is not None:
                    return result
        return str(obs_input) if obs_input else ""

    def _extract_agent_response(self, obs_output: Any) -> str:
        """Extract agent response string from observation output.

        Input formats:
            str:  ``"Hi there!"``                                    → ``"Hi there!"``
            list[dict]: ``[{"role": "assistant", "content": ...}]``  → ``"Hi!"`` (Strands via OTEL)
            dict: ``{"text": "Hi there!"}``                          → ``"Hi there!"``
            dict: ``{"message": "Hi!", "finish_reason": "..."}``     → ``"Hi!"``
            dict: ``{"output": "Hi!"}``                              → ``"Hi!"`` (LangChain CHAIN)
            dict: ``{"content": [{"text": "Hi!"}]}``                 → ``"Hi!"``
            dict: ``{"content": "Hi!"}``                             → ``"Hi!"``
            dict: ``{"messages": [{"type": "ai", "content": "Hi!"}]}``
                                                                     → ``"Hi!"`` (LangGraph output)
            None:                                                    → ``""``
        """
        if isinstance(obs_output, str):
            return obs_output
        if isinstance(obs_output, list):
            # Strands message-list format via OTEL→Langfuse
            for item in reversed(obs_output):
                if isinstance(item, dict) and item.get("role") in ("assistant", "ai"):
                    return self._extract_text_from_content(item.get("content"))
        if isinstance(obs_output, dict):
            if "text" in obs_output:
                return obs_output["text"]
            if "message" in obs_output:
                return obs_output["message"]
            if "output" in obs_output:
                return str(obs_output["output"])
            if "content" in obs_output:
                content = obs_output["content"]
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            return item["text"]
                elif isinstance(content, str):
                    return content
            if "messages" in obs_output:
                result = self._find_message_content_by_role(obs_output["messages"], ("ai", "assistant"), reverse=True)
                if result is not None:
                    return result
        return str(obs_output) if obs_output else ""

    def _extract_available_tools(self, metadata: Any) -> list[ToolConfig]:
        """Extract available tool configurations from observation metadata.

        Input format:
            metadata: {"tools": ["shell", "get_pull_request", ...], ...}
            metadata: {"tools": '["shell", "get_pull_request"]', ...}  (JSON string)
            metadata: None or {}  → []

        Returns:
            [ToolConfig(name="shell"), ToolConfig(name="get_pull_request"), ...]
        """
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
        """Extract the final agent response from the session for TaskOutput.output.

        Returns the last AgentInvocationSpan.agent_response. May be empty when
        the agent ended on a tool_use rather than producing a text response.
        """
        for trace in reversed(session.traces):
            for span in reversed(trace.spans):
                if isinstance(span, AgentInvocationSpan):
                    return span.agent_response
        return ""
