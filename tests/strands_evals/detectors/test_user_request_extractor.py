"""Tests for user request extraction."""

from datetime import datetime

from strands_evals.detectors.user_request_extractor import extract_user_requests
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolConfig,
    ToolResultContent,
    Trace,
    UserMessage,
)


def _span_info(span_id: str = "span_1") -> SpanInfo:
    now = datetime.now()
    return SpanInfo(
        session_id="sess_1",
        span_id=span_id,
        trace_id="trace_1",
        start_time=now,
        end_time=now,
    )


def _make_session(traces: list[Trace]) -> Session:
    return Session(session_id="sess_1", traces=traces)


def test_extract_from_agent_invocation_span():
    """Should extract user_prompt from AgentInvocationSpan."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info(),
                        user_prompt="Book a flight to NYC",
                        agent_response="I'll help with that.",
                        available_tools=[ToolConfig(name="search")],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Book a flight to NYC"]


def test_extract_from_inference_span():
    """Should extract TextContent from UserMessage in InferenceSpan."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    InferenceSpan(
                        span_info=_span_info(),
                        messages=[
                            UserMessage(content=[TextContent(text="What is the weather?")]),
                            AssistantMessage(content=[TextContent(text="It's sunny.")]),
                        ],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["What is the weather?"]


def test_extract_deduplicates():
    """Duplicate messages should appear only once."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info("span_1"),
                        user_prompt="Hello",
                        agent_response="Hi",
                        available_tools=[],
                    ),
                    InferenceSpan(
                        span_info=_span_info("span_2"),
                        messages=[
                            UserMessage(content=[TextContent(text="Hello")]),
                        ],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Hello"]


def test_extract_preserves_order():
    """Messages should appear in trace order."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info("span_1"),
                        user_prompt="First question",
                        agent_response="Answer 1",
                        available_tools=[],
                    ),
                ],
            ),
            Trace(
                trace_id="trace_2",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info("span_2"),
                        user_prompt="Second question",
                        agent_response="Answer 2",
                        available_tools=[],
                    ),
                ],
            ),
        ]
    )
    result = extract_user_requests(session)
    assert result == ["First question", "Second question"]


def test_extract_skips_empty_and_whitespace():
    """Empty or whitespace-only messages should be excluded."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info("span_1"),
                        user_prompt="  ",
                        agent_response="Hi",
                        available_tools=[],
                    ),
                    InferenceSpan(
                        span_info=_span_info("span_2"),
                        messages=[
                            UserMessage(content=[TextContent(text="")]),
                            UserMessage(content=[TextContent(text="Real question")]),
                        ],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Real question"]


def test_extract_empty_session():
    """Empty session should return empty list."""
    session = _make_session([])
    result = extract_user_requests(session)
    assert result == []


def test_extract_ignores_tool_results_in_user_message():
    """ToolResultContent in UserMessage should be ignored (only TextContent extracted)."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    InferenceSpan(
                        span_info=_span_info(),
                        messages=[
                            UserMessage(
                                content=[
                                    ToolResultContent(content="tool output", tool_call_id="call_1"),
                                    TextContent(text="Now continue"),
                                ]
                            ),
                        ],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Now continue"]


def test_extract_multiple_text_blocks():
    """Multiple TextContent blocks in a single UserMessage should be joined."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    InferenceSpan(
                        span_info=_span_info(),
                        messages=[
                            UserMessage(
                                content=[
                                    TextContent(text="Part one"),
                                    TextContent(text="part two"),
                                ]
                            ),
                        ],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Part one part two"]


def test_extract_strips_whitespace():
    """Leading/trailing whitespace in prompts should be stripped."""
    session = _make_session(
        [
            Trace(
                trace_id="trace_1",
                session_id="sess_1",
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info(),
                        user_prompt="  Padded prompt  ",
                        agent_response="Ok",
                        available_tools=[],
                    ),
                ],
            )
        ]
    )
    result = extract_user_requests(session)
    assert result == ["Padded prompt"]
