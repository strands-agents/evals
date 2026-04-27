"""Tests for root cause analyzer."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.detectors.root_cause_analyzer import (
    _collect_descendants,
    _is_context_exceeded,
    _parse_structured_result,
    _prune_session_to_failure_paths,
    _trace_to_roots,
    analyze_root_cause,
)
from strands_evals.types.detector import (
    FailureItem,
    FixRecommendation,
    RCAOutput,
    RCAStructuredOutput,
    RootCauseItem,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)


def _span_info(span_id: str = "span_1", parent_span_id: str | None = None) -> SpanInfo:
    now = datetime.now()
    return SpanInfo(
        session_id="sess_1",
        span_id=span_id,
        trace_id="trace_1",
        parent_span_id=parent_span_id,
        start_time=now,
        end_time=now,
    )


def _make_session(spans=None) -> Session:
    if spans is None:
        spans = [
            AgentInvocationSpan(
                span_info=_span_info("span_1"),
                user_prompt="Hello",
                agent_response="Hi there",
                available_tools=[ToolConfig(name="search")],
            ),
            InferenceSpan(
                span_info=_span_info("span_2", parent_span_id="span_1"),
                messages=[UserMessage(content=[TextContent(text="Hello")])],
            ),
            ToolExecutionSpan(
                span_info=_span_info("span_3", parent_span_id="span_1"),
                tool_call=ToolCall(name="search", arguments={"q": "test"}),
                tool_result=ToolResult(content="result"),
            ),
        ]
    return Session(
        session_id="sess_1",
        traces=[Trace(trace_id="trace_1", session_id="sess_1", spans=spans)],
    )


def _make_failures(span_ids: list[str]) -> list[FailureItem]:
    return [
        FailureItem(
            span_id=sid,
            category=["hallucination"],
            confidence=[0.9],
            evidence=["test evidence"],
        )
        for sid in span_ids
    ]


def _make_rca_structured_output(span_ids: list[str]) -> RCAStructuredOutput:
    return RCAStructuredOutput(
        root_causes=[
            RootCauseItem(
                failure_span_id=sid,
                location=sid,
                failure_causality="PRIMARY_FAILURE",
                failure_propagation_impact=["QUALITY_DEGRADATION"],
                failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
                completion_status="PARTIAL_SUCCESS",
                root_cause_explanation=f"Failure at {sid} due to test issue",
                fix_recommendation=FixRecommendation(
                    fix_type="SYSTEM_PROMPT_FIX",
                    recommendation="Add instructions to fix the issue",
                ),
            )
            for sid in span_ids
        ]
    )


# --- _parse_structured_result ---


def test_parse_structured_result_basic():
    output = _make_rca_structured_output(["span_1"])
    result = _parse_structured_result(output)
    assert len(result) == 1
    assert result[0].failure_span_id == "span_1"
    assert result[0].location == "span_1"
    assert result[0].causality == "PRIMARY_FAILURE"
    assert result[0].propagation_impact == ["QUALITY_DEGRADATION"]
    assert result[0].fix_type == "SYSTEM_PROMPT_FIX"
    assert result[0].fix_recommendation == "Add instructions to fix the issue"


def test_parse_structured_result_multiple():
    output = _make_rca_structured_output(["span_1", "span_2"])
    result = _parse_structured_result(output)
    assert len(result) == 2
    assert result[0].failure_span_id == "span_1"
    assert result[1].failure_span_id == "span_2"


def test_parse_structured_result_empty():
    output = RCAStructuredOutput(root_causes=[])
    result = _parse_structured_result(output)
    assert result == []


# --- _trace_to_roots ---


def test_trace_to_roots_single():
    parent_map = {"span_1": None}
    visited: set[str] = set()
    _trace_to_roots("span_1", parent_map, visited)
    assert visited == {"span_1"}


def test_trace_to_roots_chain():
    parent_map = {"span_3": "span_2", "span_2": "span_1", "span_1": None}
    visited: set[str] = set()
    _trace_to_roots("span_3", parent_map, visited)
    assert visited == {"span_1", "span_2", "span_3"}


def test_trace_to_roots_already_visited():
    parent_map = {"span_2": "span_1", "span_1": None}
    visited = {"span_1"}
    _trace_to_roots("span_2", parent_map, visited)
    assert visited == {"span_1", "span_2"}


# --- _collect_descendants ---


def test_collect_descendants_basic():
    children_map = {"span_1": ["span_2", "span_3"], "span_2": ["span_4"]}
    visited: set[str] = set()
    _collect_descendants("span_1", children_map, visited, max_count=10)
    assert visited == {"span_2", "span_3", "span_4"}


def test_collect_descendants_limit():
    children_map = {"span_1": ["span_2", "span_3", "span_4", "span_5"]}
    visited: set[str] = set()
    _collect_descendants("span_1", children_map, visited, max_count=2)
    assert len(visited) == 2


def test_collect_descendants_zero():
    children_map = {"span_1": ["span_2"]}
    visited: set[str] = set()
    _collect_descendants("span_1", children_map, visited, max_count=0)
    assert visited == set()


def test_collect_descendants_no_children():
    children_map: dict[str, list[str]] = {}
    visited: set[str] = set()
    _collect_descendants("span_1", children_map, visited, max_count=10)
    assert visited == set()


# --- _prune_session_to_failure_paths ---


def test_prune_keeps_failure_and_ancestors():
    spans = [
        AgentInvocationSpan(
            span_info=_span_info("root"),
            user_prompt="hello",
            agent_response="hi",
            available_tools=[],
        ),
        InferenceSpan(
            span_info=_span_info("mid", parent_span_id="root"),
            messages=[UserMessage(content=[TextContent(text="x")])],
        ),
        ToolExecutionSpan(
            span_info=_span_info("failure", parent_span_id="mid"),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="err"),
        ),
        InferenceSpan(
            span_info=_span_info("unrelated", parent_span_id="root"),
            messages=[UserMessage(content=[TextContent(text="y")])],
        ),
    ]
    session = _make_session(spans)
    pruned = _prune_session_to_failure_paths(session, ["failure"])

    kept_ids = {s.span_info.span_id for s in pruned.traces[0].spans}
    assert "root" in kept_ids
    assert "mid" in kept_ids
    assert "failure" in kept_ids
    assert "unrelated" not in kept_ids


def test_prune_empty_failures():
    session = _make_session()
    pruned = _prune_session_to_failure_paths(session, [])
    assert len(pruned.traces[0].spans) == len(session.traces[0].spans)


def test_prune_nonexistent_failure_span():
    session = _make_session()
    pruned = _prune_session_to_failure_paths(session, ["nonexistent"])
    assert len(pruned.traces[0].spans) == 0


def test_prune_does_not_mutate_original():
    session = _make_session()
    original_count = len(session.traces[0].spans)
    _prune_session_to_failure_paths(session, ["nonexistent"])
    assert len(session.traces[0].spans) == original_count


def test_prune_keeps_descendants():
    spans = [
        AgentInvocationSpan(
            span_info=_span_info("root"),
            user_prompt="hello",
            agent_response="hi",
            available_tools=[],
        ),
        InferenceSpan(
            span_info=_span_info("failure", parent_span_id="root"),
            messages=[UserMessage(content=[TextContent(text="x")])],
        ),
        ToolExecutionSpan(
            span_info=_span_info("child1", parent_span_id="failure"),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="ok"),
        ),
        ToolExecutionSpan(
            span_info=_span_info("child2", parent_span_id="failure"),
            tool_call=ToolCall(name="t", arguments={}),
            tool_result=ToolResult(content="ok"),
        ),
    ]
    session = _make_session(spans)
    pruned = _prune_session_to_failure_paths(session, ["failure"], max_descendants=10)

    kept_ids = {s.span_info.span_id for s in pruned.traces[0].spans}
    assert "child1" in kept_ids
    assert "child2" in kept_ids


# --- _is_context_exceeded ---


def test_is_context_exceeded_strands_exception():
    from strands.types.exceptions import ContextWindowOverflowException

    assert _is_context_exceeded(ContextWindowOverflowException("too big"))


def test_is_context_exceeded_string_match():
    assert _is_context_exceeded(Exception("The context window is exceeded"))
    assert _is_context_exceeded(Exception("Input too long for model"))
    assert _is_context_exceeded(Exception("max_tokens limit reached"))


def test_is_context_exceeded_unrelated():
    assert not _is_context_exceeded(Exception("Something else"))


# --- analyze_root_cause (with mocked Agent) ---


def test_analyze_root_cause_empty_failures():
    session = _make_session()
    output = analyze_root_cause(session, failures=[])
    assert isinstance(output, RCAOutput)
    assert output.root_causes == []


@patch("strands_evals.detectors.root_cause_analyzer.Agent")
def test_analyze_root_cause_direct(mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_result = MagicMock()
    mock_result.structured_output = _make_rca_structured_output(["span_2"])
    mock_agent.return_value = mock_result

    session = _make_session()
    failures = _make_failures(["span_2"])
    output = analyze_root_cause(session, failures)

    assert isinstance(output, RCAOutput)
    assert len(output.root_causes) == 1
    assert output.root_causes[0].failure_span_id == "span_2"
    assert output.root_causes[0].causality == "PRIMARY_FAILURE"


@patch("strands_evals.detectors.root_cause_analyzer.Agent")
@patch("strands_evals.detectors.root_cause_analyzer._resolve_model")
def test_analyze_root_cause_passes_model(mock_resolve, mock_agent_cls):
    mock_model = MagicMock()
    mock_resolve.return_value = mock_model
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_result = MagicMock()
    mock_result.structured_output = _make_rca_structured_output(["span_2"])
    mock_agent.return_value = mock_result

    session = _make_session()
    failures = _make_failures(["span_2"])
    analyze_root_cause(session, failures, model="us.anthropic.claude-sonnet-4-20250514-v1:0")

    mock_resolve.assert_called_once_with("us.anthropic.claude-sonnet-4-20250514-v1:0")
    mock_agent_cls.assert_called_once()
    assert mock_agent_cls.call_args.kwargs["model"] is mock_model


@patch("strands_evals.detectors.root_cause_analyzer.Agent")
def test_analyze_root_cause_context_overflow_falls_to_pruning(mock_agent_cls):
    from strands.types.exceptions import ContextWindowOverflowException

    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    pruned_result = MagicMock()
    pruned_result.structured_output = _make_rca_structured_output(["span_2"])
    mock_agent.side_effect = [
        ContextWindowOverflowException("too big"),
        pruned_result,
    ]

    session = _make_session()
    failures = _make_failures(["span_2"])
    output = analyze_root_cause(session, failures)

    assert isinstance(output, RCAOutput)
    assert len(output.root_causes) == 1


@patch("strands_evals.detectors.root_cause_analyzer.Agent")
def test_analyze_root_cause_non_context_error_raises(mock_agent_cls):
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_agent.side_effect = RuntimeError("Something else broke")

    session = _make_session()
    failures = _make_failures(["span_2"])
    with pytest.raises(RuntimeError, match="Something else broke"):
        analyze_root_cause(session, failures)


@patch("strands_evals.detectors.root_cause_analyzer.detect_failures")
@patch("strands_evals.detectors.root_cause_analyzer.Agent")
def test_analyze_root_cause_auto_detects_failures(mock_agent_cls, mock_detect):
    """When failures=None, detect_failures is called automatically."""
    from strands_evals.types.detector import FailureOutput

    mock_detect.return_value = FailureOutput(
        session_id="sess_1",
        failures=_make_failures(["span_2"]),
    )
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_result = MagicMock()
    mock_result.structured_output = _make_rca_structured_output(["span_2"])
    mock_agent.return_value = mock_result

    session = _make_session()
    output = analyze_root_cause(session, failures=None)

    mock_detect.assert_called_once()
    assert len(output.root_causes) == 1


@patch("strands_evals.detectors.root_cause_analyzer.detect_failures")
def test_analyze_root_cause_auto_detect_no_failures(mock_detect):
    """When auto-detection finds no failures, return empty RCAOutput."""
    from strands_evals.types.detector import FailureOutput

    mock_detect.return_value = FailureOutput(session_id="sess_1", failures=[])

    session = _make_session()
    output = analyze_root_cause(session, failures=None)

    assert isinstance(output, RCAOutput)
    assert output.root_causes == []


def test_analyze_root_cause_multiple_root_causes():
    """Verify multiple root causes are returned correctly."""
    output = _make_rca_structured_output(["span_1", "span_2", "span_3"])
    result = _parse_structured_result(output)
    assert len(result) == 3
    for i, sid in enumerate(["span_1", "span_2", "span_3"]):
        assert result[i].failure_span_id == sid
        assert result[i].root_cause_explanation == f"Failure at {sid} due to test issue"
