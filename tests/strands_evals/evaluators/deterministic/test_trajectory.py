from datetime import datetime, timezone

import pytest

from strands_evals.evaluators.deterministic.trajectory import ToolCalled
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


def _make_span_info(session_id="s1"):
    now = datetime.now(timezone.utc)
    return SpanInfo(trace_id="t1", span_id="sp1", session_id=session_id, start_time=now, end_time=now)


def _make_tool_span(tool_name, session_id="s1"):
    return ToolExecutionSpan(
        span_info=_make_span_info(session_id),
        tool_call=ToolCall(name=tool_name, arguments={}),
        tool_result=ToolResult(content="result"),
    )


def _make_session(tool_names, session_id="s1"):
    spans = [_make_tool_span(name, session_id) for name in tool_names]
    return Session(
        session_id=session_id,
        traces=[Trace(trace_id="t1", session_id=session_id, spans=spans)],
    )


class TestToolCalled:
    def test_tool_found_in_list_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=["search", "calculator", "format"])
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_tool_not_found_in_list_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=["search", "format"])
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_tool_found_in_session_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        session = _make_session(["search", "calculator"])
        data = EvaluationData(input="q", actual_trajectory=session)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_tool_not_found_in_session_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        session = _make_session(["search", "format"])
        data = EvaluationData(input="q", actual_trajectory=session)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_none_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=None)
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False
        assert "no trajectory" in results[0].reason

    def test_empty_list_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=[])
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_empty_session_trajectory(self):
        evaluator = ToolCalled(tool_name="calculator")
        session = Session(session_id="s1", traces=[])
        data = EvaluationData(input="q", actual_trajectory=session)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_reason_on_found(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=["calculator"])
        results = evaluator.evaluate(data)
        assert "was called" in results[0].reason

    def test_reason_on_not_found(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=["search"])
        results = evaluator.evaluate(data)
        assert "was not called" in results[0].reason

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = ToolCalled(tool_name="calculator")
        data = EvaluationData(input="q", actual_trajectory=["calculator"])
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True

    def test_to_dict(self):
        evaluator = ToolCalled(tool_name="calculator")
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "ToolCalled"
        assert d["tool_name"] == "calculator"
