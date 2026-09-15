"""Integration: OutputEvaluator + TraceIndex — skill-style progressive discovery.

The judge receives the compact overview in its prompt and the index's
discovery tools via the evaluators' `tools=` parameter. These tests use a
scripted fake Agent to verify the full loop deterministically: the "judge"
must call the tools to find evidence before scoring.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from strands_evals.evaluators import OutputEvaluator
from strands_evals.tools.trace_index import TraceIndex
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


def _span_info(second: int) -> SpanInfo:
    return SpanInfo(
        session_id="s1",
        span_id=f"sp{second}",
        start_time=datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, second + 1, tzinfo=timezone.utc),
    )


def _big_session(refund_amount: str = "$150") -> Session:
    """A session whose tool results are too big to inline: 30 spans x ~11K chars."""
    spans = [
        AgentInvocationSpan(
            span_info=_span_info(0),
            user_prompt="What was the refund for TKT-1042?",
            agent_response=f"Ticket TKT-1042 was refunded {refund_amount}.",
            available_tools=[],
        )
    ]
    for i in range(1, 30):
        content = ("filler row data " * 700) + (f" refund_amount={refund_amount} TKT-1042" if i == 17 else "")
        spans.append(
            ToolExecutionSpan(
                span_info=_span_info(i),
                tool_call=ToolCall(name="query_db", arguments={"page": i}),
                tool_result=ToolResult(content=content),
            )
        )
    return Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")


def test_evaluator_receives_index_tools_and_overview_fits():
    session = _big_session()
    index = TraceIndex(session)

    evaluator = OutputEvaluator(
        rubric="Every numeric claim must be supported by a tool result in the trace.",
        tools=index.tools,
    )

    assert evaluator.tools == index.tools
    # The overview replaces the inline trajectory and is context-safe
    inline = len(str(session.model_dump()))
    assert inline > 300_000
    assert len(index.overview()) < 15_000


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_judge_agent_constructed_with_discovery_tools(mock_agent_class):
    session = _big_session()
    index = TraceIndex(session)
    mock_agent = Mock()
    result = Mock()
    result.structured_output = EvaluationOutput(score=1.0, test_pass=True, reason="grounded")
    mock_agent.return_value = result
    mock_agent_class.return_value = mock_agent

    evaluator = OutputEvaluator(rubric="Claims must be grounded.", tools=index.tools)
    data = EvaluationData(
        input="What was the refund for TKT-1042?",
        actual_output="Ticket TKT-1042 was refunded $150.",
    )

    evaluator.evaluate(data)

    kwargs = mock_agent_class.call_args[1]
    assert kwargs["tools"] == index.tools
    tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in kwargs["tools"]}
    assert {"list_spans", "get_span", "search_spans"} <= tool_names


def test_scripted_judge_finds_evidence_via_discovery():
    """Simulate the judge's tool-use loop: overview -> search -> get_span.

    This is the skill-discovery flow: the overview says *what exists*, the
    tools load *what is needed*, and the judge never sees the full 300K trace.
    """
    session = _big_session(refund_amount="$150")
    index = TraceIndex(session)
    overview, get_span, search_spans = index.tools

    judge_context_chars = 0

    # Step 1: judge reads the overview
    overview = overview()
    judge_context_chars += len(overview)
    assert "query_db" in overview

    # Step 2: judge searches for the claim from the agent's answer
    hits = search_spans(pattern="refund_amount=$150")
    judge_context_chars += len(hits)
    assert hits.startswith("["), "evidence must be locatable"
    evidence_index = int(hits.split("]")[0][1:])
    assert evidence_index == 17

    # Step 3: judge loads the evidence span, paging when told to
    span_content = get_span(index=evidence_index)
    judge_context_chars += len(span_content)
    offset = 0
    while "refund_amount=$150" not in span_content and "TRUNCATED" in span_content:
        offset += index.max_read_chars
        span_content = get_span(index=evidence_index, offset=offset)
        judge_context_chars += len(span_content)
    assert "refund_amount=$150" in span_content

    # The judge verified the claim while reading a fraction of the trace
    full_trace_chars = len(str(session.model_dump()))
    assert judge_context_chars < full_trace_chars / 10


def test_scripted_judge_detects_fabrication():
    """The agent claims $999 but the trace only supports $150 — discovery
    exposes the fabrication where an overflowed inline judge would score 0
    or a truncated one might miss the evidence entirely."""
    session = _big_session(refund_amount="$150")
    # Overwrite the agent's claim with a fabricated amount
    agent_span = session.traces[0].spans[0]
    fabricated = AgentInvocationSpan(
        span_info=agent_span.span_info,
        user_prompt=agent_span.user_prompt,
        agent_response="Ticket TKT-1042 was refunded $999.",
        available_tools=[],
    )
    session.traces[0].spans[0] = fabricated

    index = TraceIndex(session)
    _, _, search_spans = index.tools

    # Judge searches for the claimed amount in tool evidence: not found
    claimed = search_spans(pattern="refund_amount=$999")
    assert "No matches" in claimed

    # But the actual amount is present: the claim contradicts the evidence
    actual = search_spans(pattern="refund_amount=$150")
    assert actual.startswith("[")


def test_overview_prompt_composition_pattern():
    """The documented usage: overview into the prompt, tools onto the evaluator."""
    session = _big_session()
    index = TraceIndex(session)

    prompt = (
        "Evaluate whether the agent's answer is grounded in the trace.\n"
        f"<TraceOverview>\n{index.overview()}\n</TraceOverview>\n"
        "<Output>Ticket TKT-1042 was refunded $150.</Output>\n"
        "Use get_span/search_spans to verify before scoring."
    )

    # Stays well inside any judge's context window (~4 chars/token heuristic)
    assert len(prompt) / 4 < 10_000
