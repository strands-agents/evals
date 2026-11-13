"""
Integration tests for ContextUtilizationEvaluator.

These tests make actual API calls to the judge model. The sessions are built by hand so that which
chunks the agent used is unambiguous; the tests check that a real model answers with chunk ids the
evaluator recognizes and marks the right ones.
"""

from datetime import datetime, timedelta, timezone

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import ContextUtilizationEvaluator
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)

LOGGER = "strands_evals.evaluators.context_utilization_evaluator"

DOCS = {
    "shipping.md": "Standard shipping takes 3-5 business days. Express shipping takes 1 business day and costs $15.",
    "refunds.md": "Refunds are issued within 14 days of a return. Opened software cannot be refunded.",
    "warranty.md": "Electronics carry a 2-year warranty. Batteries are covered for 6 months.",
    "stores.md": "Stores open 9am-9pm Monday to Saturday and 10am-6pm on Sunday.",
}


def _info(session_id: str, second: int) -> SpanInfo:
    t = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second)
    return SpanInfo(session_id=session_id, start_time=t, end_time=t)


def _agent(session_id: str, second: int, request: str, answer: str) -> AgentInvocationSpan:
    return AgentInvocationSpan(
        span_info=_info(session_id, second), user_prompt=request, agent_response=answer, available_tools=[]
    )


def _says(session_id: str, second: int, request: str, text: str) -> InferenceSpan:
    return InferenceSpan(
        span_info=_info(session_id, second),
        messages=[
            UserMessage(content=[TextContent(text=request)]),
            AssistantMessage(content=[TextContent(text=text)]),
        ],
    )


def _call(session_id: str, second: int, name: str, arguments: dict, output: str) -> ToolExecutionSpan:
    return ToolExecutionSpan(
        span_info=_info(session_id, second),
        tool_call=ToolCall(name=name, arguments=arguments, tool_call_id=f"{session_id}-{second}"),
        tool_result=ToolResult(content=output),
    )


def shipping_session() -> Session:
    """One turn: a search (S1.1), the shipping doc it answers from (S2.1), and a store-hours doc it ignores (S3.1)."""
    sid, request = "shipping", "How long does express shipping take and what does it cost?"
    answer = "Express shipping takes 1 business day and costs $15."
    hits = "\n---\n".join(f"{name}: {DOCS[name]}" for name in ("shipping.md", "refunds.md", "warranty.md"))
    spans = [
        _agent(sid, 0, request, answer),
        _says(sid, 1, request, "I'll search the help center."),
        _call(sid, 2, "search_docs", {"query": "express shipping"}, hits),
        _says(sid, 3, request, "shipping.md looks right; reading it in full."),
        _call(sid, 4, "read_doc", {"name": "shipping.md"}, DOCS["shipping.md"]),
        _call(sid, 5, "read_doc", {"name": "stores.md"}, DOCS["stores.md"]),
        _says(sid, 6, request, answer),
    ]
    return Session(traces=[Trace(spans=spans, trace_id=f"{sid}-1", session_id=sid)], session_id=sid)


def later_turn_session(with_turn_2: bool = True) -> Session:
    """Turn 1 reads refunds.md (S1.1), which answers it, and warranty.md (S2.1), which only turn 2's
    answer uses; turn 2 gathers nothing, so S2.1 is useful only through <later_turns>."""
    sid = "later-turn" if with_turn_2 else "later-turn-alone"
    first, second = "What is the refund window?", "How long are batteries covered?"
    turn_1 = [
        _agent(sid, 0, first, "Refunds are issued within 14 days of a return."),
        _says(sid, 1, first, "Reading the refunds and warranty policies."),
        _call(sid, 2, "read_doc", {"name": "refunds.md"}, DOCS["refunds.md"]),
        _call(sid, 3, "read_doc", {"name": "warranty.md"}, DOCS["warranty.md"]),
        _says(sid, 4, first, "Refunds are issued within 14 days of a return."),
    ]
    traces = [Trace(spans=turn_1, trace_id=f"{sid}-1", session_id=sid)]
    if with_turn_2:
        turn_2 = [_agent(sid, 10, second, "Batteries are covered for 6 months.")]
        traces.append(Trace(spans=turn_2, trace_id=f"{sid}-2", session_id=sid))
    return Session(traces=traces, session_id=sid)


SESSIONS = {
    "shipping": shipping_session,
    "later_turn": later_turn_session,
    "later_turn_alone": lambda: later_turn_session(with_turn_2=False),
}


def _task(case: Case) -> dict:
    session = SESSIONS[case.name]()
    answer = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)][-1].agent_response
    return {"output": answer, "trajectory": session}


def _check_shipping(output) -> None:
    assert output.label != "not_applicable", output.reason
    useful = set(output.metadata["useful_ids"])
    assert "S2.1" in useful  # the shipping doc it read and answered from
    assert "S3.1" not in useful  # the store hours it read and ignored
    assert 0.0 < output.score < 1.0
    assert output.metadata["judge_calls"] == 1
    assert output.metadata["dropped_ids"] == []  # every id the judge returned matched a chunk


@pytest.mark.asyncio
async def test_context_utilization_evaluator_async(caplog):
    """A real judge marks the chunks the agent used, including use in a later turn, with valid ids."""
    cases = [
        Case(name="shipping", input="How long does express shipping take and what does it cost?"),
        Case(name="later_turn", input="What is the refund window?"),
        Case(name="later_turn_alone", input="What is the refund window?"),
    ]
    experiment = Experiment(cases=cases, evaluators=[ContextUtilizationEvaluator()])

    with caplog.at_level("WARNING", logger=LOGGER):
        report = await experiment.run_evaluations_async(_task)

    outputs = {case["name"]: results[0] for case, results in zip(report.cases, report.detailed_results, strict=True)}
    _check_shipping(outputs["shipping"])
    later, alone = outputs["later_turn"], outputs["later_turn_alone"]
    assert later.label != "not_applicable", later.reason
    # S2.1 (warranty.md) is used only by turn 2's answer, which the judge sees through <later_turns>.
    assert later.metadata["useful_ids"] == ["S1.1", "S2.1"]
    assert later.score == 1.0
    assert later.metadata["judge_calls"] == 1  # turn 2 gathered nothing, so only turn 1 is judged
    assert later.metadata["dropped_ids"] == []
    # Without turn 2 the same read is unused, so the later turn is what made it count.
    assert alone.metadata["useful_ids"] == ["S1.1"]
    assert alone.score < later.score
    assert "dropped_ids" not in caplog.text  # every id the judge returned matched a chunk


def test_context_utilization_evaluator_sync(caplog):
    """The sync entry point runs the judge end to end."""
    experiment = Experiment(
        cases=[Case(name="shipping", input="How long does express shipping take and what does it cost?")],
        evaluators=[ContextUtilizationEvaluator()],
    )

    with caplog.at_level("WARNING", logger=LOGGER):
        report = experiment.run_evaluations(_task)

    _check_shipping(report.detailed_results[0][0])
    assert "dropped_ids" not in caplog.text
