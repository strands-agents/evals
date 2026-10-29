"""Judge-reliability A/B: inline trajectory vs TraceIndex progressive discovery.

Compares OutputEvaluator judging the same cases two ways:

- **inline**: the full trajectory serialized into the prompt (status quo) —
  overflows or forces the judge to skim on large traces.
- **index**: compact overview in the prompt + discovery tools
  (list_spans / get_span / search_spans) via `tools=`.

Cases seed known ground truth: grounded claims (judge should pass) and
fabricated claims whose evidence is buried mid-trace (judge should fail).
A reliable judge scores grounded high and fabricated low; an unreliable one
collapses both to the same score.

Requires live Bedrock credentials; skips otherwise.
Run: pytest tests_integ/test_trace_index_judge_reliability.py -v -s
"""

import json
import logging
from datetime import datetime, timezone

import pytest

from strands_evals.evaluators import OutputEvaluator
from strands_evals.tools.trace_index import TraceIndex
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)

logger = logging.getLogger(__name__)

RUBRIC = (
    "Score whether every factual claim in the output (amounts, ticket ids, dates, statuses) "
    "is supported by evidence in the execution trace. 1.0 = every claim has direct tool-result "
    "evidence; 0.0 = a central claim contradicts or is absent from the tool results. "
    "Verify claims against the trace evidence before scoring; do not assume."
)


def _bedrock_available() -> bool:
    try:
        import boto3

        return boto3.client("sts").get_caller_identity() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _bedrock_available(), reason="Bedrock credentials not available")


def _span_info(second: int) -> SpanInfo:
    return SpanInfo(
        session_id="s1",
        span_id=f"sp{second}",
        start_time=datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, second + 1, tzinfo=timezone.utc),
    )


def _make_session(n_tool_spans: int, evidence: str, evidence_at: int, claim: str) -> Session:
    """Session with `n_tool_spans` bulky tool results; `evidence` buried at one index."""
    spans = [
        AgentInvocationSpan(
            span_info=_span_info(0),
            user_prompt="Summarize the resolution for ticket TKT-1042.",
            agent_response=claim,
            available_tools=[],
        )
    ]
    filler_rows = [{"ticket": f"TKT-{2000 + j}", "status": "open", "note": "unrelated backlog item"} for j in range(80)]
    for i in range(1, n_tool_spans + 1):
        payload = {"page": i, "rows": filler_rows}
        if i == evidence_at:
            payload["rows"] = [*filler_rows, {"ticket": "TKT-1042", "resolution": evidence}]
        spans.append(
            ToolExecutionSpan(
                span_info=_span_info(i),
                tool_call=ToolCall(name="query_tickets", arguments={"page": i}),
                tool_result=ToolResult(content=json.dumps(payload)),
            )
        )
    return Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")


GROUNDED_CLAIM = "Ticket TKT-1042 was resolved with a $150 refund."
FABRICATED_CLAIM = "Ticket TKT-1042 was resolved with a $975 refund."
EVIDENCE = "refunded $150 to customer"

CASES = [
    ("grounded", GROUNDED_CLAIM, True),
    ("fabricated", FABRICATED_CLAIM, False),
]

# ~30 spans x ~8K chars: large enough to stress a judge, small enough to run cheaply.
N_SPANS = 30
EVIDENCE_AT = 17


def _judge_inline(session: Session, claim: str) -> dict:
    evaluator = OutputEvaluator(rubric=RUBRIC)
    trace_text = str(session.model_dump())
    data = EvaluationData(
        input="Summarize the resolution for ticket TKT-1042.",
        actual_output=f"{claim}\n\n<FullTrace>{trace_text}</FullTrace>",
    )
    try:
        out = evaluator.evaluate(data)[0]
        return {"score": out.score, "reason": out.reason, "error": None}
    except Exception as e:
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}


def _judge_explore(session: Session, claim: str) -> dict:
    index = TraceIndex(session)
    evaluator = OutputEvaluator(rubric=RUBRIC, tools=index.tools)
    data = EvaluationData(
        input="Summarize the resolution for ticket TKT-1042.",
        actual_output=f"{claim}\n\n<TraceOverview>\n{index.overview()}\n</TraceOverview>",
    )
    try:
        out = evaluator.evaluate(data)[0]
        return {"score": out.score, "reason": out.reason, "error": None}
    except Exception as e:
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}


def test_judge_reliability_inline_vs_explore():
    results = {}
    for label, claim, should_pass in CASES:
        session = _make_session(N_SPANS, EVIDENCE, EVIDENCE_AT, claim)
        results[label] = {
            "expected_pass": should_pass,
            "inline": _judge_inline(session, claim),
            "index": _judge_explore(session, claim),
        }

    logger.info("results=<%s> | judge reliability inline vs index", json.dumps(results, indent=2, default=str))

    # The index judge must separate grounded from fabricated.
    tk_grounded = results["grounded"]["index"]["score"]
    tk_fabricated = results["fabricated"]["index"]["score"]
    assert tk_grounded is not None and tk_fabricated is not None, "index judge must not error"
    assert tk_grounded > tk_fabricated, (
        f"index judge failed to separate grounded ({tk_grounded}) from fabricated ({tk_fabricated})"
    )
    assert tk_grounded >= 0.7
    assert tk_fabricated <= 0.5
