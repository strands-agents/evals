"""Integration tests for Claude Agent SDK evaluation with OpenInference instrumentation.

Creates Claude Agent SDK agents, runs them against Bedrock, captures OpenInference
traces (scope: openinference.instrumentation.claude_agent_sdk) in memory, and
evaluates using strands-evals via the OpenInferenceSessionMapper.

Requirements:
    pip install strands-agents-evals[claude]
    AWS credentials configured for Amazon Bedrock access.

Run with: pytest tests_integ/test_claude_openinference_eval.py -v
"""

import asyncio
import os
import threading

import pytest
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    CorrectnessEvaluator,
    GoalSuccessRateEvaluator,
    ToolSelectionAccuracyEvaluator,
)
from strands_evals.mappers import OpenInferenceSessionMapper, detect_otel_mapper, readable_spans_to_dicts
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.types.trace import AgentInvocationSpan, Session, ToolExecutionSpan

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

BEDROCK_ENV = {
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "ANTHROPIC_MODEL": DEFAULT_MODEL_ID,
    "AWS_REGION": os.environ.get("AWS_REGION", "us-west-2"),
}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def telemetry():
    """Setup OpenTelemetry with in-memory exporter and Claude Agent SDK instrumentation."""
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument()

    yield telemetry

    instrumentor.uninstrument()


# =============================================================================
# Helpers
# =============================================================================

# Import query lazily after instrumentation is set up (module-level import
# would bypass instrumentation). The fixture ensures instrumentation runs first.
_query_func = None
_query_lock = threading.Lock()


def _get_query():
    """Lazily import the query function after instrumentation is applied."""
    global _query_func
    if _query_func is None:
        with _query_lock:
            if _query_func is None:
                from claude_agent_sdk import query

                _query_func = query
    return _query_func


async def _run_claude_agent(prompt: str, *, agents: dict | None = None) -> str:
    """Run a Claude Agent SDK query and return the final response text."""
    query = _get_query()

    allowed_tools = ["Agent", "Bash"] if agents else ["Bash"]

    options = ClaudeAgentOptions(
        allowed_tools=allowed_tools,
        max_turns=10,
        agents=agents or {},
        env=BEDROCK_ENV,
    )

    async def _run_query() -> str:
        out = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "result"):
                if message.result:
                    out = message.result
                elif getattr(message, "is_error", False):
                    raise RuntimeError(
                        f"Claude Agent SDK run failed: subtype={getattr(message, 'subtype', None)!r} "
                        f"terminal_reason={getattr(message, 'terminal_reason', None)!r}"
                    )
        return out

    final_output = await asyncio.wait_for(_run_query(), timeout=180)

    return final_output


# =============================================================================
# Tests — Single Agent
# =============================================================================


def test_claude_single_query(telemetry):
    """Spans are captured, mapper is auto-detected, session is valid, and tool_call_ids are populated."""
    telemetry.in_memory_exporter.clear()
    response = asyncio.run(_run_claude_agent("Use bash to calculate: echo $((15 * 37)). Just give me the number."))
    spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())

    assert len(spans) > 0, "Should have captured OTEL spans"

    mapper = detect_otel_mapper(spans)
    assert isinstance(mapper, OpenInferenceSessionMapper), (
        f"Expected OpenInferenceSessionMapper but got {type(mapper).__name__}"
    )

    session = mapper.map_to_session(spans, session_id="test-single")

    assert session.session_id == "test-single"
    assert len(session.traces) > 0, "Should have at least one trace"
    assert "555" in response, f"Expected 555 in response, got: {response}"

    # Verify tool spans have tool_call_id populated and succeeded
    tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
    assert len(tool_spans) >= 1, "Expected at least one tool execution span"
    for tool_span in tool_spans:
        assert tool_span.tool_call.tool_call_id is not None, (
            f"tool_call_id should be populated, got None for tool '{tool_span.tool_call.name}'"
        )
        assert tool_span.tool_result.error is None, (
            f"tool '{tool_span.tool_call.name}' did not execute successfully: "
            f"error={tool_span.tool_result.error!r} content={tool_span.tool_result.content!r}"
        )


def test_claude_single_agent_evaluation(telemetry):
    """Single-agent session evaluates correctly via the full experiment pipeline."""
    test_cases = [
        Case[str, str](
            name="bash-calculation",
            input="Use bash to calculate: echo $((100 + 200)). Just give me the number.",
            expected_output="300",
            expected_assertion="The agent used the Bash tool to compute 100+200 and responded with 300.",
        ),
    ]

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        response = asyncio.run(_run_claude_agent(case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    experiment = Experiment(cases=test_cases, evaluators=[GoalSuccessRateEvaluator()])
    report = experiment.run_evaluations(task_function)

    assert len(report.scores) == 1
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"


# =============================================================================
# Tests — Multi Agent
# =============================================================================


def test_claude_multi_agent_evaluation(telemetry):
    """Multi-agent delegation + tool call spans evaluate correctly."""
    agents = {
        "math-specialist": AgentDefinition(
            description=(
                "Math specialist agent. Delegate any mathematical calculations "
                "including arithmetic, algebra, square roots, and numeric computations."
            ),
            prompt=(
                "You are a math specialist. Use Bash ONLY to run `python3 -c 'import math; print(...)'` "
                "for calculations. Never use Bash for anything else. "
                "Be precise and return just the numeric result."
            ),
            tools=["Bash"],
        ),
    }

    test_cases = [
        Case[str, str](
            name="multi-agent-math",
            input="Calculate the square root of 1764, then multiply that result by 3.",
            expected_output="126",
        ),
    ]

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        response = asyncio.run(_run_claude_agent(case.input, agents=agents))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    # One evaluator per extraction level: SESSION, TRACE, TOOL
    evaluators = [
        GoalSuccessRateEvaluator(),
        CorrectnessEvaluator(),
        ToolSelectionAccuracyEvaluator(),
    ]

    experiment = Experiment(cases=test_cases, evaluators=evaluators)
    report = experiment.run_evaluations(task_function)

    assert len(report.scores) == 3
    assert all(report.test_passes[:2]), f"Some evaluations failed: {report.reasons[:2]}"
    assert report.scores[2] >= 0.5, f"Tool selection accuracy too low: {report.reasons[2]}"

    session = Session.model_validate(report.cases[0]["actual_trajectory"])

    assert len(session.traces) >= 1, f"Multi-agent should produce at least 1 trace, got {len(session.traces)}"

    # Verify tool spans succeeded
    tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
    assert len(tool_spans) >= 1, "Expected at least one tool execution span from sub-agent"
    for tool_span in tool_spans:
        assert tool_span.tool_call.tool_call_id is not None, (
            f"tool_call_id should be populated, got None for tool '{tool_span.tool_call.name}'"
        )
        assert tool_span.tool_result.error is None, (
            f"tool '{tool_span.tool_call.name}' did not execute successfully: "
            f"error={tool_span.tool_result.error!r} content={tool_span.tool_result.content!r}"
        )

    executed = [s for s in tool_spans if s.tool_call.name != "Agent" and s.tool_result.error is None]
    assert executed, (
        "Expected at least one successful sub-agent tool execution, got "
        f"{[(s.tool_call.name, s.tool_result.error) for s in tool_spans]}"
    )

    agent_spans = [s for t in session.traces for s in t.spans if isinstance(s, AgentInvocationSpan)]
    assert len(agent_spans) >= 1, "Expected at least one AgentInvocationSpan"

    # Prove delegation actually happened
    delegations = [s.tool_call.arguments.get("subagent_type") for s in tool_spans if s.tool_call.name == "Agent"]
    assert "math-specialist" in delegations, f"Expected delegation to math-specialist, got {delegations}"
