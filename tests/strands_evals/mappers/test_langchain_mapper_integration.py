"""Integration tests for LangChain mappers with the evaluation pipeline.

These tests verify that the LangChain mappers (both Traceloop/OpenLLMetry and OpenInference)
correctly integrate with the strands-evals evaluation framework.
"""

import json

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.mappers import (
    LangChainOtelSessionMapper,
    OpenInferenceSessionMapper,
    detect_otel_mapper,
)
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.trace import AgentInvocationSpan, ToolExecutionSpan

TRACELOOP_SCOPE = "opentelemetry.instrumentation.langchain"
OPENINFERENCE_SCOPE = "openinference.instrumentation.langchain"


def make_traceloop_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    attributes=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
):
    """Build a span dict for Traceloop/OpenLLMetry LangChain traces."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attributes or {},
        "scope": {"name": TRACELOOP_SCOPE, "version": "0.1.0"},
        "status": {"code": "OK"},
        "span_events": [],
    }


def make_openinference_span(
    trace_id="trace-1",
    span_id="span-1",
    parent_span_id=None,
    name="test-span",
    attributes=None,
    start_time=1700000000000000000,
    end_time=1700000001000000000,
):
    """Build a span dict for OpenInference LangChain traces."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attributes or {},
        "scope": {"name": OPENINFERENCE_SCOPE, "version": "0.1.0"},
        "status": {"code": "OK"},
        "span_events": [],
    }


def create_traceloop_agent_trace(user_query: str, agent_response: str, tool_name: str, tool_result: str):
    """Create a complete Traceloop agent trace with inference, tool use, and agent invocation spans."""
    trace_id = "trace-weather-1"

    # LLM inference span with tool call
    inference_span = make_traceloop_span(
        trace_id=trace_id,
        span_id="span-llm-1",
        attributes={
            "llm.request.type": "chat",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": user_query,
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "Let me check that for you.",
            "gen_ai.completion.0.tool_calls.0.name": tool_name,
            "gen_ai.completion.0.tool_calls.0.arguments": json.dumps({"city": "Seattle"}),
            "gen_ai.completion.0.tool_calls.0.id": "tc-1",
            "llm.request.functions.0.name": tool_name,
            "llm.request.functions.0.description": "Get current weather for a city",
        },
    )

    # Tool execution span
    tool_span = make_traceloop_span(
        trace_id=trace_id,
        span_id="span-tool-1",
        attributes={
            "traceloop.span.kind": "tool",
            "traceloop.entity.name": tool_name,
            "traceloop.entity.input": json.dumps({"inputs": {"city": "Seattle"}}),
            "traceloop.entity.output": json.dumps(
                {"output": {"kwargs": {"content": tool_result, "tool_call_id": "tc-1", "status": "success"}}}
            ),
        },
    )

    # Final LLM inference span after tool result
    inference_span_2 = make_traceloop_span(
        trace_id=trace_id,
        span_id="span-llm-2",
        attributes={
            "llm.request.type": "chat",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": f"Tool result: {tool_result}",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": agent_response,
        },
    )

    # Workflow/agent invocation span
    workflow_span = make_traceloop_span(
        trace_id=trace_id,
        span_id="span-workflow-1",
        attributes={
            "traceloop.span.kind": "workflow",
            "traceloop.entity.input": json.dumps(
                {"inputs": {"messages": [{"kwargs": {"content": user_query, "type": "human"}}]}}
            ),
            "traceloop.entity.output": json.dumps(
                {"outputs": {"messages": [{"kwargs": {"content": agent_response, "type": "ai"}}]}}
            ),
        },
    )

    return [inference_span, tool_span, inference_span_2, workflow_span]


def create_openinference_agent_trace(user_query: str, agent_response: str, tool_name: str, tool_result: str):
    """Create a complete OpenInference agent trace with LLM, tool, and chain spans."""
    trace_id = "trace-weather-2"

    # LLM span with tool call
    llm_span = make_openinference_span(
        trace_id=trace_id,
        span_id="span-llm-1",
        attributes={
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": user_query,
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Let me check that for you.",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": tool_name,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": json.dumps({"city": "Seattle"}),
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "tc-1",
            "llm.tools.0.tool.json_schema": json.dumps(
                {
                    "name": tool_name,
                    "description": "Get current weather for a city",
                    "input_schema": {"type": "object"},
                }
            ),
        },
    )

    # Tool span
    tool_span = make_openinference_span(
        trace_id=trace_id,
        span_id="span-tool-1",
        name=tool_name,
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": tool_name,
            "input.value": json.dumps({"city": "Seattle"}),
            "output.value": json.dumps({"content": tool_result, "tool_call_id": "tc-1", "status": "success"}),
        },
    )

    # Final LLM span
    llm_span_2 = make_openinference_span(
        trace_id=trace_id,
        span_id="span-llm-2",
        attributes={
            "openinference.span.kind": "LLM",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": f"Tool result: {tool_result}",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": agent_response,
        },
    )

    # Chain/LangGraph span (agent invocation)
    chain_span = make_openinference_span(
        trace_id=trace_id,
        span_id="span-chain-1",
        name="LangGraph",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": json.dumps({"messages": [{"kwargs": {"content": user_query, "type": "human"}}]}),
            "output.value": json.dumps({"messages": [{"kwargs": {"content": agent_response, "type": "ai"}}]}),
        },
    )

    return [llm_span, tool_span, llm_span_2, chain_span]


class TrajectoryCheckEvaluator(Evaluator[str, str]):
    """Evaluator that checks trajectory has expected tool usage."""

    def _check_trajectory(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        """Common logic for checking trajectory."""
        trajectory = evaluation_case.actual_trajectory
        expected_tool = evaluation_case.metadata.get("expected_tool")

        if not trajectory or not hasattr(trajectory, "traces"):
            return [EvaluationOutput(score=0.0, test_pass=False, reason="No trajectory found")]

        # Find tool execution spans
        tool_found = False
        for trace in trajectory.traces:
            for span in trace.spans:
                if isinstance(span, ToolExecutionSpan):
                    if span.tool_call.name == expected_tool:
                        tool_found = True
                        break

        if tool_found:
            return [EvaluationOutput(score=1.0, test_pass=True, reason=f"Found expected tool: {expected_tool}")]
        return [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason=f"Expected tool '{expected_tool}' not found in trajectory",
            )
        ]

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._check_trajectory(evaluation_case)

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._check_trajectory(evaluation_case)


class TestMapperAutoDetection:
    """Test that detect_otel_mapper correctly identifies LangChain mappers."""

    def test_detect_traceloop_mapper(self):
        """Traceloop spans should be detected and mapped to LangChainOtelSessionMapper."""
        spans = create_traceloop_agent_trace(
            user_query="What's the weather?",
            agent_response="It's sunny.",
            tool_name="get_weather",
            tool_result="Sunny, 72F",
        )
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, LangChainOtelSessionMapper)

    def test_detect_openinference_mapper(self):
        """OpenInference spans should be detected and mapped to OpenInferenceSessionMapper."""
        spans = create_openinference_agent_trace(
            user_query="What's the weather?",
            agent_response="It's sunny.",
            tool_name="get_weather",
            tool_result="Sunny, 72F",
        )
        mapper = detect_otel_mapper(spans)
        assert isinstance(mapper, OpenInferenceSessionMapper)


class TestTraceloopMapperIntegration:
    """Integration tests for Traceloop/OpenLLMetry mapper with evaluation pipeline."""

    def test_mapper_produces_valid_session(self):
        """Verify mapper produces a session with all expected span types."""
        spans = create_traceloop_agent_trace(
            user_query="What's the weather in Seattle?",
            agent_response="The weather in Seattle is rainy, 55F.",
            tool_name="get_weather",
            tool_result="Rainy, 55F",
        )

        mapper = LangChainOtelSessionMapper()
        session = mapper.map_to_session(spans, "test-session")

        assert session.session_id == "test-session"
        assert len(session.traces) == 1

        span_types = [type(s).__name__ for s in session.traces[0].spans]
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types

    def test_evaluation_with_traceloop_mapper(self):
        """Test full evaluation pipeline with Traceloop mapper."""
        test_cases = [
            Case[str, str](
                name="weather-seattle",
                input="What's the weather in Seattle?",
                metadata={"expected_tool": "get_weather"},
            ),
        ]

        def task_function(case: Case) -> dict:
            spans = create_traceloop_agent_trace(
                user_query=case.input,
                agent_response="The weather in Seattle is rainy, 55F.",
                tool_name="get_weather",
                tool_result="Rainy, 55F",
            )
            mapper = detect_otel_mapper(spans)
            session = mapper.map_to_session(spans, session_id=case.session_id)
            return {
                "output": "The weather in Seattle is rainy, 55F.",
                "trajectory": session,
            }

        experiment = Experiment(cases=test_cases, evaluators=[TrajectoryCheckEvaluator()])
        reports = experiment.run_evaluations(task_function)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0
        assert reports[0].test_passes[0] is True

    def test_traceloop_mapper_extracts_tools(self):
        """Verify that available tools are extracted from inference spans."""
        spans = create_traceloop_agent_trace(
            user_query="What's the weather?",
            agent_response="Sunny!",
            tool_name="get_weather",
            tool_result="Sunny",
        )

        mapper = LangChainOtelSessionMapper()
        session = mapper.map_to_session(spans, "test-session")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert len(agent_spans[0].available_tools) == 1
        assert agent_spans[0].available_tools[0].name == "get_weather"


class TestOpenInferenceMapperIntegration:
    """Integration tests for OpenInference mapper with evaluation pipeline."""

    def test_mapper_produces_valid_session(self):
        """Verify mapper produces a session with all expected span types."""
        spans = create_openinference_agent_trace(
            user_query="What's the weather in Tokyo?",
            agent_response="The weather in Tokyo is clear, 68F.",
            tool_name="get_weather",
            tool_result="Clear, 68F",
        )

        mapper = OpenInferenceSessionMapper()
        session = mapper.map_to_session(spans, "test-session")

        assert session.session_id == "test-session"
        assert len(session.traces) == 1

        span_types = [type(s).__name__ for s in session.traces[0].spans]
        assert "InferenceSpan" in span_types
        assert "ToolExecutionSpan" in span_types
        assert "AgentInvocationSpan" in span_types

    def test_evaluation_with_openinference_mapper(self):
        """Test full evaluation pipeline with OpenInference mapper."""
        test_cases = [
            Case[str, str](
                name="weather-tokyo",
                input="What's the weather in Tokyo?",
                metadata={"expected_tool": "get_weather"},
            ),
        ]

        def task_function(case: Case) -> dict:
            spans = create_openinference_agent_trace(
                user_query=case.input,
                agent_response="The weather in Tokyo is clear, 68F.",
                tool_name="get_weather",
                tool_result="Clear, 68F",
            )
            mapper = detect_otel_mapper(spans)
            session = mapper.map_to_session(spans, session_id=case.session_id)
            return {
                "output": "The weather in Tokyo is clear, 68F.",
                "trajectory": session,
            }

        experiment = Experiment(cases=test_cases, evaluators=[TrajectoryCheckEvaluator()])
        reports = experiment.run_evaluations(task_function)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0
        assert reports[0].test_passes[0] is True

    def test_openinference_mapper_extracts_tools(self):
        """Verify that available tools are extracted from LLM spans."""
        spans = create_openinference_agent_trace(
            user_query="What's the weather?",
            agent_response="Sunny!",
            tool_name="get_weather",
            tool_result="Sunny",
        )

        mapper = OpenInferenceSessionMapper()
        session = mapper.map_to_session(spans, "test-session")

        agent_spans = [s for s in session.traces[0].spans if isinstance(s, AgentInvocationSpan)]
        assert len(agent_spans) == 1
        assert len(agent_spans[0].available_tools) == 1
        assert agent_spans[0].available_tools[0].name == "get_weather"


class TestMultipleCasesEvaluation:
    """Test evaluation pipeline with multiple test cases."""

    def test_multiple_traceloop_cases(self):
        """Evaluate multiple cases using Traceloop mapper."""
        test_cases = [
            Case[str, str](
                name="weather-seattle",
                input="What's the weather in Seattle?",
                metadata={"expected_tool": "get_weather"},
            ),
            Case[str, str](
                name="weather-tokyo",
                input="Tell me the weather in Tokyo",
                metadata={"expected_tool": "get_weather"},
            ),
        ]

        weather_data = {
            "Seattle": "Rainy, 55F",
            "Tokyo": "Clear, 68F",
        }

        def task_function(case: Case) -> dict:
            city = "Seattle" if "Seattle" in case.input else "Tokyo"
            weather = weather_data[city]
            spans = create_traceloop_agent_trace(
                user_query=case.input,
                agent_response=f"The weather in {city} is {weather}.",
                tool_name="get_weather",
                tool_result=weather,
            )
            mapper = detect_otel_mapper(spans)
            session = mapper.map_to_session(spans, session_id=case.session_id)
            return {
                "output": f"The weather in {city} is {weather}.",
                "trajectory": session,
            }

        experiment = Experiment(cases=test_cases, evaluators=[TrajectoryCheckEvaluator()])
        reports = experiment.run_evaluations(task_function)

        assert len(reports) == 1
        assert len(reports[0].scores) == 2
        assert all(score == 1.0 for score in reports[0].scores)
        assert all(reports[0].test_passes)
        assert reports[0].overall_score == 1.0

    def test_multiple_openinference_cases(self):
        """Evaluate multiple cases using OpenInference mapper."""
        test_cases = [
            Case[str, str](
                name="weather-london",
                input="What's the weather in London?",
                metadata={"expected_tool": "get_weather"},
            ),
            Case[str, str](
                name="weather-paris",
                input="Tell me the weather in Paris",
                metadata={"expected_tool": "get_weather"},
            ),
        ]

        weather_data = {
            "London": "Cloudy, 60F",
            "Paris": "Partly cloudy, 65F",
        }

        def task_function(case: Case) -> dict:
            city = "London" if "London" in case.input else "Paris"
            weather = weather_data[city]
            spans = create_openinference_agent_trace(
                user_query=case.input,
                agent_response=f"The weather in {city} is {weather}.",
                tool_name="get_weather",
                tool_result=weather,
            )
            mapper = detect_otel_mapper(spans)
            session = mapper.map_to_session(spans, session_id=case.session_id)
            return {
                "output": f"The weather in {city} is {weather}.",
                "trajectory": session,
            }

        experiment = Experiment(cases=test_cases, evaluators=[TrajectoryCheckEvaluator()])
        reports = experiment.run_evaluations(task_function)

        assert len(reports) == 1
        assert len(reports[0].scores) == 2
        assert all(score == 1.0 for score in reports[0].scores)
        assert all(reports[0].test_passes)


@pytest.mark.asyncio
async def test_async_evaluation_with_langchain_mappers():
    """Test async evaluation workflow with LangChain mappers."""
    test_cases = [
        Case[str, str](
            name="weather-async",
            input="What's the weather in New York?",
            metadata={"expected_tool": "get_weather"},
        ),
    ]

    async def async_task(case: Case) -> dict:
        spans = create_traceloop_agent_trace(
            user_query=case.input,
            agent_response="The weather in New York is sunny, 72F.",
            tool_name="get_weather",
            tool_result="Sunny, 72F",
        )
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {
            "output": "The weather in New York is sunny, 72F.",
            "trajectory": session,
        }

    experiment = Experiment(cases=test_cases, evaluators=[TrajectoryCheckEvaluator()])
    reports = await experiment.run_evaluations_async(async_task)

    assert len(reports) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True
