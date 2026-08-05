"""Integration tests for ADK agent evaluation via the ADKOtelSessionMapper.

Requirements:
    pip install strands-agents-evals[adk]
    Set GOOGLE_API_KEY environment variable with a valid Google AI API key.

Run with: pytest tests_integ/test_adk_eval.py -v
"""

import asyncio
import os
import uuid

import pytest
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    ConcisenessEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    GoalSuccessRateEvaluator,
    ResponseRelevanceEvaluator,
    ToolParameterAccuracyEvaluator,
    ToolSelectionAccuracyEvaluator,
)
from strands_evals.mappers import ADKOtelSessionMapper, detect_otel_mapper, readable_spans_to_dicts
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.types.trace import AgentInvocationSpan, Session, ToolExecutionSpan

# Uses Gemini 3 Flash via Google AI API (requires GOOGLE_API_KEY env var).
DEFAULT_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_MODEL = Gemini(model=DEFAULT_MODEL_ID, google_api_key=os.environ.get("GOOGLE_API_KEY", ""))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def telemetry():
    """ADK auto-instruments via the global TracerProvider."""
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    yield telemetry


@pytest.fixture
def weather_tool():
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        weather_data = {
            "seattle": "Rainy, 55F",
            "new york": "Sunny, 72F",
            "london": "Cloudy, 60F",
            "tokyo": "Clear, 68F",
        }
        city_lower = city.lower()
        for c, w in weather_data.items():
            if c in city_lower:
                return f"Weather in {city}: {w}"
        return f"Weather in {city}: Partly cloudy, 65F"

    return get_weather


@pytest.fixture
def unit_converter_tool():
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a value between units."""
        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.2f} {to_unit}"
        return f"Cannot convert from {from_unit} to {to_unit}"

    return convert_units


@pytest.fixture
def create_runner(weather_tool):
    def _create():
        agent = Agent(
            name="weather_agent",
            model=DEFAULT_MODEL,
            description="A helpful weather assistant",
            instruction="You are a weather assistant. Use the get_weather tool to look up weather. Be concise.",
            tools=[weather_tool],
        )
        return InMemoryRunner(agent=agent, app_name="test_app")

    return _create


@pytest.fixture
def create_multi_tool_runner(weather_tool, unit_converter_tool):
    def _create():
        agent = Agent(
            name="assistant_agent",
            model=DEFAULT_MODEL,
            description="A helpful assistant with weather and unit conversion tools",
            instruction=(
                "You are a helpful assistant. Use the get_weather tool for weather questions "
                "and the convert_units tool for unit conversions. Be concise."
            ),
            tools=[weather_tool, unit_converter_tool],
        )
        return InMemoryRunner(agent=agent, app_name="test_app")

    return _create


@pytest.fixture
def create_multi_agent_runner(weather_tool):
    def _create():
        weather_agent = Agent(
            name="weather_specialist",
            model=DEFAULT_MODEL,
            description="Specialist agent that looks up weather information for cities",
            instruction=(
                "You are a weather specialist. Use the get_weather tool to answer weather questions. Be concise."
            ),
            tools=[weather_tool],
        )

        root_agent = Agent(
            name="coordinator",
            model=DEFAULT_MODEL,
            description="A coordinator agent that delegates to specialists",
            instruction=(
                "You are a coordinator. For weather questions, delegate to the weather_specialist agent. "
                "Summarize the specialist's response in one sentence."
            ),
            sub_agents=[weather_agent],
        )

        return InMemoryRunner(agent=root_agent, app_name="test_app")

    return _create


# =============================================================================
# Helpers
# =============================================================================


async def _run_adk_agent(runner: InMemoryRunner, query: str, user_id: str | None = None) -> str:
    """Run an ADK agent and return the response text.

    Uses a unique user_id per call to prevent ADK from reusing conversation history.
    """
    if user_id is None:
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"

    session = await runner.session_service.create_session(app_name=runner.app_name, user_id=user_id)
    user_message = types.Content(role="user", parts=[types.Part.from_text(text=query)])

    response_text = ""
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=user_message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text

    return response_text


# =============================================================================
# Tests — Single Agent
# =============================================================================


def test_adk_single_query(telemetry, create_runner):
    """Spans are captured, mapper is auto-detected, and mapped into a valid session."""
    telemetry.in_memory_exporter.clear()

    runner = create_runner()
    response = asyncio.run(_run_adk_agent(runner, "What's the weather in Seattle?"))

    spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
    assert len(spans) > 0

    mapper = detect_otel_mapper(spans)
    assert isinstance(mapper, ADKOtelSessionMapper), f"Expected ADKOtelSessionMapper but got {type(mapper).__name__}"

    session = mapper.map_to_session(spans, session_id="test-single")

    assert session.session_id == "test-single"
    assert len(session.traces) > 0
    assert "seattle" in response.lower() or "weather" in response.lower()


def test_adk_single_agent_evaluation(telemetry, create_runner):
    """Single-agent session evaluates correctly (smoke test)."""
    test_cases = [
        Case[str, str](
            name="weather-seattle",
            input="What's the weather in Seattle?",
            expected_output="The weather in Seattle is rainy and 55F.",
            expected_assertion="The agent used the get_weather tool for Seattle and responded with the weather.",
        ),
    ]

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        runner = create_runner()
        response = asyncio.run(_run_adk_agent(runner, case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    experiment = Experiment(cases=test_cases, evaluators=[GoalSuccessRateEvaluator()])
    report = experiment.run_evaluations(task_function)

    assert len(report.scores) == 1
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"


def test_adk_single_agent_multi_tool(telemetry, create_multi_tool_runner):
    """Mapper distinguishes between different tool invocations in one session."""
    test_cases = [
        Case[str, str](
            name="multi-tool-weather",
            input="What's the weather in Seattle?",
            expected_output="The weather in Seattle is rainy and 55F.",
            expected_assertion="The agent used the get_weather tool with city='Seattle'.",
        ),
        Case[str, str](
            name="multi-tool-conversion",
            input="Convert 100 km to miles",
            expected_output="100 km is approximately 62.14 miles.",
            expected_assertion="The agent used the convert_units tool to convert 100 km to miles.",
        ),
    ]

    sessions: list = []

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        runner = create_multi_tool_runner()
        response = asyncio.run(_run_adk_agent(runner, case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        sessions.append(session)
        return {"output": response, "trajectory": session}

    evaluators = [ToolSelectionAccuracyEvaluator(), ToolParameterAccuracyEvaluator()]
    experiment = Experiment(cases=test_cases, evaluators=evaluators)
    report = experiment.run_evaluations(task_function)

    # Exact count: 2 cases × 2 evaluators = 4
    assert len(report.scores) == 4
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"

    # Verify mapped sessions have tool spans with tool_call_id populated
    for session in sessions:
        tool_spans = [s for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)]
        assert len(tool_spans) >= 1, "Expected at least one tool execution span"
        for tool_span in tool_spans:
            assert tool_span.tool_call.tool_call_id is not None, (
                f"tool_call_id should be populated, got None for tool '{tool_span.tool_call.name}'"
            )


# =============================================================================
# Tests — Multi Agent
# =============================================================================


def test_adk_multi_agent_evaluation(telemetry, create_multi_agent_runner):
    """Full evaluator coverage: delegation + tool call spans evaluate correctly at all levels."""
    test_cases = [
        Case[str, str](
            name="multi-agent-seattle",
            input="What's the weather in Seattle?",
            expected_output="The weather in Seattle is rainy and 55F.",
            expected_assertion="The agent obtained weather information for Seattle and responded with the result.",
        ),
    ]

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        runner = create_multi_agent_runner()
        response = asyncio.run(_run_adk_agent(runner, case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    evaluators = [
        GoalSuccessRateEvaluator(),
        CorrectnessEvaluator(),
        ResponseRelevanceEvaluator(),
        FaithfulnessEvaluator(),
        ConcisenessEvaluator(),
        ToolSelectionAccuracyEvaluator(),
    ]

    experiment = Experiment(cases=test_cases, evaluators=evaluators)
    report = experiment.run_evaluations(task_function)

    # Exact count: 1 case × 6 evaluators = 6
    assert len(report.scores) == 6
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"

    # Deserialize the trajectory from the first report case back into a Session object.
    session = Session.model_validate(report.cases[0]["actual_trajectory"])

    assert len(session.traces) == 1, (
        f"Multi-agent should produce 1 trace (all agents share the same OTEL trace_id), got {len(session.traces)}"
    )

    trace = session.traces[0]

    agent_spans = [s for s in trace.spans if isinstance(s, AgentInvocationSpan)]
    agent_names = {s.metadata.get("agent_name") for s in agent_spans}
    assert "coordinator" in agent_names, f"Expected coordinator agent, got {agent_names}"
    assert "weather_specialist" in agent_names, f"Expected weather_specialist agent, got {agent_names}"

    # Tool spans are scoped to their owning agent via agent_span_id
    tool_spans = [s for s in trace.spans if isinstance(s, ToolExecutionSpan)]

    coordinator_span_ids = {s.span_info.span_id for s in agent_spans if s.metadata.get("agent_name") == "coordinator"}
    specialist_span_ids = {
        s.span_info.span_id for s in agent_spans if s.metadata.get("agent_name") == "weather_specialist"
    }

    # get_weather should be owned by the specialist, not the coordinator
    for tool_span in tool_spans:
        if tool_span.tool_call.name == "get_weather":
            assert tool_span.agent_span_id in specialist_span_ids, (
                f"get_weather should be scoped to the specialist ({specialist_span_ids}), "
                f"got agent_span_id={tool_span.agent_span_id}"
            )
            assert tool_span.tool_call.tool_call_id is not None, (
                f"tool_call_id should be populated, got None for tool '{tool_span.tool_call.name}'"
            )
        elif tool_span.tool_call.name == "transfer_to_agent":
            assert tool_span.agent_span_id in coordinator_span_ids | specialist_span_ids, (
                f"transfer_to_agent should be scoped to a known agent span, got agent_span_id={tool_span.agent_span_id}"
            )

    # Coordinator tools should only be transfer_to_agent — no specialist tools leaked
    coordinator_tools = [s for s in tool_spans if s.agent_span_id in coordinator_span_ids]
    coordinator_tool_names = {s.tool_call.name for s in coordinator_tools}
    assert "get_weather" not in coordinator_tool_names, (
        f"Specialist tool leaked onto coordinator: {coordinator_tool_names}"
    )
    assert coordinator_tool_names <= {"transfer_to_agent"}, f"Unexpected tools on coordinator: {coordinator_tool_names}"

    # Specialist must have at least the get_weather call
    specialist_tools = [s for s in tool_spans if s.agent_span_id in specialist_span_ids]
    assert len(specialist_tools) >= 1, "Specialist should have at least one tool span"
    specialist_tool_names = {s.tool_call.name for s in specialist_tools}
    assert "get_weather" in specialist_tool_names, f"Expected get_weather tool, got {specialist_tool_names}"
