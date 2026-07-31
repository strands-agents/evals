"""Integration tests for ADK agent evaluation via the ADKOtelSessionMapper.

Requirements:
    pip install google-adk litellm opentelemetry-sdk strands-agents-evals

Run with: pytest tests_integ/test_adk_eval.py -v
"""

import asyncio
import uuid

import pytest
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
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

DEFAULT_MODEL = LiteLlm(model="bedrock/us.anthropic.claude-sonnet-4-6")


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
def create_agent_func(weather_tool):
    def _create_agent():
        agent = Agent(
            name="weather_agent",
            model=DEFAULT_MODEL,
            description="A helpful weather assistant",
            instruction="You are a weather assistant. Use the get_weather tool to look up weather. Be concise.",
            tools=[weather_tool],
        )
        runner = InMemoryRunner(agent=agent, app_name="test_app")
        return agent, runner

    return _create_agent


@pytest.fixture
def create_multi_tool_agent_func(weather_tool, unit_converter_tool):
    def _create_agent():
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
        runner = InMemoryRunner(agent=agent, app_name="test_app")
        return agent, runner

    return _create_agent


@pytest.fixture
def create_multi_agent_func(weather_tool):
    def _create_agents():
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

        runner = InMemoryRunner(agent=root_agent, app_name="test_app")
        return root_agent, runner

    return _create_agents


# =============================================================================
# Helpers
# =============================================================================


async def _run_adk_agent(runner: InMemoryRunner, query: str, user_id: str | None = None) -> str:
    """Run an ADK agent and return the response text.

    Uses a unique user_id per call to prevent ADK from reusing conversation history.
    """
    if user_id is None:
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"

    session = await runner.session_service.create_session(app_name="test_app", user_id=user_id)
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


def test_adk_single_query(telemetry, create_agent_func):
    """Spans are captured and mapped into a valid session."""
    telemetry.in_memory_exporter.clear()

    _, runner = create_agent_func()
    response = asyncio.run(_run_adk_agent(runner, "What's the weather in Seattle?"))

    spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
    assert len(spans) > 0

    mapper = detect_otel_mapper(spans)
    session = mapper.map_to_session(spans, session_id="test-single")

    assert session.session_id == "test-single"
    assert len(session.traces) > 0
    assert "seattle" in response.lower() or "weather" in response.lower()


def test_adk_mapper_detection(telemetry, create_agent_func):
    """ADKOtelSessionMapper is auto-detected for ADK spans."""
    telemetry.in_memory_exporter.clear()

    _, runner = create_agent_func()
    asyncio.run(_run_adk_agent(runner, "What's the weather in London?"))

    spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
    mapper = detect_otel_mapper(spans)

    assert isinstance(mapper, ADKOtelSessionMapper), f"Expected ADKOtelSessionMapper but got {type(mapper).__name__}"


def test_adk_single_agent_evaluation(telemetry, create_agent_func):
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
        _, runner = create_agent_func()
        response = asyncio.run(_run_adk_agent(runner, case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    experiment = Experiment(cases=test_cases, evaluators=[GoalSuccessRateEvaluator()])
    report = experiment.run_evaluations(task_function)

    assert len(report.scores) >= 1
    assert report.overall_score >= 0.5
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"


def test_adk_single_agent_multi_tool(telemetry, create_multi_tool_agent_func):
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

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        _, runner = create_multi_tool_agent_func()
        response = asyncio.run(_run_adk_agent(runner, case.input))

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)
        return {"output": response, "trajectory": session}

    evaluators = [ToolSelectionAccuracyEvaluator(), ToolParameterAccuracyEvaluator()]
    experiment = Experiment(cases=test_cases, evaluators=evaluators)
    report = experiment.run_evaluations(task_function)

    assert len(report.scores) >= 4
    assert report.overall_score >= 0.5
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"


# =============================================================================
# Tests — Multi Agent
# =============================================================================


def test_adk_multi_agent_evaluation(telemetry, create_multi_agent_func):
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
        _, runner = create_multi_agent_func()
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

    assert len(report.scores) >= 6
    assert report.overall_score >= 0.5
    assert all(report.test_passes), f"Some evaluations failed: {report.reasons}"
