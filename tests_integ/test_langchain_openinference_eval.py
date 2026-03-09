"""
Integration tests for LangChain agent evaluation with OpenInference instrumentation.

These tests create real LangChain agents, run them with Bedrock, capture OTEL traces
using OpenInference instrumentation, and evaluate using strands-evals.

Requirements:
    pip install strands-agents-evals[langchain]
"""

import os

import pytest
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from openinference.instrumentation.langchain import LangChainInstrumentor

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.mappers import detect_otel_mapper, readable_spans_to_dicts
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.trace import ToolExecutionSpan

# Use the same model that works in CI (same as evaluators use)
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


class ToolUsageEvaluator(Evaluator[str, str]):
    """Evaluator that checks if the expected tool was used."""

    def _check_tool_usage(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        """Common logic for checking tool usage."""
        trajectory = evaluation_case.actual_trajectory
        expected_tool = evaluation_case.metadata.get("expected_tool")

        if not trajectory or not hasattr(trajectory, "traces"):
            return [EvaluationOutput(score=0.0, test_pass=False, reason="No trajectory found")]

        for trace_obj in trajectory.traces:
            for span in trace_obj.spans:
                if isinstance(span, ToolExecutionSpan):
                    if span.tool_call.name == expected_tool:
                        return [
                            EvaluationOutput(
                                score=1.0,
                                test_pass=True,
                                reason=f"Found expected tool: {expected_tool}",
                            )
                        ]

        return [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason=f"Expected tool '{expected_tool}' not found",
            )
        ]

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._check_tool_usage(evaluation_case)

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._check_tool_usage(evaluation_case)


@pytest.fixture(scope="module")
def telemetry():
    """Setup OTEL tracing with StrandsEvalsTelemetry for OpenInference."""
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    LangChainInstrumentor().instrument()
    return telemetry


@pytest.fixture
def weather_tool():
    """Create a weather tool for testing."""

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The city name to get weather for

        Returns:
            Weather information string
        """
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
def create_agent_func(weather_tool):
    """Factory to create LangChain ReAct agents."""

    def _create_agent():
        region = os.environ.get("AWS_REGION", "us-west-2")
        llm = ChatBedrock(
            model_id=DEFAULT_MODEL_ID,
            region_name=region,
            model_kwargs={"temperature": 0},
        )
        return create_react_agent(llm, [weather_tool])

    return _create_agent


def test_openinference_single_query(telemetry, create_agent_func):
    """Test single query evaluation with OpenInference instrumentation."""
    telemetry.in_memory_exporter.clear()

    agent = create_agent_func()
    query = "What's the weather in Seattle?"

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])
    response = messages[-1].content if messages else ""

    spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
    assert len(spans) > 0, "Should have captured OTEL spans"

    mapper = detect_otel_mapper(spans)
    session = mapper.map_to_session(spans, session_id="test-single")

    assert session.session_id == "test-single"
    assert len(session.traces) > 0, "Should have at least one trace"
    assert "seattle" in response.lower() or "weather" in response.lower()


def test_openinference_evaluation_pipeline(telemetry, create_agent_func):
    """Test full evaluation pipeline with OpenInference instrumentation."""
    test_cases = [
        Case[str, str](
            name="weather-seattle",
            input="What's the weather in Seattle?",
            metadata={"expected_tool": "get_weather"},
        ),
    ]

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()

        agent = create_agent_func()
        result = agent.invoke({"messages": [HumanMessage(content=case.input)]})

        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)

        return {
            "output": response,
            "trajectory": session,
        }

    experiment = Experiment(cases=test_cases, evaluators=[ToolUsageEvaluator()])
    reports = experiment.run_evaluations(task_function)

    assert len(reports) == 1
    assert len(reports[0].scores) == 1
    # The tool should have been used
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


def test_openinference_multiple_cases(telemetry, create_agent_func):
    """Test evaluation with multiple test cases."""
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

    def task_function(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()

        agent = create_agent_func()
        result = agent.invoke({"messages": [HumanMessage(content=case.input)]})

        messages = result.get("messages", [])
        response = messages[-1].content if messages else ""

        spans = readable_spans_to_dicts(telemetry.in_memory_exporter.get_finished_spans())
        mapper = detect_otel_mapper(spans)
        session = mapper.map_to_session(spans, session_id=case.session_id)

        return {
            "output": response,
            "trajectory": session,
        }

    experiment = Experiment(cases=test_cases, evaluators=[ToolUsageEvaluator()])
    reports = experiment.run_evaluations(task_function)

    assert len(reports) == 1
    assert len(reports[0].scores) == 2
    assert reports[0].overall_score == 1.0
    assert all(reports[0].test_passes)
