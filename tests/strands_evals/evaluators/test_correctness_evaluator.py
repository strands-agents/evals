import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import CorrectnessEvaluator
from strands_evals.evaluators.correctness_evaluator import (
    CorrectnessRating,
    CorrectnessReferenceRating,
    CorrectnessReferenceScore,
    CorrectnessScore,
)
from strands_evals.mappers import (
    ADKOtelSessionMapper,
    GenericGenAISessionMapper,
    OpenInferenceSessionMapper,
)
from strands_evals.mappers.openai_agents_gen_ai_session_mapper import _OpenAIAgentsGenAISessionMapper
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)

# Real captured trace fixtures, reused from the mappers test suite. Each is a
# trajectory where a tool call happens before the final answer, so the extracted
# turn's session_history[-1] is a tool-execution list rather than the user message
# -- the exact shape that regressed the reference-mode USER QUERY in #355.
_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "mappers" / "fixtures"


def _load(name: str) -> object:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _genai_session_id(spans: list[dict]) -> str:
    """Mirror the generic mapper tests: derive session id from gen_ai.conversation.id."""
    for span in spans:
        sid = span.get("attributes", {}).get("gen_ai.conversation.id")
        if sid:
            return str(sid)
    return "test"


def _map_claude_live() -> Session:
    return OpenInferenceSessionMapper().map_to_session(_load("claude_live_spans.json"), "claude-sess")


def _map_adk_live() -> Session:
    return ADKOtelSessionMapper().map_to_session(_load("adk_live_spans.json"), "test-session-1")


def _map_smolagents_live() -> Session:
    return OpenInferenceSessionMapper().map_to_session(_load("smolagents_live_spans.json"), "smolagents-sess")


def _map_openinference_live() -> Session:
    return OpenInferenceSessionMapper().map_to_session(_load("openinference_live_spans.json"), "live-sess")


def _map_openai_agents_openinference_live() -> Session:
    return OpenInferenceSessionMapper().map_to_session(
        _load("openai_agents_openinference_live_spans.json"), "openai-agents-live-sess"
    )


def _map_openai_agents_genai_live() -> Session:
    return _OpenAIAgentsGenAISessionMapper().map_to_session(
        _load("openai_agents_genai_live_spans.json"), session_id="test"
    )


def _map_pydantic_ai_live() -> Session:
    spans = _load("pydantic_ai_live_spans.json")
    return GenericGenAISessionMapper().map_to_session(spans, session_id=_genai_session_id(spans))


def _map_otel_convention() -> Session:
    return GenericGenAISessionMapper().map_to_session(
        _load("otel_genai_convention_spans.json"), session_id="conv_otel_test_001"
    )


def _map_openinference_adot() -> Session:
    return OpenInferenceSessionMapper().map_to_session(_load("openinference_adot_spans.json"), "adot-sess")


def _map_claude_adot() -> Session:
    data = _load("claude_adot_spans.json")
    return OpenInferenceSessionMapper().map_to_session(data["spans"], data["session_id"])


def _map_openai_agents_openinference_adot() -> Session:
    return OpenInferenceSessionMapper().map_to_session(
        _load("openai_agents_openinference_adot_spans.json"), "openai-agents-adot-sess"
    )


def _map_openai_agents_genai_adot() -> Session:
    data = _load("openai_agents_genai_adot_spans.json")
    session_id = data.get("session_id", "test") if isinstance(data, dict) else "test"
    return _OpenAIAgentsGenAISessionMapper().map_to_session(data, session_id=session_id)


# Every content-bearing fixture in tests/.../mappers/fixtures, one row each, across
# the major agent frameworks and both capture shapes (live SDK export + ADOT/CloudWatch).
# Fields: (fixture id, session factory, expected user-query substring, tool_before_answer).
# tool_before_answer=True marks trajectories whose extracted final turn ends with a
# tool-execution list -- the exact shape that regressed the reference-mode USER QUERY in
# #355. Where it is False, a tool still ran but the final turn ends on a user message, so
# only the "query is recovered" guarantee applies, not the #355 trailing-list precondition.
# autogen_live_spans is covered separately: its telemetry carries no message content.
_REFERENCE_FIXTURES = [
    ("claude_live", _map_claude_live, "New York and Seattle", True),
    ("claude_adot", _map_claude_adot, "42 * 7", True),
    ("adk_live", _map_adk_live, "15 multiplied by 37", True),
    ("smolagents_live", _map_smolagents_live, "population of Tokyo", True),
    ("openinference_live", _map_openinference_live, "Taipei and Seattle", True),
    ("openinference_adot", _map_openinference_adot, "joke", False),
    ("openai_agents_openinference_live", _map_openai_agents_openinference_live, "42 multiplied by 17", True),
    ("openai_agents_openinference_adot", _map_openai_agents_openinference_adot, "42 * 7", True),
    ("openai_agents_genai_live", _map_openai_agents_genai_live, "stock price", True),
    ("openai_agents_genai_adot", _map_openai_agents_genai_adot, "42 * 7", True),
    ("pydantic_ai_live", _map_pydantic_ai_live, "weather", True),
    ("otel_genai_convention", _map_otel_convention, "Paris and London", True),
]


@pytest.fixture
def evaluation_data():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)

    tool_config = ToolConfig(name="calculator", description="Evaluate mathematical expressions")

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=[tool_config],
    )

    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )

    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        actual_trajectory=session,
        name="test",
    )


@pytest.fixture
def evaluation_data_with_reference():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)

    tool_config = ToolConfig(name="calculator", description="Evaluate mathematical expressions")

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=[tool_config],
    )

    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )

    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        expected_assertion="The agent should return the correct answer of 4.",
        actual_trajectory=session,
        name="test-reference",
    )


def test_init_with_defaults():
    evaluator = CorrectnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.reference_system_prompt is not None
    assert evaluator.system_prompt != evaluator.reference_system_prompt
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = CorrectnessEvaluator(
        version="v0", model="gpt-4", system_prompt="Custom", reference_system_prompt="Custom reference"
    )

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"
    assert evaluator.reference_system_prompt == "Custom reference"


def test_has_reference_true(evaluation_data_with_reference):
    evaluator = CorrectnessEvaluator()
    assert evaluator._has_reference(evaluation_data_with_reference) is True


def test_has_reference_false(evaluation_data):
    evaluator = CorrectnessEvaluator()
    assert evaluator._has_reference(evaluation_data) is False


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessRating(
        reasoning="The response correctly states 2+2=4", score=CorrectnessScore.PERFECTLY_CORRECT
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response correctly states 2+2=4"
    assert result[0].label == CorrectnessScore.PERFECTLY_CORRECT


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (CorrectnessScore.PERFECTLY_CORRECT, 1.0, True),
        (CorrectnessScore.PARTIALLY_CORRECT, 0.5, False),
        (CorrectnessScore.INCORRECT, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_with_reference(mock_agent_class, evaluation_data_with_reference):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(
        reasoning="The agent response matches the expected answer of 4",
        verdict=CorrectnessReferenceScore.CORRECT,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The agent response matches the expected answer of 4"
    assert result[0].label == CorrectnessReferenceScore.CORRECT


@pytest.mark.parametrize(
    "verdict,expected_value,expected_pass",
    [
        (CorrectnessReferenceScore.CORRECT, 1.0, True),
        (CorrectnessReferenceScore.INCORRECT, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_reference_score_mapping(
    mock_agent_class, evaluation_data_with_reference, verdict, expected_value, expected_pass
):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(reasoning="Test", verdict=verdict)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == verdict


# =============================================================================
# Regression for #355: reference-mode USER QUERY goes blank when a tool call
# precedes the final answer.
#
# _extract_trace_level appends the owned tool-execution list to the running
# history *before* snapshotting session_history for the turn, so the last turn's
# session_history[-1] is that list, not the UserMessage. _extract_user_prompt
# used to read only session_history[-1], so it returned "" for every such
# trajectory. These tests drive real captured spans from each supported agent
# framework through the mapper -> extractor -> evaluator path.
# =============================================================================


def _reference_case(session: Session) -> EvaluationData:
    """Wrap a mapped Session in a reference-mode case (expected_assertion set)."""
    return EvaluationData(
        input="",
        actual_output="",
        expected_assertion="The agent used its tools and produced the correct result.",
        actual_trajectory=session,
        name="reference-case",
    )


@pytest.mark.parametrize(
    "fixture_id,session_factory,expected_query_substring,tool_before_answer",
    _REFERENCE_FIXTURES,
    ids=[fid for fid, _, _, _ in _REFERENCE_FIXTURES],
)
def test_reference_prompt_has_user_query_for_fixture(
    fixture_id, session_factory, expected_query_substring, tool_before_answer
):
    """The user query must survive extraction for every captured framework trace.

    Runs the full mapper -> extractor -> evaluator path on real spans. For traces
    whose final turn ends with a tool-execution list (the #355 shape), also asserts
    that precondition so a regression that reshuffles history is caught.
    """
    session = session_factory()
    case = _reference_case(session)
    evaluator = CorrectnessEvaluator()
    parsed_input = evaluator._get_last_turn(case)

    if tool_before_answer:
        # The last history entry is a tool-execution list, so a naive
        # session_history[-1] read (the #355 bug) would miss the user message.
        assert isinstance(parsed_input.session_history[-1], list), (
            f"{fixture_id}: expected a trailing tool-execution list to exercise #355"
        )

    user_prompt = evaluator._extract_user_prompt(parsed_input)
    assert user_prompt, f"{fixture_id}: user query was blank"
    assert expected_query_substring in user_prompt

    # The rendered judge prompt must carry the query, not an empty USER QUERY block.
    prompt = evaluator._format_reference_prompt(parsed_input, case)
    assert "USER QUERY:\n\n" not in prompt
    assert f"USER QUERY:\n{user_prompt}" in prompt


def test_autogen_user_query_is_blank_because_telemetry_has_no_content():
    """AutoGen's native `autogen-core` telemetry carries no message content.

    The create_agent / invoke_agent / execute_tool spans only record agent and
    tool *names* -- no gen_ai.input.messages, output messages, or tool arguments.
    So the mapped UserMessage has empty text and the extracted query is "".
    This is a source-data limitation, not the #355 bug (which was an unreachable
    but present user message); pinned here so a future mapper change is noticed.
    """
    session = GenericGenAISessionMapper().map_to_session(_load("autogen_live_spans.json"), session_id="test")
    case = _reference_case(session)
    evaluator = CorrectnessEvaluator()
    parsed_input = evaluator._get_last_turn(case)

    assert isinstance(parsed_input.session_history[-1], list)
    assert evaluator._extract_user_prompt(parsed_input) == ""
