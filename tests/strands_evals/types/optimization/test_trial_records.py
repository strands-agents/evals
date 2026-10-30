from datetime import datetime, timezone

import pytest

from strands_evals.types.evaluation import EvaluationOutput
from strands_evals.types.optimization import TrialRecord, TrialStatus, Usage
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

START = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 10, 26, 12, 0, 1, tzinfo=timezone.utc)


def _span_info(span_id, parent_span_id=None):
    return SpanInfo(
        trace_id="trace-1",
        span_id=span_id,
        session_id="session-1",
        parent_span_id=parent_span_id,
        start_time=START,
        end_time=END,
    )


def _session():
    spans = [
        AgentInvocationSpan(
            span_info=_span_info("agent"),
            user_prompt="What is 2 + 2?",
            agent_response="4",
            available_tools=[],
        ),
        InferenceSpan(
            span_info=_span_info("chat", parent_span_id="agent"),
            messages=[
                UserMessage(content=[TextContent(text="What is 2 + 2?")]),
                AssistantMessage(content=[TextContent(text="4")]),
            ],
        ),
        ToolExecutionSpan(
            span_info=_span_info("tool", parent_span_id="agent"),
            tool_call=ToolCall(name="calculator", arguments={"expression": "2 + 2"}),
            tool_result=ToolResult(content="4"),
        ),
    ]
    return Session(traces=[Trace(spans=spans, trace_id="trace-1", session_id="session-1")], session_id="session-1")


def _usage(cache_read_tokens=None):
    return Usage(
        tokens=120,
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=None,
        model_calls=1,
        seconds=1.5,
        models=["us.anthropic.claude-sonnet-4-5"],
    )


def _record(trajectory):
    return TrialRecord(
        candidate_id="abc123def456",
        case_name="arithmetic",
        run_index=0,
        split="validation",
        status=TrialStatus.COMPLETED,
        error_type=None,
        error=None,
        attempts=1,
        earlier=[],
        output="4",
        trajectory=trajectory,
        interactions=None,
        environment_state=None,
        evaluations={"HelpfulnessEvaluator": EvaluationOutput(score=1.0, test_pass=True, reason="Correct.")},
        details={"HelpfulnessEvaluator": [EvaluationOutput(score=1.0, test_pass=True, reason="Correct.")]},
        evaluator_errors={},
        usage=_usage(),
        judge_usage=_usage(),
        rollouts=1,
    )


def test_trial_record_with_session_round_trips_through_json():
    record = _record(_session())

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    # Compared by saved form, not ==: see the xfail test below for the one class that changes.
    assert restored.model_dump() == record.model_dump()
    assert isinstance(restored.trajectory, Session)


def test_message_roles_survive_a_round_trip():
    record = _record(_session())

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert isinstance(restored.trajectory, Session)
    inference = restored.trajectory.traces[0].spans[1]
    assert isinstance(inference, InferenceSpan)
    assert [message.role for message in inference.messages] == ["user", "assistant"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "types/trace.py: InferenceSpan.messages is UserMessage | AssistantMessage with no discriminator and "
        "UserMessage.role accepts any Role, so an AssistantMessage is restored as a UserMessage with "
        "role='assistant'. Read reloaded messages by .role, not by class."
    ),
)
def test_assistant_message_is_restored_as_assistant_message():
    record = _record(_session())

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert isinstance(restored.trajectory, Session)
    inference = restored.trajectory.traces[0].spans[1]
    assert isinstance(inference, InferenceSpan)
    assert isinstance(inference.messages[1], AssistantMessage)


def test_each_span_type_is_restored_as_itself():
    record = _record(_session())

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert isinstance(restored.trajectory, Session)
    span_types = [type(span) for span in restored.trajectory.traces[0].spans]
    assert span_types == [AgentInvocationSpan, InferenceSpan, ToolExecutionSpan]


def test_trial_record_with_list_trajectory_round_trips_through_json():
    record = _record(["calculator", "validator"])

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_trial_record_without_trajectory_round_trips_through_json():
    record = _record(None)

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_status_is_restored_as_trial_status():
    record = _record(None).model_copy(
        update={"status": TrialStatus.AGENT_FAILURE, "error_type": "ValueError", "error": "boom"}
    )

    restored = TrialRecord.model_validate_json(record.model_dump_json())

    assert restored.status is TrialStatus.AGENT_FAILURE


def test_unmeasured_cache_tokens_stay_none_not_zero():
    restored = Usage.model_validate_json(_usage().model_dump_json())

    assert restored.cache_read_tokens is None
    assert restored.cache_write_tokens is None


def test_measured_zero_cache_tokens_stay_zero():
    restored = Usage.model_validate_json(_usage(cache_read_tokens=0).model_dump_json())

    assert restored.cache_read_tokens == 0
