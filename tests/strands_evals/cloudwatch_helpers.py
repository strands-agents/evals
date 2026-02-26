"""Shared test helpers for building CloudWatch body-format OTEL log records."""

import json


def make_log_record(
    trace_id="abc123",
    span_id="span-1",
    input_messages=None,
    output_messages=None,
    session_id="sess-1",
    time_nano=1000000000000000000,
):
    """Build a body-format OTEL log record dict as found in runtime log groups."""
    record = {
        "traceId": trace_id,
        "spanId": span_id,
        "timeUnixNano": time_nano,
        "body": {
            "input": {"messages": input_messages or []},
            "output": {"messages": output_messages or []},
        },
        "attributes": {"session.id": session_id},
    }
    return record


def make_user_message(text):
    """Build a user input message with double-encoded content."""
    return {"role": "user", "content": {"content": json.dumps([{"text": text}])}}


def make_assistant_text_message(text):
    """Build an assistant output message with double-encoded text content."""
    return {
        "role": "assistant",
        "content": {"message": json.dumps([{"text": text}]), "finish_reason": "end_turn"},
    }
