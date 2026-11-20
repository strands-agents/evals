"""Tests for HoneyHiveProvider with recorded HoneyHive sessions served by a mock httpx transport."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from strands_evals.providers import HoneyHiveProvider, honeyhive_provider
from strands_evals.providers.exceptions import ProviderError, SessionNotFoundError
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    TextContent,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResultContent,
    UserMessage,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
# The same calculator agent, written with an older and a current Strands Python and with Strands TypeScript.
# Each fixture's calculator tool takes different arguments.
TOOL_ARGUMENTS = {
    "honeyhive_strands_py_session.json": {"a": 25, "b": 4},
    "honeyhive_strands_py_current_session.json": {"expression": "25 * 4"},
    "honeyhive_strands_ts_session.json": {"a": 25, "b": 4},
}
FIXTURES = list(TOOL_ARGUMENTS)
TS_FIXTURE = "honeyhive_strands_ts_session.json"

_REAL_CLIENT = httpx.Client

Handler = Callable[[httpx.Request], httpx.Response]


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURE_DIR / name).read_text())["events"]


def _search(events: list[dict[str, Any]], requests: list[httpx.Request] | None = None) -> Handler:
    """Serve `events` like `POST /v1/events/search`: one page per request, plus the total `count`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        body = json.loads(request.content)
        page, limit = body["page"], body["limit"]
        return httpx.Response(200, json={"events": events[(page - 1) * limit : page * limit], "count": len(events)})

    return handler


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(honeyhive_provider, "_RETRY_BACKOFF_BASE", 0)


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], HoneyHiveProvider]:
    """Return a factory for a provider whose HTTP client sends requests to `handler`."""
    monkeypatch.setenv("HH_API_KEY", "test-key")
    monkeypatch.delenv("HH_API_URL", raising=False)

    def build(handler: Handler) -> HoneyHiveProvider:
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: _REAL_CLIENT(transport=transport, **kwargs))
        return HoneyHiveProvider()

    return build


def _spans(serve: Callable[[Handler], HoneyHiveProvider], events: list[dict[str, Any]]) -> list[Any]:
    data = serve(_search(events)).get_evaluation_data(events[0]["session_id"])
    return data["trajectory"].traces[0].spans


def _tool_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in events if e["event_name"].startswith("execute_tool"))


def _tool_message(events: list[dict[str, Any]]) -> dict[str, Any]:
    second_call = max((e for e in events if e["event_type"] == "model"), key=lambda e: len(e["inputs"]["chat_history"]))
    return next(m for m in second_call["inputs"]["chat_history"] if m["role"] == "tool")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_converts_strands_session(fixture: str, serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    events = _load(fixture)
    data = serve(_search(events)).get_evaluation_data(events[0]["session_id"])

    assert data["output"] == "25 multiplied by 4 is 100."
    # The session event and the event loop cycle spans are dropped.
    spans = data["trajectory"].traces[0].spans
    assert [type(s) for s in spans].count(InferenceSpan) == 2
    assert [s.span_info.start_time for s in spans] == sorted(s.span_info.start_time for s in spans)

    agent, tool = (next(s for s in spans if isinstance(s, t)) for t in (AgentInvocationSpan, ToolExecutionSpan))
    assert agent.user_prompt == "What is 25 * 4?"
    assert [t.name for t in agent.available_tools] == ["calculator"]
    assert tool.tool_call.name == "calculator"
    assert tool.tool_call.arguments == TOOL_ARGUMENTS[fixture]
    assert float(tool.tool_result.content) == 100
    assert tool.tool_call.tool_call_id is not None
    assert tool.tool_result.tool_call_id == tool.tool_call.tool_call_id


@pytest.mark.parametrize("fixture", FIXTURES)
def test_inference_pairs_tool_call_with_result(fixture: str, serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    first, second = (s for s in _spans(serve, _load(fixture)) if isinstance(s, InferenceSpan))

    assert first.messages[0] == UserMessage(content=[TextContent(text="What is 25 * 4?")])
    call = first.messages[-1].content[0]
    assert isinstance(call, ToolCallContent)
    assert (call.name, call.arguments) == ("calculator", TOOL_ARGUMENTS[fixture])

    assert [type(m) for m in second.messages] == [UserMessage, AssistantMessage, UserMessage, AssistantMessage]
    result = second.messages[2].content[0]
    assert isinstance(result, ToolResultContent)
    assert result.tool_call_id == call.tool_call_id
    assert result.error is None
    assert second.messages[-1].content == [TextContent(text="25 multiplied by 4 is 100.")]


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected_result"),
    [
        (
            {"parameters": {"a": 25, "b": 4}},
            {"result": '[{"json":{"stock":0}},{"text":" units"}]'},
            '{"stock": 0} units',
        ),
        ({"parameters": {"a": 25, "b": 4}}, {}, ""),
        ({"gen_ai.tool.call.arguments": '{"a": 25, "b": 4}'}, {"gen_ai.tool.call.result": "100"}, "100"),
    ],
    ids=["ts-content-blocks", "missing-result", "genai-attributes"],
)
def test_tool_span_shapes(
    serve: Callable[[Handler], HoneyHiveProvider], inputs: dict[str, Any], outputs: dict[str, Any], expected_result: str
) -> None:
    events = _load(TS_FIXTURE)
    tool_event = _tool_event(events)
    tool_event["metadata"]["tool_call_id"] = tool_event["outputs"]["tool_call_id"]
    tool_event["inputs"], tool_event["outputs"] = inputs, outputs

    tool = next(s for s in _spans(serve, events) if isinstance(s, ToolExecutionSpan))

    assert tool.tool_call.arguments == {"a": 25, "b": 4}
    assert tool.tool_call.tool_call_id == tool_event["metadata"]["tool_call_id"]
    assert tool.tool_result.content == expected_result


def test_tool_use_input_as_json_string(serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    events = _load(TS_FIXTURE)
    first_call = next(e for e in events if e["event_type"] == "model" and isinstance(e["outputs"]["content"], list))
    first_call["outputs"]["content"][0]["input"] = '{"a": 25, "b": 4}'

    first = next(s for s in _spans(serve, events) if isinstance(s, InferenceSpan))

    assert first.messages[-1].content[0].arguments == {"a": 25, "b": 4}


def test_parallel_tool_results(serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    events = _load(TS_FIXTURE)
    _tool_message(events)["content"] = json.dumps(
        [
            {"toolResult": {"toolUseId": "call_a", "status": "success", "content": [{"text": "100"}]}},
            {"toolResult": {"toolUseId": "call_b", "status": "error", "content": [{"text": "bad input"}]}},
        ]
    )

    second = [s for s in _spans(serve, events) if isinstance(s, InferenceSpan)][1]

    assert second.messages[2].content == [
        ToolResultContent(content="100", tool_call_id="call_a"),
        ToolResultContent(content="bad input", error="bad input", tool_call_id="call_b"),
    ]


def test_skips_event_that_fails_to_convert(serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    events = _load(TS_FIXTURE)
    del next(e for e in events if e["event_type"] == "model")["inputs"]["chat_history"]

    spans = _spans(serve, events)

    assert [type(s) for s in spans].count(InferenceSpan) == 1
    assert any(isinstance(s, AgentInvocationSpan) for s in spans)


def test_paginates_until_count(serve: Callable[[Handler], HoneyHiveProvider], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(honeyhive_provider, "_PAGE_SIZE", 3)
    events = _load(TS_FIXTURE)
    session_id = events[0]["session_id"]
    requests: list[httpx.Request] = []

    data = serve(_search(events, requests)).get_evaluation_data(session_id)

    assert [json.loads(r.content)["page"] for r in requests] == [1, 2, 3]
    assert requests[0].url == "https://api.dp1.us.honeyhive.ai/v1/events/search"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["filters"] == [
        {"field": "session_id", "operator": "is", "value": session_id, "type": "string"}
    ]
    assert len(data["trajectory"].traces[0].spans) == 4


def test_retries_read_timeout(serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    events = _load(TS_FIXTURE)
    search = _search(events)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return search(request)

    data = serve(handler).get_evaluation_data(events[0]["session_id"])

    assert len(calls) == 2
    assert data["output"] == "25 multiplied by 4 is 100."


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_api_key(serve: Callable[[Handler], HoneyHiveProvider], status: int) -> None:
    provider = serve(lambda request: httpx.Response(status, json={"message": "nope"}))
    with pytest.raises(ProviderError, match="project API key with read access"):
        provider.get_evaluation_data("any")


def test_missing_session(serve: Callable[[Handler], HoneyHiveProvider]) -> None:
    with pytest.raises(SessionNotFoundError):
        serve(_search([])).get_evaluation_data("00000000-0000-0000-0000-000000000000")
