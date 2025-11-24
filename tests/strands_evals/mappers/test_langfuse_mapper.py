from datetime import datetime, timezone
from unittest.mock import Mock

from strands_evals.mappers.langfuse_mapper import LangfuseSessionMapper
from strands_evals.types.trace import AgentInvocationSpan, InferenceSpan, ToolExecutionSpan


def create_mock_observation(obs_id, name, operation_name, input_data=None, output=None, metadata=None):
    """Create a mock Langfuse observation."""
    obs = Mock()
    obs.id = obs_id
    obs.name = name
    obs.input = input_data
    obs.output = output
    obs.metadata = metadata or {"attributes": {"gen_ai.operation.name": operation_name}}
    obs.start_time = datetime.now(timezone.utc)
    obs.end_time = datetime.now(timezone.utc)
    obs.created_at = datetime.now(timezone.utc)
    obs.parent_observation_id = None
    return obs


def create_mock_trace(trace_id, session_id, observations):
    """Create a mock Langfuse TraceWithFullDetails."""
    trace = Mock()
    trace.id = trace_id
    trace.session_id = session_id
    trace.observations = observations
    return trace


def test_inference_span():
    obs = create_mock_observation(
        "obs1",
        "chat",
        "chat",
        input_data=[
            {"role": "user", "content": '[{"text": "hello"}]'},
        ],
        output={"message": '[{"text": "hi"}]'},
    )

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)
    assert inference.messages[0].content[0].text == "hello"
    assert inference.messages[1].content[0].text == "hi"


def test_agent_invocation_span():
    obs = create_mock_observation(
        "obs1",
        "invoke_agent",
        "invoke_agent",
        input_data=[{"role": "user", "content": '[{"text": "2+2"}]'}],
        output={"message": "4"},
        metadata={"attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.tools": '["calc"]'}},
    )

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    agent = session.traces[0].spans[0]
    assert isinstance(agent, AgentInvocationSpan)
    assert agent.user_prompt == "2+2"
    assert agent.agent_response == "4"
    assert agent.available_tools[0].name == "calc"


def test_tool_execution_span():
    obs = create_mock_observation(
        "obs1",
        "calculate_price",
        "execute_tool",
        input_data=[{"role": "tool", "content": '{"expr": "2+2"}', "id": "t1"}],
        output={"message": '[{"text": "4"}]', "id": "t1"},
    )

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    tool = session.traces[0].spans[0]
    assert isinstance(tool, ToolExecutionSpan)
    assert tool.tool_call.name == "calculate_price"
    assert tool.tool_result.content == "4"


def test_tool_use_in_message():
    obs = create_mock_observation(
        "obs1",
        "chat",
        "chat",
        input_data=[
            {"role": "user", "content": '[{"text": "calc"}]'},
            {"role": "assistant", "content": '[{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {"x": 1}}}]'},
        ],
    )

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    msg = session.traces[0].spans[0].messages[1]
    assert msg.content[0].name == "calc"
    assert msg.content[0].tool_call_id == "t1"


def test_tool_result_in_message():
    obs = create_mock_observation(
        "obs1",
        "chat",
        "chat",
        input_data=[
            {"role": "tool", "content": '[{"toolResult": {"toolUseId": "t1", "content": [{"text": "result"}]}}]'}
        ],
    )

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    msg = session.traces[0].spans[0].messages[0]
    assert msg.content[0].content == "result"
    assert msg.content[0].tool_call_id == "t1"


def test_multiple_traces():
    obs1 = create_mock_observation(
        "obs1", "chat", "chat", input_data=[{"role": "assistant", "content": '[{"text": "a"}]'}]
    )
    obs2 = create_mock_observation(
        "obs2", "chat", "chat", input_data=[{"role": "assistant", "content": '[{"text": "b"}]'}]
    )

    trace1 = create_mock_trace("trace1", "session1", [obs1])
    trace2 = create_mock_trace("trace2", "session1", [obs2])

    session = LangfuseSessionMapper().map_to_session([trace1, trace2], "session1")

    assert len(session.traces) == 2


def test_get_agent_final_response():
    obs = create_mock_observation("obs1", "invoke_agent", "invoke_agent", output={"message": "final response"})

    trace = create_mock_trace("trace1", "session1", [obs])

    mapper = LangfuseSessionMapper()
    session = mapper.map_to_session([trace], "session1")

    final_response = mapper.get_agent_final_response(session)
    assert final_response == "final response"


def test_empty_observations():
    trace = create_mock_trace("trace1", "session1", [])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    assert len(session.traces) == 0


def test_invalid_json_handling():
    obs = create_mock_observation("obs1", "chat", "chat", input_data=[{"role": "user", "content": "invalid json"}])

    trace = create_mock_trace("trace1", "session1", [obs])

    session = LangfuseSessionMapper().map_to_session([trace], "session1")

    # Should handle gracefully and create message with empty text
    inference = session.traces[0].spans[0]
    assert isinstance(inference, InferenceSpan)
    assert len(inference.messages) == 1
    assert inference.messages[0].content[0].text == ""
