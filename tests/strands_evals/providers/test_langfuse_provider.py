"""Tests for LangfuseProvider — mocked Langfuse SDK."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.providers.exceptions import (
    ProviderError,
    SessionNotFoundError,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    ToolExecutionSpan,
)

# --- Helpers ---


def _meta(page=1, total_pages=1, total_items=10, limit=100):
    m = MagicMock(spec=["page", "limit", "total_items", "total_pages"])
    m.page, m.limit, m.total_items, m.total_pages = page, limit, total_items, total_pages
    return m


def _trace(trace_id, session_id, output=None):
    t = MagicMock()
    t.id, t.session_id, t.output = trace_id, session_id, output
    t.name, t.input, t.metadata = None, None, None
    t.timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return t


def _obs(
    obs_id,
    trace_id,
    obs_type,
    name=None,
    obs_input=None,
    obs_output=None,
    start_time=None,
    end_time=None,
    parent_observation_id=None,
    metadata=None,
    model=None,
):
    o = MagicMock()
    o.id, o.trace_id, o.type, o.name = obs_id, trace_id, obs_type, name
    o.start_time = start_time or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    o.end_time = end_time or datetime(2025, 1, 15, 10, 0, 5, tzinfo=timezone.utc)
    o.input, o.output = obs_input, obs_output
    o.parent_observation_id = parent_observation_id
    o.metadata, o.model = metadata or {}, model
    o.level, o.usage, o.usage_details = "DEFAULT", None, None
    return o


def _paginated(data, page=1, total_pages=1):
    r = MagicMock()
    r.data, r.meta = data, _meta(page=page, total_pages=total_pages, total_items=len(data))
    return r


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def provider(mock_client):
    with patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client):
        from strands_evals.providers.langfuse_provider import LangfuseProvider

        return LangfuseProvider(public_key="pk-test", secret_key="sk-test")


# --- Constructor ---


class TestConstructor:
    def test_explicit_credentials(self, mock_client):
        with patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client) as cls:
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider(public_key="pk-1", secret_key="sk-2", host="https://custom.langfuse.com")
            cls.assert_called_once_with(public_key="pk-1", secret_key="sk-2", host="https://custom.langfuse.com")

    def test_env_var_fallback(self, mock_client):
        env = {"LANGFUSE_PUBLIC_KEY": "pk-env", "LANGFUSE_SECRET_KEY": "sk-env"}
        # Remove LANGFUSE_HOST so we get the default
        clean_env = os.environ.copy()
        clean_env.pop("LANGFUSE_HOST", None)
        clean_env.update(env)
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client) as cls,
        ):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider()
            cls.assert_called_once_with(public_key="pk-env", secret_key="sk-env", host="https://us.cloud.langfuse.com")

    def test_host_env_var_fallback(self, mock_client):
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk-env",
            "LANGFUSE_SECRET_KEY": "sk-env",
            "LANGFUSE_HOST": "https://my-langfuse.example.com",
        }
        with (
            patch.dict(os.environ, env),
            patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client) as cls,
        ):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider()
            cls.assert_called_once_with(
                public_key="pk-env", secret_key="sk-env", host="https://my-langfuse.example.com"
            )

    def test_missing_credentials_raises(self):
        env = os.environ.copy()
        env.pop("LANGFUSE_PUBLIC_KEY", None)
        env.pop("LANGFUSE_SECRET_KEY", None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch("strands_evals.providers.langfuse_provider.Langfuse"),
            pytest.raises(ProviderError, match="Langfuse credentials"),
        ):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider()

    def test_default_host(self, mock_client):
        # Remove LANGFUSE_HOST so we get the default
        clean_env = os.environ.copy()
        clean_env.pop("LANGFUSE_HOST", None)
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client) as cls,
        ):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider(public_key="pk", secret_key="sk")
            assert cls.call_args[1]["host"] == "https://us.cloud.langfuse.com"

    def test_explicit_host_overrides_env(self, mock_client):
        env = {"LANGFUSE_HOST": "https://env-host.example.com"}
        with (
            patch.dict(os.environ, env),
            patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client) as cls,
        ):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            LangfuseProvider(public_key="pk", secret_key="sk", host="https://explicit.example.com")
            assert cls.call_args[1]["host"] == "https://explicit.example.com"


# --- get_evaluation_data ---


class TestGetEvaluationData:
    def test_happy_path(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs("o1", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ]
        )
        result = provider.get_evaluation_data("s1")
        assert isinstance(result["trajectory"], Session)
        assert result["trajectory"].session_id == "s1"
        assert len(result["trajectory"].traces) == 1

    def test_empty_session_raises(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([])
        with pytest.raises(SessionNotFoundError, match="s-missing"):
            provider.get_evaluation_data("s-missing")

    def test_output_from_last_agent_invocation(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs(
                    "o1",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q1"}],
                    obs_output="first",
                    start_time=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                ),
                _obs(
                    "o2",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q2"}],
                    obs_output="second",
                    start_time=datetime(2025, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
                ),
            ]
        )
        assert provider.get_evaluation_data("s1")["output"] == "second"

    def test_paginates_traces(self, provider, mock_client):
        mock_client.api.trace.list.side_effect = [
            _paginated([_trace("t1", "s1")], page=1, total_pages=2),
            _paginated([_trace("t2", "s1")], page=2, total_pages=2),
        ]
        mock_client.api.observations.get_many.side_effect = [
            _paginated([_obs("o1", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a")]),
            _paginated([_obs("o2", "t2", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="b")]),
        ]
        assert len(provider.get_evaluation_data("s1")["trajectory"].traces) == 2

    def test_paginates_observations(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.side_effect = [
            _paginated(
                [_obs("o1", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a")],
                page=1,
                total_pages=2,
            ),
            _paginated(
                [
                    _obs(
                        "o2",
                        "t1",
                        "GENERATION",
                        name="chat",
                        obs_input=[{"role": "user", "content": [{"text": "q"}]}],
                        obs_output={"role": "assistant", "content": [{"text": "a"}]},
                    )
                ],
                page=2,
                total_pages=2,
            ),
        ]
        assert len(provider.get_evaluation_data("s1")["trajectory"].traces[0].spans) == 2

    def test_wraps_api_error(self, provider, mock_client):
        mock_client.api.trace.list.side_effect = Exception("Connection refused")
        with pytest.raises(ProviderError, match="Connection refused"):
            provider.get_evaluation_data("s1")

    def test_unconvertible_observations_excluded(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs("o1", "t1", "EVENT", name="some_event"),
            ]
        )
        with pytest.raises(SessionNotFoundError):
            provider.get_evaluation_data("s1")


# --- Observation conversion ---


class TestConversion:
    def _get_spans(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")["trajectory"].traces[0].spans

    def test_generation_to_inference_span(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-gen",
                    "t1",
                    "GENERATION",
                    name="chat",
                    obs_input=[{"role": "user", "content": [{"text": "q"}]}],
                    obs_output={"role": "assistant", "content": [{"text": "a"}]},
                ),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        inference = [s for s in spans if isinstance(s, InferenceSpan)]
        assert len(inference) == 1
        assert inference[0].span_info.span_id == "o-gen"

    def test_execute_tool_to_tool_execution_span(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-tool",
                    "t1",
                    "SPAN",
                    name="execute_tool calc",
                    obs_input={"name": "calc", "arguments": {"x": "2+2"}, "toolUseId": "c1"},
                    obs_output={"result": "4", "status": "success"},
                ),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        tools = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert len(tools) == 1
        assert tools[0].tool_call.name == "calc"
        assert tools[0].tool_call.arguments == {"x": "2+2"}

    def test_invoke_agent_to_agent_invocation_span(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent my_agent",
                    obs_input=[{"text": "Hello"}],
                    obs_output="Hi there!",
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert len(agents) == 1
        assert agents[0].user_prompt == "Hello"
        assert agents[0].agent_response == "Hi there!"

    def test_unknown_type_skipped(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs("o-event", "t1", "EVENT", name="log"),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        assert len(spans) == 1
        assert isinstance(spans[0], AgentInvocationSpan)

    def test_unknown_span_name_skipped(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs("o-unk", "t1", "SPAN", name="some_other_op"),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        assert len(spans) == 1

    def test_span_info_populated(self, provider, mock_client):
        start = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, 12, 0, 10, tzinfo=timezone.utc)
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q"}],
                    obs_output="a",
                    start_time=start,
                    end_time=end,
                    parent_observation_id="o-parent",
                ),
            ],
        )
        si = spans[0].span_info
        assert si.trace_id == "t1"
        assert si.span_id == "o-agent"
        assert si.session_id == "s1"
        assert si.parent_span_id == "o-parent"
        assert si.start_time == start
        assert si.end_time == end

    def test_string_input_for_agent(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input="plain string prompt",
                    obs_output="response",
                ),
            ],
        )
        assert spans[0].user_prompt == "plain string prompt"

    def test_string_output_for_tool(self, provider, mock_client):
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-tool",
                    "t1",
                    "SPAN",
                    name="execute_tool calc",
                    obs_input={"name": "calc", "arguments": {"x": 1}},
                    obs_output="42",
                ),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        tools = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert tools[0].tool_result.content == "42"


# --- timeout and retry ---


class TestTimeoutAndRetry:
    def test_default_timeout_passed_to_api_calls(self, provider, mock_client):
        """API calls should pass request_options with default timeout."""
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ]
        )
        provider.get_evaluation_data("s1")

        # Both trace.list and observations.get_many should receive request_options
        trace_call_kwargs = mock_client.api.trace.list.call_args[1]
        assert trace_call_kwargs["request_options"] == {"timeout_in_seconds": 120}

        obs_call_kwargs = mock_client.api.observations.get_many.call_args[1]
        assert obs_call_kwargs["request_options"] == {"timeout_in_seconds": 120}

    def test_custom_timeout(self, mock_client):
        """Custom timeout should be passed through to API calls."""
        with patch("strands_evals.providers.langfuse_provider.Langfuse", return_value=mock_client):
            from strands_evals.providers.langfuse_provider import LangfuseProvider

            p = LangfuseProvider(public_key="pk", secret_key="sk", timeout=300)

        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ]
        )
        p.get_evaluation_data("s1")

        trace_call_kwargs = mock_client.api.trace.list.call_args[1]
        assert trace_call_kwargs["request_options"] == {"timeout_in_seconds": 300}

    def test_retries_on_timeout(self, provider, mock_client):
        """_fetch_all_pages should retry on timeout errors."""
        from httpx import ReadTimeout

        mock_client.api.trace.list.side_effect = [
            ReadTimeout("timed out"),
            _paginated([_trace("t1", "s1")]),
        ]
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ]
        )
        result = provider.get_evaluation_data("s1")
        assert result["trajectory"].session_id == "s1"
        assert mock_client.api.trace.list.call_count == 2

    def test_retries_exhaust_raises(self, provider, mock_client):
        """After max retries, the original error should propagate."""
        from httpx import ReadTimeout

        mock_client.api.trace.list.side_effect = ReadTimeout("timed out")
        with pytest.raises(ProviderError, match="timed out"):
            provider.get_evaluation_data("s1")


# --- _extract_agent_response edge cases ---


class TestExtractAgentResponse:
    def _get_spans(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")

    def test_message_finish_reason_dict(self, provider, mock_client):
        """When invoke_agent output is {'message': 'text', 'finish_reason': 'end_turn'}, extract message."""
        result = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q"}],
                    obs_output={"message": "Here is my response", "finish_reason": "end_turn"},
                ),
            ],
        )
        assert result["output"] == "Here is my response"

    def test_message_finish_reason_empty_message(self, provider, mock_client):
        """When invoke_agent output is {'message': '', 'finish_reason': 'tool_use'}, message is empty."""
        result = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q"}],
                    obs_output={"message": "", "finish_reason": "tool_use"},
                ),
            ],
        )
        # Empty message — output should be empty string from agent_response
        assert result["output"] == ""


class TestExtractOutputToolUse:
    def _get_result(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")

    def test_tool_use_ending_returns_empty(self, provider, mock_client):
        """When agent ended on tool_use, output is empty string."""
        result = self._get_result(
            provider,
            mock_client,
            [
                _obs(
                    "o-gen",
                    "t1",
                    "GENERATION",
                    name="chat",
                    obs_input=[{"role": "user", "content": [{"text": "hello"}]}],
                    obs_output={"role": "assistant", "content": [{"text": "I found the answer"}]},
                ),
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q"}],
                    obs_output={"message": "", "finish_reason": "tool_use"},
                ),
            ],
        )
        assert result["output"] == ""

    def test_nonempty_agent_response_used(self, provider, mock_client):
        """When agent_response has content, use it."""
        result = self._get_result(
            provider,
            mock_client,
            [
                _obs(
                    "o-gen",
                    "t1",
                    "GENERATION",
                    name="chat",
                    obs_input=[{"role": "user", "content": [{"text": "hello"}]}],
                    obs_output={"role": "assistant", "content": [{"text": "inference text"}]},
                ),
                _obs(
                    "o-agent",
                    "t1",
                    "SPAN",
                    name="invoke_agent a",
                    obs_input=[{"text": "q"}],
                    obs_output="agent says this",
                ),
            ],
        )
        assert result["output"] == "agent says this"


# --- LangChain framework support via obs.type routing ---


class TestLangChainToolType:
    """TOOL-type observations from LangChain via Langfuse."""

    def _get_spans(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")["trajectory"].traces[0].spans

    def test_tool_type_produces_tool_execution_span(self, provider, mock_client):
        """obs.type == 'TOOL' is routed to ToolExecutionSpan."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs("o-tool", "t1", "TOOL", name="add_numbers", obs_input={"a": 2, "b": 3}, obs_output="5"),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        tools = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert len(tools) == 1
        assert tools[0].tool_call.name == "add_numbers"
        assert tools[0].tool_call.arguments == {"a": 2, "b": 3}
        assert tools[0].tool_result.content == "5"

    def test_tool_type_with_dict_output(self, provider, mock_client):
        """TOOL with dict output parses result/status correctly."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-tool",
                    "t1",
                    "TOOL",
                    name="search",
                    obs_input={"query": "weather"},
                    obs_output={"result": "Sunny", "status": "success"},
                ),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        tools = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert tools[0].tool_result.content == "Sunny"
        assert tools[0].tool_result.error is None

    def test_tool_type_with_string_input(self, provider, mock_client):
        """TOOL with string input wraps it in a dict."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs("o-tool", "t1", "TOOL", name="echo", obs_input="hello", obs_output="hello"),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        tools = [s for s in spans if isinstance(s, ToolExecutionSpan)]
        assert tools[0].tool_call.name == "echo"
        assert tools[0].tool_call.arguments == {"input": "hello"}


class TestLangChainChainType:
    """CHAIN-type observations from LangChain via Langfuse."""

    def _get_spans(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")["trajectory"].traces[0].spans

    def test_root_chain_produces_agent_invocation(self, provider, mock_client):
        """Root CHAIN (parent_observation_id=None) → AgentInvocationSpan."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-chain",
                    "t1",
                    "CHAIN",
                    name="AgentExecutor",
                    obs_input={"input": "What is 2+2?"},
                    obs_output={"output": "4"},
                    parent_observation_id=None,
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert len(agents) == 1
        assert agents[0].user_prompt == "What is 2+2?"
        assert agents[0].agent_response == "4"

    def test_child_chain_is_skipped(self, provider, mock_client):
        """Non-root CHAIN (has parent) is skipped."""
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs(
                    "o-child",
                    "t1",
                    "CHAIN",
                    name="SubChain",
                    obs_input={"input": "sub"},
                    obs_output={"output": "sub-out"},
                    parent_observation_id="o-parent",
                ),
                _obs("o-agent", "t1", "SPAN", name="invoke_agent a", obs_input=[{"text": "q"}], obs_output="a"),
            ],
        )
        spans = provider.get_evaluation_data("s1")["trajectory"].traces[0].spans
        assert len(spans) == 1
        assert isinstance(spans[0], AgentInvocationSpan)

    def test_chain_with_messages_input(self, provider, mock_client):
        """CHAIN with LangChain messages-style input extracts human message."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-chain",
                    "t1",
                    "CHAIN",
                    name="LangGraph",
                    obs_input={"messages": [{"type": "human", "content": "Tell me a joke"}]},
                    obs_output={"output": "Why did the chicken cross the road?"},
                    parent_observation_id=None,
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert agents[0].user_prompt == "Tell me a joke"

    def test_chain_with_content_output(self, provider, mock_client):
        """CHAIN with content-style output."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o-chain",
                    "t1",
                    "CHAIN",
                    name="LangGraph",
                    obs_input={"input": "Hi"},
                    obs_output={"content": "Hello!"},
                    parent_observation_id=None,
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert agents[0].agent_response == "Hello!"


class TestLangChainEndToEnd:
    """Full LangChain agent trace: CHAIN + GENERATION + TOOL."""

    def test_full_langchain_trace(self, provider, mock_client):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(
            [
                _obs(
                    "o-chain",
                    "t1",
                    "CHAIN",
                    name="AgentExecutor",
                    obs_input={"input": "What is the weather?"},
                    obs_output={"output": "Sunny, 72F."},
                    parent_observation_id=None,
                ),
                _obs(
                    "o-gen",
                    "t1",
                    "GENERATION",
                    name="ChatOpenAI",
                    obs_input=[{"role": "user", "content": [{"text": "What is the weather?"}]}],
                    obs_output={"role": "assistant", "content": [{"text": "Let me check."}]},
                    parent_observation_id="o-chain",
                ),
                _obs(
                    "o-tool",
                    "t1",
                    "TOOL",
                    name="get_weather",
                    obs_input={"location": "SF"},
                    obs_output="Sunny, 72F",
                    parent_observation_id="o-chain",
                ),
            ],
        )
        result = provider.get_evaluation_data("s1")
        spans = result["trajectory"].traces[0].spans

        agent_spans = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        inference_spans = [s for s in spans if isinstance(s, InferenceSpan)]
        tool_spans = [s for s in spans if isinstance(s, ToolExecutionSpan)]

        assert len(agent_spans) == 1
        assert len(inference_spans) == 1
        assert len(tool_spans) == 1

        assert agent_spans[0].user_prompt == "What is the weather?"
        assert agent_spans[0].agent_response == "Sunny, 72F."
        assert tool_spans[0].tool_call.name == "get_weather"
        assert tool_spans[0].tool_result.content == "Sunny, 72F"
        assert result["output"] == "Sunny, 72F."


class TestStrandsOtelViaLangfuse:
    """Strands message-list format arriving via OTEL→Langfuse OTLP endpoint."""

    def _get_spans(self, provider, mock_client, observations):
        mock_client.api.trace.list.return_value = _paginated([_trace("t1", "s1")])
        mock_client.api.observations.get_many.return_value = _paginated(observations)
        return provider.get_evaluation_data("s1")["trajectory"].traces[0].spans

    def test_strands_message_list_input(self, provider, mock_client):
        """Strands invoke_agent with message-list input extracts user text."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o1",
                    "t1",
                    "SPAN",
                    name="invoke_agent my_agent",
                    obs_input=[{"role": "user", "content": '[{"text": "What\'s the weather?"}]'}],
                    obs_output="Sunny and warm!",
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert len(agents) == 1
        assert agents[0].user_prompt == "What's the weather?"

    def test_strands_message_list_input_plain_string_content(self, provider, mock_client):
        """Strands message-list where content is a plain string (not JSON)."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o1",
                    "t1",
                    "SPAN",
                    name="invoke_agent my_agent",
                    obs_input=[{"role": "user", "content": "Hello there"}],
                    obs_output="Hi!",
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert agents[0].user_prompt == "Hello there"

    def test_strands_message_list_output(self, provider, mock_client):
        """Strands invoke_agent with message-list output extracts assistant text."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o1",
                    "t1",
                    "SPAN",
                    name="invoke_agent my_agent",
                    obs_input=[{"text": "Hello"}],
                    obs_output=[{"role": "assistant", "content": '[{"text": "Hi there!"}]'}],
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert agents[0].agent_response == "Hi there!"

    def test_strands_message_list_content_as_list(self, provider, mock_client):
        """Strands message-list where content is already a parsed list."""
        spans = self._get_spans(
            provider,
            mock_client,
            [
                _obs(
                    "o1",
                    "t1",
                    "SPAN",
                    name="invoke_agent my_agent",
                    obs_input=[{"role": "user", "content": [{"text": "How are you?"}]}],
                    obs_output="I'm fine!",
                ),
            ],
        )
        agents = [s for s in spans if isinstance(s, AgentInvocationSpan)]
        assert agents[0].user_prompt == "How are you?"
