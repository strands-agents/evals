import asyncio
import itertools
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException, ModelThrottledException
from tenacity import wait_none

from strands_evals import Case, Experiment
from strands_evals.evaluators import ContextUtilizationEvaluator
from strands_evals.evaluators.context_utilization_evaluator import (
    _JUDGE_RETRY_WAIT,
    ContextUtilizationRating,
    _JudgeThrottledError,
)
from strands_evals.evaluators.context_utilization_evaluator import _estimate_tokens as chars_over_4
from strands_evals.types import EvaluationData
from strands_evals.types.evaluation import NOT_APPLICABLE
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    EvaluationLevel,
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
from strands_evals.utils import is_throttling_error

MODULE = "strands_evals.evaluators.context_utilization_evaluator"
MEASURED = {"search", "read"}
_CALL_IDS = itertools.count(1)  # every tool call gets its own id, as real traces record


@pytest.fixture
def one_chunk_per_line():
    """One chunk per line instead of the default windows, so expected token counts are easy to work out."""
    with patch.object(ContextUtilizationEvaluator, "_windows", lambda self, output: output.splitlines()):
        yield


# ---- Helpers ------------------------------------------------------------------------------


def _span_info(second: int, parent: str | None = None) -> SpanInfo:
    t = datetime(2026, 1, 1) + timedelta(seconds=second)
    return SpanInfo(session_id="s", start_time=t, end_time=t, parent_span_id=parent)


def _agent(second: int, prompt: str, response: str, parent: str | None = None) -> AgentInvocationSpan:
    return AgentInvocationSpan(
        span_info=_span_info(second, parent), user_prompt=prompt, agent_response=response, available_tools=[]
    )


def _tool(second: int, name: str, output: str, error: str | None = None) -> ToolExecutionSpan:
    return ToolExecutionSpan(
        span_info=_span_info(second),
        tool_call=ToolCall(name=name, arguments={"q": name}, tool_call_id=f"call-{next(_CALL_IDS)}"),
        tool_result=ToolResult(content=output, error=error),
    )


def _shell(second: int, command: str, output: str) -> ToolExecutionSpan:
    return ToolExecutionSpan(
        span_info=_span_info(second),
        tool_call=ToolCall(name="Bash", arguments={"command": command}, tool_call_id=f"call-{next(_CALL_IDS)}"),
        tool_result=ToolResult(content=output),
    )


def _session(*traces: list) -> Session:
    return Session(
        traces=[Trace(spans=spans, trace_id=f"t{i}", session_id="s") for i, spans in enumerate(traces)], session_id="s"
    )


def _case(session: Session, input=None, output=None) -> EvaluationData:
    return EvaluationData(input=input, actual_output=output, actual_trajectory=session)


def _two_turns(second_read: str = "conftest.py has make_client") -> Session:
    return _session(
        [_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "bug is in a.py")],
        [_agent(10, "Now add a test", "Added the test."), _tool(11, "read", second_read)],
    )


@pytest.fixture
def evaluation_data():
    spans = [
        _agent(0, "Fix the bug", "Fixed it."),
        InferenceSpan(
            span_info=_span_info(1),
            messages=[
                UserMessage(content=[TextContent(text="Fix the bug")]),
                AssistantMessage(content=[TextContent(text="Let me search first.")]),
            ],
        ),
        _tool(2, "search", "alpha beta\ngamma\n\ndelta epsilon zeta"),
        _tool(3, "read", "alpha beta\neta theta"),
        _tool(4, "edit", "file written"),
    ]
    return EvaluationData(input="Fix the bug", actual_output="Fixed it.", actual_trajectory=_session(spans))


@pytest.fixture(autouse=True)
def word_tokens():
    """One token per word, so the expected numbers can be worked out by hand."""
    with patch(f"{MODULE}._estimate_tokens", side_effect=lambda text: len(text.split())):
        yield


def _render(evaluator, case):
    """All of a case's judge prompts joined, every chunk, and the assistant reply count."""
    turns, assistant_reply_count = evaluator._render(case, case.actual_trajectory)
    pieces = [piece for turn in turns for piece in turn.pieces]
    return "\n\n".join(turn.prompt for turn in turns), pieces, assistant_reply_count


# What each mocked judge call reports as its token usage.
USAGE = {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110}


def _mock_judge(mock_agent_class, ids, reasoning="used some"):
    mock_result = _verdict(*ids, reasoning=reasoning)
    mock_agent = Mock(return_value=mock_result)
    mock_agent.invoke_async = AsyncMock(return_value=mock_result)
    mock_agent_class.return_value = mock_agent
    return mock_agent


def _verdict(*ids, reasoning="used some"):
    return Mock(
        structured_output=ContextUtilizationRating(reasoning=reasoning, useful_ids=list(ids)),
        metrics=Mock(accumulated_usage=dict(USAGE)),
    )


# ---- Construction and persistence ---------------------------------------------------------


def test_init_with_defaults():
    evaluator = ContextUtilizationEvaluator()

    assert evaluator.measured_tools is None
    assert evaluator.chunk_size_tokens == 300
    assert evaluator.max_judge_input_tokens == 900_000
    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt
    assert evaluator.evaluation_level == EvaluationLevel.SESSION_LEVEL


@patch(f"{MODULE}.Agent")
def test_judge_gets_the_system_prompt(mock_agent_class, evaluation_data):
    _mock_judge(mock_agent_class, [])

    ContextUtilizationEvaluator(measured_tools=MEASURED, system_prompt="Custom judge").evaluate(evaluation_data)

    assert mock_agent_class.call_args.kwargs["system_prompt"] == "Custom judge"


def test_chunk_size_tokens_and_max_judge_input_tokens_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size_tokens"):
        ContextUtilizationEvaluator(chunk_size_tokens=0)
    with pytest.raises(ValueError, match="max_judge_input_tokens"):
        ContextUtilizationEvaluator(max_judge_input_tokens=0)


def test_positional_arguments_follow_the_other_judge_evaluators():
    evaluator = ContextUtilizationEvaluator("v0", "my-model", "Custom judge", "cu")

    assert (evaluator.version, evaluator.model, evaluator.system_prompt) == ("v0", "my-model", "Custom judge")
    assert evaluator.get_name() == "cu"
    with pytest.raises(TypeError):
        ContextUtilizationEvaluator("v0", None, None, None, {"search"})  # measured_tools is keyword-only


def test_experiment_with_evaluator_round_trips_through_a_file(tmp_path):
    evaluator = ContextUtilizationEvaluator(measured_tools={"search", "read"}, chunk_size_tokens=50)
    path = tmp_path / "experiment.json"

    Experiment(cases=[Case(input="q")], evaluators=[evaluator]).to_file(path)
    [reloaded] = Experiment.from_file(path).evaluators

    assert isinstance(reloaded, ContextUtilizationEvaluator)
    assert reloaded.measured_tools == ["read", "search"]
    assert reloaded.chunk_size_tokens == 50


# ---- End to end through evaluate() --------------------------------------------------------


@patch(f"{MODULE}.Agent")
def test_the_judge_gets_exactly_this_prompt(mock_agent_class, evaluation_data):
    judge = _mock_judge(mock_agent_class, ["S1.1"])

    ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    assert judge.call_args.args[0] == (
        "<task>\nFix the bug\n</task>\n\n"
        "<run>\n"
        "[assistant] Let me search first.\n"
        '[call S1] search({"q": "search"})\n'
        "  <S1.1>\n    alpha beta\n    gamma\n    \n    delta epsilon zeta\n"
        '[call S2] read({"q": "read"})\n'
        "  <S2.1>\n    alpha beta\n    eta theta\n"
        '[call, not measured] edit({"q": "edit"})\n'
        "  output: file written\n"
        "</run>\n\n"
        "<final_response>\nFixed it.\n</final_response>\n\n"
        "List the ids of the useful pieces."
    )


@patch(f"{MODULE}.Agent")
def test_a_scored_row_has_exactly_this_metadata(mock_agent_class, evaluation_data):
    _mock_judge(mock_agent_class, ["S1.1", "S9.9"])
    search_id, read_id, _ = (
        span.tool_call.tool_call_id
        for span in evaluation_data.actual_trajectory.traces[0].spans
        if isinstance(span, ToolExecutionSpan)
    )

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    assert (result.score, result.test_pass, result.reason) == (pytest.approx(0.6), True, "used some")
    assert result.metadata == {
        "chunk_size_tokens": 300,
        "gathered_tokens": 10,
        "repeat_tokens": 0,
        "useful_tokens": 6,
        "per_tool": {
            "search": {
                "calls": 1,
                "gathered_tokens": 6,
                "repeat_tokens": 0,
                "useful_tokens": 6,
                "utilization": 1.0,
                "share_of_useful": 1.0,
            },
            "read": {
                "calls": 1,
                "gathered_tokens": 4,
                "repeat_tokens": 0,
                "useful_tokens": 0,
                "utilization": 0.0,
                "share_of_useful": 0.0,
            },
        },
        "per_call": [
            {
                "source": "S1",
                "tool_call_id": search_id,
                "tool": "search",
                "chunks": 1,
                "gathered_tokens": 6,
                "repeat_tokens": 0,
                "useful_tokens": 6,
            },
            {
                "source": "S2",
                "tool_call_id": read_id,
                "tool": "read",
                "chunks": 1,
                "gathered_tokens": 4,
                "repeat_tokens": 0,
                "useful_tokens": 0,
            },
        ],
        "useful_ids": ["S1.1"],
        "dropped_ids": ["S9.9"],
        "judge_calls": 1,
        "judge_input_tokens": 100,
        "judge_output_tokens": 10,
        "assistant_replies": 1,
    }


# ---- Chunking -----------------------------------------------------------------------------


def test_short_outputs_are_one_chunk_each_by_default(evaluation_data):
    prompt, pieces, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), evaluation_data)

    assert [(piece.uid, piece.text) for piece in pieces] == [
        ("S1.1", "alpha beta\ngamma\n\ndelta epsilon zeta"),
        ("S2.1", "alpha beta\neta theta"),
    ]
    assert "  <S1.1>\n    alpha beta\n    gamma" in prompt


def test_long_output_is_split_into_windows_of_whole_lines():
    output = "\n".join(f"line {n:03d}" for n in range(120))  # each line: 8 chars, 2 tokens at chars / 4
    case = _case(_session([_tool(1, "read", output)]))

    _, pieces, _ = _render(ContextUtilizationEvaluator(chunk_size_tokens=20), case)

    assert [len(piece.text.splitlines()) for piece in pieces] == [10] * 12
    assert "\n".join(piece.text for piece in pieces) == output


@patch(f"{MODULE}.Agent")
def test_the_chunk_size_is_recorded_on_each_scored_row(mock_agent_class, evaluation_data):
    _mock_judge(mock_agent_class, ["S1.1"])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED, chunk_size_tokens=50).evaluate(evaluation_data)

    assert result.metadata["chunk_size_tokens"] == 50


def test_a_line_longer_than_the_window_is_split_by_characters():
    output = "short line\n" + "x" * 250 + "\nlast line"
    case = _case(_session([_tool(1, "read", output)]))

    with patch(f"{MODULE}._estimate_tokens", chars_over_4):  # sizes in characters / 4, not words
        _, pieces, _ = _render(ContextUtilizationEvaluator(chunk_size_tokens=25), case)

    assert [piece.text for piece in pieces] == ["short line", "x" * 84, "x" * 84, "x" * 82, "last line"]


def test_chunk_sizes_and_reported_counts_use_the_same_estimate():
    output = "\n".join("y" * 39 for _ in range(60))  # each line: 40 characters with its newline, 10 tokens
    case = _case(_session([_tool(1, "read", output)]))

    with patch(f"{MODULE}._estimate_tokens", chars_over_4):
        _, pieces, _ = _render(ContextUtilizationEvaluator(chunk_size_tokens=300), case)

    assert [len(piece.text.splitlines()) for piece in pieces] == [30, 30]
    assert [piece.tokens for piece in pieces] == [299, 299]  # a "300-token" chunk reports as about 300


def test_inline_images_are_replaced_by_a_placeholder():
    image = ToolExecutionSpan(
        span_info=_span_info(1),
        tool_call=ToolCall(name="view_image", arguments={"path": "chart.png"}, tool_call_id=f"call-{next(_CALL_IDS)}"),
        tool_result=ToolResult(content="chart: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA== done"),
    )

    prompt, [piece], _ = _render(ContextUtilizationEvaluator(), _case(_session([image])))

    assert piece.text == "chart: [image] done"
    assert "base64" not in prompt


def test_failed_call_error_is_shown_but_not_measured(evaluation_data):
    failed = ToolExecutionSpan(
        span_info=_span_info(5),
        tool_call=ToolCall(name="read", arguments={"q": "missing"}, tool_call_id=f"call-{next(_CALL_IDS)}"),
        tool_result=ToolResult(content="", error="FileNotFoundError: missing"),
    )
    evaluation_data.actual_trajectory.traces[0].spans.append(failed)

    prompt, pieces, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), evaluation_data)

    assert "[call, failed]" in prompt and "FileNotFoundError: missing" in prompt
    assert [piece.uid for piece in pieces] == ["S1.1", "S2.1"]


@patch(f"{MODULE}.Agent")
def test_a_call_with_error_set_is_failed_even_when_it_has_output(mock_agent_class):
    # As StrandsInMemorySessionMapper (and the OpenInference and LangChain mappers) record a failed
    # tool: a status in `error`, the error text in `content`.
    error_text = "Error: FileNotFoundError - [Errno 2] No such file or directory: '/missing' " * 20
    session = _session(
        [
            _agent(0, "Read the config", "The config sets retries to 3."),
            _tool(1, "read", error_text, error="error"),
            _tool(2, "read", "retries: 3"),
        ]
    )
    _mock_judge(mock_agent_class, ["S1.*"])  # the successful read, fully used

    [result] = ContextUtilizationEvaluator().evaluate(_case(session))

    assert result.score == 1.0
    assert [(row["source"], row["gathered_tokens"]) for row in result.metadata["per_call"]] == [("S1", 2)]
    prompt = mock_agent_class.return_value.call_args.args[0]
    assert "[call, failed]" in prompt and "error: error" in prompt and "No such file" in prompt  # shown, not tagged


def test_output_before_an_error_is_shown_but_not_measured():
    case = _case(_session([_tool(1, "read", "first half", error="timed out"), _tool(2, "read", "other file")]))

    prompt, pieces, _ = _render(ContextUtilizationEvaluator(), case)

    assert [(piece.uid, piece.text) for piece in pieces] == [("S1.1", "other file")]
    assert "[call, failed]" in prompt and "error: timed out\n  output: first half" in prompt


# ---- Rendering ----------------------------------------------------------------------------


def test_prompt_shows_the_run_and_tags_only_measured_tools(evaluation_data):
    prompt, _, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), evaluation_data)

    assert "<task>\nFix the bug\n</task>" in prompt
    assert "[assistant] Let me search first." in prompt
    assert '[call S1] search({"q": "search"})' in prompt
    assert "[call, not measured] edit" in prompt
    assert "<final_response>\nFixed it.\n</final_response>" in prompt
    assert prompt.endswith("List the ids of the useful pieces.")


def test_the_trace_request_and_answer_win_over_the_case_input_and_output():
    # Case.input need not be the prompt, e.g. a dataset row.
    case = _case(_two_turns(), input={"session_id": "row-7"}, output="Answer from the case")

    turns, _ = ContextUtilizationEvaluator(measured_tools=MEASURED)._render(case, case.actual_trajectory)

    first, second = (turn.prompt for turn in turns)
    assert "<task>\nFix the bug\n</task>" in first and "row-7" not in first
    assert "<final_response>\nAdded the test.\n</final_response>" in second
    assert "Answer from the case" not in first + second


def test_the_case_input_and_output_fill_in_a_trace_without_them():
    session = _session([_tool(1, "read", "bug is in a.py")])  # no AgentInvocationSpan
    case = _case(session, input="Task from the case", output="Answer from the case")

    [turn], _ = ContextUtilizationEvaluator()._render(case, session)

    assert "<task>\nTask from the case\n</task>" in turn.prompt
    assert "<final_response>\nAnswer from the case\n</final_response>" in turn.prompt


def test_assistant_text_survives_a_json_round_trip(evaluation_data):
    # A text-only AssistantMessage deserializes as UserMessage(role=assistant); the text must survive.
    session = Session.model_validate_json(evaluation_data.actual_trajectory.model_dump_json())

    prompt, _, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), _case(session, input="Fix the bug"))

    assert "[assistant] Let me search first." in prompt


def test_repeated_assistant_replies_are_shown_and_counted_once(evaluation_data):
    reply = InferenceSpan(
        span_info=_span_info(1), messages=[AssistantMessage(content=[TextContent(text="Let me search first.")])]
    )
    evaluation_data.actual_trajectory.traces[0].spans.append(reply)  # the same reply, emitted twice

    prompt, _, assistant_reply_count = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), evaluation_data)

    assert prompt.count("[assistant] Let me search first.") == 1
    assert assistant_reply_count == 1


def test_a_second_request_and_sub_agent_tasks_in_one_trace_are_shown_where_made():
    spans = [
        _agent(0, "Fix the bug", "Fixed it."),
        _tool(1, "read", "bug is in a.py"),
        _agent(2, "Now add a test", "Added the test."),
        _agent(3, "Find the test helpers", "They are in conftest.py.", parent="tool-span"),
        _tool(4, "read", "conftest.py has make_client"),
    ]

    prompt, _, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), _case(_session(spans)))

    assert "<task>\nFix the bug\n</task>" in prompt
    assert prompt.index("bug is in a.py") < prompt.index("[user] Now add a test") < prompt.index("conftest.py has")
    assert "[sub-agent task] Find the test helpers" in prompt
    assert "<final_response>\nAdded the test.\n</final_response>" in prompt


def test_a_root_agent_with_a_parent_outside_the_trace_is_the_turn_request():
    # Some sources keep the root agent's parent (e.g. a server span) on every agent span.
    session = _session(
        [_agent(0, "Fix the bug", "Fixed it.", parent="srv"), _tool(1, "read", "bug is in a.py")],
        [_agent(10, "Now add a test", "Added the test.", parent="srv"), _tool(11, "read", "conftest.py")],
    )

    turns, _ = ContextUtilizationEvaluator()._render(_case(session), session)

    first, second = (turn.prompt for turn in turns)
    assert "<task>\nNow add a test\n</task>" in second and "[sub-agent task]" not in second
    assert "<final_response>\nAdded the test.\n</final_response>" in second
    assert "[later request 1] Now add a test\n[answer 1] Added the test." in first


def test_two_requests_in_one_trace_with_a_shared_outside_parent_are_both_requests():
    spans = [
        _agent(0, "Fix the bug", "Fixed it.", parent="srv"),
        _tool(1, "read", "bug is in a.py"),
        _agent(2, "Add a test", "Added it.", parent="srv"),
        _agent(3, "Find helpers", "conftest.py", parent="tool-span"),
    ]

    prompt, _, _ = _render(ContextUtilizationEvaluator(), _case(_session(spans)))

    assert "<task>\nFix the bug\n</task>" in prompt
    assert "[user] Add a test" in prompt and "[sub-agent task] Find helpers" in prompt


def test_the_same_reply_in_two_turns_is_shown_in_each():
    def done(second):
        return InferenceSpan(
            span_info=_span_info(second), messages=[AssistantMessage(content=[TextContent(text="Done.")])]
        )

    session = _session(
        [_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "bug is in a.py"), done(2)],
        [_agent(10, "Now add a test", "Added it."), _tool(11, "read", "conftest.py"), done(12)],
    )

    turns, assistant_reply_count = ContextUtilizationEvaluator()._render(_case(session), session)

    assert all("[assistant] Done." in turn.prompt for turn in turns)
    assert assistant_reply_count == 2


def test_the_first_answer_is_shown_when_a_second_request_follows_in_one_trace():
    spans = [_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "bug is in a.py"), _agent(2, "Add a test", "")]

    prompt, _, _ = _render(ContextUtilizationEvaluator(), _case(_session(spans)))

    assert prompt.index("[answer] Fixed it.") < prompt.index("[user] Add a test")


def test_a_first_request_with_no_text_keeps_its_answer():
    spans = [_agent(0, "", "Fixed it."), _tool(1, "read", "bug is in a.py"), _agent(2, "Add a test", "Added it.")]

    prompt, _, _ = _render(ContextUtilizationEvaluator(), _case(_session(spans)))

    assert "[answer] Fixed it." in prompt and "<final_response>\nAdded it.\n</final_response>" in prompt


def test_spans_with_naive_and_aware_timestamps_sort_together(evaluation_data):
    aware = datetime(2026, 1, 1, 0, 0, 9, tzinfo=timezone.utc)
    late = ToolExecutionSpan(
        span_info=SpanInfo(session_id="s", start_time=aware, end_time=aware),
        tool_call=ToolCall(name="read", arguments={}, tool_call_id=f"call-{next(_CALL_IDS)}"),
        tool_result=ToolResult(content="late read"),
    )
    evaluation_data.actual_trajectory.traces[0].spans.insert(0, late)

    _, pieces, _ = _render(ContextUtilizationEvaluator(measured_tools=MEASURED), evaluation_data)

    assert pieces[-1].text == "late read"


# ---- Judge ids ----------------------------------------------------------------------------


@pytest.mark.parametrize("whole_source", ["S1", "S1.*"])
@patch(f"{MODULE}.Agent")
def test_whole_source_ids_are_accepted_and_unknown_ids_logged(
    mock_agent_class, whole_source, evaluation_data, caplog, one_chunk_per_line
):
    _mock_judge(mock_agent_class, [whole_source, "S2.2", "S9.1"])

    with caplog.at_level("WARNING", logger=MODULE):
        [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    # S1 = all of source 1 (6 tokens); S2.2 = "eta theta" (2 tokens); S9.1 matches nothing.
    assert result.score == pytest.approx(8 / 10)
    assert result.metadata["useful_ids"] == ["S1.1", "S1.2", "S1.3", "S2.2"]
    assert "S9.1" in caplog.text


@patch(f"{MODULE}.Agent")
def test_malformed_ids_are_dropped_and_reported(mock_agent_class, evaluation_data, caplog, one_chunk_per_line):
    _mock_judge(mock_agent_class, [" S2.2 ", "s1.1", "(S1.1)", "S1.2.", "S1."])

    with caplog.at_level("WARNING", logger=MODULE):
        [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    # Split on commas and whitespace; only `<>` and a trailing `.*` are stripped, nothing else.
    assert result.metadata["useful_ids"] == ["S2.2"]
    assert result.metadata["dropped_ids"] == ["s1.1", "(S1.1)", "S1.2.", "S1."]
    assert "'s1.1'" in caplog.text and "'S1.'" in caplog.text


@pytest.mark.parametrize("packed", [["S1.1, S1.2"], ["S1.1 S1.2"], ["S1.1,S1.2"], ["<S1.1>, S1.2"]])
@patch(f"{MODULE}.Agent")
def test_several_ids_in_one_string_are_each_accepted(mock_agent_class, packed, one_chunk_per_line):
    _mock_judge(mock_agent_class, packed)
    session = _session([_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "alpha beta\ngamma delta")])

    [result] = ContextUtilizationEvaluator().evaluate(_case(session))

    assert result.metadata["useful_ids"] == ["S1.1", "S1.2"]
    assert result.metadata["dropped_ids"] == []
    assert result.score == 1.0


@patch(f"{MODULE}.Agent")
def test_ids_in_the_tagged_form_are_accepted(mock_agent_class, one_chunk_per_line):
    # The run shows chunks as <S1.1>; a judge that answers in that form still gets its ids matched.
    _mock_judge(mock_agent_class, ["<S1.1>", "<S1.2>"])
    session = _session([_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "alpha beta\ngamma delta")])

    [result] = ContextUtilizationEvaluator().evaluate(_case(session))

    assert result.score == 1.0
    assert result.metadata["useful_ids"] == ["S1.1", "S1.2"]
    assert result.metadata["dropped_ids"] == []


def test_the_prompt_asks_for_ids_without_angle_brackets():
    assert "without the angle brackets: S3.2 for the piece tagged <S3.2>" in ContextUtilizationEvaluator().system_prompt


@patch(f"{MODULE}.Agent")
def test_a_source_id_does_not_match_a_longer_source_number(mock_agent_class):
    session = _session([_tool(n, "read", f"output {n}") for n in range(1, 11)])
    _mock_judge(mock_agent_class, ["S1"])

    [result] = ContextUtilizationEvaluator().evaluate(_case(session))

    assert result.metadata["useful_ids"] == ["S1.1"]


# ---- Scoring ------------------------------------------------------------------------------


@patch(f"{MODULE}.Agent")
def test_repeats_stay_in_the_denominator_and_useful_content_counts_once(
    mock_agent_class, evaluation_data, one_chunk_per_line
):
    # S2.1 ("alpha beta") repeats S1.1: both copies are gathered, but the content is useful once.
    _mock_judge(mock_agent_class, ["S1.1", "S2.1", "S2.2"])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    assert result.metadata["useful_ids"] == ["S1.1", "S2.1", "S2.2"]  # the repeat was accepted, not dropped
    assert result.metadata["dropped_ids"] == []
    assert result.score == pytest.approx(4 / 10)
    assert result.test_pass is True  # no pass/fail threshold
    assert result.reason == "used some"
    assert (result.metadata["gathered_tokens"], result.metadata["repeat_tokens"]) == (10, 2)
    assert result.metadata["useful_tokens"] == 4  # not 6: S2.1's content was already counted with S1.1
    assert result.metadata["assistant_replies"] == 1
    search, read = result.metadata["per_tool"]["search"], result.metadata["per_tool"]["read"]
    assert (search["gathered_tokens"], search["useful_tokens"], search["utilization"]) == (6, 2, pytest.approx(2 / 6))
    assert (read["gathered_tokens"], read["repeat_tokens"], read["useful_tokens"]) == (4, 2, 2)
    assert search["share_of_useful"] == read["share_of_useful"] == 0.5


@patch(f"{MODULE}.Agent")
def test_per_call_rows_add_up_to_per_tool_rows_and_totals(mock_agent_class, evaluation_data, one_chunk_per_line):
    _mock_judge(mock_agent_class, ["S1.1", "S2.1", "S2.2"])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    per_call = result.metadata["per_call"]
    assert [(c["source"], c["tool"], c["chunks"]) for c in per_call] == [("S1", "search", 3), ("S2", "read", 2)]
    assert [(c["gathered_tokens"], c["repeat_tokens"], c["useful_tokens"]) for c in per_call] == [(6, 0, 2), (4, 2, 2)]
    for key in ("gathered_tokens", "repeat_tokens", "useful_tokens"):
        assert sum(c[key] for c in per_call) == result.metadata[key]
        for call in per_call:
            assert call[key] <= result.metadata["per_tool"][call["tool"]][key]


@patch(f"{MODULE}.Agent")
def test_a_useful_chunk_counts_all_its_tokens(mock_agent_class, evaluation_data):
    _mock_judge(mock_agent_class, ["S2.1"])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    # The read chunk (4 tokens) is used in full; the search chunks (6 tokens) are not.
    assert result.score == pytest.approx(4 / 10)
    assert result.metadata["repeat_tokens"] == 0


@patch(f"{MODULE}.Agent")
def test_share_of_useful_is_each_tools_part_of_the_useful_tokens(mock_agent_class, evaluation_data, one_chunk_per_line):
    _mock_judge(mock_agent_class, ["S1.1", "S1.3", "S2.2"])  # search: 2 + 3 tokens, read: 2

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    per_tool = result.metadata["per_tool"]
    assert per_tool["search"]["share_of_useful"] == pytest.approx(5 / 7)
    assert per_tool["read"]["share_of_useful"] == pytest.approx(2 / 7)


@patch(f"{MODULE}.Agent")
def test_real_token_estimates_give_a_score_between_zero_and_one(mock_agent_class, evaluation_data):
    _mock_judge(mock_agent_class, ["S1.1"])

    with patch(f"{MODULE}._estimate_tokens", chars_over_4):  # undo the word-count fixture
        [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    gathered, useful = result.metadata["gathered_tokens"], result.metadata["useful_tokens"]
    assert 0 < useful < gathered
    assert result.score == pytest.approx(useful / gathered)


# ---- Multi-turn sessions ------------------------------------------------------------------


def test_each_turn_gets_its_own_prompt_with_later_turns():
    case = _case(_two_turns())

    turns, _ = ContextUtilizationEvaluator(measured_tools=MEASURED)._render(case, case.actual_trajectory)

    assert [turn.number for turn in turns] == [1, 2]
    assert [[piece.uid for piece in turn.pieces] for turn in turns] == [["S1.1"], ["S2.1"]]
    first, second = (turn.prompt for turn in turns)
    assert "<task>\nFix the bug\n</task>" in first and "conftest.py" not in first.split("<later_turns>")[0]
    assert "[later request 1] Now add a test" in first and "[answer 1] Added the test." in first
    assert "<task>\nNow add a test\n</task>" in second and "<later_turns>" not in second
    assert "<final_response>\nAdded the test.\n</final_response>" in second


def test_later_turns_stop_at_the_token_budget():
    session = _session(
        [_agent(0, "First", "One."), _tool(1, "read", "first read")],
        [_agent(10, "Second", "Two.")],
        [_agent(20, "Third", "Three.")],
    )
    case = _case(session)

    # Each later-turn entry is 6 words here; a budget of 10 fits only the nearest one.
    with patch(f"{MODULE}._LATER_TURNS_TOKENS", 10):
        [turn], _ = ContextUtilizationEvaluator()._render(case, case.actual_trajectory)

    assert "[later request 1] Second" in turn.prompt
    assert "Third" not in turn.prompt


def test_turns_keep_their_session_position_when_an_earlier_turn_gathered_nothing():
    session = _session([_agent(0, "Hi", "Hello.")], [_agent(10, "Fix the bug", "Fixed it."), _tool(11, "read", "a.py")])

    turns, _ = ContextUtilizationEvaluator()._render(_case(session), session)

    assert [turn.number for turn in turns] == [2]


@patch(f"{MODULE}.Agent")
def test_turn_verdicts_are_combined_and_limited_to_their_own_chunks(mock_agent_class):
    mock_agent_class.return_value = Mock(
        side_effect=[_verdict("S1.1", "S2.1", reasoning="used the bug location"), _verdict(reasoning="ignored it")]
    )

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))

    # Turn 1 may only mark S1.1, so its "S2.1" is dropped; turn 2 marked nothing.
    assert result.metadata["useful_ids"] == ["S1.1"]
    assert result.metadata["judge_calls"] == 2
    assert result.score == pytest.approx(4 / 7)
    assert result.reason == "Turn 1: used the bug location\n\nTurn 2: ignored it"


@patch(f"{MODULE}.Agent")
def test_content_repeated_in_a_later_turn_is_useful_once(mock_agent_class):
    mock_agent_class.return_value = Mock(side_effect=[_verdict("S1.1"), _verdict("S2.1")])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(
        _case(_two_turns(second_read="bug is in a.py"))
    )

    assert (result.metadata["gathered_tokens"], result.metadata["repeat_tokens"], result.metadata["useful_tokens"]) == (
        8,
        4,
        4,
    )
    assert result.score == pytest.approx(0.5)


@patch(f"{MODULE}.Agent")
async def test_async_starts_turns_in_session_order_and_matches_their_verdicts(mock_agent_class):
    prompts = []

    async def judge(prompt, **kwargs):
        prompts.append(prompt)
        return _verdict("S1.1") if prompt.startswith("<task>\nFix the bug") else _verdict(reasoning="ignored")

    mock_agent_class.return_value = Mock(invoke_async=judge)

    [result] = await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    assert [p.startswith("<task>\nFix the bug") for p in prompts] == [True, False]
    assert result.metadata["useful_ids"] == ["S1.1"]
    assert result.score == pytest.approx(4 / 7)


# ---- Not applicable and errors ------------------------------------------------------------


@patch(f"{MODULE}.Agent")
def test_no_trajectory_is_not_applicable(mock_agent_class):
    [result] = ContextUtilizationEvaluator().evaluate(EvaluationData(input="q", actual_output="a"))

    assert result.label == NOT_APPLICABLE
    assert result.test_pass is False
    mock_agent_class.assert_not_called()


@patch(f"{MODULE}.Agent")
def test_nothing_gathered_is_not_applicable(mock_agent_class):
    session = _session([_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "", error="error")])

    [result] = ContextUtilizationEvaluator(measured_tools={"read"}).evaluate(_case(session))

    assert result.label == NOT_APPLICABLE
    assert result.test_pass is True
    assert result.reason == "no measured tool output"
    mock_agent_class.assert_not_called()


@patch(f"{MODULE}.Agent")
def test_measured_tools_that_were_never_called_are_named_in_the_reason(mock_agent_class, evaluation_data, caplog):
    with caplog.at_level("WARNING", logger=MODULE):
        [result] = ContextUtilizationEvaluator(measured_tools=["read_flie"]).evaluate(evaluation_data)

    assert result.label == NOT_APPLICABLE
    assert result.reason == (
        "no measured tool output: none of measured_tools ['read_flie'] was called (called: ['edit', 'read', 'search'])"
    )
    assert "none of the measured tools was called" in caplog.text
    mock_agent_class.assert_not_called()


def test_measured_tools_as_a_single_string_is_rejected():
    with pytest.raises(TypeError, match=r"\['search_docs'\]"):
        ContextUtilizationEvaluator(measured_tools="search_docs")


@pytest.mark.parametrize("ids", [(None, "call-2"), ("", "call-2"), ("call-1", "call-1")])
@patch(f"{MODULE}.Agent")
def test_calls_without_a_unique_tool_call_id_are_scored_with_a_warning(mock_agent_class, ids, caplog):
    spans = [_agent(0, "Fix the bug", "Fixed it."), _tool(1, "read", "bug is in a.py"), _tool(2, "read", "a.py")]
    for span, tool_call_id in zip(spans[1:], ids, strict=True):
        span.tool_call.tool_call_id = tool_call_id
    _mock_judge(mock_agent_class, ["S1.1"])

    with caplog.at_level("WARNING", logger=MODULE):
        [result] = ContextUtilizationEvaluator().evaluate(_case(_session(spans)))

    assert result.label != NOT_APPLICABLE  # source ids tell the calls apart, so scoring is unaffected
    assert [row["source"] for row in result.metadata["per_call"]] == ["S1", "S2"]
    assert result.score == pytest.approx(4 / 5)
    assert "without a unique tool_call_id" in caplog.text


@patch(f"{MODULE}.Agent")
def test_session_without_assistant_text_is_scored_with_a_warning(mock_agent_class, evaluation_data, caplog):
    # E.g. the Claude Agent SDK traced with OpenInference: tool spans but no inference spans.
    spans = [span for span in evaluation_data.actual_trajectory.traces[0].spans if not isinstance(span, InferenceSpan)]
    evaluation_data.actual_trajectory = _session(spans)
    _mock_judge(mock_agent_class, ["S1.1"])

    with caplog.at_level("WARNING", logger=MODULE):
        [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(evaluation_data)

    assert result.metadata["assistant_replies"] == 0
    assert "no assistant text between tool calls" in caplog.text


@patch(f"{MODULE}.Agent")
def test_turn_too_long_for_the_judge_is_not_applicable(mock_agent_class):
    mock_agent_class.return_value = Mock(side_effect=[_verdict("S1.1"), ContextWindowOverflowException("too long")])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))

    assert result.label == NOT_APPLICABLE
    assert result.test_pass is False
    assert result.reason == "turn 2 is too long for one judge call"
    # Turn 1's call is paid for; turn 2's failed.
    assert result.metadata == {
        "too_long_turn": 2,
        "judge_calls": 2,
        "judge_input_tokens": 100,
        "judge_output_tokens": 10,
    }


@patch(f"{MODULE}.Agent")
async def test_a_turn_estimated_too_long_is_found_before_any_judge_call(mock_agent_class):
    session = _two_turns(second_read="word " * 2_000)

    [result] = ContextUtilizationEvaluator(max_judge_input_tokens=1_000).evaluate(_case(session))
    [async_result] = await ContextUtilizationEvaluator(max_judge_input_tokens=1_000).evaluate_async(_case(session))

    assert result.label == async_result.label == NOT_APPLICABLE
    assert (
        result.metadata
        == async_result.metadata
        == {
            "too_long_turn": 2,
            "judge_calls": 0,
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
        }
    )
    mock_agent_class.assert_not_called()  # turn 1 fits but is not paid for


@patch(f"{MODULE}.Agent")
def test_sync_evaluation_stops_at_the_first_turn_that_is_too_long(mock_agent_class):
    mock_agent = Mock(side_effect=[ContextWindowOverflowException("too long")])
    mock_agent_class.return_value = mock_agent

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))

    assert result.metadata == {"too_long_turn": 1, "judge_calls": 1, "judge_input_tokens": 0, "judge_output_tokens": 0}
    assert mock_agent.call_count == 1  # turn 2 is not judged


@patch(f"{MODULE}.Agent")
async def test_async_turn_too_long_for_the_judge_is_not_applicable(mock_agent_class):
    mock_agent = Mock()
    mock_agent.invoke_async = AsyncMock(side_effect=[_verdict("S1.1"), ContextWindowOverflowException("too long")])
    mock_agent_class.return_value = mock_agent

    [result] = await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    assert result.label == NOT_APPLICABLE
    assert result.metadata["too_long_turn"] == 2


@pytest.mark.parametrize(
    "error",
    [
        ValueError("max_tokens must be <= 8192 for this model"),  # a misconfigured judge
        TimeoutError("the request took too long"),
        Exception("ValidationException: Input is too long for requested model"),  # not raised as an overflow
    ],
)
@patch(f"{MODULE}.Agent")
async def test_only_a_context_window_overflow_marks_a_turn_too_long(mock_agent_class, error):
    mock_agent_class.return_value = Mock(side_effect=error, invoke_async=AsyncMock(side_effect=error))

    with pytest.raises(type(error)):
        ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))
    with pytest.raises(type(error)):
        await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))


@patch(f"{MODULE}.Agent")
def test_other_judge_errors_are_raised(mock_agent_class):
    mock_agent_class.return_value = Mock(side_effect=RuntimeError("judge misconfigured"))

    with pytest.raises(RuntimeError, match="judge misconfigured"):
        ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))


@patch(f"{MODULE}.Agent")
async def test_async_judge_error_is_raised_and_stops_the_other_turns(mock_agent_class):
    cancelled = asyncio.Event()

    async def judge(prompt, **kwargs):
        if prompt.startswith("<task>\nFix the bug"):
            raise RuntimeError("judge misconfigured")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    mock_agent_class.return_value = Mock(invoke_async=judge)

    with pytest.raises(RuntimeError, match="judge misconfigured"):
        await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))
    assert cancelled.is_set()


@patch(f"{MODULE}.Agent")
async def test_async_turns_are_judged_concurrently_and_matched_to_their_verdicts(mock_agent_class):
    both_started = asyncio.Event()
    started = []

    async def judge(prompt, **kwargs):
        started.append(prompt)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)  # times out if the turns run one by one
        return _verdict("S1.1") if prompt.startswith("<task>\nFix the bug") else _verdict()

    mock_agent_class.return_value = Mock(invoke_async=judge)

    [result] = await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    assert result.metadata["useful_ids"] == ["S1.1"]
    assert result.metadata["judge_input_tokens"] == 200


@patch(f"{MODULE}.Agent")
async def test_async_failures_are_reported_for_the_earliest_failed_turn(mock_agent_class):
    async def judge(prompt, **kwargs):
        if prompt.startswith("<task>\nFix the bug"):
            raise ContextWindowOverflowException("too long")
        raise RuntimeError("judge misconfigured")

    mock_agent_class.return_value = Mock(invoke_async=judge)

    [result] = await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    assert result.metadata["too_long_turn"] == 1


@patch(f"{MODULE}.Agent")
async def test_judge_reply_hitting_its_output_limit_is_raised_not_marked_too_long(mock_agent_class):
    error = MaxTokensReachedException("unrecoverable state due to max_tokens limit")
    mock_agent_class.return_value = Mock(side_effect=error, invoke_async=AsyncMock(side_effect=error))

    with pytest.raises(MaxTokensReachedException):
        ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))
    with pytest.raises(MaxTokensReachedException):
        await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))


def test_judge_retry_waits_are_jittered_so_turns_throttled_together_do_not_retry_in_lockstep():
    waits = [_JUDGE_RETRY_WAIT(Mock(attempt_number=3)) for _ in range(50)]

    assert all(0 <= wait <= 4 * 2**3 for wait in waits)  # within the exponential bound for attempt 3
    assert len(set(waits)) > 1  # random, not one fixed schedule for every turn


@patch(f"{MODULE}._JUDGE_RETRY_WAIT", wait_none())
@patch(f"{MODULE}.Agent")
async def test_a_throttled_judge_call_is_retried_alone(mock_agent_class):
    throttled = ModelThrottledException("slow down")
    mock_agent_class.return_value = Mock(
        side_effect=[_verdict("S1.1"), throttled, _verdict("S2.1")],
        invoke_async=AsyncMock(side_effect=[_verdict("S1.1"), throttled, _verdict("S2.1")]),
    )
    judge = mock_agent_class.return_value

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))
    [async_result] = await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    # Three calls each: turn 1, turn 2 throttled, turn 2 again; turn 1 is not judged twice.
    assert judge.call_count == judge.invoke_async.await_count == 3
    assert result.metadata["useful_ids"] == async_result.metadata["useful_ids"] == ["S1.1", "S2.1"]
    assert result.metadata["judge_calls"] == 2


@patch(f"{MODULE}._JUDGE_RETRY_WAIT", wait_none())
@patch(f"{MODULE}.Agent")
async def test_a_judge_still_throttled_after_the_retries_raises_an_error_that_is_not_a_throttle(mock_agent_class):
    throttled = ModelThrottledException("slow down")
    mock_agent_class.return_value = Mock(side_effect=throttled, invoke_async=AsyncMock(side_effect=throttled))

    with pytest.raises(_JudgeThrottledError, match="still throttled after 6 attempts") as sync_error:
        ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))
    with pytest.raises(_JudgeThrottledError) as async_error:
        await ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate_async(_case(_two_turns()))

    assert mock_agent_class.return_value.call_count == 6
    for error in (sync_error.value, async_error.value):
        assert error.__cause__ is throttled
        assert not is_throttling_error(error)  # so Experiment does not run the whole evaluation again


@patch(f"{MODULE}._JUDGE_RETRY_WAIT", wait_none())
@patch(f"{MODULE}.Agent")
async def test_in_an_experiment_a_turn_that_stays_throttled_is_not_judged_again_with_the_others(mock_agent_class):
    async def judge(prompt, **kwargs):
        if prompt.startswith("<task>\nNow add a test"):
            raise ModelThrottledException("slow down")  # e.g. one large turn over the tokens-per-minute quota
        return _verdict("S1.1")

    calls = AsyncMock(side_effect=judge)
    mock_agent_class.return_value = Mock(invoke_async=calls)
    experiment = Experiment(cases=[Case(input="q")], evaluators=[ContextUtilizationEvaluator(measured_tools=MEASURED)])

    report = await experiment.run_evaluations_async(lambda case: {"output": "a", "trajectory": _two_turns()})

    assert calls.await_count == 1 + 6  # turn 1 once, turn 2 with its own retries; no rerun of the case
    assert report.reasons == ["Evaluator error: turn 2: the judge was still throttled after 6 attempts"]


@patch(f"{MODULE}.Agent")
def test_a_judged_turn_without_a_request_or_answer_logs_a_warning(mock_agent_class, caplog):
    _mock_judge(mock_agent_class, ["S1.1"])
    session = _session([_tool(1, "read", "bug is in a.py")])  # no AgentInvocationSpan

    with caplog.at_level("WARNING", logger=MODULE):
        ContextUtilizationEvaluator().evaluate(_case(session))

    assert "no request or answer for these turns" in caplog.text
    assert "turns=<[1]>" in caplog.text


@patch(f"{MODULE}.Agent")
def test_the_judges_token_usage_is_summed_over_its_calls(mock_agent_class):
    mock_agent_class.return_value = Mock(side_effect=[_verdict("S1.1"), _verdict("S2.1")])

    [result] = ContextUtilizationEvaluator(measured_tools=MEASURED).evaluate(_case(_two_turns()))

    assert (result.metadata["judge_input_tokens"], result.metadata["judge_output_tokens"]) == (200, 20)


@patch(f"{MODULE}.Agent")
def test_overall_score_leaves_out_nothing_measured_but_counts_rows_that_could_not_be_judged(mock_agent_class):
    _mock_judge(mock_agent_class, ["S1.1", "S2.1"])
    evaluator = ContextUtilizationEvaluator(measured_tools=MEASURED)
    scored = evaluator.evaluate(_case(_two_turns()))
    nothing_measured = ContextUtilizationEvaluator(measured_tools={"web_fetch"}).evaluate(_case(_two_turns()))
    no_session = evaluator.evaluate(EvaluationData(input="q", actual_output="a"))
    too_long = ContextUtilizationEvaluator(max_judge_input_tokens=10).evaluate(_case(_two_turns()))
    rows = [scored, nothing_measured, no_session, too_long]

    overall = EvaluationReport.calculate_overall_score([r[0].score for r in rows], rows)

    assert scored[0].score == 1.0
    assert overall == pytest.approx(1.0 / 3)  # nothing_measured left out; no_session and too_long count as 0
