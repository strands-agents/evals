"""Stage D: the automatic disclosure seam on the base Evaluator.

Covers the decision (`_resolve_disclosure_index`), the render fallback
(`_render_with_disclosure`), and the mode validation shared by every judge,
plus an end-to-end pass through TrajectoryEvaluator to confirm a fitting case
is byte-identical to before and an overflowing case hands the judge the three
trace tools.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import (
    CoherenceEvaluator,
    ConcisenessEvaluator,
    Evaluator,
    FaithfulnessEvaluator,
    GoalSuccessRateEvaluator,
    HelpfulnessEvaluator,
    ResponseRelevanceEvaluator,
    ToolParameterAccuracyEvaluator,
    ToolSelectionAccuracyEvaluator,
    TrajectoryEvaluator,
)
from strands_evals.evaluators._trace_index import TraceIndex
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)

# The preflight threshold is DEFAULT_MAX_INPUT_TOKENS (200K) * PREFLIGHT_SAFETY_MARGIN
# (0.65) ~= 130K tokens. A probe comfortably past that overflows under "auto"
# regardless of whether tiktoken or the char/4 fallback does the counting.
_OVERFLOW_PROBE = "token " * 300_000
_FITS_PROBE = "a short prompt that fits any judge window"


def _span_info(second: int) -> SpanInfo:
    return SpanInfo(
        session_id="s1",
        span_id=f"sp{second}",
        start_time=datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, second + 1, tzinfo=timezone.utc),
    )


def _small_session() -> Session:
    """A one-turn session that fits any judge window when inlined."""
    return Session(
        traces=[
            Trace(
                spans=[
                    AgentInvocationSpan(
                        span_info=_span_info(0),
                        user_prompt="What is 2 + 2?",
                        agent_response="4",
                        available_tools=[],
                    )
                ],
                trace_id="t1",
                session_id="s1",
            )
        ],
        session_id="s1",
    )


def _huge_session() -> Session:
    """A session whose serialized trajectory overflows a 200K-token judge."""
    spans = [
        AgentInvocationSpan(
            span_info=_span_info(0),
            user_prompt="Summarize the run.",
            agent_response="done",
            available_tools=[],
        )
    ]
    for i in range(1, 6):
        spans.append(
            ToolExecutionSpan(
                span_info=_span_info(i),
                tool_call=ToolCall(name="query_db", arguments={"page": i}),
                tool_result=ToolResult(content="filler row " * 60_000),
            )
        )
    return Session(traces=[Trace(spans=spans, trace_id="t1", session_id="s1")], session_id="s1")


def _case(trajectory) -> EvaluationData:
    return EvaluationData(
        input="q",
        actual_output="a",
        actual_trajectory=trajectory,
    )


# --- _validate_disclosure --------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_validate_disclosure_accepts_known_modes(mode):
    assert Evaluator._validate_disclosure(mode) == mode


def test_validate_disclosure_rejects_unknown_mode():
    with pytest.raises(ValueError, match="disclosure"):
        Evaluator._validate_disclosure("sometimes")


def test_constructor_validates_disclosure():
    with pytest.raises(ValueError, match="disclosure"):
        TrajectoryEvaluator(rubric="r", disclosure="bogus")


# --- _resolve_disclosure_index ---------------------------------------------


def test_auto_fits_does_not_disclose():
    ev = Evaluator()
    ev.disclosure = "auto"
    assert ev._resolve_disclosure_index(_case(_small_session()), _FITS_PROBE) is None


def test_auto_overflow_discloses():
    ev = Evaluator()
    ev.disclosure = "auto"
    index = ev._resolve_disclosure_index(_case(_small_session()), _OVERFLOW_PROBE)
    assert isinstance(index, TraceIndex)


def test_never_does_not_disclose_even_on_overflow():
    ev = Evaluator()
    ev.disclosure = "never"
    assert ev._resolve_disclosure_index(_case(_small_session()), _OVERFLOW_PROBE) is None


def test_always_discloses_when_session_present_even_if_it_fits():
    ev = Evaluator()
    ev.disclosure = "always"
    index = ev._resolve_disclosure_index(_case(_small_session()), _FITS_PROBE)
    assert isinstance(index, TraceIndex)


def test_no_session_never_discloses():
    ev = Evaluator()
    for mode in ("auto", "always"):
        ev.disclosure = mode
        # A non-Session trajectory (e.g. a plain list of steps) has no spans to index.
        assert ev._resolve_disclosure_index(_case(["step one", "step two"]), _OVERFLOW_PROBE) is None


# --- _render_with_disclosure -----------------------------------------------


def test_render_inline_path_is_byte_identical_and_toolless():
    ev = Evaluator()
    ev.disclosure = "auto"
    prompt, tools = ev._render_with_disclosure(
        _case(_small_session()),
        lambda idx: _FITS_PROBE if idx is None else ev._disclosed_trace_section(idx),
    )
    assert prompt == _FITS_PROBE
    assert tools == []


def test_render_disclosure_path_swaps_prompt_and_adds_three_tools():
    ev = Evaluator()
    ev.disclosure = "auto"
    prompt, tools = ev._render_with_disclosure(
        _case(_small_session()),
        lambda idx: _OVERFLOW_PROBE if idx is None else ev._disclosed_trace_section(idx),
    )
    assert prompt != _OVERFLOW_PROBE
    assert "too large to inline" in prompt
    tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools}
    assert {"list_spans", "get_span", "search_spans"} == tool_names


# --- end-to-end through TrajectoryEvaluator --------------------------------


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_fits_keeps_only_scoring_tools(mock_agent_class):
    mock_agent = Mock()
    mock_agent.return_value = Mock(structured_output=EvaluationOutput(score=1.0, test_pass=True, reason="ok"))
    mock_agent_class.return_value = mock_agent

    evaluator = TrajectoryEvaluator(rubric="r", disclosure="auto")
    evaluator.evaluate(_case(_small_session()))

    tools = mock_agent_class.call_args[1]["tools"]
    tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools}
    assert not ({"list_spans", "get_span", "search_spans"} & tool_names)


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_overflow_adds_disclosure_tools(mock_agent_class):
    mock_agent = Mock()
    mock_agent.return_value = Mock(structured_output=EvaluationOutput(score=1.0, test_pass=True, reason="ok"))
    mock_agent_class.return_value = mock_agent

    evaluator = TrajectoryEvaluator(rubric="r", disclosure="auto")
    evaluator.evaluate(_case(_huge_session()))

    tools = mock_agent_class.call_args[1]["tools"]
    tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools}
    assert {"list_spans", "get_span", "search_spans"} <= tool_names
    # The judge is handed the overview, not the inlined multi-hundred-K-token trajectory.
    prompt = mock_agent.call_args[0][0]
    assert "too large to inline" in prompt


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_never_inlines_on_overflow(mock_agent_class):
    mock_agent = Mock()
    mock_agent.return_value = Mock(structured_output=EvaluationOutput(score=1.0, test_pass=True, reason="ok"))
    mock_agent_class.return_value = mock_agent

    evaluator = TrajectoryEvaluator(rubric="r", disclosure="never")
    evaluator.evaluate(_case(_huge_session()))

    tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in mock_agent_class.call_args[1]["tools"]}
    assert not ({"list_spans", "get_span", "search_spans"} & tool_names)
    # "never" restores today's behavior: the full trajectory is inlined, so a real
    # overflow is surfaced downstream as could-not-evaluate rather than disclosed here.
    assert "too large to inline" not in mock_agent.call_args[0][0]


# --- every judge forwards the disclosure tools to its Agent -----------------

# (module under strands_evals.evaluators, factory) for the judges that take a
# Session trajectory and disclose it: each must pass the three trace tools to the
# Agent it builds. Only TrajectoryEvaluator was pinned before, so a dropped
# `tools=tools` on any of the others would ship the worst failure shape — a prompt
# that says "read the trace via the tools" with no tools attached. `disclosure=
# "always"` forces the disclosure path for any Session regardless of how each judge
# renders its prompt, so this isolates the wiring, not per-judge overflow behavior.
_TOOL_FORWARDING_JUDGES = [
    ("helpfulness_evaluator", lambda: HelpfulnessEvaluator(disclosure="always")),
    ("coherence_evaluator", lambda: CoherenceEvaluator(disclosure="always")),
    ("conciseness_evaluator", lambda: ConcisenessEvaluator(disclosure="always")),
    ("faithfulness_evaluator", lambda: FaithfulnessEvaluator(disclosure="always")),
    ("response_relevance_evaluator", lambda: ResponseRelevanceEvaluator(disclosure="always")),
    ("goal_success_rate_evaluator", lambda: GoalSuccessRateEvaluator(disclosure="always")),
    ("trajectory_evaluator", lambda: TrajectoryEvaluator(rubric="r", disclosure="always")),
    ("tool_selection_accuracy_evaluator", lambda: ToolSelectionAccuracyEvaluator(disclosure="always")),
    ("tool_parameter_accuracy_evaluator", lambda: ToolParameterAccuracyEvaluator(disclosure="always")),
]


@pytest.mark.parametrize("module, factory", _TOOL_FORWARDING_JUDGES)
def test_judge_forwards_disclosure_tools(module, factory):
    with patch(f"strands_evals.evaluators.{module}.Agent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.return_value = Mock(structured_output=Mock())
        mock_agent_class.return_value = mock_agent

        # The Agent is constructed with tools= before its (mocked) invocation, so the
        # tools are captured regardless of any downstream error from the Mock rating.
        try:
            factory().evaluate(_case(_huge_session()))
        except Exception:  # noqa: BLE001 - only the tools= wiring is under test here
            pass

        assert mock_agent_class.call_args is not None, "judge never constructed an Agent"
        tools = mock_agent_class.call_args[1]["tools"]
        tool_names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools}
        assert {"list_spans", "get_span", "search_spans"} <= tool_names


def test_always_warns_when_trajectory_is_not_a_session(caplog):
    """An explicit `always` that can't build an index shouldn't silently inline."""
    ev = Evaluator()
    ev.disclosure = "always"
    with caplog.at_level("WARNING"):
        assert ev._resolve_disclosure_index(_case(["step one", "step two"]), "") is None
    assert any("not a Session" in r.message for r in caplog.records)


def test_auto_does_not_warn_on_non_session_trajectory(caplog):
    """`auto` inlining a list is the pre-disclosure behavior, not a misuse to warn about."""
    ev = Evaluator()
    ev.disclosure = "auto"
    with caplog.at_level("WARNING"):
        assert ev._resolve_disclosure_index(_case(["step one"]), _OVERFLOW_PROBE) is None
    assert not any("not a Session" in r.message for r in caplog.records)
