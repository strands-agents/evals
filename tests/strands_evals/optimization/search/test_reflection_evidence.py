"""Tests for Reflection's evidence: the trace files, the digest, the rewards, and the versions folder."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from strands_evals.optimization import (
    AtMost,
    Criteria,
    Maximize,
    MetricRegistry,
    Minimize,
    Space,
    TextValue,
    WeightedSum,
)
from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.search.proposer import ProposalRequest
from strands_evals.optimization.search.reflection import (
    _build_evidence,
    _digest_text,
    _messages,
    _rewards,
    _text_files,
)
from strands_evals.types import EvaluationOutput
from strands_evals.types.optimization import (
    CandidateScores,
    CandidateSummary,
    Estimate,
    Measured,
    TrialRecord,
    TrialStatus,
    Usage,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCallContent,
    ToolResultContent,
    Trace,
    UserMessage,
)

REGISTRY = MetricRegistry({"Score": (0.0, 1.0), "Cost": (0.0, 1.0)})
SPACE = Space({"prompt": TextValue(default="Be helpful.\n"), "tone": TextValue(default="plain")})
T0 = datetime(2026, 11, 12, 12, 0, 0)


def trial(case, run=0, *, status="completed", score=None, tokens=100, reason="ok", output="answer", trajectory=None):
    evaluations = (
        {} if score is None else {"Score": EvaluationOutput(score=score, test_pass=score >= 0.5, reason=reason)}
    )
    return TrialRecord(
        candidate_id="p",
        case_name=case,
        run_index=run,
        split="train",
        status=TrialStatus(status),
        error="boom" if status == "agent_failure" else None,
        error_type="ValueError" if status == "agent_failure" else None,
        attempts=1,
        previous_attempts=[],
        output=output,
        trajectory=trajectory,
        interactions=None,
        environment_state=None,
        evaluations=evaluations,
        details={k: [v] for k, v in evaluations.items()},
        evaluator_errors={},
        usage=Usage.zero().model_copy(update={"tokens": tokens}),
        judge_usage={},
        rollouts=1,
    )


def scores_of(trials, criteria, extra=None):
    """The parent's train scores, as scoring gives them for these trials (one run per case)."""
    runs = {}
    for t in trials:
        counted = t.status in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE)
        failed = t.status is TrialStatus.AGENT_FAILURE
        score = (
            criteria.agent_failure_value("Score")
            if failed
            else (t.evaluations["Score"].score if t.evaluations else None)
        )
        metrics = runs.setdefault(t.case_name, {"Score": [None], "tokens": [None]})
        metrics["Score"][t.run_index] = score if counted else None
        metrics["tokens"][t.run_index] = float(t.usage.tokens) if counted else None
        for metric, values in (extra or {}).get(t.case_name, {}).items():
            metrics[metric] = values
    objective_by_run = {
        case: [criteria.objective_value({m: v[0] for m, v in ms.items() if v[0] is not None})]
        for case, ms in runs.items()
    }
    return CandidateScores(
        candidate_id="p",
        objective_by_case={},
        metrics_by_case={},
        unavailable={},
        objective_by_run=objective_by_run,
        metrics_by_run=runs,
        objective=None,
        is_eligible=True,
        failures=0,
    )


def summary(
    candidate,
    *,
    number,
    rank,
    score=0.5,
    tokens=100.0,
    eligible=True,
    trial_count=2,
    failures=0,
    estimates=None,
    feasible=True,
):
    return CandidateSummary(
        candidate_id=candidate.id,
        parent_id=candidate.parent_id,
        origin="baseline" if candidate.parent_id is None else "text",
        values=dict(candidate.values),
        number=number,
        measured_objective=score,
        measured={"Score": Measured(mean=score, ci95=None, n=2), "tokens": Measured(mean=tokens, ci95=None, n=2)},
        trial_count=trial_count,
        failures=failures,
        ranking_objective=score if eligible else None,
        term_values=[score] if eligible else None,
        rank=rank,
        estimates=estimates or {},
        violations=[] if eligible else None,
        is_feasible=feasible if eligible else None,
        is_eligible=eligible,
        changed_keys=[],
    )


def request(trials, *, criteria=None, parent=None, candidates=(), keys=("prompt",), inputs=None):
    criteria = criteria or Criteria(Maximize("Score"), [], REGISTRY)
    parent = parent or Candidate.baseline(SPACE)
    return ProposalRequest(
        space=SPACE,
        keys=list(keys),
        parent=parent,
        trials=list(trials),
        scores=scores_of(trials, criteria),
        inputs=inputs if inputs is not None else {t.case_name: f"question {t.case_name}" for t in trials},
        fit=None,
        candidates=list(candidates) or [summary(parent, number=0, rank=1)],
        n=1,
        iteration=1,
        seed=0,
        redraw=0,
        criteria=criteria,
        log=lambda *args, **kwargs: None,
    )


@pytest.fixture
def folders(tmp_path):
    traces, versions = tmp_path / "traces", tmp_path / "versions"
    traces.mkdir()
    versions.mkdir()
    return str(traces), str(versions)


# ---- rewards -------------------------------------------------------------------------------------------------------


def test_reward_is_the_bounded_objective_and_0_for_an_agent_failure():
    req = request([trial("a", score=0.8), trial("b", score=0.2), trial("c", status="agent_failure")])
    rewards, label, ranked = _rewards(req, 0.5)
    assert [r for _, r, _ in rewards] == [0.8, 0.2, 0.0]
    assert (label, ranked) == (["Score"], False)


def test_infrastructure_and_evaluator_failures_have_no_reward():
    req = request(
        [trial("a", score=0.8), trial("b", status="infrastructure_invalid"), trial("c", status="evaluator_invalid")]
    )
    rewards, _, _ = _rewards(req, 0.5)
    assert [t.case_name for t, _, _ in rewards] == ["a"]


def test_an_unbounded_metric_is_left_out_of_the_reward_and_judged_by_version(folders):
    criteria = Criteria(WeightedSum(Maximize("Score"), Minimize("tokens", scale=1000)), [], REGISTRY)
    req = request([trial("a", score=0.8, tokens=10), trial("b", score=0.4, tokens=5000)], criteria=criteria)
    rewards, label, ranked = _rewards(req, 0.5)
    assert [r for _, r, _ in rewards] == [0.8, 0.4]  # the score alone, not traded against tokens
    assert (label, ranked) == (["Score"], False)
    evidence = _build_evidence(req, 0.5, *folders)
    assert evidence.judged_by_version == ["tokens"]


def test_a_bounded_constraint_joins_the_reward_as_a_mean():
    criteria = Criteria(Maximize("Score"), [AtMost("Cost", 0.2)], REGISTRY)
    trials = [trial("a", score=0.8)]
    req = request(trials, criteria=criteria)
    req = ProposalRequest(**{**req.__dict__, "scores": scores_of(trials, criteria, {"a": {"Cost": [0.4]}})})
    rewards, label, _ = _rewards(req, 0.5)
    assert label == ["Score", "Cost"]
    assert rewards[0][1] == pytest.approx((0.8 + (1 - 0.4)) / 2)
    assert rewards[0][2]["constraints_broken"] == {"Cost <= 0.2": pytest.approx(0.2)}


def test_a_metric_with_no_value_is_left_out_of_the_reward_not_counted_as_0():
    """N/A (or a failed evaluator) on one part: the other parts' weights are rescaled, as the objective does."""
    criteria = Criteria(WeightedSum(Maximize("Score"), Maximize("Cost")), [], REGISTRY)
    trials = [trial("a", score=0.9), trial("b", score=0.4)]
    req = request(trials, criteria=criteria)
    extra = {"a": {"Cost": [None]}, "b": {"Cost": [0.6]}}
    req = ProposalRequest(**{**req.__dict__, "scores": scores_of(trials, criteria, extra)})
    rewards, _, _ = _rewards(req, 0.5)
    assert [r for _, r, _ in rewards] == [pytest.approx(0.9), pytest.approx(0.5)]


def test_a_constraint_with_no_value_is_left_out_of_the_mean():
    criteria = Criteria(Maximize("Score"), [AtMost("Cost", 0.2)], REGISTRY)
    trials = [trial("a", score=0.8)]
    req = request(trials, criteria=criteria)
    req = ProposalRequest(**{**req.__dict__, "scores": scores_of(trials, criteria, {"a": {"Cost": [None]}})})
    rewards, _, _ = _rewards(req, 0.5)
    assert rewards[0][1] == pytest.approx(0.8)


def test_a_non_finite_or_out_of_range_value_never_makes_a_reward_outside_0_1():
    """Scoring takes these out; a hand-built request still gets rewards in [0, 1]."""
    trials = [trial("a", score=0.5), trial("b", score=0.5), trial("c", score=0.5)]
    req = request(trials)
    scores = scores_of(trials, req.criteria)
    runs = {**scores.metrics_by_run, "a": {"Score": [float("nan")]}, "b": {"Score": [1.7]}}
    req = ProposalRequest(**{**req.__dict__, "scores": scores.model_copy(update={"metrics_by_run": runs})})
    rewards, _, _ = _rewards(req, 0.5)
    assert [r for _, r, _ in rewards] == [0.0, 1.0, 0.5]
    assert "Score" not in rewards[0][2]["metrics"]


def test_no_bounded_goal_metric_gives_mid_ranks():
    criteria = Criteria(Minimize("tokens"), [], REGISTRY)
    req = request([trial("a", tokens=10), trial("b", tokens=30), trial("c", tokens=20)], criteria=criteria)
    rewards, label, ranked = _rewards(req, 0.5)
    assert [r for _, r, _ in rewards] == [1.0, 0.0, 0.5]  # fewest tokens best
    assert (label, ranked) == (["tokens"], True)


def test_success_threshold_none_ranks_trials_by_objective():
    req = request([trial("a", score=0.9), trial("b", score=0.1), trial("c", score=0.5)])
    rewards, label, ranked = _rewards(req, None)
    assert [r for _, r, _ in rewards] == [1.0, 0.0, 0.5]
    assert (label, ranked) == (["Score"], True)


def test_a_failed_trial_lists_only_its_usage():
    req = request([trial("a", status="agent_failure", tokens=40)])
    ((_, _, metadata),) = _rewards(req, 0.5)[0]
    assert metadata["metrics"] == {"tokens": 40.0}
    assert metadata["reasons"] == {}
    assert metadata["objectives"] == {"Score": 0.0}  # scored at its worst end, as the run counts it
    assert metadata["error"] == "boom"


# ---- trace files and digest ----------------------------------------------------------------------------------------


def session(*messages_per_call, tool=False):
    """One trial: an agent span holding one model call per entry, each with those messages."""
    calls = []
    for i, messages in enumerate(messages_per_call):
        info = SpanInfo(
            session_id="s", start_time=T0 + timedelta(seconds=i + 1), end_time=T0 + timedelta(seconds=i + 2)
        )
        calls.append(InferenceSpan(span_info=info, messages=messages))
    agent = AgentInvocationSpan(
        span_info=SpanInfo(session_id="s", start_time=T0, end_time=T0 + timedelta(seconds=60)),
        user_prompt="q",
        agent_response="a",
        available_tools=[],
    )
    return Session(traces=[Trace(spans=[agent, *calls], trace_id="t", session_id="s")], session_id="s")


def test_trace_files_hold_the_trial_and_never_the_expected_output(folders):
    conversation = [
        UserMessage(content=[TextContent(text="question a")]),
        AssistantMessage(content=[ToolCallContent(name="search", arguments={"q": "x"}, tool_call_id="1")]),
        UserMessage(content=[ToolResultContent(content="found", tool_call_id="1")]),
        AssistantMessage(content=[TextContent(text="answer")]),
    ]
    trajectory = session(conversation[:2], conversation)
    req = request([trial("a", score=0.8, trajectory=trajectory), trial("b", score=0.2)])
    evidence = _build_evidence(req, 0.5, *folders)
    traces, _ = folders
    assert sorted(os.listdir(traces)) == ["DIGEST.md", "rollout_0000.json", "rollout_0001.json"]
    with open(os.path.join(traces, "rollout_0000.json")) as f:
        data = json.load(f)
    assert data["reward"] == 0.8
    assert data["output"] == "answer"
    assert data["task"] == {"name": "a", "repetition": 0, "input": "question a"}
    assert data["task_id"] == "rollout_0000"
    assert data["eval_result"] == {"reward": 0.8}
    assert data["metrics"]["metadata"]["reasons"] == {"Score": "ok"}
    assert data["response"]["messages"] == [
        {"role": "user", "content": [{"text": "question a"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "search", "input": {"q": "x"}}}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "1", "status": "success", "content": [{"text": "found"}]}}],
        },
        {"role": "assistant", "content": [{"text": "answer"}]},
    ]
    assert "expected" not in json.dumps(data)
    assert evidence.trials == 2
    with open(evidence.digest_file) as f:
        assert f.read() == evidence.digest_text
    assert "- rollout_0001.json: reward 0.2; ended: answer; reasons: Score: ok; 0 messages, 0 tool calls\n" in (
        evidence.digest_text
    )
    assert evidence.digest_text.endswith("## Cases where every conversation succeeded (1)\na (1/1)\n")


def test_no_trial_to_learn_from_gives_no_evidence(folders):
    assert _build_evidence(request([trial("a", status="infrastructure_invalid")]), 0.5, *folders) is None


def test_messages_take_each_agents_last_call_innermost_agent_owning_its_calls():
    outer = AgentInvocationSpan(
        span_info=SpanInfo(session_id="s", start_time=T0, end_time=T0 + timedelta(seconds=60)),
        user_prompt="q",
        agent_response="a",
        available_tools=[],
    )
    inner = AgentInvocationSpan(
        span_info=SpanInfo(session_id="s", start_time=T0 + timedelta(seconds=10), end_time=T0 + timedelta(seconds=20)),
        user_prompt="sub",
        agent_response="sub answer",
        available_tools=[],
    )

    def call(second, text):
        info = SpanInfo(
            session_id="s", start_time=T0 + timedelta(seconds=second), end_time=T0 + timedelta(seconds=second + 1)
        )
        return InferenceSpan(span_info=info, messages=[UserMessage(content=[TextContent(text=text)])])

    spans = [outer, inner, call(1, "outer 1"), call(12, "inner 1"), call(15, "inner 2"), call(30, "outer 2")]
    trajectory = Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")
    assert [m["content"][0]["text"] for m in _messages(trajectory)] == ["outer 2", "inner 2"]


def _span(cls, start, end, span_id=None, parent=None, **fields):
    info = SpanInfo(
        session_id="s",
        start_time=T0 + timedelta(seconds=start),
        end_time=T0 + timedelta(seconds=end),
        span_id=span_id,
        parent_span_id=parent,
    )
    if cls is AgentInvocationSpan:
        fields = {"user_prompt": "q", "agent_response": "a", "available_tools": [], **fields}
    return cls(span_info=info, **fields)


def _says(*texts):
    return [UserMessage(content=[TextContent(text=t)]) for t in texts]


def test_messages_of_an_agent_invoked_twice_appear_once():
    """The second invocation's last call holds the first's history: only the longer conversation is kept."""
    spans = [
        _span(AgentInvocationSpan, 0, 10),
        _span(InferenceSpan, 1, 2, messages=_says("hi", "hello")),
        _span(AgentInvocationSpan, 20, 30),
        _span(InferenceSpan, 21, 22, messages=_says("hi", "hello", "order?", "which one")),
    ]
    trajectory = Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")
    assert [m["content"][0]["text"] for m in _messages(trajectory)] == ["hi", "hello", "order?", "which one"]


def test_messages_of_parallel_agents_follow_their_parent_spans():
    """A runs 0-10 s and B 1-9 s: by time alone A's call (in B's window) would be B's; parent ids say A's."""
    spans = [
        _span(AgentInvocationSpan, 0, 10, span_id="a"),
        _span(AgentInvocationSpan, 1, 9, span_id="b"),
        _span(InferenceSpan, 2, 3, span_id="ca", parent="a", messages=_says("A's call")),
        _span(InferenceSpan, 4, 5, span_id="cb", parent="b", messages=_says("B's call")),
    ]
    trajectory = Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")
    assert [m["content"][0]["text"] for m in _messages(trajectory)] == ["A's call", "B's call"]


def test_messages_compare_naive_and_aware_span_times():
    """The agent's span has naive times and its calls aware ones (UTC): compared as UTC, no `TypeError`."""
    utc = T0.replace(tzinfo=timezone.utc)

    def call(second, text):
        info = SpanInfo(
            session_id="s", start_time=utc + timedelta(seconds=second), end_time=utc + timedelta(seconds=second + 1)
        )
        return InferenceSpan(span_info=info, messages=_says(text))

    spans = [_span(AgentInvocationSpan, 0, 60), call(1, "first"), call(30, "last")]
    trajectory = Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")
    assert [m["content"][0]["text"] for m in _messages(trajectory)] == ["last"]


def test_messages_order_agents_whose_span_times_mix_naive_and_aware():
    """Two agents, one with naive times and one with aware ones: ordering them compares UTC, no `TypeError`."""
    utc = T0.replace(tzinfo=timezone.utc)
    second = AgentInvocationSpan(
        span_info=SpanInfo(
            session_id="s",
            start_time=utc + timedelta(seconds=20),
            end_time=utc + timedelta(seconds=30),
            span_id="b",
        ),
        user_prompt="q",
        agent_response="a",
        available_tools=[],
    )
    spans = [
        second,
        _span(AgentInvocationSpan, 0, 10, span_id="a"),
        _span(InferenceSpan, 1, 2, parent="a", messages=_says("first agent")),
        _span(InferenceSpan, 21, 22, parent="b", messages=_says("second agent")),
    ]
    trajectory = Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")
    assert [m["content"][0]["text"] for m in _messages(trajectory)] == ["first agent", "second agent"]


def test_reflection_reads_only_train_trials(folders):
    """A trial of another split never reaches the evidence (invariant 2), even if handed over by mistake."""
    held_out = trial("v", score=0.1).model_copy(update={"split": "validation"})
    evidence = _build_evidence(request([trial("a", score=0.8), held_out]), 0.5, *folders)
    assert evidence.trials == 1
    assert '"v"' not in json.dumps(os.listdir(folders[0]))
    assert "v:" not in evidence.digest_text


def test_messages_keep_a_rollouts_own_list_and_read_none_as_empty():
    assert _messages([{"role": "user", "content": []}]) == [{"role": "user", "content": []}]
    assert _messages(None) == []


def test_digest_lists_failing_cases_first_then_passing():
    rows = [
        {
            "file": "rollout_0000.json",
            "case": "a",
            "repetition": 0,
            "reward": 0.2,
            "status": "completed",
            "error": None,
            "reasons": {"Score": "wrong"},
            "objectives": {"Score": 0.2},
            "ended": "no",
            "messages": 2,
            "calls": 0,
        },
        {
            "file": "rollout_0001.json",
            "case": "b",
            "repetition": 0,
            "reward": 0.9,
            "status": "completed",
            "error": None,
            "reasons": {},
            "objectives": {"Score": 0.9},
            "ended": "yes",
            "messages": 2,
            "calls": 0,
        },
    ]
    text = _digest_text(rows, 0.5, 20_000)
    assert text.startswith(
        "# Digest: the current version's 2 conversations, 2 cases\n\n1 conversations did not succeed"
    )
    assert "### a: 0 of 1 succeeded\n- rollout_0000.json: reward 0.2; ended: no; reasons: Score: wrong;" in text
    assert text.endswith("## Cases where every conversation succeeded (1)\nb (1/1)\n")


def test_digest_is_cut_to_its_limit():
    rows = [
        {
            "file": f"rollout_{i:04d}.json",
            "case": f"case-{i}",
            "repetition": 0,
            "reward": 0.0,
            "status": "completed",
            "error": None,
            "reasons": {"Score": "x" * 500},
            "objectives": {},
            "ended": "y" * 500,
            "messages": 1,
            "calls": 0,
        }
        for i in range(200)
    ]
    text = _digest_text(rows, 0.5, 2_000)
    assert len(text) <= 2_000
    assert "more lines left out at the digest's limit of 2,000 characters" in text


def test_digest_that_fits_at_the_shortest_fields_keeps_every_line():
    """A digest too long with fields cut to 75 characters but whole with them cut to 40 is returned as it is: it never
    takes the line-dropping path, which would say lines were left out while dropping none."""
    rows = [
        {
            "file": f"rollout_{i:04d}.json",
            "case": f"case-{i}",
            "repetition": 0,
            "reward": 0.0,
            "status": "completed",
            "error": None,
            "reasons": {"Score": "x" * 500},
            "objectives": {},
            "ended": "y" * 500,
            "messages": 1,
            "calls": 0,
        }
        for i in range(6)
    ]
    text = _digest_text(rows, 0.5, 998)
    assert len(text) <= 998
    assert "left out" not in text
    assert text.endswith("## Cases where every conversation succeeded (0)\n(none)\n")
    for i in range(6):
        assert f"case-{i}" in text


# ---- versions ------------------------------------------------------------------------------------------------------


def test_versions_folder_has_one_folder_per_version_with_each_text_alone(folders):
    baseline = Candidate.baseline(SPACE)
    child = baseline.child(SPACE, {"prompt": "Be brief.\n", "tone": "warm"}, origin="text", rationale="")
    unscored = baseline.child(SPACE, {"prompt": "Lost.\n"}, origin="text", rationale="")
    criteria = Criteria(Maximize("Score"), [AtMost("tokens", 150)], REGISTRY)
    candidates = [
        summary(child, number=1, rank=1, score=0.9, tokens=120.0, estimates={"Score": Estimate(mean=0.84, sd=0.03)}),
        summary(baseline, number=0, rank=2, score=0.5, tokens=200.0, trial_count=3, failures=1, feasible=False),
        summary(unscored, number=2, rank=None, eligible=False),
    ]
    req = request([trial("a", score=0.9)], criteria=criteria, parent=child, candidates=candidates)
    evidence = _build_evidence(req, 0.5, *folders)
    _, versions = folders
    assert sorted(os.listdir(versions)) == ["rank-001-version-1", "rank-002-version-0", "unranked-version-2"]
    with open(os.path.join(versions, "rank-001-version-1", "prompt.txt")) as f:
        assert f.read() == "Be brief.\n"
    with open(os.path.join(versions, "rank-001-version-1", "version.md")) as f:
        assert f.read() == (
            "# Version 1\nrank: 1\nbuilt from: version 0\nScore: 0.84 (measured 0.9)\ntokens: 120\n"
            "meets the limits: yes\n"
            'conversations: 2\n\n## Other values\n{\n "tone": "warm"\n}\n\n'
            "## Its texts, each in its own file in this folder (the text alone)\n"
            "- prompt: `prompt.txt` (10 characters)\n"
        )
    with open(os.path.join(versions, "rank-002-version-0", "version.md")) as f:
        head = f.read()
    assert head.startswith("# Version 0 (the baseline)\nrank: 2\nbuilt from: nothing (the start)\n")
    assert "meets the limits: no\n" in head  # as the ranking assessed it
    assert "conversations: 2\n" in head  # 3 trials counted, 1 of them an agent failure
    first, second, third = evidence.versions
    assert (first.is_current, first.changed, first.cells, first.meets_limits) == (
        True,
        ["prompt", "other values"],
        ["0.84 (measured 0.9)", "120"],  # the estimate and the measured mean, or the measured mean alone
        True,
    )
    assert (second.is_baseline, second.built_from, second.meets_limits, second.conversations) == (True, None, False, 2)
    assert (third.rank, third.meets_limits, third.changed) == (None, None, ["prompt"])
    assert evidence.limits == ["tokens <= 150"]
    assert evidence.goal_metrics == ["Score", "tokens"]


def test_text_file_names():
    assert _text_files(["system_prompt", "skills/triage/SKILL.md", "a b", ".hidden", "a_b", "version.md"]) == {
        "system_prompt": "system_prompt.txt",
        "skills/triage/SKILL.md": "skills_triage_SKILL.md.txt",
        "a b": "a_b.txt",
        ".hidden": "hidden.txt",
        "a_b": "a_b-2.txt",
        "version.md": "version.md.txt",
    }
