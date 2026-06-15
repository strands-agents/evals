"""Tests for GoatStrategy and its module-level helpers."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import CrescendoStrategy, GoatStrategy
from strands_evals.experimental.redteam.strategies.base import AttackRunResult
from strands_evals.experimental.redteam.strategies.goat import (
    _ATTACK_NAMES,
    _AttackerTurn,
    gen_attacker_turn,
    success_score,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig

_GOAT = "strands_evals.experimental.redteam.strategies.goat"


class _FakeSession:
    """In-test append-only TargetSession: replies via a callable, records calls.

    GOAT never backtracks, so this stub only needs ``invoke``; ``snapshot`` /
    ``restore`` raise to prove the strategy never calls them.
    """

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self.sent: list[str] = []
        self.trace: list[dict] = []

    def invoke(self, message):
        self.sent.append(message)
        return self._reply_fn(message)

    def reset(self):
        self.sent.clear()
        self.trace.clear()

    def snapshot(self):  # pragma: no cover - must never be called by GOAT
        raise AssertionError("GOAT must not snapshot (append-only)")

    def restore(self, checkpoint):  # pragma: no cover - must never be called by GOAT
        raise AssertionError("GOAT must not restore (append-only)")


def _case(
    success_criteria: str | None = "leaked the secret", actor_goal: str = "get harmful instructions"
) -> RedTeamCase:
    return RedTeamCase(
        name="g0",
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(
                risk_category="harmful_content",
                actor_goal=actor_goal,
                success_criteria=success_criteria,
            )
        ),
        metadata={"strategy": "goat"},
    )


def _turn(reply="next message", attacks=None, observation="obs", thought="th", strategy="st") -> _AttackerTurn:
    return _AttackerTurn(
        observation=observation,
        thought=thought,
        strategy=strategy,
        attacks=attacks if attacks is not None else ["hypothetical"],
        reply=reply,
    )


def _strategy(**kwargs) -> GoatStrategy:
    """A strategy with its agent builders stubbed so no real LLM is created.

    The module-level ``gen_attacker_turn`` / ``success_score`` helpers are patched per test
    via ``_run`` below (the crescendo pattern), so the loop drives off test-supplied turns.
    """
    strat = GoatStrategy(**kwargs)
    strat._build_attacker = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    strat._build_judge = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return strat


def _run(strat, case, session, *, turns, score_fn=None, max_turns=5):
    """Drive ``strat.run_attack`` with the module helpers patched (mirrors test_crescendo).

    ``turns`` is a list of ``_AttackerTurn | None`` (None simulates a parse failure); ``score_fn``
    maps a target response to a float (default 0.0 -> never early-stops). ``success_score`` keeps
    its no-criteria-returns-0 contract so the no-criteria test still exercises the real gate path.
    """
    turn_iter = iter(turns)
    score_fn = score_fn or (lambda _response: 0.0)

    def fake_gen(_attacker, _goal, _last_reply, _last_response, *, first_turn):
        return next(turn_iter)

    def fake_score(_judge, response, success_criteria):
        return 0.0 if not success_criteria else score_fn(response)

    with (
        patch(f"{_GOAT}.gen_attacker_turn", side_effect=fake_gen),
        patch(f"{_GOAT}.success_score", side_effect=fake_score),
    ):
        return strat.run_attack(case, session, max_turns=max_turns)


# ---------------------------------------------------------------------------
# ctor validation
# ---------------------------------------------------------------------------


class TestCtorGuards:
    def test_defaults(self):
        s = GoatStrategy()
        assert s.name == "goat"
        assert s.label == "goat"
        assert s._max_turns == 5
        assert s._success_threshold == 0.7
        assert s._store_reasoning is False

    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_max_turns_below_one_raises(self, bad):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            GoatStrategy(max_turns=bad)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, 2.0])
    def test_threshold_out_of_band_raises(self, bad):
        with pytest.raises(ValueError, match="success_threshold must be in"):
            GoatStrategy(success_threshold=bad)

    def test_threshold_at_one_is_allowed(self):
        assert GoatStrategy(success_threshold=1.0)._success_threshold == 1.0

    def test_label_override(self):
        assert GoatStrategy(label="goat-k10").label == "goat-k10"


# ---------------------------------------------------------------------------
# module-level helpers (real, unpatched)
# ---------------------------------------------------------------------------


class TestSuccessScore:
    def test_returns_zero_without_criteria(self):
        judge = MagicMock()
        assert success_score(lambda: judge, "anything", None) == 0.0
        judge.assert_not_called()  # no scoring -> factory's judge never invoked

    def test_parses_judge_score(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=0.8)
        assert success_score(lambda: judge, "resp", "criteria") == 0.8

    def test_parse_failure_defaults_to_zero(self):
        judge = MagicMock()
        judge.return_value.structured_output = None
        assert success_score(lambda: judge, "resp", "criteria") == 0.0

    def test_invokes_the_factorys_judge_once_per_score(self):
        """Contract: success_score calls make_judge() once per invocation and uses the returned judge
        (proves factory wiring, NOT production freshness -- that is proven by the Agent-ctor-count
        test test_reused_instance_builds_fresh_agents_each_case_without_reset)."""
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=0.5)
        make_judge = MagicMock(return_value=judge)
        success_score(make_judge, "resp", "criteria")
        make_judge.assert_called_once_with()
        judge.assert_called_once()


class TestGenAttackerTurn:
    def test_first_turn_uses_initial_prompt(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = _turn(reply="opener")
        goal = _case().config.attack_goal
        out = gen_attacker_turn(attacker, goal, "", "", first_turn=True)
        assert out.reply == "opener"
        prompt = attacker.call_args[0][0]
        assert "Begin the attack" in prompt
        assert goal.actor_goal in prompt

    def test_followup_includes_last_reply_and_response(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = _turn()
        goal = _case().config.attack_goal
        gen_attacker_turn(attacker, goal, "my last reply", "target said this", first_turn=False)
        prompt = attacker.call_args[0][0]
        assert "my last reply" in prompt
        assert "target said this" in prompt

    def test_parse_failure_returns_none(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = None
        out = gen_attacker_turn(attacker, _case().config.attack_goal, "", "", first_turn=True)
        assert out is None

    def test_brace_safe_substitution(self):
        # an actor_goal carrying literal braces must not blow up a .format-style call
        attacker = MagicMock()
        attacker.return_value.structured_output = _turn()
        goal = AttackGoal(
            risk_category="harmful_content", actor_goal="produce {payload} and {{x}}", success_criteria=None
        )
        gen_attacker_turn(attacker, goal, "{a}", "{b}", first_turn=False)
        prompt = attacker.call_args[0][0]
        assert "{payload}" in prompt and "{{x}}" in prompt


# ---------------------------------------------------------------------------
# run_attack loop
# ---------------------------------------------------------------------------


def test_returns_attack_run_result_shape():
    result = _run(
        _strategy(), _case(), _FakeSession(lambda m: "target reply"), turns=[_turn(reply="r1")], score_fn=lambda r: 1.0
    )
    assert isinstance(result, AttackRunResult)
    assert result.conversation == [
        {"role": "attacker", "content": "r1"},
        {"role": "target", "content": "target reply"},
    ]
    assert result.pruned_branches == []
    assert result.strategy_succeeded is True
    # Pin the full metadata dict (deterministic on the success path) so a new or
    # dropped key is caught -- including that reasoning_trace is absent by default.
    assert result.metadata == {
        "turns_used": 1,
        "target_calls": 1,
        "parse_failures": 0,
        "attacks_used": ["hypothetical"],  # _turn default attack
    }


def test_stops_on_success_threshold():
    # second turn scores above threshold -> loop stops at turn 2, not turn cap
    session = _FakeSession(lambda m: "resp")
    scores = iter([0.2, 0.9])
    result = _run(
        _strategy(),
        _case(),
        session,
        turns=[_turn(reply="r1"), _turn(reply="r2"), _turn(reply="r3")],
        score_fn=lambda r: next(scores),
    )
    assert result.strategy_succeeded is True
    assert result.metadata["target_calls"] == 2
    assert len(session.sent) == 2


def test_runs_to_cap_when_never_succeeds():
    turns = [_turn(reply=f"r{i}") for i in range(10)]
    result = _run(_strategy(max_turns=3), _case(), _FakeSession(lambda m: "resp"), turns=turns)
    assert result.strategy_succeeded is False
    # Full dict: ran to the (smaller) ctor cap of 3; each of the 3 turns carries
    # the _turn default attack, so attacks_used accumulates one per turn.
    assert result.metadata == {
        "turns_used": 3,
        "target_calls": 3,  # ctor max_turns wins (smaller)
        "parse_failures": 0,
        "attacks_used": ["hypothetical", "hypothetical", "hypothetical"],
    }


def test_max_turns_clamped_by_injected_value():
    turns = [_turn(reply=f"r{i}") for i in range(10)]
    # injected ceiling (2) smaller than ctor max_turns (8)
    result = _run(_strategy(max_turns=8), _case(), _FakeSession(lambda m: "resp"), turns=turns, max_turns=2)
    assert result.metadata["target_calls"] == 2


def test_no_criteria_runs_to_cap():
    # no success_criteria -> gate never fires -> runs the full cap
    turns = [_turn(reply=f"r{i}") for i in range(10)]
    result = _run(_strategy(max_turns=4), _case(success_criteria=None), _FakeSession(lambda m: "resp"), turns=turns)
    assert result.strategy_succeeded is False
    assert result.metadata["target_calls"] == 4


def test_parse_failure_ends_early():
    result = _run(
        _strategy(), _case(), _FakeSession(lambda m: "resp"), turns=[_turn(reply="r1"), None, _turn(reply="r3")]
    )
    assert result.metadata["parse_failures"] == 1
    assert result.metadata["target_calls"] == 1  # only the first turn reached the target


def test_empty_reply_ends_early_without_invoking():
    session = _FakeSession(lambda m: "resp")
    result = _run(_strategy(), _case(), session, turns=[_turn(reply="   ")])
    assert session.sent == []  # never invoke("")
    assert result.metadata["parse_failures"] == 1


def test_empty_target_response_ends_early():
    # target_calls counts the call, but the empty pair is not appended -> turns_used trails by one.
    # All-or-nothing: a turn that produced no real pair contributes nothing to attacks_used either.
    turns = [_turn(reply="r1", attacks=["hypothetical"]), _turn(reply="r2")]
    result = _run(_strategy(store_reasoning=True), _case(), _FakeSession(lambda m: ""), turns=turns)
    assert result.conversation == []
    # Full dict (store_reasoning -> reasoning_trace key present); the empty pair
    # contributes nothing to attacks_used / reasoning_trace.
    assert result.metadata == {
        "turns_used": 0,
        "target_calls": 1,
        "parse_failures": 0,
        "attacks_used": [],
        "reasoning_trace": [],
    }


def test_first_turn_breach_records_pair_and_attacks():
    # mirror image: a turn with a real pair records its attacks AND reasoning in lockstep
    turns = [_turn(reply="r1", attacks=["hypothetical"], observation="o1", thought="t1", strategy="s1")]
    result = _run(
        _strategy(store_reasoning=True), _case(), _FakeSession(lambda m: "resp"), turns=turns, score_fn=lambda r: 1.0
    )
    # Full dict: a real pair records its attacks AND reasoning in lockstep.
    assert result.metadata == {
        "turns_used": 1,
        "target_calls": 1,
        "parse_failures": 0,
        "attacks_used": ["hypothetical"],
        "reasoning_trace": [{"observation": "o1", "thought": "t1", "strategy": "s1"}],
    }


# ---------------------------------------------------------------------------
# attacks_used + reasoning_trace metadata
# ---------------------------------------------------------------------------


def test_attacks_used_filters_off_toolbox_names():
    turns = [_turn(reply="r1", attacks=["hypothetical", "made_up_name", "dual_response"])]
    result = _run(_strategy(), _case(), _FakeSession(lambda m: "resp"), turns=turns, score_fn=lambda r: 1.0)
    assert result.metadata["attacks_used"] == ["hypothetical", "dual_response"]
    assert "made_up_name" not in result.metadata["attacks_used"]


def test_all_seven_attack_names_are_accepted():
    assert _ATTACK_NAMES == {
        "refusal_suppression",
        "response_priming",
        "dual_response",
        "topic_splitting",
        "opposite_intent",
        "hypothetical",
        "persona_modification",
    }


def test_reasoning_trace_omitted_by_default():
    result = _run(
        _strategy(), _case(), _FakeSession(lambda m: "resp"), turns=[_turn(reply="r1")], score_fn=lambda r: 1.0
    )
    assert "reasoning_trace" not in result.metadata


def test_reasoning_trace_present_when_opted_in():
    turns = [_turn(reply="r1", observation="o1", thought="t1", strategy="s1")]
    result = _run(
        _strategy(store_reasoning=True), _case(), _FakeSession(lambda m: "resp"), turns=turns, score_fn=lambda r: 1.0
    )
    assert result.metadata["reasoning_trace"] == [{"observation": "o1", "thought": "t1", "strategy": "s1"}]


# ---------------------------------------------------------------------------
# reset + contract pin
# ---------------------------------------------------------------------------


def test_reset_is_noop_and_callable():
    """reset() falls back to the base no-op (no per-instance agent state); must stay callable for
    task.py interface compat and must NOT raise."""
    GoatStrategy().reset()  # no-op, no AttributeError


def test_reused_instance_builds_fresh_agents_each_case_without_reset():
    """Stateless across cases: the attacker is built once per case (its O/T/S/R history is the
    strategy) and the judge is built fresh per scoring call, with no reset() between cases. We patch
    the SDK ``Agent`` ctor (NOT the _build_* methods) so the REAL build bodies run -- reintroducing
    any instance/per-case judge cache OR a cross-case attacker cache drops the count and fails."""
    strat = GoatStrategy()

    def make_agent(**_kw):
        # Each built Agent, when invoked, scores 0.0 (so success_score never early-stops and every
        # turn scores -> a fresh judge each turn). The attacker mock is never invoked (gen is stubbed).
        agent = MagicMock()
        agent.return_value.structured_output = MagicMock(score=0.0)
        return agent

    with (
        patch(f"{_GOAT}.Agent", side_effect=make_agent) as agent_ctor,
        patch(f"{_GOAT}.gen_attacker_turn", return_value=_turn(reply="msg")),
    ):
        strat.run_attack(_case(actor_goal="goal A"), _FakeSession(lambda _m: "r"), max_turns=2)
        strat.run_attack(_case(actor_goal="goal B"), _FakeSession(lambda _m: "r"), max_turns=2)  # no reset

    # Per case: 1 attacker (built once for the case) + 1 judge per scored turn. score 0.0 never
    # early-stops, so max_turns=2 -> 2 scored turns -> 2 judges, + 1 attacker = 3 per case, 6 total.
    assert agent_ctor.call_count == 6
    system_prompts = [call.kwargs["system_prompt"] for call in agent_ctor.call_args_list]
    assert any("goal A" in p for p in system_prompts)
    assert any("goal B" in p for p in system_prompts)


def test_contract_pin_against_crescendo():
    # GOAT and crescendo share the AttackRunResult contract and the metadata core keys the
    # report reads; pin that they agree so a future divergence is caught.
    result = _run(
        _strategy(), _case(), _FakeSession(lambda m: "resp"), turns=[_turn(reply="r1")], score_fn=lambda r: 1.0
    )
    core_keys = {"turns_used", "target_calls", "parse_failures"}
    assert core_keys <= result.metadata.keys()
    assert isinstance(result, AttackRunResult)
    # both strategies expose the same run_attack signature surface
    assert hasattr(CrescendoStrategy, "run_attack")
    assert result.strategy_score == 1.0
