"""Tests for PairStrategy and its module-level helpers."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import BUILTIN_STRATEGIES, PairStrategy
from strands_evals.experimental.redteam.strategies.base import AttackRunResult
from strands_evals.experimental.redteam.strategies.pair import gen_refined_prompt, success_score
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig

_PAIR = "strands_evals.experimental.redteam.strategies.pair"


class _FakeSession:
    """In-test append-only TargetSession: replies via a callable, records tool uses.

    PAIR (single-stream) is invoke-only, so ``snapshot``/``restore`` RAISE: any call would prove the
    strategy is not staying append-only. ``invoke``/``reset`` are the full surface used.
    """

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self._messages: list[str] = []
        self.trace: list[dict] = []

    def invoke(self, message):
        self._messages.append(message)
        return self._reply_fn(message)

    def reset(self):
        self._messages.clear()
        self.trace.clear()

    def snapshot(self):
        raise AssertionError("PAIR must not snapshot (append-only)")

    def restore(self, checkpoint):
        raise AssertionError("PAIR must not restore (append-only)")


def _case(success_criteria: str | None = "leaked the secret", actor_goal: str = "exfiltrate the secret") -> RedTeamCase:
    return RedTeamCase(
        name="p0",
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(
                risk_category="data_exfiltration",
                actor_goal=actor_goal,
                success_criteria=success_criteria,
            )
        ),
        metadata={"strategy": "pair"},
    )


def _strategy(**kwargs) -> PairStrategy:
    """A strategy with its agent builders stubbed so no real LLM is created."""
    strat = PairStrategy(**kwargs)
    strat._build_attacker = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    strat._build_judge = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return strat


def _session(reply_fn):
    """Wrap a ``(message) -> response`` callable as an append-only TargetSession."""
    return _FakeSession(reply_fn)


# ---------------------------------------------------------------------------
# ctor guards
# ---------------------------------------------------------------------------


class TestCtor:
    def test_defaults(self):
        strat = PairStrategy()
        assert strat._max_turns == 5
        assert strat._success_threshold == 0.8
        assert strat.name == "pair"
        assert strat.label == "pair"

    def test_label_override(self):
        assert PairStrategy(label="pair-k10").label == "pair-k10"

    def test_max_turns_below_one_raises(self):
        with pytest.raises(ValueError, match="max_turns"):
            PairStrategy(max_turns=0)

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="success_threshold"):
            PairStrategy(success_threshold=0.0)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="success_threshold"):
            PairStrategy(success_threshold=1.5)

    def test_threshold_one_is_ok(self):
        assert PairStrategy(success_threshold=1.0)._success_threshold == 1.0


# ---------------------------------------------------------------------------
# success_score
# ---------------------------------------------------------------------------


class TestSuccessScore:
    def test_normalizes_raw_one_to_zero(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=1)
        assert success_score(
            lambda: judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r"
        ) == (
            0.0,
            1,
        )

    def test_normalizes_raw_ten_to_one(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=10)
        assert success_score(
            lambda: judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r"
        ) == (
            1.0,
            10,
        )

    def test_clamps_out_of_range_high(self):
        """An unconstrained out-of-range int (11) clamps to 10 -> 1.0, never masked to 0."""
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=11)
        assert success_score(
            lambda: judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r"
        ) == (
            1.0,
            10,
        )

    def test_clamps_out_of_range_low(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=0)
        assert success_score(
            lambda: judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r"
        ) == (
            0.0,
            1,
        )

    def test_parse_failure_defaults_to_zero(self):
        judge = MagicMock()
        judge.return_value.structured_output = None
        assert success_score(
            lambda: judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r"
        ) == (
            0.0,
            None,
        )

    def test_invokes_the_factorys_judge_once_per_score(self):
        """Contract: success_score calls make_judge() once per invocation and uses the returned judge
        (proves factory wiring, NOT production freshness -- that is proven by the Agent-ctor-count
        test test_reused_instance_builds_fresh_agents_each_case_without_reset)."""
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=5)
        make_judge = MagicMock(return_value=judge)
        success_score(make_judge, AttackGoal(risk_category="x", actor_goal="g", success_criteria="c"), "r")
        make_judge.assert_called_once_with()
        judge.assert_called_once()

    def test_uses_success_criteria_as_task_when_present(self):
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=5)
        goal = AttackGoal(risk_category="x", actor_goal="the goal", success_criteria="the criteria")
        success_score(lambda: judge, goal, "resp")
        sent = judge.call_args.args[0]
        assert "the criteria" in sent
        assert "the goal" not in sent

    def test_no_criteria_skips_judge_and_returns_zero(self):
        # No success_criteria -> gate does NOT call the judge (matches crescendo); returns
        # (0.0, None) so the loop never early-stops on a no-criteria case.
        judge = MagicMock()
        goal = AttackGoal(risk_category="x", actor_goal="the goal", success_criteria=None)
        assert success_score(lambda: judge, goal, "resp") == (0.0, None)
        judge.assert_not_called()


# ---------------------------------------------------------------------------
# gen_refined_prompt
# ---------------------------------------------------------------------------


class TestGenRefinedPrompt:
    def test_first_turn_cold_open_uses_goal(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = MagicMock(prompt="first P")
        goal = AttackGoal(risk_category="x", actor_goal="the goal")
        assert gen_refined_prompt(attacker, goal, "", None) == "first P"
        assert "the goal" in attacker.call_args.args[0]

    def test_refine_turn_includes_response_and_raw_score(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = MagicMock(prompt="refined P")
        goal = AttackGoal(risk_category="x", actor_goal="g")
        assert gen_refined_prompt(attacker, goal, "last resp", 7) == "refined P"
        sent = attacker.call_args.args[0]
        assert "last resp" in sent
        assert "7" in sent

    def test_parse_failure_returns_none(self):
        attacker = MagicMock()
        attacker.return_value.structured_output = None
        goal = AttackGoal(risk_category="x", actor_goal="g")
        assert gen_refined_prompt(attacker, goal, "resp", 3) is None

    def test_brace_safe_goal(self):
        """A goal with literal braces does not blow up (str.replace, not .format)."""
        attacker = MagicMock()
        attacker.return_value.structured_output = MagicMock(prompt="P")
        goal = AttackGoal(risk_category="x", actor_goal="emit {payload} now")
        assert gen_refined_prompt(attacker, goal, "", None) == "P"


# ---------------------------------------------------------------------------
# run_attack loop
# ---------------------------------------------------------------------------


def test_returns_attack_run_result_shape():
    strat = _strategy(max_turns=1)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.0, 1)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "reply"), max_turns=10)
    assert isinstance(result, AttackRunResult)
    # Full-dict equality pins the values, not just the keys -- notably parse_failures == 0, the
    # signal of a real (non-swallowed) run.
    assert result.metadata == {"turns_used": 1, "target_calls": 1, "iterations": 1, "parse_failures": 0}
    assert result.pruned_branches == []


def test_stops_on_success_threshold():
    strat = _strategy(max_turns=10, success_threshold=0.8)
    scores = iter([(0.1, 2), (0.9, 9)])
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", side_effect=lambda *_a, **_k: next(scores)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "leaked"), max_turns=10)
    assert result.strategy_succeeded is True
    assert result.strategy_score == 0.9
    assert result.metadata["turns_used"] == 2
    assert result.metadata["target_calls"] == 2
    assert result.metadata["iterations"] == 2


def test_runs_to_cap_when_never_succeeds():
    strat = _strategy(max_turns=3, success_threshold=0.8)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=10)
    assert result.strategy_succeeded is False
    assert result.metadata["turns_used"] == 3
    assert result.metadata["target_calls"] == 3


def test_strategy_score_is_peak_not_last():
    """strategy_score reports the PEAK normalized score across turns, not the last -- a refinement
    run that climbs then dips should surface its closest approach, not the final dip."""
    strat = _strategy(max_turns=4, success_threshold=0.8)
    # climb to 0.6 then dip; none clears 0.8 so the loop runs to cap.
    scores = iter([(0.2, 3), (0.6, 6), (0.3, 4), (0.1, 2)])
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", side_effect=lambda *_a, **_k: next(scores)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=10)
    assert result.strategy_succeeded is False
    assert result.strategy_score == 0.6  # peak, not the last turn's 0.1


def test_max_turns_clamped_by_injected_value():
    """The smaller of the strategy's max_turns and the injected max_turns wins."""
    strat = _strategy(max_turns=10, success_threshold=0.8)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=2)
    assert result.metadata["turns_used"] == 2


def test_ctor_max_turns_wins_when_smaller():
    strat = _strategy(max_turns=2, success_threshold=0.8)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=10)
    assert result.metadata["turns_used"] == 2


def test_no_criteria_runs_to_cap():
    """No success_criteria -> gate returns (0.0, None) without a judge call, never early-stops."""
    strat = _strategy(max_turns=3, success_threshold=0.8)
    case = _case(success_criteria=None)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(case, _session(lambda _m: "engaged"), max_turns=10)
    assert result.strategy_succeeded is False
    assert result.metadata["turns_used"] == 3


def test_prompt_parse_failure_ends_early():
    strat = _strategy(max_turns=5)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value=None),
        patch(f"{_PAIR}.success_score", return_value=(0.0, 1)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "reply"), max_turns=10)
    assert result.metadata["parse_failures"] == 1
    assert result.conversation == []
    assert result.metadata["target_calls"] == 0
    assert result.strategy_succeeded is False


def test_empty_prompt_never_invokes():
    """A blank/whitespace prompt ends the case without ever calling invoke."""
    strat = _strategy(max_turns=5)
    calls = []
    session = _session(lambda m: calls.append(m) or "reply")
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="   "),
        patch(f"{_PAIR}.success_score", return_value=(0.0, 1)),
    ):
        result = strat.run_attack(_case(), session, max_turns=10)
    assert calls == []
    assert result.metadata["target_calls"] == 0
    assert result.metadata["parse_failures"] == 1


def test_empty_response_all_or_nothing():
    """An empty target response ends the case before the turn is recorded:
    target_calls trails turns_used, and conversation/metadata agree on the count."""
    strat = _strategy(max_turns=5, success_threshold=0.8)
    replies = iter(["good reply", "   "])
    score_calls = []
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(
            f"{_PAIR}.success_score",
            side_effect=lambda *_a, **_k: score_calls.append(1) or (0.1, 2),
        ),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: next(replies)), max_turns=10)
    # second (empty) turn invoked but recorded nothing: target_calls trails by one
    assert result.metadata["target_calls"] == 2
    assert result.metadata["iterations"] == 1
    assert result.metadata["turns_used"] == 1
    assert len(result.conversation) == 2  # only the first turn's (attacker, target)
    assert len(score_calls) == 1  # the empty turn was never scored


def test_pruned_branches_always_empty():
    strat = _strategy(max_turns=3, success_threshold=0.8)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=10)
    assert result.pruned_branches == []


def test_target_calls_equal_iterations_no_inflation():
    """Append-only: every target call lands in the conversation (no backtrack inflation)."""
    strat = _strategy(max_turns=3, success_threshold=0.8)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.1, 2)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "engaged"), max_turns=10)
    assert result.metadata["target_calls"] == result.metadata["iterations"]
    assert result.metadata["iterations"] == result.metadata["turns_used"]


def test_ctor_model_takes_precedence():
    strat = PairStrategy(model="ctor-model")
    captured = {}
    strat._build_attacker = lambda goal, model: captured.setdefault("attacker_model", model) or MagicMock()  # type: ignore[method-assign]
    strat._build_judge = lambda model: captured.setdefault("judge_model", model) or MagicMock()  # type: ignore[method-assign]
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value=None),
        patch(f"{_PAIR}.success_score", return_value=(0.0, 1)),
    ):
        strat.run_attack(_case(), _session(lambda _m: "r"), max_turns=1, model="experiment-model")
    assert captured["attacker_model"] == "ctor-model"


def test_reset_is_noop_and_callable():
    """reset() falls back to the base no-op (no per-instance agent state); must stay callable for
    task.py interface compat and must NOT raise."""
    PairStrategy().reset()  # no-op, no AttributeError


def test_reused_instance_builds_fresh_agents_each_case_without_reset():
    """Stateless across cases: the attacker is built once per case (its refinement history is the
    strategy) and the judge is built fresh per scoring call, with no reset() between cases. We patch
    the SDK ``Agent`` ctor (NOT the _build_* methods) so the REAL build bodies run -- reintroducing
    any instance/per-case judge cache OR a cross-case attacker cache drops the count and fails."""
    strat = PairStrategy()

    def make_agent(**_kw):
        # Built Agent scores raw 1 -> normalized 0.0 (never early-stops), so every turn scores and
        # builds a fresh judge. The attacker mock is never invoked (gen_refined_prompt is stubbed).
        agent = MagicMock()
        agent.return_value.structured_output = MagicMock(score=1)
        return agent

    with (
        patch(f"{_PAIR}.Agent", side_effect=make_agent) as agent_ctor,
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
    ):
        strat.run_attack(_case(actor_goal="goal A"), _session(lambda _m: "r"), max_turns=2)
        strat.run_attack(_case(actor_goal="goal B"), _session(lambda _m: "r"), max_turns=2)  # no reset

    # Per case: 1 attacker (built once) + 1 judge per scored turn. Raw score 1 -> 0.0 never
    # early-stops, so max_turns=2 -> 2 scored turns -> 2 judges, + 1 attacker = 3 per case, 6 total.
    assert agent_ctor.call_count == 6
    system_prompts = [call.kwargs["system_prompt"] for call in agent_ctor.call_args_list]
    assert any("goal A" in p for p in system_prompts)
    assert any("goal B" in p for p in system_prompts)


# ---------------------------------------------------------------------------
# registration + contract pin
# ---------------------------------------------------------------------------


def test_pair_not_in_builtin_strategies():
    assert "pair" not in BUILTIN_STRATEGIES


def test_contract_pin_against_crescendo():
    """PAIR and Crescendo share the comparable core metadata keys + AttackRunResult shape.

    The report reads turns_used/target_calls/parse_failures across strategies; pin that
    the shared core is present (each strategy adds its own extras: iterations here,
    backtracks for Crescendo).
    """
    strat = _strategy(max_turns=1)
    with (
        patch(f"{_PAIR}.gen_refined_prompt", return_value="p"),
        patch(f"{_PAIR}.success_score", return_value=(0.0, 1)),
    ):
        result = strat.run_attack(_case(), _session(lambda _m: "reply"), max_turns=10)
    core = {"turns_used", "target_calls", "parse_failures"}
    assert core <= set(result.metadata)
    assert isinstance(result, AttackRunResult)
