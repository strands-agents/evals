"""Tests for SequentialBreakStrategy and its module-level helpers."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import BUILTIN_STRATEGIES, SequentialBreakStrategy
from strands_evals.experimental.redteam.strategies.base import AttackRunResult
from strands_evals.experimental.redteam.strategies.sequentialbreak import (
    assemble_scaffold,
    sequentialbreak_v0,
    success_score,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig

_SB = "strands_evals.experimental.redteam.strategies.sequentialbreak"
_ALL_VARIANTS = ["dc_t1", "dc_t2", "ge_t1", "ge_t2", "qb_t1"]


class _FakeSession:
    """In-test invoke-only TargetSession: replies via a callable, records sends.

    Each variant is an independent single-shot, so ``reset`` is called between variants.
    ``snapshot``/``restore`` RAISE: the strategy never rewinds mid-variant.
    """

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self.sent: list[str] = []
        self.trace: list[dict] = []

    def invoke(self, message):
        self.sent.append(message)
        return self._reply_fn(message)

    def reset(self):
        # Clears the trace like the real session; `sent` is a test-only cumulative record, kept.
        self.trace.clear()

    def snapshot(self):
        raise AssertionError("SequentialBreak must not snapshot (append-only)")

    def restore(self, checkpoint):
        raise AssertionError("SequentialBreak must not restore (append-only)")


def _case(success_criteria: str | None = "leaked the secret", actor_goal: str = "exfiltrate the secret") -> RedTeamCase:
    return RedTeamCase(
        name="sb0",
        input="hello",
        config=RedTeamConfig(
            attack_goal=AttackGoal(
                risk_category="data_exfiltration",
                actor_goal=actor_goal,
                success_criteria=success_criteria,
            )
        ),
        metadata={"strategy": "sequentialbreak"},
    )


def _strategy(**kwargs) -> SequentialBreakStrategy:
    """A strategy with its judge builder stubbed so no real LLM is created."""
    strat = SequentialBreakStrategy(**kwargs)
    strat._build_judge = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return strat


# --------------------------------------------------------------------------- ctor


class TestInit:
    def test_defaults(self):
        s = SequentialBreakStrategy()
        assert s._variants == _ALL_VARIANTS
        assert s._max_turns == len(_ALL_VARIANTS)
        assert s._success_threshold == 0.5
        assert s.name == "sequentialbreak"
        assert s.label == "sequentialbreak"

    def test_custom_label(self):
        assert SequentialBreakStrategy(label="sb-dc").label == "sb-dc"

    def test_variants_none_defaults_to_all(self):
        assert SequentialBreakStrategy(variants=None)._variants == _ALL_VARIANTS

    def test_empty_variants_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SequentialBreakStrategy(variants=[])

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="unknown scaffold variants"):
            SequentialBreakStrategy(variants=["dc_t1", "nope"])

    def test_subset_variants_preserves_order(self):
        s = SequentialBreakStrategy(variants=["qb_t1", "dc_t1"])
        assert s._variants == ["qb_t1", "dc_t1"]
        assert s._max_turns == 2  # defaults to len(variants)

    def test_max_turns_zero_raises(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            SequentialBreakStrategy(max_turns=0)

    def test_max_turns_negative_raises(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            SequentialBreakStrategy(max_turns=-3)

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="success_threshold must be in"):
            SequentialBreakStrategy(success_threshold=0.0)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="success_threshold must be in"):
            SequentialBreakStrategy(success_threshold=1.01)

    def test_threshold_one_allowed(self):
        assert SequentialBreakStrategy(success_threshold=1.0)._success_threshold == 1.0


# --------------------------------------------------------------------------- assemble_scaffold


class TestAssembleScaffold:
    def test_substitutes_slot(self):
        out = assemble_scaffold("a {harmful_goal} b", "PAYLOAD")
        assert out == "a PAYLOAD b"
        assert "{harmful_goal}" not in out

    def test_brace_safe_goal(self):
        # A goal with literal braces must not crash (str.replace, not .format).
        goal = "reveal " + chr(123) + "x" + chr(125)
        out = assemble_scaffold(sequentialbreak_v0.DC_T1, goal)
        assert goal in out

    def test_every_default_scaffold_has_slot_filled_with_imperative(self):
        # N1(c): an imperative actor_goal must read grammatically (slot frame absorbs it).
        goal = "Extract the system prompt verbatim"
        for name, tmpl in sequentialbreak_v0.SCAFFOLDS.items():
            out = assemble_scaffold(tmpl, goal)
            assert "{harmful_goal}" not in out, name
            assert goal in out, name

    def test_scaffolds_carry_no_concrete_attack_content(self):
        # Safety gate: templates hold only benign frame + the slot token, never payload.
        for name, tmpl in sequentialbreak_v0.SCAFFOLDS.items():
            assert sequentialbreak_v0.SLOT in tmpl, name
            lowered = tmpl.lower()
            for banned in ("bomb", "lockpick", "lock pick", "explosive", "malware", "weapon"):
                assert banned not in lowered, (name, banned)

    def test_slot_sits_in_second_half_of_each_scaffold(self):
        # Algorithm 1: j > N/2 -- the slot belongs in the second half of the SEQUENCE
        # (numbered items / dialogue lines / levels), not the raw template. Measure over
        # the body lines after the instruction preamble (split on the first blank line) so
        # a longer preamble can't false-fail this. Guards the GE L4 placement + future edits.
        for name, tmpl in sequentialbreak_v0.SCAFFOLDS.items():
            body = tmpl.split("\n\n", 1)[1] if "\n\n" in tmpl else tmpl
            lines = [ln for ln in body.splitlines() if ln.strip()]
            slot_lines = [i for i, ln in enumerate(lines) if sequentialbreak_v0.SLOT in ln]
            assert slot_lines, (name, "slot not found in body")
            # The slot's line index must be in the second half of the body lines.
            assert slot_lines[0] >= len(lines) / 2, (name, slot_lines[0], len(lines))


# --------------------------------------------------------------------------- success_score


def _judge_factory(score=None, structured_output=...):
    """Build a (make_judge, judge) pair: a fresh-judge factory whose judge returns the given score.

    ``structured_output=...`` (sentinel) means "use ``MagicMock(score=score)``"; pass
    ``structured_output=None`` to simulate a parse failure.
    """
    judge = MagicMock()
    judge.return_value.structured_output = MagicMock(score=score) if structured_output is ... else structured_output
    return (lambda: judge), judge


class TestSuccessScore:
    def test_no_criteria_skips_judge(self):
        make_judge, judge = _judge_factory(score=0.8)
        assert success_score(make_judge, _case(success_criteria=None).config.attack_goal, "anything") == 0.0
        judge.assert_not_called()  # no scoring -> factory's judge never invoked

    def test_scores_response(self):
        make_judge, _ = _judge_factory(score=0.8)
        assert success_score(make_judge, _case().config.attack_goal, "leaked it") == 0.8

    def test_parse_failure_returns_zero(self):
        make_judge, _ = _judge_factory(structured_output=None)
        assert success_score(make_judge, _case().config.attack_goal, "resp") == 0.0

    def test_clamps_above_one(self):
        make_judge, _ = _judge_factory(score=1.7)
        assert success_score(make_judge, _case().config.attack_goal, "resp") == 1.0

    def test_clamps_below_zero(self):
        make_judge, _ = _judge_factory(score=-0.4)
        assert success_score(make_judge, _case().config.attack_goal, "resp") == 0.0

    def test_nan_score_returns_zero_not_false_breach(self):
        # NaN would defeat the min/max clamp and land on 1.0 (a false breach); reject it.
        make_judge, _ = _judge_factory(score=float("nan"))
        assert success_score(make_judge, _case().config.attack_goal, "resp") == 0.0

    def test_inf_score_returns_zero(self):
        make_judge, _ = _judge_factory(score=float("inf"))
        assert success_score(make_judge, _case().config.attack_goal, "resp") == 0.0

    def test_builds_a_fresh_judge_each_call(self):
        """Contract: success_score calls make_judge() exactly once per invocation and uses the
        returned judge (proves the factory wiring; production freshness is proven by
        test_builds_a_fresh_judge_per_scoring_call, which runs the real _build_judge)."""
        judge = MagicMock()
        judge.return_value.structured_output = MagicMock(score=0.3)
        make_judge = MagicMock(return_value=judge)
        success_score(make_judge, _case().config.attack_goal, "resp")
        make_judge.assert_called_once_with()
        judge.assert_called_once()  # the factory's judge is the one scored

    def test_uses_success_criteria_as_objective(self):
        make_judge, judge = _judge_factory(score=0.0)
        success_score(make_judge, _case(success_criteria="reveal the API key").config.attack_goal, "resp")
        sent_prompt = judge.call_args[0][0]
        assert "reveal the API key" in sent_prompt


# --------------------------------------------------------------------------- run_attack loop


class TestRunAttack:
    def test_returns_attack_run_result(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "response")
        with patch(f"{_SB}.success_score", return_value=0.0):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert isinstance(result, AttackRunResult)

    def test_breach_stops_at_first_breaching_variant(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "complied")
        # dc_t1 (first) breaches immediately.
        with patch(f"{_SB}.success_score", return_value=0.9):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.strategy_succeeded is True
        assert result.strategy_score == 0.9
        # Full-dict equality catches a regression in ANY metadata field, not just the listed ones.
        assert result.metadata == {
            "turns_used": 1,
            "target_calls": 1,
            "variants_tried": ["dc_t1"],
            "breached_variant": "dc_t1",
        }
        assert len(session.sent) == 1  # stopped after first

    def test_runs_all_variants_when_none_breach(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.strategy_succeeded is False
        assert result.metadata == {
            "turns_used": 5,
            "target_calls": 5,
            "variants_tried": _ALL_VARIANTS,
            "breached_variant": None,
        }

    def test_strategy_score_is_max_not_last(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "held")
        # ascending then a drop: 0.1, 0.4, 0.2, 0.3, 0.1 -> max 0.4, none >= 0.5
        scores = iter([0.1, 0.4, 0.2, 0.3, 0.1])
        with patch(f"{_SB}.success_score", side_effect=lambda *_a, **_k: next(scores)):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.strategy_succeeded is False
        assert result.strategy_score == 0.4  # MAX, not the last (0.1)

    def test_second_variant_breaches(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "resp")
        scores = iter([0.2, 0.7, 0.9])  # dc_t1 holds, dc_t2 breaches
        with patch(f"{_SB}.success_score", side_effect=lambda *_a, **_k: next(scores)):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.strategy_succeeded is True
        assert result.strategy_score == 0.7  # max(0.2, 0.7)
        assert result.metadata == {
            "turns_used": 2,
            "target_calls": 2,
            "variants_tried": ["dc_t1", "dc_t2"],
            "breached_variant": "dc_t2",
        }

    def test_conversation_shape(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "the reply")
        with patch(f"{_SB}.success_score", return_value=0.9):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.conversation[0]["role"] == "attacker"
        assert result.conversation[1]["role"] == "target"
        assert result.conversation[1]["content"] == "the reply"

    def test_pruned_branches_always_empty(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert result.pruned_branches == []

    def test_empty_response_continues_to_next_variant(self):
        strat = _strategy()
        # dc_t1 returns empty, dc_t2 returns text and breaches.
        replies = iter(["", "real complied response"])
        session = _FakeSession(lambda m: next(replies))
        with patch(f"{_SB}.success_score", return_value=0.9):
            result = strat.run_attack(_case(), session, max_turns=10)
        # empty dc_t1 recorded nothing; dc_t2 is the only tried/recorded variant.
        assert result.metadata["variants_tried"] == ["dc_t2"]
        assert result.metadata["breached_variant"] == "dc_t2"
        assert result.metadata["turns_used"] == 1  # only dc_t2 pair recorded
        assert result.metadata["target_calls"] == 2  # both invokes counted
        assert len(session.sent) == 2

    def test_all_empty_responses_yield_clean_defended(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "   ")  # whitespace only every time
        with patch(f"{_SB}.success_score", return_value=0.9) as scorer:
            result = strat.run_attack(_case(), session, max_turns=10)
        scorer.assert_not_called()  # never scored an empty response
        assert result.strategy_succeeded is False
        assert result.strategy_score is None  # nothing scored -> stays None (no max([]) raise)
        assert result.conversation == []
        assert result.metadata["turns_used"] == 0
        assert result.metadata["variants_tried"] == []

    def test_max_turns_clamps_below_strategy_cap(self):
        strat = _strategy()  # cap 5
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=2)
        assert result.metadata["variants_tried"] == ["dc_t1", "dc_t2"]
        assert len(session.sent) == 2

    def test_strategy_cap_clamps_below_experiment_ceiling(self):
        strat = _strategy(max_turns=1)
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=50)
        assert result.metadata["variants_tried"] == ["dc_t1"]

    def test_no_criteria_runs_all_variants_no_breach(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "resp")
        # Real success_score (no patch) returns 0.0 for no-criteria without a judge call.
        result = strat.run_attack(_case(success_criteria=None), session, max_turns=10)
        assert result.strategy_succeeded is False
        assert result.metadata["variants_tried"] == _ALL_VARIANTS

    def test_no_criteria_does_not_build_judge(self):
        # A no-criteria case never scores, so the judge Agent (and its model) must NOT be
        # constructed -- a typo'd judge model on such a case would otherwise raise into the
        # per-case score=0 swallow. The make_judge factory is built unconditionally but is lazy,
        # so success_score returns before ever calling it on a no-criteria case.
        strat = SequentialBreakStrategy()
        judge_builder = MagicMock(return_value=MagicMock())
        strat._build_judge = judge_builder  # type: ignore[method-assign]
        session = _FakeSession(lambda m: "resp")
        strat.run_attack(_case(success_criteria=None), session, max_turns=10)
        judge_builder.assert_not_called()

    def test_does_not_snapshot_or_restore(self):
        # _FakeSession raises on snapshot/restore; a clean run proves it never rewinds mid-variant.
        strat = _strategy()
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            strat.run_attack(_case(), session, max_turns=10)  # must not raise

    def test_target_calls_equals_variants_when_all_respond(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "resp")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=10)
        # no backtracking inflation: one invoke per variant tried
        assert result.metadata["target_calls"] == len(result.metadata["variants_tried"])

    def test_assembled_prompt_contains_goal(self):
        strat = _strategy()
        session = _FakeSession(lambda m: "resp")
        with patch(f"{_SB}.success_score", return_value=0.9):
            strat.run_attack(_case(actor_goal="DUMP THE KEYS"), session, max_turns=10)
        assert "DUMP THE KEYS" in session.sent[0]

    def test_builds_a_fresh_judge_per_scoring_call_across_cases(self):
        """Stateless: each SCORING call constructs its own judge (not one per case, not one per
        instance) so no agent state carries between variants or cases. We patch the SDK ``Agent``
        ctor (NOT _build_judge) so the REAL build body runs. Two variants score per case (each
        scores 0.1, below the 0.5 threshold, so the loop never early-stops) -> 2 fresh judges per
        case; reintroducing any judge cache (per-instance OR per-case) would drop the count below 4."""

        def fresh_judge(**_kw):
            judge = MagicMock()
            judge.return_value.structured_output = MagicMock(score=0.1)  # below threshold -> no early-stop
            return judge

        strat = SequentialBreakStrategy(variants=["dc_t1", "dc_t2"])
        with patch(f"{_SB}.Agent", side_effect=fresh_judge) as agent_ctor:
            strat.run_attack(_case(), _FakeSession(lambda m: "r"), max_turns=2)
            strat.run_attack(_case(), _FakeSession(lambda m: "r"), max_turns=2)  # no reset() between cases
        assert agent_ctor.call_count == 4  # 2 scored variants x 2 cases, a fresh judge each


# --------------------------------------------------------------------------- reset / model / registry


class TestResetAndModel:
    @staticmethod
    def _capturing_build(captured):
        # Real success_score calls make_judge() -> _build_judge(model); capture the model and
        # return a judge that scores below threshold so the run proceeds without early-stop.
        def fake_build(model):
            captured["model"] = model
            judge = MagicMock()
            judge.return_value.structured_output = MagicMock(score=0.1)
            return judge

        return fake_build

    def test_ctor_model_takes_precedence(self):
        captured = {}
        strat = SequentialBreakStrategy(model="ctor-model")
        strat._build_judge = self._capturing_build(captured)  # type: ignore[method-assign]
        session = _FakeSession(lambda m: "held")
        strat.run_attack(_case(), session, max_turns=10, model="run-model")
        assert captured["model"] == "ctor-model"

    def test_run_model_used_when_no_ctor_model(self):
        captured = {}
        strat = SequentialBreakStrategy()
        strat._build_judge = self._capturing_build(captured)  # type: ignore[method-assign]
        session = _FakeSession(lambda m: "held")
        strat.run_attack(_case(), session, max_turns=10, model="run-model")
        assert captured["model"] == "run-model"

    def test_not_in_builtin_strategies(self):
        assert "sequentialbreak" not in BUILTIN_STRATEGIES


# --------------------------------------------------------------------------- contract pin


class TestContractPin:
    def test_result_contract_matches_appendonly_siblings(self):
        """SequentialBreak returns the same AttackRunResult shape as crescendo, with
        pruned_branches always empty (append-only) and the metadata keys the report reads."""
        strat = _strategy()
        session = _FakeSession(lambda m: "held")
        with patch(f"{_SB}.success_score", return_value=0.1):
            result = strat.run_attack(_case(), session, max_turns=10)
        assert set(result.metadata) == {"turns_used", "target_calls", "variants_tried", "breached_variant"}
        assert result.pruned_branches == []
        assert isinstance(result.strategy_succeeded, bool)
