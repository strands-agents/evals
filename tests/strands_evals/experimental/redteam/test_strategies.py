"""Tests for AttackStrategy implementations."""

from unittest.mock import MagicMock, patch

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import BUILTIN_STRATEGIES, PromptStrategy
from strands_evals.experimental.redteam.strategies.base import AttackRunResult
from strands_evals.experimental.redteam.strategies.target_session import TargetCheckpoint
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _FakeSession:
    """A TargetSession for forward-only strategy tests.

    PromptStrategy never backtracks, but the fake implements the full protocol
    surface (snapshot/restore too) so it stays a valid TargetSession and can't
    silently diverge from the contract the strategy is handed.
    """

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self.trace: list[dict] = []

    def invoke(self, message):
        return self._reply_fn(message)

    def reset(self):
        self.trace.clear()

    def snapshot(self):
        return TargetCheckpoint(agent_snapshot=None, trace_len=len(self.trace))

    def restore(self, checkpoint):
        del self.trace[checkpoint.trace_len :]


def _case(name: str = "c0") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="prompt_injection", actor_goal="goal")),
    )


def test_redteam_config_is_strategy_agnostic():
    """RedTeamConfig no longer carries strategy/template; cases are strategy-agnostic."""
    config = RedTeamConfig(attack_goal=AttackGoal(risk_category="prompt_injection", actor_goal="g"))
    assert not hasattr(config, "strategy")
    assert not hasattr(config, "system_prompt_template")
    assert "strategy" not in RedTeamConfig.model_fields
    assert "system_prompt_template" not in RedTeamConfig.model_fields


def test_prompt_strategy_label_defaults_to_name():
    strategy = BUILTIN_STRATEGIES["gradual_escalation"]
    assert strategy.label == strategy.name == "gradual_escalation"


def test_prompt_strategy_label_override():
    strategy = PromptStrategy("gradual_escalation", "tmpl {max_turns}", label="grad-5")
    assert strategy.name == "gradual_escalation"
    assert strategy.label == "grad-5"


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_run_attack_drives_loop(mock_simulator_cls):
    """PromptStrategy.run_attack drives the ActorSimulator loop via the injected call_target."""
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, True, False]
    act_result = MagicMock()
    act_result.structured_output.message = "follow-up attack"
    mock_sim.act.return_value = act_result
    mock_simulator_cls.return_value = mock_sim

    strategy = PromptStrategy("gradual_escalation", "prompt {actor_profile} {max_turns}")
    session = _FakeSession(lambda _m: "target reply")

    result = strategy.run_attack(_case(), session, max_turns=7)

    assert isinstance(result, AttackRunResult)
    assert len(result.conversation) == 4
    assert result.conversation[0]["role"] == "attacker"
    assert result.conversation[1]["content"] == "target reply"
    # max_turns placeholder substituted into the simulator's system prompt
    template = mock_simulator_cls.call_args.kwargs["system_prompt_template"]
    assert "{max_turns}" not in template and "7" in template


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_ctor_max_turns_caps_below_ceiling(mock_simulator_cls):
    """The strategy's own max_turns wins when smaller than the injected ceiling."""
    mock_sim = MagicMock()
    mock_sim.has_next.return_value = False
    mock_simulator_cls.return_value = mock_sim

    strategy = PromptStrategy("gradual_escalation", "p {max_turns}", max_turns=3)
    strategy.run_attack(_case(), _FakeSession(lambda _m: "r"), max_turns=50)  # ceiling 50

    assert mock_simulator_cls.call_args.kwargs["max_turns"] == 3


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_run_attack_handles_target_exception(mock_simulator_cls):
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, False]
    mock_simulator_cls.return_value = mock_sim

    def broken(_msg):
        raise RuntimeError("boom")

    result = PromptStrategy("gradual_escalation", "p {max_turns}").run_attack(
        _case(), _FakeSession(broken), max_turns=3
    )
    assert "[Error: boom]" in result.conversation[1]["content"]
