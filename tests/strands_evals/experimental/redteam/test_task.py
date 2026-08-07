"""Tests for _build_attacker_task."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies.base import AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.task import (
    MAX_ALLOWED_TURNS,
    _build_attacker_task,
)
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig


class _StubStrategy(AttackStrategy):
    """Records the injected target session and returns a canned result."""

    def __init__(self, label="stub", *, result=None, calls=None):
        super().__init__(label=label)
        self._result = result
        self.reset_count = 0
        self.calls = calls if calls is not None else []
        self.received_max_turns: int | None = None

    @property
    def name(self) -> str:
        return "stub"

    def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs) -> AttackRunResult:
        self.received_max_turns = max_turns
        # exercise the injected session so trace/output wiring is covered
        reply = target_session.invoke("ping")
        self.calls.append(reply)
        return self._result or AttackRunResult(
            conversation=[{"role": "attacker", "content": "ping"}, {"role": "target", "content": reply}],
            metadata={"turns_used": 1},
        )

    def reset(self) -> None:
        self.reset_count += 1


def _case(name: str = "c0", label: str = "stub") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="prompt_injection", actor_goal="goal")),
        metadata={"strategy": label},
    )


def _by_label(*strategies: AttackStrategy) -> dict[str, AttackStrategy]:
    return {s.label: s for s in strategies}


class _FakeSession:
    """Minimal TargetSession a test can pass straight to _build_attacker_task."""

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self.trace: list[dict] = []

    def invoke(self, message):
        return self._reply_fn(message)

    def reset(self):
        self.trace.clear()

    def snapshot(self):
        return None

    def restore(self, checkpoint):
        pass


def test_run_attack_receives_max_allowed_turns_ceiling():
    """task_fn passes MAX_ALLOWED_TURNS as the hard ceiling; the strategy owns its budget."""
    from strands_evals.experimental.redteam.task import MAX_ALLOWED_TURNS

    strat = _StubStrategy()
    _build_attacker_task(_FakeSession(lambda _msg: "ok"), _by_label(strat))(_case())
    assert strat.received_max_turns == MAX_ALLOWED_TURNS


def test_task_fn_calls_run_attack_and_maps_result():
    strat = _StubStrategy()
    task = _build_attacker_task(_FakeSession(lambda _msg: "target reply"), _by_label(strat))

    result = task(_case())

    # Only the keys the base Experiment reads are returned; run stats flow via run_meta.
    assert set(result) == {"output", "trajectory"}
    assert result["output"][1]["content"] == "target reply"
    assert strat.reset_count == 1


def test_task_fn_records_run_stats_into_run_meta():
    """Strategy metadata reaches the experiment via run_meta, not the returned dict."""
    strat = _StubStrategy()
    run_meta: dict[str, dict] = {}
    task = _build_attacker_task(_FakeSession(lambda _msg: "target reply"), _by_label(strat), run_meta=run_meta)

    task(_case("c0"))

    assert run_meta["c0"]["turns_used"] == 1


def test_task_fn_resets_strategy_each_case():
    strat = _StubStrategy()
    task = _build_attacker_task(_FakeSession(lambda _msg: "ok"), _by_label(strat))
    task(_case("c0"))
    task(_case("c1"))
    assert strat.reset_count == 2


def test_task_fn_missing_strategy_label_raises():
    task = _build_attacker_task(_FakeSession(lambda _msg: "ok"), _by_label(_StubStrategy()))
    case = _case()
    case.metadata = {}  # no strategy label
    with pytest.raises(ValueError, match="strategy"):
        task(case)


def test_task_fn_unknown_strategy_label_raises():
    task = _build_attacker_task(_FakeSession(lambda _msg: "ok"), _by_label(_StubStrategy(label="stub")))
    with pytest.raises(ValueError, match="missing"):
        task(_case(label="missing"))


def test_task_fn_session_trace_becomes_trajectory():
    """The session's tool trace is returned as the trajectory."""

    def reply(_msg):
        session.trace.append({"name": "lookup", "input": {"id": "1"}})
        return "reply"

    session = _FakeSession(reply)
    task = _build_attacker_task(session, _by_label(_StubStrategy()))
    result = task(_case())

    assert result["trajectory"] == [{"name": "lookup", "input": {"id": "1"}}]


def test_task_fn_trajectory_is_a_copy_not_the_live_trace():
    """A passed-in session is reused across cases, and each case's reset() clears
    its trace in place. The returned trajectory must be a copy, so a later case's
    reset() does not retroactively empty a prior case's recorded trajectory."""

    def reply(_msg):
        session.trace.append({"name": "lookup", "input": {}})
        return "reply"

    session = _FakeSession(reply)
    task = _build_attacker_task(session, _by_label(_StubStrategy()))

    result0 = task(_case("c0"))
    assert result0["trajectory"] == [{"name": "lookup", "input": {}}]
    # second case resets (clears) the same session's trace in place ...
    task(_case("c1"))
    # ... but case 0's recorded trajectory must be untouched, and a distinct object
    assert result0["trajectory"] == [{"name": "lookup", "input": {}}]
    assert result0["trajectory"] is not session.trace


def test_task_fn_bare_callable_target_raises_type_error():
    """A bare callable can't snapshot/restore, so it is rejected up front."""
    task = _build_attacker_task(lambda _msg: "reply", _by_label(_StubStrategy()))
    with pytest.raises(TypeError, match="TargetSession"):
        task(_case())


def test_task_fn_session_missing_trace_raises_type_error():
    """A session with all four methods but no `trace` is rejected up front, not
    accepted and then crashed on `session.trace` (which the engine would swallow
    into a misleading 'defended' verdict)."""

    class _NoTrace:
        def invoke(self, _m):
            return "r"

        def reset(self):
            pass

        def snapshot(self):
            return None

        def restore(self, _c):
            pass

    task = _build_attacker_task(_NoTrace(), _by_label(_StubStrategy()))
    with pytest.raises(TypeError, match="TargetSession"):
        task(_case())


def test_task_fn_resets_agent_to_clean_baseline_per_case():
    """Agent target is rolled back to its as-constructed baseline before each case.

    The baseline is snapshotted ONCE at task-build time, then every case resets via
    load_snapshot(baseline) -- covering all session fields, not just messages."""
    from strands import Agent

    agent = MagicMock(spec=Agent)
    agent.messages = MagicMock()
    agent.messages.__len__.return_value = 0
    agent.return_value = "ok"
    baseline = agent.take_snapshot.return_value
    task = _build_attacker_task(agent, _by_label(_StubStrategy()))

    # baseline captured exactly once, before any case
    assert agent.take_snapshot.call_count == 1

    task(_case("c0"))
    task(_case("c1"))

    # each case rolls the target back to that same baseline
    assert agent.load_snapshot.call_count == 2
    agent.load_snapshot.assert_called_with(baseline)


def test_max_allowed_turns_constant_present():
    assert MAX_ALLOWED_TURNS >= 50


def test_parallel_task_fn_calls_factory_per_case():
    """`parallel=True` with `agent_factory` builds a fresh target for every case."""
    factory_calls: list[_FakeSession] = []

    def factory():
        sess = _FakeSession(lambda _msg: "ok")
        factory_calls.append(sess)
        return sess

    task = _build_attacker_task(
        agent=None,
        by_label=_by_label(_StubStrategy()),
        agent_factory=factory,
        parallel=True,
    )
    task(_case("c0"))
    task(_case("c1"))
    task(_case("c2"))
    assert len(factory_calls) == 3


def test_parallel_task_fn_requires_agent_factory():
    """Parallel runs always require an explicit `agent_factory` -- there is no deepcopy fallback.

    Strands targets carry non-deepcopyable client state (the default `BedrockModel` holds an
    httplib pool with thread locks), so the runner refuses parallel without a factory regardless
    of whether `agent` is an `Agent`, `MultiAgentBase`, or a custom `TargetSession`. Surfacing
    the requirement at build time beats crashing mid-run.
    """
    from strands import Agent

    real_agent = Agent(model=None, callback_handler=None)
    with pytest.raises(TypeError, match="agent_factory"):
        _build_attacker_task(
            agent=real_agent,
            by_label=_by_label(_StubStrategy()),
            parallel=True,
        )

    sess = _FakeSession(lambda _msg: "ok")
    with pytest.raises(TypeError, match="agent_factory"):
        _build_attacker_task(
            agent=sess,
            by_label=_by_label(_StubStrategy()),
            parallel=True,
        )


def test_factory_path_skips_baseline_snapshot():
    """When `agent_factory` is supplied, the build step never calls `take_snapshot` on the source
    `agent` -- the factory is the source of truth for clean state."""
    from strands import Agent

    agent = MagicMock(spec=Agent)
    agent.messages = MagicMock()
    agent.messages.__len__.return_value = 0

    def factory():
        target = MagicMock(spec=Agent)
        target.messages = MagicMock()
        target.messages.__len__.return_value = 0
        target.return_value = "ok"
        return target

    _build_attacker_task(
        agent=agent,
        by_label=_by_label(_StubStrategy()),
        agent_factory=factory,
    )
    # the factory path neither captures a baseline nor mutates the seed agent.
    assert agent.take_snapshot.call_count == 0


def test_task_fn_routes_multi_agent_base_to_multi_agent_session():
    """A MultiAgentBase target is wrapped in StrandsMultiAgentSession (not
    StrandsAgentSession) and the baseline is captured once at build time."""
    from strands.multiagent.base import MultiAgentBase

    from strands_evals.experimental.redteam.strategies.target_session import (
        StrandsMultiAgentSession,
        _MultiAgentSnapshot,
    )

    class _FakeOrch(MultiAgentBase):
        id = "root"

        def __init__(self):
            self.nodes = {}  # no leaves -- the strategy stub never invokes
            self.serialize_calls = 0
            self.deserialize_calls = 0

        async def invoke_async(self, task, invocation_state=None, **kwargs):
            raise NotImplementedError

        def __call__(self, task, invocation_state=None, **kwargs):
            return MagicMock(results={}, __str__=lambda self=None: "ok")

        def serialize_state(self):
            self.serialize_calls += 1
            return {"type": "fake"}

        def deserialize_state(self, payload):
            self.deserialize_calls += 1

    root = _FakeOrch()
    captured: dict[str, Any] = {}

    class _CapturingStrategy(_StubStrategy):
        def run_attack(self, case, target_session, *, max_turns, model=None, **kwargs):
            captured["session"] = target_session
            return AttackRunResult(conversation=[], metadata={"turns_used": 0})

    task = _build_attacker_task(root, _by_label(_CapturingStrategy()))
    # baseline serialize_state captured once before any case runs
    assert root.serialize_calls == 1
    task(_case("c0"))
    # routed to the multi-agent session
    assert isinstance(captured["session"], StrandsMultiAgentSession)
    # baseline composite has the expected shape
    assert isinstance(captured["session"]._baseline, _MultiAgentSnapshot)
