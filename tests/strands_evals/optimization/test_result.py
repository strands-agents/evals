from strands_evals.optimization import AgentSpend, RunCost, Spend, Usage
from strands_evals.optimization.result import _cost_rows, _run_cost


def _usage(tokens, calls=1):
    return Usage(
        tokens=tokens,
        input_tokens=tokens,
        output_tokens=0,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=calls,
        seconds=1.0,
        models=[],
    ).model_dump()


def _attempt(status, *, usage=0, judges=0, rollouts=1):
    return {
        "status": status,
        "error_type": None,
        "error": None,
        "usage": _usage(usage, calls=int(usage > 0)),
        "judge_usage": _usage(judges, calls=int(judges > 0)),
        "pass_index": 0,
        "rollouts": rollouts,
    }


def _trial(split, status, *, usage, judges=None, errors=None, previous=()):
    return {
        "split": split,
        "status": status,
        "usage": _usage(usage),
        "judge_usage": {name: _usage(tokens) for name, tokens in (judges or {}).items()},
        "evaluator_errors": dict(errors or {}),
        "previous_attempts": list(previous),
    }


def test_the_agent_and_the_judges_count_every_attempt_each_in_its_own_role():
    trials = [
        # A re-run pass replaced an infrastructure failure: the agent ran twice.
        _trial(
            "train", "completed", usage=100, judges={"J": 10}, previous=[_attempt("infrastructure_invalid", usage=30)]
        ),
        # Throttled once inside the attempt, then completed with one evaluator failed.
        _trial(
            "train",
            "completed",
            usage=60,
            judges={"J": 10, "K": 4},
            errors={"K": "ValueError"},
            previous=[_attempt("infrastructure_invalid", usage=5, rollouts=0)],
        ),
        # A judge-only re-run: the output's usage stays on the new record; the replaced attempt keeps its failed
        # judge's spend only.
        _trial(
            "validation", "completed", usage=50, judges={"J": 10, "K": 5}, previous=[_attempt("completed", judges=7)]
        ),
        # Every evaluator failed, and failed again when judged again.
        _trial(
            "validation",
            "evaluator_invalid",
            usage=40,
            judges={"J": 3, "K": 3},
            errors={"J": "ValueError", "K": "ValueError"},
            previous=[_attempt("evaluator_invalid", judges=6)],
        ),
        _trial("test", "agent_failure", usage=20),
        # Infrastructure failed, and failed again on its re-run pass.
        _trial("test", "infrastructure_invalid", usage=15, previous=[_attempt("infrastructure_invalid", usage=15)]),
    ]
    cost = _run_cost(trials, [], blocks=[])

    assert {split: (s.usage.tokens, s.trials, s.attempts, s.failed) for split, s in cost.agent.items()} == {
        "train": (100 + 30 + 60 + 5, 2, 4, 2),
        "validation": (50 + 40, 2, 2, 0),  # judge-only re-runs don't run the agent
        "test": (20 + 15 + 15, 2, 3, 3),
    }
    assert {split: (s.usage.tokens, s.failed) for split, s in cost.judges.items()} == {
        "train": (10 + 10 + 4, 1),
        "validation": (10 + 5 + 7 + 3 + 3 + 6, 1 + 2),
        "test": (0, 0),
    }
    assert cost.agent["train"].usage.model_calls == 4  # two trials, each with an earlier attempt
    assert cost.proposers == {}


def test_each_block_gets_its_proposers_spend_and_failed_attempts():
    events = [
        {"event": "proposer_usage", "block": "text", "usage": _usage(10)},  # the returned patches' usage
        {"event": "proposer_retry", "block": "text", "error_type": "ValueError", "error": "bad"},
        {"event": "proposer_usage", "block": "text", "usage": _usage(5)},  # spend on no returned patch
        {"event": "proposer_error", "block": "text", "error_type": "ValueError", "error": "bad"},
        {"event": "proposer_usage", "block": "structure", "usage": _usage(3)},
        {"event": "proposer_retry", "block": "other", "error_type": "ValueError", "error": "bad"},
        {"event": "trial", "record": {}},
    ]
    cost = _run_cost([], events, blocks=["numbers", "structure", "text"])

    # Every block, in the search's order, then a block only the events name.
    assert {block: (s.usage.tokens, s.failed) for block, s in cost.proposers.items()} == {
        "numbers": (0, 0),
        "structure": (3, 0),
        "text": (15, 2),
        "other": (0, 1),
    }
    assert list(cost.proposers) == ["numbers", "structure", "text", "other"]
    assert cost.total.tokens == 18
    assert cost.agent["train"] == AgentSpend(usage=Usage.zero(), failed=0, trials=0, attempts=0)


def test_the_summary_table_has_a_row_per_role_and_the_total():
    def spend(tokens, failed=0):
        return Spend(usage=Usage.model_validate(_usage(tokens, calls=2)), failed=failed)

    def agent(tokens, failed=0):
        return AgentSpend(usage=spend(tokens).usage, failed=failed, trials=3, attempts=3 + failed)

    cost = RunCost(
        agent={"train": agent(100, failed=1), "validation": agent(20), "test": agent(10)},
        judges={"train": spend(8), "validation": spend(4), "test": spend(2)},
        proposers={"text": spend(50, failed=2)},
    )
    assert _cost_rows(cost) == [
        ["", "train", "validation", "test", "model calls", "attempts", "failed attempts"],
        ["agent", "100", "20", "10", "6", "10", "1"],
        ["judges", "8", "4", "2", "6", "-", "0"],
        ["proposer text", "50", "-", "-", "2", "-", "2"],
        ["total", "158", "24", "12", "14", "-", "-"],
    ]
