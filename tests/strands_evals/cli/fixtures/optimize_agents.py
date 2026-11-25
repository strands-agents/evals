"""Tunable agents on a scripted model for `strands-evals optimize` tests (no network).

The model answers a case right with a probability set by its temperature and whether its system prompt asks for
the exact answer, as the optimizer's smoke test does; every run gives the same numbers.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator

from strands import Agent
from strands.context_manager import Offload
from strands.models import Model

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    TPE,
    AtMost,
    BayesBenchModel,
    Block,
    FinalPick,
    Float,
    Int,
    IterationRecord,
    LowerBound,
    Maximize,
    OneOf,
    Search,
    Split,
    TopRanked,
    choice,
    component,
    tunable,
)
from strands_evals.types import EvaluationOutput

EXACT = "Give the exact answer."


def context_manager(manager: dict) -> dict:
    """The agent's `context_manager=` for a `manager` bundle: summarize at a utilization, else keep the newest."""
    if "context_utilization" in manager:
        strategy = Offload.summarize("*").when(context_utilization=manager["context_utilization"], preserve_recent=4)
    else:
        strategy = Offload.drop("*").when(preserve_recent=manager["preserve_recent"])
    return {"strategies": [strategy], "archive": False}


class World(Model):
    """The agent's scripted model."""

    def __init__(self, temperature: float, manager: dict | None) -> None:
        self.temperature, self.manager = temperature, manager or {}
        self.config = {"model_id": "scripted-world", "context_window_limit": 200_000}

    def update_config(self, **config):
        self.config.update(config)

    def get_config(self):
        return self.config

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterator[dict]:
        question = messages[-1]["content"][0]["text"]
        p = 0.95 - 0.8 * abs(self.temperature - 0.25)
        if "context_utilization" in self.manager:
            p, tokens = p + 0.03, int(300 + 100 * self.manager["context_utilization"])
        else:
            tokens = 500 + 5 * self.manager.get("preserve_recent", 40)
        if EXACT not in (system_prompt or ""):
            p *= 0.4
        right = random.Random(f"{question}:{self.temperature:.4f}:{tokens}:{system_prompt}").random() < p
        reply = question.replace("question", "answer") if right else "I'm not sure."
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": tokens - 20, "outputTokens": 20, "totalTokens": tokens}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


class Correct(Evaluator):
    """Exact match against the expected output."""

    async def evaluate_async(self, data):
        hit = str(data.actual_output).strip() == data.expected_output
        return [EvaluationOutput(score=float(hit), test_pass=hit, reason="right" if hit else "wrong answer")]

    def evaluate(self, data):
        raise NotImplementedError


@tunable(
    manager=choice(
        window=component(dict, preserve_recent=Int(4, 40, default=40)),
        summary=component(dict, context_utilization=Float(0.1, 0.8, default=0.3)),
    ),
    temperature=Float(0.0, 1.0, default=0.9),
)
def manager_agent(manager=None, temperature=0.9):
    """A space without text: the default search is TPE only, so no proposer calls a model."""
    return Agent(
        model=World(temperature, manager),
        context_manager=context_manager(manager),
        system_prompt=EXACT,
        callback_handler=None,
    )


def plain_agent():
    """Not tunable: `optimize run` rejects it."""
    return Agent(model=World(0.25, None), callback_handler=None)


world = World(0.25, None)
"""A scripted model instance for portable configs, referenced as `modelModule`."""


# `harness_agent.py` (an agent file) as a tunable factory, for the parity test of the two routes.

INSTRUCTIONS = ["You answer questions.", f"You answer questions. {EXACT}"]
BUILT: list[str] = []


@tunable(instructions=OneOf(*INSTRUCTIONS), effort=OneOf("auto", "low"))
def harness_factory(instructions, effort):
    """The agent of `harness_agent.py`, its instructions and effort tunable."""
    from strands_harness import create_harness

    return create_harness(
        model=World(0.25, None),
        instructions=instructions,
        effort=effort,
        builtin_tools=[],
        builtin_plugins=[],
        skills=False,
        session=False,
        memory=False,
        callback_handler=None,
    )


# Objects for the `MODULE:ATTR` flags.


class FirstRanked(TopRanked):
    """`TopRanked` under another name, so a test can see a `--selection` reached the run."""

    name = "first_ranked"


class Scores(BayesBenchModel):
    """`BayesBenchModel` under another name, so a test can see a `--score-model` reached the run."""

    name = "scores"


first_ranked = FirstRanked()
scores = Scores()
search = Search([Block(TPE(n_startup_trials=1), "temperature")])
final_pick = FinalPick(n_finalists=2, rule=LowerBound(q=0.2))
split = Split(  # matched to the experiment's cases by name
    train=[Case(name=f"q{i:02d}", input="") for i in range(8)],
    validation=[Case(name=f"q{i:02d}", input="") for i in range(8, 14)],
    test=[Case(name=f"q{i:02d}", input="") for i in range(14, 20)],
)
objective = Maximize("Correct")
constraints = [AtMost("tokens", 450)]

ITERATIONS: list[int] = []


async def record_iteration(record: IterationRecord) -> None:
    """An async `--on-iteration` callback."""
    ITERATIONS.append(record.iteration)


# The `--build-task` + `--space` route: the same agent as `manager_agent`, declared without a factory.

space = {
    "manager": choice(
        window=component(dict, preserve_recent=Int(4, 40, default=40)),
        summary=component(dict, context_utilization=Float(0.1, 0.8, default=0.3)),
    ),
    "temperature": Float(0.0, 1.0, default=0.9),
}


def build_task(ctx):
    """One task per candidate; a fresh agent per trial."""

    async def task(case):
        manager = ctx.resolve("manager")
        agent = Agent(
            model=World(ctx.values["temperature"], manager),
            context_manager=context_manager(manager),
            system_prompt=EXACT,
            callback_handler=None,
        )
        return {"output": str(await agent.invoke_async(case.input))}

    return task


# A `--rollout` for the `--config` route.

ROLLOUTS: list[str] = []


async def harness_rollout(agent_factory, case):
    """Builds the trial's harness through `agent_factory`, as the default does, and records the case."""
    ROLLOUTS.append(case.name)
    agent = agent_factory()
    return {"output": str(await agent.invoke_async(case.input))}


def no_environment(case):
    """An `--environment` that provides no resources."""
    return {}
