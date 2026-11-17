"""Tunable agents on a scripted model for `strands-evals optimize` tests (no network).

The model answers a case right with a probability set by its temperature and whether its system prompt asks for
the exact answer, as the optimizer's smoke test does; every run gives the same numbers.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.models import Model

from strands_evals.evaluators import Evaluator
from strands_evals.optimization import Float, Int, choice, component, tunable
from strands_evals.types import EvaluationOutput

EXACT = "Give the exact answer."


class World(Model):
    """The agent's scripted model."""

    def __init__(self, temperature: float, manager: object) -> None:
        self.temperature, self.manager = temperature, manager
        self.config = {"model_id": "scripted-world"}

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
        if isinstance(self.manager, SummarizingConversationManager):
            p, tokens = p + 0.03, int(300 + 100 * self.manager.summary_ratio)
        else:
            tokens = 500 + 5 * getattr(self.manager, "window_size", 40)
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
        window=component(SlidingWindowConversationManager, window_size=Int(4, 40)),
        summary=component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8)),
    ),
    temperature=Float(0.0, 1.0, default=0.9),
)
def manager_agent(manager=None, temperature=0.9):
    """A space without text: the default search is TPE only, so no proposer calls a model."""
    return Agent(
        model=World(temperature, manager), conversation_manager=manager, system_prompt=EXACT, callback_handler=None
    )


def plain_agent():
    """Not tunable: `optimize run` rejects it."""
    return Agent(model=World(0.25, None), callback_handler=None)


PROMPT_AGENT = """\
from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager

from strands_evals.optimization import FileText, Float, tunable
from tests.strands_evals.cli.fixtures.optimize_agents import World


@tunable(prompt=FileText("prompt.md"), temperature=Float(0.0, 1.0, default=0.3))
def prompt_agent(prompt, temperature=0.3):
    model = World(temperature, SummarizingConversationManager())
    return Agent(model=model, system_prompt=prompt, callback_handler=None)
"""
"""An `agent.py` with a prompt file, written into a test's project folder (a file marker reads its file at import)."""


world = World(0.25, None)
"""A scripted model instance for portable configs, referenced as `modelModule`."""
