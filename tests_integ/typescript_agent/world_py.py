"""world.ts in Python: right when the prompt asks for the exact answer (except every 10th), else every 3rd."""

import re

from strands.models import Model

EXACT = "Give the exact answer."


class Deterministic(Model):
    """The scripted model of world.ts."""

    def __init__(self):
        self.config = {"model_id": "scripted-deterministic"}

    def update_config(self, **config):
        self.config.update(config)

    def get_config(self):
        return self.config

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        question = "".join(block.get("text", "") for block in messages[-1]["content"])
        found = re.search(r"\d+", question)
        n = int(found.group()) if found else 0
        right = (n % 10 != 0) if EXACT in (system_prompt or "") else (n % 3 == 0)
        reply = question.replace("question", "answer") if right else "I'm not sure."
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": 680, "outputTokens": 20, "totalTokens": 700}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}
