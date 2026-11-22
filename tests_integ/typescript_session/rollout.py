"""A two-turn rollout for `strands-evals optimize --rollout`: one conversation, the answer on the second turn.

The scripted model (`world.ts`) only answers from the second user turn of one conversation, so this rollout
scores well exactly when the session API really holds one agent open across turns.
"""

from typing import Any

from strands_evals import Case


async def two_turns(agent_factory: Any, case: Case) -> dict[str, Any]:
    """Warm the conversation up with one turn, then ask the case's question on the second."""
    agent = agent_factory()
    async with agent.session() as session:
        opening = await session.turn(f"here it comes: {case.input}")
        if opening != "I'm not sure.":  # the first turn of a fresh conversation never answers
            raise RuntimeError(f"the opening turn answered {opening!r}: the conversation did not start fresh")
        return {"output": await session.turn(case.input)}
