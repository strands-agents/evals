"""agent.ts in Python: the same harness agent on the same scripted model, for the parity test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strands_harness import create_harness  # noqa: E402
from world_py import Deterministic  # noqa: E402

agent = create_harness(
    model=Deterministic(),
    instructions="You answer questions.",
    effort="auto",
    builtin_tools=[],
    builtin_plugins=[],
    skills=False,
)
