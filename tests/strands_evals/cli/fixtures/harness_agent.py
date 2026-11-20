"""An agent file as `/export` writes one: the agent built by `create_harness`, on the scripted model (no network).

`optimize_agents.harness_factory` is the same agent as a tunable factory.
"""

from strands_harness import create_harness

from tests.strands_evals.cli.fixtures.optimize_agents import BUILT, World

BUILT.append("file")  # each run of the file, so a test can count them

agent = create_harness(
    model=World(0.25, None),
    instructions="You answer questions.",
    effort="auto",
    builtin_tools=[],
    builtin_plugins=[],
    skills=False,
)
