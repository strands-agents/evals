"""Adapts agent configuration into task callables compatible with Experiment.run_evaluations."""

from typing import Any

from strands import Agent

from .case import Case
from .mappers.strands_in_memory_session_mapper import StrandsInMemorySessionMapper
from .telemetry import StrandsEvalsTelemetry


class AgentTask:
    """A task that creates a fresh Agent per case and invokes it with case.input.

    Accepts the same keyword arguments as strands.Agent.

    Example:
        task = AgentTask(model="us.anthropic.claude-sonnet-4-20250514-v1:0", tools=[calculator])
        experiment.run_evaluations(task=task)
    """

    def __init__(self, **agent_kwargs: Any):
        self._agent_kwargs = agent_kwargs

    def _create_agent(self) -> Agent:
        return Agent(**self._agent_kwargs)

    def __call__(self, case: Case) -> dict[str, Any]:
        agent = self._create_agent()
        result = agent(case.input)
        return {"output": str(result)}


class TracedAgentTask(AgentTask):
    """An AgentTask that also collects OpenTelemetry spans and maps them to a Session.

    Use this when your evaluators need trajectory data (e.g., TrajectoryEvaluator).

    Example:
        task = TracedAgentTask(model="us.anthropic.claude-sonnet-4-20250514-v1:0", tools=[calculator])
        experiment.run_evaluations(task=task)
    """

    def __init__(self, **agent_kwargs: Any):
        super().__init__(**agent_kwargs)
        self._telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        self._mapper = StrandsInMemorySessionMapper()

    def __call__(self, case: Case) -> dict[str, Any]:
        self._telemetry.in_memory_exporter.clear()

        agent = self._create_agent()
        result = agent(case.input)

        spans = list(self._telemetry.in_memory_exporter.get_finished_spans())
        session = self._mapper.map_to_session(spans, case.session_id)

        return {"output": str(result), "trajectory": session}
