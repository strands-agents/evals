"""Test-only ``--agent`` entry-point shapes for the CLI.

These targets are imported by ``module:attr`` strings during CLI tests.
None of them invoke real models — agents are stub callables and "Agent"
subclasses register themselves under :class:`strands.Agent` only by name.
"""

from __future__ import annotations

from typing import Any

from strands_evals.case import Case


class _StubAgentResult:
    """Minimal stand-in for :class:`strands.agent.agent_result.AgentResult`."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


class _CallableAgent:
    """Bare callable that mimics ``agent(prompt)`` without inheriting from Agent."""

    def __init__(self, response: str = "stub-response", trace_attributes: dict | None = None) -> None:
        self.response = response
        self.trace_attributes = trace_attributes or {}
        self.calls: list[Any] = []

    def __call__(self, prompt: Any, **_: Any) -> _StubAgentResult:
        self.calls.append(prompt)
        return _StubAgentResult(self.response)


# --- module-level entry-point shapes ----------------------------------------

#: Pre-built "agent" instance — classifier should treat this as the
#: ``agent_instance`` shape only when it inherits from ``strands.Agent``;
#: tests assert that this NON-subclassed callable falls into ``callable_no_arg``
#: when used as a factory or rejected when not.
simple_callable_agent = _CallableAgent("hello from simple_callable_agent")


def build_agent() -> _CallableAgent:
    """Zero-arg factory returning a stub agent. Classified as ``callable_no_arg``."""
    return _CallableAgent("hello from build_agent")


def build_agent_for_case(case: Case) -> _CallableAgent:
    """One-arg factory keyed off ``case.metadata``. Classified as ``callable_one_arg``."""
    response = (case.metadata or {}).get("expected_response", f"response for {case.name}")
    return _CallableAgent(response)


def two_arg_factory(case: Case, extra: str) -> _CallableAgent:
    """Two-arg factory — should be rejected by the classifier."""
    return _CallableAgent(f"{extra}: {case.input}")
