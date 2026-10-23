"""Trial status from the real exception object."""

from __future__ import annotations

from ...types.optimization import TrialStatus


def classify(error: BaseException) -> TrialStatus:
    """`INFRASTRUCTURE_INVALID` for throttling, network, and credential errors; `AGENT_FAILURE` otherwise."""
    raise NotImplementedError
