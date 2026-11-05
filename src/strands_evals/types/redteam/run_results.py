"""Run stats a red team task records for `RedTeamReport` and `AttackSuccessEvaluator` to read."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..evaluation import EvaluationData

# Name of the task-output `environment_state` entry `RedTeamReport` reads run stats from.
RUN_RESULTS = "redteam_run_results"

# Run-stats key a strategy sets when its attacker model declined its role, and the conversation role of the
# refused message it keeps for debugging (never sent, so it is the last entry and has no target reply). The
# target was never fully tested, so `AttackSuccessEvaluator` and `RedTeamReport` treat the attack as
# inconclusive, never defended.
ATTACKER_REFUSED = "attacker_refused"

# Run-stats key counting attacker messages the refusal judge failed to judge; each was sent anyway, so a
# non-zero count means refusal detection was degraded for the case.
REFUSAL_JUDGE_FAILURES = "refusal_judge_failures"


def _run_results(case: dict[str, Any] | EvaluationData) -> dict[str, Any]:
    """Return the case's `RUN_RESULTS` environment state, falling back to `metadata` for older reports.

    Accepts a serialized report row or a live `EvaluationData`. Custom tasks own `environment_state`, so
    entries may be `EnvironmentState` objects or plain dicts, and a non-dict state reads as `{}`.
    """
    if isinstance(case, dict):
        entries, metadata = case.get("actual_environment_state"), case.get("metadata")
    else:
        entries, metadata = case.actual_environment_state, case.metadata
    for entry in entries or []:
        if isinstance(entry, dict):
            name, state = entry.get("name"), entry.get("state")
        else:
            name, state = getattr(entry, "name", None), getattr(entry, "state", None)
        if name == RUN_RESULTS:
            return state if isinstance(state, dict) else {}
    return metadata or {}
