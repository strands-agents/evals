"""Trial records for the optimizer: one case executed once, its status and usage."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..evaluation import EnvironmentState, EvaluationOutput, Interaction
from ..trace import Session

SplitName = Literal["train", "validation", "test"]


class TrialStatus(str, Enum):
    """What happened in a trial, decided from what happened, never inferred from a score.

    `VERIFIER_INVALID` means every evaluator failed, with at least one evaluator. A trial with
    zero evaluators (usage-only objective) is `COMPLETED`.
    """

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    VERIFIER_INVALID = "verifier_invalid"


class Usage(BaseModel):
    """Model usage summed over `chat` spans (never `invoke_agent` spans, which carry running totals).

    For a trial: the last attempt's agent spans, judge spans excluded. The same type holds
    `judge_usage`, `search_cost`, `judge_cost`, `retry_cost`, and `agent_cost`.

    Field names differ from `types.trace.TokenUsage` on purpose: they are the user-facing metric names
    (`USAGE_METRICS`), and `input_tokens` here includes cache tokens, which `TokenUsage.input_tokens`
    doesn't on every provider.

    Attributes:
        tokens: Everything the model processed, including cache (span `total_tokens`).
        input_tokens: Full prompt including cache, computed as `total_tokens - output_tokens`.
        output_tokens: Everything generated: text, tool-call arguments, reasoning.
        cache_read_tokens: Part of `input_tokens`; None when the provider never reported it
            (unmeasured, not 0).
        cache_write_tokens: Part of `input_tokens`; None when the provider never reported it
            (unmeasured, not 0).
        model_calls: Number of `chat` spans.
        seconds: Duration of the `rollout` span (environment setup and teardown excluded).
        models: Model ids that made the calls, first use first, from each `chat` span's
            `gen_ai.request.model`; `[]` when no span has it. Summing two usages joins the lists in
            first-use order. For `Patch.usage`, the proposer agent's configured model id.
    """

    model_config = ConfigDict(frozen=True)

    tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    model_calls: int
    seconds: float
    models: list[str]

    @classmethod
    def zero(cls) -> Usage:
        """No usage: every count 0, cache tokens None (unmeasured), no models; the start of every sum."""
        raise NotImplementedError

    def __add__(self, other: Usage) -> Usage:
        """Field by field; cache tokens None + None = None, None + x = x; `models` joined in first-use order.

        Totals are `sum(usages, Usage.zero())`.
        """
        raise NotImplementedError


class AttemptRecord(BaseModel):
    """One earlier attempt of a trial, kept for reporting; no `Session`.

    Two sources, both oldest first: attempts `Experiment` retried on throttling inside one task span,
    and a saved record that a later re-run pass replaced (its own `earlier` list, then itself).

    Attributes:
        status: What that attempt ended as (`infrastructure_invalid`, or `verifier_invalid` before a
            judge-only re-run).
        error_type: The exception type, if it raised.
        error: The exception message, if it raised.
        usage: What the agent spent in it; zero for a judge-only re-run, which reuses the task output.
        judge_usage: What its judges spent; zero for a throttled attempt `Experiment` retried.
        pass_index: The `TrialRunner` pass it ran in (0 = first; retried throttling keeps its pass).
        rollouts: What it cost the budget, as `TrialRecord.rollouts`.
    """

    model_config = ConfigDict(frozen=True)

    status: TrialStatus
    error_type: str | None
    error: str | None
    usage: Usage
    judge_usage: Usage
    pass_index: int
    rollouts: int


class TrialRecord(BaseModel):
    """One case x one run of one candidate.

    Stores no objective: scoring computes each run's objective from `evaluations` and `usage`.
    Each evaluator name is in exactly one of `evaluations` and `evaluator_errors`; `details` has the same
    keys as `evaluations`: each evaluator's own outputs (one per tool call or turn for tool- and
    turn-level evaluators), so reports rebuilt from the records match what `Experiment` returned and
    Reflection sees per-row reasons. Status, metrics, and usage read the last attempt; `earlier` is for
    reporting and the budget only, and `attempts == len(earlier) + 1`. `output` is saved as returned when
    JSON-serializable, else as `str(output)`.

    `rollouts` is what the last attempt cost the budget: 1 when the agent ran (a first pass or a re-run
    pass), 0 for a judge-only re-run on the saved output and for an attempt `Experiment` retried on
    throttling. The trial's total is `rollouts + sum(a.rollouts for a in earlier)`; resume rebuilds `spent`
    from the last record per key this way, with nothing counted twice, because a replacing record's
    `earlier` holds the replaced record and its own `earlier`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    case_name: str
    run_index: int
    split: SplitName
    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    earlier: list[AttemptRecord]
    output: Any
    trajectory: list[Any] | Session | None
    interactions: list[Interaction] | None
    environment_state: list[EnvironmentState] | None
    evaluations: dict[str, EvaluationOutput]
    details: dict[str, list[EvaluationOutput]]
    evaluator_errors: dict[str, str]
    usage: Usage
    judge_usage: Usage
    rollouts: int
