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

    `EVALUATOR_INVALID` means every evaluator failed, with at least one evaluator. A trial with
    zero evaluators (usage-only objective) is `COMPLETED`.
    """

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    EVALUATOR_INVALID = "evaluator_invalid"


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
        return cls(
            tokens=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=None,
            cache_write_tokens=None,
            model_calls=0,
            seconds=0.0,
            models=[],
        )

    def __add__(self, other: Usage) -> Usage:
        """Field by field; cache tokens None + None = None, None + x = x; `models` joined in first-use order.

        Totals are `sum(usages, Usage.zero())`.
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            tokens=self.tokens + other.tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=_add_measured(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_add_measured(self.cache_write_tokens, other.cache_write_tokens),
            model_calls=self.model_calls + other.model_calls,
            seconds=self.seconds + other.seconds,
            models=self.models + [model for model in other.models if model not in self.models],
        )


def _add_measured(left: int | None, right: int | None) -> int | None:
    """Sum two counts that may be unmeasured: None only when both are."""
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


class AttemptRecord(BaseModel):
    """One earlier attempt of a trial, kept for reporting; no `Session`.

    Two sources, both oldest first: attempts `Experiment` retried on throttling inside one task span,
    and a saved record that a later re-run pass replaced (its own `previous_attempts`, then itself).

    Attributes:
        status: What that attempt ended as. Only a retried status occurs: `infrastructure_invalid`,
            or `completed` / `evaluator_invalid` with failed evaluators before a judge-only re-run;
            `agent_failure` is final and never retried.
        error_type: The exception type, if it raised.
        error: The exception message, if it raised.
        usage: What the agent spent in it; zero when a judge-only re-run reused its output, since that
            usage stays on the new record, so it is counted once.
        judge_usage: What its failed judges spent; judges whose scores the new record kept count there
            instead, so this is only wasted work. Zero for a throttled attempt `Experiment` retried.
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
    Reflection sees per-row reasons. Status, metrics, and usage read the last attempt;
    `previous_attempts` is for reporting and the budget only, and
    `attempts == len(previous_attempts) + 1`. `output` is saved as returned when JSON-serializable,
    else as `str(output)`.

    `rollouts` is what the last attempt cost the budget: 1 when the agent ran (a first pass or a re-run
    pass), 0 for a judge-only re-run on the saved output and for an attempt `Experiment` retried on
    throttling. The trial's total is `rollouts + sum(a.rollouts for a in previous_attempts)`; resume rebuilds
    `spent` from the last record per key this way, with nothing counted twice, because a replacing
    record's `previous_attempts` holds the replaced record and its own `previous_attempts`.

    A failed evaluator never changes the status: the trial stays `completed` with it in
    `evaluator_errors` (every evaluator failed is `evaluator_invalid`); `infrastructure_invalid` is the
    agent side only. `judge_usage` is keyed by evaluator name, like `evaluations`: each judge's spend
    for the score in `evaluations` (a failed evaluator's spend is under its name too), including scores
    kept from a record a judge-only re-run replaced; the trial's judge spend is the sum.
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
    previous_attempts: list[AttemptRecord]
    output: Any
    trajectory: list[Any] | Session | None
    interactions: list[Interaction] | None
    environment_state: list[EnvironmentState] | None
    evaluations: dict[str, EvaluationOutput]
    details: dict[str, list[EvaluationOutput]]
    evaluator_errors: dict[str, str]
    usage: Usage
    judge_usage: dict[str, Usage]
    rollouts: int
