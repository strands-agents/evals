"""Per-trial trace capture: the trial's `Session`, usage, and evaluator outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ...types.optimization import Usage
from ...types.trace import Session


@dataclass(frozen=True)
class EvaluatorView:
    """What the `evaluator {name}` spans of one trial say.

    `labels` come from each evaluator span's `gen_ai.evaluation.score.label`, which `Experiment`
    computes but doesn't return. `errors` holds exception type names for the record only; whether a
    judge failure was infrastructure comes from `TrialRunner`'s evaluator proxy, never from these names.
    """

    judge_usage: Usage
    errors: dict[str, str]
    labels: dict[str, str]


class TraceCapture:
    """One per run; keeps only spans of registered trial traces."""

    def __init__(self) -> None:
        """Install or attach the in-memory exporter; must run before any `Experiment` is built."""
        raise NotImplementedError

    def register(self, trace_id: int) -> None:
        """Keep this trace's spans; spans of unregistered traces are dropped at export."""
        raise NotImplementedError

    def agent_view(self, trace_id: int, attempt_span_id: int) -> tuple[Session | None, Usage]:
        """The `Session` and usage of the given attempt's subtree, judge spans excluded.

        Args:
            trace_id: The trial's trace.
            attempt_span_id: The last attempt's span; only its subtree is read.

        Returns:
            The agent's `Session` (None without agent spans) and its usage, judge spans excluded.
        """
        raise NotImplementedError

    def attempt_usage(self, trace_id: int, attempt_span_id: int) -> Usage:
        """One earlier attempt's usage (`TrialLogEntry.earlier`), by `agent_view`'s rules; for `AttemptRecord`.

        Args:
            trace_id: The trial's trace.
            attempt_span_id: That attempt's span; only its subtree is read.
        """
        raise NotImplementedError

    def evaluator_view(self, trace_id: int) -> EvaluatorView:
        """One walk over the trial's `evaluator {name}` spans: judge usage, errors, and labels."""
        raise NotImplementedError

    def release(self, trace_ids: Iterable[int]) -> None:
        """Drop only these trials' spans."""
        raise NotImplementedError
