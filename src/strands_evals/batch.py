"""Batch evaluation over sessions discovered from a TraceProvider.

Composes `TraceProvider.list_sessions` (session discovery) with
`TraceProvider.get_evaluation_data` (per-session retrieval) and an `Experiment`
so callers can evaluate every session matching a filter in a single call,
instead of writing the discover -> build cases -> run boilerplate by hand.
"""

import asyncio
import logging
from collections.abc import Callable

from .case import Case
from .evaluators.evaluator import Evaluator
from .experiment import Experiment
from .providers.trace_provider import SessionFilter, TraceProvider
from .types.evaluation import TaskOutput
from .types.evaluation_report import EvaluationReport

logger = logging.getLogger(__name__)


def evaluate_sessions(
    provider: TraceProvider,
    evaluators: list[Evaluator],
    session_filter: SessionFilter | None = None,
    *,
    max_workers: int = 1,
) -> EvaluationReport:
    """Discover sessions from a provider and evaluate them all.

    Discovers session IDs via `provider.list_sessions(session_filter)`, wraps each
    in a `Case` keyed on the session ID, and runs the given evaluators through an
    `Experiment`. The per-case task fetches trace data via
    `provider.get_evaluation_data(session_id)`.

    Args:
        provider: A TraceProvider whose backend supports session discovery
            (i.e. overrides `list_sessions`).
        evaluators: Evaluators to run against each discovered session.
        session_filter: Optional filter narrowing which sessions to evaluate. If
            None, provider-specific defaults apply.
        max_workers: Maximum number of parallel workers. Defaults to 1 (sequential),
            matching `Experiment.run_evaluations`. Providers that share a single
            network client or a `TracedHandler` exporter should keep this at 1.

    Returns:
        A single `EvaluationReport` flattened across every (session, evaluator) pair,
        with each row tagged by its evaluator via the `evaluator` field on `cases`.

    Raises:
        NotImplementedError: If the provider does not support session discovery.
        ProviderError: If the provider is unreachable or returns an error.
    """
    session_ids = list(provider.list_sessions(session_filter))
    logger.debug("session_count=<%d> | discovered sessions for batch evaluation", len(session_ids))

    cases: list[Case] = [Case(name=session_id, input=session_id, session_id=session_id) for session_id in session_ids]

    experiment = Experiment(cases=cases, evaluators=evaluators)

    task: Callable[[Case], TaskOutput] = provider.as_task()

    return asyncio.run(experiment.run_evaluations_async(task, max_workers=max_workers))
