"""Session diagnosis: detect failures and analyze root causes."""

import logging
from collections.abc import Iterable

from strands.models.model import Model

from .._async import bounded_gather, run_async
from ..types.detector import ConfidenceLevel, DiagnosisResult
from ..types.trace import Session
from .failure_detector import detect_failures_async
from .root_cause_analyzer import analyze_root_cause_async

logger = logging.getLogger(__name__)


def diagnose_session(
    session: Session,
    *,
    model: Model | str | None = None,
    confidence_threshold: ConfidenceLevel = ConfidenceLevel.LOW,
) -> DiagnosisResult:
    """Run failure detection and root cause analysis on a session.

    Synchronous wrapper around `diagnose_session_async`. Safe to call from
    sync code or from inside a running event loop (Jupyter, FastAPI, async
    tests) — the work runs on a dedicated worker thread with its own loop.

    Args:
        session: The Session object to diagnose.
        model: Model for LLM-based detectors. None uses default.
        confidence_threshold: Minimum confidence for failure detection.

    Returns:
        DiagnosisResult with failures and root causes.
    """
    return run_async(
        lambda: diagnose_session_async(
            session,
            model=model,
            confidence_threshold=confidence_threshold,
        )
    )


async def diagnose_session_async(
    session: Session,
    *,
    model: Model | str | None = None,
    confidence_threshold: ConfidenceLevel = ConfidenceLevel.LOW,
    max_workers: int = 5,
) -> DiagnosisResult:
    """Async variant of `diagnose_session`.

    Pipeline: detect_failures_async → analyze_root_cause_async (if failures found).

    Args:
        session: The Session object to diagnose.
        model: Model for LLM-based detectors. None uses default.
        confidence_threshold: Minimum confidence for failure detection.
        max_workers: Maximum concurrent LLM calls inside per-session
            chunked detection / RCA. See `detect_failures_async`.

    Returns:
        DiagnosisResult with failures and root causes.
    """
    failure_output = await detect_failures_async(
        session,
        confidence_threshold=confidence_threshold,
        model=model,
        max_workers=max_workers,
    )

    root_causes = []
    if failure_output.failures:
        rca_output = await analyze_root_cause_async(
            session,
            failures=failure_output.failures,
            model=model,
            max_workers=max_workers,
        )
        root_causes = rca_output.root_causes

    return DiagnosisResult(
        session_id=session.session_id,
        failures=failure_output.failures,
        root_causes=root_causes,
    )


async def diagnose_sessions_async(
    sessions: Iterable[Session],
    *,
    model: Model | str | None = None,
    confidence_threshold: ConfidenceLevel = ConfidenceLevel.LOW,
    max_workers: int = 5,
    per_session_workers: int = 5,
) -> list[DiagnosisResult | None]:
    """Diagnose many sessions concurrently.

    Sessions fan out under a `max_workers` semaphore; per-session chunk/window
    fan-out is bounded by `per_session_workers`. Sessions that raise are
    logged and replaced with `None` in the result so one bad session does
    not poison the batch.

    Args:
        sessions: Sessions to diagnose. Order is preserved in the output.
        model: Model for LLM-based detectors.
        confidence_threshold: Minimum confidence for failure detection.
        max_workers: Maximum sessions diagnosed concurrently.
        per_session_workers: Maximum concurrent LLM calls inside each session.

    Returns:
        List of `DiagnosisResult` (or `None` if that session failed),
        in the same order as `sessions`.
    """
    sessions_list = list(sessions)

    async def _diagnose(session: Session) -> DiagnosisResult:
        return await diagnose_session_async(
            session,
            model=model,
            confidence_threshold=confidence_threshold,
            max_workers=per_session_workers,
        )

    gathered = await bounded_gather(
        (_diagnose(s) for s in sessions_list),
        max_workers,
        return_exceptions=True,
    )

    results: list[DiagnosisResult | None] = []
    for session, outcome in zip(sessions_list, gathered, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("Diagnosis failed for session %s: %s", session.session_id, outcome)
            results.append(None)
        else:
            results.append(outcome)
    return results
