"""Session diagnosis: detect failures and analyze root causes."""

from strands.models.model import Model

from ..types.detector import ConfidenceLevel, DiagnosisResult
from ..types.trace import Session
from .failure_detector import detect_failures
from .root_cause_analyzer import analyze_root_cause


def diagnose_session(
    session: Session,
    *,
    model: Model | str | None = None,
    confidence_threshold: ConfidenceLevel = "low",
) -> DiagnosisResult:
    """Run failure detection and root cause analysis on a session.

    Pipeline: detect_failures → analyze_root_cause (if failures found).

    Args:
        session: The Session object to diagnose.
        model: Model for LLM-based detectors. None uses default.
        confidence_threshold: Minimum confidence for failure detection.

    Returns:
        DiagnosisResult with failures and root causes.
    """
    failure_output = detect_failures(
        session,
        confidence_threshold=confidence_threshold,
        model=model,
    )

    root_causes = []
    if failure_output.failures:
        rca_output = analyze_root_cause(
            session,
            failures=failure_output.failures,
            model=model,
        )
        root_causes = rca_output.root_causes

    return DiagnosisResult(
        session_id=session.session_id,
        failures=failure_output.failures,
        root_causes=root_causes,
    )
