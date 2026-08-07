"""Detectors for analyzing agent execution traces.

Detectors answer "why did my agent behave this way?" by analyzing Session
traces for failures, summarizing execution, extracting user requests, and
performing root cause analysis.
"""

from ..types.detector import (
    ConfidenceLevel,
    DiagnosisConfig,
    DiagnosisResult,
    DiagnosisTrigger,
    FailureDetectionStructuredOutput,
    FailureItem,
    FailureOutput,
    RCAItem,
    RCAOutput,
    RCAStructuredOutput,
)
from .diagnosis import diagnose_session, diagnose_session_async, diagnose_sessions_async
from .failure_detector import detect_failures, detect_failures_async
from .root_cause_analyzer import analyze_root_cause, analyze_root_cause_async

__all__ = [
    # Core detectors
    "detect_failures",
    "detect_failures_async",
    "analyze_root_cause",
    "analyze_root_cause_async",
    # Diagnosis
    "diagnose_session",
    "diagnose_session_async",
    "diagnose_sessions_async",
    "DiagnosisConfig",
    "DiagnosisResult",
    "DiagnosisTrigger",
    # Types
    "ConfidenceLevel",
    "FailureOutput",
    "FailureItem",
    "FailureDetectionStructuredOutput",
    "RCAOutput",
    "RCAItem",
    "RCAStructuredOutput",
]
