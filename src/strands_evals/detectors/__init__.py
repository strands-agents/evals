"""Detectors for analyzing agent execution traces.

Detectors answer "why did my agent behave this way?" by analyzing Session
traces for failures, summarizing execution, extracting user requests, and
performing root cause analysis.
"""

from ..types.detector import (
    ConfidenceLevel,
    DiagnosisConfig,
    DiagnosisResult,
    FailureDetectionStructuredOutput,
    FailureItem,
    FailureOutput,
    RCAItem,
    RCAOutput,
    RCAStructuredOutput,
)
from .diagnosis import diagnose_session
from .failure_detector import detect_failures
from .root_cause_analyzer import analyze_root_cause

__all__ = [
    # Core detectors
    "detect_failures",
    "analyze_root_cause",
    # Diagnosis
    "diagnose_session",
    "DiagnosisConfig",
    "DiagnosisResult",
    # Types
    "ConfidenceLevel",
    "FailureOutput",
    "FailureItem",
    "FailureDetectionStructuredOutput",
    "RCAOutput",
    "RCAItem",
    "RCAStructuredOutput",
]
