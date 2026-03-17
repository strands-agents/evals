"""Detectors for analyzing agent execution traces.

Detectors answer "why did my agent behave this way?" by analyzing Session
traces for failures, summarizing execution, extracting user requests, and
performing root cause analysis.
"""

from ..types.detector import (
    ConfidenceLevel,
    DiagnosisResult,
    FailureDetectionStructuredOutput,
    FailureItem,
    FailureOutput,
    RCAItem,
    RCAOutput,
    RCAStructuredOutput,
    SummaryOutput,
)
from .failure_detector import detect_failures
from .user_request_extractor import extract_user_requests

__all__ = [
    # Core detectors
    "detect_failures",
    "extract_user_requests",
    # Types
    "ConfidenceLevel",
    "DiagnosisResult",
    "FailureOutput",
    "FailureItem",
    "FailureDetectionStructuredOutput",
    "SummaryOutput",
    "RCAOutput",
    "RCAItem",
    "RCAStructuredOutput",
]
