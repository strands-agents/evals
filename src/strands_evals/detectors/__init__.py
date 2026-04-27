"""Detectors for analyzing agent execution traces.

Detectors answer "why did my agent behave this way?" by analyzing Session
traces for failures and summarizing execution.
"""

from ..types.detector import (
    ConfidenceLevel,
    FailureDetectionStructuredOutput,
    FailureItem,
    FailureOutput,
)
from .failure_detector import detect_failures

__all__ = [
    # Core detectors
    "detect_failures",
    # Types
    "ConfidenceLevel",
    "FailureOutput",
    "FailureItem",
    "FailureDetectionStructuredOutput",
]
