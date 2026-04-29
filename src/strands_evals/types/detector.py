"""Pydantic models for detectors.

Includes both output types (FailureOutput, etc.) and LLM structured
output schemas (FailureDetectionStructuredOutput, etc.).
"""

from typing import Literal

from pydantic import BaseModel, Field

# Confidence levels used across detectors
ConfidenceLevel = Literal["low", "medium", "high"]


class FailureItem(BaseModel):
    """A single detected failure.

    ``confidence`` is a list of floats (one per category) in the range [0.0, 1.0].
    Strings from the LLM (``"low"|"medium"|"high"``) are mapped to numeric
    values at parse time so downstream thresholding and merging can use
    direct numeric comparisons.
    """

    span_id: str = Field(description="Span where failure occurred")
    category: list[str] = Field(description="Failure classifications")
    confidence: list[float] = Field(description="Confidence per category in [0.0, 1.0]")
    evidence: list[str] = Field(description="Evidence per category")


class FailureOutput(BaseModel):
    """Output from detect_failures()."""

    session_id: str
    failures: list[FailureItem] = Field(default_factory=list)


class FailureError(BaseModel):
    """LLM output schema: single failure entry."""

    location: str
    category: list[str]
    confidence: list[Literal["low", "medium", "high"]]
    evidence: list[str]


class FailureDetectionStructuredOutput(BaseModel):
    """LLM output schema: failure detection result."""

    errors: list[FailureError]
