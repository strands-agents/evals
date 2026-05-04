"""Pydantic models for detectors.

Includes both output types (FailureOutput, RCAOutput, etc.) and
LLM structured output schemas (FailureDetectionStructuredOutput, etc.).
"""

from typing import Literal

from pydantic import BaseModel, Field
from strands.models.model import Model

# Confidence levels used across detectors
ConfidenceLevel = Literal["low", "medium", "high"]


DiagnosisTrigger = Literal["on_failure", "always"]


class DiagnosisConfig(BaseModel):
    """Configuration for detectors in an experiment.

    Attributes:
        trigger: When to run diagnosis — "on_failure" or "always".
        model: The model to use for diagnosis.
        confidence_threshold: Minimum confidence level for failure detection.
    """

    trigger: DiagnosisTrigger = "on_failure"
    model: Model | str | None = None
    confidence_threshold: ConfidenceLevel = "medium"

    model_config = {"arbitrary_types_allowed": True}


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


class RCAItem(BaseModel):
    """A single root cause finding."""

    failure_span_id: str = Field(description="The failure span this explains")
    location: str = Field(description="Span where root cause originated")
    causality: str = Field(description="PRIMARY_FAILURE | SECONDARY_FAILURE | TERTIARY_FAILURE")
    propagation_impact: list[str] = Field(default_factory=list)
    failure_detection_timing: str = Field(
        description="IMMEDIATELY_AT_OCCURRENCE | SEVERAL_STEPS_LATER | ONLY_AT_TASK_END | SILENT_UNDETECTED"
    )
    completion_status: str = Field(description="COMPLETE_SUCCESS | PARTIAL_SUCCESS | COMPLETE_FAILURE")
    root_cause_explanation: str
    fix_type: str = Field(description="SYSTEM_PROMPT_FIX | TOOL_DESCRIPTION_FIX | OTHERS")
    fix_recommendation: str


class RCAOutput(BaseModel):
    """Output from analyze_root_cause()."""

    root_causes: list[RCAItem] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    """Output from diagnose_session()."""

    session_id: str
    failures: list[FailureItem] = Field(default_factory=list)
    root_causes: list[RCAItem] = Field(default_factory=list)

    @property
    def recommendations(self) -> list[str]:
        """Deduplicated fix recommendations from all root causes."""
        seen: set[str] = set()
        result: list[str] = []
        for rc in self.root_causes:
            rec = rc.fix_recommendation.strip()
            if rec and rec not in seen:
                result.append(rec)
                seen.add(rec)
        return result


class FailureError(BaseModel):
    """LLM output schema: single failure entry."""

    location: str
    category: list[str]
    confidence: list[Literal["low", "medium", "high"]]
    evidence: list[str]


class FailureDetectionStructuredOutput(BaseModel):
    """LLM output schema: failure detection result."""

    errors: list[FailureError]


class FixRecommendation(BaseModel):
    """LLM output schema: fix recommendation."""

    model_config = {"populate_by_name": True}

    fix_type: Literal["SYSTEM_PROMPT_FIX", "TOOL_DESCRIPTION_FIX", "OTHERS"] = Field(
        ..., alias="Fix Type", description="Type of fix needed"
    )
    recommendation: str = Field(
        ...,
        alias="Recommendation",
        description="Brief, actionable fix suggestion (1-2 sentences)",
    )


class RootCauseItem(BaseModel):
    """LLM output schema: single root cause entry."""

    model_config = {"populate_by_name": True}

    failure_span_id: str = Field(
        ...,
        alias="Failure Span ID",
        description="The span_id from execution_failures that this root cause addresses (1:1 mapping)",
    )
    location: str = Field(
        ...,
        alias="Location",
        description="Exact span_id where the root cause failure occurred",
    )
    failure_causality: Literal["PRIMARY_FAILURE", "SECONDARY_FAILURE", "TERTIARY_FAILURE", "UNCLEAR"] = Field(
        ...,
        alias="Failure Causality",
        description="Causality classification of the failure",
    )
    failure_propagation_impact: list[
        Literal[
            "TASK_TERMINATION",
            "QUALITY_DEGRADATION",
            "INCORRECT_PATH",
            "STATE_CORRUPTION",
            "NO_PROPAGATION",
            "UNCLEAR",
        ]
    ] = Field(
        ...,
        alias="Failure Propagation Impact",
        description="List of impact types on task execution",
    )
    failure_detection_timing: Literal[
        "IMMEDIATELY_AT_OCCURRENCE",
        "SEVERAL_STEPS_LATER",
        "ONLY_AT_TASK_END",
        "SILENT_UNDETECTED",
    ] = Field(
        ...,
        alias="Failure Detection Timing",
        description="When the failure was detected in the execution",
    )
    completion_status: Literal["COMPLETE_SUCCESS", "PARTIAL_SUCCESS", "COMPLETE_FAILURE"] = Field(
        ..., alias="Completion Status", description="Overall task completion status"
    )
    root_cause_explanation: str = Field(
        ...,
        alias="Root Cause Explanation",
        description="Concise explanation of the fundamental issue (2-3 sentences)",
    )
    fix_recommendation: FixRecommendation = Field(
        ...,
        alias="Fix Recommendation",
        description="Structured recommendation for addressing the root cause",
    )


class RCAStructuredOutput(BaseModel):
    """LLM output schema: root cause analysis result."""

    root_causes: list[RootCauseItem] = Field(
        ..., description="List of all identified root causes in the execution trace"
    )
