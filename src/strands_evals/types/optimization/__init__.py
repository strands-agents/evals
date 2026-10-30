"""Data records for the optimizer. Imports nothing from `strands_evals.optimization` at runtime."""

from .estimates import Estimate
from .results import IterationRecord, ProposalOutcome, StopReason
from .scores import (
    CandidateLabel,
    CandidateScores,
    CaseSplit,
    Measured,
    PairedComparison,
    Range,
    SplitScores,
    TestReport,
    ValidationReport,
)
from .search import CandidateSummary, Evidence, Observation, ObservationOutcome, Patch, Value, Values
from .trials import AttemptRecord, SplitName, TrialRecord, TrialStatus, Usage

__all__ = [
    "AttemptRecord",
    "CandidateLabel",
    "CandidateScores",
    "CandidateSummary",
    "CaseSplit",
    "Estimate",
    "Evidence",
    "IterationRecord",
    "Measured",
    "Observation",
    "ObservationOutcome",
    "PairedComparison",
    "Patch",
    "ProposalOutcome",
    "Range",
    "SplitName",
    "SplitScores",
    "StopReason",
    "TestReport",
    "TrialRecord",
    "TrialStatus",
    "Usage",
    "ValidationReport",
    "Value",
    "Values",
]
