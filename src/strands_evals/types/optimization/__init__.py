"""Data records for the optimizer. Imports nothing from `strands_evals.optimization` at runtime."""

from .estimates import Estimate
from .results import IterationRecord, ProposalOutcome, StopReason
from .scores import (
    LABEL_OUTCOMES,
    CandidateLabel,
    CandidateScores,
    CaseSplit,
    LabelOutcome,
    Measured,
    PairedComparison,
    Range,
    SplitScores,
    TestReport,
    ValidationReport,
    near,
)
from .search import CandidateSummary, Evidence, Observation, ObservationOutcome, Patch, Value, Values
from .trials import SPLIT_NAMES, AttemptRecord, SplitName, TrialRecord, TrialStatus, Usage

__all__ = [
    "AttemptRecord",
    "CandidateLabel",
    "CandidateScores",
    "CandidateSummary",
    "CaseSplit",
    "Estimate",
    "Evidence",
    "IterationRecord",
    "LABEL_OUTCOMES",
    "LabelOutcome",
    "Measured",
    "Observation",
    "ObservationOutcome",
    "PairedComparison",
    "Patch",
    "ProposalOutcome",
    "Range",
    "SPLIT_NAMES",
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
    "near",
]
