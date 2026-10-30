"""Eval-driven optimization of agent prompts, components, and numeric knobs."""

from ..types.optimization import (
    AttemptRecord,
    CandidateScores,
    CandidateSummary,
    CaseSplit,
    Estimate,
    Evidence,
    IterationRecord,
    Measured,
    Observation,
    ObservationOutcome,
    PairedComparison,
    Patch,
    ProposalOutcome,
    Range,
    SplitName,
    StopReason,
    TestReport,
    TrialRecord,
    TrialStatus,
    Usage,
    ValidationReport,
    Value,
    Values,
)
from .candidate import Candidate
from .declaration.files import FileText, JsonField, SkillLibrary
from .declaration.markers import Choice, Component, Float, Int, OneOf, TextValue, choice, component
from .declaration.space import FileSource, Marker, ParamInfo, ParamKind, Space
from .declaration.tools import ToolDescriptions, tool_descriptions
from .declaration.tunable import BuildContext, TunableFactory, tunable
from .end_phase import FinalPick
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn
from .execution.status import InfrastructureError
from .optimizer import Optimizer
from .result import CandidateView, Grain, OptimizationResult
from .score_model.bayesbench import BayesBenchModel
from .score_model.design import design
from .score_model.effects import effects
from .score_model.protocols import Fit, FitData, MetricModel, ScoreModel
from .score_model.ranking import Assessment, LowerBound, Mean, PickRule, p_met
from .scoring.metrics import MetricRegistry
from .scoring.objectives import (
    AtLeast,
    AtMost,
    Constraint,
    Criteria,
    Maximize,
    MetricTarget,
    Minimize,
    Objective,
    Pareto,
    WeightedSum,
)
from .scoring.rewards import RewardFunction, RewardFunctionEvaluator, RewardResult
from .search.blocks import Alternate, Block, Joint, Nested, Schedule, ScheduleContext, Search, Step
from .search.proposer import ProposalRequest, Proposer
from .search.reflection import Reflection
from .search.thompson import Thompson
from .search.tpe import TPE
from .selection.base import Selection, SelectionContext
from .selection.top_ranked import TopRanked
from .setup.budget import Budget
from .setup.splits import Split

__all__ = [
    "Alternate",
    "Assessment",
    "AtLeast",
    "AtMost",
    "AttemptRecord",
    "BayesBenchModel",
    "Block",
    "Budget",
    "BuildContext",
    "Candidate",
    "CandidateScores",
    "CandidateSummary",
    "CandidateView",
    "CaseSplit",
    "Choice",
    "Component",
    "Constraint",
    "Criteria",
    "EnvironmentFn",
    "Estimate",
    "Evidence",
    "FileSource",
    "FileText",
    "FinalPick",
    "Fit",
    "FitData",
    "Float",
    "Grain",
    "InfrastructureError",
    "Int",
    "IterationRecord",
    "Joint",
    "JsonField",
    "LowerBound",
    "Marker",
    "Maximize",
    "Mean",
    "Measured",
    "MetricModel",
    "MetricRegistry",
    "MetricTarget",
    "Minimize",
    "Nested",
    "Objective",
    "Observation",
    "ObservationOutcome",
    "OneOf",
    "OptimizationResult",
    "Optimizer",
    "PairedComparison",
    "ParamInfo",
    "ParamKind",
    "Pareto",
    "Patch",
    "PickRule",
    "ProposalOutcome",
    "ProposalRequest",
    "Proposer",
    "Range",
    "Reflection",
    "RewardFunction",
    "RewardFunctionEvaluator",
    "RewardResult",
    "RolloutFn",
    "Schedule",
    "ScheduleContext",
    "ScoreModel",
    "Search",
    "Selection",
    "SelectionContext",
    "SkillLibrary",
    "Space",
    "Split",
    "SplitName",
    "Step",
    "StopReason",
    "TPE",
    "TaskFn",
    "TestReport",
    "TextValue",
    "Thompson",
    "ToolDescriptions",
    "TopRanked",
    "TrialRecord",
    "TrialStatus",
    "TunableFactory",
    "Usage",
    "ValidationReport",
    "Value",
    "Values",
    "WeightedSum",
    "choice",
    "component",
    "design",
    "effects",
    "p_met",
    "tool_descriptions",
    "tunable",
]
