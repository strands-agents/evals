"""Eval-driven optimization of agent prompts, components, and numeric knobs."""

from ..types.optimization import (
    AcceptanceVerdict,
    IterationRecord,
    MergePlan,
    Observation,
    Patch,
    ProposalOutcome,
    ProposalRequest,
    ReflectionRecord,
    SelectionContext,
    SelectionStep,
    TrialStatus,
    Usage,
)
from .acceptance.budget import Budget
from .acceptance.gate import Gate
from .candidates.candidate import Candidate
from .declaration.files import FileText, JsonField, SkillLibrary
from .declaration.markers import Float, Int, OneOf, TextValue, choice, component
from .declaration.space import ParamInfo, ParamKind, Space
from .declaration.tools import tool_descriptions
from .declaration.tunable import TunableFactory, tunable
from .optimizer import Optimizer
from .result import CandidateView, OptimizationResult
from .rewards import RewardFunction, RewardFunctionEvaluator
from .scoring.objectives import AtLeast, AtMost, Maximize, Minimize, metric
from .search.blocks import Block, Search
from .search.method import Method
from .search.reflection import Reflection
from .search.schedules import Alternate, Joint
from .search.tpe import TPE
from .selection.base import Selection
from .selection.frontier import CaseFrontier
from .selection.greedy import Greedy

__all__ = [
    "AcceptanceVerdict",
    "Alternate",
    "AtLeast",
    "AtMost",
    "Block",
    "Budget",
    "Candidate",
    "CandidateView",
    "CaseFrontier",
    "FileText",
    "Float",
    "Gate",
    "Greedy",
    "Int",
    "IterationRecord",
    "JsonField",
    "Joint",
    "Maximize",
    "MergePlan",
    "Method",
    "Minimize",
    "Observation",
    "OneOf",
    "OptimizationResult",
    "Optimizer",
    "ParamInfo",
    "ParamKind",
    "Patch",
    "ProposalOutcome",
    "ProposalRequest",
    "Reflection",
    "ReflectionRecord",
    "RewardFunction",
    "RewardFunctionEvaluator",
    "Search",
    "Selection",
    "SelectionContext",
    "SelectionStep",
    "SkillLibrary",
    "Space",
    "TPE",
    "TextValue",
    "TrialStatus",
    "TunableFactory",
    "Usage",
    "choice",
    "component",
    "metric",
    "tool_descriptions",
    "tunable",
]
