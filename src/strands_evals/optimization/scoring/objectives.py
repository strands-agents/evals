"""Objectives, constraints, and turning trials into scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...evaluators.evaluator import Evaluator
from ...types.optimization import Splits, SplitScores, TrialRecord
from .metrics import MetricRegistry


class MetricExpr:
    """A linear combination of metrics, e.g. `0.8*metric("Helpfulness") - 0.0001*metric("tokens")`.

    Supports `+`/`-` with another expression, `*`/`/` by a number, and unary `-`; anything else
    raises `TypeError`. Per case, N/A evaluator terms drop and the remaining evaluator weights are
    rescaled to the same total |weight|; usage terms are never rescaled. All evaluator terms N/A, or
    any usage term unmeasured, leaves the case unmeasured.
    """

    terms: tuple[tuple[str, float], ...]

    def __init__(self, terms: tuple[tuple[str, float], ...]):
        """`terms` are (metric name as written, weight) pairs; built by `metric()` and the operators."""
        raise NotImplementedError

    def __add__(self, other: MetricExpr) -> MetricExpr:
        raise NotImplementedError

    def __sub__(self, other: MetricExpr) -> MetricExpr:
        raise NotImplementedError

    def __mul__(self, weight: float) -> MetricExpr:
        raise NotImplementedError

    def __rmul__(self, weight: float) -> MetricExpr:
        raise NotImplementedError

    def __truediv__(self, divisor: float) -> MetricExpr:
        raise NotImplementedError

    def __neg__(self) -> MetricExpr:
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"0.8*Helpfulness - 0.0001*tokens"`; stored in the record."""
        raise NotImplementedError


@dataclass(frozen=True)
class Objective:
    """What to optimize, returned by `Maximize` / `Minimize`."""

    expr: MetricExpr
    direction: Literal["maximize", "minimize"]

    def __str__(self) -> str:
        """E.g. `"Maximize(0.8*Helpfulness - 0.0001*tokens)"`; stored in `run.json`."""
        raise NotImplementedError


@dataclass(frozen=True)
class Constraint:
    """A bound that can only reject, returned by `AtMost` / `AtLeast`."""

    expr: MetricExpr
    op: Literal["<=", ">="]
    bound: float

    def __str__(self) -> str:
        """E.g. `"tokens <= 4000"`; stored in `run.json`."""
        raise NotImplementedError


def metric(name: str | Evaluator | type[Evaluator]) -> MetricExpr:
    """An evaluator score or usage metric; an evaluator class or instance resolves via `get_name()`."""
    raise NotImplementedError


def Maximize(target: str | MetricExpr) -> Objective:
    raise NotImplementedError


def Minimize(target: str | MetricExpr) -> Objective:
    raise NotImplementedError


def AtMost(target: str | MetricExpr, bound: float) -> Constraint:
    raise NotImplementedError


def AtLeast(target: str | MetricExpr, bound: float) -> Constraint:
    raise NotImplementedError


class Criteria:
    """The objective plus the constraints; names resolve through the registry at construction."""

    def __init__(self, objectives: Objective | None, constraints: Sequence[Constraint], registry: MetricRegistry):
        """`objectives=None` means maximize the mean of every evaluator's score."""
        raise NotImplementedError

    def scalar(self, metrics: Mapping[str, float]) -> float | None:
        """The objective for one case or run, always higher-is-better (`Minimize` negates)."""
        raise NotImplementedError

    def violations(self, per_case: Mapping[str, Mapping[str, float]]) -> list[str]:
        """One message per violated constraint; a constraint never measured counts as violated.

        `per_case` maps case name -> metric name -> value.
        """
        raise NotImplementedError


def score(candidate_id: str, records: Sequence[TrialRecord], criteria: Criteria, splits: Splits) -> SplitScores:
    """Group records by split and average per case.

    A run's scalar is measured only when its status is completed, so a crash never scores.
    """
    raise NotImplementedError
