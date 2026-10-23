"""Admission on the selection split: constraints, then coverage, then the selection strategy."""

from __future__ import annotations

from collections.abc import Sequence

from ..types.optimization import Admission, CandidateScores, SelectionContext
from .scoring.objectives import Criteria
from .selection.base import Selection


def admit(
    candidate: CandidateScores,
    parent: CandidateScores,
    *,
    selection_cases: Sequence[str],
    criteria: Criteria,
    selection: Selection,
    ctx: SelectionContext,
    min_coverage: float,
) -> Admission:
    """Selection-split scores only, never the holdout.

    Constraints and coverage always run first, so a custom `Selection` can't skip them.
    """
    raise NotImplementedError
