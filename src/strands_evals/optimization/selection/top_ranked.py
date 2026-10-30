"""TopRanked: the parent is the first of the ranking."""

from __future__ import annotations

from typing import ClassVar

from .base import Selection, SelectionContext


class TopRanked(Selection):
    """The default: `ctx.ranking[0]`.

    Feasible candidates first, by the highest objective (each goal metric's estimate resolved to a tenth of
    its sd, combined by `Criteria`; measured train means without a fit or with any metric flagged); none
    feasible -> the least violation. Ties go to the earlier candidate, so the baseline wins when nothing
    beats it at the model's resolution.
    """

    name: ClassVar[str] = "top_ranked"

    def choose(self, ctx: SelectionContext) -> str:
        raise NotImplementedError
