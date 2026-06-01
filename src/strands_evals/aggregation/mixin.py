"""Mixin that gives ``EvaluationReport`` its pandas-style surface.

The upstream ``EvaluationReport`` adds this mixin to its bases::

    class EvaluationReport(DataFrameMixin, BaseModel):
        ...

That one-line change is the only edit to the core type. All behavior lives
here and in ``coerce`` / ``frame``, so the report stays a thin data model and
the data-frame machinery is isolated and independently testable.

Single-report usage collapses one report's case dimension::

    report.group_by("evaluator").agg(mean_score=("score", "mean"))

Multi-report usage flattens first, then groups::

    ReportFrame.from_reports(reports).group_by("evaluator").agg(...)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import coerce
from .frame import ReportFrame, ReportGroupBy


class DataFrameMixin:
    """Pandas-style operations for a report. Expects the host to expose the
    ``EvaluationReport`` fields (``evaluator_name``, ``scores``, ``cases``,
    ``test_passes``, ``reasons``)."""

    def to_dataframe(self) -> pd.DataFrame:
        """Coerce this report into a one-row-per-case ``DataFrame``."""
        return coerce.to_dataframe(self)  # type: ignore[arg-type]

    def frame(self) -> ReportFrame:
        """Return this report as a :class:`ReportFrame`."""
        return ReportFrame(self.to_dataframe())

    def group_by(self, by: str | list[str], **kwargs: Any) -> ReportGroupBy:
        """Group this report's rows. See :meth:`ReportFrame.group_by`."""
        return self.frame().group_by(by, **kwargs)

    def agg(self, *args: Any, **kwargs: Any) -> ReportFrame:
        """Aggregate this report's rows without grouping (whole-frame agg)."""
        result = self.to_dataframe().agg(*args, **kwargs)
        frame = result.to_frame().T if isinstance(result, pd.Series) else result
        return ReportFrame(frame.reset_index(drop=True))

    def filter(self, predicate: Any) -> ReportFrame:
        """Filter this report's rows. See :meth:`ReportFrame.filter`."""
        return self.frame().filter(predicate)

    def describe(self, *args: Any, **kwargs: Any) -> ReportFrame:
        """Describe this report's numeric columns."""
        return self.frame().describe(*args, **kwargs)
