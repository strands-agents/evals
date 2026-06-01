"""Pandas-style aggregation for evaluation reports.

The design treats an ``EvaluationReport`` as tabular data: a single
``to_dataframe`` touch point coerces it into a ``pandas.DataFrame``, and all
grouping / aggregation / filtering / description piggybacks on pandas rather
than being reimplemented.

Public API:

- ``to_dataframe(report)``: coerce one report into a one-row-per-case frame.
- ``flatten(reports)``: low-level — combine many reports into one
  ``pandas.DataFrame``. Prefer ``ReportFrame.from_reports(reports)`` for the
  wrapped, chainable form.
- ``ReportFrame``: a pandas-backed view with ``group_by`` / ``agg`` /
  ``filter`` / ``describe``, terminal ``display``, and JSON ``to_file`` /
  ``from_file``. Unrecognized methods delegate to the underlying frame.
- ``ReportGroupBy``: result of ``ReportFrame.group_by``; ``agg`` returns a
  ``ReportFrame``.
- ``DataFrameMixin``: added to ``EvaluationReport`` to expose these methods on
  the report itself (``report.group_by(...)``). For multiple reports use
  ``ReportFrame.from_reports(reports)`` (``EvaluationReport.flatten`` already
  exists upstream with different semantics).

Comparison (deltas, A/B, statistical tests) is intentionally out of scope
here; it is a separate, composable layer that operates on the aggregated
frame.
"""

from .coerce import flatten, to_dataframe
from .frame import ReportFrame, ReportGroupBy
from .mixin import DataFrameMixin

__all__ = [
    "to_dataframe",
    "flatten",
    "ReportFrame",
    "ReportGroupBy",
    "DataFrameMixin",
]
