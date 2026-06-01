"""Lightweight wrappers around ``pandas`` for evaluation results.

``ReportFrame`` wraps a ``pandas.DataFrame`` of evaluation rows and exposes
the familiar pandas surface (``group_by``, ``agg``, ``filter``, ``describe``)
plus a terminal ``display`` and JSON ``to_file`` / ``from_file``. Any method
not defined here is delegated straight to the underlying frame, and a
``DataFrame`` result is re-wrapped — so the full pandas vocabulary is
available without reimplementing it.

``group_by`` returns a ``ReportGroupBy``, a thin wrapper over a pandas
``DataFrameGroupBy`` whose aggregating methods return a ``ReportFrame``.

Note: ``ReportFrame.filter`` filters *rows* by a boolean mask or predicate,
which is the operation evaluation users want. This intentionally differs from
``pandas.DataFrame.filter`` (which selects columns by label); use ``.df`` for
the raw pandas object if the column-label behavior is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rich.console import Console
from rich.table import Table

from . import coerce


class ReportGroupBy:
    """Thin wrapper over a pandas ``DataFrameGroupBy``.

    Aggregating calls (``agg`` and any pandas groupby method that returns a
    ``DataFrame``) are re-wrapped as a ``ReportFrame`` so the result keeps the
    display and serialization helpers.
    """

    def __init__(self, groupby: "pd.core.groupby.DataFrameGroupBy") -> None:
        self._groupby = groupby

    def agg(self, *args: Any, **kwargs: Any) -> "ReportFrame":
        """Aggregate. Accepts the same arguments as ``DataFrameGroupBy.agg``.

        Named aggregation reads cleanly, e.g.::

            frame.group_by("evaluator").agg(
                mean_score=("score", "mean"),
                pass_rate=("passed", "mean"),
                n=("score", "count"),
            )
        """
        result = self._groupby.agg(*args, **kwargs)
        return ReportFrame(result.reset_index())

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._groupby, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return ReportFrame(result.reset_index())
            return result

        return wrapper


class ReportFrame:
    """A pandas-backed view over evaluation rows.

    Construct via :func:`strands_evals.aggregation.to_dataframe` /
    :func:`strands_evals.aggregation.flatten`, or the ``EvaluationReport``
    methods that delegate here. Wrap an existing frame directly with
    ``ReportFrame(df)``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def from_reports(cls, reports) -> "ReportFrame":
        """Combine many reports into one ``ReportFrame``, one row per case.

        This is the entry point for results spanning multiple evaluators or
        runs: each report's cases become rows, evaluator identity is preserved
        per row, then ``group_by`` collapses whichever dimensions you choose::

            ReportFrame.from_reports(reports).group_by("evaluator").agg(...)

        Note: this is distinct from ``EvaluationReport.flatten``, which merges
        reports into a single combined report and does not preserve per-report
        evaluator identity.
        """
        return cls(coerce.flatten(reports))

    # -- access ---------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The underlying pandas ``DataFrame`` (no copy)."""
        return self._df

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying frame."""
        return self._df.copy()

    # -- core operations ------------------------------------------------

    def group_by(self, by: str | list[str], **kwargs: Any) -> ReportGroupBy:
        """Group rows by one or more columns. Mirrors ``DataFrame.groupby``.

        ``kwargs`` are forwarded to pandas; ``as_index=False`` is not needed —
        the index is reset when aggregations are wrapped back up.
        """
        return ReportGroupBy(self._df.groupby(by, **kwargs))

    def filter(
        self, predicate: pd.Series | Callable[[pd.DataFrame], pd.Series]
    ) -> "ReportFrame":
        """Filter *rows* by a boolean mask or a callable returning one.

        Examples::

            frame.filter(frame.df["passed"])
            frame.filter(lambda d: d["score"] >= 0.5)
        """
        mask = predicate(self._df) if callable(predicate) else predicate
        return ReportFrame(self._df[mask].reset_index(drop=True))

    def describe(self, *args: Any, **kwargs: Any) -> "ReportFrame":
        """Descriptive statistics over numeric columns. Mirrors ``describe``."""
        return ReportFrame(self._df.describe(*args, **kwargs).reset_index())

    # -- serialization --------------------------------------------------

    def to_file(self, path: str) -> None:
        """Write the frame to a JSON file (records orientation).

        Args:
            path: Output path. ``.json`` is appended when no suffix is given.

        Raises:
            ValueError: If the path has a non-``.json`` extension.
        """
        file_path = Path(path)
        if file_path.suffix:
            if file_path.suffix != ".json":
                raise ValueError(
                    f"Only .json is supported. Got path with extension: {path}."
                )
        else:
            file_path = file_path.with_suffix(".json")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_json(file_path, orient="records", indent=2)

    @classmethod
    def from_file(cls, path: str) -> "ReportFrame":
        """Load a frame previously written by :meth:`to_file`."""
        file_path = Path(path)
        if file_path.suffix != ".json":
            raise ValueError(f"Only .json is supported. Got file: {path}.")
        return cls(pd.read_json(file_path, orient="records"))

    # -- display --------------------------------------------------------

    def display(self, title: str = "Evaluation Results") -> None:
        """Render the frame to the terminal as a rich table."""
        console = Console()
        if self._df.empty:
            console.print("[dim]No rows to display.[/dim]")
            return

        table = Table(title=title, show_lines=False)
        for col in self._df.columns:
            justify = "right" if pd.api.types.is_numeric_dtype(
                self._df[col]
            ) else "left"
            table.add_column(str(col), justify=justify)

        for _, row in self._df.iterrows():
            table.add_row(*[_format_cell(v) for v in row])
        console.print(table)

    # -- passthrough ----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return ReportFrame(result)
            return result

        return wrapper

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"ReportFrame(rows={len(self._df)}, columns={list(self._df.columns)})"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
