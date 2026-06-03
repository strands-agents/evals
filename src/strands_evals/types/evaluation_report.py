import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from ..display.display_console import CollapsibleTableReportDisplay
from ..types.evaluation import EvaluationOutput

# Stable column names for the tabular view of a report.
CASE = "case"
EVALUATOR = "evaluator"
SCORE = "score"
PASSED = "passed"
REASON = "reason"
METADATA_PREFIX = "metadata."

_BASE_COLUMNS = [CASE, EVALUATOR, SCORE, PASSED, REASON]


def _case_key(case: dict, index: int) -> str:
    """Resolve a case's grouping key: metadata case_key, else name, else index."""
    metadata = case.get("metadata") or {}
    return metadata.get("case_key") or case.get("name") or f"case#{index}"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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

            report.group_by("evaluator").agg(
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

    Construct via ``EvaluationReport.to_dataframe()`` / ``.frame()`` or the
    ``EvaluationReport`` methods that delegate here. Wrap an existing frame
    directly with ``ReportFrame(df)``.

    Any method not defined here delegates to the underlying frame and a
    ``DataFrame`` result is re-wrapped, so the full pandas vocabulary is
    available. Use ``.df`` for the raw pandas object.

    Note: ``ReportFrame.filter`` filters *rows* by a boolean mask or predicate,
    which is what evaluation users want; this differs from
    ``pandas.DataFrame.filter`` (which selects columns by label).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

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
        """Group rows by one or more columns. Mirrors ``DataFrame.groupby``."""
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
            justify = "right" if pd.api.types.is_numeric_dtype(self._df[col]) else "left"
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


class EvaluationReport(BaseModel):
    """
    A report of the evaluation of a task.

    Attributes:
        evaluator_name: The name of the evaluator that produced this report.
        overall_score: The overall score of the task.
        scores: A list of the score for each test case in order.
        cases: A list of records for each test case.
        test_passes: A list of booleans indicating whether the test pass or fail.
        reasons: A list of reason for each test case.
    """

    evaluator_name: str = ""
    overall_score: float
    scores: list[float]
    cases: list[dict]
    test_passes: list[bool]
    reasons: list[str] = []
    detailed_results: list[list[EvaluationOutput]] = []
    diagnoses: list[dict | None] = []
    recommendations: list[str | None] = []

    @classmethod
    def flatten(cls, reports: list["EvaluationReport"]) -> "EvaluationReport":
        """Flatten multiple evaluation reports into a single report.

        Each case is stamped with its source ``evaluator`` so that the tabular
        view (:meth:`to_dataframe`) preserves per-report evaluator identity even
        though the combined report's own ``evaluator_name`` is ``"Combined"``.
        That is what makes ``EvaluationReport.flatten(reports).group_by(
        "evaluator")`` group by the original evaluators.
        """
        if not reports:
            return cls(overall_score=0.0, scores=[], cases=[], test_passes=[])

        scores, cases, passes, reasons, detailed, diags, recs = [], [], [], [], [], [], []

        for report in reports:
            evaluator = report.evaluator_name or "Unknown"
            for i, case in enumerate(report.cases):
                cases.append({**case, "evaluator": evaluator})
                scores.append(report.scores[i] if i < len(report.scores) else 0.0)
                passes.append(report.test_passes[i] if i < len(report.test_passes) else False)
                reasons.append(report.reasons[i] if i < len(report.reasons) else "")
                detailed.append(report.detailed_results[i] if i < len(report.detailed_results) else [])
                diags.append(report.diagnoses[i] if i < len(report.diagnoses) else None)
                recs.append(report.recommendations[i] if i < len(report.recommendations) else None)

        return cls(
            evaluator_name="Combined",
            overall_score=sum(scores) / len(scores) if scores else 0.0,
            scores=scores,
            cases=cases,
            test_passes=passes,
            reasons=reasons,
            detailed_results=detailed,
            diagnoses=diags,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # DataFrame surface
    #
    # An EvaluationReport behaves like a pandas DataFrame: a single
    # `to_dataframe` touch point coerces it to a frame, and grouping /
    # aggregation / filtering / description piggyback on pandas. Comparison
    # (deltas, A/B, statistical tests) is a separate, composable layer that
    # operates on the frame these produce.
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Coerce this report into a one-row-per-case ``DataFrame``.

        Columns: ``case``, ``evaluator``, ``score``, ``passed``, ``reason``,
        plus one ``metadata.<key>`` column per metadata key. The ``evaluator``
        column prefers a case-level ``evaluator`` (set by :meth:`flatten`),
        falling back to this report's ``evaluator_name``. A report with no
        cases yields an empty frame with the base columns present.
        """
        cases = self.cases or []
        scores = self.scores or []
        passes = self.test_passes or []
        reasons = self.reasons or []

        rows: list[dict] = []
        for i, case in enumerate(cases):
            metadata = case.get("metadata") or {}
            row = {
                CASE: _case_key(case, i),
                EVALUATOR: case.get("evaluator") or self.evaluator_name or "Unknown",
                SCORE: float(scores[i]) if i < len(scores) else float("nan"),
                PASSED: bool(passes[i]) if i < len(passes) else False,
                REASON: reasons[i] if i < len(reasons) else "",
            }
            for key, value in metadata.items():
                # case_key is already promoted to the `case` column; don't duplicate.
                if key == "case_key":
                    continue
                row[f"{METADATA_PREFIX}{key}"] = value
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=_BASE_COLUMNS)

        frame = pd.DataFrame(rows)
        meta_cols = [c for c in frame.columns if c.startswith(METADATA_PREFIX)]
        ordered = [c for c in _BASE_COLUMNS if c in frame.columns] + sorted(meta_cols)
        return frame[ordered]

    def frame(self) -> ReportFrame:
        """Return this report as a :class:`ReportFrame`."""
        return ReportFrame(self.to_dataframe())

    def group_by(self, by: str | list[str], **kwargs: Any) -> ReportGroupBy:
        """Group this report's rows. See :meth:`ReportFrame.group_by`.

        Single report::

            report.group_by("evaluator").agg(mean_score=("score", "mean"))

        Across reports, flatten first::

            EvaluationReport.flatten(reports).group_by("evaluator").agg(...)
        """
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

    @staticmethod
    def _format_input_for_display(input_value: object) -> str:
        """Format an input value for display, handling multimodal inputs gracefully.

        Detects serialized multimodal inputs (dicts with 'media' and 'instruction' keys)
        and renders a readable summary instead of dumping raw binary or base64 data.
        """
        if isinstance(input_value, dict) and "media" in input_value and "instruction" in input_value:
            instruction = input_value.get("instruction", "")
            context = input_value.get("context")
            media = input_value.get("media")

            # Describe media without dumping raw data
            if isinstance(media, list):
                media_desc = f"[{len(media)} media item(s)]"
            elif isinstance(media, dict):
                media_desc = "[1 media item]"
            elif isinstance(media, str) and len(media) > 200:
                media_desc = f"[media: {media[:50]}...]"
            else:
                media_desc = f"[media: {media}]"

            parts = [f"instruction: {instruction}"]
            if context:
                parts.append(f"context: {context}")
            parts.append(media_desc)
            return " | ".join(parts)
        return str(input_value)

    def _display(
        self,
        static: bool = True,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
        include_recommendations: bool = False,
    ):
        """
        Render an interface of the report with as much details as configured using Rich.

        Args:
            static: Whether to render the interface as interactive or static.
            include_input (Defaults to True): Include the input in the display.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.
            include_recommendations (Defaults to False): Include diagnosis recommendations in the display.

        Note:
            This method provides an interactive console interface where users can expand or collapse
            individual test cases to view more or less detail.
        """
        report_data = {}
        for i in range(len(self.scores)):
            name = self.cases[i].get("name", f"Test {i + 1}")
            reason = self.reasons[i] if i < len(self.reasons) else "N/A"
            details_dict = {"name": name}
            # Include evaluator column for flattened reports (right after name)
            if "evaluator" in self.cases[i]:
                details_dict["evaluator"] = self.cases[i]["evaluator"]
            details_dict["score"] = f"{self.scores[i]:.2f}"
            details_dict["test_pass"] = self.test_passes[i]
            details_dict["reason"] = reason
            if include_input:
                details_dict["input"] = self._format_input_for_display(self.cases[i].get("input"))
            if include_actual_output:
                details_dict["actual_output"] = str(self.cases[i].get("actual_output"))
            if include_expected_output:
                details_dict["expected_output"] = str(self.cases[i].get("expected_output"))
            if include_actual_trajectory:
                details_dict["actual_trajectory"] = str(self.cases[i].get("actual_trajectory"))
            if include_expected_trajectory:
                details_dict["expected_trajectory"] = str(self.cases[i].get("expected_trajectory"))
            if include_actual_interactions:
                details_dict["actual_interactions"] = str(self.cases[i].get("actual_interactions"))
            if include_expected_interactions:
                details_dict["expected_interactions"] = str(self.cases[i].get("expected_interactions"))
            if include_meta:
                details_dict["metadata"] = str(self.cases[i].get("metadata"))
            if include_recommendations:
                rec = self.recommendations[i] if i < len(self.recommendations) else None
                if rec is not None:
                    details_dict["recommendation"] = rec

            report_data[str(i)] = {
                "details": details_dict,
                "detailed_results": self.detailed_results[i] if i < len(self.detailed_results) else [],  # NEW
                "expanded": False,
            }

        display_console = CollapsibleTableReportDisplay(items=report_data, overall_score=self.overall_score)
        display_console.run(static=static)

    def display(
        self,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
        include_recommendations: bool = False,
    ):
        """
        Render the report with as much details as configured using Rich. Use run_display if want
        to interact with the table.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.
            include_recommendations (Defaults to False): Include diagnosis recommendations in the display.
        """
        self._display(
            static=True,
            include_input=include_input,
            include_actual_output=include_actual_output,
            include_expected_output=include_expected_output,
            include_expected_trajectory=include_expected_trajectory,
            include_actual_trajectory=include_actual_trajectory,
            include_actual_interactions=include_actual_interactions,
            include_expected_interactions=include_expected_interactions,
            include_meta=include_meta,
            include_recommendations=include_recommendations,
        )

    def run_display(
        self,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
        include_recommendations: bool = False,
    ):
        """
        Render the report interactively with as much details as configured using Rich.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.
            include_recommendations (Defaults to False): Include diagnosis recommendations in the display.
        """
        self._display(
            static=False,
            include_input=include_input,
            include_actual_output=include_actual_output,
            include_expected_output=include_expected_output,
            include_expected_trajectory=include_expected_trajectory,
            include_actual_trajectory=include_actual_trajectory,
            include_actual_interactions=include_actual_interactions,
            include_expected_interactions=include_expected_interactions,
            include_meta=include_meta,
            include_recommendations=include_recommendations,
        )

    def to_dict(self):
        """
        Returns a dictionary representation of the report.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an EvaluationReport instance from a dictionary.

        Args:
            data: A dictionary containing the report data.
        """
        return cls.model_validate(data)

    def to_file(self, path: str):
        """
        Write the report to a JSON file.

        Args:
            path: The file path where the report will be saved. Can be:
                  - A filename only (e.g., "foo.json" or "foo") - saves in current working directory
                  - A relative path (e.g., "relative_path/foo.json") - saves relative to current working directory
                  - An absolute path (e.g., "/path/to/dir/foo.json") - saves in exact directory

                  If no extension is provided, ".json" will be added automatically.
                  Only .json format is supported.

        Raises:
            ValueError: If the path has a non-JSON extension.
        """
        file_path = Path(path)

        if file_path.suffix:
            if file_path.suffix != ".json":
                raise ValueError(
                    f"Only .json format is supported. Got path with extension: {path}. "
                    f"Please use a .json extension or provide a path without an extension."
                )
        else:
            file_path = file_path.with_suffix(".json")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str):
        """
        Create an EvaluationReport instance from a JSON file.

        Args:
            path: Path to the JSON file.

        Return:
            An EvaluationReport object.

        Raises:
            ValueError: If the file does not have a .json extension.
        """
        file_path = Path(path)

        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}. Please provide a path with .json extension."
            )

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)
