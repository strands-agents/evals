"""Cross-run trend analysis for sequential experiment reports.

A single `EvaluationReport` captures one run: its `overall_score` and the per-case
`test_passes` behind it. What it cannot say is whether the agent is getting better or worse,
because each run is a fresh report with no link to the ones before it. A sequence like
`0.91 -> 0.85 -> 0.78 -> 0.71` is a steady regression, but nothing connects those four numbers.

`ExperimentTrendReport` closes that gap. Given several reports in run order, it fits a slope to
`overall_score` and to the pass rate, labels each direction, and flags a regression when either
is trending down. The analysis is pure: it reads reports and returns a report, and never runs an
experiment or mutates one.
"""

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .types.evaluation_report import EvaluationReport

# A slope this shallow is noise, not a trend. Runs jitter a little even when nothing changed, so a
# tiny non-zero slope should read as "stable" rather than a spurious improvement or regression.
_STABLE_SLOPE_EPSILON = 0.005


@dataclass
class RunPoint:
    """One run's headline numbers, in the order it was observed.

    Attributes:
        run_id: An identifier for the run. When built from files, this is the file stem.
        overall_score: The run's `EvaluationReport.overall_score`.
        pass_rate: Fraction of cases that passed, from `test_passes`. Falls back to
            `overall_score` when a report carries no `test_passes`.
    """

    run_id: str
    overall_score: float
    pass_rate: float


@dataclass
class MetricTrend:
    """The shape of one metric across runs.

    Attributes:
        slope: Ordinary-least-squares slope of the metric against run index. Positive means the
            metric rose over the window; negative means it fell. Zero for a single run.
        direction: `"improving"`, `"degrading"`, or `"stable"`. Slopes within
            `_STABLE_SLOPE_EPSILON` of zero read as stable.
        first: The metric's value on the earliest run in the window.
        last: The metric's value on the latest run in the window.
        window: How many runs contributed to the fit.
    """

    slope: float
    direction: str
    first: float
    last: float
    window: int


@dataclass
class ExperimentTrendReport:
    """The result of analyzing a sequence of runs.

    Attributes:
        runs: The `RunPoint`s that were analyzed, in run order.
        overall_score_trend: Trend of `overall_score` across the runs.
        pass_rate_trend: Trend of the pass rate across the runs.
        any_regression: True when either metric is degrading.
    """

    runs: list[RunPoint]
    overall_score_trend: MetricTrend
    pass_rate_trend: MetricTrend
    any_regression: bool


def _direction(slope: float) -> str:
    """Label a slope, treating near-zero slopes as stable."""
    if slope > _STABLE_SLOPE_EPSILON:
        return "improving"
    if slope < -_STABLE_SLOPE_EPSILON:
        return "degrading"
    return "stable"


def _trend(values: Sequence[float]) -> MetricTrend:
    """Fit a slope to one metric's values, which are already in run order.

    A single run has no slope to fit, so its trend is stable with a zero slope. `first` and `last`
    still report that run's value.
    """
    if not values:
        raise ValueError("cannot compute a trend from zero runs")

    if len(values) == 1:
        return MetricTrend(slope=0.0, direction="stable", first=values[0], last=values[0], window=1)

    xs = list(range(len(values)))
    slope, _ = statistics.linear_regression(xs, values)
    return MetricTrend(
        slope=slope,
        direction=_direction(slope),
        first=values[0],
        last=values[-1],
        window=len(values),
    )


def _pass_rate(report: EvaluationReport) -> float:
    """A report's pass rate, falling back to `overall_score` when it has no `test_passes`."""
    if report.test_passes:
        return sum(report.test_passes) / len(report.test_passes)
    return report.overall_score


class ExperimentTrendAnalyzer:
    """Fit trends across a window of sequential experiment runs.

    The analyzer never runs an experiment; it reads reports that were already produced (typically
    saved with `EvaluationReport.to_file`) and reports how their headline metrics move. Runs are
    taken in the order given — sort the paths yourself if filenames do not sort chronologically.
    """

    def __init__(self, reports: Sequence[EvaluationReport], run_ids: Sequence[str] | None = None):
        """Build an analyzer from in-memory reports.

        Args:
            reports: `EvaluationReport`s in run order (earliest first).
            run_ids: Optional labels, one per report. Defaults to `run_0`, `run_1`, ....

        Raises:
            ValueError: If `run_ids` is given but its length does not match `reports`.
        """
        if run_ids is not None and len(run_ids) != len(reports):
            raise ValueError(f"run_ids has {len(run_ids)} entries but there are {len(reports)} reports")

        self._reports = list(reports)
        self._run_ids = list(run_ids) if run_ids is not None else [f"run_{i}" for i in range(len(reports))]

    @classmethod
    def from_files(cls, report_paths: Sequence[str | Path], window: int | None = None) -> "ExperimentTrendAnalyzer":
        """Build an analyzer from saved report files.

        Args:
            report_paths: Paths to JSON reports written by `EvaluationReport.to_file`, in run
                order (earliest first). Sort them yourself if needed.
            window: Keep only the most recent `window` runs. `None` keeps them all.

        Returns:
            An analyzer over the (optionally windowed) reports, each run id taken from its file stem.

        Raises:
            ValueError: If `window` is not a positive integer.
        """
        if window is not None and window <= 0:
            raise ValueError(f"window must be a positive integer, got {window}")

        paths = [Path(p) for p in report_paths]
        if window is not None:
            paths = paths[-window:]

        reports = [EvaluationReport.from_file(str(p)) for p in paths]
        run_ids = [p.stem for p in paths]
        return cls(reports, run_ids)

    def analyze(self) -> ExperimentTrendReport:
        """Compute the trend report over the configured runs.

        Returns:
            An `ExperimentTrendReport` with per-run points and the two metric trends.

        Raises:
            ValueError: If there are no runs to analyze.
        """
        if not self._reports:
            raise ValueError("cannot analyze zero runs")

        runs = [
            RunPoint(run_id=run_id, overall_score=report.overall_score, pass_rate=_pass_rate(report))
            for run_id, report in zip(self._run_ids, self._reports, strict=True)
        ]

        overall_trend = _trend([r.overall_score for r in runs])
        pass_trend = _trend([r.pass_rate for r in runs])

        return ExperimentTrendReport(
            runs=runs,
            overall_score_trend=overall_trend,
            pass_rate_trend=pass_trend,
            any_regression=overall_trend.direction == "degrading" or pass_trend.direction == "degrading",
        )

    def to_dict(self) -> dict:
        """Return the analysis as a JSON-serializable dict."""
        report = self.analyze()
        return {
            "runs": [vars(run) for run in report.runs],
            "overall_score_trend": vars(report.overall_score_trend),
            "pass_rate_trend": vars(report.pass_rate_trend),
            "any_regression": report.any_regression,
        }

    def to_file(self, path: str) -> None:
        """Write the analysis to a JSON file, adding a `.json` suffix when none is given."""
        file_path = Path(path)
        if file_path.suffix and file_path.suffix != ".json":
            raise ValueError(f"Only .json format is supported. Got path with extension: {path}.")
        if not file_path.suffix:
            file_path = file_path.with_suffix(".json")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
