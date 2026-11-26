import json

import pytest

from strands_evals.trend import (
    ExperimentTrendAnalyzer,
    ExperimentTrendReport,
    MetricTrend,
    RunPoint,
    _direction,
    _pass_rate,
    _trend,
)
from strands_evals.types.evaluation_report import EvaluationReport


def _report(overall_score: float, test_passes: list[bool] | None = None) -> EvaluationReport:
    """A minimal report carrying just the fields the analyzer reads."""
    passes = test_passes if test_passes is not None else []
    scores = [1.0 if p else 0.0 for p in passes]
    cases = [{"name": f"case_{i}"} for i in range(len(passes))]
    return EvaluationReport(overall_score=overall_score, scores=scores, cases=cases, test_passes=passes)


class TestDirection:
    """Near-zero slopes read as stable so run-to-run jitter is not a false trend."""

    def test_clear_rise_is_improving(self):
        assert _direction(0.1) == "improving"

    def test_clear_fall_is_degrading(self):
        assert _direction(-0.1) == "degrading"

    def test_zero_is_stable(self):
        assert _direction(0.0) == "stable"

    def test_tiny_positive_slope_is_stable(self):
        assert _direction(0.001) == "stable"

    def test_tiny_negative_slope_is_stable(self):
        assert _direction(-0.001) == "stable"

    def test_epsilon_boundary_is_stable(self):
        assert _direction(0.005) == "stable"


class TestTrend:
    """Fit a slope to one metric's run-ordered values."""

    def test_monotonic_decline_is_degrading(self):
        trend = _trend([0.91, 0.85, 0.78, 0.71])
        assert trend.direction == "degrading"
        assert trend.slope < 0
        assert trend.first == 0.91
        assert trend.last == 0.71
        assert trend.window == 4

    def test_monotonic_rise_is_improving(self):
        trend = _trend([0.5, 0.6, 0.7, 0.8])
        assert trend.direction == "improving"
        assert trend.slope > 0

    def test_flat_values_are_stable(self):
        trend = _trend([0.8, 0.8, 0.8])
        assert trend.direction == "stable"
        assert trend.slope == 0.0

    def test_single_run_is_stable_with_zero_slope(self):
        trend = _trend([0.75])
        assert trend == MetricTrend(slope=0.0, direction="stable", first=0.75, last=0.75, window=1)

    def test_empty_values_raise(self):
        with pytest.raises(ValueError, match="zero runs"):
            _trend([])


class TestPassRate:
    """Pass rate comes from test_passes, falling back to overall_score."""

    def test_pass_rate_from_test_passes(self):
        assert _pass_rate(_report(0.9, [True, True, False, True])) == 0.75

    def test_falls_back_to_overall_score_when_no_passes(self):
        assert _pass_rate(_report(0.42, [])) == 0.42


class TestAnalyzer:
    """End-to-end trend analysis over a window of reports."""

    def test_detects_regression_across_runs(self):
        reports = [_report(s, [True] * round(s * 4) + [False] * (4 - round(s * 4))) for s in (0.75, 0.5, 0.25, 0.0)]
        result = ExperimentTrendAnalyzer(reports).analyze()

        assert isinstance(result, ExperimentTrendReport)
        assert result.overall_score_trend.direction == "degrading"
        assert result.pass_rate_trend.direction == "degrading"
        assert result.any_regression is True
        assert [r.run_id for r in result.runs] == ["run_0", "run_1", "run_2", "run_3"]

    def test_improving_run_has_no_regression(self):
        reports = [_report(s) for s in (0.2, 0.4, 0.6, 0.8)]
        result = ExperimentTrendAnalyzer(reports).analyze()
        assert result.overall_score_trend.direction == "improving"
        assert result.any_regression is False

    def test_stable_run_has_no_regression(self):
        reports = [_report(0.8) for _ in range(3)]
        result = ExperimentTrendAnalyzer(reports).analyze()
        assert result.overall_score_trend.direction == "stable"
        assert result.any_regression is False

    def test_regression_flags_when_only_pass_rate_degrades(self):
        # overall_score held flat while the pass rate slid: still a regression to surface.
        reports = [
            _report(0.8, [True, True, True, True]),
            _report(0.8, [True, True, True, False]),
            _report(0.8, [True, True, False, False]),
        ]
        result = ExperimentTrendAnalyzer(reports).analyze()
        assert result.overall_score_trend.direction == "stable"
        assert result.pass_rate_trend.direction == "degrading"
        assert result.any_regression is True

    def test_custom_run_ids(self):
        reports = [_report(0.9), _report(0.8)]
        result = ExperimentTrendAnalyzer(reports, run_ids=["nightly-1", "nightly-2"]).analyze()
        assert [r.run_id for r in result.runs] == ["nightly-1", "nightly-2"]

    def test_mismatched_run_ids_raise(self):
        with pytest.raises(ValueError, match="run_ids has 1 entries"):
            ExperimentTrendAnalyzer([_report(0.9), _report(0.8)], run_ids=["only-one"])

    def test_analyze_empty_raises(self):
        with pytest.raises(ValueError, match="zero runs"):
            ExperimentTrendAnalyzer([]).analyze()

    def test_single_run_is_stable(self):
        result = ExperimentTrendAnalyzer([_report(0.7)]).analyze()
        assert result.overall_score_trend.window == 1
        assert result.overall_score_trend.direction == "stable"
        assert result.any_regression is False


class TestFromFiles:
    """Load reports written by EvaluationReport.to_file."""

    def _write(self, tmp_path, name, overall_score, passes):
        report = _report(overall_score, passes)
        path = tmp_path / f"{name}.json"
        report.to_file(str(path))
        return path

    def test_reads_reports_and_uses_file_stems_as_run_ids(self, tmp_path):
        paths = [
            self._write(tmp_path, "run_0", 0.9, [True, True]),
            self._write(tmp_path, "run_1", 0.5, [True, False]),
        ]
        result = ExperimentTrendAnalyzer.from_files(paths).analyze()
        assert [r.run_id for r in result.runs] == ["run_0", "run_1"]
        assert result.overall_score_trend.direction == "degrading"

    def test_window_keeps_most_recent_runs(self, tmp_path):
        paths = [self._write(tmp_path, f"run_{i}", 0.9 - i * 0.1, [True]) for i in range(5)]
        result = ExperimentTrendAnalyzer.from_files(paths, window=2).analyze()
        assert [r.run_id for r in result.runs] == ["run_3", "run_4"]
        assert result.overall_score_trend.window == 2

    def test_non_positive_window_raises(self, tmp_path):
        paths = [self._write(tmp_path, "run_0", 0.9, [True])]
        with pytest.raises(ValueError, match="window must be a positive integer"):
            ExperimentTrendAnalyzer.from_files(paths, window=0)


class TestSerialization:
    """The analysis serializes to JSON for downstream reporting."""

    def test_to_dict_shape(self):
        reports = [_report(0.9, [True]), _report(0.7, [False])]
        data = ExperimentTrendAnalyzer(reports).to_dict()
        assert set(data) == {"runs", "overall_score_trend", "pass_rate_trend", "any_regression"}
        assert data["runs"][0] == {"run_id": "run_0", "overall_score": 0.9, "pass_rate": 1.0}
        assert set(data["overall_score_trend"]) == {"slope", "direction", "first", "last", "window"}

    def test_to_file_round_trips(self, tmp_path):
        reports = [_report(0.9), _report(0.6)]
        out = tmp_path / "trend.json"
        ExperimentTrendAnalyzer(reports).to_file(str(out))

        data = json.loads(out.read_text())
        assert data["overall_score_trend"]["direction"] == "degrading"
        assert data["any_regression"] is True

    def test_to_file_rejects_non_json_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Only .json format is supported"):
            ExperimentTrendAnalyzer([_report(0.9)]).to_file(str(tmp_path / "trend.txt"))

    def test_run_point_and_metric_trend_are_dataclasses(self):
        point = RunPoint(run_id="r", overall_score=0.5, pass_rate=0.5)
        assert vars(point) == {"run_id": "r", "overall_score": 0.5, "pass_rate": 0.5}
