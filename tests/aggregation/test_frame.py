"""Tests for aggregation/frame.py — ReportFrame and ReportGroupBy."""

from __future__ import annotations

import pandas as pd
import pytest

from strands_evals.aggregation import ReportFrame, ReportGroupBy


def _frame() -> ReportFrame:
    return ReportFrame(
        pd.DataFrame(
            {
                "case": ["foo", "bar", "foo", "bar"],
                "evaluator": ["GoalSuccess", "GoalSuccess", "ToolAccuracy", "ToolAccuracy"],
                "score": [0.9, 0.6, 0.8, 0.4],
                "passed": [True, True, True, False],
            }
        )
    )


# ---------------------------------------------------------------------------
# group_by / agg
# ---------------------------------------------------------------------------


def test_group_by_returns_report_groupby():
    assert isinstance(_frame().group_by("evaluator"), ReportGroupBy)


def test_agg_named_aggregation_returns_report_frame():
    result = _frame().group_by("evaluator").agg(
        mean_score=("score", "mean"),
        pass_rate=("passed", "mean"),
        n=("score", "count"),
    )
    assert isinstance(result, ReportFrame)
    df = result.df
    assert set(df["evaluator"]) == {"GoalSuccess", "ToolAccuracy"}
    goal = df[df["evaluator"] == "GoalSuccess"].iloc[0]
    assert goal["mean_score"] == pytest.approx(0.75)
    assert goal["pass_rate"] == pytest.approx(1.0)
    assert goal["n"] == 2


def test_agg_result_has_group_key_as_column_not_index():
    result = _frame().group_by("evaluator").agg(mean_score=("score", "mean"))
    assert "evaluator" in result.df.columns


def test_group_by_multiple_columns():
    result = _frame().group_by(["evaluator", "case"]).agg(mean_score=("score", "mean"))
    assert len(result.df) == 4  # 2 evaluators x 2 cases


def test_groupby_passthrough_method_rewrapped():
    # A pandas groupby method (mean) flows through and is re-wrapped.
    result = _frame().group_by("evaluator").mean(numeric_only=True)
    assert isinstance(result, ReportFrame)
    assert "score" in result.df.columns


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_with_mask():
    f = _frame()
    result = f.filter(f.df["passed"])
    assert isinstance(result, ReportFrame)
    assert len(result) == 3
    assert result.df["passed"].all()


def test_filter_with_callable():
    result = _frame().filter(lambda d: d["score"] >= 0.8)
    assert len(result) == 2
    assert (result.df["score"] >= 0.8).all()


def test_filter_resets_index():
    result = _frame().filter(lambda d: d["score"] >= 0.8)
    assert list(result.df.index) == [0, 1]


# ---------------------------------------------------------------------------
# describe / passthrough
# ---------------------------------------------------------------------------


def test_describe_returns_report_frame():
    result = _frame().describe()
    assert isinstance(result, ReportFrame)
    # describe puts the stat labels (count/mean/...) into an "index" column
    assert "index" in result.df.columns


def test_passthrough_head_rewrapped():
    result = _frame().head(2)
    assert isinstance(result, ReportFrame)
    assert len(result) == 2


def test_passthrough_non_dataframe_result_not_wrapped():
    # .shape returns a tuple -> returned as-is
    assert _frame().shape == (4, 4)


def test_len_and_df_property():
    f = _frame()
    assert len(f) == 4
    assert isinstance(f.df, pd.DataFrame)


def test_to_pandas_returns_copy():
    f = _frame()
    copy = f.to_pandas()
    copy.loc[0, "score"] = 0.0
    assert f.df.loc[0, "score"] == 0.9  # original unchanged


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_to_file_and_from_file_roundtrip(tmp_path):
    path = tmp_path / "agg.json"
    original = _frame().group_by("evaluator").agg(mean_score=("score", "mean"))
    original.to_file(str(path))
    assert path.exists()
    loaded = ReportFrame.from_file(str(path))
    pd.testing.assert_frame_equal(
        original.df.reset_index(drop=True),
        loaded.df.reset_index(drop=True),
        check_dtype=False,
    )


def test_to_file_appends_json_suffix(tmp_path):
    path = tmp_path / "agg"
    _frame().to_file(str(path))
    assert (tmp_path / "agg.json").exists()


def test_to_file_rejects_non_json_extension(tmp_path):
    with pytest.raises(ValueError):
        _frame().to_file(str(tmp_path / "agg.csv"))


def test_from_file_rejects_non_json_extension(tmp_path):
    with pytest.raises(ValueError):
        ReportFrame.from_file(str(tmp_path / "agg.csv"))


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


def test_display_runs_on_populated_frame(capsys):
    _frame().display()
    out = capsys.readouterr().out
    assert "GoalSuccess" in out


def test_display_runs_on_empty_frame(capsys):
    ReportFrame(pd.DataFrame(columns=["case", "score"])).display()
    out = capsys.readouterr().out
    assert "No rows" in out


def test_repr_is_informative():
    assert "ReportFrame" in repr(_frame())
