import json
import tempfile
from pathlib import Path

import pytest

from strands_evals.types.evaluation_report import EvaluationReport


class TestEvaluationReportEvaluatorName:
    """Tests for the evaluator_name field on EvaluationReport."""

    def test_evaluator_name_default_empty(self):
        """Test that evaluator_name defaults to empty string."""
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "test"}],
            test_passes=[True],
        )
        assert report.evaluator_name == ""

    def test_evaluator_name_set(self):
        """Test that evaluator_name can be set."""
        report = EvaluationReport(
            evaluator_name="ResponseRelevance",
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "test"}],
            test_passes=[True],
        )
        assert report.evaluator_name == "ResponseRelevance"

    def test_evaluator_name_serialization(self):
        """Test that evaluator_name is included in serialization."""
        report = EvaluationReport(
            evaluator_name="TestEvaluator",
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "test"}],
            test_passes=[True],
        )
        data = report.to_dict()
        assert data["evaluator_name"] == "TestEvaluator"

    def test_evaluator_name_deserialization(self):
        """Test that evaluator_name is restored from dict."""
        data = {
            "evaluator_name": "TestEvaluator",
            "overall_score": 0.5,
            "scores": [0.5],
            "cases": [{"name": "test"}],
            "test_passes": [True],
        }
        report = EvaluationReport.from_dict(data)
        assert report.evaluator_name == "TestEvaluator"

    def test_evaluator_name_backward_compatible(self):
        """Test that reports without evaluator_name can still be loaded."""
        data = {
            "overall_score": 0.5,
            "scores": [0.5],
            "cases": [{"name": "test"}],
            "test_passes": [True],
        }
        report = EvaluationReport.from_dict(data)
        assert report.evaluator_name == ""


class TestEvaluationReportFlatten:
    """Tests for the flatten() classmethod."""

    def test_flatten_empty_list(self):
        """Test flattening an empty list returns empty report."""
        flattened = EvaluationReport.flatten([])
        assert flattened.evaluator_name == ""
        assert flattened.overall_score == 0.0
        assert flattened.scores == []
        assert flattened.cases == []
        assert flattened.test_passes == []

    def test_flatten_single_report(self):
        """Test flattening a single report."""
        report = EvaluationReport(
            evaluator_name="Evaluator1",
            overall_score=0.8,
            scores=[0.9, 0.7],
            cases=[
                {"name": "case-1", "input": "input1"},
                {"name": "case-2", "input": "input2"},
            ],
            test_passes=[True, True],
            reasons=["reason1", "reason2"],
        )

        flattened = EvaluationReport.flatten([report])

        assert flattened.evaluator_name == "Combined"
        assert flattened.overall_score == 0.8
        assert len(flattened.cases) == 2
        assert flattened.cases[0]["evaluator"] == "Evaluator1"
        assert flattened.cases[1]["evaluator"] == "Evaluator1"

    def test_flatten_multiple_reports(self):
        """Test flattening multiple reports."""
        report1 = EvaluationReport(
            evaluator_name="ResponseRelevance",
            overall_score=0.85,
            scores=[0.9, 0.8],
            cases=[
                {"name": "case-1", "input": "input1"},
                {"name": "case-2", "input": "input2"},
            ],
            test_passes=[True, True],
            reasons=["relevant", "relevant"],
        )

        report2 = EvaluationReport(
            evaluator_name="Equals",
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[
                {"name": "case-1", "input": "input1"},
                {"name": "case-2", "input": "input2"},
            ],
            test_passes=[False, False],
            reasons=["not equal", "not equal"],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        assert flattened.evaluator_name == "Combined"
        assert flattened.overall_score == pytest.approx(0.425)
        assert len(flattened.cases) == 4
        assert len(flattened.scores) == 4
        assert len(flattened.test_passes) == 4
        assert len(flattened.reasons) == 4

    def test_flatten_preserves_case_data(self):
        """Test that flattening preserves all case data."""
        report = EvaluationReport(
            evaluator_name="Test",
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "case-1", "input": "test input", "metadata": {"key": "value"}}],
            test_passes=[True],
            reasons=["test reason"],
        )

        flattened = EvaluationReport.flatten([report])

        assert flattened.cases[0]["name"] == "case-1"
        assert flattened.cases[0]["input"] == "test input"
        assert flattened.cases[0]["metadata"] == {"key": "value"}
        assert flattened.cases[0]["evaluator"] == "Test"

    def test_flatten_adds_evaluator_field(self):
        """Test that flatten adds evaluator field to each case."""
        report1 = EvaluationReport(
            evaluator_name="Eval1",
            overall_score=1.0,
            scores=[1.0],
            cases=[{"name": "case-1"}],
            test_passes=[True],
        )
        report2 = EvaluationReport(
            evaluator_name="Eval2",
            overall_score=0.0,
            scores=[0.0],
            cases=[{"name": "case-1"}],
            test_passes=[False],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        assert flattened.cases[0]["evaluator"] == "Eval1"
        assert flattened.cases[1]["evaluator"] == "Eval2"

    def test_flatten_averages_scores(self):
        """Test that overall_score is the average of all scores."""
        report1 = EvaluationReport(
            evaluator_name="Eval1",
            overall_score=1.0,
            scores=[1.0, 1.0],
            cases=[{"name": "c1"}, {"name": "c2"}],
            test_passes=[True, True],
        )
        report2 = EvaluationReport(
            evaluator_name="Eval2",
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[{"name": "c1"}, {"name": "c2"}],
            test_passes=[False, False],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        # (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5
        assert flattened.overall_score == pytest.approx(0.5)

    def test_flatten_handles_missing_evaluator_name(self):
        """Test that flatten handles reports without evaluator_name."""
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "case-1"}],
            test_passes=[True],
        )

        flattened = EvaluationReport.flatten([report])

        assert flattened.cases[0]["evaluator"] == "Unknown"

    def test_flatten_handles_mismatched_lengths(self):
        """Test that flatten handles reports with mismatched array lengths gracefully."""
        report = EvaluationReport(
            evaluator_name="Test",
            overall_score=0.5,
            scores=[0.5],  # Only 1 score
            cases=[{"name": "c1"}, {"name": "c2"}],  # 2 cases
            test_passes=[True],  # Only 1 pass
            reasons=[],  # No reasons
        )

        flattened = EvaluationReport.flatten([report])

        assert len(flattened.cases) == 2
        assert flattened.scores[0] == 0.5
        assert flattened.scores[1] == 0.0  # Default for missing
        assert flattened.test_passes[0] is True
        assert flattened.test_passes[1] is False  # Default for missing
        assert flattened.reasons[0] == ""
        assert flattened.reasons[1] == ""

    def test_flatten_preserves_detailed_results(self):
        """Test that flatten preserves detailed_results."""
        from strands_evals.types.evaluation import EvaluationOutput

        detailed = [EvaluationOutput(score=0.5, test_pass=True, reason="detail")]
        report = EvaluationReport(
            evaluator_name="Test",
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "case-1"}],
            test_passes=[True],
            detailed_results=[detailed],
        )

        flattened = EvaluationReport.flatten([report])

        assert len(flattened.detailed_results) == 1
        assert flattened.detailed_results[0] == detailed

    def test_flatten_preserves_diagnoses_and_recommendations(self):
        """Test that flatten preserves diagnoses and recommendations."""
        report1 = EvaluationReport(
            evaluator_name="Eval1",
            overall_score=0.0,
            scores=[0.0],
            cases=[{"name": "case-1"}],
            test_passes=[False],
            diagnoses=[{"session_id": "s1", "failures": [], "root_causes": []}],
            recommendations=["Fix the prompt"],
        )
        report2 = EvaluationReport(
            evaluator_name="Eval2",
            overall_score=1.0,
            scores=[1.0],
            cases=[{"name": "case-2"}],
            test_passes=[True],
            diagnoses=[None],
            recommendations=[None],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        assert len(flattened.diagnoses) == 2
        assert flattened.diagnoses[0] == {"session_id": "s1", "failures": [], "root_causes": []}
        assert flattened.diagnoses[1] is None
        assert flattened.recommendations[0] == "Fix the prompt"
        assert flattened.recommendations[1] is None

    def test_flatten_does_not_modify_original(self):
        """Test that flatten does not modify the original reports."""
        original_case = {"name": "case-1", "input": "test"}
        report = EvaluationReport(
            evaluator_name="Test",
            overall_score=0.5,
            scores=[0.5],
            cases=[original_case],
            test_passes=[True],
        )

        flattened = EvaluationReport.flatten([report])

        # Original should not have evaluator field
        assert "evaluator" not in original_case
        # Flattened should have it
        assert "evaluator" in flattened.cases[0]


class TestFormatInputForDisplay:
    """Tests for _format_input_for_display static method."""

    def test_non_multimodal_input_passthrough(self):
        assert EvaluationReport._format_input_for_display("plain text") == "plain text"

    def test_non_dict_input(self):
        assert EvaluationReport._format_input_for_display(42) == "42"

    def test_dict_without_media_key(self):
        result = EvaluationReport._format_input_for_display({"instruction": "test"})
        assert result == "{'instruction': 'test'}"

    def test_multimodal_dict_with_list_media(self):
        input_val = {
            "media": [{"source": "a"}, {"source": "b"}],
            "instruction": "Describe these images",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "instruction: Describe these images" in result
        assert "[2 media item(s)]" in result

    def test_multimodal_dict_with_dict_media(self):
        input_val = {
            "media": {"source": "abc123", "format": "png"},
            "instruction": "Describe this image",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "instruction: Describe this image" in result
        assert "[1 media item]" in result

    def test_multimodal_dict_with_long_string_media(self):
        long_base64 = "a" * 300
        input_val = {
            "media": long_base64,
            "instruction": "Describe",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "[media: " + "a" * 50 + "...]" in result

    def test_multimodal_dict_with_short_string_media(self):
        input_val = {
            "media": "image.png",
            "instruction": "Describe",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "[media: image.png]" in result

    def test_multimodal_dict_with_context(self):
        input_val = {
            "media": {"source": "abc"},
            "instruction": "Analyze",
            "context": "You are a botanist",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "instruction: Analyze" in result
        assert "context: You are a botanist" in result
        assert "[1 media item]" in result

    def test_multimodal_dict_without_context(self):
        input_val = {
            "media": {"source": "abc"},
            "instruction": "Describe",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        assert "context" not in result

    def test_multimodal_dict_parts_joined_with_pipe(self):
        input_val = {
            "media": {"source": "abc"},
            "instruction": "Describe",
            "context": "Context here",
        }
        result = EvaluationReport._format_input_for_display(input_val)

        parts = result.split(" | ")
        assert len(parts) == 3
        assert parts[0] == "instruction: Describe"
        assert parts[1] == "context: Context here"
        assert parts[2] == "[1 media item]"


class TestEvaluationReportFileOperations:
    """Tests for file save/load with evaluator_name."""

    def test_to_file_includes_evaluator_name(self):
        """Test that to_file includes evaluator_name in JSON."""
        report = EvaluationReport(
            evaluator_name="TestEval",
            overall_score=0.5,
            scores=[0.5],
            cases=[{"name": "test"}],
            test_passes=[True],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.to_file(str(path))

            with open(path) as f:
                data = json.load(f)

            assert data["evaluator_name"] == "TestEval"

    def test_from_file_loads_evaluator_name(self):
        """Test that from_file loads evaluator_name."""
        data = {
            "evaluator_name": "LoadedEval",
            "overall_score": 0.5,
            "scores": [0.5],
            "cases": [{"name": "test"}],
            "test_passes": [True],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            with open(path, "w") as f:
                json.dump(data, f)

            report = EvaluationReport.from_file(str(path))

            assert report.evaluator_name == "LoadedEval"

    def test_flattened_report_roundtrip(self):
        """Test that a flattened report can be saved and loaded."""
        report1 = EvaluationReport(
            evaluator_name="Eval1",
            overall_score=1.0,
            scores=[1.0],
            cases=[{"name": "case-1"}],
            test_passes=[True],
            reasons=["pass"],
        )
        report2 = EvaluationReport(
            evaluator_name="Eval2",
            overall_score=0.0,
            scores=[0.0],
            cases=[{"name": "case-1"}],
            test_passes=[False],
            reasons=["fail"],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flattened.json"
            flattened.to_file(str(path))
            loaded = EvaluationReport.from_file(str(path))

            assert loaded.evaluator_name == "Combined"
            assert loaded.overall_score == pytest.approx(0.5)
            assert len(loaded.cases) == 2
            assert loaded.cases[0]["evaluator"] == "Eval1"
            assert loaded.cases[1]["evaluator"] == "Eval2"
