"""Tests for the EvaluationOutput.status field and its effect on aggregation."""

import pytest

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


class TestEvaluationOutputStatusField:
    """Tests for the status field on EvaluationOutput."""

    def test_default_status_is_graded(self):
        output = EvaluationOutput(score=1.0, test_pass=True, reason="good")
        assert output.status == "graded"

    def test_status_could_not_evaluate(self):
        output = EvaluationOutput(
            score=0.0,
            test_pass=False,
            reason="No tool errors in trajectory",
            status="could_not_evaluate",
        )
        assert output.status == "could_not_evaluate"

    def test_status_informational(self):
        output = EvaluationOutput(
            score=0.0,
            test_pass=False,
            reason="Referenced tickets: TICKET-123, TICKET-456",
            status="informational",
        )
        assert output.status == "informational"

    def test_status_serialization_roundtrip(self):
        output = EvaluationOutput(
            score=0.5,
            test_pass=True,
            reason="partial",
            status="could_not_evaluate",
        )
        data = output.model_dump()
        assert data == {
            "score": 0.5,
            "test_pass": True,
            "reason": "partial",
            "label": None,
            "status": "could_not_evaluate",
        }

        restored = EvaluationOutput.model_validate(data)
        assert restored == output

    def test_status_included_in_model_dump_when_default(self):
        """When status is 'graded', it still appears in model_dump (Pydantic default behavior)."""
        output = EvaluationOutput(score=1.0, test_pass=True)
        assert output.model_dump() == {
            "score": 1.0,
            "test_pass": True,
            "reason": None,
            "label": None,
            "status": "graded",
        }

    def test_backward_compatible_deserialization(self):
        """Old data without a status field should deserialize with default 'graded'."""
        data = {"score": 0.8, "test_pass": True, "reason": "good"}
        output = EvaluationOutput.model_validate(data)
        assert output == EvaluationOutput(score=0.8, test_pass=True, reason="good", status="graded")


class TestDefaultAggregatorWithStatus:
    """Tests for _default_aggregator filtering on status."""

    def test_all_graded_outputs(self):
        outputs = [
            EvaluationOutput(score=1.0, test_pass=True, reason="pass"),
            EvaluationOutput(score=0.5, test_pass=False, reason="partial"),
        ]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        assert score == pytest.approx(0.75)
        assert passed is False
        assert "pass" in reason
        assert "partial" in reason

    def test_mixed_graded_and_could_not_evaluate(self):
        outputs = [
            EvaluationOutput(score=1.0, test_pass=True, reason="pass", status="graded"),
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="No errors in trajectory",
                status="could_not_evaluate",
            ),
        ]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        # Only the graded output should be counted
        assert score == pytest.approx(1.0)
        assert passed is True
        assert "pass" in reason
        assert "No errors in trajectory" not in reason

    def test_all_could_not_evaluate(self):
        outputs = [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="No errors",
                status="could_not_evaluate",
            ),
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="Missing data",
                status="could_not_evaluate",
            ),
        ]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        assert score == 0.0
        assert passed is False
        assert reason == "No gradable evaluation outputs produced"

    def test_all_informational(self):
        outputs = [
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="Referenced: TICKET-1",
                status="informational",
            ),
        ]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        assert score == 0.0
        assert passed is False
        assert reason == "No gradable evaluation outputs produced"

    def test_mixed_graded_and_informational(self):
        outputs = [
            EvaluationOutput(score=0.8, test_pass=True, reason="good", status="graded"),
            EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason="Info only",
                status="informational",
            ),
            EvaluationOutput(score=0.6, test_pass=True, reason="ok", status="graded"),
        ]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        # Only graded: (0.8 + 0.6) / 2 = 0.7
        assert score == pytest.approx(0.7)
        assert passed is True
        assert "good" in reason
        assert "ok" in reason
        assert "Info only" not in reason

    def test_empty_outputs_list(self):
        score, passed, reason = Evaluator._default_aggregator([])
        assert score == 0.0
        assert passed is False
        assert reason == "No gradable evaluation outputs produced"

    def test_single_graded_output(self):
        outputs = [EvaluationOutput(score=0.9, test_pass=True, reason="excellent")]
        score, passed, reason = Evaluator._default_aggregator(outputs)
        assert score == pytest.approx(0.9)
        assert passed is True
        assert reason == "excellent"


class TestEvaluationReportFlattenWithStatus:
    """Tests for EvaluationReport.flatten() respecting statuses."""

    def test_flatten_excludes_non_graded_from_overall_score(self):
        report1 = EvaluationReport(
            overall_score=1.0,
            scores=[1.0],
            cases=[{"name": "case-1", "evaluator": "Eval1"}],
            test_passes=[True],
            statuses=["graded"],
        )
        report2 = EvaluationReport(
            overall_score=0.0,
            scores=[0.0],
            cases=[{"name": "case-1", "evaluator": "Eval2"}],
            test_passes=[False],
            statuses=["could_not_evaluate"],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        # Only the graded score (1.0) should count
        assert flattened.overall_score == pytest.approx(1.0)
        # Both scores are still in the list
        assert flattened.scores == [1.0, 0.0]
        # Statuses are preserved
        assert flattened.statuses == ["graded", "could_not_evaluate"]

    def test_flatten_all_non_graded_gives_zero(self):
        report = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[{"name": "c1", "evaluator": "E"}, {"name": "c2", "evaluator": "E"}],
            test_passes=[False, False],
            statuses=["could_not_evaluate", "informational"],
        )

        flattened = EvaluationReport.flatten([report])
        assert flattened.overall_score == 0.0

    def test_flatten_missing_statuses_defaults_to_graded(self):
        """Legacy reports without statuses should behave as all graded (validator pads them)."""
        report = EvaluationReport(
            overall_score=0.8,
            scores=[0.9, 0.7],
            cases=[{"name": "c1", "evaluator": "E"}, {"name": "c2", "evaluator": "E"}],
            test_passes=[True, True],
            # No statuses field - validator pads to ["graded", "graded"]
        )

        # Validator should have padded statuses
        assert report.statuses == ["graded", "graded"]

        flattened = EvaluationReport.flatten([report])
        assert flattened.overall_score == pytest.approx(0.8)
        assert flattened.statuses == ["graded", "graded"]

    def test_flatten_preserves_statuses_in_roundtrip(self):
        """Statuses survive to_dict/from_dict roundtrip."""
        report = EvaluationReport(
            overall_score=0.5,
            scores=[1.0, 0.0],
            cases=[{"name": "c1"}, {"name": "c2"}],
            test_passes=[True, False],
            statuses=["graded", "could_not_evaluate"],
        )
        data = report.to_dict()
        restored = EvaluationReport.from_dict(data)
        assert restored.statuses == ["graded", "could_not_evaluate"]

    def test_flatten_multiple_reports_mixed_statuses(self):
        report1 = EvaluationReport(
            overall_score=0.5,
            scores=[0.5, 1.0],
            cases=[{"name": "c1", "evaluator": "E1"}, {"name": "c2", "evaluator": "E1"}],
            test_passes=[True, True],
            statuses=["graded", "graded"],
        )
        report2 = EvaluationReport(
            overall_score=0.0,
            scores=[0.0, 0.0],
            cases=[{"name": "c1", "evaluator": "E2"}, {"name": "c2", "evaluator": "E2"}],
            test_passes=[False, False],
            statuses=["could_not_evaluate", "could_not_evaluate"],
        )

        flattened = EvaluationReport.flatten([report1, report2])

        # Only graded scores: (0.5 + 1.0) / 2 = 0.75
        assert flattened.overall_score == pytest.approx(0.75)
        assert len(flattened.scores) == 4
        assert flattened.statuses == ["graded", "graded", "could_not_evaluate", "could_not_evaluate"]


class TestEvaluationReportFileRoundtripWithStatus:
    """Tests for file serialization preserving status."""

    def test_report_with_statuses_roundtrip(self, tmp_path):
        report = EvaluationReport(
            overall_score=0.9,
            scores=[0.9, 0.0],
            cases=[{"name": "c1", "evaluator": "E"}, {"name": "c2", "evaluator": "E"}],
            test_passes=[True, False],
            reasons=["good", "could not evaluate"],
            statuses=["graded", "could_not_evaluate"],
        )

        path = tmp_path / "report.json"
        report.to_file(str(path))
        loaded = EvaluationReport.from_file(str(path))

        assert loaded.overall_score == pytest.approx(0.9)
        assert loaded.statuses == ["graded", "could_not_evaluate"]

    def test_legacy_report_without_statuses_loads(self, tmp_path):
        """Reports saved before the status feature should load with padded 'graded' statuses."""
        import json

        data = {
            "overall_score": 0.5,
            "scores": [0.5],
            "cases": [{"name": "c1"}],
            "test_passes": [True],
            "reasons": ["ok"],
        }
        path = tmp_path / "legacy.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = EvaluationReport.from_file(str(path))
        assert loaded.statuses == ["graded"]


class TestEvaluatorWithStatusEndToEnd:
    """End-to-end tests showing evaluators using the status field."""

    def test_evaluator_returning_could_not_evaluate(self):
        """An evaluator that cannot grade should return status='could_not_evaluate'."""

        class ErrorHandlingEvaluator(Evaluator[str, str]):
            def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
                # Simulate: no errors in trajectory, nothing to evaluate
                if evaluation_case.actual_trajectory is None:
                    return [
                        EvaluationOutput(
                            score=0.0,
                            test_pass=False,
                            reason="No tool errors occurred in trajectory",
                            status="could_not_evaluate",
                        )
                    ]
                return [EvaluationOutput(score=1.0, test_pass=True, reason="Handled correctly")]

        evaluator = ErrorHandlingEvaluator()

        # Case with no trajectory (cannot evaluate)
        case_no_traj = EvaluationData(input="test", actual_output="output")
        outputs = evaluator.evaluate(case_no_traj)
        assert len(outputs) == 1
        assert outputs[0].status == "could_not_evaluate"

        # Aggregation should report no gradable outputs
        score, passed, reason = evaluator.aggregator(outputs)
        assert score == 0.0
        assert passed is False
        assert reason == "No gradable evaluation outputs produced"

    def test_evaluator_returning_informational(self):
        """An evaluator that produces informational results."""

        class TicketEvaluator(Evaluator[str, str]):
            def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
                return [
                    EvaluationOutput(
                        score=0.0,
                        test_pass=False,
                        reason="Referenced tickets: TICKET-123",
                        status="informational",
                    )
                ]

        evaluator = TicketEvaluator()
        outputs = evaluator.evaluate(EvaluationData(input="test", actual_output="output"))

        score, passed, reason = evaluator.aggregator(outputs)
        assert score == 0.0
        assert passed is False
        assert reason == "No gradable evaluation outputs produced"

    def test_mixed_outputs_in_single_evaluate_call(self):
        """An evaluator returning a mix of graded and non-graded in one call."""

        class MultiOutputEvaluator(Evaluator[str, str]):
            def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
                return [
                    EvaluationOutput(score=0.8, test_pass=True, reason="criterion 1 met", status="graded"),
                    EvaluationOutput(
                        score=0.0,
                        test_pass=False,
                        reason="criterion 2 not applicable",
                        status="could_not_evaluate",
                    ),
                    EvaluationOutput(score=0.6, test_pass=True, reason="criterion 3 met", status="graded"),
                ]

        evaluator = MultiOutputEvaluator()
        outputs = evaluator.evaluate(EvaluationData(input="test", actual_output="output"))

        score, passed, reason = evaluator.aggregator(outputs)
        # Only graded: (0.8 + 0.6) / 2 = 0.7
        assert score == pytest.approx(0.7)
        assert passed is True
        assert "criterion 1 met" in reason
        assert "criterion 3 met" in reason
        assert "criterion 2 not applicable" not in reason

    def test_custom_aggregator_can_override_status_filtering(self):
        """Users can still set a custom aggregator that ignores status."""

        def custom_aggregator(outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
            # Include all outputs regardless of status
            avg = sum(o.score for o in outputs) / len(outputs) if outputs else 0.0
            return avg, avg >= 0.5, "custom"

        evaluator = Evaluator()
        evaluator.aggregator = custom_aggregator

        outputs = [
            EvaluationOutput(score=1.0, test_pass=True, status="graded"),
            EvaluationOutput(score=0.0, test_pass=False, status="could_not_evaluate"),
        ]
        score, passed, reason = evaluator.aggregator(outputs)
        assert score == pytest.approx(0.5)
        assert passed is True
        assert reason == "custom"


class TestRollUpStatus:
    """Tests for the _roll_up_status helper function in experiment.py."""

    def test_empty_outputs_returns_graded(self):
        from strands_evals.experiment import _roll_up_status

        assert _roll_up_status([]) == "graded"

    def test_all_graded_returns_graded(self):
        from strands_evals.experiment import _roll_up_status

        outputs = [
            EvaluationOutput(score=1.0, test_pass=True, status="graded"),
            EvaluationOutput(score=0.5, test_pass=False, status="graded"),
        ]
        assert _roll_up_status(outputs) == "graded"

    def test_mixed_graded_and_non_graded_returns_graded(self):
        from strands_evals.experiment import _roll_up_status

        outputs = [
            EvaluationOutput(score=0.0, test_pass=False, status="could_not_evaluate"),
            EvaluationOutput(score=1.0, test_pass=True, status="graded"),
        ]
        assert _roll_up_status(outputs) == "graded"

    def test_all_could_not_evaluate_returns_first_status(self):
        from strands_evals.experiment import _roll_up_status

        outputs = [
            EvaluationOutput(score=0.0, test_pass=False, status="could_not_evaluate"),
            EvaluationOutput(score=0.0, test_pass=False, status="could_not_evaluate"),
        ]
        assert _roll_up_status(outputs) == "could_not_evaluate"

    def test_all_informational_returns_first_status(self):
        from strands_evals.experiment import _roll_up_status

        outputs = [
            EvaluationOutput(score=0.0, test_pass=False, status="informational"),
        ]
        assert _roll_up_status(outputs) == "informational"

    def test_mixed_non_graded_returns_first_status(self):
        from strands_evals.experiment import _roll_up_status

        outputs = [
            EvaluationOutput(score=0.0, test_pass=False, status="informational"),
            EvaluationOutput(score=0.0, test_pass=False, status="could_not_evaluate"),
        ]
        assert _roll_up_status(outputs) == "informational"


class TestEvaluationReportStatusesValidator:
    """Tests for the statuses list padding validator on EvaluationReport."""

    def test_statuses_padded_when_shorter_than_scores(self):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.5, 1.0, 0.8],
            cases=[{"name": "c1"}, {"name": "c2"}, {"name": "c3"}],
            test_passes=[True, True, True],
            statuses=["graded"],
        )
        assert report.statuses == ["graded", "graded", "graded"]

    def test_statuses_padded_when_empty(self):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.5, 1.0],
            cases=[{"name": "c1"}, {"name": "c2"}],
            test_passes=[True, True],
        )
        assert report.statuses == ["graded", "graded"]

    def test_statuses_not_modified_when_already_correct_length(self):
        report = EvaluationReport(
            overall_score=0.5,
            scores=[0.5, 1.0],
            cases=[{"name": "c1"}, {"name": "c2"}],
            test_passes=[True, True],
            statuses=["graded", "could_not_evaluate"],
        )
        assert report.statuses == ["graded", "could_not_evaluate"]

    def test_invalid_status_value_rejected(self):
        """Literal typing should reject invalid status values via Pydantic validation."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            EvaluationReport(
                overall_score=0.5,
                scores=[0.5],
                cases=[{"name": "c1"}],
                test_passes=[True],
                statuses=["invalid_status"],
            )
