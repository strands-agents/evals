from strands_evals.classification import RiskLabel, classify_risk, classify_task_risk
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput


def _passed() -> EvaluationOutput:
    return EvaluationOutput(score=1.0, test_pass=True)


def _failed() -> EvaluationOutput:
    return EvaluationOutput(score=0.0, test_pass=False)


def _not_applicable() -> EvaluationOutput:
    return EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)


class TestClassifyRisk:
    """Turn repeated `(case, evaluator)` runs into a single stability label."""

    def test_all_passed_is_pass(self):
        assert classify_risk([_passed(), _passed(), _passed()]) is RiskLabel.PASS

    def test_all_failed_is_bug(self):
        assert classify_risk([_failed(), _failed(), _failed()]) is RiskLabel.BUG

    def test_mixed_pass_and_fail_is_flaky(self):
        assert classify_risk([_passed(), _failed(), _passed()]) is RiskLabel.FLAKY

    def test_empty_is_cne(self):
        assert classify_risk([]) is RiskLabel.CNE

    def test_all_not_applicable_is_cne(self):
        assert classify_risk([_not_applicable(), _not_applicable()]) is RiskLabel.CNE

    def test_single_pass_is_pass(self):
        assert classify_risk([_passed()]) is RiskLabel.PASS

    def test_single_fail_is_bug(self):
        assert classify_risk([_failed()]) is RiskLabel.BUG

    def test_not_applicable_rows_are_dropped_before_classification(self):
        """A run that had nothing to judge must not tip the verdict toward pass."""
        results = [_not_applicable(), _failed(), _not_applicable()]
        assert classify_risk(results) is RiskLabel.BUG

    def test_not_applicable_does_not_create_flakiness(self):
        """Gradable rows all agree; the not-applicable row alone must not read as a mix."""
        results = [_passed(), _passed(), _not_applicable()]
        assert classify_risk(results) is RiskLabel.PASS

    def test_gradable_uses_property_not_score(self):
        """Rows are graded by the `not_applicable` property, regardless of their score value."""
        results = [
            EvaluationOutput(score=0.0, test_pass=True, label="Yes"),
            EvaluationOutput(score=0.0, test_pass=False),
        ]
        assert classify_risk(results) is RiskLabel.FLAKY


class TestClassifyTaskRisk:
    """Roll several evaluator verdicts on one case up to a worst-of verdict."""

    def test_bug_dominates_flaky(self):
        assert classify_task_risk([RiskLabel.BUG, RiskLabel.FLAKY, RiskLabel.FLAKY]) is RiskLabel.BUG

    def test_flaky_dominates_pass(self):
        assert classify_task_risk([RiskLabel.PASS, RiskLabel.FLAKY, RiskLabel.PASS]) is RiskLabel.FLAKY

    def test_pass_dominates_cne(self):
        assert classify_task_risk([RiskLabel.CNE, RiskLabel.PASS, RiskLabel.CNE]) is RiskLabel.PASS

    def test_all_pass_is_pass(self):
        assert classify_task_risk([RiskLabel.PASS, RiskLabel.PASS]) is RiskLabel.PASS

    def test_all_cne_is_cne(self):
        assert classify_task_risk([RiskLabel.CNE, RiskLabel.CNE]) is RiskLabel.CNE

    def test_empty_is_cne(self):
        assert classify_task_risk([]) is RiskLabel.CNE

    def test_single_label_is_returned(self):
        assert classify_task_risk([RiskLabel.FLAKY]) is RiskLabel.FLAKY


class TestRiskLabel:
    """The label is a string enum so it serializes and compares like its value."""

    def test_values(self):
        assert RiskLabel.BUG.value == "bug"
        assert RiskLabel.FLAKY.value == "flaky"
        assert RiskLabel.CNE.value == "cne"
        assert RiskLabel.PASS.value == "pass"

    def test_is_str_enum(self):
        assert RiskLabel.BUG == "bug"


class TestEndToEnd:
    """The two utilities compose: per-evaluator verdicts feed the case rollup."""

    def test_bug_and_flaky_evaluators_make_a_bug_case(self):
        correctness = classify_risk([_failed(), _failed(), _failed()])
        faithfulness = classify_risk([_passed(), _failed(), _passed()])
        assert correctness is RiskLabel.BUG
        assert faithfulness is RiskLabel.FLAKY
        assert classify_task_risk([correctness, faithfulness]) is RiskLabel.BUG
