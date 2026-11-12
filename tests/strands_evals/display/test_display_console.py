from strands_evals.display.display_console import _pass_rate_text
from strands_evals.types.evaluation import COULD_NOT_EVALUATE, NOT_APPLICABLE, EvaluationOutput


def _case(test_pass: bool, rows: list[EvaluationOutput]) -> dict:
    return {"details": {"test_pass": test_pass}, "detailed_results": rows}


def _unjudged() -> EvaluationOutput:
    return EvaluationOutput(
        score=0.0, test_pass=False, label=NOT_APPLICABLE, metadata={COULD_NOT_EVALUATE: "judge_context_window_exceeded"}
    )


def test_pass_rate_counts_every_case():
    items = {0: _case(True, [EvaluationOutput(score=1.0, test_pass=True)]), 1: _case(False, [])}
    assert _pass_rate_text(items) == "Pass Rate: 0.5"


def test_unjudged_case_counts_as_not_passed_and_is_named():
    """Matches the CLI summary ("1/2 passed, 1 could not evaluate") and `--fail-on=any`."""
    items = {0: _case(True, [EvaluationOutput(score=1.0, test_pass=True)]), 1: _case(False, [_unjudged()])}
    assert _pass_rate_text(items) == "Pass Rate: 0.5 (1 could not evaluate)"


def test_case_with_nothing_to_judge_still_counts_as_passed():
    declined = EvaluationOutput(score=0.0, test_pass=True, reason="no skill invoked", label=NOT_APPLICABLE)
    assert _pass_rate_text({0: _case(True, [declined])}) == "Pass Rate: 1.0"


def test_empty_report_has_zero_pass_rate():
    assert _pass_rate_text({}) == "Pass Rate: 0.0"
