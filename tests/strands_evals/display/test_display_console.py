from strands_evals.display.display_console import _case_is_applicable
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput


def _case(test_pass, detailed_results):
    return {"details": {"test_pass": test_pass}, "detailed_results": detailed_results}


def test_case_with_no_detailed_rows_is_applicable():
    # An empty detailed_results is a genuine (failed) evaluation, so it stays in the rate.
    assert _case_is_applicable(_case(False, [])) is True


def test_case_all_rows_not_applicable_and_passing_is_dropped():
    rows = [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)]
    assert _case_is_applicable(_case(True, rows)) is False


def test_not_applicable_row_that_failed_is_kept():
    # not_applicable but test_pass=False means a real failure (e.g. missing trajectory).
    rows = [EvaluationOutput(score=0.0, test_pass=False, label=NOT_APPLICABLE)]
    assert _case_is_applicable(_case(False, rows)) is True


def test_case_with_a_judged_row_is_applicable():
    rows = [EvaluationOutput(score=1.0, test_pass=True, reason="judged")]
    assert _case_is_applicable(_case(True, rows)) is True
