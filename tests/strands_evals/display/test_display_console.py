from io import StringIO
from typing import Any

import pytest
from rich.cells import cell_len
from rich.console import Console
from strands.decisions import DecisionMetadata, DecisionModel, DecisionResult, Question

from strands_evals.display import display_console
from strands_evals.display.display_console import CollapsibleTableReportDisplay, _pass_rate_text
from strands_evals.evaluators import Band, Cascade, Choice, Equals, Noul, Scale, TypedRubricEvaluator
from strands_evals.types.evaluation import COULD_NOT_EVALUATE, NOT_APPLICABLE, EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport


def _case(test_pass: bool, rows: list[EvaluationOutput]) -> dict:
    return {"details": {"test_pass": test_pass}, "detailed_results": rows}


def _unjudged() -> EvaluationOutput:
    return EvaluationOutput(
        score=0.0, test_pass=False, label=NOT_APPLICABLE, metadata={COULD_NOT_EVALUATE: "judge_context_window_exceeded"}
    )


def test_pass_rate_counts_every_case():
    items = {0: _case(True, [EvaluationOutput(score=1.0, test_pass=True)]), 1: _case(False, [])}
    assert _pass_rate_text(items) == "Pass Rate: 0.50"


def test_unjudged_case_counts_as_not_passed_and_is_named():
    """Matches the CLI summary ("1/2 passed, 1 could not evaluate") and `--fail-on=any`."""
    items = {0: _case(True, [EvaluationOutput(score=1.0, test_pass=True)]), 1: _case(False, [_unjudged()])}
    assert _pass_rate_text(items) == "Pass Rate: 0.50 (1 could not evaluate)"


def test_case_with_nothing_to_judge_still_counts_as_passed():
    declined = EvaluationOutput(score=0.0, test_pass=True, reason="no skill invoked", label=NOT_APPLICABLE)
    assert _pass_rate_text({0: _case(True, [declined])}) == "Pass Rate: 1.00"


def test_empty_report_has_zero_pass_rate():
    assert _pass_rate_text({}) == "Pass Rate: 0.00"


def test_pass_rate_is_rounded_like_the_overall_score():
    items = {i: _case(i < 2, [EvaluationOutput(score=1.0, test_pass=i < 2)]) for i in range(3)}
    assert _pass_rate_text(items) == "Pass Rate: 0.67"


class StubDecisionModel(DecisionModel):
    """Return distributions through the public DecisionModel contract."""

    def __init__(self, model_id: str | None = "checkpoint-1", provider_name: str | None = "stub") -> None:
        self.model_id = model_id
        self.provider_name = provider_name

    async def _ask(self, state: Any, questions: dict[str, Question], **kwargs: Any) -> DecisionResult:
        distributions = {
            "safety": {"true": 0.2, "false": 0.8},
            "quality": {"poor": 0.1, "ok": 0.2, "good": 0.7},
            "risk": {"refuse": 0.1, "skip": 0.9},
        }
        # A backend's committed answer may differ from the most likely option.
        answers = {"safety": True, "quality": "good", "risk": "skip"}
        return DecisionResult(
            answers={key: answers[key] for key in questions},
            metadata=DecisionMetadata(
                latency_ms=1,
                provider_name=self.provider_name,
                model_id=self.model_id,
                probabilities={key: distributions[key] for key in questions},
                confidence={key: 0.6 for key in questions},
            ),
        )


@pytest.fixture
def rubric() -> TypedRubricEvaluator:
    return TypedRubricEvaluator(
        {
            "safety": Noul("Is the response harmful?", expected=False),
            "quality": Scale("How good is the response?", levels=["poor", "ok", "good"]),
            "risk": Choice("How is risk handled?", options={"refuse": 1.0, "skip": None}),
        },
        decision_model=StubDecisionModel(),
    )


@pytest.fixture
def captured_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    console = Console(file=StringIO(), record=True, width=180, color_system=None)
    monkeypatch.setattr(display_console, "console", console)
    return console


def _report(rows: list[EvaluationOutput]) -> EvaluationReport:
    return EvaluationReport(
        overall_score=0.8,
        scores=[0.8],
        test_passes=[True],
        cases=[{"name": "example", "evaluator": "rubric", "input": "hi"}],
        reasons=["case reason"],
        detailed_results=[rows],
    )


def _rubric_outputs(rubric: TypedRubricEvaluator) -> list[EvaluationOutput]:
    return rubric.evaluate(EvaluationData(input="hi", actual_output="answer"))


@pytest.mark.parametrize("single_question", [True, False], ids=["single", "multiple"])
def test_static_report_shows_real_rubric_results(
    rubric: TypedRubricEvaluator, captured_console: Console, single_question: bool
) -> None:
    rows = _rubric_outputs(rubric)
    report = _report(rows[:1] if single_question else rows)
    before = report.model_dump_json()

    report.display(include_input=False)

    text = captured_console.export_text()
    assert "decision_model" in text and "stub / checkpoint-1" in text
    assert "safety: Score=0.80" in text
    assert "Question: Is the response harmful?" in text
    assert "Label: false" in text
    assert "Raw answer: True" in text
    assert "Selected option: false" in text
    assert "Probabilities: true=0.2, false=0.8" in text
    assert "Confidence: 0.6" in text
    if not single_question:
        assert "quality: Score=0.80" in text
        assert "Raw answer: good" in text
        assert "Probabilities: poor=0.1, ok=0.2, good=0.7" in text
        assert "risk: Score=n/a" in text
        assert f"Label: {NOT_APPLICABLE}" in text
        assert "Raw answer: skip" in text
        assert "Selected option: n/a" in text
        assert "Probabilities: refuse=0.1, skip=0.9" in text
    assert report.model_dump_json() == before


def test_interactive_report_shows_question_details_only_after_expansion(
    rubric: TypedRubricEvaluator, captured_console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(_rubric_outputs(rubric)[:1])

    monkeypatch.setattr(display_console.Prompt, "ask", lambda *args, **kwargs: "q")
    report.run_display(include_input=False)
    collapsed = captured_console.export_text(clear=True)
    assert "stub / checkpoint-1" in collapsed
    assert "Raw answer:" not in collapsed

    choices = iter(["0", "q"])
    monkeypatch.setattr(display_console.Prompt, "ask", lambda *args, **kwargs: next(choices))
    report.run_display(include_input=False)
    assert "Raw answer: True" in captured_console.export_text()


@pytest.mark.parametrize("fallback", [False, True], ids=["fast", "fallback"])
def test_cascade_shows_routing_and_the_fast_model(
    rubric: TypedRubricEvaluator, captured_console: Console, fallback: bool
) -> None:
    cascade = Cascade(
        fast=rubric,
        fallback=Equals(value="answer"),
        escalate_when=Band(question="safety", low=0.1 if fallback else 0.4, high=0.3 if fallback else 0.6),
    )
    rows = cascade.evaluate(EvaluationData(input="hi", actual_output="answer"))
    _report(rows).display(include_input=False)

    text = captured_console.export_text()
    assert "path" in text and "p_gate" in text and "0.2" in text
    assert ("fallback" if fallback else "fast") in text
    assert "stub / checkpoint-1" in text
    assert text.count("Raw answer: True") == 1
    if fallback:
        assert "fast: stub / checkpoint-1" in text
        assert "Fast Rubric Audit for Case 0" in text
        assert "Metric 1: Score=1.00" in text
    else:
        assert "fast: stub / checkpoint-1" in text
        assert "Fast Rubric Audit" not in text


def test_typed_fallback_keeps_both_models_and_deduplicates_the_fast_audit(
    rubric: TypedRubricEvaluator, captured_console: Console
) -> None:
    fallback = TypedRubricEvaluator(rubric.questions, decision_model=StubDecisionModel(model_id="checkpoint-2"))
    cascade = Cascade(fast=rubric, fallback=fallback, escalate_when=Band(question="safety", low=0.1, high=0.3))
    rows = cascade.evaluate(EvaluationData(input="hi", actual_output="answer"))
    assert len(rows) == 3  # Every verdict has a copy of the same fast audit.

    _report(rows).display(include_input=False)

    text = captured_console.export_text()
    assert "fast: stub / checkpoint-1; fallback: stub / checkpoint-2" in text
    assert text.count("Fast Rubric Audit for Case 0") == 1
    assert text.count("Raw answer: True") == 2  # One verdict and one audited fast answer.


@pytest.mark.parametrize("provider_name", ["stub", None], ids=["provider", "backend-class"])
def test_missing_model_id_is_reported_explicitly(captured_console: Console, provider_name: str | None) -> None:
    rubric = TypedRubricEvaluator(
        {"safety": Noul("Is it harmful?", expected=False)},
        decision_model=StubDecisionModel(model_id=None, provider_name=provider_name),
    )
    _report(_rubric_outputs(rubric)).display(include_input=False)
    text = captured_console.export_text()
    assert f"{provider_name or 'StubDecisionModel'} (model not reported)" in text
    assert "checkpoint-1" not in text


def test_mixed_report_uses_consistent_columns_without_mutating_items(
    rubric: TypedRubricEvaluator, captured_console: Console
) -> None:
    items = {
        "plain": {
            "details": {"name": "ordinary", "score": "1.00", "test_pass": True},
            "detailed_results": [EvaluationOutput(score=1.0, test_pass=True)],
            "expanded": False,
        },
        "rubric": {
            "details": {"name": "typed", "score": "0.80", "test_pass": True, "recommendation": "review"},
            "detailed_results": _rubric_outputs(rubric)[:1],
            "expanded": False,
        },
    }
    display = CollapsibleTableReportDisplay(items, overall_score=0.9)
    display.display_items()
    text = captured_console.export_text()
    assert "decision_model" in text and "recommendation" in text
    assert "ordinary" in text and "typed" in text and "stub / checkpoint-1" in text
    assert "decision_model" not in items["rubric"]["details"]
    assert "Raw answer:" not in text


def test_empty_report_can_be_rendered(captured_console: Console) -> None:
    EvaluationReport(overall_score=0, scores=[], test_passes=[], cases=[]).display()
    assert "Pass Rate: 0.00" in captured_console.export_text()


def test_rubric_text_is_rendered_literally(rubric: TypedRubricEvaluator, captured_console: Console) -> None:
    row = _rubric_outputs(rubric)[0]
    row.metadata["question"] = "[bold]literal question[/bold]"
    row.metadata["answer"] = "[red]literal answer[/red]"
    _report([row]).display(include_input=False)
    text = captured_console.export_text()
    assert "[bold]literal question[/bold]" in text
    assert "[red]literal answer[/red]" in text


@pytest.mark.parametrize(
    "metadata",
    [
        {"path": "/tmp/artifact.txt"},
        {"path": "/tmp/artifact.txt", "p_gate": "not a probability"},
        {"p_gate": 0.2},
        {"path": "custom-stage", "p_gate": 0.2},
    ],
)
def test_custom_metadata_does_not_create_cascade_columns(captured_console: Console, metadata: dict[str, Any]) -> None:
    row = EvaluationOutput(score=1.0, test_pass=True, metadata=metadata)
    _report([row]).display(include_input=False)

    text = captured_console.export_text()
    assert "path" not in text
    assert "p_gate" not in text
    assert "decision_model" not in text


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("p_gate", "bad gate", "bad gate"),
        ("confidence", "bad confidence", "Confidence: bad confidence"),
        ("probabilities", {"true": None, "false": "bad"}, "Probabilities: true=None, false=bad"),
        ("probabilities", "not a map", "Probabilities: not a map"),
    ],
)
def test_malformed_optional_numbers_are_displayed_without_changing_the_report(
    rubric: TypedRubricEvaluator, captured_console: Console, field: str, value: Any, expected: str
) -> None:
    row = _rubric_outputs(rubric)[0]
    row.metadata.update({"path": "fast", "p_gate": 0.2, field: value})
    report = _report([row])
    before = report.model_dump_json()

    report.display(include_input=False)

    assert expected in captured_console.export_text()
    assert report.model_dump_json() == before


@pytest.mark.parametrize("audit", [None, "not a list", [{"score": "bad"}]])
def test_invalid_fast_audit_warns_and_preserves_the_verdict(
    rubric: TypedRubricEvaluator, captured_console: Console, caplog: pytest.LogCaptureFixture, audit: Any
) -> None:
    cascade = Cascade(
        fast=rubric, fallback=Equals(value="answer"), escalate_when=Band(question="safety", low=0.1, high=0.3)
    )
    rows = cascade.evaluate(EvaluationData(input="hi", actual_output="answer"))
    rows[0].metadata["fast_rows"] = audit
    report = _report(rows).model_copy(update={"scores": [1.0], "overall_score": 1.0})
    before = report.model_dump_json()

    report.display(include_input=False)

    text = captured_console.export_text()
    assert "1.00" in text and "fallback" in text
    assert "Fast Rubric Audit" not in text
    assert "fast rubric audit could not be displayed" in caplog.text
    assert len(caplog.records) == 1
    assert report.model_dump_json() == before


def test_generic_expanded_metrics_show_labels_and_not_applicable_scores(captured_console: Console) -> None:
    items = {
        "0": {
            "details": {"name": "ordinary", "score": "1.00", "test_pass": True},
            "expanded": True,
            "detailed_results": [
                EvaluationOutput(score=0, test_pass=True, label=NOT_APPLICABLE),
                EvaluationOutput(score=1, test_pass=True, label="passed"),
            ],
        }
    }
    CollapsibleTableReportDisplay(items, overall_score=1.0).display_items()

    text = captured_console.export_text()
    assert "Metric 1: Score=n/a" in text
    assert f"Label: {NOT_APPLICABLE}" in text
    assert "Metric 2: Score=1.00" in text
    assert "Label: passed" in text


@pytest.mark.parametrize("width", [40, 80])
def test_narrow_report_wraps_question_details_without_losing_values(
    rubric: TypedRubricEvaluator, captured_console: Console, width: int
) -> None:
    captured_console.width = width

    _report(_rubric_outputs(rubric)).display(include_input=False)

    text = captured_console.export_text()
    for value in ("checkpoint-1", "true=0.2", "false=0.8", "poor=0.1", "skip=0.9", "Raw answer: True"):
        assert value in text
    assert all(cell_len(line) <= width for line in text.splitlines())


@pytest.mark.parametrize("field", ["question", "answer", "model_id"])
def test_unmatched_markup_in_rubric_metadata_is_literal(
    rubric: TypedRubricEvaluator, captured_console: Console, field: str
) -> None:
    row = _rubric_outputs(rubric)[0]
    row.metadata[field] = "[/x]"

    _report([row]).display(include_input=False)

    assert "[/x]" in captured_console.export_text()


def test_unmatched_markup_in_generic_input_and_reasons_is_literal(
    captured_console: Console, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [EvaluationOutput(score=1, test_pass=True, reason="metric [/x]")] * 2
    report = _report(rows)
    report.cases[0]["input"] = "input [/x]"
    report.reasons[0] = "case [/x]"
    choices = iter(["0", "q"])
    monkeypatch.setattr(display_console.Prompt, "ask", lambda *args, **kwargs: next(choices))

    report.run_display()

    text = captured_console.export_text()
    for value in ("input [/x]", "case [/x]", "metric [/x]"):
        assert value in text
