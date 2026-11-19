import dataclasses
import json
import random

import pytest
from strands.decisions import DecisionMetadata, DecisionModel, DecisionResult

from strands_evals import Case, Experiment
from strands_evals.evaluators import Band, Cascade, Choice, Equals, Evaluator, Noul, Scale, TypedRubricEvaluator
from strands_evals.types import EvaluationData
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput


class StubDecisionModel(DecisionModel):
    """Returns canned distributions; answers default to the most likely option."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    async def _ask(self, state, questions, **kwargs):
        self.calls += 1
        answers = {}
        for question_id, distribution in self.probabilities.items():
            top = max(distribution, key=distribution.get)
            answers[question_id] = top == "true" if questions[question_id]["choices"] == "boolean" else top
        return DecisionResult(
            answers=answers,
            metadata=DecisionMetadata(
                latency_ms=1, provider_name="stub", model_id="stub-1", probabilities=self.probabilities
            ),
        )


class StubJudge(Evaluator):
    """A fallback with a score mapping, like the built-in LLM judges."""

    _score_mapping = {"Harmful": 0.0, "Not Harmful": 1.0}

    def __init__(self, name=None, *, rows=None, error=None):
        super().__init__(name=name)
        # Private, so the base `to_dict` writes only the name.
        self._rows = rows if rows is not None else [EvaluationOutput(score=1.0, test_pass=True, reason="judge says ok")]
        self._error = error
        self._calls_made = []

    @property
    def calls(self):
        return len(self._calls_made)

    def evaluate(self, evaluation_case):
        self._calls_made.append(evaluation_case)
        if self._error:
            raise self._error
        return list(self._rows)


def _case():
    return EvaluationData(input="What is 2+2?", actual_output="4")


def _cascade(probabilities, questions=None, band=None, judge=None):
    questions = questions or {"harmful": Noul("Is `actual_output` harmful?", expected=False)}
    rubric = TypedRubricEvaluator(questions, decision_model=StubDecisionModel(probabilities))
    return Cascade(
        fast=rubric,
        fallback=judge or StubJudge(),
        escalate_when=band or Band(question="harmful", low=0.2, high=0.8),
    )


def _noul(p_true):
    return {"harmful": {"true": p_true, "false": 1 - p_true}}


# --- Band -------------------------------------------------------------------------------------


@pytest.mark.parametrize("p, inside", [(0.19, False), (0.2, True), (0.5, True), (0.8, True), (0.81, False)])
def test_band_contains_its_ends(p, inside):
    assert Band(question="q", low=0.2, high=0.8).contains(p) is inside


def test_band_may_be_a_single_point():
    assert Band(question="q", low=0.5, high=0.5).contains(0.5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"question": "q", "low": 0.8, "high": 0.2}, "must not be greater than high"),
        ({"question": "q", "low": -0.1, "high": 0.2}, "greater than or equal to 0"),
        ({"question": "q", "low": 0.1, "high": 1.5}, "less than or equal to 1"),
        ({"question": "q", "low": float("nan"), "high": 0.2}, "finite"),
        ({"question": " ", "low": 0.1, "high": 0.2}, "must not be blank"),
        ({"question": "q", "low": "0.1", "high": 0.2}, "valid number"),
        ({"question": "q", "low": 0.1, "high": 0.2, "hihg": 0.3}, "hihg"),
    ],
)
def test_band_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Band(**kwargs)


def test_band_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Band(question="q", low=0.1, high=0.2).low = 0.3


# --- constructor ------------------------------------------------------------------------------


def test_fast_must_be_a_typed_rubric():
    with pytest.raises(TypeError, match="fast must be a TypedRubricEvaluator"):
        Cascade(fast=StubJudge(), fallback=StubJudge(), escalate_when=Band(question="q1", low=0, high=1))


def test_fallback_must_be_an_evaluator():
    rubric = TypedRubricEvaluator([Noul("q")], decision_model=None)
    with pytest.raises(TypeError, match="fallback must be an Evaluator"):
        Cascade(fast=rubric, fallback="judge", escalate_when=Band(question="q1", low=0, high=1))


def test_band_question_must_be_in_the_rubric():
    rubric = TypedRubricEvaluator({"harmful": Noul("q")}, decision_model=None)
    with pytest.raises(ValueError, match=r"'q1' is not in the fast rubric; its question ids are \['harmful'\]"):
        Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="q1", low=0, high=1))


def test_band_may_be_given_as_a_dict():
    rubric = TypedRubricEvaluator([Noul("q")], decision_model=None)
    cascade = Cascade(fast=rubric, fallback=StubJudge(), escalate_when={"question": "q1", "low": 0.1, "high": 0.2})
    assert cascade.escalate_when == Band(question="q1", low=0.1, high=0.2)


def test_invalid_band_dict_is_rejected():
    rubric = TypedRubricEvaluator([Noul("q")], decision_model=None)
    with pytest.raises(ValueError, match="must not be greater than high"):
        Cascade(fast=rubric, fallback=StubJudge(), escalate_when={"question": "q1", "low": 0.3, "high": 0.2})


def test_arguments_are_keyword_only():
    rubric = TypedRubricEvaluator({"harmful": Noul("q")}, decision_model=None)
    with pytest.raises(TypeError):
        Cascade(rubric, StubJudge(), Band(question="harmful", low=0.2, high=0.8))


def test_fallback_may_not_be_a_cascade():
    inner = _cascade(_noul(0.5))
    rubric = TypedRubricEvaluator({"harmful": Noul("q")}, decision_model=None)
    with pytest.raises(TypeError, match="nesting cascades is not supported"):
        Cascade(fast=rubric, fallback=inner, escalate_when=Band(question="harmful", low=0, high=1))


def test_fallback_may_be_another_typed_rubric():
    rubric = TypedRubricEvaluator({"harmful": Noul("q")}, decision_model=None)
    other = TypedRubricEvaluator({"other": Noul("q")}, decision_model=None)
    assert Cascade(fast=rubric, fallback=other, escalate_when=Band(question="harmful", low=0, high=1)).fallback is other


@pytest.mark.parametrize(
    "question",
    [
        Scale("q", levels=["poor", "ok", "good"]),
        Choice("q", options={"a": 1.0, "b": 0.5, "c": 0.0}),
    ],
)
def test_band_below_the_lowest_possible_p_gate_is_rejected(question, caplog):
    rubric = TypedRubricEvaluator({"gate": question}, decision_model=None)
    with pytest.raises(ValueError, match=r"below 0\.3333, the lowest possible p_gate .* 3 options"):
        Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=0.3))
    # A band that reaches 1/3 can contain a p_gate, so it is accepted without a warning.
    caplog.clear()
    Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=1 / 3))
    assert "only cases where the question does not apply" not in caplog.text


# A Choice with 3 credited options and 1 "does not apply" option. A case it applies to has more
# than half its probability on the credited options, so its p_gate is above 0.5 / 3.
_WITH_NA = Choice("q", options={"a": 1.0, "b": 0.5, "c": 0.0, "n/a": None})


@pytest.mark.parametrize("high, warns", [(0.0, True), (0.16, True), (0.5 / 3, True), (0.17, False), (1.0, False)])
def test_band_no_applicable_case_can_reach_only_warns_for_a_choice_with_a_none_credit(high, warns, caplog):
    rubric = TypedRubricEvaluator({"gate": _WITH_NA}, decision_model=None)
    caplog.clear()

    cascade = Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=high))

    assert cascade.escalate_when.high == high
    assert ("only cases where the question does not apply will escalate" in caplog.text) is warns
    if warns:
        assert "question_id=<gate>" in caplog.text


def test_choice_with_a_none_credit_is_not_rejected_at_the_plain_1_over_k_floor():
    # 1 / 4 options would reject high=0.2 for a Choice with no `None` credit.
    rubric = TypedRubricEvaluator({"gate": _WITH_NA}, decision_model=None)
    Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=0.2))


def test_choice_with_a_single_credited_option_warns_at_one_half(caplog):
    rubric = TypedRubricEvaluator({"gate": Choice("q", options={"a": 1.0, "n/a": None})}, decision_model=None)
    caplog.clear()
    Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=0.5))
    assert "only cases where the question does not apply will escalate" in caplog.text


def test_any_band_is_reachable_for_a_noul():
    rubric = TypedRubricEvaluator({"gate": Noul("q")}, decision_model=None)
    Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=0.01))


def test_evaluation_level_follows_the_fallback():
    from strands_evals.evaluators import HarmfulnessEvaluator

    rubric = TypedRubricEvaluator([Noul("q")], decision_model=None)
    cascade = Cascade(fast=rubric, fallback=HarmfulnessEvaluator(), escalate_when=Band(question="q1", low=0, high=1))
    assert cascade.evaluation_level == HarmfulnessEvaluator.evaluation_level


def test_parts_are_read_only():
    cascade = _cascade(_noul(0.1))
    with pytest.raises(AttributeError):
        cascade.fast = cascade.fast
    with pytest.raises(AttributeError):
        cascade.escalate_when = Band(question="harmful", low=0, high=1)


# --- routing ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confident_case_stays_on_the_fast_path():
    judge = StubJudge()
    cascade = _cascade(_noul(0.05), judge=judge)

    rows = await cascade.evaluate_async(_case())

    assert judge.calls == 0
    assert len(rows) == 1
    assert rows[0].score == pytest.approx(0.95)
    assert rows[0].metadata["path"] == "fast"
    assert rows[0].metadata["p_gate"] == pytest.approx(0.05)
    assert rows[0].metadata["band"] == {"question": "harmful", "low": 0.2, "high": 0.8}
    assert rows[0].metadata["evaluated_by"] == "TypedRubricEvaluator"
    assert "fast_rows" not in rows[0].metadata


@pytest.mark.asyncio
async def test_uncertain_case_escalates_and_keeps_the_fallback_reasoning():
    judge = StubJudge(rows=[EvaluationOutput(score=0.0, test_pass=False, reason="repeats the insult")])
    cascade = _cascade(_noul(0.5), judge=judge)

    rows = await cascade.evaluate_async(_case())

    assert judge.calls == 1
    assert [(row.score, row.test_pass, row.reason) for row in rows] == [(0.0, False, "repeats the insult")]
    metadata = rows[0].metadata
    assert metadata["path"] == "fallback"
    assert metadata["p_gate"] == pytest.approx(0.5)
    assert [row["metadata"]["question_id"] for row in metadata["fast_rows"]] == ["harmful"]
    json.dumps(metadata)  # the audit trail stays writable by `EvaluationReport.to_file`


@pytest.mark.asyncio
@pytest.mark.parametrize("p_true, escalated", [(0.2, True), (0.8, True), (0.19, False), (0.81, False)])
async def test_band_ends_escalate(p_true, escalated):
    judge = StubJudge()
    await _cascade(_noul(p_true), judge=judge).evaluate_async(_case())
    assert judge.calls == int(escalated)


@pytest.mark.asyncio
async def test_noul_gates_on_p_true_whatever_the_expected_answer():
    # P(true)=0.1 is outside the band even though P(expected=False)=0.9.
    judge = StubJudge()
    rows = await _cascade(_noul(0.1), band=Band(question="harmful", low=0.0, high=0.15), judge=judge).evaluate_async(
        _case()
    )
    assert judge.calls == 1
    assert rows[0].metadata["p_gate"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_scale_and_choice_gate_on_the_most_likely_option():
    questions = {
        "complete": Scale("How complete is `actual_output`?", levels=["poor", "ok", "good"]),
        "tone": Choice("Tone of `actual_output`?", options={"kind": 1.0, "rude": 0.0}),
    }
    probabilities = {
        "complete": {"poor": 0.1, "ok": 0.3, "good": 0.6},
        "tone": {"kind": 0.45, "rude": 0.55},
    }
    for question, p_gate in [("complete", 0.6), ("tone", 0.55)]:
        cascade = _cascade(probabilities, questions=questions, band=Band(question=question, low=0.9, high=1.0))
        rows = await cascade.evaluate_async(_case())
        assert [row.metadata["path"] for row in rows] == ["fast", "fast"]
        assert [row.metadata["p_gate"] for row in rows] == pytest.approx([p_gate, p_gate])


_RISK = Choice(
    "How does `actual_output` handle the risky part of `input`?",
    options={"refuses": 1.0, "hedges": 0.5, "complies": 0.0, "nothing_risky": None},
)


def _risk(refuses, hedges, complies, nothing_risky):
    return {"risk": {"refuses": refuses, "hedges": hedges, "complies": complies, "nothing_risky": nothing_risky}}


@pytest.mark.asyncio
async def test_choice_p_gate_ignores_the_does_not_apply_option():
    # "does not apply" (0.40) is the largest option, but the row reports refuses (0.35).
    probabilities = _risk(0.35, 0.15, 0.10, 0.40)
    questions = {"risk": _RISK}

    inside = _cascade(probabilities, questions, Band(question="risk", low=0.30, high=0.38))
    rows = await inside.evaluate_async(_case())
    assert rows[0].metadata["path"] == "fallback"
    assert rows[0].metadata["p_gate"] == pytest.approx(0.35)
    assert rows[0].metadata["fast_rows"][0]["label"] == "refuses"
    assert "refuses (p=0.35)" in rows[0].metadata["fast_rows"][0]["reason"]

    # A band around 0.40 used to catch this case through the N/A option; it no longer does.
    around_na = _cascade(probabilities, questions, Band(question="risk", low=0.39, high=0.41))
    rows = await around_na.evaluate_async(_case())
    assert rows[0].metadata["path"] == "fast"
    assert rows[0].metadata["p_gate"] == pytest.approx(0.35)
    assert rows[0].label == "refuses"
    assert "refuses (p=0.35)" in rows[0].reason


@pytest.mark.asyncio
async def test_p_gate_is_the_probability_of_the_label_the_row_reports():
    # Over random distributions, `p_gate` always equals the `p=` of the reported label.
    rng = random.Random(7)
    questions = {"risk": _RISK, "tone": Scale("Tone?", levels=["cold", "neutral", "warm"])}
    checked = 0
    for _ in range(200):
        risk = [rng.random() for _ in range(4)]
        tone = [rng.random() for _ in range(3)]
        if sum(risk[:3]) == 0 or risk[3] / sum(risk) >= 0.5:
            continue  # the row is not applicable, which a separate test covers
        risk_labels = ["refuses", "hedges", "complies", "nothing_risky"]
        probabilities = {
            "risk": {k: v / sum(risk) for k, v in zip(risk_labels, risk, strict=True)},
            "tone": {k: v / sum(tone) for k, v in zip(["cold", "neutral", "warm"], tone, strict=True)},
        }
        for gate in ("risk", "tone"):
            cascade = _cascade(probabilities, questions, Band(question=gate, low=0.0, high=1.0))
            rows = await cascade.evaluate_async(_case())
            row = next(r for r in rows[0].metadata["fast_rows"] if r["metadata"]["question_id"] == gate)
            assert rows[0].metadata["p_gate"] == probabilities[gate][row["label"]]
            assert f"(p={probabilities[gate][row['label']]:.2f})" in row["reason"]
        checked += 1
    assert checked > 100  # the loop is not vacuous


@pytest.mark.asyncio
async def test_tied_credited_options_gate_on_the_first_declared():
    probabilities = _risk(0.2, 0.2, 0.2, 0.4)
    cascade = _cascade(probabilities, {"risk": _RISK}, Band(question="risk", low=0.0, high=1.0))

    rows = await cascade.evaluate_async(_case())

    fast_row = rows[0].metadata["fast_rows"][0]
    assert fast_row["label"] == "refuses"
    assert rows[0].metadata["p_gate"] == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_only_does_not_apply_cases_escalate_with_a_band_no_applicable_case_reaches():
    judge = StubJudge()
    questions = {"risk": _RISK}
    band = Band(question="risk", low=0.0, high=0.05)
    paths = {}
    for name, probabilities in {
        "unsure": _risk(0.30, 0.25, 0.20, 0.25),
        "confident": _risk(0.90, 0.04, 0.03, 0.03),
        "not_applicable": _risk(0.02, 0.01, 0.02, 0.95),
        "half_not_applicable": _risk(0.20, 0.15, 0.15, 0.50),
    }.items():
        rows = await _cascade(probabilities, questions, band, judge=judge).evaluate_async(_case())
        paths[name] = rows[0].metadata["path"]

    assert paths == {
        "unsure": "fast",
        "confident": "fast",
        "not_applicable": "fallback",
        "half_not_applicable": "fallback",
    }
    assert judge.calls == 2


@pytest.mark.asyncio
async def test_not_applicable_row_records_the_highest_credited_probability():
    cascade = _cascade(_risk(0.03, 0.01, 0.01, 0.95), {"risk": _RISK}, Band(question="risk", low=0.0, high=0.05))
    rows = await cascade.evaluate_async(_case())
    assert rows[0].metadata["path"] == "fallback"
    assert rows[0].metadata["p_gate"] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_inapplicable_gating_question_escalates():
    questions = {"safety": Choice("Risky part of `input`?", options={"refuses": 1.0, "nothing_risky": None})}
    probabilities = {"safety": {"refuses": 0.05, "nothing_risky": 0.95}}
    judge = StubJudge()
    cascade = _cascade(probabilities, questions=questions, band=Band(question="safety", low=0.4, high=0.6), judge=judge)

    rows = await cascade.evaluate_async(_case())

    assert judge.calls == 1
    assert rows[0].metadata["path"] == "fallback"
    assert rows[0].metadata["fast_rows"][0]["label"] == NOT_APPLICABLE


@pytest.mark.asyncio
async def test_fast_rows_carry_no_escalation_reason():
    rows = await _cascade(_noul(0.05)).evaluate_async(_case())
    assert [row.metadata["path"] for row in rows] == ["fast"]
    assert "escalation_reason" not in rows[0].metadata


@pytest.mark.asyncio
async def test_band_escalation_records_the_band_reason():
    rows = await _cascade(_noul(0.5)).evaluate_async(_case())
    assert rows[0].metadata["escalation_reason"] == "band"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "band_high, p_gate_in_band",
    [(0.05, True), (0.5, True), (0.02, False)],
    ids=["band-also-matches", "wide-band", "band-misses"],
)
async def test_not_applicable_reason_wins_whether_or_not_p_gate_is_in_the_band(band_high, p_gate_in_band):
    # The credited options hold 0.04 (top: 0.03), and "does not apply" holds 0.96.
    band = Band(question="risk", low=0.0, high=band_high)
    cascade = _cascade(_risk(0.03, 0.005, 0.005, 0.96), {"risk": _RISK}, band)

    rows = await cascade.evaluate_async(_case())

    assert band.contains(rows[0].metadata["p_gate"]) is p_gate_in_band
    assert rows[0].metadata["path"] == "fallback"
    assert rows[0].metadata["escalation_reason"] == "not_applicable"


@pytest.mark.asyncio
async def test_reason_tells_a_band_escalation_from_a_not_applicable_one():
    band = Band(question="risk", low=0.30, high=0.40)
    cases = {
        "in_band": (_risk(0.35, 0.25, 0.20, 0.20), "band"),
        "not_applicable": (_risk(0.02, 0.01, 0.02, 0.95), "not_applicable"),
        "confident": (_risk(0.80, 0.10, 0.05, 0.05), None),
    }
    for name, (probabilities, reason) in cases.items():
        rows = await _cascade(probabilities, {"risk": _RISK}, band).evaluate_async(_case())
        assert rows[0].metadata.get("escalation_reason") == reason, name
        assert rows[0].metadata["path"] == ("fast" if reason is None else "fallback"), name


@pytest.mark.asyncio
async def test_every_fallback_row_gets_the_reason_and_the_metadata_stays_json():
    judge = StubJudge(
        rows=[
            EvaluationOutput(score=1.0, test_pass=True, reason="a"),
            EvaluationOutput(score=0.0, test_pass=False, reason="b", metadata={"judge": "x"}),
        ]
    )
    rows = await _cascade(_noul(0.5), judge=judge).evaluate_async(_case())
    assert [row.metadata["escalation_reason"] for row in rows] == ["band", "band"]
    assert rows[1].metadata["judge"] == "x"
    json.dumps([row.metadata for row in rows])


@pytest.mark.asyncio
async def test_a_fallback_row_survives_a_report_round_trip(tmp_path):
    model = StubDecisionModel(_risk(0.02, 0.01, 0.02, 0.95))
    rubric = TypedRubricEvaluator({"risk": _RISK}, decision_model=model)
    cascade = Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="risk", low=0, high=0.05))

    report = await Experiment(cases=[Case(name="c", input="hi")], evaluators=[cascade]).run_evaluations_async(
        lambda case: "ok"
    )
    row = report[0].detailed_results[0][0] if isinstance(report, list) else report.detailed_results[0][0]

    assert row.metadata["escalation_reason"] == "not_applicable"
    assert EvaluationOutput.model_validate_json(row.model_dump_json()).metadata == row.metadata


@pytest.mark.asyncio
async def test_fallback_error_fails_the_case():
    cascade = _cascade(_noul(0.5), judge=StubJudge(error=RuntimeError("judge down")))
    with pytest.raises(RuntimeError, match="judge down"):
        await cascade.evaluate_async(_case())


@pytest.mark.asyncio
async def test_fallback_rows_are_not_mutated():
    original = EvaluationOutput(score=1.0, test_pass=True, reason="ok", metadata={"judge": "x"})
    await _cascade(_noul(0.5), judge=StubJudge(rows=[original])).evaluate_async(_case())
    assert original.metadata == {"judge": "x"}


def test_sync_evaluate_matches_async():
    rows = _cascade(_noul(0.05)).evaluate(_case())
    assert rows[0].metadata["path"] == "fast"


# --- aggregation and labels -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_path_uses_the_rubric_aggregator_and_label():
    questions = {
        "harmful": Noul("Is `actual_output` harmful?", expected=False, weight=3.0),
        "answers": Noul("Does `actual_output` answer `input`?"),
    }
    probabilities = {"harmful": {"true": 0.0, "false": 1.0}, "answers": {"true": 0.6, "false": 0.4}}
    cascade = _cascade(probabilities, questions=questions)
    rows = await cascade.evaluate_async(_case())

    score, passed, _ = cascade.aggregator(rows)
    assert score == pytest.approx((3 * 1.0 + 0.6) / 4)
    assert passed
    assert cascade._get_label(score, passed, rows) == "YES"


@pytest.mark.asyncio
async def test_fallback_path_uses_the_fallback_aggregator_and_label():
    judge = StubJudge(rows=[EvaluationOutput(score=0.0, test_pass=False, reason="harmful")])
    cascade = _cascade(_noul(0.5), judge=judge)
    rows = await cascade.evaluate_async(_case())

    assert cascade.aggregator(rows) == (0.0, False, "harmful")
    assert cascade._get_label(0.0, False, rows) == "Harmful"


def test_empty_fallback_rows_use_the_fallback_aggregator():
    assert _cascade(_noul(0.5)).aggregator([]) == (0.0, False, "No evaluation outputs produced")


# --- experiment and serialization -------------------------------------------------------------


def test_runs_in_an_experiment():
    judge = StubJudge(rows=[EvaluationOutput(score=0.0, test_pass=False, reason="judge")])
    model = StubDecisionModel(_noul(0.5))
    rubric = TypedRubricEvaluator({"harmful": Noul("Is `actual_output` harmful?")}, decision_model=model)
    cascade = Cascade(fast=rubric, fallback=judge, escalate_when=Band(question="harmful", low=0.4, high=0.6))

    report = Experiment(cases=[Case(name="c", input="hi")], evaluators=[cascade]).run_evaluations(lambda case: "ok")

    assert report.scores == [0.0]
    assert report.test_passes == [False]
    assert report.detailed_results[0][0].metadata["path"] == "fallback"
    assert model.calls == 1 and judge.calls == 1


def test_to_dict_round_trips_through_experiment():
    cascade = Cascade(
        fast=TypedRubricEvaluator({"harmful": Noul("q", expected=False)}, decision_model=None, pass_threshold=0.7),
        fallback=StubJudge(name="judge"),
        escalate_when=Band(question="harmful", low=0.1, high=0.3),
        name="harm",
    )
    data = cascade.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["escalate_when"] == {"question": "harmful", "low": 0.1, "high": 0.3}
    assert data["fast"]["decision_model"] is None

    experiment = Experiment(cases=[Case(input="hi")], evaluators=[cascade])
    loaded = Experiment.from_dict(experiment.to_dict(), custom_evaluators=[StubJudge]).evaluators[0]

    assert isinstance(loaded, Cascade)
    assert loaded.name == "harm"
    assert loaded.escalate_when == cascade.escalate_when
    assert loaded.fast.questions == cascade.fast.questions
    assert loaded.fast.to_dict() == cascade.fast.to_dict()
    assert isinstance(loaded.fallback, StubJudge) and loaded.fallback.name == "judge"
    assert loaded.fast.decision_model is None


def test_from_dict_rebuilds_only_declared_evaluator_args():
    # Values shaped like a serialized evaluator stay plain data everywhere else.
    rubric = TypedRubricEvaluator({"evaluator_type": Noul("q")}, decision_model=None)
    equals = Equals(value={"evaluator_type": "Equals", "value": 1})
    experiment = Experiment(cases=[Case(input="hi")], evaluators=[rubric, equals])

    loaded = Experiment.from_dict(experiment.to_dict()).evaluators

    assert list(loaded[0].questions) == ["evaluator_type"]
    assert loaded[1].value == {"evaluator_type": "Equals", "value": 1}


def test_band_that_only_escalates_not_applicable_cases_survives_a_round_trip(caplog):
    rubric = TypedRubricEvaluator({"risk": _RISK}, decision_model=None)
    cascade = Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="risk", low=0, high=0.05))
    experiment = Experiment(cases=[Case(input="hi")], evaluators=[cascade])

    caplog.clear()
    loaded = Experiment.from_dict(experiment.to_dict(), custom_evaluators=[StubJudge]).evaluators[0]

    assert loaded.escalate_when == cascade.escalate_when
    assert "only cases where the question does not apply will escalate" in caplog.text


def test_to_dict_omits_name_when_unset():
    assert "name" not in _cascade(_noul(0.1)).to_dict()
