import dataclasses
import json

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
        Choice("q", options={"a": 1.0, "b": 0.5, "n/a": None}),
    ],
)
def test_band_below_the_lowest_possible_p_gate_is_rejected(question):
    rubric = TypedRubricEvaluator({"gate": question}, decision_model=None)
    with pytest.raises(ValueError, match=r"below 0\.3333, the lowest possible p_gate .* 3 options"):
        Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=0.3))
    # A band that reaches 1/3 can contain a p_gate, so it is accepted.
    Cascade(fast=rubric, fallback=StubJudge(), escalate_when=Band(question="gate", low=0, high=1 / 3))


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


def test_to_dict_omits_name_when_unset():
    assert "name" not in _cascade(_noul(0.1)).to_dict()
