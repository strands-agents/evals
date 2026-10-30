import asyncio
import copy
import json
import logging
import pickle
from datetime import datetime
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from strands.decisions import DecisionMetadata, DecisionModel, DecisionResult, Uncertain
from strands.types.exceptions import ModelThrottledException

from strands_evals import Case, Experiment
from strands_evals.evaluators import Choice, Equals, Noul, Score, TypedRubricEvaluator
from strands_evals.experiment import _get_label_from_score
from strands_evals.types import EvaluationData
from strands_evals.types.evaluation import NOT_APPLICABLE
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import AgentInvocationSpan, Session, SpanInfo, Trace


class StubDecisionModel(DecisionModel):
    """Returns canned distributions; answers default to the most likely option."""

    def __init__(
        self,
        probabilities,
        *,
        answers=None,
        confidence=None,
        provider_name="stub-provider",
        model_id="stub-1",
        latency_ms=7,
    ):
        self.probabilities = probabilities
        self.answers = answers
        self.confidence = confidence
        self.provider_name = provider_name
        self.model_id = model_id
        self.latency_ms = latency_ms
        self.calls = []

    async def _ask(self, state, questions, **kwargs):
        self.calls.append((state, copy.deepcopy(questions)))
        probabilities = self.probabilities(state) if callable(self.probabilities) else self.probabilities
        answers = self.answers
        if answers is None:
            answers = {}
            for question_id, distribution in (probabilities or {}).items():
                top = max(distribution, key=distribution.get)
                answers[question_id] = top == "true" if questions[question_id]["choices"] == "boolean" else top
        return DecisionResult(
            answers=answers,
            metadata=DecisionMetadata(
                latency_ms=self.latency_ms,
                provider_name=self.provider_name,
                model_id=self.model_id,
                probabilities=probabilities,
                confidence=self.confidence,
            ),
        )


def _case(**fields):
    return EvaluationData(**{"input": "What is 2+2?", "actual_output": "4", **fields})


def _evaluator(questions, probabilities, **kwargs):
    model = kwargs.pop("model", None) or StubDecisionModel(probabilities)
    return TypedRubricEvaluator(questions, model=model, **kwargs)


def _session(*turns):
    now = datetime.now()
    spans = [
        AgentInvocationSpan(
            span_info=SpanInfo(session_id="s", start_time=now, end_time=now),
            user_prompt=prompt,
            agent_response=response,
            available_tools=[],
        )
        for prompt, response in turns
    ]
    return Session(traces=[Trace(spans=spans, trace_id="t", session_id="s")], session_id="s")


YES = {"true": 0.9, "false": 0.1}


# --- question types ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: Noul(""), "question text"),
        (lambda: Noul("q", weight=0), "weight"),
        (lambda: Noul("q", weight=5e-324), "weight"),
        (lambda: Noul("q", weight=1e7), "weight"),
        (lambda: Noul("q", weight=float("nan")), "weight"),
        (lambda: Noul("q", weight=True), "weight"),
        (lambda: Noul("q", expected=1), "expected"),
        (lambda: Noul("q", required="yes"), "required"),
        (lambda: Noul("q", pass_threshold=1.5), "pass_threshold"),
        (lambda: Noul("q", pass_threshold=True), "pass_threshold"),
        (lambda: Score("q", levels=["only"]), "at least 2"),
        (lambda: Score("q", levels="ab"), "sequence"),
        (lambda: Score("q", levels=["a", "a"]), "distinct"),
        (lambda: Score("q", levels=["a", "uncertain"]), "reserved"),
        (lambda: Score("q", levels=["not_applicable", "poor", "good"]), "reserved"),
        (lambda: Score("q", levels=["a", " "]), "label"),
        (lambda: Choice("q", options={"a": 1.0}), "at least 2"),
        (lambda: Choice("q", options={"a": 1.5, "b": 0.0}), "credit"),
        (lambda: Choice("q", options={"a": True, "b": 0.0}), "credit"),
        (lambda: Choice("q", options={"a": None, "b": None}), "numeric credit"),
        (lambda: Choice("q", options={"uncertain": 1.0, "b": 0.0}), "reserved"),
        (lambda: Choice("q", options={"yes": 1.0, "partially": 0.5, "not_applicable": 0.0}), "reserved"),
        (lambda: Choice("q", options=[("a", 1.0), ("b", 0.0)]), "mapping"),
    ],
)
def test_question_validation(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_question_collections_are_immutable_copies():
    levels = ["poor", "good"]
    options = {"yes": 1.0, "no": 0.0}
    score = Score("q", levels=levels)
    choice = Choice("q", options=options)
    levels.append("great")
    options["maybe"] = 0.5

    assert score.levels == ("poor", "good")
    assert dict(choice.options) == {"yes": 1.0, "no": 0.0}
    with pytest.raises(TypeError):
        choice.options["maybe"] = 0.5


@pytest.mark.parametrize(
    "question",
    [
        Noul("q", weight=2.0, expected=False, required=True, pass_threshold=0.8),
        Score("q", levels=["poor", "ok", "good"]),
        Choice("q", options={"a": 1.0, "b": 0.5, "n/a": None}, required=True),
    ],
)
def test_question_to_dict_round_trips(question):
    data = question.to_dict()
    assert json.loads(json.dumps(data)) == data
    rebuilt = TypedRubricEvaluator([data], model=None).questions["q1"]
    assert rebuilt == question


def test_score_allows_more_than_ten_levels():
    assert len(Score("Rate it from 0 to 10", levels=[str(i) for i in range(11)]).levels) == 11


def test_invalid_tagged_question_names_its_id():
    with pytest.raises(ValueError, match=r"question <q2> \| question weight"):
        TypedRubricEvaluator([Noul("ok"), {"type": "noul", "text": "q", "weight": 0}], model=None)
    with pytest.raises(TypeError, match=r"question <q1> \| .*wieght"):
        TypedRubricEvaluator([{"type": "noul", "text": "q", "wieght": 2}], model=None)


@pytest.mark.parametrize(
    "item, error",
    [({"type": "slider", "text": "q"}, ValueError), ({"type": "noul", "txt": "q"}, TypeError), ("q", TypeError)],
)
def test_unknown_question_shapes_are_rejected(item, error):
    with pytest.raises(error):
        TypedRubricEvaluator([item], model=None)


# --- constructor ------------------------------------------------------------------------------


def test_model_is_keyword_only_and_required():
    with pytest.raises(TypeError):
        TypedRubricEvaluator([Noul("q")])


def test_explicit_none_model_warns(caplog):
    with caplog.at_level(logging.WARNING):
        evaluator = TypedRubricEvaluator([Noul("q")], model=None)
    assert evaluator.model is None
    assert "no decision model attached" in caplog.text


def test_model_must_be_a_decision_model():
    with pytest.raises(TypeError, match="DecisionModel"):
        TypedRubricEvaluator([Noul("q")], model="global.anthropic.claude-sonnet-4-6")
    evaluator = TypedRubricEvaluator([Noul("q")], model=None)
    with pytest.raises(TypeError, match="DecisionModel"):
        evaluator.model = object()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"questions": []}, "questions"),
        ({"state_fields": []}, "state_fields"),
        ({"state_fields": "input"}, "state_fields"),
        ({"state_fields": ["input", "output"]}, "unknown state_fields"),
        ({"state_fields": ["input", "input"]}, "distinct"),
        ({"pass_threshold": 1.1}, "pass_threshold"),
        ({"pass_threshold": False}, "pass_threshold"),
    ],
)
def test_constructor_validation(kwargs, message):
    args = {"questions": [Noul("q")], "model": None, **kwargs}
    with pytest.raises(ValueError, match=message):
        TypedRubricEvaluator(**args)


def test_state_fields_list_is_stored_as_tuple():
    evaluator = TypedRubricEvaluator([Noul("q")], model=None, state_fields=["input"])
    assert evaluator.state_fields == ("input",)


# --- state ------------------------------------------------------------------------------------


class Order(BaseModel):
    item: str
    quantity: int


async def test_state_keeps_field_order_and_sorts_nested_keys():
    model = StubDecisionModel({"q1": YES})
    evaluator = TypedRubricEvaluator(
        [Noul("q")], model=model, state_fields=["metadata", "input", "actual_output", "expected_output"]
    )
    await evaluator.evaluate_async(_case(metadata={"b": 1, "a": [Order(item="x", quantity=2)]}))

    state = model.calls[0][0]
    assert state == json.dumps(
        {
            "metadata": {"a": [{"item": "x", "quantity": 2}], "b": 1},
            "input": "What is 2+2?",
            "actual_output": "4",
            "expected_output": None,
        },
        ensure_ascii=False,
    )
    assert list(json.loads(state)) == ["metadata", "input", "actual_output", "expected_output"]


async def test_state_is_identical_across_calls():
    model = StubDecisionModel({"q1": YES})
    evaluator = TypedRubricEvaluator([Noul("q")], model=model, state_fields=["input", "metadata"])
    await evaluator.evaluate_async(_case(metadata={"y": 1, "x": 2}))
    await evaluator.evaluate_async(_case(metadata={"x": 2, "y": 1}))
    assert model.calls[0][0] == model.calls[1][0]


async def test_actual_output_required_only_when_selected():
    evaluator = _evaluator([Noul("q")], {"q1": YES})
    with pytest.raises(ValueError, match="actual_output"):
        await evaluator.evaluate_async(_case(actual_output=None))

    input_only = _evaluator([Noul("q")], {"q1": YES}, state_fields=["input"])
    assert len(await input_only.evaluate_async(_case(actual_output=None))) == 1


@pytest.mark.parametrize(
    "metadata",
    [{"x": float("nan")}, {"x": b"bytes"}, {"x": object()}, {"x": "\ud800"}, {"x": {1: "a", "1": "b"}}],
)
async def test_unencodable_state_names_the_field(metadata):
    model = StubDecisionModel({"q1": YES})
    evaluator = TypedRubricEvaluator([Noul("q")], model=model, state_fields=["metadata"])
    with pytest.raises(ValueError, match="state field <metadata>"):
        await evaluator.evaluate_async(_case(metadata=metadata))
    assert model.calls == []


async def test_key_collision_error_names_the_likely_cause():
    evaluator = _evaluator([Noul("q")], {"q1": YES}, state_fields=["metadata"])
    with pytest.raises(ValueError, match="dict keys of mixed types"):
        await evaluator.evaluate_async(_case(metadata={"x": {1: "a", "1": "b"}}))


def test_question_naming_an_unselected_field_warns(caplog):
    with caplog.at_level(logging.WARNING):
        TypedRubricEvaluator(
            [
                Noul("Does `actual_output` match `expected_output`?"),
                Noul("Does `actual_output` call `search`?"),
                Noul("Is `input` answered?"),
            ],
            model=None,
        )
    warnings = [record.getMessage() for record in caplog.records if "not in state_fields" in record.getMessage()]
    assert warnings == [
        "evaluator=<TypedRubricEvaluator>, question_id=<q1>, fields=<expected_output> | question names fields "
        "that are not in state_fields so the decision model cannot see them"
    ]


async def test_trace_fields_come_from_the_last_turn():
    model = StubDecisionModel({"q1": YES})
    evaluator = TypedRubricEvaluator([Noul("q")], model=model, state_fields=["user_prompt", "assistant_response"])
    session = _session(("first question", "first answer"), ("second question", "second answer"))
    await evaluator.evaluate_async(_case(actual_trajectory=session))

    assert json.loads(model.calls[0][0]) == {"assistant_response": "second answer", "user_prompt": "second question"}


@pytest.mark.parametrize("trajectory, error", [(None, TypeError), (["tool"], TypeError), (_session(), ValueError)])
async def test_trace_fields_require_a_session_with_a_turn(trajectory, error):
    evaluator = _evaluator([Noul("q")], {"q1": YES}, state_fields=["assistant_response"])
    with pytest.raises(error):
        await evaluator.evaluate_async(_case(actual_trajectory=trajectory))


async def test_case_fields_do_not_need_a_trace():
    evaluator = _evaluator([Noul("q")], {"q1": YES})
    rows = await evaluator.evaluate_async(_case(actual_trajectory=["not", "a", "session"]))
    assert len(rows) == 1


# --- questions sent to the backend ------------------------------------------------------------


async def test_one_call_with_ids_choices_and_unchanged_text():
    model = StubDecisionModel(
        {
            "q1": YES,
            "q2": {"poor": 0.1, "ok": 0.2, "good": 0.7},
            "q3": {"b": 0.6, "a": 0.4},
        }
    )
    evaluator = TypedRubricEvaluator(
        [
            Noul("Is `actual_output` right?"),
            Score("How good?", levels=["poor", "ok", "good"]),
            Choice("Which?", options={"b": 0.0, "a": 1.0}),
        ],
        model=model,
    )
    await evaluator.evaluate_async(_case())

    assert len(model.calls) == 1
    assert model.calls[0][1] == {
        "q1": {"question": "Is `actual_output` right?", "choices": "boolean", "uncertainty": False},
        "q2": {"question": "How good?", "choices": ["poor", "ok", "good"], "uncertainty": False},
        "q3": {"question": "Which?", "choices": ["b", "a"], "uncertainty": False},
    }


async def test_questions_are_rebuilt_for_every_call():
    class MutatingModel(StubDecisionModel):
        async def _ask(self, state, questions, **kwargs):
            result = await super()._ask(state, questions, **kwargs)
            questions["q1"]["choices"].append("tampered")
            return result

    model = MutatingModel({"q1": {"a": 1.0, "b": 0.0}})
    evaluator = TypedRubricEvaluator([Choice("q", options={"a": 1.0, "b": 0.0})], model=model)
    await evaluator.evaluate_async(_case())
    await evaluator.evaluate_async(_case())
    assert model.calls[1][1]["q1"]["choices"] == ["a", "b"]


# --- response validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "probabilities, answers, confidence, message",
    [
        ({"q1": YES}, {}, None, "no answer"),
        (None, {"q1": True}, None, "no probability distribution"),
        ({}, {"q1": True}, None, "no probability distribution"),
        ({"q1": {"true": 1.0}}, {"q1": True}, None, "cover exactly"),
        ({"q1": {"true": 0.5, "false": 0.5, "maybe": 0.0}}, {"q1": True}, None, "cover exactly"),
        ({"q1": {"true": 1.2, "false": -0.2}}, {"q1": True}, None, r"\[0, 1\]"),
        ({"q1": {"true": float("nan"), "false": 0.5}}, {"q1": True}, None, r"\[0, 1\]"),
        ({"q1": {"true": True, "false": 0.0}}, {"q1": True}, None, r"\[0, 1\]"),
        ({"q1": {"true": 0.6, "false": 0.6}}, {"q1": True}, None, "sum to 1"),
        ({"q1": YES}, {"q1": True}, {"q1": 2.0}, "confidence"),
    ],
)
async def test_malformed_responses_fail_the_whole_rubric(probabilities, answers, confidence, message):
    model = StubDecisionModel(probabilities, answers=answers, confidence=confidence)
    evaluator = TypedRubricEvaluator([Noul("q")], model=model)
    with pytest.raises(ValueError, match=message):
        await evaluator.evaluate_async(_case())


async def test_committed_answer_need_not_be_the_most_likely():
    model = StubDecisionModel({"q1": {"true": 0.4, "false": 0.6}}, answers={"q1": True})
    rows = await TypedRubricEvaluator([Noul("q")], model=model).evaluate_async(_case())
    assert rows[0].score == pytest.approx(0.4)
    assert rows[0].label == "false"
    assert rows[0].metadata["answer"] is True


async def test_uncertain_answer_despite_disabled_uncertainty_is_an_error():
    model = StubDecisionModel({"q1": {"true": 0.55, "false": 0.45}}, answers={"q1": Uncertain("low confidence")})
    with pytest.raises(ValueError, match="uncertainty was disabled"):
        await TypedRubricEvaluator([Noul("q")], model=model).evaluate_async(_case())


# --- scoring ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected, distribution, score, label",
    [
        (True, {"true": 0.9, "false": 0.1}, 0.9, "true"),
        (False, {"true": 0.9, "false": 0.1}, 0.1, "true"),
        (False, {"true": 0.2, "false": 0.8}, 0.8, "false"),
        (True, {"true": 0.5, "false": 0.5}, 0.5, "true"),
    ],
)
async def test_noul_scores_probability_of_expected_answer(expected, distribution, score, label):
    row = (await _evaluator([Noul("q", expected=expected)], {"q1": distribution}).evaluate_async(_case()))[0]
    assert row.score == pytest.approx(score)
    assert row.label == label


@pytest.mark.parametrize(
    "distribution, score, label",
    [
        ({"poor": 0.0, "ok": 0.0, "good": 1.0}, 1.0, "good"),
        ({"poor": 1.0, "ok": 0.0, "good": 0.0}, 0.0, "poor"),
        ({"poor": 0.2, "ok": 0.36, "good": 0.44}, (0.36 + 2 * 0.44) / 2, "good"),
        ({"poor": 0.5, "ok": 0.0, "good": 0.5}, 0.5, "poor"),
    ],
)
async def test_score_is_the_expected_level(distribution, score, label):
    question = Score("q", levels=["poor", "ok", "good"])
    row = (await _evaluator([question], {"q1": distribution}).evaluate_async(_case()))[0]
    assert row.score == pytest.approx(score)
    assert row.label == label


RISK = {"refuses": 1.0, "hedges": 0.5, "complies": 0.0, "nothing_risky": None}


@pytest.mark.parametrize(
    "distribution, score, label",
    [
        # A near three-way split scores near the mean credit, not the top option's full credit.
        ({"refuses": 0.35, "hedges": 0.32, "complies": 0.33, "nothing_risky": 0.0}, 0.51, "refuses"),
        # The N/A option is the most likely but holds under half the mass: score the credited ones.
        ({"refuses": 0.35, "hedges": 0.05, "complies": 0.2, "nothing_risky": 0.4}, 0.625, "refuses"),
        # Ties between credited options go to declaration order.
        ({"refuses": 0.0, "hedges": 0.5, "complies": 0.5, "nothing_risky": 0.0}, 0.25, "hedges"),
    ],
)
async def test_choice_is_the_expected_credit(distribution, score, label):
    row = (await _evaluator([Choice("q", options=RISK)], {"q1": distribution}).evaluate_async(_case()))[0]
    assert row.score == pytest.approx(score)
    assert row.label == label
    assert row.metadata["selected_option"] == label


@pytest.mark.parametrize(
    "options, distribution",
    [
        (RISK, {"refuses": 0.3, "hedges": 0.1, "complies": 0.1, "nothing_risky": 0.5}),
        (
            {"pass": 1.0, "fail": 0.0, "off_topic": None, "empty": None},
            {"pass": 0.3, "fail": 0.1, "off_topic": 0.3, "empty": 0.3},
        ),
    ],
)
async def test_choice_is_not_applicable_at_half_the_mass(options, distribution):
    row = (await _evaluator([Choice("q", options=options)], {"q1": distribution}).evaluate_async(_case()))[0]
    assert row.label == NOT_APPLICABLE
    assert row.score == 0.0
    assert row.test_pass is True
    assert row.metadata["selected_option"] is None


async def test_score_overshoot_is_clamped():
    # Sums to 1 + 8e-7, inside the 1e-6 tolerance, so the expected level is 1.0000004 before clamping.
    distribution = {"poor": 0.0, "ok": 0.0000008, "good": 1.0}
    question = Score("q", levels=["poor", "ok", "good"])
    row = (await _evaluator([question], {"q1": distribution}).evaluate_async(_case()))[0]
    assert row.score == 1.0


@pytest.mark.parametrize("own_threshold, passed", [(None, True), (0.9, False)])
async def test_row_passes_at_its_own_threshold_or_the_evaluators(own_threshold, passed):
    question = Noul("q", pass_threshold=own_threshold)
    row = (await _evaluator([question], {"q1": {"true": 0.6, "false": 0.4}}).evaluate_async(_case()))[0]
    assert row.test_pass is passed
    assert row.metadata["pass_threshold"] == (own_threshold if own_threshold is not None else 0.5)


async def test_score_exactly_at_threshold_passes():
    row = (await _evaluator([Noul("q")], {"q1": {"true": 0.5, "false": 0.5}}).evaluate_async(_case()))[0]
    assert row.test_pass is True


# --- metadata and reasons ---------------------------------------------------------------------


async def test_row_metadata_shape():
    model = StubDecisionModel({"q1": YES}, confidence={"q1": 0.8}, latency_ms=42)
    question = Noul("q", weight=2.0, required=True)
    row = (await TypedRubricEvaluator([question], model=model).evaluate_async(_case()))[0]

    assert row.metadata == {
        "evaluated_by": "TypedRubricEvaluator",
        "question_id": "q1",
        "question_type": "noul",
        "question": "q",
        "weight": 2.0,
        "required": True,
        "pass_threshold": 0.5,
        "answer": True,
        "selected_option": "true",
        "probabilities": YES,
        "confidence": 0.8,
        "provider_name": "stub-provider",
        "model_id": "stub-1",
        "latency_ms": 42,
    }
    assert row.reason == "true (p=0.90); P(true)=0.90"


async def test_backend_class_recorded_only_without_provider_name():
    model = StubDecisionModel({"q1": YES}, provider_name=None, model_id=None)
    row = (await TypedRubricEvaluator([Noul("q")], model=model).evaluate_async(_case()))[0]
    assert row.metadata["backend"] == "StubDecisionModel"
    assert row.metadata["confidence"] is None


async def test_rows_are_deterministic_for_a_fixed_response():
    evaluator = _evaluator(
        [Noul("q"), Choice("c", options=RISK)],
        {"q1": YES, "q2": {"refuses": 0.35, "hedges": 0.05, "complies": 0.2, "nothing_risky": 0.4}},
    )
    first = await evaluator.evaluate_async(_case())
    second = await evaluator.evaluate_async(_case())
    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]
    assert first[1].reason == "expected credit 0.62; most likely refuses (p=0.35); not applicable mass 0.40"


# --- aggregation ------------------------------------------------------------------------------


def _rows(evaluator, probabilities):
    evaluator.model = StubDecisionModel(probabilities)
    return evaluator.evaluate(_case())


def test_composite_is_the_weighted_mean_and_can_outweigh_a_failure():
    evaluator = TypedRubricEvaluator([Noul("a", weight=3.0), Noul("b", weight=1.0)], model=None)
    rows = _rows(evaluator, {"q1": {"true": 1.0, "false": 0.0}, "q2": {"true": 0.0, "false": 1.0}})
    score, passed, reason = evaluator.aggregator(rows)
    assert score == pytest.approx(0.75)
    assert passed is True
    assert reason.startswith("q1: true (p=1.00)")


def test_failed_required_question_fails_the_case_without_changing_the_score():
    evaluator = TypedRubricEvaluator(
        [Noul("quality", weight=3.0), Noul("safety", weight=1.0, required=True, pass_threshold=0.9)], model=None
    )
    rows = _rows(evaluator, {"q1": {"true": 1.0, "false": 0.0}, "q2": {"true": 0.8, "false": 0.2}})
    score, passed, reason = evaluator.aggregator(rows)
    assert score == pytest.approx(0.95)
    assert passed is False
    assert reason.startswith("required question q2 failed | ")


def test_not_applicable_required_question_does_not_block():
    evaluator = TypedRubricEvaluator([Noul("quality"), Choice("safety", options=RISK, required=True)], model=None)
    rows = _rows(
        evaluator,
        {"q1": YES, "q2": {"refuses": 0.0, "hedges": 0.0, "complies": 0.1, "nothing_risky": 0.9}},
    )
    score, passed, _ = evaluator.aggregator(rows)
    assert score == pytest.approx(0.9)
    assert passed is True


def test_all_not_applicable_case_passes_with_placeholder_score():
    evaluator = TypedRubricEvaluator([Choice("safety", options=RISK)], model=None)
    rows = _rows(evaluator, {"q1": {"refuses": 0.0, "hedges": 0.0, "complies": 0.0, "nothing_risky": 1.0}})
    score, passed, reason = evaluator.aggregator(rows)
    assert (score, passed) == (0.0, True)
    assert "not applicable" in reason


def test_extreme_weights_stay_finite():
    evaluator = TypedRubricEvaluator([Noul("a", weight=1e6), Noul("b", weight=1e-6)], model=None)
    rows = _rows(evaluator, {"q1": {"true": 0.5, "false": 0.5}, "q2": {"true": 0.5, "false": 0.5}})
    score, _, _ = evaluator.aggregator(rows)
    assert score == pytest.approx(0.5)


def test_weights_come_from_the_rubric_not_row_metadata():
    evaluator = TypedRubricEvaluator([Noul("a", weight=3.0), Noul("b", weight=1.0)], model=None)
    rows = _rows(evaluator, {"q1": {"true": 1.0, "false": 0.0}, "q2": {"true": 0.0, "false": 1.0}})
    rows[1].metadata["weight"] = 1000.0
    assert evaluator.aggregator(rows)[0] == pytest.approx(0.75)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_wrong_row_count_is_an_error(count):
    rows = _rows(TypedRubricEvaluator([Noul("a"), Noul("b"), Noul("c")], model=None), {"q1": YES, "q2": YES, "q3": YES})
    evaluator = TypedRubricEvaluator([Noul("a"), Noul("b")], model=None)
    with pytest.raises(ValueError, match="rubric rows"):
        evaluator.aggregator(rows[:count])


# --- running ----------------------------------------------------------------------------------


async def test_running_without_a_model_fails_before_any_work():
    evaluator = TypedRubricEvaluator([Noul("q")], model=None, state_fields=["assistant_response"])
    with pytest.raises(RuntimeError, match="no decision model attached"):
        await evaluator.evaluate_async(_case(actual_trajectory=None))


def test_sync_and_async_give_the_same_rows():
    evaluator = _evaluator([Noul("q"), Score("s", levels=["a", "b"])], {"q1": YES, "q2": {"a": 0.3, "b": 0.7}})
    assert evaluator.evaluate(_case()) == asyncio.run(evaluator.evaluate_async(_case()))


async def test_concurrent_cases_do_not_share_state():
    def probabilities(state):
        p = 0.9 if json.loads(state)["input"] == "good" else 0.1
        return {"q1": {"true": p, "false": round(1 - p, 10)}}

    evaluator = TypedRubricEvaluator([Noul("q")], model=StubDecisionModel(probabilities))
    inputs = ["good", "bad"] * 10
    results = await asyncio.gather(*(evaluator.evaluate_async(_case(input=value)) for value in inputs))
    assert [rows[0].score for rows in results] == pytest.approx([0.9 if value == "good" else 0.1 for value in inputs])


async def test_backend_errors_propagate():
    class FailingModel(StubDecisionModel):
        async def _ask(self, state, questions, **kwargs):
            raise ConnectionError("backend down")

    evaluator = TypedRubricEvaluator([Noul("q")], model=FailingModel({}))
    with pytest.raises(ConnectionError):
        await evaluator.evaluate_async(_case())


# --- serialization ----------------------------------------------------------------------------


def test_to_dict_writes_the_rubric_but_never_the_model(caplog):
    evaluator = TypedRubricEvaluator(
        [Noul("q", weight=2.0), Choice("c", options={"a": 1.0, "b": None})],
        model=StubDecisionModel({}),
        state_fields=["input"],
        pass_threshold=0.7,
        name="quality",
    )
    with caplog.at_level(logging.WARNING):
        data = evaluator.to_dict()

    assert data == {
        "evaluator_type": "TypedRubricEvaluator",
        "questions": {
            "q1": {
                "type": "noul",
                "text": "q",
                "weight": 2.0,
                "required": False,
                "pass_threshold": None,
                "expected": True,
            },
            "q2": {
                "type": "choice",
                "text": "c",
                "weight": 1.0,
                "required": False,
                "pass_threshold": None,
                "options": {"a": 1.0, "b": None},
            },
        },
        "model": None,
        "state_fields": ["input"],
        "pass_threshold": 0.7,
        "name": "quality",
    }
    assert "model_id" not in data
    assert "re-attach it with evaluator.model" in caplog.text


def test_to_dict_does_not_warn_without_a_model(caplog):
    evaluator = TypedRubricEvaluator([Noul("q")], model=None)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        evaluator.to_dict()
    assert caplog.text == ""


def test_rebuilding_from_to_dict_gives_an_equal_config():
    original = TypedRubricEvaluator(
        [Noul("q", required=True, pass_threshold=0.9), Score("s", levels=["a", "b", "c"])],
        model=StubDecisionModel({}),
        state_fields=("input", "expected_output"),
        pass_threshold=0.6,
        name="rubric",
    )
    data = json.loads(json.dumps(original.to_dict()))
    args = {key: value for key, value in data.items() if key != "evaluator_type"}
    assert TypedRubricEvaluator(**args).to_dict() == original.to_dict()


# --- through Experiment -----------------------------------------------------------------------

SAFETY = Choice("How does `actual_output` handle risk?", options=RISK, required=True, pass_threshold=0.9)


def _risk_model(safety):
    """Quality always answers yes; the safety distribution depends on the case input."""
    return StubDecisionModel(lambda state: {"q1": YES, "q2": safety[json.loads(state)["input"]]})


SAFETY_BY_INPUT = {
    "safe": {"refuses": 1.0, "hedges": 0.0, "complies": 0.0, "nothing_risky": 0.0},
    "unsafe": {"refuses": 0.0, "hedges": 0.0, "complies": 1.0, "nothing_risky": 0.0},
    "benign": {"refuses": 0.0, "hedges": 0.0, "complies": 0.0, "nothing_risky": 1.0},
}


def _experiment(evaluators, inputs=("safe", "unsafe", "benign")):
    return Experiment(cases=[Case(name=value, input=value) for value in inputs], evaluators=evaluators)


def _echo(case):
    return case.input


def test_experiment_uses_the_weighted_aggregator_and_keeps_every_row():
    evaluator = TypedRubricEvaluator([Noul("Is it helpful?", weight=3.0), SAFETY], model=_risk_model(SAFETY_BY_INPUT))
    report = _experiment([evaluator]).run_evaluations(_echo)

    assert report.scores == pytest.approx([(3 * 0.9 + 1.0) / 4, (3 * 0.9) / 4, 0.9])
    # "unsafe" fails on the required question; "benign" passes because safety is not applicable.
    assert report.test_passes == [True, False, True]
    assert [len(rows) for rows in report.detailed_results] == [2, 2, 2]
    assert report.detailed_results[2][1].label == NOT_APPLICABLE


def test_experiment_async_matches_sync():
    evaluator = TypedRubricEvaluator([Noul("Is it helpful?"), SAFETY], model=_risk_model(SAFETY_BY_INPUT))
    sync_report = _experiment([evaluator]).run_evaluations(_echo)
    async_report = asyncio.run(_experiment([evaluator]).run_evaluations_async(_echo, max_workers=3))
    assert async_report.scores == pytest.approx(sync_report.scores)
    assert async_report.test_passes == sync_report.test_passes


def test_named_rubrics_coexist_with_other_evaluators():
    quality = TypedRubricEvaluator([Noul("q")], model=StubDecisionModel({"q1": YES}), name="quality")
    tone = TypedRubricEvaluator([Noul("t")], model=StubDecisionModel({"q1": {"true": 0.2, "false": 0.8}}), name="tone")
    report = _experiment([quality, tone, Equals(value="safe")], inputs=["safe"]).run_evaluations(_echo)

    by_name = {row["evaluator"]: score for row, score in zip(report.cases, report.scores, strict=True)}
    assert by_name == pytest.approx({"quality": 0.9, "tone": 0.2, "Equals": 1.0})


def test_all_not_applicable_case_is_left_out_of_the_overall_score():
    evaluator = TypedRubricEvaluator(
        [Choice("c", options=RISK)],
        model=StubDecisionModel(lambda state: {"q1": SAFETY_BY_INPUT[json.loads(state)["input"]]}),
    )
    report = _experiment([evaluator], inputs=["safe", "benign"]).run_evaluations(_echo)
    assert report.scores == [1.0, 0.0]
    assert report.overall_score == 1.0


def test_backend_error_fails_only_that_evaluator():
    class FailingModel(StubDecisionModel):
        async def _ask(self, state, questions, **kwargs):
            raise ConnectionError("backend down")

    rubric = TypedRubricEvaluator([Noul("q")], model=FailingModel({}))
    report = _experiment([rubric, Equals(value="safe")], inputs=["safe"]).run_evaluations(_echo)

    rows = {row["evaluator"]: index for index, row in enumerate(report.cases)}
    assert report.test_passes[rows["TypedRubricEvaluator"]] is False
    assert "backend down" in report.reasons[rows["TypedRubricEvaluator"]]
    assert report.test_passes[rows["Equals"]] is True


def test_throttled_backend_is_retried():
    class ThrottledOnceModel(StubDecisionModel):
        async def _ask(self, state, questions, **kwargs):
            if not self.calls:
                self.calls.append("throttled")
                raise ModelThrottledException("slow down")
            return await super()._ask(state, questions, **kwargs)

    model = ThrottledOnceModel({"q1": YES})
    evaluator = TypedRubricEvaluator([Noul("q")], model=model)
    with (
        patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01),
        patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02),
    ):
        report = _experiment([evaluator], inputs=["safe"]).run_evaluations(_echo)

    assert len(model.calls) == 2
    assert report.scores == pytest.approx([0.9])


def test_failed_required_question_is_labelled_no_in_telemetry():
    labels = []

    def recording(*args, **kwargs):
        labels.append(_get_label_from_score(*args, **kwargs))
        return labels[-1]

    evaluator = TypedRubricEvaluator([Noul("Is it helpful?", weight=3.0), SAFETY], model=_risk_model(SAFETY_BY_INPUT))
    with patch("strands_evals.experiment._get_label_from_score", side_effect=recording):
        report = _experiment([evaluator], inputs=["unsafe"]).run_evaluations(_echo)

    # The composite clears 0.5, so a score-based label would say YES for a failing case.
    assert report.scores[0] >= 0.5
    assert report.test_passes == [False]
    assert labels == ["NO"]


def test_experiment_file_round_trip_and_model_reattachment(tmp_path):
    rubric = TypedRubricEvaluator(
        [Noul("Is it helpful?"), SAFETY], model=_risk_model(SAFETY_BY_INPUT), state_fields=["input"], name="rubric"
    )
    path = tmp_path / "experiment.json"
    _experiment([rubric, Equals(value="safe")], inputs=["safe"]).to_file(str(path))

    loaded = Experiment.from_file(str(path))
    reloaded = next(evaluator for evaluator in loaded.evaluators if isinstance(evaluator, TypedRubricEvaluator))
    assert reloaded.to_dict() == rubric.to_dict()

    before = loaded.run_evaluations(_echo)
    rows = {row["evaluator"]: index for index, row in enumerate(before.cases)}
    assert "no decision model attached" in before.reasons[rows["rubric"]]
    assert before.test_passes[rows["Equals"]] is True

    reloaded.model = _risk_model(SAFETY_BY_INPUT)
    after = loaded.run_evaluations(_echo)
    assert after.scores[rows["rubric"]] == pytest.approx((0.9 + 1.0) / 2)


def test_row_metadata_survives_report_json_and_flatten(tmp_path):
    first = TypedRubricEvaluator([Noul("q")], model=StubDecisionModel({"q1": YES}), name="first")
    second = TypedRubricEvaluator([Noul("q")], model=StubDecisionModel({"q1": YES}), name="second")
    reports = [_experiment([evaluator], inputs=["safe"]).run_evaluations(_echo) for evaluator in (first, second)]

    flattened = EvaluationReport.flatten(reports)
    path = tmp_path / "report.json"
    flattened.to_file(str(path))
    loaded = EvaluationReport.from_file(str(path))

    assert [rows[0].metadata for rows in loaded.detailed_results] == [
        rows[0].metadata for report in reports for rows in report.detailed_results
    ]
    assert loaded.detailed_results[0][0].metadata["evaluated_by"] == "TypedRubricEvaluator"


# --- question ids -----------------------------------------------------------------------------


async def test_mapping_keys_become_question_ids():
    model = StubDecisionModel({"helpful": YES, "safety": SAFETY_BY_INPUT["unsafe"]})
    evaluator = TypedRubricEvaluator({"helpful": Noul("Is it helpful?"), "safety": SAFETY}, model=model)
    rows = await evaluator.evaluate_async(_case())

    assert list(model.calls[0][1]) == ["helpful", "safety"]
    assert [row.metadata["question_id"] for row in rows] == ["helpful", "safety"]
    _, passed, reason = evaluator.aggregator(rows)
    assert passed is False
    assert reason.startswith("required question safety failed | helpful: true (p=0.90)")


def test_mapping_ids_survive_to_dict_and_insertion():
    evaluator = TypedRubricEvaluator({"safety": SAFETY, "helpful": Noul("h")}, model=None)
    data = evaluator.to_dict()
    assert list(data["questions"]) == ["safety", "helpful"]

    # Inserting a question keeps every existing id pointing at the same question.
    rebuilt = TypedRubricEvaluator({"tone": Noul("t"), **data["questions"]}, model=None)
    assert list(rebuilt.questions) == ["tone", "safety", "helpful"]
    assert rebuilt.questions["safety"] == SAFETY
    with pytest.raises(TypeError):
        rebuilt.questions["safety"] = Noul("x")


@pytest.mark.parametrize("questions", [{"": Noul("q")}, {" ": Noul("q")}, {1: Noul("q")}, {}])
def test_invalid_question_ids_are_rejected(questions):
    with pytest.raises(ValueError, match="question"):
        TypedRubricEvaluator(questions, model=None)


# --- copying ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question", [Noul("q", weight=2.0), Score("q", levels=["a", "b"]), Choice("q", options=RISK, required=True)]
)
def test_questions_deep_copy_and_pickle(question):
    assert copy.deepcopy(question) == question
    assert pickle.loads(pickle.dumps(question)) == question


def test_evaluator_with_a_choice_deep_copies():
    evaluator = TypedRubricEvaluator({"safety": SAFETY}, model=None)
    assert copy.deepcopy(evaluator).to_dict() == evaluator.to_dict()
