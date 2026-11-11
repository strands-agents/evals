import copy
import pickle

import pytest

from strands_evals.types.typed_rubric import Choice, Noul, Scale

RISK = {"refuses": 1.0, "hedges": 0.5, "complies": 0.0, "nothing_risky": None}


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: Noul(""), "blank"),
        (lambda: Noul("   "), "blank"),
        (lambda: Noul("q", weight=0), "weight"),
        (lambda: Noul("q", weight=5e-324), "weight"),
        (lambda: Noul("q", weight=1e7), "weight"),
        (lambda: Noul("q", weight=float("nan")), "weight"),
        (lambda: Noul("q", weight=True), "weight"),
        (lambda: Noul("q", expected=1), "expected"),
        (lambda: Noul("q", required="yes"), "required"),
        (lambda: Noul("q", pass_threshold=1.5), "pass_threshold"),
        (lambda: Noul("q", pass_threshold=True), "pass_threshold"),
        (lambda: Scale("q", levels=["only"]), "at least 2"),
        (lambda: Scale("q", levels="ab"), "valid tuple"),
        (lambda: Scale("q", levels=["a", 1]), "valid string"),
        (lambda: Scale("q", levels=["a", "a"]), "distinct"),
        (lambda: Scale("q", levels=["a", "uncertain"]), "reserved"),
        (lambda: Scale("q", levels=["not_applicable", "poor", "good"]), "reserved"),
        (lambda: Scale("q", levels=["a", " "]), "blank"),
        (lambda: Choice("q", options={"a": 1.0}), "at least 2"),
        (lambda: Choice("q", options={"a": 1.5, "b": 0.0}), "less than or equal to 1"),
        (lambda: Choice("q", options={"a": True, "b": 0.0}), "valid number"),
        (lambda: Choice("q", options={"a": None, "b": None}), "numeric credit"),
        (lambda: Choice("q", options={"uncertain": 1.0, "b": 0.0}), "reserved"),
        (lambda: Choice("q", options={"yes": 1.0, "partially": 0.5, "not_applicable": 0.0}), "reserved"),
        (lambda: Choice("q", options=[("a", 1.0), ("b", 0.0)]), "valid dictionary"),
        (lambda: Noul("q", wieght=2.0), "wieght"),
    ],
)
def test_question_validation(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_question_collections_are_immutable_copies():
    levels = ["poor", "good"]
    options = {"yes": 1.0, "no": 0.0}
    score = Scale("q", levels=levels)
    choice = Choice("q", options=options)
    levels.append("great")
    options["maybe"] = 0.5

    assert score.levels == ("poor", "good")
    assert dict(choice.options) == {"yes": 1.0, "no": 0.0}
    with pytest.raises(TypeError):
        choice.options["maybe"] = 0.5


def test_scale_allows_more_than_ten_levels():
    assert len(Scale("Rate it from 0 to 10", levels=[str(i) for i in range(11)]).levels) == 11


@pytest.mark.parametrize(
    "question", [Noul("q", weight=2.0), Scale("q", levels=["a", "b"]), Choice("q", options=RISK, required=True)]
)
def test_questions_deep_copy_and_pickle(question):
    assert copy.deepcopy(question) == question
    assert pickle.loads(pickle.dumps(question)) == question
