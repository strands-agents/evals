"""Question types for `TypedRubricEvaluator`.

Each type holds one rubric question and its scoring settings. Validation happens on construction;
the scoring formulas live in `strands_evals.evaluators.typed_rubric_evaluator`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer, TypeAdapter
from pydantic.dataclasses import dataclass

from .evaluation import NOT_APPLICABLE

# "uncertain" is reserved by the SDK for its `Uncertain` answer. `NOT_APPLICABLE` is the row label
# that `EvaluationOutput.not_applicable` checks, so an option with that name would make a scored
# row look unjudged and drop it from the composite and the required gate.
_RESERVED_LABELS = ("uncertain", NOT_APPLICABLE)

# Unknown keyword arguments fail, as do NaN and infinity. Strictness is set per field rather than
# here, so `levels` still accepts a list and tagged dicts loaded from JSON still decode.
_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False)


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _not_reserved(value: str) -> str:
    if value in _RESERVED_LABELS:
        raise ValueError(f"{value!r} is reserved; reserved labels are {list(_RESERVED_LABELS)}")
    return value


def _distinct(levels: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(levels)) != len(levels):
        raise ValueError("levels must be distinct")
    return levels


def _credited_and_frozen(options: Mapping[str, float | None]) -> Mapping[str, float | None]:
    if all(credit is None for credit in options.values()):
        raise ValueError("at least one option needs a numeric credit")
    return MappingProxyType(dict(options))


# `strict=True` keeps bools and numeric strings out of numbers, and numbers out of bools.
_Text = Annotated[str, Field(strict=True), AfterValidator(_not_blank)]
_Label = Annotated[str, Field(strict=True), AfterValidator(_not_blank), AfterValidator(_not_reserved)]
_Weight = Annotated[float, Field(strict=True, ge=1e-6, le=1e6)]
_Threshold = Annotated[float, Field(strict=True, ge=0, le=1)]
_Flag = Annotated[bool, Field(strict=True)]
_Credit = Annotated[float, Field(strict=True, ge=0, le=1)]


@dataclass(frozen=True, config=_CONFIG)
class Noul:
    """A yes/no question, scored as the probability of the expected answer.

    The name comes from the Jev decision-model API, where a yes/no question is a "noul" (DeepEval's
    JevEval uses the same name). It is sent to the backend as an SDK `"boolean"` question.

    Attributes:
        text: The question, naming the state fields it reads, e.g. "Does `actual_output` answer `input`?".
        weight: Relative weight in the composite score, in [1e-6, 1e6].
        expected: The answer that earns full credit.
        required: When True, the case fails if this question fails, whatever the composite.
        pass_threshold: This question's pass threshold. None uses the evaluator's.
    """

    text: _Text
    weight: _Weight = 1.0
    expected: _Flag = True
    required: _Flag = False
    pass_threshold: _Threshold | None = None
    question_type: Literal["Noul"] = Field("Noul", kw_only=True, repr=False)


@dataclass(frozen=True, config=_CONFIG)
class Score:
    """A rating on an ordered scale, scored as the expected level normalised to [0, 1].

    The backend picks among the levels as a categorical choice. The ordering is used only by the
    scoring formula, `sum(index * p) / (len(levels) - 1)`, so this is not native ordinal inference.

    Attributes:
        text: The question, naming the state fields it reads.
        levels: At least 2 distinct labels, worst first. Any sequence; stored as a tuple. No upper
            bound: `Score` is sent as a choice question, and each backend enforces its own limit.
        weight: Relative weight in the composite score, in [1e-6, 1e6].
        required: When True, the case fails if this question fails, whatever the composite.
        pass_threshold: This question's pass threshold. None uses the evaluator's.
    """

    text: _Text
    levels: Annotated[tuple[_Label, ...], Field(min_length=2), AfterValidator(_distinct)]
    weight: _Weight = 1.0
    required: _Flag = False
    pass_threshold: _Threshold | None = None
    question_type: Literal["Score"] = Field("Score", kw_only=True, repr=False)


@dataclass(frozen=True, config=_CONFIG)
class Choice:
    """A pick among labelled options, scored as the expected credit over applicable options.

    Options with a `None` credit mean "does not apply". When they hold at least half the
    probability, the row is `NOT_APPLICABLE`. Otherwise the score is
    `sum(p * credit) / sum(p)` over the credited options, and the row's label is the most likely
    credited option.

    Attributes:
        text: The question, naming the state fields it reads.
        options: At least 2 labels mapped to a credit in [0, 1], or None for "does not apply". At
            least one credit must be a number. Stored as a read-only copy, order kept.
        weight: Relative weight in the composite score, in [1e-6, 1e6].
        required: When True, the case fails if this question applies and fails, whatever the composite.
        pass_threshold: This question's pass threshold. None uses the evaluator's.
    """

    text: _Text
    options: Annotated[
        Mapping[_Label, _Credit | None],
        Field(min_length=2),
        AfterValidator(_credited_and_frozen),
        # The read-only proxy is written out as a plain dict.
        PlainSerializer(dict, return_type=dict[str, float | None]),
    ]
    weight: _Weight = 1.0
    required: _Flag = False
    pass_threshold: _Threshold | None = None
    question_type: Literal["Choice"] = Field("Choice", kw_only=True, repr=False)

    def __reduce__(self) -> tuple[Any, ...]:
        # The read-only options proxy cannot be pickled or deep-copied; rebuild through validation.
        return (Choice, (self.text, dict(self.options), self.weight, self.required, self.pass_threshold))


RubricQuestion = Noul | Score | Choice
"""Any question type a `TypedRubricEvaluator` accepts."""

# Validates a question, or decodes the tagged dict that `_QUESTION_ADAPTER.dump_python` writes.
_QUESTION_ADAPTER: TypeAdapter[RubricQuestion] = TypeAdapter(
    Annotated[RubricQuestion, Field(discriminator="question_type")]
)
