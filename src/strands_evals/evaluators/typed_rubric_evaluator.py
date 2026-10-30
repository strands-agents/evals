"""Typed-rubric scoring through a decision model.

A rubric is a set of typed questions (`Noul`, `Score`, `Choice`). One `DecisionModel.ask` call
answers all of them, and each question's probability distribution is turned into a score with a
fixed formula, so the same rubric and backend response always give the same rows. The formulas
match DeepEval's JevEval: `Noul` scores P(expected), `Score` the expected level, and `Choice` the
expected credit over its applicable options.

Requires a `DecisionModel` that reports per-question probabilities in
`DecisionResult.metadata.probabilities`. Backends that only return committed answers are rejected.
"""

import json
import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from pydantic import BaseModel
from strands.decisions import DecisionModel, DecisionResult, Question, Uncertain

from .._async import run_async
from ..extractors import TraceExtractor
from ..types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

# Marks the rows this evaluator produced, under the same key `StructuredOutputSimilarity` uses.
_MARKER = "TypedRubricEvaluator"

_MIN_WEIGHT = 1e-6
_MAX_WEIGHT = 1e6
# No upper bound: `Score` is sent as a choice question, and each backend enforces its own limit.
_MIN_LEVELS = 2
# "uncertain" is reserved by the SDK for its `Uncertain` answer. `NOT_APPLICABLE` is the row label
# that `EvaluationOutput.not_applicable` checks, so an option with that name would make a scored
# row look unjudged and drop it from the composite and the required gate.
_RESERVED_LABELS = ("uncertain", NOT_APPLICABLE)
_PROBABILITY_TOLERANCE = 1e-6

_CASE_FIELDS = ("input", "actual_output", "expected_output", "expected_assertion", "metadata")
_TRACE_FIELDS = ("user_prompt", "assistant_response")
_FIELD_REFERENCE = re.compile(r"`([A-Za-z_]+)`")


@dataclass(frozen=True)
class _Verdict:
    """One question's scored result. `score` is None when the question does not apply."""

    score: float | None
    selected: str | None
    reason: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_text(value: Any, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")


def _require_bool(value: Any, what: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be a bool")


def _require_threshold(value: Any, what: str) -> None:
    if not _is_number(value) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{what} must be a number in [0, 1]")


def _validate_common(question: "RubricQuestion") -> None:
    _require_text(question.text, "question text")
    if not _is_number(question.weight) or not _MIN_WEIGHT <= question.weight <= _MAX_WEIGHT:
        raise ValueError(f"question weight must be a number in [{_MIN_WEIGHT}, {_MAX_WEIGHT}]")
    _require_bool(question.required, "question required")
    if question.pass_threshold is not None:
        _require_threshold(question.pass_threshold, "question pass_threshold")


def _validate_labels(labels: Sequence[str], what: str) -> None:
    for label in labels:
        _require_text(label, f"{what} label")
        if label in _RESERVED_LABELS:
            raise ValueError(f"{what} label {label!r} is reserved; reserved labels are {list(_RESERVED_LABELS)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{what} labels must be distinct")


def _most_likely(labels: Sequence[str], distribution: Mapping[str, float]) -> str:
    """Return the most likely label, breaking ties by declaration order."""
    return max(labels, key=lambda label: distribution[label])


def _common_dict(question: "RubricQuestion") -> dict[str, Any]:
    return {
        "text": question.text,
        "weight": question.weight,
        "required": question.required,
        "pass_threshold": question.pass_threshold,
    }


@dataclass(frozen=True)
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

    text: str
    weight: float = 1.0
    expected: bool = True
    required: bool = False
    pass_threshold: float | None = None

    type: ClassVar[str] = "noul"

    def __post_init__(self) -> None:
        _validate_common(self)
        _require_bool(self.expected, "Noul expected")

    def to_dict(self) -> dict[str, Any]:
        """Return the tagged, JSON-safe form that `TypedRubricEvaluator` accepts back."""
        return {"type": self.type, **_common_dict(self), "expected": self.expected}

    def _choices(self) -> Literal["boolean"]:
        return "boolean"

    def _labels(self) -> tuple[str, ...]:
        return ("true", "false")

    def _score(self, distribution: Mapping[str, float]) -> _Verdict:
        selected = "true" if distribution["true"] >= distribution["false"] else "false"
        expected = "true" if self.expected else "false"
        score = distribution[expected]
        reason = f"{selected} (p={distribution[selected]:.2f}); P({expected})={score:.2f}"
        return _Verdict(score=score, selected=selected, reason=reason)


@dataclass(frozen=True)
class Score:
    """A rating on an ordered scale, scored as the expected level normalised to [0, 1].

    The backend picks among the levels as a categorical choice. The ordering is used only by the
    scoring formula, `sum(index * p) / (len(levels) - 1)`, so this is not native ordinal inference.

    Attributes:
        text: The question, naming the state fields it reads.
        levels: At least 2 distinct labels, worst first. Stored as a tuple.
        weight: Relative weight in the composite score, in [1e-6, 1e6].
        required: When True, the case fails if this question fails, whatever the composite.
        pass_threshold: This question's pass threshold. None uses the evaluator's.
    """

    text: str
    levels: Sequence[str]
    weight: float = 1.0
    required: bool = False
    pass_threshold: float | None = None

    type: ClassVar[str] = "score"

    def __post_init__(self) -> None:
        _validate_common(self)
        if isinstance(self.levels, str) or not isinstance(self.levels, Sequence):
            raise ValueError("Score levels must be a sequence of strings")
        object.__setattr__(self, "levels", tuple(self.levels))
        if len(self.levels) < _MIN_LEVELS:
            raise ValueError(f"Score needs at least {_MIN_LEVELS} levels")
        _validate_labels(self.levels, "Score level")

    def to_dict(self) -> dict[str, Any]:
        """Return the tagged, JSON-safe form that `TypedRubricEvaluator` accepts back."""
        return {"type": self.type, **_common_dict(self), "levels": list(self.levels)}

    def _choices(self) -> list[str]:
        return list(self.levels)

    def _labels(self) -> tuple[str, ...]:
        return tuple(self.levels)

    def _score(self, distribution: Mapping[str, float]) -> _Verdict:
        top = len(self.levels) - 1
        expected_level = sum(index * distribution[level] for index, level in enumerate(self.levels))
        selected = _most_likely(self.levels, distribution)
        reason = (
            f"expected level {expected_level:.2f} of {top}; most likely {selected} (p={distribution[selected]:.2f})"
        )
        # The probabilities may sum to slightly over 1 (within tolerance), which can push the
        # expected level just past the top; clamp that overshoot.
        return _Verdict(score=min(expected_level / top, 1.0), selected=selected, reason=reason)


@dataclass(frozen=True)
class Choice:
    """A pick among labelled options, scored as the expected credit over applicable options.

    Options with a `None` credit mean "does not apply". When they hold at least half the
    probability, the row is `NOT_APPLICABLE`. Otherwise the score is
    `sum(p * credit) / sum(p)` over the credited options, and the row's label is the most likely
    credited option.

    Attributes:
        text: The question, naming the state fields it reads.
        options: At least 2 distinct labels mapped to a credit in [0, 1], or None for "does not
            apply". At least one credit must be a number. Stored as a read-only copy, order kept.
        weight: Relative weight in the composite score, in [1e-6, 1e6].
        required: When True, the case fails if this question applies and fails, whatever the composite.
        pass_threshold: This question's pass threshold. None uses the evaluator's.
    """

    text: str
    options: Mapping[str, float | None]
    weight: float = 1.0
    required: bool = False
    pass_threshold: float | None = None

    type: ClassVar[str] = "choice"

    def __post_init__(self) -> None:
        _validate_common(self)
        if not isinstance(self.options, Mapping):
            raise ValueError("Choice options must be a mapping of label to credit")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        if len(self.options) < 2:
            raise ValueError("Choice needs at least 2 options")
        _validate_labels(list(self.options), "Choice option")
        for label, credit in self.options.items():
            if credit is not None and (not _is_number(credit) or not math.isfinite(credit) or not 0 <= credit <= 1):
                raise ValueError(f"Choice option {label!r} credit must be a number in [0, 1] or None")
        if all(credit is None for credit in self.options.values()):
            raise ValueError("Choice needs at least one option with a numeric credit")

    def __reduce__(self) -> tuple[Any, ...]:
        # `MappingProxyType` cannot be pickled or deep-copied; rebuild through the constructor.
        return (Choice, (self.text, dict(self.options), self.weight, self.required, self.pass_threshold))

    def to_dict(self) -> dict[str, Any]:
        """Return the tagged, JSON-safe form that `TypedRubricEvaluator` accepts back."""
        return {"type": self.type, **_common_dict(self), "options": dict(self.options)}

    def _choices(self) -> list[str]:
        return list(self.options)

    def _labels(self) -> tuple[str, ...]:
        return tuple(self.options)

    def _score(self, distribution: Mapping[str, float]) -> _Verdict:
        not_applicable_mass = sum(distribution[label] for label, credit in self.options.items() if credit is None)
        if not_applicable_mass >= 0.5:
            return _Verdict(score=None, selected=None, reason=f"not applicable (p={not_applicable_mass:.2f})")

        credited = {label: credit for label, credit in self.options.items() if credit is not None}
        credited_mass = sum(distribution[label] for label in credited)
        expected_credit = sum(distribution[label] * credit for label, credit in credited.items()) / credited_mass
        selected = _most_likely(list(credited), distribution)
        reason = f"expected credit {expected_credit:.2f}; most likely {selected} (p={distribution[selected]:.2f})"
        if not_applicable_mass > 0:
            reason += f"; not applicable mass {not_applicable_mass:.2f}"
        return _Verdict(score=expected_credit, selected=selected, reason=reason)


RubricQuestion = Noul | Score | Choice
"""Any question type a `TypedRubricEvaluator` accepts."""

# Adding a question type means adding it here and to `RubricQuestion`.
_QUESTION_CLASSES = (Noul, Score, Choice)
_QUESTION_TYPES: dict[str, type[RubricQuestion]] = {cls.type: cls for cls in _QUESTION_CLASSES}


def _question_from(question_id: str, item: RubricQuestion | Mapping[str, Any]) -> RubricQuestion:
    """Accept a question, or the tagged dict its `to_dict` writes. Errors name the question id."""
    if isinstance(item, _QUESTION_CLASSES):
        return item
    if not isinstance(item, Mapping):
        names = ", ".join(cls.__name__ for cls in _QUESTION_CLASSES)
        raise TypeError(f"question <{question_id}> | must be a {names} or tagged dict, got {type(item).__name__}")
    fields = dict(item)
    tag = fields.pop("type", None)
    question_cls = _QUESTION_TYPES.get(tag) if isinstance(tag, str) else None
    if question_cls is None:
        raise ValueError(f"question <{question_id}> | unknown type {tag!r}; expected one of {sorted(_QUESTION_TYPES)}")
    try:
        return question_cls(**fields)
    except TypeError as error:
        raise TypeError(f"question <{question_id}> | {error}") from error
    except ValueError as error:
        raise ValueError(f"question <{question_id}> | {error}") from error


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _encode_state(state: Mapping[str, Any]) -> str:
    """Render the state as deterministic JSON text, naming the field that cannot be encoded.

    Top-level fields keep their `state_fields` order, so the author controls what the model reads
    first; keys inside each value are sorted. `sort_keys` compares raw keys before converting them,
    so colliding keys such as `1` and `"1"` raise rather than being silently merged. That
    `TypeError` reads like a comparison error, so the message names the cause too.
    """
    options: dict[str, Any] = {"sort_keys": True, "ensure_ascii": False, "allow_nan": False, "default": _json_default}
    members = []
    for field, value in state.items():
        try:
            encoded = json.dumps(value, **options)
            # Rejects unpaired surrogates; UnicodeEncodeError is a ValueError.
            encoded.encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"state field <{field}> cannot be encoded as JSON "
                f"(unsupported value, or dict keys of mixed types such as 1 and '1'): {error}"
            ) from error
        members.append(f"{json.dumps(field)}: {encoded}")
    return "{" + ", ".join(members) + "}"


def _validate_distribution(question_id: str, question: RubricQuestion, distribution: Any) -> None:
    if not isinstance(distribution, Mapping) or set(distribution) != set(question._labels()):
        raise ValueError(
            f"question <{question_id}> | probabilities must cover exactly {list(question._labels())}, "
            f"got {list(distribution) if isinstance(distribution, Mapping) else distribution!r}"
        )
    for label, probability in distribution.items():
        if not _is_number(probability) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"question <{question_id}> | probability for {label!r} must be a number in [0, 1]")
    if abs(sum(distribution.values()) - 1) > _PROBABILITY_TOLERANCE:
        raise ValueError(f"question <{question_id}> | probabilities must sum to 1")


class TypedRubricEvaluator(Evaluator[InputT, OutputT]):
    """Score a case against a typed rubric with one decision-model call.

    Questions are given as a mapping of id to question, or as a sequence that gets the ids `q1`,
    `q2`, ... in order. Prefer ids for rubrics that will change: anything keyed by a question id
    keeps pointing at the same question when others are added.

    Each question produces one `EvaluationOutput` row whose score comes from the backend's
    probability distribution. The case score is the weighted mean of the applicable rows. The case
    passes when that mean reaches `pass_threshold` and no applicable required question failed.

    The state sent to the backend is JSON text holding only `state_fields`:

    - `input`, `actual_output`, `expected_output`, `expected_assertion`, `metadata`: read from the
      case. `actual_output` must be set when selected; the others may be null.
    - `user_prompt`, `assistant_response`: read from the last turn of `actual_trajectory`, which
      must then be a `Session`.

    `model` is required. Pass `None` only when the model will be attached later through
    `evaluator.model`, which is how `Experiment.from_dict` rebuilds this evaluator: `to_dict`
    saves the rubric but never the model.

    Prefer `evaluate_async` (which `Experiment` uses). The sync `evaluate` runs on a separate event
    loop, so a backend holding a loop-bound HTTP client is not safe across sync calls. Configure a
    finite request timeout on the backend's client.

    Example:
        ```python
        evaluator = TypedRubricEvaluator(
            questions={
                "answers": Noul("Does `actual_output` answer `input`?", weight=2.0),
                "safety": Choice(
                    "How does `actual_output` handle the risky part of `input`?",
                    options={"refuses": 1.0, "hedges": 0.5, "complies": 0.0, "nothing_risky": None},
                    required=True,
                    pass_threshold=0.9,
                ),
            },
            model=decision_model,
        )
        ```
    """

    # `_get_label_from_score` labels the case from `test_pass`, which accounts for required
    # questions and the configured threshold, rather than from `score >= 0.5`.
    _label_from_test_pass: ClassVar[bool] = True

    def __init__(
        self,
        questions: Mapping[str, RubricQuestion | Mapping[str, Any]] | Sequence[RubricQuestion | Mapping[str, Any]],
        *,
        model: DecisionModel | None,
        state_fields: Sequence[str] = ("input", "actual_output"),
        pass_threshold: float = 0.5,
        name: str | None = None,
    ):
        """Create a typed-rubric evaluator.

        Args:
            questions: A mapping of question id to question, or a sequence of questions that gets
                the ids `q1`, `q2`, ... in order. Each question may also be the tagged dict its
                `to_dict` writes.
            model: The decision model that answers the rubric. `None` leaves the evaluator without
                a model until one is set through `evaluator.model`.
            state_fields: Fields to send to the model. Any sequence; stored as a tuple.
            pass_threshold: Threshold for the composite score, and for questions that set none.
            name: Instance name for reports and telemetry. Defaults to the class name.

        Raises:
            TypeError: If `model` is not a `DecisionModel` or None, or a question has the wrong type.
            ValueError: If the rubric, fields or threshold are invalid.
        """
        if isinstance(questions, Mapping):
            entries = list(questions.items())
        elif isinstance(questions, Sequence) and not isinstance(questions, str):
            entries = [(f"q{index}", item) for index, item in enumerate(questions, start=1)]
        else:
            raise ValueError("questions must be a mapping of id to question, or a sequence of questions")
        if not entries:
            raise ValueError("questions must not be empty")
        for question_id, _ in entries:
            _require_text(question_id, "question id")
        if isinstance(state_fields, str) or not isinstance(state_fields, Sequence) or not state_fields:
            raise ValueError("state_fields must be a non-empty sequence of field names")
        fields = tuple(state_fields)
        unknown = [field for field in fields if field not in _CASE_FIELDS + _TRACE_FIELDS]
        if unknown:
            raise ValueError(f"unknown state_fields {unknown}; supported: {list(_CASE_FIELDS + _TRACE_FIELDS)}")
        if len(set(fields)) != len(fields):
            raise ValueError("state_fields must be distinct")
        _require_threshold(pass_threshold, "pass_threshold")

        uses_trace = any(field in _TRACE_FIELDS for field in fields)
        super().__init__(trace_extractor=TraceExtractor(EvaluationLevel.TRACE_LEVEL) if uses_trace else None, name=name)
        self._question_ids = tuple(question_id for question_id, _ in entries)
        self._questions = tuple(_question_from(question_id, item) for question_id, item in entries)
        self._state_fields = fields
        self._warn_on_unselected_field_references()
        self._uses_trace = uses_trace
        self._pass_threshold = float(pass_threshold)
        self.model = model
        self.aggregator = self._aggregate

    @property
    def model(self) -> DecisionModel | None:
        """The decision model that answers the rubric, or None until one is attached."""
        return self._model

    @model.setter
    def model(self, value: DecisionModel | None) -> None:
        if value is not None and not isinstance(value, DecisionModel):
            raise TypeError(f"model must be a strands.decisions.DecisionModel or None, got {type(value).__name__}")
        if value is None:
            logger.warning(
                "evaluator=<%s> | no decision model attached, set evaluator.model before running", self.get_name()
            )
        self._model = value

    @property
    def questions(self) -> Mapping[str, RubricQuestion]:
        """The rubric questions keyed by id, in rubric order. Read-only."""
        return MappingProxyType(dict(self._entries()))

    @property
    def state_fields(self) -> tuple[str, ...]:
        """The fields sent to the decision model."""
        return self._state_fields

    @property
    def pass_threshold(self) -> float:
        """Threshold for the composite score, and for questions that set none."""
        return self._pass_threshold

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score the case synchronously by running `evaluate_async` on a separate event loop."""
        return run_async(lambda: self.evaluate_async(evaluation_case))

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score the case with one decision-model call, returning one row per question.

        Raises:
            RuntimeError: If no decision model is attached.
            TypeError: If a trace field is selected and `actual_trajectory` is not a `Session`.
            ValueError: If the state cannot be built or the response is incomplete or malformed.
        """
        model = self._model
        if model is None:
            raise RuntimeError(f"evaluator <{self.get_name()}> has no decision model attached; set evaluator.model")
        state = self._build_state(evaluation_case)
        result = await model.ask(state, self._sdk_questions())
        distributions = self._validate_result(result)
        return [
            self._row(question_id, question, result, distributions[question_id])
            for question_id, question in self._entries()
        ]

    def to_dict(self) -> dict:
        """Serialize the rubric configuration. The decision model is never written.

        `"model": None` lets `Experiment.from_dict` call the constructor; re-attach the model with
        `evaluator.model = ...` after loading.
        """
        if self._model is not None:
            logger.warning(
                "evaluator=<%s> | decision model is not serialized, re-attach it with evaluator.model after loading",
                self.get_name(),
            )
        data: dict[str, Any] = {
            "evaluator_type": self.get_type_name(),
            "questions": {question_id: question.to_dict() for question_id, question in self._entries()},
            # Not `model_id`: `Experiment.from_dict` turns that into a generative model string.
            # An explicit None satisfies the keyword-only `model` argument on reload.
            "model": None,
            "state_fields": list(self._state_fields),
            "pass_threshold": self._pass_threshold,
        }
        if self.name is not None:
            data["name"] = self.name
        return data

    def _warn_on_unselected_field_references(self) -> None:
        # Only known field names in backticks count, so ordinary code spans in a question are ignored.
        known = set(_CASE_FIELDS + _TRACE_FIELDS)
        for question_id, question in self._entries():
            unselected = sorted(
                {name for name in _FIELD_REFERENCE.findall(question.text) if name in known} - set(self._state_fields)
            )
            if unselected:
                logger.warning(
                    "evaluator=<%s>, question_id=<%s>, fields=<%s> | question names fields that are not in "
                    "state_fields so the decision model cannot see them",
                    self.get_name(),
                    question_id,
                    ", ".join(unselected),
                )

    def _entries(self) -> Iterator[tuple[str, RubricQuestion]]:
        return zip(self._question_ids, self._questions, strict=True)

    def _sdk_questions(self) -> dict[str, Question]:
        # Built fresh per call so a backend that mutates its input cannot affect another call.
        # Uncertainty is disabled: scores come from the distribution, so an `Uncertain` answer
        # would only restate the backend's confidence threshold.
        return {
            question_id: {"question": question.text, "choices": question._choices(), "uncertainty": False}
            for question_id, question in self._entries()
        }

    def _build_state(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        trace_values: dict[str, str] = {}
        if self._uses_trace:
            turn = self._get_last_turn(evaluation_case)
            trace_values = {
                "user_prompt": self._extract_user_prompt(turn),
                "assistant_response": turn.agent_response.text,
            }

        state: dict[str, Any] = {}
        for field in self._state_fields:
            if field in trace_values:
                state[field] = trace_values[field]
                continue
            value = getattr(evaluation_case, field)
            if field == "actual_output" and value is None:
                raise ValueError("state field <actual_output> is selected but the case has no actual_output")
            state[field] = value
        return _encode_state(state)

    def _validate_result(self, result: DecisionResult) -> dict[str, Mapping[str, float]]:
        """Check the whole response before any row is built, so a rubric is never partly scored.

        Returns:
            The validated distribution for each question id.
        """
        probabilities = result.metadata.probabilities
        confidence = result.metadata.confidence
        distributions: dict[str, Mapping[str, float]] = {}
        for question_id, question in self._entries():
            # The committed answer is recorded but never scored, so only its presence is checked.
            # It is not required to be the most likely option; the `DecisionModel` contract does
            # not promise that, although the System One backends enforce it themselves.
            if question_id not in result.answers:
                raise ValueError(f"question <{question_id}> | decision model returned no answer")
            if isinstance(result.answers[question_id], Uncertain):
                raise ValueError(
                    f"question <{question_id}> | decision model answered Uncertain although uncertainty was disabled"
                )
            if probabilities is None or question_id not in probabilities:
                raise ValueError(
                    f"question <{question_id}> | decision model returned no probability distribution; "
                    "TypedRubricEvaluator requires a backend that reports probabilities"
                )
            _validate_distribution(question_id, question, probabilities[question_id])
            # Confidence is only recorded, but a NaN in row metadata would make the report
            # unwritable, since `EvaluationReport.to_file` rejects non-strict JSON.
            if confidence is not None and question_id in confidence:
                _require_threshold(confidence[question_id], f"question <{question_id}> confidence")
            distributions[question_id] = probabilities[question_id]
        return distributions

    def _row(
        self,
        question_id: str,
        question: RubricQuestion,
        result: DecisionResult,
        distribution: Mapping[str, float],
    ) -> EvaluationOutput:
        metadata = result.metadata
        verdict = question._score(distribution)
        threshold = question.pass_threshold if question.pass_threshold is not None else self._pass_threshold

        row_metadata: dict[str, Any] = {
            "evaluated_by": _MARKER,
            "question_id": question_id,
            "question_type": question.type,
            "question": question.text,
            "weight": question.weight,
            "required": question.required,
            "pass_threshold": threshold,
            "answer": result.answers[question_id],
            "selected_option": verdict.selected,
            "probabilities": dict(distribution),
            "confidence": metadata.confidence.get(question_id) if metadata.confidence else None,
            "provider_name": metadata.provider_name,
            "model_id": metadata.model_id,
            "latency_ms": metadata.latency_ms,
        }
        if metadata.provider_name is None:
            row_metadata["backend"] = type(self._model).__name__

        if verdict.score is None:
            return EvaluationOutput(
                score=0.0, test_pass=True, reason=verdict.reason, label=NOT_APPLICABLE, metadata=row_metadata
            )
        return EvaluationOutput(
            score=verdict.score,
            test_pass=verdict.score >= threshold,
            reason=verdict.reason,
            label=verdict.selected,
            metadata=row_metadata,
        )

    def _aggregate(self, outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        """Weighted mean over applicable rows; an applicable required question that fails fails the case.

        Raises:
            ValueError: If there is not exactly one row per question, including an empty list.
        """
        if len(outputs) != len(self._questions):
            raise ValueError(f"expected {len(self._questions)} rubric rows, got {len(outputs)}")

        rows = list(zip(self._question_ids, self._questions, outputs, strict=True))
        reason = " | ".join(f"{question_id}: {output.reason}" for question_id, _, output in rows if output.reason)
        applicable = [
            (question_id, question, output) for question_id, question, output in rows if not output.not_applicable
        ]
        if not applicable:
            return (0.0, True, reason)

        total_weight = sum(question.weight for _, question, _ in applicable)
        composite = sum(question.weight * output.score for _, question, output in applicable) / total_weight
        required_failures = [
            question_id for question_id, question, output in applicable if question.required and not output.test_pass
        ]
        if required_failures:
            reason = f"required question {', '.join(required_failures)} failed | {reason}"
        return (composite, composite >= self._pass_threshold and not required_failures, reason)
