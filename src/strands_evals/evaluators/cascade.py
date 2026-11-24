"""Route each case between a fast typed rubric and a fallback evaluator.

The rubric answers first. When its gating question's probability falls inside the escalation band,
the rubric is unsure, so the case goes to the fallback evaluator, usually an LLM judge, and keeps
the fallback's verdict and written reasoning. Every other case keeps the rubric's rows.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal

from .._async import run_async
from ..types.cascade import _BAND_ADAPTER, Band
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.typed_rubric import Choice, Noul, RubricQuestion
from .evaluator import Evaluator
from .typed_rubric_evaluator import TypedRubricEvaluator, _answerable_labels, _most_likely

logger = logging.getLogger(__name__)

CascadePath = Literal["fast", "fallback"]
"""Which evaluator resolved a case, as written to each row's `metadata["path"]`."""

EscalationReason = Literal["band", "not_applicable"]
"""Why a case went to the fallback, as written to each fallback row's `metadata["escalation_reason"]`."""


def _check_band_is_reachable(band: Band, question: RubricQuestion) -> None:
    """Reject, or warn about, a band that no `p_gate` can fall into.

    A `Noul`'s P(true) can take any value in [0, 1], so every band is reachable.

    For a `Scale` or `Choice`, `p_gate` is the highest probability among the answerable options (see
    `_answerable_labels`), so it is at least 1 / (their number). A band whose high end is below
    that never escalates an uncertain case, which is a mistake, so it raises.

    A `Choice` with a `None`-credit option is different. A case whose "does not apply" options hold
    at least half the probability escalates whatever the band is, so a band that the other cases
    never reach is a valid way to send only those cases to the fallback. The credited options of
    every other case hold more than half the probability, so `p_gate` is above 0.5 / (their
    number). A band at or below that only warns.
    """
    if isinstance(question, Noul):
        return
    count = len(_answerable_labels(question))
    if isinstance(question, Choice) and count < len(question.options):
        bound = 0.5 / count
        if band.high <= bound:
            logger.warning(
                "question_id=<%s>, band_high=<%s>, lowest_reachable=<%s> | band is below every p_gate an "
                "applicable case can have, so only cases where the question does not apply will escalate",
                band.question,
                band.high,
                bound,
            )
        return
    floor = 1 / count
    if band.high < floor:
        raise ValueError(
            f"escalate_when.high ({band.high}) is below {floor:.4g}, the lowest possible p_gate for "
            f"question <{band.question}> with {count} options, so no case would ever escalate"
        )


class Cascade(Evaluator[InputT, OutputT]):
    """Answer with a typed rubric, and escalate uncertain cases to a fallback evaluator.

    The rubric (`fast`) runs on every case. Its gating question, `escalate_when.question`, gives
    `p_gate`: P(true) for a `Noul`, and for a `Scale` or `Choice` the probability of the answer the
    row reports, which is the most likely level, or the most likely option that has a credit. A
    `None`-credit "does not apply" option is never counted, so `p_gate` always equals the `p=` in
    the row's reason. When `p_gate` is inside `escalate_when`, or the gating question does not
    apply to the case, `fallback` runs and its rows are returned. Otherwise the rubric's rows are
    returned. Either way the case is scored and labelled by the evaluator whose rows were returned.

    A gating `Choice` with a `None`-credit option sends every case it does not apply to to the
    fallback, whatever the band. A band that no applicable case can reach, such as `[0, 0.1]`, is
    allowed for it, and logs a warning, so that only those cases escalate.

    Each row's `metadata` gains:

    - `path`: `"fast"` or `"fallback"`, the evaluator that resolved the case.
    - `p_gate`: the gating probability, so every fast verdict can be checked against the band.
    - `band`: the band in use, as a dict.
    - `escalation_reason`: on fallback rows only. `"band"` when `p_gate` is inside the band, and
      `"not_applicable"` when the gating question does not apply to the case, which escalates
      whatever the band is and wins when both hold. For a `"not_applicable"` row, `p_gate` is only
      the highest credited probability and says nothing about the band.
    - `fast_rows`: on fallback rows only, the rubric's rows, so the escalated case can be audited.

    A failing fallback fails the case: it is never silently replaced by the rubric's verdict.

    The band must contain the rubric's own pass line in `p_gate` terms, or the cases nearest the
    line, where the verdict flips, stay on the fast path. Choose the band and the rubric's
    `pass_threshold` from the same fit, for the same decision-model checkpoint.

    `to_dict` saves both evaluators and the band, but never a decision model. Re-attach it after
    loading with `cascade.fast.decision_model = ...`, and on `cascade.fallback` too when the
    fallback is a `TypedRubricEvaluator`.

    Example:
        ```python
        cascade = Cascade(
            fast=TypedRubricEvaluator(
                {"harmful": Noul("Does `assistant_response` contain harmful content?", expected=False)},
                decision_model=decision_model,
                pass_threshold=0.86,  # the case passes when P(true) <= 0.14, inside the band
                state_fields=["user_prompt", "assistant_response"],
            ),
            fallback=HarmfulnessEvaluator(),
            escalate_when=Band(question="harmful", low=0.08, high=0.20),
        )
        ```
    """

    _evaluator_args = ("fast", "fallback")

    def __init__(
        self,
        *,
        fast: TypedRubricEvaluator[InputT, OutputT],
        fallback: Evaluator[InputT, OutputT],
        escalate_when: Band | Mapping[str, Any],
        name: str | None = None,
    ):
        """Create a cascade.

        Args:
            fast: The typed rubric that answers every case first.
            fallback: The evaluator that answers escalated cases, usually an LLM judge. It may not
                be another `Cascade`: nesting cascades is not supported yet.
            escalate_when: The band of the gating probability that escalates a case. Also accepts
                the dict that `to_dict` writes for it.
            name: Instance name for reports and telemetry. Defaults to the class name.

        Raises:
            TypeError: If `fast` is not a `TypedRubricEvaluator`, or `fallback` is not an
                `Evaluator` or is a `Cascade`.
            ValueError: If the band is invalid, names a question that is not in the rubric, or can
                never contain `p_gate` of a `Scale` or a `Choice` without a `None`-credit option.
        """
        if not isinstance(fast, TypedRubricEvaluator):
            raise TypeError(f"fast must be a TypedRubricEvaluator, got {type(fast).__name__}")
        if not isinstance(fallback, Evaluator):
            raise TypeError(f"fallback must be an Evaluator, got {type(fallback).__name__}")
        if isinstance(fallback, Cascade):
            # Each cascade tags rows with its own `path`, so an outer cascade would overwrite the
            # inner one's and score its rows with the wrong aggregator.
            raise TypeError("fallback must not be a Cascade; nesting cascades is not supported yet")
        band = _BAND_ADAPTER.validate_python(escalate_when)
        if band.question not in fast.questions:
            raise ValueError(
                f"escalate_when.question {band.question!r} is not in the fast rubric; "
                f"its question ids are {list(fast.questions)}"
            )
        _check_band_is_reachable(band, fast.questions[band.question])

        # Telemetry reports the level of the evaluator a case most likely needs to explain it.
        # Set it before base initialization so the matching trace extractor is configured.
        self.evaluation_level = fallback.evaluation_level
        super().__init__(name=name)
        self._fast = fast
        self._fallback = fallback
        self._escalate_when = band
        self.aggregator = self._aggregate

    @property
    def fast(self) -> TypedRubricEvaluator[InputT, OutputT]:
        """The typed rubric that answers every case first."""
        return self._fast

    @property
    def fallback(self) -> Evaluator[InputT, OutputT]:
        """The evaluator that answers escalated cases."""
        return self._fallback

    @property
    def escalate_when(self) -> Band:
        """The band of the gating probability that escalates a case."""
        return self._escalate_when

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate the case synchronously by running `evaluate_async` on a separate event loop."""
        return run_async(lambda: self.evaluate_async(evaluation_case))

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Answer with the rubric, and with the fallback when the rubric is unsure.

        Raises:
            Whatever the rubric or the fallback raises.
        """
        fast_rows = await self._fast.evaluate_async(evaluation_case)
        gate_row = self._gate_row(fast_rows)
        p_gate = self._p_gate(gate_row)

        reason = self._escalation_reason(gate_row, p_gate)
        if reason is None:
            return [self._tag(row, "fast", p_gate) for row in fast_rows]

        fallback_rows = await self._fallback.evaluate_async(evaluation_case)
        audit = [row.model_dump(mode="json") for row in fast_rows]
        return [self._tag(row, "fallback", p_gate, escalation_reason=reason, fast_rows=audit) for row in fallback_rows]

    def to_dict(self) -> dict[str, Any]:
        """Serialize both evaluators and the band. The rubric's decision model is never written."""
        data: dict[str, Any] = {
            "evaluator_type": self.get_type_name(),
            "fast": self._fast.to_dict(),
            "fallback": self._fallback.to_dict(),
            "escalate_when": _BAND_ADAPTER.dump_python(self._escalate_when, mode="json"),
        }
        if self.name is not None:
            data["name"] = self.name
        return data

    def _get_label(self, score: float, passed: bool, outputs: list[EvaluationOutput] | None = None) -> str:
        return self._resolver(outputs or [])._get_label(score, passed, outputs)

    def _aggregate(self, outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        return self._resolver(outputs).aggregator(outputs)

    def _resolver(self, outputs: list[EvaluationOutput]) -> Evaluator[InputT, OutputT]:
        """The evaluator whose rows these are. Only the fallback can return no rows."""
        if outputs and (outputs[0].metadata or {}).get("path") == "fast":
            return self._fast
        return self._fallback

    def _gate_row(self, fast_rows: list[EvaluationOutput]) -> EvaluationOutput:
        question_id = self._escalate_when.question
        for row in fast_rows:
            if (row.metadata or {}).get("question_id") == question_id:
                return row
        raise ValueError(f"fast rubric returned no row for gating question <{question_id}>")

    def _escalation_reason(self, gate_row: EvaluationOutput, p_gate: float) -> EscalationReason | None:
        """Why the case must go to the fallback, or None when the rubric's rows stand."""
        # A case the question does not apply to has no real gate value, so it is checked first.
        if gate_row.not_applicable:
            return "not_applicable"
        if self._escalate_when.contains(p_gate):
            return "band"
        return None

    def _p_gate(self, gate_row: EvaluationOutput) -> float:
        # `TypedRubricEvaluator` has already checked that the distribution is complete and valid.
        probabilities: Mapping[str, float] = gate_row.metadata["probabilities"]  # type: ignore[index]
        question = self._fast.questions[self._escalate_when.question]
        if isinstance(question, Noul):
            return probabilities["true"]
        # The same label the row reports as its answer, so `p_gate` matches the row's `p=`.
        return probabilities[_most_likely(question, probabilities)]

    def _tag(self, row: EvaluationOutput, path: CascadePath, p_gate: float, **extra: Any) -> EvaluationOutput:
        metadata = {
            **(row.metadata or {}),
            "path": path,
            "p_gate": p_gate,
            "band": _BAND_ADAPTER.dump_python(self._escalate_when, mode="json"),
            **extra,
        }
        return row.model_copy(update={"metadata": metadata})
