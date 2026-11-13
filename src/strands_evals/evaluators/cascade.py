"""Route each case between a fast typed rubric and a fallback evaluator.

The rubric answers first. When its gating question's probability falls inside the escalation band,
the rubric is unsure, so the case goes to the fallback evaluator, usually an LLM judge, and keeps
the fallback's verdict and written reasoning. Every other case keeps the rubric's rows.
"""

from collections.abc import Mapping
from typing import Any, Literal

from .._async import run_async
from ..types.cascade import _BAND_ADAPTER, Band
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.typed_rubric import Noul, RubricQuestion, Scale
from .evaluator import Evaluator
from .typed_rubric_evaluator import TypedRubricEvaluator

Path = Literal["fast", "fallback"]
"""Which evaluator resolved a case, as written to each row's `metadata["path"]`."""


def _check_band_is_reachable(band: Band, question: RubricQuestion) -> None:
    """Reject a band that no `p_gate` can fall into, since it would never escalate a case.

    For a `Scale` or `Choice`, `p_gate` is the most likely option's probability, which is never
    below 1 / (number of options). A `Noul`'s P(true) can take any value in [0, 1].
    """
    if isinstance(question, Noul):
        return
    option_count = len(question.levels) if isinstance(question, Scale) else len(question.options)
    floor = 1 / option_count
    if band.high < floor:
        raise ValueError(
            f"escalate_when.high ({band.high}) is below {floor:.4g}, the lowest possible p_gate for "
            f"question <{band.question}> with {option_count} options, so no case would ever escalate"
        )


class Cascade(Evaluator[InputT, OutputT]):
    """Answer with a typed rubric, and escalate uncertain cases to a fallback evaluator.

    The rubric (`fast`) runs on every case. Its gating question, `escalate_when.question`, gives
    `p_gate`: P(true) for a `Noul`, the most likely option's probability for a `Scale` or
    `Choice`. When `p_gate` is inside `escalate_when`, or the gating question does not apply to
    the case, `fallback` runs and its rows are returned. Otherwise the rubric's rows are returned.
    Either way the case is scored and labelled by the evaluator whose rows were returned.

    Each row's `metadata` gains:

    - `path`: `"fast"` or `"fallback"`, the evaluator that resolved the case.
    - `p_gate`: the gating probability, so every fast verdict can be checked against the band.
    - `band`: the band in use, as a dict.
    - `fast_rows`: on fallback rows only, the rubric's rows, so the escalated case can be audited.

    A failing fallback fails the case: it is never silently replaced by the rubric's verdict.

    `to_dict` saves both evaluators and the band, but never the rubric's decision model. Re-attach
    it after loading with `cascade.fast.decision_model = ...`.

    Example:
        ```python
        cascade = Cascade(
            fast=TypedRubricEvaluator(
                {"harmful": Noul("Does `assistant_response` contain harmful content?", expected=False)},
                decision_model=decision_model,
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
        fast: TypedRubricEvaluator[InputT, OutputT],
        fallback: Evaluator[InputT, OutputT],
        escalate_when: Band | Mapping[str, Any],
        *,
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
                never contain `p_gate` for that question.
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

        super().__init__(name=name)
        self._fast = fast
        self._fallback = fallback
        self._escalate_when = band
        # Telemetry reports the level of the evaluator a case most likely needs to explain it.
        self.evaluation_level = fallback.evaluation_level
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

        if gate_row.not_applicable or self._escalate_when.contains(p_gate):
            fallback_rows = await self._fallback.evaluate_async(evaluation_case)
            audit = [row.model_dump(mode="json") for row in fast_rows]
            return [self._tag(row, "fallback", p_gate, fast_rows=audit) for row in fallback_rows]
        return [self._tag(row, "fast", p_gate) for row in fast_rows]

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

    def _p_gate(self, gate_row: EvaluationOutput) -> float:
        # `TypedRubricEvaluator` has already checked that the distribution is complete and valid.
        probabilities: Mapping[str, float] = gate_row.metadata["probabilities"]  # type: ignore[index]
        if isinstance(self._fast.questions[self._escalate_when.question], Noul):
            return probabilities["true"]
        return max(probabilities.values())

    def _tag(self, row: EvaluationOutput, path: Path, p_gate: float, **extra: Any) -> EvaluationOutput:
        metadata = {
            **(row.metadata or {}),
            "path": path,
            "p_gate": p_gate,
            "band": _BAND_ADAPTER.dump_python(self._escalate_when, mode="json"),
            **extra,
        }
        return row.model_copy(update={"metadata": metadata})
