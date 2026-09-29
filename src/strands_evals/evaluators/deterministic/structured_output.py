"""Field-level scoring of structured output against ground truth.

`Equals` compares structured output with whole-object `==`, so it scores 0.0 or 1.0 and
names nothing. `StructuredOutputSimilarity` compares field by field, with type-aware
comparators and order-independent list matching, so the score reflects how wrong the
output is and the reason names the field to fix. Deterministic and offline.

Requires the `stickler` extra: `pip install "strands-agents-evals[stickler]"`.
"""

import importlib
import logging
from typing import Any, Iterator, Mapping

from pydantic import BaseModel, ValidationError
from strands.agent.agent_result import AgentResult

from ...types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.evaluation_report import EvaluationReport
from ..evaluator import Evaluator

logger = logging.getLogger(__name__)


def _stickler() -> tuple[Any, Any]:
    """Import stickler on first use, so `import strands_evals` does not pay for it.

    Returns:
        The `eval_for` and `aggregate_from_comparisons` callables.

    Raises:
        ImportError: If the `stickler` extra is not installed.
    """
    try:
        # optional extra: stickler-eval
        from stickler import aggregate_from_comparisons, eval_for
    except ImportError as exc:
        raise ImportError(
            "StructuredOutputSimilarity requires the 'stickler-eval' package. Install "
            'it with: pip install "strands-agents-evals[stickler]"'
        ) from exc
    return eval_for, aggregate_from_comparisons


def _resolve_model_cls(model_cls: type[BaseModel] | str) -> type[BaseModel]:
    """Accept a model class, or the dotted path `to_dict` writes for one.

    Raises:
        TypeError: If the value does not resolve to a Pydantic model class.
    """
    if isinstance(model_cls, str):
        model_cls = _import_dotted_path(model_cls)
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError(f"model_cls must be a Pydantic model class; got {model_cls!r}")
    return model_cls


def _import_dotted_path(path: str) -> Any:
    """Resolve `module.QualName`, where a nested class makes the qualname itself dotted.

    Each split is tried from the longest module prefix down, since the module boundary
    cannot be found by splitting at the last dot.

    Raises:
        ImportError: If the module exists but fails to import, e.g. a missing dependency
            inside it. That is the real cause, so it is not reported as a bad path.
        TypeError: If no split resolves.
    """
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            resolved: Any = importlib.import_module(module_name)
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            if missing is not None and missing != module_name and not module_name.startswith(f"{missing}."):
                raise
            continue
        for attribute in parts[split:]:
            resolved = getattr(resolved, attribute, None)
            if resolved is None:
                break
        else:
            return resolved
    raise TypeError(
        f"model_cls string must be a dotted path to an importable model class, "
        f"such as 'myapp.models.Invoice'; could not resolve {path!r}"
    )


def _passed(result: Any, match_threshold: float) -> bool:
    """Whether a case passes: the score clears the bar AND the populated fields were found.

    The score alone is not enough. It credits a field blank on both sides as a match, so on
    a sparse schema a prediction that found nothing can clear the bar (10 optional fields,
    2 populated, empty prediction: score 0.80, recall 0.00). Recall only counts fields that
    had a value, which is what stickler recommends gating on for sparse schemas.

    "Something to find" is `tp + fn + fd`: stickler counts a wrong value as `fd`, not `fn`.
    When there is nothing to find, the case passes unless the prediction invented values.
    With something to find, invented extras are not penalized; `precision` records them.
    """
    if not bool(result.matched):
        return False
    # stickler's result shape rather than a documented accessor; pinned by `<2.0.0`.
    overall = (result.raw or {}).get("confusion_matrix", {}).get("overall", {})
    findable = overall.get("tp", 0) + overall.get("fn", 0) + overall.get("fd", 0)
    if findable == 0:
        return overall.get("fa", 0) == 0
    return result.recall is not None and result.recall >= match_threshold


def _rows(report: EvaluationReport, evaluator: str | None) -> Iterator[tuple[Mapping[str, Any], EvaluationOutput]]:
    """Yield `(case, row)` pairs from a report, for one evaluator when `evaluator` is set."""
    for index, rows in enumerate(report.detailed_results):
        case: Mapping[str, Any] = report.cases[index] if index < len(report.cases) else {}
        if evaluator is not None and case.get("evaluator") != evaluator:
            continue
        for row in rows:
            yield case, row


class StructuredOutputSimilarity(Evaluator[InputT, OutputT]):
    """Scores structured output against ground truth, field by field.

    Comparison configuration is inferred from the Pydantic model the agent already passes
    as `structured_output_model`; `explain` shows every inferred choice. Each case yields
    one row: the weighted score on `score`, and on `metadata` the per-field scores,
    `precision`, `recall`, `f1` and the raw comparison. Because the detail lives in the
    report, `metrics` and `per_case` read a report, including one reloaded from JSON.

    Args:
        model_cls: The Pydantic model the agent emits, or the dotted path `to_dict` writes.
        match_threshold: Similarity at which an object counts as a match. Drives list
            element matching, and bounds both halves of `test_pass`: the score (wrong
            values) and recall (missing values). At 0.7 an agent may leave up to 30% of
            the populated fields empty. For separate policies, gate on `metadata["recall"]`.
        weight_hints: Weight fields by name-based heuristics (ids and amounts count more).
            Off by default, so weights stay uniform.
        name: Evaluator name, forwarded to `Evaluator`; what `metrics(evaluator=...)`
            filters on.

    Raises:
        ImportError: If the `stickler` extra is not installed.
        TypeError: If `model_cls` does not resolve to a Pydantic model class.
    """

    def __init__(
        self,
        model_cls: type[BaseModel] | str,
        *,
        match_threshold: float = 0.7,
        weight_hints: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize the evaluator. See the class docstring for arguments."""
        super().__init__(name=name)
        eval_for, _ = _stickler()
        self._model_cls = _resolve_model_cls(model_cls)
        self.match_threshold = match_threshold
        self.weight_hints = weight_hints
        # Built once here, so a model stickler cannot handle fails at construction.
        self._spec = eval_for(
            self._model_cls,
            match_threshold=match_threshold,
            weight_hints=weight_hints,
        )

    @property
    def model_cls(self) -> type[BaseModel]:
        """The Pydantic model every case is compared as."""
        return self._model_cls

    def to_dict(self) -> dict:
        """Convert the evaluator into a dictionary, with `model_cls` as a dotted path.

        A model no other process can import (defined in `__main__` or inside a function)
        is still written, with a warning, as `OutputEvaluator.to_dict` does for tools.

        Returns:
            dict: The evaluator's information.
        """
        _dict = super().to_dict()
        path = f"{self._model_cls.__module__}.{self._model_cls.__qualname__}"
        if self._model_cls.__module__ == "__main__" or "<locals>" in self._model_cls.__qualname__:
            logger.warning(
                "model_cls=<%s> | not importable by dotted path, so from_file() will not reload this "
                "evaluator, move the model to an importable module to make the experiment portable",
                path,
            )
        _dict["model_cls"] = path
        return _dict

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Compare one case's actual output against its expected output.

        Args:
            evaluation_case: The case to score.

        Returns:
            One row. `NOT_APPLICABLE` when there is no ground truth or it belongs to
            another schema; score 0 with the reason when the output will not validate.
        """
        name = self._model_cls.__name__
        if evaluation_case.expected_output is None:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=True,
                    reason="expected_output is None, so there was no ground truth to compare against",
                    label=NOT_APPLICABLE,
                )
            ]

        dropped = self._foreign_fields(evaluation_case.expected_output)
        if dropped:
            # Converting would drop these fields, and an all-optional model would then
            # compare blank against blank and score a perfect 1.0.
            other = type(evaluation_case.expected_output).__name__
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=True,
                    reason=(
                        f"expected_output is {other}, whose fields {', '.join(dropped)} are not "
                        f"on {name}, so there was nothing for this evaluator to judge"
                    ),
                    label=NOT_APPLICABLE,
                )
            ]

        try:
            expected = self._coerce(evaluation_case.expected_output, "expected_output")
            actual = self._coerce(evaluation_case.actual_output, "actual_output")
        except (ValidationError, TypeError) as exc:
            logger.debug("case=<%s>, model=<%s> | output did not validate", evaluation_case.name, name)
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"could not compare as {name}: {exc}",
                    label=name,
                    metadata={"error": "validation_failed"},
                )
            ]

        result = self._spec.evaluate(expected, actual)
        passed = _passed(result, self.match_threshold)

        logger.debug(
            "case=<%s>, model=<%s>, score=<%.4f>, matched=<%s> | scored structured output",
            evaluation_case.name,
            name,
            result.overall_score,
            result.matched,
        )

        return [
            EvaluationOutput(
                score=result.overall_score,
                test_pass=passed,
                reason=self._weakest(result.field_scores),
                label=name,
                metadata={
                    "field_scores": dict(result.field_scores),
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "matched": result.matched,
                    # `prediction_raw` is the bulkiest part and does not affect field_metrics.
                    "comparison": {k: v for k, v in result.raw.items() if k != "prediction_raw"},
                },
            )
        ]

    @staticmethod
    def metrics(report: EvaluationReport, *, evaluator: str | None = None) -> Any:
        """Per-field metrics across every case in a report.

        A nested path only counts documents whose parent item scored at or above
        `match_threshold`; below it, the item counts as one wrong unit under the parent
        path. Cases whose output did not validate carry no comparison and are skipped.

        Args:
            report: The report to aggregate, in memory or reloaded from JSON.
            evaluator: Evaluator name to read, for a report carrying more than one.

        Returns:
            stickler's `ProcessEvaluation`. Its `field_metrics` is keyed by dotted path
            (`line_items.sku`), with tp/tn/fn/fa/fd counts, precision, recall, F1 and
            accuracy.

        Raises:
            ValueError: If the rows come from more than one evaluator and `evaluator` is
                not set. Merging two schemas would union their field paths.
        """
        _, aggregate_from_comparisons = _stickler()
        scored = [
            (case, row.metadata["comparison"])
            for case, row in _rows(report, evaluator)
            if row.metadata and "comparison" in row.metadata
        ]
        names = sorted({str(case.get("evaluator")) for case, _ in scored})
        if len(names) > 1:
            raise ValueError(
                f"report carries comparisons from {len(names)} evaluators ({', '.join(names)}); "
                f"pass evaluator=<name> to aggregate one schema at a time"
            )
        return aggregate_from_comparisons([comparison for _, comparison in scored])

    @staticmethod
    def per_case(report: EvaluationReport, *, evaluator: str | None = None) -> list[Mapping[str, Any]]:
        """Per-document field scores in case order, as a flat table.

        Nested list children have no per-leaf score, so they appear in `metrics` only.
        Cases whose output did not validate are omitted.

        Args:
            report: The report to read.
            evaluator: Evaluator name to read, for a report carrying more than one.

        Returns:
            One entry per scored case: `case`, `model`, `overall_score`, `test_pass`,
            `matched`, `precision`, `recall`, `f1` and `field_scores`. `matched` is
            stickler's score-only verdict; `test_pass` also requires recall.
        """
        return [
            {
                "case": case.get("name"),
                "model": row.label,
                "overall_score": row.score,
                "test_pass": row.test_pass,
                "matched": row.metadata["matched"],
                "precision": row.metadata["precision"],
                "recall": row.metadata["recall"],
                "f1": row.metadata["f1"],
                "field_scores": dict(row.metadata["field_scores"]),
            }
            for case, row in _rows(report, evaluator)
            if row.metadata and "field_scores" in row.metadata
        ]

    def explain(self) -> dict[str, dict[str, Any]]:
        """The comparator, threshold and weight chosen for each field path, and why.

        Returns:
            One entry per dotted field path.
        """
        return self._spec.explain()

    def _foreign_fields(self, value: Any) -> list[str]:
        """Fields a model of another class has that `model_cls` lacks.

        Empty when nothing would be lost converting, including the same class defined
        twice (a re-run notebook cell), so the test is on fields, not class identity.
        """
        if not isinstance(value, BaseModel) or isinstance(value, self._model_cls):
            return []
        return sorted(set(type(value).model_fields) - set(self._model_cls.model_fields))

    def _coerce(self, value: Any, which: str) -> BaseModel:
        """Accept a model instance, an `AgentResult`, a dict, or a JSON string."""
        cls = self._model_cls
        if isinstance(value, AgentResult):
            if value.structured_output is None:
                raise TypeError(
                    f"{which} is an AgentResult with no structured_output; call the agent with "
                    f"structured_output_model={cls.__name__}"
                )
            value = value.structured_output
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            dropped = self._foreign_fields(value)
            if dropped:
                raise TypeError(
                    f"{which} is {type(value).__name__}, whose fields {', '.join(dropped)} are not on {cls.__name__}"
                )
            return cls.model_validate(value.model_dump())
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, str):
            return cls.model_validate_json(value)
        raise TypeError(
            f"{which} must be a {cls.__name__} instance, an AgentResult, a dict, or a JSON string; "
            f"got {type(value).__name__}"
        )

    @staticmethod
    def _weakest(field_scores: Mapping[str, float], limit: int = 4) -> str:
        """Name the weakest fields, so a low score says what to fix."""
        imperfect = sorted(
            ((name, score) for name, score in field_scores.items() if score < 1.0),
            key=lambda pair: pair[1],
        )
        if not imperfect:
            return "all fields matched"
        listed = "; ".join(f"{name}={score:.2f}" for name, score in imperfect[:limit])
        if len(imperfect) > limit:
            listed += f"; (+{len(imperfect) - limit} more)"
        return f"weakest fields: {listed}"
