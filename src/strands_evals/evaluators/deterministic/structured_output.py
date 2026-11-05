"""Field-level scoring of structured output against ground truth.

`Equals` compares structured output with whole-object `==`, so it scores 0.0 or 1.0 and
names nothing. `StructuredOutputSimilarity` compares field by field, with type-aware
comparators and order-independent list matching, so the score reflects how wrong the
output is and the reason names the field to fix. Deterministic and offline.

Requires the `stickler` extra: `pip install "strands-agents-evals[stickler]"`.
"""

import importlib
import json
import logging
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import VERSION as PYDANTIC_VERSION
from pydantic import AliasChoices, AliasPath, BaseModel, RootModel, ValidationError
from strands.agent.agent_result import AgentResult

from ...types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.evaluation_report import EvaluationReport
from ..evaluator import Evaluator

if TYPE_CHECKING:
    from stickler.utils.process_evaluation import ProcessEvaluation

logger = logging.getLogger(__name__)

# Marks the rows this evaluator scored. `metadata` is open to every evaluator, so rows are
# selected on this value rather than on key names another evaluator may also write.
_MARKER = "StructuredOutputSimilarity"
# The field detail on a marked row. A marked row carries all of it, or it is corrupted.
_DETAIL_KEYS = frozenset({"field_scores", "matched", "precision", "recall", "f1", "comparison"})


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
        TypeError: If the value does not resolve to a Pydantic model class with named fields.
    """
    if isinstance(model_cls, str):
        model_cls = _import_dotted_path(model_cls)
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError(f"model_cls must be a Pydantic model class; got {model_cls!r}")
    if issubclass(model_cls, RootModel) or not model_cls.model_fields:
        # Values are recognized by their field names, and a RootModel stores its value under
        # none: its cached form has no `root` key, so it could not be read back.
        raise TypeError(f"model_cls must have named fields to compare; {model_cls.__name__} has none")
    return model_cls


def _accepted_keys(model_cls: type[BaseModel]) -> frozenset[str]:
    """Every top-level key pydantic reads a field of `model_cls` from, validating by name.

    Names always count, since values are validated with `by_name=True`. A validation alias
    replaces the alias for reading, and neither counts when the model sets
    `validate_by_alias=False`.
    """
    by_alias = model_cls.model_config.get("validate_by_alias", True)
    keys: set[str] = set()
    for name, info in model_cls.model_fields.items():
        keys.add(name)
        if not by_alias:
            continue
        alias = info.validation_alias if info.validation_alias is not None else info.alias
        for choice in alias.choices if isinstance(alias, AliasChoices) else [alias]:
            if isinstance(choice, str):
                keys.add(choice)
            elif isinstance(choice, AliasPath) and choice.path and isinstance(choice.path[0], str):
                keys.add(choice.path[0])
    return frozenset(keys)


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


class StructuredOutputReport(EvaluationReport):
    """An `EvaluationReport` that can roll up `StructuredOutputSimilarity`'s field detail.

    The rollups are methods rather than functions because the detail they read is on the
    report: each row's `metadata`. Bind the report to the experiment and `run_evaluations`
    returns it, with no cast:

        report = Experiment(
            cases=cases,
            evaluators=[StructuredOutputSimilarity(Invoice)],
            report_cls=StructuredOutputReport,
        ).run_evaluations(task)

        report.metrics()    # per-field confusion matrix
        report.per_case()   # per-document field scores

    Reading detail off the report rather than off the evaluator is what makes both survive
    `flatten` and `model_dump_json`. Nothing binds this class automatically: the CLI builds
    its own `Experiment`, and `Experiment.to_dict` keeps only cases and evaluators, so both
    write the base class. Reload their JSON through this class to roll it up.

        StructuredOutputReport.from_file("report.json").metrics()
    """

    def metrics(self, *, evaluator: str | None = None) -> "ProcessEvaluation":
        """Per-field metrics across every scored case in the report.

        A nested path only counts documents whose parent item scored at or above
        `match_threshold`; below it, the item counts as one wrong unit under the parent
        path. Rows without field detail are skipped: values that could not be read as the
        model, cases with no ground truth, and rows from other evaluators.

        Args:
            evaluator: Evaluator name to read, for a report carrying more than one.

        Returns:
            stickler's `ProcessEvaluation`. Its `field_metrics` is keyed by dotted path
            (`line_items.sku`), with `tp`/`tn`/`fn`/`fa`/`fp`/`fd` counts and `cm_precision`,
            `cm_recall`, `cm_f1` and `cm_accuracy`.

        Raises:
            ValueError: If the scored rows compare more than one model, or come from more
                than one evaluator with `evaluator` unset, since merging would union their
                field paths. Also if there is nothing to roll up; see `per_case`.
        """
        _, aggregate_from_comparisons = _stickler()
        scored = self._scored_rows(evaluator)
        names = sorted({str(case.get("evaluator")) for case, _, _ in scored})
        if len(names) > 1:
            raise ValueError(
                f"report carries comparisons from {len(names)} evaluators ({', '.join(names)}); "
                f"pass evaluator=<name> to aggregate one schema at a time"
            )
        models = sorted({str(row.label) for _, row, _ in scored})
        if len(models) > 1:
            raise ValueError(
                f"report compares {len(models)} models ({', '.join(models)}) under one evaluator name; "
                f"call metrics() on each report before flattening them, or give the evaluators distinct names"
            )
        return aggregate_from_comparisons([detail["comparison"] for _, _, detail in scored])

    def per_case(self, *, evaluator: str | None = None) -> list[dict[str, Any]]:
        """Per-document field scores in case order, as a flat table.

        Nested list children have no per-leaf score, so they appear in `metrics` only.
        Rows without field detail are omitted, as in `metrics`.

        Args:
            evaluator: Evaluator name to read, for a report carrying more than one.

        Returns:
            One entry per scored case: `case`, `evaluator`, `model`, `overall_score`, `test_pass`,
            `matched`, `precision`, `recall`, `f1` and `field_scores`. `matched` is
            stickler's score-only verdict; `test_pass` also requires recall.

        Raises:
            ValueError: If `evaluator` names no evaluator in the report, if a row carries
                only part of the field detail, or if no row was scored at all.
        """
        return [
            {
                "case": case.get("name"),
                "evaluator": case.get("evaluator"),
                "model": row.label,
                "overall_score": row.score,
                "test_pass": row.test_pass,
                "matched": detail["matched"],
                "precision": detail["precision"],
                "recall": detail["recall"],
                "f1": detail["f1"],
                "field_scores": dict(detail["field_scores"]),
            }
            for case, row, detail in self._scored_rows(evaluator)
        ]

    def _scored_rows(self, evaluator: str | None) -> list[tuple[Mapping[str, Any], EvaluationOutput, dict[str, Any]]]:
        """The `(case, row, detail)` triples for scored rows, for one evaluator when `evaluator` is set.

        The single rule both rollups select by, so they never disagree about which rows count.
        It returns rows or raises: an empty result would read the same as a clean one.

        Raises:
            ValueError: If `evaluator` names no evaluator in the report, if a row carries only
                part of the field detail, or if nothing was scored.
        """
        if evaluator is not None:
            present = {str(case["evaluator"]) for case in self.cases if case.get("evaluator") is not None}
            if evaluator not in present:
                raise ValueError(
                    f"no rows from evaluator {evaluator!r}; this report carries: "
                    f"{', '.join(sorted(present)) or '(none)'}"
                )
        scored: list[tuple[Mapping[str, Any], EvaluationOutput, dict[str, Any]]] = []
        for index, rows in enumerate(self.detailed_results):
            case: Mapping[str, Any] = self.cases[index] if index < len(self.cases) else {}
            if evaluator is not None and str(case.get("evaluator")) != evaluator:
                continue
            for row in rows:
                detail = row.metadata or {}
                if detail.get("evaluated_by") != _MARKER:
                    continue
                missing = _DETAIL_KEYS - detail.keys()
                if missing:
                    raise ValueError(
                        f"row for case {case.get('name')!r} is missing field detail: {', '.join(sorted(missing))}"
                    )
                scored.append((case, row, detail))
        if not scored:
            source = f"evaluator {evaluator!r}" if evaluator is not None else "this report"
            raise ValueError(
                f"{source} has no scored structured-output rows: every row is from another evaluator, "
                f"had no ground truth, or could not be read as the model"
            )
        return scored


class StructuredOutputSimilarity(Evaluator[InputT, OutputT]):
    """Scores structured output against ground truth, field by field.

    Comparison configuration is inferred from the Pydantic model the agent already passes
    as `structured_output_model`; `explain` shows every inferred choice. Each case yields
    one row: the weighted score on `score`, and on `metadata` the per-field scores,
    `precision`, `recall`, `f1`, `matched`, the raw comparison, and `evaluated_by`, which
    marks the row as this evaluator's. A case that cannot be compared scores 0 with
    `metadata={"error": "unreadable"}` and the reason. Because the detail lives in the
    report, the rollups are methods on `StructuredOutputReport` — pass that as the
    experiment's `report_cls` to get them.

    Both sides of a case are read as `model_cls`. An instance is compared as it is, and an
    `AgentResult` by its `structured_output`. Anything else (a dict, a JSON string, or an
    instance of another class) is read by pydantic, as a result store or experiment file
    hands back the instances it was given: by field name or alias, lax about types, and
    ignoring keys the model does not define. A value with keys, none of which is a field
    of `model_cls`, is another schema and fails, so a case gets the same result in memory
    and read back from a file.

    Args:
        model_cls: The Pydantic model the agent emits, or the dotted path `to_dict` writes.
        match_threshold: Similarity at which an object counts as a match. Drives list
            element matching, and bounds both halves of `test_pass`: the score (wrong
            values) and recall (missing values). At 0.7 an agent may leave up to 30% of
            the populated fields empty. For separate policies, gate on `metadata["recall"]`.
        weight_hints: Weight fields by name-based heuristics (ids and amounts count more).
            Off by default, so weights stay uniform.
        name: Evaluator name, forwarded to `Evaluator`; what
            `StructuredOutputReport.metrics(evaluator=...)` filters on.

    Raises:
        ImportError: If the `stickler` extra is not installed.
        TypeError: If `model_cls` does not resolve to a Pydantic model class with named fields.
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
        if tuple(int(part) for part in PYDANTIC_VERSION.split(".")[:2]) < (2, 12):
            raise ImportError(
                f"StructuredOutputSimilarity requires pydantic>=2.12 (found {PYDANTIC_VERSION}). Install it "
                'with: pip install "strands-agents-evals[stickler]"'
            )
        self._model_cls = _resolve_model_cls(model_cls)
        self._accepted_keys = _accepted_keys(self._model_cls)
        # Read-only: `_spec` is built from both, so changing one later would split the verdict.
        self._match_threshold = match_threshold
        self._weight_hints = weight_hints
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

    @property
    def match_threshold(self) -> float:
        """Similarity at which an object counts as a match; see the class docstring."""
        return self._match_threshold

    @property
    def weight_hints(self) -> bool:
        """Whether fields are weighted by name-based heuristics."""
        return self._weight_hints

    def to_dict(self) -> dict:
        """Convert the evaluator into a dictionary, with `model_cls` as a dotted path.

        A model no other process can import (defined in `__main__` or inside a function)
        is still written, with a warning, since `from_file` then fails with the path named.

        Returns:
            dict: The evaluator's information.
        """
        _dict = super().to_dict()
        path = f"{self._model_cls.__module__}.{self._model_cls.__qualname__}"
        if self._model_cls.__module__ == "__main__" or "<locals>" in self._model_cls.__qualname__:
            logger.warning(
                "model_cls=<%s> | model is not importable by dotted path | move it to an importable "
                "module so the experiment file can reload this evaluator",
                path,
            )
        _dict["model_cls"] = path
        _dict["match_threshold"] = self._match_threshold
        _dict["weight_hints"] = self._weight_hints
        return _dict

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Compare one case's actual output against its expected output.

        Args:
            evaluation_case: The case to score.

        Returns:
            One row. `NOT_APPLICABLE` when there is no ground truth; score 0 with the reason
            when either side cannot be read as `model_cls`.
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

        try:
            expected = self._read(evaluation_case.expected_output, "expected_output")
            actual = self._read(evaluation_case.actual_output, "actual_output")
        except (ValueError, TypeError) as exc:
            logger.debug("case=<%s>, model=<%s> | could not read case as model", evaluation_case.name, name)
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"could not compare as {name}: {exc}",
                    label=name,
                    metadata={"error": "unreadable"},
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
                    "evaluated_by": _MARKER,
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

    def explain(self) -> dict[str, dict[str, Any]]:
        """The comparator, threshold and weight chosen for each field path, and why.

        Returns:
            One entry per dotted field path.
        """
        return self._spec.explain()

    def _read(self, value: Any, which: str) -> BaseModel:
        """Read one side of a case as `model_cls`; see the class docstring for the rules.

        Another model is read from its JSON form, the form a result store hands back, so it
        reads the same both ways.

        Raises:
            TypeError: If the value is not a form this evaluator reads, or is another schema.
            ValueError: If it is a string that is not JSON, or does not validate.
        """
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
        form = type(value).__name__
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        elif isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{which} is a string that is not JSON: {exc}") from exc
            form = f"JSON {type(value).__name__}"
        if not isinstance(value, dict):
            raise TypeError(
                f"{which} must be an instance of {cls.__name__}, an AgentResult, a dict, or a JSON object; got {form}"
            )
        if value and not {str(key) for key in value} & self._accepted_keys:
            # Validated, it would be a blank document, and blank against blank scores 1.0.
            keys = ", ".join(sorted(map(str, value)))
            raise TypeError(f"{which} ({form}) has keys {keys}, none of which is a field of {cls.__name__}")
        try:
            return cls.model_validate(value, strict=False, extra="ignore", by_name=True)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or '(root)'}: {error['msg']}"
                for error in exc.errors(include_url=False)
            )
            raise ValueError(f"{which} did not validate: {errors}") from exc

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
