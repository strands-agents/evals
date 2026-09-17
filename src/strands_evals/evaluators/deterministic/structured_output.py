"""Field-level comparison of structured output against ground truth.

`Equals` compares structured output with whole-object `==`, scoring 0.0 or 1.0. An
extraction that gets nine of ten fields right is indistinguishable from one that
gets none right, and a reordered list counts as wrong. `StructuredOutput` compares
field by field with type-aware comparators, order-independent list matching and
per-field thresholds, so the score reflects how wrong the output is and names which
field to fix.

Deterministic and offline: no LLM judge, no credentials, no per-call cost.

Requires the `stickler` extra:

```bash
pip install "strands-agents-evals[stickler]"
```
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator

logger = logging.getLogger(__name__)

# optional extra: stickler-eval
try:
    from stickler import aggregate_from_comparisons, eval_for

    _STICKLER_AVAILABLE = True
    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised by the extras gate
    _STICKLER_AVAILABLE = False
    _IMPORT_ERROR = exc


@dataclass(frozen=True)
class _CaseResult:
    """What one comparison produced, kept for reading after the run.

    `EvaluationOutput` carries four scalars, so it cannot express a per-field
    breakdown. Keeping the comparison here instead means `per_case` can return the
    real numbers rather than a reconstruction.
    """

    name: str | None
    model_cls: type
    overall_score: float
    matched: bool
    test_pass: bool
    # Computed by stickler alongside the score. Unlike `overall_score` these ignore
    # correct absence, so they stay meaningful on a sparse schema; `recall` is the
    # one `test_pass` gates on.
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    field_scores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _passed(result: Any, match_threshold: float) -> bool:
    """Whether a case passes: the score cleared the bar AND the real values were found.

    `result.matched` alone is not enough. It is `overall_score >= match_threshold`, and
    `overall_score` credits a field absent on *both* sides with 1.0 at full weight -- a
    value the model correctly left blank is a value it got right. On a sparse extraction
    schema, which is the common case, those uninformative fields outvote the informative
    ones, so a prediction that found *nothing* can clear the threshold:

        10 optional fields, 2 populated in ground truth, prediction returns nothing
        -> overall_score 0.80, matched True, recall 0.00

    and it gets worse as the schema widens, so raising `match_threshold` cannot fix it:
    the threshold required tends to 1.0 as the optional tail grows.

    Recall closes it. Recall counts only fields that had a value to find, so correct
    absence neither helps nor hurts it, and it reads 0.0 for a blank extraction. This is
    what stickler's own guidance recommends gating on for sparse schemas:
    https://awslabs.github.io/stickler/Getting-Started/thresholds-and-metrics/#sparse-objects

    When there was nothing to find at all, the score verdict stands on its own -- two
    genuinely empty objects are a match, not a failure. That case is detected as
    `tp + fn == 0` rather than by a null recall: stickler reports recall as 0.0, not
    None, when the denominator is empty, so gating on recall alone would fail a document
    whose ground truth was legitimately blank.
    """
    if not bool(result.matched):
        return False
    overall = (result.raw or {}).get("confusion_matrix", {}).get("overall", {})
    findable = overall.get("tp", 0) + overall.get("fn", 0)
    if findable == 0:
        return True
    return result.recall is not None and result.recall >= match_threshold


def _distinct_names(classes: Mapping[type, Any]) -> dict[type, str]:
    """Map each class to a display name, unique within this set.

    Keyed on `__name__` normally, because that is what a caller wants to read. Two
    distinct classes can share a name, though -- two modules each defining an
    `Invoice` is ordinary -- and a plain `__name__` key silently collapses them,
    with the later one overwriting the earlier rollup entirely rather than merging
    it.

    Colliding names fall back to `module.QualName`, then to a numeric suffix. The
    second fallback is not paranoia: classes built with `type()` in one module,
    which is how dynamic models are made, share both `__module__` and
    `__qualname__`, so qualifying alone does not separate them.

    Args:
        classes: The classes to name, used as a set via its keys.

    Returns:
        A mapping from class to display name, unique across the input.
    """
    ordered = list(classes)

    def bucket(key_of: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cls in ordered:
            counts[key_of(cls)] = counts.get(key_of(cls), 0) + 1
        return counts

    plain = bucket(lambda c: c.__name__)
    qualified = bucket(lambda c: f"{c.__module__}.{c.__qualname__}")

    names: dict[type, str] = {}
    seen: dict[str, int] = {}
    for cls in ordered:
        if plain[cls.__name__] == 1:
            name = cls.__name__
        else:
            name = f"{cls.__module__}.{cls.__qualname__}"
            if qualified[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                name = f"{name}#{seen[name]}"
        names[cls] = name
    return names


def _require_stickler() -> None:
    """Raise a directed ImportError when the optional extra is missing."""
    if not _STICKLER_AVAILABLE:
        raise ImportError(
            "StructuredOutput requires the 'stickler-eval' package. Install it "
            'with: pip install "strands-agents-evals[stickler]"'
        ) from _IMPORT_ERROR


class StructuredOutput(Evaluator[InputT, OutputT]):
    """Scores structured output against ground truth, field by field.

    Comparison configuration is inferred from the Pydantic model itself -- the same
    model the agent already passes as `structured_output_model` -- so no schema or
    annotation work is required. Every inferred decision is inspectable via
    `explain`.

    Returns one `EvaluationOutput` per case, carrying the weighted overall score.
    Field-level detail is read from the evaluator rather than squeezed through that
    type, which has only four scalar fields: `per_case` for per-document field
    scores, `metrics` for the dataset-wide confusion matrix. Both read comparisons
    already performed, so neither runs anything twice.

    Args:
        model_cls: The Pydantic model the agent emits. Optional. When given, every
            case is coerced to it and anything that will not validate raises, which
            is what a single-schema suite wants. When omitted, the class is inferred
            per case and `metrics` partitions its rollup by class, so a suite mixing
            output types stays readable.
        match_threshold: Similarity at or above which an object counts as a match.
            Drives the per-element matching of `list[Model]` fields, and one half of
            `test_pass`. A case passes when the weighted score clears this AND recall
            does, because the score alone credits fields absent on both sides and so
            would pass a prediction that found nothing on a sparse schema. See
            `_passed`.
        weight_hints: When True, weight fields by name-based importance heuristics
            (ids and amounts count for more). Off by default so weights stay uniform
            and metrics are not skewed by guessed business criticality.
        name: Optional evaluator name, forwarded to `Evaluator`.

    Raises:
        ImportError: If the `stickler` extra is not installed.
    """

    def __init__(
        self,
        model_cls: type[BaseModel] | None = None,
        *,
        match_threshold: float = 0.7,
        weight_hints: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize the evaluator. See the class docstring for argument detail."""
        _require_stickler()
        super().__init__(name=name)
        self.declared_cls = model_cls
        self.match_threshold = match_threshold
        self.weight_hints = weight_hints

        self._specs: dict[type, Any] = {}
        # One _CaseResult per evaluated case. Appended to and never read during a
        # run. `list.append` is atomic under the GIL, so this needs no lock even
        # though the harness calls evaluate() from up to `max_workers` threads via
        # asyncio.to_thread (default 10). Everything is read afterwards, on one
        # thread, by metrics() and per_case().
        #
        # Append-only is a correctness argument, not a measured one: a
        # read-modify-write mutant of this line did NOT lose results across 1000
        # cases at 32 workers on CPython 3.12, so a race test cannot stand in for
        # it. `test_the_accumulator_is_append_only` pins the structure instead.
        #
        # `evaluate_async` is deliberately NOT overridden. The base class's is
        # `await asyncio.to_thread(self.evaluate, ...)`, which is what puts this on
        # worker threads at all; a synchronous override would run the comparison
        # inline on the event loop, making the sentence above false and the
        # concurrency test vacuous, and would block every other case's task on a
        # CPU-bound Hungarian match.
        self._results: list[_CaseResult] = []

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Compare one case's actual output against its expected output.

        Args:
            evaluation_case: The case to score.

        Returns:
            A single-element list carrying the weighted score for the case.
        """
        cls = self._resolve_cls(evaluation_case)
        spec = self._spec_for(cls)

        expected = self._coerce(evaluation_case.expected_output, cls, "expected_output")
        actual = self._coerce(evaluation_case.actual_output, cls, "actual_output")

        result = spec.evaluate(expected, actual)

        # `prediction_raw` is dropped: only the confidence accumulators consume it,
        # and they need `field_comparisons` alongside it. Keeping it without them
        # makes aggregate_from_comparisons warn on every call, and it is the
        # bulkiest part of the result. field_metrics is identical without it.
        passed = _passed(result, self.match_threshold)
        self._results.append(
            _CaseResult(
                name=evaluation_case.name,
                model_cls=cls,
                overall_score=result.overall_score,
                matched=result.matched,
                test_pass=passed,
                precision=result.precision,
                recall=result.recall,
                f1=result.f1,
                field_scores=dict(result.field_scores),
                raw={k: v for k, v in result.raw.items() if k != "prediction_raw"},
            )
        )

        logger.debug(
            "case=<%s>, model=<%s>, score=<%.4f>, matched=<%s> | scored structured output",
            evaluation_case.name,
            cls.__name__,
            result.overall_score,
            result.matched,
        )

        return [
            EvaluationOutput(
                score=result.overall_score,
                # Not `result.matched` alone: that credits absent-on-both fields, so a
                # blank extraction passes on a sparse schema. See `_passed`.
                test_pass=passed,
                reason=self._weakest(result.field_scores),
                # The model class, so a mixed-schema run can be grouped by output
                # type. Two caveats worth knowing. `EvaluationOutput.label` is
                # documented as the categorical label for the score, and the
                # harness derives its own span label from the score independently,
                # so the two describe different things. And this is the raw
                # `__name__`, while `per_case` and `metrics` use the disambiguated
                # name: with two same-named classes the label cannot be joined
                # back to its rollup. The disambiguated name is not available here
                # because it depends on which classes appear across the whole run,
                # which is not known until the run ends.
                label=cls.__name__,
            )
        ]

    def metrics(self) -> dict[str, Any]:
        """Per-field metrics across every case evaluated so far.

        Keyed by model class name, so a suite mixing output types gets one rollup
        per type. Feeding two schemas into a single rollup unions their field paths,
        which makes a field present in half the documents read as missed in the rest.

        Each value carries a `field_metrics` mapping keyed by dotted path
        (`line_items.sku`) with the five-category counts (tp/tn/fn/fa/fd) plus
        precision, recall, F1 and accuracy.

        Read after the run. A nested path's counts only cover documents whose parent
        pair scored at or above `match_threshold`: below that, threshold gating
        treats the pair as atomic and emits no field breakdown, so a nested field
        has a smaller denominator than the document count.

        Returns:
            One rollup per model class, keyed by display name.
        """
        by_cls: dict[type, list[dict[str, Any]]] = {}
        for case in list(self._results):
            by_cls.setdefault(case.model_cls, []).append(case.raw)
        keys = _distinct_names(by_cls)
        return {keys[cls]: aggregate_from_comparisons(raws) for cls, raws in by_cls.items()}

    def per_case(self) -> list[Mapping[str, Any]]:
        """Per-document field scores, in the order the cases completed.

        `EvaluationOutput` has four scalar fields, so the harness's
        `report.detailed_results` can only ever echo what `evaluate` returned.
        Reading from the retained comparison instead means the real per-field
        numbers are available without squeezing them through that shape, and
        without a second comparison pass.

        Nested list children are absent from `field_scores` because no per-leaf
        score is emitted for them; use `metrics` for those, which reports their
        counts and precision/recall/F1.

        Do NOT positionally zip this with `report.scores`. That is index-ordered
        while this is completion-ordered, and the harness records a failed
        evaluator as a zero-score row without `evaluate` ever appending here, so
        the two can differ in length as well as order. Join on `case` instead.

        Returns:
            One entry per case with `case`, `model`, `overall_score`, `test_pass`,
            `matched`, `precision`, `recall`, `f1` and `field_scores`. On a sparse
            schema read `recall` or `f1` rather than `overall_score`: the score credits
            fields absent on both sides, those three do not.
        """
        keys = _distinct_names({case.model_cls: None for case in self._results})
        return [
            {
                "case": case.name,
                "model": keys[case.model_cls],
                "overall_score": case.overall_score,
                # `test_pass` is the verdict the harness reported. `matched` is
                # stickler's score-only view, kept because it is the number the
                # `overall_score` column is thresholded against; the two differ
                # exactly when correct absence carried a prediction that found
                # nothing. See `_passed`.
                "test_pass": case.test_pass,
                "matched": case.matched,
                "precision": case.precision,
                "recall": case.recall,
                "f1": case.f1,
                "field_scores": dict(case.field_scores),
            }
            for case in list(self._results)
        ]

    def reset(self) -> None:
        """Drop accumulated results.

        Evaluator instances are shared across cases and may be reused across
        experiments, so a stateful evaluator needs an explicit way to clear.

        Clears the inferred-spec cache as well as the results. Keeping the specs
        meant `explain()` still raised "ambiguous across 2 inferred schemas" after a
        reset, naming a class that contributed nothing to the current results.
        """
        self._results.clear()
        self._specs.clear()

    def explain(self) -> dict[str, dict[str, Any]]:
        """Per-field comparison config and why it was chosen.

        Keyed by dotted path, so nested decisions are auditable too.

        Returns:
            One entry per field path, carrying the comparator, threshold, weight and
            the provenance of each choice.

        Raises:
            RuntimeError: If no model is available, or if several inferred schemas
                make the answer ambiguous.
        """
        if self.declared_cls is not None:
            return self._spec_for(self.declared_cls).explain()
        if not self._specs:
            raise RuntimeError(
                "explain() needs a model: pass model_cls to the constructor, or evaluate at least one case first."
            )
        if len(self._specs) > 1:
            names = ", ".join(sorted(c.__name__ for c in self._specs))
            raise RuntimeError(
                f"explain() is ambiguous across {len(self._specs)} inferred schemas "
                f"({names}). Pass model_cls to select one."
            )
        return next(iter(self._specs.values())).explain()

    def _resolve_cls(self, evaluation_case: EvaluationData[InputT, OutputT]) -> type[BaseModel]:
        if self.declared_cls is not None:
            return self.declared_cls
        # expected_output first, deliberately. Ground truth defines the fields being
        # measured; the agent's output does not get to narrow that. If it is
        # inferred from `actual_output` instead, an agent returning a model with
        # fewer fields silently drops the missing ones from the comparison and
        # scores 1.0 -- a perfect result for output that omitted a field, which is
        # the exact failure this evaluator exists to catch. Preferring expected
        # makes `_coerce` raise on that input instead.
        for value in (evaluation_case.expected_output, evaluation_case.actual_output):
            if isinstance(value, BaseModel):
                return type(value)
        raise TypeError(
            f"{type(self).__name__} could not infer a model class from this case. Pass "
            f"model_cls to the constructor, or supply outputs as Pydantic model "
            f"instances rather than dicts or JSON strings."
        )

    def _spec_for(self, cls: type[BaseModel]) -> Any:
        spec = self._specs.get(cls)
        if spec is None:
            spec = eval_for(
                cls,
                match_threshold=self.match_threshold,
                weight_hints=self.weight_hints,
            )
            self._specs[cls] = spec
        return spec

    def _coerce(self, value: Any, cls: type[BaseModel], which: str) -> BaseModel:
        """Accept a model instance, a dict, or a JSON string."""
        if isinstance(value, cls):
            return value
        if isinstance(value, BaseModel):
            # A different model class carrying the same fields.
            return cls.model_validate(value.model_dump())
        if isinstance(value, dict):
            return cls.model_validate(value)
        if isinstance(value, str):
            return cls.model_validate_json(value)
        raise TypeError(
            f"{type(self).__name__} needs {which} as a {cls.__name__} instance, dict, "
            f"or JSON string; got {type(value).__name__}"
        )

    @staticmethod
    def _weakest(field_scores: Mapping[str, float], limit: int = 4) -> str:
        """Name the weakest fields so a low case score is actionable."""
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
