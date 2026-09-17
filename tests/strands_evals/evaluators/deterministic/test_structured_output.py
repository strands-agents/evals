"""StructuredOutput scores structured output field by field, via stickler.

The tests that matter here cover claims the harness itself does not enforce: that
the case score is stickler's weighted score rather than a mean, that per-field
detail is available without a second comparison pass, that a mixed-schema suite
does not produce a merged rollup, and that the evaluator is safe at the harness's
default concurrency.
"""

import asyncio
import inspect
import threading

import pytest
from pydantic import BaseModel, Field, ValidationError

pytest.importorskip("stickler", reason="requires the stickler extra")

from strands_evals import Case, Experiment  # noqa: E402
from strands_evals.evaluators import Equals, Evaluator, StructuredOutput  # noqa: E402
from strands_evals.types.evaluation import EvaluationData  # noqa: E402


class LineItem(BaseModel):
    sku: str | None = None
    unit_price: float | None = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class Receipt(BaseModel):
    merchant: str
    tax: float


def _invoice(iid="INV-1", vendor="Acme Corporation", total=100.0, sku="SKU-1", price=100.0):
    return Invoice(
        invoice_id=iid,
        vendor_name=vendor,
        total_amount=total,
        line_items=[LineItem(sku=sku, unit_price=price)],
    )


def _case(name, expected):
    return Case(name=name, input="", expected_output=expected, metadata={"name": name})


def _run(evaluator, pairs, **kwargs):
    """Run pairs of (expected, actual) through the real harness."""
    cases = [_case(name, exp) for name, (exp, _) in pairs.items()]
    actual = {name: act for name, (_, act) in pairs.items()}
    return asyncio.run(
        Experiment(cases=cases, evaluators=[evaluator]).run_evaluations_async(
            lambda c: actual[c.metadata["name"]], **kwargs
        )
    )


def _data(expected, actual):
    return EvaluationData(input="", invocation_input="", expected_output=expected, actual_output=actual)


class TestPerCaseOutput:
    """One output per case, carrying stickler's weighted score."""

    def test_one_output_per_case(self):
        evaluator = StructuredOutput(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice()))

        assert len(outputs) == 1
        assert outputs[0].label == "Invoice"

    def test_score_is_sticklers_weighted_overall(self):
        """Not a mean of per-field scores; the weighted score stickler computed.

        An earlier draft emitted one output per field and installed a custom
        aggregator to recombine them, which required looking weights up by field
        name because EvaluationOutput carries none. Reporting overall_score
        directly removes that whole mechanism.
        """
        evaluator = StructuredOutput(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec_for(Invoice).evaluate(gt, pred).overall_score
        outputs = evaluator.evaluate(_data(gt, pred))

        assert outputs[0].score == pytest.approx(expected)

    def test_reason_names_the_weakest_fields(self):
        evaluator = StructuredOutput(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(iid="INV-9")))

        assert "invoice_id" in (outputs[0].reason or "")

    def test_no_custom_aggregator_is_installed(self):
        """The framework default is correct for a single output, so leave it.

        Overriding it was only necessary while several outputs per case had to be
        recombined.
        """
        evaluator = StructuredOutput(Invoice)

        assert evaluator.aggregator is Evaluator._default_aggregator


class TestPerCaseDetail:
    """Field detail comes from the evaluator, not from EvaluationOutput."""

    def test_per_case_carries_field_scores(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"doc-a": (_invoice(), _invoice(vendor="Acme Corp"))})

        (entry,) = evaluator.per_case()
        assert entry["case"] == "doc-a"
        assert entry["model"] == "Invoice"
        assert set(entry["field_scores"]) == set(Invoice.model_fields)

    def test_per_case_preserves_the_case_name(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {f"doc-{i}": (_invoice(), _invoice()) for i in range(3)})

        assert {e["case"] for e in evaluator.per_case()} == {"doc-0", "doc-1", "doc-2"}

    def test_per_case_runs_no_extra_comparisons(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})

        before = len(evaluator._results)
        evaluator.per_case()
        evaluator.per_case()
        assert len(evaluator._results) == before

    def test_per_case_is_empty_after_reset(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})
        evaluator.reset()

        assert evaluator.per_case() == []


class TestCaseScore:
    def test_case_score_equals_stickler_overall_score(self):
        evaluator = StructuredOutput(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec_for(Invoice).evaluate(gt, pred).overall_score
        report = _run(evaluator, {"a": (gt, pred)})

        assert report.scores[0] == pytest.approx(expected)

    def test_perfect_match_says_so(self):
        evaluator = StructuredOutput(Invoice)
        report = _run(evaluator, {"a": (_invoice(), _invoice())})

        assert report.scores[0] == pytest.approx(1.0)
        assert report.test_passes[0] is True
        assert report.reasons[0] == "all fields matched"


class TestBeatsEquals:
    """The reason this integration exists."""

    def test_equals_cannot_rank_what_stickler_separates(self):
        pairs = {
            "near": (_invoice(), _invoice(vendor="Acme Corp")),
            "far": (_invoice(), _invoice(iid="X", vendor="Zeta", total=9.0, sku="Z")),
        }
        stickler_report = _run(StructuredOutput(Invoice), pairs)
        equals_report = _run(Equals(), pairs)

        assert len(set(equals_report.scores)) == 1  # both wrong, indistinguishable
        assert stickler_report.scores[0] > stickler_report.scores[1]


class TestDatasetRollup:
    def test_metrics_includes_nested_paths(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})

        paths = evaluator.metrics()["Invoice"].field_metrics
        assert "line_items" in paths
        assert "line_items.sku" in paths

    def test_metrics_runs_no_extra_comparisons(self):
        """One comparison per case, feeding both the outputs and the rollup."""
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {f"c{i}": (_invoice(), _invoice()) for i in range(5)})

        assert len(evaluator._results) == 5
        evaluator.metrics()
        evaluator.metrics()
        assert len(evaluator._results) == 5

    def test_five_categories_are_populated(self):
        """FN is a missed field, FA an invented one, FD a wrong one."""
        gt = _invoice(total=100.0)
        missing = _invoice(total=None)  # FN on total_amount
        wrong = _invoice(total=55.0)  # FD on total_amount

        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"missing": (gt, missing), "wrong": (gt, wrong)})

        total = evaluator.metrics()["Invoice"].field_metrics["total_amount"]
        assert total["fn"] == 1
        assert total["fd"] == 1

    def test_reset_clears_the_rollup(self):
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})
        assert evaluator.metrics()

        evaluator.reset()
        assert evaluator.metrics() == {}


class TestMixedSchemas:
    def test_rollup_is_partitioned_by_class(self):
        """A merged rollup would union field paths and misreport denominators."""
        evaluator = StructuredOutput()  # no declared model_cls
        pairs = {
            "inv": (_invoice(), _invoice()),
            "rec": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=2.0)),
        }
        _run(evaluator, pairs)
        metrics = evaluator.metrics()

        assert set(metrics) == {"Invoice", "Receipt"}
        assert set(metrics["Receipt"].field_metrics) == {"merchant", "tax"}
        assert "merchant" not in metrics["Invoice"].field_metrics
        assert all(pe.document_count == 1 for pe in metrics.values())

    def test_declared_model_cls_rejects_a_foreign_shape(self):
        """Strict mode must fail loudly rather than coerce nonsense."""
        evaluator = StructuredOutput(Invoice)

        with pytest.raises(ValidationError):
            evaluator.evaluate(_data(Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0)))

    def test_explain_is_ambiguous_across_inferred_schemas(self):
        evaluator = StructuredOutput()
        _run(
            evaluator,
            {
                "inv": (_invoice(), _invoice()),
                "rec": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0)),
            },
        )

        with pytest.raises(RuntimeError, match="ambiguous"):
            evaluator.explain()

    def test_a_narrower_actual_does_not_score_perfectly(self):
        """Ground truth defines the fields, not the agent's output.

        Inferring the schema from `actual_output` let an agent returning a model
        with fewer fields drop the missing ones from the comparison entirely and
        score 1.0 -- a perfect result for output that omitted a field, which is
        the exact failure this evaluator exists to catch. Resolving from
        `expected_output` instead makes the coercion raise.
        """

        class Narrower(BaseModel):
            invoice_id: str

        evaluator = StructuredOutput()  # inferred
        with pytest.raises(ValidationError):
            evaluator.evaluate(_data(_invoice(), Narrower(invoice_id="INV-1")))

    def test_schema_is_resolved_from_expected_not_actual(self):
        evaluator = StructuredOutput()
        evaluator.evaluate(_data(_invoice(), _invoice()))

        assert evaluator.per_case()[0]["model"] == "Invoice"

    def test_classes_sharing_a_name_get_separate_rollups(self):
        """Two modules each defining `Invoice` is ordinary.

        Keying the rollup on `__name__` alone collapsed them, and because it was
        built by a dict comprehension the later class *overwrote* the earlier
        rather than merging it, losing those documents silently. Same-module
        classes built with `type()` share `__module__` and `__qualname__` too, so
        qualifying alone is not enough either.
        """

        def make(fields):
            return type("Invoice", (BaseModel,), {"__annotations__": fields})

        first, second = make({"a": str}), make({"b": str})
        evaluator = StructuredOutput()
        _run(
            evaluator,
            {
                "one": (first(a="x"), first(a="x")),
                "two": (second(b="y"), second(b="ZZ")),
            },
        )

        metrics = evaluator.metrics()
        assert len(metrics) == 2, f"rollups collapsed: {list(metrics)}"
        assert all(pe.document_count == 1 for pe in metrics.values())
        assert {tuple(sorted(pe.field_metrics)) for pe in metrics.values()} == {("a",), ("b",)}

    def test_a_unique_class_name_stays_unqualified(self):
        """Disambiguation must not make the common case ugly."""
        evaluator = StructuredOutput(Invoice)
        _run(evaluator, {"a": (_invoice(), _invoice())})

        assert list(evaluator.metrics()) == ["Invoice"]
        assert evaluator.per_case()[0]["model"] == "Invoice"

    def test_inference_needs_a_model_instance(self):
        evaluator = StructuredOutput()

        with pytest.raises(TypeError, match="could not infer"):
            evaluator.evaluate(_data({"invoice_id": "X"}, {"invoice_id": "X"}))


class TestConcurrency:
    """The accumulator has to survive the harness running evaluate() on threads.

    `Evaluator.evaluate_async` is `await asyncio.to_thread(self.evaluate, ...)`, and
    `run_evaluations_async` defaults to `max_workers=10`, so anything accumulated
    across cases is shared mutable state on many threads. `list.append` is atomic
    under the GIL and needs no lock; a read-modify-write is not guaranteed to be.
    That is a correctness argument rather than a measured one -- see
    `test_the_accumulator_is_append_only`.
    """

    def test_evaluate_really_runs_on_worker_threads(self):
        """The premise of everything below.

        If `evaluate_async` were overridden with a synchronous body, the base
        class's `to_thread` offload would be bypassed, every case would run inline
        on the event loop, and the retention test below would be single-threaded and
        therefore vacuous. Asserting the thread count is what keeps that from
        happening silently.
        """
        seen: set[str] = set()

        class Probe(StructuredOutput):
            def evaluate(self, evaluation_case):
                seen.add(threading.current_thread().name)
                return super().evaluate(evaluation_case)

        n = 40
        _run(
            Probe(Invoice),
            {f"c{i}": (_invoice(), _invoice()) for i in range(n)},
            max_workers=10,
        )

        assert seen, "evaluate() never ran"
        assert seen != {"MainThread"}, (
            "evaluate() ran only on the main thread, so the harness's to_thread "
            "offload was bypassed -- is evaluate_async overridden?"
        )

    def test_no_results_are_lost_at_the_harness_default(self):
        """Every case is retained when the harness runs them concurrently.

        A retention check, not a proof against every possible race: a
        read-modify-write mutant of the append survives this test on CPython 3.12,
        so it cannot stand in for the correctness argument in the class docstring.
        """
        n = 200
        evaluator = StructuredOutput(Invoice)
        report = _run(
            evaluator,
            {f"c{i}": (_invoice(), _invoice()) for i in range(n)},
            max_workers=10,
        )

        assert len(report.scores) == n
        assert len(evaluator._results) == n
        assert evaluator.metrics()["Invoice"].document_count == n

    def test_the_accumulator_is_append_only(self):
        """Pins the property the docstring rests on, since a race test cannot.

        `evaluate` must only ever `append`. Rebinding `_results`, or mutating it
        through a read-modify-write, is the shape that is not guaranteed atomic --
        and measurement will not catch it reliably, so the structure is asserted
        instead.
        """
        source = inspect.getsource(StructuredOutput.evaluate)
        assert "self._results.append(" in source
        assert "self._results =" not in source


class TestCoercion:
    @pytest.mark.parametrize(
        "actual",
        [
            {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
            '{"invoice_id": "INV-1", "vendor_name": "Acme Corporation"}',
        ],
        ids=["dict", "json-string"],
    )
    def test_accepts_dicts_and_json_strings(self, actual):
        evaluator = StructuredOutput(Invoice)
        outputs = evaluator.evaluate(_data(Invoice(invoice_id="INV-1", vendor_name="Acme Corporation"), actual))

        assert outputs[0].score == pytest.approx(1.0)

    def test_rejects_an_unusable_type(self):
        evaluator = StructuredOutput(Invoice)

        with pytest.raises(TypeError, match="instance, dict, or JSON string"):
            evaluator.evaluate(_data(_invoice(), 42))


class TestExplain:
    def test_covers_nested_paths(self):
        config = StructuredOutput(Invoice).explain()

        assert "line_items.sku" in config
        assert config["invoice_id"]["comparator"] == "ExactComparator"

    def test_list_of_models_is_not_reported_as_a_string_comparator(self):
        """A List[StructuredModel] has no single comparator."""
        config = StructuredOutput(Invoice).explain()

        assert config["line_items"]["comparator"] != "LevenshteinComparator"

    def test_needs_a_model(self):
        with pytest.raises(RuntimeError, match="needs a model"):
            StructuredOutput().explain()


class SparseInvoice(BaseModel):
    """The common extraction shape: two fields that matter, a long optional tail.

    `overall_score` credits a field absent on both sides with 1.0 at full weight, so
    the tail dominates the mean. These fields exist to make that measurable.
    """

    invoice_id: str
    vendor_name: str
    invoice_date: str | None = None
    total_amount: float | None = None
    tax_amount: float | None = None
    po_number: str | None = None
    payment_terms: str | None = None
    shipping_address: str | None = None
    billing_address: str | None = None
    notes: str | None = None


class TestPassRequiresFindingSomething:
    """`test_pass` gates on recall as well as the score.

    Every test here fails against a `test_pass = result.matched` implementation, and
    against stickler 0.7.0 where `matched` meant all-fields-AND. They are what makes
    the `stickler-eval>=1.0.0` floor in pyproject.toml mean something.
    """

    def test_a_prediction_that_found_nothing_does_not_pass(self):
        """The defect this gate exists for.

        Ground truth has 2 of 10 fields; the agent returns nothing. The 8 fields blank
        on both sides each score 1.0, so the mean is 0.80 and clears the 0.7 default
        while recall is 0.00. Passing here would tell a CI gate that an agent which
        extracted nothing is fine.
        """
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        evaluator = StructuredOutput(SparseInvoice)
        outputs = evaluator.evaluate(_data(gt, blank))

        assert outputs[0].score > 0.7, "the score really does clear the threshold"
        assert outputs[0].test_pass is False, "but nothing was found, so it must not pass"

        entry = evaluator.per_case()[0]
        assert entry["recall"] == 0.0
        assert entry["matched"] is True, "stickler's score-only verdict still says match"
        assert entry["test_pass"] is False, "the evaluator's verdict disagrees, correctly"

    def test_widening_the_optional_tail_does_not_buy_a_pass(self):
        """Severity scales with schema width, so a higher threshold cannot fix it.

        The blank-prediction score is the blank fraction of the schema, which tends to
        1.0 as the optional tail grows. Only a recall gate is stable under that.
        """
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        for threshold in (0.7, 0.75, 0.8):
            evaluator = StructuredOutput(SparseInvoice, match_threshold=threshold)
            outputs = evaluator.evaluate(_data(gt, blank))
            assert outputs[0].test_pass is False, f"blank passed at threshold {threshold}"

    def test_a_good_extraction_still_passes(self):
        """The gate must not cost a correct sparse extraction its pass."""
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")

        evaluator = StructuredOutput(SparseInvoice)
        outputs = evaluator.evaluate(_data(gt, gt.model_copy()))

        assert outputs[0].test_pass is True
        assert evaluator.per_case()[0]["recall"] == 1.0

    def test_nothing_to_find_is_a_pass_not_a_failure(self):
        """A document whose ground truth is legitimately blank.

        stickler reports recall as 0.0 rather than None when the denominator is empty,
        so a naive recall gate would fail this. The guard reads `tp + fn == 0` instead.
        """

        class AllOptional(BaseModel):
            a: str | None = None
            b: str | None = None

        evaluator = StructuredOutput(AllOptional)
        outputs = evaluator.evaluate(_data(AllOptional(), AllOptional()))

        assert outputs[0].score == 1.0
        assert outputs[0].test_pass is True

    def test_per_case_exposes_the_metrics_the_verdict_rests_on(self):
        """recall/precision/f1 come from the retained comparison, not a second pass."""
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        evaluator = StructuredOutput(SparseInvoice)
        evaluator.evaluate(_data(gt, gt.model_copy()))

        entry = evaluator.per_case()[0]
        for key in ("test_pass", "matched", "precision", "recall", "f1"):
            assert key in entry, f"per_case() should expose {key}"
