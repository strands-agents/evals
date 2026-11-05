"""StructuredOutputSimilarity scores structured output field by field, via stickler.

The tests that matter here cover claims the harness itself does not enforce: that the
case score is stickler's weighted score rather than a mean, that the field-level detail
rides on the report and therefore survives serialization, that a case whose output will
not validate still produces a row, and that `test_pass` gates on recall.

Skipped as a module when the `stickler` extra is absent, via `pytestmark`, which keeps
the imports at the top of the file where the house style wants them.
"""

import asyncio
import builtins
import datetime
import enum
import json
import logging
import tempfile
import threading
from decimal import Decimal
from importlib.util import find_spec
from pathlib import Path

import pytest
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    create_model,
    model_validator,
)
from pydantic.alias_generators import to_camel
from strands.agent.agent_result import AgentResult

import strands_evals
from strands_evals import Case, Experiment, StructuredOutputReport
from strands_evals import evaluators as evaluators_package
from strands_evals.evaluators import Equals, Evaluator, StructuredOutputSimilarity
from strands_evals.evaluators.deterministic import structured_output as structured_output_module
from strands_evals.local_file_task_result_store import LocalFileTaskResultStore
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport

pytestmark = pytest.mark.skipif(find_spec("stickler") is None, reason="requires the stickler extra")


class LineItem(BaseModel):
    sku: str | None = None
    unit_price: float | None = None


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float | None = None
    line_items: list[LineItem] = []


class Receipt(BaseModel):
    merchant: str
    tax: float


class Nested:
    """Holder for a nested model class, whose `__qualname__` is itself dotted."""

    class Inner(BaseModel):
        a: str


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
        Experiment(cases=cases, evaluators=[evaluator], report_cls=StructuredOutputReport).run_evaluations_async(
            lambda c: actual[c.metadata["name"]], **kwargs
        )
    )


def _data(expected, actual):
    return EvaluationData(input="", invocation_input="", expected_output=expected, actual_output=actual)


def _reload(report):
    """Round-trip a report through JSON, as `strands-evals run --output` does."""
    return StructuredOutputReport.model_validate_json(report.model_dump_json())


class TestPerCaseOutput:
    """One output per case, carrying stickler's weighted score."""

    def test_one_output_per_case(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice()))

        assert len(outputs) == 1
        assert outputs[0].label == "Invoice"

    def test_score_is_sticklers_weighted_overall(self):
        """Not a mean of per-field scores; the weighted score stickler computed."""
        evaluator = StructuredOutputSimilarity(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec.evaluate(gt, pred).overall_score
        outputs = evaluator.evaluate(_data(gt, pred))

        assert outputs[0].score == pytest.approx(expected)

    def test_reason_names_the_weakest_fields(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(iid="INV-9")))

        assert "invoice_id" in (outputs[0].reason or "")

    def test_no_custom_aggregator_is_installed(self):
        """The framework default is correct for a single output, so leave it."""
        evaluator = StructuredOutputSimilarity(Invoice)

        assert evaluator.aggregator is Evaluator._default_aggregator


class TestDetailRidesOnTheReport:
    """Field detail lives on the row, so it stays with the report.

    That makes it visible to `strands-evals run --output`, separate per run of one instance,
    and in case order.
    """

    def test_row_metadata_carries_field_scores(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(_invoice(), _invoice(vendor="Acme Corp")))

        metadata = outputs[0].metadata
        assert set(metadata["field_scores"]) == set(Invoice.model_fields)
        for key in ("precision", "recall", "f1", "matched", "comparison"):
            assert key in metadata, f"metadata should carry {key}"

    def test_per_case_reads_the_report(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"doc-a": (_invoice(), _invoice(vendor="Acme Corp"))})

        (entry,) = report.per_case()
        assert entry["case"] == "doc-a"
        assert entry["model"] == "Invoice"
        assert set(entry["field_scores"]) == set(Invoice.model_fields)

    def test_per_case_is_in_case_order(self):
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {f"doc-{i}": (_invoice(), _invoice()) for i in range(5)},
            max_workers=5,
        )

        assert [e["case"] for e in report.per_case()] == [f"doc-{i}" for i in range(5)]

    def test_detail_survives_serialization(self):
        """A saved report rolls up the same as the one in memory."""
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice(vendor="Acme Corp"))})
        reloaded = _reload(report)

        assert reloaded.per_case() == report.per_case()
        assert reloaded.metrics().field_metrics == report.metrics().field_metrics

    def test_reusing_one_instance_does_not_blend_runs(self):
        """Two runs of one instance roll up separately."""
        evaluator = StructuredOutputSimilarity(Invoice)
        first = _run(evaluator, {"a": (_invoice(), _invoice())})
        second = _run(evaluator, {"b": (_invoice(), _invoice())})

        assert first.metrics().document_count == 1
        assert second.metrics().document_count == 1
        assert [e["case"] for e in second.per_case()] == ["b"]


class TestBindingTheRollupReport:
    """The rollups are methods on `StructuredOutputReport`, bound via `report_cls`."""

    def test_run_evaluations_returns_the_rollup_report(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})

        assert type(report) is StructuredOutputReport

    def test_without_report_cls_the_report_has_no_rollups(self):
        """Not a defect to fix, the reason `from_file` is the documented route for a CLI run.

        `report_cls` is not carried by `Experiment.to_dict`/`from_dict`, and the CLI builds its
        own `Experiment` without it, so an unbound run yields the base class.
        """
        cases = [_case("a", _invoice())]
        report = Experiment(cases=cases, evaluators=[StructuredOutputSimilarity(Invoice)]).run_evaluations(
            lambda c: _invoice()
        )

        assert type(report) is EvaluationReport

    def test_a_report_written_unbound_rolls_up_after_reload(self):
        """The `strands-evals run --output` path: plain report out, subclass reads it back.

        Report JSON does not depend on `report_cls`, so the rollups a bound run computes in
        memory are the same ones a reload computes from the file.
        """
        pairs = {"a": (_invoice(), _invoice(vendor="Acme Corp")), "b": (_invoice(), _invoice(total=55.0))}
        bound = _run(StructuredOutputSimilarity(Invoice), pairs)

        cases = [_case(name, expected) for name, (expected, _) in pairs.items()]
        actual = {name: act for name, (_, act) in pairs.items()}
        unbound = Experiment(cases=cases, evaluators=[StructuredOutputSimilarity(Invoice)]).run_evaluations(
            lambda c: actual[c.metadata["name"]]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "report.json")
            unbound.to_file(path)
            reloaded = StructuredOutputReport.from_file(path)

        assert type(reloaded) is StructuredOutputReport
        assert reloaded.per_case() == bound.per_case()
        assert reloaded.metrics().field_metrics == bound.metrics().field_metrics


class TestCaseScore:
    def test_case_score_equals_stickler_overall_score(self):
        evaluator = StructuredOutputSimilarity(Invoice, weight_hints=True)
        gt, pred = _invoice(), _invoice(iid="INV-9", vendor="Acme Corp")

        expected = evaluator._spec.evaluate(gt, pred).overall_score
        report = _run(evaluator, {"a": (gt, pred)})

        assert report.scores[0] == pytest.approx(expected)

    def test_perfect_match_says_so(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})

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
        stickler_report = _run(StructuredOutputSimilarity(Invoice), pairs)
        equals_report = _run(Equals(), pairs)

        assert len(set(equals_report.scores)) == 1  # both wrong, indistinguishable
        assert stickler_report.scores[0] > stickler_report.scores[1]


class TestDatasetRollup:
    def test_metrics_includes_nested_paths(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})

        paths = report.metrics().field_metrics
        assert "line_items" in paths
        assert "line_items.sku" in paths

    def test_metrics_counts_every_case(self):
        n = 5
        report = _run(StructuredOutputSimilarity(Invoice), {f"c{i}": (_invoice(), _invoice()) for i in range(n)})

        assert report.metrics().document_count == n

    def test_five_categories_are_populated(self):
        """FN is a missed field, FA an invented one, FD a wrong one."""
        gt = _invoice(total=100.0)
        missing = _invoice(total=None)  # FN on total_amount
        wrong = _invoice(total=55.0)  # FD on total_amount

        report = _run(StructuredOutputSimilarity(Invoice), {"missing": (gt, missing), "wrong": (gt, wrong)})

        total = report.metrics().field_metrics["total_amount"]
        assert total["fn"] == 1
        assert total["fd"] == 1

    def test_an_unfiltered_mixed_report_raises_rather_than_merging(self):
        """`report.metrics()` is the first thing anyone types, so it must not merge.

        Merging unions the field paths, so `document_count` counts the whole suite and each
        schema's fields read as absent across the other's documents. Two evaluators also
        means this is the `flatten` path, so it doubles as proof that `flatten` returns
        `report_cls` rather than the base class.
        """
        cases = [_case("inv", _invoice()), _case("rec", Receipt(merchant="M", tax=1.0))]
        actual = {"inv": _invoice(), "rec": Receipt(merchant="M", tax=1.0)}
        report = Experiment(
            cases=cases,
            evaluators=[
                StructuredOutputSimilarity(Invoice, name="inv"),
                StructuredOutputSimilarity(Receipt, name="rec"),
            ],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: actual[c.metadata["name"]])

        assert type(report) is StructuredOutputReport

        with pytest.raises(ValueError, match="pass evaluator="):
            report.metrics()

        assert report.metrics(evaluator="inv").document_count == 1

    def test_two_schemas_get_separate_rollups(self):
        """A merged rollup would union field paths and misreport denominators.

        The README says one `Experiment` per schema, so this is not a recommended layout.
        It is pinned because nothing stops a caller building it, and `evaluator=` has to
        keep the rollups apart when they do.
        """
        cases = [_case("inv", _invoice()), _case("rec", Receipt(merchant="M", tax=1.0))]
        actual = {"inv": _invoice(), "rec": Receipt(merchant="M", tax=2.0)}
        report = Experiment(
            cases=cases,
            evaluators=[
                StructuredOutputSimilarity(Invoice, name="invoice"),
                StructuredOutputSimilarity(Receipt, name="receipt"),
            ],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: actual[c.metadata["name"]])

        invoice = report.metrics(evaluator="invoice")
        receipt = report.metrics(evaluator="receipt")

        assert set(receipt.field_metrics) == {"merchant", "tax"}
        assert "merchant" not in invoice.field_metrics


class TestAskingForSomethingThatIsNotThere:
    """A rollup that finds nothing says so, instead of returning an empty that reads clean."""

    def test_an_unknown_evaluator_name_raises_rather_than_reading_empty(self):
        """An empty rollup is indistinguishable from a suite where nothing could be scored.

        `evaluator=` is a free-form string, so a typo is the likely way to arrive here, and
        a silent empty sends the caller looking for the fault in their data. The message
        lists the names the report does carry, as the mixed-report error does.
        """
        report = _run(StructuredOutputSimilarity(Invoice, name="inv"), {"a": (_invoice(), _invoice())})

        with pytest.raises(ValueError, match="no rows from evaluator 'Inv'; this report carries: inv"):
            report.metrics(evaluator="Inv")

        with pytest.raises(ValueError, match="no rows from evaluator 'Inv'"):
            report.per_case(evaluator="Inv")

    def test_a_row_carrying_partial_detail_raises_from_both_rollups(self):
        """`evaluate` writes all of the field detail or none, so a partial row is a corrupted report.

        Skipping it let the two rollups disagree: dropping one key emptied `per_case` while
        `metrics` still counted the row. Both now select through one rule, and raise.
        """
        report = _run(StructuredOutputSimilarity(Invoice), {"doc-a": (_invoice(), _invoice())})
        report.detailed_results[0][0].metadata.pop("matched")

        with pytest.raises(ValueError, match="row for case 'doc-a' is missing field detail: matched"):
            report.per_case()
        with pytest.raises(ValueError, match="row for case 'doc-a' is missing field detail: matched"):
            report.metrics()

    def test_naming_an_evaluator_without_field_detail_raises(self):
        """`Equals` is in the report but has nothing to roll up; an empty result would read clean."""
        cases = [_case("a", _invoice())]
        report = Experiment(
            cases=cases,
            evaluators=[StructuredOutputSimilarity(Invoice, name="inv"), Equals()],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: _invoice())

        with pytest.raises(ValueError, match="evaluator 'Equals' has no scored structured-output rows"):
            report.metrics(evaluator="Equals")
        with pytest.raises(ValueError, match="evaluator 'Equals' has no scored structured-output rows"):
            report.per_case(evaluator="Equals")

        assert report.metrics(evaluator="inv").document_count == 1

    def test_a_report_where_nothing_was_scored_raises(self):
        """Every prediction failing to validate is a result to report, not an empty rollup."""
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), "prose"), "b": (_invoice(), "prose")})

        with pytest.raises(ValueError, match="this report has no scored structured-output rows"):
            report.metrics()
        with pytest.raises(ValueError, match="this report has no scored structured-output rows"):
            report.per_case()

    def test_a_non_string_evaluator_tag_still_filters(self):
        """A hand-edited report can carry a non-string tag; filtering compares strings on both sides."""
        report = _run(StructuredOutputSimilarity(Invoice, name="1"), {"a": (_invoice(), _invoice())})
        report.cases[0]["evaluator"] = 1

        assert report.metrics(evaluator="1").document_count == 1
        assert len(report.per_case(evaluator="1")) == 1


class TestFlattenedReports:
    """`flatten` is the documented way to merge reports, so the schema guard has to hold across it."""

    def test_flattening_two_schemas_under_one_name_raises(self):
        """Two separately run reports share the default evaluator name, so a name check alone passed.

        `metrics()` then returned one rollup over both schemas' field paths. The guard also
        compares the model each row was scored as.
        """
        invoice = _run(StructuredOutputSimilarity(Invoice), {"inv": (_invoice(), _invoice())})
        receipt = _run(
            StructuredOutputSimilarity(Receipt),
            {"rec": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0))},
        )
        merged = StructuredOutputReport.flatten([invoice, receipt])

        assert type(merged) is StructuredOutputReport
        with pytest.raises(ValueError, match=r"compares 2 models \(Invoice, Receipt\)"):
            merged.metrics()
        assert [entry["model"] for entry in merged.per_case()] == ["Invoice", "Receipt"]

    def test_flattening_reports_of_one_schema_rolls_up(self):
        first = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), _invoice())})
        second = _run(StructuredOutputSimilarity(Invoice), {"b": (_invoice(), _invoice(vendor="Acme Corp"))})
        merged = StructuredOutputReport.flatten([first, second])

        assert merged.metrics().document_count == 2
        assert [entry["case"] for entry in merged.per_case()] == ["a", "b"]


class TestModelClassIsRequired:
    """The class is what reads a case back: `Experiment.from_file()` datasets and a warm
    `LocalFileTaskResultStore` both hand values back as dicts, which carry no class.
    """

    def test_model_cls_is_required(self):
        with pytest.raises(TypeError):
            StructuredOutputSimilarity()

    def test_rejects_a_non_model(self):
        with pytest.raises(TypeError, match="Pydantic model class"):
            StructuredOutputSimilarity(dict)

    def test_an_unsupported_model_fails_at_construction(self):
        """A model stickler cannot compare fails at construction, not on every case."""

        class Node(BaseModel):
            name: str
            child: "Node | None" = None

        Node.model_rebuild()

        with pytest.raises(TypeError, match="recursive"):
            StructuredOutputSimilarity(Node)


class TestSerialization:
    """The evaluator round-trips through `Experiment.to_file()` and `from_file()`."""

    def test_to_dict_emits_a_dotted_path(self):
        evaluator = StructuredOutputSimilarity(Invoice)

        assert evaluator.to_dict() == {
            "evaluator_type": "StructuredOutputSimilarity",
            "model_cls": f"{Invoice.__module__}.Invoice",
            "match_threshold": 0.7,
            "weight_hints": False,
        }

    def test_the_settings_the_comparison_is_built_from_are_read_only(self):
        """Changing one after construction would split the verdict from the saved settings."""
        evaluator = StructuredOutputSimilarity(Invoice, match_threshold=0.8, weight_hints=True)

        with pytest.raises(AttributeError):
            evaluator.match_threshold = 0.4
        with pytest.raises(AttributeError):
            evaluator.weight_hints = False
        assert (evaluator.match_threshold, evaluator.weight_hints) == (0.8, True)

    def test_round_trips_through_file(self, tmp_path):
        path = tmp_path / "experiment.json"
        evaluator = StructuredOutputSimilarity(Invoice, match_threshold=0.8)
        Experiment(cases=[_case("a", _invoice())], evaluators=[evaluator]).to_file(path)

        # No custom_evaluators=: the registry resolves it.
        reloaded = Experiment.from_file(path)
        (restored,) = reloaded._evaluators

        assert isinstance(restored, StructuredOutputSimilarity)
        assert restored.model_cls is Invoice
        assert (restored.match_threshold, restored.weight_hints) == (0.8, False)

    def test_a_reloaded_experiment_scores_the_same(self, tmp_path):
        path = tmp_path / "experiment.json"
        Experiment(
            cases=[_case("a", _invoice())],
            evaluators=[StructuredOutputSimilarity(Invoice)],
        ).to_file(path)

        report = Experiment.from_file(path).run_evaluations(lambda c: _invoice(vendor="Acme Corp"))

        assert report.scores[0] == pytest.approx(
            StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), _invoice(vendor="Acme Corp")))[0].score
        )


class TestCasesWithNothingToScore:
    """A case the evaluator cannot score returns a row, rather than raising."""

    def test_missing_expected_output_is_not_applicable(self):
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (None, _invoice())})

        (row,) = report.detailed_results[0]
        assert row.label == NOT_APPLICABLE
        assert row.not_applicable is True
        assert report.test_passes[0] is True, "declining to judge passes, so the row is droppable"

    def test_an_unparseable_output_is_a_scored_failure(self):
        """Prose instead of structured output is an eval failure, not a crash."""
        report = _run(StructuredOutputSimilarity(Invoice), {"a": (_invoice(), "sorry, I could not read it")})

        (row,) = report.detailed_results[0]
        assert row.score == 0.0
        assert row.test_pass is False
        assert row.reason.startswith("could not compare as Invoice: actual_output is a string that is not JSON")
        assert row.metadata == {"error": "unreadable"}
        assert "Evaluator error" not in (row.reason or ""), "the harness never saw an exception"

    def test_a_failed_case_is_excluded_from_the_rollup_but_visible(self):
        """The counts can differ, but the report says which case was dropped."""
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {"good": (_invoice(), _invoice()), "bad": (_invoice(), "prose")},
        )

        assert len(report.scores) == 2
        assert report.metrics().document_count == 1
        assert [e["case"] for e in report.per_case()] == ["good"]
        assert any(row.metadata == {"error": "unreadable"} for rows in report.detailed_results for row in rows)

    def test_a_foreign_ground_truth_is_a_visible_failure(self):
        """A case belonging to another schema fails, rather than being skipped.

        Reading it as the model would drop every unrecognized key, because Pydantic ignores
        extras, so an all-optional model came out blank on both sides and scored a perfect
        1.0. Skipping it instead would hide it: a not-applicable row passes and leaves the
        mean, and the same case read back from a result store is a dict, which must get the
        same answer.
        """
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {"a": (Receipt(merchant="M", tax=1.0), Receipt(merchant="M", tax=1.0))},
        )

        (row,) = report.detailed_results[0]
        assert (row.score, row.test_pass, row.label) == (0.0, False, "Invoice")
        assert row.not_applicable is False
        assert row.reason == (
            "could not compare as Invoice: expected_output (Receipt) has keys merchant, tax, "
            "none of which is a field of Invoice"
        )

    def test_a_redefined_class_with_the_same_fields_still_scores(self):
        """Re-running a notebook cell creates a new class object with the same fields.

        Instances of the old class fail `isinstance` against the new one, but they are read
        through their fields, so they are scored.
        """
        old = create_model("Invoice", invoice_id=(str, ...), vendor_name=(str | None, None))
        new = create_model("Invoice", invoice_id=(str, ...), vendor_name=(str | None, None))
        assert old is not new

        outputs = StructuredOutputSimilarity(new).evaluate(
            _data(old(invoice_id="I1", vendor_name="Acme"), old(invoice_id="I1", vendor_name="Acme"))
        )

        assert outputs[0].label == "Invoice"
        assert outputs[0].score == 1.0

    def test_an_agent_result_is_unwrapped(self):
        """Returning the agent's result directly is the natural way to write the task."""
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": []},
            metrics=None,
            state={},
            structured_output=_invoice(vendor="Acme Corp"),
        )

        from_result = StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), result))[0]
        from_model = StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), _invoice(vendor="Acme Corp")))[0]

        assert from_result.score == pytest.approx(from_model.score)

    def test_an_agent_result_without_structured_output_says_how_to_fix_it(self):
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "prose"}]},
            metrics=None,
            state={},
        )

        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), result))

        assert outputs[0].score == 0.0
        assert "structured_output_model=Invoice" in (outputs[0].reason or "")

    def test_a_mixed_schema_suite_fails_its_cross_pairs_but_rolls_up_each_schema(self):
        """Every evaluator runs over every case; cross pairs sharing no field name fail visibly.

        Not a supported layout -- the README says one `Experiment` per schema -- and schemas
        that share a field name are read as each other. Read blank, the cross pairs would pass
        at 1.0 and double each schema's `document_count`; failing them says to split the suite,
        and each schema's rollup still counts only its own case.
        """

        class Sparse(BaseModel):
            invoice_id: str | None = None
            vendor_name: str | None = None

        class Person(BaseModel):
            name: str | None = None

        cases = [
            _case("inv", Sparse(invoice_id="INV-1", vendor_name="Acme")),
            _case("person", Person(name="Ada")),
        ]
        actual = {"inv": Sparse(invoice_id="INV-1", vendor_name="Acme"), "person": Person(name="Ada")}
        report = Experiment(
            cases=cases,
            evaluators=[
                StructuredOutputSimilarity(Sparse, name="invoice"),
                StructuredOutputSimilarity(Person, name="person"),
            ],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: actual[c.metadata["name"]])

        assert report.metrics(evaluator="invoice").document_count == 1
        assert report.metrics(evaluator="person").document_count == 1
        assert sorted(report.test_passes) == [False, False, True, True]
        assert report.overall_score == pytest.approx(0.5), "the cross pairs are failures, not dropped rows"


class TestConcurrency:
    def test_evaluate_runs_on_worker_threads(self):
        """`evaluate_async` is deliberately not overridden.

        The base class's is `await asyncio.to_thread(self.evaluate, ...)`. A synchronous
        override would run the comparison inline on the event loop, blocking every other
        case's task on a CPU-bound Hungarian match.
        """
        seen: set[str] = set()

        class Probe(StructuredOutputSimilarity):
            def evaluate(self, evaluation_case):
                seen.add(threading.current_thread().name)
                return super().evaluate(evaluation_case)

        _run(Probe(Invoice), {f"c{i}": (_invoice(), _invoice()) for i in range(40)}, max_workers=10)

        assert seen, "evaluate() never ran"
        assert seen != {"MainThread"}, (
            "evaluate() ran only on the main thread, so the harness's to_thread offload "
            "was bypassed -- is evaluate_async overridden?"
        )

    def test_no_results_are_lost_at_the_harness_default(self):
        n = 200
        report = _run(
            StructuredOutputSimilarity(Invoice),
            {f"c{i}": (_invoice(), _invoice()) for i in range(n)},
            max_workers=10,
        )

        assert len(report.scores) == n
        assert report.metrics().document_count == n


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
        evaluator = StructuredOutputSimilarity(Invoice)
        outputs = evaluator.evaluate(_data(Invoice(invoice_id="INV-1", vendor_name="Acme Corporation"), actual))

        assert outputs[0].score == pytest.approx(1.0)

    def test_an_unusable_type_is_reported_not_raised(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        scalar = evaluator.evaluate(_data(_invoice(), 42))[0]
        unknown = evaluator.evaluate(_data(_invoice(), object()))[0]

        expected = (
            "could not compare as Invoice: actual_output must be an instance of Invoice, "
            "an AgentResult, a dict, or a JSON object; got"
        )
        assert (scalar.score, scalar.reason) == (0.0, f"{expected} int")
        assert (unknown.score, unknown.reason) == (0.0, f"{expected} object")

    def test_a_model_class_accepts_its_own_dotted_path(self):
        evaluator = StructuredOutputSimilarity(f"{Invoice.__module__}.Invoice")

        assert evaluator.model_cls is Invoice

    def test_a_nested_model_class_round_trips(self):
        """`to_dict` emits `__qualname__`, so a nested class has a dotted qualname.

        Splitting at the last dot read `myapp.models.Outer` as the module name and raised
        ModuleNotFoundError instead of reloading.
        """
        evaluator = StructuredOutputSimilarity(Nested.Inner)
        path = evaluator.to_dict()["model_cls"]

        assert path.endswith(".Nested.Inner")
        assert StructuredOutputSimilarity(path).model_cls is Nested.Inner

    def test_an_unresolvable_path_raises_type_error(self):
        with pytest.raises(TypeError, match="could not resolve"):
            StructuredOutputSimilarity("no.such.module.Model")

    def test_a_broken_module_reports_its_own_import_error(self, tmp_path, monkeypatch):
        """A missing dependency inside the target module is not a bad path.

        Reporting "could not resolve" for it hid the real ModuleNotFoundError.
        """
        (tmp_path / "broken_fixture_module.py").write_text(
            "import totally_absent_dependency\n\nclass Model:\n    pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ModuleNotFoundError, match="totally_absent_dependency"):
            StructuredOutputSimilarity("broken_fixture_module.Model")

    def test_an_unimportable_model_warns_at_save_time(self, caplog):
        """A model defined in a function cannot be reloaded from its dotted path."""

        def make():
            class Local(BaseModel):
                a: str | None = None

            return Local

        with caplog.at_level(logging.WARNING):
            path = StructuredOutputSimilarity(make()).to_dict()["model_cls"]

        assert "<locals>" in path
        assert "not importable by dotted path" in caplog.text

    def test_a_bare_name_raises_type_error(self):
        with pytest.raises(TypeError, match="could not resolve"):
            StructuredOutputSimilarity("Invoice")


class TestExplain:
    """Config derived from the model class, so it stays on the evaluator."""

    def test_covers_nested_paths(self):
        config = StructuredOutputSimilarity(Invoice).explain()

        assert "line_items.sku" in config
        assert config["invoice_id"]["comparator"] == "ExactComparator"

    def test_list_of_models_is_not_reported_as_a_string_comparator(self):
        """A List[StructuredModel] has no single comparator."""
        config = StructuredOutputSimilarity(Invoice).explain()

        assert config["line_items"]["comparator"] != "LevenshteinComparator"

    def test_reads_the_same_before_and_after_a_run(self):
        evaluator = StructuredOutputSimilarity(Invoice)
        before = evaluator.explain()
        _run(evaluator, {"a": (_invoice(), _invoice(vendor="Acme Corp"))})

        assert evaluator.explain() == before


class SparseInvoice(BaseModel):
    """The common extraction shape: two fields that matter, a long optional tail.

    `overall_score` credits a field absent on both sides with 1.0 at full weight, so the
    tail dominates the mean. These fields exist to make that measurable.
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
    against stickler 0.7.0 where `matched` meant all-fields-AND. They are what makes the
    `stickler-eval>=1.0.0` floor in pyproject.toml mean something.
    """

    def test_a_prediction_that_found_nothing_does_not_pass(self):
        """The defect this gate exists for.

        Ground truth has 2 of 10 fields; the agent returns nothing. The 8 fields blank on
        both sides each score 1.0, so the mean is 0.80 and clears the 0.7 default while
        recall is 0.00. Passing here would tell a CI gate that an agent which extracted
        nothing is fine.
        """
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, blank))

        assert outputs[0].score > 0.7, "the score really does clear the threshold"
        assert outputs[0].test_pass is False, "but nothing was found, so it must not pass"

        metadata = outputs[0].metadata
        assert metadata["recall"] == 0.0
        assert metadata["matched"] is True, "stickler's score-only verdict still says match"

    def test_widening_the_optional_tail_does_not_buy_a_pass(self):
        """Severity scales with schema width, so a higher threshold cannot fix it."""
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        blank = SparseInvoice(invoice_id="", vendor_name="")

        for threshold in (0.7, 0.75, 0.8):
            outputs = StructuredOutputSimilarity(SparseInvoice, match_threshold=threshold).evaluate(_data(gt, blank))
            assert outputs[0].test_pass is False, f"blank passed at threshold {threshold}"

    def test_a_good_extraction_still_passes(self):
        """The gate must not cost a correct sparse extraction its pass."""
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, gt.model_copy()))

        assert outputs[0].test_pass is True
        assert outputs[0].metadata["recall"] == 1.0

    def test_nothing_to_find_is_a_pass_not_a_failure(self):
        """A document whose ground truth is legitimately blank.

        stickler reports recall as 0.0 rather than None when the denominator is empty, so
        a naive recall gate would fail this. The guard reads the matrix instead.
        """

        class AllOptional(BaseModel):
            a: str | None = None
            b: str | None = None

        outputs = StructuredOutputSimilarity(AllOptional).evaluate(_data(AllOptional(), AllOptional()))

        assert outputs[0].score == 1.0
        assert outputs[0].test_pass is True

    def test_every_populated_field_wrong_does_not_pass(self):
        """stickler counts a wrong value as `fd`, not `fn`.

        So `tp + fn` is not the number of fields there were to find, and reading it as
        such made the "nothing to find" shortcut fire on a document whose ground truth was
        fully populated and entirely mis-extracted -- the same false pass the recall gate
        exists to close, reached through wrong values instead of missing ones.
        """
        gt = SparseInvoice(invoice_id="INV-8842", vendor_name="Acme Corporation")
        wrong = SparseInvoice(invoice_id="ZZZ-0000", vendor_name="Zeta Industries GmbH")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, wrong))

        matrix = outputs[0].metadata["comparison"]["confusion_matrix"]["overall"]
        assert (matrix["tp"], matrix["fn"]) == (0, 0), "the shortcut's old condition still holds"
        assert matrix["fd"] == 2, "the populated fields are wrong, not missing"
        assert outputs[0].score > 0.7, "the score really does clear the threshold"
        assert outputs[0].test_pass is False, "nothing was extracted correctly, so it must not pass"

    def test_fabricating_fields_for_a_blank_document_does_not_pass(self):
        """Nothing to find, values invented anyway.

        On a wide schema the invented fields are a minority, so the score clears the
        threshold while precision is 0.0. Nothing corroborates the prediction.
        """
        fields = {f"f{index}": (str | None, None) for index in range(20)}
        Wide = create_model("Wide", **fields)
        fabricated = Wide(**{f"f{index}": "made up" for index in range(5)})

        outputs = StructuredOutputSimilarity(Wide).evaluate(_data(Wide(), fabricated))

        assert outputs[0].score > 0.7, "the score really does clear the threshold"
        assert outputs[0].metadata["precision"] == 0.0
        assert outputs[0].test_pass is False

    def test_finding_everything_still_passes_when_extra_fields_are_invented(self):
        """The documented limit of gating on recall alone.

        Ground truth is often annotated sparsely, so failing an agent for extracting a
        field the annotation merely omitted would be wrong. `precision` is on the row for
        anyone wanting the stricter check.
        """
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        richer = SparseInvoice(invoice_id="INV-1", vendor_name="Acme", po_number="PO-9", notes="n")

        outputs = StructuredOutputSimilarity(SparseInvoice).evaluate(_data(gt, richer))

        assert outputs[0].metadata["recall"] == 1.0
        assert outputs[0].test_pass is True
        assert outputs[0].metadata["precision"] < 1.0, "precision shows what the gate ignores"

    def test_per_case_exposes_the_metrics_the_verdict_rests_on(self):
        gt = SparseInvoice(invoice_id="INV-1", vendor_name="Acme")
        report = _run(StructuredOutputSimilarity(SparseInvoice), {"a": (gt, gt.model_copy())})

        (entry,) = report.per_case()
        for key in ("test_pass", "matched", "precision", "recall", "f1"):
            assert key in entry, f"per_case() should expose {key}"


class TestWithoutTheExtra:
    def test_the_import_error_names_the_extra(self, monkeypatch):
        """Constructing without stickler installed must say how to fix it."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "stickler":
                raise ImportError("No module named 'stickler'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=r"strands-agents-evals\[stickler\]"):
            StructuredOutputSimilarity(Invoice)


class _CamelLine(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    sku_code: str | None = None
    unit_count: int | None = None


class _CamelInvoice(BaseModel):
    """Aliased the way an agent emitting camelCase JSON is: aliases only, no populate_by_name."""

    model_config = ConfigDict(alias_generator=to_camel)
    invoice_id: str
    vendor_name: str | None = None
    line_items: list[_CamelLine] = []


class _AllOptional(BaseModel):
    """Sparse: most fields blank in ground truth, which is where blank-vs-blank scores 1.0."""

    invoice_id: str | None = None
    vendor_name: str | None = None
    po_number: str | None = None
    notes: str | None = None
    total_amount: float | None = None


class _CamelAllOptional(BaseModel):
    """Aliased and all-optional: a dict by field name must not read as blank on both sides."""

    model_config = ConfigDict(alias_generator=to_camel)
    vendor_name: str | None = None
    total_amount: float | None = None


class _Status(str, enum.Enum):
    PAID = "paid"
    OPEN = "open"


class _TypedValues(BaseModel):
    """Values whose form changes between a model instance, a dict and JSON."""

    invoice_date: datetime.date | None = None
    status: _Status | None = None
    amount: Decimal | None = None


class _ValidationAliased(BaseModel):
    """Keys pydantic accepts only through validation aliases, not `alias`."""

    model_config = ConfigDict(alias_generator=AliasGenerator(validation_alias=to_camel))
    vendor_name: str | None = None
    invoice_id: str | None = Field(None, validation_alias=AliasChoices("invoiceNo", "invoice_id"))


class _ExtraAllowed(BaseModel):
    """Extra keys are kept by pydantic but are not fields, so they must not count as an overlap."""

    model_config = ConfigDict(extra="allow")
    invoice_id: str | None = None
    vendor_name: str | None = None


class _Strict(BaseModel):
    """Strict models reject a date given as a string in Python mode, which is how stores hold it."""

    model_config = ConfigDict(strict=True)
    invoice_date: datetime.date | None = None
    paid_at: datetime.datetime | None = Field(None, strict=True)


class _Tier(enum.Enum):
    GOLD = "gold"
    SILVER = "silver"


class _PlainEnum(BaseModel):
    tier: _Tier | None = None
    vendor_name: str | None = None


class _Forbidding(BaseModel):
    """Annotation keys in ground truth must not make a forbidding model fail validation."""

    model_config = ConfigDict(extra="forbid")
    invoice_id: str | None = None
    vendor_name: str | None = None


class _Person(BaseModel):
    patient: str | None = None
    diagnosis: str | None = None


# (model, correct ground truth, a prediction wrong on every populated field), as field-name dicts
_SHAPES = {
    "plain-nested": (
        Invoice,
        {
            "invoice_id": "INV-1",
            "vendor_name": "Acme Corporation",
            "total_amount": 100.0,
            "line_items": [{"sku": "SKU-1", "unit_price": 100.0}],
        },
        {
            "invoice_id": "ZZZ-9",
            "vendor_name": "Zeta Industries GmbH",
            "total_amount": 9.0,
            "line_items": [{"sku": "QQQ", "unit_price": 1.0}],
        },
    ),
    "aliased-nested": (
        _CamelInvoice,
        {
            "invoice_id": "INV-1",
            "vendor_name": "Acme Corporation",
            "line_items": [{"sku_code": "SKU-1", "unit_count": 4}],
        },
        {
            "invoice_id": "ZZZ-9",
            "vendor_name": "Zeta Industries GmbH",
            "line_items": [{"sku_code": "QQQ", "unit_count": 9}],
        },
    ),
    "aliased-all-optional": (
        _CamelAllOptional,
        {"vendor_name": "Acme Corporation", "total_amount": 5.0},
        {"vendor_name": "Zeta Industries GmbH", "total_amount": 99.0},
    ),
    "typed-values": (
        _TypedValues,
        {"invoice_date": datetime.date(2026, 3, 14), "status": _Status.PAID, "amount": Decimal("12.50")},
        {"invoice_date": datetime.date(2025, 1, 1), "status": _Status.OPEN, "amount": Decimal("99.00")},
    ),
    "validation-aliased": (
        _ValidationAliased,
        {"vendor_name": "Acme Corporation", "invoice_id": "INV-1"},
        {"vendor_name": "Zeta Industries GmbH", "invoice_id": "ZZZ-9"},
    ),
    "extra-allowed": (
        _ExtraAllowed,
        {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
        {"invoice_id": "ZZZ-9", "vendor_name": "Zeta Industries GmbH"},
    ),
    "strict-typed": (
        _Strict,
        {"invoice_date": datetime.date(2026, 3, 14), "paid_at": datetime.datetime(2026, 3, 14, 9, 30)},
        {"invoice_date": datetime.date(2025, 1, 1), "paid_at": datetime.datetime(2025, 1, 1, 8, 0)},
    ),
    "plain-enum": (
        _PlainEnum,
        {"tier": _Tier.GOLD, "vendor_name": "Acme Corporation"},
        {"tier": _Tier.SILVER, "vendor_name": "Zeta Industries GmbH"},
    ),
    "forbid-extra": (
        _Forbidding,
        {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
        {"invoice_id": "ZZZ-9", "vendor_name": "Zeta Industries GmbH"},
    ),
    "all-optional": (
        _AllOptional,
        {"invoice_id": "INV-1", "vendor_name": "Acme Corporation"},
        {"invoice_id": "ZZZ-9", "vendor_name": "Zeta Industries GmbH"},
    ),
}


def _forms(instance, with_agent_result=False):
    """Every representation a case value reaches the evaluator in."""
    forms = {
        "instance": instance,
        "dict-by-name": instance.model_dump(),
        "dict-by-alias": instance.model_dump(by_alias=True),
        "json-by-name": instance.model_dump_json(),
        "json-by-alias": instance.model_dump_json(by_alias=True),
    }
    if with_agent_result:
        forms["agent-result"] = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": []},
            metrics=None,
            state={},
            structured_output=instance,
        )
    return forms


@pytest.mark.parametrize("shape", sorted(_SHAPES))
class TestEveryInputForm:
    """Every ground-truth form against every prediction form, per model shape.

    Feeding one representation at a time misses false passes that only one form produces,
    such as an aliased model read from a dict by field name. This covers the forms a value
    reaches the evaluator in and the model shapes listed in `_SHAPES`; a new shape or form
    belongs here rather than in a one-off test.
    """

    def _instances(self, shape):
        model, truth, wrong = _SHAPES[shape]
        return model, model.model_validate(truth, by_name=True), model.model_validate(wrong, by_name=True)

    def _score_all(self, model, truths, predictions):
        evaluator = StructuredOutputSimilarity(model)
        return [
            (e, a, evaluator.evaluate(_data(expected, actual))[0])
            for e, expected in truths.items()
            for a, actual in predictions.items()
        ]

    def _variants(self, model, truth):
        """Predictions that leave one populated optional field blank."""
        # Required fields cannot be blank in a valid instance; the wrong-value property covers them.
        populated = {
            k: v
            for k, v in truth.model_dump().items()
            if v not in (None, [], "") and not model.model_fields[k].is_required()
        }
        variants = {}
        for field, value in populated.items():
            blank = model.model_validate(
                {**truth.model_dump(), field: None if not isinstance(value, list) else []}, by_name=True
            )
            variants[f"{field}-blank"] = blank
        return variants

    def test_a_correct_prediction_scores_one_in_every_form(self, shape):
        model, truth, _ = self._instances(shape)
        rows = self._score_all(model, _forms(truth, True), _forms(truth, True))
        bad = [(e, a, o.score, o.reason) for e, a, o in rows if not (o.score == 1.0 and o.test_pass)]
        assert bad == [], f"correct prediction did not score 1.0 / pass: {bad}"

    def test_a_wrong_prediction_never_passes_in_any_form(self, shape):
        model, truth, wrong = self._instances(shape)
        rows = self._score_all(model, _forms(truth, True), _forms(wrong, True))
        bad = [(e, a, o.score, o.reason) for e, a, o in rows if o.test_pass or o.score == 1.0]
        assert bad == [], f"wrong prediction passed: {bad}"

    def test_a_prediction_missing_any_one_field_never_scores_one(self, shape):
        model, truth, _ = self._instances(shape)
        for variant, prediction in self._variants(model, truth).items():
            rows = self._score_all(model, _forms(truth, True), _forms(prediction, True))
            bad = [(e, a, o.score) for e, a, o in rows if o.score == 1.0]
            assert bad == [], f"{variant} scored 1.0: {bad}"

    def test_ground_truth_of_another_schema_fails_in_every_form(self, shape):
        model, truth, _ = self._instances(shape)
        rows = self._score_all(model, _forms(_Person(patient="Jo", diagnosis="flu"), True), _forms(truth))
        bad = [(e, o.label, o.score, o.reason) for e, _, o in rows if o.test_pass or o.score > 0.0 or o.not_applicable]
        assert bad == [], f"foreign ground truth did not fail: {bad}"

    def test_a_prediction_of_another_schema_never_passes(self, shape):
        model, truth, _ = self._instances(shape)
        rows = self._score_all(model, _forms(truth, True), _forms(_Person(patient="Jo", diagnosis="flu"), True))
        bad = [(e, a, o.score, o.reason) for e, a, o in rows if o.test_pass or o.score > 0.0]
        assert bad == [], f"foreign prediction scored: {bad}"

    def test_extra_annotation_keys_on_ground_truth_are_ignored(self, shape):
        """Dataset files carry keys like an annotator id; they are ignored, not failed."""
        model, truth, wrong = self._instances(shape)
        annotated = {**truth.model_dump(mode="json"), "annotator": "bob"}
        truths = {"dict": annotated, "json": json.dumps(annotated)}
        correct = self._score_all(model, truths, _forms(truth))
        assert [(e, a, o.score) for e, a, o in correct if not (o.score == 1.0 and o.test_pass)] == []
        incorrect = self._score_all(model, truths, _forms(wrong))
        assert [(e, a, o.label, o.score) for e, a, o in incorrect if o.test_pass or o.label == NOT_APPLICABLE] == []

    @pytest.mark.parametrize("prediction", ["correct", "wrong", "foreign-truth"])
    def test_a_warm_cache_and_a_reloaded_experiment_score_like_a_cold_run(self, shape, prediction, tmp_path):
        """The two places a case becomes a field-name dict between runs.

        Every direction matters: a cached dict must not turn a wrong prediction into a pass, a
        strict model must not turn a correct one into a failure, and ground truth of another
        schema must fail as a dict just as it does as an instance. The dataset file is written
        with JSON-mode values, as a dataset file holds them: `Experiment.to_file` itself
        cannot serialize a `date` in a case, for any evaluator.
        """
        model, truth, wrong = self._instances(shape)
        answer = wrong if prediction == "wrong" else truth
        if prediction == "foreign-truth":
            truth = _Person(patient="Jo", diagnosis="flu")
        cases = [_case("c", truth)]

        def run(experiment, **kwargs):
            return asyncio.run(experiment.run_evaluations_async(lambda c: answer, **kwargs))

        store = LocalFileTaskResultStore(tmp_path / "store")
        cold = run(Experiment(cases=cases, evaluators=[StructuredOutputSimilarity(model)]), evaluation_data_store=store)
        warm = run(Experiment(cases=cases, evaluators=[StructuredOutputSimilarity(model)]), evaluation_data_store=store)
        path = tmp_path / "experiment.json"
        file_cases = [_case("c", truth.model_dump(mode="json"))]
        Experiment(cases=file_cases, evaluators=[StructuredOutputSimilarity(model)]).to_file(path)
        reloaded = run(Experiment.from_file(path))

        assert cold.test_passes == [prediction == "correct"]
        assert warm.scores == pytest.approx(cold.scores) and warm.test_passes == cold.test_passes
        assert reloaded.scores == pytest.approx(cold.scores) and reloaded.test_passes == cold.test_passes


class _GenericMetadataEvaluator(Evaluator):
    """A sibling evaluator using the obvious metadata names, which this evaluator also uses."""

    def evaluate(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=True, metadata={"precision": 1.0, "recall": 1.0, "f1": 1.0})]


class TestSharingAReportWithOtherEvaluators:
    def test_a_sibling_evaluator_using_generic_metadata_keys_is_ignored(self):
        """`metadata` is open to every evaluator, so rows are selected on `evaluated_by`.

        Selecting on any detail key read the sibling's well-formed row as a corrupted one of ours.
        """
        report = Experiment(
            cases=[_case("c1", _invoice())],
            evaluators=[StructuredOutputSimilarity(Invoice), _GenericMetadataEvaluator()],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: _invoice())

        assert report.metrics(evaluator="StructuredOutputSimilarity").document_count == 1
        assert [entry["evaluator"] for entry in report.per_case()] == ["StructuredOutputSimilarity"]

    def test_per_case_names_the_evaluator_of_each_row(self):
        """Two instances on one report would otherwise produce identical-looking rows."""
        report = Experiment(
            cases=[_case("c1", _invoice())],
            evaluators=[
                StructuredOutputSimilarity(Invoice, name="strict"),
                StructuredOutputSimilarity(Invoice, name="loose", match_threshold=0.5),
            ],
            report_cls=StructuredOutputReport,
        ).run_evaluations(lambda c: _invoice(vendor="Acme Corp"))

        assert sorted(entry["evaluator"] for entry in report.per_case()) == ["loose", "strict"]

    def test_the_reason_names_the_side_that_failed_to_validate(self):
        """A broken ground-truth row must not read as a failing prediction."""
        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data({"invoice_id": "I1"}, _invoice()))

        assert (outputs[0].reason or "").startswith("could not compare as Invoice: expected_output did not validate")


def test_the_report_is_importable_from_the_top_level_package():
    """Beside `EvaluationReport`, which it subclasses; the `evaluators` path keeps working."""
    assert "StructuredOutputReport" in strands_evals.__all__
    assert evaluators_package.StructuredOutputReport is StructuredOutputReport


class TestReadingValues:
    """Reading behaviors the input-form sweep cannot express."""

    def test_a_redefined_enum_reads_by_value(self):
        """Re-running a notebook cell redefines the enum too; its members must still read."""

        def define():
            class Status(enum.Enum):
                PAID = "paid"

            class Inv(BaseModel):
                status: Status | None = None

            return Inv, Status

        old_model, old_status = define()
        new_model, new_status = define()
        outputs = StructuredOutputSimilarity(new_model).evaluate(
            _data(new_model(status=new_status.PAID), old_model(status=old_status.PAID))
        )

        assert outputs[0].score == 1.0 and outputs[0].test_pass

    def test_a_computed_field_on_a_redefined_class_is_ignored(self):
        """`model_dump` includes computed fields, which a forbidding model would reject."""

        class Computing(BaseModel):
            invoice_id: str | None = None
            vendor_name: str | None = None

            @computed_field
            @property
            def upper(self) -> str:
                return (self.vendor_name or "").upper()

        prediction = Computing(invoice_id="INV-1", vendor_name="Acme")
        outputs = StructuredOutputSimilarity(_Forbidding).evaluate(
            _data(_Forbidding(invoice_id="INV-1", vendor_name="Acme"), prediction)
        )

        assert outputs[0].score == 1.0 and outputs[0].test_pass

    def test_a_nested_computed_field_on_a_forbidding_model_survives_a_warm_cache(self, tmp_path):
        """The store writes computed fields; the forbidding inner model rejects them on reload."""

        class Line(BaseModel):
            model_config = ConfigDict(extra="forbid")
            unit: float
            count: int

            @computed_field
            @property
            def total(self) -> float:
                return self.unit * self.count

        class Order(BaseModel):
            lines: list[Line] = []

        truth = Order(lines=[Line(unit=2.0, count=3)])
        store = LocalFileTaskResultStore(tmp_path)
        runs = [
            asyncio.run(
                Experiment(
                    cases=[_case("c", truth)], evaluators=[StructuredOutputSimilarity(Order)]
                ).run_evaluations_async(lambda c: truth, evaluation_data_store=store)
            )
            for _ in ("cold", "warm")
        ]

        assert [(run.scores[0], run.test_passes[0]) for run in runs] == [(1.0, True), (1.0, True)]

    def test_an_extra_key_inside_a_forbidding_union_member_is_ignored(self):
        """Extras are ignored at every level, so a union member's annotation cannot fail the case."""

        class Cat(BaseModel):
            model_config = ConfigDict(extra="forbid")
            lives: int

        class Dog(BaseModel):
            model_config = ConfigDict(extra="forbid")
            barks: bool

        class Pet(BaseModel):
            pet: Cat | Dog | None = None

        outputs = StructuredOutputSimilarity(Pet).evaluate(
            _data({"pet": {"lives": 9, "annotator": "bob"}}, Pet(pet=Cat(lives=9)))
        )

        assert outputs[0].score == 1.0 and outputs[0].test_pass

    def test_a_shared_key_scores_on_the_schema_fields_and_ignores_the_rest(self):
        """Pinned on purpose: only a value with no field of the model is another schema.

        Treating partial overlap as foreign would fail annotated datasets. The evaluator
        scores what its schema defines.
        """

        class Keyed(BaseModel):
            id: str | None = None
            vendor: str | None = None

        outputs = StructuredOutputSimilarity(Keyed).evaluate(
            _data({"id": "A", "vendor": "Acme"}, {"id": "A", "vendor": "Acme", "receipt_no": "x"})
        )

        assert outputs[0].label == "Keyed" and outputs[0].score == 1.0

    def test_a_validation_alias_key_reads(self):
        camel = {"vendorName": "Acme Corporation", "invoiceNo": "INV-1"}
        outputs = StructuredOutputSimilarity(_ValidationAliased).evaluate(
            _data(camel, _ValidationAliased.model_validate(camel))
        )

        assert outputs[0].score == 1.0 and outputs[0].test_pass

    def test_bytes_in_a_python_dict_read_natively(self):
        class Blob(BaseModel):
            model_config = ConfigDict(val_json_bytes="base64")
            blob: bytes | None = None

        outputs = StructuredOutputSimilarity(Blob).evaluate(_data(Blob(blob=b"hello"), {"blob": b"hello"}))

        assert outputs[0].score == 1.0 and outputs[0].test_pass

    def test_a_normalizing_validator_still_reads_dicts(self):
        """A validator that rewrites a field on every input does not make dict cases unreadable."""

        class Classification(BaseModel):
            label: str = "unknown"

            @model_validator(mode="after")
            def lower(self):
                self.label = self.label.lower()
                return self

        evaluator = StructuredOutputSimilarity(Classification)
        wrong = evaluator.evaluate(_data({"label": "Positive"}, {"label": "negative"}))[0]
        right = evaluator.evaluate(_data({"label": "Positive"}, '{"label": "POSITIVE"}'))[0]

        assert (wrong.label, wrong.test_pass) == ("Classification", False)
        assert (right.score, right.test_pass) == (1.0, True)

    def test_ground_truth_stating_a_default_value_is_read(self):
        """`count: 0` is a real annotation, even though it equals the default."""

        class Counted(BaseModel):
            count: int = 0

        outputs = StructuredOutputSimilarity(Counted).evaluate(_data({"count": 0}, {"count": 3}))

        assert outputs[0].label == "Counted" and outputs[0].test_pass is False

    def test_two_blank_documents_are_a_match(self):
        """By design: an empty object is a blank document of this schema, not another schema."""
        outputs = StructuredOutputSimilarity(_AllOptional).evaluate(_data("{}", {}))

        assert outputs[0].label == "_AllOptional" and outputs[0].score == 1.0

    def test_an_old_pydantic_is_reported_at_construction(self, monkeypatch):
        """`model_validate(extra=...)` needs 2.12; otherwise every case would fail on a TypeError."""
        monkeypatch.setattr(structured_output_module, "PYDANTIC_VERSION", "2.11.7")

        with pytest.raises(ImportError, match=r"requires pydantic>=2\.12 \(found 2\.11\.7\)"):
            StructuredOutputSimilarity(Invoice)

    @pytest.mark.parametrize("model_cls", [RootModel[dict[str, str]], create_model("Empty")], ids=["root", "empty"])
    def test_a_model_without_named_fields_is_rejected_at_construction(self, model_cls):
        """Values are recognized by field name; a RootModel's cached form has no `root` key."""
        with pytest.raises(TypeError, match="must have named fields to compare"):
            StructuredOutputSimilarity(model_cls)


class TestAValueOfAnotherSchemaFails:
    """A value is another schema when it has keys and none of them is a field of the model.

    Such a value fails, on either side and in any form, with a reason naming its keys. The
    rule is on keys alone, so it does not depend on what the model's validators do.
    """

    def test_the_reason_lists_the_keys_and_the_form(self):
        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data('{"patient": "Jo"}', _invoice()))

        assert outputs[0].reason == (
            "could not compare as Invoice: expected_output (JSON dict) has keys patient, "
            "none of which is a field of Invoice"
        )

    def test_an_agent_result_holding_another_model_fails(self):
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": []},
            metrics=None,
            state={},
            structured_output=Receipt(merchant="M", tax=1.0),
        )
        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data(_invoice(), result))

        assert outputs[0].test_pass is False
        assert "actual_output (Receipt) has keys merchant, tax" in (outputs[0].reason or "")

    def test_a_key_only_the_alias_reads_is_not_a_field_when_aliases_are_off(self):
        class NameOnly(BaseModel):
            model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)
            invoice_id: str | None = Field(None, alias="InvoiceID")

        outputs = StructuredOutputSimilarity(NameOnly).evaluate(_data({"invoice_id": "1"}, {"InvoiceID": "1"}))

        assert outputs[0].test_pass is False
        assert "has keys InvoiceID, none of which is a field of NameOnly" in (outputs[0].reason or "")

    def test_a_validation_alias_replaces_the_alias_for_reading(self):
        """pydantic reads `total_in`, not `Total`; a value keyed only by `Total` is another schema."""

        class Overridden(BaseModel):
            total: float | None = Field(None, alias="Total", validation_alias="total_in")

        truth = {"total_in": 5.0}
        assert StructuredOutputSimilarity(Overridden).evaluate(_data(truth, {"total_in": 5.0}))[0].score == 1.0
        unreadable = StructuredOutputSimilarity(Overridden).evaluate(_data({"Total": 5.0}, Overridden()))[0]
        assert unreadable.test_pass is False and "has keys Total" in (unreadable.reason or "")

    def test_a_validator_that_sets_a_field_for_any_input_does_not_make_a_foreign_value_readable(self):
        """A field a validator sets for any input does not make a foreign value read as the model."""

        class Derived(BaseModel):
            name: str | None = None
            length: int = 0

            @model_validator(mode="after")
            def measure(self):
                self.length = len(self.name or "")
                return self

        outputs = StructuredOutputSimilarity(Derived).evaluate(_data({"other": 1}, {"zzz": 1}))

        assert outputs[0].test_pass is False and outputs[0].score == 0.0
        correct = StructuredOutputSimilarity(Derived).evaluate(_data({"name": "Ada"}, {"name": "Ada"}))
        assert correct[0].score == 1.0

    def test_keys_only_a_validator_understands_fail_rather_than_being_guessed_at(self):
        """Pinned on purpose: the rule reads keys, not validators, so it cannot misjudge one.

        A model that maps `full_name` onto its fields still scores its own instances, and a
        result store writes field names, so only a hand-written dataset meets this, and it
        is told which keys were not recognized.
        """

        class Name(BaseModel):
            first: str | None = None
            last: str | None = None

            @model_validator(mode="before")
            @classmethod
            def split(cls, value):
                if isinstance(value, dict) and "full_name" in value:
                    first, last = value["full_name"].split(" ", 1)
                    return {"first": first, "last": last}
                return value

        evaluator = StructuredOutputSimilarity(Name)
        source_form = evaluator.evaluate(_data({"full_name": "Ada Lovelace"}, Name(first="Ada", last="Lovelace")))[0]
        instances = evaluator.evaluate(_data(Name(first="Ada", last="Lovelace"), Name(first="Ada", last="Lovelace")))[0]

        assert source_form.test_pass is False and "has keys full_name" in (source_form.reason or "")
        assert (instances.score, instances.test_pass) == (1.0, True)

    @pytest.mark.parametrize(
        "ground_truth",
        [
            "2x2 is 4.",
            '["a", "b"]',
            ["a", "b"],
            4,
            (1, 2),
            {1, 2},
            Decimal("4"),
            b'{"invoice_id": "I"}',
            RootModel[int](5),
        ],
        ids=["prose", "json-list", "list", "int", "tuple", "set", "decimal", "bytes", "scalar-root-model"],
    )
    def test_ground_truth_that_is_not_an_object_fails(self, ground_truth):
        """Not skipped: a not-applicable row passes and leaves the mean, which would hide it."""
        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data(ground_truth, _invoice()))

        assert (outputs[0].score, outputs[0].test_pass, outputs[0].label) == (0.0, False, "Invoice")
        assert (outputs[0].reason or "").startswith("could not compare as Invoice: expected_output ")

    def test_a_malformed_json_object_fails_with_the_parse_error(self):
        outputs = StructuredOutputSimilarity(Invoice).evaluate(_data('{"invoice_id": "INV-1"', _invoice()))

        assert outputs[0].test_pass is False
        assert (outputs[0].reason or "").startswith(
            "could not compare as Invoice: expected_output is a string that is not JSON"
        )
