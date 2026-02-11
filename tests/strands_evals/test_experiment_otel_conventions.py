"""Property-based tests for OTel test semantic conventions on Experiment.

Feature: otel-test-semantic-conventions
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from strands_evals import Case, Experiment


# Strategy: non-empty strings for experiment names (printable, no surrogates)
name_strategy = st.text(min_size=1, max_size=200, alphabet=st.characters(categories=("L", "N", "P", "S", "Z")))


@settings(max_examples=100)
@given(name=name_strategy)
def test_property_1_name_stored_on_construction(name: str):
    """Property 1: Name stored on construction.

    For any non-empty string s, constructing Experiment(name=s) should result in experiment.name == s.

    Feature: otel-test-semantic-conventions, Property 1: Name stored on construction

    **Validates: Requirements 1.1**
    """
    experiment = Experiment(name=name)
    assert experiment.name == name


@settings(max_examples=100)
@given(name=st.text(max_size=200, alphabet=st.characters(categories=("L", "N", "P", "S", "Z"))))
def test_property_2_serialization_round_trip_preserves_name(name: str):
    """Property 2: Serialization round-trip preserves name.

    For any valid Experiment with any name string, calling to_dict and then from_dict
    on the result should produce an Experiment whose name equals the original experiment's name.

    Feature: otel-test-semantic-conventions, Property 2: Serialization round-trip preserves name

    **Validates: Requirements 1.3, 1.4, 1.6**
    """
    case = Case(input="test_input", expected_output="test_output")
    experiment = Experiment(cases=[case], name=name)

    serialized = experiment.to_dict()
    restored = Experiment.from_dict(serialized)

    assert restored.name == experiment.name


import uuid
from unittest.mock import MagicMock, call, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from strands_evals.types import EvaluationOutput


# --- Strategies for Property 3 ---

# Reusable strategy for case names (non-empty printable strings)
case_name_strategy = st.text(min_size=1, max_size=100, alphabet=st.characters(categories=("L", "N", "P", "S")))

# Strategy for session IDs (non-empty printable strings)
session_id_strategy = st.text(min_size=1, max_size=100, alphabet=st.characters(categories=("L", "N", "P", "S")))


def _make_mock_span():
    """Create a mock span that works as a context manager."""
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


def _make_mock_evaluator(pass_result: bool = True):
    """Create a mock evaluator that returns a controlled result."""
    evaluator = MagicMock()
    evaluator.get_type_name.return_value = "MockEvaluator"
    evaluator.evaluate.return_value = [
        EvaluationOutput(score=1.0 if pass_result else 0.0, test_pass=pass_result, reason="mock")
    ]
    evaluator.aggregator.return_value = (1.0 if pass_result else 0.0, pass_result, "mock reason")
    return evaluator


@settings(max_examples=100)
@given(
    exp_name=name_strategy,
    case_names=st.lists(case_name_strategy, min_size=1, max_size=5),
    session_ids=st.lists(session_id_strategy, min_size=1, max_size=5),
)
def test_property_3_case_spans_contain_required_test_attributes(
    exp_name: str, case_names: list[str], session_ids: list[str]
):
    """Property 3: Case spans contain all required test.* attributes (sync path).

    For any Experiment with any name and any non-empty list of Cases (with arbitrary names
    and session_ids), running evaluations (sync) should produce case-level spans where each
    span contains test.suite.name equal to the experiment name, test.suite.run.id as a valid
    UUID4, test.case.name equal to the case name, and test.case.id equal to the case session_id.

    Feature: otel-test-semantic-conventions, Property 3: Case spans contain all required test.* attributes

    **Validates: Requirements 2.1, 2.2**
    """
    # Align lists: zip to shortest length
    paired = list(zip(case_names, session_ids))

    cases = [
        Case(name=cn, session_id=sid, input="input", expected_output="input")
        for cn, sid in paired
    ]

    evaluator = _make_mock_evaluator(pass_result=True)
    experiment = Experiment(cases=cases, evaluators=[evaluator], name=exp_name)

    # Track spans created by start_as_current_span
    spans = []

    def mock_start_span(name, attributes=None):
        span = _make_mock_span()
        span._test_span_name = name
        span._test_initial_attributes = attributes or {}
        spans.append(span)
        return span

    with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
        experiment.run_evaluations(lambda c: c.input)

    # Filter to eval_case spans only
    case_spans = [s for s in spans if s._test_span_name.startswith("eval_case ")]

    assert len(case_spans) == len(paired), (
        f"Expected {len(paired)} eval_case spans, got {len(case_spans)}"
    )

    for i, (cn, sid) in enumerate(paired):
        span = case_spans[i]
        attrs = span._test_initial_attributes

        # test.suite.name must equal experiment name
        assert attrs.get("test.suite.name") == exp_name, (
            f"test.suite.name mismatch: expected {exp_name!r}, got {attrs.get('test.suite.name')!r}"
        )

        # test.suite.run.id must be a valid UUID4
        run_id = attrs.get("test.suite.run.id")
        assert run_id is not None, "test.suite.run.id is missing"
        parsed = uuid.UUID(run_id, version=4)
        assert str(parsed) == run_id, f"test.suite.run.id is not a valid UUID4: {run_id!r}"

        # test.case.name must equal the case name
        assert attrs.get("test.case.name") == cn, (
            f"test.case.name mismatch: expected {cn!r}, got {attrs.get('test.case.name')!r}"
        )

        # test.case.id must equal the case session_id
        assert attrs.get("test.case.id") == sid, (
            f"test.case.id mismatch: expected {sid!r}, got {attrs.get('test.case.id')!r}"
        )

    # All eval_case spans must share the same run_id
    run_ids = [s._test_initial_attributes.get("test.suite.run.id") for s in case_spans]
    assert len(set(run_ids)) == 1, f"Expected all case spans to share one run_id, got {set(run_ids)}"


@settings(max_examples=100)
@given(aggregate_pass=st.booleans())
def test_property_4_evaluator_span_result_status_matches_pass_fail(aggregate_pass: bool):
    """Property 4: Evaluator span result status matches pass/fail (sync path).

    For any evaluator result where aggregate_pass is a boolean, the evaluator span should
    have test.case.result.status set to "pass" when aggregate_pass is True and "fail" when
    aggregate_pass is False.

    Feature: otel-test-semantic-conventions, Property 4: Evaluator span result status matches pass/fail

    **Validates: Requirements 2.3**
    """
    case = Case(name="test_case", input="input", expected_output="input")
    evaluator = _make_mock_evaluator(pass_result=aggregate_pass)
    experiment = Experiment(cases=[case], evaluators=[evaluator], name="test_exp")

    # Track spans
    spans = []

    def mock_start_span(name, attributes=None):
        span = _make_mock_span()
        span._test_span_name = name
        span._test_initial_attributes = attributes or {}
        spans.append(span)
        return span

    with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
        experiment.run_evaluations(lambda c: c.input)

    # Filter to evaluator spans
    eval_spans = [s for s in spans if s._test_span_name.startswith("evaluator ")]

    assert len(eval_spans) == 1, f"Expected 1 evaluator span, got {len(eval_spans)}"

    eval_span = eval_spans[0]

    # Check set_attributes was called with test.case.result.status
    eval_span.set_attributes.assert_called()
    set_attrs_call = eval_span.set_attributes.call_args
    attrs_dict = set_attrs_call[0][0] if set_attrs_call[0] else set_attrs_call[1]

    expected_status = "pass" if aggregate_pass else "fail"
    assert attrs_dict.get("test.case.result.status") == expected_status, (
        f"Expected test.case.result.status={expected_status!r}, "
        f"got {attrs_dict.get('test.case.result.status')!r}"
    )


# --- Property 3 (async variant) and Property 5: Async path tests ---

import asyncio

import pytest


@settings(max_examples=100)
@given(
    exp_name=name_strategy,
    case_names=st.lists(case_name_strategy, min_size=1, max_size=5, unique=True),
    session_ids=st.lists(session_id_strategy, min_size=1, max_size=5),
)
async def test_property_3_async_case_spans_contain_required_test_attributes(
    exp_name: str, case_names: list[str], session_ids: list[str]
):
    """Property 3 (async variant): Case spans contain all required test.* attributes.

    For any Experiment with any name and any non-empty list of Cases (with arbitrary names
    and session_ids), running evaluations async should produce case-level spans where each
    span contains test.suite.name equal to the experiment name, test.suite.run.id as a valid
    UUID4, test.case.name equal to the case name, and test.case.id equal to the case session_id.

    Feature: otel-test-semantic-conventions, Property 3: Case spans contain all required test.* attributes

    **Validates: Requirements 3.1, 3.2**
    """
    # Align lists: zip to shortest length
    paired = list(zip(case_names, session_ids))

    cases = [
        Case(name=cn, session_id=sid, input="input", expected_output="input")
        for cn, sid in paired
    ]

    evaluator = _make_mock_evaluator(pass_result=True)
    experiment = Experiment(cases=cases, evaluators=[evaluator], name=exp_name)

    # Track spans created by start_as_current_span
    spans = []

    def mock_start_span(name, attributes=None):
        span = _make_mock_span()
        span._test_span_name = name
        span._test_initial_attributes = attributes or {}
        # Collect set_attributes calls into a merged dict for easy inspection
        span._test_set_attributes_merged = {}
        original_set_attributes = span.set_attributes

        def tracking_set_attributes(attrs):
            span._test_set_attributes_merged.update(attrs)
            return original_set_attributes(attrs)

        span.set_attributes = MagicMock(side_effect=tracking_set_attributes)
        spans.append(span)
        return span

    with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):

            async def async_task(c):
                return c.input

            await experiment.run_evaluations_async(async_task)

    # Filter to execute_case spans only (async path uses execute_case, not eval_case)
    case_spans = [s for s in spans if s._test_span_name.startswith("execute_case ")]

    assert len(case_spans) == len(paired), (
        f"Expected {len(paired)} execute_case spans, got {len(case_spans)}"
    )

    for cn, sid in paired:
        # Find the span for this case (order may vary in async)
        matching = [s for s in case_spans if s._test_span_name == f"execute_case {cn}"]
        assert len(matching) >= 1, f"No execute_case span found for case {cn!r}"
        span = matching[0]

        # In async path, test.* attributes are set via set_attributes (not initial attributes)
        attrs = span._test_set_attributes_merged

        # test.suite.name must equal experiment name
        assert attrs.get("test.suite.name") == exp_name, (
            f"test.suite.name mismatch: expected {exp_name!r}, got {attrs.get('test.suite.name')!r}"
        )

        # test.suite.run.id must be a valid UUID4
        run_id = attrs.get("test.suite.run.id")
        assert run_id is not None, "test.suite.run.id is missing"
        parsed = uuid.UUID(run_id, version=4)
        assert str(parsed) == run_id, f"test.suite.run.id is not a valid UUID4: {run_id!r}"

        # test.case.name must equal the case name
        assert attrs.get("test.case.name") == cn, (
            f"test.case.name mismatch: expected {cn!r}, got {attrs.get('test.case.name')!r}"
        )

        # test.case.id must equal the case session_id
        assert attrs.get("test.case.id") == sid, (
            f"test.case.id mismatch: expected {sid!r}, got {attrs.get('test.case.id')!r}"
        )

    # All execute_case spans must share the same run_id
    run_ids = [s._test_set_attributes_merged.get("test.suite.run.id") for s in case_spans]
    assert len(set(run_ids)) == 1, f"Expected all case spans to share one run_id, got {set(run_ids)}"


@settings(max_examples=100)
@given(
    exp_name=name_strategy,
    case_names=st.lists(case_name_strategy, min_size=1, max_size=3),
    session_ids=st.lists(session_id_strategy, min_size=1, max_size=3),
    pass_results=st.lists(st.booleans(), min_size=1, max_size=3),
)
async def test_property_5_existing_gen_ai_attributes_preserved_async(
    exp_name: str, case_names: list[str], session_ids: list[str], pass_results: list[bool]
):
    """Property 5: Existing gen_ai.evaluation.* attributes preserved (async path).

    For any Experiment run (async), all spans that previously carried gen_ai.evaluation.*
    attributes should still carry those same attributes with the same values. The addition
    of test.* attributes should not remove or alter any existing attribute.

    Feature: otel-test-semantic-conventions, Property 5: Existing gen_ai.evaluation.* attributes preserved

    **Validates: Requirements 3.3, 3.4**
    """
    # Align all lists to shortest length
    min_len = min(len(case_names), len(session_ids), len(pass_results))
    paired = list(zip(case_names[:min_len], session_ids[:min_len], pass_results[:min_len]))

    cases = [
        Case(name=cn, session_id=sid, input="input", expected_output="input")
        for cn, sid, _ in paired
    ]

    # Create evaluators that return the specified pass/fail results
    evaluator = MagicMock()
    evaluator.get_type_name.return_value = "MockEvaluator"
    evaluator.evaluation_level = None

    # Track spans
    spans = []

    def mock_start_span(name, attributes=None):
        span = _make_mock_span()
        span._test_span_name = name
        span._test_set_attributes_calls = []
        original_set_attributes = span.set_attributes

        def tracking_set_attributes(attrs):
            span._test_set_attributes_calls.append(dict(attrs))
            return original_set_attributes(attrs)

        span.set_attributes = MagicMock(side_effect=tracking_set_attributes)
        spans.append(span)
        return span

    # We need to make evaluate_async return different results per case
    call_count = [0]

    async def mock_evaluate_async(evaluation_case):
        idx = min(call_count[0], len(paired) - 1)
        pass_result = paired[idx][2]
        call_count[0] += 1
        return [EvaluationOutput(score=1.0 if pass_result else 0.0, test_pass=pass_result, reason="mock")]

    evaluator.evaluate_async = MagicMock(side_effect=mock_evaluate_async)
    evaluator.aggregator = MagicMock(
        side_effect=lambda outputs: (outputs[0].score, outputs[0].test_pass, outputs[0].reason)
    )

    experiment = Experiment(cases=cases, evaluators=[evaluator], name=exp_name)

    with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):

            async def async_task(c):
                return c.input

            await experiment.run_evaluations_async(async_task)

    # Check execute_case spans have gen_ai.evaluation.* attributes preserved
    case_spans = [s for s in spans if s._test_span_name.startswith("execute_case ")]
    assert len(case_spans) == len(paired)

    for span in case_spans:
        # Merge all set_attributes calls
        all_attrs = {}
        for call_attrs in span._test_set_attributes_calls:
            all_attrs.update(call_attrs)

        # Verify gen_ai.evaluation.* attributes are present on case spans
        assert "gen_ai.evaluation.data.input" in all_attrs, (
            f"gen_ai.evaluation.data.input missing from execute_case span"
        )
        assert "gen_ai.evaluation.data.expected_output" in all_attrs, (
            f"gen_ai.evaluation.data.expected_output missing from execute_case span"
        )
        assert "gen_ai.evaluation.data.actual_output" in all_attrs, (
            f"gen_ai.evaluation.data.actual_output missing from execute_case span"
        )

        # Verify test.* attributes are also present (coexistence)
        assert "test.suite.name" in all_attrs, "test.suite.name missing from execute_case span"
        assert "test.suite.run.id" in all_attrs, "test.suite.run.id missing from execute_case span"
        assert "test.case.name" in all_attrs, "test.case.name missing from execute_case span"
        assert "test.case.id" in all_attrs, "test.case.id missing from execute_case span"

    # Check evaluator spans have gen_ai.evaluation.* attributes preserved
    eval_spans = [s for s in spans if s._test_span_name.startswith("evaluator ")]
    assert len(eval_spans) == len(paired)

    for span in eval_spans:
        all_attrs = {}
        for call_attrs in span._test_set_attributes_calls:
            all_attrs.update(call_attrs)

        # Verify gen_ai.evaluation.* attributes on evaluator spans
        assert "gen_ai.evaluation.score.label" in all_attrs, (
            f"gen_ai.evaluation.score.label missing from evaluator span"
        )
        assert "gen_ai.evaluation.score.value" in all_attrs, (
            f"gen_ai.evaluation.score.value missing from evaluator span"
        )
        assert "gen_ai.evaluation.test_pass" in all_attrs, (
            f"gen_ai.evaluation.test_pass missing from evaluator span"
        )
        assert "gen_ai.evaluation.explanation" in all_attrs, (
            f"gen_ai.evaluation.explanation missing from evaluator span"
        )

        # Verify test.case.result.status is also present (coexistence)
        assert "test.case.result.status" in all_attrs, (
            f"test.case.result.status missing from evaluator span"
        )
        # Verify the status value is valid
        assert all_attrs["test.case.result.status"] in ("pass", "fail"), (
            f"test.case.result.status has invalid value: {all_attrs['test.case.result.status']!r}"
        )


# ============================================================================
# Task 5.2: Unit tests for edge cases and backward compatibility
# ============================================================================


class TestDefaultName:
    """Test default name is 'unnamed_experiment' when not provided.

    Validates: Requirements 4.1, 6.1
    """

    def test_default_name_when_not_provided(self):
        experiment = Experiment()
        assert experiment.name == "unnamed_experiment"

    def test_default_name_with_cases_and_evaluators(self):
        case = Case(input="x", expected_output="y")
        experiment = Experiment(cases=[case])
        assert experiment.name == "unnamed_experiment"


class TestFromDictLegacy:
    """Test from_dict with legacy dict (no name key) defaults correctly.

    Validates: Requirements 1.5, 6.3
    """

    def test_from_dict_without_name_key(self):
        legacy_dict = {
            "cases": [{"input": "hello", "expected_output": "world"}],
            "evaluators": [{"evaluator_type": "Evaluator"}],
        }
        experiment = Experiment.from_dict(legacy_dict)
        assert experiment.name == "unnamed_experiment"

    def test_from_dict_with_name_key(self):
        data = {
            "name": "my_experiment",
            "cases": [{"input": "hello", "expected_output": "world"}],
            "evaluators": [{"evaluator_type": "Evaluator"}],
        }
        experiment = Experiment.from_dict(data)
        assert experiment.name == "my_experiment"


class TestNoWrapperSpan:
    """Test no wrapper span is created (verify span names).

    Validates: Requirements 4.1, 4.2
    """

    def test_sync_no_wrapper_span(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            experiment.run_evaluations(lambda c: c.input)

        span_names = [s._test_span_name for s in spans]
        # No wrapper span like "test_suite_run" should exist
        for name in span_names:
            assert not name.startswith("test_suite_run"), f"Unexpected wrapper span: {name}"
        # Only expected span types should be present
        allowed_prefixes = ("eval_case ", "task_execution", "evaluator ")
        for name in span_names:
            assert any(name.startswith(p) for p in allowed_prefixes), (
                f"Unexpected span name: {name!r}"
            )

    @pytest.mark.asyncio
    async def test_async_no_wrapper_span(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):
                await experiment.run_evaluations_async(lambda c: c.input)

        span_names = [s._test_span_name for s in spans]
        for name in span_names:
            assert not name.startswith("test_suite_run"), f"Unexpected wrapper span: {name}"
        allowed_prefixes = ("execute_case ", "evaluator ")
        for name in span_names:
            assert any(name.startswith(p) for p in allowed_prefixes), (
                f"Unexpected span name: {name!r}"
            )


class TestNoAddEventForTestData:
    """Test no add_event calls for test.* data.

    Validates: Requirements 5.1
    """

    def test_sync_no_add_event_for_test_data(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            experiment.run_evaluations(lambda c: c.input)

        for span in spans:
            # add_event should never be called with test.* attribute keys
            if span.add_event.called:
                for call_obj in span.add_event.call_args_list:
                    args, kwargs = call_obj
                    event_attrs = kwargs.get("attributes", {})
                    if len(args) > 1:
                        event_attrs = args[1]
                    for key in event_attrs:
                        assert not key.startswith("test."), (
                            f"add_event called with test.* attribute: {key}"
                        )

    @pytest.mark.asyncio
    async def test_async_no_add_event_for_test_data(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):
                await experiment.run_evaluations_async(lambda c: c.input)

        for span in spans:
            if span.add_event.called:
                for call_obj in span.add_event.call_args_list:
                    args, kwargs = call_obj
                    event_attrs = kwargs.get("attributes", {})
                    if len(args) > 1:
                        event_attrs = args[1]
                    for key in event_attrs:
                        assert not key.startswith("test."), (
                            f"add_event called with test.* attribute: {key}"
                        )


class TestRunIdFormat:
    """Test run_id is valid UUID4 format.

    Validates: Requirements 2.1, 3.1
    """

    def test_sync_run_id_is_valid_uuid4(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            span._test_initial_attributes = attributes or {}
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            experiment.run_evaluations(lambda c: c.input)

        case_spans = [s for s in spans if s._test_span_name.startswith("eval_case ")]
        assert len(case_spans) == 1
        run_id = case_spans[0]._test_initial_attributes["test.suite.run.id"]
        parsed = uuid.UUID(run_id, version=4)
        assert str(parsed) == run_id

    @pytest.mark.asyncio
    async def test_async_run_id_is_valid_uuid4(self):
        case = Case(name="c1", input="input", expected_output="input")
        evaluator = _make_mock_evaluator(pass_result=True)
        experiment = Experiment(cases=[case], evaluators=[evaluator], name="test")

        spans = []

        def mock_start_span(name, attributes=None):
            span = _make_mock_span()
            span._test_span_name = name
            span._test_set_attributes_merged = {}

            def tracking_set_attributes(attrs):
                span._test_set_attributes_merged.update(attrs)

            span.set_attributes = MagicMock(side_effect=tracking_set_attributes)
            spans.append(span)
            return span

        with patch.object(experiment._tracer, "start_as_current_span", side_effect=mock_start_span):
            with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):
                await experiment.run_evaluations_async(lambda c: c.input)

        case_spans = [s for s in spans if s._test_span_name.startswith("execute_case ")]
        assert len(case_spans) == 1
        run_id = case_spans[0]._test_set_attributes_merged["test.suite.run.id"]
        parsed = uuid.UUID(run_id, version=4)
        assert str(parsed) == run_id


# ============================================================================
# Property 6: Backward-compatible construction produces identical evaluation results
# ============================================================================


@settings(max_examples=100)
@given(
    num_cases=st.integers(min_value=1, max_value=5),
    pass_pattern=st.lists(st.booleans(), min_size=1, max_size=5),
)
def test_property_6_backward_compatible_construction_produces_identical_evaluation_results(
    num_cases: int, pass_pattern: list[bool]
):
    """Property 6: Backward-compatible construction produces identical evaluation results.

    For any valid list of Cases and Evaluators, constructing an Experiment without the name
    parameter and running evaluations should produce the same EvaluationReport scores, passes,
    and reasons as the current implementation.

    Feature: otel-test-semantic-conventions, Property 6: Backward-compatible construction produces identical evaluation results

    **Validates: Requirements 6.1, 6.2**
    """
    # Align pass_pattern to num_cases
    effective_pattern = [pass_pattern[i % len(pass_pattern)] for i in range(num_cases)]

    cases = [
        Case(name=f"case_{i}", input=f"input_{i}", expected_output=f"input_{i}")
        for i in range(num_cases)
    ]

    # Create evaluator that returns results based on the pattern
    call_counter = [0]

    def make_evaluator():
        evaluator = MagicMock()
        evaluator.get_type_name.return_value = "MockEvaluator"

        def evaluate_side_effect(evaluation_case):
            idx = min(call_counter[0], len(effective_pattern) - 1)
            p = effective_pattern[idx]
            call_counter[0] += 1
            return [EvaluationOutput(score=1.0 if p else 0.0, test_pass=p, reason="mock reason")]

        evaluator.evaluate.side_effect = evaluate_side_effect
        evaluator.aggregator.side_effect = lambda outputs: (outputs[0].score, outputs[0].test_pass, outputs[0].reason)
        return evaluator

    def mock_start_span(name, **kwargs):
        return _make_mock_span()

    # Run with explicit name
    call_counter[0] = 0
    evaluator_named = make_evaluator()
    experiment_named = Experiment(cases=cases, evaluators=[evaluator_named], name="my_experiment")

    with patch.object(experiment_named._tracer, "start_as_current_span", side_effect=mock_start_span):
        reports_named = experiment_named.run_evaluations(lambda c: c.input)

    # Run without name (default)
    call_counter[0] = 0
    evaluator_default = make_evaluator()
    experiment_default = Experiment(cases=cases, evaluators=[evaluator_default])

    with patch.object(experiment_default._tracer, "start_as_current_span", side_effect=mock_start_span):
        reports_default = experiment_default.run_evaluations(lambda c: c.input)

    # Both should produce identical evaluation results
    assert len(reports_named) == len(reports_default)
    for r_named, r_default in zip(reports_named, reports_default):
        assert r_named.scores == r_default.scores, (
            f"Scores differ: {r_named.scores} vs {r_default.scores}"
        )
        assert r_named.test_passes == r_default.test_passes, (
            f"Passes differ: {r_named.test_passes} vs {r_default.test_passes}"
        )
        assert r_named.reasons == r_default.reasons, (
            f"Reasons differ: {r_named.reasons} vs {r_default.reasons}"
        )
        assert r_named.overall_score == r_default.overall_score, (
            f"Overall scores differ: {r_named.overall_score} vs {r_default.overall_score}"
        )
