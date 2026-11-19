"""Tests for the Markdown `strands-evals ui` writes: values must read exactly as the page prints them."""

from __future__ import annotations

import json

import pytest

from strands_evals.cli._ui_markdown import (
    experiment_markdown,
    report_markdown,
    session_markdown,
    stringify,
    value_text,
)


@pytest.mark.parametrize(
    "written,printed",
    [
        ("1.0", "1"),
        ("-0.0", "0"),
        ("100", "100"),
        ("0.30000000000000004", "0.30000000000000004"),
        ("123.456", "123.456"),
        ("1e20", "100000000000000000000"),
        ("1e21", "1e+21"),
        ("1.5e300", "1.5e+300"),
        ("0.000001", "0.000001"),
        ("1e-7", "1e-7"),
        ("-2.5e-10", "-2.5e-10"),
        ("9007199254740993", "9007199254740992"),
        ("12345678901234567890", "12345678901234567000"),
        ("1e400", "null"),
        ("NaN", "null"),
        ("true", "true"),
        ("null", "null"),
        (r'"\u00e9 \u0001 \" \\ \n \ud83d\ude00 \ud800"', r'"é \u0001 \" \\ \n 😀 \ud800"'),
        ("[]", "[]"),
        ("{}", "{}"),
        ("[1, [], {}]", "[\n  1,\n  [],\n  {}\n]"),
        (
            '{"b": 1, "10": 2, "2": 3, "01": 4, "-1": 5}',
            '{\n  "2": 3,\n  "10": 2,\n  "b": 1,\n  "01": 4,\n  "-1": 5\n}',
        ),
        ('{"a": {"b": [true]}}', '{\n  "a": {\n    "b": [\n      true\n    ]\n  }\n}'),
    ],
)
def test_stringify_prints_a_value_as_javascript_does(written, printed):
    """Each expectation is what `JSON.stringify(JSON.parse(written), null, 2)` gives in a browser."""
    assert stringify(json.loads(written)) == printed


def test_value_text_leaves_text_alone_and_prints_nothing_for_none():
    assert [value_text("as written\n"), value_text(None), value_text({"a": 1})] == [
        "as written\n",
        "",
        '{\n  "a": 1\n}',
    ]


@pytest.mark.parametrize(
    "score,printed",
    [
        (0.125, "0.13"),
        (0.135, "0.14"),
        (1.005, "1.00"),
        (0.005, "0.01"),
        (-0.001, "-0.00"),
        (1, "1.00"),
        (None, "none"),
    ],
)
def test_a_score_is_rounded_as_the_page_rounds_it(score, printed):
    """Each expectation is `score.toFixed(2)` in a browser, which rounds the exact binary value half up."""
    text = report_markdown("r.json", {"overall_score": score, "cases": []}, None)

    assert f"- Overall score: {printed}\n" in text


@pytest.mark.parametrize(
    "experiment,link,printed",
    [
        (
            "e.json",
            {"status": "partial", "detail": "1 of 3 report case(s) are not in this experiment"},
            "e.json (partial: 1 of 3 report case(s) are not in this experiment)",
        ),
        (
            None,
            {"status": "tie", "detail": "2 experiments hold every case of this report and are equally near: a, b"},
            "not paired: 2 experiments hold every case of this report and are equally near: a, b",
        ),
        (None, None, "none found in this folder; a report made in code has none"),
    ],
    ids=["paired", "a tie", "none found"],
)
def test_a_report_says_how_well_its_experiment_fits_and_why_it_has_none(experiment, link, printed):
    text = report_markdown("r.json", {"overall_score": 1, "cases": []}, experiment, link)

    assert f"- Experiment: {printed}\n\n## Failing rows" in text


def test_a_value_cannot_close_the_fence_it_is_written_in():
    report = {
        "cases": [{"name": "a", "evaluator": "E", "input": "see ```` four", "actual_output": {"code": "```"}}],
        "scores": [0],
        "test_passes": [False],
    }

    text = report_markdown("r.json", report, None)

    assert "Input:\n\n`````text\nsee ```` four\n`````\n" in text
    assert 'Actual output:\n\n````json\n{\n  "code": "```"\n}\n````\n' in text


def test_a_name_stays_on_one_line_and_cannot_break_the_table():
    report = {"cases": [{"name": "a|b\nc", "evaluator": " two\t words "}], "scores": [1], "test_passes": [True]}

    text = report_markdown("r.json", report, None)

    assert text.endswith("| a\\|b c | two words | 1.00 | passed |\n")
    assert "- Evaluators: two words\n" in text


def test_a_row_is_not_applicable_only_when_every_output_declined_and_passed():
    declined = {"score": 1, "test_pass": True, "label": "not_applicable"}
    report = {
        "cases": [{"name": name, "evaluator": "E"} for name in ("declined", "could not judge", "mixed")],
        "scores": [1, 0, 1],
        "test_passes": [True, False, True],
        "detailed_results": [
            [declined],
            [{**declined, "test_pass": False}],
            [declined, {"score": 1, "test_pass": True}],
        ],
    }

    text = report_markdown("r.json", report, None)

    assert "- Rows: 3 (1 passed, 1 failed, 1 not applicable)\n" in text
    assert text.endswith(
        "| declined | E | 1.00 | not applicable |\n"
        "| could not judge | E | 0.00 | failed |\n"
        "| mixed | E | 1.00 | passed |\n"
    )


def test_an_experiment_with_nothing_in_it_says_so():
    assert experiment_markdown("e.json", {"cases": [], "evaluators": []}, []) == (
        "# Experiment: e.json\n\n- Cases: 0\n- Evaluators: 0\n- Reports: none found in this folder\n\n"
        "## Evaluators\n\nNone.\n\n## Cases\n\nNone.\n"
    )


@pytest.mark.parametrize(
    "content,printed",
    [
        ("in transit", "in transit"),
        ('{"status": "in transit", "items": [1, 2.0]}', '{\n "status": "in transit",\n "items": [\n  1,\n  2\n ]\n}'),
        ('\ufeff {"output": "in transit"}\n', "in transit"),
        ('{"output": 7}', '{\n "output": 7\n}'),
        ('{"output": "a", "more": 1}', '{\n "output": "a",\n "more": 1\n}'),
        ('{"half": ', '{"half": '),
        ('{"n": NaN}', '{"n": NaN}'),
        ("[] and then some", "[] and then some"),
        ({"status": ["in transit"]}, '{\n "status": [\n  "in transit"\n ]\n}'),
        (None, "null"),
    ],
)
def test_tool_output_reads_as_the_page_reads_it(content, printed):
    """Each expectation is what `blockText` in `assets/ui-trace.js` gives for the same tool result."""
    span = {"span_type": "execute_tool", "tool_call": {"name": "lookup"}, "tool_result": {"content": content}}

    text = session_markdown("s.json", {"session_id": "s", "traces": [{"spans": [span]}]}, None, [])

    assert text.endswith(f"Tool result:\n\n```text\n{printed}\n```\n")


@pytest.mark.parametrize(
    "seconds,printed",
    [(4, "- Time: 4s\n"), (4.25, "- Time: 4.25s\n"), (1.005, "- Time: 1.00s\n"), (123.5, "- Time: 124s\n"), (None, "")],
)
def test_a_session_gives_its_time_as_the_page_prints_it(seconds, printed):
    """Each expectation is `nice(seconds)` in a browser; a session whose spans carry no times has no line for it."""
    text = session_markdown("s.json", {"session_id": "s", "traces": []}, seconds, [])

    assert text == f"# Session: s.json\n\n- Session: s\n- Traces: 0\n- Spans: 0\n{printed}- Diagnoses: none found\n"


@pytest.mark.parametrize(
    "usage,printed",
    [
        ({"inputTokens": 9999.5, "outputTokens": 0}, "10000 in · 0 out tokens"),
        ({"inputTokens": 12345, "outputTokens": 1234567}, "12k in · 1.23M out tokens"),
        ({"inputTokens": 5, "cacheReadInputTokens": 0}, "5 in tokens"),
        ({"outputTokens": 2, "cacheReadInputTokens": 20500}, "2 out · 21k cache read tokens"),
    ],
)
def test_token_counts_are_shortened_as_the_page_shortens_them(usage, printed):
    """Each expectation is `usageLine` in `assets/ui-trace.js` for the same usage."""
    message = {"role": "assistant", "content": [], "metadata": {"usage": usage}}
    span = {"span_type": "inference", "messages": [message]}

    text = session_markdown("s.json", {"session_id": "s", "traces": [{"spans": [span]}]}, None, [])

    assert text.endswith(f"#### assistant\n\n{printed}\n")
