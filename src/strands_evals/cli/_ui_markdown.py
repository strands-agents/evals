"""A report, an experiment and a session as Markdown, and the description of the server served at `/llms.txt`.

This is the only writer of that Markdown. A file is read at its address with `.md` added, and the page sends
what only it holds (one row, one case, a candidate's report, an experiment with unsaved edits) to be written.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any

_TICKS = re.compile("`+")
_SPACE = re.compile(r"[ \t\r\n]+")
_BREAKS = re.compile(r"[\r\n]+")

# Each pair of fields a report row or a case can carry, in the order the page shows them.
_ASPECTS = ("output", "trajectory", "interactions", "environment_state")

# The longest value written out whole, in characters. A longer one is cut and says so: the file has the rest.
VALUE_LIMIT = 4000

# The deepest a document may nest and still be written. Each level indents the levels inside it, so a small
# file nested thousands deep would be laid out as hundreds of megabytes before any value was cut.
DEPTH_LIMIT = 200

# Whether what is being written was sent by its holder and not read from the folder.
_POSTED: ContextVar[bool] = ContextVar("posted", default=False)


@contextmanager
def posted() -> Iterator[None]:
    """Write a document its sender holds: nothing is said of a file that has more of it, or of a scan."""
    token = _POSTED.set(True)
    try:
        yield
    finally:
        _POSTED.reset(token)


def value_text(value: Any) -> str:
    """A value as the page reads it: text as it is, anything structured as indented JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def _one_line(value: Any) -> str:
    return _SPACE.sub(" ", value_text(value)).strip(" ")


def _named(name: str) -> str:
    """The name of a file as it is, on one line: after a line break it would read as a heading."""
    return _BREAKS.sub(" ", name)


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _score(value: Any) -> str:
    """Two decimals, with a half rounded up as the page rounds it."""
    if not _is_number(value) or (isinstance(value, float) and not math.isfinite(value)):
        return "none"
    # A float is rounded from the number it stores, which is what the page shows. One too large to store exactly
    # is taken as written, so it does not print the digits it never had.
    number = Decimal(repr(value)) if isinstance(value, float) and abs(value) >= 2**53 else Decimal(value)
    with localcontext() as context:
        context.prec = max(28, number.adjusted() + 4)
        return str(number.quantize(Decimal("0.01"), ROUND_HALF_UP))


def _fenced(text: str, lang: str) -> str:
    """Text in a fenced block whose fence is longer than any run of backticks inside it."""
    ticks = "`" * max(3, 1 + max((len(run) for run in _TICKS.findall(text)), default=0))
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _capped(text: str, lang: str) -> list[str]:
    """Text in a fence, cut at `VALUE_LIMIT` characters with a line saying how much was left out."""
    if len(text) <= VALUE_LIMIT:
        return [_fenced(text, lang)]
    rest = "" if _POSTED.get() else ": the file has the rest"
    return [_fenced(text[:VALUE_LIMIT], lang), f"Cut after {VALUE_LIMIT} of {len(text)} characters{rest}."]


def _lang(value: Any) -> str:
    return "text" if isinstance(value, str) else "json"


def _value(value: Any) -> list[str]:
    return _capped(value_text(value), _lang(value))


def _whole(value: Any) -> str:
    """A document written out in full, however long: a diagnosis file is read whole or not at all."""
    return _fenced(value_text(value), _lang(value))


def _block(label: str, value: Any) -> list[str]:
    return [f"{label}:", *_value(value)] if _present(value) else []


def _cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _at(values: Any, index: int) -> Any:
    values = _list(values)
    return values[index] if index < len(values) else None


def _name(value: Any, index: int) -> str:
    return _one_line(value) if _present(value) else f"#{index}"


def _spans(value: Any) -> list | None:
    """The spans of a recorded session, trace after trace, or None when the value is not one."""
    if not isinstance(value, dict) or not isinstance(value.get("traces"), list):
        return None
    return [span for trace in value["traces"] for span in _list(_dict(trace).get("spans"))]


def _trajectory(label: str, value: Any) -> list[str]:
    spans = _spans(value)
    if spans is None:
        return _block(label, value)
    return [f"{label}: a recorded session of {_count(len(spans), 'span')}, kept in the report file."]


def _case_blocks(case: dict, sides: Sequence[str]) -> list[str]:
    """What a case asked for and, in a report, what came back."""
    parts = _block("Input", case.get("input")) + _block("Assertion", case.get("expected_assertion"))
    for aspect in _ASPECTS:
        for side in sides:
            label = f"{side.capitalize()} {aspect.replace('_', ' ')}"
            value = case.get(f"{side}_{aspect}")
            parts += _trajectory(label, value) if aspect == "trajectory" else _block(label, value)
    return parts


def _rows(doc: dict) -> list[dict]:
    """One row per case and evaluator, read as the page reads them.

    A case with no name is numbered by its place among the rows of its evaluator, so it has one name under each.
    """
    rows = []
    place: dict[str, int] = {}
    for index, case in enumerate(_list(doc.get("cases"))):
        case = _dict(case)
        evaluator = _one_line(case["evaluator"]) if _present(case.get("evaluator")) else "(unnamed)"
        place[evaluator] = place.get(evaluator, -1) + 1
        outputs = [_dict(output) for output in _list(_at(doc.get("detailed_results"), index))]
        declined = bool(outputs) and all(o.get("label") == "not_applicable" and o.get("test_pass") for o in outputs)
        rows.append(
            {
                "case": case,
                "name": _name(case.get("name"), place[evaluator]),
                "evaluator": evaluator,
                "score": _at(doc.get("scores"), index),
                "reason": _at(doc.get("reasons"), index) or "",
                "outputs": outputs,
                "verdict": "not applicable"
                if declined
                else "passed"
                if _at(doc.get("test_passes"), index)
                else "failed",
                "diagnosis": _at(doc.get("diagnoses"), index),
                "recommendation": _at(doc.get("recommendations"), index),
            }
        )
    return rows


def _row_parts(row: dict, more: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """A row under its heading: what the evaluator made of it, then the case it judged.

    `more` is facts about where the row is from, for a row written on its own.
    """
    outputs = row["outputs"]
    facts = [f"- Score: {_score(row['score'])}", f"- Verdict: {row['verdict']}"]
    only = outputs[0] if len(outputs) == 1 else {}
    if isinstance(only.get("label"), str) and only["label"]:
        facts.append(f"- Label: {_one_line(only['label'])}")
    several = outputs if len(outputs) > 1 else []
    for n, output in enumerate(several):
        label = output.get("label")
        facts.append(
            f"- Output {n + 1} of {len(outputs)}: score {_score(output.get('score'))}, "
            + ("passed" if output.get("test_pass") else "failed")
            + (f", {_one_line(label)}" if isinstance(label, str) and label else "")
        )
    judged = ["\n".join([*facts, *more])]
    judged += _block("Reason", row["reason"] if _present(row["reason"]) else only.get("reason"))
    for n, output in enumerate(several):
        judged += _block(f"Reason, output {n + 1} of {len(outputs)}", output.get("reason"))
    judged += _block("Diagnosis", row["diagnosis"]) + _block("Recommendation", row["recommendation"])
    return judged, _case_blocks(row["case"], ("expected", "actual"))


def _failing_parts(rows: Sequence[dict], heading: str) -> list[str]:
    """Each failing row in full. A case that fails under several evaluators is written out under the first only."""
    parts: list[str] = []
    written: dict[tuple[str, ...], str] = {}
    for row in rows:
        if row["verdict"] != "failed":
            continue
        title = f"{row['name']} · {row['evaluator']}"
        judged, case = _row_parts(row)
        first = written.setdefault((row["name"], *case), title)
        if case and first != title:
            case = [f'The case is the same as under "{first}" above.']
        parts += [f"{heading} {title}", *judged, *case]
    return parts


def _document(parts: Iterable[str]) -> str:
    return "\n\n".join(parts) + "\n"


def _made_from(experiment: str | None, link: dict | None) -> str:
    """The experiment a report was paired with and how well it fits, or why it has none, as the scan found it."""
    if experiment:
        return _named(experiment) + (f" ({link['status']}: {_named(link['detail'])})" if link else "")
    if link:
        return f"not paired: {_named(link['detail'])}"
    return "none found in this folder; a report made in code has none"


def report_markdown(path: str, doc: dict, experiment: str | None, link: dict | None = None) -> str:
    """A report: its figures, each failing row in full, then every row in one table.

    `experiment` and `link` are the scan's pairing of it: the experiment's path, and the status of the fit.
    """
    rows = _rows(doc)
    verdicts = [row["verdict"] for row in rows]
    facts = [
        f"- Overall score: {_score(doc.get('overall_score'))}",
        f"- Rows: {len(rows)} ({verdicts.count('passed')} passed, {verdicts.count('failed')} failed, "
        f"{verdicts.count('not applicable')} not applicable)",
        f"- Cases: {len({row['name'] for row in rows})}",
        f"- Evaluators: {', '.join(dict.fromkeys(row['evaluator'] for row in rows)) or 'none'}",
    ]
    # A report that was sent with no pairing was not looked for in a scan, so nothing is said of one.
    if experiment or link or not _POSTED.get():
        facts.append(f"- Experiment: {_made_from(experiment, link)}")
    failing = _failing_parts(rows, "###")
    table = ["| Case | Evaluator | Score | Verdict |", "| --- | --- | --- | --- |"]
    table += [
        f"| {_cell(row['name'])} | {_cell(row['evaluator'])} | {_score(row['score'])} | {row['verdict']} |"
        for row in rows
    ]
    return _document(
        [f"# Report: {_named(path)}", "\n".join(facts), "## Failing rows", *(failing or ["None."])]
        + ["## All rows", "\n".join(table)]
    )


def row_markdown(doc: dict, index: int, report: str, experiment: str | None) -> str | None:
    """One row of a report on its own, or None when the report has no such row.

    `report` and `experiment` are how to name the two files to the reader.
    """
    rows = _rows(doc)
    if not 0 <= index < len(rows):
        return None
    row = rows[index]
    where = [f"- Report: `{_named(report)}`", *([f"- Experiment: `{_named(experiment)}`"] if experiment else [])]
    judged, case = _row_parts(row, where)
    return _document([f"# {row['name']} · {row['evaluator']}", *judged, *case])


def failing_prompt(doc: dict, report: str, experiment: str | None, command: str | None) -> str:
    """Every failing row of a report, under what can be said of them: what failed, where the files are, and
    `command`, which runs one of the cases again. Which agent was evaluated is in neither file, so it is not named.
    """
    rows = _rows(doc)
    failed = sum(row["verdict"] == "failed" for row in rows)
    facts = [f"- Report: `{_named(report)}`"]
    if experiment:
        facts.append(f"- Experiment, which holds the cases and the evaluators: `{_named(experiment)}`")
    if command:
        facts.append(
            f"- To run one case again: `{_named(command)}`. "
            "`MODULE:ATTR` is a placeholder: put the import path of the agent under test there."
        )
    return _document(
        [
            f"{'This row fails' if failed == 1 else f'These {failed} rows fail'} in an evaluation report. "
            "Each is below with what was asked, what was expected, what came back and the evaluator's reason. "
            "Find why each fails. Neither file names the agent that was evaluated: the cause is in that agent, "
            "or in the case or the evaluator where the expectation itself is wrong.",
            "\n".join(facts),
            *_failing_parts(rows, "##"),
        ]
    )


def _evaluator_parts(evaluator: dict, index: int) -> list[str]:
    kind, name = evaluator.get("evaluator_type"), evaluator.get("name")
    title = _name(name if _present(name) else kind, index)
    if _present(name) and _present(kind):
        title += f" ({_one_line(kind)})"
    settings = {key: value for key, value in evaluator.items() if key not in ("evaluator_type", "name")}
    return [f"### {title}", *(_block("Settings", settings) if settings else [])]


def _case_parts(case: dict, index: int, heading: str) -> list[str]:
    return [
        f"{heading} {_name(case.get('name'), index)}",
        *_case_blocks(case, ("expected",)),
        *_block("Metadata", case.get("metadata")),
    ]


def experiment_markdown(path: str, doc: dict, reports: Sequence[str]) -> str:
    """An experiment: its evaluators with their settings, then every case."""
    cases = [_dict(case) for case in _list(doc.get("cases"))]
    evaluators = [e for e in _list(doc.get("evaluators")) if isinstance(e, dict)]
    facts = [
        f"- Cases: {len(cases)}",
        f"- Evaluators: {len(evaluators)}",
        f"- Reports: {', '.join(_named(made) for made in sorted(reports)) or 'none found in this folder'}",
    ]
    parts = [f"# Experiment: {_named(path)}", "\n".join(facts), "## Evaluators"]
    parts += [part for i, e in enumerate(evaluators) for part in _evaluator_parts(e, i)] or ["None."]
    parts.append("## Cases")
    listed = [part for i, case in enumerate(cases) for part in _case_parts(case, i, "###")]
    return _document(parts + (listed or ["None."]))


def case_markdown(case: Any, index: int, experiment: str) -> str:
    """One case of an experiment on its own. `index` names it when it has no name of its own."""
    heading, *blocks = _case_parts(_dict(case), index, "#")
    return _document([heading, f"- Experiment: `{_named(experiment)}`", *blocks])


# ── a session, and a trace on its own ──


def _round(value: float) -> int:
    """To the nearest whole number, with a half rounded up as the page rounds it."""
    return math.floor(value + 0.5)


def _nice(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(_round(value)) if abs(value) >= 100 else f"{_round(value * 100) / 100:.2f}"


def _compact(n: float) -> str:
    if abs(n) >= 1e6:
        return f"{_score(n / 1e6)}M"
    return f"{_round(n / 1e3)}k" if abs(n) >= 1e4 else str(_round(n))


def _not_json(constant: str) -> Any:
    raise ValueError(f"{constant} is not JSON")


def _tool_output(value: Any) -> str:
    """Tool output as the page reads it: JSON in a string is laid out, and a lone `{"output": "..."}` is its text."""
    if not isinstance(value, str):
        return json.dumps(value, indent=1, ensure_ascii=False)
    text = value.strip()
    if not text or text[0] not in "[{" or text[-1] not in "]}":
        return value
    try:
        parsed = json.loads(text, parse_constant=_not_json)
    except (ValueError, RecursionError):
        return value
    if isinstance(parsed, dict) and len(parsed) == 1 and isinstance(parsed.get("output"), str):
        return parsed["output"]
    return json.dumps(parsed, indent=1, ensure_ascii=False)


def _call(name: Any, arguments: Any) -> list[str]:
    # What a session names is written outside a fence, where a line break could start a heading.
    named = _one_line(name) if _present(name) else "?"
    return [f"Tool call `{named}`:", *_capped(json.dumps(arguments, indent=2, ensure_ascii=False), "json")]


def _result(text: str, status: Any) -> list[str]:
    return [f"Tool result{f' ({_one_line(status)})' if _present(status) else ''}:", *_capped(text, "text")]


def _span_result(result: dict) -> list[str]:
    error = result.get("error")
    text = error if isinstance(error, str) else _tool_output(result.get("content"))
    return _result(text, "error" if error else None)


def _content(block: Any) -> list[str]:
    """One block of a message, in either vocabulary: the Strands blocks or a session's `content_type` ones."""
    if not isinstance(block, dict):
        return _capped(value_text(block), "text" if isinstance(block, str) or block is None else "json")
    if isinstance(block.get("text"), str):
        return _value(block["text"])
    if block.get("toolUse"):
        use = _dict(block["toolUse"])
        return _call(use.get("name"), use.get("input"))
    if block.get("content_type") == "tool_use":
        return _call(block.get("name"), block.get("arguments"))
    if block.get("toolResult"):
        result = _dict(block["toolResult"])
        parts = []
        for item in _list(result.get("content")):
            if isinstance(item, dict):
                item = item["json"] if "json" in item else item["text"] if isinstance(item.get("text"), str) else item
            parts.append(_tool_output(item))
        status = result.get("status")
        return _result("\n".join(parts), status if status != "success" else None)
    if block.get("content_type") == "tool_result":
        return _span_result(block)
    if block.get("reasoningContent"):
        text = _dict(_dict(block["reasoningContent"]).get("reasoningText")).get("text")
        return ["Reasoning:", *_capped(text if isinstance(text, str) else "", "text")]
    return _value(block)


def _is_count(value: Any) -> bool:
    """A number a float can hold. A longer integer, like NaN and Infinity, is no count of anything."""
    return _is_number(value) and abs(value) <= sys.float_info.max


def _usage(usage: Any) -> list[str]:
    if not isinstance(usage, dict):
        return []
    parts = [
        f"{_compact(usage[key])} {word}"
        for key, word in (("inputTokens", "in"), ("outputTokens", "out"))
        if _is_count(usage.get(key))
    ]
    cached = usage.get("cacheReadInputTokens")
    if _is_count(cached) and cached:
        parts.append(f"{_compact(cached)} cache read")
    return [" · ".join(parts) + " tokens"] if parts else []


def _message(message: Any, heading: str) -> list[str]:
    message = _dict(message)
    content = message.get("content")
    if not isinstance(content, list):
        content = [{"text": value_text(content)}]
    role = message.get("role")
    parts = [
        f"{heading} {_one_line(role) if _present(role) else 'unknown'}",
        *_usage(_dict(message.get("metadata")).get("usage")),
    ]
    return parts + [part for block in content for part in _content(block)]


def _span(span: Any, index: int, heading: str) -> list[str]:
    """One span under a heading that carries its `span_id`, which is what a diagnosis names it by."""
    held = _dict(span)
    kind = held.get("span_type") or "span"
    span_id = _dict(held.get("span_info")).get("span_id")
    named = f" (span_id: {_one_line(span_id)})" if _present(span_id) else ""

    def head(name: str) -> str:
        return f"{heading} Span {index}: {name}{named}"

    if kind == "inference" and isinstance(held.get("messages"), list):
        return [head("inference")] + [part for message in held["messages"] for part in _message(message, f"{heading}#")]
    if kind == "execute_tool":
        call = _dict(held.get("tool_call"))
        return [
            head("tool"),
            *_call(call.get("name"), call.get("arguments")),
            *_span_result(_dict(held.get("tool_result"))),
        ]
    if kind == "invoke_agent":
        return [
            head("invoke agent"),
            *_block("System prompt", held.get("system_prompt")),
            *_block("Prompt", held.get("user_prompt")),
            *_block("Response", held.get("agent_response")),
        ]
    return [head(_one_line(kind)), *_value(span)]


def session_markdown(path: str, doc: dict, seconds: float | None, diagnoses: Sequence[tuple[str, Any]]) -> str:
    """A session: what it holds, each diagnosis made of it as `(path, document)`, then every trace span by span."""
    traces = [_list(_dict(trace).get("spans")) for trace in _list(doc.get("traces"))]
    facts = [
        f"- Session: {_one_line(doc.get('session_id'))}",
        f"- Traces: {len(traces)}",
        f"- Spans: {sum(map(len, traces))}",
    ]
    if seconds is not None:
        facts.append(f"- Time: {_nice(seconds)}s")
    facts.append(f"- Diagnoses: {', '.join(_named(made) for made, _ in diagnoses) or 'none found'}")
    parts = [f"# Session: {_named(path)}", "\n".join(facts)]
    for made, diagnosis in diagnoses:
        parts += [f"## Diagnosis: {_named(made)}", _whole(diagnosis)]
    first = 0
    for at, spans in enumerate(traces):
        trace_id = _dict(doc["traces"][at]).get("trace_id")
        named = f": {_one_line(trace_id)}" if _present(trace_id) else ""
        parts.append(f"## Trace {at + 1} of {len(traces)}{named}")
        parts += [part for i, span in enumerate(spans) for part in _span(span, first + i, "###")] or ["No spans."]
        first += len(spans)
    return _document(parts)


def trace_markdown(doc: Any, cost: str | None = None) -> str:
    """A recorded session or a list of messages on its own, as the conversation it was.

    `cost` is what the page says it took, for the line under the heading.
    """
    said = [_named(cost)] if cost else []
    spans = _spans(doc)
    if spans is not None:
        parts = [f"# Trace: {_count(len(spans), 'span')} of a recorded session", *said]
        return _document(parts + [part for i, span in enumerate(spans) for part in _span(span, i, "##")])
    if isinstance(doc, list):
        parts = [f"# Trace: {_count(len(doc), 'message')}", *said]
        return _document(
            parts + [part for i, message in enumerate(doc) for part in _message(message, f"## Message {i + 1}:")]
        )
    return _document(["# Trace", *_value(doc)])


def diagnosis_markdown(path: str, doc: dict, session: str | None) -> str:
    """A diagnosis as it was written. `session` is the file of the session it was made from, when that was found."""
    session_id = doc.get("session_id")
    where = f", in {_named(session)}" if session else ", which is not in the folder"
    facts = [
        f"- Session: {_one_line(session_id) + where if _present(session_id) else 'not named'}",
        f"- Failures: {len(_list(doc.get('failures')))}",
        f"- Root causes: {len(_list(doc.get('root_causes')))}",
    ]
    return _document([f"# Diagnosis: {_named(path)}", "\n".join(facts), _whole(doc)])


# What sits under each root of the page's addresses. `ui-route.js` is where they are parsed.
PAGE_ADDRESSES = {
    "/reports": [
        ("", "every report file found"),
        ("/<report file>", "one report"),
        ("/<report file>/rows/<case>/<evaluator>", "one row of it, opened and in view"),
        ("/<report file>/rows/<case>", "the same, for a row whose case has no evaluator named"),
    ],
    "/experiments": [
        ("", "every experiment file found"),
        ("/<experiment file>", "one experiment, in the cases editor"),
        ("/<experiment file>/cases/<name>", "the same, on one case"),
    ],
    "/sessions": [
        ("", "every recorded session found, and each diagnosis whose session is not in the folder"),
        ("/<session file>", "one session, trace by trace, under the diagnoses made of it"),
    ],
    "/optimization": [
        ("", "the optimizer runs found"),
        (
            "/<run directory>/<view>",
            "one run; the view is result, setup, iterations, candidates or trade-off",
        ),
        ("/<run directory>/candidates/<id>", "one candidate of it, opened and in view"),
        ("/<run directory>/candidates/<id>/report", "the cases that candidate ran, as a report"),
    ],
}

_FORM_HOLDS = {
    ("report", ".md"): "its figures, each failing row with its input, expected and actual output and the "
    "evaluator's reason, then every row",
    ("experiment", ".md"): "its evaluators with their settings, then every case",
    ("session", ".md"): "what it recorded, each diagnosis made of it, then every span of every trace: the prompts, "
    "the model's messages, the tool calls and their results",
    ("report", ".json"): "the report file's document",
    ("experiment", ".json"): "the experiment file's document",
    ("session", ".json"): "the session file's document",
}


def llms_text(
    routes: dict[str, str], page_roots: Sequence[str], data_roots: dict[str, str], data_forms: Sequence[str]
) -> str:
    """What this server answers, for a reader that is a program: `routes` maps each route to what it returns."""
    forms = [
        f"- `{root}<{kind} file>{form}`: {_FORM_HOLDS[kind, form]}"
        for form in data_forms
        for root, kind in data_roots.items()
    ]
    pages = [
        f"- `{root}{rest}`: {holds}" for root in page_roots for rest, holds in PAGE_ADDRESSES.get(root, [("", "")])
    ]
    return _document(
        [
            "# strands-evals ui",
            "> A local page over a folder of strands-evals experiments, reports, recorded sessions and optimizer runs. "
            "What it shows can be read without a browser, as Markdown or as JSON.",
            "This description, the Markdown and JSON forms, `POST /api/markdown`, the shape of `/api/scan` and the "
            "prompt the page copies are experimental: they may change before a release.",
            "## Token",
            "Every address here needs this server's token, sent as the `X-Evals-Token` header or as `?t=<token>`. "
            "The token is in the link the server printed when it started. The cookie a browser holds opens "
            "the page only: it does not open the Markdown, the JSON or any `/api` route.",
            "## Markdown and JSON",
            "A report, an experiment or a session is read by adding a form to its page address. A file is named "
            "by its path under the served folder, which ends in `.json`; `/api/scan` lists them. A diagnosis file "
            "is read the same way, under `/sessions/`.",
            "\n".join(forms),
            f"In Markdown a value longer than {VALUE_LIMIT} characters is cut, with a line saying how much was left "
            "out: the JSON form has all of it. A report writes a case that fails under several evaluators once. "
            "Every row and every span is written, however many there are. A diagnosis file is written whole; "
            "one held in a report row is cut like any other value. "
            f"A document nested more than {DEPTH_LIMIT} levels deep is not written: its Markdown answers 422.",
            "`POST /api/markdown` writes the same Markdown for a document sent to it. The page uses it for what "
            "has no file of its own: one row, one case, the failing rows as a prompt, a trace inside a report, a "
            "candidate's report, and an experiment with edits that are not saved.",
            "## Routes",
            "\n".join(f"- `{route}`: {holds}" for route, holds in routes.items()),
            "## Page addresses",
            "These answer with the page itself, for a browser. Each one names one thing, so a link to it "
            "can be pasted into a conversation. Asked for without `text/html` in `Accept`, one answers 406 with "
            "a few lines that name the forms above.",
            "\n".join(pages),
        ]
    )
