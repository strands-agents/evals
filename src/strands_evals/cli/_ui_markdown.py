"""A report, an experiment and a session as Markdown, and the description of the server served at `/llms.txt`.

The page writes the same Markdown for its copy actions (`assets/ui-markdown.js`), and
`tests/ui/markdown.mjs` holds the two to the same text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_INDEX_KEY = re.compile(r"0|[1-9][0-9]*")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
_TICKS = re.compile("`+")
_SPACE = re.compile(r"[ \t\r\n]+")

# Each pair of fields a report row or a case can carry, in the order the page shows them.
_ASPECTS = ("output", "trajectory", "interactions", "environment_state")


def _truthy(value: Any) -> bool:
    """Whether JavaScript takes a parsed JSON value as true."""
    if value is None or value is False or value == "":
        return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value == value and value != 0
    return True


def _number(value: float) -> str:
    """A number as JavaScript prints it: whole numbers without a point, exponents only past 1e21 and 1e-7."""
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    _, digits, exponent = Decimal(repr(abs(value))).normalize().as_tuple()
    text = "".join(map(str, digits))
    point = len(text) + int(exponent)
    if len(text) <= point <= 21:
        body = text + "0" * (point - len(text))
    elif 0 < point <= 21:
        body = f"{text[:point]}.{text[point:]}"
    elif -6 < point <= 0:
        body = "0." + "0" * -point + text
    else:
        fraction = f".{text[1:]}" if len(text) > 1 else ""
        body = f"{text[0]}{fraction}e{'+' if point > 0 else '-'}{abs(point - 1)}"
    return ("-" if value < 0 else "") + body


def _key_order(obj: dict) -> list[str]:
    """Keys in the order JavaScript keeps them: array indices first, ascending, then as written."""
    index = sorted((key for key in obj if _INDEX_KEY.fullmatch(key) and int(key) < 2**32 - 1), key=int)
    listed = set(index)
    return index + [key for key in obj if key not in listed]


def stringify(value: Any, indent: str = "", step: str = "  ") -> str:
    """`JSON.stringify(value, null, 2)`, character for character. `step` is the indent of one level."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", json.dumps(value, ensure_ascii=False))
    if isinstance(value, int):
        if abs(value) < 2**53:
            return str(value)
        try:
            return _number(float(value))
        except OverflowError:
            return "null"
    if isinstance(value, float):
        return _number(value)
    inner = indent + step
    if isinstance(value, dict):
        items = [f"{inner}{stringify(key)}: {stringify(value[key], inner, step)}" for key in _key_order(value)]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}" if items else "{}"
    items = [inner + stringify(item, inner, step) for item in value]
    return "[\n" + ",\n".join(items) + f"\n{indent}]" if items else "[]"


def value_text(value: Any) -> str:
    """A value as the page reads it: text as it is, anything structured as indented JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else stringify(value)


def _one_line(value: Any) -> str:
    return _SPACE.sub(" ", value_text(value)).strip(" ")


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _score(value: Any) -> str:
    """Two decimals, rounded as JavaScript's `toFixed(2)` rounds."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return "none"
    return ("-" if value < 0 else "") + str(Decimal(abs(value)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def _fenced(text: str, lang: str) -> str:
    """Text in a fenced block whose fence is longer than any run of backticks inside it."""
    ticks = "`" * max(3, 1 + max((len(run) for run in _TICKS.findall(text)), default=0))
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _fence(value: Any) -> str:
    return _fenced(value_text(value), "text" if isinstance(value, str) else "json")


def _block(label: str, value: Any) -> list[str]:
    return [f"{label}:", _fence(value)] if _present(value) else []


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


def _session_spans(value: Any) -> int | None:
    """How many spans a recorded session holds, or None when the value is not one."""
    if not isinstance(value, dict) or not isinstance(value.get("traces"), list):
        return None
    return sum(len(t["spans"]) for t in value["traces"] if isinstance(t, dict) and isinstance(t.get("spans"), list))


def _trajectory(label: str, value: Any) -> list[str]:
    spans = _session_spans(value)
    if spans is None:
        return _block(label, value)
    return [f"{label}: a recorded session of {_count(spans, 'span')}, kept in the report file."]


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
        declined = bool(outputs) and all(
            o.get("label") == "not_applicable" and _truthy(o.get("test_pass")) for o in outputs
        )
        passed = _truthy(_at(doc.get("test_passes"), index))
        reason = _at(doc.get("reasons"), index)
        rows.append(
            {
                "case": case,
                "name": _name(case.get("name"), place[evaluator]),
                "evaluator": evaluator,
                "score": _at(doc.get("scores"), index),
                "reason": reason if _truthy(reason) else "",
                "outputs": outputs,
                "verdict": "not applicable" if declined else "passed" if passed else "failed",
                "diagnosis": _at(doc.get("diagnoses"), index),
                "recommendation": _at(doc.get("recommendations"), index),
            }
        )
    return rows


def _row_parts(row: dict, heading: str) -> list[str]:
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
            + ("passed" if _truthy(output.get("test_pass")) else "failed")
            + (f", {_one_line(label)}" if isinstance(label, str) and label else "")
        )
    parts = [f"{heading} {row['name']} · {row['evaluator']}", "\n".join(facts)]
    parts += _block("Reason", row["reason"] if _present(row["reason"]) else only.get("reason"))
    for n, output in enumerate(several):
        parts += _block(f"Reason, output {n + 1} of {len(outputs)}", output.get("reason"))
    parts += _block("Diagnosis", row["diagnosis"]) + _block("Recommendation", row["recommendation"])
    return parts + _case_blocks(row["case"], ("expected", "actual"))


def _document(parts: Iterable[str]) -> str:
    return "\n\n".join(parts) + "\n"


def _made_from(experiment: str | None, link: dict | None) -> str:
    """The experiment a report was paired with and how well it fits, or why it has none, as the scan found it."""
    if experiment:
        return experiment + (f" ({link['status']}: {link['detail']})" if link else "")
    return f"not paired: {link['detail']}" if link else "none found in this folder; a report made in code has none"


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
        f"- Experiment: {_made_from(experiment, link)}",
    ]
    failing = [part for row in rows if row["verdict"] == "failed" for part in _row_parts(row, "###")]
    table = ["| Case | Evaluator | Score | Verdict |", "| --- | --- | --- | --- |"]
    table += [
        f"| {_cell(row['name'])} | {_cell(row['evaluator'])} | {_score(row['score'])} | {row['verdict']} |"
        for row in rows
    ]
    return _document(
        [f"# Report: {path}", "\n".join(facts), "## Failing rows", *(failing or ["None."])]
        + ["## All rows", "\n".join(table)]
    )


def _evaluator_parts(evaluator: dict, index: int) -> list[str]:
    kind, name = evaluator.get("evaluator_type"), evaluator.get("name")
    title = _name(name if _present(name) else kind, index)
    if _present(name) and _present(kind):
        title += f" ({_one_line(kind)})"
    settings = {key: value for key, value in evaluator.items() if key not in ("evaluator_type", "name")}
    return [f"### {title}", *(_block("Settings", settings) if settings else [])]


def experiment_markdown(path: str, doc: dict, reports: Sequence[str]) -> str:
    """An experiment: its evaluators with their settings, then every case."""
    cases = [_dict(case) for case in _list(doc.get("cases"))]
    evaluators = [e for e in _list(doc.get("evaluators")) if isinstance(e, dict)]
    facts = [
        f"- Cases: {len(cases)}",
        f"- Evaluators: {len(evaluators)}",
        f"- Reports: {', '.join(sorted(reports)) or 'none found in this folder'}",
    ]
    parts = [f"# Experiment: {path}", "\n".join(facts), "## Evaluators"]
    parts += [part for i, e in enumerate(evaluators) for part in _evaluator_parts(e, i)] or ["None."]
    parts.append("## Cases")
    listed = [part for i, case in enumerate(cases) for part in _case_parts(case, i, "###")]
    return _document(parts + (listed or ["None."]))


def _case_parts(case: dict, index: int, heading: str) -> list[str]:
    return [
        f"{heading} {_name(case.get('name'), index)}",
        *_case_blocks(case, ("expected",)),
        *_block("Metadata", case.get("metadata")),
    ]


# ── a session ── `mdSpan`, `mdMessage`, `mdContent` and the helpers they read in `assets/ui-markdown.js`,
# `assets/ui-trace.js` and `assets/ui-core.js`, line for line.

# What JavaScript's `trim` strips.
_JS_SPACE = "\t\n\x0b\x0c\r \xa0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff" + "".join(map(chr, range(0x2000, 0x200B)))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _as_text(value: Any) -> str:
    """`String(value)` for a value that is not a list."""
    if isinstance(value, dict):
        return "[object Object]"
    return value if isinstance(value, str) else stringify(value)


def _round(value: float) -> int:
    return math.floor(value + 0.5)


def _nice(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(_round(value)) if abs(value) >= 100 else f"{_round(value * 100) / 100:.2f}"


def _compact(n: float) -> str:
    if abs(n) >= 1e6:
        return f"{_score(n / 1e6)}M"
    return f"{_round(n / 1e3)}k" if abs(n) >= 1e4 else str(_round(n))


def _no_constant(name: str) -> Any:
    raise ValueError(name)


def _block_text(value: Any) -> str:
    """Tool output as the page reads it: JSON in a string is laid out, and a lone `{"output": "..."}` is its text."""
    if not isinstance(value, str):
        return stringify(value, step=" ")
    text = value.strip(_JS_SPACE)
    if not text or text[0] not in "[{" or text[-1] not in "]}":
        return value
    try:
        parsed = json.loads(text, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        return value
    if isinstance(parsed, dict) and len(parsed) == 1 and isinstance(parsed.get("output"), str):
        return parsed["output"]
    return stringify(parsed, step=" ")


def _call(name: Any, arguments: Any) -> list[str]:
    return [f"Tool call `{_as_text(name) if _truthy(name) else '?'}`:", _fenced(stringify(arguments), "json")]


def _result(text: str, status: Any) -> list[str]:
    return [f"Tool result{f' ({_as_text(status)})' if _truthy(status) else ''}:", _fenced(text, "text")]


def _span_result(result: dict) -> list[str]:
    error = result.get("error")
    text = error if isinstance(error, str) else _block_text(result.get("content"))
    return _result(text, "error" if _truthy(error) else None)


def _content(block: Any) -> list[str]:
    """One block of a message, in either vocabulary: the Strands blocks or a session's `content_type` ones."""
    if not isinstance(block, dict | list):
        return [_fenced("" if block is None else _as_text(block), "text")]
    held = _dict(block)
    if isinstance(held.get("text"), str):
        return [_fence(held["text"])]
    if _truthy(held.get("toolUse")):
        use = _dict(held["toolUse"])
        return _call(use.get("name"), use.get("input"))
    if held.get("content_type") == "tool_use":
        return _call(held.get("name"), held.get("arguments"))
    if _truthy(held.get("toolResult")):
        result = _dict(held["toolResult"])
        parts = []
        for item in _list(result.get("content")):
            if isinstance(item, dict):
                item = item["json"] if "json" in item else item["text"] if isinstance(item.get("text"), str) else item
            parts.append(_block_text(item))
        status = result.get("status")
        return _result("\n".join(parts), status if _truthy(status) and status != "success" else None)
    if held.get("content_type") == "tool_result":
        return _span_result(held)
    if _truthy(held.get("reasoningContent")):
        text = _dict(_dict(held["reasoningContent"]).get("reasoningText")).get("text")
        return ["Reasoning:", _fenced(text if isinstance(text, str) else "", "text")]
    return [_fence(block)]


def _usage(usage: Any) -> list[str]:
    if not isinstance(usage, dict):
        return []
    parts = [
        f"{_compact(usage[key])} {word}"
        for key, word in (("inputTokens", "in"), ("outputTokens", "out"))
        if _is_number(usage.get(key))
    ]
    cached: Any = usage.get("cacheReadInputTokens")
    if _is_number(cached) and _truthy(cached):
        parts.append(f"{_compact(cached)} cache read")
    return [" · ".join(parts) + " tokens"] if parts else []


def _message(message: Any, heading: str) -> list[str]:
    message = _dict(message)
    content = message.get("content")
    if not isinstance(content, list):
        content = [{"text": "" if content is None else _as_text(content)}]
    role = message.get("role")
    parts = [
        f"{heading} {_as_text(role) if _truthy(role) else 'unknown'}",
        *_usage(_dict(message.get("metadata")).get("usage")),
    ]
    return parts + [part for block in content for part in _content(block)]


def _span(span: Any, index: int, heading: str) -> list[str]:
    held = _dict(span)
    kind = held.get("span_type") if _truthy(held.get("span_type")) else "span"
    if kind == "inference" and isinstance(held.get("messages"), list):
        return [f"{heading} Span {index}: inference"] + [
            part for message in held["messages"] for part in _message(message, f"{heading}#")
        ]
    if kind == "execute_tool":
        call = _dict(held.get("tool_call"))
        return [
            f"{heading} Span {index}: tool",
            *_call(call.get("name"), call.get("arguments")),
            *_span_result(_dict(held.get("tool_result"))),
        ]
    if kind == "invoke_agent":
        return [
            f"{heading} Span {index}: invoke agent",
            *_block("System prompt", held.get("system_prompt")),
            *_block("Prompt", held.get("user_prompt")),
            *_block("Response", held.get("agent_response")),
        ]
    return [f"{heading} Span {index}: {_as_text(kind)}", _fence(span)]


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
    facts.append(f"- Diagnoses: {', '.join(made for made, _ in diagnoses) or 'none found'}")
    parts = [f"# Session: {path}", "\n".join(facts)]
    for made, diagnosis in diagnoses:
        parts += [f"## Diagnosis: {made}", _fence(diagnosis)]
    first = 0
    for at, spans in enumerate(traces):
        trace_id = _dict(doc["traces"][at]).get("trace_id")
        named = f": {_one_line(trace_id)}" if _present(trace_id) else ""
        parts.append(f"## Trace {at + 1} of {len(traces)}{named}")
        parts += [part for i, span in enumerate(spans) for part in _span(span, first + i, "###")] or ["No spans."]
        first += len(spans)
    return _document(parts)


def diagnosis_markdown(path: str, doc: dict, session: str | None) -> str:
    """A diagnosis as it was written. `session` is the file of the session it was made from, when that was found."""
    session_id = doc.get("session_id")
    where = f", in {session}" if session else ", which is not in the folder"
    facts = [
        f"- Session: {_one_line(session_id) + where if _present(session_id) else 'not named'}",
        f"- Failures: {len(_list(doc.get('failures')))}",
        f"- Root causes: {len(_list(doc.get('root_causes')))}",
    ]
    return _document([f"# Diagnosis: {path}", "\n".join(facts), _fence(doc)])


# What sits under each root of the page's addresses. `ui-route.js` is where they are parsed.
PAGE_ADDRESSES = {
    "/reports": [
        ("", "every report file found"),
        ("/<report file>", "one report"),
        ("/<report file>/rows/<case>/<evaluator>", "one row of it, opened and in view"),
        ("/<report file>/rows/<case>", "the same, for a row whose case has no evaluator named"),
        ("/compare?r=<report file>&r=<report file>", "two or more reports side by side, oldest first"),
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
            "one run; the view is setup, overview, progress, candidates, tunables or trade-off",
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
            "## Token",
            "Every address here needs this server's token, sent as the `X-Evals-Token` header or as `?t=<token>`. "
            "The token is in the link the server printed when it started. The cookie a browser holds opens "
            "the page only: it does not open the Markdown, the JSON or any `/api` route.",
            "## Markdown and JSON",
            "A report, an experiment or a session is read by adding a form to its page address. A file is named "
            "by its path under the served folder, which ends in `.json`; `/api/scan` lists them. A diagnosis file "
            "is read the same way, under `/sessions/`. A comparison is a page only: read each of its reports.",
            "\n".join(forms),
            "## Routes",
            "\n".join(f"- `{route}`: {holds}" for route, holds in routes.items()),
            "## Page addresses",
            "These answer with the page itself, for a browser. Each one names one thing, so a link to it "
            "can be pasted into a conversation.",
            "\n".join(pages),
        ]
    )
