"""A report and an experiment as Markdown, and the description of the server served at `/llms.txt`.

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
_ASPECT_LABEL = {
    "output": "output",
    "trajectory": "trajectory",
    "interactions": "interactions",
    "environment_state": "environment state",
}


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


def stringify(value: Any, indent: str = "") -> str:
    """`JSON.stringify(value, null, 2)`, character for character."""
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
    inner = indent + "  "
    if isinstance(value, dict):
        items = [f"{inner}{stringify(key)}: {stringify(value[key], inner)}" for key in _key_order(value)]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}" if items else "{}"
    items = [inner + stringify(item, inner) for item in value]
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


def _fence(value: Any) -> str:
    """A value in a fenced block whose fence is longer than any run of backticks inside it."""
    text = value_text(value)
    ticks = "`" * max(3, 1 + max((len(run) for run in _TICKS.findall(text)), default=0))
    return f"{ticks}{'text' if isinstance(value, str) else 'json'}\n{text}\n{ticks}"


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
            label = f"{side.capitalize()} {_ASPECT_LABEL[aspect]}"
            value = case.get(f"{side}_{aspect}")
            parts += _trajectory(label, value) if aspect == "trajectory" else _block(label, value)
    return parts


def _rows(doc: dict) -> list[dict]:
    """One row per case and evaluator, read as the page reads them."""
    rows = []
    for index, case in enumerate(_list(doc.get("cases"))):
        case = _dict(case)
        outputs = [_dict(output) for output in _list(_at(doc.get("detailed_results"), index))]
        declined = bool(outputs) and all(
            o.get("label") == "not_applicable" and _truthy(o.get("test_pass")) for o in outputs
        )
        passed = _truthy(_at(doc.get("test_passes"), index))
        reason = _at(doc.get("reasons"), index)
        rows.append(
            {
                "case": case,
                "name": _name(case.get("name"), index),
                "evaluator": _one_line(case["evaluator"]) if _present(case.get("evaluator")) else "(unnamed)",
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


def report_markdown(path: str, doc: dict, experiment: str | None) -> str:
    """A report: its figures, each failing row in full, then every row in one table."""
    rows = _rows(doc)
    verdicts = [row["verdict"] for row in rows]
    facts = [
        f"- Overall score: {_score(doc.get('overall_score'))}",
        f"- Rows: {len(rows)} ({verdicts.count('passed')} passed, {verdicts.count('failed')} failed, "
        f"{verdicts.count('not applicable')} not applicable)",
        f"- Cases: {len({row['name'] for row in rows})}",
        f"- Evaluators: {', '.join(dict.fromkeys(row['evaluator'] for row in rows)) or 'none'}",
        f"- Experiment: {experiment or 'none found, so this report was run from code'}",
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
        f"- Reports: {', '.join(sorted(reports)) or 'none, so it has not been run'}",
    ]
    parts = [f"# Experiment: {path}", "\n".join(facts), "## Evaluators"]
    parts += [part for i, e in enumerate(evaluators) for part in _evaluator_parts(e, i)] or ["None."]
    parts.append("## Cases")
    listed = [part for i, case in enumerate(cases) for part in case_parts(case, i, "###")]
    return _document(parts + (listed or ["None."]))


def case_parts(case: dict, index: int, heading: str) -> list[str]:
    return [
        f"{heading} {_name(case.get('name'), index)}",
        *_case_blocks(case, ("expected",)),
        *_block("Metadata", case.get("metadata")),
    ]


# What sits under each root of the page's addresses. `ui-route.js` is where they are parsed.
PAGE_ADDRESSES = {
    "/reports": [
        ("", "every report file found"),
        ("/<report file>", "one report"),
        ("/<report file>/rows/<case>/<evaluator>", "one row of it, opened and in view"),
    ],
    "/experiments": [
        ("", "every experiment file found"),
        ("/<experiment file>", "one experiment, in the cases editor"),
        ("/<experiment file>/cases/<name>", "the same, on one case"),
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
    ("report", ".json"): "the report file's document",
    ("experiment", ".json"): "the experiment file's document",
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
            "> A local page over a folder of strands-evals experiments, reports and optimizer runs. "
            "What it shows can be read without a browser, as Markdown or as JSON.",
            "## Token",
            "Every address here needs this server's token, sent as the `X-Evals-Token` header or as `?t=<token>`. "
            "The token is in the link the server printed when it started. The cookie a browser holds opens "
            "the page only: it does not open the Markdown, the JSON or any `/api` route.",
            "## Markdown and JSON",
            "A report or an experiment is read by adding a form to its page address. A file is named by its "
            "path under the served folder, which ends in `.json`; `/api/scan` lists them.",
            "\n".join(forms),
            "## Routes",
            "\n".join(f"- `{route}`: {holds}" for route, holds in routes.items()),
            "## Page addresses",
            "These answer with the page itself, for a browser. Each one names one thing, so a link to it "
            "can be pasted into a conversation.",
            "\n".join(pages),
        ]
    )
