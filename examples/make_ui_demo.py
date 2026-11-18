"""Write a synthetic folder with something to show in every view of `strands-evals ui`.

usage: python examples/make_ui_demo.py DIR, then strands-evals ui DIR
"""

import copy
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from strands_evals.experiment import Experiment
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)

REPO = Path(__file__).resolve().parent.parent
START = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)

# name: (input, expected output, tools the agent should call or None, seconds the run took)
SUPPORT: dict[str, tuple] = {
    "order_status": ("Where is my order 11023?", "Order 11023 is in transit.", ["lookup_order"], 2.4),
    "refund_window": ("Can I still get a refund?", "You are eligible for a refund within 30 days.", None, 1.1),
    "password_reset": ("I can't log in.", "I sent a reset link to the address on file.", ["send_reset_link"], 3.3),
    "cancel_subscription": (
        "Cancel my plan and refund me.",
        "Your subscription is cancelled and a refund was issued.",
        ["find_account", "cancel_plan", "issue_refund"],
        6.8,
    ),
    "warranty_question": ("Is the kettle still under warranty?", "Yes, until March 2027.", ["lookup_order"], 2.9),
    "angry_customer": ("This is the third time I am asking!", "I am sorry. I have escalated this.", None, 1.6),
}
EVALUATORS: list[dict] = [
    {"evaluator_type": "Equals"},
    {"evaluator_type": "Contains", "value": "refund", "case_sensitive": False},
    {"evaluator_type": "OutputEvaluator", "name": "tone", "rubric": "Score 1.0 when the reply is polite and specific."},
    {"evaluator_type": "TrajectoryEvaluator", "rubric": "Score 1.0 when the expected tools were called, in order."},
]
# What each evaluator made of each case in the current report: (score, verdict), verdict one of pass, fail, na.
JUDGED: dict[str, dict[str, tuple[float, str]]] = {
    "Equals": {"password_reset": (0.0, "fail"), "angry_customer": (0.0, "fail")},
    "Contains": {"order_status": (0.0, "fail"), "password_reset": (0.0, "fail"), "warranty_question": (0.0, "fail")},
    "tone": {"angry_customer": (0.35, "fail"), "cancel_subscription": (0.8, "pass"), "password_reset": (0.6, "pass")},
    "TrajectoryEvaluator": {
        "refund_window": (1.0, "na"),
        "angry_customer": (1.0, "na"),
        "cancel_subscription": (0.67, "fail"),
    },
}
# The same cases a week earlier: these rows read differently then.
EARLIER = {
    ("Equals", "order_status"): (0.0, "fail"),
    ("Equals", "password_reset"): (1.0, "pass"),
    ("tone", "angry_customer"): (0.7, "pass"),
    ("tone", "refund_window"): (0.55, "pass"),
    ("TrajectoryEvaluator", "cancel_subscription"): (0.33, "fail"),
}


def write(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def case(index: int, name: str, text: str, expected: str, tools: list[str] | None = None, **metadata) -> dict:
    document = {
        "name": name,
        "session_id": f"00000000-0000-4000-8000-{index:012d}",
        "input": text,
        "expected_output": expected,
    }
    if tools:
        document["expected_trajectory"] = tools
    if metadata:
        document["metadata"] = metadata
    return document


def experiment(cases: list[dict], evaluators: list[dict]) -> dict:
    document = {"cases": cases, "evaluators": evaluators}
    Experiment.from_dict(copy.deepcopy(document))
    return document


def session(name: str, text: str, answer: str, tools: list[str], seconds: float) -> dict:
    """A recorded run: the agent span, then a model call and a tool call for each tool."""

    def info(number: int, begin: float, end: float) -> SpanInfo:
        return SpanInfo(
            trace_id=f"trace-{name}",
            span_id=f"span-{number}",
            session_id=name,
            start_time=START + timedelta(seconds=begin),
            end_time=START + timedelta(seconds=end),
        )

    spans: list = [
        AgentInvocationSpan(
            span_info=info(0, 0, seconds),
            user_prompt=text,
            agent_response=answer,
            available_tools=[],
            system_prompt="You are the support assistant for Northwind Outfitters.",
        )
    ]
    step = seconds / (len(tools) + 1)
    for number, tool in enumerate(tools):
        call = ToolCall(name=tool, arguments={"customer": "c-1042"}, tool_call_id=f"call-{number}")
        begin = number * step
        asked = AssistantMessage(
            content=[TextContent(text=f"I will use {tool}."), ToolCallContent(**call.model_dump())]
        )
        spans.append(InferenceSpan(span_info=info(2 * number + 1, begin, begin + step * 0.6), messages=[asked]))
        result = ToolResult(content=json.dumps({"ok": True, "tool": tool}), tool_call_id=call.tool_call_id)
        spans.append(
            ToolExecutionSpan(
                span_info=info(2 * number + 2, begin + step * 0.6, begin + step), tool_call=call, tool_result=result
            )
        )
    recorded = Session(session_id=name, traces=[Trace(trace_id=f"trace-{name}", session_id=name, spans=spans)])
    return recorded.model_dump(mode="json")


def report(cases: dict[str, tuple], judged: dict[tuple[str, str], tuple[float, str]], traced: bool) -> dict:
    """One row per case and evaluator. A row not in `judged` passed with a score of 1."""
    rows = []
    for evaluator in EVALUATORS:
        label = evaluator.get("name") or evaluator["evaluator_type"]
        for name, (text, expected, tools, seconds) in cases.items():
            score, verdict = judged.get((label, name), (1.0, "pass"))
            wrong = label == "Equals" and verdict == "fail"
            answer = "I'm not able to help with that." if wrong else expected
            trajectory = session(name, text, answer, tools, seconds) if traced and tools else tools
            row = {
                "name": name,
                "input": text,
                "expected_output": expected,
                "actual_output": answer,
                "expected_trajectory": tools,
                "actual_trajectory": trajectory,
                "evaluator": label,
                "evaluator_type": evaluator["evaluator_type"],
            }
            reason = {"pass": "meets the check", "fail": "does not meet the check", "na": "nothing to judge"}[verdict]
            output = {
                "score": score,
                "test_pass": verdict != "fail",
                "reason": reason,
                "label": "not_applicable" if verdict == "na" else None,
                "metadata": None,
            }
            rows.append((row, output, verdict))
    counted = [output["score"] for _, output, verdict in rows if verdict != "na"]
    document = {
        "overall_score": round(sum(counted) / len(counted), 3),
        "cases": [row for row, _, _ in rows],
        "scores": [output["score"] for _, output, _ in rows],
        "test_passes": [output["test_pass"] for _, output, _ in rows],
        "reasons": [output["reason"] for _, output, _ in rows],
        "detailed_results": [[output] for _, output, _ in rows],
    }
    EvaluationReport.from_dict(copy.deepcopy(document))
    return document


def optimizer_runs(out: Path) -> None:
    """A finished run from the page's own test fixture, the cases it split up, and the same run part-way."""
    path = REPO / "tests" / "ui" / "fixtures" / "make_optimizer_run.py"
    spec = importlib.util.spec_from_file_location("make_optimizer_run", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.OUT = out / "tuning" / "support-prompt-tuning"
    fixture.main()

    names = fixture.TRAIN + fixture.VALIDATION + fixture.TEST
    cases = [
        case(100 + i, name, fixture.conversation(name)[0], fixture.answer("c10", name)) for i, name in enumerate(names)
    ]
    judge = [{"evaluator_type": "OutputEvaluator", "name": "Helpfulness", "rubric": "Score how helpful the reply is."}]
    write(out / "tuning" / "support-cases.experiment.json", experiment(cases, judge))

    running = out / "tuning" / "still-running"
    shutil.copytree(fixture.OUT / "transcripts", running / "transcripts", dirs_exist_ok=True)
    config = json.loads((fixture.OUT / "run.json").read_text())
    write(running / "run.json", {**config, "run_id": "support-tuning-still-running"})
    events = [json.loads(line) for line in (fixture.OUT / "events.jsonl").read_text().splitlines()]
    done = ("finalists", "picked", "test", "stop")
    kept = [event for event in events if event.get("iteration", 0) <= 3 and event["event"] not in done]
    (running / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in kept))


def main(out: Path) -> None:
    cases = [
        case(i, name, text, expected, tools) for i, (name, (text, expected, tools, _)) in enumerate(SUPPORT.items())
    ]
    cases[0]["metadata"] = {"priority": "normal"}
    now = {(label, name): judged for label, by_case in JUDGED.items() for name, judged in by_case.items()}

    # A run with a report, and the same experiment's report from a week earlier to compare it against.
    write(out / "support" / "support.experiment.json", experiment(cases, EVALUATORS))
    write(out / "support" / "support.report.json", report(SUPPORT, now, traced=True))
    write(out / "support-last-week" / "support.experiment.json", experiment(cases, EVALUATORS))
    write(out / "support-last-week" / "support.report.json", report(SUPPORT, {**now, **EARLIER}, traced=False))

    # Cases still in draft, for the review pass; nothing has run them yet.
    drafts = [
        case(20, "gift_card_balance", "How much is left on my gift card?", "You have $25.00 left."),
        case(21, "change_address", "I moved. Update my address.", "Your address is updated.", status="draft"),
        case(22, "bulk_order_quote", "Quote me 500 mugs.", "A specialist will send a quote.", status="draft"),
        case(
            23, "store_hours", "When are you open?", "9 to 6, Monday to Saturday.", status="draft", source="generated"
        ),
    ]
    write(out / "drafts" / "onboarding.experiment.json", experiment(drafts, EVALUATORS[:2]))

    # A report that no longer matches the experiment beside it, and one with no experiment at all.
    two = dict(list(SUPPORT.items())[:2])
    write(out / "edited-since" / "billing.experiment.json", experiment(cases[2:5], EVALUATORS[:1]))
    write(out / "edited-since" / "billing.report.json", report(dict(list(SUPPORT.items())[1:4]), now, traced=False))
    write(out / "report-only" / "nightly.report.json", report(two, now, traced=False))

    # The experiments shipped as examples: written by hand, not run.
    shutil.copytree(REPO / "examples" / "experiments", out / "examples", dirs_exist_ok=True)

    # Files the scan reads and sets aside.
    write(out / "notes.json", {"just": "notes"})
    (out / "half-written.json").write_text('{"cases": [')

    optimizer_runs(out)
    sys.stdout.write(f"\nwrote {out}\nopen it with: strands-evals ui {out}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
