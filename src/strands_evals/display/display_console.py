import logging
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..types.evaluation import EvaluationOutput, all_could_not_evaluate

console = Console()
logger = logging.getLogger(__name__)


def _pass_rate_text(items: dict) -> str:
    """The panel's pass rate, counted the same way as the CLI summary and `--fail-on`.

    Every case counts, and a case the evaluator could not judge (e.g. its judge's context window
    overflowed) counts as not passed. Those cases are also named next to the rate, so a low rate
    from unjudged cases does not read as the agent failing.
    """
    cases = list(items.values())
    passed = sum(1 for case in cases if case["details"]["test_pass"])
    rate = passed / len(cases) if cases else 0.0
    unjudged = sum(1 for case in cases if all_could_not_evaluate(case.get("detailed_results") or []))
    return f"Pass Rate: {rate:.2f}" + (f" ({unjudged} could not evaluate)" if unjudged else "")


def _is_rubric_row(row: EvaluationOutput) -> bool:
    return (row.metadata or {}).get("evaluated_by") == "TypedRubricEvaluator"


def _fmt(value: Any) -> str:
    """Format recorded numbers without rejecting malformed optional metadata."""
    try:
        return f"{value:.4g}"
    except (TypeError, ValueError):
        return str(value)


def _fast_rubric_rows(outputs: list[EvaluationOutput]) -> list[EvaluationOutput]:
    """Read the audit once; Cascade repeats it on every fallback verdict row."""
    for output in outputs:
        metadata = output.metadata or {}
        if metadata.get("path") == "fallback" and "fast_rows" in metadata:
            audit = metadata["fast_rows"]
            if not isinstance(audit, list):
                logger.warning(
                    "audit_type=<%s> | fast rubric audit could not be displayed | expected a list",
                    type(audit).__name__,
                )
                return []
            try:
                rows = [EvaluationOutput.model_validate(entry) for entry in audit]
            except ValidationError as error:
                logger.warning("error=<%s> | fast rubric audit could not be displayed", error)
                return []
            return [row for row in rows if _is_rubric_row(row)]
    return []


def _decision_model_text(rows: list[EvaluationOutput]) -> str:
    """Describe only the identity the backend reported, including missing model ids."""
    identities = []
    for row in rows:
        metadata = row.metadata or {}
        provider = metadata.get("provider_name") or metadata.get("backend")
        model_id = metadata.get("model_id")
        identity = " / ".join(str(value) for value in (provider, model_id) if value)
        if not model_id:
            identity = f"{identity} (model not reported)" if identity else "not reported"
        if identity not in identities:
            identities.append(identity)
    return "; ".join(identities)


def _verdict_details(
    details: dict[str, Any], outputs: list[EvaluationOutput], fast_rows: list[EvaluationOutput]
) -> dict[str, Any]:
    """Project stored verdict metadata into optional summary columns."""
    details = dict(details)
    rubric_rows = [row for row in outputs if _is_rubric_row(row)]
    if fast_rows:
        details["decision_model"] = f"fast: {_decision_model_text(fast_rows)}"
        if rubric_rows:
            details["decision_model"] += f"; fallback: {_decision_model_text(rubric_rows)}"
    elif rubric_rows:
        identity = _decision_model_text(rubric_rows)
        stage = (rubric_rows[0].metadata or {}).get("path")
        if stage in ("fast", "fallback") and all((row.metadata or {}).get("path") == stage for row in rubric_rows):
            identity = f"{stage}: {identity}"
        details["decision_model"] = identity

    for field in ("path", "p_gate"):
        values = []
        for row in outputs:
            metadata = row.metadata or {}
            if metadata.get("path") not in ("fast", "fallback"):
                continue
            value = metadata.get(field)
            if value is not None:
                text = _fmt(value) if field == "p_gate" else str(value)
                if text not in values:
                    values.append(text)
        if values:
            details[field] = "; ".join(values)
    return details


def _metrics_tree(outputs: list[EvaluationOutput], title: str) -> Tree:
    """Render verdicts, adding typed-rubric fields when the row identifies itself as one."""
    tree = Tree(Text(title, style="bold cyan"))
    for index, result in enumerate(outputs, start=1):
        metadata = result.metadata or {}
        name = str(metadata.get("question_id", f"Metric {index}")) if _is_rubric_row(result) else f"Metric {index}"
        status = "✅" if result.test_pass else "❌"
        score = "n/a" if result.not_applicable else f"{result.score:.2f}"
        node = tree.add(Text(f"{name}: Score={score} {status}", style="yellow"))
        if result.label is not None:
            node.add(Text(f"Label: {result.label}"))
        if _is_rubric_row(result):
            if metadata.get("question") is not None:
                node.add(Text(f"Question: {metadata['question']}"))
            if "answer" in metadata:
                node.add(Text(f"Raw answer: {metadata['answer']}"))
            if "selected_option" in metadata:
                selected = metadata["selected_option"]
                node.add(Text(f"Selected option: {selected if selected is not None else 'n/a'}"))
            probabilities = metadata.get("probabilities")
            if probabilities is not None:
                if isinstance(probabilities, Mapping):
                    text = ", ".join(f"{label}={_fmt(value)}" for label, value in probabilities.items())
                else:
                    text = str(probabilities)
                node.add(Text(f"Probabilities: {text}"))
            if metadata.get("confidence") is not None:
                node.add(Text(f"Confidence: {_fmt(metadata['confidence'])}"))
            node.add(Text(f"Decision model: {_decision_model_text([result])}"))
        if result.reason:
            node.add(Text(result.reason, style="dim"))
    return tree


class CollapsibleTableReportDisplay:
    """
    Interactive console display for evaluation reports with expandable/collapsible test case details.

    This class provides an interactive rich console interface for displaying evaluation results
    with the ability to expand or collapse individual test cases to show or hide details.

    Attributes:
        items: Dictionary of test cases with their details and expansion state
        overall_score: The overall evaluation score
        include_input: Whether to display input values in the table
        include_output: Whether to display output values in the table


    items should follow the following structure:
        {
        "0": {"details": {
                "name": str,
                "score": float,
                "test_pass: bool,
                "reason": str,
                ... # will display everything that's given like actual_output etc.
                },
                "detailed_results": list[EvaluationOutput],
            },
            "expanded": bool
        }
    """

    def __init__(self, items: dict, overall_score: float) -> None:
        """
        Initialize the collapsible table display.

        Args:
            items: Dictionary of test cases with their details and expansion state
            overall_score: The overall evaluation score
        """
        self.items = items
        self.overall_score = overall_score

    def display_items(self, static: bool = False) -> None:
        """
        Display the evaluation report as a rich table with expandable/collapsible rows.

        Renders a table showing test case results with expansion indicators.
        Expanded rows show full details, while collapsed rows show minimal information.

        Args:
            static: Show typed-rubric details even when the case is collapsed.
        """
        overall_score_string = f"[bold blue]Overall Score: {self.overall_score:.2f}[/bold blue]"
        overall_pass_rate = f"[bold blue]{_pass_rate_text(self.items)}[/bold blue]"
        spacing = "           "
        console.print(Panel(f"{overall_score_string}{spacing}{overall_pass_rate}", title="📊 Evaluation Report"))

        # Create Table and headers
        table = Table(title="Test Case Results", show_lines=True)
        colors_mapping = {
            "index": "cyan",
            "name": "magenta",
            "score": "green",
        }
        rows = []
        for key, item in self.items.items():
            outputs = item.get("detailed_results") or []
            fast_rows = _fast_rubric_rows(outputs)
            details = _verdict_details(item["details"], outputs, fast_rows)
            rows.append((key, item, details, outputs, fast_rows))
        headers = ["index"] + list(dict.fromkeys(header for _, _, details, _, _ in rows for header in details))
        for header in headers:
            if header in colors_mapping:
                table.add_column(header, style=colors_mapping[header])
            else:
                table.add_column(header, style="yellow")

        for key, item, case, _, _ in rows:
            symbol = "▼" if item["expanded"] else "▶"
            pass_status = "✅" if case["test_pass"] else "❌"

            # Build renderables dynamically based on headers order
            renderables: list[str | Text] = [f"{symbol} {key}"]  # index column
            for header in headers[1:]:
                if header not in case:
                    renderables.append("")
                elif header == "test_pass":
                    renderables.append(pass_status)
                elif item["expanded"]:
                    renderables.append(Text(str(case[header])))
                else:
                    # Show actual values for core fields, "..." for rest when collapsed
                    if header in ("name", "evaluator", "score", "decision_model", "path", "p_gate"):
                        renderables.append(Text(str(case[header])))
                    else:
                        renderables.append("...")
            table.add_row(*renderables)

        console.print(table)

        for key, item, _, outputs, fast_rows in rows:
            has_rubric = any(_is_rubric_row(row) for row in outputs)
            if item["expanded"] or (static and (has_rubric or fast_rows)):
                if len(outputs) > 1 or has_rubric or fast_rows:
                    console.print(_metrics_tree(outputs, f"📋 Detailed Metrics for Case {key}"))
                if fast_rows:
                    console.print(_metrics_tree(fast_rows, f"📋 Fast Rubric Audit for Case {key}"))

    def run(self, static: bool = False) -> None:
        """
        Run the interactive display loop. If static, then the terminal will only display the report.

        Args:
            static: Whether to display only or allow interaction with the report.

        Provides an interactive console interface where users can:
        - Expand/collapse individual test cases by entering their number
        - Expand all test cases with 'o'
        - Collapse all test cases with 'c'
        - Quit the interactive view with 'q'
        """
        while True:
            console.clear()
            self.display_items(static=static)

            if static:
                return

            choice = Prompt.ask(
                "\nEnter the test case number to expand/collapse it, o to expand all, "
                "and c to collapse all (q to quit)."
            )

            if choice.lower() == "q":
                break

            if choice.lower() == "o":
                for key in self.items:
                    self.items[key]["expanded"] = True
            elif choice.lower() == "c":
                for key in self.items:
                    self.items[key]["expanded"] = False
            else:
                if choice in self.items:
                    self.items[choice]["expanded"] = not self.items[choice]["expanded"]
