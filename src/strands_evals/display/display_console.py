from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


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
                }
            },
            "expanded": bool
        }
    """

    def __init__(self, items: dict, overall_score: float):
        """
        Initialize the collapsible table display.

        Args:
            items: Dictionary of test cases with their details and expansion state
            overall_score: The overall evaluation score
        """
        self.items = items
        self.overall_score = overall_score

    def display_items(self):
        """
        Display the evaluation report as a rich table with expandable/collapsible rows.

        Renders a table showing test case results with expansion indicators.
        Expanded rows show full details, while collapsed rows show minimal information.
        """
        overall_score_string = f"[bold blue]Overall Score: {self.overall_score:.2f}[/bold blue]"
        pass_count = sum([1 if case["details"]["test_pass"] else 0 for case in self.items.values()])
        pass_rate = pass_count / len(self.items)
        overall_pass_rate = f"[bold blue]Pass Rate: {pass_rate}[/bold blue]"
        spacing = "           "
        console.print(Panel(f"{overall_score_string}{spacing}{overall_pass_rate}", title="📊 Evaluation Report"))

        # Create Table and headers
        table = Table(title="Test Case Results", show_lines=True)
        colors_mapping = {
            "index": "cyan",
            "name": "magenta",
            "score": "green",
        }
        headers = ["index"] + list(self.items["0"]["details"].keys())
        for header in headers:
            if header in colors_mapping:
                table.add_column(header, style=colors_mapping[header])
            else:
                table.add_column(header, style="yellow")

        for key, item in self.items.items():
            symbol = "▼" if item["expanded"] else "▶"
            case = item["details"]
            pass_status = "✅" if case["test_pass"] else "❌"
            other_fields = list(case.values())[3:]
            if item["expanded"]:  # We always to render at least the index, name, score, test_pass, and reason
                renderables = [
                    f"{symbol} {key}",
                    case.get("name", f"Test {key}"),
                    case.get("score"),
                    pass_status,
                ] + other_fields
            else:
                renderables = [
                    f"{symbol} {key}",
                    case.get("name", f"Test {key}"),
                    case.get("score"),
                    pass_status,
                ] + len(other_fields) * ["..."]
            table.add_row(*renderables)
        console.print(table)

    def run(self, static: bool = False):
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
            self.display_items()

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
