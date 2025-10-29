import json
import os

from pydantic import BaseModel
from typing_extensions import TypeVar

from ..display.display_console import CollapsibleTableReportDisplay
from ..types.evaluation import EvaluationOutput

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class EvaluationReport(BaseModel):
    """
    A report of the evaluation of a task.

    Attributes:
        overall_score: The overall score of the task.
        scores: A list of the score for each test case in order.
        cases: A list of records for each test case.
        test_passes: A list of booleans indicating whether the test pass or fail.
        reasons: A list of reason for each test case.
    """

    overall_score: float
    scores: list[float]
    cases: list[dict]
    test_passes: list[bool]
    reasons: list[str] = []
    detailed_results: list[list[EvaluationOutput]] = []

    def _display(
        self,
        static: bool = True,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
    ):
        """
        Render an interface of the report with as much details as configured using Rich.

        Args:
            static: Whether to render the interface as interactive or static.
            include_input (Defaults to True): Include the input in the display.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.

        Note:
            This method provides an interactive console interface where users can expand or collapse
            individual test cases to view more or less detail.
        """
        report_data = {}
        for i in range(len(self.scores)):
            name = self.cases[i].get("name", f"Test {i + 1}")
            reason = self.reasons[i] if i < len(self.reasons) else "N/A"
            details_dict = {
                "name": name,
                "score": f"{self.scores[i]:.2f}",
                "test_pass": self.test_passes[i],
                "reason": reason,
            }
            if include_input:
                details_dict["input"] = str(self.cases[i].get("input"))
            if include_actual_output:
                details_dict["actual_output"] = str(self.cases[i].get("actual_output"))
            if include_expected_output:
                details_dict["expected_output"] = str(self.cases[i].get("expected_output"))
            if include_actual_trajectory:
                details_dict["actual_trajectory"] = str(self.cases[i].get("actual_trajectory"))
            if include_expected_trajectory:
                details_dict["expected_trajectory"] = str(self.cases[i].get("expected_trajectory"))
            if include_actual_interactions:
                details_dict["actual_interactions"] = str(self.cases[i].get("actual_interactions"))
            if include_expected_interactions:
                details_dict["expected_interactions"] = str(self.cases[i].get("expected_interactions"))
            if include_meta:
                details_dict["metadata"] = str(self.cases[i].get("metadata"))

            report_data[str(i)] = {
                "details": details_dict,
                "detailed_results": self.detailed_results[i] if i < len(self.detailed_results) else [],  # NEW
                "expanded": False,
            }

        display_console = CollapsibleTableReportDisplay(items=report_data, overall_score=self.overall_score)
        display_console.run(static=static)

    def display(
        self,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
    ):
        """
        Render the report with as much details as configured using Rich. Use run_display if want
        to interact with the table.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.
        """
        self._display(
            static=True,
            include_input=include_input,
            include_actual_output=include_actual_output,
            include_expected_output=include_expected_output,
            include_expected_trajectory=include_expected_trajectory,
            include_actual_trajectory=include_actual_trajectory,
            include_actual_interactions=include_actual_interactions,
            include_expected_interactions=include_expected_interactions,
            include_meta=include_meta,
        )

    def run_display(
        self,
        include_input: bool = True,
        include_actual_output: bool = False,
        include_expected_output: bool = False,
        include_expected_trajectory: bool = False,
        include_actual_trajectory: bool = False,
        include_actual_interactions: bool = False,
        include_expected_interactions: bool = False,
        include_meta: bool = False,
    ):
        """
        Render the report interactively with as much details as configured using Rich.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_actual_output (Defaults to False): Include the actual output in the display.
            include_expected_output (Defaults to False): Include the expected output in the display.
            include_expected_trajectory (Defaults to False): Include the expected trajectory in the display.
            include_actual_trajectory (Defaults to False): Include the actual trajectory in the display.
            include_actual_interactions (Defaults to False): Include the actual interactions in the display.
            include_expected_interactions (Defaults to False): Include the expected interactions in the display.
            include_meta (Defaults to False): Include metadata in the display.
        """
        self._display(
            static=False,
            include_input=include_input,
            include_actual_output=include_actual_output,
            include_expected_output=include_expected_output,
            include_expected_trajectory=include_expected_trajectory,
            include_actual_trajectory=include_actual_trajectory,
            include_actual_interactions=include_actual_interactions,
            include_expected_interactions=include_expected_interactions,
            include_meta=include_meta,
        )

    def to_dict(self):
        """
        Returns a dictionary representation of the report.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an EvaluationReport instance from a dictionary.

        Args:
            data: A dictionary containing the report data.
        """
        return cls.model_validate(data)

    def to_file(self, file_name: str, format: str = "json", directory: str = "report_files"):
        """
        Write the report to a file.

        Args:
            file_name: Name of the file without extension.
            format: The format of the file to be saved.
            directory: Directory to save the file (default: "report_files").
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if format == "json":
            with open(f"{directory}/{file_name}.json", "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def from_file(cls, file_path: str, format: str = "json"):
        """
        Create an EvaluationReport instance from a file.

        Args:
            file_path: Path to the file.
            format: The format of the file to be read.

        Return:
            An EvaluationReport object.
        """
        if format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return cls.from_dict(data)
