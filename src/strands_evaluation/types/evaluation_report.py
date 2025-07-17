from pydantic import BaseModel
from typing_extensions import TypeVar
import json
import os
from ..evaluators.utils.display_console import CollapsibleTableReportDisplay

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
    reasons: list[str] | None = []

    def _display(self, static: bool = True, include_input: bool = True, include_output: bool = True,
                include_expected_output: bool = False, include_expected_trajectory: bool = False,
                include_actual_trajectory: bool = False, include_meta: bool = False):
        """
        Render an interface of the report with as much details as configured using Rich.

        Args:
            static: Whether to render the interface as interactive or static.
            include_input: Whether to include the input in the display. Defaults to True.
            include_output: Whether to include the actual output in the display. Defaults to True.
            include_expected_output: Whether to include the expected output in the display. Defaults to False.
            include_expected_trajectory: Whether to include the expected trajectory in the display. Defaults to False.
            include_actual_trajectory: Whether to include the actual trajectory in the display. Defaults to False.
            include_meta: Whether to include metadata in the display. Defaults to False.
            
        Note:
            This method provides an interactive console interface where users can expand or collapse
            individual test cases to view more or less detail.
        """
        report_data = {}
        for i in range(len(self.scores)):
            name = self.cases[i].get("name", f"Test {i+1}")
            reason = self.reasons[i] if i < len(self.reasons) else "N/A"
            details_dict = {
                            "name": name,
                            "score": f"{self.scores[i]:.2f}",
                            "test_pass": self.test_passes[i],
                            "reason": reason,
                            }
            if include_input:
                details_dict["input"] = str(self.cases[i].get("input"))
            if include_output:
                details_dict["actual_output"] = str(self.cases[i].get("actual_output"))
            if include_expected_output:
                details_dict["expected_output"] = str(self.cases[i].get("expected_output"))
            if include_actual_trajectory:
                details_dict["actual_trajectory"] = str(self.cases[i].get("actual_trajectory"))
            if include_expected_trajectory:
                details_dict["expected_trajectory"] = str(self.cases[i].get("expected_trajectory"))
            if include_meta:
                details_dict["metadata"] = str(self.cases[i].get("metadata"))
            
            report_data[str(i)] = {
                "details": details_dict,
                "expanded": False
                }
        
        display_console = CollapsibleTableReportDisplay(items = report_data, overall_score = self.overall_score)
        display_console.run(static = static)

    def display(self, include_input: bool = True, include_output: bool = True,
                include_expected_output: bool = False, include_expected_trajectory: bool = False,
                include_actual_trajectory: bool = False, include_meta: bool = False):
        """
        Render the report with as much details as configured using Rich. Use run_display if want
        to interact with the table.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_output: Whether to include the actual output in the display. Defaults to True.
            include_expected_output: Whether to include the expected output in the display. Defaults to False.
            include_expected_trajectory: Whether to include the expected trajectory in the display. Defaults to False.
            include_actual_trajectory: Whether to include the actual trajectory in the display. Defaults to False.
            include_meta: Whether to include metadata in the display. Defaults to False.
        """
        self._display(static = True, include_input = include_input, include_output = include_output,
                      include_expected_output = include_expected_output,
                      include_expected_trajectory = include_expected_trajectory,
                      include_actual_trajectory = include_actual_trajectory, include_meta = include_meta)
    
    def run_display(self, include_input: bool = True, include_output: bool = True,
                include_expected_output: bool = False, include_expected_trajectory: bool = False,
                include_actual_trajectory: bool = False, include_meta: bool = False):
        """
        Render the report interactively with as much details as configured using Rich.

        Args:
            include_input: Whether to include the input in the display. Defaults to True.
            include_output: Whether to include the actual output in the display. Defaults to True.
            include_expected_output: Whether to include the expected output in the display. Defaults to False.
            include_expected_trajectory: Whether to include the expected trajectory in the display. Defaults to False.
            include_actual_trajectory: Whether to include the actual trajectory in the display. Defaults to False.
            include_meta: Whether to include metadata in the display. Defaults to False.
        """
        self._display(static = False, include_input = include_input, include_output = include_output,
                      include_expected_output = include_expected_output,
                      include_expected_trajectory = include_expected_trajectory,
                      include_actual_trajectory = include_actual_trajectory, include_meta = include_meta)

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

    def to_file(self, file_name: str, format: str, directory: str = "report_files"):
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
                json.dump(self.to_dict(), f, indent = 2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_file(cls, file_path: str, format: str):
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