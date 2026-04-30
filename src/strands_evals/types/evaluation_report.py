import json
from pathlib import Path

from pydantic import BaseModel

from ..display.display_console import CollapsibleTableReportDisplay
from ..types.evaluation import EvaluationOutput


class EvaluationReport(BaseModel):
    """
    A report of the evaluation of a task.

    Attributes:
        evaluator_name: The name of the evaluator that produced this report.
        overall_score: The overall score of the task.
        scores: A list of the score for each test case in order.
        cases: A list of records for each test case.
        test_passes: A list of booleans indicating whether the test pass or fail.
        reasons: A list of reason for each test case.
    """

    evaluator_name: str = ""
    overall_score: float
    scores: list[float]
    cases: list[dict]
    test_passes: list[bool]
    reasons: list[str] = []
    detailed_results: list[list[EvaluationOutput]] = []

    @classmethod
    def flatten(cls, reports: list["EvaluationReport"]) -> "EvaluationReport":
        """Flatten multiple evaluation reports into a single report."""
        if not reports:
            return cls(overall_score=0.0, scores=[], cases=[], test_passes=[])

        scores, cases, passes, reasons, detailed = [], [], [], [], []

        for report in reports:
            evaluator = report.evaluator_name or "Unknown"
            for i, case in enumerate(report.cases):
                cases.append({**case, "evaluator": evaluator})
                scores.append(report.scores[i] if i < len(report.scores) else 0.0)
                passes.append(report.test_passes[i] if i < len(report.test_passes) else False)
                reasons.append(report.reasons[i] if i < len(report.reasons) else "")
                detailed.append(report.detailed_results[i] if i < len(report.detailed_results) else [])

        return cls(
            evaluator_name="Combined",
            overall_score=sum(scores) / len(scores) if scores else 0.0,
            scores=scores,
            cases=cases,
            test_passes=passes,
            reasons=reasons,
            detailed_results=detailed,
        )

    @staticmethod
    def _format_input_for_display(input_value: object) -> str:
        """Format an input value for display, handling multimodal inputs gracefully.

        Detects serialized multimodal inputs (dicts with 'media' and 'instruction' keys)
        and renders a readable summary instead of dumping raw binary or base64 data.
        """
        if isinstance(input_value, dict) and "media" in input_value and "instruction" in input_value:
            instruction = input_value.get("instruction", "")
            context = input_value.get("context")
            media = input_value.get("media")

            # Describe media without dumping raw data
            if isinstance(media, list):
                media_desc = f"[{len(media)} media item(s)]"
            elif isinstance(media, dict):
                media_desc = "[1 media item]"
            elif isinstance(media, str) and len(media) > 200:
                media_desc = f"[media: {media[:50]}...]"
            else:
                media_desc = f"[media: {media}]"

            parts = [f"instruction: {instruction}"]
            if context:
                parts.append(f"context: {context}")
            parts.append(media_desc)
            return " | ".join(parts)
        return str(input_value)

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
            details_dict = {"name": name}
            # Include evaluator column for flattened reports (right after name)
            if "evaluator" in self.cases[i]:
                details_dict["evaluator"] = self.cases[i]["evaluator"]
            details_dict["score"] = f"{self.scores[i]:.2f}"
            details_dict["test_pass"] = self.test_passes[i]
            details_dict["reason"] = reason
            if include_input:
                details_dict["input"] = self._format_input_for_display(self.cases[i].get("input"))
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

    def to_file(self, path: str):
        """
        Write the report to a JSON file.

        Args:
            path: The file path where the report will be saved. Can be:
                  - A filename only (e.g., "foo.json" or "foo") - saves in current working directory
                  - A relative path (e.g., "relative_path/foo.json") - saves relative to current working directory
                  - An absolute path (e.g., "/path/to/dir/foo.json") - saves in exact directory

                  If no extension is provided, ".json" will be added automatically.
                  Only .json format is supported.

        Raises:
            ValueError: If the path has a non-JSON extension.
        """
        file_path = Path(path)

        if file_path.suffix:
            if file_path.suffix != ".json":
                raise ValueError(
                    f"Only .json format is supported. Got path with extension: {path}. "
                    f"Please use a .json extension or provide a path without an extension."
                )
        else:
            file_path = file_path.with_suffix(".json")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str):
        """
        Create an EvaluationReport instance from a JSON file.

        Args:
            path: Path to the JSON file.

        Return:
            An EvaluationReport object.

        Raises:
            ValueError: If the file does not have a .json extension.
        """
        file_path = Path(path)

        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}. Please provide a path with .json extension."
            )

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)
