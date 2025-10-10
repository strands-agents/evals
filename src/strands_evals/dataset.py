import asyncio
import json
import os
from collections.abc import Callable

from typing_extensions import Any, Generic, TypeVar

from .case import Case
from .evaluators.evaluator import Evaluator
from .evaluators.interactions_evaluator import InteractionsEvaluator
from .evaluators.output_evaluator import OutputEvaluator
from .evaluators.trajectory_evaluator import TrajectoryEvaluator
from .telemetry import get_tracer, serialize
from .types.evaluation import EvaluationData
from .types.evaluation_report import EvaluationReport

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Dataset(Generic[InputT, OutputT]):
    """
    A collection of test cases, representing a dataset.

    Dataset organizes a collection of test cases and evaluate them all with
    the defined evaluator on some task.

    Attributes:
        cases: A list of test cases in the dataset.
        evaluator: The evaluator to be used on the test cases.

    Example:
        dataset = Dataset[str, str](
            cases=[
                Case(name="Simple Knowledge",
                        input="What is the capital of France?",
                        expected_output="The capital of France is Paris.",
                        expected_trajectory=[],
                        metadata={"category": "knowledge"}),
               Case(name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator],
                        metadata={"category": "math"})
            ],
            evaluator=OutputEvaluator(rubric = "The output is relevant and complete. 0 if the output is incorrect or irrelevant.")
        )
    """

    def __init__(
        self, cases: list[Case[InputT, OutputT]] | None = None, evaluator: Evaluator[InputT, OutputT] | None = None,
    ):
        self._cases = cases or []
        self._evaluator = evaluator or Evaluator()
        self._tracer = get_tracer()

    @property
    def cases(self) -> list[Case[InputT, OutputT]]:
        """
        Get a deep copy of all test cases in the dataset.

        Returns deep copies to prevent accidental mutation of the original test cases.
        Users can safely modify the returned cases without affecting the dataset.

        Returns:
            List of Case objects (deep copies) containing all test cases in the dataset
        """
        return [case.model_copy(deep=True) for case in self._cases]

    @property
    def evaluator(self) -> Evaluator[InputT, OutputT]:
        """
        Get the evaluator used for assessing test case performance.

        Returns:
            The evaluator instance configured for this dataset
        """
        return self._evaluator

    @cases.setter
    def cases(self, new_cases: list[Case[InputT, OutputT]]):
        """
        Set the test cases for this dataset.

        Args:
            new_cases: List of Case objects to use as the dataset's test cases
        """
        self._cases = new_cases

    @evaluator.setter
    def evaluator(self, new_evaluator: Evaluator[InputT, OutputT]):
        """
        Set the evaluator for assessing test case performance.

        Args:
            new_evaluator: Evaluator instance to use for evaluating test cases
        """
        self._evaluator = new_evaluator

    def _run_task(
        self, task: Callable[[InputT], OutputT | dict[str, Any]], case: Case[InputT, OutputT]
    ) -> EvaluationData[InputT, OutputT]:
        """
        Run the task with the inputs from the test case.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
            case: The test case containing neccessary information to run the task

        Return:
            An EvaluationData record containing the input and actual output, name, expected output, and metadata.
        """
        if asyncio.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")

        evaluation_context = EvaluationData(
            name=case.name,
            input=case.input,
            expected_output=case.expected_output,
            expected_trajectory=case.expected_trajectory,
            expected_interactions=case.expected_interactions,
            metadata=case.metadata,
        )
        task_output = task(case.input)
        if isinstance(task_output, dict):  # could be evaluating the trajectory as well
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
            evaluation_context.actual_interactions = task_output.get("interactions")
            new_input = task_output.get("input", None)  # allows the user to update the input in the task function
            if new_input is not None:
                evaluation_context.input = new_input
        else:  # evaluating only the output
            evaluation_context.actual_output = task_output
        return evaluation_context

    async def _run_task_async(
        self, task: Callable[[InputT], OutputT | dict[str, Any]], case: Case[InputT, OutputT]
    ) -> EvaluationData[InputT, OutputT]:
        """
        Run the task with the inputs from the test case asynchronously.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
                The task can either run synchronously or asynchronously.
            case: The test case containing neccessary information to run the task

        Return:
            An EvaluationData record containing the input and actual output, name, expected output, and metadata.
        """
        # Create evaluation context
        evaluation_context = EvaluationData(
            name=case.name,
            input=case.input,
            expected_output=case.expected_output,
            expected_trajectory=case.expected_trajectory,
            expected_interactions=case.expected_interactions,
            metadata=case.metadata,
        )

        # Handle both async and sync tasks
        if asyncio.iscoroutinefunction(task):
            task_output = await task(case.input)
        else:
            # Run sync function in separate thread to avoid blocking
            task_output = await asyncio.to_thread(task, case.input)

        if isinstance(task_output, dict):
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
            evaluation_context.actual_interactions = task_output.get("interactions")
            new_input = task_output.get("input", None)  # allows the user to update the input in the task function
            if new_input is not None:
                evaluation_context.input = new_input
        else:
            evaluation_context.actual_output = task_output

        return evaluation_context

    async def _worker(self, queue: asyncio.Queue, task: Callable, results: list):
        """
        Worker that processes cases from the queue. Run evaluation on the task.

        Args:
            queue: Queue containing cases to process
            task: Task function to run on each case
            results: List to store results
        """
        while True:
            try:
                case = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            case_name = case.name or f"case_{len(results)}"
            
            ## Evaluation with tracing ##
            with self._tracer.start_as_current_span(
                f"eval_case {case_name}",
                attributes={
                    "gen_ai.eval.case.name": case_name,
                    "gen_ai.eval.case.input": serialize(case.input),
                }
            ) as case_span:
                try:
                    # Task execution span
                    with self._tracer.start_as_current_span(
                        "task_execution",
                        attributes={
                            "gen_ai.eval.task.type": "agent_task",
                            "gen_ai.eval.case.name": case_name,
                        }
                    ) as task_span:
                        evaluation_context = await self._run_task_async(task, case)
                        task_span.set_attributes({
                            "gen_ai.eval.data.input": serialize(evaluation_context.input),
                            "gen_ai.eval.data.expected_output": serialize(evaluation_context.expected_output),
                            "gen_ai.eval.data.actual_output": serialize(evaluation_context.actual_output),
                            "gen_ai.eval.data.has_trajectory": evaluation_context.actual_trajectory is not None,
                            "gen_ai.eval.data.has_interactions": evaluation_context.actual_interactions is not None,
                        })
                    
                    # Evaluator execution span
                    with self._tracer.start_as_current_span(
                        f"evaluator {self.evaluator.get_type_name()}",
                        attributes={
                            "gen_ai.eval.evaluator.type": self.evaluator.get_type_name(),
                            "gen_ai.eval.case.name": case_name,
                        }
                    ) as eval_span:
                        evaluation_output = await self.evaluator.evaluate_async(evaluation_context)
                        eval_span.set_attributes({
                            "gen_ai.eval.output.score": evaluation_output.score,
                            "gen_ai.eval.output.test_pass": evaluation_output.test_pass,
                            "gen_ai.eval.output.reason": evaluation_output.reason or "",
                        })
                    
                    # Store results
                    results.append(
                        {
                            "case": evaluation_context.model_dump(),
                            "test_pass": evaluation_output.test_pass,
                            "score": evaluation_output.score,
                            "reason": evaluation_output.reason or "",
                        }
                    )
                    
                except Exception as e:
                    case_span.record_exception(e)
                    results.append(
                        {
                            "case": case.model_dump(),
                            "test_pass": False,
                            "score": 0,
                            "reason": f"An error occurred: {str(e)}",
                        }
                    )
                finally:
                    queue.task_done()

    def run_evaluations(self, task: Callable[[InputT], OutputT | dict[str, Any]]) -> EvaluationReport:
        """
        Run the evaluations for all of the test cases with the evaluator.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.

        Return:
            An EvaluationReport containing the overall score, individual case results, and basic feedback for each test case.
        """
        scores = []
        test_passes = []
        cases = []
        reasons = []
        
        for case in self._cases:
            case_name = case.name or f"case_{len(cases)}"
            
            # Create a separate trace for each case
            with self._tracer.start_as_current_span(
                f"eval_case {case_name}",
                attributes={
                    "gen_ai.eval.case.name": case_name,
                    "gen_ai.eval.case.input": serialize(case.input),
                }
            ) as case_span:
                try:
                    # Task execution span
                    with self._tracer.start_as_current_span(
                        "task_execution",
                        attributes={
                            "gen_ai.eval.task.type": "agent_task",
                            "gen_ai.eval.case.name": case_name,
                        }
                    ) as task_span:
                        evaluation_context = self._run_task(task, case)
                        task_span.set_attributes({
                            "gen_ai.eval.data.input": serialize(evaluation_context.input),
                            "gen_ai.eval.data.expected_output": serialize(evaluation_context.expected_output),
                            "gen_ai.eval.data.actual_output": serialize(evaluation_context.actual_output),
                            "gen_ai.eval.data.has_trajectory": evaluation_context.actual_trajectory is not None,
                            "gen_ai.eval.data.has_interactions": evaluation_context.actual_interactions is not None,
                        })
                    
                    # Evaluator execution span
                    with self._tracer.start_as_current_span(
                        f"evaluator {self.evaluator.get_type_name()}",
                        attributes={
                            "gen_ai.eval.evaluator.type": self.evaluator.get_type_name(),
                            "gen_ai.eval.case.name": case_name,
                        }
                    ) as eval_span:
                        evaluation_output = self.evaluator.evaluate(evaluation_context)
                        eval_span.set_attributes({
                            "gen_ai.eval.output.score": evaluation_output.score,
                            "gen_ai.eval.output.test_pass": evaluation_output.test_pass,
                            "gen_ai.eval.output.reason": evaluation_output.reason or "",
                        })
                    
                    cases.append(evaluation_context.model_dump())
                    test_passes.append(evaluation_output.test_pass)
                    scores.append(evaluation_output.score)
                    reasons.append(evaluation_output.reason or "")
                    
                except Exception as e:
                    case_span.record_exception(e)
                    cases.append(case.model_dump())
                    test_passes.append(False)
                    scores.append(0)
                    reasons.append(f"An error occured : {str(e)}")

        report = EvaluationReport(
            overall_score=sum(scores) / len(scores) if len(scores) else 0,
            scores=scores,
            test_passes=test_passes,
            cases=cases,
            reasons=reasons,
        )

        return report

    async def run_evaluations_async(self, task: Callable, max_workers: int = 10) -> EvaluationReport:
        """
        Run evaluations asynchronously using a queue for parallel processing.

        Args:
            task: The task function to run on each case. This function should take in InputT and returns either OutputT or {"output": OutputT, "trajectory": ...}.
            The task can either run synchronously or asynchronously.
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            EvaluationReport containing evaluation results
        """
        queue = asyncio.Queue()
        results = []

        for case in self._cases:
            queue.put_nowait(case)

        num_workers = min(max_workers, len(self._cases))

        # Create and start workers
        workers = [asyncio.create_task(self._worker(queue, task, results)) for _ in range(num_workers)]

        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Process results
        scores = [r["score"] for r in results]
        test_passes = [r["test_pass"] for r in results]
        cases = [r["case"] for r in results]
        reasons = [r["reason"] for r in results]

        # Create and return report
        return EvaluationReport(
            overall_score=sum(scores) / len(scores) if scores else 0,
            scores=scores,
            test_passes=test_passes,
            cases=cases,
            reasons=reasons,
        )

    def to_dict(self) -> dict:
        """
        Convert the dataset to a dictionary.

        Return:
            A dictionary representation of the dataset.
        """
        return {"cases": [case.model_dump() for case in self._cases], "evaluator": self.evaluator.to_dict()}

    def to_file(self, file_name: str, format: str = "json", directory: str = "dataset_files"):
        """
        Write the dataset to a file.

        Args:
            file_name: Name of the file without extension.
            format: The format of the file to be saved.
            directory: Directory to save the file (default: "dataset_files").
        """
        os.makedirs(directory, exist_ok=True)
        if format == "json":
            with open(f"{directory}/{file_name}.json", "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise Exception(f"Format {format} is not supported.")

    @classmethod
    def from_dict(cls, data: dict, custom_evaluators: list[Evaluator] = None):
        """
        Create a dataset from a dictionary.

        Args:
            data: A dictionary representation of the dataset.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            A Dataset object.
        """
        custom_evaluators = custom_evaluators or []
        cases = [Case.model_validate(case_data) for case_data in data["cases"]]
        default_evaluators = {
            "Evaluator": Evaluator,
            "OutputEvaluator": OutputEvaluator,
            "TrajectoryEvaluator": TrajectoryEvaluator,
            "InteractionsEvaluator": InteractionsEvaluator,
        }
        all_evaluators = {**default_evaluators, **{v.get_type_name(): v for v in custom_evaluators}}

        evaluator_type = data["evaluator"]["evaluator_type"]
        evaluator_args = {k: v for k, v in data["evaluator"].items() if k != "evaluator_type"}

        if evaluator_type in all_evaluators:
            evaluator = all_evaluators[evaluator_type](**evaluator_args)
        else:
            raise Exception(
                f"Cannot find {evaluator_type}. Make sure the evaluator type is spelled correctly and all relevant custom evaluators are passed in."
            )

        return cls(cases=cases, evaluator=evaluator)

    @classmethod
    def from_file(cls, file_path: str, format: str = "json", custom_evaluators: list[Evaluator] = None):
        """
        Create a dataset from a file.

        Args:
            file_path: Path to the file.
            format: The format of the file to be read.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            A Dataset object.
        """
        if format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            raise Exception(f"Format {format} is not supported.")

        return cls.from_dict(data, custom_evaluators)
