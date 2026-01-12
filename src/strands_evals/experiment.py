import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from opentelemetry.trace import format_trace_id
from typing_extensions import Any, Generic, TypeVar

from .case import Case
from .evaluators.evaluator import Evaluator
from .evaluators.interactions_evaluator import InteractionsEvaluator
from .evaluators.output_evaluator import OutputEvaluator
from .evaluators.trajectory_evaluator import TrajectoryEvaluator
from .telemetry import get_tracer, serialize
from .telemetry._cloudwatch_logger import _send_to_cloudwatch
from .types.evaluation import EvaluationData
from .types.evaluation_report import EvaluationReport

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _get_label_from_score(evaluator: Evaluator, score: float) -> str:
    """
    Get the label from score using evaluator's _score_mapping if available.
    If no mapping exists, returns "YES" for scores >= 0.5, "NO" otherwise.

    Args:
        evaluator: The evaluator instance
        score: The numeric score
        default_label: Default label to return if provided and no mapping found

    Returns:
        The label corresponding to the score
    """
    if hasattr(evaluator, "_score_mapping") and evaluator._score_mapping:
        # Create reverse mapping from score to label
        reverse_mapping = {v: k for k, v in evaluator._score_mapping.items()}
        # Find the score in the mapping
        if score in reverse_mapping:
            return str(reverse_mapping[score])

    # Otherwise, return YES/NO based on score
    return "YES" if score >= 0.5 else "NO"


class Experiment(Generic[InputT, OutputT]):
    """
    An evaluation experiment containing test cases and evaluators.

    Experiment organizes a collection of test cases and evaluates them all with
    the defined evaluators on some task.

    Attributes:
        cases: A list of test cases in the experiment.
        evaluators: The list of evaluators to be used on the test cases.

    Example:
        experiment = Experiment[str, str](
            cases=[
                Case(name="Simple Knowledge",
                        input="What is the capital of France?",
                        expected_output="The capital of France is Paris.",
                        expected_trajectory=[],
                        metadata={"category": "knowledge"}),
               Case(name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator"],
                        metadata={"category": "math"})
            ],
            evaluators=[
                OutputEvaluator(
                    rubric=(
                        "The output is relevant and complete. 0 if the output is incorrect or irrelevant."
                    )
                )
            ]
        )
    """

    def __init__(
        self,
        cases: list[Case[InputT, OutputT]] | None = None,
        evaluators: list[Evaluator[InputT, OutputT]] | None = None,
    ):
        self._cases = cases or []
        self._evaluators = evaluators or [Evaluator()]
        self._tracer = get_tracer()
        # self._logger = get_logger(__name__)

        self._config_id = os.environ.get("EVALUATION_RESULTS_LOG_GROUP", "default-strands-evals")

    @property
    def cases(self) -> list[Case[InputT, OutputT]]:
        """
        Get a deep copy of all test cases in the experiment.

        Returns deep copies to prevent accidental mutation of the original test cases.
        Users can safely modify the returned cases without affecting the experiment.

        Returns:
            List of Case objects (deep copies) containing all test cases in the experiment
        """
        return [case.model_copy(deep=True) for case in self._cases]

    @property
    def evaluators(self) -> list[Evaluator[InputT, OutputT]]:
        """
        Get the evaluators used for assessing test case performance.

        Returns:
            The list of evaluator instances configured for this experiment
        """
        return self._evaluators

    @cases.setter
    def cases(self, new_cases: list[Case[InputT, OutputT]]):
        """
        Set the test cases for this experiment.

        Args:
            new_cases: List of Case objects to use as the experiment's test cases
        """
        self._cases = new_cases

    @evaluators.setter
    def evaluators(self, new_evaluators: list[Evaluator[InputT, OutputT]]):
        """
        Set the evaluators for assessing test case performance.

        Args:
            new_evaluators: List of Evaluator instances to use for evaluating test cases
        """
        self._evaluators = new_evaluators

    def _run_task(
        self, task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]], case: Case[InputT, OutputT]
    ) -> EvaluationData[InputT, OutputT]:
        """
        Run the task with the inputs from the test case.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either
                OutputT or {"output": OutputT, "trajectory": ...}.
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
        # TODO afarn - I think this should take kwargs not just 'case'
        task_output = task(case)
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
        self, task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]], case: Case[InputT, OutputT]
    ) -> EvaluationData[InputT, OutputT]:
        """
        Run the task with the inputs from the test case asynchronously.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either
                OutputT or {"output": OutputT, "trajectory": ...}. The task can either run synchronously
                or asynchronously.
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
            task_output = await task(case)
        else:
            # Run sync function in separate thread to avoid blocking
            task_output = await asyncio.to_thread(task, case)

        if isinstance(task_output, dict):
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
            evaluation_context.actual_interactions = task_output.get("interactions")
            # allows the user to update the input in the task function
            new_input = task_output.get("input", None)
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
            trace_id = None

            try:
                with self._tracer.start_as_current_span(
                    f"execute_case {case_name}",
                ) as case_span:
                    evaluation_context = await self._run_task_async(task, case)
                    case_span.set_attributes(
                        {
                            "gen_ai.evaluation.data.input": serialize(evaluation_context.input),
                            "gen_ai.evaluation.data.expected_output": serialize(evaluation_context.expected_output),
                            "gen_ai.evaluation.data.actual_output": serialize(evaluation_context.actual_output),
                            "gen_ai.evaluation.data.has_trajectory": (evaluation_context.actual_trajectory is not None),
                            "gen_ai.evaluation.data.has_interactions": (
                                evaluation_context.actual_interactions is not None
                            ),
                        }
                    )
                    trace_id = format_trace_id(case_span.get_span_context().trace_id)

                # Evaluate with each evaluator
                evaluator_results = []
                for evaluator in self._evaluators:
                    with self._tracer.start_as_current_span(
                        f"evaluator {evaluator.get_type_name()}",
                    ) as eval_span:
                        evaluation_outputs = await evaluator.evaluate_async(evaluation_context)
                        (aggregate_score, aggregate_pass, aggregate_reason) = evaluator.aggregator(evaluation_outputs)

                        try:
                            label = _get_label_from_score(evaluator, aggregate_score)
                        except Exception:
                            label = "UNKNOWN"

                        eval_span.set_attributes(
                            {
                                "gen_ai.evaluation.score.label": label,
                                "gen_ai.evaluation.score.value": str(aggregate_score),
                                "gen_ai.evaluation.test_pass": aggregate_pass,
                                "gen_ai.evaluation.explanation": aggregate_reason or "",
                            }
                        )

                        evaluator_results.append(
                            {
                                "evaluator_name": evaluator.get_type_name(),
                                "test_pass": aggregate_pass,
                                "score": aggregate_score,
                                "reason": aggregate_reason or "",
                                "detailed_results": evaluation_outputs,
                            }
                        )

                        # CloudWatch logging for this evaluator
                        try:
                            evaluator_full_name = f"Custom.{evaluator.get_type_name()}"
                            region = os.environ.get("AWS_REGION", "us-east-1")
                            _config_arn = f"arn:aws:strands:{region}::strands-evaluation-empty-config/{self._config_id}"
                            _evaluator_arn = f"arn:aws:strands-evals:::evaluator/{evaluator_full_name}"

                            log_data = {
                                "gen_ai.evaluation.name": evaluator_full_name,
                                "gen_ai.evaluation.score.value": str(aggregate_score),
                                "gen_ai.evaluation.explanation": aggregate_reason or "",
                                "gen_ai.evaluation.score.label": label,
                                "gen_ai.response.id": trace_id,
                                "aws.bedrock_agentcore.evaluator.rating_scale": "Numerical",
                                "aws.bedrock_agentcore.evaluation_level": evaluator.evaluation_level or "Trace",
                                "event.name": "gen_ai.evaluation.result",
                                "aws.bedrock_agentcore.online_evaluation_config.arn": _config_arn,
                                "aws.bedrock_agentcore.online_evaluation_config.name": "strands-local-evaluation",
                                "aws.bedrock_agentcore.evaluator.arn": _evaluator_arn,
                                "session.id": case.session_id,
                            }

                            agent_observability_enabled = os.environ.get("AGENT_OBSERVABILITY_ENABLED", "")
                            if agent_observability_enabled:
                                _send_to_cloudwatch(
                                    message="gen_ai.evaluation.result",
                                    log_data=log_data,
                                    trace_id=trace_id,
                                    evaluator_name=evaluator_full_name,
                                    score=aggregate_score,
                                    config_id=self._config_id,
                                    label=label,
                                )
                        except Exception as e:
                            logger.debug(f"Skipping CloudWatch logging: {str(e)}")

                # Store results
                results.append(
                    {
                        "case": evaluation_context.model_dump(),
                        "evaluator_results": evaluator_results,
                    }
                )

            except Exception as e:
                # Handle task execution errors
                evaluator_results = []
                for evaluator in self._evaluators:
                    evaluator_results.append(
                        {
                            "evaluator_name": evaluator.get_type_name(),
                            "test_pass": False,
                            "score": 0,
                            "reason": f"An error occurred: {str(e)}",
                            "detailed_results": [],
                        }
                    )
                results.append(
                    {
                        "case": case.model_dump(),
                        "evaluator_results": evaluator_results,
                    }
                )
            finally:
                queue.task_done()

    def run_evaluations(
        self, task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]]
    ) -> list[EvaluationReport]:
        """
        Run the evaluations for all of the test cases with all evaluators.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either
                OutputT or {"output": OutputT, "trajectory": ...}.

        Return:
            A list of EvaluationReport objects, one for each evaluator, containing the overall score,
            individual case results, and basic feedback for each test case.
        """
        evaluator_data: dict[str, dict[str, list]] = {
            evaluator.get_type_name(): {
                "scores": [],
                "test_passes": [],
                "cases": [],
                "reasons": [],
                "detailed_results": [],
            }
            for evaluator in self._evaluators
        }

        for case in self._cases:
            case_name = case.name or f"case_{len(evaluator_data[self._evaluators[0].get_type_name()]['cases'])}"

            with self._tracer.start_as_current_span(
                f"eval_case {case_name}",
                attributes={
                    "gen_ai.evaluation.case.name": case_name,
                    "gen_ai.evaluation.case.input": serialize(case.input),
                },
            ) as case_span:
                # Task execution span - execute once
                try:
                    with self._tracer.start_as_current_span(
                        "task_execution",
                        attributes={
                            "gen_ai.evaluation.task.type": "agent_task",
                            "gen_ai.evaluation.case.name": case_name,
                        },
                    ) as task_span:
                        evaluation_context = self._run_task(task, case)
                        task_span.set_attributes(
                            {
                                "gen_ai.evaluation.data.input": serialize(evaluation_context.input),
                                "gen_ai.evaluation.data.expected_output": serialize(evaluation_context.expected_output),
                                "gen_ai.evaluation.data.actual_output": serialize(evaluation_context.actual_output),
                                "gen_ai.evaluation.data.has_trajectory": (
                                    evaluation_context.actual_trajectory is not None
                                ),
                                "gen_ai.evaluation.data.has_interactions": (
                                    evaluation_context.actual_interactions is not None
                                ),
                            }
                        )
                except Exception as e:
                    # Task execution failed - record failure for all evaluators and skip to next case
                    case_span.record_exception(e)
                    for evaluator in self._evaluators:
                        eval_name = evaluator.get_type_name()
                        evaluator_data[eval_name]["cases"].append(case.model_dump())
                        evaluator_data[eval_name]["test_passes"].append(False)
                        evaluator_data[eval_name]["scores"].append(0)
                        evaluator_data[eval_name]["reasons"].append(f"Task execution error: {str(e)}")
                        evaluator_data[eval_name]["detailed_results"].append([])
                    continue

                # Evaluate with each evaluator using the same task output
                # Each evaluator handles its own exceptions independently
                for evaluator in self._evaluators:
                    eval_name = evaluator.get_type_name()
                    try:
                        with self._tracer.start_as_current_span(
                            f"evaluator {evaluator.get_type_name()}",
                            attributes={
                                "gen_ai.evaluation.name": evaluator.get_type_name(),
                                "gen_ai.evaluation.case.name": case_name,
                            },
                        ) as eval_span:
                            evaluation_outputs = evaluator.evaluate(evaluation_context)
                            (aggregate_score, aggregate_pass, aggregate_reason) = evaluator.aggregator(
                                evaluation_outputs
                            )
                            eval_span.set_attributes(
                                {
                                    "gen_ai.evaluation.score.value": aggregate_score,
                                    "gen_ai.evaluation.test_pass": aggregate_pass,
                                    "gen_ai.evaluation.explanation": aggregate_reason or "",
                                }
                            )

                            evaluator_data[eval_name]["cases"].append(evaluation_context.model_dump())
                            evaluator_data[eval_name]["test_passes"].append(aggregate_pass)
                            evaluator_data[eval_name]["scores"].append(aggregate_score)
                            evaluator_data[eval_name]["reasons"].append(aggregate_reason or "")
                            evaluator_data[eval_name]["detailed_results"].append(evaluation_outputs)
                    except Exception as e:
                        # Individual evaluator failed - only affects this evaluator
                        evaluator_data[eval_name]["cases"].append(evaluation_context.model_dump())
                        evaluator_data[eval_name]["test_passes"].append(False)
                        evaluator_data[eval_name]["scores"].append(0)
                        evaluator_data[eval_name]["reasons"].append(f"Evaluator error: {str(e)}")
                        evaluator_data[eval_name]["detailed_results"].append([])

        reports = []
        for evaluator in self._evaluators:
            eval_name = evaluator.get_type_name()
            data = evaluator_data[eval_name]
            report = EvaluationReport(
                overall_score=sum(data["scores"]) / len(data["scores"]) if len(data["scores"]) else 0,
                scores=data["scores"],
                test_passes=data["test_passes"],
                cases=data["cases"],
                reasons=data["reasons"],
                detailed_results=data["detailed_results"],
            )
            reports.append(report)

        return reports

    async def run_evaluations_async(self, task: Callable, max_workers: int = 10) -> list[EvaluationReport]:
        """
        Run evaluations asynchronously using a queue for parallel processing.

        Args:
            task: The task function to run on each case. This function should take in InputT and returns
                either OutputT or {"output": OutputT, "trajectory": ...}. The task can either run
                synchronously or asynchronously.
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            List of EvaluationReport objects, one for each evaluator, containing evaluation results
        """
        queue: asyncio.Queue[Case[InputT, OutputT]] = asyncio.Queue()
        results: list[Any] = []

        for case in self._cases:
            queue.put_nowait(case)

        num_workers = min(max_workers, len(self._cases))

        workers = [asyncio.create_task(self._worker(queue, task, results)) for _ in range(num_workers)]

        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Organize results by evaluator
        evaluator_data: dict[str, dict[str, list]] = {
            evaluator.get_type_name(): {
                "scores": [],
                "test_passes": [],
                "cases": [],
                "reasons": [],
                "detailed_results": [],
            }
            for evaluator in self._evaluators
        }

        for result in results:
            case_data = result["case"]
            for eval_result in result["evaluator_results"]:
                eval_name = eval_result["evaluator_name"]
                evaluator_data[eval_name]["cases"].append(case_data)
                evaluator_data[eval_name]["scores"].append(eval_result["score"])
                evaluator_data[eval_name]["test_passes"].append(eval_result["test_pass"])
                evaluator_data[eval_name]["reasons"].append(eval_result["reason"])
                evaluator_data[eval_name]["detailed_results"].append(eval_result["detailed_results"])

        reports = []
        for evaluator in self._evaluators:
            eval_name = evaluator.get_type_name()
            data = evaluator_data[eval_name]
            scores = data["scores"]
            report = EvaluationReport(
                overall_score=sum(scores) / len(scores) if scores else 0,
                scores=scores,
                test_passes=data["test_passes"],
                cases=data["cases"],
                reasons=data["reasons"],
                detailed_results=data["detailed_results"],
            )
            reports.append(report)

        return reports

    def to_dict(self) -> dict:
        """
        Convert the experiment to a dictionary.

        Return:
            A dictionary representation of the experiment.
        """
        return {
            "cases": [case.model_dump() for case in self._cases],
            "evaluators": [evaluator.to_dict() for evaluator in self._evaluators],
        }

    def to_file(self, path: str):
        """
        Write the experiment to a JSON file.

        Args:
            path: The file path where the experiment will be saved. Can be:
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

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict, custom_evaluators: list[type[Evaluator]] | None = None):
        """
        Create an experiment from a dictionary.

        Args:
            data: A dictionary representation of the experiment.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            An Experiment object.
        """
        custom_evaluators = custom_evaluators or []
        cases: list[Case] = [Case.model_validate(case_data) for case_data in data["cases"]]
        default_evaluators: dict[str, type[Evaluator]] = {
            "Evaluator": Evaluator,
            "OutputEvaluator": OutputEvaluator,
            "TrajectoryEvaluator": TrajectoryEvaluator,
            "InteractionsEvaluator": InteractionsEvaluator,
        }
        all_evaluators: dict[str, type[Evaluator]] = {
            **default_evaluators,
            **{v.get_type_name(): v for v in custom_evaluators},
        }

        evaluators = []
        for evaluator_dict in data["evaluators"]:
            evaluator_type = evaluator_dict["evaluator_type"]
            evaluator_args = {k: v for k, v in evaluator_dict.items() if k != "evaluator_type"}

            if "model_id" in evaluator_args:
                evaluator_args["model"] = evaluator_args.pop("model_id")

            if evaluator_type in all_evaluators:
                evaluator = all_evaluators[evaluator_type](**evaluator_args)
                evaluators.append(evaluator)
            else:
                raise Exception(
                    f"Cannot find {evaluator_type}. Make sure the evaluator type is spelled correctly and "
                    f"all relevant custom evaluators are passed in."
                )

        return cls(cases=cases, evaluators=evaluators)

    @classmethod
    def from_file(cls, path: str, custom_evaluators: list[type[Evaluator]] | None = None):
        """
        Create an experiment from a JSON file.

        Args:
            path: Path to the JSON file.
            custom_evaluators: A list of relevant custom evaluators.

        Return:
            An Experiment object.

        Raises:
            ValueError: If the file does not have a .json extension.
        """
        file_path = Path(path)

        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}. Please provide a path with .json extension."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data, custom_evaluators)
