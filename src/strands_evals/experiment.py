import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from opentelemetry.trace import format_trace_id
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import Any, Generic

from .case import Case
from .evaluation_data_store import EvaluationDataStore
from .evaluators.deterministic import Contains, Equals, StartsWith, StateEquals, ToolCalled
from .evaluators.evaluator import Evaluator
from .evaluators.interactions_evaluator import InteractionsEvaluator
from .evaluators.output_evaluator import OutputEvaluator
from .evaluators.trajectory_evaluator import TrajectoryEvaluator
from .telemetry import get_tracer, serialize
from .telemetry._cloudwatch_logger import _send_to_cloudwatch
from .types.evaluation import EvaluationData, InputT, OutputT
from .types.evaluation_report import EvaluationReport
from .utils import is_throttling_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Retry configuration for handling throttling
_MAX_RETRY_ATTEMPTS = 6
_INITIAL_RETRY_DELAY = 4
_MAX_RETRY_DELAY = 240  # 4 minutes


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

    def _validate_case_names(self) -> None:
        """Validate that all cases have unique, non-None names.

        Raises:
            ValueError: If any case is missing a name or if names are not unique.
        """
        missing_names = [i for i, case in enumerate(self._cases) if case.name is None]
        if missing_names:
            raise ValueError(
                f"All cases must have a name when using an evaluation_data_store. "
                f"Cases at indices {missing_names} are missing names."
            )
        case_names = [case.name for case in self._cases]
        if len(case_names) != len(set(case_names)):
            raise ValueError("All case names must be unique when using an evaluation_data_store.")

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
            expected_environment_state=case.expected_environment_state,
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
            evaluation_context.actual_environment_state = task_output.get("environment_state")
            # allows the user to update the input in the task function
            new_input = task_output.get("input", None)
            if new_input is not None:
                evaluation_context.input = new_input
        else:
            evaluation_context.actual_output = task_output

        return evaluation_context

    async def _execute_task(
        self,
        task: Callable,
        case: Case[InputT, OutputT],
        case_name: str,
    ) -> EvaluationData[InputT, OutputT]:
        """Execute a task for a single case with retry logic, tracing, and optional result caching.

        Args:
            task: The task function to run on the case.
            case: The test case to execute.
            case_name: Display name for the case (used in spans and logs).

        Returns:
            EvaluationData containing the task's input, expected output, and actual output.

        Raises:
            Exception: The original exception from the last retry attempt if all retries are exhausted.
        """

        @retry(
            retry=retry_if_exception(is_throttling_error),
            stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
            reraise=True,
        )
        async def _run_with_retry(task=task, case=case):
            return await self._run_task_async(task, case)

        with self._tracer.start_as_current_span(
            f"task_execution {case_name}",
            attributes={
                "gen_ai.evaluation.task.type": "agent_task",
                "gen_ai.evaluation.case.name": case_name,
            },
        ) as task_span:
            try:
                evaluation_context = await _run_with_retry()
            except RetryError as e:
                original_exception = e.last_attempt.exception()
                if original_exception is None:
                    original_exception = Exception(f"Task execution failed after {_MAX_RETRY_ATTEMPTS} retries")
                logger.error(
                    f"Max retry attempts ({_MAX_RETRY_ATTEMPTS}) exceeded for task execution "
                    f"on case {case_name}. Last error: {str(original_exception)}"
                )
                raise original_exception from e

            task_span.set_attributes(
                {
                    "gen_ai.evaluation.data.input": serialize(evaluation_context.input),
                    "gen_ai.evaluation.data.expected_output": serialize(evaluation_context.expected_output),
                    "gen_ai.evaluation.data.actual_output": serialize(evaluation_context.actual_output),
                    "gen_ai.evaluation.data.has_trajectory": (evaluation_context.actual_trajectory is not None),
                    "gen_ai.evaluation.data.has_interactions": (evaluation_context.actual_interactions is not None),
                }
            )
        return evaluation_context

    async def _run_evaluator(
        self,
        evaluator: Evaluator,
        evaluation_context: EvaluationData,
        case_name: str,
        trace_id: str,
        session_id: str,
    ) -> dict:
        """Run a single evaluator against a case's evaluation context.

        Handles retry with exponential backoff, tracing, CloudWatch logging,
        and error isolation (failures are recorded as results, not raised).

        Args:
            evaluator: The evaluator to run.
            evaluation_context: The task result data to evaluate.
            case_name: Display name for the case (used in spans and logs).
            trace_id: The trace ID for CloudWatch logging.
            session_id: The session ID for CloudWatch logging.

        Returns:
            A dict with evaluator_name, test_pass, score, reason, and detailed_results.
        """

        @retry(
            retry=retry_if_exception(is_throttling_error),
            stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
            reraise=True,
        )
        async def _evaluate_with_retry(evaluator=evaluator, evaluation_context=evaluation_context):
            outputs = await evaluator.evaluate_async(evaluation_context)
            (score, passed, reason) = evaluator.aggregator(outputs)
            return outputs, float(score), passed, reason

        try:
            with self._tracer.start_as_current_span(
                f"evaluator {evaluator.get_type_name()}",
                attributes={
                    "gen_ai.evaluation.name": evaluator.get_type_name(),
                    "gen_ai.evaluation.case.name": case_name,
                },
            ) as eval_span:
                (
                    evaluation_outputs,
                    aggregate_score,
                    aggregate_pass,
                    aggregate_reason,
                ) = await _evaluate_with_retry()

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
                        "aws.bedrock_agentcore.evaluation_level": (evaluator.evaluation_level or "Trace"),
                        "event.name": "gen_ai.evaluation.result",
                        "aws.bedrock_agentcore.online_evaluation_config.arn": _config_arn,
                        "aws.bedrock_agentcore.online_evaluation_config.name": ("strands-local-evaluation"),
                        "aws.bedrock_agentcore.evaluator.arn": _evaluator_arn,
                        "session.id": session_id,
                    }

                    agent_observability_enabled = os.environ.get("AGENT_OBSERVABILITY_ENABLED", "")
                    if agent_observability_enabled:
                        _send_to_cloudwatch(
                            message="gen_ai.evaluation.result",
                            log_data=log_data,
                            trace_id=trace_id,
                            evaluator_name=evaluator_full_name,
                            score=cast(float, aggregate_score),
                            config_id=self._config_id,
                            label=label,
                        )
                except Exception as e:
                    logger.debug(f"Skipping CloudWatch logging: {str(e)}")

                return {
                    "evaluator_name": evaluator.get_type_name(),
                    "test_pass": aggregate_pass,
                    "score": aggregate_score,
                    "reason": aggregate_reason or "",
                    "detailed_results": evaluation_outputs,
                }

        except RetryError as e:
            # Max retries exceeded
            original_exception = e.last_attempt.exception()
            if original_exception is None:
                original_exception = Exception(
                    f"Evaluator {evaluator.get_type_name()} failed after {_MAX_RETRY_ATTEMPTS} retries"
                )
            logger.error(
                f"Max retry attempts ({_MAX_RETRY_ATTEMPTS}) exceeded for evaluator "
                f"{evaluator.get_type_name()} on case {case_name}. "
                f"Last error: {str(original_exception)}"
            )
            return {
                "evaluator_name": evaluator.get_type_name(),
                "test_pass": False,
                "score": 0,
                "reason": f"Evaluator error: {str(original_exception)}",
                "detailed_results": [],
            }
        except Exception as e:
            # Catch non-throttling errors and record as failure (error isolation)
            return {
                "evaluator_name": evaluator.get_type_name(),
                "test_pass": False,
                "score": 0,
                "reason": f"Evaluator error: {str(e)}",
                "detailed_results": [],
            }

    async def _worker(
        self,
        queue: asyncio.Queue,
        task: Callable,
        results: list,
        evaluation_data_store: EvaluationDataStore | None = None,
    ):
        """
        Worker that processes cases from the queue. Run evaluation on the task.

        Args:
            queue: Queue containing cases to process
            task: Task function to run on each case
            results: List to store results
            evaluation_data_store: Optional store for loading/saving evaluation data
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
                    attributes={
                        "gen_ai.evaluation.case.name": case_name,
                        "gen_ai.evaluation.case.input": serialize(case.input),
                    },
                ) as case_span:
                    try:
                        # Try loading cached result from store
                        cached_result = None
                        if evaluation_data_store is not None:
                            cached_result = await asyncio.to_thread(evaluation_data_store.load, case.name)

                        if cached_result is not None:
                            evaluation_context = cached_result
                        else:
                            evaluation_context = await self._execute_task(task, case, case_name)
                            if evaluation_data_store is not None:
                                await asyncio.to_thread(evaluation_data_store.save, case.name, evaluation_context)
                        trace_id = format_trace_id(case_span.get_span_context().trace_id)

                        # Evaluate with each evaluator
                        evaluator_results = []
                        for evaluator in self._evaluators:
                            result = await self._run_evaluator(
                                evaluator, evaluation_context, case_name, trace_id, case.session_id
                            )
                            evaluator_results.append(result)

                        # Store results
                        results.append(
                            {
                                "case": evaluation_context.model_dump(),
                                "evaluator_results": evaluator_results,
                            }
                        )

                    except Exception as e:
                        case_span.record_exception(e)
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
        self,
        task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]],
        evaluation_data_store: EvaluationDataStore | None = None,
    ) -> list[EvaluationReport]:
        """
        Run the evaluations for all of the test cases with all evaluators.

        Delegates to run_evaluations_async with max_workers=1 for sequential execution.

        Args:
            task: The task to run the test case on. This function should take in InputT and returns either
                OutputT or {"output": OutputT, "trajectory": ...}.
            evaluation_data_store: Optional store for loading/saving evaluation data. When provided, cached
                results are loaded instead of running the task, and new results are saved after task execution.

        Return:
            A list of EvaluationReport objects, one for each evaluator, containing the overall score,
            individual case results, and basic feedback for each test case.
        """
        if asyncio.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")

        return asyncio.run(self.run_evaluations_async(task, max_workers=1, evaluation_data_store=evaluation_data_store))

    async def run_evaluations_async(
        self, task: Callable, max_workers: int = 10, evaluation_data_store: EvaluationDataStore | None = None
    ) -> list[EvaluationReport]:
        """
        Run evaluations asynchronously using a queue for parallel processing.

        Args:
            task: The task function to run on each case. This function should take in InputT and returns
                either OutputT or {"output": OutputT, "trajectory": ...}. The task can either run
                synchronously or asynchronously.
            max_workers: Maximum number of parallel workers (default: 10)
            evaluation_data_store: Optional store for loading/saving evaluation data. When provided, cached
                results are loaded instead of running the task, and new results are saved after task execution.

        Returns:
            List of EvaluationReport objects, one for each evaluator, containing evaluation results
        """
        if evaluation_data_store is not None:
            self._validate_case_names()

        queue: asyncio.Queue[Case[InputT, OutputT]] = asyncio.Queue()
        results: list[Any] = []

        for case in self._cases:
            queue.put_nowait(case)

        num_workers = min(max_workers, len(self._cases))

        workers = [
            asyncio.create_task(self._worker(queue, task, results, evaluation_data_store)) for _ in range(num_workers)
        ]

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
                evaluator_name=eval_name,
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
            "Equals": Equals,
            "Contains": Contains,
            "StartsWith": StartsWith,
            "StateEquals": StateEquals,
            "ToolCalled": ToolCalled,
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

