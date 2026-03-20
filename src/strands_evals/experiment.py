import asyncio
import inspect
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

    def _record_evaluator_result(
        self,
        evaluator_data: dict[str, dict[str, list]],
        eval_name: str,
        case_data: dict,
        test_pass: bool,
        score: float,
        reason: str,
        detailed_results: list,
    ):
        """
        Record a single evaluator result in the evaluator_data dictionary.

        Args:
            evaluator_data: Dictionary to store evaluator results
            eval_name: Name of the evaluator
            case_data: Case data (already serialized as dict)
            test_pass: Whether the test passed
            score: Evaluation score
            reason: Reason/explanation for the result
            detailed_results: Detailed evaluation outputs
        """
        evaluator_data[eval_name]["cases"].append(case_data)
        evaluator_data[eval_name]["test_passes"].append(test_pass)
        evaluator_data[eval_name]["scores"].append(score)
        evaluator_data[eval_name]["reasons"].append(reason)
        evaluator_data[eval_name]["detailed_results"].append(detailed_results)

    def _build_reports(self, evaluator_data: dict[str, dict[str, list]]) -> list[EvaluationReport]:
        """Build EvaluationReport objects from collected evaluator data.

        Args:
            evaluator_data: Dictionary keyed by evaluator name containing scores, test_passes,
                cases, reasons, and detailed_results lists.

        Returns:
            A list of EvaluationReport objects, one per evaluator.
        """
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

    def _init_evaluator_data(self) -> dict[str, dict[str, list]]:
        """Initialize the evaluator data dictionary for collecting results."""
        return {
            evaluator.get_type_name(): {
                "scores": [],
                "test_passes": [],
                "cases": [],
                "reasons": [],
                "detailed_results": [],
            }
            for evaluator in self._evaluators
        }

    def _log_evaluation_to_cloudwatch(
        self,
        evaluator: Evaluator,
        eval_span: Any,
        evaluation_context: EvaluationData,
        score: float,
        reason: str,
    ):
        """Log evaluation result to CloudWatch if observability is enabled.

        Args:
            evaluator: The evaluator that produced the result
            eval_span: The OpenTelemetry span for this evaluation
            evaluation_context: The evaluation data being evaluated
            score: The aggregate evaluation score
            reason: The aggregate evaluation reason
        """
        try:
            label = _get_label_from_score(evaluator, score)
        except Exception:
            label = "UNKNOWN"

        eval_span.set_attributes({"gen_ai.evaluation.score.label": label})

        try:
            trace_id = format_trace_id(eval_span.get_span_context().trace_id)
            session_id = (evaluation_context.metadata or {}).get("session_id", "")
            evaluator_full_name = f"Custom.{evaluator.get_type_name()}"
            region = os.environ.get("AWS_REGION", "us-east-1")
            _config_arn = f"arn:aws:strands:{region}::strands-evaluation-empty-config/{self._config_id}"
            _evaluator_arn = f"arn:aws:strands-evals:::evaluator/{evaluator_full_name}"

            log_data = {
                "gen_ai.evaluation.name": evaluator_full_name,
                "gen_ai.evaluation.score.value": str(score),
                "gen_ai.evaluation.explanation": reason or "",
                "gen_ai.evaluation.score.label": label,
                "gen_ai.response.id": trace_id,
                "aws.bedrock_agentcore.evaluator.rating_scale": "Numerical",
                "aws.bedrock_agentcore.evaluation_level": evaluator.evaluation_level or "Trace",
                "event.name": "gen_ai.evaluation.result",
                "aws.bedrock_agentcore.online_evaluation_config.arn": _config_arn,
                "aws.bedrock_agentcore.online_evaluation_config.name": "strands-local-evaluation",
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
                    score=cast(float, score),
                    config_id=self._config_id,
                    label=label,
                )
        except Exception as e:
            logger.debug("error=<%s> | skipping cloudwatch logging", e)

    def _set_task_span_attributes(self, task_span: Any, evaluation_context: EvaluationData) -> None:
        """Set standard attributes on a task execution span."""
        task_span.set_attributes(
            {
                "gen_ai.evaluation.data.input": serialize(evaluation_context.input),
                "gen_ai.evaluation.data.expected_output": serialize(evaluation_context.expected_output),
                "gen_ai.evaluation.data.actual_output": serialize(evaluation_context.actual_output),
                "gen_ai.evaluation.data.has_trajectory": evaluation_context.actual_trajectory is not None,
                "gen_ai.evaluation.data.has_interactions": evaluation_context.actual_interactions is not None,
            }
        )

    def _extract_task_error(self, e: Exception, case_name: str) -> str:
        """Extract error message from a task execution exception and log it."""
        if isinstance(e, RetryError):
            original_exception = e.last_attempt.exception()
            if original_exception is None:
                original_exception = Exception(f"Task execution failed after {_MAX_RETRY_ATTEMPTS} retries")
            error_msg = str(original_exception)
        else:
            error_msg = str(e)
        logger.error("case=<%s>, error=<%s> | task execution failed", case_name, error_msg)
        return error_msg

    def _build_failed_evaluation_data(
        self, case: Case[InputT, OutputT], error_msg: str
    ) -> EvaluationData[InputT, OutputT]:
        """Build an EvaluationData record for a failed task execution."""
        return EvaluationData(
            name=case.name,
            input=case.input,
            actual_output=None,
            expected_output=case.expected_output,
            expected_trajectory=case.expected_trajectory,
            expected_interactions=case.expected_interactions,
            expected_environment_state=case.expected_environment_state,
            metadata={**(case.metadata or {}), "task_error": error_msg},
        )

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
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")

        metadata = {**(case.metadata or {}), "session_id": case.session_id}
        evaluation_context = EvaluationData(
            name=case.name,
            input=case.input,
            expected_output=case.expected_output,
            expected_trajectory=case.expected_trajectory,
            expected_interactions=case.expected_interactions,
            expected_environment_state=case.expected_environment_state,
            metadata=metadata,
        )
        task_output = task(case)
        if isinstance(task_output, dict):  # could be evaluating the trajectory as well
            evaluation_context.actual_output = task_output.get("output")
            evaluation_context.actual_trajectory = task_output.get("trajectory")
            evaluation_context.actual_interactions = task_output.get("interactions")
            evaluation_context.actual_environment_state = task_output.get("environment_state")
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
        metadata = {**(case.metadata or {}), "session_id": case.session_id}
        evaluation_context = EvaluationData(
            name=case.name,
            input=case.input,
            expected_output=case.expected_output,
            expected_trajectory=case.expected_trajectory,
            expected_interactions=case.expected_interactions,
            expected_environment_state=case.expected_environment_state,
            metadata=metadata,
        )

        # Handle both async and sync tasks
        if inspect.iscoroutinefunction(task):
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

    def run_tasks(
        self, task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]]
    ) -> list[EvaluationData[InputT, OutputT]]:
        """Run the task against all cases and return the evaluation data.

        Executes the task function against each case with retry logic for throttling errors,
        without running any evaluators. The returned EvaluationData can be passed to
        run_evaluators() later, enabling reuse of task results across multiple evaluation runs.

        Args:
            task: The task to run on each case. Takes a Case and returns either
                OutputT or {"output": OutputT, "trajectory": ..., "interactions": ..., "environment_state": ...}.

        Returns:
            A list of EvaluationData objects, one per case. On task failure, the corresponding
            EvaluationData will have actual_output=None and the error in metadata["task_error"].
        """
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_tasks_async instead.")

        results: list[EvaluationData[InputT, OutputT]] = []

        for case in self._cases:
            case_name = case.name or f"case_{len(results)}"

            with self._tracer.start_as_current_span(
                f"run_task {case_name}",
                attributes={
                    "gen_ai.evaluation.case.name": case_name,
                    "gen_ai.evaluation.case.input": serialize(case.input),
                },
            ) as run_task_span:

                @retry(
                    retry=retry_if_exception(is_throttling_error),
                    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
                    wait=wait_exponential(multiplier=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
                    reraise=True,
                )
                def _run_task_with_retry(task=task, case=case):
                    return self._run_task(task, case)

                try:
                    with self._tracer.start_as_current_span(
                        "task_execution",
                        attributes={
                            "gen_ai.evaluation.task.type": "agent_task",
                            "gen_ai.evaluation.case.name": case_name,
                        },
                    ) as task_span:
                        evaluation_context = _run_task_with_retry()
                        self._set_task_span_attributes(task_span, evaluation_context)
                        results.append(evaluation_context)
                except Exception as e:
                    error_msg = self._extract_task_error(e, case_name)
                    run_task_span.record_exception(e)
                    results.append(self._build_failed_evaluation_data(case, error_msg))

        return results

    async def _task_worker(
        self,
        queue: asyncio.Queue,
        task: Callable,
        results: list[EvaluationData[InputT, OutputT] | None],
    ):
        """Worker that processes cases from the queue for task execution only.

        Args:
            queue: Queue containing (index, case) tuples to process
            task: Task function to run on each case
            results: Pre-allocated list to store EvaluationData results by index
        """
        while True:
            try:
                index, case = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            case_name = case.name or f"case_{index}"

            try:
                with self._tracer.start_as_current_span(
                    f"run_task {case_name}",
                    attributes={
                        "gen_ai.evaluation.case.name": case_name,
                        "gen_ai.evaluation.case.input": serialize(case.input),
                    },
                ) as run_task_span:

                    @retry(
                        retry=retry_if_exception(is_throttling_error),
                        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
                        wait=wait_exponential(multiplier=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
                        reraise=True,
                    )
                    async def _run_task_with_retry(task=task, case=case):
                        return await self._run_task_async(task, case)

                    try:
                        with self._tracer.start_as_current_span(
                            "task_execution",
                            attributes={
                                "gen_ai.evaluation.task.type": "agent_task",
                                "gen_ai.evaluation.case.name": case_name,
                            },
                        ) as task_span:
                            evaluation_context = await _run_task_with_retry()
                            self._set_task_span_attributes(task_span, evaluation_context)
                            results[index] = evaluation_context
                    except Exception as e:
                        error_msg = self._extract_task_error(e, case_name)
                        run_task_span.record_exception(e)
                        results[index] = self._build_failed_evaluation_data(case, error_msg)
            finally:
                queue.task_done()

    async def run_tasks_async(
        self,
        task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]],
        max_workers: int = 10,
    ) -> list[EvaluationData[InputT, OutputT]]:
        """Run the task against all cases asynchronously and return the evaluation data.

        Args:
            task: The task to run on each case. Can be sync or async.
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            A list of EvaluationData objects, one per case. On task failure, the corresponding
            EvaluationData will have actual_output=None and the error in metadata["task_error"].
        """
        queue: asyncio.Queue[tuple[int, Case[InputT, OutputT]]] = asyncio.Queue()
        results: list[EvaluationData[InputT, OutputT] | None] = [None] * len(self._cases)

        for index, case in enumerate(self._cases):
            queue.put_nowait((index, case))

        num_workers = min(max_workers, len(self._cases))

        workers = [asyncio.create_task(self._task_worker(queue, task, results)) for _ in range(num_workers)]

        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        return cast(list[EvaluationData[InputT, OutputT]], results)

    def run_evaluators(self, evaluation_data: list[EvaluationData[InputT, OutputT]]) -> list[EvaluationReport]:
        """Run evaluators on pre-computed evaluation data.

        Evaluates each EvaluationData item with all configured evaluators, without executing
        any task functions. Use with run_tasks() to separate task execution from evaluation.

        Args:
            evaluation_data: Pre-computed evaluation data, e.g. from run_tasks() or deserialized
                from a previous run.

        Returns:
            A list of EvaluationReport objects, one per evaluator.
        """
        evaluator_data = self._init_evaluator_data()

        for evaluation_context in evaluation_data:
            first_eval_name = self._evaluators[0].get_type_name()
            case_name = evaluation_context.name or f"case_{len(evaluator_data[first_eval_name]['cases'])}"

            # If task execution failed, record failure for all evaluators without running them
            task_error = (evaluation_context.metadata or {}).get("task_error")
            if task_error:
                for evaluator in self._evaluators:
                    self._record_evaluator_result(
                        evaluator_data=evaluator_data,
                        eval_name=evaluator.get_type_name(),
                        case_data=evaluation_context.model_dump(),
                        test_pass=False,
                        score=0,
                        reason=f"Task execution error: {task_error}",
                        detailed_results=[],
                    )
                continue

            with self._tracer.start_as_current_span(
                f"eval_case {case_name}",
                attributes={
                    "gen_ai.evaluation.case.name": case_name,
                    "gen_ai.evaluation.case.input": serialize(evaluation_context.input),
                },
            ):
                for evaluator in self._evaluators:
                    eval_name = evaluator.get_type_name()

                    @retry(
                        retry=retry_if_exception(is_throttling_error),
                        stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
                        wait=wait_exponential(multiplier=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
                        reraise=True,
                    )
                    def _evaluate_with_retry(evaluator=evaluator, evaluation_context=evaluation_context):
                        outputs = evaluator.evaluate(evaluation_context)
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
                            evaluation_outputs, aggregate_score, aggregate_pass, aggregate_reason = (
                                _evaluate_with_retry()
                            )
                            eval_span.set_attributes(
                                {
                                    "gen_ai.evaluation.score.value": str(aggregate_score),
                                    "gen_ai.evaluation.test_pass": aggregate_pass,
                                    "gen_ai.evaluation.explanation": aggregate_reason or "",
                                }
                            )

                            self._record_evaluator_result(
                                evaluator_data=evaluator_data,
                                eval_name=eval_name,
                                case_data=evaluation_context.model_dump(),
                                test_pass=aggregate_pass,
                                score=aggregate_score,
                                reason=aggregate_reason or "",
                                detailed_results=evaluation_outputs,
                            )

                            self._log_evaluation_to_cloudwatch(
                                evaluator=evaluator,
                                eval_span=eval_span,
                                evaluation_context=evaluation_context,
                                score=aggregate_score,
                                reason=aggregate_reason or "",
                            )
                    except RetryError as e:
                        original_exception = e.last_attempt.exception()
                        if original_exception is None:
                            original_exception = Exception(
                                f"Evaluator {evaluator.get_type_name()} failed after {_MAX_RETRY_ATTEMPTS} retries"
                            )
                        logger.error(
                            "evaluator=<%s>, case=<%s>, retries=<%s>, error=<%s> | max retry attempts exceeded",
                            evaluator.get_type_name(),
                            case_name,
                            _MAX_RETRY_ATTEMPTS,
                            original_exception,
                        )
                        self._record_evaluator_result(
                            evaluator_data=evaluator_data,
                            eval_name=eval_name,
                            case_data=evaluation_context.model_dump(),
                            test_pass=False,
                            score=0,
                            reason=f"Evaluator error: {str(original_exception)}",
                            detailed_results=[],
                        )
                    except Exception as e:
                        self._record_evaluator_result(
                            evaluator_data=evaluator_data,
                            eval_name=eval_name,
                            case_data=evaluation_context.model_dump(),
                            test_pass=False,
                            score=0,
                            reason=f"Evaluator error: {str(e)}",
                            detailed_results=[],
                        )

        return self._build_reports(evaluator_data)

    async def _evaluator_worker(
        self,
        queue: asyncio.Queue,
        results: list[dict | None],
    ):
        """Worker that processes EvaluationData from the queue for evaluation only.

        Args:
            queue: Queue containing (index, EvaluationData) tuples to evaluate
            results: Pre-allocated list to store per-case evaluator results by index
        """
        while True:
            try:
                index, evaluation_context = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            case_name = evaluation_context.name or f"case_{index}"

            # If task execution failed, record failure for all evaluators
            task_error = (evaluation_context.metadata or {}).get("task_error")
            if task_error:
                evaluator_results = []
                for evaluator in self._evaluators:
                    evaluator_results.append(
                        {
                            "evaluator_name": evaluator.get_type_name(),
                            "test_pass": False,
                            "score": 0,
                            "reason": f"Task execution error: {task_error}",
                            "detailed_results": [],
                        }
                    )
                results[index] = {
                    "case": evaluation_context.model_dump(),
                    "evaluator_results": evaluator_results,
                }
                queue.task_done()
                continue

            try:
                evaluator_results = []
                with self._tracer.start_as_current_span(
                    f"eval_case {case_name}",
                    attributes={
                        "gen_ai.evaluation.case.name": case_name,
                        "gen_ai.evaluation.case.input": serialize(evaluation_context.input),
                    },
                ):
                    for evaluator in self._evaluators:

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

                                evaluator_results.append(
                                    {
                                        "evaluator_name": evaluator.get_type_name(),
                                        "test_pass": aggregate_pass,
                                        "score": aggregate_score,
                                        "reason": aggregate_reason or "",
                                        "detailed_results": evaluation_outputs,
                                    }
                                )

                                self._log_evaluation_to_cloudwatch(
                                    evaluator=evaluator,
                                    eval_span=eval_span,
                                    evaluation_context=evaluation_context,
                                    score=aggregate_score,
                                    reason=aggregate_reason or "",
                                )

                        except RetryError as e:
                            original_exception = e.last_attempt.exception()
                            if original_exception is None:
                                original_exception = Exception(
                                    f"Evaluator {evaluator.get_type_name()} failed after {_MAX_RETRY_ATTEMPTS} retries"
                                )
                            logger.error(
                                "evaluator=<%s>, case=<%s>, retries=<%s>, error=<%s> | max retry attempts exceeded",
                                evaluator.get_type_name(),
                                case_name,
                                _MAX_RETRY_ATTEMPTS,
                                original_exception,
                            )
                            evaluator_results.append(
                                {
                                    "evaluator_name": evaluator.get_type_name(),
                                    "test_pass": False,
                                    "score": 0,
                                    "reason": f"Evaluator error: {str(original_exception)}",
                                    "detailed_results": [],
                                }
                            )
                        except Exception as e:
                            evaluator_results.append(
                                {
                                    "evaluator_name": evaluator.get_type_name(),
                                    "test_pass": False,
                                    "score": 0,
                                    "reason": f"Evaluator error: {str(e)}",
                                    "detailed_results": [],
                                }
                            )

                results[index] = {
                    "case": evaluation_context.model_dump(),
                    "evaluator_results": evaluator_results,
                }
            finally:
                queue.task_done()

    async def run_evaluators_async(
        self,
        evaluation_data: list[EvaluationData[InputT, OutputT]],
        max_workers: int = 10,
    ) -> list[EvaluationReport]:
        """Run evaluators on pre-computed evaluation data asynchronously.

        Args:
            evaluation_data: Pre-computed evaluation data, e.g. from run_tasks_async() or deserialized
                from a previous run.
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            A list of EvaluationReport objects, one per evaluator.
        """
        queue: asyncio.Queue[tuple[int, EvaluationData[InputT, OutputT]]] = asyncio.Queue()
        results: list[dict | None] = [None] * len(evaluation_data)

        for index, data in enumerate(evaluation_data):
            queue.put_nowait((index, data))

        num_workers = min(max_workers, len(evaluation_data)) if evaluation_data else 0

        workers = [asyncio.create_task(self._evaluator_worker(queue, results)) for _ in range(num_workers)]

        await queue.join()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Organize results by evaluator
        evaluator_data = self._init_evaluator_data()

        for result in results:
            if result is None:
                continue
            case_data = result["case"]
            for eval_result in result["evaluator_results"]:
                eval_name = eval_result["evaluator_name"]
                evaluator_data[eval_name]["cases"].append(case_data)
                evaluator_data[eval_name]["scores"].append(eval_result["score"])
                evaluator_data[eval_name]["test_passes"].append(eval_result["test_pass"])
                evaluator_data[eval_name]["reasons"].append(eval_result["reason"])
                evaluator_data[eval_name]["detailed_results"].append(eval_result["detailed_results"])

        return self._build_reports(evaluator_data)

    def run_evaluations(
        self, task: Callable[[Case[InputT, OutputT]], OutputT | dict[str, Any]]
    ) -> list[EvaluationReport]:
        """Run tasks and evaluators in sequence.

        Convenience method equivalent to run_evaluators(run_tasks(task)).

        Args:
            task: The task to run on each case. Takes a Case and returns either
                OutputT or {"output": OutputT, "trajectory": ...}.

        Returns:
            A list of EvaluationReport objects, one per evaluator.
        """
        return self.run_evaluators(self.run_tasks(task))

    async def run_evaluations_async(self, task: Callable, max_workers: int = 10) -> list[EvaluationReport]:
        """Run tasks and evaluators asynchronously in sequence.

        Convenience method equivalent to run_evaluators_async(run_tasks_async(task)).

        Args:
            task: The task function to run on each case. Can be sync or async.
            max_workers: Maximum number of parallel workers (default: 10)

        Returns:
            A list of EvaluationReport objects, one per evaluator.
        """
        evaluation_data = await self.run_tasks_async(task, max_workers)
        return await self.run_evaluators_async(evaluation_data, max_workers)

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
