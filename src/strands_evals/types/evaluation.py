from pydantic import BaseModel
from typing_extensions import Any, Generic, TypedDict, TypeVar, Union

from .trace import Session

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Interaction(TypedDict, total=False):
    """
    Represents a single interaction in a multi-agent or multi-step system.

    Used to capture the communication flow and dependencies between different
    components (agents, tools, or processing nodes) during task execution.
    All fields are optional to accommodate different interaction patterns.

    Attributes:
        node_name: Identifier for the agent, tool, or component involved in this interaction
        dependencies: List of other nodes/components this interaction depends on or references
        messages: Sequence of messages, responses, or communication exchanged during this interaction

    Example:
        interaction = {
            "node_name": "calculator_agent",
            "dependencies": ["input_parser", "math_validator"],
            "messages": ["Calculate 2+2"]
        }
    """

    node_name: str
    dependencies: list
    messages: list


class TaskOutput(TypedDict, total=False):
    """
    Structured output format for task functions that return complex results.

    Used when task functions need to return more than just the output response,
    such as trajectory or interaction history. All fields are optional
    to support different task complexity levels.

    Attributes:
        output: The primary response or result from the task
        trajectory: Sequence of steps, tools, or actions taken during task execution
        interactions: Communication flow between agents or components during execution
        input: A new input to replace the original in the evaluation, will not mutate the original test case

    Example:
        task_result = {
            "output": "The answer is 42",
            "trajectory": ["calculator", "validator"],
            "interactions": [{"node_name": "math_agent", "messages": ["Computing..."]}]
        }
    """

    output: Any
    trajectory: list[Any]
    interactions: list[Interaction]
    input: Any


class EvaluationData(BaseModel, Generic[InputT, OutputT]):
    """
    A record of all of the context for the evaluator to evaluate a test case.

    Attributes:
        input: The input to the task. eg. the query to the agent
        actual_output: The actual response given the input.
        expected_output: The expected response given the input.
        actual_trajectory: The actual trajectory of a task given the input.
        expected_trajectory: The expected trajectory of a task given the input.
        name: The name of the test case. This will be used to identify the test in the summary report.
        metadata: Additional information about the test case.
        actual_interactions: The actual interaction sequence given the input.
        expected_interactions: The expected interaction sequence given the input.
    """

    input: InputT
    actual_output: OutputT | None = None
    name: str | None = None
    expected_output: OutputT | None = None
    expected_trajectory: Union[list[Any], Session, None] = None
    actual_trajectory: Union[list[Any], Session, None] = None
    metadata: dict[str, Any] | None = None
    actual_interactions: list[Interaction] | None = None
    expected_interactions: list[Interaction] | None = None


class EvaluationOutput(BaseModel):
    """
    Structured output for LLM-based judge.

    Attributes:
        score: The score of the test case.
        test_pass: Whether the test pass or fail.
        reason: The reason for the score for each test case.
    """

    score: float
    test_pass: bool
    reason: str | None = None
