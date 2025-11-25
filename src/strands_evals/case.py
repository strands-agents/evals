import uuid

from pydantic import BaseModel, Field
from typing_extensions import Any, Generic, TypeVar

from .types.evaluation import Interaction

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Case(BaseModel, Generic[InputT, OutputT]):
    """
    A single test case, representing a row in an Experiment.

    Each test case represents a single test scenario with inputs to test.
    Optionally, a test case may contains a name, expected outputs, expected trajectory, expected interactions
    and arbitrary metadata.

    Attributes:
        input: The input to the task. eg. the query to the agent
        name: The name of the test case. This will be used to identify the test in the summary report.
        session_id: The session ID for the test case. Automatically generates a UUID4 if not provided.
        expected_output: The expected response given the input. eg. the agent's response
        expected_trajectory: The expected trajectory of a task given the input. eg. sequence of tools
        expected_interactions: The expected interaction sequence given the input (ideal for multi-agent systems).
        metadata: Additional information about the test case.

    Example:
        case = Case[str,str](name="Simple Math",
                        input="What is 2x2?",
                        expected_output="2x2 is 4.",
                        expected_trajectory=["calculator],
                        metadata={"category": "math"})

        simple_test_case = Case(input="What is 2x2?")

        case_with_interaction = Case(
                    input="What is 2x2?",
                    expected_interactions=[
                        {"agent_1":"Hello, what would you like to do?"},
                        {"agent_2":"What is 2x2?"}
                    ]
                )
    """

    name: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: InputT
    expected_output: OutputT | None = None
    expected_trajectory: list[Any] | None = None
    expected_interactions: list[Interaction] | None = None
    metadata: dict[str, Any] | None = None
