import inspect
import logging

from typing_extensions import Any, Generic, TypeGuard, TypeVar

from ..extractors import TraceExtractor
from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.trace import AssistantMessage, Context, EvaluationLevel, Session, TextContent, ToolConfig, UserMessage

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """

    # Optional: subclasses can set this to enable trace parsing
    evaluation_level: EvaluationLevel | None = None
    _trace_extractor: TraceExtractor | None = None

    def __init__(self, trace_extractor: TraceExtractor | None = None):
        """Initialize evaluator with optional custom trace extractor.

        Args:
            trace_extractor: Custom trace extractor. If None and evaluation_level is set,
                           a default TraceExtractor will be created.
        """
        self.aggregator = self._default_aggregator
        if trace_extractor:
            self._trace_extractor = trace_extractor
        elif self.evaluation_level:
            self._trace_extractor = TraceExtractor(self.evaluation_level)

    @staticmethod
    def _default_aggregator(outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        avg_score = sum(o.score for o in outputs) / len(outputs)
        all_pass = all(o.test_pass for o in outputs)
        combined_reason = " | ".join(o.reason for o in outputs if o.reason)
        return avg_score, all_pass, combined_reason

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses, especially if you want to run evaluations asynchronously."
        )

    def _parse_trajectory(self, evaluation_case: EvaluationData[InputT, OutputT]) -> Any:
        """Parse Session trajectory using TraceExtractor."""
        if not self._trace_extractor:
            raise ValueError("No trace extractor configured. Set evaluation_level or provide trace_extractor.")

        trajectory = evaluation_case.actual_trajectory
        if not isinstance(trajectory, Session):
            raise TypeError(
                f"Trace parsing requires actual_trajectory to be a Session object, got {type(trajectory).__name__}."
            )

        return self._trace_extractor.extract(trajectory)

    def _format_tools(self, tools: list[ToolConfig]) -> str:
        """Format available tools for prompt display."""
        return "\n".join([f"- {tool.name}: {tool.description or 'No description'}" for tool in tools])

    def _format_session_history(self, contexts: list[Context]) -> str:
        """Format session history with tool executions for prompt display."""
        lines = []
        for ctx in contexts:
            lines.append(f"User: {ctx.user_prompt.text}")
            if ctx.tool_execution_history:
                for tool_exec in ctx.tool_execution_history:
                    lines.append(f"Action: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                    lines.append(f"Tool: {tool_exec.tool_result.content}")
            lines.append(f"Assistant: {ctx.agent_response.text}")
        return "\n".join(lines)

    def _has_text_content(self, msg: UserMessage | AssistantMessage) -> TypeGuard[UserMessage | AssistantMessage]:
        """Check if a message object has accessible text content.

        Args:
            msg: Message object to check (UserMessage or AssistantMessage)

        Returns:
            True if msg has content attribute with at least one item that is TextContent
        """
        return (
            hasattr(msg, "content")
            and bool(msg.content)
            and len(msg.content) > 0
            and isinstance(msg.content[0], TextContent)
        )

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the name of the evaluator type.

        Returns:
            str: The name of the evaluator type.
        """
        return cls.__name__

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.

        Returns:
            dict: A dictionary containing the evaluator's information. Omit private attributes
            (attributes starting with '_') and attributes with default values.
        """
        _dict = {"evaluator_type": self.get_type_name()}

        # Get default values from __init__ signature
        sig = inspect.signature(self.__class__.__init__)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
        exclude_attrs = {"aggregator"}
        for k, v in self.__dict__.items():
            if not k.startswith("_") and k not in exclude_attrs and (k not in defaults or v != defaults[k]):
                _dict[k] = v
        return _dict
