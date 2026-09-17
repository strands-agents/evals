import asyncio
import inspect
import logging
from collections.abc import Callable

from strands.models.model import Model
from typing_extensions import Any, Generic, Literal, TypeGuard, cast, get_args

from ..detectors.chunking import would_exceed_context
from ..detectors.constants import DEFAULT_MAX_INPUT_TOKENS
from ..extractors import TraceExtractor
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import (
    AssistantMessage,
    Context,
    EvaluationLevel,
    Session,
    TextContent,
    ToolConfig,
    ToolLevelInput,
    TraceLevelInput,
    UserMessage,
)
from ._trace_index import TraceIndex

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# The `disclosure` knob on judge evaluators. Exported as a type so the public
# kwarg is statically checkable; `DISCLOSURE_MODES` is derived from it (single
# source of truth) and used by the runtime validator.
DisclosureMode = Literal["auto", "always", "never"]
DISCLOSURE_MODES: tuple[str, ...] = get_args(DisclosureMode)

# Judge input context windows (tokens) by model-id substring. Intentionally
# coarse: an unknown model falls back to DEFAULT_MAX_INPUT_TOKENS. A wrong guess
# only shifts when disclosure engages; it does not change the score of a fitting
# case, and under "auto" a large-enough underestimate simply inlines and lets the
# judge raise its own context-length error (see `_render_with_disclosure`).
_JUDGE_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("nova-micro", 128_000),
    ("nova-lite", 300_000),
    ("nova-pro", 300_000),
    ("claude", 200_000),
)


class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """

    # Optional: subclasses can set this to enable trace parsing
    evaluation_level: EvaluationLevel | None = None
    _trace_extractor: TraceExtractor | None = None

    # Trace-disclosure mode for judges that inline a Session trajectory. Subclasses
    # that accept a `disclosure` argument set it per-instance; this class default
    # keeps `self.disclosure` resolvable for evaluators that don't expose the knob.
    disclosure: DisclosureMode = "auto"

    def __init__(self, trace_extractor: TraceExtractor | None = None, name: str | None = None):
        """Initialize evaluator with optional custom trace extractor.

        Args:
            trace_extractor: Custom trace extractor. If None and evaluation_level is set,
                           a default TraceExtractor will be created.
            name: Instance-level identifier used as the evaluator tag in
                `EvaluationReport.cases[i]["evaluator"]` and as
                `gen_ai.evaluation.name` on emitted spans/logs. When two
                instances of the same class run in one experiment (e.g.,
                `Contains(value="x")` and `Contains(value="y")`), distinct
                names keep their results from colliding. Defaults to the
                class name when unset.
        """
        self.aggregator = self._default_aggregator
        self.name = name
        if trace_extractor:
            self._trace_extractor = trace_extractor
        elif self.evaluation_level:
            self._trace_extractor = TraceExtractor(self.evaluation_level)

    def _get_model_id(self, model: Model | str | None) -> str:
        """Extract model_id from a Model instance or string for serialization.

        This helper method should be called in subclass __init__ methods that accept a model parameter.

        Args:
            model: Model instance, string model ID, or None

        Returns:
            The model ID string, DEFAULT_BEDROCK_MODEL_ID if None, or empty string for invalid types
        """
        if isinstance(model, str):
            return model
        elif isinstance(model, Model) and hasattr(model, "config") and isinstance(model.config, dict):
            return model.config.get("model_id", "")
        elif model is None:
            return DEFAULT_BEDROCK_MODEL_ID
        else:
            return ""

    @staticmethod
    def _validate_disclosure(disclosure: str) -> DisclosureMode:
        """Validate a `disclosure` argument, returning it (narrowed) when valid."""
        if disclosure not in DISCLOSURE_MODES:
            raise ValueError(f"disclosure must be one of {DISCLOSURE_MODES}, got {disclosure!r}")
        return cast(DisclosureMode, disclosure)

    def _judge_window_tokens(self) -> int:
        """Resolve the judge model's input context window in tokens.

        Falls back to `DEFAULT_MAX_INPUT_TOKENS` for models not in the table, so the
        overflow preflight uses the window of the model actually judging rather than
        assuming the default everywhere (a larger judge should disclose less often).
        """
        model_id = self._get_model_id(getattr(self, "model", None)).lower()
        for needle, window in _JUDGE_CONTEXT_WINDOWS:
            if needle in model_id:
                return window
        return DEFAULT_MAX_INPUT_TOKENS

    @staticmethod
    def _session_of(evaluation_case: EvaluationData[InputT, OutputT]) -> Session | None:
        """The Session trajectory to index for disclosure, or None when there isn't one."""
        trajectory = evaluation_case.actual_trajectory
        return trajectory if isinstance(trajectory, Session) else None

    def _render_with_disclosure(
        self,
        evaluation_case: EvaluationData[InputT, OutputT],
        render: Callable[[TraceIndex | None], str],
    ) -> tuple[str, list[Any]]:
        """Render a judge prompt, falling back to trace tools when it would overflow.

        `render(None)` builds the prompt with the trajectory inlined (today's
        behavior); `render(index)` builds it with a paged ``<TraceOverview>`` in
        place of the inlined trajectory. Returns ``(prompt, tools)``: on the inline
        path `tools` is empty and the prompt is byte-identical to before, so a case
        that fits the judge window is unchanged. On the disclosure path the judge
        gets the overview plus `list_spans` / `get_span` / `search_spans` and reads
        the trace on demand instead of overflowing on an inlined dump.

        Modes (`self.disclosure`): ``"auto"`` discloses only on a preflight overflow;
        ``"always"`` discloses whenever a Session trajectory is present; ``"never"``
        always inlines, restoring the prior behavior where a real overflow surfaces
        as the judge model's own context-length error.

        Only ``"auto"`` needs the size probe, so the inline prompt is built once and
        reused as both the probe and the fitting-case result. Under ``"always"`` /
        ``"never"`` the decision is size-independent, so the (potentially large)
        inline render is skipped unless it is the one actually returned.
        """
        if self.disclosure == "auto":
            inline_prompt = render(None)
            index = self._resolve_disclosure_index(evaluation_case, inline_prompt)
            if index is None:
                return inline_prompt, []
            return render(index), list(index.tools)
        # "always" / "never": the probe is irrelevant, so don't serialize it.
        index = self._resolve_disclosure_index(evaluation_case, "")
        if index is None:
            return render(None), []
        return render(index), list(index.tools)

    def _resolve_disclosure_index(
        self, evaluation_case: EvaluationData[InputT, OutputT], inline_probe: str
    ) -> TraceIndex | None:
        """Decide whether to disclose, returning a `TraceIndex` to use or None to inline.

        `inline_probe` is the prompt (or its trajectory-bearing part) that would be
        inlined; under ``"auto"`` it is preflighted against the judge model's window.
        Judges whose prompt is assembled in pieces (e.g. one row per decision) call
        this once per case and reuse the returned index across every piece.

        The probe covers only the rendered user prompt, not the judge's system
        prompt or (on the disclosure path) the tool schemas, so it slightly
        under-counts the true request size. `PREFLIGHT_SAFETY_MARGIN` (< 1.0)
        absorbs that; a residual underestimate near the boundary just inlines and
        lets the judge raise its own context-length error rather than mis-scoring.
        """
        if self.disclosure == "never":
            return None
        session = self._session_of(evaluation_case)
        if session is None:
            return None
        if self.disclosure == "auto" and not would_exceed_context(inline_probe, self._judge_window_tokens()):
            return None
        return TraceIndex(session)

    def _disclosed_trace_section(self, trace_index: TraceIndex) -> str:
        """The trace-overview block that substitutes for an inlined trajectory.

        Wraps `TraceIndex.for_judge()`'s paged overview with an instruction telling
        the judge to read spans through the tools and verify claims before scoring.
        """
        overview_section, _ = trace_index.for_judge()
        return (
            "The full trajectory is too large to inline. Read it through the trace tools "
            "(list_spans, get_span, search_spans) and verify every claim against the spans "
            f"before scoring.\n{overview_section}"
        )

    @staticmethod
    def _default_aggregator(outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        # Handle empty outputs list to avoid division by zero
        if not outputs:
            return (0.0, False, "No evaluation outputs produced")

        avg_score = sum(o.score for o in outputs) / len(outputs)
        all_pass = all(o.test_pass for o in outputs)
        combined_reason = " | ".join(o.reason for o in outputs if o.reason)
        return avg_score, all_pass, combined_reason

    @staticmethod
    def _aggregate_dropping_na(outputs: list[EvaluationOutput]) -> tuple[float, bool, str]:
        """Average only the rows that carry a verdict.

        For evaluators that emit one row per decision and can find some of those decisions
        unjudgeable, the not-applicable rows score 0.0 as a placeholder. Averaging that in would
        report a case with one perfectly judged decision and one unjudgeable one as half right.
        Set `self.aggregator` to this in `__init__` to opt in.
        """
        scored = [o for o in outputs if not o.not_applicable]
        if not scored:
            reason = " | ".join(o.reason for o in outputs if o.reason) or "not applicable"
            # Carry the rows' own verdicts: "nothing to judge" passes, but absent data fails.
            all_pass = all(o.test_pass for o in outputs) if outputs else True
            return (0.0, all_pass, reason)
        avg = sum(o.score for o in scored) / len(scored)
        all_pass = all(o.test_pass for o in scored)
        reason = " | ".join(o.reason for o in scored if o.reason)
        return avg, all_pass, reason

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

        Delegates to evaluate() via asyncio.to_thread by default, ensuring subclasses
        that only implement evaluate() work in the async path.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.
        """
        return await asyncio.to_thread(self.evaluate, evaluation_case)

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

    def _get_last_turn(self, evaluation_case: EvaluationData[InputT, OutputT]) -> TraceLevelInput:
        """Extract the most recent turn from the conversation for evaluation."""
        parsed_inputs = self._parse_trajectory(evaluation_case)
        if not parsed_inputs:
            raise ValueError(
                "No turn-level inputs could be parsed from the trajectory. "
                "Ensure actual_trajectory is a Session with at least one AgentInvocationSpan."
            )
        return parsed_inputs[-1]

    def _extract_user_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Extract user prompt from last message in session history.

        Args:
            parsed_input: Trace-level input containing session history

        Returns:
            User prompt text, or empty string if not available
        """
        if not parsed_input.session_history:
            return ""

        last_msg = parsed_input.session_history[-1]
        if not isinstance(last_msg, list) and self._has_text_content(last_msg):
            first_content = last_msg.content[0]
            if isinstance(first_content, TextContent):
                return first_content.text

        return ""

    def _format_tools(self, tools: list[ToolConfig]) -> str:
        """Format available tools for prompt display, including parameter schemas."""
        tool_lines = []
        for tool in tools:
            desc = tool.description or "No description"
            if tool.parameters:
                params = tool.parameters
                properties = params.get("properties", {})
                required = params.get("required", [])
                param_details = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else ""
                    param_details.append(f"    - {param_name} ({param_type}{req_marker}): {param_desc}")
                if param_details:
                    params_str = "\n".join(param_details)
                    tool_lines.append(f"- {tool.name}: {desc}\n  Parameters:\n{params_str}")
                else:
                    tool_lines.append(f"- {tool.name}: {desc}")
            else:
                tool_lines.append(f"- {tool.name}: {desc}")
        return "\n".join(tool_lines)

    def _format_session_history(self, contexts: list[Context], trace_index: TraceIndex | None = None) -> str:
        """Format session history with tool executions for prompt display.

        When `trace_index` is provided the history is too large to inline, so the
        paged trace-overview block is returned in its place and the judge reads the
        spans through the trace tools instead.
        """
        if trace_index is not None:
            return self._disclosed_trace_section(trace_index)
        lines = []
        for ctx in contexts:
            lines.append(f"User: {ctx.user_prompt.text}")
            if ctx.tool_execution_history:
                for tool_exec in ctx.tool_execution_history:
                    lines.append(f"Action: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                    lines.append(f"Tool: {tool_exec.tool_result.content}")
            lines.append(f"Assistant: {ctx.agent_response.text}")
        return "\n".join(lines)

    def _tool_level_disclosure(
        self, evaluation_case: EvaluationData[InputT, OutputT], tool_inputs: list[ToolLevelInput]
    ) -> tuple[TraceIndex | None, list[Any]]:
        """Resolve disclosure once for a tool-level case, shared across every tool call.

        Every tool call in a case is judged against the same session history, so the
        overflow decision and the `TraceIndex` are made once here — using the first
        tool call's rendered prompt as the ``"auto"`` size probe — and reused across
        the loop, instead of rebuilding a `TraceIndex` (re-flatten + re-sort every
        span) on each iteration. Returns ``(index, tools)`` to thread into
        `_format_tool_level_prompt` and the judge `Agent`.
        """
        if not tool_inputs:
            return None, []
        probe = self._format_tool_level_prompt(tool_inputs[0]) if self.disclosure == "auto" else ""
        index = self._resolve_disclosure_index(evaluation_case, probe)
        return index, (list(index.tools) if index is not None else [])

    def _format_tool_level_prompt(self, tool_input: ToolLevelInput, trace_index: TraceIndex | None = None) -> str:
        """Format evaluation prompt from tool-level input.

        When `trace_index` is provided the conversation history would overflow the
        judge, so the paged trace-overview block replaces the inlined history; the
        available-tools list and the target tool call are always kept inline.
        """
        parts = []

        # Format available tools
        if tool_input.available_tools:
            parts.append(f"## Available tool-calls\n{self._format_tools(tool_input.available_tools)}")
        else:
            logger.debug(
                "span_id=<%s> | no available tools resolved for tool-level evaluation",
                tool_input.span_info.span_id,
            )
            parts.append(
                "## Available tool-calls\n"
                "No tool list could be resolved for this agent. "
                "Evaluate the tool call based on the user's request and conversation context."
            )

        # Format previous conversation history
        if trace_index is not None:
            parts.append(f"## Previous conversation history\n{self._disclosed_trace_section(trace_index)}")
        elif tool_input.session_history:
            history_lines = []
            for msg in tool_input.session_history:
                if isinstance(msg, list):
                    # Handle tool execution lists
                    for tool_exec in msg:
                        history_lines.append(f"Tool call: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                        history_lines.append(f"Tool result: {tool_exec.tool_result.content}")
                else:
                    text = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
                    history_lines.append(f"{msg.role.value.capitalize()}: {text}")
            history_str = "\n".join(history_lines)
            parts.append(f"## Previous conversation history\n{history_str}")

        # Format target tool call to evaluate
        tool_details = tool_input.tool_execution_details
        tool_call_str = f"Tool call: {tool_details.tool_call.name}({tool_details.tool_call.arguments})"
        parts.append(f"## Target tool-call to evaluate\n{tool_call_str}")

        return "\n\n".join(parts)

    def _format_trace_level_prompt(self, parsed_input: TraceLevelInput, trace_index: TraceIndex | None = None) -> str:
        """Format evaluation prompt from parsed turn data.

        When `trace_index` is provided the conversation history would overflow the
        judge, so the paged trace-overview block replaces the inlined history; the
        assistant's response being judged is always kept inline.
        """
        parts = []

        if trace_index is not None:
            parts.append(f"# Conversation History:\n{self._disclosed_trace_section(trace_index)}")
        elif parsed_input.session_history:
            history_lines = []
            for msg in parsed_input.session_history:
                if isinstance(msg, list):
                    # Handle tool execution lists
                    for tool_exec in msg:
                        history_lines.append(f"Tool call: {tool_exec.tool_call.name}({tool_exec.tool_call.arguments})")
                        history_lines.append(f"Tool result: {tool_exec.tool_result.content}")
                else:
                    text = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
                    history_lines.append(f"{msg.role.value.capitalize()}: {text}")
            history_str = "\n".join(history_lines)
            parts.append(f"# Conversation History:\n{history_str}")

        parts.append(f"# Assistant's Response:\n{parsed_input.agent_response.text}")

        return "\n\n".join(parts)

    def _has_text_content(self, msg: UserMessage | AssistantMessage) -> TypeGuard[UserMessage | AssistantMessage]:
        """Check if a message object has accessible text content.

        Args:
            msg: Message object to check (UserMessage or AssistantMessage)

        Returns:
            True if msg has content attribute with at least one TextContent block.
            Note: TextContent may not be at index 0 due to tool calls or other content types.
        """
        if not hasattr(msg, "content") or not msg.content:
            return False

        # Check if ANY content block is TextContent, not just the first
        return any(isinstance(content_block, TextContent) for content_block in msg.content)

    def _extract_text_content(self, msg: UserMessage | AssistantMessage) -> str:
        """Extract and concatenate text from all TextContent blocks in a message.

        Args:
            msg: Message object containing content blocks

        Returns:
            Concatenated text from all TextContent blocks, or empty string if none found.
            Multiple text blocks are joined with a space.
            Note: Iterates through all content blocks since TextContent may not be first.
        """
        if not hasattr(msg, "content") or not msg.content:
            return ""

        # Collect all TextContent blocks - there could be multiple
        text_blocks = []
        for content_block in msg.content:
            if isinstance(content_block, TextContent):
                text_blocks.append(content_block.text)

        # Join multiple text blocks with space
        return " ".join(text_blocks) if text_blocks else ""

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the name of the evaluator type.

        Returns:
            str: The name of the evaluator type.
        """
        return cls.__name__

    def get_name(self) -> str:
        """Get the instance-level evaluator name, falling back to the class name.

        Used for the per-row `evaluator` tag in `EvaluationReport` and the
        `gen_ai.evaluation.name` OTel attribute. `get_type_name()` is still
        used for class-keyed lookups such as `from_dict` registry resolution.

        Returns:
            str: The instance name if set, otherwise the class name.
        """
        return self.name or self.get_type_name()

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
            if not k.startswith("_") and k not in exclude_attrs:
                # Handle model attribute specially
                if k == "model":
                    if isinstance(v, Model):
                        # Serialize Model instance to model_id
                        _dict["model_id"] = self._get_model_id(v)
                    elif v is None:
                        # model=None means "resolve the default at runtime". Omit it (like any
                        # other default-valued field) so reload restores None rather than pinning
                        # the judge to whatever DEFAULT_BEDROCK_MODEL_ID happens to be.
                        pass
                    else:
                        # Explicit string model ID, include as-is
                        _dict[k] = v
                elif k not in defaults or v != defaults[k]:
                    _dict[k] = v
        return _dict
