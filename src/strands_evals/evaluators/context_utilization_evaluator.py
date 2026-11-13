import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models.model import Model
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException
from tenacity import AsyncRetrying, Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    InferenceSpan,
    Role,
    Session,
    TextContent,
    ToolCall,
    ToolExecutionSpan,
    Trace,
    _find_root_agent_span,
    _to_aware_utc,
)
from ..utils import is_throttling_error
from .evaluator import Evaluator
from .prompt_templates.context_utilization import get_template

logger = logging.getLogger(__name__)

# Inline images (e.g. a `view_image` result) reach the trace as base64 data URLs. The judge cannot
# read them as text, and their characters would dominate both the prompt and the token counts.
_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+")
_IMAGE_PLACEHOLDER = "[image]"
# Chunks are runs of whole lines up to a token budget. Every token count here (chunk sizes, the
# reported counts, the budgets below) is characters / 4, so results do not depend on which tokenizer
# is installed and a 300-token chunk reports as about 300.
_CHUNK_TOKENS = 300
_CHARS_PER_TOKEN = 4
# Default for `max_judge_input_tokens`: a turn estimated above it is not sent to the judge (the
# default judge has a 1M-token context).
_MAX_JUDGE_INPUT_TOKENS = 900_000
# A throttled judge call is retried on its own, so a long session does not lose the turns already
# judged (Experiment retries the whole evaluation). Same attempts and bounds as Experiment's retries,
# with full jitter: `evaluate_async` judges several turns at once, and without it turns throttled
# together would all retry at the same moments and keep the judge throttled.
_JUDGE_RETRY_ATTEMPTS = 6
_JUDGE_RETRY_WAIT = wait_random_exponential(multiplier=4, max=240)
# Turns judged at once by `evaluate_async` (`evaluate` judges them one after another).
_MAX_CONCURRENT_TURNS = 8
# Each turn is judged on its own; the requests and answers after it are shown, clipped, so content
# gathered now and used later still counts.
_LATER_TURNS_TOKENS = 6000
_LATER_TURN_CHARS = 600
_COUNTS = ("gathered_tokens", "repeat_tokens", "useful_tokens")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN) if text else 0


class _JudgeThrottledError(RuntimeError):
    """A judge call still throttled after the evaluator's own retries.

    Raised from the throttling error instead of re-raising it: `Experiment` retries an evaluation
    whose error is a throttle, and doing so would judge every turn of the session again, the turns
    that already passed included.
    """


def _input_too_long(error: BaseException) -> bool:
    """The judge's input did not fit its context window.

    Only Strands' `ContextWindowOverflowException` (its model providers raise it for the providers'
    own "input is too long" errors). Matching error text instead would also catch a misconfigured
    judge (e.g. "max_tokens must be <= 8192") or a timeout that "took too long", and report it as a
    too-long turn instead of failing. An `EventLoopException` is judged by the error it wraps.
    """
    if isinstance(error, EventLoopException):
        return _input_too_long(error.original_exception)
    return isinstance(error, ContextWindowOverflowException)


def _clip(text: str, limit: int = _LATER_TURN_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


class ContextUtilizationRating(BaseModel):
    """Structured output for context utilization evaluation."""

    reasoning: str = Field(description="Brief reasoning about which gathered information the agent used")
    useful_ids: list[str] = Field(description="Ids of the useful pieces, e.g. 'S3.2', or 'S3.*' for all of S3")


@dataclass(frozen=True)
class _Piece:
    """One tagged chunk of gathered tool output."""

    uid: str
    source: str
    text: str
    tokens: int
    call: ToolCall


@dataclass
class _TurnDraft:
    """One trace as it is read: its request and answer, the rendered run, and the pieces it gathered."""

    request: str = ""
    answer: str = ""
    run: list[str] = field(default_factory=list)
    pieces: list[_Piece] = field(default_factory=list)


@dataclass
class _RenderState:
    """Running totals while a session is rendered: the last source number used (so chunk ids stay
    unique between turns) and the assistant replies shown."""

    sources: int = 0
    assistant_replies: int = 0


@dataclass(frozen=True)
class _Turn:
    """One judge call: the turn's position in the session, its prompt, and the pieces it may mark."""

    number: int
    prompt: str
    pieces: list[_Piece]


class ContextUtilizationEvaluator(Evaluator[InputT, OutputT]):
    """Measures how much of the information an agent gathered it actually used.

    The output of each measured tool call (every successful call by default, or those of the tools
    named in `measured_tools`) is split into chunks, each tagged with an id: runs of whole lines of
    about `chunk_size_tokens` tokens (300 by default). A failed call (`error` set) is shown to the judge
    but never measured. Each turn of the session with measured output (each `Trace`: one user
    request and the agent's work on it) is judged in its own call, which sees that turn's request,
    run and answer plus the later requests and answers, clipped, and returns the ids of the chunks
    the agent used.

    The score is `useful tokens / gathered tokens` over every measured call: a useful chunk counts
    all its tokens. A chunk identical to an earlier one (ignoring whitespace) stays in the
    denominator, since the agent paid for every copy, but counts once in the numerator. Overlapping
    but different chunks, such as two reads of a file from different lines, are counted separately.
    There is no pass/fail threshold, since a sensible one depends on the tools measured: a scored row
    always has `test_pass=True`.

    `metadata` on a scored row:
        chunk_size_tokens: The rough chunk size, in tokens, the row was measured with (the evaluator's
            setting, the same for every row it scores); compare only rows with the same value.
        gathered_tokens, repeat_tokens, useful_tokens: Totals over every measured call;
            `repeat_tokens` are the tokens of chunks identical (ignoring whitespace) to an earlier one.
        per_call: One row per measured call that returned output (a call with empty output has no
            chunks and no row, though it uses up a source id): `source` (its id, e.g. "S3"),
            `tool_call_id` (from the trace; a warning is logged when it is missing or repeated),
            `tool`, `chunks` and the three counts, so calls can be grouped afterwards by any rule.
        per_tool: The three counts of the `per_call` rows totalled per tool, plus `calls`,
            `utilization` (useful / gathered tokens) and `share_of_useful` (its useful tokens / all
            useful tokens).
        useful_ids: The chunk ids the judge marked, in session order.
        dropped_ids: Ids the judge returned that match no chunk of their turn (ignored, and logged);
            a low score with entries here may come from the judge's answer, not the agent.
        judge_calls: Judge calls made, one per turn with measured tool output.
        judge_input_tokens, judge_output_tokens: The judge's token usage over those calls, as its model
            reported it (0 when it reported none).
        assistant_replies: Distinct assistant replies in the session, in judged turns or not.

    Not-applicable rows (`label="not_applicable"`, score 0.0). One that passes (nothing to judge) is
    left out of `EvaluationReport.overall_score`; one that fails (the session could not be judged)
    counts there as 0, so a failure to judge is not hidden:
        - no `Session` trajectory (`test_pass=False`, counted);
        - nothing gathered by a measured tool (`test_pass=True`, left out);
        - a turn too long for one judge call (`test_pass=False`, counted; `metadata["too_long_turn"]`): its
          estimated size is over `max_judge_input_tokens` (checked before any judge call, so it costs
          nothing), or the judge raised `ContextWindowOverflowException`; the other turns alone would
          count only part of what the agent gathered. `metadata` also has `judge_calls` (calls
          started, the failing one and, in `evaluate_async`, cancelled ones included; 0 when the
          estimate caught it) and the token usage of the calls that finished. Any other judge error
          is raised.

    Before comparing scores: they depend on chunk size and on how complete the trace is (exporters
    that truncate tool output show the judge less), so compare runs evaluated the same way. With no
    assistant text between tool calls (`assistant_replies == 0`) the score is a lower bound. Sub-agent
    work counts with the main agent's, and token counts are estimates (characters / 4).
    """

    evaluation_level = EvaluationLevel.SESSION_LEVEL

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
        *,
        measured_tools: Iterable[str] | None = None,
        chunk_size_tokens: int = _CHUNK_TOKENS,
        max_judge_input_tokens: int = _MAX_JUDGE_INPUT_TOKENS,
    ):
        """Initialize the evaluator.

        Args:
            version: Prompt template version.
            model: Judge model.
            system_prompt: Overrides the template's system prompt.
            name: Instance-level evaluator name.
            measured_tools: Names of the tools whose output is measured (e.g. search, read, memory
                lookup). Output of other tools is shown to the judge but not measured. None measures
                every tool, so write and edit confirmations count as gathered too. A failed call
                (`error` set) is never measured.
            chunk_size_tokens: Rough size of a chunk, in tokens (characters / 4, the same estimate as
                the reported counts): a chunk is consecutive lines of output up to about this many
                tokens, and a single longer line is split into near-equal parts. Smaller chunks measure
                more finely but mean more judge decisions.
            max_judge_input_tokens: Largest estimated judge input (system prompt plus one turn's
                prompt, characters / 4) sent in one call; a session with a larger turn gets a
                not-applicable row before any judge call. Set it below your judge model's context
                window, or lower to cap the cost of one call.
        """
        if chunk_size_tokens < 1:
            raise ValueError(f"chunk_size_tokens must be at least 1, got {chunk_size_tokens}")
        if max_judge_input_tokens < 1:
            raise ValueError(f"max_judge_input_tokens must be at least 1, got {max_judge_input_tokens}")
        if isinstance(measured_tools, str):  # a str is an Iterable[str] too: its characters would be the names
            raise TypeError(f"measured_tools takes a collection of tool names, e.g. [{measured_tools!r}]")
        super().__init__(name=name)
        # A sorted list rather than a set, so `to_dict` / `Experiment.to_file` can write it as JSON.
        self.measured_tools = sorted(set(measured_tools)) if measured_tools is not None else None
        self.chunk_size_tokens = chunk_size_tokens
        self.max_judge_input_tokens = max_judge_input_tokens
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        # A session that gathered nothing has nothing to measure; averaging its placeholder 0.0
        # in would report it as one that wasted everything it read.
        self.aggregator = self._aggregate_dropping_na

    # ---- Entry points -----------------------------------------------------------------------

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        prepared = self._prepare(evaluation_case)
        if isinstance(prepared, EvaluationOutput):
            return [prepared]
        turns, assistant_reply_count = prepared
        results = []
        for calls, turn in enumerate(turns, start=1):
            try:
                results.append(self._judge(turn))
            except Exception as e:
                if not _input_too_long(e):
                    raise
                # Stop here: the session gets no score, so the remaining turns need not be judged.
                return [self._too_long_row(turn, calls, results)]
        return [self._score(turns, results, assistant_reply_count)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        prepared = self._prepare(evaluation_case)
        if isinstance(prepared, EvaluationOutput):
            return [prepared]
        turns, assistant_reply_count = prepared
        limit = asyncio.Semaphore(_MAX_CONCURRENT_TURNS)
        calls = 0

        async def judge(turn: _Turn) -> AgentResult:
            nonlocal calls
            async with limit:
                calls += 1
                return await self._judge_async(turn)

        tasks = [asyncio.ensure_future(judge(turn)) for turn in turns]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            # One failure decides the session (no score, or an error), so stop the other judge calls.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            done = [task.result() for task in tasks if task.done() and not task.cancelled() and not task.exception()]
            # Of the turns that failed before the rest were cancelled, the earliest in the session
            # decides (close to `evaluate`, which stops at its first failure).
            failed, error = next(
                (turn, task.exception())
                for turn, task in zip(turns, tasks, strict=True)
                if not task.cancelled() and task.exception() is not None
            )
            if error is not None and _input_too_long(error):
                return [self._too_long_row(failed, calls, done)]
            # Keep the error's own cause (e.g. the throttle behind `_JudgeThrottledError`).
            failure = cast(BaseException, error)
            raise failure from failure.__cause__
        return [self._score(turns, list(results), assistant_reply_count)]

    def _prepare(self, evaluation_case: EvaluationData[InputT, OutputT]) -> tuple[list[_Turn], int] | EvaluationOutput:
        """The per-turn judge calls and the assistant reply count, or the not-applicable row to return."""
        session = evaluation_case.actual_trajectory
        if not isinstance(session, Session):
            return self._not_applicable_row("no Session trajectory provided", test_pass=False)
        turns, assistant_reply_count = self._render(evaluation_case, session)
        if not turns:
            return self._not_applicable_row(self._nothing_measured_reason(session), test_pass=True)
        # Scoring does not need `tool_call_id` (chunks are keyed by source id), but `per_call` rows are
        # matched back to spans by it; say so when it cannot be, e.g. in a hand-built session.
        ids = list({piece.source: piece.call.tool_call_id for turn in turns for piece in turn.pieces}.values())
        if not all(ids) or len(set(ids)) < len(ids):
            logger.warning(
                "session_id=<%s> | measured tool calls without a unique tool_call_id | "
                "per_call rows cannot be matched to spans by tool_call_id",
                session.session_id,
            )
        # Checked before any judge call, so a session that cannot be scored costs nothing. A judge
        # that still reports its input too long (an estimate miss, or a smaller model) is handled
        # where it is called.
        too_long = next(
            (
                turn
                for turn in turns
                if _estimate_tokens(self.system_prompt + turn.prompt) > self.max_judge_input_tokens
            ),
            None,
        )
        if too_long is not None:
            return self._too_long_row(too_long, judge_calls=0, results=[])
        if not assistant_reply_count:
            logger.warning(
                "session_id=<%s> | no assistant text between tool calls | score is a lower bound",
                session.session_id,
            )
        return turns, assistant_reply_count

    def _nothing_measured_reason(self, session: Session) -> str:
        """Why a session has no measured output, naming the tools when `measured_tools` matched none called."""
        called = sorted({s.tool_call.name for t in session.traces for s in t.spans if isinstance(s, ToolExecutionSpan)})
        if self.measured_tools is None or set(self.measured_tools) & set(called):
            return "no measured tool output"
        logger.warning(
            "session_id=<%s>, measured_tools=<%s>, tools_called=<%s> | none of the measured tools was called",
            session.session_id,
            self.measured_tools,
            called,
        )
        return f"no measured tool output: none of measured_tools {self.measured_tools} was called (called: {called})"

    def _judge(self, turn: _Turn) -> AgentResult:
        """Judge one turn, retrying this call alone when the judge is throttled.

        Raises `_JudgeThrottledError` once the retries run out (see its docstring).
        """
        try:
            for attempt in Retrying(**self._retry_policy()):
                with attempt:
                    return self._new_judge()(turn.prompt, structured_output_model=ContextUtilizationRating)
        except Exception as e:
            if is_throttling_error(e):
                raise self._still_throttled(turn) from e
            raise
        raise AssertionError("unreachable: Retrying re-raises the last error")

    async def _judge_async(self, turn: _Turn) -> AgentResult:
        """`_judge` for `evaluate_async`: one turn, retried alone when the judge is throttled."""
        try:
            async for attempt in AsyncRetrying(**self._retry_policy()):
                with attempt:
                    return await self._new_judge().invoke_async(
                        turn.prompt, structured_output_model=ContextUtilizationRating
                    )
        except Exception as e:
            if is_throttling_error(e):
                raise self._still_throttled(turn) from e
            raise
        raise AssertionError("unreachable: AsyncRetrying re-raises the last error")

    @staticmethod
    def _still_throttled(turn: _Turn) -> _JudgeThrottledError:
        return _JudgeThrottledError(
            f"turn {turn.number}: the judge was still throttled after {_JUDGE_RETRY_ATTEMPTS} attempts"
        )

    @staticmethod
    def _retry_policy() -> dict[str, Any]:
        return {
            "retry": retry_if_exception(is_throttling_error),
            "stop": stop_after_attempt(_JUDGE_RETRY_ATTEMPTS),
            "wait": _JUDGE_RETRY_WAIT,
            "reraise": True,
        }

    def _new_judge(self) -> Agent:
        """A fresh judge per call (each turn and each retry), so no prompt or verdict carries over."""
        return Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)

    @staticmethod
    def _not_applicable_row(reason: str, test_pass: bool) -> EvaluationOutput:
        return EvaluationOutput(score=0.0, test_pass=test_pass, reason=reason, label=NOT_APPLICABLE)

    @classmethod
    def _too_long_row(cls, turn: _Turn, judge_calls: int, results: list[AgentResult]) -> EvaluationOutput:
        """No verdict: the turn does not fit in a single judge call, so the session is not scored."""
        row = cls._not_applicable_row(f"turn {turn.number} is too long for one judge call", False)
        row.metadata = {"too_long_turn": turn.number, "judge_calls": judge_calls, **cls._judge_usage(results)}
        return row

    @staticmethod
    def _judge_usage(results: list[AgentResult]) -> dict[str, int]:
        """The judge's input and output tokens over its calls, from each result's metrics."""
        totals = {"judge_input_tokens": 0, "judge_output_tokens": 0}
        for result in results:
            usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None)
            if isinstance(usage, dict):
                totals["judge_input_tokens"] += int(usage.get("inputTokens", 0))
                totals["judge_output_tokens"] += int(usage.get("outputTokens", 0))
        return totals

    # ---- Rendering: Session -> one judge prompt per turn -------------------------------------

    def _render(self, evaluation_case: EvaluationData[InputT, OutputT], session: Session) -> tuple[list[_Turn], int]:
        """One judge call per turn with measured tool output, and the number of assistant replies shown.

        Chunk ids are numbered across the whole session, so they stay unique between turns. The
        requests and answers come from the trace; the case's input and output only fill in a missing
        first request or last answer, since they need not be the prompt (e.g. a dict or a row id).
        """
        state = _RenderState()
        drafts = [self._draft_turn(trace, state) for trace in session.traces]
        if drafts:
            if not drafts[0].request and evaluation_case.input is not None:
                drafts[0].request = str(evaluation_case.input)
            if not drafts[-1].answer and evaluation_case.actual_output is not None:
                drafts[-1].answer = str(evaluation_case.actual_output)
        missing = [
            n for n, draft in enumerate(drafts, start=1) if draft.pieces and not (draft.request and draft.answer)
        ]
        if missing:
            logger.warning(
                "session_id=<%s>, turns=<%s> | no request or answer for these turns in the trace or case | "
                "judged from the run alone",
                session.session_id,
                missing,
            )
        clipped = [(_clip(draft.request), _clip(draft.answer)) for draft in drafts]  # once, not once per turn
        turns = [
            _Turn(number=number, prompt=self._turn_prompt(draft, clipped[number:]), pieces=draft.pieces)
            for number, draft in enumerate(drafts, start=1)
            if draft.pieces
        ]
        return turns, state.assistant_replies

    def _draft_turn(self, trace: Trace, state: _RenderState) -> _TurnDraft:
        """Read one trace in time order into its request, answer, rendered run and chunks.

        Updates `state`: one source number per measured call, one count per assistant reply shown.
        """
        draft = _TurnDraft()
        # Consecutive inference spans can end in the same assistant message: skip a reply identical to
        # the last one shown.
        last_assistant_text = ""
        agents = [span for span in trace.spans if isinstance(span, AgentInvocationSpan)]
        # Root agents have no parent. When a source keeps a parent outside the trace on every agent
        # (e.g. a server span), the roots are the agents sharing the earliest agent's parent.
        roots = [a for a in agents if a.span_info.parent_span_id is None]
        if not roots and (earliest := _find_root_agent_span(agents)):
            roots = [a for a in agents if a.span_info.parent_span_id == earliest.span_info.parent_span_id]
        seen_root = False
        for span in sorted(trace.spans, key=lambda s: _to_aware_utc(s.span_info.start_time)):
            if isinstance(span, AgentInvocationSpan):
                text = span.user_prompt.strip()
                if not any(span is root for root in roots):
                    if text:
                        draft.run.append(f"[sub-agent task] {text}")
                elif not seen_root:
                    seen_root = True
                    draft.request, draft.answer = text, span.agent_response
                else:  # a later root agent in the same trace: show the answer so far, then the new request
                    if draft.answer:
                        draft.run.append(f"[answer] {draft.answer}")
                    if text:
                        draft.run.append(f"[user] {text}")
                    draft.answer = span.agent_response or draft.answer
            elif isinstance(span, InferenceSpan):
                text = self._assistant_text(span)
                if text and text != last_assistant_text:
                    state.assistant_replies += 1
                    draft.run.append(f"[assistant] {text}")
                    last_assistant_text = text
            elif isinstance(span, ToolExecutionSpan):
                sid = None
                if self._measured(span):
                    state.sources += 1
                    sid = f"S{state.sources}"
                lines, pieces = self._render_call(span, sid)
                draft.run.extend(lines)
                draft.pieces.extend(pieces)
        return draft

    def _measured(self, span: ToolExecutionSpan) -> bool:
        """Whether a call's output is gathered: the call did not fail and its tool is measured.

        A call with `error` set failed. Mappers put a status there (e.g. "error") and the error text
        in `content`, so content alone cannot tell a failed call from a successful one.
        """
        failed = bool(span.tool_result.error)
        return not failed and (self.measured_tools is None or span.tool_call.name in self.measured_tools)

    def _render_call(self, span: ToolExecutionSpan, sid: str | None) -> tuple[list[str], list[_Piece]]:
        """One tool call's run lines, and its chunks tagged under `sid` when the call is measured."""
        call = f"{span.tool_call.name}({json.dumps(span.tool_call.arguments, default=str)})"
        output = _DATA_URL.sub(_IMAGE_PLACEHOLDER, span.tool_result.content)
        if span.tool_result.error:
            # A failed call is shown, since the agent saw it, but its output is not gathered context.
            failed = f"[call, failed] {call}\n  error: {span.tool_result.error}"
            return [failed + (f"\n  output: {output}" if output else "")], []
        lines: list[str] = []
        pieces: list[_Piece] = []
        if sid is None:
            lines.append(f"[call, not measured] {call}\n  output: {output}")
        else:
            lines.append(f"[call {sid}] {call}")
            for n, text in enumerate(self._split(output), start=1):
                piece = _Piece(f"{sid}.{n}", sid, text, _estimate_tokens(text), span.tool_call)
                pieces.append(piece)
                lines.append(f"  <{piece.uid}>\n" + "\n".join(f"    {line}" for line in text.splitlines()))
        return lines, pieces

    def _split(self, output: str) -> list[str]:
        """The chunks of one call's output, empty ones dropped."""
        return [chunk.strip("\n") for chunk in self._windows(output) if chunk.strip()]

    def _windows(self, output: str) -> list[str]:
        """Consecutive lines of up to about `chunk_size_tokens` tokens; a longer single line is split by characters."""
        chunks: list[str] = []
        current: list[str] = []
        size = 0
        for line in output.splitlines():
            cost = _estimate_tokens(line + "\n")
            if cost > self.chunk_size_tokens:  # e.g. minified JSON or a long log line: chunks of its own
                if current:
                    chunks.append("\n".join(current))
                    current, size = [], 0
                # Near-equal parts, so no tiny tail becomes a chunk of its own.
                step = math.ceil(len(line) / math.ceil(len(line) / (self.chunk_size_tokens * _CHARS_PER_TOKEN)))
                chunks.extend(line[k : k + step] for k in range(0, len(line), step))
                continue
            if current and size + cost > self.chunk_size_tokens:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += cost
        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _assistant_text(span: InferenceSpan) -> str:
        """Text of the model's reply in an inference span (its last message, when the assistant's).

        Checks `role`, not the class: the message union has no discriminator, so a text-only
        `AssistantMessage` comes back as a `UserMessage` with `role=assistant` after a JSON round-trip.
        """
        if not span.messages or span.messages[-1].role != Role.ASSISTANT:
            return ""
        return "\n".join(c.text for c in span.messages[-1].content if isinstance(c, TextContent)).strip()

    def _turn_prompt(self, draft: _TurnDraft, later: list[tuple[str, str]]) -> str:
        """The judge prompt for one turn: request, run, answer, then the later turns if any."""
        prompt = f"<task>\n{draft.request}\n</task>\n\n<run>\n" + "\n".join(draft.run) + "\n</run>\n\n"
        prompt += f"<final_response>\n{draft.answer}\n</final_response>\n\n"
        later_text = self._later_turns(later)
        if later_text:
            prompt += f"<later_turns>\n{later_text}\n</later_turns>\n\n"
        return prompt + "List the ids of the useful pieces."

    @staticmethod
    def _later_turns(later: list[tuple[str, str]]) -> str:
        """The (clipped) requests and answers after a turn, nearest first, up to a token budget."""
        entries: list[str] = []
        budget = _LATER_TURNS_TOKENS
        for k, (request, answer) in enumerate(later, start=1):
            entry = f"[later request {k}] {request}\n[answer {k}] {answer}"
            cost = _estimate_tokens(entry)
            if cost > budget:
                break
            entries.append(entry)
            budget -= cost
        return "\n".join(entries)

    # ---- Scoring: judge verdicts -> EvaluationOutput -----------------------------------------

    def _score(self, turns: list[_Turn], results: list[AgentResult], assistant_reply_count: int) -> EvaluationOutput:
        """Score every gathered chunk against the judge's verdict on its turn."""
        ratings = [cast(ContextUtilizationRating, result.structured_output) for result in results]
        useful: set[str] = set()
        dropped: list[str] = []
        for turn, rating in zip(turns, ratings, strict=True):
            turn_useful, turn_dropped = self._useful_uids(rating.useful_ids, turn.pieces)
            useful |= turn_useful
            dropped += turn_dropped
        per_call = self._per_call([piece for turn in turns for piece in turn.pieces], useful)
        per_tool = self._per_tool(per_call)
        gathered, repeated, used = (sum(row[name] for row in per_call) for name in _COUNTS)
        for row in per_tool.values():
            row["share_of_useful"] = row["useful_tokens"] / used if used else 0.0
        score = used / gathered if gathered else 0.0
        if len(turns) == 1:
            reason = ratings[0].reasoning
        else:
            reason = "\n\n".join(
                f"Turn {turn.number}: {rating.reasoning}" for turn, rating in zip(turns, ratings, strict=True)
            )
        metadata: dict[str, Any] = {
            "chunk_size_tokens": self.chunk_size_tokens,
            "gathered_tokens": gathered,
            "repeat_tokens": repeated,
            "useful_tokens": used,
            "per_tool": per_tool,
            "per_call": per_call,
            "useful_ids": sorted(useful, key=self._uid_key),
            "dropped_ids": dropped,
            "judge_calls": len(turns),
            **self._judge_usage(results),
            "assistant_replies": assistant_reply_count,
        }
        return EvaluationOutput(score=score, test_pass=True, reason=reason, metadata=metadata)

    @staticmethod
    def _useful_uids(ids: list[str], pieces: list[_Piece]) -> tuple[set[str], list[str]]:
        """The turn's chunk ids the judge marked, and the ids (after splitting) that match none.

        Accepts `S3.2`, and `S3.*` / `S3` for every chunk of source 3. Angle brackets are stripped, since
        the run shows chunks tagged as `<S3.2>`, and a string holding several ids separated by commas
        or whitespace (`"S3.1, S3.2"`) is read as each of them.
        """
        known = {piece.uid for piece in pieces}
        by_source: dict[str, list[str]] = {}
        for piece in pieces:
            by_source.setdefault(piece.source, []).append(piece.uid)
        result: set[str] = set()
        dropped: list[str] = []
        for raw in (part for entry in ids for part in re.split(r"[,\s]+", entry) if part):
            item = raw.strip("<>").removesuffix(".*")
            if item in known:
                result.add(item)
            elif item in by_source:
                result.update(by_source[item])  # a whole source, e.g. `S3` or `S3.*`
            else:
                dropped.append(raw)
        if dropped:
            logger.warning("dropped_ids=<%s> | judge returned ids that match no piece", dropped)
        return result, dropped

    @staticmethod
    def _per_call(pieces: list[_Piece], useful: set[str]) -> list[dict[str, Any]]:
        """Gathered, repeated and useful tokens per measured call, from one pass over the chunks.

        Content used through several copies counts once, credited to its first useful copy, so these
        rows add up to the session's totals.
        """
        per_call: dict[str, dict[str, Any]] = {}
        seen: set[bytes] = set()  # digests of whitespace-normalized chunk text, not a second copy of it
        counted: set[bytes] = set()
        for piece in pieces:
            if piece.source not in per_call:
                per_call[piece.source] = {
                    "source": piece.source,
                    "tool_call_id": piece.call.tool_call_id,
                    "tool": piece.call.name,
                    "chunks": 0,
                    **dict.fromkeys(_COUNTS, 0),
                }
            row = per_call[piece.source]
            key = hashlib.blake2b(" ".join(piece.text.split()).encode(), digest_size=16).digest()
            row["chunks"] += 1
            row["gathered_tokens"] += piece.tokens
            if key in seen:
                row["repeat_tokens"] += piece.tokens
            seen.add(key)
            if piece.uid in useful and key not in counted:
                row["useful_tokens"] += piece.tokens
                counted.add(key)
        return list(per_call.values())

    @staticmethod
    def _per_tool(per_call: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """The three counts of the `per_call` rows totalled per tool, with `calls` and `utilization`.

        `_score` adds `share_of_useful`.
        """
        per_tool: dict[str, dict[str, Any]] = {}
        for row in per_call:
            tool = per_tool.setdefault(row["tool"], {"calls": 0, **dict.fromkeys(_COUNTS, 0)})
            tool["calls"] += 1
            for name in _COUNTS:
                tool[name] += row[name]
        for tool in per_tool.values():
            tool["utilization"] = tool["useful_tokens"] / tool["gathered_tokens"] if tool["gathered_tokens"] else 0.0
        return per_tool

    @staticmethod
    def _uid_key(uid: str) -> tuple[int, ...]:
        """Sort `S12.3` after `S2.10`: by source number, then chunk number."""
        return tuple(int(n) for n in uid[1:].split("."))
