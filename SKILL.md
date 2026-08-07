---
name: strands-evals
description: Use when authoring evaluations with strands-agents-evals. Activates on tasks involving Case/Experiment construction, picking evaluators (Output, Trajectory, Helpfulness, Faithfulness, Coherence, Conciseness, ResponseRelevance, Harmfulness, Refusal, Stereotyping, InstructionFollowing, GoalSuccessRate, ToolSelection/ParameterAccuracy, Multimodal*), trace-based evaluation with mappers (CloudWatch, OpenSearch, OpenInference, LangChain OTel, Strands in-memory), simulators (ActorSimulator, ToolSimulator), failure detection and root-cause analysis (detect_failures, analyze_root_cause, diagnose_session), chaos / fault-injection testing (ChaosCase, ChaosExperiment, ChaosPlugin), red-team evaluation (RedTeamExperiment, AdversarialCaseGenerator, AttackSuccessEvaluator, attack strategies), or auto test-case generation (ExperimentGenerator). Trigger phrases include "evaluate this agent", "score the trajectory", "simulate a user", "diagnose the session", "generate test cases", "LLM-as-a-Judge", "inject tool failures", "chaos test", "red team this agent", "jailbreak / adversarial". Skip for general LLM eval theory unrelated to this package.
version: 0.2.0
---

# strands-evals — Authoring Evaluations

Patterns for using the `strands-agents-evals` package to evaluate AI agents and LLM applications. For repo conventions, contribution rules, prompt versioning, and review checklists, see `AGENTS.md` at the repo root. This skill stays focused on authoring.

## Routing Map

| Goal | Section |
| --- | --- |
| Score a one-shot response | OutputEvaluator |
| Score tool/action sequences | TrajectoryEvaluator |
| Score a multi-turn trace | Trace-based evaluators |
| Multi-turn conversation testing | ActorSimulator |
| Replace real tools during eval | ToolSimulator |
| Diagnose a failing session | Detectors |
| Inject tool failures / response corruption | Chaos Testing |
| Red-team an agent for safety bypasses | Red Team |
| Auto-generate test cases | ExperimentGenerator |
| Image-to-text evaluation | Multimodal |
| Build a custom evaluator | Custom evaluator |
| Async / parallel runs | Async Execution |
| Cache task results across runs | Result Caching |

## Core Building Blocks

```python
from strands_evals import Case, Experiment

case = Case[str, str](
    name="capital-france",
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    metadata={"category": "knowledge"},
)
experiment = Experiment[str, str](cases=[case], evaluators=[...])
report = experiment.run_evaluations(task_function)
report.run_display()
```

`task_function(case: Case)` returns either a string output or, for trace-based evaluators, a dict like `{"output": ..., "trajectory": Session}`.

`run_evaluations()` always returns a single `EvaluationReport`. With one evaluator, the report is keyed to that evaluator. With multiple, results are flattened into one report and each row is tagged via `report.cases[i]["evaluator"]`.

Persist experiments:

```python
experiment.to_file("my_eval", "json")
loaded = Experiment.from_file("./experiment_files/my_eval.json", "json")
```

## OutputEvaluator (rubric-based LLM-as-judge)

```python
from strands_evals.evaluators import OutputEvaluator

OutputEvaluator(
    rubric="Score 1.0 for accurate+complete, 0.5 partial, 0.0 incorrect.",
    include_inputs=True,
    model="global.anthropic.claude-sonnet-4-6",  # default judge
)
```

Use when scoring free-form text against a rubric and traces aren't needed.

## TrajectoryEvaluator

Always extract trajectory; don't pass full `agent.messages` because of context overflow risk:

```python
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor

trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
evaluator.update_trajectory_description(
    tools_use_extractor.extract_tools_description(agent, is_short=True)
)
return {"output": str(response), "trajectory": trajectory}
```

Built-in scorers usable inside the rubric: `exact_match_scorer`, `in_order_match_scorer`, `any_order_match_scorer`. Pick:
- **exact** for strict pipelines.
- **in_order** when sequence matters but interleaving is allowed.
- **any_order** when only the set of tools matters.

## Trace-Based Evaluators (need a Session)

Capture spans with telemetry, then map them to a `Session`:

```python
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

def task_function(case):
    telemetry.in_memory_exporter.clear()
    agent = Agent(
        trace_attributes={"session.id": case.session_id, "gen_ai.conversation.id": case.session_id},
        callback_handler=None,
    )
    response = agent(case.input)
    spans = telemetry.in_memory_exporter.get_finished_spans()
    session = StrandsInMemorySessionMapper().map_to_session(spans, session_id=case.session_id)
    return {"output": str(response), "trajectory": session}
```

### Shortcut: `@eval_task` + `TracedHandler`

For the common Strands-in-memory case, the `@eval_task(TracedHandler())` decorator collects spans and maps them to a `Session` automatically. The decorated function can return an `Agent` (auto-invoked with `case.input`), a string, or a dict.

```python
from strands import Agent
from strands_evals import eval_task, TracedHandler

@eval_task(TracedHandler())
def task(case):
    return Agent(model="...", tools=[...])  # auto-invoked, telemetry captured

report = experiment.run_evaluations(task)
```

`TracedHandler` shares one in-memory exporter across calls — safe with `run_evaluations` (sequential) or `run_evaluations_async(max_workers=1)`. For concurrent runs (`max_workers > 1`), give each worker its own handler instance via a per-call factory or fall back to the manual capture pattern above. Pass a different `mapper=` to `TracedHandler` if you need a non-default `SessionMapper`.

Pick the evaluator by scope:

| Scope | Evaluators |
| --- | --- |
| Tool-level | `ToolSelectionAccuracyEvaluator`, `ToolParameterAccuracyEvaluator` |
| Trace-level (last turn) | `CorrectnessEvaluator`, `HelpfulnessEvaluator`, `FaithfulnessEvaluator`, `CoherenceEvaluator`, `ConcisenessEvaluator`, `ResponseRelevanceEvaluator`, `HarmfulnessEvaluator`, `RefusalEvaluator`, `StereotypingEvaluator`, `InstructionFollowingEvaluator` |
| Session-level (full conversation) | `GoalSuccessRateEvaluator` |
| Multi-agent interactions and handoffs | `InteractionsEvaluator` (output-based) |

Helpfulness uses a seven-level scale, 0.0 Not helpful to 1.0 Above and beyond. Correctness uses a three-level rubric in basic mode, or binary CORRECT/INCORRECT in reference mode when `expected_assertion` is set on the case. Conciseness uses three levels. Coherence uses five levels. Harmfulness, Refusal, Stereotyping, InstructionFollowing are binary.

For traces from external systems pick the matching mapper:
- `CloudWatchSessionMapper` paired with `CloudWatchProvider` and `CloudWatchLogsParser`
- `LangChainOtelSessionMapper`
- `OpenInferenceSessionMapper`
- `OpenSearchSessionMapper` paired with `OpenSearchProvider`
- `LangfuseProvider` for fetching from Langfuse

Trace types live in `strands_evals.types.trace`: `Session`, `Trace`, and `SpanUnion = InferenceSpan | ToolExecutionSpan | AgentInvocationSpan`.

## Multimodal (image-to-text)

```python
from strands_evals.types import ImageData, MultimodalInput
from strands_evals.evaluators import (
    MultimodalCorrectnessEvaluator,
    MultimodalFaithfulnessEvaluator,
    MultimodalInstructionFollowingEvaluator,
    MultimodalOverallQualityEvaluator,
    MultimodalOutputEvaluator,  # base for custom multimodal rubrics
)

case = Case[MultimodalInput, str](
    name="img-1",
    input=MultimodalInput(
        media=ImageData(source="path/to/image.png"),
        instruction="Describe the image in detail.",
    ),
)
```

`ImageData.source` accepts: file path, base64 string, data URL, HTTP URL, PIL Image, or raw bytes. Reference-based rubric is auto-selected when `expected_output` is set, otherwise reference-free.

## Deterministic Evaluators (no LLM)

Fast, free, exact:

```python
from strands_evals.evaluators.deterministic import (
    Contains, Equals, StartsWith,   # output checks
    ToolCalled,                     # trajectory check
    StateEquals,                    # environment-state check
)
```

Use these as cheap pre-filters before LLM judges or alongside them.

## ActorSimulator (multi-turn user simulation)

```python
from strands_evals import ActorSimulator  # UserSimulator is an alias

simulator = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=10)

while simulator.has_next():
    memory_exporter.clear()
    agent_response = agent(user_message)
    turn_spans = list(memory_exporter.get_finished_spans())
    all_spans.extend(turn_spans)
    user_message = str(simulator.act(str(agent_response)).structured_output.message)
```

Pair with `GoalSuccessRateEvaluator` to verify the simulated user achieved their objective. Combine with trace-level evaluators for per-turn quality.

## ToolSimulator (LLM-powered fake tools)

```python
from pydantic import BaseModel, Field
from strands_evals.simulation.tool_simulator import ToolSimulator

tool_simulator = ToolSimulator()

class HVACResponse(BaseModel):
    temperature: float = Field(..., description="Target temp F")
    mode: str
    status: str = "success"

@tool_simulator.tool(
    share_state_id="room_environment",
    initial_state_description="Room: 68F, humidity 45%, HVAC off",
    output_schema=HVACResponse,
)
def hvac_controller(temperature: float, mode: str) -> dict: ...

agent = Agent(tools=[tool_simulator.get_tool("hvac_controller")])
```

The decorated function body is never executed. The LLM produces schema-validated responses. Tools that share `share_state_id` see one consistent state. Use this for sensor and controller pairs so reads reflect prior writes.

## Detectors (failure detection + RCA)

In an experiment, attach `DiagnosisConfig`:

```python
from strands_evals import DiagnosisConfig
from strands_evals.types.detector import ConfidenceLevel, DiagnosisTrigger

Experiment(
    cases=cases,
    evaluators=[HelpfulnessEvaluator()],
    diagnosis_config=DiagnosisConfig(
        trigger=DiagnosisTrigger.ON_FAILURE,         # or ALWAYS
        confidence_threshold=ConfidenceLevel.MEDIUM, # LOW, MEDIUM, HIGH
    ),
)
```

Standalone on a `Session`:

```python
from strands_evals.detectors import detect_failures, analyze_root_cause, diagnose_session

# End-to-end
result = diagnose_session(session, confidence_threshold=ConfidenceLevel.MEDIUM)
for rc in result.root_causes:
    print(rc.fix_type, rc.fix_recommendation)

# Or step by step
failures = detect_failures(session, confidence_threshold=ConfidenceLevel.MEDIUM)
if failures.failures:
    rca = analyze_root_cause(session, failures=failures.failures)
# analyze_root_cause auto-runs detection if failures is omitted
rca = analyze_root_cause(session)
```

Display recommendations on the report:

```python
report.display(include_recommendations=True)
```

## Chaos Testing (deterministic fault injection)

Inject tool failures and response corruption to evaluate resilience. Effects fire via Strands' native hook system; the user's task body stays chaos-free.

Use `@eval_task(TracedHandler())` so the chaos-aware evaluators have a `Session` to score against — they call `_get_last_turn()` on `actual_trajectory` and will raise without one:

```python
from strands import Agent
from strands_evals import Case, eval_task, TracedHandler
from strands_evals.chaos import (
    ChaosCase, ChaosExperiment, ChaosPlugin,
    Timeout, NetworkError, ExecutionError, ValidationError,   # pre-hook (cancel call)
    TruncateFields, RemoveFields, CorruptValues,              # post-hook (corrupt response)
)

base = [Case(name="flight_search", input="Find flights to Tokyo")]
effect_maps = {
    "search_timeout":  {"tool_effects": {"search_tool":   [Timeout()]}},
    "db_truncate":     {"tool_effects": {"database_tool": [TruncateFields(max_length=20)]}},
}
chaos_cases = ChaosCase.expand(base, effect_maps, include_no_effect_baseline=True)

@eval_task(TracedHandler())
def task(case):
    return Agent(tools=[search_tool, database_tool], plugins=[ChaosPlugin()])

report = ChaosExperiment(cases=chaos_cases, evaluators=[...]).run_evaluations(task=task)
```

Effect categories:
- **Pre-hook (cancel before execution):** `Timeout`, `NetworkError`, `ExecutionError`, `ValidationError` — each takes an `error_message`. First pre-hook effect wins.
- **Post-hook (corrupt response):** `TruncateFields(max_length=...)`, `RemoveFields(remove_ratio=...)`, `CorruptValues(...)` — applied in order to the tool's dict response.

**One effect per tool per `ChaosCase`.** `ChaosCase` validates `len(effects_list) <= 1` per tool and raises `ValueError` otherwise. To test multiple effects on the same tool, use separate `ChaosCase` instances (`ChaosCase.expand` will produce them from distinct entries in `effect_maps`).

Pair with chaos-aware evaluators in `strands_evals.evaluators.chaos`:

```python
from strands_evals.evaluators.chaos import (
    FailureCommunicationEvaluator,   # did the agent tell the user?
    PartialCompletionEvaluator,      # did it deliver what it could?
    RecoveryStrategyEvaluator,       # did it retry / fall back well?
)
```

All three are trace-based — they need `actual_trajectory` to be a `Session`, which is why the task above uses `TracedHandler`.

### Combining ToolSimulator with Chaos

`ChaosPlugin` operates on Strands `@tool` calls regardless of whether the implementation is real or simulated. Wrap simulated tools in `ToolSimulator` and pass `ChaosPlugin()` alongside as usual:

```python
from strands_evals.simulation.tool_simulator import ToolSimulator

tool_simulator = ToolSimulator()

@tool_simulator.tool(output_schema=SearchResponse)
def search_tool(query: str) -> dict: ...

@eval_task(TracedHandler())
def task(case):
    return Agent(
        tools=[tool_simulator.get_tool("search_tool")],
        plugins=[ChaosPlugin()],
    )
```

The simulator generates the tool's response; `ChaosPlugin` then applies pre-hook cancellations or post-hook corruption to that response based on the active `ChaosCase`. Useful when you want chaos coverage without standing up real backends.

### Parallel chaos runs

`ChaosExperiment` inherits `run_evaluations_async`. The `@eval_task(TracedHandler())` form above is sequential-only — `TracedHandler.before()` calls `exporter.clear()` on a shared in-memory exporter, so concurrent workers wipe each other's in-flight spans. For `max_workers > 1`, capture spans manually and let the mapper partition them by session ID. `StrandsInMemorySessionMapper` filters spans by `session.id` / `gen_ai.conversation.id` when those attributes are present, so stamping them on the agent gives you per-case isolation against a shared exporter:

```python
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
mapper = StrandsInMemorySessionMapper()

def task(case):
    agent = Agent(
        tools=[search_tool, database_tool],
        plugins=[ChaosPlugin()],
        trace_attributes={
            "session.id": case.session_id,
            "gen_ai.conversation.id": case.session_id,
        },
    )
    output = str(agent(case.input))
    spans = telemetry.in_memory_exporter.get_finished_spans()
    session = mapper.map_to_session(spans, case.session_id)
    return {"output": output, "trajectory": session}

report = await ChaosExperiment(cases=chaos_cases, evaluators=[...]).run_evaluations_async(
    task=task, max_workers=10,
)
```

Note: this leaves spans in the exporter across cases (no `clear()`); memory grows with case count. Acceptable for typical eval runs, but flush manually if you're sweeping thousands of cases.

`ChaosCase.expand(base_cases, effect_maps, include_no_effect_baseline=True)` produces the Cartesian product (cases × effect maps) plus an optional baseline run per case. `ChaosPlugin` reads the active case from a `ContextVar` set by `ChaosExperiment` — do not instantiate cases by hand inside a plain `Experiment` and expect effects to fire.

## Red Team (adversarial evaluation)

Lives under `strands_evals.experimental.redteam`. Runs the case × strategy cross-product against a target agent and produces a `RedTeamReport`.

```python
from strands import Agent
from strands_evals.experimental.redteam import (
    RedTeamExperiment, AdversarialCaseGenerator, AttackSuccessEvaluator,
    CrescendoStrategy, GoatStrategy, PairStrategy,
    BadLikertJudgeStrategy, SequentialBreakStrategy, PromptStrategy,
    AttackGoal, RedTeamConfig, RISK_CATEGORIES,
)

target = Agent(model=..., system_prompt=..., tools=[...])

# 1. Generate cases tailored to the target (or hand-author RedTeamCase)
cases = AdversarialCaseGenerator(model=judge_model).generate_cases(
    agent=target,
    risk_categories=["prompt_injection", "data_exfiltration"],
    num_cases=5,
)

# 2. Run case × strategy cross-product
experiment = RedTeamExperiment(
    cases=cases,
    agent=target,
    attack_strategies=[CrescendoStrategy(max_turns=10), PairStrategy(max_turns=8)],
    evaluators=[AttackSuccessEvaluator(model=judge_model, pass_threshold=0.3)],
    model=judge_model,
)
report = experiment.run_evaluations()
report.display()
```

Targets accepted: a `strands.Agent`, a `strands.multiagent.Graph`/`Swarm` (any `MultiAgentBase`), or a custom `TargetSession` Protocol implementer. For parallel runs (`max_workers > 1`) pass `agent_factory=` instead of `agent=` — Strands clients carry non-deepcopyable state.

Built-in strategies:

| Strategy | Use |
| --- | --- |
| `PromptStrategy(label, system_prompt_template)` | Single-prompt attacker; one is registered as `BUILTIN_STRATEGIES["gradual_escalation"]` |
| `CrescendoStrategy(max_turns=...)` | Multi-turn ramp from benign to harmful |
| `GoatStrategy(...)` | Generative Offensive Agent Tester loop |
| `PairStrategy(max_turns=...)` | Prompt Automatic Iterative Refinement |
| `BadLikertJudgeStrategy(...)` | Likert-scale judge-prompt attack |
| `SequentialBreakStrategy(...)` | Narrative-scaffold attack (PR #254) |

Targets and sessions in `redteam.strategies`: `StrandsAgentSession`, `StrandsMultiAgentSession`, `TargetCheckpoint`, `TargetSession` (Protocol).

Cases are typed `RedTeamCase` carrying a `RedTeamConfig(attack_goal=AttackGoal(risk_category=..., actor_goal=..., severity=..., success_criteria=...), traits={...})`. `RISK_CATEGORIES` is the canonical category list for case generation.

`AttackSuccessEvaluator` is the default — an LLM-as-judge with continuous 0.0-1.0 scoring, structured-output severity (`refused | partial | substantial | full`), and `pass_threshold` (default 0.3, where pass = score below threshold = attack failed).

`RedTeamReport` adds case-centric grouping: one `AttackResult` per case, plus `GroupedSummary` aggregations exposed via `report.by_risk_category()` and `report.by_strategy()`. Severity is recorded on each `AttackResult` (no `by_severity()` aggregator). `trajectory` holds raw tool I/O — sanitize before sharing if tools return sensitive data.

**Hard turn cap:** `task.py` enforces `MAX_ALLOWED_TURNS = 50` regardless of a strategy's own `max_turns`. A `CrescendoStrategy(max_turns=100)` will still stop at 50 inside `RedTeamExperiment`. Lower turn budgets honor the strategy setting.

Stability: `experimental.redteam` APIs may change in a minor release. Breaking changes (renames, removed args, changed defaults) go through a deprecation cycle with a `DeprecationWarning` for at least one minor version.

## ExperimentGenerator (auto test-case generation)

```python
from strands_evals.generators import ExperimentGenerator
from strands_evals.evaluators import TrajectoryEvaluator

tool_context = """
Available tools:
- calculator(expression: str) -> float
- web_search(query: str) -> str
- file_read(path: str) -> str
"""

generator = ExperimentGenerator[str, str](str, str)
experiment = await generator.from_context_async(
    context=tool_context,
    num_cases=10,
    evaluator=TrajectoryEvaluator,
    task_description="Math + research assistant with tools",
    num_topics=3,  # spread cases across topics
)
experiment.to_file("generated_experiment", "json")
```

## Custom Evaluator

```python
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class PolicyComplianceEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        violations = self._check(evaluation_case.actual_output)
        if not violations:
            return [EvaluationOutput(score=1.0, test_pass=True, reason="compliant", label="compliant")]
        return [EvaluationOutput(
            score=0.0,
            test_pass=False,
            reason=f"violations: {', '.join(violations)}",
            label="non_compliant",
        )]

    def _check(self, response: str) -> list[str]: ...
```

For LLM-backed custom evaluators, route through `strands.Agent`. The package treats this as a hard rule, see `AGENTS.md`. Use `agent(prompt, structured_output_model=PydanticModel)` for structured scoring. Default judge model is `global.anthropic.claude-sonnet-4-6`.

## Async Execution

`run_evaluations` is sync; under the hood it delegates to `run_evaluations_async(max_workers=1)`. For parallel runs:

```python
report = await experiment.run_evaluations_async(task, max_workers=10)
```

- `task` may be sync or async — async tasks require `run_evaluations_async`; passing one to `run_evaluations` raises `ValueError`.
- `max_workers` defaults to 10; the runner caps it at `len(cases)`.
- `TracedHandler` shares a single in-memory exporter, so it is **not** safe under `max_workers > 1`. Either run sequentially or build a per-call `TracedHandler` inside the task.
- Red-team parallel runs must use `agent_factory=` (see Red Team section).

## Result Caching

`Experiment.run_evaluations` and `run_evaluations_async` accept `evaluation_data_store=` to cache per-case `EvaluationData` and skip cases that already have results.

```python
from strands_evals import LocalFileTaskResultStore  # writes one JSON per case

store = LocalFileTaskResultStore("./results")
report = experiment.run_evaluations(task, evaluation_data_store=store)
```

For non-filesystem backends (S3, DB, etc.), implement the `EvaluationDataStore` Protocol — it requires only `load(case_name) -> EvaluationData | None` and `save(case_name, result)`.

## Authoring Best Practices

- **Diversify cases**: knowledge, reasoning, tool use, multi-turn, edge cases, safety.
- **Combine evaluators**: cheap deterministic + LLM judges + trace-based, scoped to what you actually need.
- **Always extract trajectories** before passing to evaluators. Never feed raw `agent.messages`.
- **Set `session.id` and `gen_ai.conversation.id`** in `trace_attributes` so mappers can group spans.
- **Run multiple times** for non-determinism. LLM judges have variance, baseline statistically.
- **Use a stronger judge for harder evals**, override `model=...` per evaluator when the default is not enough.
- **Persist experiments to JSON** for reproducibility and version-control alongside agent configs.

## Type Annotations

Use Python built-in generics (PEP 585) and `|` unions (PEP 604). The package targets Python >=3.10, so these work without `from __future__ import annotations`.

```python
# Good
def f(items: list[str], opts: dict[str, int]) -> tuple[str, ...] | None: ...
def g(model: Model | str | None = None) -> list[EvaluationOutput]: ...

# Avoid
from typing import List, Dict, Tuple, Optional, Union
def f(items: List[str], opts: Dict[str, int]) -> Optional[Tuple[str, ...]]: ...
def g(model: Union[Model, str, None] = None) -> List[EvaluationOutput]: ...
```

Only import from `typing` for symbols without a built-in equivalent: `Any`, `Callable`, `Iterable`, `Sequence`, `Mapping`, `Protocol`, `TypedDict`, `TypeVar`, `Generic`, `Literal`, `cast`, `overload`, `Self`.

## Pointers

- Repo conventions, contribution rules, prompt versioning, review checklist: `AGENTS.md`
- CLI workflow (`strands-evals run` / `validate` / `diagnose` / `report` / `generate` / `fetch`): see the README "Command-Line Interface" section and `--help` on each subcommand. `fetch` has provider-scoped sub-subcommands (`fetch cloudwatch`, `fetch langfuse`, `fetch opensearch`) that emit a Session JSON to stdout or `-o PATH`, ready to pipe into `strands-evals diagnose -`.
- Logging style: `STYLE_GUIDE.md`
- Human contributor guide: `CONTRIBUTING.md`
- User docs: https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/quickstart/
