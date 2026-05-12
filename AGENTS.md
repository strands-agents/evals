# AGENTS.md

This document provides context, patterns, and guidelines for AI coding assistants working in this repository. For human contributors, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Product Overview

**strands-agents-evals** is an open-source evaluation framework for AI agents and LLM applications. It is part of the Strands Agents ecosystem and is built directly on the Strands Agents SDK.

**Core Features:**
- LLM-as-a-Judge evaluators with built-in rubrics (correctness, faithfulness, helpfulness, etc.)
- Multimodal evaluators (image-to-text tasks)
- Deterministic evaluators (output/trajectory/environment-state)
- Trace-based evaluation over OpenTelemetry sessions from CloudWatch, Langfuse, OpenSearch, LangChain, and Strands in-memory
- Multi-turn conversation simulators (ActorSimulator / UserSimulator / ToolSimulator)
- Failure detection and root-cause analysis (`detectors` module)
- Automated experiment / test-case generation
- Experiment management with JSON serialization

## Relationship to the Strands Agents SDK

strands-evals depends on **`strands-agents`** (the Python SDK) for all LLM interactions. Any change in this repo that touches model calls, tools, sessions, traces, hooks, or structured output should be interpreted against the SDK's public API.

- SDK source (external, not in this repo): [`strands-agents/sdk-python`](https://github.com/strands-agents/sdk-python) under `src/strands/`
- Key SDK surfaces actually imported by this repo (grep-verified):
  - `strands.Agent`, called as `Agent(model=..., system_prompt=..., callback_handler=None)` and invoked via `agent(prompt, structured_output_model=PydanticModel)`
  - `strands.models.model.Model` and provider classes
  - `strands.agent.agent_result.AgentResult`, `strands.agent.conversation_manager.SlidingWindowConversationManager`
  - `strands.tools.decorator` (`DecoratedFunctionTool`, `FunctionToolMetadata`) and the `@tool` decorator from `strands`
  - `strands.types.content.Message`
  - `strands.types.exceptions` (`EventLoopException`, `ModelThrottledException`)
- **Not used as SDK imports (do not add to the list without verifying a real import):**
  - `strands.types.traces` — this repo defines its own `strands_evals.types.trace` for session/span modeling; the SDK's trace types are not imported.
  - `strands.telemetry` — the evals repo has its own `strands_evals.telemetry` module. The only reference to `strands.telemetry` is the literal string `"strands.telemetry.tracer"` used as an OpenTelemetry scope name in `mappers/constants.py`.

**Review guidance:** When reviewing a PR in this repo, if the change uses any SDK feature (agents, tools, models, streaming, structured output, sessions, hooks, telemetry), verify the usage matches the current SDK public API. If the SDK source is available locally (e.g., via a sibling checkout), scan it; otherwise rely on the imported symbols and docstrings. Flag usages that depend on private SDK internals (anything under `_*` modules) or deprecated experimental surfaces.

## Directory Structure

```
strands-evals/
│
├── src/strands_evals/                        # Main package source code
│   ├── __init__.py                           # Public API exports
│   ├── case.py                               # Case[InputT, OutputT] test scenario
│   ├── experiment.py                         # Experiment orchestration + run_evaluations
│   ├── eval_task_handler.py                  # EvalTaskHandler, TracedHandler, @eval_task
│   ├── evaluation_data_store.py              # EvaluationDataStore
│   ├── local_file_task_result_store.py       # LocalFileTaskResultStore
│   ├── _async.py                             # Async helpers
│   ├── utils.py                              # Shared utilities
│   │
│   ├── evaluators/                           # Evaluator implementations
│   │   ├── evaluator.py                      # Base Evaluator[InputT, OutputT]
│   │   ├── coherence_evaluator.py
│   │   ├── conciseness_evaluator.py
│   │   ├── correctness_evaluator.py
│   │   ├── faithfulness_evaluator.py
│   │   ├── goal_success_rate_evaluator.py
│   │   ├── harmfulness_evaluator.py
│   │   ├── helpfulness_evaluator.py
│   │   ├── instruction_following_evaluator.py
│   │   ├── interactions_evaluator.py
│   │   ├── output_evaluator.py
│   │   ├── refusal_evaluator.py
│   │   ├── response_relevance_evaluator.py
│   │   ├── stereotyping_evaluator.py
│   │   ├── tool_parameter_accuracy_evaluator.py
│   │   ├── tool_selection_accuracy_evaluator.py
│   │   ├── trajectory_evaluator.py
│   │   ├── multimodal_*.py                   # Multimodal evaluators (MLLM-as-a-Judge)
│   │   ├── deterministic/                    # Non-LLM evaluators
│   │   │   ├── output.py
│   │   │   ├── trajectory.py
│   │   │   └── environment_state.py
│   │   └── prompt_templates/                 # Per-evaluator prompt modules
│   │       └── {evaluator_name}/{name}_v0.py # Exports SYSTEM_PROMPT constant
│   │
│   ├── detectors/                            # Failure detection + RCA
│   │   ├── failure_detector.py               # detect_failures (model.stream text mode)
│   │   ├── root_cause_analyzer.py            # analyze_root_cause (Agent + structured output)
│   │   ├── diagnosis.py                      # diagnose_session pipeline
│   │   ├── chunking.py                       # Context-overflow fallback
│   │   ├── utils.py                          # _resolve_model, _serialize_session, etc.
│   │   ├── constants.py
│   │   └── prompt_templates/
│   │       ├── failure_detection/
│   │       └── root_cause/                   # root_cause_v0 + root_cause_merge_v0
│   │
│   ├── simulation/                           # Multi-turn simulators
│   │   ├── actor_simulator.py                # ActorSimulator (acts as user)
│   │   ├── tool_simulator.py                 # ToolSimulator (LLM-powered tool responses)
│   │   ├── profiles/actor_profile.py
│   │   ├── tools/goal_completion.py
│   │   └── prompt_templates/
│   │
│   ├── generators/                           # Test-case/experiment generation
│   │   ├── experiment_generator.py
│   │   ├── topic_planner.py
│   │   └── prompt_template/prompt_templates.py
│   │
│   ├── extractors/                           # Extract trajectories from sessions
│   │   ├── trace_extractor.py
│   │   ├── tools_use_extractor.py
│   │   ├── graph_extractor.py
│   │   └── swarm_extractor.py
│   │
│   ├── mappers/                              # Session providers -> Session type
│   │   ├── session_mapper.py                 # Base interface
│   │   ├── cloudwatch_session_mapper.py
│   │   ├── cloudwatch_parser.py
│   │   ├── langchain_otel_session_mapper.py
│   │   ├── openinference_session_mapper.py
│   │   ├── opensearch_session_mapper.py
│   │   └── strands_in_memory_session_mapper.py
│   │
│   ├── providers/                            # External trace providers
│   │   ├── trace_provider.py                 # Base interface
│   │   ├── cloudwatch_provider.py
│   │   ├── langfuse_provider.py
│   │   ├── opensearch_provider.py
│   │   └── exceptions.py
│   │
│   ├── telemetry/                            # OpenTelemetry integration
│   │   ├── tracer.py                         # get_tracer
│   │   ├── config.py                         # StrandsEvalsTelemetry
│   │   └── _cloudwatch_logger.py
│   │
│   ├── display/
│   │   └── display_console.py                # Rich console rendering for reports
│   │
│   ├── tools/
│   │   └── evaluation_tools.py               # Helpers used by evaluators
│   │
│   └── types/                                # Public type definitions
│       ├── evaluation.py                     # EvaluationData / EvaluationOutput
│       ├── evaluation_report.py
│       ├── trace.py                          # Session, Trace, SpanUnion
│       ├── detector.py                       # FailureItem, RCAItem, DiagnosisResult, ...
│       ├── multimodal.py
│       └── simulation/
│           ├── actor.py
│           └── tool.py
│
├── tests/strands_evals/                      # Unit tests (mirrors src/strands_evals/)
│   ├── evaluators/
│   │   └── deterministic/
│   ├── detectors/
│   ├── simulation/
│   ├── extractors/
│   ├── generators/
│   ├── mappers/
│   ├── providers/
│   ├── telemetry/
│   ├── tools/
│   └── types/
│
├── tests_integ/                              # Integration tests (real providers)
│
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── STYLE_GUIDE.md
└── AGENTS.md                                 # This file
```

### Directory Purposes

- **`src/strands_evals/`**: All production code
- **`tests/strands_evals/`**: Unit tests mirroring src/ structure
- **`tests_integ/`**: Integration tests using real trace providers and model endpoints
- **`docs are not vendored yet`**: CONTRIBUTING.md and STYLE_GUIDE.md are the canonical references

**IMPORTANT**: After making changes that affect the directory structure, update this section to reflect the current state of the repository.

## Core Architecture Patterns

### Evaluator Pattern

All evaluators subclass `strands_evals.evaluators.Evaluator[InputT, OutputT]`:

- Public methods: `evaluate()` and `evaluate_async()`
- Return type: `list[EvaluationOutput]` with fields `score`, `test_pass`, `reason`, `label`
- LLM calls use Strands Agent directly:
  ```python
  Agent(model=self.model, system_prompt=..., callback_handler=None)
  ```
- `model` parameter type: `Union[Model, str, None]`, supports all 13+ Strands model providers
- Structured output pattern: `agent(prompt, structured_output_model=PydanticModel)`

### Default Judge Model

`DEFAULT_BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"` in `src/strands_evals/evaluators/evaluator.py`. Updated to sonnet-4-6 in PR #215.

### Prompt Management

- **No Jinja2**. Prompts are Python string constants in versioned modules.
- Pattern: `evaluators/prompt_templates/{name}/{name}_v0.py` exports `SYSTEM_PROMPT`.
- `get_template(version)` returns the module whose `SYSTEM_PROMPT` is used.
- The `detectors/prompt_templates/root_cause/` directory has both `root_cause_v0.py` and `root_cause_merge_v0.py`, with `get_template` and `get_merge_template` selectors.

### Case and Experiment

- `Case[InputT, OutputT]`: one test scenario (`input`, `expected_output`, optional `trajectory`, `metadata`)
- `Experiment[InputT, OutputT]`: collection of Cases plus evaluators
- Entry point: `experiment.run_evaluations(task_function)` returns a list of reports

### Session / Trace Types

Defined in `strands_evals.types.trace`:

- `Session` → `list[Trace]` → `list[SpanUnion]`
- `SpanUnion = InferenceSpan | ToolExecutionSpan | AgentInvocationSpan`
- Mapping functions live in `mappers/`, one per external source.

### Detectors Module Public API

Exported from `detectors/__init__.py`:

- `detect_failures(session, *, confidence_threshold, model)` — uses `model.stream()` in text mode (tool-use structured output was observed to degrade detection quality). Falls back to chunked windows on context overflow.
- `analyze_root_cause(session, failures=None, *, model)` — uses Strands Agent + structured output (`RCAStructuredOutput`). Three-tier fallback: direct → failure-path pruning (ancestors + up to 10 descendants) → chunked windows merged by LLM.
- `diagnose_session(session, *, model, confidence_threshold)` — thin pipeline: `detect_failures` → `analyze_root_cause` → `DiagnosisResult`.

Shared helpers live in `detectors/utils.py`. Types live in `types/detector.py`. `ConfidenceLevel` is an `Enum` (`.LOW` / `.MEDIUM` / `.HIGH`), not a `Literal`.

## Coding Patterns and Best Practices

### Code Conventions

- **No `_internal/` directories anywhere** — this is an open-source repo, visible surface should be deliberate.
- Private functions and modules use the `_` prefix.
- **No litellm dependency** — all LLM calls go through `strands.Agent`. Do not introduce litellm.
- Core runtime dependencies: `strands-agents`, `pydantic`, `rich`, `boto3`, `tenacity`, `opentelemetry-*`.

### Type Annotations

- Type annotations are required on all public signatures.
- Use `typing` / `typing_extensions` for complex types.
- Mypy runs via `hatch fmt --linter`. Strict checks disabled selectively in `pyproject.toml` for Generic-class patterns; see `[tool.mypy]` overrides.

### Docstrings

Use Google-style docstrings for public functions, classes, and modules.

### Logging Style

See [STYLE_GUIDE.md](./STYLE_GUIDE.md). Structured fields first, human-readable message after a pipe:

```python
logger.debug("user_id=<%s>, action=<%s> | user performed action", user_id, action)
```

- Use `%s` string interpolation, not f-strings (logging performance).
- Lowercase message, no punctuation.
- Separate multiple statements with `|`.

### Import Organization

Auto-organized by ruff/isort:
1. Standard library
2. Third-party
3. Local

Imports must live at the top of the file.

### Naming

- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_` prefix

### Error Handling

- Provide clear error messages with context.
- Do not swallow exceptions silently.
- For context-exceeded model errors in `detectors/`, use `_is_context_exceeded` from `detectors/utils.py`.

## Testing Patterns

### Unit Tests (`tests/strands_evals/`)

- Mirror the `src/strands_evals/` structure exactly.
- Mock external dependencies (LLMs, AWS services, trace providers).
- `moto` is declared as a test dependency in `pyproject.toml` for AWS mocking, but no test file currently imports it. Treat it as available, not established prior art.
- Use `pytest-asyncio` (`asyncio_mode = "auto"`) for async tests.

### Integration Tests (`tests_integ/`)

- Use real model providers and trace backends.
- Require credentials via environment variables.

### File Naming

- Unit: `test_{module}.py` in `tests/strands_evals/{path}/`
- Integration: `test_{feature}.py` in `tests_integ/`

### Running Tests

```bash
hatch test                           # Unit tests
hatch test -c                        # With coverage
hatch run test-integ                 # Integration tests
hatch test tests/strands_evals/evaluators/    # Targeted
hatch test --all                     # All supported Python versions (3.10-3.13)
```

## Development Commands

```bash
# Environment
hatch shell

# Formatting & linting
hatch fmt --formatter                 # Format
hatch fmt --linter                    # Lint (ruff + mypy)

# Testing
hatch test
hatch test -c
hatch run test-integ

# Full readiness check
hatch run prepare                     # lint + format + test-lint + test --all

# Pre-commit
pre-commit install -t pre-commit -t commit-msg
pre-commit run --all-files
```

## PR and Commit Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep PRs focused. Describe the **why** more than the **what**.
- Title should be short, body carries detail.
- CI runs `test-lint` and the full test matrix; don't merge with failures.

## Things to Do

- Use explicit return types for all functions.
- Write Google-style docstrings for public APIs.
- Use structured logging format.
- Mirror `src/` structure in `tests/`.
- Run `hatch fmt --formatter` and `hatch fmt --linter` before committing.
- Route all LLM calls through `strands.Agent`.
- Version prompt templates as Python string constants in `{name}_v0.py`.
- When editing `detectors/`, keep the three-tier RCA fallback and the `model.stream()` text-mode path for failure detection intact unless the PR explicitly targets those behaviors.

## Things NOT to Do

- Don't create `_internal/` directories — this is open source.
- Don't introduce litellm or wrap model providers outside `strands.Agent`.
- Don't use Jinja2 for prompts; use Python string constants.
- Don't use f-strings in logging calls.
- Don't add punctuation or capital letters to log messages.
- Don't rely on private SDK internals (`strands._*`).
- Don't put unit tests outside `tests/strands_evals/`.
- Don't commit without running pre-commit hooks.

## Review Guidance for AI Agents

When reviewing a PR in this repo, scan for all of the following:

1. **Evaluator changes**: confirm the class still extends `Evaluator[InputT, OutputT]`, still returns `list[EvaluationOutput]`, and still uses `Agent(...)` for LLM calls. Confirm the default model has not silently regressed from the current default.
2. **Prompt changes**: check that a new version module (`{name}_v1.py`) was added rather than overwriting `_v0.py`, if the PR is billed as a prompt change rather than a bugfix.
3. **Detector changes**: preserve the text-mode streaming in `failure_detector.py` and the three-tier fallback in `root_cause_analyzer.py` unless explicitly targeted.
4. **Session/trace schema**: if the PR changes fields on `Session`, `Trace`, or `SpanUnion`, every mapper in `mappers/` must still produce valid instances. Check each.
5. **SDK usage**: if the PR calls into `strands.*`, verify the symbol exists in the current SDK public API and is not under an `_` module or an experimental surface. When the SDK source is available as a sibling checkout (`../sdk-python/` or similar), read the relevant SDK file and confirm the signature matches.
6. **Dependencies**: no new litellm, no new Jinja2, no new `_internal/` directories.
7. **Style & logging**: structured logging with `%s` interpolation, lowercase messages, no f-strings in logger calls.
8. **Tests**: new code needs a mirrored test file in `tests/strands_evals/...`. Async code uses `pytest-asyncio` auto mode.

## Additional Resources

- [CONTRIBUTING.md](./CONTRIBUTING.md) — Human contributor guidelines
- [STYLE_GUIDE.md](./STYLE_GUIDE.md) — Logging and style conventions
- [Strands Agents Documentation](https://strandsagents.com/)
- [Strands Agents Python SDK](https://github.com/strands-agents/sdk-python)
- [Strands Agents Tools](https://github.com/strands-agents/tools)
- [Strands Agents Samples](https://github.com/strands-agents/samples)
