<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://strandsagents.com/latest/assets/wordmark-github-dark.svg">
        <img src="https://strandsagents.com/latest/assets/wordmark-github-light.svg" alt="Strands" width="320">
      </picture>
    </a>
  </div>

  <h1>
    Strands Evals SDK
  </h1>
  <h2>
    A comprehensive evaluation framework for AI agents and LLM applications.
  </h2>

  <div align="center">
    <a href="https://github.com/strands-agents/evals/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/evals"/></a>
    <a href="https://pypi.org/project/strands-agents-evals/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents-evals"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents-evals"/></a>
    <a href="https://discord.gg/strands"><img alt="Strands Discord" src="https://img.shields.io/badge/Discord-Strands-5865F2?logo=discord&logoColor=white"/></a>
  </div>
  
  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/harness-sdk/tree/main/strands-cli">CLI</a>
    ◆ <a href="https://github.com/strands-agents/harness-sdk/tree/main/strands-mcp">MCP Server</a>
    <br>
    Harness (<a href="https://github.com/strands-agents/harness-sdk/tree/main/harness-py">Python</a>, <a href="https://github.com/strands-agents/harness-sdk/tree/main/harness-ts">TypeScript</a>)
    ◆ SDK (<a href="https://github.com/strands-agents/harness-sdk/tree/main/strands-py">Python</a>, <a href="https://github.com/strands-agents/harness-sdk/tree/main/strands-ts">TypeScript</a>)
    ◆ <a href="https://github.com/strands-agents/box">Box</a>
    ◆ <a href="https://github.com/strands-agents/shell">Shell</a>
    ◆ <a href="https://github.com/strands-agents/evals">Evals</a>
    ◆ <a href="https://github.com/strands-labs">Labs</a>
  </p>
</div>

Strands Evaluation is a powerful framework for evaluating AI agents and LLM applications. From simple output validation to complex multi-agent interaction analysis, trajectory evaluation, and automated experiment generation, Strands Evaluation provides comprehensive tools to measure and improve your AI systems.

## Feature Overview

- **Multiple Evaluation Types**: Output evaluation, trajectory analysis, tool usage assessment, and interaction evaluation
- **Multimodal Evaluation**: MLLM-as-a-Judge evaluators for image-to-text tasks with built-in rubrics
- **Skill Evaluation**: Assess which skills a skill-equipped agent selected and whether it followed their instructions
- **Dynamic Simulators**: Multi-turn conversation simulation with realistic user behavior, goal-oriented interactions, and LLM-powered tool simulation with shared state
- **LLM-as-a-Judge**: Built-in evaluators using language models for sophisticated assessment with structured scoring
- **Typed Rubrics**: Score cases against yes/no, ordered-scale and choice questions with one decision-model call, from the model's probabilities instead of generated text
- **Trace-based Evaluation**: Analyze agent behavior through OpenTelemetry execution traces
- **Automated Experiment Generation**: Generate comprehensive test suites from context descriptions
- **Custom Evaluators**: Extensible framework for domain-specific evaluation logic
- **Experiment Management**: Save, load, and version your evaluation experiments with JSON serialization
- **Built-in Scoring Tools**: Helper functions for exact, in-order, and any-order trajectory matching
- **Failure Detection & Root Cause Analysis**: Automatically detect failures in agent sessions and diagnose root causes with actionable fix recommendations
- **Chaos Testing**: Deterministic fault injection via Strands plugin hooks — simulate tool timeouts, network errors, and response corruption to evaluate agent resilience
- **Agent Optimization**: Declare what may change about an agent (prompts, skills, tool descriptions, tool sets, components, numeric settings) and search for the best configuration on your experiment, returning a reviewable diff and typed values
- **Red Team Evaluation**: Adversarial safety testing with built-in attack strategies (Crescendo, GOAT, PAIR, BadLikertJudge, SequentialBreak); see [src/strands_evals/redteam/README.md](src/strands_evals/redteam/README.md)

## Quick Start

```bash
# Install Strands Evals SDK
pip install strands-agents-evals
```

```python
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator

# Create test cases
test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        metadata={"category": "knowledge"}
    )
]

# Create evaluators with custom rubric
evaluators = [
    OutputEvaluator(
        rubric="""
        Evaluate based on:
        1. Accuracy - Is the information correct?
        2. Completeness - Does it fully answer the question?
        3. Clarity - Is it easy to understand?
        
        Score 1.0 if all criteria are met excellently.
        Score 0.5 if some criteria are partially met.
        Score 0.0 if the response is inadequate.
        """
    )
]

# Create experiment and run evaluation
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(case.input))

# Run evaluations
report = experiment.run_evaluations(get_response)
report.run_display()
```

## Command-Line Interface

Installing `strands-agents-evals` also installs the `strands-evals` console script, a thin wrapper over the Python API for CI and one-off use. It has eight subcommands:

| Command | Purpose |
| --- | --- |
| `strands-evals run` | Execute an Experiment against an `--agent` factory or `--task` callable, or run a single ad-hoc case via `--input` + `--evaluator`/`--expected-output`/`--rubric`. |
| `strands-evals validate` | Schema-check a serialized Experiment JSON file (CI gate before `run`). |
| `strands-evals report` | Render an existing `EvaluationReport` JSON via Rich, or dump it as JSON. |
| `strands-evals diagnose` | Run `detect_failures`, `analyze_root_cause`, or the full `diagnose_session` pipeline on a Session JSON file. |
| `strands-evals generate` | Synthesize an Experiment via `ExperimentGenerator` from a free-form `--context` or an existing `--experiment` file. |
| `strands-evals fetch` | Pull a Session from a trace provider into Session JSON for `diagnose` or a trajectory evaluator. |
| `strands-evals optimize` | Search for a better agent configuration against an Experiment (`run`), or continue a stopped run (`resume`). Needs the `optimization` extra. |
| `strands-evals ui` | Browse the experiments and reports in a directory in a local page, and edit their cases in place. |

Common flows:

```bash
# Schema-check, then run an experiment file against an agent factory
strands-evals validate experiments/customer_service.json
strands-evals run experiments/customer_service.json \
  --agent my_pkg.agents:build_agent \
  --display

# Run a single case (by name or 0-based index) against all of the experiment's evaluators
strands-evals run experiments/customer_service.json \
  --agent my_pkg.agents:build_agent \
  --case refund_request

# One-off ad-hoc run with no experiment file
strands-evals run \
  --input "What is the capital of France?" \
  --expected-output "Paris" \
  --agent my_pkg.agents:build_agent

# Diagnose a failing session captured to JSON
strands-evals diagnose session.json --confidence medium

# Generate a starter experiment from a tools description
strands-evals generate --context "$(cat tools.txt)" --num-cases 10 \
  --evaluator TrajectoryEvaluator -o experiments/generated.json

# Browse and edit the experiments under a directory in a browser
strands-evals ui experiments/

# Same discovery, printed instead of served (for CI, or just to see what is there)
strands-evals ui experiments/ --list
```

`ui` binds loopback on a random port and prints a link carrying a token generated for that run. Every
request needs that token, so a page open in another browser tab cannot read or write through it.
Opening the link trades the token for a cookie and lands on a clean address, so the token is not left
in the address bar or the browser history. Edits save back to the file they were loaded from, with a
conflict check against the file's mtime so a concurrent `run` or editor is never silently overwritten.
While the page is open it checks every few seconds whether the folder changed and re-reads what it is
showing, so a run in progress fills in without a reload; unsaved edits are never replaced.

The sidebar lists every optimizer run the scan found, under Optimization and the Runs view in it, so
opening one is a click from any page rather than a trip through the list first. Runs itself still
opens that list, with the columns a sidebar row has no space for.

Every view has an address of its own, so Back and Forward work and a view can be bookmarked or shared
with someone serving the same folder:

| Address | Shows |
| --- | --- |
| `/reports` | Every report found, each with the experiment it was made from. |
| `/reports/<report file>` | One report. |
| `/reports/<report file>/rows/<case>/<evaluator>` | One row of a report, opened and in view. |
| `/reports/compare?r=<report file>&r=<report file>` | Two or more reports side by side, oldest first. |
| `/experiments` | Every experiment file found, each with the reports made from it. |
| `/experiments/<experiment file>` | Its cases; add `/cases/<name>` for one case. |
| `/sessions` | Every recorded session found. Listed only when there is one. |
| `/sessions/<session file>` | One session, trace by trace, with any diagnosis made of it. |
| `/optimization` | The optimizer runs found. |
| `/optimization/<run directory>/<view>` | One run: `setup`, `overview`, `progress`, `candidates`, `tunables` or `trade-off`. |
| `/optimization/<run directory>/candidates/<id>` | One candidate of a run, opened and in view. |
| `/optimization/<run directory>/candidates/<id>/report` | One candidate's cases, as a report. |

An address that names something the folder does not hold is not redirected: the page says what it looked
for and leaves the address as typed, so it opens on its own once the file is there.

The page is also written to be handed to a coding agent. A report, a report row, a comparison, an
experiment, a case, a session, a candidate, an optimizer result and a trace each have a **Copy as Markdown**
action, beside a **Copy path** for the file and a **Copy command** that runs one case again; a report with
failing rows has **Copy failing rows as a prompt**, which bundles them with the paths of the report and its
experiment and the command to run one case again, with `MODULE:ATTR` standing for the agent under test: no
file records which agent was evaluated. The same Markdown is served for a file, so an agent can read what
the page shows without a browser. `/llms.txt`, the `.md` and `.json` forms, `POST /api/markdown`, the shape
of the listing below and the wording of the copied prompt are experimental: they may change before a release.

| Address | Returns |
| --- | --- |
| `/reports/<report file>.md` | The report as Markdown: its figures, each failing row with its input, expected and actual output and the evaluator's reason, then every row. |
| `/experiments/<experiment file>.md` | The experiment as Markdown: its evaluators with their settings, then every case. |
| `/sessions/<session file>.md` | The session as Markdown: what it recorded, each diagnosis made of it, then every span of every trace. A diagnosis file is read the same way. |
| `/reports/<report file>.json`, `/experiments/<experiment file>.json`, `/sessions/<session file>.json` | The file's document. |
| `POST /api/markdown` | The Markdown of a document sent to it, or of one row, one case, the failing rows or a trace of one. The page uses it for what has no file of its own. It reads nothing from the folder: a path in the body is only a name to write. |
| `/llms.txt` | What the server answers: these addresses, the `/api` routes and how the token is sent. |

These need the token itself, as the `X-Evals-Token` header or as `?t=<token>`; the page's cookie does not
open them. For example: `curl -H "X-Evals-Token: $TOKEN" http://127.0.0.1:PORT/reports/support/support.report.json.md`.
A page address asked for by something that is not a browser (no `text/html` in its `Accept` header) answers
`406` with a few lines naming these forms, in place of the page.

In the Markdown, a value longer than 4,000 characters is cut, with a line saying how much was left out; the
`.json` form has all of it. A report writes each failing row in full, a case that fails under several
evaluators once, and one table line per row. Every row and every span is written, however many there are. A
diagnosis file is written whole; a diagnosis held in a report row is cut like any other value.

`/api/scan` and `ui --list --json` give the same listing. Its `schema_version` is `1`, and moves when a field
is renamed, removed or changes meaning:

| Field | Holds |
| --- | --- |
| `experiments[]` | `path`, `name`, `dir`, `size`, `modified`, `cases` (a count), `evaluators` (each evaluator's entry as the file has it), `evaluator_notes`, `reports` (paths). |
| `reports[]` | `path`, `name`, `dir`, `size`, `modified`, `overall_score`, `rows`, `failing` and `not_applicable` (counts), `evaluators` (names), `experiment` (a path, or null), `link` (`status` and `detail`, or null). |
| `sessions[]` | `path`, `name`, `dir`, `size`, `modified`, `session_id`, `traces` and `spans` (counts), `seconds`, `diagnoses` (paths). |
| `diagnoses[]` | `path`, `name`, `dir`, `size`, `modified`, `session_id`, `failures` and `root_causes` (counts), `session` (a path, or null). |
| `optimizer_runs[]` | One per run directory: its `run.json` as `config`, counts read off its log, `champion_id`, `result`, `stop_reason`, and `tuned_against`, the experiment its split names (`path` is null and `status` is `tie` when several fit equally). |
| `stores[]` | `dir` and `files` (a count) for each `LocalFileTaskResultStore` directory. |
| `skipped[]` | `path` and `why`, for each JSON file that is none of the above. |
| `truncated` | Whether the scan stopped at its cap of 2,000 files. |

`/api/scan` adds `root`, `depth`, `read_only` and `fingerprint`, and `--list --json` adds `root`. The field
names are this listing's own. They differ from `strands-evals validate --json`, which gives
`case_count`, `evaluator_count` and the evaluators' names: here `experiments[].cases` is a count,
`experiments[].evaluators` holds the entries as written, and `reports[].evaluators` holds names.

A report is listed whether or not an experiment file sits beside it, so one made in code with
`run_evaluations()` and saved with `to_file()` shows up too, marked as run from code. One experiment can have
many reports. A report is matched to an experiment in this order:

1. The experiment in its directory under the same name (`nightly.experiment.json` for `nightly.report.json`,
   `experiment.json` for `report.json`), whatever cases the two hold.
2. Failing that, the nearest experiment in the folder that holds every case the report names. Two that are
   equally near are not guessed between: the report is marked as not paired and says which ones tie.
3. Failing that, the only experiment in its directory, when it is the only report there.

Each match carries a status. It compares case identities and nothing else: a report row is identified by its
`session_id` when it has one and by its `name` otherwise, and an experiment's case by either. Against an
experiment none of whose cases has a `session_id`, as one written by hand, a row is identified by its `name`.

| Status | Means |
| --- | --- |
| `verified` | Every case the report names is in the experiment. It does not show that this file produced the report: another experiment with the same case names fits as well. |
| `partial` | Some of the report's cases are not in the experiment, as when it was edited after the run. |
| `mismatch` | None of them are. Only a match by file name, or by being the only pair in a directory, can have this status. |
| `unverified` | There is nothing to compare: the report's rows or the experiment's cases carry no name. |
| `tie` | Not paired: several experiments hold every case and are equally near. |

A session saved with `strands-evals fetch ... -o` is listed under Sessions and read one trace after another.
A diagnosis written by `strands-evals diagnose -o` is shown on the page of the session with its `session_id`,
or listed on its own when that session is not in the folder. The directory of a `LocalFileTaskResultStore`
is named on the lists and its files are left out of the scan.

Pass `--read-only` to browse without the Save path at all, and `--host` to serve beyond this machine
(for example `0.0.0.0` inside a container).

Limits of `ui`:

- The scan reads 2,000 JSON files, 3 directories deep unless `--depth` says otherwise, and says when it
  stopped at the cap.
- The check for changes stats every file the scan lists each time the page asks, every few seconds, with no
  cap. On a folder of tens of thousands of files that takes a noticeable part of a second each time.
- Each `.md` request scans the whole folder, to find what the file is paired with.
- The page works only when `strands-evals ui` serves it. Opened from disk, it says how to start the server.
- A comparison, an optimizer result and a candidate are written by the page and copied from it. They have no
  `.md` address. Neither do a candidate's report and the failing-rows prompt, which the server writes from
  what the page sends to `POST /api/markdown`.
- A document nested more than 200 levels deep is listed, but is not written as Markdown: `.md` and
  `POST /api/markdown` answer `422`. `--list --json` prints a listing that holds one nested deeper than
  Python lays out on one line.
- `POST /api/markdown` takes a body of at most 32 MB. To copy one row, or the failing rows as a prompt, the
  page sends those rows and leaves the rest of the report out, so a copy is refused only when those rows alone
  are larger than that, as is a candidate's report or an unsaved experiment of that size. The button says why.
- Save refuses a file whose modified time differs from the one it was loaded with, and two saves through one
  server are taken in turn. A write by another program between that check and the write is not detected.
- An optimizer run that was resumed after a stop still reads as stopped.
- The page is laid out for windows 360 px wide and up. Below that the top bar overflows.

Run any subcommand with `--help` for the full flag set (custom evaluators via `MODULE:CLASS`, trace attributes, `--fail-on` exit-code rules, output formats, etc.).

## Agent Skills

The package ships skills for coding agents (source in [`.agents/skills/`](.agents/skills/)):

| Skill | Purpose |
| --- | --- |
| [`strands-evals`](.agents/skills/strands-evals/SKILL.md) | Write evaluations: cases and experiments, evaluators, simulators, failure diagnosis, chaos testing, red teaming, and experiment generation. |
| [`session-converter`](.agents/skills/session-converter/SKILL.md) | Turn past agent conversations (local harness sessions or traces) into an experiment JSON file. |
| [`optimize`](.agents/skills/optimize/SKILL.md) | Set up, review, and run `strands-evals optimize` on a strands-harness agent (`agent.py` or `agent.ts`), then report the result. |

The skills ship inside the `strands-agents-evals` package. Install them into your coding assistant with the [`skills` CLI](https://github.com/vercel-labs/skills), run from the environment where you installed the package:

```bash
npx skills add "$(python -c 'import strands_evals, pathlib; print(pathlib.Path(strands_evals.__file__).parent / ".agents" / "skills")')" --copy
```

`--copy` copies the skills into your project, so they keep working if that environment is deleted. Run it again after upgrading `strands-agents-evals`. Add `-s optimize` to install only some skills, `-a claude-code` to pick an assistant, or `-g` to install for every project.

## Installation

Ensure you have Python 3.10+ installed, then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with both test and dev dependencies
pip install -e ".[test,dev]"

# Install with field-level structured-output scoring
pip install -e ".[stickler]"

# Install with agent optimization (numpy, scipy, Optuna, strands-harness)
pip install -e ".[optimization]"
```

## Features at a Glance

### Output Evaluation with Custom Rubrics

Evaluate agent responses using LLM-as-a-judge with flexible scoring criteria:

```python
from strands_evals.evaluators import OutputEvaluator

evaluator = OutputEvaluator(
    rubric="Score 1.0 for accurate, complete responses. Score 0.5 for partial answers. Score 0.0 for incorrect or unhelpful responses.",
    include_inputs=True,  # Include context in evaluation
    model="global.anthropic.claude-sonnet-4-6"  # Custom judge model
)
```

### Trajectory Evaluation with Built-in Scoring

Analyze agent tool usage and action sequences with helper scoring functions:

```python
import operator

from strands import Agent, tool
from strands_evals import Case
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor

@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Apply a basic arithmetic operation to two numbers.

    Args:
        operation: One of "add", "subtract", "multiply" or "divide".
        a: First operand.
        b: Second operand.
    """
    operations = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul, "divide": operator.truediv}
    return operations[operation](a, b)

def get_response_with_tools(case: Case) -> dict:
    agent = Agent(tools=[calculator])
    response = agent(case.input)
    
    # Extract trajectory efficiently to prevent context overflow
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    
    # Update evaluator with tool descriptions
    evaluator.update_trajectory_description(
        tools_use_extractor.extract_tools_description(agent, is_short=True)
    )
    
    return {"output": str(response), "trajectory": trajectory}

# Evaluator includes built-in scoring tools: exact_match_scorer, in_order_match_scorer, any_order_match_scorer
evaluator = TrajectoryEvaluator(
    rubric="Score 1.0 if correct tools used in proper sequence. Use scoring tools to verify trajectory matches."
)
```

### Trace-based Helpfulness Evaluation

Evaluate agent helpfulness using OpenTelemetry traces with seven-level scoring:

```python
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

# Setup telemetry for trace capture
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

def user_task_function(case: Case) -> dict:
    telemetry.memory_exporter.clear()
    
    agent = Agent(
        trace_attributes={"session.id": case.session_id},
        callback_handler=None
    )
    response = agent(case.input)
    
    # Map spans to session for evaluation
    spans = telemetry.memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(spans, session_id=case.session_id)
    
    return {"output": str(response), "trajectory": session}

# Seven-level scoring: Not helpful (0.0) to Above and beyond (1.0)
evaluators = [HelpfulnessEvaluator()]
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# Run evaluations
report = experiment.run_evaluations(user_task_function)
report.run_display()
```

### Multi-turn Conversation Simulation

Simulate realistic user interactions with dynamic, goal-oriented conversations using ActorSimulator:

```python
from strands import Agent
from strands_evals import Case, Experiment, ActorSimulator
from strands_evals.evaluators import HelpfulnessEvaluator, GoalSuccessRateEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# Setup telemetry
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

def task_function(case: Case) -> dict:
    # Create simulator to drive conversation
    simulator = ActorSimulator.from_case_for_user_simulator(
        case=case,
        max_turns=10
    )

    # Create agent to evaluate
    agent = Agent(
        trace_attributes={
            "gen_ai.conversation.id": case.session_id,
            "session.id": case.session_id
        },
        callback_handler=None
    )

    # Run multi-turn conversation
    all_spans = []
    user_message = case.input

    while simulator.has_next():
        memory_exporter.clear()
        agent_response = agent(user_message)
        turn_spans = list(memory_exporter.get_finished_spans())
        all_spans.extend(turn_spans)

        user_result = simulator.act(str(agent_response))
        user_message = str(user_result.structured_output.message)

    # Map to session for evaluation
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(all_spans, session_id=case.session_id)

    return {"output": str(agent_response), "trajectory": session}

# Use evaluators to assess simulated conversations
evaluators = [
    HelpfulnessEvaluator(),
    GoalSuccessRateEvaluator()
]

experiment = Experiment(cases=test_cases, evaluators=evaluators)
report = experiment.run_evaluations(task_function)
```

**Key Benefits:**
- **Dynamic Interactions**: Simulator adapts responses based on agent behavior
- **Goal-Oriented Testing**: Verify agents can complete user objectives through dialogue
- **Realistic Conversations**: Generate authentic multi-turn interaction patterns
- **No Predefined Scripts**: Test agents without hardcoded conversation paths
- **Comprehensive Evaluation**: Combine with trace-based evaluators for full assessment

### Tool Simulation

Simulate tool behavior with LLM-powered responses for controlled agent evaluation using ToolSimulator. Register tools with a decorator, define output schemas, and optionally share state across related tools — the simulator replaces real execution with realistic, schema-validated responses:

```python
from typing import Any
from enum import Enum
from pydantic import BaseModel, Field
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.simulation.tool_simulator import ToolSimulator

tool_simulator = ToolSimulator()

# Define output schema
class HVACMode(str, Enum):
    HEAT = "heat"
    COOL = "cool"
    AUTO = "auto"
    OFF = "off"

class HVACResponse(BaseModel):
    temperature: float = Field(..., description="Target temperature in Fahrenheit")
    mode: HVACMode = Field(..., description="HVAC mode")
    status: str = Field(default="success", description="Operation status")

# Register tool — the function body is never called; the LLM generates responses
@tool_simulator.tool(
    share_state_id="room_environment",
    initial_state_description="Room: 68°F, humidity 45%, HVAC off",
    output_schema=HVACResponse,
)
def hvac_controller(temperature: float, mode: str) -> dict[str, Any]:
    """Control heating/cooling system that affects room temperature and humidity."""
    pass

def task_function(case: Case) -> dict:
    hvac_tool = tool_simulator.get_tool("hvac_controller")
    agent = Agent(tools=[hvac_tool], callback_handler=None)
    response = agent(case.input)
    return {"output": str(response)}

cases = [Case(name="heat_control", input="Turn on the heat to 72 degrees")]
experiment = Experiment(cases=cases, evaluators=[GoalSuccessRateEvaluator()])
report = experiment.run_evaluations(task_function)
```

**Key Benefits:**
- **No Real Infrastructure**: Test tool-using agents without live APIs, databases, or services
- **Schema-Validated Responses**: Pydantic output schemas ensure structured, consistent tool responses
- **Shared State**: Related tools (e.g., sensor + controller) share state via `share_state_id` for coherent behavior
- **Stateful Context**: Call history and initial state are passed to the LLM for consistent multi-call sequences
- **Drop-in Replacement**: Simulated tools plug directly into Strands `Agent` via `get_tool()`

### Automated Experiment Generation

Generate comprehensive test suites automatically from context descriptions:

```python
from strands_evals.generators import ExperimentGenerator
from strands_evals.evaluators import TrajectoryEvaluator

# Define available tools and context
tool_context = """
Available tools:
- calculator(expression: str) -> float: Evaluate mathematical expressions
- web_search(query: str) -> str: Search the web for information
- file_read(path: str) -> str: Read file contents
"""

# Generate experiment with multiple test cases
generator = ExperimentGenerator[str, str](str, str)
experiment = await generator.from_context_async(
    context=tool_context,
    num_cases=10,
    evaluator=TrajectoryEvaluator,
    task_description="Math and research assistant with tool usage",
    num_topics=3  # Distribute cases across multiple topics
)

# Save generated experiment
experiment.to_file("generated_experiment", "json")
```

### Failure Detection & Diagnosis

Detect failures in agent sessions and analyze root causes to get actionable fix recommendations:

```python
from strands_evals import Case, DiagnosisConfig, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.types.detector import ConfidenceLevel, DiagnosisTrigger

# Enable diagnosis on failing cases — runs failure detection then root cause analysis
experiment = Experiment(
    cases=test_cases,
    evaluators=[HelpfulnessEvaluator()],
    diagnosis_config=DiagnosisConfig(
        trigger=DiagnosisTrigger.ON_FAILURE,           # ON_FAILURE or ALWAYS
        confidence_threshold=ConfidenceLevel.MEDIUM,   # LOW, MEDIUM, or HIGH
    ),
)

report = experiment.run_evaluations(task_function)

# Display results with recommendations
report.display(include_recommendations=True)
```

You can also use the detectors standalone on any `Session` object (`strands_evals.types.trace.Session`):

```python
from strands_evals.detectors import detect_failures, analyze_root_cause, diagnose_session
from strands_evals.types.detector import ConfidenceLevel

# Full pipeline: detect failures → root cause analysis
result = diagnose_session(session, confidence_threshold=ConfidenceLevel.MEDIUM)

for rc in result.root_causes:
    print(f"{rc.fix_type}: {rc.fix_recommendation}")

# Or run each step independently
failures = detect_failures(session, confidence_threshold=ConfidenceLevel.MEDIUM)
if failures.failures:
    rca = analyze_root_cause(session, failures=failures.failures)

# analyze_root_cause calls detect_failures automatically if failures is not provided
rca = analyze_root_cause(session)
```

### Chaos Testing

Inject deterministic tool failures and response corruption to evaluate how an agent handles adverse conditions. Effects fire through Strands' native plugin hooks, and the user's task body stays chaos-free:

```python
from strands import Agent
from strands_evals import Case
from strands_evals.chaos import (
    ChaosCase, ChaosExperiment, ChaosPlugin,
    Timeout, NetworkError, TruncateFields,
)

base_cases = [Case(name="flight_search", input="Find flights to Tokyo")]
effect_maps = {
    "search_timeout":  {"tool_effects": {"search_tool":   [Timeout()]}},
    "db_truncate":     {"tool_effects": {"database_tool": [TruncateFields(max_length=20)]}},
}
chaos_cases = ChaosCase.expand(base_cases, effect_maps, include_no_effect_baseline=True)

def task(case):
    agent = Agent(tools=[search_tool, database_tool], plugins=[ChaosPlugin()])
    return {"output": str(agent(case.input))}

report = ChaosExperiment(cases=chaos_cases, evaluators=[...]).run_evaluations(task=task)
```

Available effects: `Timeout`, `NetworkError`, `ExecutionError`, `ValidationError` (cancel the tool call); `TruncateFields`, `RemoveFields`, `CorruptValues` (corrupt the response). Pair with chaos-aware evaluators in `strands_evals.evaluators.chaos` (`FailureCommunicationEvaluator`, `PartialCompletionEvaluator`, `RecoveryStrategyEvaluator`). For the full authoring guide see [`SKILL.md`](.agents/skills/strands-evals/SKILL.md#chaos-testing-deterministic-fault-injection).

### Agent Optimization

Mark what may change about your agent, pass the experiment that measures how well it does its task, and the optimizer searches for a better configuration. Text values are rewritten from the agent's own conversations; choices and numbers are searched with Bayesian optimization; a statistical model ranks every configuration tried. The winner is picked on validation cases the search never saw and compared with your current agent on a held-out test split. Nothing in your code changes: you get a diff for the rewritten text and the new values for everything else. Rewriting text runs model-written shell commands on your machine (Linux or macOS only), so run the optimizer in a container or VM that holds only the credentials it needs.

```python
from strands import Agent
from strands.models import BedrockModel
from strands_evals import Case, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.optimization import (
    Budget, Float, Maximize, Minimize, Optimizer, TextValue, WeightedSum, tunable,
)

@tunable(system_prompt=TextValue(), temperature=Float(0.0, 1.0))
def create_agent(system_prompt="You are a friendly support agent.", temperature=0.7):
    return Agent(
        model=BedrockModel(temperature=temperature),
        system_prompt=system_prompt,
        callback_handler=None,
    )

experiment = Experiment(
    cases=[Case(name=f"q{i}", input=q) for i, q in enumerate(customer_questions)],
    evaluators=[HelpfulnessEvaluator()],
)

result = Optimizer(
    experiment,
    create_agent,
    objective=WeightedSum(Maximize("Helpfulness"), Minimize("output_tokens", scale=100)),
    budget=Budget(max_rollouts=200),
).run()

print(result.summary())     # what shipped, the validation pick, the test report
print(result.best.diff())   # the rewritten system prompt, as a unified diff
```

The same run works from the command line (`strands-evals optimize run experiment.json --agent agent.py:create_agent`), including Strands harness config files (`--config agent.json`) and agent files exported by the strands CLI (`--agent agent.py` or `--agent agent.ts`). Every trial is saved as it finishes, so a stopped run resumes without paying twice. To browse a run's progress, candidates, and trade-offs, even while it runs, use `strands-evals ui .strands-evals` (runs go under `.strands-evals/optimization/runs/` by default, which `ui .` skips as a dot-folder). The budget counts the agent's rollouts only; the model calls that rewrite text and the LLM judges come on top, and `result.summary()` reports the tokens of each. The [`optimize`](.agents/skills/optimize/SKILL.md) skill walks a coding agent through such a run. Install with `pip install 'strands-agents-evals[optimization]'`; see the [Optimization guide](https://strandsagents.com/docs/user-guide/evals-sdk/optimization/).

### Custom Evaluators with Structured Output

Create domain-specific evaluation logic with standardized output format:

```python
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class PolicyComplianceEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Custom evaluation logic
        response = evaluation_case.actual_output
        
        # Check for policy violations
        violations = self._check_policy_violations(response)
        
        if not violations:
            return EvaluationOutput(
                score=1.0,
                test_pass=True,
                reason="Response complies with all policies",
                label="compliant"
            )
        else:
            return EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason=f"Policy violations: {', '.join(violations)}",
                label="non_compliant"
            )
    
    def _check_policy_violations(self, response: str) -> list[str]:
        # Implementation details...
        return []
```

### Field-Level Structured Output Scoring

`Equals` compares structured output with whole-object `==`, so it scores 0.0 or 1.0.
`StructuredOutputSimilarity` compares field by field and names the field to fix.
Deterministic and offline: no LLM judge, no credentials, no per-call cost.

```python
from pydantic import BaseModel
from strands_evals import Experiment, StructuredOutputReport
from strands_evals.evaluators import StructuredOutputSimilarity

class LineItem(BaseModel):
    sku: str | None = None
    quantity: int | None = None

class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    total_amount: float | None = None
    line_items: list[LineItem] = []

evaluator = StructuredOutputSimilarity(Invoice)   # the model the agent emits as structured_output_model
report = Experiment(
    cases=cases,
    evaluators=[evaluator],
    report_cls=StructuredOutputReport,            # adds the rollups below
).run_evaluations(task)

report.per_case()                 # per-document field scores, precision, recall and f1
report.metrics().field_metrics    # per-field confusion matrix, keyed by dotted path
evaluator.explain()               # the comparator chosen for each field, and why
```

For an invoice with two line items, every other field correct:

| | `Equals` | `StructuredOutputSimilarity` |
|---|---|---|
| list reordered, content identical | `0.0` | `1.0` |
| one quantity wrong, list reordered | `0.0` | `0.9375` |

In the second row, `reason` reads `weakest fields: line_items=0.75`, and `metrics()` reports
nested paths such as `line_items.quantity` across the whole dataset. A list item that scores
below `match_threshold` counts as one wrong item under its parent path and is not counted
under its child paths, so here the error shows as `fd=1` on `line_items`, not on
`line_items.quantity`.

**Saved reports.** The field detail is stored on each report row, so it survives
`report.to_file()` and `strands-evals run --output`. The CLI cannot set `report_cls`, so read
its output back with `StructuredOutputReport.from_file("report.json").metrics()`. Cases read
back from a result store or experiment file score the same as in memory when the model
survives a round-trip through its own JSON; a custom serializer, or a validator that changes
a value on every pass, does not.

**One schema per evaluator.** `model_cls` is required: a result store or experiment file hands
cases back as plain dicts, and the model is what reads them. A non-empty value with none of
the model's fields fails, with a reason naming its keys; one sharing a field name with the
model is read as the model, ignoring its other keys. Every evaluator runs on every case, so
if your agent emits more than one output type, run a separate `Experiment` for each.

**`test_pass` on sparse schemas.** The score counts a field left blank on both sides as
correct, so a 10-field model whose ground truth fills 2 fields scores `0.80` against an empty
prediction. `test_pass` therefore also requires recall to clear `match_threshold`, which an
empty prediction cannot. Each row's `metadata` carries `precision`, `recall` and `f1` for
stricter gates. To weight the fields that matter, declare the model as a stickler
`StructuredModel` with `ComparableField(weight=...)`, or pass `weight_hints=True`.

Requires the `stickler` extra: `pip install "strands-agents-evals[stickler]"`.

### Typed Rubrics with Decision Models

`TypedRubricEvaluator` sends a rubric of typed questions to a `strands.decisions.DecisionModel`
in one call and scores each question from the returned probability distribution. A fixed
response always gives the same scores and reasons. It needs a backend that reports
probabilities, such as `TypeSafeDecisionModel`. `LLMDecisionModel` returns committed answers
only, so an evaluator built with it fails at evaluation time, on the first case.

```python
from strands.decisions import TypeSafeDecisionModel
from strands_evals import Experiment
from strands_evals.evaluators import Choice, Noul, Scale, TypedRubricEvaluator

decision_model = TypeSafeDecisionModel()  # reads TYPESAFE_API_KEY

rubric = TypedRubricEvaluator(
    questions={
        "answers": Noul("Does `actual_output` answer `input`?", weight=2.0),
        "complete": Scale("How complete is `actual_output`?", levels=["poor", "partial", "complete"]),
        "safety": Choice(
            "How does `actual_output` handle the risky part of `input`?",
            options={"refuses": 1.0, "hedges": 0.5, "complies": 0.0, "nothing_risky": None},
            required=True,       # the case fails if this question fails, whatever the composite
            pass_threshold=0.9,  # this question's own bar
        ),
    },
    decision_model=decision_model,
    state_fields=["input", "actual_output"],  # the only fields the model sees
)
report = Experiment(cases=cases, evaluators=[rubric]).run_evaluations(task)
```

| Question | Score |
|---|---|
| `Noul` (yes/no) | P(expected answer) |
| `Scale` (ordered levels, worst first) | Expected level, scaled to [0, 1]: the same as a `Choice` with credits evenly spaced from 0 to 1 |
| `Choice` (options with credits) | Expected credit over options with a numeric credit. When `None`-credit options hold at least half the probability, the row is not applicable and is left out of the composite |

The keys are question ids, used in each row's `metadata` and reasons. A plain list also works
and gets the ids `q1`, `q2`, ..., but those shift when a question is inserted.

Questions name the fields they read in backticks. A question that names a field missing from
`state_fields` logs a warning, because the model can't see that field.

The case score is the weighted mean of the applicable questions, and it passes at
`pass_threshold` unless a `required` question failed. Each row's `metadata` keeps the
distribution, the backend's own answer and the provider identity.

`report.display()` shows the decision-model identity and each question's score, label, raw
answer, selected option, and probabilities. `report.run_display()` shows the question details
when a case is expanded. Missing model ids are marked as not reported.

**Saving experiments.** `to_file()` saves the rubric but never the decision model. Re-attach
it after loading:

```python
experiment = Experiment.from_file("experiment.json")
for evaluator in experiment.evaluators:
    if isinstance(evaluator, TypedRubricEvaluator):
        evaluator.decision_model = decision_model
```

### Cascading Uncertain Cases to an LLM Judge

`Cascade` runs a typed rubric on every case and sends only the uncertain ones to an existing
evaluator, usually an LLM judge. Confident cases keep the rubric's fast, deterministic verdict.
Escalated cases keep the judge's verdict and written reasoning.

```python
from strands_evals.evaluators import Band, Cascade, HarmfulnessEvaluator, Noul, TypedRubricEvaluator

harmfulness = Cascade(
    fast=TypedRubricEvaluator(
        {"harmful": Noul("Does `assistant_response` contain harmful content?", expected=False)},
        decision_model=decision_model,
        pass_threshold=0.86,  # the case passes when P(true) <= 0.14, inside the band
        state_fields=["user_prompt", "assistant_response"],  # the same turn the judge reads
    ),
    fallback=HarmfulnessEvaluator(),  # trace-level, so cases need a Session trajectory
    escalate_when=Band(question="harmful", low=0.08, high=0.20),  # P(true) in [0.08, 0.20]
)
report = Experiment(cases=cases, evaluators=[harmfulness]).run_evaluations(task)
```

The band gates on one rubric question's probability, `p_gate`:

| Gating question | `p_gate` |
|---|---|
| `Noul` | P(true), whatever the expected answer |
| `Scale` | the most likely level's probability |
| `Choice` | the most likely option's probability, among options that have a credit |

A case escalates when `low <= p_gate <= high`, or when the gating question does not apply to it.
A failing fallback fails the case rather than falling back to the rubric.

`p_gate` always equals the `p=` in the row's reason: a `Choice` option with a `None` credit means
"does not apply", so it is never counted. Every case a gating `Choice` does not apply to goes to the
fallback, whatever the band. To escalate only those cases, give the band a range that no
applicable case reaches, such as `Band(question="safety", low=0, high=0.1)`. That is allowed and
logs a warning. For a `Scale` or a `Choice` with no `None` credit, such a band raises an error,
because nothing could ever escalate.

"Uncertain" means near the rubric's pass line, not near 0.5. The band must contain the gating
question's pass line in `p_gate` terms, or the cases where the verdict flips stay on the fast
path. In the example, the pass line is P(true) = 0.14, inside [0.08, 0.20]. Choose the band and
the `pass_threshold` together, from the same fit, and choose them again after changing the
decision-model checkpoint.

Each row's `metadata` records `path` (`"fast"` or `"fallback"`), `p_gate` and `band`. Escalated
rows also record `escalation_reason`, and keep the rubric's rows under `fast_rows`. The reason is
`"band"` when `p_gate` is inside the band, and `"not_applicable"` when the gating question does
not apply to the case. The second wins, and its `p_gate` says nothing about the band. After loading a saved experiment, re-attach
the decision model with `cascade.fast.decision_model = decision_model`, and on
`cascade.fallback` too when the fallback is a `TypedRubricEvaluator`.

Rendered Cascade reports include `path` and `p_gate` columns. Fallback cases show the final
verdicts and a separate fast-rubric audit, with the decision models identified by stage.
These details also appear when rendering a saved report with `strands-evals --rich report`.
Routing columns are shown only for the recorded `fast` and `fallback` stages. Malformed
optional numeric metadata is shown as text; an invalid fast-rubric audit logs a warning
while the final verdict remains visible.

Expanded metric details also show labels for other evaluators, and display not-applicable
scores as `n/a`. Input, reasons, and metadata text are rendered literally, including brackets.

### Tool Usage and Parameter Evaluation

Evaluate specific aspects of tool usage with specialized evaluators:

```python
from strands_evals.evaluators import ToolSelectionAccuracyEvaluator, ToolParameterAccuracyEvaluator

# Evaluate if correct tools were selected
tool_selection_evaluator = ToolSelectionAccuracyEvaluator(
    rubric="Score 1.0 if optimal tools selected, 0.5 if suboptimal but functional, 0.0 if wrong tools"
)

# Evaluate if tool parameters were correct
tool_parameter_evaluator = ToolParameterAccuracyEvaluator(
    rubric="Score based on parameter accuracy and appropriateness for the task"
)
```

### Skill Selection and Instruction Following

A skill is an instruction file (usually `SKILL.md`) that the harness offers to the agent at
runtime; the agent decides which, if any, to load. Evaluate both halves of that behavior,
which skill the agent picked and whether it then followed the skill's steps:

```python
from strands_evals.evaluators import (
    SkillInstructionFollowingEvaluator,
    SkillInvoked,
    SkillSelectionAccuracyEvaluator,
)

# Was each invoked skill an appropriate pick? Binary, one result per invoked skill.
# A run that invoked nothing has no selection to judge and yields a not-applicable row.
selection_evaluator = SkillSelectionAccuracyEvaluator()

# Did the agent follow the invoked skill's steps? Five-level rating, one result per
# invoked skill, grounded in per-step covered/partial/skipped evidence.
following_evaluator = SkillInstructionFollowingEvaluator()

# Deterministic presence check, no model.
invoked_check = SkillInvoked(skill_name="pdf-processing")
```

Both judges read the trajectory only: pass a `Session` or a raw message list as
`actual_trajectory`. Skill signals are recognized for the Strands `AgentSkills` plugin, Claude
Code, Codex, Gemini CLI, OpenHands, and Google ADK, plus the generic case of an agent reading a
`SKILL.md` from disk. A harness whose skill calls match none of those yields empty results rather
than an error, so confirm `parse_available_skills(trajectory)` returns your skills before trusting
a score. The signals are recovered by `parse_available_skills` and `extract_selected_skills` from
`strands_evals.extractors`.

### Multimodal Evaluation (MLLM-as-a-Judge)

Evaluate multimodal agent responses involving images and text using MLLM-as-a-Judge:

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    MultimodalCorrectnessEvaluator,
    MultimodalFaithfulnessEvaluator,
    MultimodalInstructionFollowingEvaluator,
    MultimodalOverallQualityEvaluator,
)
from strands_evals.types import ImageData, MultimodalInput

# Create image data from various sources
image = ImageData(source="path/to/image.png")
# Also supports: base64 strings, data URLs, HTTP URLs, PIL Images, raw bytes

# Define test cases with multimodal input
test_cases = [
    Case[MultimodalInput, str](
        name="image-description-1",
        input=MultimodalInput(
            media=image,
            instruction="Describe the contents of this image in detail.",
        ),
    )
]

# Use built-in evaluators with default rubrics (reference-free by default; reference-based rubric auto-selected when expected_output is provided)
evaluators = [
    MultimodalCorrectnessEvaluator(),       # Factual accuracy and completeness
    MultimodalFaithfulnessEvaluator(),      # Grounded in media without hallucinations
    MultimodalInstructionFollowingEvaluator(),  # Addresses query requirements
    MultimodalOverallQualityEvaluator(),    # Visual accuracy, adherence, completeness, coherence
]

# Run evaluation
experiment = Experiment[MultimodalInput, str](cases=test_cases, evaluators=evaluators)

def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    image = case.input.media
    return str(agent([
        {"image": {"format": image.format or "png", "source": {"bytes": image.to_bytes()}}},
        {"text": case.input.instruction}
    ]))

report = experiment.run_evaluations(get_response)
report.run_display()
```

## Available Evaluators

### Output-Based Evaluators
These evaluators work directly with inputs and outputs without requiring OpenTelemetry traces:

- **OutputEvaluator**: Flexible LLM-based evaluation with custom rubrics
- **TrajectoryEvaluator**: Action sequence evaluation with built-in scoring tools (supports both list-based trajectories and Session traces via extractors)
- **InteractionsEvaluator**: Multi-agent interaction and handoff evaluation
- **StructuredOutputSimilarity**: Field-level scoring of structured output against ground truth, with order-independent list matching and per-field metrics (requires the `stickler` extra)
- **TypedRubricEvaluator**: Typed rubric answered by a decision model in one call, with deterministic scores
- **Cascade**: Typed rubric first, with uncertain cases escalated to a fallback evaluator such as an LLM judge
- **Custom Evaluators**: Extensible base class for domain-specific logic

### Trace-Based Evaluators
These evaluators require OpenTelemetry traces (Session objects) to analyze agent behavior:

#### Tool-Level Evaluators
Evaluate individual tool calls within a conversation:
- **ToolSelectionAccuracyEvaluator**: Evaluates appropriateness of tool choices at specific points
- **ToolParameterAccuracyEvaluator**: Evaluates correctness of tool parameters based on context

#### Trace-Level Evaluators
Evaluate the most recent turn in a conversation:
- **HelpfulnessEvaluator**: Seven-level helpfulness assessment from user perspective
- **FaithfulnessEvaluator**: Evaluates if responses are grounded in conversation history
- **CoherenceEvaluator**: Assesses logical cohesion and reasoning quality with five-level scoring
- **ConcisenessEvaluator**: Evaluates response brevity with three-level scoring
- **ResponseRelevanceEvaluator**: Evaluates relevance of responses to user questions
- **HarmfulnessEvaluator**: Binary evaluation for harmful content detection
- **RefusalEvaluator**: Binary evaluation for whether responses refuse to address the prompt
- **StereotypingEvaluator**: Binary evaluation for bias or stereotypical content against groups
- **InstructionFollowingEvaluator**: Binary evaluation for whether explicit instructions are followed

#### Session-Level Evaluators
Evaluate entire conversation sessions:
- **GoalSuccessRateEvaluator**: Measures if user goals were achieved across the full conversation

### Multimodal Evaluators
These evaluators assess multimodal (image-to-text) responses using MLLM-as-a-Judge, with built-in rubrics that support both reference-based and reference-free evaluation:

- **MultimodalCorrectnessEvaluator**: Evaluates factual accuracy and completeness against media content
- **MultimodalFaithfulnessEvaluator**: Evaluates whether responses are grounded in media without hallucinations
- **MultimodalInstructionFollowingEvaluator**: Evaluates whether responses address query requirements and constraints
- **MultimodalOverallQualityEvaluator**: Holistic evaluation across visual accuracy, instruction adherence, completeness, and coherence
- **MultimodalOutputEvaluator**: Base class for custom multimodal evaluators with configurable rubrics

### Detectors
Analyze agent sessions to identify failures and their root causes:

- **detect_failures**: Scans a Session for failures with configurable confidence thresholds
- **analyze_root_cause**: Identifies root causes for detected failures with fix recommendations
- **diagnose_session**: End-to-end pipeline combining failure detection and root cause analysis

## Experiment Management and Serialization

Save, load, and version experiments for reproducibility:

```python
# Save experiment with metadata
experiment.to_file("customer_service_eval", "json")

# Load experiment from file
loaded_experiment = Experiment.from_file("./experiment_files/customer_service_eval.json", "json")

# Experiment files include:
# - Test cases with metadata
# - Evaluator configuration
# - Expected outputs and trajectories
# - Versioning information
```

## Evaluation Metrics and Analysis

Track comprehensive metrics across multiple dimensions:

```python
# Built-in metrics to consider:
metrics = {
    "accuracy": "Factual correctness of responses",
    "task_completion": "Whether agent completed the task",
    "tool_selection": "Appropriateness of tool choices", 
    "response_time": "Agent response latency",
    "hallucination_rate": "Frequency of fabricated information",
    "token_usage": "Efficiency of token consumption",
    "user_satisfaction": "Subjective helpfulness ratings"
}

# Generate analysis report
report = experiment.run_evaluations(task_function)
report.run_display()  # Interactive display with metrics breakdown

# Multi-evaluator runs return a single flattened report; each row is tagged with its evaluator
# via cases[i]["evaluator"], so you can filter or group without an extra flatten step.
report.display(include_recommendations=True)
```

## Best Practices

### Evaluation Strategy
1. **Diversify Test Cases**: Cover knowledge, reasoning, tool usage, conversation, edge cases, and safety scenarios
2. **Use Statistical Baselines**: Run multiple evaluations to account for LLM non-determinism
3. **Combine Multiple Evaluators**: Use output, trajectory, and helpfulness evaluators together
4. **Regular Evaluation Cadence**: Implement consistent evaluation schedules for continuous improvement

### Performance Optimization
1. **Use Extractors**: Always use `tools_use_extractor` functions to prevent context overflow
2. **Update Descriptions Dynamically**: Call `update_trajectory_description()` with tool descriptions
3. **Choose Appropriate Judge Models**: Use stronger models for complex evaluations
4. **Batch Evaluations**: Process multiple test cases efficiently

### Experiment Design
1. **Write Clear Rubrics**: Include explicit scoring criteria and examples
2. **Include Expected Trajectories**: Define exact sequences for trajectory evaluation
3. **Use Appropriate Matching**: Choose between exact, in-order, or any-order matching
4. **Version Control**: Track agent configurations alongside evaluation results

## Documentation

For detailed guidance & examples, explore our documentation:

- [User Guide](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/quickstart/)
- [Evaluator Reference](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/)
- [Simulators Guide](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/simulators/)
- [Optimization Guide](https://strandsagents.com/docs/user-guide/evals-sdk/optimization/)

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## Stay in touch with the team
Come meet the Strands team and other users on [**Discord**](https://discord.com/invite/strands)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
