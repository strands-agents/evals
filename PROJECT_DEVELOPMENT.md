# Project development with Strands Evals

`strands-evals run --project` evaluates a source-defined agent against captured user conversations. `strands-evals optimize` is a bounded, instructions-only proof of concept: it evaluates replacement instructions and writes a patch for manual review. It never applies the patch or edits project source.

These commands execute **trusted project code**, not sandboxed configuration. Imports, factories, tools, and external runners can access your files, credentials, network, and other resources. Review the source and runner before using them. Agent rollouts, judges, and proposal generation can incur provider costs.

## Generate review-only draft cases

```sh
strands-evals generate --project ./agent.py --num-cases 12 --model MODEL_ID > drafts.json
```

Project mode inspects the agent's instructions and tool schemas and delegates generation
to `ExperimentGenerator`. Python projects load directly; TypeScript projects use the same
`--runner` executable as project evaluation. Source inspection executes trusted project code
but does not invoke the agent. The model is explicit and generation may incur provider costs.

The result is `{ "schemaVersion": 1, "drafts": [...] }`, with each draft containing an `id`,
`assertion`, `userTurns`, and `metadata.reviewed: false`. No dataset files are created.
Review and approve each draft before converting it into a runnable capture case. The
`strands dev` UI provides a Keep/Edit/Skip review flow. Project mode accepts 1–50 cases and
rejects missing assertions, duplicate prompts, and an incorrect case count. `--output` is
not accepted in project mode: use stdout redirection for a draft artifact.

## Evaluate a project

```bash
# Default: an LLM judge checks each case's assertion against the final answer.
strands-evals run --project ./my-agent/agent.py --cases ./captures \
  --output ./report.json

# Deterministic alternative: each assertion is a literal expected substring.
strands-evals run --project ./my-agent/agent.py --cases ./captures \
  --evaluator contains --output ./contains-report.json

# Deterministic alternative: use the same substring for every case.
strands-evals run --project ./my-agent/agent.py --cases ./captures \
  --contains "Paris" --output ./paris-report.json
```

The default `OutputEvaluator` judges whether the **final output** satisfies the assertion, using the complete user conversation as context; it does not require verbatim wording. This is not a deterministic string comparison. Configure the credentials and model environment required by your agent and the SDK's default judge before running it.

`--evaluator contains` instead checks for the case's assertion as a case-sensitive substring and returns 1 or 0. `--contains TEXT` selects a fixed case-sensitive substring without an LLM judge. With `--evaluator contains --contains TEXT`, the fixed text takes precedence. Deterministic judging does **not** make the agent itself offline or free.

`--evaluator` can be repeated for other built-in evaluator names or custom `MODULE:CLASS` evaluators. Explicit evaluators take precedence over `--rubric`; without evaluators or `--contains`, `--rubric TEXT` replaces the default judge rubric. Do not use an explanatory assertion such as “the answer should name Paris” as a literal substring unless that is actually the output you require.

### Capture case files

Point `--cases` at a nonempty directory containing **only `.json` files**, with no subdirectories. Files are loaded in filename order. Each file is one capture-v1-style object, not a serialized `Experiment`:

```json
{
  "id": "capital-followup",
  "assertion": "The final answer identifies Paris as the capital of France.",
  "userTurns": [
    {"content": "Let's talk about France."},
    {"content": "What is its capital?"}
  ],
  "trajectory": {"turns": [], "spans": []},
  "metadata": {"category": "geography"}
}
```

- `id` must be a unique, nonempty string across the directory.
- `assertion` must be a nonempty string. It supplies both the expected assertion and expected output for evaluation.
- `userTurns` must be a nonempty array of objects with nonempty string `content` fields.
- `trajectory` is optional; if present, it must be an object with `turns` and `spans` arrays.
- `metadata` is optional and must be an object.

Only user text is replayed. Captured assistant responses and spans are **not** replayed as agent history or used as the newly generated trajectory. A fresh agent processes the user turns in order, retaining its conversation within that case; the last response is the output graded. Captured trajectories are validated but not used to drive execution. There is no `strands-evals capture` subcommand; supply compatible artifacts from your capture workflow or author the JSON files directly.

### Python source contract

An explicit project file may end in `.py`, `.ts`, `.js`, `.mts`, or `.mjs`. A directory must contain exactly one of `agent.py`, `main.py`, `agent.ts`, `index.ts`, `agent.js`, or `index.js`; otherwise pass the file explicitly. Non-Python source requires an external runner.

For native Python execution, export a callable `agent`, or a callable `create_agent` factory returning one. If both exist, the factory takes precedence. The agent receives one text argument per user turn. Awaitable results are supported; each response is converted to a string. Keep interactive loops and CLI startup behind an `if __name__ == "__main__":` guard.

For example, an agent that also supports instruction optimization:

```python
from pathlib import Path

from strands import Agent


def create_agent(instructions=None):
    if instructions is None:
        instructions = Path(__file__).with_name("instructions.txt").read_text(encoding="utf-8")
    return Agent(system_prompt=instructions)
```

Execution uses the entrypoint's parent as the working directory and exposes project/package imports. Each case reloads the entrypoint and constructs a fresh agent. Rollouts and grading are serial; `--max-workers` does not change project execution. This is **in-process isolation**, not a fresh interpreter or sandbox: newly imported modules under the project directory are cleaned up, and the working directory, import path, and collector settings are restored, but existing module state and arbitrary side effects are not fully isolated. Use an external runner if you need a fresh process per case.

### External runner protocol

```bash
strands-evals run --project ./my-agent/agent.ts --cases ./captures \
  --runner /absolute/path/to/runner --contains "Paris"
```

`--runner` names an executable, not a shell command with arguments. Use an executable wrapper script if arguments are needed. It is supported for Python too. Use an absolute executable path for optimization: the optimizer fingerprints the runner by reading that path.

Each request launches a fresh process with the entrypoint directory as its working directory and the inherited environment. The runner receives one newline-terminated JSON object on stdin. Normal execution has no `action` field:

```json
{"project":"/absolute/path/to/agent.ts","case_id":"capital-followup","user_turns":["Let's talk about France.","What is its capital?"]}
```

The runner must execute those turns in order and return an object with a string `output` and, optionally, an array `trajectory`:

```json
{"output":"Paris is the capital of France.","trajectory":[{"role":"assistant","content":"Paris is the capital of France."}]}
```

Return a zero process exit code on success. Write diagnostics to stderr and a single JSON response line to stdout. The implementation parses the **last stdout line**, but do not rely on extra stdout chatter. Invalid JSON, non-object responses, non-string output, non-array trajectory, nonzero exit codes, and the 300-second subprocess timeout fail execution. There is no corresponding built-in timeout for native Python calls.

For optimization, the runner must additionally support instruction inspection:

```json
{"project":"/absolute/path/to/agent.ts","action":"inspect"}
```

Respond with the actual nonempty runtime instructions:

```json
{"instructions":"You are a helpful geography assistant.\n"}
```

Optimization execution requests also carry `"overrides":{"instructions":"replacement instructions"}`. The runner must apply that value at runtime, without rewriting source. No other override keys are supported. Inspection and each case run use separate processes; persistence between them must not be assumed.

`--collector-endpoint URL` accepts an absolute HTTP(S) URL and sets the OTLP endpoint, traces endpoint (`URL/v1/traces`), exporter (`otlp`), and protocol (`http/protobuf`) in the execution environment. It does not start a collector or guarantee that an agent has configured telemetry export.

### Reports, failures, and project-mode flags

`run` emits an `EvaluationReport` JSON to stdout, or to `--output PATH`, and a summary to stderr. `--display` and `--interactive` use the existing report display. Reports contain scores, pass flags, reasons, and case records; they are not a separate replayable capture archive.

Project mode requires `--cases` and cannot be combined with an experiment positional argument, `--input`, `--input-file`, `--agent`, or `--task`. Do not assume every flag shown in the shared `run --help` applies to projects: project mode does not use `--expected-output`, `--name`, `--trace-attributes`, `--data-store`, or diagnosis configuration to execute/evaluate its cases, and is always serial.

The default `--fail-on any` exits 1 if any evaluation fails. `--fail-on threshold:0.8` exits 1 when the overall score is below 0.8; `--fail-on none` or `--exit-zero` suppress score-based failure, not execution/input errors. Invalid project inputs and runner protocol errors return 2; other project execution exceptions return 1. Source execution happens before grading, so a failed rollout is not silently turned into a successful score. Judge failures may be represented as failed evaluation rows; inspect report reasons rather than only the average score.

## Optimize instructions

Keep baseline instructions in an existing UTF-8 file **inside the project root**. The root is the directory passed as the project, or the parent of an explicit entrypoint. In a real run, the file's text must exactly match the extracted runtime instructions, including whitespace and trailing newlines. Extraction uses the Python agent's string `system_prompt` or captured `strands_harness.create_harness(..., instructions=...)` construction; external runners use the inspection response. It does not guess a prompt by searching source literals.

Native Python replacement instructions are passed as `create_agent(instructions=...)`, used to reconstruct a captured harness with changed instructions, or assigned to an exported agent's string `system_prompt`. Ensure your factory/runner actually honors the override. Only instructions are tuned: model selection, tools, source code, and other configuration are not optimization targets.

### Plan before spending

```bash
strands-evals optimize --project ./my-agent --cases ./captures \
  --instructions-file ./my-agent/instructions.txt \
  --budget 3 --holdout-size 5 --dry-run
```

Exactly one project is required, either `--project PATH` or positional `PATH`. `--dry-run` validates the input file/dataset, split sizes, and supplied proposal file, fingerprints source, and prints a JSON plan. It does not import project source, inspect runtime instructions, invoke the runner, call models, or create a run directory. It is **not** proof that the source, runtime match, runner, credentials, or resume manifest will work in a real run.

The plan includes a maximum of `budget + 3` evaluation batches and `(budget + 1) * training_cases + 2 * holdout_cases` rollouts. Proposal generation adds up to `budget` model requests; each rollout may make many agent/tool/model calls, and judges can add further calls. `monetary_estimate` is `null` because provider pricing and token usage are unknown. The budget bounds candidate count, **not tokens, time, or dollars**.

### Propose and evaluate

```bash
# Explicit proposal model; agent and default assertion judge may also call models.
strands-evals optimize ./my-agent --cases ./captures \
  --instructions-file ./my-agent/instructions.txt \
  --proposal-model YOUR_MODEL_ID --budget 3

# No proposal model or LLM judge: provide one candidate and a fixed substring gate.
# Agent/tool execution can still incur costs.
strands-evals optimize ./my-agent --cases ./captures \
  --instructions-file ./my-agent/instructions.txt \
  --proposal-file ./candidate.txt --budget 1 --contains "Paris"
```

A real run needs `--proposal-model` or `--proposal-file`; there is no implicit proposal model. A plain-text proposal file supplies one replacement, so use `--budget 1`. A `.json` proposal file supplies an array of nonempty instruction strings, with at least `--budget` entries. If both options are given, the file supplies the proposals. Generated proposals see current best instructions and training evaluation feedback, not holdout feedback. Default grading is the same assertion judge as project `run`; `--contains TEXT` substitutes the fixed deterministic gate. Optimize does not expose `run`'s custom evaluator/rubric flags.

The selection rules are deliberately strict:

1. Deterministically split cases into disjoint training and holdout sets. Holdout defaults to five cases and must contain at least five, plus at least one training case (at least six total). `--allow-small-holdout` permits smaller positive holdouts **for testing only**.
2. Evaluate baseline instructions on training cases. Evaluate each of up to `--budget` proposals (default three, minimum one) on training cases.
3. Replace the current best only on a strictly higher training score. Ties are rejected. Reaching a good score does not end the budget early.
4. If training produced an improvement, evaluate the original baseline and the best candidate on holdout. Accept only a strictly higher holdout score. A holdout tie rejects the candidate; no training improvement means no holdout evaluation or accepted patch.

These are average-score gates, not guarantees that every case improves or that the change is statistically significant. Errors, incomplete reports, non-finite scores, or evaluator/task error reasons fail optimization rather than count as valid improvements.

### Artifacts and resume limits

A real run stores `.strands/runs/<run-id>/manifest.json` inside the project root and returns JSON state on stdout. It records the baseline text, options, split indices, proposals, reports, fingerprint, and status. On acceptance it also writes `winner.instructions.txt` and `winner.patch`, a unified diff targeting the instructions file. On rejection, `accepted` is false and `patch_path` is null. Successful completion returns 0 even when no proposal is accepted; bad inputs/`ValueError`/filesystem errors return 2, and other runtime failures return 3.

Review the evidence and patch yourself. **No source file is automatically edited, and there is no automatic patch application.** Treat saved prompts, feedback, reports, and generated text as potentially sensitive and untrusted.

```bash
# Reuse the exact original invocation/options, adding the printed run ID.
strands-evals optimize ./my-agent --cases ./captures \
  --instructions-file ./my-agent/instructions.txt \
  --proposal-file ./candidate.txt --budget 1 --contains "Paris" \
  --resume RUN_ID
```

`--resume` accepts a run ID or a path to an existing directory directly under that project's `.strands/runs`. Resume checks manifest version and fingerprints of project source, baseline instructions, dataset, and options. Keep paths and all recorded options unchanged, including budget, holdout size, runner and proposal settings. Dataset file bytes are fingerprinted as well as normalized cases, so even JSON formatting changes can invalidate resume.

Project fingerprinting includes regular nonsymlink files under the root, excluding `.strands`, `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.pytest_cache`, `dist`, and `build`. Dependencies, symlink targets, environment/credentials, provider behavior, and files outside the root are not comprehensively fingerprinted. Put reports and scratch outputs outside the root or in `.strands` to avoid accidentally changing its fingerprint. Source is checked again before uncached evaluations; this is a change guard, not an immutable snapshot.

Persistence is deliberately limited:

- Manifest writes use a temporary file, flush/fsync, and atomic replacement. This is not a transactional database, multi-process lock, or guarantee against every crash scenario. Do not run concurrent writers against one run.
- Completed proposals and evaluations are reused. The CLI checkpoints completed individual cases as well as aggregate reports, so a completed case need not be rerun after interruption. It does not checkpoint inside a case, a model call, or a tool execution.
- An interruption after an external call but before its checkpoint can repeat that call and its cost/side effects. Filesystem/process state and provider/model state are not restored.
- Recorded proposal failures consume a budget slot. Recorded evaluation failures and proposal failures are re-raised on resume; resume is not an automatic retry mechanism. Start a new run after correcting a recorded failure.
- Winner text and patch are written separately from the final manifest, not atomically as a group. There is no separate durable call ledger, token/cost accounting, or general rollback. Reports preserve evaluation evidence, not a full independent trace/capture store.

Use `strands-evals run --help` and `strands-evals optimize --help` for the current parser options. Common verbosity/format flags are accepted, but project reports and optimization results remain JSON unless a project report display is explicitly requested.
