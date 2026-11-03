# Project development with Strands Evals

`strands-evals run --project` evaluates a source-defined agent against captured user conversations. `strands-evals optimize` searches for changes to the keys that you declare. It follows the planned optimization API (#442). It writes a diff for review and edits source only when you run `--apply`.

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

`--runner` names an executable, not a shell command with arguments. Use an executable wrapper script if arguments are needed. It is supported for Python too.

Each request launches a fresh process with the entrypoint directory as its working directory and the inherited environment. The runner receives one newline-terminated JSON object on stdin. Normal execution has no `action` field:

```json
{"project":"/absolute/path/to/agent.ts","case_id":"capital-followup","user_turns":["Let's talk about France.","What is its capital?"]}
```

The runner must execute those turns in order and return an object with a string `output` and, optionally, an array `trajectory`:

```json
{"output":"Paris is the capital of France.","trajectory":[{"role":"assistant","content":"Paris is the capital of France."}]}
```

Return a zero process exit code on success. Write diagnostics to stderr and a single JSON response line to stdout. The implementation parses the **last stdout line**, but do not rely on extra stdout chatter. Invalid JSON, non-object responses, non-string output, non-array trajectory, nonzero exit codes, and the 300-second subprocess timeout fail execution. There is no corresponding built-in timeout for native Python calls.

When an optimization space has an `instructions` key with a file, the runner must also support inspection:

```json
{"project":"/absolute/path/to/agent.ts","action":"inspect"}
```

Respond with the actual nonempty runtime instructions:

```json
{"instructions":"You are a helpful geography assistant.\n"}
```

Optimization requests also carry `"overrides": {"<key>": <value>}`, with the value of every declared key. The runner must apply each value at runtime, without a change to source. A runner that ignores a key makes that key useless to the search. Inspection and each case run use separate processes. Do not keep state between them.

`--collector-endpoint URL` accepts an absolute HTTP(S) URL and sets the OTLP endpoint, traces endpoint (`URL/v1/traces`), exporter (`otlp`), and protocol (`http/protobuf`) in the execution environment. It does not start a collector or guarantee that an agent has configured telemetry export.

### Reports, failures, and project-mode flags

`run` emits an `EvaluationReport` JSON to stdout, or to `--output PATH`, and a summary to stderr. `--display` and `--interactive` use the existing report display. Reports contain scores, pass flags, reasons, and case records; they are not a separate replayable capture archive.

Project mode requires `--cases` and cannot be combined with an experiment positional argument, `--input`, `--input-file`, `--agent`, or `--task`. Do not assume every flag shown in the shared `run --help` applies to projects: project mode does not use `--expected-output`, `--name`, `--trace-attributes`, `--data-store`, or diagnosis configuration to execute/evaluate its cases, and is always serial.

The default `--fail-on any` exits 1 if any evaluation fails. `--fail-on threshold:0.8` exits 1 when the overall score is below 0.8; `--fail-on none` or `--exit-zero` suppress score-based failure, not execution/input errors. Invalid project inputs and runner protocol errors return 2; other project execution exceptions return 1. Source execution happens before grading, so a failed rollout is not silently turned into a successful score. Judge failures may be represented as failed evaluation rows; inspect report reasons rather than only the average score.

## Optimize an agent

`strands-evals optimize` follows the planned optimization API (#442) with an interim engine. The engine uses the planned records and run folder. It has no score model yet, so it ranks candidates by measured means.

### Declare what can change

Write a space file. Each key names one marker from the plan. The CLI supports these markers:

| Marker | Fields | Proposer |
| --- | --- | --- |
| `FileText` | `path`, `size_cap`, `edit_instructions` | Reflection |
| `TextValue` | `default`, `size_cap`, `edit_instructions` | Reflection |
| `Int` | `min_value`, `max_value`, `default`, `log` | Random |
| `Float` | `min_value`, `max_value`, `default`, `log` | Random |
| `OneOf` | `options`, `default` | Random |

```json
{
  "instructions": {"marker": "FileText", "path": "instructions.md"},
  "temperature": {"marker": "Float", "min_value": 0.0, "max_value": 1.0, "default": 0.7},
  "model": {"marker": "OneOf", "options": ["model-a", "model-b"]}
}
```

- A `FileText` path is relative to the project root. The file must be inside the project. Its text is the baseline.
- `--instructions-file FILE` is a shortcut for one `FileText` key named `instructions`.
- `JsonField`, `SkillLibrary`, `choice`, and `component` are planned. The CLI rejects them at this time.
- The `text` block uses the Reflection proposer. The `numbers` block uses `Random`, a stand-in for the planned `TPE` search.

The agent must accept every key at runtime, for each candidate and for the baseline:

- A Python project exports `create_agent(**keys)`. Give each parameter a default, so that the instructions check below can call it without arguments.
- A Python project that calls `strands_harness.create_harness(...)` gets the keys as harness options.
- An external runner gets `"overrides"` in each request. See the runner protocol above.

When the space has an `instructions` key with a file, the CLI compares that file with the instructions that the agent uses at runtime. If the two are different, the run stops before it spends.

### Plan before you spend

```bash
strands-evals optimize ./my-agent --cases ./captures --space ./space.json \
  --proposal-model MODEL_ID --dry-run
```

`--dry-run` reads the space and the cases, then prints a JSON plan. It does not import source, call models, or make a run folder. The plan gives the split, the objective, and these rollout counts:

- `baseline`: the rollouts that score the baseline on train.
- `validation_reserve`: `(finalists + 1) x validation cases x runs`.
- `test_reserve`: `2 x test cases x runs`.
- `search`: what the search can spend. `max_candidates` is `search / per_candidate`.

One rollout runs one case one time. `--max-rollouts` (default 500) does not count judge calls or proposer calls. `monetary_estimate` is `null` because the token use is not known.

### Run

```bash
strands-evals optimize ./my-agent --cases ./captures --space ./space.json \
  --proposal-model MODEL_ID --max-rollouts 200
```

The run follows the planned loop:

1. Split the cases into train, validation, and test. `--split` sets the shares (default `0.4,0.3,0.3`). `--seed` sets the shuffle.
2. Score the baseline on train.
3. Select the best train candidate as the parent. Each block suggests patches from that parent (the planned `Joint` schedule). A block asks again, up to 3 times, after a duplicate, an invalid patch, or an empty reply. It asks only for replacements, and a reply with fewer patches than asked for is valid. A failed proposer call runs again up to 3 times. After that, an infrastructure error stops the search with `proposer_error`, and another error counts as an empty reply.
4. Score each new candidate on train.
5. Stop on `budget`, `iterations` (`--max-iterations`), `exhausted`, or `proposer_error`.
6. Measure the top `--finalists` (default 3) and the baseline on validation. The highest validation score ships. A tie keeps the baseline.
7. Compare the winner with the baseline on test. The test result is for the report only. It does not change what ships.

Without a score model, a candidate can win on validation by chance. Read the test report before you apply a change. `--runs-per-candidate N` measures each case N times, which helps with noisy judges.

The default judge is the assertion judge of `run --project`. `--evaluator contains` and `--contains TEXT` select substring checks. `--objective maximize:METRIC` or `minimize:METRIC` selects what to optimize. Repeat the flag for a weighted sum, with `:weight=N` and `:scale=N`. The default maximizes every evaluator. `seconds` is the only usage metric in project mode, because project mode does not capture chat spans yet.

An agent error counts as the worst score. `--crash-score none` counts it as missing, and a finite number sets the score. A throttle, network, or `InfrastructureError` failure runs again, up to `--rerun-passes` times (default 2). Re-runs use only the rollouts that the reserves leave. For an external runner, the CLI finds these failures in the error text of the runner. A failed judge runs again on the saved output, with no rollout. If the baseline cannot be measured on train, the run stops before the search and exits 3.

`--proposal-file FILE` replaces the proposers for an offline run. The file is a JSON list of patches, `{"updates": {"<key>": <value>}, "rationale": "..."}`. A plain-text file, or a list of strings, gives texts for a space with one text key.

### Read the results

Each run writes `<project>/.strands-evals/optimization/runs/<run-id>/`. `--base-dir` sets another parent folder. `--name` sets a prefix for the run ID.

| File | Contents |
| --- | --- |
| `run.json` | The run configuration. Resume compares it. |
| `events.jsonl` | One event per line, in the planned event names. Read it for live progress. |
| `transcripts/` | Each trajectory one time, by content hash. |
| `effects.json` | Empty until the score model lands. |
| `best.diff` | A unified diff of the changed texts, for review. |
| `best.values.json` | The changed numbers and choices, as `{"<key>": {"old": ..., "new": ...}}`. |
| `result.json` | The result. stdout gets the same JSON. |

The result has the planned `OptimizationResult` fields: `improved`, `best`, `baseline`, `validation`, `test`, `stop_reason`, `iterations`, `rollouts_spent`, and the costs. It also has `diff` and `changed_values`. stderr gets the run folder first, then one line for each iteration.

The exit code is 0 when the run completes, also when nothing improves. It is 2 for bad input, and 3 for a runtime failure or a setup stop.

### Resume a killed run

```bash
strands-evals optimize ./my-agent --cases ./captures --space ./space.json \
  --proposal-model MODEL_ID --resume ./my-agent/.strands-evals/optimization/runs/RUN_ID
```

Give the same project, runner, cases, space, objective, judge, and proposer flags. If one changed, resume stops and names it. The event log gets a `resume` event. Resume reads the split, the seed, the finalists, and the other settings from `run.json`, so do not give them again. A killed run can get a larger `--max-rollouts`, but not a smaller one. It can also get a new `--max-iterations`. A later resume keeps the last budget that you gave.

Resume uses every saved trial again. It discards a killed iteration and runs that iteration again. A finished run does not continue: resume returns its result again, with no new rollouts. To search more, start a new run.

### Apply the winner

```bash
strands-evals optimize --apply ./my-agent/.strands-evals/optimization/runs/RUN_ID --dry-run
strands-evals optimize --apply ./my-agent/.strands-evals/optimization/runs/RUN_ID
```

`--apply` writes the `FileText` values of the winner into their files. Each file must still have the text that the run started with, else the command stops. Numbers, choices, and `TextValue` texts stay unchanged: the output lists them under `manual`. `--candidate ID` applies another candidate. With `--dry-run`, the command lists the files and writes nothing.

Use `strands-evals run --help` and `strands-evals optimize --help` for the current parser options. Common verbosity/format flags are accepted, but project reports and optimization results remain JSON unless a project report display is explicitly requested.
