---
name: optimize
description: Use when the user wants to tune a strands-workspace agent (agent.py or agent.ts built with create_harness / createHarness) against one of its experiments, such as its instructions, model, effort, tools, or plugins, with `strands-evals optimize`. Sets up the run from a free-text request or step by step (experiment, knobs, objective, constraints, case split, budget), confirms a review summary, runs the agent through the installed Strands CLI's evals bridge with progress, and saves a report.
version: 0.2.0
---

# optimize

Sets up and runs `strands-evals optimize run` on the agent of a `strands-workspace`, the same way the Strands CLI runs it: the CLI's evals bridge loads the agent, so the knobs, the optimizer environment, and the run files match the CLI's.

The optimizer is experimental and isn't on evals `main` yet; the bridge installs the build the Strands CLI pins.

`<skill_dir>` below is the folder holding this file. `scripts/prepare.py` finds the CLI and the workspace, reads the knobs, and writes the run files.

## Steps

Ask every question in plain chat: one step at a time, numbered options with the recommended one first and marked `(Recommended)`, and always accept a free-form answer (ex. "2 steps", "drop effort", "tokens <= 4000"). Check each answer against the rules of its step before moving on; when it breaks one, say which rule and ask again. If you can't wait for a reply (ex. a non-interactive run), take each recommended value, say so, and continue.

### 1. Find the inputs and ask for a request

From the folder holding `strands-workspace` (or pass the workspace path last), run with Python 3.10 or later:

```bash
python3 <skill_dir>/scripts/prepare.py inspect [path/to/strands-workspace]
```

It prints JSON with:

- `cli`: the Strands CLI it found. It resolves `strands-dev`, then `strands`, on PATH to its package and needs the CLI's evals bridge (`dist/src/dev/evals-agent` and the TypeScript runner). When none has it, it exits with code 2: tell the user to install a Strands CLI that includes them, and stop.
- `agent`: the agent file, its root folder, and its language. It is `agent.entrypoint` of `strands-workspace/strands-dev/.config` when that file exists, else `strands-workspace/agent.py` or `strands-workspace/agent.ts`, which must call `create_harness(...)` or `createHarness(...)`. A Python agent needs `strands-harness` in the workspace's `.venv`.
- `experiments`: each `strands-workspace/strands-dev/experiments/<name>/<name>.json`, with its case count and evaluator names (each `evaluator_type` is a metric name).
- `knobs`: what step 2 can tune.

Then ask the first question:

Question: Describe the optimization you want, or answer "step by step".
1. Step by step (Recommended): I walk you through the experiment, the knobs, the goal, the cases, and the budget.
2. Describe it: for example, "make the support agent shorter without losing helpfulness, about 100 rollouts".

Next, unless the request names one, ask which experiment to use, listing each with its cases and evaluators, recommending the one with the most cases. With none, tell the user to add an experiment under `strands-workspace/strands-dev/experiments/` and stop.

If the user describes it, decide every value of steps 2 to 5 from that text and the files alone, using each step's recommendation rule where the text says nothing, ask no setup question, and go to step 6. A value the text asks for that breaks a rule of its step is changed to the closest valid one and flagged in the review.

### 2. Knobs: what may change

Offer the `knobs` from step 1, the Strands CLI's catalog read from the agent's `create_harness` / `createHarness` call. Each has a `key` (passed to `--tune`), a `label`, a `kind`, the `current` value, and `detail`:

- **text** (`instructions`, `systemPrompt`): rewritten by Reflection, the proposer model.
- **models** (`model`): a choice of the current model and the model ids the user names (`--model ID`, ex. `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`). With no other model id it tunes nothing.
- **choice** (`effort`, `caching`, `backgroundTasks`, `contextManager`, `skills`): one of `choices`, ex. effort `auto · low · medium · high · xhigh · max` for the agent's model.
- **switches** (`builtinTools`, `tools`, `mcpServers`, `builtinPlugins`, `plugins`, `interventions`, or a `skills` list): each entry in `entries` is kept or left out.

A knob with `unavailable` can't be tuned; show it with its reason (ex. set in code, or the system prompt replaces the instructions). `session` and `memory` are off in every trial.

**Recommend** the knobs with `on_by_default` (the instructions). When the goal or the request is about cost, length, or speed, also recommend effort, caching, and the context manager when the agent has them. Recommend switches only for entries the cases don't need. Recommend a model choice only with model ids the user named.

**Rules.** At least one knob must be tuned. When a text knob is tuned, warn: Reflection writes shell commands and runs them on this machine with the user's environment, credentials, and network. It is not a sandbox, and text in the cases can carry prompt injection. Run it in a container or VM that has only the credentials the run needs.

### 3. Goal: objective, constraints, crashes, and pick

**Metrics** are the experiment's evaluator names (`Helpfulness` resolves to `HelpfulnessEvaluator`) and the usage metrics `tokens`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `seconds`, and `model_calls`. A TypeScript agent runs outside the optimizer's process: it reports usage, but evaluators that read the trace (ex. `Helpfulness`, `TrajectoryEvaluator`, `ToolCalled`) are refused, so pick an experiment whose evaluators read only the answer (ex. `OutputEvaluator`).

**Objective.** Each goal is one `--objective {maximize,minimize}:METRIC[:weight=X][:scale=X]`. Several goals form a weighted sum; `--pareto` combines them as a Pareto front instead (needs 2 or more). With no `--objective`, the run maximizes the mean of every evaluator. The options:

1. The mean of every evaluator: no `--objective`.
2. One evaluator only (when there are several): `--objective maximize:<Evaluator>`.
3. Fewer tokens, each evaluator at least 0.8: `--objective minimize:tokens` with `--constraint '<Evaluator>>=0.8'` per evaluator.
4. Less time, each evaluator at least 0.8: `--objective minimize:seconds` with the same constraints.
5. The evaluators and fewer tokens: `--objective maximize:<Evaluator>` per evaluator and `--objective minimize:tokens:scale=1000`.
6. The evaluators and less time: the same with `--objective minimize:seconds:scale=10`.
7. The evaluators, fewer tokens, and less time: both usage goals.
8. The best trade-offs of the evaluators and tokens: option 5 with `--pareto`.

Recommend option 1 unless the request is about length or cost (option 5, or `output_tokens` when it's about answer length) or speed (option 6). `scale=X` is the size of a meaningful change: `scale=100` on `output_tokens` weighs 100 tokens like the whole 0-1 range of an evaluator.

Objective rules: every metric resolves to an evaluator of the experiment or a usage metric; a metric appears once; a usage metric in a weighted sum or a Pareto front needs `scale=`; `weight=` and `scale=` count only when there are several goals; a metric name with `:` can't go into a spec; an experiment with no evaluator needs a usage goal. When every goal is a usage metric and no constraint puts a floor on an evaluator, warn that a broken agent can look best, and recommend option 3's constraints.

**Constraints.** Each is `--constraint 'METRIC<=X'` or `--constraint 'METRIC>=X'` (ex. `tokens<=4000`, `Correctness>=0.8`); only a candidate meeting every constraint on validation ships. Recommend none, except the floors above. A metric name with `<`, `>`, or `=` can't go into a constraint. With a constraint, ask how sure each must be to hold, `--feasibility-threshold`: 0.8, 0.9 (default, recommended), 0.95, or 0.99.

**Crashes.** `--agent-failure-score`: `worst` (default, recommended) scores a crashed run at each evaluator's worst value; `none` counts it as no data; a number scores it that value.

**Pick.** How the champion is picked from the finalists on validation: the posterior mean (default, recommended), a 25% lower bound to ship cautiously, or a 10% lower bound to ship more cautiously. A lower bound is set up in step 7 with `--final-pick N:0.25` or `--final-pick N:0.1`, N from step 5, and replaces `--n-finalists`.

### 4. Cases: the split

Train cases guide the search, validation picks the winner, and test checks it on cases nothing else saw. Choose the train share from 0.3, 0.4, 0.5, 0.6, 0.7, and 0.8, then how the rest divides: half test (1 test case per validation case), a third test (1 test case per 2 validation cases), or no test. Pass `--split TRAIN,VALIDATION,TEST`, with validation `(1 - TRAIN) × (1 - test part)` and test `(1 - TRAIN) × test part`. The shares must sum to 1 within 1e-9, so write enough digits (ex. `0.6,0.266666666667,0.133333333333`).

Show each choice as case counts: each share times the case count, rounded down, with the cases left over going to the largest remainders. Recommend 0.5 train with half test (`0.5,0.25,0.25`), the Strands CLI's default, or the nearest choice that leaves at least 2 validation cases and, with test cases, at least 2 test cases.

Rules: each share is from 0 to 1 and train and validation are above 0. The optimizer refuses a split with fewer than 1 train or 1 validation case. Fewer than 2 validation or test cases: warn that the scores can be luck, and say how many cases to add to reach 2 of each. Cases without a unique `name` are named in a copy of the experiment, so the user's file doesn't change.

### 5. Budget

Ask each, recommending the default:

| Setting | Flag | Choices | Default |
| --- | --- | --- | --- |
| budget: steps | `--max-iterations N` | 0, 1, 2, 4, 6, 8 | 4 |
| per step: candidates | `--proposals-per-iteration N` | 2, 4, 6, 8 | 4 |
| runs of each case per candidate | `--runs-per-candidate N` | 1, 2, 3, 5 | 1 |
| finalists on validation | `--n-finalists N` | 1, 2, 3, 5 | 3 |
| parallel trials | `--max-workers N` | 1, 2, 4, 8, 16 | 4 |

0 steps measures only the baseline: `--max-iterations` must be at least 1, so pass no `--max-iterations` and no `--reflection-model`, and set it up in step 7 with `--baseline-only`. Under `--pareto`, at least 5 finalists go to validation.

Estimate the rollouts (one case run once) and show them with each answer:

```text
rollouts = runs × (train + steps × candidates × train + (finalists + 1) × validation + tested × test)
```

`finalists` is at least 5 under `--pareto`, and `tested` is 2 (the champion and the baseline), or `finalists + 1` under `--pareto`. Judge calls are about rollouts × evaluators, and Reflection's own model calls come on top; neither counts against the budget.

The run passes `--max-rollouts 500`, the Strands CLI's cap. When the estimate is above 500, lower the steps, candidates, or runs, or raise `--max-rollouts` if the user asks. `--max-rollouts` must cover the baseline and the end phase, `runs × (train + (finalists + 1) × validation + tested × test)`. The optimizer warns when one step's trials, `candidates × train × runs`, exceed the parallel trials: steps then run in batches and take longer.

### 6. Review

Show the setup as one summary, then the two commands of step 7, and ask the user to start or to name a row to change. Go back to that row's step, then show the review again.

```text
optimize · <experiment> · <cases> cases · <train> train / <validation> validation / <test> test
  agent        strands-workspace/agent.py
  tune         instructions, effort
  models       <the agent's model> [and N other models]
  goal         <the objective, or "the mean of every evaluator">
  constraints  <tokens <= 4000, 90% sure that each constraint holds, or "none">
  crashes      a crashed run scores the worst value
  pick         the posterior mean of 3 finalists
  cases        5 train / 3 validation / 2 test
  budget       4 steps × 4 candidates, 1 run of each case
  parallel     4 trials at once
  ~<rollouts> rollouts · ~<judge calls> judge calls · price unknown
```

Under the summary, list every warning from steps 2 to 5, and say that the run's model, judge, and proposer calls are paid and use the user's credentials, and that the first run installs the optimizer (about a minute, needs git and network).

**Setup command**, from the same folder as step 1:

```bash
python3 <skill_dir>/scripts/prepare.py setup [path/to/strands-workspace] --experiment <name> \
  --tune <knob key> [--tune ...] [--model <model id> ...] [--baseline-only] [--final-pick N:Q]
```

**Optimize flags**, appended to the `command` that setup prints:

```text
--split <TRAIN,VALIDATION,TEST> [--max-iterations <steps>] --proposals-per-iteration <candidates> \
--runs-per-candidate <runs> --max-workers <parallel> --max-rollouts 500 --seed 0 \
[--objective ... [--pareto]] [--constraint ...] \
[--feasibility-threshold X] [--agent-failure-score none|X] [--n-finalists N] [--reflection-model MODEL_ID]
```

- Pass `--feasibility-threshold`, `--agent-failure-score`, and `--n-finalists` only when they differ from 0.9, `worst`, and 3, and `--n-finalists` only without `--final-pick`.
- `--reflection-model MODEL_ID` changes the model rewriting text (default Bedrock `global.anthropic.claude-sonnet-5-5`); pass it only when the user asks and a text knob is tuned.

### 7. Run

**Setup.** Run the setup command. It prints progress to stderr and JSON to stdout:

- `python` (`OPT_PY` below): the optimizer environment. Setup reuses the Strands CLI's environment at `strands-workspace/strands-dev/tmp/.state/evals-env`, or creates it from the workspace's `.venv` Python (a TypeScript agent: `python3`, or `$PYTHON`) and installs the pinned optimizer once into a shared folder under `~/.strands/temporary-evals/`. A Python agent's packages are added after the optimizer's.
- `folder`: the run's folder, `strands-workspace/strands-dev/experiments/<name>/optimizations/<stamp>/`, where the Strands CLI keeps its own runs. The experiments folder's `.gitignore` already leaves out `optimizations/`, so nothing lands in git. Its `.state/` holds `space.json` (the tuned knobs as text, choice, and switch variables keyed by `create_harness` option, snake_case in Python) and `task.py`, which the CLI's own code writes and which loads the agent through the bridge.
- `variables`: the knobs that `space.json` tunes. Check they match the review.
- `command`: `cd` to the agent root, put the optimizer's `bin` first on `PATH`, and start `strands-evals optimize run` with `--agent`, `--rollout`, `--search` (with `--baseline-only`), `--final-pick` (with `--final-pick`), `--base-dir`, and `--rich`.
- `resume`: the same start for `strands-evals optimize resume`, for exit code 130.

If setup exits with code 2, show its last stderr lines. For a failed install, ask the user to check the network, the pip index settings, and that git is installed; for a missing `.venv` or `strands-harness`, ask them to install the agent's dependencies in `strands-workspace/.venv`.

**Start.** Run `command` plus the optimize flags in the background, with stdout, stderr, and the exit code in their own files in `folder`:

```bash
F=<folder>
(<command> <optimize flags> > "$F/run.out" 2> "$F/run.log"; echo $? > "$F/run.exit") &
```

If you can't run a background process, run it in the foreground with a long timeout and show the output when it ends. As soon as `run.log` prints `strands-evals optimize: run folder <RUN_DIR>`, tell the user the folder.

**Progress.** Until `$F/run.exit` exists, about once a minute, relay each new `strands-evals optimize: iteration N (<blocks>): P proposals, R rollouts` line from `run.log`, and show the run's status:

```bash
"$OPT_PY" <skill_dir>/scripts/progress.py <RUN_DIR>
```

It prints the phase (measuring the baseline, the search iteration, or the end phase), the elapsed time, the rollouts spent of the budget, the baseline and the best objective measured on train, and the latest candidates with their measured objective and the proposer's reason. The best measured on train isn't what ships: the end phase picks the champion on validation.

**Exit codes**, from `$F/run.exit`:

- **0**: the run finished. Go to step 8.
- **3 with `<RUN_DIR>/result.json`**: the search stopped on a proposer error, but the end phase ran and the result is valid. Show the error from `run.log`, then go to step 8.
- **3 without `result.json`**: the run stopped before it finished, most often at setup. Show the error line from `run.log`. When it says the baseline's train cases are unmeasured or every baseline trial failed, the agent or a model service failed (ex. `NoCredentialsError`, `Could not load credentials`): ask the user to check that the agent answers a case and that the credentials are fresh, then run setup and the command again.
- **2**: bad input (a flag, the agent, a trace-level evaluator with a TypeScript agent, or a space variable). Show the error, fix the value in its step, and show the review again.
- **130**, or the process is gone without writing `run.exit` (killed): continue with `<resume> <RUN_DIR> --rich`, started the same way. Finished trials aren't run again, and only `--max-rollouts`, `--max-iterations`, and `--max-workers` may change on resume.

### 8. Report

Write the report and print its path:

```bash
"$OPT_PY" <skill_dir>/scripts/report.py <RUN_DIR>
```

It writes `<RUN_DIR>/report.md`: what ships, the test check (the champion against the baseline on the test cases, with its 90% range and whether the gain is likely real), the changed values, the diff, the optimizer's summary, and the run's files. Give the user its path to open, and the headline: the candidate that ships, or that no candidate beat the current agent.

Nothing in the user's project changes. The champion's changes are keyed like `space.json`: `<RUN_DIR>/best.diff` holds each changed text (ex. `instructions`) as a diff under its key, and `<RUN_DIR>/best.values.json` holds each changed choice or switch as `{"old": ..., "new": ...}`, where a switch key such as `builtin_tools_0` is `false` when that entry is left out. Apply them to the agent's `create_harness` / `createHarness` call only when the user asks, and suggest evaluating the agent again after.
