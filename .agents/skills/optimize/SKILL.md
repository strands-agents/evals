---
name: optimize
description: Use when the user wants to tune a strands-workspace agent (agent.py or agent.ts built with create_harness / createHarness) against one of its experiments, such as its instructions, model, effort, tools, or plugins, with `strands-evals optimize`. Sets up the run from a free-text request or step by step (experiment, knobs, objective, constraints, case split, budget, crashes and pick), confirms a review summary, runs the agent through the installed Strands CLI's evals bridge with progress, and saves a report.
version: 0.5.0
---

# optimize

Sets up and runs `strands-evals optimize run` on the agent of a `strands-workspace`, the same way the Strands CLI runs it: the CLI's evals bridge loads the agent, so the knobs, the optimizer environment, and the run files match the CLI's.

The optimizer is experimental and isn't on evals `main` yet; the bridge installs the build the Strands CLI pins.

`<skill_dir>` below is the folder holding this file. `scripts/prepare.mjs` loads the installed Strands CLI's own modules to find the workspace's agent, read its knobs, prepare the optimizer environment, and write the run files; everything else is the optimizer's own output.

## Steps

Ask every question in plain chat, one step at a time, unless a step says to ask its questions together in one message. Give numbered options with the recommended one first and marked `(Recommended)`, and end every list with a short "Your own" option, so the free-form path is always visible; always accept a free-form answer (ex. "2 steps", "drop effort", "tokens at most 4000"). Check each answer against the rules of its step before moving on; when it breaks one, say which rule and ask again. If you can't wait for a reply (ex. a non-interactive run), take each recommended value, say so, and continue.

**What the user sees is plain language.** Options and their descriptions name what changes and what it costs (case counts, rollout estimates), never a flag, a knob key, or a file path. Flags appear only in each step's **Build** notes, which are for you, and in the final command of the review (step 8).

### 1. Find the inputs and ask for a request

From the folder holding `strands-workspace` (or pass the workspace path last), run with the `node` the Strands CLI uses:

```bash
node <skill_dir>/scripts/prepare.mjs inspect [path/to/strands-workspace]
```

It prints JSON with:

- `cli`: the Strands CLI it found. It resolves `strands-dev`, then `strands`, on PATH to its package and needs the CLI's evals bridge (`dist/src/dev/evals-agent` and the TypeScript runner). When none has it, it exits with code 2: tell the user to install a Strands CLI that includes them, and stop.
- `agent`: the agent file, its root folder, and its language. It is `agent.entrypoint` of `strands-workspace/strands-dev/.config` when that file exists, else `strands-workspace/agent.py` or `strands-workspace/agent.ts`, which must call `create_harness(...)` or `createHarness(...)`. A Python agent needs `strands-harness` in the workspace's `.venv`.
- `experiments`: each `strands-workspace/strands-dev/experiments/<name>/<name>.json`, with its case count and evaluator names (each `evaluator_type` is a metric name).
- `knobs`: what step 2 can tune.

Then ask the first question:

```text
Describe the optimization you want, or answer "step by step".
1. Step by step (Recommended): I walk you through the experiment, what may change, the goal, the cases, and the budget.
2. Describe it: for example, "make the support agent shorter without losing helpfulness, about 100 rollouts".
3. Your own: anything else, for example "just show me what can be tuned".
```

Next, unless the request names one, ask which experiment to use, listing each by name with its cases and evaluators (ex. `capitals: 6 cases, scored by OutputEvaluator`), recommending the one with the most cases, and end with "Your own: the path to another experiment file". With none, tell the user to add an experiment under `strands-workspace/strands-dev/experiments/` and stop.

If the user describes it, decide every value of steps 2 to 7 from that text and the files alone, using each step's recommendation rule where the text says nothing, ask no setup question, and go to step 8. A value the text asks for that breaks a rule of its step is changed to the closest valid one and flagged in the review.

### 2. Knobs: what may change

List **every** knob from step 1 that has no `unavailable`, by its `label`, with its current value and what the optimizer may try (`detail`). Mark each recommended knob `(Recommended)`; the user picks any number of them. After the list, name the untunable knobs in one line with their reason, only when one of them is something the user might expect to tune (ex. "system prompt: set in code").

```text
What may the optimizer change? Pick any (ex. "1, 3").
1. instructions (Recommended): rewritten by the proposer model; now "Answer geography questions in one short sentence."
2. model: try other models next to bedrock/global.anthropic.claude-sonnet-4-5…; name the models to try
3. effort: auto, low, medium, high, xhigh, or max; now auto
4. prompt caching: auto, on, or off; now auto
5. built-in tools: keep or drop each of read
6. background tasks: on or off; now off
7. context manager: auto, agentic, or off; now auto
8. skills: on or off; now on
9. built-in plugins: keep or drop each of todos, environment
10. Your own: for example, "only the instructions and effort low or medium"
```

The kinds behind the labels:

- **text** (instructions, system prompt): rewritten by Reflection, the proposer model.
- **models** (model): a choice of the current model and the model ids the user names. With no other model id it tunes nothing, so ask for the ids when the user picks it.
- **choice** (effort, prompt caching, background tasks, context manager, skills): one of `choices`.
- **switches** (built-in tools, tools, MCP servers, built-in plugins, plugins, interventions, or a skills list): each entry in `entries` is kept or left out.

Session and memory are off in every trial.

**Recommend** the knobs with `recommended: true` (the instructions). When the goal or the request is about cost, length, or speed, also recommend effort, prompt caching, and the context manager when the agent has them. Recommend switches only for entries the cases don't need. Recommend a model choice only with model ids the user named.

**Rules.** At least one knob must be tuned. When a text knob is tuned, warn: Reflection writes shell commands and runs them on this machine with the user's environment, credentials, and network. It is not a sandbox, and text in the cases can carry prompt injection. Run it in a container or VM that has only the credentials the run needs.

**Build.** Each picked knob is one `--tune <key>` of the setup command, and each model id to try is one `--model <id>`.

### 3. Objective: what to improve

**Metrics** are the experiment's evaluator names (`Helpfulness` resolves to `HelpfulnessEvaluator`) and the usage metrics tokens, input tokens, output tokens, cache read tokens, cache write tokens, seconds, and model calls. A TypeScript agent runs outside the optimizer's process: it reports usage, but evaluators that read the trace (ex. `Helpfulness`, `TrajectoryEvaluator`, `ToolCalled`) are refused, so pick an experiment whose evaluators read only the answer (ex. `OutputEvaluator`).

Ask what to improve. The options hold only objectives; limits come in step 4:

```text
What should the optimizer improve?
1. The mean score of every evaluator (Recommended)
2. One evaluator only: <Evaluator>
3. Fewer tokens
4. Less time
5. The evaluators and fewer tokens, weighed together
6. The evaluators and less time, weighed together
7. The evaluators, fewer tokens, and less time, weighed together
8. The best trade-offs between the evaluators and tokens, without weighing them
9. Your own: for example, "fewer output tokens, OutputEvaluator counting twice as much"
```

Offer option 2 once per evaluator, and only when there are several. Recommend option 1 unless the request is about length or cost (option 5, or option 5 with output tokens when it's about answer length) or speed (option 6). Offer "shorter answers" (output tokens) in place of tokens when the request is about answer length.

A "Your own" goal uses the metrics of step 4. Objective rules: every metric resolves to an evaluator of the experiment or a usage metric; a metric appears once; an experiment with no evaluator needs a usage goal (option 3 or 4); option 8 needs 2 goals or more; a metric name with `:` can't be used.

**Build.** Each goal is one `--objective {maximize,minimize}:METRIC[:weight=X][:scale=X]`; several goals form a weighted sum.

- Option 1: no `--objective`.
- Option 2: `--objective maximize:<Evaluator>`.
- Options 3 and 4: `--objective minimize:tokens` or `--objective minimize:seconds`.
- Options 5 to 7: `--objective maximize:<Evaluator>` per evaluator plus `--objective minimize:tokens:scale=1000` and/or `--objective minimize:seconds:scale=10`.
- Option 8: option 5 plus `--pareto`.
- Option 9: one `--objective` per goal the user names, plus `--pareto` when they ask for trade-offs without weighing.

A usage metric in a weighted sum or a Pareto front needs `scale=`, the size of a meaningful change: `scale=100` on `output_tokens` weighs 100 tokens like the whole 0-1 range of an evaluator. `weight=` and `scale=` count only when there are several goals.

### 4. Constraints: limits on top of the objective

Ask which constraints to add on top of the objective from step 3, with what can be constrained listed under the options. Name the experiment's own evaluators in the first line of that list. Only a candidate meeting every constraint on validation ships.

```text
Which constraints should the winner also meet?
1. None (Recommended)
2. Every evaluator at least 0.8: OutputEvaluator at least 0.8
3. Your own, for example "OutputEvaluator at least 0.9" or "tokens at most 4000"

A constraint is "at least" or "at most" a number, on the average per run of one case:
- OutputEvaluator: its score, from 0 (worst) to 1 (best)
- tokens: all tokens the agent's model used to answer one case (input plus output)
- input tokens, output tokens: the tokens it read (cached ones included) and the tokens it wrote
- cache read tokens, cache write tokens: input tokens read from or written to the prompt cache
- seconds: how long the agent took to answer one case
- model calls: how many times the agent called its model for one case
The judge's own calls don't count toward tokens, seconds, or model calls.
```

An evaluator with its own score range shows that range instead of 0 to 1. A TypeScript agent reports tokens, seconds, and model calls too, but its experiment can't use evaluators that read the agent's steps (ex. Helpfulness, TrajectoryEvaluator, ToolCalled); with one of those, say the experiment needs a Python agent or answer-only evaluators.

When the objective is usage only (step 3, option 3 or 4), recommend option 2 instead and say why: with nothing but tokens or time to improve, an agent that answers badly, or not at all, can look best, so a minimum score per evaluator keeps the answers good.

Rules: each constraint is one metric from the list above, "at least" or "at most", and a number; a metric name with `<`, `>`, or `=` can't be used.

When at least one constraint was added, ask how sure each must be to hold:

```text
How sure must the optimizer be that each constraint holds?
1. 90% (Recommended)
2. 80%
3. 95%
4. 99%
5. Your own: another level above 0% and below 100%, ex. 85%
```

**Build.** Each constraint is `--constraint 'METRIC>=X'` or `--constraint 'METRIC<=X'`; the confidence is `--feasibility-threshold X`, passed only when it isn't 0.9.

### 5. Cases: the split

Train cases guide the search, validation picks the winner, and test checks it on cases nothing else saw. Offer the train share from 30%, 40%, 50%, 60%, 70%, and 80%, then how the rest divides: half test, a third test, or no test; end each list with "Your own" (ex. "4 train, 1 validation, 1 test"). Show each choice as case counts (ex. `3 train / 2 validation / 1 test`): each share times the case count, rounded down, with the cases left over going to the largest remainders. Recommend half train with half of the rest for test, the Strands CLI's default, or the nearest choice that leaves at least 2 validation cases and, with test cases, at least 2 test cases.

Rules: train and validation get a share above 0. The optimizer refuses a split with fewer than 1 train or 1 validation case. Fewer than 2 validation or test cases: warn that the scores can be luck, and say how many cases to add to reach 2 of each. Cases without a unique `name` are named in a copy of the experiment, so the user's file doesn't change.

**Build.** `--split TRAIN,VALIDATION,TEST`, with validation `(1 - TRAIN) × (1 - test part)` and test `(1 - TRAIN) × test part`. The shares must sum to 1 within 1e-9, so write enough digits (ex. `0.6,0.266666666667,0.133333333333`).

### 6. Budget

Ask all five settings together in one message, each with one line on what it does, then what a rollout is and the estimate for the recommended set, worked out with the case counts from step 5. Take one reply for all of them; a setting the reply leaves out keeps its recommended value. Recompute the estimate from the reply and show it the same way before moving on; when a value breaks a rule, say which and ask only for that one again.

```text
Budget: answer any of these in one reply (ex. "2 steps, 2 runs"); the rest keep the recommended value.
1. Steps: rounds of proposing and testing new versions of the agent.
   0 (measure only the current agent), 1, 2, 4 (Recommended), 6, 8, or your own
2. Candidates per step: new versions tried in each round.
   2, 4 (Recommended), 6, 8, or your own
3. Runs of each case: how often each case is repeated per version, to average out noise.
   1 (Recommended), 2, 3, 5, or your own
4. Finalists: the best versions re-checked on the validation cases to pick the winner.
   1, 2, 3 (Recommended), 5, or your own
5. Trials at once: how many runs go in parallel; changes speed, not cost.
   1, 2, 4 (Recommended), 8, 16, or your own

A rollout is one run of the agent on one case; each is judged once per evaluator.
rollouts = runs × (current agent on train + steps × candidates × train
                   + (finalists + current agent) × validation + 2 × test)
         = 1 × (3 + 4 × 4 × 3 + (3 + 1) × 2 + 2 × 1) = 61 rollouts, ~61 judge calls
with 6 cases: 3 train / 2 validation / 1 test.
```

Under the best trade-offs objective (step 3, option 8), at least 5 finalists go to validation and test checks every finalist and the current agent, so the last term is `(finalists + 1) × test`; say so in the formula line. Judge calls are about rollouts × evaluators, and Reflection's own model calls come on top; neither counts against the budget.

Rules: the cap is 500 rollouts, the Strands CLI's. When the estimate is above 500, recommend fewer steps, candidates, or runs, or raise the cap if the user asks; the cap must cover the parts outside the steps, `runs × (train + (finalists + 1) × validation + tested × test)`. When one step's trials, `candidates × train × runs`, exceed the trials at once, say that steps then run in batches and take longer. 0 steps measures only the baseline and needs no proposer model. Under the best trade-offs objective, recommend 5 finalists and refuse fewer.

**Build.** Steps are `--max-iterations N`, candidates `--proposals-per-iteration N`, runs `--runs-per-candidate N`, finalists `--n-finalists N` (passed only when it isn't 3, and never with a lower-bound pick), trials at once `--max-workers N`, and the cap `--max-rollouts 500`. 0 steps passes no `--max-iterations` and no `--reflection-model`, and adds `--baseline-only` to the setup command.

### 7. Crashes and pick

Ask both questions together in one message and take one reply; a question left out keeps its recommended value. N is the finalists from step 6.

```text
Two last choices: answer either or both in one reply (ex. "crashes 2, pick 1").
Crashes: how should a run that crashes count?
1. As the worst score of each evaluator (Recommended)
2. As no data, left out of the scores
3. Your own: a score you give, ex. 0.5
Pick: how should the winner be chosen from the 3 finalists on validation?
1. The best average score (Recommended)
2. Cautiously: by a low estimate of each score, 75% sure the score is at least that
3. More cautiously: by a lower estimate, 90% sure the score is at least that
4. Your own: another level above 50% and below 100%, ex. 80% sure
```

**Build.** Crashes: option 1 passes nothing, option 2 `--agent-failure-score none`, option 3 `--agent-failure-score X`. Pick: option 1 passes nothing; options 2 and 3 add `--final-pick N:0.25` or `--final-pick N:0.1` to the setup command and replace `--n-finalists`; a level of X% sure in option 4 is `--final-pick N:(1 - X)`, ex. 80% is `N:0.2`.

### 8. Review

Show the setup as one summary in plain language, then the commands, and ask the user to start or to name a row to change. Go back to that row's step, then show the review again.

```text
optimize · <experiment> · <cases> cases · <train> train / <validation> validation / <test> test
  agent        strands-workspace/agent.py
  tune         instructions, effort
  models       <the agent's model> [and N other models]
  goal         <the objective, or "the mean of every evaluator">
  constraints  <tokens at most 4000, 90% sure that each holds, or "none">
  cases        5 train / 3 validation / 2 test
  budget       4 steps × 4 candidates, 1 run of each case, 3 finalists
  parallel     4 trials at once
  crashes      a crashed run scores the worst value
  pick         the best average score of 3 finalists
  ~<rollouts> rollouts · ~<judge calls> judge calls · price unknown
```

Under the summary, list every warning from steps 2 to 7, and say that the run's model, judge, and proposer calls are paid and use the user's credentials, and that the first run installs the optimizer (about a minute, needs git and network).

Then show the commands, built from the **Build** notes:

**Setup command**, from the same folder as step 1:

```bash
node <skill_dir>/scripts/prepare.mjs setup [path/to/strands-workspace] --experiment <name> \
  --tune <knob key> [--tune ...] [--model <model id> ...] [--baseline-only] [--final-pick N:Q]
```

**Optimize flags**, appended to the `command` that setup prints:

```text
--split <TRAIN,VALIDATION,TEST> [--max-iterations <steps>] --proposals-per-iteration <candidates> \
--runs-per-candidate <runs> --max-workers <parallel> --max-rollouts 500 --seed 0 \
[--objective ... [--pareto]] [--constraint ...] \
[--feasibility-threshold X] [--agent-failure-score none|X] [--n-finalists N] [--reflection-model MODEL_ID]
```

`--reflection-model MODEL_ID` changes the model rewriting text (default Bedrock `global.anthropic.claude-sonnet-5-5`); pass it only when the user asks and a text knob is tuned.

### 9. Run

**Setup.** Run the setup command. It prints progress to stderr and JSON to stdout. On the way it prepares the optimizer environment, as the Strands CLI does: it reuses the Strands CLI's environment at `strands-workspace/strands-dev/tmp/.state/evals-env`, or creates it from the workspace's `.venv` Python (a TypeScript agent: `python3`, or `$PYTHON`) and installs the pinned optimizer once into a shared folder under `~/.strands/temporary-evals/`. A Python agent's packages are added after the optimizer's. The JSON holds:

- `folder`: the run's folder, `strands-workspace/strands-dev/experiments/<name>/optimizations/<stamp>/`, where the Strands CLI keeps its own runs. The experiments folder's `.gitignore` already leaves out `optimizations/`, so nothing lands in git. Its `.state/` holds `space.json` (the tuned knobs as text, choice, and switch variables keyed by `create_harness` option, snake_case in Python) and `task.py`, which the CLI's own code writes and which loads the agent through the bridge.
- `variables`: the knobs that `space.json` tunes. Check they match the review.
- `command`: `cd` to the agent root, put the optimizer's `bin` first on `PATH`, and start `strands-evals optimize run` with `--agent`, `--rollout`, `--search` (with `--baseline-only`), `--final-pick` (with `--final-pick`), `--base-dir`, and `--rich`.
- `resume`: the same start for `strands-evals optimize resume`, for exit code 130.

If setup exits with code 2, show its last stderr lines. For a failed install, ask the user to check the network, the pip index settings, and that git is installed; for a missing `.venv` or `strands-harness`, ask them to install the agent's dependencies in `strands-workspace/.venv`.

**Start.** Run `command` plus the optimize flags in the background, with stdout, stderr, and the exit code in their own files in `folder`:

```bash
F=<folder>
(<command> <optimize flags> > "$F/summary.txt" 2> "$F/run.log"; echo $? > "$F/run.exit") &
```

If you can't run a background process, run it in the foreground with a long timeout and show the output when it ends. As soon as `run.log` prints `strands-evals optimize: run folder <RUN_DIR>`, tell the user the folder.

**Progress**, as the Strands CLI shows it: until `$F/run.exit` exists, about once a minute, relay each new line of `run.log` that starts with `strands-evals optimize:`, such as `iteration N (<blocks>): P proposals, R rollouts`, and any warning or error line, in plain words (ex. "step 2 of 4 done: 4 new versions tried, 61 rollouts so far"). Read new lines with `tail -n +<next line> "$F/run.log"`.

**Exit codes**, from `$F/run.exit`:

- **0**: the run finished. Go to step 10.
- **3 with `<RUN_DIR>/result.json`**: the search stopped on a proposer error, but the end phase ran and the result is valid. Show the error from `run.log`, then go to step 10.
- **3 without `result.json`**: the run stopped before it finished, most often at setup. Show the error line from `run.log`. When it says the baseline's train cases are unmeasured or every baseline trial failed, the agent or a model service failed (ex. `NoCredentialsError`, `Could not load credentials`): ask the user to check that the agent answers a case and that the credentials are fresh, then run setup and the command again.
- **2**: bad input (a flag, the agent, a trace-level evaluator with a TypeScript agent, or a space variable). Show the error, fix the value in its step, and show the review again.
- **130**, or the process is gone without writing `run.exit` (killed): continue with `(<resume> <RUN_DIR> --rich > "$F/summary.txt" 2>> "$F/run.log"; echo $? > "$F/run.exit") &`, then follow it the same way. Finished trials aren't run again, and only `--max-rollouts`, `--max-iterations`, and `--max-workers` may change on resume.

### 10. Report

The report is `$F/summary.txt`, the optimizer's own summary (`--rich`): why the search stopped, what ships, every finalist on validation, the test check (the champion against the current agent on the test cases, with its 90% interval), the effects of each option, and the costs. Give the user its path to open, and the headline in plain words: the version that ships, or that no version beat the current agent; whether the test gain is likely real (the whole interval above 0), likely worse (below 0), or may be luck (it includes 0, and more test cases narrow it).

Nothing in the user's project changes. The champion's changes are keyed like `space.json`: `<RUN_DIR>/best.diff` holds each changed text (ex. `instructions`) as a diff under its key, and `<RUN_DIR>/best.values.json` holds each changed choice or switch as `{"old": ..., "new": ...}`, where a switch key such as `builtin_tools_0` is `false` when that entry is left out. Apply them to the agent's `create_harness` / `createHarness` call only when the user asks, and suggest evaluating the agent again after.
