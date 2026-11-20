---
name: optimizer
description: Use when the user wants to tune an agent against a strands-evals experiment, such as its prompt, skills, model, effort, or numeric settings, with `strands-evals optimize`. Sets up the run from a free-text request or step by step (knobs, objective, constraints, case split, budget), confirms a review summary, installs the optimizer if needed, runs it with progress, and saves a report.
version: 0.1.0
---

# optimizer

Sets up and runs `strands-evals optimize run` with the options, defaults, and checks of the `/optimize` tab of `strands dev`, for any agent the optimizer can load.

The optimizer is experimental and isn't on evals `main` yet. This skill runs the build pinned by `strands dev`:

```text
strands-agents-evals[optimization] @ git+https://github.com/arielnabavian/evals@5a41cdabe8090bad9f5200a03039c1beeca0cb68
```

`<skill_dir>` below is the folder holding this file.

## Steps

Ask every question in plain chat: one step at a time, numbered options with the recommended one first and marked `(Recommended)`, and always accept a free-form answer (ex. "2 steps", "drop effort", "tokens <= 4000"). Check each answer against the rules of its step before moving on; when it breaks one, say which rule and ask again. If you can't wait for a reply (ex. a non-interactive run), take each recommended value, say so, and continue.

### 1. Find the inputs and ask for a request

Find the experiment and the agent in the working directory, or where the user points, without opening whole files (use `ls`, `grep`, `jq`).

- **Experiment**: a JSON file with `cases` and `evaluators` (`strands-evals run` reads the same file). Every case needs a unique `name`. Note the case count and each evaluator's `evaluator_type`, which is its metric name. A `.toml` run file of `strands-evals optimize run` holds the flags of an earlier setup: take its keys as the starting answers.
- **Agent**, one of three routes:
  - **Portable config** (`--config PATH`): a strands-harness config JSON with an `"instructions"` key (it may be empty). Needs no Python of the user's.
  - **Tunable factory** (`--agent MODULE:ATTR`): a Python function decorated with `@tunable(...)` from `strands_evals.optimization`. `MODULE:ATTR` is `path/to/file.py:attr` or a dotted module. A plain factory is rejected.
  - **Task builder** (`--build-task MODULE:ATTR --space MODULE:ATTR`): a `build_task(ctx)` and its `{name: marker}` space, for agents the factory form doesn't fit. Every marker needs `default=`.

Only the portable config and the factory route can be recommended to a user who has neither: a strands-harness agent can be written as a portable config, and a Python Strands agent can take `@tunable` markers on its factory's parameters.

Then ask the first question:

Question: Describe the optimization you want, or answer "step by step".
1. Step by step (Recommended): I walk you through the knobs, the goal, the cases, and the budget.
2. Describe it: for example, "make the support agent shorter without losing helpfulness, about 100 rollouts".

If more than one experiment or agent fits, or none does, and the request doesn't say which, ask which to use next.

If the user describes it, decide every value of steps 2 to 5 from that text and the files alone, using each step's recommendation rule where the text says nothing, ask no setup question, and go to step 6. A value the text asks for that breaks a rule of its step is changed to the closest valid one and flagged in the review.

### 2. Knobs: what may change

**Portable config.** Each knob is a dotted path in the config. The run tunes `instructions` unless `--no-tune-instructions` is passed, and each knob adds one `--tune`:

- **Text** (`--tune PATH`): a text field, rewritten by Reflection, the proposer model. `--tune skills` makes each `SKILL.md` in the config's local `skills` folder editable (one local path only).
- **Choice** (`--tune 'PATH=JSON_LIST'`): one of plain JSON values, ex. `--tune 'effort=["auto","off","low"]'` or `--tune 'caching=[true,false]'`. The list must include the config's current value, which is the baseline. Good candidates are `model` (the current model plus the model ids the user names, ex. `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`), `effort`, `caching`, and `contextManager`.
- **Number** (`--tune PATH=int:LO..HI` or `--tune PATH=float:LO..HI`): an inclusive range, ex. `--tune builtinTools.subagent.maxDepth=int:0..2`. The field must be written in the config, and its value, the baseline, must lie in the range.

Each choice and range is checked against the harness's own schema before the run, so a value no trial could take stops the run with exit code 2 and spends nothing. This route can't tune model parameters (temperature), choose among constructed components, or reach `subagents`; those need the factory route.

**Factory or task builder.** The knobs are the markers already in the code: `TextValue`, `FileText`, `JsonField`, `SkillLibrary`, and `tool_descriptions` are text; `OneOf` and `choice` are choices; `Int` and `Float` are numbers. List them with their baselines. To add, change, or drop a knob, propose the edit to the marker in the user's code and make it only after they agree.

**Recommend** `instructions` (or the factory's prompt marker) by default, as `strands dev` does. When the goal or the request is about cost, length, or speed, also recommend the effort, caching, and context-manager settings the agent has. Recommend the `skills` folder when the agent has one and the cases exercise it. Recommend a model choice only with model ids the user named.

**Rules.** At least one knob must be tuned. When a text knob is tuned, warn: Reflection writes shell commands and runs them on this machine with the user's environment, credentials, and network. It is not a sandbox, and text in the cases can carry prompt injection. Run it in a container or VM that has only the credentials the run needs.

### 3. Goal: objective, constraints, crashes, and pick

**Metrics** are the experiment's evaluator names (`Helpfulness` resolves to `HelpfulnessEvaluator`) and the usage metrics `tokens`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `seconds`, and `model_calls`. Usage metrics and trace-level evaluators (ex. Helpfulness) need the agent to be Strands, running in the optimizer's process.

**Objective.** Each goal is one `--objective {maximize,minimize}:METRIC[:weight=X][:scale=X]`. Several goals form a weighted sum; `--pareto` combines them as a Pareto front instead (needs 2 or more). With no `--objective`, the run maximizes the mean of every evaluator. The options, in the order `strands dev` offers them:

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

**Pick.** How the champion is picked from the finalists on validation: the posterior mean (default, recommended), a 25% lower bound to ship cautiously, or a 10% lower bound to ship more cautiously. A lower bound needs a `FinalPick` object, so write a file such as `.strands-evals/optimization/final_pick.py`, with N from step 5:

```python
from strands_evals.optimization import FinalPick, LowerBound

final_pick = FinalPick(n_finalists=N, rule=LowerBound(q=0.25))  # q=0.1 for the 10% bound
```

and pass `--final-pick <absolute path>:final_pick` instead of `--n-finalists`.

### 4. Cases: the split

Train cases guide the search, validation picks the winner, and test checks it on cases nothing else saw. Choose the train share from 0.3, 0.4, 0.5, 0.6, 0.7, and 0.8, then how the rest divides: half test (1 test case per validation case), a third test (1 test case per 2 validation cases), or no test. Pass `--split TRAIN,VALIDATION,TEST`, with validation `(1 - TRAIN) × (1 - test part)` and test `(1 - TRAIN) × test part`. The shares must sum to 1 within 1e-9, so write enough digits (ex. `0.6,0.266666666667,0.133333333333`).

Show each choice as case counts: each share times the case count, rounded down, with the cases left over going to the largest remainders. Recommend 0.5 train with half test (`0.5,0.25,0.25`), the `strands dev` default, or the nearest choice that leaves at least 2 validation cases and, with test cases, at least 2 test cases.

Rules: each share is from 0 to 1 and train and validation are above 0. The optimizer refuses a split with fewer than 1 train or 1 validation case. Fewer than 2 validation or test cases: warn that the scores can be luck, and say how many cases to add to reach 2 of each.

### 5. Budget

Ask each, recommending the default:

| Setting | Flag | Choices | Default |
| --- | --- | --- | --- |
| budget: steps | `--max-iterations N` | 0, 1, 2, 4, 6, 8 | 4 |
| per step: candidates | `--proposals-per-iteration N` | 2, 4, 6, 8 | 4 |
| runs of each case per candidate | `--runs-per-candidate N` | 1, 2, 3, 5 | 1 |
| finalists on validation | `--n-finalists N` | 1, 2, 3, 5 | 3 |
| parallel trials | `--max-workers N` | 1, 2, 4, 8, 16 | 4 |

0 steps measures only the baseline: `--max-iterations` must be at least 1, so pass `--search <skill_dir>/assets/hooks.py:baseline_only` instead, and no `--reflection-model`. Under `--pareto`, at least 5 finalists go to validation.

Estimate the rollouts (one case run once) and show them with each answer:

```text
rollouts = runs × (train + steps × candidates × train + (finalists + 1) × validation + tested × test)
```

`finalists` is at least 5 under `--pareto`, and `tested` is 2 (the champion and the baseline), or `finalists + 1` under `--pareto`. Judge calls are about rollouts × evaluators, and Reflection's own model calls come on top; neither counts against the budget.

The run passes `--max-rollouts 500`, the `strands dev` cap. When the estimate is above 500, lower the steps, candidates, or runs, or raise `--max-rollouts` if the user asks. `--max-rollouts` must cover the baseline and the end phase, `runs × (train + (finalists + 1) × validation + tested × test)`. The optimizer warns when one step's trials, `candidates × train × runs`, exceed the parallel trials: steps then run in batches and take longer.

### 6. Review

Show the setup as one summary, then the exact command, and ask the user to start or to name a row to change. Go back to that row's step, then show the review again.

```text
optimize · <experiment> · <cases> cases · <train> train / <validation> validation / <test> test
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

Under the summary, list every warning from steps 2 to 5, say that the run's model, judge, and proposer calls are paid and use the user's credentials, and say whether step 7 will install the optimizer.

Build the command from the answers. Run it from the folder holding the experiment, with absolute paths for the hook files:

```bash
"$OPT_PY" -m strands_evals.cli optimize run <experiment.json> \
  <route: --config <config.json> [--tune ...] | --agent <file.py:attr> | --build-task <file.py:attr> --space <file.py:attr>> \
  --split <TRAIN,VALIDATION,TEST> --max-iterations <steps> --proposals-per-iteration <candidates> \
  --runs-per-candidate <runs> --max-workers <parallel> --max-rollouts 500 --seed 0 \
  [--objective ... [--pareto]] [--constraint ...] \
  [--feasibility-threshold X] [--agent-failure-score none|X] [--n-finalists N | --final-pick <file.py>:final_pick] \
  --name <name> --rich
```

- Pass `--feasibility-threshold`, `--agent-failure-score`, and `--n-finalists` only when they differ from 0.9, `worst`, and 3.
- `--name` is the run folder's prefix: use the experiment's file name, with only lowercase letters, digits, `-`, and `_`.
- `--reflection-model MODEL_ID` changes the model rewriting text (default Bedrock `global.anthropic.claude-sonnet-5-5`); pass it only when the user asks and a text knob is tuned.
- `--rich` prints the result as text; without it, piped output is JSON.

### 7. Run

**Environment.** `OPT_PY` is the Python that runs the optimizer. Try the Python that runs the agent first (the project's `.venv/bin/python`, or `python3` for a portable config):

```bash
"$AGENT_PY" -c "import strands_evals.optimization" && OPT_PY="$AGENT_PY"
```

If that import fails, install the pinned build into its own environment, so the user's environment keeps its packages. It needs Python 3.10 or later, git, and network access:

```bash
"$AGENT_PY" -m venv .strands-evals/optimizer-env
OPT_PY="$PWD/.strands-evals/optimizer-env/bin/python"   # Scripts/python.exe on Windows
"$OPT_PY" -m pip install --disable-pip-version-check --prefer-binary \
  'strands-agents-evals[optimization] @ git+https://github.com/arielnabavian/evals@5a41cdabe8090bad9f5200a03039c1beeca0cb68'
```

For the factory and task-builder routes, the optimizer imports the user's code, so also let the new environment read the agent's packages, after its own:

```bash
"$AGENT_PY" -c "import site; print('\n'.join(site.getsitepackages()))" \
  | sed 's/.*/import site; site.addsitedir("&")/' \
  > "$("$OPT_PY" -c 'import site; print(site.getsitepackages()[0])')/agent-packages.pth"
"$OPT_PY" -c "import strands_evals.optimization"
```

If the install fails, show pip's last lines and ask the user to check the network, the pip index settings, and that git is installed. The environment and the run folders live under `.strands-evals/`; if the project is a git repository and doesn't ignore that folder, suggest adding it to `.gitignore`.

**Start.** Run the command in the background, with stdout, stderr, and the exit code in their own files:

```bash
mkdir -p .strands-evals/optimization
L=.strands-evals/optimization/<name>
(<command> > "$L.out" 2> "$L.err"; echo $? > "$L.exit") &
```

If you can't run a background process, run it in the foreground with a long timeout and show the output when it ends. As soon as stderr prints `strands-evals optimize: run folder <RUN_DIR>`, tell the user the folder.

**Progress.** Until `$L.exit` exists, about once a minute, relay each new `strands-evals optimize: iteration N (<blocks>): P proposals, R rollouts` line from stderr, and show the run's status:

```bash
"$OPT_PY" <skill_dir>/scripts/progress.py <RUN_DIR>
```

It prints the phase (measuring the baseline, the search iteration, or the end phase), the elapsed time, the rollouts spent of the budget, the baseline and the best objective measured on train, and the latest candidates with their measured objective and the proposer's reason. The best measured on train isn't what ships: the end phase picks the champion on validation.

**Exit codes**, from `$L.exit`:

- **0**: the run finished. Go to step 8.
- **3 with `<RUN_DIR>/result.json`**: the search stopped on a proposer error, but the end phase ran and the result is valid. Show the error from stderr, then go to step 8.
- **3 without `result.json`**: the run stopped before it finished, most often at setup. Show the error line from stderr. When it says the baseline's train cases are unmeasured, the agent or a model service failed (ex. `NoCredentialsError`): ask the user to check that the agent answers a case and that the credentials are fresh, then run again.
- **2**: bad input (a flag, a hook, the run file, or a missing optimizer). Show the error, fix the value in its step, and show the review again.
- **130**, or the process is gone without writing `$L.exit` (killed): continue with `"$OPT_PY" -m strands_evals.cli optimize resume <RUN_DIR> --rich`. Finished trials aren't run again. Only `--max-rollouts`, `--max-iterations`, and `--max-workers` may change on resume, and a `--config` run whose config file changed since is refused.

### 8. Report

Write the report and print its path:

```bash
"$OPT_PY" <skill_dir>/scripts/report.py <RUN_DIR>
```

It writes `<RUN_DIR>/report.md`: what ships, the test check (the champion against the baseline on the test cases, with its 90% range and whether the gain is likely real), the changed values, the diff, the optimizer's summary, and the run's files. Give the user its path to open, and the headline: the candidate that ships, or that no candidate beat the current agent.

Nothing in the user's project changes. The champion's text changes are in `<RUN_DIR>/best.diff`, on the real file paths, and every typed value is in `<RUN_DIR>/best.values.json`. Apply them only when the user asks, and suggest evaluating the agent again after.
