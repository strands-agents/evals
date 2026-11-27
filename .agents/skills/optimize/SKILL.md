---
name: optimize
description: Use when the user wants to tune a strands-harness agent (agent.py built with create_harness, or agent.ts built with createHarness) against an experiment with `strands-evals optimize`, such as its instructions, model, effort, caching, or context manager. Sets up the run from the user's prompt (step by step when there is none), confirms a review (unless the user asks to run it, or the run is headless), runs it with progress, and reports the result.
version: 0.1.0
---

# optimize

Sets up and runs `strands-evals optimize run` on an agent built with strands-harness: a Python `agent.py` (`create_harness`) or a TypeScript `agent.ts` (`createHarness`). Unless the file has its own tunable factory, every trial runs the agent's file again, with a fresh session and a copy of the memory folder, and only the chosen knobs change in the call that builds `agent`:

- **Python**: the skill writes a temporary tunable factory, `$F/tunable_agent.py` (step 7), and passes it to `--agent`. The user's file is never edited.
- **Python with its own tunable factory** (a function marked `@tunable(...)`, or `tunable(fn, inject=...)`): the skill passes that factory.
- **TypeScript**: the skill passes the file with one `--tune` flag per knob; it runs in a Node.js worker.

`<skill_dir>` below is the folder holding this file. The skill uses the Python of the installed `strands-evals` command (step 1).

## Steps

Ask in plain chat, one step at a time unless a step says to ask its questions together. Every list has at most three numbered options before a short "Your own" option, the default first and marked `(Default)`, except the knobs question (every tunable knob, step 2) and the budget's settings (step 5); always accept a free-form answer (ex. "2 steps", "drop effort", "tokens at most 4000"). When an answer breaks a rule of its step, say which and ask again.

**Setting up from the request.** When the message that started the skill says how to optimize (ex. "make the support agent shorter without losing helpfulness, about 100 rollouts"), ask no setup question: take every value it gives (the experiment, steps 2 to 5, and the review rows), infer the rest from it and the files, give the others their defaults, and go to the review (step 6). Only a skill started without such a prompt (ex. `/optimize` alone, or "optimize my agent") goes step by step, skipping a question the user already answered. A value that breaks a rule becomes the closest valid one and is flagged in the review.

**Running without questions.** When the user explicitly asks to run, in any message (ex. "just run it", "2 steps, then start it", "no need to confirm"; "optimize my agent" isn't one), or you can't wait for a reply (a headless run, ex. `claude -p` or a scheduled job; Claude Code sets `CLAUDE_CODE_SESSION_ATTENDED=0`, so check `printenv CLAUDE_CODE_SESSION_ATTENDED` before asking anything), stop asking: give every value still missing its default and skip only the review's wait (step 6). The review itself, with its warnings, is still written to the user before the run starts. Still stop at exit code 2 (step 7).

In every setup, what has no default (the agent or experiment file when none is found, the model ids for a model choice, an install step) is asked for, or in a headless run named as missing before stopping.

Questions and their options are plain language: they name what changes and what it costs, never a flag, an option name in code, or a file path. Flags live in each step's **Build** notes, which are for you, and appear to the user only in the review's command (step 6).

The case split, how sure each constraint must be, how crashes count, and how the winner is picked aren't asked: they take their defaults and show as review rows the user can change (step 6).

### 1. Find the inputs and the experiment

Look only in these places, with these commands from the current folder; don't search the tree or open other files. When something isn't there, ask for it instead of looking further.

```bash
ls strands-dev/.config strands-workspace/strands-dev/.config agent.py agent.ts strands-workspace/agent.py strands-workspace/agent.ts 2>/dev/null
E=$(find strands-dev/experiments strands-workspace/strands-dev/experiments -mindepth 2 -maxdepth 2 \
  -name '*.json' ! -name '*.report.json' 2>/dev/null)
[ -n "$E" ] && echo "$E" || find . strands-workspace -maxdepth 1 -name '*.json' 2>/dev/null | xargs grep -l '"evaluators"' 2>/dev/null
```

The first line lists the agent files; the rest lists the workspace's experiments, or else the JSON files with evaluators in the current folder and `strands-workspace/`.

- **Agent.** With a `.config`, the agent is its `agent.entrypoint`, relative to its `agent.path` (itself relative to the `.config`'s folder). Otherwise it is the `agent.py` or `agent.ts` listed. Its `agent` must be built by a call in the file: `agent = create_harness(...)` in Python, or an exported `agent` from `createHarness(...)` in TypeScript (also `.mts`, `.js`, `.mjs`). With no agent, or several, ask for the path of "the file that builds your agent with create_harness or createHarness".
- **TypeScript** only: Node.js 22.18 or later must be on `PATH`, and the agent file must be able to import `@strands-agents/harness`. Check from the project with `node --version` and `node --input-type=module -e "import('@strands-agents/harness')"`; when either fails, give the user the fix (install a newer Node.js, or `npm install @strands-agents/harness`) and wait, without installing anything yourself.
- **Project.** The `.config`'s agent path, else the agent file's folder. Commands run from there.
- **Python.** The interpreter of the `strands-evals` command on `PATH`, for both languages. The optimizer imports a Python agent in that strands-evals process, so it must also have the agent's packages. `strands-harness` comes with the optimizer's `[optimization]` extra, as it did under the Strands CLI, so the agent's code must work with the version that extra pins; the check below catches an install without the extra:

  ```bash
  SE=$(command -v strands-evals) && SE=$(realpath "$SE") && PY="$(dirname "$SE")/python" \
    && { [ -x "$PY" ] || PY=$(head -1 "$SE" | sed 's/^#!//; s/ .*//'); } \
    && "$PY" -c "import strands_evals.optimization, strands_harness" && strands-evals optimize run --help > /dev/null
  ```

  When there is no `strands-evals`, the import fails, or the command doesn't start (ex. a moved environment left a dead interpreter path in it), don't install anything yourself: give the user this command for the Python their agent runs with (ex. their activated environment's `python`), say that its `optimization` extra pins `strands-harness` to 0.4.x, and wait until they have run it:

  ```bash
  python -m pip install 'strands-agents-evals[optimization]>=1.12.0'
  ```
- **Tunable factory**, Python only. Read the agent file and list its tunable factories: module-level names bound to a function decorated with `@tunable(...)` (under any import alias or module prefix, ex. `@tn(...)` or `@opt.tunable(...)`), or assigned the result of `tunable(fn, inject=...)` (with or without a type annotation), including one defined inside a module-level `if` block. Don't count a `@tunable` in a comment or docstring, a name rebound later to something else, or a function with another decorator above `@tunable` (it no longer returns the factory). A factory the file imports from another module counts too when the file's own code or comments say it is tunable; when you can't tell, ask the user.

  With one name, the run uses that factory; with several, ask which, up to three, the first as the default. With none, the agent is the `create_harness` agent above. Read the markers in the factory's `@tunable(...)` (or `inject=`) for the review's tune row: one per name, ex. `OneOf("auto", "low")` is a choice and `TextValue()` a text the proposer rewrites. A `JsonField(path, pointer)` is a field of a JSON file whose baseline is the file's value: with an `Int` or `Float`, a number in that range (ex. `JsonField("agent.json", "/temperature", Float(0.0, 1.0))`); with a `OneOf`, a choice among its options; on a boolean, a choice of true or false; otherwise a text the proposer rewrites.
- **Experiments.** The files the second line lists. For each, read only its counts: `jq '{cases: (.cases | length), evaluators: [.evaluators[].evaluator_type]}' <file>` (each `evaluator_type` is a metric name); skip a file `jq` can't read that way.

Then, in a step-by-step setup, ask which experiment to use: up to three, by name with their cases and evaluators (ex. `capitals: 6 cases, scored by OutputEvaluator`), the most cases first as the default, then "Your own: another experiment's name or path", naming any others found in one line under the list. With none found, ask for the path of an experiment file (the JSON file with cases and evaluators that `strands-evals run` reads), and stop if there is none.

**Few cases.** Once the experiment is known, check its case count. With fewer than 20 cases, say that the optimizer splits them into train, validation, and test, so a few cases per part can't tell versions apart (a judge's score moves by a tenth or more between runs on 3 cases), and ask:

```text
This experiment has N cases; 20 or more give the optimizer enough to compare versions.
1. Add cases first: write more like the current ones (Default)
2. Add cases from past conversations with the agent
3. Continue with N cases
```

Option 1 uses the `strands-evals` skill (its experiment generator, or cases written from the existing ones), option 2 the `session-converter` skill; either writes a new experiment file, then come back here with it. Running without questions, continue with the cases there are and put this in the review's warnings.

### 2. Knobs: what may change

Skip this step with a tunable factory (step 1): its markers already say what may change. Still give the Reflection warning under **Rules** when one of them is a text marker (`TextValue`, `FileText`, `SkillLibrary`, `tool_descriptions`, or a `JsonField` with no `Int`, `Float`, or `OneOf` on a string, array, object, or null field).

Read the agent's `create_harness(...)` or `createHarness(...)` call and offer the knobs below with their current values. An option the call doesn't pass has the harness default as its current value (effort and context manager auto; caching on; skills from the project's `.agent/skills` folder when it exists; background tasks off; the model the harness's own). In TypeScript, background tasks can be tuned only when the call writes them. The system prompt is a text knob only when the call passes one, and then replaces the instructions. Name in one not-tunable line a knob set in code (a variable, a call, a model instance). Memory and session aren't offered.

**Switch each on or off.** For a list the agent loads, offer to switch its entries on or off one by one, never the whole list at once. That covers the skills (each skill folder), the tools, the built-in tools (unset: shell, read, write, edit, web_fetch, web_search, programmatic_tool_caller, subagent), the built-in plugins (unset: environment in Python; todos and environment in TypeScript), the plugins, and the MCP servers when the call writes them by name. Name the entries in the knob's line. An entry set in code that can't be named (ex. a tool without a name) can't be switched; with many entries, say that each switch is one more setting to search, so a larger budget helps.

**Skills and tools** can also have their text rewritten by the proposer model; never write a new skill or tool, create a skills folder, or add skills to the call:

- **Skill wording** (each skill's `SKILL.md`) needs one local skills folder holding at least one skill: `skills` written as a folder path, or a list of one. In Python, an unset `skills` (or `True`) also counts when `<project>/.agent/skills` holds a skill; in TypeScript it doesn't.
- **Switching skills** needs skills the agent actually loads: a folder as above, other sources (ex. URLs), or an unset `skills` (or `True`) with skills in `<project>/.agent/skills`. With none, say "skills: none attached" in the not-tunable line.
- **Tool descriptions**, Python only: the built-in tools the agent loads and tools in the call's `tools` written as `@tool` functions (or other tools whose description can be set); not MCP servers' tools.

When the user picks skills, or tools or built-in tools, and their text can be rewritten, ask this follow-up before step 3, once for skills and once for tools (for skills in TypeScript, without option 3: one run can't do both there):

```text
How may the optimizer change the skills?
1. Rewrite their wording (Default): the proposer model rewrites each skill's instructions; the skills stay on
2. Switch each one on or off: try the agent with some of its skills left out
3. Both: rewrite their wording, and also try leaving some out
4. Your own: for example, "rewrite only the geo skill"
```

For tools, it reads "descriptions" and "what each tool says it does" (ex. "rewrite only the lookup tool's description"). With only switching possible, the knob reads ex. "skills: switch each on or off (geo, maps); now all on" and the follow-up isn't asked. Rewriting, alone or with switching, is a text knob (see **Rules**).

**Effort levels** depend on the model: offer auto, low, medium, and high, which most models accept, and take others through "Your own". Say that a level the model doesn't offer makes its trials fail and spends their rollouts. Effort isn't tunable when the model is a model instance rather than an id string: the harness ignores effort for those.

List every tunable knob, then "Your own", then the not-tunable line. The user may pick several: use the host's multi-select question UI when it has one, with the defaults pre-selected; otherwise accept several numbers in one reply.

```text
What may the optimizer change? Pick any (ex. "1, 3").
1. instructions (Default): rewritten by the proposer model; now "Answer geography questions in one short sentence."
2. model: try other models next to bedrock/global.anthropic.claude-sonnet-4-5...; name the models to try
3. effort: auto, low, medium, or high; now auto
4. prompt caching: auto, on, or off; now on
5. context manager: auto, agentic, or off; now auto
6. background tasks: on or off; now off
7. skills: rewrite their wording, or switch each on or off (geo, maps); now all on
8. tools: rewrite their descriptions, or switch each on or off (lookup, search); now all on
9. built-in tools: rewrite their descriptions, or switch each on or off (shell, read, write, edit, web_fetch, web_search, programmatic_tool_caller, subagent); now all on
10. Your own: for example, "only the instructions and effort low or medium"
Every trial starts with a fresh session and a copy of the memory folder, so trials don't share state.
```

A choice the user narrows (ex. "effort low or medium") always keeps the current value. With model picked, ask for the model ids to try.

**Defaults**: the instructions, or the system prompt when the agent passes one. When the request is about cost, length, or speed, effort, prompt caching, and the context manager are defaults too. A model choice is a default only with model ids the user named.

**Rules.** At least one knob is tuned. When a text knob is tuned, warn: Reflection, the proposer, writes shell commands and runs them on this machine with the user's environment, credentials, and network. It is not a sandbox, and text in the cases can carry prompt injection. Run it in a container or VM that has only the credentials the run needs.

**Build, Python.** Each knob is one entry of `MARKERS` in `$F/tunable_agent.py` (step 7), named as the call names the argument, and each switch one entry of `SWITCHES`; `CURRENT` holds each plain knob's value in the call:

- Instructions: `"instructions": TextValue(default=CURRENT["instructions"])`, with `""` when the call sets none (the harness treats both alike). The system prompt: `"system_prompt": TextValue(default=CURRENT["system_prompt"])`.
- A choice: `OneOf(...)` over its options, the current value among them, ex. `"effort": OneOf("auto", "low", "high", default=CURRENT["effort"])`, for `model`, `effort`, `caching`, `context_manager` (its off is `False`), or `background_tasks`. For an argument the call leaves out, the current value is its harness default, with `None` for the model and background tasks.
- Skill wording: `"skills": SkillLibrary("<absolute skills folder>")` (a relative folder in the call is taken from the project folder), with `only=("geo",)` for named skills.
- Each switch: `"<parameter>": ("<argument>", "<entry name>")` in `SWITCHES`, ex. `"skill_geo": ("skills", "geo")` or `"builtin_shell": ("builtin_tools", "shell")`, with a parameter name that is a Python identifier. An entry's name is a skill's folder name, a built-in's name, a tool's name, a plugin's `name` (else its class name), or an MCP server's key.
- Tool descriptions: the tools' names in `DESCRIBED`, built-in (ex. `"read"`) or the agent's own, ex. `DESCRIBED = ("lookup", "search")`, or `()` when none.

**Build, TypeScript.** Each knob is a `--tune` over an argument of the call, named as the call names it (ex. `--tune systemPrompt`, `contextManager`, `backgroundTasks`). The instructions are tuned unless `--no-tune-instructions` is passed. A choice is `--tune 'OPTION=JSON_LIST'` with a list that includes the current value, for `model`, `effort`, `caching`, `contextManager` (its off is `false`), or `backgroundTasks`, ex. `--tune 'effort=["auto","low","high"]'`. For an option the call leaves out, the current value is its harness default, with the harness's default model id for the model (ex. `bedrock/global.anthropic.claude-opus-5`). Skill wording is `--tune skills` (the call's folder, read from the project folder). TypeScript has no separate switch per entry: switching is one choice among sets of entries, `--tune 'OPTION=JSON_LIST'` whose options are lists, the call's own list first, for `builtinTools`, `builtinPlugins`, or `skills` (a skills folder as the list of its skill subfolders, ex. `--tune 'skills=[["./skills/geo","./skills/maps"],["./skills/geo"],["./skills/maps"],[]]'`). With at most 4 entries, offer every set; with more, the full set and each set with one entry left out. Say that in TypeScript the search picks among these sets rather than switching each entry freely. Tools and plugins written in code can't be switched there.

A tunable factory, the user's or the skill's, takes no `--tune` and no `--no-tune-instructions` (the optimizer refuses both).

### 3. Objective: what to improve

**Metrics** are the experiment's evaluator names (`Helpfulness` resolves to `HelpfulnessEvaluator`) and the usage metrics tokens, input tokens, output tokens, cache read tokens, cache write tokens, seconds, and model calls.

Ask what to improve; limits come in step 4. For options 1 to 3, ask the follow-up in one short message.

```text
What should the optimizer improve?
1. Maximize a metric: an evaluator's score, or the mean of every evaluator (Default)
2. Minimize a metric: ex. tokens, output tokens, seconds, or model calls
3. Balance several metrics: weigh them together, or find the best trade-offs without weighing
4. Your own: for example, "fewer output tokens"
```

Follow-ups, in plain words:

- **Maximize**: which metric; default the mean of every evaluator, or the one evaluator the request names.
- **Minimize**: which usage metric; default tokens, output tokens when the request is about answer length, or seconds when it's about speed.
- **Balance**: which metrics, at least two (ex. "OutputEvaluator and tokens"); whether to weigh them (default), with how much each counts (ex. "OutputEvaluator twice as much as tokens"), or find the best trade-offs; for tokens or time, how big a change matters (default 1000 tokens, 10 seconds), said as the trade it makes: with 1000 tokens, 100 more tokens per case cost as much as 0.1 of an evaluator's 0 to 1 score. When a text knob is tuned, add that a rewritten text often adds a few hundred tokens to every case, so at 1000 a rewrite must gain about as many tenths of a score to win; suggest a larger size (ex. 5000) unless cost matters most.

**Default**: the mean of every evaluator. When the request is about length or cost, balancing the evaluators and tokens (output tokens when it's about answer length); about speed, the evaluators and seconds.

**Rules.** Every metric is one listed above; a metric appears once; an experiment with no evaluator needs a usage goal (option 2); balancing needs two metrics or more; a metric name with `:` can't be used.

**Build.** Each goal is one `--objective {maximize,minimize}:METRIC[:weight=X][:scale=X]` (maximize an evaluator, minimize a usage metric). The mean of every evaluator passes no `--objective`. Several goals form a weighted sum, with `weight=` from how much each counts, or, with `--pareto`, a front of the best trade-offs (also for a "Your own" asking for trade-offs without weighing). A limit in a "Your own" answer (ex. "scoring >= 0.8") becomes a constraint of step 4, not a goal. With several goals, each usage metric needs `scale=`, the size of a meaningful change (default `scale=1000` for tokens, `scale=10` for seconds; `scale=100` on output tokens weighs 100 tokens like an evaluator's whole 0-1 range).

### 4. Constraints: limits on top of the objective

Ask which constraints the winner must also meet, with what can be constrained listed under the options, the experiment's own evaluators first. Only a candidate meeting every constraint on validation ships.

```text
Which constraints should the winner also meet?
1. None (Default)
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

An evaluator with its own score range shows that range instead of 0 to 1.

**Default**: none, or option 2 when the objective only minimizes a usage metric (step 3, option 2); then say why: with only tokens or time to improve, an agent that answers badly, or not at all, can look best.

**A limit guards, the goal improves.** A constraint keeps the winner from getting worse; it doesn't make the search try to raise the score. So when a constraint sets an evaluator's lowest score and the objective only minimizes usage metrics, also maximize that evaluator: the objective becomes a balance of the evaluator and the usage metric (step 3, with the usage metric's default size), and say so (ex. "I'll also maximize Helpfulness, so the search tries to keep answers good while cutting tokens, not only stay above the limit").

**The current agent's scores.** A constraint the current agent already misses changes the search: until a version meets it, versions are ranked by how close they come, not by the goal (ex. minimizing tokens is ignored while the search chases a score), and when no version meets it on validation, the current agent ships though it misses it too. So before asking, read the current agent's scores from the newest earlier run of this experiment, when there is one (`<run folders>` is `optimization/runs/*/` from the project, or `strands-dev/experiments/<name>/optimizations/*/record/*/` in a strands-dev workspace, step 7; `<experiment>` the experiment file's absolute path):

```bash
for D in $(ls -td <run folders> 2>/dev/null); do
  [ "$(jq -r .experiment_file "$D/cli.json" 2>/dev/null)" = "<experiment>" ] || continue
  jq -r 'select(.event=="label" and .iteration==0) | .label.measured | to_entries[] | "\(.key) \(.value.mean)"' "$D/events.jsonl"
  break
done
```

With a score, show it under each metric of the list (ex. `OutputEvaluator: its score, from 0 (worst) to 1 (best); your agent scored 0.70 last time`), and make option 2 "every evaluator at least its current score minus 0.05" (ex. `OutputEvaluator at least 0.65`), so the constraint guards against getting worse. Say the score comes from that run, so it is stale if the agent changed since. Without an earlier run, keep option 2 as written and say that a limit above what the agent scores today has this effect; a run with 0 steps (step 5) measures it.

When a constraint is at or above the current agent's score, or above it minus 0.05 (judges vary that much between runs on a few cases), warn in the review: the search ignores the goal until a version meets it, and the current agent may ship. A limit the user wants to reach rather than keep (ex. "get it to 0.9") fits the objective better (step 3, maximize).

**Rules.** Each constraint is one metric from the list, "at least" or "at most", and a number; a metric name with `<`, `>`, or `=` can't be used.

**Build.** Each constraint is `--constraint 'METRIC>=X'` or `--constraint 'METRIC<=X'`.

### 5. Budget

Ask all six settings in one message. A setting the reply leaves out keeps its default.

```text
Budget: change any of these in one reply (ex. "200 rollouts, 2 runs"); the rest keep their default.
1. Max rollouts: the most rollouts (one run of the agent on one case) the whole optimization may spend. Default 500.
2. Steps: rounds of proposing and testing new versions of the agent. Default no limit: steps go on until the rollouts run out.
3. Candidates per step: new versions tried in each round. Default 4.
4. Runs of each case: how often each case is repeated per version, to average out noise. Default 1.
5. Finalists: the best versions re-checked on the validation cases to pick the winner. Default 3.
6. Trials at once: how many runs go in parallel; changes speed, not cost. Default 4.
```

When the experiment has fewer than 20 cases and an evaluator is an LLM judge (any but the deterministic `Equals`, `Contains`, `StartsWith`, `ToolCalled`, `SkillInvoked`, `StateEquals`, `StructuredOutputSimilarity`), add under the list: "With N cases and an LLM judge, 2 runs of each case average out the judge's noise, at twice the rollouts." The default stays 1.

Don't show a rollout estimate. When the user asks how a setting relates to the max rollouts, use these, from the case counts of the split (step 6): a run spends `runs × (train × (1 + steps × candidates) + validation × (finalists + 1) + test × tested)` rollouts (one run of the agent on one case), with `tested` 2 (the winner and the current agent), and never more than the max rollouts. The least a run needs is that for 0 steps plus room to re-run failed trials: `2 × ceil(runs × validation ÷ 20)` for each of the finalists + 1, and `2 × ceil(runs × test ÷ 20)` for each of the `tested`. The steps that fit are `ceil(floor((max rollouts − the least a run needs) ÷ (runs × train)) ÷ candidates)`: the search stops when what's left can't pay for one more candidate, and the last step may try fewer. With a tunable factory whose markers hold a `choice(...)` of 2 or more components, each round's first step tries one candidate, so more steps fit than that. Judge calls are about rollouts × evaluators. Neither the judge's nor Reflection's own calls count against the budget. Under best trade-offs (step 3, balancing without weighing), at least 5 finalists are re-checked, so the finalists default is 5 and `tested` is the finalists plus 1; say so in step 3's follow-up.

**Rules**, stated only when an answer breaks one: every setting is a whole number; steps are 0 or more, and 0 measures only the current agent (no new versions, no proposer model); the others are at least 1, and finalists at least 5 under best trade-offs. The max rollouts must at least cover the least a run needs (the optimizer refuses less). When a step limit is more than the steps that fit, say that the rollouts run out first, after about the steps that fit. When one step's trials (candidates × train cases × runs) exceed the trials at once, say that steps run in batches and take longer.

**Build.** The max rollouts are `--max-rollouts N` (always passed), steps `--max-iterations N` (only with a step limit), candidates `--proposals-per-iteration N`, runs `--runs-per-candidate N`, finalists `--n-finalists N` (only when it isn't 3, and never with a cautious pick, step 6), and trials at once `--max-workers N`. 0 steps passes no `--max-iterations` and no `--reflection-model`, but `--search "<hooks file>:baseline_only"` (step 7).

### 6. Review

**Skipping the confirmation.** When the run goes without questions (**Running without questions**, under **Steps**), don't wait for a reply here, but don't skip the review either: before starting the run, write the summary, every warning (the case counts, a constraint at or near the current agent's score or with no score known, the runs of each case), and the command as a message, say the run is starting, and only then go to step 7. A warning given only in the final report comes too late to act on. In a headless run, where nobody reads until the end, put the review's warnings first in the final report (step 8) instead.

Otherwise, show the setup as one summary, then the command, and ask the user to start or to name a row to change. For a row of steps 2 to 5, go back to that step (a tunable factory's tune row changes only in its markers: propose the code edit, and make it only when the user agrees); for a default row, take the new value here and check it against its rules below. Then show the review again.

For example (fill every row from this run's values; mark a row `(default)` only when it kept its default):

```text
optimize · capitals · 6 cases · 3 train / 2 validation / 1 test
  agent        strands-workspace/agent.py
  tune         instructions
  models       bedrock/global.anthropic.claude-sonnet-4-5…
  goal         the mean of every evaluator
  constraints  none
  cases        3 train / 2 validation / 1 test (default split)
  budget       ≤500 rollouts, no step limit, 4 candidates per step, 1 run, 3 finalists (default)
  parallel     4 trials at once (default)
  crashes      a crashed run scores the worst value (default)
  pick         the best average score of the 3 finalists (default)
You can change any row, including cases, crashes, and pick, before I start.
```

With constraints, their row also says how sure each must be (ex. `tokens at most 4000, 90% sure each holds (default)`). Under the summary, list every warning from steps 2 to 5 and from the rows below, and say that the model, judge, and proposer calls are paid and use the user's credentials; with no step limit, add that the run usually spends nearly all of the max rollouts, with a proposer call per step.

The default rows:

- **Cases**: train guides the search, validation picks the winner, and test checks it on cases nothing else saw. Default: half train, and half of the rest test; when that leaves fewer than 2 validation cases or, with test cases, fewer than 2 test cases, the nearest of: train 30%, 40%, 50%, 60%, 70%, or 80%, with the rest split half test, a third test, or no test, that doesn't. Show case counts: each share times the case count, rounded down, with the leftover cases going to the largest remainders. The user's own split may be shares or counts (ex. "4 train, 1 validation, 1 test"). Rules: at least 1 train and 1 validation case; with fewer than 2 validation or test cases, warn that the scores can be luck and say how many cases to add to reach 2 of each. With fewer than 10 train or 5 validation cases, warn that differences smaller than about a tenth of a score can be noise, so the winner may be luck; with constraints, add that a constraint is hard to confirm 90% sure on so few cases, so the current agent often ships. Say how many cases to add to reach 10 train and 5 validation (ex. 20 cases with the default split).
- **How sure each constraint must be**, only with constraints: default 90%; the user's own is above 0% and below 100%.
- **Crashes**: default, a crashed run scores each evaluator's worst value. The user may instead leave crashed runs out of the scores, or give them a score (ex. 0.5).
- **Pick**: how the winner is chosen from the finalists on validation. Default, the best average score. The user may instead pick cautiously, by a low estimate of each score that it reaches with a sureness above 50% and below 100% (ex. 75% or 90% sure).

**Build.**

- Cases: always `--split TRAIN,VALIDATION,TEST` with the shares the row shows, the default included (the optimizer's own default, 0.4 / 0.3 / 0.3, differs from this row's); validation `(1 - TRAIN) × (1 - test part)` and test `(1 - TRAIN) × test part`, summing to 1 within 1e-9, so write enough digits (ex. `0.6,0.266666666667,0.133333333333`); counts become shares by dividing by the case count.
- Sureness: `--feasibility-threshold X`, only when it isn't 0.9.
- Crashes: nothing by default, `--agent-failure-score none` to leave crashed runs out, `--agent-failure-score X` for a score.
- Pick: nothing by default; X% sure writes `final_pick` into the hooks file (step 7) with `q = 1 - X` (ex. 75% is 0.25) and N the finalists, and passes `--final-pick "<hooks file>:final_pick"`.
- Proposer model: `--reflection-model MODEL_ID` (default Bedrock `global.anthropic.claude-sonnet-5-5`), only when the user asks and a text knob is tuned (or the factory has a text marker).

The command is step 7's **Start**, with the flags of the **Build** notes of steps 2 to 6 and `--seed 0` after `--rich`.

### 7. Run

**Folders.** Two folders per run: the optimizer's run folder `<RUN_DIR>` (`run.json`, `events.jsonl`, the result), which the optimizer creates and names under `<runs>`, and the skill's folder `F` (`summary.txt`, `run.log`, `run.exit`, `tunable_agent.py`, the hooks file, the path of `<RUN_DIR>` in `run_dir`), with `<stamp>` the UTC time as `YYYYMMDD-HHMMSS`:

- For an experiment of a strands-dev workspace, where the Strands CLI keeps its runs: `F` is `strands-dev/experiments/<name>/optimizations/<stamp>/` and `<runs>` is `$F/record`.
- Otherwise: `<runs>` is the optimizer's default, `<project>/optimization/runs/` (the run passes no `--base-dir`), so a run sits where `strands-evals optimize` and `strands-evals ui` put and look for runs; `F` is `<project>/optimization/skill/<stamp>/`.

Suggest adding `optimization/` to the project's `.gitignore` when it's a git repository that doesn't ignore it. Every case needs a unique `name`: when one doesn't have it, write a copy with every case named (keep existing names, add `case-<n>` to the rest, `-2`, `-3` to repeats) as `$F/experiment.json` and run that.

**tunable_agent.py**, Python without its own tunable factory: copy [assets/tunable_agent.py](assets/tunable_agent.py) to `$F/tunable_agent.py` (`cp <skill_dir>/assets/tunable_agent.py "$F/tunable_agent.py"`) and edit the copy where its `# Edit:` comments say, with the knobs of step 2's **Build**; never edit the asset itself. Set `AGENT_FILE` to the agent file's absolute path and `CALL` to the position of the call that builds `agent` among the file's `create_harness` calls (0 when it has one); leave out what isn't tuned, with `SWITCHES = {}` when nothing is switched. Keep the file with the run folder: a resumed run loads it again.

The run passes `--agent "$F/tunable_agent.py:create_agent"`. A file with its own tunable factory passes `--agent <agent file>:<factory name>` instead, and a TypeScript file `--agent <agent file>` with its `--tune` flags.

**Hooks file**, only for 0 steps or a cautious pick: with the skill's `tunable_agent.py`, append the part each needs to it; otherwise write it as `$F/hooks.py`. `<hooks file>` in the flags is that file:

```python
from strands_evals.optimization import FinalPick, LowerBound, Schedule, Search


class BaselineOnly(Schedule):  # 0 steps: measure only the current agent
    name = "baseline_only"

    def next(self, ctx):
        return None


baseline_only = Search([], schedule=BaselineOnly())
final_pick = FinalPick(n_finalists=N, rule=LowerBound(q=Q))  # a cautious pick
```

**Start.** Only after the review (step 6) is written to the user, in a message of its own, also when running without questions. Run in the background from the project, with stdout, stderr, and the exit code in their own files:

```bash
cd <project> && touch "$F/run.started" && (strands-evals optimize run <experiment> \
  --agent <agent, as above> [--base-dir "$F/record"] --rich <flags of steps 2 to 6> \
  > "$F/summary.txt" 2> "$F/run.log"; echo $? > "$F/run.exit") &
```

`--base-dir "$F/record"` only in a strands-dev workspace. If you can't run a background process, run it in the foreground with a long timeout and show the output when it ends. `<RUN_DIR>` is the folder the optimizer creates under `<runs>` once the run starts, before it measures anything: within a few seconds of starting, before any other progress check, take it from `run.log`'s `strands-evals optimize: run folder <RUN_DIR>` line, else the folder `find <runs> -mindepth 1 -maxdepth 1 -type d -newer "$F/run.started"` lists (with several, the one whose `cli.json` `agent` is this run's). Write its absolute path to `$F/run_dir` right away (a run killed later is resumed through it) and tell the user.

**Progress**: until `$F/run.exit` exists, about once a minute, relay each new `run.log` line that starts with `strands-evals optimize:` (ex. `iteration N (<blocks>): P proposals, R rollouts`, with R that step's rollouts, so keep a running total from the baseline's train cases) and any warning or error, in plain words (ex. "step 2 done: 4 new versions tried, 27 of 500 rollouts so far"). Read new lines with `tail -n +<next line> "$F/run.log"`.

**Exit codes**, from `$F/run.exit`:

- **0**: the run finished. Go to step 8.
- **3 with a `strands-evals optimize: continue with ... resume <RUN_DIR>` line in `run.log`**: the search stopped on a proposer error (ex. Reflection's model failed) before picking a winner; the rollouts spent are kept. Show the error, and tell the user the run can continue where it stopped once the cause is fixed (ex. fresh credentials for the proposer's model). Don't start a new run.
- **3 without that line**: a setup or runtime failure. Show the error line. When it says the baseline's train cases are unmeasured or every baseline trial failed, the agent or a model service failed (ex. `NoCredentialsError`): ask the user to check that the agent answers a case and that the credentials are fresh, then run again in a new run folder. When the first error is `tunable_agent.py`'s `didn't build agent from create_harness call`, check `CALL` against the file, and that the file builds `agent` when imported (not only under `if __name__ == "__main__":`); fix `CALL` or ask the user, then run again in a new run folder.
- **2**: bad input (a flag, the experiment, a TypeScript `--tune` the agent's call can't take, or an agent the optimizer can't load: a `tunable_agent.py` whose markers fail, ex. a choice without its current value or a skills folder without a skill, the user's tunable factory whose markers fail, ex. a `JsonField` on a number without its `Int` or `Float`, or given a `JsonChoice`, a file whose `agent` no `createHarness` call built, or a TypeScript agent without Node.js 22.18 or `@strands-agents/harness`). Show the error, fix the value in its step or review row (or the marker in `tunable_agent.py`; in the user's factory, propose the marker fix and make it only when the user agrees), and show the review again.
- **130**, or the process is gone without writing `run.exit` (killed): tell the user the run stopped early and give its folder.

After a 3 with the `continue` line or a 130, say the user can ask to resume it (see "Resuming a run").

**The UI.** To browse a run, its progress included, start `strands-evals ui` from the project. It scans 3 folders deep by default, which finds `optimization/runs/<run>/`; when `<RUN_DIR>` sits deeper below the project (a strands-dev workspace: `strands-dev/experiments/<name>/optimizations/<stamp>/record/<run>/`, 7 deep), pass `--depth N` with N its number of folders below the project, or the run isn't listed.

### 8. Report

The report is `$F/summary.txt`, the optimizer's own summary: why the search stopped, what ships, every finalist on validation, the test check (the winner against the current agent on the test cases, with its 90% interval), the effects of each option, and the costs. In a headless run, start the report with the review's warnings (step 6), as warnings about this setup (nobody saw them before the run) and any change the skill made to the request (ex. also maximizing a guarded evaluator, step 4). When the current agent misses a constraint on validation, say the limit is above what it scores today and suggest its measured score minus 0.05 as the limit (a guard, step 4), not a search for a version that reaches it. Then give the user its path and the headline in plain words: the version that ships, or that none beat the current agent; whether the test gain is likely real (the whole interval above 0), likely worse (below 0), or may be luck (it includes 0; more test cases narrow it).

A run can exit 0 before its test check ran: skipped because the budget couldn't afford it, or waiting to re-run trials that hit infrastructure failures (ex. throttling). `jq -r '.test_reason // "done"' "<RUN_DIR>/result.json"` prints `done`, or why the test check didn't fully run. When it isn't `done`, tell the user in plain words: what ships is final; when the reason is the budget or trials waiting to re-run, resuming the run can still run the test check, with more rollouts allowed when the budget was the reason.

Nothing in the user's project changes. `<RUN_DIR>/best.diff` holds each changed text (ex. `instructions`) as a diff under its option name, and `<RUN_DIR>/best.values.json` each changed choice as `{"old": ..., "new": ...}`. A skill's rewritten wording is under its `SKILL.md` path in the skills folder, a factory's changed `JsonField` (a number or choice included) is a hunk on its JSON file's path in `best.diff` and under its key in `best.values.json`, and a switch turned off (ex. `skill_maps` false) means leaving that entry out of the call's list. Apply them to the agent's `create_harness` or `createHarness` call (and its skill and JSON files) only when the user asks, and suggest evaluating the agent again after.

## Resuming a run

Resume only when the user explicitly asks; never on your own. A run can be resumed when it was killed (130, or no `run.exit`), stopped on a proposer error (3 with the `continue` line), or finished before its test check ran (step 8); any other failed run starts over as a new run. Skip steps 1 to 6: take the folder the user names (the skill's `F`, or the run folder `<RUN_DIR>`), else the newest resumable run of the experiment's `optimizations/` (or `<project>/optimization/skill/`), and confirm it with the user. `<RUN_DIR>` is the path in `$F/run_dir`, or in a strands-dev workspace the one folder under `$F/record` (step 7); without one the run never started, so there is nothing to resume. Given a `<RUN_DIR>`, `F` is the skill folder whose `run_dir` names it; a run started without the skill has none, so write its logs to a new `<project>/optimization/skill/<stamp>/`. With `<project>` as in step 1, run:

```bash
cd <project> && (strands-evals optimize resume "<RUN_DIR>" --rich \
  [--max-rollouts N] [--max-iterations N] [--max-workers N] \
  > "$F/summary.txt" 2>> "$F/run.log"; echo $? > "$F/run.exit") &
```

Those three are the only settings that may change on resume (a larger `--max-rollouts` lets a skipped test check run). Finished trials aren't run again. Follow it as in step 7 and report as in step 8.
