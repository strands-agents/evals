---
name: session-converter
description: Use when the user wants to turn past local agent conversations (Codex, Claude Code, Kiro CLI, Strands CLI, or any other harness) into strands-evals experiments. Locates the relevant session files on the local filesystem, extracts their episodes (turns with a focused user goal), suggests evaluators, and writes an experiment JSON file. Skip for traces stored in trace providers (CloudWatch, Langfuse, OpenSearch).
version: 0.1.0
---

# session-converter

Converts recorded local agent sessions into a strands-evals experiment.

## Steps

### 1. Locate sessions

Find every session file on the local filesystem that matches the user's criteria. Don't open whole files to do it: session files can be large, so read metadata, use `ls -t`, `find -newermt`, `grep`, `jq`, or targeted reads. Read the criteria from the prompt:

- **Harness**: default to all harnesses.
- **Date range**: resolve relative dates in the user's local time zone; session timestamps are UTC. A session matches if any of its activity falls in the range; mtime alone isn't enough.
- **Workspace**: "here" means the current working directory; default to all workspaces.
- **Content**: defer filtering on conversation content to step 2.

The following session file paths are best-effort and can change between releases. If a path is missing or looks different, check the linked doc.

**Claude Code** ([docs](https://code.claude.com/docs/en/claude-directory#application-data))
- Session file: `~/.claude/projects/<project>/<session_uuid>.jsonl`. `<project>` is the working directory with every non-alphanumeric character replaced by `-`. Pass `<project>` to tools as a full path, since a leading `-` parses as an option.
- Some records include activity times in their `timestamp` field.
- Ignore `*.orphaned-*.jsonl`, `*.jsonl.superseded-*`, and `<session_uuid>/subagents/*.jsonl`.

**Strands CLI** ([docs](https://github.com/strands-agents/harness-sdk/blob/main/strands-cli/README.md))
- Session file: `<directory>/<session_id>/scopes/*/*/snapshots/snapshot_latest.json`. Its `createdAt` and mtime show only the last activity.
- The oldest `snapshots/immutable_history/snapshot_<uuid7>.json` shows the first activity. When that directory is empty, treat the first activity as unknown.
- To find session directories, read `roots[]` in `~/.strands/cli/session-roots.json`. Each entry has a `directory` (where sessions live) and a `workspace` (the repo it ran in). The catalog is best-effort and can miss entries, so also check `<workspace>/.agent/sessions`.

**Kiro CLI** ([docs](https://kiro.dev/docs/cli/chat/session-management/))
- Session directory: `~/.kiro/sessions/<workspace_hash>/sess_<session_uuid>/`, holding the conversation in `messages.jsonl` and metadata in `session.json`. The metadata includes `workspacePaths` (on newer versions), `title`, `createdAt`, and `lastModifiedAt`.
- To search one workspace, compute its hash instead of scanning every directory:
  ```bash
  printf '%s' "<absolute_workspace_path>" | openssl dgst -sha256 -r | cut -c1-16
  ```
- Sessions started outside a workspace live under `~/.kiro/sessions/_global/`.
- Some versions also write a flat store: `~/.kiro/sessions/cli/<session_uuid>.jsonl` with a `<session_uuid>.json` metadata sidecar.

**Codex**
- Session file: `~/.codex/sessions/<year>/<month>/<day>/rollout-<timestamp>-<session_uuid>.jsonl`.
- Read the `session_meta` record to get `payload.session_id`, `payload.cwd`, and the initial `payload.timestamp`.
- Match a workspace against `payload.cwd`. `~/.codex/history.jsonl` and `~/.codex/state_*.sqlite` are history and local-state stores, not session transcripts.

**Other harnesses**: find the harness's session store (its docs, or recently modified `*.json` / `*.jsonl` files under its home directory) and apply the same prompt criteria.

List the matching session file paths, grouped by harness. Include their conversation id, harness type, and session timestamp. If nothing matches, say which locations you searched and what criteria you applied and any near misses, then ask the user whether to widen the search.

### 2. Process sessions

Delegate the session files from step 1 to subagents that follow [assets/segmenter.md](assets/segmenter.md). Give each subagent up to 10 session file paths (mutually exclusive between subagents), the segmenter document's absolute path, each session file's timestamp, and any content criteria from the prompt. If that would take more than 5 subagents, spawn 5 and split the files evenly among them. The user's prompt can override either cap. Wait for every subagent to complete before continuing to the next step.

If you cannot spawn subagents, follow the segmenter document's steps yourself for every session file.

### 3. Suggest evaluators

Read the strands-evals authoring skill at [../../../SKILL.md](../../../SKILL.md) to learn what each evaluator scores. If it is missing, read [../../../AGENTS.md](../../../AGENTS.md) instead.

Use the episode summaries from step 2 for their topics and their `expected_assertion`s for the user's intent. Every evaluator runs on every case, so pick 3-5 evaluators that fit the episodes taken together.:

- **GoalSuccessRateEvaluator**: most episodes work toward a goal over several turns (follow-ups, corrections, retries).
- **CorrectnessEvaluator**: most episodes ask for a verifiable answer (a fact, a fix, a computed result).
- **HelpfulnessEvaluator**: most episodes are open-ended requests for help, where the answer has no single right form.
- **FaithfulnessEvaluator**: users corrected claims that didn't match the files, docs, or tool results the agent read.
- **InstructionFollowingEvaluator**: user inputs carry explicit constraints (format, length, scope, things not to do).
- **ConcisenessEvaluator**: users asked for shorter answers or cut off verbose ones.
- **ResponseRelevanceEvaluator**: users had to redirect the agent back to what they asked.
- **CoherenceEvaluator**: episodes need long, multi-step explanations or reasoning.
- **HarmfulnessEvaluator**, **RefusalEvaluator**, **StereotypingEvaluator**: only when episodes touch sensitive content, or the agent refused or should have.
- **ToolSelectionAccuracyEvaluator**: users corrected which tool or action the agent chose (read the wrong file, ran the wrong command).
- **ToolParameterAccuracyEvaluator**: users corrected the arguments the agent used (wrong path, query, flag, or value).
- **SkillSelectionAccuracyEvaluator**, **SkillInstructionFollowingEvaluator**: episodes invoke skills, or users corrected which skill the agent used or how it followed one. Runs that invoke no skill score as not applicable.

Only CorrectnessEvaluator and GoalSuccessRateEvaluator read `expected_assertion` (reference mode); the rest need no expected values. Don't suggest:

- **ToolCalled**, **SkillInvoked**: they check one exact tool or skill name on every case, which rarely fits arbitrary episodes.
- **Contains**, **StartsWith**, **StateEquals**: they check the same value on every case, which rarely fits arbitrary episodes.
- **Equals**, **StructuredOutputSimilarity**: they need `expected_output` on every case, which this skill doesn't set.
- **OutputEvaluator**, **TrajectoryEvaluator**: they need a custom rubric; the built-in judges above cover the same ground.
- **InteractionsEvaluator**: it scores multi-agent handoffs, which single-agent sessions don't have.
- **Multimodal\*** evaluators: they need image inputs, and episode inputs are text.
- **FailureCommunicationEvaluator**, **PartialCompletionEvaluator**, **RecoveryStrategyEvaluator**: they score chaos runs with injected faults.
- **AttackSuccessEvaluator**: it scores red-team runs.

### 4. Output the experiment file

Write the experiment JSON file to the directory the user gives, defaulting to `./experiments`. Create the directory if it doesn't exist. Name the file `<timestamp>_<uuid>.json`, where `<timestamp>` is the current UTC time as `YYYYMMDDTHHMMSSZ` and `<uuid>` is a random UUID4.

```json
{
  "cases": [
    {
      "name": "<name>",
      "input": "<input>",
      "expected_assertion": "<expected_assertion>",
      "metadata": {
        "session_file": "<session_file>",
        "conversation_id": "<conversation_id>",
        "timestamp": "<timestamp>",
        "episode_index": "<episode_index>",
      }
    }
  ],
  "evaluators": [
    {"evaluator_type": "<evaluator>"}
  ]
}
```

Write one case per kept episode from step 2, and one evaluator entry per evaluator from step 3.

- `name`: a short kebab-case phrase generated from the episode's `summary`, unique within the file.
- `input`: episode's user input
- `session_file` the file path of the episode's session.
- `conversation_id`: the conversation id of the episode's session.
- `timestamp`: the episode timestamp.
- `episode_index`: the episode's index within its session.
- `evaluator`: the name of a chosen evaluator.

## Output

Report the experiment file path, the number of cases, and the total number of sessions.
