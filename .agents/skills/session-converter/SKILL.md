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

If no episodes were returned, report the dropped reasons and ask the user whether to widen the search.

### 3. Suggest evaluators

Group the `failure`s from every episode's error analysis in step 2 into the failure modes below, assigning each failure to exactly one mode (the most specific that fits), and count the episodes that show each mode. Every evaluator runs on every case, so suggest an evaluator only when its failure mode shows up in at least 10% of all episodes. Map each failure mode to its evaluator:

- **GoalSuccessRateEvaluator**: the episode's `summary` outcome is abandoned, unresolved, or cut off mid-work, or the user needed several retries to reach the goal (ex. "still failing, third time I'm asking").
- **CorrectnessEvaluator**: the agent gave a wrong fact, fix, or computed result (ex. "that's not the default, it's 30 seconds").
- **HelpfulnessEvaluator**: the answer was accurate but didn't help, so the user re-asked or asked for more detail or examples (ex. "ok but how do I actually use it?").
- **FaithfulnessEvaluator**: the agent made claims that didn't match the files, docs, or tool results it read (ex. "there's no retry option in that file").
- **InstructionFollowingEvaluator**: the agent broke an explicit constraint in the user input (format, length, scope, things not to do) (ex. "I said don't touch the tests").
- **ConcisenessEvaluator**: users asked for shorter answers or cut off verbose ones (ex. "too long, just give me the command").
- **ResponseRelevanceEvaluator**: users had to redirect the agent back to what they asked (ex. "I asked about the parser, not the CLI").
- **CoherenceEvaluator**: the agent's explanation or reasoning contradicted itself or didn't follow logically (ex. "you just said the opposite above").
- **HarmfulnessEvaluator**: the agent produced or repeated harmful language, such as insults, hate, or violence (ex. "don't call my coworker an idiot in the reply").
- **StereotypingEvaluator**: the agent produced biased or stereotyped content (ex. "why did you assume the engineer is a he?").
- **ToolSelectionAccuracyEvaluator**: the agent chose the wrong tool or action (ex. "why did you run the whole test suite? I asked you to read the file").
- **ToolParameterAccuracyEvaluator**: the agent passed tool arguments it made up rather than took from the user or earlier results (ex. "where did that path come from? I never mentioned it"). Wrong-but-plausible arguments belong under GoalSuccessRate.
- **SkillSelectionAccuracyEvaluator**: the agent used the wrong skill (ex. "use the code-review skill, not implement").
- **SkillInstructionFollowingEvaluator**: the agent didn't follow the skill it used (ex. "the skill says to ask before posting").

Only suggest evaluators listed above. If no failure mode reaches 10% of episodes, suggest GoalSuccessRateEvaluator alone.

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
        "episode_index": <episode_index>
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
- `input`: the episode's user input.
- `session_file`: the file path of the episode's session.
- `conversation_id`: the conversation id listed for the episode's session file.
- `timestamp`: the episode timestamp.
- `episode_index`: the episode's 0-based position among the episodes returned for its session, as an integer.
- `evaluator`: the name of a chosen evaluator.

## Output

Report the experiment file path, the number of cases, and the total number of sessions. List each suggested evaluator with its failure mode's episode count and one `evidence` quote, and name any modes that appeared but fell below 10%. Tell the user to run it with `strands-evals run <file> --agent <module>:<factory>`.
