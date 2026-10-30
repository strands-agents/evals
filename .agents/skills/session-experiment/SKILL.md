---
name: session-experiment
description: Use when the user wants to turn past local agent conversations (Claude Code, Kiro CLI, Strands CLI, or any other harness) into strands-evals experiments. Locates the relevant session files on the local filesystem and extracts their episodes (turns with a focused user goal). Skip for traces stored in trace providers (CloudWatch, Langfuse, OpenSearch).
version: 0.1.0
---

# session-experiment

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

**Other harnesses**: find the harness's session store (its docs, or recently modified `*.json` / `*.jsonl` files under its home directory) and apply the same prompt criteria.

List the matching session file paths, grouped by harness. Include their conversation id, harness type, and session timestamp. If nothing matches, say which locations you searched and what criteria you applied and any near misses, then ask the user whether to widen the search.

### 2. Process sessions

Delegate the session files from step 1 to subagents that follow [assets/segmenter.md](assets/segmenter.md). Give each subagent up to 10 session file paths (mutually exclusive between subagents), the segmenter document's absolute path, each session file's timestamp, and any content criteria from the prompt. If that would take more than 5 subagents, spawn 5 and split the files evenly among them. The user's prompt can override either cap. Wait for every subagent to complete before continuing to the next step.

If you cannot spawn subagents, follow the segmenter document's steps yourself for every session file.

## Output

Stop after step 2. Report the outputs from all subagent runs (or your own output from step 2 if no subagents were spawned). Converting these sessions into an experiment is not part of this skill yet, so don't write experiment code; ask the user what they want to do with the results you found.
