# segmenter

Processes the session files you were given, along with any content criteria and how to handle context-dependent inputs (fill, drop, or keep).

## Steps

### 1. Read sessions

For each session file you were given, use `jq` to extract the user inputs and assistant messages, as `{role, text, timestamp, position}`. `position` is the record's line number, or its index in `data.messages[]` for Strands CLI. Skip tool calls, tool results, reasoning, and harness bookkeeping. Session files can be large, so stream them through `jq` instead of reading them whole. Derive path names with shell parameter expansion (`${f##*/}`) instead of `basename`.

If the known fields below don't fit, print a condensed view of the file's shape:

```bash
jq -c '[paths(scalars) | map(if type == "number" then "[]" else . end) | join(".")] | unique' <file> | sort | uniq -c | sort -rn | head -20
```

From it, find the field that marks each record's type (user, assistant, tool, ...) and the field holding its text, then filter on them: `jq -c 'select(<is user or assistant text>) | {role: <role>, text: <text>}' <file>`. Known fields, best-effort:

- **Claude Code**: user is `type == "user"` and `origin.kind == "human"` (without `origin`, skip `isMeta`, `tool_result`-only, interrupt, and `<task-notification>` lines). Assistant is `type == "assistant"`. The text is in `message.content`, a string or `text` blocks. Slash command input is in `<command-args>`.
- **Strands CLI**: `data.messages[]` by `role`. The text is in `content[].text`; skip messages whose `content[]` has no `text` block (`toolUse`/`toolResult`-only messages are tool traffic, not turns).
- **Kiro CLI**: user is `payload.type == "user"`; assistant is `payload.type == "assistant"` with `payload.operationType == "Say"`. The text is in `payload.content`; keep leading `#[[...]]` tokens (ex. `#[[terminal]]`). This format is for `sess_*/messages.jsonl` files. The flat `cli/<session_uuid>.jsonl` store uses a different record shape; use the shape-discovery command above for it.
- **Codex**: select `type == "response_item"` records where `payload.type == "message"` and `payload.role` is `"user"` or `"assistant"`. The record `timestamp` is the turn timestamp. Join the `text` fields from `payload.content[]` blocks whose type is `"input_text"` or `"output_text"`; skip tool calls and all non-text content blocks. Ignore `"developer"` messages and any setup-only user entries that contain injected instructions or environment context rather than a user goal.

If you were given content criteria (ex. a conversation topic), keep only the sessions that match them. Match conversation criteria against the extracted turns. Keep the user inputs in full and the first 500 characters of each assistant message for each session file.

### 2. Segment conversations

Split each session's turns from step 1 into episodes: turns that serve one user goal. Walk the user inputs in session file order and apply this rubric to each:

- **Starts a new episode**: a new task, an unrelated question, or a switch to a different artifact or topic.
- **Continues the episode**: a follow-up on the assistant's answer, clarification, correction, retry, approval, answer to the assistant's question, or a change in scope of the same deliverable. This includes inputs that can't stand alone (ex. "do it", "accurate?"), errors pasted from running the assistant's suggestion, requests to undo the assistant's change, and returning to the goal after a brief side question.
- **No goal**: drop episodes with nothing to evaluate: no request (ex. greetings, thanks), a request that only changes local setup (ex. "install the session-converter skill"), a request that depends on a specific skill, MCP, or tool another agent may not have, or a request to modify resources (ex. "add type hints to parser.py", "rebase onto main", "post the review"). Episodes requiring common tools, like file, shell, and web tools, may be kept.

### 3. Handle context-dependent inputs

A context-dependent input points at a specific project, file, build, or artifact without naming it (ex. "this repository", "the graph primitive", "what caused this?" on a pasted CI log). Test: could an agent with only the input, a shell, and web access find what the request is about?

If context-dependent inputs should be dropped, drop episodes whose opening input is context-dependent and skip the rest of this step.

If context-dependent inputs should be filled, for each episode whose opening input is context-dependent or has IDE context attached (see below), find what it refers to, using only what the session recorded before that input or attached to it:

- **Earlier turns**: the turns from step 1 before the input. If an assistant message was cut at 500 characters, re-read it in full by its `position`.
- **Workspace path**: the directory the session ran in. Use it to make paths in the rewritten input absolute (ex. `<workspace>/tests/test_parser.py`), and to resolve references like "here". Known fields, best-effort:
  - **Claude Code**: `cwd` on the session's first record. Later records follow the agent's `cd` commands, so their `cwd` may be a subdirectory.
  - **Strands CLI**: the part of the session file path before `/.agent/sessions/`. Otherwise, the `workspace` of the `~/.strands/cli/session-roots.json` entry whose `directory` holds the session.
  - **Kiro CLI**: `workspacePaths` in the sibling `session.json`. It's empty for sessions under `_global/`.
  - **Codex**: `payload.cwd` in the `session_meta` record.
- **Tool results**: the prior tool calls and their results. Use the input's `position` to stream the records since the previous user input with `jq`, and look further back only if the reference isn't there. Tool names differ between harnesses, so match on arguments like file paths and commands. If a result was truncated or offloaded, use the call's arguments to name what it refers to. Known fields, best-effort:
  - **Claude Code**: calls are `message.content[]` blocks with `type == "tool_use"` (`name`, `input`, `id`). Results are `type == "tool_result"` blocks in user records, matched by `tool_use_id`, with the text in `content`.
  - **Strands CLI**: calls are `content[].toolUse` (`name`, `input`, `toolUseId`). Results are `content[].toolResult`, matched by `toolUseId`, with the text in `content[].text` or `content[].json`.
  - **Kiro CLI**: calls are `payload.type == "tool_call"` (`toolName`, `args`, `toolCallId`). Results are `payload.type == "tool_result"`, matched by `toolCallId`, with the text in `payload.content`.
  - **Codex**: calls are `response_item` records with `payload.type` `"function_call"` or `"custom_tool_call"` (`name`, `arguments`, `call_id`). Results are `"function_call_output"` or `"custom_tool_call_output"`, matched by `call_id`, with the text in `payload.output`.
- **IDE context**: context from an attached IDE. Known fields, best-effort:
  - **Claude Code**: IDE selections are `attachment.type == "selected_lines_in_ide"` records whose `parentUuid` is the `uuid` of the input's record (`displayPath`, `lineStart`, `lineEnd`, `content`).
  - **Kiro CLI**: IDE context is `payload.contextItems[]` on the user record (`name`, `uri`, `content`), marked inline in `payload.content` by a leading token like `#[[currentlyOpenFiles:active]]` or `#[[terminal]]`.

Rewrite the input so it stands alone: replace references like "this file" with the absolute path, and add the earlier turn, file excerpt, or command output it relies on, trimmed to what the user had seen or pointed at when they wrote the input, not to what later turns showed mattered. Keep the user's request and wording otherwise. If the session didn't record the context, keep the input as written.

## Output

Write each remaining episode in this format to a new temp file (`mktemp`), and return only the file's path to the main agent.

```
Session: <session_file_path>
<dropped_reason>

Episode 0
Summary: <summary>
Timestamp: <timestamp>
Input:
---
<input>
---
Expected assertion: <expected_assertion>
Error analysis:
- <failure>: "<evidence>"

Episode 1
...
```

- `session_file_path`: processed session file's absolute path
- `dropped_reason`: if the session was dropped (ex. no episodes found, does not fit content criteria), explain why. Otherwise, omit it.
- `summary`: one line with the goal, what happened, and the outcome (resolved, abandoned, unresolved, or cut off mid-work).
- `timestamp`: ISO 8601 timestamp for the first user input in the episode, or session timestamp if unavailable.
- `input`: the opening user input, verbatim, or as rewritten in step 3, without Kiro `#[[...]]` tokens. If the user resent or rephrased it before the assistant replied, use the last version.
- `expected_assertion`: generate it from the episode's later user inputs and assistant messages, in session file order. State what a successful response to the opening input must do, as checkable statements: user corrections, rejections, and requests to explain the assistant's wording reveal unmet expectations, and user acceptance confirms what was right. Include only what the opening input asked for; later inputs that add new questions or scope don't add statements. Ground every statement in the episode, and don't copy the assistant's answer.
- `failure`: one line per distinct way the assistant fell short of the user's goal in the episode, in your own words (ex. "claimed a function existed that wasn't in the file", "ignored the request to keep it short"). Describe what went wrong.
- `evidence`: the user input, quoted verbatim and trimmed to one sentence, that shows the failure.

If the user accepted the work without corrections, write `Error analysis: none` on a single line with no bullets.
