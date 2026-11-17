# segmenter

Processes the session files you were given, along with any content criteria and how to handle context-dependent inputs (fill, drop, or keep).

## Steps

### 1. Read sessions

For each session file you were given, extract every user and assistant turn as `{role, text, timestamp, position}`. `position` locates the turn's record in the file: its `jq` `input_line_number` for `.jsonl` files, or its `jq` path for `.json` files (ex. `.data.messages[3]`, `.traces[0].spans[2]`). Order turns by timestamp (keep turns with null timestamps in place). Skip tool calls, tool results, reasoning, and harness bookkeeping. Session files can be large, so stream them through `jq` instead of reading them whole. Derive path names with shell parameter expansion (`${f##*/}`) instead of `basename`.

Count the paths whose value is `"user"` or `"assistant"`. The most frequent path with both values is the role field.

```bash
jq -r 'path(.. | strings | select(. == "user" or . == "assistant")) | map(if type == "number" then "[]" else . end) | join(".")' <file> | sort | uniq -c | sort -rn | head -5
```

If nothing prints, the roles may sit inside JSON-encoded strings, so decode them first by piping the file through `jq -c 'walk(if type == "string" then (fromjson? // .) else . end)'`. Otherwise, the session may use other role names (ex. `human`, `ai`, `model`). Sessions may also name turns by field instead of by role value (ex. `user_prompt` and `agent_response` on an agent span) as well.

Tool results and bookkeeping messages may have the same roles, so discover the marker that sets apart human/assistant messages (ex. `origin.kind == "human"`). Read around the role matches to find the fields holding the message text and timestamp. The text may be a plain string or a list of blocks, since some sessions mix both. Using the discovered paths, build a filter: `select((<role field> == "user" or <role field> == "assistant") and <marker>) | {role: <role field>, text: <text field>, timestamp: <timestamp or null>, position: <position>}`. Discard records with no text content, tool-only messages, records marked as reasoning, and records that hold harness bookkeeping rather than text the user typed (ex. `isMeta` records, `<command-name>`, `<task-notification>`, `[Request interrupted by user` lines, `# AGENTS.md instructions`, `<environment_context>`). For slash commands, keep the input in `<command-args>`.

If records repeat earlier turns (ex. each model call replays the conversation history, or an agent span and the model calls nested under it hold the same exchange), keep each turn once. Take each turn from the outermost record (ex. outer agent span in a trace) that holds it, with that record's timestamp (ex. an agent span's `user_prompt` and `agent_response` with its start time).

If you were given content criteria (ex. a conversation topic), keep only the sessions that match them. Match conversation criteria against the extracted turns. Keep the user inputs in full and the first 500 characters of each assistant message for each session file.

### 2. Segment conversations

Split each session's turns from step 1 into episodes: turns that serve one user goal. Walk the user inputs in order and apply this rubric to each:

- **Starts a new episode**: a new task, an unrelated question, or a switch to a different artifact or topic.
- **Continues the episode**: a follow-up on the assistant's answer, clarification, correction, retry, approval, answer to the assistant's question, or a change in scope of the same deliverable. This includes inputs that can't stand alone (ex. "do it", "accurate?"), errors pasted from running the assistant's suggestion, requests to undo the assistant's change, and returning to the goal after a brief side question.
- **No goal**: drop episodes with nothing to evaluate: no request (ex. greetings, thanks), a request that only changes local setup (ex. "install the session-converter skill"), a request that depends on a specific skill, MCP, or tool another agent may not have, or a request to modify resources (ex. "add type hints to parser.py", "rebase onto main", "post the review"). Episodes requiring common tools, like file, shell, and web tools, may be kept unless they fall into the categories above.

### 3. Handle context-dependent inputs

A context-dependent input points at a specific project, file, build, IDE selection, or artifact without naming it (ex. "this repository", "the graph primitive", "what caused this?" on a pasted CI log, "#[[terminal]] why did this fail?"). Test: could an agent with only the input, a shell, and web access find what the request is about?

If context-dependent inputs should be dropped, drop episodes whose opening input is context-dependent and skip the rest of this step.

If context-dependent inputs should be filled, for each episode whose opening input is context-dependent or has IDE context attached (see below), find what it refers to, using only what the session recorded before that input or attached to it:

- **Earlier turns**: the turns from step 1 before the input. If an assistant message was cut at 500 characters, re-read it in full by its `position`.
- **Workspace path**: the directory the session ran in. Use it to make paths in the rewritten input absolute (ex. `<workspace>/tests/test_parser.py`), and to resolve references like "here". Look for a working-directory field (ex. `cwd`, `workspace`, `workspacePaths`) on the first records, or a span or resource attribute in traces (ex. `metadata.cwd`). If no record has one, check the file path and sibling metadata files; Strands CLI stores it in the file path before `/.agent/sessions/` or in `~/.strands/cli/session-roots.json`, while Kiro CLI stores it in the sibling `session.json`.
- **Tool results**: the prior tool calls and their results. Use `jq` to stream the records timestamped between the previous user input and this one (or between their `position`s if records have no timestamps), and look further back only if the reference isn't there. Calls (ex. `tool_use`, `toolUse`, `tool_call`, `function_call`) and results (ex. `tool_result`, `toolResult`, `function_call_output`) are usually linked by a shared id field (ex. `tool_use_id`, `toolUseId`, `toolCallId`, `call_id`). Find it by searching the file for one call's id (ex. `gen_ai.tool.call.id`, `metadata.tool_id`). In traces, they're usually tool-execution spans rather than message blocks (ex. `execute_tool` spans with `tool_call` and `tool_result`). Tool names differ between sources, so match on arguments like file paths and commands. If a result was truncated or offloaded, use the call's arguments to name what it refers to.
- **IDE context**: files, selections, or terminal output attached from an IDE. Look for attachments or context items (ex. `attachment`, `contextItems`, `selected_lines_in_ide`) on the input's record, or on records linked to it (ex. by `parentUuid` or `parentObservationId`).

Rewrite the input so it stands alone: replace references like "this file" with the absolute path, and add the earlier turn, file excerpt, or command output it relies on, trimmed to what the user had seen or pointed at when they wrote the input, not to what later turns showed mattered. Strip Kiro `#[[...]]` tokens. Keep the user's request and wording otherwise. If the session didn't record the context, keep the input as written.

## Output

Write a new temp file (`mktemp`) with the following format. Return the file path to the main agent.

```
Session: <session_file_path_0>
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

Session: <session_file_path_1>
<dropped_reason>
...

Episode 0
...
```

- `session_file_path_N`: processed session file's absolute path
- `dropped_reason`: if the session was dropped (ex. no episodes found, does not fit content criteria), explain why. Otherwise, omit it.
- `summary`: one line with the goal, what happened, and the outcome (resolved, abandoned, unresolved, or cut off mid-work).
- `timestamp`: ISO 8601 timestamp for the first user input in the episode, or session timestamp if unavailable.
- `input`: the opening user input, verbatim. If the input was rewritten in step 3, use that instead. If the user resent or rephrased it before the assistant replied, use the last version.
- `expected_assertion`: generate it from the episode's later user inputs and assistant messages, in order. State the end state that means the opening input's request succeeded, as checkable statements: user corrections, rejections, and requests to explain the assistant's wording reveal unmet expectations, and user acceptance confirms what was right. Include only what the opening input asked for; later inputs that add new questions or scope don't add statements. Ground every statement in the episode, and don't copy the assistant's answer.
- `failure`: one line per distinct way the assistant fell short of the user's goal in the episode, in your own words (ex. "claimed a function existed that wasn't in the file", "ignored the request to keep it short"). Describe what went wrong.
- `evidence`: the user input, quoted verbatim and trimmed to one sentence, that shows the failure.

If the user accepted the work without corrections, write `Error analysis: none` on a single line with no bullets. Episode numbering restarts at 0 for each session.
