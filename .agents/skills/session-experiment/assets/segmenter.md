# segmenter

Processes the session files you were given, along with any content criteria.

## Steps

### 1. Read sessions

For each session file you were given, use `jq` to extract the user inputs and assistant responses, in order, as `{role, text}`. Skip tool calls, tool results, reasoning, and harness bookkeeping. Session files can be large, so stream them through `jq` instead of reading them whole. Derive path names with shell parameter expansion (`${f##*/}`) instead of `basename`.

If the known fields below don't fit, print a condensed view of the file's shape:

```bash
jq -c '[paths(scalars) | map(if type == "number" then "[]" else . end) | join(".")] | unique' <file> | sort | uniq -c | sort -rn | head -20
```

From it, find the field that marks each record's type (user, assistant, tool, ...) and the field holding its text, then filter on them: `jq -c 'select(<is user or assistant text>) | {role: <role>, text: <text>}' <file>`. Known fields, best-effort:

- **Claude Code**: user is `type == "user"` and `origin.kind == "human"` (without `origin`, skip `isMeta`, `tool_result`-only, interrupt, and `<task-notification>` lines). Assistant is `type == "assistant"`. The text is in `message.content`, a string or `text` blocks. Slash command input is in `<command-args>`.
- **Strands CLI**: `data.messages[]` by `role`. The text is in `content[].text`; skip messages whose `content[]` has no `text` block (`toolUse`/`toolResult`-only messages are tool traffic, not turns).
- **Kiro CLI**: user is `payload.type == "user"`; assistant is `payload.type == "assistant"` with `payload.operationType == "Say"`. The text is in `payload.content`. This format is for `sess_*/messages.jsonl` files. The flat `cli/<session_uuid>.jsonl` store uses a different record shape; use the shape-discovery command above for it.

If you were given content criteria (ex. a conversation topic), keep only the sessions that match them. Match conversation criteria against the extracted turns. Keep the user inputs in full and the first 500 characters of each assistant message for each session file.

### 2. Segment conversations

Split each session's turns from step 1 into episodes: turns that serve one user goal. Walk the user inputs in session file order and apply this rubric to each:

- **Starts a new episode**: a new task, an unrelated question, or a switch to a different artifact or topic.
- **Continues the episode**: a follow-up, clarification, correction, retry, approval, answer to the assistant's question, or a change in scope of the same deliverable. This includes inputs that can't stand alone (ex. "do it", "accurate?"), errors pasted from running the assistant's suggestion, requests to undo the assistant's change, and returning to the goal after a brief side question.
- **No goal**: drop episodes with nothing to evaluate: no request (ex. greetings, thanks), a request that only changes local setup (ex. "install the session-experiment skill"), or a request that depends on a specific skill, MCP, or tool another agent may not have. Episodes requiring common tools, like file, shell, and web tools, may be kept.

## Output

Return every episode to the main agent in this format.

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

Episode 1
...
```

- `session_file_path`: processed session file's absolute path
- `dropped_reason`: if the session was dropped (ex. no episodes found, does not fit content criteria), explain why. Otherwise, omit it.
- `summary`: one line with the goal, what happened, and the outcome (resolved, abandoned, unresolved, or cut off mid-work).
- `timestamp`: ISO 8601 timestamp for the first user input in the episode, or session timestamp if unavailable.
- `input`: the opening user input, verbatim. If the user resent or rephrased it before the assistant replied, use the last version.
- `expected_assertion`: generate it from the episode's later user inputs and assistant messages, in session file order. State what a successful response to the opening input must do, as checkable statements: user corrections, rejections, and requests to explain the assistant's wording reveal unmet expectations, and user acceptance confirms what was right. Ground every statement in the episode, and don't copy the assistant's answer.
