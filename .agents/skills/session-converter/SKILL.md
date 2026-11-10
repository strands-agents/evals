---
name: session-converter
description: Use when the user wants to turn past agent conversations (Codex, Claude Code, Kiro CLI, Strands CLI, or any other harness) stored in local session files into strands-evals experiments. Skip for traces stored in trace providers (CloudWatch, Langfuse, OpenSearch).
version: 0.1.0
---

# session-converter

Converts recorded local agent sessions into a strands-evals experiment.

## Steps

### 1. Ask clarifying questions

Ask each question below that the prompt doesn't answer, in one batch, with your harness's question tool if it has one, otherwise as plain-text questions with numbered options, and wait for the answers. If you can't wait for a reply (ex. a non-interactive run), state that you're assuming each question's first option and continue.

**Harnesses.** Ask when the prompt doesn't say which harnesses to search (ex. "my sessions from last week").

Question: Which agent harnesses' sessions should I include?
1. All (Recommended): every harness with sessions on this machine (ex. Claude Code, Codex, Kiro CLI, Strands CLI).
2. Only <current harness>: only sessions from the harness running this skill.

**Context-dependent inputs.** A context-dependent input is an episode's opening user message that points at a specific project, file, build, or earlier turn without naming it, so another agent can't tell what it's about (ex. "now do the same for the parser", "why is this test failing?", "what errors does the graph primitive throw?").

Question: How should I handle user messages that depend on outside context (ex. "this file", "that test")?
1. Fill (Recommended): rewrite the input to include the context it relies on, so it stands alone.
2. Drop: filter out episodes whose input depends on outside context.
3. Keep: keep the input as written.

**URL checks.** The validation step may request URLs to verify if they are still valid.

Question: Should I check that URLs named in the cases still resolve?
1. Check public (Recommended): request each public `https` URL and flag cases whose URLs are gone.
2. Check all: also request internal, localhost, and credential-bearing URLs (ex. intranet docs the agent under test can reach).
3. Skip: make no network requests.

### 2. Locate sessions

Find every session file on the local filesystem that matches the user's criteria. Don't open whole files to do it: session files can be large, so read metadata, use `ls -t`, `find -newermt`, `grep`, `jq`, or targeted reads. Read the criteria from the prompt:

- **Harness**: the harnesses named in the prompt or chosen in step 1.
- **Date range**: resolve relative dates in the user's local time zone; session timestamps are UTC. A session matches if any of its activity falls in the range; mtime alone isn't enough.
- **Workspace**: "here" means the current working directory; default to all workspaces.
- **Content**: defer filtering on conversation content to step 3.

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

### 3. Process sessions

Delegate the session files from step 2 to subagents that follow [assets/segmenter.md](assets/segmenter.md). Give each subagent up to 10 session file paths (mutually exclusive between subagents), the segmenter document's absolute path, each session file's timestamp, any content criteria from the prompt, and the context-dependent input handling from step 1 as fill, drop, or keep (map any other answer to the closest of the three). If that would take more than 5 subagents, spawn 5 and split the files evenly among them. The user's prompt can override either cap. Wait for every subagent to complete, then read the episode files whose paths they return before continuing to the next step.

If you cannot spawn subagents, follow the segmenter document's steps yourself for every session file.

If no episodes were returned, report the dropped reasons and ask the user whether to widen the search.

### 4. Suggest evaluators

Group the `failure`s from every episode's error analysis in step 3 into the failure modes below, assigning each failure to exactly one mode (the most specific that fits), and count the episodes that show each mode. Every evaluator runs on every case, so suggest an evaluator only when its failure mode shows up in at least 10% of all episodes. Map each failure mode to its evaluator:

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

### 5. Output the experiment file

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

Write one case per kept episode from step 3, and one evaluator entry per evaluator from step 4.

- `name`: a short kebab-case phrase generated from the episode's `summary`, unique within the file.
- `input`: the episode's user input.
- `session_file`: the file path of the episode's session.
- `conversation_id`: the conversation id listed for the episode's session file.
- `timestamp`: the episode timestamp.
- `episode_index`: the episode's 0-based position among the episodes returned for its session, as an integer.
- `evaluator`: the name of a chosen evaluator.

### 6. Validate the experiment file

Run the following validations on the generated experiment file:

**Secrets** Find and remove secrets. Use the following command to check for known token formats and credential assignments.

```bash
jq -r --arg p '\ba[ks]ia[0-9a-z]{16}|\bgh[pousr]_[a-z0-9]{36,}|\bgithub_pat_[a-z0-9_]{22,}|\bglpat-[a-z0-9_-]{20,}|\b[sr]k[-_][a-z0-9_-]{20,}|\baiza[0-9a-z_-]{35}|\bnpm_[a-z0-9]{36}|\bhf_[a-z0-9]{30,}|\bxox[abprs]-[a-z0-9-]{10,}|hooks\.slack\.com/services/[a-z0-9/]+|accountkey=[a-z0-9+/=]{20,}|\beyJ[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}|-----BEGIN [a-z ]*PRIVATE KEY( BLOCK)?-----|bearer [a-z0-9._~+/-]{16,}|authorization:\s*basic [a-z0-9+/=]{8,}|://[^/:@\s]+:[^/@\s]+@|\b(iqojb3jpz2lu|fwogzxivyxdz)[a-z0-9+/=]{20,}|\babsk[a-z0-9+/=]{40,}|\bxapp-[a-z0-9-]{20,}|\bya29\.[a-z0-9_-]{30,}|\bgocspx-[a-z0-9_-]{20,}|\bwhsec_[a-z0-9]{24,}|\bsg\.[a-z0-9_-]{22}\.[a-z0-9_-]{43}|\bpypi-age[a-z0-9_-]{50,}|\bdo[por]_v1_[a-f0-9]{64}|\bshp(at|ca|pa|ss)_[a-f0-9]{32}|\blin_api_[a-z0-9]{40}|\bdapi[a-f0-9]{32}|[?&]sig=[a-z0-9%+/=]{20,}|(aws_secret_access_key|client_secret|secret_key|access_token|refresh_token|api[_-]?key|passw(or)?d)["\x27]?\s*[:=]\s*["\x27]?(?![<${(]|os\.environ)[^\s"\x27,]{8,}' '.cases[] | .name as $n | .. | strings | match($p; "gi") | "\($n): \(.string)"' <experiment_file>
```

Then, read each case for secrets the pattern doesn't cover (ex. API keys, passwords, or cookies assigned in config or prose, internal hostnames with credentials). Replace each secret value with `<redacted>` and keep the surrounding text. Leave clear placeholders (ex. `<your-api-key>`) as they are. Run the search again and confirm every remaining match contains `<redacted>` or is a placeholder you kept.

**Duplicates.** List the cases sorted by normalized input, so near-duplicates sit next to each other:

```bash
jq -r '.cases[] | [(.input | ascii_downcase | gsub("[^a-z0-9]+"; " ") | .[0:100]), .name] | @tsv' <experiment_file> | sort
```

Treat cases as duplicates when their inputs make the same request about the same target and differ only in wording, typos, or added context (ex. "can kiro cli spawn subagents" and "can kiro cli spawn subagents?"). Inputs that share wording but point at different files, URLs, or repos aren't duplicates. Read the full inputs of adjacent rows and of cases with similar names, since the sort only groups matching prefixes. In each group, keep the case whose input best stands alone, or the earliest `timestamp` on a tie, and remove the rest.

**Actionability.** Read each case's `input` and `expected_assertion` as the agent running it would, with only a shell and web access. Mark a case inactionable if it matches any of the following (but don't remove them):

- Needs a specific skill, MCP, data only the original user's session had (ex. a private dashboard, an earlier conversation), or resource modification (ex. writes files).
- A path, URL, or repo symbol it names is gone. Check with `test -e <path>`, `curl -sL -o /dev/null -w "%{http_code}" <url>`, or `grep -rlw "<name>" <repo>`. Follow the URL checks answer from step 1: only curl public `https` URLs for check public, curl every URL for check all, don't curl for skip.
- A claim it makes about a tool, file, or format no longer holds (ex. "Kiro's `session.json` has no workspace path field"). Check the current file (ex. `jq 'keys' <file>`).
- Points at a project, file, build, artifact, or earlier turn without naming it. Skip this check if the user chose to keep context-dependent inputs as-is.

**Structure.** Run `strands-evals validate <experiment_file>`. If it fails, fix the file and run it again until it passes.

### 7. Handoff to human

Report the experiment file path. List each suggested evaluator with its failure mode's episode count out of all episodes from step 3 and one `evidence` quote and name any modes that appeared but fell below 10%. Say how many secrets you redacted and in which cases, without repeating the values. List each removed duplicate by `name` with the case it duplicated. List each inactionable case by `name` with its reason, and recommend dropping it. Tell the user to run it with `strands-evals run <file> --agent <module>:<factory>`.

Delete all temp files or directories you created during this run (ex. the episode files from step 3). Don't delete the experiment file or the session files. Report how many files you deleted.

Ask the user to review the experiment file and whether they want any of the post-generation actions below (drop cases, smoke-test the experiment). Wait for their response. If you can't wait for a reply, keep every case, say which ones you'd drop, and skip the other actions.

## Post-generation

Run these only if the user asks for them after step 7.

### Drop cases

Remove the cases the user names (ex. the inactionable cases from step 6) from the experiment file. Then run `strands-evals validate <experiment_file>` again and fix any issues that occur.

### Smoke-test

Run the first case against a default Strands agent to check that the experiment runs end-to-end:

```bash
d=$(mktemp -d) && cat > "$d/default_agent.py" <<'EOF'
from strands import Agent

def build_agent() -> Agent:
    return Agent(callback_handler=None)
EOF
strands-evals run <experiment_file> --case 0 --agent "$d/default_agent.py:build_agent"
```

Report whether the run completed and each evaluator's result. A failing score is fine here; this tests the file, not the agent. If it errors (ex. missing model credentials), report the error. Then delete the temp directory.
