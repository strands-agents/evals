"""Reflection prompt template v0: the system prompt, the submit instructions, the task message, and the messages for
values over their limits and for a refused submission.

Uses `str.format` and plain Python for the parts that vary.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SYSTEM_PROMPT = """\
You are an expert in agent harnesses. You improve the texts an agent is built from, such as its system prompt and \
its tools' descriptions, so that the agent does better on the goal of the study that measures it.

You learn from two kinds of evidence:

1. **The versions measured so far**, ranked best first for the study's goal, each with its results over all of its \
conversations: what kinds of text have done well, and which have not. What a version costs, such as its tokens, is \
judged here, on whole versions.
2. **The current version's conversations**, each labeled by how well that one conversation did: where the current \
version's agent goes wrong, and what it does when it succeeds.

## Your Folders

Read only your trajectory folder and the versions folder, and write only in your working folder, `{work_folder}`: \
it is your shell's working directory, and also HOME and TMPDIR. Do not read or write anywhere else: other studies run \
on this machine, and their files are no evidence for this one.

## Context Window Management

Trajectory files and version files can be large. Before reading ANY file:
```bash
ls -lh <file_path>
```
- Files < 10KB: Safe to read fully
- Files > 10KB: Use `head`, `tail`, `grep` with limits

## Your Task

1. Compare the versions: what do the higher-ranked ones say, or leave out, that the lower-ranked ones do not?
2. Compare the current version's conversations that went well with those that did not: what goes wrong, and what \
text would have prevented it?
3. Write a new version of the parameters you change, each in full. You may add, rewrite, reorder, shorten or remove \
anything: keep what the evidence shows works, and change what it gives you a reason to change."""

# Appended to the system prompt, a replaced one included: how to hand the new values back.
SUBMIT_INSTRUCTIONS = """

## Submitting the Optimized Parameters

You have access to a `submit_optimized_params` tool. When you have finished
analyzing the traces and crafted the optimized parameters:

1. Write each optimized parameter to a separate file in your working folder using the shell tool
2. Call `submit_optimized_params(file_path_dict={"param_name": "<working folder>/<file>", ...})` to submit

Each file must contain ONLY the parameter value, nothing else.
Do NOT include the parameters in your response text — use the tool to submit them.

Example for system prompt optimization:
```
# Write the optimized prompt to a file in your working folder (the shell's working directory)
echo '...' > system_prompt.txt
# Submit it
submit_optimized_params(file_path_dict={"system_prompt": "<working folder>/system_prompt.txt"})
```"""

# Sent in the same conversation when a submitted value is over its limit, so the agent shortens what it wrote rather
# than starting again.
OVER_LIMIT = (
    "{what} over {whose} limit: {details}. A value over its limit is discarded and the current value kept. Shorten "
    "{it} to fit, keeping what matters most (check each file with `wc -c`), write {it} to {its} file in your working "
    "folder again, and submit with submit_optimized_params."
)

# Sent in the same conversation when the agent's turn ended with nothing accepted and its last submission refused
# (a value carrying a version's header): `refusal` is what the submit tool answered.
REFUSED = (
    "{refusal}\n\nYour turn ended with nothing submitted. Fix the file(s) in your working folder and submit again with "
    "submit_optimized_params."
)

_TASK_OPENING = """\
Write a new version of the parameters below, so that the agent does better on this study's goal. You have two kinds \
of evidence: the versions measured so far, and the current version's conversations.

## The versions measured so far

"""

_DIGEST = (
    "**Digest**: `{digest_file}` lists every conversation in the folder in one file: every case with a conversation "
    "that did not succeed, how each of those conversations ended and the evaluators' reasons, then the cases where "
    "every conversation succeeded. Read it first, then open the conversations it points you to (each file holds its "
    "messages once, in `response.messages`).\n\n"
)

_STEPS_1_TO_3 = """\
---

## Analysis Steps

**Step 1: Learn from the versions**

Read the table above. Compare the texts of higher-ranked versions with lower-ranked ones, and a version with the one \
it was built from: what do the versions that did well say, or leave out?
```bash
ls {versions_folder} | head -40
diff -r {versions_folder}/rank-001-version-* {versions_folder}/rank-002-version-*
```

**Step 2: Check the traces' sizes**
```bash
ls -lh {traces_folder}/*.json | head -20
```

**Step 3: Categorize the traces by their label**
"""

_LABELS_FROM_DIGEST = """\
```bash
cat {digest_file}
```
"""

_LABELS_FROM_TRACES = """\
```bash
for f in {traces_folder}/*.json; do
    echo -n "$f: "
    grep -oE '"reward":\\s*[0-9.-]+' "$f" | head -1
done 2>/dev/null | head -30
```
"""

_STEPS_4_ON = """
**Step 4: Examine sufficient traces**

| Folder Size | Minimum Coverage |
|-------------|------------------|
| ≤ 10 traces | Examine ALL |
| 11-30 traces | At least 70% |
| > 30 traces | At least 50% |

Balance coverage between the traces that went well and those that did not.

**Step 5: Extract insights**

Focus on:
- What do the versions that did well have in common, and what did the versions that did badly try?
- What do the current version's successful conversations do that the failed ones don't, and what mistakes do the \
failed ones make?
- What patterns appear across MULTIPLE traces (not one-off issues)?

Do not repeat a change the versions show did badly. Put each insight in the ONE parameter that reaches the agent when \
it matters: guidance about one item's selection, inputs or results belongs in that item's text; guidance about the \
whole task belongs in the text that governs the whole task.

**Step 6: Write and submit the new version**

Write each parameter you change in full: you may add, rewrite, reorder, shorten or remove anything. Change only what \
the evidence gives you a reason to change, and at least one parameter. Write each changed parameter's complete new \
value, and nothing else, to its own file in your working folder (to start from a version's text, copy its \
`<name>.txt`; a value that carries a version's header from `version.md` is refused):
```bash
cat > {work_folder}/optimized_<name>.txt << 'PARAM_EOF'
[the complete new value]
PARAM_EOF
```

Before submitting, check the length of every file whose parameter has a character limit, and shorten it until it \
fits:
```bash
wc -c < {work_folder}/optimized_<name>.txt
```
A value over its limit is discarded and the current value kept.

Then submit them under their exact names:
```
submit_optimized_params(file_path_dict={{"<name>": "{work_folder}/optimized_<name>.txt", ...}})
```

A parameter you do not submit keeps its current value.

---

## Criteria for Good Insights

### Actionability
Each insight should have:
- Clear **trigger condition** (WHEN...)
- Specific **action** (DO...)

Reject vague insights like "be careful" or "consider edge cases".

### Coverage
- Address different failure modes observed
- Prioritize patterns that appear in multiple traces

### Appropriate Generalization
- Keep actions concrete and specific
- Generalize conditions when pattern applies broadly
- The agent will meet tasks you have not seen: no rule that holds only for one task's names, ids or values

### A tool's description
Preserve its correct facts and, where the traces support it, cover: **use when** (the trigger, prerequisites, how it \
differs from similar items), **inputs** (meanings, formats, constraints) and **returns** (important fields, empty or \
partial results, side effects, recoverable errors). Keep it local to the tool: no multi-step task workflow, no \
unsupported guarantees."""


@dataclass(frozen=True)
class VersionRow:
    """One row of the versions table.

    Attributes:
        rank: 1 = best; None when the version can't be ranked (a result is missing).
        number: The version's display number.
        built_from: The display number of the version it was built from; None for the baseline.
        is_current: Whether its conversations are the ones in the trajectory folder.
        is_baseline: Whether it is the baseline.
        changed: What it changed from the version it was built from.
        cells: Each goal metric, formatted, in the table's column order: the fit's estimate and the measured mean
            (`0.84 (measured 0.9)`), or the measured mean alone.
        meets_limits: Whether it meets the constraints, as the ranking assessed it; None when unknown.
        conversations: How many of its conversations completed.
    """

    rank: int | None
    number: int
    built_from: int | None
    is_current: bool
    is_baseline: bool
    changed: Sequence[str]
    cells: Sequence[str]
    meets_limits: bool | None
    conversations: int


@dataclass(frozen=True)
class TextVariable:
    """One parameter the agent may rewrite.

    Attributes:
        name: Its key, the name it is submitted under.
        value: Its current value, in full.
        edit_instructions: What it is, for the agent; None gives "text the agent reads".
        size_cap: Its limit in characters; None for no limit.
    """

    name: str
    value: str
    edit_instructions: str | None
    size_cap: int | None


def system_prompt(work_folder: str) -> str:
    """The default system prompt, without the submit instructions."""
    return SYSTEM_PROMPT.format(work_folder=work_folder)


def task_message(
    *,
    versions: Sequence[VersionRow],
    versions_folder: str,
    goal_metrics: Sequence[str],
    limits: Sequence[str],
    traces_folder: str,
    digest_file: str | None,
    label_metrics: Sequence[str],
    label_ranked: bool,
    judged_by_version: Sequence[str],
    variables: Sequence[TextVariable],
    work_folder: str,
) -> str:
    """The task message.

    Args:
        versions: The versions table's rows, best first.
        versions_folder: The folder holding each version's parameters, one file per version.
        goal_metrics: The table's metric columns.
        limits: The constraints, as sentences; empty for none (then the table has no "Meets the limits" column).
        traces_folder: The folder holding the current version's conversations.
        digest_file: The digest of every conversation; None for none.
        label_metrics: The metrics a conversation's reward is computed from.
        label_ranked: Whether the reward is a rank among the current version's conversations.
        judged_by_version: Goal metrics left out of the reward, judged on whole versions instead.
        variables: The parameters the agent may rewrite.
        work_folder: The agent's working folder.
    """
    parts = [_TASK_OPENING]
    limits_note = f" (its limits: {'; '.join(limits)})" if limits else ""
    parts.append(
        f"{len(versions)} versions, best first for this study's goal{limits_note}. Each version's results are, for "
        "each metric, the study's estimate of its mean over all of its conversations, which also draws on the "
        "versions like it, then its measured mean (a version with no estimate shows its measured mean alone). The "
        "current version (marked ←) is the one whose conversations are in "
        f"the trajectory folder. Each version's parameters, in full, are in `{versions_folder}`: one folder per "
        "version, named by its rank, with its results in `version.md` and each parameter alone in a file of its own "
        "(`<name>.txt`, listed in `version.md`).\n\n"
    )
    limits_column = "Meets the limits | " if limits else ""
    parts.append(
        f"| Rank | Version | Built from | Changed | {''.join(f'{m} | ' for m in goal_metrics)}{limits_column}"
        "Conversations |\n"
    )
    parts.append(f"|---|---|---|---|{'---|' * len(goal_metrics)}{'---|' if limits else ''}---|\n")
    for v in versions:
        version = f"{v.number}{' ←' if v.is_current else ''}{' (baseline)' if v.is_baseline else ''}"
        built_from = "-" if v.built_from is None else v.built_from
        changed = ", ".join(v.changed) if v.changed else "-"
        meets = ("-" if v.meets_limits is None else "yes" if v.meets_limits else "no") + " | " if limits else ""
        parts.append(
            f"| {'-' if v.rank is None else v.rank} | {version} | {built_from} | {changed} | "
            f"{''.join(f'{c} | ' for c in v.cells)}{meets}{v.conversations} |\n"
        )
    parts.append(f"\n\n## The current version's conversations\n\n**Trajectory Folder**: {traces_folder}\n\n")
    if digest_file:
        parts.append(_DIGEST.format(digest_file=digest_file))
    ranked = ", ranked among the current version's conversations" if label_ranked else ""
    parts.append(
        "Each trace's `reward` labels that one conversation, from 0 (worst) to 1 (best): its "
        f"{' and '.join(label_metrics)}{ranked}."
    )
    if judged_by_version:
        one = len(judged_by_version) == 1
        parts.append(
            f" A conversation's {' and '.join(judged_by_version)} {'is' if one else 'are'} not in its label: compare "
            f"the versions for {'that' if one else 'those'}."
        )
    parts.append(f"\n\n## The parameters ({len(variables)})\n")
    for var in variables:
        what = var.edit_instructions or "text the agent reads"
        if var.size_cap:
            limit = f"at most {var.size_cap} characters (it is {len(var.value)} now)"
        else:
            limit = f"none (it is {len(var.value)} characters now)"
        parts.append(
            f'\n<parameter name="{var.name}">\n{var.value}\n</parameter>\n\n'
            f"- **What it is**: {what}\n- **Its limit**: {limit}\n"
        )
    parts.append("\n\n")
    parts.append(_STEPS_1_TO_3.format(versions_folder=versions_folder, traces_folder=traces_folder))
    if digest_file:
        parts.append(_LABELS_FROM_DIGEST.format(digest_file=digest_file))
    else:
        parts.append(_LABELS_FROM_TRACES.format(traces_folder=traces_folder))
    parts.append(_STEPS_4_ON.format(work_folder=work_folder))
    return "".join(parts)


def over_limit_message(over: Mapping[str, tuple[int, int]]) -> str:
    """The message for values over their limits: `over` maps a name to (its length, its limit)."""
    details = "; ".join(
        f"`{k}` is {n:,} characters, its limit {lim:,} ({n - lim:,} over)" for k, (n, lim) in over.items()
    )
    one = len(over) == 1
    names = ", ".join(f"`{k}`" for k in over)
    return OVER_LIMIT.format(
        what=f"{names} {'is' if one else 'are'}",
        whose="its" if one else "their",
        details=details,
        it="it" if one else "them",
        its="its" if one else "their",
    )
