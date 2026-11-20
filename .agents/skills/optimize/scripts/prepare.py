"""Find the Strands CLI's evals bridge and a strands-workspace, list its experiments and knobs, and set up a run.

`inspect` prints the workspace's agent, experiments, and knobs as JSON. `setup` prepares the optimizer environment,
writes the run's space.json and task.py with the Strands CLI's own code, and prints the start of the command as JSON.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE = "strands-workspace"
DEV_FOLDER = "strands-dev"
AGENT_FILES = {"agent.py": "python", "agent.ts": "typescript"}
HARNESS_CALLS = {"python": "create_harness", "typescript": "createHarness"}
CLI_COMMANDS = ("strands-dev", "strands")
BRIDGE_FOLDER = Path("dist/src/dev")
BRIDGE_FILES = (
    "evals-agent/strands_dev_evals/harness.py",
    "evals-agent/strands_dev_evals/space.py",
    "evals-agent/entry-files.js",
    "temporary-evals-fork/typescript-runner-main.js",
    "temporary-evals-fork/knobs.js",
    "temporary-evals-fork/environment.js",
    "temporary-evals-fork/layout.js",
    "temporary-evals-fork/experiment-files.js",
    "temporary-evals-fork/evals-process.js",
)
PREVIEW = 80

# Runs in the Strands CLI's Node, so the knobs, the environment, and task.py come from the CLI's own code.
NODE_SCRIPT = r"""
import { existsSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

const reply = process.stdout.write.bind(process.stdout)
process.stdout.write = process.stderr.write.bind(process.stderr)
const input = JSON.parse(await new Promise((done) => {
  let text = ''
  process.stdin.setEncoding('utf8').on('data', (chunk) => (text += chunk)).on('end', () => done(text))
}))
const load = (path) => import(new URL(path, input.bridge).href)
const workspace = {
  directory: input.directory,
  root: input.root,
  entrypoint: input.entrypoint,
  config: { language: input.language },
}
const fail = (message) => {
  process.stderr.write(`error: ${message}\n`)
  process.exit(2)
}

const setup = async (knobs) => {
  const { temporarySpaceVariables } = await load('temporary-evals-fork/knobs.js')
  const tunable = knobs.filter((knob) => !knob.unavailable).map((knob) => knob.key)
  const unknown = input.tune.filter((key) => !tunable.includes(key))
  if (unknown.length > 0) fail(`cannot tune ${unknown.join(', ')}; the tunable knobs are ${tunable.join(', ')}`)
  const enabled = knobs.map((knob) => ({ ...knob, enabled: input.tune.includes(knob.key) }))
  const variables = temporarySpaceVariables(enabled, input.models)
  if (variables.length === 0) fail('the chosen knobs give nothing to tune; a model needs another model id to try')
  const layout = await load('temporary-evals-fork/layout.js')
  const { temporaryRunnableExperiment } = await load('temporary-evals-fork/experiment-files.js')
  const { prepareTemporaryOptimizerEnvironment } = await load('temporary-evals-fork/environment.js')
  const { temporaryTypescriptRunnerOf } = await load('temporary-evals-fork/evals-process.js')
  const { writeEvalsOptimizeEntry } = await load('evals-agent/entry-files.js')
  const python = await prepareTemporaryOptimizerEnvironment(
    { workspaceDirectory: input.directory, projectRoot: input.root, language: input.language },
    { onProgress: (line) => process.stderr.write(`${line}\n`) }
  )
  await layout.ensureTemporaryExperimentsFolder(input.directory)
  const home = layout.temporaryExperimentHome(input.directory, input.experiment)
  const files = layout.temporaryOptimizationFiles(home, layout.temporaryRunStamp(new Date()))
  if (existsSync(files.folder)) fail(`${files.folder} already exists; wait a second, then try again`)
  await mkdir(dirname(files.space), { recursive: true })
  await writeFile(files.space, `${JSON.stringify({ variables }, null, 2)}\n`)
  const experiment = resolve(await temporaryRunnableExperiment(input.experiment, files.copy))
  const flags = await writeEvalsOptimizeEntry(files.task, {
    agentFile: input.entrypoint,
    experimentFile: experiment,
    customEvaluators: [],
    ...temporaryTypescriptRunnerOf(workspace),
    spaceFile: files.space,
    settings: ['instructions'],
    baselineOnly: input.baselineOnly,
    ...(input.finalPick ? { finalPick: input.finalPick } : {}),
  })
  return { python, folder: files.folder, record: files.record, experiment, flags, variables }
}

try {
  const { readTemporaryKnobs } = await load('temporary-evals-fork/knobs.js')
  const knobs = await readTemporaryKnobs(workspace)
  const output = input.action === 'setup' ? await setup(knobs) : knobs
  reply(JSON.stringify(output))
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}
"""


class PrepareError(Exception):
    """A problem the user must fix; the message says how."""


def find_bridge() -> Path:
    """The Strands CLI's dist/src/dev folder, from the first CLI on PATH that ships every bridge file."""
    tried = []
    for command in CLI_COMMANDS:
        found = shutil.which(command)
        if found is None:
            continue
        bridge = Path(found).resolve().parent.parent / BRIDGE_FOLDER
        missing = [name for name in BRIDGE_FILES if not (bridge / name).is_file()]
        if not missing:
            return bridge
        tried.append(f"{command} at {bridge.parent.parent.parent} lacks {', '.join(missing)}")
    found_text = "; ".join(tried) if tried else f"none of {', '.join(CLI_COMMANDS)} is on PATH"
    raise PrepareError(
        f"no installed Strands CLI includes the evals bridge ({found_text}). "
        "Install a Strands CLI build that ships dist/src/dev/evals-agent and the TypeScript runner, then try again."
    )


def find_agent(workspace: Path) -> dict[str, str]:
    """The agent file, its root, and its language: from strands-dev/.config when present, else agent.py or agent.ts."""
    binding = workspace / DEV_FOLDER / ".config"
    if binding.is_file():
        try:
            agent = json.loads(binding.read_text(encoding="utf-8"))["agent"]
            root = (binding.parent / agent["path"]).resolve()
            entrypoint, language = root / agent["entrypoint"], agent["language"]
        except (ValueError, KeyError, TypeError) as e:
            raise PrepareError(f"{binding} has no agent path, entrypoint, and language ({e!r})") from e
        if language not in HARNESS_CALLS:
            raise PrepareError(f"{binding} names the language {language!r}, not python or typescript")
    else:
        present = [(workspace / name, kind) for name, kind in AGENT_FILES.items() if (workspace / name).is_file()]
        if len(present) != 1:
            names = " and ".join(AGENT_FILES)
            raise PrepareError(f"{workspace} needs exactly one of {names}, found {len(present)}")
        (entrypoint, language), root = present[0], workspace.resolve()
    if not entrypoint.is_file():
        raise PrepareError(f"the agent file {entrypoint} does not exist")
    if HARNESS_CALLS[language] not in entrypoint.read_text(encoding="utf-8"):
        raise PrepareError(f"{entrypoint} does not call {HARNESS_CALLS[language]}(...), which the optimizer reads")
    return {"file": str(entrypoint.resolve()), "root": str(root), "language": language}


def list_experiments(workspace: Path) -> list[dict[str, Any]]:
    """Each strands-dev/experiments/<name>/<name>.json with its case count and evaluator names."""
    folder = workspace / DEV_FOLDER / "experiments"
    found: list[dict[str, Any]] = []
    for home in sorted(folder.iterdir()) if folder.is_dir() else []:
        file = (home / f"{home.name}.json").resolve()
        if home.name.startswith(".") or not file.is_file():
            continue
        try:
            document = json.loads(file.read_text(encoding="utf-8"))
        except ValueError as e:
            found.append({"name": home.name, "file": str(file), "problem": f"not JSON ({e})"})
            continue
        evaluators = [evaluator.get("evaluator_type") for evaluator in document.get("evaluators") or []]
        found.append(
            {"name": home.name, "file": str(file), "cases": len(document.get("cases") or []), "evaluators": evaluators}
        )
    return found


def run_node(bridge: Path, workspace: Path, agent: dict[str, str], request: dict[str, Any]) -> Any:
    """Run NODE_SCRIPT with the CLI's modules and return its JSON output; its progress goes to stderr."""
    node = shutil.which("node")
    if node is None:
        raise PrepareError("node is not on PATH; the Strands CLI needs it")
    payload = {
        "bridge": bridge.as_uri() + "/",
        "directory": str((workspace / DEV_FOLDER).resolve()),
        "root": agent["root"],
        "entrypoint": agent["file"],
        "language": agent["language"],
        **request,
    }
    done = subprocess.run(
        [node, "--input-type=module", "-e", NODE_SCRIPT],
        input=json.dumps(payload),
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise PrepareError(f"the Strands CLI stopped with exit code {done.returncode}; its error is above")
    return json.loads(done.stdout)


def preview(knob: dict[str, Any]) -> dict[str, Any]:
    """The knob's fields worth showing, with long text cut; `on_by_default` is what the CLI turns on first."""
    shown = {key: knob[key] for key in ("key", "label", "kind", "detail", "unavailable", "choices") if key in knob}
    if "current" in knob:
        current = knob["current"]
        long = isinstance(current, str) and len(current) > PREVIEW
        shown["current"] = current[:PREVIEW] + "…" if long else current
    if "entries" in knob:
        shown["entries"] = [entry["label"] for entry in knob["entries"]]
    return {**shown, "on_by_default": knob["enabled"]}


def final_pick(text: str) -> dict[str, float]:
    finalists, _, lower_bound = text.partition(":")
    return {"finalists": int(finalists), "lowerBound": float(lower_bound)}


def optimizer(python: str, root: str) -> str:
    """`strands-evals optimize` started as the Strands CLI starts it: from the agent root, its env's bin on PATH."""
    region = os.environ.get("AWS_REGION")
    words = [
        f"cd {shlex.quote(root)} &&",
        f'PATH={shlex.quote(str(Path(python).parent))}:"$PATH"',
        *([f"AWS_DEFAULT_REGION={shlex.quote(region)}"] if region else []),
        f"{shlex.quote(python)} -m strands_evals.cli optimize",
    ]
    return " ".join(words)


def commands(setup: dict[str, Any], root: str) -> dict[str, str]:
    """The start of the run command, and the resume command missing only its run folder."""
    start = optimizer(setup["python"], root)
    run = [setup["experiment"], *setup["flags"], "--base-dir", setup["record"]]
    return {"command": f"{start} run {shlex.join(run)} --rich", "resume": f"{start} resume"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("inspect", "setup"))
    parser.add_argument("workspace", nargs="?", type=Path, default=Path(WORKSPACE), help=f"default ./{WORKSPACE}")
    parser.add_argument("--experiment", help="setup: an experiment name from inspect")
    parser.add_argument("--tune", action="append", default=[], help="setup: a knob key from inspect; repeat")
    parser.add_argument("--model", action="append", default=[], help="setup: another model id to try; repeat")
    parser.add_argument("--baseline-only", action="store_true", help="setup: 0 steps, measure only the baseline")
    parser.add_argument("--final-pick", type=final_pick, help="setup: FINALISTS:LOWER_BOUND, ex. 3:0.25")
    args = parser.parse_args()
    try:
        workspace: Path = args.workspace
        if not (workspace / DEV_FOLDER).is_dir() and not any((workspace / name).is_file() for name in AGENT_FILES):
            raise PrepareError(f"{workspace.resolve()} is not a strands-workspace")
        bridge = find_bridge()
        agent = find_agent(workspace)
        experiments = list_experiments(workspace)
        if args.action == "inspect":
            knobs = run_node(bridge, workspace, agent, {"action": "knobs"})
            report = {"cli": str(bridge.parent.parent.parent), "agent": agent, "experiments": experiments}
            sys.stdout.write(json.dumps({**report, "knobs": [preview(knob) for knob in knobs]}, indent=2) + "\n")
            return 0
        if args.experiment is None:
            raise PrepareError("setup needs --experiment NAME, a name from inspect")
        chosen = next((e for e in experiments if e["name"] == args.experiment), None)
        if chosen is None:
            names = ", ".join(e["name"] for e in experiments) or "none"
            raise PrepareError(f"no experiment named {args.experiment}; the workspace has {names}")
        request = {
            "action": "setup",
            "experiment": chosen["file"],
            "tune": args.tune or ["instructions"],
            "models": args.model,
            "baselineOnly": args.baseline_only,
            "finalPick": args.final_pick,
        }
        setup = run_node(bridge, workspace, agent, request)
        result = {key: setup[key] for key in ("python", "folder", "record", "variables")}
        sys.stdout.write(json.dumps({**result, **commands(setup, agent["root"])}, indent=2) + "\n")
        return 0
    except PrepareError as e:
        sys.stderr.write(f"prepare.py: error: {e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
