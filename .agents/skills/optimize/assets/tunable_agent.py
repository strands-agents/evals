"""A temporary tunable factory over AGENT_FILE, written by the optimize skill; the agent file isn't changed."""

import copy
import runpy
import shutil
import sys
import tempfile
import threading
import weakref
from contextvars import ContextVar
from pathlib import Path

import strands_harness
import strands_harness.agent
from strands_harness import defaults

from strands_evals.optimization import OneOf, SkillLibrary, TextValue, tool_descriptions, tunable

# Edit: the agent file, its call, and each plain knob's value in that call (SKILL.md step 2, Build).
AGENT_FILE = "/absolute/path/agent.py"
CALL = 0  # the create_harness call that builds `agent`, counted from 0 in the order the file makes them
CURRENT = {"instructions": "You answer geography questions.", "effort": "auto"}  # each plain knob's value in that call
if str(Path(AGENT_FILE).parent) not in sys.path:  # so the agent file's own imports of its sibling modules work
    sys.path.insert(0, str(Path(AGENT_FILE).parent))
_REAL = strands_harness.agent.create_harness
_LOCK = threading.Lock()  # create_harness is swapped while the file runs
_RUN = ContextVar("tunable_agent_run", default=None)  # (agents built so far, changes) while the file runs
_UNSET = {  # what the harness uses for a list argument the call leaves out
    "skills": True,
    "builtin_tools": list(defaults.DEFAULT_BUILTIN_TOOLS),
    "builtin_plugins": list(defaults.DEFAULT_BUILTIN_PLUGINS),
    "tools": [],
    "plugins": [],
    "mcp_servers": {},
}


class Edit:
    """A list argument's change: `source` replaces its value (ex. a skills copy), `drop` leaves out named entries."""

    def __init__(self, source=None, drop=()):
        self.source, self.drop = source, set(drop)

    def apply(self, value):
        value = self.source if self.source is not None else value
        if not self.drop:
            return value
        if isinstance(value, dict):
            return {key: entry for key, entry in value.items() if key not in self.drop}
        return [entry for entry in _entries(value) if _name(entry) not in self.drop]


def _entries(value):
    """A list argument's entries; a skills folder (or True: the default folder) as the skills it holds."""
    if value is True:
        value = defaults.DEFAULT_SKILLS_DIR
    if isinstance(value, (str, Path)):
        value = [value]
    entries = []
    for entry in value or []:
        folder = Path(entry).expanduser() if isinstance(entry, (str, Path)) else None
        if folder is not None and folder.is_dir() and not (folder / "SKILL.md").is_file():
            entries += sorted(str(skill.parent) for skill in folder.glob("*/SKILL.md"))
        else:
            entries.append(entry)
    return entries


def _name(entry):
    """An entry's name: a skill's folder, a built-in's name, or a tool's or plugin's own name."""
    if isinstance(entry, (str, Path)):
        return Path(entry).name
    for attribute in ("tool_name", "name", "__name__"):
        name = getattr(entry, attribute, None)
        if isinstance(name, str):
            return name
    return type(entry).__name__


def _fresh(kwargs, folder):
    """An empty session folder and a copy of the memory folder, so trials share no state."""
    for key in ("session", "memory"):
        value = kwargs.get(key, True)
        if value is False or value is None or not isinstance(value, (bool, dict)):
            continue  # off, or a manager object that every trial then shares
        config = dict(value) if isinstance(value, dict) else {}
        if key == "memory":
            source = Path(config.get("dir", defaults.DEFAULT_MEMORY_DIR)).expanduser()
            if source.is_dir():
                shutil.copytree(source, folder / key)
        kwargs[key] = {**config, "dir": str(folder / key)}
    kwargs.setdefault("callback_handler", None)  # Strands' default prints every streamed answer
    return kwargs


def _create_harness(**kwargs):
    """create_harness while the file runs; the real one for any other call (ex. a subagent's child agent)."""
    run = _RUN.get()
    if run is None:
        return _REAL(**kwargs)
    built, changes = run
    if len(built) == CALL:
        for key, value in changes.items():
            kwargs[key] = value.apply(kwargs.get(key, _UNSET.get(key))) if isinstance(value, Edit) else value
    folder = Path(tempfile.mkdtemp(prefix="tunable-agent-"))
    try:
        agent = _REAL(**_fresh(kwargs, folder))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    weakref.finalize(agent, shutil.rmtree, folder, True)  # also runs at exit
    built.append(agent)
    return agent


def _build(changes):
    """Run the agent file again with `changes` on its agent's create_harness call; return its `agent`."""
    built = []
    with _LOCK:
        token = _RUN.set((built, changes))
        strands_harness.create_harness = strands_harness.agent.create_harness = _create_harness
        try:
            module = runpy.run_path(AGENT_FILE, run_name="_tunable_agent_source")
        finally:
            strands_harness.create_harness = strands_harness.agent.create_harness = _REAL
            _RUN.reset(token)
    if len(built) <= CALL or module.get("agent") is not built[CALL]:
        raise RuntimeError(f"{AGENT_FILE} didn't build agent from create_harness call {CALL}")
    return module["agent"]


def _described(names):
    """The agent's tools named in `names`, its own and the built-in ones, from one build (no model call)."""
    tools = _build({}).tool_registry.registry
    missing = [name for name in names if name not in tools]
    if missing:
        raise RuntimeError(f"{AGENT_FILE}: the agent of create_harness call {CALL} has no tools named {missing}")
    return [tools[name] for name in names]


def _describe(agent, tools):
    """Give the agent's own copies of its tools the descriptions of `tools` (a tool switched off is skipped)."""
    registry = agent.tool_registry.registry
    for tool in tools:
        if tool.tool_name in registry:
            described = copy.copy(registry[tool.tool_name])  # a built-in tool may be shared between agents
            described.tool_spec = {**copy.deepcopy(described.tool_spec), "description": tool.tool_spec["description"]}
            registry[tool.tool_name] = described
    return agent


# Edit: one marker per knob (SKILL.md step 2, Build); drop the lines of knobs that aren't tuned.
MARKERS = {
    "instructions": TextValue(default=CURRENT["instructions"]),
    "effort": OneOf("auto", "low", "high", default=CURRENT["effort"]),
    "skills": SkillLibrary("/absolute/path/skills"),  # skill wording
}
# Edit: the tools whose descriptions may change, by name (the agent's own or built-in); () when none.
DESCRIBED = ("lookup",)
if DESCRIBED:
    MARKERS["tool_descriptions"] = tool_descriptions(*_described(DESCRIBED))
# Edit: one on/off switch per entry, as parameter -> (create_harness argument, entry name); every entry is on now.
SWITCHES = {"skill_geo": ("skills", "geo"), "skill_maps": ("skills", "maps"), "tool_lookup": ("tools", "lookup")}


@tunable(**MARKERS, **{param: OneOf(True, False, default=True) for param in SWITCHES})
def create_agent(**values):
    changes = {name: values[name] for name in CURRENT if name in values and values[name] != CURRENT[name]}
    if "skills" in MARKERS:
        changes["skills"] = Edit(source=str(values["skills"]))  # the candidate's copy of the skills folder
    for option in {option for option, _ in SWITCHES.values()}:
        off = {entry for param, (of, entry) in SWITCHES.items() if of == option and not values[param]}
        if off:  # every entry on builds the call as written
            changes.setdefault(option, Edit()).drop = off
    return _describe(_build(changes), values.get("tool_descriptions", []))


# Edit: append the hooks of SKILL.md step 7 here for 0 steps or a cautious pick.
