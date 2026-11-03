"""Offline coverage for source projects and captured user-turn replay."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_evals.cli._project import ProjectError, ProjectRunner, evaluate_project, load_project_cases
from strands_evals.cli.main import main
from strands_evals.evaluators.deterministic import Equals


@pytest.fixture
def project(tmp_path):
    source = tmp_path / "agent.py"
    source.write_text(
        "from pathlib import Path\n"
        "from helper import suffix\n"
        "class Fake:\n"
        '    system_prompt = "original"\n'
        "    def __init__(self): self.turns = []\n"
        "    def __call__(self, turn):\n"
        '        assert Path("marker").is_file()\n'
        "        self.turns.append(turn)\n"
        '        return "|".join(self.turns) + suffix\n'
        "agent = Fake()\n"
    )
    (tmp_path / "helper.py").write_text('suffix = "!"\n')
    (tmp_path / "marker").touch()
    cases = tmp_path / "cases"
    cases.mkdir()
    for number in (1, 2):
        (cases / f"{number}.json").write_text(
            json.dumps(
                {
                    "id": str(number),
                    "assertion": "one|two!",
                    "userTurns": [{"content": "one"}, {"content": "two"}],
                    "trajectory": {"turns": [{"role": "assistant", "content": "NEVER REPLAY"}], "spans": []},
                    "metadata": {"test": True},
                }
            )
        )
    return source, cases


def test_fresh_multiturn_source_and_environment(project, monkeypatch):
    source, directory = project
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "before")
    runner = ProjectRunner(source, collector_endpoint="http://localhost:4318")
    cwd = Path.cwd()
    path = sys.path[:]
    cases = load_project_cases(directory)
    assert runner.get_instructions() == "original"
    report = evaluate_project(runner, cases, [Equals()])
    assert report.test_passes == [True, True]
    assert Path.cwd() == cwd
    assert sys.path == path
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "before"
    assert runner.run(cases[0], {"instructions": "changed"})["output"] == "one|two!"


def test_run_json_and_failure(project, capsys):
    source, directory = project
    assert main(["run", "--project", str(source), "--cases", str(directory), "--evaluator", "equals"]) == 0
    assert json.loads(capsys.readouterr().out)["test_passes"] == [True, True]
    assert main(["run", "--project", str(source), "--cases", str(directory), "--contains", "absent"]) == 1
    assert json.loads(capsys.readouterr().out)["test_passes"] == [False, False]
    assert main(["run", "--project", str(source), "--cases", str(directory), "--evaluator", "contains"]) == 0


@pytest.mark.parametrize("mutation", ["empty", "file", "text", "duplicate", "turns", "trajectory", "assertion"])
def test_invalid_case_directory(project, mutation):
    _, directory = project
    first = directory / "1.json"
    if mutation == "empty":
        for path in directory.iterdir():
            path.unlink()
    elif mutation == "file":
        directory = first
    elif mutation == "text":
        (directory / "bad.txt").touch()
    else:
        data = json.loads(first.read_text())
        if mutation == "duplicate":
            data["id"] = "2"
        elif mutation == "turns":
            data["userTurns"] = [{"content": 1}]
        elif mutation == "trajectory":
            data["trajectory"] = []
        else:
            data["assertion"] = ""
        first.write_text(json.dumps(data))
    with pytest.raises(ProjectError):
        load_project_cases(directory)


def test_source_failure_and_validation(project, capsys):
    source, directory = project
    source.write_text('raise RuntimeError("source failed")\n')
    assert main(["run", "--project", str(source), "--cases", str(directory), "--evaluator", "equals"]) == 1
    assert "source failed" in capsys.readouterr().err
    with pytest.raises(ProjectError, match="HTTP"):
        ProjectRunner(source, collector_endpoint="not-a-url")
    with pytest.raises(ProjectError, match="entry point"):
        ProjectRunner(directory)


def test_external_runner_protocol_and_inspect(project, monkeypatch):
    source, directory = project
    requests = []

    def run(argv, **kwargs):
        requests.append(json.loads(kwargs["input"]))
        assert kwargs["env"]["OTEL_TRACES_EXPORTER"] == "otlp"
        assert kwargs["timeout"] == 300
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"instructions": "prompt"}
                if requests[-1].get("action") == "inspect"
                else {"output": "one|two!", "trajectory": []}
            ),
            stderr="",
        )

    monkeypatch.setattr("strands_evals.cli._project.subprocess.run", run)
    runner = ProjectRunner(source, runner=sys.executable, collector_endpoint="http://localhost:4318")
    assert runner.get_instructions() == "prompt"
    assert runner.run(load_project_cases(directory)[0], {"instructions": "next"})["output"] == "one|two!"
    assert requests[-1]["user_turns"] == ["one", "two"]
    assert requests[-1]["overrides"] == {"instructions": "next"}


def test_create_harness_reconstruction(project, monkeypatch):
    source, directory = project
    source.write_text('from strands_harness import create_harness\nagent = create_harness(instructions="domain")\n')
    calls = []

    def create_harness(**kwargs):
        calls.append(kwargs)
        return type(
            "Fake",
            (),
            {
                "system_prompt": "framework: " + kwargs["instructions"],
                "__call__": lambda self, prompt: self.system_prompt,
            },
        )()

    module = SimpleNamespace(create_harness=create_harness)
    monkeypatch.setitem(sys.modules, "strands_harness", module)
    runner = ProjectRunner(source)
    assert runner.get_instructions() == "domain"
    assert (
        runner.run(load_project_cases(directory)[0], {"instructions": "replacement"})["output"]
        == "framework: replacement"
    )
    assert module.create_harness is create_harness
    assert calls[-1] == {"instructions": "replacement"}


def test_package_imports_and_restore_on_error(tmp_path, monkeypatch):
    package = tmp_path / "local_project"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "helper.py").write_text('prompt = "package prompt"\n')
    source = package / "agent.py"
    source.write_text(
        "from .helper import prompt\n"
        "class Fake:\n"
        "    system_prompt = prompt\n"
        '    def __call__(self, turn): raise RuntimeError("failed")\n'
        "agent = Fake()\n"
    )
    from strands_evals.case import Case

    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "original")
    original = Path.cwd()
    runner = ProjectRunner(source, collector_endpoint="http://localhost:4318/v1/traces")
    assert runner.get_instructions() == "package prompt"
    with pytest.raises(RuntimeError, match="failed"):
        runner.run(Case(name="failure", input=["hello"]))
    assert Path.cwd() == original
    assert os.environ["OTEL_TRACES_EXPORTER"] == "original"


def test_external_runner_error_and_bad_protocol(project, monkeypatch):
    source, directory = project
    case = load_project_cases(directory)[0]
    runner = ProjectRunner(source, runner=sys.executable)
    monkeypatch.setattr(
        "strands_evals.cli._project.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=7, stdout="", stderr="broken source"),
    )
    with pytest.raises(ProjectError, match="broken source"):
        runner.run(case)
    monkeypatch.setattr(
        "strands_evals.cli._project.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout='{"output": 42}', stderr=""),
    )
    with pytest.raises(ProjectError, match="string output"):
        runner.run(case)
