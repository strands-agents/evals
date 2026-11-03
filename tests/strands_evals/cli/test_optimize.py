import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from strands_evals.case import Case
from strands_evals.cli._project import ProjectError
from strands_evals.cli.commands import optimize
from strands_evals.cli.main import main
from strands_evals.optimization.execution.status import InfrastructureError

AGENT = """\
from pathlib import Path


def create_agent(instructions=None, temperature=0.7):
    if instructions is None:
        instructions = Path(__file__).with_name("instructions.md").read_text(encoding="utf-8")

    class Echo:
        system_prompt = instructions

        def __call__(self, prompt):
            return "Paris" if "geography" in instructions and temperature < 0.5 else "unknown"

    return Echo()
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "agent"
    root.mkdir()
    (root / "agent.py").write_text(AGENT)
    (root / "instructions.md").write_text("You are a helpful assistant.\n")
    cases = tmp_path / "cases"
    cases.mkdir()
    for index in range(10):
        (cases / f"{index:02d}.json").write_text(
            json.dumps({"id": f"case-{index:02d}", "assertion": "Paris", "userTurns": [{"content": "Capital?"}]})
        )
    (tmp_path / "space.json").write_text(
        json.dumps(
            {
                "instructions": {"marker": "FileText", "path": "instructions.md"},
                "temperature": {"marker": "Float", "min_value": 0.0, "max_value": 1.0, "default": 0.7},
            }
        )
    )
    (tmp_path / "patches.json").write_text(
        json.dumps(
            [
                {"updates": {"instructions": "You are a geography assistant.\n"}, "rationale": "domain"},
                {
                    "updates": {"instructions": "You are a geography assistant.\n", "temperature": 0.2},
                    "rationale": "both",
                },
            ]
        )
    )
    return tmp_path


def arguments(project, *extra):
    return [
        "optimize",
        str(project / "agent"),
        "--cases",
        str(project / "cases"),
        "--space",
        str(project / "space.json"),
        "--evaluator",
        "contains",
        "--proposal-file",
        str(project / "patches.json"),
        *extra,
    ]


def test_dry_run_prints_the_plan_without_loading_source(project, monkeypatch, capsys):
    monkeypatch.setattr(
        "strands_evals.cli.commands.optimize.ProjectRunner", Mock(side_effect=AssertionError("must not load"))
    )

    assert main(arguments(project, "--dry-run", "--max-rollouts", "60")) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["rollouts"] == {
        "baseline": 4,
        "per_candidate": 4,
        "validation_reserve": 12,
        "test_reserve": 6,
        "search": 38,
        "max_candidates": 9,
    }
    assert plan["objective"] == "Maximize(Contains)"
    assert [param["key"] for param in plan["space"]] == ["instructions", "temperature"]
    assert "default" not in plan["space"][0] and plan["space"][0]["chars"] == 29
    assert not (project / "agent" / ".strands-evals").exists()


def test_optimizes_end_to_end_then_applies_the_pick(project, capsys):
    assert main(arguments(project, "--name", "demo")) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["improved"] is True
    assert result["best"]["values"] == {"instructions": "You are a geography assistant.\n", "temperature": 0.2}
    assert result["changed_values"] == {"temperature": {"old": 0.7, "new": 0.2}}
    run_dir = Path(result["run_dir"])
    assert run_dir.parent == project / "agent" / ".strands-evals" / "optimization" / "runs"
    assert run_dir.name.startswith("demo-")
    assert (run_dir / "best.diff").read_text() == result["diff"] and "+You are a geography" in result["diff"]

    assert main(["optimize", "--apply", str(run_dir)]) == 0

    applied = json.loads(capsys.readouterr().out)
    assert applied["manual"] == {"temperature": {"old": 0.7, "new": 0.2}}
    assert (project / "agent" / "instructions.md").read_text() == "You are a geography assistant.\n"


def test_resume_of_a_finished_run_returns_the_same_result(project, capsys):
    assert main(arguments(project)) == 0
    first = json.loads(capsys.readouterr().out)

    assert main(arguments(project, "--resume", first["run_dir"])) == 0

    assert json.loads(capsys.readouterr().out) == first


def test_resume_rejects_a_changed_objective(project, capsys):
    assert main(arguments(project)) == 0
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]

    code = main(arguments(project, "--resume", run_dir, "--objective", "minimize:seconds"))

    assert code == 2
    assert "objective changed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--resume", "RUN", "--split", "0.5,0.5"], "--split: read from run.json on resume"),
        (["--candidate", "abc"], "--candidate needs --apply"),
        (["--finalists", "0"], "--finalists must be at least 1"),
        (["--split", "0.5,0.6"], "sum to 1"),
        (["--objective", "maximize:tokens"], "not measured in project mode"),
        (["--max-rollouts", "10"], "below the 22 rollouts"),
        (["--crash-score", "bad"], "--crash-score"),
        (["--crash-score", "nan"], "finite number"),
        (["--name", "Bad Name"], "--name"),
    ],
)
def test_rejects_bad_flags_with_exit_code_2(project, capsys, extra, message):
    assert main(arguments(project, *extra)) == 2
    assert message in capsys.readouterr().err


def test_needs_a_space_and_cases(project, capsys):
    assert main(["optimize", str(project / "agent"), "--cases", str(project / "cases")]) == 2
    assert "--space FILE or --instructions-file FILE" in capsys.readouterr().err


def test_text_keys_need_a_proposer_for_a_real_run(project, capsys):
    command = [
        "optimize",
        str(project / "agent"),
        "--cases",
        str(project / "cases"),
        "--space",
        str(project / "space.json"),
    ]

    assert main(command) == 2
    assert "--proposal-model" in capsys.readouterr().err
    assert main([*command, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["warnings"]


def test_rejects_an_instructions_file_that_the_agent_does_not_use(project, capsys):
    (project / "agent" / "other.md").write_text("Something else.\n")
    command = arguments(project)
    command[command.index("--space") : command.index("--space") + 2] = [
        "--instructions-file",
        str(project / "agent" / "other.md"),
    ]

    assert main(command) == 2
    assert "does not match the instructions" in capsys.readouterr().err


def test_a_missing_optimization_extra_exits_3_with_the_install_command(project, monkeypatch, capsys):
    monkeypatch.setattr("strands_evals.cli._optimize.scoring.importlib.util.find_spec", lambda name: None)

    assert main(arguments(project)) == 3
    assert "strands-agents-evals[optimization]" in capsys.readouterr().err


def test_apply_rejects_project_flags(project, capsys):
    assert main(["optimize", str(project / "agent"), "--apply", "RUN"]) == 2
    assert "--apply takes only" in capsys.readouterr().err


KILLABLE_AGENT = """\
from pathlib import Path

HERE = Path(__file__).parent


def create_agent(instructions=None, temperature=0.7):
    if instructions is None:
        instructions = (HERE / "instructions.md").read_text(encoding="utf-8")

    class Echo:
        def __call__(self, prompt):
            counter = HERE / "calls.txt"
            count = int(counter.read_text()) + 1 if counter.exists() else 1
            counter.write_text(str(count))
            if (HERE / "kill").exists() and count == 6:
                raise KeyboardInterrupt
            return "Paris" if "geography" in instructions and temperature < 0.5 else "unknown"

    Echo.system_prompt = instructions
    return Echo()
"""


def test_resume_of_a_killed_run_finishes_it_and_logs_the_resume(project, capsys):
    (project / "agent" / "agent.py").write_text(KILLABLE_AGENT)
    (project / "agent" / "kill").touch()
    with pytest.raises(KeyboardInterrupt):
        main(arguments(project))
    (run_dir,) = (project / "agent" / ".strands-evals" / "optimization" / "runs").iterdir()
    (project / "agent" / "kill").unlink()
    capsys.readouterr()

    assert main(arguments(project, "--resume", str(run_dir))) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["improved"] is True
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("resume") == 1
    assert (run_dir / "effects.json").is_file()


def test_resume_rejects_a_changed_judge(project, capsys):
    assert main(arguments(project)) == 0
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]
    command = arguments(project, "--resume", run_dir)
    command[command.index("--evaluator") : command.index("--evaluator") + 2] = ["--contains", "Paris"]

    assert main(command) == 2
    assert "judge changed" in capsys.readouterr().err


def test_a_factory_without_defaults_skips_the_instructions_check(project, capsys):
    source = (project / "agent" / "agent.py").read_text()
    (project / "agent" / "agent.py").write_text(
        source.replace(
            "def create_agent(instructions=None, temperature=0.7):", "def create_agent(instructions, temperature):"
        )
    )

    assert main(arguments(project)) == 0

    captured = capsys.readouterr()
    assert "cannot compare instructions.md" in captured.err
    assert json.loads(captured.out)["improved"] is True


@pytest.mark.parametrize(
    ("runner", "message", "raised"),
    [
        ("/bin/runner", "Runner failed (1): ThrottlingException: slow down", InfrastructureError),
        ("/bin/runner", "Runner failed (1): ModelStreamErrorException", InfrastructureError),
        ("/bin/runner", "Runner failed (1): connect ECONNREFUSED 127.0.0.1:443", InfrastructureError),
        ("/bin/runner", "Runner failed (1): TypeError in the agent", ProjectError),
        (None, "Project does not support these runtime overrides", ProjectError),
    ],
)
def test_runner_text_marks_transient_failures_for_a_rerun(runner, message, raised):
    fake = Mock(runner=runner)
    fake.run.side_effect = ProjectError(message)

    with pytest.raises(raised):
        optimize._rollout(fake)({"instructions": "x"}, Case(name="one", input=["hi"]))


def test_resume_rejects_a_smaller_budget(project, capsys):
    assert main(arguments(project, "--max-rollouts", "100")) == 0
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]

    assert main(arguments(project, "--resume", run_dir, "--max-rollouts", "60")) == 2
    assert "at least the saved 100" in capsys.readouterr().err


def test_stdout_carries_only_the_json_when_the_agent_writes_to_file_descriptor_1(project, capfd):
    source = (project / "agent" / "agent.py").read_text()
    noisy = source.replace(
        "        def __call__(self, prompt):\n",
        "        def __call__(self, prompt):\n            __import__('os').write(1, b'noise from a tool\\n')\n",
    )
    assert noisy != source
    (project / "agent" / "agent.py").write_text(noisy)

    assert main(arguments(project)) == 0

    captured = capfd.readouterr()
    assert json.loads(captured.out)["improved"] is True
    assert "noise from a tool" in captured.err
