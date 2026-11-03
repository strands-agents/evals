"""Offline project draft generation and backwards-compatible CLI parsing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from strands_evals.cli import _project_generation
from strands_evals.cli.commands import generate
from strands_evals.cli.main import _build_parser, main


@pytest.fixture
def generation(monkeypatch):
    inspector = Mock()
    inspector.return_value.get_generation_context.return_value = {
        "instructions": "Help users check deliveries.",
        "tools": [{"name": "track", "inputSchema": {"json": {"type": "object"}}}],
    }
    generator = Mock()
    generator.return_value.from_context_async = AsyncMock(
        return_value=SimpleNamespace(
            cases=[SimpleNamespace(input="Where is parcel 123?", expected_output="Uses tracking to report status.")]
        )
    )
    monkeypatch.setattr(_project_generation, "ProjectRunner", inspector)
    monkeypatch.setattr(_project_generation, "ExperimentGenerator", generator)
    return inspector, generator


def test_project_drafts_use_inspected_context(generation, capsys, tmp_path):
    inspector, generator = generation
    assert (
        main(["generate", "--project", str(tmp_path), "--runner", "runner", "--model", "model-id", "--num-cases", "1"])
        == 0
    )
    inspector.assert_called_once_with(str(tmp_path), runner="runner")
    inspector.return_value.get_generation_context.assert_called_once_with()
    generator.assert_called_once_with(input_type=str, output_type=str, model="model-id")
    call = generator.return_value.from_context_async.call_args.kwargs
    assert json.loads(call["context"]) == inspector.return_value.get_generation_context.return_value
    assert call["num_cases"] == 1
    assert "plain-language assertion" in call["task_description"]
    assert "one realistic user" in call["task_description"]
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "schemaVersion": 1,
        "drafts": [
            {
                "id": "draft-001",
                "assertion": "Uses tracking to report status.",
                "userTurns": [{"role": "user", "content": "Where is parcel 123?"}],
                "metadata": {"source": "strands-evals-generator", "reviewed": False},
            }
        ],
    }
    assert "unreviewed" in captured.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--model", ""],
        ["--model", "m", "--num-cases", "0"],
        ["--model", "m", "--num-cases", "51"],
        ["--model", "m", "--output", "cases.json"],
        ["--model", "m", "--num-topics", "2"],
        ["--model", "m", "--evaluator", "OutputEvaluator"],
        ["--model", "m", "--extra-information", "extra"],
        ["--model", "m", "--custom-evaluator", "mod:Class"],
    ],
)
def test_project_validation_precedes_all_calls(generation, flags, capsys):
    inspector, generator = generation
    assert main(["generate", "--project", "agent.py", *flags]) == 2
    inspector.assert_not_called()
    generator.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [SimpleNamespace(input="question", expected_output=None)],
        [SimpleNamespace(input="question", expected_output=" ")],
        [SimpleNamespace(input="", expected_output="assertion")],
        [SimpleNamespace(input=["turn"], expected_output="assertion")],
        [SimpleNamespace(input="question", expected_output={"assertion": "not plain text"})],
    ],
)
def test_malformed_drafts_fail_without_stdout(generation, cases, capsys):
    _, generator = generation
    generator.return_value.from_context_async.return_value = SimpleNamespace(cases=cases)
    assert main(["generate", "--project", "agent.py", "--model", "m", "--num-cases", "1"]) == 3
    assert capsys.readouterr().out == ""


def test_duplicate_prompts_rejected(generation, capsys):
    _, generator = generation
    generator.return_value.from_context_async.return_value = SimpleNamespace(
        cases=[
            SimpleNamespace(input="question", expected_output="assertion"),
            SimpleNamespace(input=" question ", expected_output="different assertion"),
        ]
    )
    assert main(["generate", "--project", "agent.py", "--model", "m", "--num-cases", "2"]) == 3
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "context",
    [
        {"instructions": "", "tools": []},
        {"instructions": "real", "tools": "bad"},
        {"instructions": "real", "tools": ["bad"]},
    ],
)
def test_bad_inspection_cannot_trigger_generation(generation, context, capsys):
    inspector, generator = generation
    inspector.return_value.get_generation_context.return_value = context
    assert main(["generate", "--project", "agent.py", "--model", "m"]) == 3
    generator.assert_not_called()
    assert capsys.readouterr().out == ""


def test_generation_failure_does_not_fabricate_drafts(generation, capsys):
    _, generator = generation
    generator.return_value.from_context_async.side_effect = ValueError("model refused")
    assert main(["generate", "--project", "agent.py", "--model", "m"]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model refused" in captured.err


def test_help_and_parsing_do_not_call_project_or_model(generation, capsys):
    inspector, generator = generation
    parser = _build_parser()
    args = parser.parse_args(["generate", "--project", "agent.py", "--model", "m"])
    assert args.num_cases == 5
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])
    assert exc.value.code == 0
    assert "stdout-only" in capsys.readouterr().out
    inspector.assert_not_called()
    generator.assert_not_called()


@pytest.mark.parametrize(
    "flags",
    [[], ["--context", "text", "--project", "agent.py"], ["--experiment", "source.json", "--project", "agent.py"]],
)
def test_required_exclusive_source(flags):
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["generate", *flags])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["context", "experiment"])
def test_legacy_generation_keeps_existing_flags(monkeypatch, capsys, mode):
    generator = Mock()
    experiment = Mock(cases=[1], evaluators=[])
    experiment.to_dict.return_value = {"cases": []}
    generator.return_value.from_context_async = AsyncMock(return_value=experiment)
    generator.return_value.from_experiment_async = AsyncMock(return_value=experiment)
    monkeypatch.setattr(generate, "ExperimentGenerator", generator)
    source = Mock()
    monkeypatch.setattr(generate.Experiment, "from_file", Mock(return_value=source))
    flags = (
        ["--num-topics", "3", "--evaluator", "OutputEvaluator"]
        if mode == "context"
        else ["--extra-information", "extra"]
    )
    assert main(["generate", f"--{mode}", "input", "--num-cases", "2", "--task-description", "task", *flags]) == 0
    assert json.loads(capsys.readouterr().out) == {"cases": []}
    generator.assert_called_once_with(input_type=str, output_type=str, model=None)
    method = getattr(generator.return_value, f"from_{mode}_async")
    assert method.call_args.kwargs["num_cases"] == 2
    assert method.call_args.kwargs["task_description"] == "task"


def test_runner_rejected_outside_project(generation, capsys):
    inspector, generator = generation
    assert main(["generate", "--context", "text", "--runner", "runner"]) == 2
    assert "--runner requires --project" in capsys.readouterr().err
    inspector.assert_not_called()
    generator.assert_not_called()


def test_inspection_and_generation_noise_stays_off_stdout(generation, capsys):
    inspector, generator = generation
    context = inspector.return_value.get_generation_context.return_value
    experiment = generator.return_value.from_context_async.return_value

    def inspect():
        print("project import noise")  # noqa: T201 - test stdout isolation
        return context

    async def generate_cases(**kwargs):
        print("generator noise")  # noqa: T201 - test stdout isolation
        return experiment

    inspector.return_value.get_generation_context.side_effect = inspect
    generator.return_value.from_context_async.side_effect = generate_cases
    assert main(["generate", "--project", "agent.py", "--model", "m", "--num-cases", "1"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["schemaVersion"] == 1
    assert "project import noise" in captured.err
    assert "generator noise" in captured.err


@pytest.mark.parametrize("count", [1, 50])
def test_project_case_count_bounds(generation, count):
    inspector, generator = generation
    inspector.return_value.get_generation_context.return_value["tools"] = []
    generator.return_value.from_context_async.return_value = SimpleNamespace(
        cases=[
            SimpleNamespace(input=f"Question {index}", expected_output=f"Asserts behavior {index}")
            for index in range(count)
        ]
    )
    drafts = _project_generation.generate_project_drafts("agent.py", None, count, "m", "Check edge cases")
    assert len(drafts["drafts"]) == count
    assert "Check edge cases" in generator.return_value.from_context_async.call_args.kwargs["task_description"]
