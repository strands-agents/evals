"""Tests for the anonymous usage reporting of the `strands-evals` console script."""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

import pytest

from strands_evals.cli import _common, _usage
from strands_evals.cli.main import _build_parser, main


class _Stream(io.StringIO):
    def __init__(self, interactive: bool) -> None:
        super().__init__()
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Turn reporting on, and collect each event instead of sending it."""
    monkeypatch.delenv("STRANDS_EVALS_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setattr(_usage, "cli_version", lambda: "0.4.2")

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(_usage, "_send", events.append)
    return events


@pytest.fixture
def reported(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Collect the (command, outcome) pair that `main` reports, without building an event."""
    pairs: list[tuple[str, str]] = []
    monkeypatch.setattr(_usage, "report", lambda command, outcome: pairs.append((command, outcome)))
    return pairs


def _subcommands() -> set[str]:
    """Read the subcommand names from the parser the CLI builds."""
    actions = [a for a in _build_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(actions) == 1
    return set(actions[0].choices)


class TestEnabled:
    def test_reporting_is_on_with_neither_switch_set(self):
        assert _usage.enabled({}) is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", " 0 "])
    def test_the_package_switch_turns_reporting_off(self, value: str):
        assert _usage.enabled({"STRANDS_EVALS_TELEMETRY": value}) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_the_package_switch_leaves_reporting_on(self, value: str):
        assert _usage.enabled({"STRANDS_EVALS_TELEMETRY": value}) is True

    # The convention is shared with other tools, so any value they give meaning to still means opt out here.
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "please"])
    def test_do_not_track_turns_reporting_off(self, value: str):
        assert _usage.enabled({"DO_NOT_TRACK": value}) is False

    @pytest.mark.parametrize("value", ["0", "", "false"])
    def test_do_not_track_set_to_off_leaves_reporting_on(self, value: str):
        assert _usage.enabled({"DO_NOT_TRACK": value}) is True

    def test_either_switch_is_enough(self):
        assert _usage.enabled({"STRANDS_EVALS_TELEMETRY": "1", "DO_NOT_TRACK": "1"}) is False


class TestCommandName:
    # The server holds the same list as an enum and rejects a name it does not know, so the copy here has to match
    # what the CLI registers. A new subcommand needs the server updated first.
    def test_the_command_list_holds_every_subcommand_the_cli_registers(self):
        assert _usage.COMMANDS == _subcommands()

    def test_every_registered_subcommand_reports_its_own_name(self):
        for command in _subcommands():
            assert _usage.command_name(command) == command

    @pytest.mark.parametrize("command", [None, "", "eval", "run --agent app:agent", "RUN"])
    def test_anything_else_reports_unknown(self, command: str | None):
        assert _usage.command_name(command) == "unknown"


class TestOutcomeFor:
    # EXIT_FAILURES means the evaluation found failures, which is a working CLI. Reporting it as `failed` would count
    # the results of an evaluation as a broken tool.
    def test_an_evaluation_that_found_failures_is_a_completed_cli_run(self):
        assert _usage.outcome_for(_common.EXIT_FAILURES) == "completed"

    def test_a_clean_run_completed(self):
        assert _usage.outcome_for(_common.EXIT_OK) == "completed"

    @pytest.mark.parametrize("exit_code", [_common.EXIT_BAD_INPUT, _common.EXIT_RUNTIME_ERROR])
    def test_a_cli_that_could_not_do_its_work_failed(self, exit_code: int):
        assert _usage.outcome_for(exit_code) == "failed"

    def test_an_exit_code_no_one_mapped_failed(self):
        assert _usage.outcome_for(130) == "failed"

    def test_an_exit_code_never_reports_an_outcome_the_server_rejects(self):
        for exit_code in range(-1, 256):
            assert _usage.outcome_for(exit_code) in _usage.OUTCOMES


class TestCliVersion:
    # The version of the package is dynamic, so a dev build or a release candidate carries a label the server rejects.
    # A prerelease label can also carry a branch, a user, or a host name.
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.4.2", "0.4.2"),
            ("0.4.2.dev3", "0.4.2"),
            ("1.2.3rc1", "1.2.3"),
            ("0.4.2+sha.3f8a1c", "0.4.2"),
            ("0.4.2.dev3+gabc1234.d20261124", "0.4.2"),
            ("0.4", "0.4.0"),
            ("01.02.03", "1.2.3"),
            ("1234.1234.1234", "1234.1234.1234"),
        ],
    )
    def test_a_version_is_normalized_to_major_minor_patch(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
    ):
        monkeypatch.setattr(_usage, "distribution_version", lambda _: raw)
        assert _usage.cli_version() == expected

    @pytest.mark.parametrize("raw", ["", "unknown", "12345.1.1", "v1.2.3", "1"])
    def test_a_version_with_no_numeric_prefix_is_not_reported(self, monkeypatch: pytest.MonkeyPatch, raw: str):
        monkeypatch.setattr(_usage, "distribution_version", lambda _: raw)
        assert _usage.cli_version() is None

    def test_a_package_that_is_not_installed_is_not_reported(self, monkeypatch: pytest.MonkeyPatch):
        def raise_not_found(_: str) -> str:
            raise _usage.PackageNotFoundError("strands-agents-evals")

        monkeypatch.setattr(_usage, "distribution_version", raise_not_found)
        assert _usage.cli_version() is None


class TestOsFamily:
    @pytest.mark.parametrize(
        ("system", "expected"),
        [("Linux", "linux"), ("Darwin", "macos"), ("Windows", "windows"), ("FreeBSD", "other"), ("", "other")],
    )
    def test_the_platform_maps_to_one_of_the_four_names_the_server_takes(
        self, monkeypatch: pytest.MonkeyPatch, system: str, expected: str
    ):
        monkeypatch.setattr(_usage.platform, "system", lambda: system)
        assert _usage.os_family() == expected


class TestBuildEvent:
    def test_the_event_holds_only_the_subcommand_the_outcome_and_the_context(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_usage, "cli_version", lambda: "0.4.2")
        monkeypatch.setattr(_usage.platform, "system", lambda: "Linux")

        event = _usage.build_event("run", "completed", stream=_Stream(interactive=False))

        assert event == {
            "v": 1,
            "event": "command/1",
            "context": {"cli_version": "0.4.2", "os": "linux", "interactive": False},
            "data": {"command": "run", "outcome": "completed"},
        }

    # The CLI already reads this to choose between its rich and its JSON output, so the value is exact rather than
    # inferred: it separates a person at a terminal from a pipeline without saying anything about who the caller is.
    @pytest.mark.parametrize("interactive", [True, False])
    def test_a_terminal_is_reported_from_the_stream_the_output_format_follows(
        self, monkeypatch: pytest.MonkeyPatch, interactive: bool
    ):
        monkeypatch.setattr(_usage, "cli_version", lambda: "0.4.2")
        event = _usage.build_event("run", "completed", stream=_Stream(interactive=interactive))

        assert event is not None
        assert event["context"]["interactive"] is interactive  # type: ignore[index]

    def test_no_event_is_built_when_the_version_is_not_reportable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_usage, "cli_version", lambda: None)
        assert _usage.build_event("run", "completed", stream=_Stream(interactive=False)) is None


class TestReport:
    def test_an_event_is_sent_once(self, sent: list[dict[str, Any]]):
        _usage.report("run", "completed", stream=_Stream(interactive=False))
        assert len(sent) == 1
        assert sent[0]["data"] == {"command": "run", "outcome": "completed"}

    @pytest.mark.parametrize("env", [{"STRANDS_EVALS_TELEMETRY": "0"}, {"DO_NOT_TRACK": "1"}])
    def test_a_switch_that_is_off_sends_nothing(self, sent: list[dict[str, Any]], env: dict[str, str]):
        _usage.report("run", "completed", env=env, stream=_Stream(interactive=False))
        assert sent == []

    def test_a_version_that_cannot_be_reported_sends_nothing(
        self, monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]]
    ):
        monkeypatch.setattr(_usage, "cli_version", lambda: None)
        _usage.report("run", "completed", stream=_Stream(interactive=False))
        assert sent == []

    # The caller is the exit path of the CLI, so no failure here may change what the user's command returns.
    @pytest.mark.parametrize("error", [OSError("connection refused"), TimeoutError(), ValueError("bad response")])
    def test_a_send_that_fails_is_dropped_and_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]], error: Exception, caplog
    ):
        def fail(_: dict[str, object]) -> None:
            raise error

        monkeypatch.setattr(_usage, "_send", fail)
        with caplog.at_level("DEBUG", logger="strands_evals.cli._usage"):
            _usage.report("run", "completed", stream=_Stream(interactive=False))

        assert "usage event dropped" in caplog.text

    def test_the_request_is_one_small_json_post_to_the_tenant_route(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STRANDS_EVALS_TELEMETRY", raising=False)
        monkeypatch.setattr(_usage, "cli_version", lambda: "0.4.2")

        requests: list[Any] = []

        def capture(request: Any, timeout: float) -> Any:
            requests.append((request, timeout))
            raise OSError("no network in a test")

        monkeypatch.setattr(_usage.urllib.request, "urlopen", capture)
        _usage.report("validate", "completed", stream=_Stream(interactive=False))

        request, timeout = requests[0]
        assert request.full_url == "https://telemetry.strandsagents.com/strands-evals/ping"
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert timeout == _usage.TIMEOUT_SECONDS
        # The route drops a body over 2 KiB before any validator sees it.
        assert len(request.data) < 2048
        assert json.loads(request.data)["event"] == "command/1"


class TestMainReports:
    @pytest.fixture(autouse=True)
    def _skip_the_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the subcommand itself, so these tests are about the reporting only."""
        self._exit_code = _common.EXIT_OK
        monkeypatch.setattr(_common, "run_command", lambda func, args: self._exit_code)

    @pytest.mark.parametrize(
        ("exit_code", "outcome"),
        [
            (_common.EXIT_OK, "completed"),
            (_common.EXIT_FAILURES, "completed"),
            (_common.EXIT_BAD_INPUT, "failed"),
            (_common.EXIT_RUNTIME_ERROR, "failed"),
        ],
    )
    def test_a_subcommand_that_ends_reports_its_name_and_its_outcome(
        self, reported: list[tuple[str, str]], exit_code: int, outcome: str
    ):
        self._exit_code = exit_code
        assert main(["validate", "experiment.json"]) == exit_code
        assert reported == [("validate", outcome)]

    # `ui` serves until someone stops it, so an end would never be reported. It reports once at start instead.
    def test_a_subcommand_that_serves_reports_that_it_started(self, reported: list[tuple[str, str]]):
        main(["ui", "experiments/"])
        assert reported == [("ui", "started")]

    def test_an_interrupt_is_reported_and_still_ends_the_process(
        self, monkeypatch: pytest.MonkeyPatch, reported: list[tuple[str, str]]
    ):
        def interrupt(func: Any, args: Any) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(_common, "run_command", interrupt)
        with pytest.raises(KeyboardInterrupt):
            main(["run", "experiment.json", "--agent", "app:agent"])

        assert reported == [("run", "interrupted")]

    def test_every_outcome_the_cli_reports_is_one_the_server_accepts(self, reported: list[tuple[str, str]]):
        main(["validate", "experiment.json"])
        main(["ui", "experiments/"])
        assert [outcome for _, outcome in reported] == ["completed", "started"]
        for _, outcome in reported:
            assert outcome in _usage.OUTCOMES

    # A nested subparser of its own would overwrite `args.command` if it used the same dest.
    @pytest.mark.parametrize(
        "argv",
        [["fetch", "langfuse", "--session-id", "s1"], ["optimize", "run", "experiment.json", "--agent", "app:agent"]],
    )
    def test_a_nested_subcommand_still_reports_the_top_level_name(
        self, reported: list[tuple[str, str]], argv: list[str]
    ):
        main(argv)
        assert reported == [(argv[0], "completed")]
