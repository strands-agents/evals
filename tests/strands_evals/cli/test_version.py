"""Tests for ``strands-evals version``."""

from __future__ import annotations

from strands_evals.cli.main import main


def test_version_prints_package_and_python(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "strands-agents-evals" in captured.out
    assert "python" in captured.out
