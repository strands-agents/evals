import importlib
import sys

import pytest


@pytest.mark.parametrize("missing", ["numpy", "scipy", "optuna"])
def test_importing_the_package_without_the_extra_names_the_install_command(monkeypatch, missing):
    for name in [
        m for m in sys.modules if m == "strands_evals.optimization" or m.startswith("strands_evals.optimization.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r"pip install 'strands-agents-evals\[optimization\]'"):
        importlib.import_module("strands_evals.optimization")
