"""Allow `python -m strands_evals.cli` to invoke the CLI.

`python -m strands_evals` also works via the top-level
:mod:`strands_evals.__main__` shim, which delegates here.
"""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
