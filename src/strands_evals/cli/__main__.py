"""Allow `python -m strands_evals` to invoke the CLI."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
