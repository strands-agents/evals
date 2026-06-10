"""Command-line interface for strands-agents-evals.

Exposes the `strands-evals` console script. Each subcommand is a thin
wrapper over the public library API in :mod:`strands_evals`.
"""

from .main import main

__all__ = ["main"]
