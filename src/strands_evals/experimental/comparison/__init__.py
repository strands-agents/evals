"""Experimental utilities for comparing evaluation reports.

Install the optional `comparison` extra to use this package.
"""

try:
    import scipy  # noqa: F401  # optional extra: comparison
except ImportError as error:
    raise ImportError(
        "strands_evals.experimental.comparison needs the optional dependencies; install them with "
        "pip install 'strands-agents-evals[comparison]'"
    ) from error

from .pairing import PairedScore, pair_reports
from .summary import EvaluatorSummary, summarize_pairs

__all__ = [
    "EvaluatorSummary",
    "PairedScore",
    "pair_reports",
    "summarize_pairs",
]
