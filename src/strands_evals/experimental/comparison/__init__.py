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

from .pairing import PairedScore, PairedTrialScore, aggregate_trial_pairs, pair_report_trials, pair_reports
from .summary import EvaluatorSummary, EvaluatorVerdict, summarize_pairs

__all__ = [
    "EvaluatorSummary",
    "EvaluatorVerdict",
    "PairedScore",
    "PairedTrialScore",
    "aggregate_trial_pairs",
    "pair_reports",
    "pair_report_trials",
    "summarize_pairs",
]
