"""Experimental utilities for comparing evaluation reports.

Install the optional `comparison` extra to use this package.
"""

try:
    from scipy.stats import PermutationMethod  # optional extra: comparison
except ImportError as error:
    raise ImportError(
        "strands_evals.experimental.comparison needs the optional dependencies; install them with "
        "pip install 'strands-agents-evals[comparison]'"
    ) from error

from .compare import PairedComparisonResult, PairedComparisonVerdict, compare_paired


def _validate_scipy_api() -> None:
    try:
        PermutationMethod(rng=0)
    except TypeError as error:
        raise ImportError(
            "strands_evals.experimental.comparison requires scipy>=1.15; install it with "
            "pip install 'strands-agents-evals[comparison]'"
        ) from error


_validate_scipy_api()

__all__ = [
    "PairedComparisonResult",
    "PairedComparisonVerdict",
    "compare_paired",
]
