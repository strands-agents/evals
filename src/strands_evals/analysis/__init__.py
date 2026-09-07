"""Analysis utilities for evaluation reports.

This module provides tools for analyzing evaluation results at the report
level, complementing the per-session detectors module.
"""

from .failure_cohorts import CohortAnalysis, FailureCohort, analyze_failure_cohorts, print_cohort_summary

__all__ = [
    "FailureCohort",
    "CohortAnalysis",
    "analyze_failure_cohorts",
    "print_cohort_summary",
]
