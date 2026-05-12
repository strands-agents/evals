"""Shared constants for detectors."""

# Default model for all detectors
DEFAULT_DETECTOR_MODEL = "global.anthropic.claude-sonnet-4-6"

# Token estimation and context window
CHARS_PER_TOKEN = 3
DEFAULT_MAX_INPUT_TOKENS = 200_000

PREFLIGHT_SAFETY_MARGIN = 0.65
CHUNK_SAFETY_MARGIN = 0.500

# Failure detector chunking
CHUNK_OVERLAP_SPANS = 5
CHUNK_MIN_NEW_CONTENT_RATIO = 0.5
MIN_CHUNK_SIZE = 2

# Map the LLM's categorical confidence strings to numeric values.
CONFIDENCE_MAP: dict[str, float] = {"low": 0.5, "medium": 0.75, "high": 0.9}

# RCA pruning and chunking
RCA_MAX_DESCENDANTS = 10
RCA_WINDOW_SPLIT_FACTOR = 3
RCA_MIN_WINDOW_SIZE = 5
