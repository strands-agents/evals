"""Shared constants for detectors."""

# Default model for all detectors
DEFAULT_DETECTOR_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Token estimation and context window
CHARS_PER_TOKEN = 4
CONTEXT_SAFETY_MARGIN = 0.75
DEFAULT_MAX_INPUT_TOKENS = 128_000

# Failure detector chunking
CHUNK_OVERLAP_SPANS = 2
MIN_CHUNK_SIZE = 5
