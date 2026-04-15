"""Shared constants for detectors."""

# Default model for all detectors
DEFAULT_DETECTOR_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Token estimation and context window
CHARS_PER_TOKEN = 3
DEFAULT_MAX_INPUT_TOKENS = 200_000

# Fallback multiplier used only until the runtime calibration completes.
# Once calibrate_token_ratio() runs, this is replaced by the measured ratio.
TIKTOKEN_FALLBACK_MULTIPLIER = 1.4

PREFLIGHT_SAFETY_MARGIN = 0.85
CHUNK_SAFETY_MARGIN = 0.70

# Failure detector chunking
CHUNK_OVERLAP_SPANS = 5
CHUNK_MIN_NEW_CONTENT_RATIO = 0.5
MIN_CHUNK_SIZE = 2
