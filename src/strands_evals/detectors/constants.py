"""Shared constants for detectors."""

# Default model for all detectors
DEFAULT_DETECTOR_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Token estimation and context window
CHARS_PER_TOKEN = 3
DEFAULT_MAX_INPUT_TOKENS = 200_000

# Dual safety margins:
# - Pre-flight margin is generous: try direct analysis more often since it's
#   always higher quality. If the estimate is wrong, the try/except fallback
#   catches it and routes to chunking (cost: one wasted LLM call).
# - Chunk margin controls per-chunk token budget. Higher = more context per
#   chunk (better quality) but more overflow risk. 0.75 balances both.
PREFLIGHT_SAFETY_MARGIN = 0.85
CHUNK_SAFETY_MARGIN = 0.75

# Failure detector chunking
CHUNK_OVERLAP_SPANS = 5
CHUNK_MIN_NEW_CONTENT_RATIO = 0.5
MIN_CHUNK_SIZE = 5
