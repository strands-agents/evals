"""Shared constants for detectors."""

# Default model for all detectors
DEFAULT_DETECTOR_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Token estimation and context window
CHARS_PER_TOKEN = 3
DEFAULT_MAX_INPUT_TOKENS = 200_000

# Safety margins, expressed as a fraction of ``DEFAULT_MAX_INPUT_TOKENS``.
#
# ``estimate_tokens()`` uses the Strands SDK's tiktoken (cl100k_base) count,
# which systematically *underestimates* Anthropic Claude's actual tokenizer
# for JSON-heavy payloads by ~40%. These margins bake in that gap (via a
# ``1 / 1.4`` factor) so the effective budget stays inside the real context
# window. If the SDK ever switches to a Claude-native counter, bump these
# back up to ``0.85`` / ``0.70``.
#
#   ``0.85 / 1.4 ≈ 0.607``  → preflight threshold  (~121K cl100k ≈ 170K Claude)
#   ``0.70 / 1.4 ≈ 0.500``  → per-chunk budget     (~100K cl100k ≈ 140K Claude)
PREFLIGHT_SAFETY_MARGIN = 0.65
CHUNK_SAFETY_MARGIN = 0.500

# Failure detector chunking
CHUNK_OVERLAP_SPANS = 5
CHUNK_MIN_NEW_CONTENT_RATIO = 0.5
MIN_CHUNK_SIZE = 2

# Map the LLM's categorical confidence strings to numeric values.
# Ported verbatim from AgentCoreLens; any future prompt that changes the
# confidence alphabet must update this in lockstep.
CONFIDENCE_MAP: dict[str, float] = {"low": 0.5, "medium": 0.75, "high": 0.9}
