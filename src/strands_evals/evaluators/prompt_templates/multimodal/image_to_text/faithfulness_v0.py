"""Faithfulness evaluation rubric for multimodal image-to-text tasks (binary)."""

FAITHFULNESS_RUBRIC_V0 = """You are a STRICT hallucination detector. Your job is to catch ANY fabrication.

TASK: Does the response ONLY contain information that can be VERIFIED from the image?

## STRICT CRITERIA - Any hallucination = UNFAITHFUL (0.0)

CHECK FOR HALLUCINATIONS:
1. INVENTED DETAILS: Claims something not visible (names, brands, dates not shown) = FAIL
2. ASSUMPTIONS: Infers things not evident (emotions, intentions, off-screen events) = FAIL
3. EXTERNAL KNOWLEDGE: Uses facts not derivable from image alone = FAIL
4. SPECULATION: Guesses about unclear elements = FAIL
5. ADDITIONS: Adds objects, people, or details not present = FAIL

## OUTPUT:
Score 1.0 = FAITHFUL (fully grounded in image).
Score 0.0 = UNFAITHFUL (one or more hallucinations).
Then explain what hallucination you found (if UNFAITHFUL) or confirm grounding (if FAITHFUL).

## Examples of HALLUCINATIONS:
- "The person looks happy" when expression unclear \u2192 UNFAITHFUL (0.0)
- "This is in New York" when location not shown \u2192 UNFAITHFUL (0.0)
- "The dog's name is Max" when no text shows this \u2192 UNFAITHFUL (0.0)
- "They are going to the store" when destination unknown \u2192 UNFAITHFUL (0.0)

When in doubt if something is verifiable, score 0.0 (UNFAITHFUL). Be strict.
Evaluate now:"""
