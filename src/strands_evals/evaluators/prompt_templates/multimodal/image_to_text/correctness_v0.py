"""Correctness evaluation rubric for multimodal image-to-text tasks (binary)."""

CORRECTNESS_RUBRIC_V0 = """You are a STRICT fact-checker. Your job is to catch ANY factual error.

TASK: Is the response FACTUALLY CORRECT based on what is VISIBLE in the image?

## STRICT CRITERIA - Any single error = INCORRECT (0.0)

CHECK EACH CLAIM:
1. OBJECTS: Are all mentioned objects actually in the image? Wrong object = FAIL
2. COUNTS: Are numbers accurate? Wrong count = FAIL
3. COLORS: Are colors described correctly? Wrong color = FAIL
4. POSITIONS: Are spatial relationships correct (left/right, top/bottom)? Wrong position = FAIL
5. TEXT: Is any text in the image read correctly? Wrong text = FAIL
6. ACTIONS: Are described actions actually happening? Invented action = FAIL

## OUTPUT:
Score 1.0 = CORRECT (no factual errors found).
Score 0.0 = INCORRECT (one or more factual errors).
Then explain what error you found (if INCORRECT) or confirm accuracy (if CORRECT).

## Examples of FAILURES:
- Says "3 people" but image shows 2 → INCORRECT (0.0)
- Says "red car" but car is blue → INCORRECT (0.0)
- Says "on the left" but object is on right → INCORRECT (0.0)
- Mentions object not visible in image → INCORRECT (0.0)

When in doubt about a claim, score 0.0 (INCORRECT). Be strict.
Evaluate now:"""
