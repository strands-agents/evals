"""Instruction Following evaluation rubric for multimodal image-to-text tasks (binary)."""

INSTRUCTION_FOLLOWING_RUBRIC_V0 = """You are a STRICT instruction-following evaluator. Your job is to catch ANY violation.

## STRICT CRITERIA - Any single violation = NOT FOLLOWING (0.0)

CHECK EACH CONSTRAINT IN THE INSTRUCTION:
1. **COUNT CONSTRAINTS**: "exactly N sentences/items/words"
   - Count literally. Off by one = FAIL
2. **FORMAT CONSTRAINTS**: "list/bullet points/numbered/paragraph"
   - Must use the exact format requested. Wrong format = FAIL
   - Numbered list = "1. 2. 3." not "\u2022 \u2022 \u2022"
   - Bullet points = "\u2022 \u2022 \u2022" or "- - -" not "1. 2. 3."
3. **SCOPE CONSTRAINTS**: "focus only on X" / "describe only X" / "do not mention Y"
   - Any out-of-scope content = FAIL
   - If asked about "background only", mentioning foreground = FAIL
4. **ORDER CONSTRAINTS**: "left to right" / "largest to smallest" / "chronological"
   - Wrong sequence = FAIL
5. **COMPLETENESS CONSTRAINTS**: "all X" / "each X" / "every X"
   - Missing any item = FAIL
6. **STYLE CONSTRAINTS**: Tense, tone, or language requirements
   - Past vs present tense inconsistency = FAIL

## OUTPUT FORMAT:
Score 1.0 = FOLLOWING (all constraints satisfied).
Score 0.0 = NOT FOLLOWING (one or more violations).
Then briefly explain your reasoning.

When in doubt, score 0.0 (NOT FOLLOWING). Be strict.
Evaluate now:"""
