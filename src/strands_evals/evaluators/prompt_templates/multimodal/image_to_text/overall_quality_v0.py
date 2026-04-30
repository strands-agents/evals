"""Overall Quality evaluation rubric for multimodal image-to-text tasks."""

OVERALL_QUALITY_RUBRIC_V0 = """You are evaluating the overall quality of an AI assistant's response to a user instruction about an image. Examine the image carefully before scoring.

Evaluate the response across FOUR dimensions, then provide an overall score:

1. VISUAL ACCURACY: Does the response correctly describe what is actually shown in the image?
2. INSTRUCTION ADHERENCE: Does the response directly address the user's question or instruction?
3. COMPLETENESS: Does the response cover the key aspects relevant to the instruction?
4. COHERENCE & HELPFULNESS: Is the response well-organized and genuinely useful?

Scoring Rubric:
0.0 (Very Poor) — Fails on most dimensions: factually wrong about the image, off-topic, and unhelpful.
0.25 (Poor) — Addresses the instruction weakly; contains multiple visual inaccuracies or significant omissions.
0.5 (Average) — Partially correct and on-topic but misses important details or contains minor errors about the image.
0.75 (Good) — Mostly accurate, relevant, and complete; may have minor imprecisions that do not materially affect quality.
1.0 (Excellent) — Accurate, comprehensive, directly addresses the instruction, and provides a clear and helpful answer grounded in the image.

**CRITICAL GUIDANCE FOR SCORES 0.75 AND 1.0:**
- A score of 1.0 should be awarded whenever the response correctly answers the core question and demonstrates solid understanding of the image content, even if it lacks minor details. Do not penalize a response for small omissions if the key information is present and correct.
- A score of 0.75 is for responses that are mostly correct but have one noticeable gap or minor inaccuracy.
- Avoid defaulting to 0.5 when a response is largely correct — if the answer addresses the main point accurately, it deserves 0.75 or 1.0.
- Reserve 0.5 for responses that are only partially correct or miss significant aspects of the question.
- Reserve 0.25 for responses with clear factual errors about image content or major omissions.
- Reserve 0.0 for responses that are fundamentally wrong, off-topic, or useless.

**IMPORTANT: When judging correctness, prioritize whether the response answers the user's actual question correctly. A response that reaches the right conclusion or gives correct key information should score high even if it lacks exhaustive detail. Do NOT downgrade a response simply because you can identify minor additional details that were omitted — only downgrade if those omissions materially harm the usefulness of the answer.**

--- CALIBRATION EXAMPLES ---

Example 1 (Score: 1.0):
Instruction: "What kind of chart is shown and what does it represent?"
Response: "The image shows a bar chart comparing quarterly revenue for three product lines (A, B, C) over 2024. Product A leads in Q3 with $4.2M, while Product C shows steady growth across all quarters."
Why 1.0: Correctly identifies the chart type, reads specific data points accurately from the image, and provides a useful summary.

Example 2 (Score: 0.5):
Instruction: "What kind of chart is shown and what does it represent?"
Response: "This is a bar chart showing some financial data over time."
Why 0.5: Correctly identifies the chart type but is vague — misses product lines, specific values, and the quarterly structure.

Example 3 (Score: 0.0):
Instruction: "What kind of chart is shown and what does it represent?"
Response: "The pie chart shows the distribution of customer demographics across different age groups."
Why 0.0: Misidentifies the chart type (pie vs bar) and hallucinates content not present in the image.

Example 4 (Score: 1.0):
Instruction: "Why would you suspect the person eating this is not diabetic?"
Response: "Based on the image, we would suspect the person eating this is not diabetic because the dessert appears to be high in sugar and carbohydrates — items a diabetic would typically avoid or consume in very limited quantities."
Why 1.0: Correctly identifies the key visual element (sugary dessert) and provides a logically sound, complete answer to the question. Even though more details could be added, the core answer is accurate and helpful.

Example 5 (Score: 0.75):
Instruction: "What material is the suitcase made of?"
Response: "The suitcase appears to be made of hard-shell plastic with a textured surface."
Why 0.75: Gets the key material correct. Minor uncertainty is acceptable. Does not need to enumerate every component to earn a high score.

--- END CALIBRATION EXAMPLES ---

Use the examples above to calibrate your scoring. First, briefly analyze the response (2-3 sentences). Then provide your score using 0.0, 0.25, 0.5, 0.75, or 1.0."""
