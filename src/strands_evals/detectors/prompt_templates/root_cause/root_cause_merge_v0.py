"""Root cause merge prompt template v0.

Uses Python f-string rendering for prompt assembly.
"""

SYSTEM_PROMPT = """\
# ROLE
You are an expert at synthesizing multiple partial root cause analyses into a comprehensive whole.

# TASK
You are provided with root cause analyses from different segments of a long agent execution trace. \
Your task is to merge these analyses into a single, coherent root cause analysis that captures all \
failures while eliminating duplicates."""

USER_PROMPT = "Merge and deduplicate the root causes from all chunks into a single comprehensive analysis"


def build_prompt(
    chunk_results: list[str],
    execution_failures_json: str = "",
    meta_data: str = "",
) -> str:
    """Build the merge prompt from chunk results.

    Args:
        chunk_results: List of JSON strings from per-chunk RCA.
        execution_failures_json: Serialized failure list for reference.
        meta_data: Optional execution context/summary.
    """
    parts: list[str] = []

    if execution_failures_json:
        parts.append(f"""## Tagged Failures
```json
{execution_failures_json}
```""")

    if meta_data:
        parts.append(f"""## Execution Context
{meta_data}""")

    parts.append("# PARTIAL ROOT CAUSE ANALYSES")
    for i, result in enumerate(chunk_results, 1):
        parts.append(f"""## Chunk {i} Analysis
{result}""")

    parts.append("""\
# INSTRUCTIONS
1. Parse each chunk's root cause analysis (JSON format)
2. Identify duplicate failures across chunks (same Failure Span ID or same root cause)
3. For duplicates, keep the most complete analysis (including Fix Recommendation)
4. Merge all unique failures into a single JSON array
5. Maintain the original output format and attribute definitions
6. Ensure causality relationships are consistent across the merged result
7. Preserve all Failure Span ID and Location references exactly as they appear
8. Preserve Fix Recommendation objects from the chunk analyses
9. CRITICAL: Ensure each Failure Span ID appears exactly once in the merged output (1:1 mapping)

# DEDUPLICATION RULES
- If the same Failure Span ID appears in multiple chunks, keep only one entry (prefer the most detailed)
- If different Failure Span IDs describe the same root cause, keep both entries (do not merge)
- Maintain chronological ordering by span occurrence when possible
- CRITICAL: The output must maintain 1:1 mapping between Failure Span ID and root cause entries

# OUTPUT FORMAT
Generate ONLY valid JSON as an array of root cause objects:
[
    {
      "Failure Span ID": "span_id_from_execution_failures",
      "Location": "span_id_where_root_cause_occurred",
      "Failure Causality": "PRIMARY_FAILURE|SECONDARY_FAILURE|TERTIARY_FAILURE|UNCLEAR",
      "Failure Propagation Impact": ["TASK_TERMINATION|QUALITY_DEGRADATION|INCORRECT_PATH|STATE_CORRUPTION|NO_PROPAGATION|UNCLEAR"],
      "Failure Detection Timing": "IMMEDIATELY_AT_OCCURRENCE|SEVERAL_STEPS_LATER|ONLY_AT_TASK_END|SILENT_UNDETECTED",
      "Completion Status": "COMPLETE_SUCCESS|PARTIAL_SUCCESS|COMPLETE_FAILURE",
      "Root Cause Explanation": "...",
      "Fix Recommendation": {
        "Fix Type": "SYSTEM_PROMPT_FIX|TOOL_DESCRIPTION_FIX|OTHERS",
        "Recommendation": "..."
      }
    }
]

CRITICAL:
- Output ONLY the JSON array, no markdown formatting, no additional text
- Ensure all Failure Span IDs are preserved exactly as they appear in the chunks
- Ensure all Location span_ids are preserved exactly as they appear in the chunks
- Maintain consistency in causality classifications
- Eliminate true duplicates (same Failure Span ID) but preserve distinct failures
- Preserve Fix Recommendation objects with their Fix Type and Recommendation fields
- Each Failure Span ID must appear exactly once in the output (1:1 mapping)

Your merged analysis:""")

    return "\n\n".join(parts)
