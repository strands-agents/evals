"""
Prompt templates for function tool response generation in Strands Evals.

This module contains prompt templates used to generate realistic tool responses during
agent evaluation scenarios. These templates enable LLM-powered simulation of tool
behavior when actual tools are not available or when consistent, controllable responses
are needed for evaluation purposes.
"""

from textwrap import dedent

FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """
You are simulating a function tool call for agent evaluation. Generate a realistic response based on the function name, 
parameters, and context.

## Function Tool Information
Tool Name: {tool_name}
Parameters: {parameters}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the function name and parameters to understand what this tool should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic response that would be returned by such a function
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON that represents the function's return value

## Response Format
Return a JSON object that represents what this function would return. Examples:

For data retrieval functions:
```json
{{
  "status": "success",
  "data": {{
    "result": "retrieved data",
    "count": 42
  }}
}}
```

For action functions:
```json
{{
  "status": "success", 
  "message": "Action completed successfully",
  "transaction_id": "txn_12345"
}}
```

For calculation functions:
```json
{{
  "result": 156.78,
  "unit": "meters"
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)
