"""
Prompt templates for tool response generation in Strands Evals.

This module contains prompt templates used to generate realistic tool responses during
agent evaluation scenarios. These templates enable LLM-powered simulation of tool
behavior when actual tools are not available or when consistent, controllable responses
are needed for evaluation purposes.
"""

from textwrap import dedent

TOOL_RESPONSE_GENERATION_PROMPT = dedent(
        """You are simulating the execution of the tool '{tool_name}'.

Tool Input Schema:
{input_schema}

Output Schema:
{output_schema}

User Input Payload:
{user_payload}

Available State Context:
{previous_responses}

IMPORTANT:
- Simulate what this tool would return when called with the provided parameters
- Use the state context and call history to provide consistent responses
- Validate parameters against the tool schema strictly
- Validate the output against the output schema if provided
- Return realistic responses that match the tool's purpose and output schema


VALIDATION RULES:
1. Check required parameters from the schema - if any are missing, return a validation error
2. Check parameter types - if any have wrong types, return a validation error
3. Only return success if all parameters are valid

Generate only valid JSON with no markdown or explanation.
"""
)