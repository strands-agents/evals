"""
Prompt templates for tool response generation in Strands Evals.

This module contains prompt templates used to generate realistic tool responses during
agent evaluation scenarios. These templates enable LLM-powered simulation of tool
behavior when actual tools are not available or when consistent, controllable responses
are needed for evaluation purposes.
"""

from textwrap import dedent

FUNCTION_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """
You are simulating a function tool call for agent evaluation. Generate a realistic response based on the function name, parameters, and context.

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

MCP_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """You are simulating an MCP (Model Context Protocol) tool call for agent evaluation. Generate a realistic response based on the tool name, input payload, and context.

## MCP Tool Information  
Tool Name: {tool_name}
Input Payload: {mcp_payload}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the tool name and input payload to understand what this MCP tool should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic response following MCP response format
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON in MCP response format

## MCP Response Format
MCP tools return responses in this format:

For successful operations:
```json
{{
  "content": [
    {{
      "type": "text",
      "text": "Operation completed successfully. Retrieved 5 items."
    }}
  ]
}}
```

For data operations:
```json
{{
  "content": [
    {{
      "type": "text", 
      "text": "Found user profile for john.doe"
    }},
    {{
      "type": "resource",
      "resource": {{
        "uri": "user://john.doe",
        "name": "John Doe Profile",
        "mimeType": "application/json"
      }}
    }}
  ]
}}
```

For errors:
```json
{{
  "isError": true,
  "content": [
    {{
      "type": "text",
      "text": "Error: User not found"
    }}
  ]
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)

API_TOOL_RESPONSE_GENERATION_PROMPT = dedent(
    """You are simulating an API tool call for agent evaluation. Generate a realistic HTTP response based on the API endpoint, method, payload, and context.

## API Tool Information
Tool Name: {tool_name}  
Path: {path}
Method: {method}
Request Payload: {api_payload}

## Initial State Context
{initial_state_description}

## Current State & Previous Tool Responses (for context)
{previous_responses}

## Instructions
1. Analyze the API path, method, and payload to understand what this endpoint should do
2. Use the initial state description to understand the starting context and available data
3. Generate a realistic HTTP response with appropriate status code and data
4. Consider the previous responses to maintain consistency in the simulation
5. Ensure responses are consistent with the established state and realistic for the domain
6. Return valid JSON in HTTP response format

## HTTP Response Format
API responses should include status codes and appropriate data:

For successful GET requests:
```json
{{
  "status": 200,
  "data": {{
    "id": 123,
    "name": "Example Item",
    "created_at": "2024-01-15T10:30:00Z"
  }}
}}
```

For successful POST requests:
```json
{{
  "status": 201,
  "data": {{
    "id": 456,
    "message": "Resource created successfully"
  }}
}}
```

For errors:
```json
{{
  "status": 404,
  "error": {{
    "type": "not_found",
    "title": "Not Found", 
    "detail": "The requested resource could not be found"
  }}
}}
```

For validation errors:
```json
{{
  "status": 400,
  "error": {{
    "type": "validation_error",
    "title": "Bad Request",
    "detail": "Missing required field: email"
  }}
}}
```

Generate only valid JSON with no markdown code blocks or additional explanation."""
)
