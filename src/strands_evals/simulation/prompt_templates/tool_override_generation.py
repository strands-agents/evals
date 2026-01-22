"""
Prompt template for tool override generation in Strands Evals.

This module contains the prompt template used to analyze test scenarios and determine
optimal tool simulation strategies for agent evaluation workflows. It applies scientific
tool categorization to ensure consistent and appropriate simulation decisions across
different tool types and usage contexts.
"""

from textwrap import dedent

TOOL_OVERRIDE_GENERATION_PROMPT = dedent(
    """You are an expert at analyzing test scenarios and determining optimal tool simulation strategies for agent evaluation workflows.

Your primary objective is to apply SCIENTIFIC TOOL CATEGORIZATION and ensure CONSISTENCY in tool simulation decisions.

## Scenario
{scenario}

## Available Tools
{tools_json}

## Scientific Tool Categorization Framework

Based on comprehensive analysis of MCP servers and tool libraries, tools fall into four primary categories:

### CATEGORY 1: COMPUTE TOOLS (Default: REAL)
**Characteristics**: Pure computational functions, no side effects, no state changes
**Examples**: Mathematical operations, FFT, string manipulation, date formatting, validation
**Simulation Strategy**: Connect directly via MCP - these are safe and deterministic
**Rationale**: No external dependencies, consistent results, low security risk

### CATEGORY 2: DATABASE/PERSISTENT STATE TOOLS (Default: SIMULATE)
**Characteristics**: CRUD operations, booking systems, inventory management, resource allocation
**Examples**: create_booking(), update_inventory(), delete_user(), query_orders()
**Simulation Strategy**: MUST use synthetic/dummy databases with relevant test data
**Rationale**: Cannot connect to production DBs; subsequent operations depend on consistent state
**Critical Rule**: If ANY tool modifies a resource, ALL tools operating on that resource MUST be simulated

### CATEGORY 3: ML MODEL TOOLS (Default: CONTEXT-DEPENDENT)
**Characteristics**: Calls to other ML models, AI services, content generation
**Examples**: image_generator(), text_classifier(), sentiment_analyzer(), llm_call()
**Simulation Strategy**: Evaluate based on scenario requirements and cost considerations
**Rationale**: May need human supervision; consider latency and cost implications

### CATEGORY 4: SPECIALIZED TOOLS (Default: SIMULATE)
**Characteristics**: External integrations, infrastructure operations, specialized hardware
**Examples**: 3D renderers, CAD functions, game engines, deployment tools, notification services
**Simulation Strategy**: Require specialized support; simulate unless explicitly needed
**Rationale**: Complex dependencies, potential side effects, specialized environments

## Consistency Rules (CRITICAL)

**RULE 1 - Resource State Consistency**: 
If tool A modifies resource R, then ALL tools B, C, D that operate on resource R MUST have the same simulation decision.
Example: cancel_flight(booking_id) simulated → get_flight_status(booking_id) must also be simulated

**RULE 2 - Workflow Integrity**: 
Tools in the same logical workflow should maintain consistent simulation decisions to preserve end-to-end test validity.

**RULE 3 - External Service Consistency**: 
If one tool calls external service S, related tools calling service S should have consistent simulation decisions.

## Instructions

For EACH tool, analyze:

1. **Category Classification**: Determine which of the 4 categories (1-4) this tool belongs to
2. **Resource Dependencies**: Identify what resources/services this tool operates on
3. **Consistency Impact**: List other tools that must have matching simulation decisions
4. **Simulation Decision**: Apply category defaults, then adjust for consistency rules

## Failure Conditions Specification

Configure failure simulation with these parameters:

{{
  "enabled": true,              // Whether failure simulation is enabled (boolean)
  "error_rate": 0.15,          // Error rate between 0.0 and 1.0 (float)
  "error_type": "timeout",     // Error type (see allowed values below)
  "error_message": "Custom error message"  // Optional custom error message (string)
}}

### Examples of Error Types:
- `"timeout"` - Request timeout errors
- `"execution_error"` - General execution failures  
- `"network_error"` - Network connectivity issues
- `"authentication_error"` - Authentication failures
- `"authorization_error"` - Permission denied errors
- `"rate_limit_error"` - Rate limiting errors
- `"internal_error"` - Internal system errors

### Failure Rate Guidelines:
- **0.0** - No failures (disabled)
- **0.01-0.05** - Low failure rate (1-5%) - production-like
- **0.1-0.2** - Medium failure rate (10-20%) - stress testing
- **0.3+** - High failure rate (30%+) - chaos engineering

## Response Format

{{
  "scenario_summary": "Brief summary of the scenario and testing objectives",
  "resource_groups": {{
    "group_name": {{
      "description": "What this resource group represents",
      "tools": ["tool1", "tool2", "tool3"],
      "simulation_decision": true,
      "rationale": "Why all tools in this group have the same decision"
    }}
  }},
  "tool_overrides": [
    {{
      "tool_name": "name_of_tool",
      "category": 1,
      "category_rationale": "Category 1: Compute tool - pure mathematical operation with no side effects",
      "resource_dependencies": ["resource_name"],
      "consistency_requirements": ["related_tool1", "related_tool2"],
      "should_simulate": false,
      "failure_conditions": {{
        "enabled": false,
        "error_rate": 0.0,
        "error_type": "execution_error",
        "error_message": "Error message if failure occurs"
      }},
      "rationale": "Final decision rationale considering category and consistency rules"
    }}
  ]
}}

Generate only valid JSON with no markdown code blocks or additional explanation."""
)
