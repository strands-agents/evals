"""Constants for OTEL trace mappers.

Scope names identify which instrumentation library produced the spans.
Attribute keys and values are used for span type detection and data extraction.
"""

# --- Instrumentation scope names ---
SCOPE_LANGCHAIN_OTEL = "opentelemetry.instrumentation.langchain"
SCOPE_OPENINFERENCE = "openinference.instrumentation.langchain"
SCOPE_OPENINFERENCE_SMOLAGENTS = "openinference.instrumentation.smolagents"
SCOPE_OPENINFERENCE_CLAUDE_AGENT_SDK = "openinference.instrumentation.claude_agent_sdk"
SCOPE_ADK = "gcp.vertex.agent"
SCOPE_STRANDS = "strands.telemetry.tracer"

# All scopes that should route to OpenInferenceSessionMapper
SCOPES_OPENINFERENCE_FAMILY = frozenset(
    [
        SCOPE_OPENINFERENCE,
        SCOPE_OPENINFERENCE_SMOLAGENTS,
        SCOPE_OPENINFERENCE_CLAUDE_AGENT_SDK,
    ]
)

# --- OTEL semantic convention attribute keys ---
ATTR_LLM_REQUEST_TYPE = "llm.request.type"

# --- Traceloop/OpenLLMetry attribute keys ---
ATTR_TRACELOOP_SPAN_KIND = "traceloop.span.kind"
ATTR_TRACELOOP_ENTITY_NAME = "traceloop.entity.name"
ATTR_TRACELOOP_ENTITY_INPUT = "traceloop.entity.input"
ATTR_TRACELOOP_ENTITY_OUTPUT = "traceloop.entity.output"

# --- Traceloop/OpenLLMetry attribute values ---
KIND_WORKFLOW = "workflow"
KIND_TOOL = "tool"
LLM_TYPE_CHAT = "chat"

# --- ADOT body field values (LangChain serialization) ---
ADOT_ROLE_UNKNOWN = "unknown"
ADOT_LANGGRAPH_NAME = "LangGraph"
ADOT_TOOL_CALL_WITH_CONTEXT = "tool_call_with_context"
ADOT_INPUT_STR_KEY = "input_str"
