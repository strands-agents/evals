from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent

from strands_evals import Case, Dataset
from strands_evals.evaluators import FaithfulnessEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# ======================================
# SETUP TELEMETRY
# ======================================
telemetry = StrandsEvalsTelemetry()
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
telemetry.tracer_provider.add_span_processor(span_processor)


# ======================================
# SETUP AND RUN STRANDS EVAL
# ======================================


def user_task_function(query: str) -> str:
    agent = Agent(callback_handler=None)
    agent_response = agent(query)
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id="test-session")

    return {"output": str(agent_response), "trajectory": session}


test_cases = [
    Case[str, str](name="knowledge-1", input="What is the capital of France?", metadata={"category": "knowledge"}),
    Case[str, str](name="knowledge-2", input="What color is the ocean?", metadata={"category": "knowledge"}),
]

evaluator = FaithfulnessEvaluator()

dataset = Dataset[str, str](cases=test_cases, evaluator=evaluator)

report = dataset.run_evaluations(user_task_function)
report.run_display()
