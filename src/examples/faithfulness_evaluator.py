from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent

from strands_evals import Case, Experiment
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


def user_task_function(case: Case) -> str:
    agent = Agent(
        trace_attributes={"gen_ai.conversation.id": case.session_id, "session.id": case.session_id},
        callback_handler=None,
    )
    agent_response = agent(case.input)
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(agent_response), "trajectory": session}


test_cases = [
    Case[str, str](name="knowledge-1", input="What is the capital of France?", metadata={"category": "knowledge"}),
    Case[str, str](name="knowledge-2", input="What color is the ocean?", metadata={"category": "knowledge"}),
]

evaluators = [FaithfulnessEvaluator()]

experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

reports = experiment.run_evaluations(user_task_function)
reports[0].run_display()
